# Speculative Decoding in Practice: How Draft Models, Medusa and EAGLE Buy Speed, and Where They Lose It

All three methods exploit the same asymmetry: generating one token at a time leaves the accelerator idle because decoding is memory-bandwidth-bound, so a cheap guesser proposes several tokens and the expensive target model checks them all in a single parallel forward pass. The differences are in where the guesses come from — a second smaller model, a set of extra output heads bolted onto the frozen backbone, or an autoregressive drafter that works in the target's own feature space — and those differences drive the acceptance rate, the overhead, and the engineering risk.

The realistic ceiling is **2–3× on a single stream** and considerably less under load. In a production engine at batch 32, EAGLE-3 returns **1.90×** on Llama 3.3 70B but only **1.06×** on Qwen3 235B, and an n-gram baseline returns **0.88×** — slower than not speculating at all [1]. The gap between configurations is larger than the gap between methods.

## The shared mechanism, and why the output does not change

A draft model *q* proposes γ tokens autoregressively. The target *p* scores all γ in parallel. Each drafted token *x* is accepted with probability `min(1, p(x)/q(x))`; on the first rejection, the offending token is resampled from the normalised residual `norm(max(0, p − q))`, and everything after the rejection is discarded [2][3]. The bookkeeping is exact rather than approximate: the accepted mass contributes `min(p,q)` and the rejection path contributes `p − min(p,q)`, so the emitted token is distributed as `p` for **any** pair of distributions. Leviathan et al. prove this in an appendix, Chen et al. prove the same result independently as their Theorem 1, and the argument extends to temperature, top-k, nucleus and argmax once those are expressed as adjusted distributions [2][3].

That exactness is free of any assumption about the draft's quality; what draft quality buys is *speed*, through the acceptance rate. With mean acceptance α and draft-to-target latency ratio *c*, one verification pass yields

```
E[tokens per pass] = (1 − α^(γ+1)) / (1 − α)
speedup            = (1 − α^(γ+1)) / ((1 − α)(γc + 1))
```

Under those formulas the draft's job is to be right often and cheap per token. The paper's own worked examples give **3.69×** at α = 0.8, γ = 5, and **6.86×** at α = 0.9, γ = 10 [2]. They also show the fragility: plugging α = 0.5, γ = 5 and c = 0.10 into the same expression gives only **1.31×** (calculated from the formula in [2]). As *c* → 0 the speedup is bounded by `1/(1 − α)`, which is why a very small, very accurate draft is the ideal case and why a draft that is cheap but frequently wrong is not.

The physical reason it works is memory traffic. Target weights and KV-cache reads can be issued once per verification iteration instead of once per token, so the memory accesses attributable to the target shrink by roughly the number of tokens accepted in that pass [2]. This is also why the method is framed as a *latency* optimisation for medium-to-low QPS, memory-bound workloads rather than a peak-throughput win [4].

One caveat on the algebra: it treats acceptance as independent and identically distributed across positions, and a 352-experiment study on LLAMA-65B and OPT-66B reports that this assumption is inaccurate because tokens within a sentence are correlated. The authors replace it with a latency-aware model, `Throughput = TAR / (t_target^d + t_draft^d)`, validated to within **3.5%** of measured throughput on 4×A100 80GB, and find that **draft latency and depth, not draft language-modelling accuracy, dominate** [5]. A draft with fewer, wider layers beats a deeper one at equal parameter count.

## Three ways to produce a draft

| | Draft-model decoding | Medusa | EAGLE |
|---|---|---|---|
| Draft source | Separate small LM, often distilled [3][6] | K single-layer heads on frozen backbone [7] | 1 decoder layer on fused target features [8][9] |
| Verification | Chain of γ tokens, one target pass [2] | Tree over per-head top-k Cartesian product [7] | Tree attention; static then dynamic [10][9] |
| Reported acceptance | α 83.6–89.5% at T=0.8 [11] | 3.0–3.5 tokens/step [7] | τ 2.4–6.6, config-dependent [9][1] |
| Reported speedup | 1.9–2.5× Chinchilla 70B [3] | 1.5–2.8× Vicuna [7][12][13] | 2.1–3.8×; up to 6.5× claimed [8][9] |
| Extra cost | Whole second model + KV cache [14][15] | ~5 GPU-hours heads, 1.22× per-step [7] | 0.24–0.99B params, 1–2 days [8] |

### Draft-model decoding: the reference implementation

The draft is typically a smaller version of the target, distilled or simply trained on the same corpus. Chen et al. used a 4B, 8-layer draft against Chinchilla 70B at batch 1, and the measured latency ratio was about **0.13** — 1.8 ms/token for the draft against 14.1 ms/token for the target (calculated from the cited per-token latencies) [3]. That ratio is what makes the scheme pay: it is the *c* in the denominator of the speedup formula.

Draft quality is trainable, and distillation is the standard lever. DistillSpec aligns a 77M T5-Small draft with a 3B T5-XL target and reports **10–45% speedups over standard speculative decoding**, plus 6.4×/10.7× latency reduction when the target itself is also distilled [6]. A temperature-matched recipe — Llama-68M draft, Llama-2-13B-chat target, KD temperatures aligned to the decoding temperature — reaches **α = 83.6% and 2.34×** on Alpaca and **α = 89.5% and 5.62×** on GSM8K at decoding temperature 0.8 [11].

There is a cheaper variant: self-speculation skips a subset of the target's own layers to draft, then verifies with the full model. It needs no second model, no extra memory and no network training — only a layer-skip mask found by Bayesian optimisation in about 2.5 hours — and on LLaMA-2-13B/CNN-DM it reaches **acceptance 0.919 and 1.57×**, against 0.592 and 1.01× for random skipping [16]. Block efficiency peaks at roughly half the layers skipped.

### Medusa: heads instead of a model

Medusa appends K decoding heads to the backbone's final hidden state; head *k* predicts token *t+k+1* while the original LM head predicts *t+1*. Each head is a single feed-forward layer with a residual connection, initialised so its predictions begin identical to the base LM head, and the backbone stays frozen in Medusa-1 [7]. Because the heads are trained on the same model's features, there is no draft-model pretraining, no second model to serve, and no draft-target distribution shift to manage.

Candidates come from the **Cartesian product of each head's top-k predictions**, flattened into a tree and verified in one forward pass by a tree-causal mask in which each node attends only to its ancestors [7][12]. Five heads with one layer each suffice; with optimised trees, three or four are enough, and redundant heads can be dropped without overhead [7]. The tree matters as much as the heads: a **sparse 64-node tree beats a dense 256-node tree** because verification overhead is compute-bound, so extra candidates cost more than they return [7]. Training Medusa-1 on Vicuna-7B took about **5 GPU-hours on a single A100 PCIe** with 60k ShareGPT samples [7]; the Hugging Face TGI recipe uses 3 heads and 1 layer over 4 GPUs and warns that out-of-domain training data can make the speedup "minimal or nonexistent" [17].

The cost of dropping the second model is that Medusa does not preserve the target distribution. Its non-greedy path uses an entropy-threshold "typical acceptance" rule, and the project documentation states plainly that relaxing the distribution-matching requirement is what makes non-greedy generation faster than greedy [18][7].

### EAGLE: drafting in feature space, with better trees

EAGLE drafts autoregressively not over tokens but over the target's **second-to-top-layer feature vector**, feeding back both the sampled token and a predicted feature, and it obtains draft tokens through the target's own frozen LM head [8]. Only a small autoregressive head is trained — 0.24B, 0.37B, 0.56B and 0.99B parameters for 7B, 13B, 33B and 70B targets respectively, with training of the 70B head taking 1–2 days on 4×A100 40G [8].

EAGLE-2 leaves the head and its training untouched and changes only inference-time tree construction: it exploits the fact that draft confidence is well calibrated to acceptance (on Alpaca with Vicuna-7B, tokens with confidence below 0.05 were accepted about 4% of the time, while tokens above 0.95 were accepted about 98% of the time), expands the highest-value nodes, reranks, and keeps a context-dependent tree instead of EAGLE-1's fixed shape [10]. That single change is worth 20–40% over EAGLE-1 and takes the reported range to **3.05–4.26× at temperature 1** [10].

EAGLE-3 changes the training objective rather than the tree. It fuses low-, mid- and high-level target features, projects them to hidden size, adds the sampled-token embedding, and runs a single Transformer decoder layer; "training-time test" simulates multi-step drafting during training by feeding drafts back as inputs with adjusted attention masks, and removing the feature-prediction loss is what frees the drafter to use intermediate-layer features. Draft-tree depth rises from 6 to 8 at unchanged node count, so EAGLE-3 checkpoints are not interchangeable with earlier heads [9].

## What the papers report, and why the numbers are not comparable

| Configuration | Setting | Acceptance | Gain |
|---|---|---|---|
| Draft model, Chinchilla 70B + 4B draft, K=4 | XSum, nucleus p=0.8, batch 1 | 14.1 → 7.52 ms/token | 1.92× [3] |
| Draft model, same pair | HumanEval, T=0.8 | 14.1 → 5.73 ms/token | 2.46× [3] |
| Draft model, Llama-68M + Llama-2-13B-chat | Alpaca / GSM8K, T=0.8 | α 83.6% / 89.5% | 2.34× / 5.62× [11] |
| Medusa-2, Vicuna-7B | MT-Bench, batch 1 | 3.47 tokens/step, 1.22× overhead | 2.83× [7] |
| Medusa-2, Vicuna-13B | MT-Bench, T=1 | τ 3.26 | 2.65× [19] |
| Medusa, Vicuna-7B | MT-Bench, greedy, A100, batch 1 | accepted length 2.52 | 1.87× [12] |
| Medusa, Vicuna-7B | Spec-Bench avg, greedy | accepted length 2.15 | 1.50× [12] |
| EAGLE, Vicuna-7B | MT-Bench, greedy | — | 2.90× [8] |
| EAGLE-3, Vicuna-13B | T=0 | τ 6.62 | up to 6.5× [9] |
| EAGLE-3, Llama-3.3-70B | SPEED-Bench, BS=32, T=0 / T=1 | AL 2.44 / 2.37 | 1.90× / 1.75× [1] |
| EAGLE-3, Qwen3-235B | SPEED-Bench, BS=32 | AL 2.36 | 1.06× [1] |

**The acceptance column cannot be used to rank methods.** The cleanest same-configuration comparison available — Vicuna-7B-v1.3, FP16, batch 1, greedy, one RTX 3090, mean of three runs over 480 samples — ranks EAGLE at **2.08×** (76.30 tok/s), draft-model speculative sampling with a 68M drafter at **1.77×** (64.85 tok/s) and Medusa at **1.48×** (54.30 tok/s) against a 36.74 tok/s baseline [13]. Move the identical recipe to an A100 80GB and Medusa rises to **2.42×** while EAGLE reaches **2.39×** — the order flips, because the bottleneck was hardware utilisation, not method quality [13]. EAGLE also falls to **1.74×** at temperature 1 and to **1.74×** in FP32, versus 2.39× in FP16 [13]. Within a single method, task mix alone moves the number by 0.5–0.9× (EAGLE: 1.79× on translation to 2.44× on maths; Medusa: 1.29× on RAG to 1.69× on maths) [13]. Medusa is absent from that benchmark's temperature-1 panel entirely, so the two have no matched comparison under sampling [13].

The acceptance-length spread is similarly configuration-bound. EAGLE-3's own report gives mean acceptance lengths of **6.62** (Vicuna-13B), **6.23** (LLaMA-3.1-8B-Instruct), **5.88** (LLaMA-3.3-70B) and **5.84** (DeepSeek-R1-Distill-LLaMA-8B) at temperature 0 [9]; SPEED-Bench measures **2.44** for Llama 3.3 70B at batch 32 with a 3-token draft [1]; ARC-Decode measures **3.26** for Medusa-2 on Vicuna-13B/MT-Bench at temperature 1 [19]; a controlled study on Qwen3-8B at batch 16 measures **5.98** for EAGLE-3 with a 7/10/16 tree under low entropy falling to **3.67** under high entropy [20]. A public vLLM issue report puts the same nominal cell — Qwen3-32B, EAGLE-3, GSM8K, 3 drafted tokens, greedy — at 55% acceptance and 2.65 accepted tokens in vLLM against roughly 44% and 2.32 in a different harness, which says the engine implementation moves τ as much as the method does.

The practical consequence is that the Medusa-versus-EAGLE-versus-draft-model question has no stable scalar answer. EAGLE's authors report it 1.5–1.6× faster than Medusa and 2.1–3.8× over vanilla [8]; an independent re-implementation on Vicuna-7B found EAGLE achieved a higher acceptance length than Hydra++ but only **comparable end-to-end throughput**, attributing the gap to EAGLE querying a full self-attention block per candidate position rather than once per step [21]. Higher acceptance is not the same as higher throughput.

## Where the speedup goes in real serving

### Batch size is the dominant tax

Speculation trades serial latency for parallel verification work, and that trade inverts as the batch fills. EAGLE on Llama2-Chat 70B falls from **3.01× at batch 1 to 2.40× at batch 4** with only 1.99× throughput gain [8]; Hydra++ falls from **2.70× at batch 1 to 1.63× at batch 8** [21]; EAGLE on Llama3.1-8B/GSM8K falls from **1.73× at batch 1 to 1.21× at batch 128** in a production vLLM stack on H100, where verification accounts for **42–95% of runtime** [22]. Wider trees degrade faster: a k=21, depth-3 tree drops below 1× by batch 64 while a k=3 chain stays above it [22]. SGLang's adaptive controller encodes the same logic in its documentation, lowering speculative steps at high batch because the cost of each wasted draft step is multiplied across all sequences [23].

### Temperature and sampling

Acceptance is a function of how peaked the target distribution is, so raising temperature erodes the win. For a distilled draft pair the swing from temperature 0 to 1 was **29–31% relative speedup** in a peer-reviewed study, with temperature-aligned distillation narrowing the gap [11]. For EAGLE-3 the effect is measured on Qwen3-8B at batch 16: acceptance falls from **5.98 to 5.35** under low entropy and from **4.30 to 3.67** under high entropy when moving from greedy to sampled drafting, taking speedup from **3.36× to 2.92×** and **2.48× to 2.05×** — the penalty is largest exactly where speculative decoding is already weakest [20].

Medusa has no comparable curve. Its paper argues from an entropy-threshold argument that higher temperature should *lengthen* accepted sequences under typical acceptance, but the only temperature-controlled measurement is a single run at 0.7, and the claim is stated rather than plotted [7]. Treat any statement about Medusa's temperature behaviour as the authors' reasoning, not a measurement.

### Long context and memory bandwidth

The usual story — speculation helps when decoding is memory-bound and hurts when it is compute-bound — reverses again at long context. Once KV-cache loading dominates, verification cost approaches the cost of plain target decoding, and there is a **critical sequence length** (theoretically around 4,000 tokens for Llama-3.1-8B with γ=3 on 8×A100) above which speedup *increases* with batch size instead of falling [24]. The same work reports up to **2.51×** end-to-end at 100k prefill and batch 41 on 8×H100, and notes that at that scale a draft model can occupy **38–140% of the target's memory footprint** because its KV dimensions are less favourable — so compressed-KV self-speculation reaches 2.01× at batch 128 while a Llama-3.2-1B draft only reaches 1.81×, even though the small draft wins at batch 16–32 [24].

Tree verification also interacts badly with paged KV accounting. Draft tokens consume extra cache pages and count toward the token budget, and because allocators round up to blocks of B tokens while speculation windows are typically 2–12 tokens, speculative KV pressure can inflate by roughly **B/m** — up to about 8× at m=2 with B=16 in one analysis [25]. That analysis is a preprint reporting simulated capacity, not measured token throughput, so treat the multiplier as an accounting argument rather than a benchmark.

### Where it stops paying

Four failure modes recur. First, a model for which good drafts are hard: Qwen3 235B + EAGLE-3 returns **1.06×** at batch 32 with a perfectly respectable 2.36 acceptance length [1]. Second, prompt-overlap workloads where n-gram lookup wins outright — on code editing with high prompt/answer overlap, n-gram beats EAGLE/EAGLE-3 by up to **53–100%** [22], yet the same n-gram method returns **0.88×** on the general SPEED-Bench workload [1]. Third, drafts that are too short: in a vLLM-on-AMD walkthrough, drafting one token at a time returned **0.71×** despite 86.3% acceptance, because the per-step overhead exceeds what one extra token repays; gains only appeared at 5–7 drafted tokens. That last figure comes from a vendor blog rather than a peer-reviewed benchmark, so treat the specific numbers as illustrative. Fourth, overly optimistic measurement: synthetic random-token inputs overestimate speculative-decoding throughput by **23%** relative to real workloads [1].

## Implementation cost and serving support

### Training and memory

The three families sit on different cost curves. Medusa-1 trains in about 5 GPU-hours on one A100 PCIe with a frozen backbone [7]. EAGLE trains 0.24–0.99B parameters for 1–2 days on four A100s at 70B scale, and EAGLE-3's banded-diagonal attention mask is not expressible as a standard causal mask — Nebius implements it with block-sparse FlashAttention, cutting GPU memory utilisation from **44% to 25%** and raising training throughput from **6,900 to 11,200 tok/s** on an 8×H100 pod [26]. Draft-model decoding pays for a whole second model and its KV cache; SpecDec's drafter roughly doubles model-weight memory [14], and a measured two-model deployment added about **2.9 GiB at 1K context and 3.1 GiB at 16K** [15]. Self-speculation sidesteps the memory cost entirely at the price of a layer-skip search [16].

Medusa's own memory delta is nowhere reported. The paper gives no GB figure for the heads or the candidate tree, only the ~5 GPU-hour training cost and the observation that each head is a single layer [7]. A self-run reproduction reports a **~180 MB** head-only checkpoint and 52 GB/GPU training peak for Llama-3.1-8B, and a third-party analysis times each Medusa head at 0.3 ms against a 28 ms base decoding step [27]. Those are useful brackets, not an official measurement, and the head-checkpoint figure comes from a self-published set of training notes.

### Framework support is the real differentiator in 2026

EAGLE is first-class almost everywhere. vLLM's speculative-decoding documentation lists EAGLE, MTP, DSpine/DFlash, draft model, n-gram, suffix and FLy, with `--speculative-config` method values `draft_model`, `ngram`, `suffix`, `mtp`, `eagle3` and `dflash` [4]. SGLang documents EAGLE-2 and EAGLE-3 alongside UNO, DFlash, STANDALONE and NGRAM, and reports 1×H100 throughput of 158.34 tok/s without speculation, 244.10 with EAGLE-2 and 373.25 with EAGLE-3 (batch, context and temperature unstated) [28].

Medusa is not in either list. It does not appear among vLLM's documented methods or method values, and the sanctioned extension route is a custom speculator registered through a plugin entry point [4][29]. SGLang's documentation likewise omits it [28]. TensorRT-LLM is the exception: `MedusaDecodingConfig` shipped in release 0.8.0, extended with in-flight batching and per-sequence rewind in 0.9.0, Qwen2 support in 0.15.0 and LLM-API support in 0.16.0; the original release gated it to `temperature = 0`, Python runtime only, Ampere or newer [30]. Its example configuration uses `max_batch_size=1` [31]. No release note from 1.0 through 1.4 (2026-11) mentions Medusa at all — silence rather than deprecation, but also no evidence of active maintenance [30]. Draft-model decoding took a harder hit in the same engine: release 1.2 removed the separate draft-engine path behind EAGLE-3, MTP-Eagle and Draft-Target, leaving only one-model implementations where the drafter is a submodule [30].

A self-published vLLM plugin benchmark is the main recent evidence of Medusa running at concurrency: Llama-3.1-8B on one H100 at concurrency 16 and temperature 0.7, reporting 139.7 versus 58.2 tok/s (**2.40×**), 78% draft-token acceptance and mean accepted length 3.71 with per-head acceptance declining 91/83/74/66/57% across five heads [32]. It is a single third-party measurement, not a framework-supported configuration.

### Exactness is conditional in every real stack

vLLM describes speculative decoding as "theoretically lossless up to the precision limits of hardware numerics" and explicitly flags its FLy verification path as not distribution-preserving [29]. SGLang's exact rejection-sampling path requires a chain draft with `--speculative-eagle-topk 1`, so it is incompatible with tree top-k greater than 1, and its overlap scheduler only supports topk=1 [28]. Medusa's default non-greedy path is relaxed by design [18].

What relaxation buys and costs is now measured, though not for Medusa specifically. The generic trade-off is one-for-one: raising acceptance by an amount *E* costs exactly *E* in total-variation distance from the target, and the lossless case is recovered at the weight that zeroes the relaxation [33][34]. Empirically the fidelity bill is small but not zero: on Llama-3.1-8B at temperature 1, a risk-bounded acceptance rule lifted MMLU-Pro acceptance length from 2.81 to 3.50 and speedup from 1.60× to 1.88× at unchanged 32.3% accuracy, while HumanEval pass@10 moved 69.9 to 68.2 and one paired GSM8K case returned 8 where the gold answer was 28 [19]. A pre-registered calibrated instrument on Qwen2.5-7B found a relaxed arm statistically equivalent to bf16 (−0.02, CI −0.14 to +0.10) without measurably raising acceptance (0.952 versus 0.950) [35]. Nucleus-based relaxation improved task accuracy in 15 of 30 cells and lost in 14, ranging −1.58 to +4.14 points [34]. Medusa's own quality evidence is a set of GPT-4-judged MT-Bench deltas of +0.01, −0.07, −0.14 and +0.05 across four models [7] — small, but not uniformly positive, and not a controlled comparison against its own exact path.

## Choosing between them

- **Choose EAGLE or EAGLE-3 when you need the highest single-stream speedup and want a supported path.** It has the strongest reported numbers and is the only one of the three with first-class support in vLLM, SGLang and TensorRT-LLM [4][28][30]. Budget the training run and, for EAGLE-3, the block-sparse attention work [8][26].
- **Choose draft-model decoding when you already own a good small model, or when the workload has high draft-target agreement.** It is the best performer on Llama3-70B in independent testing [22], but check your engine: the separate-draft-engine path was removed from TensorRT-LLM in 1.2 [30].
- **Choose self-speculation when memory or training budget is the binding constraint.** Layer-skipping needs no second model, no extra memory and no network training, and reaches 1.57× at 0.919 acceptance on LLaMA-2-13B [16].
- **Choose Medusa only on TensorRT-LLM, or where single-stream latency dominates.** Its head cost is small and its tree mechanism is sound, but the engine support has narrowed to one vendor, the documented example runs at batch size 1 [31], and its relaxed acceptance path has never been compared against its own exact path on identical prompts.
- **Do not use speculation at high batch on a compute-bound deployment, or on prompt-overlap tasks where n-gram lookup is available.** Verification can be 42–95% of runtime [22], a 235B model can return 1.06× [1], and n-gram can beat a trained drafter outright on code editing [22] while losing badly elsewhere [1].

## What the numbers do not settle

Three gaps matter for anyone sizing a deployment. **Medusa has no token-level acceptance rate and no VRAM figure in GB** anywhere in its paper, its repository, or the benchmarking literature — it reports tokens per decoding step and per-step latency overhead, and nothing about memory [7]. **Acceptance length is not comparable across papers** for any of the three methods: the same nominal model and task reads differently in different engines, and the benchmark that comes closest to a controlled three-way comparison reports speedup and tokens/s but never publishes acceptance length [13][36]. And **no source measures Medusa's typical-acceptance path against its own rejection-sampling path on identical prompts**, so the size of the fidelity concession that makes Medusa's non-greedy mode fast remains an inference from general relaxation results [33][34][19] rather than a direct measurement.

The direction of the answer is nonetheless stable: speculative decoding reliably converts spare compute into lower latency at small batch, the three families differ far less than the hardware, batch size and sampling settings they run under, and the method you can actually deploy — which in 2026 means EAGLE on a mainstream engine — matters more than the method with the best headline multiplier.

## References

- **[1] raw.githubusercontent.com** - "SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding". [Source](https://raw.githubusercontent.com/mlresearch/v306/main/assets/abramovich26a/abramovich26a.pdf) (accessed 2026-11-27)
- **[2] arxiv.org** - "Fast Inference from Transformers via Speculative Decoding". [Source](https://arxiv.org/abs/2211.17192) (accessed 2026-11-27)
- **[3] arxiv.org** - "Accelerating Large Language Model Decoding with Speculative Sampling". [Source](https://arxiv.org/pdf/2302.01318) (accessed 2026-11-27)
- **[4] docs.vllm.ai** - "Speculative Decoding". [Source](https://docs.vllm.ai/en/latest/features/speculative_decoding/) (accessed 2026-11-27)
- **[5] aclanthology.org** - "Decoding Speculative Decoding". [Source](https://aclanthology.org/2025.naacl-long.328.pdf) (accessed 2026-11-27)
- **[6] proceedings.iclr.cc** - "DistillSpec: Improving Speculative Decoding via Knowledge Distillation". [Source](https://proceedings.iclr.cc/paper_files/paper/2024/file/8766fbc68e1ed1cdef712ce273e0a363-Paper-Conference.pdf) (accessed 2026-11-27)
- **[7] arxiv.org** - "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads". [Source](https://arxiv.org/pdf/2401.10774) (accessed 2026-11-27)
- **[8] raw.githubusercontent.com** - "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty". [Source](https://raw.githubusercontent.com/mlresearch/v235/main/assets/li24bt/li24bt.pdf) (accessed 2026-11-27)
- **[9] arxiv.org** - "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test". [Source](https://arxiv.org/html/2503.01840) (accessed 2026-11-27)
- **[10] aclanthology.org** - "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees". [Source](https://aclanthology.org/2024.emnlp-main.422.pdf) (accessed 2026-11-27)
- **[11] aclanthology.org** - "Temperature-Centric Investigation of Speculative Decoding with Knowledge Distillation". [Source](https://aclanthology.org/2024.findings-emnlp.767.pdf) (accessed 2026-11-27)
- **[12] aclanthology.org** - "Amphista: Bi-directional Multi-head Decoding for Accelerating LLM Inference". [Source](https://aclanthology.org/2025.naacl-long.450.pdf) (accessed 2026-11-27)
- **[13] aclanthology.org** - "Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding". [Source](https://aclanthology.org/2024.findings-acl.456.pdf) (accessed 2026-11-27)
- **[14] arxiv.org** - "Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation". [Source](https://arxiv.org/abs/2203.16487v6) (accessed 2026-11-27)
- **[15] arxiv.org** - "Memory-Efficient Speculative Decoding with Quantized Draft KV Cache". [Source](https://arxiv.org/html/2611.20565v1) (accessed 2026-11-27)
- **[16] aclanthology.org** - "Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding". [Source](https://aclanthology.org/2024.acl-long.607.pdf) (accessed 2026-11-27)
- **[17] huggingface.co** - "Train Medusa · Hugging Face". [Source](https://huggingface.co/docs/text-generation-inference/basic_tutorials/train_medusa) (accessed 2026-11-27)
- **[18] GitHub** - "GitHub - FasterDecoding/Medusa: Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads". [Source](https://github.com/FasterDecoding/Medusa) (accessed 2026-11-27)
- **[19] raw.githubusercontent.com** - "ARC-Decode: Accelerated Decoding with Risk-Bounded Acceptance". [Source](https://raw.githubusercontent.com/mlresearch/v306/main/assets/li26gh/li26gh.pdf) (accessed 2026-11-27)
- **[20] arxiv.org** - "When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding". [Source](https://arxiv.org/pdf/2610.24993) (accessed 2026-11-27)
- **[21] arxiv.org** - "Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding". [Source](https://arxiv.org/pdf/2402.05109) (accessed 2026-11-27)
- **[22] arxiv.org** - "Speculative Decoding: Performance or Illusion?". [Source](https://arxiv.org/pdf/2601.11580) (accessed 2026-11-27)
- **[23] SGLang Documentation** - "Adaptive Speculative Decoding". [Source](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding) (accessed 2026-11-27)
- **[24] proceedings.iclr.cc** - "MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding". [Source](https://proceedings.iclr.cc/paper_files/paper/2025/file/13f972adf12bdf886583d48cd528002f-Paper-Conference.pdf) (accessed 2026-11-27)
- **[25] techrxiv.org** - "Transactional KV Caching for Speculative Decoding under Paged KV Memory". [Source](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.177101038.80960856) (accessed 2026-11-27)
- **[26] nebius.com** - "Training speculative decoders: removing the logits and attention bottlenecks". [Source](https://nebius.com/blog/posts/training-speculative-decoders) (accessed 2026-11-27)
- **[27] GitHub** - "speculative-decoding-medusa/docs/training_notes.md at main · sudsho/speculative-decoding-medusa". [Source](https://github.com/sudsho/speculative-decoding-medusa/blob/main/docs/training_notes.md) (accessed 2026-11-27)
- **[28] SGLang Documentation** - "Speculative Decoding - SGLang Documentation". [Source](https://docs.sglang.io/docs/advanced_features/speculative_decoding) (accessed 2026-11-27)
- **[29] GitHub** - "vllm/docs/features/speculative_decoding/README.md at main · vllm-project/vllm". [Source](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md) (accessed 2026-11-27)
- **[30] nvidia.github.io** - "Release Notes". [Source](https://nvidia.github.io/TensorRT-LLM/release-notes.html) (accessed 2026-11-27)
- **[31] nvidia.github.io** - "Generate Text Using Medusa Decoding". [Source](https://nvidia.github.io/TensorRT-LLM/1.0.0rc1/examples/llm_medusa_decoding.html) (accessed 2026-11-27)
- **[32] GitHub** - "GitHub - sudsho/speculative-decoding-medusa: Medusa-2 speculative decoding with multiple draft heads as a vLLM plugin; reference implementation, speedups...". [Source](https://github.com/sudsho/speculative-decoding-medusa) (accessed 2026-11-27)
- **[33] raw.githubusercontent.com** - "DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification". [Source](https://raw.githubusercontent.com/mlresearch/v300/main/assets/wang26d/wang26d.pdf) (accessed 2026-11-27)
- **[34] arxiv.org** - "Nucleus Speculative Decoding: Plausibility-Aware Verification Beyond Exact Distribution". [Source](https://arxiv.org/pdf/2610.07822) (accessed 2026-11-27)
- **[35] arxiv.org** - "A Calibrated Instrument for Measuring How Inference". [Source](https://arxiv.org/pdf/2609.18005) (accessed 2026-11-27)
- **[36] GitHub** - "GitHub - hemingkx/Spec-Bench: Spec-Bench: A Comprehensive Benchmark and Unified Evaluation Platform for Speculative Decoding (ACL 2024 Findings)". [Source](https://github.com/hemingkx/Spec-Bench) (accessed 2026-11-27)