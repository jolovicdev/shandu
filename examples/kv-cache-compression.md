# KV-Cache Compression for Long-Context LLM Inference: Quantization, Eviction, and Low-Rank Methods Compared

## Executive Summary

Quantization is the only one of the three families that production inference engines actually ship, and it is also the family with the most consistent accuracy-per-byte story in the literature. Eviction buys larger nominal compression ratios but degrades sharply on long-context and multi-turn workloads, and low-rank/latent methods achieve the highest compression ratios only when stacked with quantization or baked into pretraining.

The strongest single anchor is KVQuant, which reports **3.7×, 4.8×, and 6.9× KV-cache reduction at 4-, 3-, and 2-bit** with under 0.02, 0.1, and 0.5 Wikitext-2 perplexity degradation respectively across LLaMA/Llama-2/Llama-3/Mistral [1]. At 128K sequence length, LLaMA-7B's KV cache falls from 64.0 GB in fp16 to 13.3 GB at 3-bit with 1% outliers and 9.3 GB at 2-bit [1]. Those savings are estimated from bit-width assumptions rather than measured end-to-end, but the perplexity numbers are measured.

Eviction methods report headline ratios of 5–10× (H2O at a 20% budget [2]) and ~92% compression at a 1024-token cache on LongBench (SnapKV [3]), yet peer-reviewed matched-budget work shows the picture darkens with context length: at 15% retention on RULER, SnapKV scores 12.82 at 128K against a full-cache 62.30, while a learned policy (LKV) holds 55.34 [4]. SCBench reaches a similar conclusion, arguing that sub-O(n) memory is "almost infeasible in multi-turn decoding" because the critical KVs change per query [5].

Low-rank and latent methods reach the highest compression — DeepSeek-V2's MLA cuts KV cache 93.3% versus DeepSeek 67B [6], and Palu reaches 91.25% when combined with 2-bit quantization [7] — but the pure low-rank operating point is modest (Eigen Attention tops out at 40% reduction [8]) and post-hoc conversion carries real costs.

**Bottom line:** for a serving deployment today, FP8 KV cache is the safe default-adjacent lever (roughly 2× memory at minimal measured accuracy cost), with 4-bit integer schemes as the research frontier for larger savings. Eviction is a research technique with documented long-context failure modes, and low-rank is either a pretraining decision (MLA) or a research method that needs quantization stacked on top to be competitive.

## Key Findings

1. **Quantization scales predictably with bit-width, and the degradation curve is steep at 2-bit.** KVQuant's measured Wikitext-2 perplexity on LLaMA-7B moves from 5.68 (fp16) to 5.69 (4-bit), 5.75 (3-bit), and 6.01 (2-bit) [1]. Naive integer baselines at the same widths are catastrophic — int3 gives 10.87 and int2 gives 11,779 — so the method's gains come from granularity and outlier handling, not bit-width alone [1].

2. **Granularity choices dominate quantization quality.** In KVQuant's ablation, per-channel Key plus per-token Value quantization improves perplexity by 3.82 over per-token quantization of both at 3-bit; quantizing Keys before RoPE rather than after adds another 0.82; keeping 1% of outliers in fp16 adds 0.19 [1].

3. **Long-context accuracy at 3-bit is close to fp16; 2-bit is not.** On LLaMA-2-7B-32K, KVQuant at 3.33 average bits scores 31.21 on LongBench versus 31.96 for fp16, and 53.65 versus 56.40 on RULER; at 2.33 bits RULER falls to 36.54 [1]. KIVI at 3.05 bits scores 30.04 and 39.78 on the same pair, which KVQuant cites as roughly 14% better RULER at similar width [1].

4. **Eviction's accuracy depends heavily on the policy and the evaluation protocol.** H2O matches full-cache accuracy at a 20% budget on OPT-30B HELM tasks, while a recency-only policy collapses (PiQA 57.94 versus 80.09) [2]. StreamingLLM's recency-plus-sink hybrid restores Llama-2-13B perplexity from 5158.07 to 5.40 on a 65K-token book [9], but its accuracy on StreamEval falls to zero once the query–answer distance exceeds the cache size [9].

5. **Attention-score eviction degrades badly at long context and under question-agnostic evaluation.** Ada-KV reports SnapKV's LongBench average dropping from 49.09 to 42.86 when moving from question-aware to question-agnostic scoring at a 2048 budget [10]. LKV's RULER sweep at 15% retention shows SnapKV at 12.82 and PyramidKV at 10.53 at 128K, against 62.30 for full cache [4].

6. **Low-rank alone is a moderate-compression technique; its high-compression results come from stacking.** Eigen Attention reports up to 40% KV reduction with average accuracy drops of 1%/2%/3% at 0.8×/0.7×/0.6× across nine models [8]. Palu reports 50% compression at Llama-2-7B WikiText-2 perplexity 6.01 versus 5.47 baseline, and 91.25% compression at 5.76 perplexity when combined with 2-bit quantization [7].

7. **At matched memory, quantization often beats low-rank alone.** Palu's own table shows KVQuant at 3-bit reaching 5.35 perplexity at 81.25% compression, while Palu-50% in fp16 reaches 5.63 at only 50% compression [7]. Palu-30% plus 3-bit reaches 5.33 at 86.87%, roughly tied with KVQuant at higher compression [7].

8. **Production engines ship quantization only, as opt-in flags.** vLLM exposes FP8 KV cache through `kv_cache_dtype` with per-tensor or per-head scaling [11]; SGLang adds FP4 formats (`nvfp4`, `fp4_mx_block16`) [12]; TensorRT-LLM supports FP8/INT8 per-tensor KV and a prototype NVFP4 mixed-precision mode [13][14]. No engine documentation shows token-level eviction or low-rank KV as a user-facing feature.

9. **FP4 KV cache can break hard reasoning while leaving easy benchmarks flat.** SGLang's own table shows GPT-OSS-120B scoring 0.7533 on aime25 at KV16 versus 0.3533 at KV4, while gsm8k stays at 0.9161 versus 0.9152 [12].

## Detailed Analysis

### 1. Quantization: bit-width, granularity, and the accuracy cliff

#### 1.1 What the savings actually are

KVQuant's headline ratios are 3.7×, 4.8×, and 6.9× at 4-, 3-, and 2-bit [1]. These are not raw bit-width ratios: the effective widths including scaling and outlier overhead are about 4.32–4.35 bits at "4-bit", 3.32–3.35 at "3-bit", and 2.32–2.35 at "2-bit" [1]. The paper's KV-cache byte figures are estimated from those bit-width assumptions, not measured on hardware: LLaMA-7B at 128K sequence length goes from 64.0 GB fp16 to 17.3 GB (4-bit), 13.3 GB (3-bit with 1% outliers), and 9.3 GB (2-bit) [1]. The same estimates put LLaMA-7B at 1M tokens on a single A100-80GB at 2-bit [1].

Atom takes a different route: asymmetric INT4 at per-attention-head granularity, fused into FlashInfer with paged attention [15]. Its KV-specific accuracy cost is small — 0.12 WikiText-2 perplexity on Llama-7B (6.04 to 6.16) on top of W4A4 weights and activations [15] — but the paper reports no KV-specific compression ratio and no long-context benchmarks, so it anchors the granularity question rather than the savings question [15].

#### 1.2 Why granularity matters more than bit-width

The ablation is the most useful part of KVQuant for practitioners. At 3-bit on LLaMA-7B, quantizing Keys per-channel and Values per-token instead of both per-token improves perplexity by 3.82 (7.05 versus 10.87) [1]. Moving Key quantization before RoPE rather than after adds 0.82 (6.23 versus 7.05) [1]. Non-uniform datatypes add 0.29, and isolating 1% of outliers in fp16 adds 0.19 [1]. The mechanism is that RoPE rotates Keys after they are written, so quantizing the rotated representation mixes positional information into the quantization error.

This is why the naive baselines are so bad: int3 gives 10.87 and int2 gives 11,779 on LLaMA-7B, while KVQuant's tuned 3-bit gives 5.75 and 2-bit gives 6.01 [1]. A team that simply casts the cache to INT4 without per-channel Key scaling and outlier handling should expect the naive numbers, not the tuned ones.

#### 1.3 The accuracy curve on long-context benchmarks

On LLaMA-2-7B-32K, KVQuant at 3.33 average bits scores 31.21 on LongBench against 31.96 for fp16, and 53.65 on RULER against 56.40 [1]. At 2.33 bits, RULER drops to 36.54 — a loss of roughly 20 points from fp16, calculated from the cited figures [1]. KIVI at 3.05 bits scores 30.04 on LongBench and 39.78 on RULER [1].

The independent benchmark by Agrawal and Mayer puts realized compression at about 5.16–5.32× for KIVI 2-bit and 3.15–3.19× for KIVI 4-bit, measured as prefill KV memory on an A100 40GB [16]. On Llama-3.1-8B, KIVI 4-bit is the most accuracy-stable method tested: HotpotQA 55.99 versus 55.72 for full cache, Multi-news 26.95 versus 27.15 [16]. KIVI 2-bit is close but not free: 54.42 on HotpotQA and 42.94 on 2WikiMQA versus 55.72 and 44.27 [16].

TurboQuant at 3-bit and 4-bit shows the opposite pattern in the same benchmark: it holds up on QA but collapses on summarization, with Llama Multi-news ROUGE at 16.39 and 16.16 against 27.15 for full cache [16]. The authors note the summarization runs were executed once because each takes three to four days [16], so those two cells carry more noise than the rest of the table.

#### 1.4 Production quantization formats

vLLM documents FP8 KV cache through `kv_cache_dtype`, with `auto` as the unquantized default and `fp8_e4m3`/`fp8_e5m2` as opt-ins [11]. Per-attention-head scaling requires the Flash Attention backend plus llm-compressor calibration; a `--kv-cache-dtype-skip-layers` flag keeps sensitive layers such as sliding-window attention at native precision [11]. The vLLM v0.8.2 documentation states that FP8 roughly doubles KV-cache space but gives no latency improvement at that version because fused dequantization and attention were not yet implemented [11].

SGLang supports FP8 E4M3/E5M2 and FP4 (`nvfp4`, `fp4_mx_block16`) via `--kv-cache-dtype`, with per-tensor scaling only [12]. Its documented savings are about 1.78× more tokens than FP8 and 3.56× more tokens than BF16 for FP4 block-16, computed from bit-width ratios including scaling overhead [12]. The accuracy table is the cautionary part: GPT-OSS-120B aime25 falls from 0.7533 at KV16 to 0.3533 at KV4, and gpqa_diamond from 0.5081 to 0.3202, while gsm8k is essentially unchanged [12]. SGLang's own guidance is to prefer `fp8_e4m3` for accuracy and warns that degradation may be worse in long-context scenarios [12].

TensorRT-LLM's older attention documentation lists INT8 and FP8 KV cache via `QuantMode.INT8_KV_CACHE` and `QuantMode.FP8_KV_CACHE`, with per-tensor scaling only in that version [14]. Its current KV Cache System page documents a prototype mixed-precision mode that stores full-attention history in NVFP4 with one FP8 scale per 16 elements while retaining the first and last N tokens in the compute dtype, selected through `dtype` values such as `mixed_nvfp4_bfloat16` [13][17]. That mode requires the TRT-LLM or FlashInfer backend with `use_kv_cache_manager_v2`, and does not support speculative decoding, beam search, sparse attention, cold-page compression, or context parallelism [17].

### 2. Eviction: high headline ratios, fragile at long context

#### 2.1 The three policy shapes

Eviction methods differ mainly in what signal they use to decide which tokens to keep. H2O accumulates attention scores across decoding steps and keeps a mix of "heavy hitters" and recent tokens [2]. StreamingLLM is purely positional: four initial attention-sink tokens plus a rolling window of recent tokens [9]. SnapKV and PyramidKV score tokens from an observation window at the end of the prompt, with SnapKV using per-head voting plus max-pooling and PyramidKV varying the per-layer budget [3][18].

The policy choice matters more than the budget. In H2O's OPT-30B table at a 20% budget, full cache scores 80.09 on PiQA and the recency-only "Local" baseline scores 57.94, while H2O scores 79.22 [2]. Keeping only heavy hitters or only local tokens degrades 2.85%–22.75% versus full cache; combining both recovers full performance [2]. H2O also reports that some 0/1-shot tasks need 30–40% budget to match full cache [2].

StreamingLLM's sink-token result is the cleanest demonstration that pure recency fails: on Llama-2-13B over a 65K-token book, a pure 1024-token window gives perplexity 5158.07, while four sink tokens plus a 1020-token window gives 5.40, close to the sliding-window-with-recomputation oracle at 5.43 [9]. But the same paper shows accuracy on StreamEval falling to zero once the query–answer distance exceeds the cache size, and StreamingLLM underperforming simple truncation on LongBench (NarrativeQA 11.6 versus 18.7) [9].

#### 2.2 What the long-context numbers look like

SnapKV reports roughly 92% average compression at a 1024-token prompt cache and 68% at 4096 on LongBench, where average input is about 13K tokens [3]. At a matched 4096-token budget it beats H2O on most tasks: Mistral-7B HotpotQA 42.32 versus 36.54, 2WikiMQA 27.93 versus 20.60 [3]. On Command-R at 128K with 32× compression, NIAH scores 9.819 versus 9.866 baseline, a 0.5% drop [3].

PyramidKV at a 128-token per-layer budget on LLaMA-3-8B averages 37.25 versus 41.46 for full cache, ahead of SnapKV at 35.50, H2O at 35.37, and StreamingLLM at 32.00 [18]. At 2048 tokens per layer it matches full cache (41.49 versus 41.46) [18]. Its measured KV memory at 8192 tokens is 107M at a 128-token cache (1.6% of full) versus 6848M for full cache [18].

The peer-reviewed evaluations are harsher. Ada-KV reports SnapKV's LongBench average falling from 49.09 to 42.86 when moving from question-aware to question-agnostic scoring at a 2048 budget [10]. LKV, at a single matched 15% retention ratio, reports LongBench 46.73 versus 47.48 for full cache on Llama-3.1-8B, but RULER at 128K of 55.34 versus 62.30 [4]. At the same 15% budget, SnapKV scores 38.05 on LongBench and 12.82 at 128K on RULER; PyramidKV scores 32.12 and 10.53 [4]. The gap between learned and heuristic policies widens as compression increases: LKV holds 43.93 at a 10% budget while SnapKV falls to 34.62 [4].

SCBench adds a second matched-budget lane from 1/32 to 1/2 compression. On Llama-3.1-8B multi-turn, KIVI at 1/8 averages 32.0 against 48.7 for full attention, while SnapKV at 1/32 averages 20.9, PyramidKV 21.0, and StreamingLLM 15.5 [5]. The paper's argument is structural: sub-O(n) memory is "almost infeasible in multi-turn decoding" because the critical KVs vary per query [5].

#### 2.3 Eviction in engines

No engine documentation shows token-level eviction as a shipped feature. TensorRT-LLM's eviction is block-level prioritized LRU: blocks get priority 0–100 with a default of 35, only leaf blocks can be evicted, and `host_cache_size` offloads to CPU memory with a default of 0 [13][17]. Its older attention documentation describes a cyclic rolling-buffer KV cache that overwrites least-recently-used entries beyond `max_attention_window_size`, and a StreamingLLM mode with sink tokens enabled via a `streamingllm` flag [14]. vLLM's prefix caching uses LRU block eviction, which is cache management rather than token eviction.

### 3. Low-rank and latent methods: highest ceiling, narrowest operating point

#### 3.1 Post-hoc low-rank

Eigen Attention performs SVD on per-layer, per-head query/key/value representation matrices from a calibration set and merges the basis into the projection weights, so Keys and Values are stored at reduced dimension [8]. It reports up to 40% KV reduction with average perplexity increases of 0.32, 0.69, and 1.79 and accuracy drops of 1%, 2%, and 3% at 0.8×, 0.7×, and 0.6× compression across OPT, MPT, Llama-2, and Llama-3 models [8]. At 40% compression, Keys compress 54% while Values compress only 26% [8]. Larger models tolerate it better: OPT-66B loses about 1% accuracy at 40% compression, while Llama-3-8B loses 7 points [8].

Palu decomposes K/V projection weights offline and caches the low-dimensional latent states, reconstructing on the fly [7]. At 50% compression on Llama-2-7B, WikiText-2 perplexity is 6.01 versus 5.47 baseline with group-head decomposition at group size 4, and zero-shot average is 62.61 versus 65.05 [7]. Joint decomposition across all heads is more accurate (5.62 perplexity, 64.82 zero-shot) but costs more at reconstruction time [7]. On LongBench, Palu at 30% compression loses under 1% on average (Mistral-7B 42.54 to 41.55), but at 50% it falls to 36.23 [7].

#### 3.2 Latent caching in pretrained architectures

DeepSeek-V2's Multi-head Latent Attention caches a 512-dimension latent vector plus a 64-dimension decoupled RoPE key per head instead of full per-head K/V, reducing KV cache 93.3% versus DeepSeek 67B [6]. In the paper's large-MoE ablation, MLA uses about 4% of MHA's per-token KV elements (34.6K versus 860.2K) while matching or beating MHA on BBH (50.7 versus 46.6), MMLU (59.0 versus 57.5), C-Eval (59.2 versus 57.9), and CMMLU (62.5 versus 60.7) [6]. The deployed model additionally applies 6-bit-average KV quantization [6]. This is an architecture decision made at pretraining, not a post-hoc compression method, and the paper reports no matched-ratio comparison against eviction or quantization alone.

MHA2MLA attempts to retrofit MLA onto existing models via partial-RoPE removal plus joint SVD, with fine-tuning on 0.6%–1% of pretraining tokens [19]. It reports 92.19% KV reduction on Llama2-7B with about a 1% LongBench drop, and 96.87% when combined with Int4 HQQ [19]. At roughly 87.5% compression it beats Int2 quantization on LongBench (25.1 versus 21.2 for HQQ and 18.5 for Quanto) [19]. The fine-tuning requirement is the catch: this is not a drop-in serving flag.

#### 3.3 The contested head-to-head

A 2026 preprint claimed that per-channel INT4 quantization dominates low-rank projection at matched storage — for example, Mistral 7B at 512 bits giving 9.42 perplexity for INT4 versus 44.19 for rank-32 fp16 [20]. Its own v2 retracts the matched-storage comparison, stating that the compared arms are weight-perturbation interventions that reconstruct dense floating-point weights and do not implement packed cache storage [20]. The matched-ratio claim is therefore not validated, and the low-rank-versus-quantization question has to be answered from each method's own tables.

Those tables point in a consistent direction. Palu's own comparison shows KVQuant at 3-bit reaching 5.35 perplexity at 81.25% compression while Palu-50% in fp16 reaches 5.63 at 50% compression [7]. Eigen Attention's matched-ratio table shows quantized standard attention beating quantized Eigen at large cache sizes but losing as precision drops [8]. The pattern is that low-rank and quantization are complementary: low-rank removes dimensions, quantization reduces the precision of what remains, and stacking them reaches the highest compression at acceptable accuracy.

### 4. The trade-off curve across families

No single study benchmarks all three families at matched budgets on the same long-context suite. The closest is Agrawal and Mayer's benchmark, which covers quantization, eviction, and merging but explicitly excludes low-rank methods as not drop-in [16]. SCBench covers quantization and eviction but no low-rank [5]. The cross-family curve therefore has to be stitched from each paper's own comparisons, and should be read as indicative rather than measured.

| Family | Representative method | Reported compression | Reported accuracy cost | Production status |
|---|---|---|---|---|
| Quantization | KVQuant 3-bit | 4.8× [1] | <0.1 PPL; LongBench 31.21 vs 31.96 [1] | FP8 shipped; INT4 research [11][12] |
| Quantization | KIVI 4-bit | 3.15–3.19× [16] | HotpotQA 55.99 vs 55.72 [16] | Research; FP8 is the shipped analogue |
| Quantization | SGLang FP4 | ~3.56× vs BF16 [12] | aime25 0.3533 vs 0.7533 [12] | Opt-in flag [12] |
| Eviction | H2O 20% budget | ~5× [2] | Matches full cache on HELM [2] | Not shipped as token eviction |
| Eviction | SnapKV 1024-token cache | ~92% on LongBench [3] | RULER 128K 12.82 vs 62.30 | Not shipped |
| Eviction | LKV 15% retention | 6.6× at 200K [4] | LongBench 46.73 vs 47.48 [4] | Not shipped |
| Low-rank | Eigen Attention 0.6× | 40% reduction [8] | ~3% accuracy drop [8] | Not shipped |
| Low-rank | Palu 50% | 2× [7] | PPL 6.01 vs 5.47 [7] | Not shipped |
| Low-rank | Palu-30% + 3-bit | 86.87% [7] | PPL 5.33 vs 5.12 [7] | Not shipped |
| Latent | DeepSeek-V2 MLA | 93.3% [6] | Matches or beats MHA [6] | Only for MLA-native checkpoints |

The shape of the curve that emerges: at moderate compression (roughly 2–4×), quantization and eviction are both close to lossless, and quantization is the safer choice because its failure mode is graceful. At high compression (8× and above), eviction's accuracy collapses on long-context and multi-turn workloads [4][5], while quantization at 2-bit degrades but does not collapse [1]. Low-rank alone sits in the middle and is dominated by quantization at matched memory in Palu's own table [7]; low-rank stacked with quantization reaches the highest compression at modest cost [7][19].

### 5. Practical constraints

**Kernel support and latency.** vLLM's v0.8.2 documentation states that FP8 KV cache gave no latency improvement at that version because fused dequantization and attention were not implemented [11]. SGLang's FP4 prefill path requires SM100 hardware, a page size divisible by 4, and an attention head dimension divisible by 64 [12]. TensorRT-LLM's mixed-precision NVFP4 mode gathers two pools into a staging copy each forward, with per-step cost proportional to the longest sequence in the batch [17].

**Prefix caching interaction.** Quantized KV interacts badly with prefix caching in at least one documented case: a vLLM issue reports `--kv-cache-dtype fp8` raising attention block size from 800 to 1600 on a hybrid Mamba model and collapsing prefix-cache hit rate from 31.7% to about 10%. That is a single user's community report, not official documentation, and should be treated as a lead rather than a confirmed engine property. SGLang's 2026 H2 quantization roadmap lists open tasks for FP4 KV prefix-cache correctness and an NVFP4+FP8 KV corruption blocker [21].

**Eviction and paged attention.** Eviction methods that drop tokens mid-sequence conflict with paged attention's fixed block allocation, which is one reason engines implement block-level LRU rather than token-level eviction [13]. TensorRT-LLM's cyclic KV cache overwrites least-recently-used entries but does not work with beam search [14].

**Low-rank and serving.** Palu requires a custom Triton kernel for RoPE attention because reconstruction must happen online [7]. MHA2MLA requires fine-tuning on 0.6%–1% of pretraining tokens [19]. Neither is a configuration flag.

## Risks & Counterpoints

The evidence for quantization is the strongest of the three families, but it is not uniform. KVQuant's byte-level savings are estimated from bit-width assumptions rather than measured on hardware [1], and its long-context numbers come from a single model family (LLaMA-2-7B-32K) [1]. The independent benchmark that measures realized compression covers only 7–8B models on a single A100 40GB [16], so the ratios may not transfer to larger models or different hardware.

The eviction literature has a systematic evaluation problem. H2O and StreamingLLM were evaluated mostly on short-context HELM and lm-eval tasks [2][9], and the long-context numbers that do exist come from later papers evaluating them as baselines. SnapKV's strongest claims come from its own paper [3]; the independent re-implementation that found a 20% budget to be a "sweet spot" tested only 2–3.5K-token contexts on a single task [22], which is far short of the 32K+ regime SnapKV targets.

The low-rank evidence is thinner and partly contested. The one preprint that claimed quantization dominates rank reduction retracted its matched-storage comparison in v2 [20]. Palu and Eigen Attention each compare against their own chosen baselines, and neither reports RULER numbers at 32K–128K, so low-rank behavior at extreme context is unmeasured relative to eviction's documented collapse. DeepSeek-V2's MLA results are the authors' own architecture evaluated by its developers [6], and the compression is not available to models that were not pretrained with it.

Finally, the production picture is narrower than the research picture. FP8 KV cache is shipped and documented, but the accuracy evidence for it in engine documentation is qualitative — vLLM's page says FP8 E4M3 "typically only minimally degrades inference accuracy" without numbers [11]. The concrete accuracy tables come from SGLang's FP4 documentation [12], which shows that aggressive formats can halve reasoning scores on hard benchmarks.

## Open Questions

**A true three-family matched-budget study.** The dominance question cannot be settled without a single independent benchmark that runs quantization, eviction, and low-rank methods at the same KV budgets on the same LongBench or RULER suite. The two closest benchmarks each exclude low-rank [16][5], so the cross-family curve in this report is stitched from per-paper comparisons.

**Low-rank behavior at extreme context.** Palu and Eigen Attention report LongBench or short-context perplexity but no RULER results at 32K–128K [7][8]. Whether low-rank degrades gracefully like quantization or collapses like eviction at long context is unknown.

**Quantization and prefix caching.** No official vLLM or SGLang documentation states whether FP8 or FP4 quantized KV blocks are hashable and reusable in prefix caching, or what the measured throughput overhead is. The available signals are a community issue report and a roadmap listing open correctness tasks [21].

**llama.cpp KV-cache quantization.** No retrieved page documents llama.cpp's `--cache-type-k`/`--cache-type-v` flags or Q8_0/Q4_0 KV formats, so this report cannot confirm what that engine ships.

## References

- **[1] arxiv.org** - "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization". [Source](https://arxiv.org/html/2401.18079v4) (accessed 2026-10-28)
- **[2] arxiv.org** - "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models". [Source](https://arxiv.org/pdf/2306.14048) (accessed 2026-10-28)
- **[3] arxiv.org** - "SnapKV: LLM Knows What You are Looking for Before Generation". [Source](https://arxiv.org/html/2404.14469v2) (accessed 2026-10-28)
- **[4] aclanthology.org** - "LKV: End-to-End Learning of Head-wise Budgets and Token Selection for LLM KV Cache Eviction". [Source](https://aclanthology.org/2026.emnlp-main.1183.pdf) (accessed 2026-10-28)
- **[5] arxiv.org** - "SCBENCH: A KV CACHE-CENTRIC ANALYSIS OF LONG-CONTEXT METHODS". [Source](https://arxiv.org/pdf/2412.10319) (accessed 2026-10-28)
- **[6] arxiv.org** - "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model". [Source](https://arxiv.org/html/2405.04434v2) (accessed 2026-10-28)
- **[7] arxiv.org** - "Palu: Compressing KV-Cache with Low-Rank Projection". [Source](https://arxiv.org/html/2407.21118v1) (accessed 2026-10-28)
- **[8] aclanthology.org** - "Eigen Attention: Attention in Low-Rank Space for KV Cache Compression". [Source](https://aclanthology.org/2024.findings-emnlp.899.pdf) (accessed 2026-10-28)
- **[9] arxiv.org** - "Efficient Streaming Language Models with Attention Sinks". [Source](https://arxiv.org/html/2309.17453v4) (accessed 2026-10-28)
- **[10] papers.nips.cc** - "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference". [Source](https://papers.nips.cc/paper_files/paper/2025/file/a40ff56daab9f4808b1e18350c8a11ce-Paper-Conference.pdf) (accessed 2026-10-28)
- **[11] docs.vllm.ai** - "Quantized KV Cache". [Source](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) (accessed 2026-10-28)
- **[12] SGLang Documentation** - "Quantized KV Cache". [Source](https://docs.sglang.io/docs/advanced_features/quantized_kv_cache) (accessed 2026-10-28)
- **[13] nvidia.github.io** - "KV Cache System". [Source](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html) (accessed 2026-10-28)
- **[14] nvidia.github.io** - "Multi-Head, Multi-Query, and Group-Query Attention". [Source](https://nvidia.github.io/TensorRT-LLM/1.0.0rc5/advanced/gpt-attention.html) (accessed 2026-10-28)
- **[15] proceedings.mlsys.org** - "Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving". [Source](https://proceedings.mlsys.org/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf) (accessed 2026-10-28)
- **[16] arxiv.org** - "Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving". [Source](https://arxiv.org/pdf/2607.05399) (accessed 2026-10-28)
- **[17] GitHub** - "TensorRT-LLM/docs/source/features/kvcache.md at main · NVIDIA/TensorRT-LLM". [Source](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md) (accessed 2026-10-28)
- **[18] arxiv.org** - "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling". [Source](https://arxiv.org/html/2406.02069v2) (accessed 2026-10-28)
- **[19] arxiv.org** - "Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs". [Source](https://arxiv.org/abs/2502.14837v2) (accessed 2026-10-28)
- **[20] arxiv.org** - "Quantization Dominates Rank Reduction for KV-Cache Compression". [Source](https://arxiv.org/pdf/2604.11501v1) (accessed 2026-10-28)
- **[21] GitHub** - "[Roadmap] Quantization 2026 H2 · Issue #31783 · sgl-project/sglang". [Source](https://github.com/sgl-project/sglang/issues/31783) (accessed 2026-10-28)
- **[22] GitHub** - "GitHub - AayushSalvi/SnapKV-analysis: DSC 291 Efficient AI - Improving KV Cache Compression Techniques". [Source](https://github.com/AayushSalvi/SnapKV-analysis) (accessed 2026-10-28)