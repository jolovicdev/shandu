# KV-Cache Compression for Long-Context Inference: Quantization, Eviction, and Low-Rank Methods Compared

**Quantization is the only family that ships in production engines, and it is also the family with the best-documented accuracy-per-byte at moderate compression.** Eviction buys larger nominal compression ratios but degrades sharply on long-context and multi-turn workloads, and its strongest results depend on seeing the question before compressing. Low-rank and latent methods reach the highest compression ratios, but only when stacked on top of quantization, and they require calibration or fine-tuning that no serving engine exposes as a user-facing feature. No single study benchmarks all three families at matched budgets on one long-context suite, so the cross-family ranking below is stitched from each paper's own comparisons and should be read as directional.

## The three families side by side

| | Quantization | Eviction / token dropping | Low-rank / latent |
|---|---|---|---|
| **Mechanism** | Store K/V at lower bit-width; dequantize in the attention kernel | Keep a subset of tokens (by attention score, recency, or a learned policy); discard the rest | Project K/V into a smaller latent space; reconstruct on the fly or attend in latent space |
| **Reported memory savings** | 3.7× at 4-bit, 4.8× at 3-bit, 6.9× at 2-bit [1]; realized 3.2–5.3× in an independent benchmark [2] | 4× at a 0.75 keep-ratio [2]; 6.6× at 15% retention [3]; 92% at a 1,024-token cache on ~13K-token inputs [4] | 40% reduction (0.6×) [5]; 50% [6]; 93.3% for pretrained MLA [7]; 92–97% when stacked with quantization [8][6] |
| **Reported accuracy loss** | <0.02 / <0.1 / <0.5 WikiText-2 PPL at 4/3/2-bit [1]; near-parity on LongBench at 4-bit [2] | Negligible at moderate budgets on LongBench [4][9]; collapses beyond 32K on RULER [3]; ~15–36% ROUGE loss on summarization [2] | +0.32 to +1.79 PPL at 0.8×/0.7×/0.6× [5]; ~1% LongBench drop at 92% reduction with fine-tuning [8] |
| **Production adoption** | Shipped, opt-in: FP8 in vLLM, SGLang, TGI, TensorRT-LLM; FP4 experimental in SGLang and TensorRT-LLM [10][11][12] | Not shipped as token eviction; engines ship block-level LRU and cyclic/StreamingLLM windows [12][13] | Not shipped for general models; MLA latent caching only for MLA-native checkpoints [7] |

## Quantization: the best-characterized family

### Memory savings and granularity

KVQuant is the most complete published accounting. It reports **3.7×, 4.8×, and 6.9× KV-cache reduction at 4-, 3-, and 2-bit** respectively, with the bit-widths being averages that include scaling and outlier overhead (nuq4-1% ≈ 4.32–4.35 bits, nuq3-1% ≈ 3.32–3.35 bits, nuq2-1% ≈ 2.32–2.35 bits) [1]. At 128K sequence length, LLaMA-7B's cache falls from 64.0 GB in fp16 to 13.3 GB at nuq3-1% and 9.3 GB at nuq2-1%; the paper's context-length projections are estimates from bit-width assumptions, not measured allocations [1].

Granularity choices carry most of the accuracy. On LLaMA-7B at 3-bit, quantizing keys per-channel before RoPE and values per-token improves perplexity by **3.82 points** over per-token quantization of both; moving key quantization before RoPE adds 0.82; keeping 1% of per-vector outliers in fp16 adds 0.19 [1]. Atom takes a different route, quantizing asymmetrically at **per-attention-head** granularity and reporting only a 0.12 WikiText-2 perplexity increase on Llama-7B (6.04 → 6.16) on top of its W4A4 weight/activation scheme [14]. Atom does not report a KV-specific compression ratio or any long-context benchmark [14].

### Accuracy at each bit-width

The degradation curve is steep at the bottom. KVQuant's own baselines show uniform INT3 at 10.87 PPL and INT2 at 11,779 PPL on LLaMA-7B against a 5.68 fp16 baseline, versus 5.75 and 6.01 for KVQuant-3bit-1% and -2bit-1% [1]. On long-context suites, KVQuant-3bit-1% scores 31.21 vs 31.96 fp16 on LongBench and 53.65 vs 56.40 on RULER for LLaMA-2-7B-32K, while 2-bit falls to 36.54 on RULER [1].

An independent benchmark by Agrawal and Mayer, run on a single A100 40GB with Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3, finds **KIVI 4-bit the most accuracy-stable method across models and workloads** (Llama Multi-news 26.95 vs 27.15 all-KV; HotpotQA 55.99 vs 55.72), with realized compression of 3.15–3.19× [2]. KIVI 2-bit reaches 5.16–5.32× compression with small but visible losses [2]. The same benchmark shows aggressive quantization is not uniformly safe: TurboQuant at an effective 3.25 bits collapses Llama Multi-news ROUGE to 16.39 from 27.15 and drops TriviaQA by roughly 7 points [2]. That is a preprint, not yet peer-reviewed, and its summarization runs for TurboQuant were executed once [2].

## Eviction: large nominal ratios, fragile under long context

### Attention-score policies

H2O keeps a mix of accumulated-attention "heavy hitters" and recent tokens. At a **20% KV budget it matches full-cache accuracy** on OPT-30B across HELM and lm-eval tasks (PiQA 79.22 vs 80.09, COPA 85.00 vs 81.00, Winogrande 71.67 vs 71.51), while a recency-only baseline collapses (PiQA 57.94) [15]. Keeping only heavy hitters or only local tokens degrades 2.85–22.75%, and some 0/1-shot tasks need 30–40% budget [15]. These evaluations are short-context classification and QA tasks, not LongBench or RULER.

SnapKV scores tokens from an observation window at the end of the prompt. On LongBench with ~13K-token average inputs it reports **~92% compression at a 1,024-token cache and ~68% at 4,096, "with negligible drops in accuracy"**, and at a matched 4,096-token budget it beats H2O on most tasks (Mistral HotpotQA 42.32 vs 36.54; 2WikiMQA 27.93 vs 20.6) [4]. On Command-R at 128K with 32× compression it scores 9.819 vs 9.866 on NIAH, a 0.5% drop [4]. PyramidKV, which varies the per-layer budget, averages 37.25 vs 41.46 full cache at a 128-token per-layer budget on LLaMA-3-8B, ahead of SnapKV (35.50), H2O (35.37) and StreamingLLM (32.00); at 2,048 tokens it matches full cache (41.49 vs 41.46) [9].

### Recency policies

StreamingLLM keeps 4 attention-sink tokens plus a rolling window. It restores Llama-2-13B PG-19 perplexity from **5158.07 under a pure window to 5.40**, close to the sliding-window-with-recomputation oracle at 5.43 [16]. Its failure mode is structural: accuracy on StreamEval falls to zero once the query–answer distance exceeds the cache size, and it underperforms simple truncation on LongBench (NarrativeQA 11.6 vs 18.7) [16].

### What peer-reviewed matched-budget work shows

Ada-KV (NeurIPS 2025) quantifies how much eviction depends on seeing the question. SnapKV's LongBench average drops from **49.09 question-aware to 42.86 question-agnostic** at a 2,048-token budget, and its question-agnostic RULER average at 20% budget is 44.02, rising to 53.29 with adaptive head-wise budgeting [17]. LKV (EMNLP 2026) at 15% retention scores 46.73 vs 47.48 full cache on LongBench for Llama-3.1-8B, but **55.34 vs 62.30 at 128K on RULER**, while heuristic eviction collapses outright at that length (SnapKV 12.82, PyramidKV 10.53) [3]. LKV also measures the memory side directly: at 200K context, KV storage falls from 25.0 GB to 3.75 GB at 15% budget, a 6.6× reduction [3].

SCBench (ICLR 2025) adds a second matched-budget lane across compression rates from 1/32 to 1/2. On Llama-3.1-8B multi-turn, KIVI at 1/8 averages 32.0 against full attention's 48.7, while SnapKV 20.9, PyramidKV 21.0 and StreamingLLM 15.5 at 1/32 [18]. The paper's conclusion is that sub-linear-memory methods are "almost infeasible in multi-turn decoding" because the critical KVs change with each query [18].

## Low-rank and latent methods: highest ratios, heaviest prerequisites

### Post-training low-rank projection

Palu SVD-decomposes the K/V projection weights and caches the latent states. At **50% compression** on Llama-2-7B it reports WikiText-2 perplexity 5.47 → 6.01 (group-head, group size 4) and zero-shot average 65.05 → 62.61; the joint-head variant is much better (5.62 PPL, 64.82 average) but costs more to reconstruct [6]. On LongBench, 30% compression costs under 1% (Mistral-7B 42.54 → 41.55), but 50% falls to 36.23 [6]. Combined with 2-bit per-token quantization, Palu reaches **91.25% compression (11.4×) at 5.76 PPL**, against KVQuant 2-bit at 6.95 PPL and 87.5% compression [6].

Palu's own tables also show low-rank alone is not uniformly better than quantization. At 3-bit, KVQuant (5.35 PPL, 81.25% compression) beats Palu-50% in fp16 (5.63 PPL, 50% compression) on accuracy per byte, while Palu-30%+3-bit (5.33 PPL, 86.87%) is roughly tied with KVQuant-3bit at higher compression [6]. A separate preprint by Salfati claimed quantization dominates rank reduction at matched storage, but its own v2 retracts the matched-storage comparison, stating the compared arms are weight-perturbation interventions that do not implement packed cache storage [19]. That head-to-head claim is therefore unvalidated in either direction.

### Eigenbasis projection

Eigen Attention (Findings of EMNLP 2024) merges SVD-derived per-head bases into the projection weights and reports **up to 40% KV reduction** with average perplexity increases of 0.32 / 0.69 / 1.79 and accuracy drops of 1% / 2% / 3% at 0.8× / 0.7× / 0.6× across nine models [5]. At 40% compression keys compress 54% while values compress only 26%, and larger models tolerate it better (OPT-66B loses 1% accuracy) [5]. Its own matched-ratio comparisons put eviction ahead at moderate compression (H2O at 0.4× roughly equals Eigen at 0.6×: Llama-2-7B 0.64 vs 0.63) and show quantized standard attention beating quantized Eigen at large cache sizes but losing as precision drops; combining Eigen with H2O reaches 0.24× cache [5].

### Latent caching (MLA)

DeepSeek-V2's Multi-head Latent Attention caches a 512-dimension latent plus a 64-dimension decoupled RoPE key, **cutting KV cache 93.3%** versus DeepSeek 67B and using about 4% of MHA's per-token KV elements in the large-MoE ablation (34.6K vs 860.2K) while matching or beating MHA on BBH, MMLU, C-Eval and CMMLU [7]. The deployed model adds 6-bit-average KV quantization on top [7]. This is a pretrained architecture, not a post-hoc compression, and the paper reports no matched-ratio comparison against eviction or quantization-only [7]. Post-hoc conversion is possible but costly: MHA2MLA reports 92.19% reduction on Llama2-7B with a ~1% LongBench drop, but requires full-parameter fine-tuning on 0.6–1% of pretraining tokens [8].

## The trade-off curve, and where each family wins

The honest summary is that **no single technique dominates across all settings**, which is the conclusion of a March 2026 survey that compiles per-paper numbers without running a matched experiment [20]. What the stitched evidence supports:

- **Moderate compression (roughly 2–4×), accuracy-critical:** quantization wins. KIVI 4-bit is the most stable method in the only independent multi-method LongBench benchmark [2], and KVQuant-3bit sits within 0.1 PPL of fp16 [1]. Eviction at comparable budgets is competitive on question-aware LongBench [4][9] but loses once the question is hidden [17].
- **High compression (8× and above), single long request:** eviction and low-rank both reach ratios quantization cannot, but with different failure modes. Eviction's accuracy collapses beyond 32K on RULER [3]; low-rank at 50% costs several LongBench points [6], and its best numbers come from stacking with quantization [8][6].
- **Multi-turn and shared-prefix serving:** fixed-size quantization degrades gracefully while eviction degrades badly, because the tokens that matter change per query [18].
- **Extreme compression with acceptable quality:** the strongest published combination is low-rank plus quantization — Palu-30%+2-bit at 91.25% compression and 5.76 PPL [6], or MHA2MLA+Int4 at 96.87% reduction with a 2.4-point LongBench drop [8].

A caveat on the curve: the two matched-budget benchmarks that cover multiple families both exclude low-rank methods, one because they are not drop-in [2] and one by scope [18]. Low-rank's position relative to the other two at identical byte budgets therefore rests on each low-rank paper's own chosen baselines.

## What production engines actually ship

Quantization is universal and opt-in; nothing else is.

| Engine | KV quantization | Flag / format | Eviction or low-rank |
|---|---|---|---|
| vLLM | FP8 only | `--kv-cache-dtype` = `auto`/`fp8`/`fp8_e4m3`/`fp8_e5m2`; per-tensor or per-head with Flash Attention plus llm-compressor calibration [10] | None; only LRU block eviction in prefix caching |
| SGLang | FP8 and experimental FP4 | `--kv-cache-dtype` with `nvfp4`, `fp4_mx_block16`; per-tensor scaling only [11] | None documented |
| TensorRT-LLM | FP8/INT8 per-tensor; prototype NVFP4 mixed precision | `KvCacheConfig.dtype`; `mixed_nvfp4_bfloat16`/`_fp8`/`_float16` retain first/last N tokens in compute dtype [12][13] | Prioritized LRU block eviction (priority 0–100, default 35), host offload, deprecated StreamingLLM sink tokens [12][13] |
| TGI | FP8 | `--kv-cache-dtype` = `fp8_e4m3fn`/`fp8_e5m2`, defaulting to model dtype | LRU of cached prefixes only |

SGLang's own documentation gives the clearest accuracy picture for FP4: GPT-OSS-120B scores 0.7533 on aime25 at KV16 versus **0.3533 at KV4**, while gsm8k stays flat, and the docs warn degradation may be worse in long-context scenarios [11]. Documented memory savings are ~3.56× more tokens than BF16 for FP4 and ~1.78× more than FP8 [11].

No engine documents low-rank or latent KV compression as a user-facing feature. MLA latent caching is supported only for MLA-native DeepSeek-family checkpoints; post-hoc conversion remains research code [7][8].

## Practical constraints

**Kernel support and latency.** vLLM's FP8 KV cache historically gave no latency improvement because fused dequantization plus attention was not implemented [10]. TensorRT-LLM's prototype mixed-precision mode pays a per-step cost: each forward gathers two pools into a staging copy sized to the longest sequence in the batch [12]. On the eviction side, SnapKV has the best throughput in the independent benchmark, on par with or above all-KV, while TurboQuant incurs the highest time-to-first-token (1351 ms vs 1125 ms on Llama Qasper 8K+) [2].

**Paged attention and prefix caching.** Quantized KV interacts badly with prefix caching in documented cases. A vLLM issue reports `--kv-cache-dtype fp8` raising attention block size from 800 to 1600 on a hybrid Mamba model and collapsing prefix-cache hit rate from 31.7% to about 10% — community evidence from a single user, not an official statement. SGLang's 2026 H2 roadmap lists open tasks for FP4 KV prefix-cache correctness and an NVFP4+FP8 KV corruption blocker. These are leads rather than settled findings; no official engine page states the interaction directly.

**Prerequisites.** Low-rank methods need calibration data and, for the best results, fine-tuning [8][6][5]. Eviction needs an observation window or question access to score tokens well, and its accuracy depends on that access [17].

## Which family fits which regime

- **Default production serving, 2–4× memory relief, minimal risk:** FP8 KV cache. It is the only option with vendor support across four engines, and 4-bit-class quantization is the most accuracy-stable family in independent testing [2].
- **Very long single requests where some quality loss is acceptable:** attention-score eviction at moderate budgets, accepting that accuracy falls off beyond 32K and on summarization [3][2].
- **Maximum compression with a calibration budget:** low-rank stacked on quantization — Palu-30%+2-bit or an MLA conversion — not low-rank alone [6][8].
- **Multi-turn or shared-prefix workloads:** avoid token eviction; fixed-size quantization degrades far more gracefully [18].

## Evidence gaps

Three limits bound these conclusions. First, no study benchmarks quantization, eviction and low-rank at matched budgets on one long-context suite; the two multi-family benchmarks exclude low-rank [2][18]. Second, low-rank methods have no published RULER results at 32K–128K, so their behavior at extreme context is unmeasured against eviction's documented collapse. Third, the interaction between quantized KV and prefix caching is documented only in issue threads and roadmaps, not in official engine documentation.

## References

- **[1] arxiv.org** - "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization". [Source](https://arxiv.org/html/2401.18079v4) (accessed 2026-10-28)
- **[2] arxiv.org** - "Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving". [Source](https://arxiv.org/pdf/2607.05399) (accessed 2026-10-28)
- **[3] aclanthology.org** - "LKV: End-to-End Learning of Head-wise Budgets and Token Selection for LLM KV Cache Eviction". [Source](https://aclanthology.org/2026.emnlp-main.1183.pdf) (accessed 2026-10-28)
- **[4] arxiv.org** - "SnapKV: LLM Knows What You are Looking for Before Generation". [Source](https://arxiv.org/html/2404.14469v2) (accessed 2026-10-28)
- **[5] aclanthology.org** - "Eigen Attention: Attention in Low-Rank Space for KV Cache Compression". [Source](https://aclanthology.org/2024.findings-emnlp.899.pdf) (accessed 2026-10-28)
- **[6] arxiv.org** - "Palu: Compressing KV-Cache with Low-Rank Projection". [Source](https://arxiv.org/html/2407.21118v1) (accessed 2026-10-28)
- **[7] arxiv.org** - "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model". [Source](https://arxiv.org/html/2405.04434v2) (accessed 2026-10-28)
- **[8] arxiv.org** - "Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs". [Source](https://arxiv.org/abs/2502.14837v2) (accessed 2026-10-28)
- **[9] arxiv.org** - "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling". [Source](https://arxiv.org/html/2406.02069v2) (accessed 2026-10-28)
- **[10] docs.vllm.ai** - "Quantized KV Cache". [Source](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) (accessed 2026-10-28)
- **[11] SGLang Documentation** - "Quantized KV Cache". [Source](https://docs.sglang.io/docs/advanced_features/quantized_kv_cache) (accessed 2026-10-28)
- **[12] nvidia.github.io** - "KV Cache System". [Source](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html) (accessed 2026-10-28)
- **[13] nvidia.github.io** - "Multi-Head, Multi-Query, and Group-Query Attention". [Source](https://nvidia.github.io/TensorRT-LLM/1.0.0rc5/advanced/gpt-attention.html) (accessed 2026-10-28)
- **[14] proceedings.mlsys.org** - "Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving". [Source](https://proceedings.mlsys.org/paper_files/paper/2024/file/5edb57c05c81d04beb716ef1d542fe9e-Paper-Conference.pdf) (accessed 2026-10-28)
- **[15] arxiv.org** - "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models". [Source](https://arxiv.org/pdf/2306.14048) (accessed 2026-10-28)
- **[16] arxiv.org** - "Efficient Streaming Language Models with Attention Sinks". [Source](https://arxiv.org/html/2309.17453v4) (accessed 2026-10-28)
- **[17] papers.nips.cc** - "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference". [Source](https://papers.nips.cc/paper_files/paper/2025/file/a40ff56daab9f4808b1e18350c8a11ce-Paper-Conference.pdf) (accessed 2026-10-28)
- **[18] arxiv.org** - "SCBENCH: A KV CACHE-CENTRIC ANALYSIS OF LONG-CONTEXT METHODS". [Source](https://arxiv.org/pdf/2412.10319) (accessed 2026-10-28)
- **[19] arxiv.org** - "Quantization Dominates Rank Reduction for KV-Cache Compression". [Source](https://arxiv.org/pdf/2604.11501v1) (accessed 2026-10-28)
- **[20] arxiv.org** - "KV Cache Optimization Strategies for Scalable and Efficient LLM Inference". [Source](https://arxiv.org/pdf/2603.20397) (accessed 2026-10-28)