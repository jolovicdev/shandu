# KV-Cache Compression for Long-Context LLM Inference: Quantization, Eviction, and Low-Rank Methods Compared

## Executive Summary

The KV cache is the dominant memory cost of long-context serving, and three families of methods attack it. The evidence points to a clear production answer: **quantization is the only family that ships in mainstream inference engines, and FP8 is the default operating point.** vLLM, SGLang, and TensorRT-LLM all expose KV-cache quantization as an opt-in flag, with FP8 as the recommended format and 4-bit formats (NVFP4, MXFP4, INT4) arriving as hardware-gated options [1][2][3][4]. Eviction appears in production only as StreamingLLM-style sink-token and rolling-window attention, not as the H2O/SnapKV/PyramidKV policies that dominate the academic literature [3]. Low-rank and latent methods (Palu, CommonKV, MHA2MLA, X-EcoMLA) achieve the largest compression ratios but require fine-tuning or distillation for the best results and do not appear in any engine's documented feature set [5][6][7].

On accuracy, the families separate cleanly at matched budgets. 4-bit quantization is near-lossless across peer-reviewed papers — KVQuant-4bit-1% stays within 0.02 perplexity of FP16 on WikiText-2 across LLaMA-7B through 65B [8], and KIVI-4 matches 16-bit on LongBench (44.59 vs 44.52) [9]. 3-bit is viable only with outlier isolation [8][10]. 2-bit is method-dependent, ranging from near-baseline (RotateKV, KVQuant-1%) to catastrophic collapse (uniform INT2, QuaRot-2bit) [11][10]. Eviction degrades gracefully at light budgets but collapses at aggressive ones, and its reported numbers are the most contested in the literature [12][13][14]. Low-rank methods sit between: Palu reaches 50% compression at 5.63 vs 5.12 perplexity on Llama2-7B [5], while MLA-style projection reaches 92% compression at a ~1% LongBench drop but only after fine-tuning [6].

The bottom line for a practitioner: **FP8 KV cache is production-ready today; 4-bit KV is production-ready on specific hardware with accuracy caveats; eviction and low-rank are research-grade for general serving, with eviction viable only as a fixed-window/sink mechanism and low-rank viable only if you control the model weights.**

## Key Findings

1. **The baseline is brutal and linear.** KV cache bytes scale as 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_element [8][9]. For LLaMA-7B at 128K tokens in FP16, the KV cache is 64 GB against 12.6 GB of weights [8]. An 8B model at 128K needs roughly 16 GiB, "on par with the weights themselves" [13].

2. **GQA is the pre-compression lever that already happened.** Eight KV heads against 64 query heads gives an 8× reduction before any compression method runs. PagedAttention does not reduce per-token cost — it cuts fragmentation, lifting KV memory utilization from 20.4–38.2% to 96.3% at a 20–26% attention-kernel latency cost [15].

3. **4-bit quantization is the safe operating point.** KVQuant-4bit-1% is within 0.02 PPL of FP16 across LLaMA-7B–65B at 3.7× savings [8]; KIVI-4 matches 16-bit on LongBench [9]; RotateKV-4bit shows a 0.01 PPL delta on LLaMA-2-13B [10]; Coupled Quantization's CQ-2c8b gives 5.70 vs 5.68 FP16 [11].

4. **3-bit needs outlier handling; 2-bit is a coin flip.** KVQuant-3bit-1% stays under 0.1 PPL at 4.8× savings [8], but without outlier isolation KVQuant-3b collapses to 11.21 PPL on LLaMA-2-7B and 12,323 on PTB in an independent re-run [16]. At 2-bit, RotateKV reaches 4.84 vs 4.57 on LLaMA-2-13B [10] while QuaRot-2bit hits 21.43 vs 5.75 on LLaMA-3-8B [10].

5. **Eviction's headline numbers do not survive independent re-runs.** H2O's own paper reports 20% budget ≈ full-cache accuracy (COPA 84.00 vs 85.00) at up to 5× reduction [12]. An independent 2026 re-run on Llama-3.1-8B LongBench at B=128 puts H2O at 38.10, SnapKV at 42.22, and PyramidKV at 41.64 against FullKV 46.24 [13]. PyramidKV's claimed parity at 2048 entries (41.49 vs 41.46) is contradicted by LKV's 32.12 vs 47.48 at 15% retention [17][14].

6. **Low-rank methods reach the highest ratios but need training.** Palu (post-training SVD on projection weights) gives 50% compression at 5.63 vs 5.12 PPL and 1.61× attention speedup [5]. MHA2MLA cuts Llama2-7B KV by 92.19% with a ~1% LongBench drop, but requires fine-tuning on 0.6–1% of pretraining tokens [6]. X-EcoMLA reaches 6.4× with no average-score loss but needs distillation plus DPO [18].

7. **Production engines ship quantization only.** vLLM documents FP8 e4m3/e5m2 with per-tensor or per-head scaling [1]; SGLang adds experimental FP4 [2]; TensorRT-LLM supports INT8 and FP8 per-tensor only, plus StreamingLLM sink tokens and a cyclic rolling buffer [3]. No supplied evidence shows H2O, SnapKV, PyramidKV, KIVI, KVQuant, or Palu shipping as a default, flag, or plugin in any of these engines.

8. **Kernel economics explain the adoption pattern.** FP8 runs inside the attention kernel: vLLM measures Llama-3.1-8B on H100 at 54% of the BF16 ITL slope and +14.9% throughput at concurrency 8 [19]. 4-bit KV dequantization is compute-bound on A100 — QServe measures naive KV4 at 0.88× of KV8, reaching 1.49× only after hand-optimization [20] — and Hopper has no INT4 tensor-core MMA [21].

9. **Engine-level accuracy for FP8 is small but not zero.** vLLM reports ≤1–2 points degradation on reasoning benchmarks and 97–98% AUC@128k recovery with uncalibrated scales [19]. SGLang's own tables show FP4 KV collapsing on hard benchmarks (GPT-OSS-120B aime25 0.7533 KV16 → 0.3533 KV4) while gsm8k holds [2].

## Detailed Analysis

### 1. The Baseline: What the KV Cache Costs

Every savings claim is a ratio against a baseline that is linear in sequence length and batch size. The formula is 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_element [8][9]. KVQuant's worked example: LLaMA-7B at 128K tokens in FP16 needs 64.0 GB of KV cache against 12.6 GB of weights; LLaMA-65B at 128K needs 320 GB, at 1M needs 2,560 GB [8]. These are estimates from the formula, not measurements.

PagedAttention's contribution is orthogonal to compression. Before vLLM, contiguous pre-allocation used only 20.4–38.2% of KV memory for actual token states; paged blocks bound waste to within one block, reaching 96.3% utilization [15]. The cost is a 20–26% higher attention-kernel latency [15]. This matters for compression because it sets the denominator: a method that claims "5× reduction" is reducing a cache that paged allocation has already made 2–4× more efficient in practice.

GQA is the largest single lever already deployed. Llama-3 70B uses 8 KV heads against 64 query heads, an 8× reduction before any compression. Without GQA, 128K on a 70B model would need roughly 320 GB of FP16 KV. This is why the compression literature increasingly reports against GQA baselines, and why methods that assume MHA (like some eviction policies) show different trade-offs on modern models.

### 2. Quantization: Bit-Widths, Granularities, and Reported Losses

Quantization is the best-evidenced family, with multiple peer-reviewed primaries reporting full tables.

**Granularity is the design axis.** KIVI quantizes keys per-channel and values per-token, with group size 32 and a 128-token full-precision residual window [9]. Its ablation shows why: at 2-bit, per-channel keys with per-token values gives 63.53 on CoQA vs 2.88 for the reverse, because key reconstruction error is 4.55 per-channel vs 13.67 per-token [9]. KVQuant adds pre-RoPE key quantization (0.82 PPL improvement), sensitivity-weighted non-uniform datatypes (0.33 PPL gain over NormalFloat), and per-vector dense-and-sparse outlier isolation (0.19 PPL improvement at 1% outliers) [8]. RotateKV adds outlier-aware rotation and attention-sink-aware quantization, keeping sink tokens in FP16 [10].

**The bit-width ladder is consistent across papers.** At 4-bit, KVQuant-4bit-1% is within 0.02 PPL of FP16 across LLaMA-7B–65B at 3.7× savings [8]; KIVI-4 matches 16-bit on LongBench (Llama2-7B 44.59 vs 44.52) [9]; RotateKV-4bit shows 0.01 PPL delta on LLaMA-2-13B [10]; CQ-2c8b gives 5.70 vs 5.68 [11]. At 3-bit, KVQuant-3bit-1% stays under 0.1 PPL at 4.8× savings, scoring 31.21 vs 31.96 on LongBench and 53.65 vs 56.40 on RULER for LLaMA-2-7B-32K [8]. At 2-bit, the spread widens: RotateKV reaches 4.84 vs 4.57 on LLaMA-2-13B with <1.7% GSM8K drop and 3.97× peak memory reduction [10]; KVQuant-2bit-1% gives 6.01 vs 5.68 on LLaMA-7B at 6.9× savings but RULER drops to 36.54 vs 56.40 [8].

**The failure modes are documented.** Uniform INT2 collapses to 11,779 PPL on LLaMA-7B [11]. QuaRot-2bit hits 21.43 vs 5.75 on LLaMA-3-8B [10]. KIVI at group size 128 drops LongBench TriviaQA to 38.6 vs 87.4 [10]. MILLION's re-run shows KVQuant-3b without outlier isolation at 11.21 PPL on LLaMA-2-7B and 12,323 on PTB [16]. The lesson: 2-bit is not a bit-width, it is a family of methods with wildly different outlier handling.

**Engine-level quantization is FP8-first.** vLLM's docs document only FP8, with per-tensor or per-attention-head scaling; per-head requires FlashAttention plus llm-compressor calibration [1]. SGLang adds FP4 (nvfp4, fp4_mx_block16) with per-tensor scaling only [2]. TensorRT-LLM supports INT8 and FP8 with per-tensor scaling only [3]. vLLM's release notes show the frontier moving to hardware-gated formats: NVFP4 on SM8x/SM12x/SM90/SM100, MXFP8 for DeepSeek-V4.1-Flash on SM100, Triton INT4 per-token-head, ROCm ultraquant_4bit at ~2× FP8 density [4].

### 3. Eviction: Selection Policies and the Re-Run Problem

Eviction policies select which tokens to keep. The main families are attention-score (H2O's heavy hitters, SnapKV's observation-window voting), recency (StreamingLLM's sink-plus-window), learned (LKV's head-wise budgets), and hybrid (PyramidKV's per-layer allocation).

**H2O's own numbers are strong.** At 20% budget on OPT-30B, COPA is 84.00 vs 85.00 full cache, OpenBookQA 43.00 vs 43.20, PiQA 78.45 vs 78.51, Winogrande 69.06 vs 70.24, at up to 5× memory reduction [12]. Recency-only collapses (COPA 48.00) and strided baselines collapse (COPA 50.00) [12]. The motivation is attention sparsity above 95% [12].

**Independent re-runs are harsher.** On Llama-3.1-8B LongBench at B=128, SnapKV averages 42.22, PyramidKV 41.64, H2O 38.10, StreamingLLM 36.51, against FullKV 46.24 [13]. At B=32, SnapKV falls to 35.85 and H2O to 33.16 [13]. SnapKV's gov_report ROUGE-L is 22.3 vs 33.8 full cache [13]. LKV reports PyramidKV at 32.12 vs 47.48 full cache at 15% retention, and DuoAttention at 31.03 [14]. PyramidKV's own paper claims 41.49 vs 41.46 at 2048 entries [17] — a direct conflict with LKV's 32.12 vs 47.48 that no supplied source resolves.

**Failure modes are structural.** Eviction is irreversible: "the evicted KV embeddings become inaccessible in the subsequent decoding steps" [12]. An ACL 2026 study shows eviction bias causes system-prompt leakage even without adversarial prompting, with StreamingLLM and SnapKV showing the starkest bias [22]. Rigid per-layer priors collapse on retrieval: PyramidKV scores 44.65 and DuoAttention 29.00 vs 99.50 full cache on PassageRetrieval-en [14]. PyramidKV's own appendix notes per-layer budgets are ineffective under standard vLLM paged attention without per-layer block tables [17].

**Deferring eviction helps.** Draft-guided eviction (DGE) defers eviction by one decode step, recovering SnapKV from 35.34 to 44.17 averaged over 15 backbone×budget cells, within 0.77 of full cache [13]. This suggests the timing of eviction matters as much as the selection policy.

### 4. Low-Rank and Latent Methods: Highest Ratios, Highest Barriers

Low-rank methods compress the KV cache by projecting it into a smaller latent space. The schemes differ in where the decomposition happens.

**Palu** applies SVD to the key/value projection weight matrices post-training, caching the smaller latent representation and reconstructing full K/V on the fly [5]. At 50% compression on Llama2-7B, J-LRD gives 5.62 vs 5.47 PPL, G-LRD 6.01, M-LRD 6.75 [5]. Combined with 2-bit quantization, Palu-30% reaches 91.25% compression at 5.76 PPL vs KVQuant-2bit's 6.95 at 87.5% [5]. The paper claims 1.19 PPL improvement over KVQuant at 30% less memory [5].

**CommonKV** applies cross-layer SVD, concatenating adjacent layers' Wk/Wv matrices and sharing a latent matrix [7]. It is training-free and holds 71.59 vs 73.55 full KV at 0.5 ratio on Llama-3.1-8B, beating Palu's 68.79 [7]. It degrades sharply past 0.6 (68.19 vs Palu 50.59) [7].

**MLA-style projection** gives the largest ratios but needs training. MHA2MLA cuts Llama2-7B KV by 92.19% with a ~1% LongBench drop (25.1 vs 27.4) after fine-tuning on 0.6–1% of pretraining tokens, and beats 2-bit quantization at matched ~87.5% compression (25.1 vs HQQ 21.2, Quanto 18.5) [6]. X-EcoMLA reaches 6.4× (15.6% KV) with no average-score loss on Llama3.2-1B and 10.67× with ~1.7–2× throughput on MI300, but requires distillation plus DPO [18].

**The matched-budget comparison is thin.** Palu's head-to-head against KVQuant is the clearest: at 2-bit, Palu-30% gives 5.76 vs KVQuant-2bit's 6.95 at 30% less memory [5]. MHA2MLA's comparison against Int2 quantization at matched compression favors the low-rank approach (25.1 vs 21.2) [6]. But no supplied harness runs a low-rank method alongside a quantizer and an evictor at equal measured bytes.

### 5. Cross-Family Comparison at Matched Settings

Three independent unified harnesses compare quantizers against evictors under one model and benchmark, and all converge on the same qualitative ordering.

| Method family | Representative method | Compression | Accuracy at matched budget | Source type |
|---|---|---|---|---|
| 4-bit quantization | KVQuant-4bit-1% | 3.7× | <0.02 PPL delta [8] | Peer-reviewed |
| 4-bit quantization | KIVI-4 | ~4× | 44.59 vs 44.52 LongBench [9] | Peer-reviewed |
| 3-bit quantization | KVQuant-3bit-1% | 4.8× | <0.1 PPL; RULER 53.65 vs 56.40 [8] | Peer-reviewed |
| 2-bit quantization | RotateKV-2bit | ~4× | 4.84 vs 4.57 PPL [10] | Peer-reviewed |
| 2-bit quantization | KVQuant-2bit-1% | 6.9× | 6.01 vs 5.68 PPL; RULER 36.54 vs 56.40 [8] | Peer-reviewed |
| Eviction (attention) | H2O | 5× | 38.10 vs 46.24 LongBench B=128 [13] | Independent re-run |
| Eviction (attention) | SnapKV | 4× | 42.22 vs 46.24 LongBench B=128 [13] | Independent re-run |
| Eviction (learned) | LKV | 6.6× | 46.73 vs 47.48 LongBench R=0.15 [14] | Preprint |
| Low-rank (post-training) | Palu-50% | 2× | 5.63 vs 5.12 PPL [5] | Preprint |
| Low-rank (cross-layer) | CommonKV-0.5 | 2× | 71.59 vs 73.55 LongBench [7] | Preprint |
| Latent (fine-tuned) | MHA2MLA | 12.8× | 25.1 vs 27.4 LongBench [6] | Preprint |

KVDiagnosis runs eight compressors across five families on one Qwen3-8B with fixed splits, producing 59,800 compressed runs and 12,520 full-cache-correct→compressed-wrong rows [23]. At the light tier, TOVA, AdaKV, SnapKV, and QuantizedCache sit within 0.3 score points but their failure rates span 0.8–2.0% — a 2.5× difference hidden by aggregate quality [23]. At the aggressive tier, TOVA/AdaKV score 79.1/80.0 vs StreamingLLM 33.3 and ChunkKV 54.8, while QuantizedCache falls from 89.3 (4b) to 14.6 (2b) [23]. Critically, KVDiagnosis refuses cross-mechanism ranking because 75/50/25% retention and 8b/4b/2b are not byte-equivalent [23].

HqeKV matches quantization and eviction by average bit-width on Llama-3.1-8B and Qwen3-8B over LongBench [24]. HqeKV-3.25 averages 49.98 vs CAKE-3.25 at 40.53 on Llama, and at 2-bit HqeKV-2 (49.71) is on par with FP16 (49.76), KIVI-2 (48.96), and OTT-2 (49.43) [24]. But ZipCache/KVTuner averages are computed only over non-OOM/non-ERROR columns, and no low-rank method is included [24].

SCBench shows benchmark choice dominates the reported trade-off [25]. SnapKV scores 4 on InfiniteBench retrieval and 100 on LongBench retrieval but 0 on SCBench multi-request and 0 across all five turns [25]. On Llama-3.1-8B multi-turn average, full attention is 48.7 vs KIVI (1/8) 32.0, SnapKV (1/32) 20.9, and StreamingLLM (1/32) 15.5 [25]. On Qwen2.5-72B multi-request, StreamingLLM and SnapKV fall to 2.2 and 3.3 vs 45.8 full [25]. The mechanism: critical KVs are stable within a turn but shift across queries, so query-conditioned eviction fails on follow-ups [25].

MoE-nD is the only supplied harness reporting a matched-memory head-to-head between a quantizer and an evictor: on DeepSeek-R1-Distill-Qwen-7B at 16k LongBench inputs, per-layer-quantization-only reaches 5.9 at 139 MB while the eviction-routed variant reaches 12.0 at 136 MB vs 11.5 uncompressed at 1.9 GB [26]. It explicitly excludes low-rank projection and defers SnapKV/H2O comparisons to a follow-up [26].

### 6. Production Engine Mapping

The production picture is unambiguous: engines ship quantization, not eviction or low-rank.

| Engine | KV quantization formats | Granularity | Eviction / low-rank | Form |
|---|---|---|---|---|
| vLLM | FP8 e4m3/e5m2; NVFP4; MXFP8/MXFP4; Triton INT4 per-token-head; ROCm ultraquant_4bit [1][4] | Per-tensor or per-head [1] | StreamingLLM sink tokens on some backends [3] | Opt-in flag (`kv_cache_dtype`) [1] |
| SGLang | FP8 e4m3/e5m2; experimental NVFP4, FP4 block-16 [2] | Per-tensor only [2] | None documented | Opt-in flag (`--kv-cache-dtype`) [2] |
| TensorRT-LLM | INT8, FP8 [3] | Per-tensor only [3] | StreamingLLM sink tokens; cyclic rolling buffer [3] | Config flag (`kv_cache_quant_mode`) [3] |

vLLM's attention-backend registry shows the real support matrix: FlashAttention FA2 accepts only fp16/bf16 KV, FA3/FA4 add fp8/fp8_e4m3; TRITON_ATTN adds int4_per_token_head, int8_per_token_head, fp8_per_token_head; FlashInfer adds nvfp4_4over6 [27]. Block sizes are backend-constrained (FA multiples of 16, FlashInfer 16–1024, FlashMLA 64), and a pinned block size not a multiple of 128 makes FA ineligible for head_dim=256 [27].

vLLM's release notes show the frontier moving fast: GLM-5.x defaults to FP8 KV on SM100, halving KV memory per rank at +2.3–5.5% throughput and 7–11% TTFT cost per the PR benchmark [4]. DeepSeek-V4.1-Flash uses NVFP4 compressed KV as the SM100 default [4]. ROCm gains 4-bit ultraquant_4bit at ~2× FP8 density [4].

No supplied evidence shows H2O, SnapKV, PyramidKV, KIVI, KVQuant, or Palu shipping as a default, flag, or plugin in any of these engines. KVCache-Factory lists porting to nano-vllm/mini-sglang as future work [28].

### 7. Why Adoption Follows This Pattern: Kernel Economics

The adoption pattern is explained by kernel economics, not by accuracy.

**FP8 is cheap because it runs inside the attention kernel.** vLLM measures Llama-3.1-8B on H100 at 54% of the BF16 ITL slope (4.37e-05 → 2.37e-05 ms/token, break-even ~7k tokens) and +14.9% output throughput (517.5 vs 450.3 tok/s) at concurrency 8 [19]. But head_dim=256 regresses prefill TTFT ~1.6× from two-level accumulation, and a Hopper accumulation bug dropped 128k NIAH from 91% (BF16) to 13% (FP8), restored to 89% by the fix [19].

**4-bit KV kernels are not free.** QServe measures naive KV4 attention at 0.48 ms vs 0.42 ms for KV8 on A100 (0.88×) because dequantization saturates the CUDA-core roofline, reaching 0.28 ms (1.49×) only after FP16 ops, bit tricks, and prefetching [20]. FuTransformer notes Hopper has no INT4 tensor-core MMA, so INT4 must be expanded into 16-bit operands [21]. Oaken shows the same on GPU: implementing its algorithm on GPU suffers long quant/dequant latency due to CUDA warp divergence [29].

**Paged allocation interacts with compression.** QServe follows vLLM/TensorRT-LLM paged KV caches but requires per-head dynamic KV quantization for 4-bit accuracy, storing FP16 scales and zero points per head immediately after quantized KV features in each page [20]. PyramidKV's per-layer budgets are ineffective under standard vLLM paged attention without per-layer block tables [17]. TensorRT-LLM's KV Cache Connector documents concrete constraints: with a connector attached, host_cache_size/disk_cache_size above zero fail at bring-up, and enable_block_reuse=True must be set or the connector's prefix is never honored [30].

### 8. The Academic-to-Production Gap

The gap between benchmark results and production deployment is real and documented.

**Engine-level accuracy is quantified.** vLLM reports ≤1–2 points degradation on reasoning benchmarks (lowest recovery 97% GPQA:Diamond) and 97–98% AUC@128k recovery on MRCR with uncalibrated per-tensor scales, with a systematic downward shift for Kimi-K2.5 on FlashMLA/H200 motivating calibration [19]. SGLang's own tables show FP4 KV collapsing on hard benchmarks: GPT-OSS-120B aime25 0.7533 KV16 → 0.3533 KV4, while gsm8k holds [2].

**Real-workload evidence is emerging.** The vLLM project's TurboQuant study finds FP8 KV gives 2× capacity with negligible loss and BF16-matching throughput, while aggressive variants lose ~20 points on Qwen3-30B-Thinking reasoning and ~30% relative AUC on 256k retrieval (33.5/31.2 vs 45.8 BF16) [31]. TurboQuant adds 10–68% latency and 66–80% of BF16 throughput because it dequantizes before attention [31].

**RAG faithfulness degrades silently.** An independent RAG audit finds INT8 near-lossless but INT4 silently degrades faithfulness on accuracy-preserved examples (231 worsenings vs 24 improvements on RGB, p<10−38; 173 vs 31 on HotpotQA), with up to 6% degenerate generations, and realized savings of only ~1.9× (INT8) and ~3.6× (INT4) at K=5 due to scale/zero-point overhead [32].

**Agent workloads are harder.** CommitKV shows that at a 4096-token budget it beats the best compressed baseline by 19.85/5.49/1.54 points on Qwen3-14B/Qwen3.6-27B/DeepSeek-R1-Distill-Llama-8B but still trails FullKV (46.54 vs 50.62 on Qwen3-14B), and at 2048 tokens snapshot baselines collapse (SnapKV 13.80, R-KV 16.21, TriAttention 13.79 vs FullKV 50.62) while CommitKV reaches 38.45 [33].

## Risks & Counterpoints

**The cross-family ranking is not matched-protocol.** KVDiagnosis, HqeKV, SCBench, and MoE-nD all omit low-rank methods, so the low-rank column of the comparison table rests on per-paper numbers with different baselines [23][24][25][26]. No supplied harness runs a 4-bit quantizer, an eviction method, and a low-rank method at equal measured bytes on the same model and benchmark.

**The PyramidKV/SnapKV 2048-budget dispute is unresolved.** PyramidKV's own paper reports 41.49 vs 41.46 at 2048 entries [17]; LKV reports 32.12 vs 47.48 at 15% retention [14]. No supplied source adjudicates the conflict, and the official repos and course-project reproduction do not supply matched-budget numbers.

**Eviction accuracy claims are contested.** H2O's own paper reports near-parity at 20% budget [12]; independent re-runs place H2O 8 points below full cache at B=128 [13]. The discrepancy likely reflects different models, benchmarks, and budget definitions, but the practical implication is that eviction's reported numbers should not be taken at face value.

**Low-rank methods need training for the best results.** Palu is post-training [5], but MHA2MLA requires fine-tuning on 0.6–1% of pretraining tokens [6] and X-EcoMLA requires distillation plus DPO [18]. The training-free options (CommonKV) degrade sharply past 0.6 compression [7].

**Engine-level accuracy for 4-bit KV is thin.** SGLang's FP4 tables and FuTransformer's PPL deltas are the only low-bit accuracy data, and neither covers vLLM's shipped 4-bit paths (Triton INT4 per-token-head, ROCm ultraquant_4bit, TurboQuant) on long-context benchmarks [2][21][4].

**Practitioner-reported accuracy in real RAG and agent workloads is limited.** The RAG audit and CommitKV use benchmark datasets, and the Huawei/ETH reuse study shows aggregate F1 can be inflated by up to 42% from queries the full-prefill baseline cannot answer [32][33][34].

## Open Questions

**A matched-protocol harness including low-rank.** The single largest gap is a harness that runs a 4-bit quantizer, an eviction method, and a low-rank method at equal measured bytes on the same model, context length, and benchmark. KVDiagnosis, HqeKV, SCBench, and MoE-nD all omit low-rank [23][24][25][26].

**Direct matched-memory comparisons of SnapKV and H2O against quantization.** MoE-nD explicitly defers this to a follow-up [26], and SCBench does not provide it (KIVI at 1/8 vs eviction at 1/32) [25].

**An independent re-run of PyramidKV and SnapKV at 2048-entry budgets.** The conflict between PyramidKV's claimed parity (41.49 vs 41.46) [17] and LKV's 32.12 vs 47.48 at 15% retention [14] remains unresolved.

**Engine-level accuracy for vLLM's shipped 4-bit KV paths.** The vLLM TurboQuant study covers TurboQuant variants but not the other shipped low-bit paths (Triton INT4 per-token-head, ROCm ultraquant_4bit), and SGLang's FP4 tables cover a different engine [31][2].

**Production-traffic accuracy data for eviction or low-bit quantization in RAG and multi-turn agent serving.** The RAG audit and CommitKV use benchmark datasets, and the Huawei/ETH reuse study shows aggregate F1 can be inflated by up to 42% from queries the full-prefill baseline cannot answer [32][33][34].

**Whether MLA-style latent caching is deployable outside DeepSeek-family models without full fine-tuning.** MHA2MLA's 0.6–1% pretraining-token requirement [6] and the absence of an open-sourced tensor-parallel MLA inference framework leave this open.

## Recommendation

**Production-ready today:** FP8 KV cache in vLLM, SGLang, and TensorRT-LLM. It is opt-in, hardware-gated, and delivers ~2× capacity with ≤1–2 points degradation on reasoning benchmarks and 97–98% AUC@128k recovery with uncalibrated scales [1][2][3][19]. Calibrate scales with llm-compressor for best results [1].

**Production-ready with caveats:** 4-bit KV on hardware that supports it (NVFP4 on SM90/SM100, MXFP4, Triton INT4 per-token-head, ROCm ultraquant_4bit). Accuracy is near-lossless on simple benchmarks but degrades on hard reasoning and long-context retrieval [2][31]. Validate on your workload before enabling.

**Research-grade for general serving:** eviction (H2O, SnapKV, PyramidKV) and low-rank (Palu, CommonKV). Eviction is viable only as a fixed-window/sink mechanism (StreamingLLM-style) or with deferred eviction [13][3]. Low-rank is viable only if you control the model weights and can afford fine-tuning or distillation [6][18].

**Not recommended for production:** 2-bit KV without outlier isolation, uniform INT2, and any eviction policy at aggressive budgets on multi-turn or agent workloads [11][10][25][33].

## References

- **[1] docs.vllm.ai** - "Quantized KV Cache". [Source](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/) (accessed 2026-10-28)
- **[2] SGLang Documentation** - "Quantized KV Cache". [Source](https://docs.sglang.io/docs/advanced_features/quantized_kv_cache) (accessed 2026-10-28)
- **[3] nvidia.github.io** - "Multi-Head, Multi-Query, and Group-Query Attention". [Source](https://nvidia.github.io/TensorRT-LLM/1.0.0rc0/advanced/gpt-attention.html) (accessed 2026-10-28)
- **[4] GitHub** - "Releases · vllm-project/vllm". [Source](https://github.com/vllm-project/vllm/releases) (accessed 2026-10-28)
- **[5] arxiv.org** - "Palu: Compressing KV-Cache with Low-Rank Projection". [Source](https://arxiv.org/html/2407.21118v1) (accessed 2026-10-28)
- **[6] arxiv.org** - "Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs". [Source](https://arxiv.org/pdf/2502.14837) (accessed 2026-10-28)
- **[7] arxiv.org** - "CommonKV: Compressing KV Cache with Cross-layer Parameter Sharing". [Source](https://arxiv.org/abs/2508.16134v1) (accessed 2026-10-28)
- **[8] arxiv.org** - "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization". [Source](https://arxiv.org/pdf/2401.18079) (accessed 2026-10-28)
- **[9] arxiv.org** - "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache". [Source](https://arxiv.org/pdf/2402.02750) (accessed 2026-10-28)
- **[10] ijcai.org** - "RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations". [Source](https://www.ijcai.org/proceedings/2025/0690.pdf) (accessed 2026-10-28)
- **[11] arxiv.org** - "KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization". [Source](https://arxiv.org/pdf/2405.03917) (accessed 2026-10-28)
- **[12] arxiv.org** - "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models". [Source](https://arxiv.org/pdf/2306.14048) (accessed 2026-10-28)
- **[13] arxiv.org** - "When to Evict, Not What to Keep: Draft-Guided Eviction for Training-Free KV-Cache Compression". [Source](https://arxiv.org/pdf/2609.33334) (accessed 2026-10-28)
- **[14] arxiv.org** - "LKV: End-to-End Learning of Head-wise Budgets and Token Selection for LLM KV Cache Eviction". [Source](https://arxiv.org/pdf/2605.06676) (accessed 2026-10-28)
- **[15] arxiv.org** - "Efficient Memory Management for Large Language Model Serving with PagedAttention". [Source](https://arxiv.org/pdf/2309.06180) (accessed 2026-10-28)
- **[16] arxiv.org** - "MILLION: MasterIng Long-Context LLM Inference Via Outlier-Immunized KV Product QuaNtization". [Source](https://arxiv.org/abs/2504.03661v2) (accessed 2026-10-28)
- **[17] arxiv.org** - "PyramidKV: Dynamic KV Cache Compression based on Pyra- midal Information Funneling". [Source](https://arxiv.org/abs/2406.02069v4) (accessed 2026-10-28)
- **[18] arxiv.org** - "X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression". [Source](https://arxiv.org/abs/2503.11132v4) (accessed 2026-10-28)
- **[19] vllm.ai** - "The State of FP8 KV-Cache and Attention Quantization in vLLM". [Source](https://vllm.ai/blog/2026-04-22-fp8-kvcache) (accessed 2026-10-28)
- **[20] arxiv.org** - "QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving". [Source](https://arxiv.org/abs/2405.04532v3) (accessed 2026-10-28)
- **[21] arxiv.org** - "FuTransformer: FP8 prefill and INT4 decode for Llama-style inference on Hopper". [Source](https://arxiv.org/pdf/2610.23796) (accessed 2026-10-28)
- **[22] aclanthology.org** - "The Pitfalls of KV Cache Compression". [Source](https://aclanthology.org/2026.acl-long.1926.pdf) (accessed 2026-10-28)
- **[23] arxiv.org** - "KVDiagnosis: A Diagnostic Benchmark for KV-Cache Compression in Long-Context Language Models". [Source](https://arxiv.org/pdf/2608.09412) (accessed 2026-10-28)
- **[24] aclanthology.org** - "HqeKV: Towards Hybrid Quantization and Eviction for KV Cache in Long-Context LLM Inference". [Source](https://aclanthology.org/2026.findings-acl.201.pdf) (accessed 2026-10-28)
- **[25] arxiv.org** - "SCBENCH: A KV CACHE-CENTRIC ANALYSIS OF LONG-CONTEXT METHODS". [Source](https://arxiv.org/abs/2412.10319v2) (accessed 2026-10-28)
- **[26] arxiv.org** - "MoE-nD: Per-Layer Mixture-of-Experts Routing for Multi-Axis KV Cache Compression". [Source](https://arxiv.org/html/2604.17695v1) (accessed 2026-10-28)
- **[27] docs.vllm.ai** - "Attention Backend Feature Support". [Source](https://docs.vllm.ai/en/latest/design/attention_backends/) (accessed 2026-10-28)
- **[28] GitHub** - "GitHub - Zefan-Cai/KVCache-Factory: Unified KV Cache Compression Methods for Auto-Regressive Models". [Source](https://github.com/Zefan-Cai/KVCache-Factory) (accessed 2026-10-28)
- **[29] dl.acm.org** - "Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization". [Source](https://dl.acm.org/doi/pdf/10.1145/3695053.3731019) (accessed 2026-10-28)
- **[30] nvidia.github.io** - "KV Cache Connector". [Source](https://nvidia.github.io/TensorRT-LLM/features/kv-cache-connector.html) (accessed 2026-10-28)
- **[31] vLLM Blog** - "A First Comprehensive Study of TurboQuant: Accuracy and Performance". [Source](https://vllm-project.github.io/2026/05/11/turboquant.html) (accessed 2026-10-28)
- **[32] arxiv.org** - "Faithfulness Is Not Free: Auditing Offline KV-Cache Quantization in Retrieval-Augmented Generation". [Source](https://arxiv.org/pdf/2608.30996v1) (accessed 2026-10-28)
- **[33] arxiv.org** - "CommitKV: Lifecycle-Aware KV Cache Compression via Commit Transitions for Multi-Turn Agents". [Source](https://arxiv.org/html/2608.07855v1) (accessed 2026-10-28)
- **[34] arxiv.org** - "Evaluating the accuracy of KV cache reuse techniques". [Source](https://arxiv.org/pdf/2609.31415) (accessed 2026-10-28)