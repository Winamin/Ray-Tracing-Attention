# RT MLP with FP16 or BF16 Tensor operands

The `BF16` branch exposes two independent MLP candidates:

* `--mlp-backend rt_tensor`: power-of-two-scaled FP16 operands and FP32 matrix accumulation.
* `--mlp-backend rt_bfloat`: native BF16 operands and FP32 matrix accumulation.

Both preserve the original model payload and immutable RT weight-block geometry.
RT candidate identities populate direct block-address slots. Weight decoding and
input loading consume those slots; no sorting or runtime material partition is
performed. Missing, foreign or invalid receipts stop the request with error
`4096`. Every output receiver retains every original coordinate contribution.

Tensor multiplication uses 32 output channels, 128 token rows and 32 input
coordinates per tile. Batches below 17 rows retain the original RT transmission
path, including single-token decoding. The fixed 128-row Tensor tile was slower
at eight rows on the measured RTX 5060 Ti. Row tails load zero padding and store
only the requested rows. Operand-range overflow retains an FP32 scalar fallback.

Native BF16 requires a compiler that supports `GL_EXT_bfloat16`, Vulkan
`VK_KHR_shader_bfloat16` type and cooperative-matrix features, and the existing
cooperative-matrix2 features. The runtime checks workgroup and shared-memory
limits and creates the actual device pipelines before reporting an active Tensor
backend. Unsupported devices retain RT transmission. `/rt` and runtime metadata
report the effective backend, operand type, accumulator type and minimum row
count. FP16 is never reported as BF16.

RT performs BVH traversal and intersection. Tensor Core performs matrix
multiply-add. Shader instructions still decode the source payload, convert
operands, manage data movement, compute SiLU/gating and add the residual. Other
pre-existing QKV, recurrent-state and attention Tensor paths retain their existing
formats; choosing `rt_bfloat` changes the MLP operands only.

The original FP32 error bound is recorded as a comparison, rather than an
adoption gate. Coverage, zero-input behavior and finite outputs remain required.
The measurements below pin the worker hash, check SPIR-V for actual BF16/FP16
matrix and RT instructions, compare original-payload FP64 activations and
outputs, measure each MLP shape and full prefill, and capture finite 32-layer
states and generated token identities. Speed and output differences are
measured separately for each backend.

## Measurements on 2026-11-12

RTX 5060 Ti, original 32-layer QwenFlash payload, no cache, full V membership,
and three warmed measurements per shape. Full prefill GPU medians include the
other inference stages, but exclude worker initialization and pipeline compilation.

| Prefill tokens | RT transmission | RT + FP16 Tensor | RT + BF16 Tensor |
|---:|---:|---:|---:|
| 32 | 680.101 ms | 363.723 ms | 422.538 ms |
| 128 | 2586.950 ms | 462.599 ms | 602.509 ms |
| 512 | 10435.957 ms | 1802.214 ms | 1962.868 ms |

For layers 0 and 31 with 1, 7, 8, 16, 17 and 32 diagnostic rows, the largest
original-payload FP64 output errors were `2.42692e-4` for FP16 and `2.36272e-3`
for BF16. Both exceed the old FP32 bound at Tensor row counts. All receiver
coverage and finite-output checks passed; the original RT computation remains
active for smaller batches.

All three RT backends generated the same sixteen token identities for the tested
32-token prefix, and every captured prefill layer and final decode state was
finite. This is a single-prompt comparison; it does not establish model-wide
quality equivalence. The `standard` matrix-only backend remained faster in the
independent MLP check (layer 0, 32 rows: `1.307 ms`), but it does not retain MLP RT
queries and had a larger FP64 output error (`8.43660e-3`) in that check.
