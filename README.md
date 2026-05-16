# Work Trial: FP4 Quantized Linear Kernel

## Goal

Implement and evaluate an FP4 quantized linear layer for transformer inference.

Coding agents and public references are allowed, with one caveat: we want to see how you think.

As such, you should understand and be prepared to explain all code you submit. You should also be prepared to explain why you steered the project in certain directions.

## Deliverables

1. A repository implementing the requirements below.
2. A short PowerPoint summarizing your approach, results, and conclusions.

You will hop on a call with a member of our team to discuss your approach and conclusions.

## Requirements

### 1. Quantization

Implement:

```python
quantize(weight, *, format, block_size, scale_mode) -> QuantizedTensor
dequantize(qweight) -> torch.Tensor
```

Support at least two FP4 formats, per-block scaling, two scale modes (`absmax` plus one of your choice), packing two 4-bit values per byte, ragged dimensions, and non-contiguous input tensors.

You choose the FP4 encoding, please document it according to floating point conventions: sign bit, exponent/mantissa layout, representable values, and how values are rounded or clipped.

### 2. Quantized Linear

Implement:

```python
fp4_linear(x, qweight, bias=None) -> torch.Tensor
```

Compare naive dequantize-then-matmul against an optimized path of your choice (Triton, CUDA, CUTLASS, or hybrid). The optimized path should avoid at least some unnecessary memory movement versus the naive approach. Be prepared to explain the tradeoff you made.

### 3. Tests

Include tests for:

- correctness of packing
- quantization/dequantization error;
- both scale modes;
- ragged shapes;
- non-contiguous tensors;
- `fp4_linear` correctness against a dequantized reference

Include both deterministic and randomized cases.

### 4. Microbenchmarks

Benchmark decode-like and prefill-like shapes. Report latency, speedup versus a BF16/FP16 baseline, and memory footprint including scale overhead. Use sound methodology, such as warmup, synchronization, and repeated runs.

Characterize quantization error across your FP4 formats, scale modes, and block sizes. Discuss what you find: how do these choices trade off against each other in speed, memory, and accuracy?

### 5. Model Integration

Integrate your quantized linear into a transformer model (e.g. Qwen3) and evaluate on a few MMLU subjects. Compare FP4 accuracy against a BF16/FP16 baseline across your configurations.

Also examine per-layer output drift, particularly whether attention and MLP layers respond differently to quantization, and whether sensitivity varies by model depth.

Discuss which configuration would you choose for deployment, and why? We're especially interested in cases where local error and downstream accuracy don't correlate cleanly.

## 6. References

List papers, repositories, blog posts, and any generated code you used. For generated code, briefly note what was generated, what you changed, and how you validated it.