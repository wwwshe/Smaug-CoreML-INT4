# Smaug-CoreML-INT4

[한국어](README.md) | **English**

Three neural-network components of [abacusai/Smaug-AI-Image-Detector](https://huggingface.co/abacusai/Smaug-AI-Image-Detector), converted to Core ML (ML Program) with INT4-quantized weights.

> [!IMPORTANT]
> **These three models alone do not perform end-to-end image detection.**
> Image preprocessing (cropping, patchifying), language-model input assembly (special-token embeddings) and post-processing (pooling, normalization, classification head, sigmoid) are not included and must be implemented separately. See [End-to-end inference flow](#end-to-end-inference-flow).

> [!WARNING]
> **Inference from Swift / on iOS devices has not been verified.** All results below were obtained with Python (`coremltools`) on macOS using `CPU_ONLY`.

## About the original model

Smaug-AI-Image-Detector, by Abacus.AI, is a fine-tune of Google's [EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2) that outputs the probability that an image is AI-generated. The original is a PyTorch / sentence-transformers model with about 439M parameters. For accuracy and benchmarks, see the [original model card](https://huggingface.co/abacusai/Smaug-AI-Image-Detector). This repository does not reproduce or verify those benchmark results.

Converted from upstream revision `c7c95951e04a87b884a4ad479401b1128c3903bd`.

## Files

| File | Role | Package size | Largest file (`weight.bin`) |
|---|---|---|---|
| `SmaugVision_INT4.mlpackage` | Vision tower: image patches → image soft tokens | 105.37 MB (100.48 MiB) | 104,822,400 B (99.96 MiB) |
| `SmaugEmbedder_INT4.mlpackage` | Multimodal embedder: vision tokens (768) → language-model space (512) | 0.25 MB (0.24 MiB) | 248,064 B |
| `SmaugLanguage_INT4.mlpackage` | Language model: token-embedding sequence → per-token hidden states (768) | 87.31 MB (83.26 MiB) | 86,624,832 B (82.61 MiB) |

For reference, the FP32 Core ML conversions before INT4 quantization used about 640 MiB (Vision), 524 MiB (Language) and 1.5 MiB (Embedder) on disk (`du`). The FP32 conversions are not included.

## Input / output specification

All shapes are static. The original pipeline always uses 528 × 528 crops, so the per-crop shapes are the same for every image size. Batch size is 1; with multiple crops, call each model once per crop.

### SmaugVision_INT4

| | Name | Shape | Type |
|---|---|---|---|
| Input | `pixel_values` | `[1, 1260, 768]` | Float32 |
| Input | `pixel_position_ids` | `[1, 1260, 2]` | Int32 |
| Output | `_to_copy_219` | `[121, 768]` (observed) | Float32 |

- 1260 = soft-token budget 140 × 3 × 3 pooling kernel. A 528 × 528 crop yields 33 × 33 = 1089 patches of 16 × 16; the remaining 171 entries are padding.
- Each patch holds 16 × 16 × 3 = 768 values. Pixels are rescaled to `[0, 1]` with no mean/std normalization.
- `pixel_position_ids` holds the (x, y) patch coordinates; padding patches are `(-1, -1)`.
- The output shape is not recorded in the model spec; the observed output was `[121, 768]` (11 × 11 pooled tokens, padding removed).

### SmaugEmbedder_INT4

| | Name | Shape | Type |
|---|---|---|---|
| Input | `inputs_embeds` | `[121, 768]` | Float32 |
| Output | `linear` | `[121, 512]` | Float32 |

### SmaugLanguage_INT4

| | Name | Shape | Type |
|---|---|---|---|
| Input | `inputs_embeds` | `[1, 125, 512]` | Float32 |
| Input | `attention_mask` | `[1, 125]` | Int32 |
| Output | `linear_217` | `[1, 125, 768]` | Float32 |

- 125 = 4 special tokens + 121 image tokens. `attention_mask` is all ones.

## End-to-end inference flow

This flow follows the original `smaug_image_detect.py` and sentence-transformers configuration. **Only the steps marked ★ are handled by the Core ML models in this repository**; everything else must be implemented by you.

1. **Crops** (implement yourself)
   - Apply EXIF orientation and convert to RGB.
   - If the shorter side is larger than 576, downscale the shorter side to 576 (LANCZOS).
   - If the shorter side is smaller than 528, upscale the shorter side to 528 (BICUBIC, rounded up).
   - `crops=1`: one centre 528 × 528 crop. `crops=3`: start, centre and end crops along the longer side.
2. **Patchify** (implement yourself): build `pixel_values` and `pixel_position_ids` exactly like the Hugging Face `Gemma4ImageProcessor` used by the original model (patch_size 16, rescale 1/255, no normalization), padded to 1260 entries. The value layout inside each patch must match the original implementation exactly.
3. ★ **SmaugVision_INT4** → `[121, 768]`
4. ★ **SmaugEmbedder_INT4** → `[121, 512]`
5. **Assemble language-model input** (implement yourself)
   - Token order: `<bos>` (2), `<|image>` (255999), image token (258880) × 121, `<image|>` (258882), `<eos>` (1).
   - Put the step 4 output at the image-token positions, and the original model's input-embedding layer (`get_input_embeddings()`) output at the 4 special-token positions, to form `[1, 125, 512]`.
   - **The special-token embeddings are not included in the Core ML models.** Extract them from the original model in advance.
6. ★ **SmaugLanguage_INT4** → `[1, 125, 768]`
7. **Post-processing** (implement yourself)
   - Mean pooling over all 125 tokens using the attention mask → `[768]`
   - L2 normalization
   - Dense 768 → 1 (with bias, no activation). The weights are in `3_Dense/model.safetensors` of the original repository and **are not included here.**
   - With multiple crops, average the logits, then apply sigmoid to get the AI-generated probability. The original model card suggests 0.5 as the default threshold.

## Conversion process

1. Load the original model in FP32 with `SmaugImageDetect.from_pretrained(..., device="cpu")`.
2. Wrap the `vision_tower`, `embed_vision` and `language_model` sub-modules and export each with `torch.export.export(strict=False)`.
   - For Vision and Language, the graph was lowered to the ATEN dialect with `run_decompositions`, and `aten.__and__.Tensor` nodes, which Core ML does not support, were replaced with `aten.logical_and.default`.
   - Embedder and Language were exported with inputs captured during real image inference; Vision was exported with random tensors of the same shape.
3. Convert with `coremltools.convert`: `convert_to="mlprogram"`, `minimum_deployment_target=iOS18`, `compute_precision=FLOAT32`.
4. Quantize weights to INT4 with `coremltools.optimize.coreml.linear_quantize_weights`: `mode="linear_symmetric"`, `dtype="int4"`, `granularity="per_block"`, `block_size=32`.

Environment: macOS 27.0.1 (arm64), Python 3.11.17, torch 2.7.0, transformers 5.20.2, coremltools 9.0

The conversion scripts are not included in this repository.

## Verification results

**Method**
- The original PyTorch model (FP32, CPU) was compared with the Core ML models (`coremltools`, `ComputeUnit.CPU_ONLY`).
- One test image was used (1278 × 1230 PNG, not included).
- Per component: the real inputs to each sub-module were captured during original inference, fed to the Core ML model, and the outputs compared.
- End-to-end: the original Python pipeline (including pre- and post-processing) was kept, and only the outputs of the three sub-modules were replaced with Core ML outputs before comparing the final probability. **Pre- and post-processing therefore came from the original Python code; no separate implementation of them has been verified.**

**Per component (vs. original PyTorch)**

| Component | FP32 Core ML relative L2 error | INT4 relative L2 error | INT4 cosine similarity | INT4 max abs error |
|---|---|---|---|---|
| Vision | 9.9e-4 | 0.353 | 0.9377 | 3956 |
| Embedder | 1.6e-7 | 0.088 | 0.9962 | 0.548 |
| Language | 5.0e-4 | 0.298 | 0.9547 | 244 |

All INT4 outputs were finite (no NaN/Inf). The FP32 conversion closely matches the original; most of the per-component error comes from INT4 quantization.

**Final probability (same single image)**

| Setting | Original PyTorch | With the 3 INT4 Core ML models | Absolute difference |
|---|---|---|---|
| `crops=1` | 0.999835 | 0.999807 | 0.000028 |
| `crops=3` | 0.999857 | 0.999839 | 0.000018 |

> [!CAUTION]
> The close final-probability match comes from **one image whose probability is saturated near 1**. Given the per-component errors, the same agreement is not guaranteed for other images or for images near the decision boundary (around 0.5). Accuracy on benchmark datasets was not measured.

**Not verified**
- Loading and inference from Swift on iOS / iPadOS / macOS
- Results on GPU or Neural Engine compute units (`ALL`, `CPU_AND_NE`, etc.)
- Independent implementations of pre- and post-processing
- Accuracy across many images or benchmark datasets
- Inference speed and memory usage

## Usage

### Download

Model weights are stored with Git LFS.

```bash
git lfs install
```

```bash
git clone https://github.com/wwwshe/Smaug-CoreML-INT4.git
```

### Python (coremltools)

This example only shows how each model is called. Build the actual inputs as described in [End-to-end inference flow](#end-to-end-inference-flow).

```python
import numpy as np
import coremltools as ct

vision = ct.models.MLModel("SmaugVision_INT4.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
embedder = ct.models.MLModel("SmaugEmbedder_INT4.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
language = ct.models.MLModel("SmaugLanguage_INT4.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)

# pixel_values: float32 [1, 1260, 768], pixel_position_ids: int32 [1, 1260, 2]
image_tokens = vision.predict({
    "pixel_values": pixel_values,
    "pixel_position_ids": pixel_position_ids,
})["_to_copy_219"]                                   # [121, 768]

projected = embedder.predict({"inputs_embeds": image_tokens})["linear"]   # [121, 512]

# inputs_embeds: [1, 125, 512] = 4 special-token embeddings + 121 projected tokens
hidden = language.predict({
    "inputs_embeds": inputs_embeds,
    "attention_mask": np.ones((1, 125), dtype=np.int32),
})["linear_217"]                                     # [1, 125, 768]

# Then implement mean pooling → L2 normalization → Dense (768 → 1) → sigmoid yourself
```

### Xcode / Swift

The `.mlpackage` files can be added to an Xcode project; the minimum deployment target is iOS 18 / macOS 15. **Loading and inference from Swift, and Swift implementations of pre- and post-processing, have not been verified.**

## Distribution

- GitHub warns about files larger than 50 MiB and blocks regular Git pushes of files larger than 100 MiB.
- The `weight.bin` of `SmaugVision_INT4` is 99.96 MiB, very close to that limit, and the `weight.bin` of `SmaugLanguage_INT4` (82.61 MiB) triggers the warning.
- `.gitattributes` therefore stores `weights/*.bin` and `*.mlmodel` inside `*.mlpackage` with Git LFS. The total is about 193 MB.
- GitHub Free includes 10 GiB of Git LFS storage and 10 GiB of bandwidth per month. Each clone uses about 0.19 GB, so if many downloads are expected, also attaching an archive to GitHub Releases is recommended.

## License

- The model files in this repository are derivatives of the original weights and are distributed under the [Apache License 2.0](LICENSE).
- Both the original [Smaug-AI-Image-Detector](https://huggingface.co/abacusai/Smaug-AI-Image-Detector) (Abacus.AI) and its base model [EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2) (Google DeepMind) are licensed under Apache License 2.0.
- Attribution and a description of modifications are in [NOTICE](NOTICE).
- The base model card asks deployers to follow the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
- This project is not affiliated with or endorsed by Abacus.AI, Google, or Apple.

## Limitations

- According to the original model card, brand-new generators or generators built to evade detection may get through, and a real photo with a small AI-edited region may be scored as real.
- A score is evidence, not proof. Do not use it as the only basis for decisions about people or their work.
- INT4 quantization changes per-component outputs. Check accuracy on your own data before using these models in production.

## Citation

Original model:

```bibtex
@misc{abacusai2026smaugaiimagedetector,
  title  = {Smaug-AI-Image-Detector},
  author = {Abacus.AI},
  year   = {2026},
  url    = {https://huggingface.co/abacusai/Smaug-AI-Image-Detector}
}
```
