# Smaug-CoreML-INT4

**한국어** | [English](README.en.md)

[abacusai/Smaug-AI-Image-Detector](https://huggingface.co/abacusai/Smaug-AI-Image-Detector)의 신경망 구성요소 3개를 Core ML(ML Program) 형식으로 변환하고 가중치를 INT4로 양자화한 모델입니다.

> [!IMPORTANT]
> **이 저장소의 모델 3개만으로는 이미지 판별이 완성되지 않습니다.**
> 이미지 전처리(크롭, 패치 분할), 언어 모델 입력 조립(특수 토큰 임베딩), 후처리(풀링, 정규화, 분류 헤드, sigmoid)는 포함되어 있지 않으며 사용하는 쪽에서 별도로 구현해야 합니다. 자세한 내용은 [전체 추론 흐름](#전체-추론-흐름)을 참고하세요.

> [!WARNING]
> **Swift / iOS 기기에서의 추론은 검증되지 않았습니다.** 아래 검증 결과는 모두 macOS의 Python(`coremltools`)에서 `CPU_ONLY`로 실행한 결과입니다.

## 원본 모델 소개

Smaug-AI-Image-Detector는 Abacus.AI가 Google의 [EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2)를 파인튜닝한, 이미지가 AI로 생성되었을 확률을 출력하는 모델입니다. 원본은 PyTorch / sentence-transformers 형식이며 약 439M 파라미터입니다. 모델 성능과 벤치마크는 [원본 모델 카드](https://huggingface.co/abacusai/Smaug-AI-Image-Detector)를 참고하세요. 이 저장소는 원본의 벤치마크 성능을 재현하거나 검증하지 않았습니다.

변환 기준 원본 리비전: `c7c95951e04a87b884a4ad479401b1128c3903bd`

## 구성 파일

| 파일 | 역할 | 패키지 크기 | 가장 큰 파일 (`weight.bin`) |
|---|---|---|---|
| `SmaugVision_INT4.mlpackage` | Vision Tower: 이미지 패치 → 이미지 소프트 토큰 | 105.37 MB (100.48 MiB) | 104,822,400 B (99.96 MiB) |
| `SmaugEmbedder_INT4.mlpackage` | Multimodal Embedder: 비전 토큰(768) → 언어 모델 공간(512) | 0.25 MB (0.24 MiB) | 248,064 B |
| `SmaugLanguage_INT4.mlpackage` | Language Model: 토큰 임베딩 시퀀스 → 토큰별 은닉 상태(768) | 87.31 MB (83.26 MiB) | 86,624,832 B (82.61 MiB) |

참고로 INT4 양자화 전 FP32 Core ML 변환본은 디스크 사용량(`du`) 기준 Vision 약 640 MiB, Language 약 524 MiB, Embedder 약 1.5 MiB였습니다. FP32 변환본은 이 저장소에 포함하지 않습니다.

## 입출력 규격

모든 입력 형태는 고정(static)입니다. 원본 파이프라인은 항상 528×528 크롭을 사용하므로 크롭 하나를 처리할 때의 형태가 이미지 크기와 관계없이 동일합니다. 배치 크기는 1이며, 크롭이 여러 개면 모델을 크롭마다 한 번씩 호출해야 합니다.

### SmaugVision_INT4

| 구분 | 이름 | 형태 | 타입 |
|---|---|---|---|
| 입력 | `pixel_values` | `[1, 1260, 768]` | Float32 |
| 입력 | `pixel_position_ids` | `[1, 1260, 2]` | Int32 |
| 출력 | `_to_copy_219` | `[121, 768]` (실측) | Float32 |

- 1260 = 소프트 토큰 예산 140 × 풀링 커널 3×3. 528×528 크롭은 16×16 패치 33×33 = 1089개이며, 나머지 171개는 패딩입니다.
- 각 패치는 16×16×3 = 768개 값이며, 픽셀 값은 `[0, 1]` 범위로 rescale만 하고 mean/std 정규화는 하지 않습니다.
- `pixel_position_ids`는 패치의 (x, y) 좌표이며 패딩 패치는 `(-1, -1)`입니다.
- 출력 형태는 모델 명세에 기록되어 있지 않으며, 실제 실행 결과 `[121, 768]`(11×11 풀링 토큰, 패딩 제거 후)이었습니다.

### SmaugEmbedder_INT4

| 구분 | 이름 | 형태 | 타입 |
|---|---|---|---|
| 입력 | `inputs_embeds` | `[121, 768]` | Float32 |
| 출력 | `linear` | `[121, 512]` | Float32 |

### SmaugLanguage_INT4

| 구분 | 이름 | 형태 | 타입 |
|---|---|---|---|
| 입력 | `inputs_embeds` | `[1, 125, 512]` | Float32 |
| 입력 | `attention_mask` | `[1, 125]` | Int32 |
| 출력 | `linear_217` | `[1, 125, 768]` | Float32 |

- 125 = 특수 토큰 4개 + 이미지 토큰 121개. `attention_mask`는 모두 1입니다.

## 전체 추론 흐름

원본 `smaug_image_detect.py`와 sentence-transformers 구성을 기준으로 정리한 흐름입니다. **★ 표시 단계만 이 저장소의 Core ML 모델이 담당**하며, 나머지는 직접 구현해야 합니다.

1. **크롭 생성** (직접 구현)
   - EXIF 방향 보정 후 RGB로 변환합니다.
   - 짧은 변이 576보다 크면 짧은 변을 576으로 축소합니다(LANCZOS).
   - 짧은 변이 528보다 작으면 짧은 변을 528로 확대합니다(BICUBIC, 올림).
   - `crops=1`: 중앙 528×528 1장. `crops=3`: 긴 변 방향의 시작·중앙·끝 3장.
2. **패치 분할** (직접 구현): 원본의 Hugging Face `Gemma4ImageProcessor`(patch_size 16, rescale 1/255, 정규화 없음)와 동일한 순서와 레이아웃으로 `pixel_values`, `pixel_position_ids`를 만들고 1260개로 패딩합니다. 패치 내부 값 배열 순서는 원본 구현과 정확히 일치해야 합니다.
3. ★ **SmaugVision_INT4** → `[121, 768]`
4. ★ **SmaugEmbedder_INT4** → `[121, 512]`
5. **언어 모델 입력 조립** (직접 구현)
   - 토큰 순서: `<bos>`(2), `<|image>`(255999), 이미지 토큰(258880) × 121, `<image|>`(258882), `<eos>`(1)
   - 이미지 토큰 자리에는 4단계 출력을, 특수 토큰 4개 자리에는 원본 모델의 입력 임베딩 레이어(`get_input_embeddings()`) 출력 벡터를 넣어 `[1, 125, 512]`를 만듭니다.
   - **특수 토큰 임베딩은 Core ML 모델에 포함되어 있지 않습니다.** 원본 모델에서 미리 추출해 두어야 합니다.
6. ★ **SmaugLanguage_INT4** → `[1, 125, 768]`
7. **후처리** (직접 구현)
   - attention mask 기준 mean pooling(125개 토큰 전체) → `[768]`
   - L2 정규화
   - Dense 768 → 1 (bias 포함, 활성화 없음). 가중치는 원본 저장소의 `3_Dense/model.safetensors`에 있으며 **이 저장소에 포함되어 있지 않습니다.**
   - 크롭이 여러 개면 로짓을 평균한 뒤 sigmoid를 적용해 AI 생성 확률을 얻습니다. 원본 모델 카드는 0.5를 기본 임계값으로 제시합니다.

## 변환 과정

1. 원본 모델을 `SmaugImageDetect.from_pretrained(..., device="cpu")`로 FP32 로드합니다.
2. `vision_tower`, `embed_vision`, `language_model` 하위 모듈을 각각 래퍼로 감싸 `torch.export.export(strict=False)`로 내보냅니다.
   - Vision과 Language는 `run_decompositions`로 ATEN dialect로 변환한 뒤, Core ML이 지원하지 않는 `aten.__and__.Tensor` 노드를 `aten.logical_and.default`로 바꿨습니다.
   - Embedder와 Language는 실제 이미지 추론 중 캡처한 입력으로, Vision은 같은 형태의 임의 텐서로 export했습니다.
3. `coremltools.convert`로 변환했습니다: `convert_to="mlprogram"`, `minimum_deployment_target=iOS18`, `compute_precision=FLOAT32`.
4. `coremltools.optimize.coreml.linear_quantize_weights`로 가중치를 INT4로 양자화했습니다: `mode="linear_symmetric"`, `dtype="int4"`, `granularity="per_block"`, `block_size=32`.

변환 환경: macOS 27.0.1 (arm64), Python 3.11.17, torch 2.7.0, transformers 5.20.2, coremltools 9.0

변환 스크립트는 이 저장소에 포함하지 않았습니다.

## 검증 결과

**검증 방법**
- 원본 PyTorch 모델(FP32, CPU)과 Core ML 모델(`coremltools`, `ComputeUnit.CPU_ONLY`)을 비교했습니다.
- 테스트 이미지는 1장(1278×1230 PNG, 저장소 미포함)입니다.
- 개별 구간 비교: 원본 추론 중 각 하위 모듈의 실제 입력을 캡처해 같은 입력을 Core ML 모델에 넣고 출력을 비교했습니다.
- 전체 비교: 원본 Python 파이프라인(전처리·후처리 포함)을 그대로 두고, 세 하위 모듈의 출력만 Core ML 출력으로 교체해 최종 확률을 비교했습니다. **즉 전처리·후처리는 원본 Python 구현을 사용했으며, 별도로 구현한 전처리·후처리는 검증하지 않았습니다.**

**개별 구간 (원본 PyTorch 대비)**

| 구간 | FP32 Core ML 상대 L2 오차 | INT4 상대 L2 오차 | INT4 코사인 유사도 | INT4 최대 절대 오차 |
|---|---|---|---|---|
| Vision | 9.9e-4 | 0.353 | 0.9377 | 3956 |
| Embedder | 1.6e-7 | 0.088 | 0.9962 | 0.548 |
| Language | 5.0e-4 | 0.298 | 0.9547 | 244 |

모든 INT4 출력은 유한값(NaN/Inf 없음)이었습니다. FP32 변환은 원본과 거의 일치하며, 구간별 오차의 대부분은 INT4 양자화에서 발생합니다.

**최종 확률 (같은 이미지 1장)**

| 설정 | 원본 PyTorch | INT4 Core ML 3개 사용 | 절대 차이 |
|---|---|---|---|
| `crops=1` | 0.999835 | 0.999807 | 0.000028 |
| `crops=3` | 0.999857 | 0.999839 | 0.000018 |

> [!CAUTION]
> 최종 확률이 일치한 것은 **이미지 1장, 그것도 확률이 거의 1로 포화된 경우**의 결과입니다. 구간별 오차가 작지 않으므로, 판정 경계(0.5 부근)의 이미지나 다른 이미지에서 같은 수준의 일치를 보장하지 않습니다. 벤치마크 데이터셋 정확도는 측정하지 않았습니다.

**검증하지 않은 항목**
- Swift / iOS / iPadOS 기기에서의 로드 및 추론
- GPU, Neural Engine(`ALL`, `CPU_AND_NE` 등) 연산 장치에서의 결과
- 별도로 구현한 전처리·후처리
- 여러 이미지 및 벤치마크 데이터셋 기준 정확도
- 추론 속도와 메모리 사용량

## 사용 방법

### 다운로드

모델 가중치는 Git LFS로 관리합니다.

```bash
git lfs install
```

```bash
git clone https://github.com/wwwshe/Smaug-CoreML-INT4.git
```

### Python (coremltools)

아래는 각 모델을 호출하는 형태만 보여 주는 예시입니다. 실제 입력은 [전체 추론 흐름](#전체-추론-흐름)대로 만들어야 합니다.

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

# inputs_embeds: [1, 125, 512] = 특수 토큰 임베딩 4개 + projected 121개
hidden = language.predict({
    "inputs_embeds": inputs_embeds,
    "attention_mask": np.ones((1, 125), dtype=np.int32),
})["linear_217"]                                     # [1, 125, 768]

# 이후 mean pooling → L2 정규화 → Dense(768→1) → sigmoid 를 직접 구현
```

### Xcode / Swift

`.mlpackage`를 Xcode 프로젝트에 추가할 수 있으며, 최소 배포 대상은 iOS 18 / macOS 15입니다. 다만 **Swift에서의 로드와 추론, 전처리·후처리 구현은 검증하지 않았습니다.**

## 배포 방식 검토

- GitHub는 50 MiB를 넘는 파일에 경고하고, 100 MiB를 넘는 파일은 일반 Git push를 차단합니다.
- `SmaugVision_INT4`의 `weight.bin`은 99.96 MiB로 한도에 매우 가깝고, `SmaugLanguage_INT4`의 `weight.bin`(82.61 MiB)도 경고 대상입니다.
- 따라서 `.gitattributes`에서 `*.mlpackage`의 `weights/*.bin`과 `*.mlmodel`을 Git LFS로 지정했습니다. 총 용량은 약 193 MB입니다.
- GitHub Free 계정의 Git LFS 무료 한도는 저장 공간 10 GiB, 월 대역폭 10 GiB입니다. clone 1회에 약 0.19 GB가 사용되므로 다운로드가 많을 것으로 예상되면 GitHub Releases에 압축 파일을 함께 올리는 방식을 권장합니다.

## 라이선스

- 이 저장소의 모델 파일은 원본 가중치의 파생물이며 [Apache License 2.0](LICENSE)으로 배포합니다.
- 원본 [Smaug-AI-Image-Detector](https://huggingface.co/abacusai/Smaug-AI-Image-Detector)(Abacus.AI)와 기반 모델 [EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2)(Google DeepMind) 모두 Apache License 2.0입니다.
- 저작자 표시와 수정 내역은 [NOTICE](NOTICE)에 있습니다.
- 기반 모델 카드는 배포자에게 [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) 준수를 요구합니다.
- 이 프로젝트는 Abacus.AI, Google, Apple과 관련이 없으며 이들의 보증을 받지 않았습니다.

## 한계 및 주의사항

- 원본 모델 카드에 따르면 새로운 생성기나 탐지 회피용 생성기는 탐지하지 못할 수 있고, 일부만 AI로 편집된 실제 사진은 실제 사진으로 판정될 수 있습니다.
- 점수는 증거일 뿐 증명이 아닙니다. 사람이나 그 사람의 작업물에 대한 결정의 유일한 근거로 사용하지 마세요.
- INT4 양자화로 구간별 출력이 원본과 달라지므로, 서비스에 사용하기 전에 실제 데이터로 정확도를 따로 확인하세요.

## 인용

원본 모델:

```bibtex
@misc{abacusai2026smaugaiimagedetector,
  title  = {Smaug-AI-Image-Detector},
  author = {Abacus.AI},
  year   = {2026},
  url    = {https://huggingface.co/abacusai/Smaug-AI-Image-Detector}
}
```
