# Weekly Paper Report: 20260315

> ISO 2026-W11




## 1. Weekly Briefing


이번 주 평가 논문 수가 181건으로 전주 대비 54.7% 급증하며 4주 연속 상승세를 이어갔으며, 특히 CS.CV 분야 논문이 109건(전주 대비 +39건)으로 두드러진 증가세를 보였습니다. 다만 평균 점수가 75.78점(-0.64점)으로 소폭 하락하며 양적 확대 속 질적 관리가 필요한 상황입니다.



| Metric | Value | WoW |
|--------|-------|-----|
| Evaluated | 181 | - |
| Tier 1 | 90 | - |
| cs.CV | 109 | 39 |
| Keyword Hits | 0 | 0 |
| Avg Score | 75.78 | -0.64 |



### 4-Week Trend (uptrend)

| Week | Papers | cs.CV | Avg Score |
|------|--------|-------|-----------|

| W1 | 108 | 39 | 74.17 |

| W2 | 107 | 71 | 76.4 |

| W3 | 117 | 70 | 76.42 |

| W4 | 181 | 109 | 75.78 |





### Top Categories

| Category | Count | WoW % |
|----------|-------|-------|

| cs.CV | 109 | 55.7 |

| cs.AI | 40 | 81.8 |

| cs.LG | 24 | 20.0 |

| cs.RO | 22 | -8.3 |

| eess.IV | 8 | 14.3 |





- Graduated Reminds: 274
- Active Reminds: 21




### Tech Radar








#### TF-IDF Keywords

`video` (STABLE) `image` (RISING) `reasoning` (RISING) `data` (RISING) `motion` (RISING) `generation` (RISING) `visual` (RISING) `time` (STABLE) `training` (RISING) `real` (RISING) `multi` (RISING) `diffusion` (NEW) `learning` (RISING) `quality` (RISING) `language` (RISING) `spatial` (NEW) `human` (RISING) `segmentation` (RISING) `detection` (STABLE) `system` (RISING) `knowledge` (DISAPPEARED) `semantic` (DISAPPEARED) `tasks` (DISAPPEARED) `dataset` (DISAPPEARED) `social` (DISAPPEARED) `aware` (DISAPPEARED) `graph` (DISAPPEARED) `memory` (DISAPPEARED) `control` (DISAPPEARED) `object` (DISAPPEARED) `reconstruction` (DISAPPEARED) `classification` (DISAPPEARED) `task` (DISAPPEARED) `accuracy` (DISAPPEARED) `temporal` (DISAPPEARED) `multimodal` (DISAPPEARED) `camera` (DISAPPEARED) `scale` (DISAPPEARED) `images` (DISAPPEARED) `vision` (DISAPPEARED) `alignment` (DISAPPEARED) `long` (DISAPPEARED) 







## 2. Top Papers


### 1. Chain of Event-Centric Causal Thought for Physically Plausible Video Generation

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.09094v1](http://arxiv.org/abs/2603.09094v1)

- Physically Plausible Video Generation (PPVG) has emerged as a promising avenue for modeling real-world physical phenomena. PPVG requires an understanding of commonsense knowledge, which remains a challenge for video diffusion models. Current approaches leverage commonsense reasoning capability of large language models to embed physical concepts into prompts. However, generation models often render physical phenomena as a single moment defined by prompts, due to the lack of conditioning mechanisms for modeling causal progression. In this paper, we view PPVG as generating a sequence of causally connected and dynamically evolving events. To realize this paradigm, we design two key modules: (1) Physics-driven Event Chain Reasoning. This module decomposes the physical phenomena described in prompts into multiple elementary event units, leveraging chain-of-thought reasoning. To mitigate causal ambiguity, we embed physical formulas as constraints to impose deterministic causal dependencies during reasoning. (2) Transition-aware Cross-modal Prompting (TCP). To maintain continuity between events, this module transforms causal event units into temporally aligned vision-language prompts. It summarizes discrete event descriptions to obtain causally consistent narratives, while progressively synthesizing visual keyframes of individual events by interactive editing. Comprehensive experiments on PhyGenBench and VideoPhy benchmarks demonstrate that our framework achieves superior performance in generating physically plausible videos across diverse physical domains. Our code will be released soon.
- 물리적으로 타당한 영상 생성 기술이 스포츠 하이라이트 자동 편집 및 보정에 직접적으로 적용 가능


### 2. Mobile-GS: Real-time Gaussian Splatting for Mobile Devices

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.11531v1](http://arxiv.org/abs/2603.11531v1)

- 3D Gaussian Splatting (3DGS) has emerged as a powerful representation for high-quality rendering across a wide range of applications.However, its high computational demands and large storage costs pose significant challenges for deployment on mobile devices. In this work, we propose a mobile-tailored real-time Gaussian Splatting method, dubbed Mobile-GS, enabling efficient inference of Gaussian Splatting on edge devices. Specifically, we first identify alpha blending as the primary computational bottleneck, since it relies on the time-consuming Gaussian depth sorting process. To solve this issue, we propose a depth-aware order-independent rendering scheme that eliminates the need for sorting, thereby substantially accelerating rendering. Although this order-independent rendering improves rendering speed, it may introduce transparency artifacts in regions with overlapping geometry due to the scarcity of rendering order. To address this problem, we propose a neural view-dependent enhancement strategy, enabling more accurate modeling of view-dependent effects conditioned on viewing direction, 3D Gaussian geometry, and appearance attributes. In this way, Mobile-GS can achieve both high-quality and real-time rendering. Furthermore, to facilitate deployment on memory-constrained mobile platforms, we also introduce first-order spherical harmonics distillation, a neural vector quantization technique, and a contribution-based pruning strategy to reduce the number of Gaussian primitives and compress the 3D Gaussian representation with the assistance of neural networks. Extensive experiments demonstrate that our proposed Mobile-GS achieves real-time rendering and compact model size while preserving high visual quality, making it well-suited for mobile applications.
- 모바일 기용 실시간 가우시안 스플래팅으로 엣지 디바이스에서 고품질 3D 재구현 가능


### 3. AsyncMDE: Real-Time Monocular Depth Estimation via Asynchronous Spatial Memory

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.10438v1](http://arxiv.org/abs/2603.10438v1)

- Foundation-model-based monocular depth estimation offers a viable alternative to active sensors for robot perception, yet its computational cost often prohibits deployment on edge platforms. Existing methods perform independent per-frame inference, wasting the substantial computational redundancy between adjacent viewpoints in continuous robot operation. This paper presents AsyncMDE, an asynchronous depth perception system consisting of a foundation model and a lightweight model that amortizes the foundation model's computational cost over time. The foundation model produces high-quality spatial features in the background, while the lightweight model runs asynchronously in the foreground, fusing cached memory with current observations through complementary fusion, outputting depth estimates, and autoregressively updating the memory. This enables cross-frame feature reuse with bounded accuracy degradation. At a mere 3.83M parameters, it operates at 237 FPS on an RTX 4090, recovering 77% of the accuracy gap to the foundation model while achieving a 25X parameter reduction. Validated across indoor static, dynamic, and synthetic extreme-motion benchmarks, AsyncMDE degrades gracefully between refreshes and achieves 161FPS on a Jetson AGX Orin with TensorRT, clearly demonstrating its feasibility for real-time edge deployment.
- 에지 디바이스용 실시간 심도 추정 시스템으로 스포츠 장면의 3차원 공간 이해에 필수적이며, rk3588 기반 하드웨어에 최적화되어 있음


### 4. HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.12222v1](http://arxiv.org/abs/2603.12222v1)

- Vision Transformers require significant computational resources and memory bandwidth, severely limiting their deployment on edge devices. While recent structured pruning methods successfully reduce theoretical FLOPs, they typically operate at a single structural granularity and rely on complex, multi-stage pipelines with post-hoc thresholding to satisfy sparsity budgets. In this paper, we propose Hierarchical Auto-Pruning (HiAP), a continuous relaxation framework that discovers optimal sub-networks in a single end-to-end training phase without requiring manual importance heuristics or predefined per-layer sparsity targets. HiAP introduces stochastic Gumbel-Sigmoid gates at multiple granularities: macro-gates to prune entire attention heads and FFN blocks, and micro-gates to selectively prune intra-head dimensions and FFN neurons. By optimizing both levels simultaneously, HiAP addresses both the memory-bound overhead of loading large matrices and the compute-bound mathematical operations. HiAP naturally converges to stable sub-networks using a loss function that incorporates both structural feasibility penalties and analytical FLOPs. Extensive experiments on ImageNet demonstrate that HiAP organically discovers highly efficient architectures, and achieves a competitive accuracy-efficiency Pareto frontier for models like DeiT-Small, matching the performance of sophisticated multi-stage methods while significantly simplifying the deployment pipeline.
- 엣지 디바이스에서 AI 모델을 효율적으로 실행하여 실시간 처리 성능 극대화


### 5. Detect Anything in Real Time: From Single-Prompt Segmentation to Multi-Class Detection

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.11441v1](http://arxiv.org/abs/2603.11441v1)

- Recent advances in vision-language modeling have produced promptable detection and segmentation systems that accept arbitrary natural language queries at inference time. Among these, SAM3 achieves state-of-the-art accuracy by combining a ViT-H/14 backbone with cross-modal transformer decoding and learned object queries. However, SAM3 processes a single text prompt per forward pass. Detecting N categories requires N independent executions, each dominated by the 439M-parameter backbone. We present Detect Anything in Real Time (DART), a training-free framework that converts SAM3 into a real-time multi-class detector by exploiting a structural invariant: the visual backbone is class-agnostic, producing image features independent of the text prompt. This allows the backbone computation to be shared between all classes, reducing its cost from O(N) to O(1). Combined with batched multi-class decoding, detection-only inference, and TensorRT FP16 deployment, these optimizations yield 5.6x cumulative speedup at 3 classes, scaling to 25x at 80 classes, without modifying any model weight. On COCO val2017 (5,000 images, 80 classes), DART achieves 55.8 AP at 15.8 FPS (4 classes, 1008x1008) on a single RTX 4080, surpassing purpose-built open-vocabulary detectors trained on millions of box annotations. For extreme latency targets, adapter distillation with a frozen encoder-decoder achieves 38.7 AP with a 13.9 ms backbone. Code and models are available at https://github.com/mkturkcan/DART.
- 실시간 다중 클래스 검색 기술로 스포츠 장면에서 선수와 객체를 즉시 식별하여 하이라이트 자동 생성 가능


### 6. Streaming Autoregressive Video Generation via Diagonal Distillation

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.09488v1](http://arxiv.org/abs/2603.09488v1)

- Large pretrained diffusion models have significantly enhanced the quality of generated videos, and yet their use in real-time streaming remains limited. Autoregressive models offer a natural framework for sequential frame synthesis but require heavy computation to achieve high fidelity. Diffusion distillation can compress these models into efficient few-step variants, but existing video distillation approaches largely adapt image-specific methods that neglect temporal dependencies. These techniques often excel in image generation but underperform in video synthesis, exhibiting reduced motion coherence, error accumulation over long sequences, and a latency-quality trade-off. We identify two factors that result in these limitations: insufficient utilization of temporal context during step reduction and implicit prediction of subsequent noise levels in next-chunk prediction (i.e., exposure bias). To address these issues, we propose Diagonal Distillation, which operates orthogonally to existing approaches and better exploits temporal information across both video chunks and denoising steps. Central to our approach is an asymmetric generation strategy: more steps early, fewer steps later. This design allows later chunks to inherit rich appearance information from thoroughly processed early chunks, while using partially denoised chunks as conditional inputs for subsequent synthesis. By aligning the implicit prediction of subsequent noise levels during chunk generation with the actual inference conditions, our approach mitigates error propagation and reduces oversaturation in long-range sequences. We further incorporate implicit optical flow modeling to preserve motion quality under strict step constraints. Our method generates a 5-second video in 2.61 seconds (up to 31 FPS), achieving a 277.3x speedup over the undistilled model.
- 물리적으로 타당한 스포츠 하이라이트 자동 생성 기술로 경기 장면의 인과적 연결을 유지하며 자연스러운 편집 가능


### 7. CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation

- **Score**: 98.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.09418v1](http://arxiv.org/abs/2603.09418v1)

- State-of-the-art whole-body pose estimators often lack robustness, producing anatomically implausible predictions in challenging scenes. We posit this failure stems from spurious correlations learned from visual context, a problem we formalize using a Structural Causal Model (SCM). The SCM identifies visual context as a confounder that creates a non-causal backdoor path, corrupting the model's reasoning. We introduce the Causal Intervention Graph Pose (CIGPose) framework to address this by approximating the true causal effect between visual evidence and pose. The core of CIGPose is a novel Causal Intervention Module: it first identifies confounded keypoint representations via predictive uncertainty and then replaces them with learned, context-invariant canonical embeddings. These deconfounded embeddings are processed by a hierarchical graph neural network that reasons over the human skeleton at both local and global semantic levels to enforce anatomical plausibility. Extensive experiments show CIGPose achieves a new state-of-the-art on COCO-WholeBody. Notably, our CIGPose-x model achieves 67.0\% AP, surpassing prior methods that rely on extra training data. With the additional UBody dataset, CIGPose-x is further boosted to 67.5\% AP, demonstrating superior robustness and data efficiency. The codes and models are publicly available at https://github.com/53mins/CIGPose.
- 전신 자세 추정 기술은 스포츠 동작 분석에 직접적으로 활용 가능한 핵심 기술이다


### 8. TrainDeeploy: Hardware-Accelerated Parameter-Efficient Fine-Tuning of Small Transformer Models at the Extreme Edge

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.09511v1](http://arxiv.org/abs/2603.09511v1)

- On-device tuning of deep neural networks enables long-term adaptation at the edge while preserving data privacy. However, the high computational and memory demands of backpropagation pose significant challenges for ultra-low-power, memory-constrained extreme-edge devices. These challenges are further amplified for attention-based models due to their architectural complexity and computational scale. We present TrainDeeploy, a framework that unifies efficient inference and on-device training on heterogeneous ultra-low-power System-on-Chips (SoCs). TrainDeeploy provides the first complete on-device training pipeline for extreme-edge SoCs supporting both Convolutional Neural Networks (CNNs) and Transformer models, together with multiple training strategies such as selective layer-wise fine-tuning and Low-Rank Adaptation (LoRA). On a RISC-V-based heterogeneous SoC, we demonstrate the first end-to-end on-device fine-tuning of a Compact Convolutional Transformer (CCT), achieving up to 11 trained images per second. We show that LoRA reduces dynamic memory usage by 23%, decreases the number of trainable parameters and gradients by 15x, and reduces memory transfer volume by 1.6x compared to full backpropagation. TrainDeeploy achieves up to 4.6 FLOP/cycle on CCT (0.28M parameters, 71-126M FLOPs) and up to 13.4 FLOP/cycle on Deep-AE (0.27M parameters, 0.8M FLOPs), while expanding the scope of prior frameworks to support both CNN and Transformer models with parameter-efficient tuning on extreme-edge platforms.
- rk3588 기반 edge device에서 AI 모델 효율적 튜닝을 위한 하드웨어 가속 기술


### 9. CycleULM: A unified label-free deep learning framework for ultrasound localisation microscopy

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.09840v1](http://arxiv.org/abs/2603.09840v1)

- Super-resolution ultrasound via microbubble (MB) localisation and tracking, also known as ultrasound localisation microscopy (ULM), can resolve microvasculature beyond the acoustic diffraction limit. However, significant challenges remain in localisation performance and data acquisition and processing time. Deep learning methods for ULM have shown promise to address these challenges, however, they remain limited by in vivo label scarcity and the simulation-to-reality domain gap. We present CycleULM, the first unified label-free deep learning framework for ULM. CycleULM learns a physics-emulating translation between the real contrast-enhanced ultrasound (CEUS) data domain and a simplified MB-only domain, leveraging the power of CycleGAN without requiring paired ground truth data. With this translation, CycleULM removes dependence on high-fidelity simulators or labelled data, and makes MB localisation and tracking substantially easier. Deployed as modular plug-and-play components within existing pipelines or as an end-to-end processing framework, CycleULM delivers substantial performance gains across both in silico and in vivo datasets. Specifically, CycleULM improves image contrast (contrast-to-noise ratio) by up to 15.3 dB and sharpens CEUS resolution with a 2.5{\times} reduction in the full width at half maximum of the point spread function. CycleULM also improves MB localisation performance, with up to +40% recall, +46% precision, and a -14.0 μm mean localisation error, yielding more faithful vascular reconstructions. Importantly, CycleULM achieves real-time processing throughput at 18.3 frames per second with order-of-magnitude speed-ups (up to ~14.5{\times}). By combining label-free learning, performance enhancement, and computational efficiency, CycleULM provides a practical pathway toward robust, real-time ULM and accelerates its translation to clinical applications.
- 실시간 처리 성능과 물리 모델링 기술이 스포츠 영상 분석에 직접적으로 적용 가능


### 10. LCAMV: High-Accuracy 3D Reconstruction of Color-Varying Objects Using LCA Correction and Minimum-Variance Fusion in Structured Light

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.10456v1](http://arxiv.org/abs/2603.10456v1)

- Accurate 3D reconstruction of colored objects with structured light (SL) is hindered by lateral chromatic aberration (LCA) in optical components and uneven noise characteristics across RGB channels. This paper introduces lateral chromatic aberration correction and minimum-variance fusion (LCAMV), a robust 3D reconstruction method that operates with a single projector-camera pair without additional hardware or acquisition constraints. LCAMV analytically models and pixel-wise compensates LCA in both the projector and camera, then adaptively fuses multi-channel phase data using a Poisson-Gaussian noise model and minimum-variance estimation. Unlike existing methods that require extra hardware or multiple exposures, LCAMV enables fast acquisition. Experiments on planar and non-planar colored surfaces show that LCAMV outperforms grayscale conversion and conventional channel-weighting, reducing depth error by up to 43.6\%. These results establish LCAMV as an effective solution for high-precision 3D reconstruction of nonuniformly colored objects.
- 색상 보정 및 최소 분산 융합 기술로 스포츠 장면의 정밀 3D 재구성 및 영상 보정에 적용 가능






### Graduated Reminds


- **Helios: Real Real-Time Long Video Generation Model** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.04379v1](http://arxiv.org/abs/2603.04379v1)

- **Real Eyes Realize Faster: Gaze Stability and Pupil Novelty for Efficient Egocentric Learning** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.04098v1](http://arxiv.org/abs/2603.04098v1)

- **AI+HW 2035: Shaping the Next Decade** (score: None, graduated: 2026-03-11)
  [http://arxiv.org/abs/2603.05225v1](http://arxiv.org/abs/2603.05225v1)

- **Toward Native ISAC Support in O-RAN Architectures for 6G** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03607v1](http://arxiv.org/abs/2603.03607v1)

- **InfinityStory: Unlimited Video Generation with World Consistency and Character-Aware Shot Transitions** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03646v1](http://arxiv.org/abs/2603.03646v1)

- **Yolo-Key-6D: Single Stage Monocular 6D Pose Estimation with Keypoint Enhancements** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03879v1](http://arxiv.org/abs/2603.03879v1)

- **SSR: A Generic Framework for Text-Aided Map Compression for Localization** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.04272v1](http://arxiv.org/abs/2603.04272v1)

- **Logics-Parsing-Omni Technical Report** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09677v1](http://arxiv.org/abs/2603.09677v1)

- **Exploring Challenges in Developing Edge-Cloud-Native Applications Across Multiple Business Domains** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03738v1](http://arxiv.org/abs/2603.03738v1)

- **Scalable Injury-Risk Screening in Baseball Pitching From Broadcast Video** (score: None, graduated: 2026-03-11)
  [http://arxiv.org/abs/2603.04864v1](http://arxiv.org/abs/2603.04864v1)

- **TrainDeeploy: Hardware-Accelerated Parameter-Efficient Fine-Tuning of Small Transformer Models at the Extreme Edge** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09511v1](http://arxiv.org/abs/2603.09511v1)

- **Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding** (score: None, graduated: 2026-03-11)
  [http://arxiv.org/abs/2603.04977v1](http://arxiv.org/abs/2603.04977v1)

- **Lambdas at the Far Edge: a Tale of Flying Lambdas and Lambdas on Wheels** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.04008v1](http://arxiv.org/abs/2603.04008v1)

- **NS-VLA: Towards Neuro-Symbolic Vision-Language-Action Models** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09542v1](http://arxiv.org/abs/2603.09542v1)

- **Towards Unified Multimodal Interleaved Generation via Group Relative Policy Optimization** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09538v1](http://arxiv.org/abs/2603.09538v1)

- **VisionPangu: A Compact and Fine-Grained Multimodal Assistant with 1.7B Parameters** (score: None, graduated: 2026-03-11)
  [http://arxiv.org/abs/2603.04957v1](http://arxiv.org/abs/2603.04957v1)

- **A Multi-Prototype-Guided Federated Knowledge Distillation Approach in AI-RAN Enabled Multi-Access Edge Computing System** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09727v1](http://arxiv.org/abs/2603.09727v1)

- **Guiding Diffusion-based Reconstruction with Contrastive Signals for Balanced Visual Representation** (score: None, graduated: 2026-03-11)
  [http://arxiv.org/abs/2603.04803v1](http://arxiv.org/abs/2603.04803v1)

- **Two Teachers Better Than One: Hardware-Physics Co-Guided Distributed Scientific Machine Learning** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09032v1](http://arxiv.org/abs/2603.09032v1)

- **A Text-Native Interface for Generative Video Authoring** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09072v1](http://arxiv.org/abs/2603.09072v1)

- **Optimal Short Video Ordering and Transmission Scheduling for Reducing Video Delivery Cost in Peer-to-Peer CDNs** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03938v1](http://arxiv.org/abs/2603.03938v1)

- **From Ideal to Real: Stable Video Object Removal under Imperfect Conditions** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09283v1](http://arxiv.org/abs/2603.09283v1)

- **RIVER: A Real-Time Interaction Benchmark for Video LLMs** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03985v1](http://arxiv.org/abs/2603.03985v1)

- **WikiCLIP: An Efficient Contrastive Baseline for Open-domain Visual Entity Recognition** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09921v1](http://arxiv.org/abs/2603.09921v1)

- **EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.04090v1](http://arxiv.org/abs/2603.04090v1)

- **Fine-grained Motion Retrieval via Joint-Angle Motion Images and Token-Patch Late Interaction** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09930v1](http://arxiv.org/abs/2603.09930v1)

- **Adaptive Enhancement and Dual-Pooling Sequential Attention for Lightweight Underwater Object Detection with YOLOv10** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03807v1](http://arxiv.org/abs/2603.03807v1)

- **A Baseline Study and Benchmark for Few-Shot Open-Set Action Recognition with Feature Residual Discrimination** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.04125v1](http://arxiv.org/abs/2603.04125v1)

- **M3GCLR: Multi-View Mini-Max Infinite Skeleton-Data Game Contrastive Learning For Skeleton-Based Action Recognition** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09367v1](http://arxiv.org/abs/2603.09367v1)

- **DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03744v1](http://arxiv.org/abs/2603.03744v1)

- **Decoder-Free Distillation for Quantized Image Restoration** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09624v1](http://arxiv.org/abs/2603.09624v1)

- **Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09681v1](http://arxiv.org/abs/2603.09681v1)

- **SURE: Semi-dense Uncertainty-REfined Feature Matching** (score: None, graduated: 2026-03-11)
  [http://arxiv.org/abs/2603.04869v1](http://arxiv.org/abs/2603.04869v1)

- **PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09216v1](http://arxiv.org/abs/2603.09216v1)

- **Scaling Dense Event-Stream Pretraining from Visual Foundation Models** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03969v1](http://arxiv.org/abs/2603.03969v1)

- **CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09418v1](http://arxiv.org/abs/2603.09418v1)

- **Trainable Bitwise Soft Quantization for Input Feature Compression** (score: None, graduated: 2026-03-11)
  [http://arxiv.org/abs/2603.05172v1](http://arxiv.org/abs/2603.05172v1)

- **Chain of Event-Centric Causal Thought for Physically Plausible Video Generation** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09094v1](http://arxiv.org/abs/2603.09094v1)

- **HE-VPR: Height Estimation Enabled Aerial Visual Place Recognition Against Scale Variance** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.04050v1](http://arxiv.org/abs/2603.04050v1)

- **Evolving Prompt Adaptation for Vision-Language Models** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09493v1](http://arxiv.org/abs/2603.09493v1)

- **Agentic Peer-to-Peer Networks: From Content Distribution to Capability and Action Sharing** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03753v1](http://arxiv.org/abs/2603.03753v1)

- **CycleULM: A unified label-free deep learning framework for ultrasound localisation microscopy** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09840v1](http://arxiv.org/abs/2603.09840v1)

- **TemporalDoRA: Temporal PEFT for Robust Surgical Video Question Answering** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09696v1](http://arxiv.org/abs/2603.09696v1)

- **Semi-Supervised Generative Learning via Latent Space Distribution Matching** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.04223v1](http://arxiv.org/abs/2603.04223v1)

- **Separators in Enhancing Autoregressive Pretraining for Vision Mamba** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03806v1](http://arxiv.org/abs/2603.03806v1)

- **Semantic Bridging Domains: Pseudo-Source as Test-Time Connector** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03844v1](http://arxiv.org/abs/2603.03844v1)

- **When to Lock Attention: Training-Free KV Control in Video Diffusion** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09657v1](http://arxiv.org/abs/2603.09657v1)

- **M2Diff: Multi-Modality Multi-Task Enhanced Diffusion Model for MRI-Guided Low-Dose PET Enhancement** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09075v1](http://arxiv.org/abs/2603.09075v1)

- **Point Cloud Feature Coding for Object Detection over an Error-Prone Cloud-Edge Collaborative System** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03890v1](http://arxiv.org/abs/2603.03890v1)

- **Streaming Autoregressive Video Generation via Diagonal Distillation** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09488v1](http://arxiv.org/abs/2603.09488v1)

- **Architecture and evaluation protocol for transformer-based visual object tracking in UAV applications** (score: None, graduated: 2026-03-10)
  [http://arxiv.org/abs/2603.03904v1](http://arxiv.org/abs/2603.03904v1)

- **DCAU-Net: Differential Cross Attention and Channel-Spatial Feature Fusion for Medical Image Segmentation** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09530v1](http://arxiv.org/abs/2603.09530v1)

- **Person Detection and Tracking from an Overhead Crane LiDAR** (score: None, graduated: 2026-03-11)
  [http://arxiv.org/abs/2603.04938v1](http://arxiv.org/abs/2603.04938v1)

- **MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging** (score: None, graduated: 2026-03-12)
  [http://arxiv.org/abs/2603.09116v1](http://arxiv.org/abs/2603.09116v1)







### Notable Authors

| Author | Papers | Avg Score |
|--------|--------|-----------|

| Zixuan Wang | 4 | 81.2 |

| Hao Li | 3 | 86.1 |

| Jae-Sang Hyun | 3 | 84.7 |

| Fangfu Liu | 2 | 96.0 |

| Yueqi Duan | 2 | 96.0 |

| Yi Wang | 2 | 92.0 |

| Jie Zhang | 2 | 90.8 |

| Cewu Lu | 2 | 90.0 |

| Kyomin Sohn | 2 | 88.0 |

| Xin Gu | 2 | 88.0 |







## 3. Trends



### cappic-ai

| Date | Avg Score |
|------|-----------|

| 2026-02-16 | 78.09 |

| 2026-02-17 | 72.39 |

| 2026-02-18 | 74.52 |

| 2026-02-18 | 75.53 |

| 2026-02-19 | 62.97 |

| 2026-02-23 | 75.83 |

| 2026-02-24 | 85.2 |

| 2026-02-25 | 68.0 |

| 2026-02-26 | 76.58 |

| 2026-03-02 | 76.66 |

| 2026-03-03 | 79.6 |

| 2026-03-04 | 75.8 |

| 2026-03-05 | 77.33 |

| 2026-03-10 | 75.92 |

| 2026-03-11 | 75.21 |

| 2026-03-12 | 76.22 |



