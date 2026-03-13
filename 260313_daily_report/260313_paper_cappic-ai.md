# CAPP!C_AI 논문 리포트 (2026-03-13)

> 수집 92 | 필터 86 | 폐기 7 | 평가 82 | 출력 61 | 기준 50점

검색 윈도우: 2026-03-12T00:00:00+00:00 ~ 2026-03-13T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 37

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: Detect Anything in Real Time: From Single-Prompt Segmentation to Multi-Class Detection

- arXiv: http://arxiv.org/abs/2603.11441v1
- PDF: https://arxiv.org/pdf/2603.11441v1
- 코드: https://github.com/mkturkcan/DART
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 100.0 (llm_adjusted:100 = base:88 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Recent advances in vision-language modeling have produced promptable detection and segmentation systems that accept arbitrary natural language queries at inference time. Among these, SAM3 achieves state-of-the-art accuracy by combining a ViT-H/14 backbone with cross-modal transformer decoding and learned object queries. However, SAM3 processes a single text prompt per forward pass. Detecting N categories requires N independent executions, each dominated by the 439M-parameter backbone. We present Detect Anything in Real Time (DART), a training-free framework that converts SAM3 into a real-time multi-class detector by exploiting a structural invariant: the visual backbone is class-agnostic, producing image features independent of the text prompt. This allows the backbone computation to be shared between all classes, reducing its cost from O(N) to O(1). Combined with batched multi-class decoding, detection-only inference, and TensorRT FP16 deployment, these optimizations yield 5.6x cumulative speedup at 3 classes, scaling to 25x at 80 classes, without modifying any model weight. On COCO val2017 (5,000 images, 80 classes), DART achieves 55.8 AP at 15.8 FPS (4 classes, 1008x1008) on a single RTX 4080, surpassing purpose-built open-vocabulary detectors trained on millions of box annotations. For extreme latency targets, adapter distillation with a frozen encoder-decoder achieves 38.7 AP with a 13.9 ms backbone. Code and models are available at https://github.com/mkturkcan/DART.

**선정 근거**
실시간 다중 클래스 검색 기술로 스포츠 장면에서 선수와 객체를 즉시 식별하여 하이라이트 자동 생성 가능

**활용 인사이트**
경기 중 선수 동작, 공 위치, 득점 순간 등을 실시간으로 감지하여 자동으로 하이라이트 편집 기능 구현

## 2위: Mobile-GS: Real-time Gaussian Splatting for Mobile Devices

- arXiv: http://arxiv.org/abs/2603.11531v1
- PDF: https://arxiv.org/pdf/2603.11531v1
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 100.0 (llm_adjusted:100 = base:95 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
3D Gaussian Splatting (3DGS) has emerged as a powerful representation for high-quality rendering across a wide range of applications.However, its high computational demands and large storage costs pose significant challenges for deployment on mobile devices. In this work, we propose a mobile-tailored real-time Gaussian Splatting method, dubbed Mobile-GS, enabling efficient inference of Gaussian Splatting on edge devices. Specifically, we first identify alpha blending as the primary computational bottleneck, since it relies on the time-consuming Gaussian depth sorting process. To solve this issue, we propose a depth-aware order-independent rendering scheme that eliminates the need for sorting, thereby substantially accelerating rendering. Although this order-independent rendering improves rendering speed, it may introduce transparency artifacts in regions with overlapping geometry due to the scarcity of rendering order. To address this problem, we propose a neural view-dependent enhancement strategy, enabling more accurate modeling of view-dependent effects conditioned on viewing direction, 3D Gaussian geometry, and appearance attributes. In this way, Mobile-GS can achieve both high-quality and real-time rendering. Furthermore, to facilitate deployment on memory-constrained mobile platforms, we also introduce first-order spherical harmonics distillation, a neural vector quantization technique, and a contribution-based pruning strategy to reduce the number of Gaussian primitives and compress the 3D Gaussian representation with the assistance of neural networks. Extensive experiments demonstrate that our proposed Mobile-GS achieves real-time rendering and compact model size while preserving high visual quality, making it well-suited for mobile applications.

**선정 근거**
모바일 기용 실시간 가우시안 스플래팅으로 엣지 디바이스에서 고품질 3D 재구현 가능

**활용 인사이트**
스포츠 장면을 3D로 재구현하여 다각도에서 볼 수 있는 VR/AR 콘텐츠 생성 및 영상 보정 기능 강화

## 3위: HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers

- arXiv: http://arxiv.org/abs/2603.12222v1
- PDF: https://arxiv.org/pdf/2603.12222v1
- 발행일: 2026-03-12
- 카테고리: cs.CV, cs.LG
- 점수: final 100.0 (llm_adjusted:100 = base:92 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Vision Transformers require significant computational resources and memory bandwidth, severely limiting their deployment on edge devices. While recent structured pruning methods successfully reduce theoretical FLOPs, they typically operate at a single structural granularity and rely on complex, multi-stage pipelines with post-hoc thresholding to satisfy sparsity budgets. In this paper, we propose Hierarchical Auto-Pruning (HiAP), a continuous relaxation framework that discovers optimal sub-networks in a single end-to-end training phase without requiring manual importance heuristics or predefined per-layer sparsity targets. HiAP introduces stochastic Gumbel-Sigmoid gates at multiple granularities: macro-gates to prune entire attention heads and FFN blocks, and micro-gates to selectively prune intra-head dimensions and FFN neurons. By optimizing both levels simultaneously, HiAP addresses both the memory-bound overhead of loading large matrices and the compute-bound mathematical operations. HiAP naturally converges to stable sub-networks using a loss function that incorporates both structural feasibility penalties and analytical FLOPs. Extensive experiments on ImageNet demonstrate that HiAP organically discovers highly efficient architectures, and achieves a competitive accuracy-efficiency Pareto frontier for models like DeiT-Small, matching the performance of sophisticated multi-stage methods while significantly simplifying the deployment pipeline.

**선정 근거**
엣지 디바이스에서 AI 모델을 효율적으로 실행하여 실시간 처리 성능 극대화

**활용 인사이트**
rk3588 엣지 디바이스에서 비전 트랜스포머 모델을 경량화하여 저지연 실시간 분석 구현

## 4위: OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams

- arXiv: http://arxiv.org/abs/2603.12265v1
- PDF: https://arxiv.org/pdf/2603.12265v1
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Modern visual agents require representations that are general, causal, and physically structured to operate in real-time streaming environments. However, current vision foundation models remain fragmented, specializing narrowly in image semantic perception, offline temporal modeling, or spatial geometry. This paper introduces OmniStream, a unified streaming visual backbone that effectively perceives, reconstructs, and acts from diverse visual inputs. By incorporating causal spatiotemporal attention and 3D rotary positional embeddings (3D-RoPE), our model supports efficient, frame-by-frame online processing of video streams via a persistent KV-cache. We pre-train OmniStream using a synergistic multi-task framework coupling static and temporal representation learning, streaming geometric reconstruction, and vision-language alignment on 29 datasets. Extensive evaluations show that, even with a strictly frozen backbone, OmniStream achieves consistently competitive performance with specialized experts across image and video probing, streaming geometric reconstruction, complex video and spatial reasoning, as well as robotic manipulation (unseen at training). Rather than pursuing benchmark-specific dominance, our work demonstrates the viability of training a single, versatile vision backbone that generalizes across semantic, spatial, and temporal reasoning, i.e., a more meaningful step toward general-purpose visual understanding for interactive and embodied agents.

**선정 근거**
연속 스트림에서 지각, 재구성 및 행동을 통합하는 실시간 비전 백본 기술

**활용 인사이트**
스포츠 촬영 및 분석에 필요한 실시간 비전 처리, 공간 이해, 동작 분석을 통합적으로 지원하여 하드웨어 통합에 최적화

## 5위: Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training

- arXiv: http://arxiv.org/abs/2603.12255v1
- PDF: https://arxiv.org/pdf/2603.12255v1
- 발행일: 2026-03-12
- 카테고리: cs.CV, cs.LG
- 점수: final 96.0 (llm_adjusted:95 = base:82 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Humans perceive and understand real-world spaces through a stream of visual observations. Therefore, the ability to streamingly maintain and update spatial evidence from potentially unbounded video streams is essential for spatial intelligence. The core challenge is not simply longer context windows but how spatial information is selected, organized, and retained over time. In this paper, we propose Spatial-TTT towards streaming visual-based spatial intelligence with test-time training (TTT), which adapts a subset of parameters (fast weights) to capture and organize spatial evidence over long-horizon scene videos. Specifically, we design a hybrid architecture and adopt large-chunk updates parallel with sliding-window attention for efficient spatial video processing. To further promote spatial awareness, we introduce a spatial-predictive mechanism applied to TTT layers with 3D spatiotemporal convolution, which encourages the model to capture geometric correspondence and temporal continuity across frames. Beyond architecture design, we construct a dataset with dense 3D spatial descriptions, which guides the model to update its fast weights to memorize and organize global 3D spatial signals in a structured manner. Extensive experiments demonstrate that Spatial-TTT improves long-horizon spatial understanding and achieves state-of-the-art performance on video spatial benchmarks. Project page: https://liuff19.github.io/Spatial-TTT.

**선정 근거**
스트리밍 비전 처리 기술이 스포츠 촬영 및 공간 이해에 직접적으로 적용 가능합니다

## 6위: BiGain: Unified Token Compression for Joint Generation and Classification

- arXiv: http://arxiv.org/abs/2603.12240v1
- PDF: https://arxiv.org/pdf/2603.12240v1
- 코드: https://github.com/Greenoso/BiGain
- 발행일: 2026-03-12
- 카테고리: cs.CV, cs.LG
- 점수: final 96.0 (llm_adjusted:95 = base:82 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Acceleration methods for diffusion models (e.g., token merging or downsampling) typically optimize synthesis quality under reduced compute, yet often ignore discriminative capacity. We revisit token compression with a joint objective and present BiGain, a training-free, plug-and-play framework that preserves generation quality while improving classification in accelerated diffusion models. Our key insight is frequency separation: mapping feature-space signals into a frequency-aware representation disentangles fine detail from global semantics, enabling compression that respects both generative fidelity and discriminative utility. BiGain reflects this principle with two frequency-aware operators: (1) Laplacian-gated token merging, which encourages merges among spectrally smooth tokens while discouraging merges of high-contrast tokens, thereby retaining edges and textures; and (2) Interpolate-Extrapolate KV Downsampling, which downsamples keys/values via a controllable interextrapolation between nearest and average pooling while keeping queries intact, thereby conserving attention precision. Across DiT- and U-Net-based backbones and ImageNet-1K, ImageNet-100, Oxford-IIIT Pets, and COCO-2017, our operators consistently improve the speed-accuracy trade-off for diffusion-based classification, while maintaining or enhancing generation quality under comparable acceleration. For instance, on ImageNet-1K, with 70% token merging on Stable Diffusion 2.0, BiGain increases classification accuracy by 7.15% while improving FID by 0.34 (1.85%). Our analyses indicate that balanced spectral retention, preserving high-frequency detail and low/mid-frequency semantics, is a reliable design rule for token compression in diffusion models. To our knowledge, BiGain is the first framework to jointly study and advance both generation and classification under accelerated diffusion, supporting lower-cost deployment.

**선정 근거**
엣지 장치에서 실시간 영역 분할 기술은 스포츠 촬영 및 분석에 직접적으로 적용 가능하다.

**활용 인사이트**
PicoSAM3을 활용해 경기 중 주요 선수나 공의 움직임을 실시간으로 추적하고 분석할 수 있다.

## 7위: InSpatio-WorldFM: An Open-Source Real-Time Generative Frame Model

- arXiv: http://arxiv.org/abs/2603.11911v1
- PDF: https://arxiv.org/pdf/2603.11911v1
- 코드: https://inspatio.github.io/worldfm/
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 94.4 (llm_adjusted:93 = base:80 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
We present InSpatio-WorldFM, an open-source real-time frame model for spatial intelligence. Unlike video-based world models that rely on sequential frame generation and incur substantial latency due to window-level processing, InSpatio-WorldFM adopts a frame-based paradigm that generates each frame independently, enabling low-latency real-time spatial inference. By enforcing multi-view spatial consistency through explicit 3D anchors and implicit spatial memory, the model preserves global scene geometry while maintaining fine-grained visual details across viewpoint changes. We further introduce a progressive three-stage training pipeline that transforms a pretrained image diffusion model into a controllable frame model and finally into a real-time generator through few-step distillation. Experimental results show that InSpatio-WorldFM achieves strong multi-view consistency while supporting interactive exploration on consumer-grade GPUs, providing an efficient alternative to traditional video-based world models for real-time world simulation.

**선정 근거**
Efficient video tokenization technology applicable for sports highlight extraction on edge devices

**활용 인사이트**
EVATok을 사용해 스포츠 영상의 동적 부분에 토큰을 집중 배분하여 하이라이트 추출 효율을 극대화할 수 있다.

## 8위: SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics

- arXiv: http://arxiv.org/abs/2603.12193v1
- PDF: https://arxiv.org/pdf/2603.12193v1
- 발행일: 2026-03-12
- 카테고리: cs.RO, cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Active perception and manipulation are crucial for robots to interact with complex scenes. Existing methods struggle to unify semantic-driven active perception with robust, viewpoint-invariant execution. We propose SaPaVe, an end-to-end framework that jointly learns these capabilities in a data-efficient manner. Our approach decouples camera and manipulation actions rather than placing them in a shared action space, and follows a bottom-up training strategy: we first train semantic camera control on a large-scale dataset, then jointly optimize both action types using hybrid data. To support this framework, we introduce ActiveViewPose-200K, a dataset of 200k image-language-camera movement pairs for semantic camera movement learning, and a 3D geometry-aware module that improves execution robustness under dynamic viewpoints. We also present ActiveManip-Bench, the first benchmark for evaluating active manipulation beyond fixed-view settings. Extensive experiments in both simulation and real-world environments show that SaPaVe outperforms recent vision-language-action models such as GR00T N1 and \(π_0\), achieving up to 31.25\% higher success rates in real-world tasks. These results show that tightly coupled perception and execution, when trained with decoupled yet coordinated strategies, enable efficient and generalizable active manipulation. Project page: https://lmzpai.github.io/SaPaVe

**선정 근거**
가속화된 확산 모델 기술이 엣지 디바이스에서 스포츠 영상 처리에 적용 가능

**활용 인사이트**
BiGain 프레임워크로 스포츠 영상의 품질을 유지하면서 실시간으로 보정하고 분석을 동시에 수행할 수 있다.

## 9위: PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation

- arXiv: http://arxiv.org/abs/2603.11917v1
- PDF: https://arxiv.org/pdf/2603.11917v1
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Real-time, on-device segmentation is critical for latency-sensitive and privacy-aware applications such as smart glasses and Internet-of-Things devices. We introduce PicoSAM3, a lightweight promptable visual segmentation model optimized for edge and in-sensor execution, including deployment on the Sony IMX500 vision sensor. PicoSAM3 has 1.3 M parameters and combines a dense CNN architecture with region of interest prompt encoding, Efficient Channel Attention, and knowledge distillation from SAM2 and SAM3. On COCO and LVIS, PicoSAM3 achieves 65.45% and 64.01% mIoU, respectively, outperforming existing SAM-based and edge-oriented baselines at similar or lower complexity. The INT8 quantized model preserves accuracy with negligible degradation while enabling real-time in-sensor inference at 11.82 ms latency on the IMX500, fully complying with its memory and operator constraints. Ablation studies show that distillation from large SAM models yields up to +14.5% mIoU improvement over supervised training and demonstrate that high-quality, spatially flexible promptable segmentation is feasible directly at the sensor level.

**선정 근거**
PicoSAM3의 실시간 영역 분할 기술은 스포츠 장면에서 선수나 중요 객체를 식별하는 데 직접적으로 활용 가능하며, 엣지 장치에서의 저지연 처리가 실시간 하이라이트 생성에 필수적입니다.

**활용 인사이트**
PicoSAM3을 rk3588에 탑재하여 경기 중 선수나 공을 실시간으로 추적하고, 이를 기반으로 자동으로 하이라이트 장면을 식별하고 편집할 수 있습니다. 11.82ms의 낮은 지연 시간은 실시간 분석을 가능하게 하여 훈련이나 경기 중에도 즉각적인 피드백을 제공할 수 있습니다.

## 10위: EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation

- arXiv: http://arxiv.org/abs/2603.12267v1
- PDF: https://arxiv.org/pdf/2603.12267v1
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Autoregressive (AR) video generative models rely on video tokenizers that compress pixels into discrete token sequences. The length of these token sequences is crucial for balancing reconstruction quality against downstream generation computational cost. Traditional video tokenizers apply a uniform token assignment across temporal blocks of different videos, often wasting tokens on simple, static, or repetitive segments while underserving dynamic or complex ones. To address this inefficiency, we introduce $\textbf{EVATok}$, a framework to produce $\textbf{E}$fficient $\textbf{V}$ideo $\textbf{A}$daptive $\textbf{Tok}$enizers. Our framework estimates optimal token assignments for each video to achieve the best quality-cost trade-off, develops lightweight routers for fast prediction of these optimal assignments, and trains adaptive tokenizers that encode videos based on the assignments predicted by routers. We demonstrate that EVATok delivers substantial improvements in efficiency and overall quality for video reconstruction and downstream AR generation. Enhanced by our advanced training recipe that integrates video semantic encoders, EVATok achieves superior reconstruction and state-of-the-art class-to-video generation on UCF-101, with at least 24.4% savings in average token usage compared to the prior state-of-the-art LARP and our fixed-length baseline.

**선정 근거**
Efficient video tokenization technology applicable for sports highlight extraction on edge devices

## 11위: Multimodal Emotion Recognition via Bi-directional Cross-Attention and Temporal Modeling

- arXiv: http://arxiv.org/abs/2603.11971v1
- PDF: https://arxiv.org/pdf/2603.11971v1
- 발행일: 2026-03-12
- 카테고리: cs.CV, cs.AI
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Emotion recognition in in-the-wild video data remains a challenging problem due to large variations in facial appearance, head pose, illumination, background noise, and the inherently dynamic nature of human affect. Relying on a single modality, such as facial expressions or speech, is often insufficient to capture these complex emotional cues. To address this issue, we propose a multimodal emotion recognition framework for the Expression (EXPR) Recognition task in the 10th Affective Behavior Analysis in-the-wild (ABAW) Challenge.   Our approach leverages large-scale pre-trained models, namely CLIP for visual encoding and Wav2Vec 2.0 for audio representation learning, as frozen backbone networks. To model temporal dependencies in facial expression sequences, we employ a Temporal Convolutional Network (TCN) over fixed-length video windows. In addition, we introduce a bi-directional cross-attention fusion module, in which visual and audio features interact symmetrically to enhance cross-modal contextualization and capture complementary emotional information. A lightweight classification head is then used for final emotion prediction. We further incorporate a text-guided contrastive objective based on CLIP text features to encourage semantically aligned visual representations.   Experimental results on the ABAW 10th EXPR benchmark show that the proposed framework provides a strong multimodal baseline and achieves improved performance over unimodal modeling. These results demonstrate the effectiveness of combining temporal visual modeling, audio representation learning, and cross-modal fusion for robust emotion recognition in unconstrained real-world environments.

**선정 근거**
다중 모달 접근법과 시간적 모델링이 스포츠 선수의 감정 반응과 경기 상황 분석에 적용 가능하여 경기 전략 수립에 도움을 줄 수 있습니다.

**활용 인사이트**
CLIP과 Wav2Vec 2.0을 결합한 비주얼-오디오 크로스 어텐션 모델을 스포츠 경기 영상에 적용하여 선수들의 감정 패턴을 실시간으로 분석하고 경기 흐름을 예측할 수 있습니다.

## 12위: Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning

- arXiv: http://arxiv.org/abs/2603.11460v1
- PDF: https://arxiv.org/pdf/2603.11460v1
- 코드: https://github.com/ermitaju1/STaRC
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 90.4 (llm_adjusted:88 = base:85 + bonus:+3)
- 플래그: 코드 공개

**개요**
Existing retrieval-augmented approaches for Dense Video Captioning (DVC) often fail to achieve accurate temporal segmentation aligned with true event boundaries, as they rely on heuristic strategies that overlook ground truth event boundaries. The proposed framework, \textbf{STaRC}, overcomes this limitation by supervising frame-level saliency through a highlight detection module. Note that the highlight detection module is trained on binary labels derived directly from DVC ground truth annotations without the need for additional annotation. We also propose to utilize the saliency scores as a unified temporal signal that drives retrieval via saliency-guided segmentation and informs caption generation through explicit Saliency Prompts injected into the decoder. By enforcing saliency-constrained segmentation, our method produces temporally coherent segments that align closely with actual event transitions, leading to more accurate retrieval and contextually grounded caption generation. We conduct comprehensive evaluations on the YouCook2 and ViTT benchmarks, where STaRC achieves state-of-the-art performance across most of the metrics. Our code is available at https://github.com/ermitaju1/STaRC

**선정 근거**
시각적 중요도를 기반으로 한 하이라이트 감지 기술은 스포츠 하이라이트 편집에 직접적으로 적용 가능하여 자동으로 주요 장면을 추출하고 편집할 수 있습니다.

**활용 인사이트**
STaRC 프레임워크를 활용하여 스포츠 경기 영상에서 실시간으로 중요한 순간을 감지하고 자동으로 하이라이트 영상을 생성하며, 이를 SNS 플랫폼과 연동하여 콘텐츠를 공유할 수 있습니다.

## 13위: $Ψ_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation

- arXiv: http://arxiv.org/abs/2603.12263v1
- PDF: https://arxiv.org/pdf/2603.12263v1
- 발행일: 2026-03-12
- 카테고리: cs.RO
- 점수: final 88.8 (llm_adjusted:86 = base:78 + bonus:+8)
- 플래그: 실시간, 코드 공개

**개요**
We introduce $Ψ_0$ (Psi-Zero), an open foundation model to address challenging humanoid loco-manipulation tasks. While existing approaches often attempt to address this fundamental problem by co-training on large and diverse human and humanoid data, we argue that this strategy is suboptimal due to the fundamental kinematic and motion disparities between humans and humanoid robots. Therefore, data efficiency and model performance remain unsatisfactory despite the considerable data volume. To address this challenge, \ours\;decouples the learning process to maximize the utility of heterogeneous data sources. Specifically, we propose a staged training paradigm with different learning objectives: First, we autoregressively pre-train a VLM backbone on large-scale egocentric human videos to acquire generalizable visual-action representations. Then, we post-train a flow-based action expert on high-quality humanoid robot data to learn precise robot joint control. Our research further identifies a critical yet often overlooked data recipe: in contrast to approaches that scale with noisy Internet clips or heterogeneous cross-embodiment robot datasets, we demonstrate that pre-training on high-quality egocentric human manipulation data followed by post-training on domain-specific real-world humanoid trajectories yields superior performance. Extensive real-world experiments demonstrate that \ours\ achieves the best performance using only about 800 hours of human video data and 30 hours of real-world robot data, outperforming baselines pre-trained on more than 10$\times$ as much data by over 40\% in overall success rate across multiple tasks. We will open-source the entire ecosystem to the community, including a data processing and training pipeline, a humanoid foundation model, and a real-time action inference engine.

**선정 근거**
인간형 로봇의 움직임 및 조작을 위한 오픈소스 기반 모델은 향후 스포츠 활동과 상호작용하는 피지컬 AI 하드웨어 개발에 기반이 될 수 있습니다.

**활용 인사이트**
Ψ_0 모델의 단계적 학습 패러다임을 차용하여 스포츠 영상 데이터로 사전 학습 후, 실제 스포츠 장치 데이터로 후 학습하여 스포츠 특화 AI 하드웨어를 개발할 수 있습니다.

## 14위: CFD-HAR: User-controllable Privacy through Conditional Feature Disentanglement

- arXiv: http://arxiv.org/abs/2603.11526v1
- PDF: https://arxiv.org/pdf/2603.11526v1
- 발행일: 2026-03-12
- 카테고리: cs.LG
- 점수: final 88.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Modern wearable and mobile devices are equipped with inertial measurement units (IMUs). Human Activity Recognition (HAR) applications running on such devices use machine-learning-based, data-driven techniques that leverage such sensor data. However, sensor-data-driven HAR deployments face two critical challenges: protecting sensitive user information embedded in sensor data in accordance with users' privacy preferences and maintaining high recognition performance with limited labeled samples. This paper proposes a technique for user-controllable privacy through feature disentanglement-based representation learning at the granular level for dynamic privacy filtering. We also compare the efficacy of our technique against few-shot HAR using autoencoder-based representation learning. We analyze their architectural designs, learning objectives, privacy guarantees, data efficiency, and suitability for edge Internet of Things (IoT) deployment. Our study shows that CFD-based HAR provides explicit, tunable privacy protection controls by separating activity and sensitive attributes in the latent space, whereas autoencoder-based few-shot HAR offers superior label efficiency and lightweight adaptability but lacks inherent privacy safeguards. We further examine the security implications of both approaches in continual IoT settings, highlighting differences in susceptibility to representation leakage and embedding-level attacks. The analysis reveals that neither paradigm alone fully satisfies the emerging requirements of next-generation IoT HAR systems. We conclude by outlining research directions toward unified frameworks that jointly optimize privacy preservation, few-shot adaptability, and robustness for trustworthy IoT intelligence.

**선정 근거**
사용자 프라이버시 보호 기능을 갖춘 인간 활동 인식 기술은 스포츠 동작 분석에 직접적으로 적용 가능하며 개인정보 보호와 분석 성능을 동시에 만족시킵니다.

**활용 인사이트**
CFD-HAR 기술을 스포츠 선수의 동작 분석에 적용하여 민감한 개인정보는 분리하고 동작 패턴만 분석함으로써 선수 개인정보 보호와 동작 분석 효율성을 동시에 높일 수 있습니다.

## 15위: Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception

- arXiv: http://arxiv.org/abs/2603.11556v1
- PDF: https://arxiv.org/pdf/2603.11556v1
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 88.0 (llm_adjusted:85 = base:85 + bonus:+0)

**개요**
Image aesthetic enhancement aims to perceive aesthetic deficiencies in images and perform corresponding editing operations, which is highly challenging and requires the model to possess creativity and aesthetic perception capabilities. Although recent advancements in image editing models have significantly enhanced their controllability and flexibility, they struggle with enhancing image aesthetic. The primary challenges are twofold: first, following editing instructions with aesthetic perception is difficult, and second, there is a scarcity of "perfectly-paired" images that have consistent content but distinct aesthetic qualities. In this paper, we propose Dual-supervised Image Aesthetic Enhancement (DIAE), a diffusion-based generative model with multimodal aesthetic perception. First, DIAE incorporates Multimodal Aesthetic Perception (MAP) to convert the ambiguous aesthetic instruction into explicit guidance by (i) employing detailed, standardized aesthetic instructions across multiple aesthetic attributes, and (ii) utilizing multimodal control signals derived from text-image pairs that maintain consistency within the same aesthetic attribute. Second, to mitigate the lack of "perfectly-paired" images, we collect "imperfectly-paired" dataset called IIAEData, consisting of images with varying aesthetic qualities while sharing identical semantics. To better leverage the weak matching characteristics of IIAEData during training, a dual-branch supervision framework is also introduced for weakly supervised image aesthetic enhancement. Experimental results demonstrate that DIAE outperforms the baselines and obtains superior image aesthetic scores and image content consistency scores.

**선정 근거**
이미지 미적 향상 기술이 스포츠 영상 보정에 직접적으로 적용 가능하며 촬영된 영상의 품질을 향상시켜 시각적 효과를 극대화할 수 있습니다.

**활용 인사이트**
DIAE 모델을 스포츠 경기 영상에 적용하여 실시간으로 영상의 미적 품질을 향상시키고, 다중 모달 감지를 통해 경기의 중요한 순간을 강조하여 시각적 하이라이트를 생성할 수 있습니다.

## 16위: Bridging Discrete Marks and Continuous Dynamics: Dual-Path Cross-Interaction for Marked Temporal Point Processes

- arXiv: http://arxiv.org/abs/2603.11462v1
- PDF: https://arxiv.org/pdf/2603.11462v1
- 코드: https://github.com/AONE-NLP/NEXTPP
- 발행일: 2026-03-12
- 카테고리: cs.LG, cs.AI
- 점수: final 88.0 (llm_adjusted:85 = base:82 + bonus:+3)
- 플래그: 코드 공개

**개요**
Predicting irregularly spaced event sequences with discrete marks poses significant challenges due to the complex, asynchronous dependencies embedded within continuous-time data streams.Existing sequential approaches capture dependencies among event tokens but ignore the continuous evolution between events, while Neural Ordinary Differential Equation (Neural ODE) methods model smooth dynamics yet fail to account for how event types influence future timing.To overcome these limitations, we propose NEXTPP, a dual-channel framework that unifies discrete and continuous representations via Event-granular Neural Evolution with Cross-Interaction for Marked Temporal Point Processes. Specifically, NEXTPP encodes discrete event marks via a self-attention mechanism, simultaneously evolving a latent continuous-time state using a Neural ODE. These parallel streams are then fused through a crossattention module to enable explicit bidirectional interaction between continuous and discrete representations. The fused representations drive the conditional intensity function of the neural Hawkes process, while an iterative thinning sampler is employed to generate future events. Extensive evaluations on five real-world datasets demonstrate that NEXTPP consistently outperforms state-of-the-art models. The source code can be found at https://github.com/AONE-NLP/NEXTPP.

**선정 근거**
이 논문은 스포츠 경기의 주요 순간을 자동으로 식별하는 데 적용 가능한 시간적 이벤트 예측 기술을 제안합니다. 이는 우리 AI 촬영 에지 디바이스가 자동으로 중요 장면을 포착하고 편집하는 핵심 기술이 될 수 있습니다.

**활용 인사이트**
NEXTPP 프레임워크를 이용해 경기 중 발생하는 불규칙한 이벤트 시퀀스를 실시간으로 분석하여 중요한 순간을 예측하고, 이를 바탕으로 카메라가 자동으로 초점을 맞추고 하이라이트 영상을 생성할 수 있습니다. 이를 통해 사용자는 수동으로 중요 장면을 선택할 필요 없이 최적의 콘텐츠를 자동으로 얻을 수 있습니다.

## 17위: SoulX-LiveAct: Towards Hour-Scale Real-Time Human Animation with Neighbor Forcing and ConvKV Memory

- arXiv: http://arxiv.org/abs/2603.11746v1
- PDF: https://arxiv.org/pdf/2603.11746v1
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 86.4 (llm_adjusted:83 = base:78 + bonus:+5)
- 플래그: 실시간

**개요**
Autoregressive (AR) diffusion models offer a promising framework for sequential generation tasks such as video synthesis by combining diffusion modeling with causal inference. Although they support streaming generation, existing AR diffusion methods struggle to scale efficiently. In this paper, we identify two key challenges in hour-scale real-time human animation. First, most forcing strategies propagate sample-level representations with mismatched diffusion states, causing inconsistent learning signals and unstable convergence. Second, historical representations grow unbounded and lack structure, preventing effective reuse of cached states and severely limiting inference efficiency. To address these challenges, we propose Neighbor Forcing, a diffusion-step-consistent AR formulation that propagates temporally adjacent frames as latent neighbors under the same noise condition. This design provides a distribution-aligned and stable learning signal while preserving drifting throughout the AR chain. Building upon this, we introduce a structured ConvKV memory mechanism that compresses the keys and values in causal attention into a fixed-length representation, enabling constant-memory inference and truly infinite video generation without relying on short-term motion-frame memory. Extensive experiments demonstrate that our approach significantly improves training convergence, hour-scale generation quality, and inference efficiency compared to existing AR diffusion methods. Numerically, LiveAct enables hour-scale real-time human animation and supports 20 FPS real-time streaming inference on as few as two NVIDIA H100 or H200 GPUs. Quantitative results demonstrate that our method attains state-of-the-art performance in lip-sync accuracy, human animation quality, and emotional expressiveness, with the lowest inference cost.

**선정 근거**
실시간 인간 애니메이션 기술은 스포츠 동작 분석에 적용 가능하다.

**활용 인사이트**
20 FPS 실시간 스트리밍으로 스포츠 선수의 동작을 분석하고 하이라이트 장면을 자동으로 생성할 수 있으며, ConvKV 메모리 메커니즘으로 장시간 경기 기록을 효율적으로 처리할 수 있습니다.

## 18위: MV-SAM3D: Adaptive Multi-View Fusion for Layout-Aware 3D Generation

- arXiv: http://arxiv.org/abs/2603.11633v1
- PDF: https://arxiv.org/pdf/2603.11633v1
- 코드: https://github.com/devinli123/MV-SAM3D
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 86.4 (llm_adjusted:83 = base:70 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Recent unified 3D generation models have made remarkable progress in producing high-quality 3D assets from a single image. Notably, layout-aware approaches such as SAM3D can reconstruct multiple objects while preserving their spatial arrangement, opening the door to practical scene-level 3D generation. However, current methods are limited to single-view input and cannot leverage complementary multi-view observations, while independently estimated object poses often lead to physically implausible layouts such as interpenetration and floating artifacts.   We present MV-SAM3D, a training-free framework that extends layout-aware 3D generation with multi-view consistency and physical plausibility. We formulate multi-view fusion as a Multi-Diffusion process in 3D latent space and propose two adaptive weighting strategies -- attention-entropy weighting and visibility weighting -- that enable confidence-aware fusion, ensuring each viewpoint contributes according to its local observation reliability. For multi-object composition, we introduce physics-aware optimization that injects collision and contact constraints both during and after generation, yielding physically plausible object arrangements. Experiments on standard benchmarks and real-world multi-object scenes demonstrate significant improvements in reconstruction fidelity and layout plausibility, all without any additional training. Code is available at https://github.com/devinli123/MV-SAM3D.

**선정 근거**
Multi-view 3D generation technology applicable for sports camera systems

**활용 인사이트**
다중 관점 융합 기술로 스포츠 경기의 3D 장면을 재구성하고 물리적 타당성을 보장하여 경기 전략 분석과 시각적 콘텐츠 제작에 활용할 수 있습니다.

## 19위: Resource-Efficient Iterative LLM-Based NAS with Feedback Memory

- arXiv: http://arxiv.org/abs/2603.12091v1
- PDF: https://arxiv.org/pdf/2603.12091v1
- 발행일: 2026-03-12
- 카테고리: cs.LG, cs.AI
- 점수: final 86.4 (llm_adjusted:83 = base:75 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Neural Architecture Search (NAS) automates network design, but conventional methods demand substantial computational resources. We propose a closed-loop pipeline leveraging large language models (LLMs) to iteratively generate, evaluate, and refine convolutional neural network architectures for image classification on a single consumer-grade GPU without LLM fine-tuning. Central to our approach is a historical feedback memory inspired by Markov chains: a sliding window of $K{=}5$ recent improvement attempts keeps context size constant while providing sufficient signal for iterative learning. Unlike prior LLM optimizers that discard failure trajectories, each history entry is a structured diagnostic triple -- recording the identified problem, suggested modification, and resulting outcome -- treating code execution failures as first-class learning signals. A dual-LLM specialization reduces per-call cognitive load: a Code Generator produces executable PyTorch architectures while a Prompt Improver handles diagnostic reasoning. Since both the LLM and architecture training share limited VRAM, the search implicitly favors compact, hardware-efficient models suited to edge deployment. We evaluate three frozen instruction-tuned LLMs (${\leq}7$B parameters) across up to 2000 iterations in an unconstrained open code space, using one-epoch proxy accuracy on CIFAR-10, CIFAR-100, and ImageNette as a fast ranking signal. On CIFAR-10, DeepSeek-Coder-6.7B improves from 28.2% to 69.2%, Qwen2.5-7B from 50.0% to 71.5%, and GLM-5 from 43.2% to 62.0%. A full 2000-iteration search completes in ${\approx}18$ GPU hours on a single RTX~4090, establishing a low-budget, reproducible, and hardware-aware paradigm for LLM-driven NAS without cloud infrastructure.

**선정 근거**
에지 장치용 효율적인 모델 설계가 스포츠 촬영 에지 디바이스에 적용 가능한 기술

**활용 인사이트**
단일 소비자급 GPU에서 18시간 만에 최적의 경량 모델을 설계하여 rk3588 에지 디바이스에 적합한 스포츠 분석 모델을 효율적으로 개발할 수 있습니다.

## 20위: Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning

- arXiv: http://arxiv.org/abs/2603.11439v1
- PDF: https://arxiv.org/pdf/2603.11439v1
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Dense Video Captioning (DVC) is a challenging multimodal task that involves temporally localizing multiple events within a video and describing them with natural language. While query-based frameworks enable the simultaneous, end-to-end processing of localization and captioning, their reliance on shared queries often leads to significant multi-task interference between the two tasks, as well as temporal redundancy in localization. In this paper, we propose utilizing role-specific queries that separate localization and captioning into independent components, allowing each to exclusively learn its role. We then employ contrastive alignment to enforce semantic consistency between the corresponding outputs, ensuring coherent behavior across the separated queries. Furthermore, we design a novel suppression mechanism in which mutual temporal overlaps across queries are penalized to tackle temporal redundancy, supervising the model to learn distinct, non-overlapping event regions for more precise localization. Additionally, we introduce a lightweight module that captures core event concepts to further enhance semantic richness in captions through concept-level representations. We demonstrate the effectiveness of our method through extensive experiments on major DVC benchmarks YouCook2 and ActivityNet Captions.

**선정 근거**
비디오 캡셔닝 기술은 스포츠 영상 분석에 적용 가능

**활용 인사이트**
역할별 쿼리 기반으로 스포츠 경기 중 다양한 이벤트를 정확히 식별하고 설명하며, 중첩 억제 메커니즘으로 중복되는 장면을 줄여 하이라이트 편집 효율을 높일 수 있습니다.

## 21위: INFACT: A Diagnostic Benchmark for Induced Faithfulness and Factuality Hallucinations in Video-LLMs

- arXiv: http://arxiv.org/abs/2603.11481v1
- PDF: https://arxiv.org/pdf/2603.11481v1
- 발행일: 2026-03-12
- 카테고리: cs.CV, cs.AI
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Despite rapid progress, Video Large Language Models (Video-LLMs) remain unreliable due to hallucinations, which are outputs that contradict either video evidence (faithfulness) or verifiable world knowledge (factuality). Existing benchmarks provide limited coverage of factuality hallucinations and predominantly evaluate models only in clean settings. We introduce \textsc{INFACT}, a diagnostic benchmark comprising 9{,}800 QA instances with fine-grained taxonomies for faithfulness and factuality, spanning real and synthetic videos. \textsc{INFACT} evaluates models in four modes: Base (clean), Visual Degradation, Evidence Corruption, and Temporal Intervention for order-sensitive items. Reliability under induced modes is quantified using Resist Rate (RR) and Temporal Sensitivity Score (TSS). Experiments on 14 representative Video-LLMs reveal that higher Base-mode accuracy does not reliably translate to higher reliability in the induced modes, with evidence corruption reducing stability and temporal intervention yielding the largest degradation. Notably, many open-source baselines exhibit near-zero TSS on factuality, indicating pronounced temporal inertia on order-sensitive questions.

**선정 근거**
비디오 LLM 기술이 스포츠 비디오 분석에 적용 가능합니다

**활용 인사이트**
스포츠 경기 영상에서의 환경 변화나 증거 왜곡에도 정확한 분석을 제공하여 하이라이트 편집 정확도 향상

## 22위: Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation

- arXiv: http://arxiv.org/abs/2603.12247v1
- PDF: https://arxiv.org/pdf/2603.12247v1
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 84.8 (llm_adjusted:81 = base:78 + bonus:+3)
- 플래그: 코드 공개

**개요**
Reinforcement learning (RL) has emerged as a promising paradigm for enhancing image editing and text-to-image (T2I) generation. However, current reward models, which act as critics during RL, often suffer from hallucinations and assign noisy scores, inherently misguiding the optimization process. In this paper, we present FIRM (Faithful Image Reward Modeling), a comprehensive framework that develops robust reward models to provide accurate and reliable guidance for faithful image generation and editing. First, we design tailored data curation pipelines to construct high-quality scoring datasets. Specifically, we evaluate editing using both execution and consistency, while generation is primarily assessed via instruction following. Using these pipelines, we collect the FIRM-Edit-370K and FIRM-Gen-293K datasets, and train specialized reward models (FIRM-Edit-8B and FIRM-Gen-8B) that accurately reflect these criteria. Second, we introduce FIRM-Bench, a comprehensive benchmark specifically designed for editing and generation critics. Evaluations demonstrate that our models achieve superior alignment with human judgment compared to existing metrics. Furthermore, to seamlessly integrate these critics into the RL pipeline, we formulate a novel "Base-and-Bonus" reward strategy that balances competing objectives: Consistency-Modulated Execution (CME) for editing and Quality-Modulated Alignment (QMA) for generation. Empowered by this framework, our resulting models FIRM-Qwen-Edit and FIRM-SD3.5 achieve substantial performance breakthroughs. Comprehensive experiments demonstrate that FIRM mitigates hallucinations, establishing a new standard for fidelity and instruction adherence over existing general models. All of our datasets, models, and code have been publicly available at https://firm-reward.github.io.

**선정 근거**
이미지 편집 및 생성 기술이 프로젝트의 영상 보정 및 편집 부분에 직접 적용 가능

**활용 인사이트**
FIRM 프레임워크를 활용하여 스포츠 영상의 품질을 향상시키고 자연스러운 보정 효과 구현

## 23위: LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference

- arXiv: http://arxiv.org/abs/2603.11605v1
- PDF: https://arxiv.org/pdf/2603.11605v1
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 84.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
Human motion is highly expressive and naturally aligned with language, yet prevailing methods relying heavily on joint text-motion embeddings struggle to synthesize temporally accurate, detailed motions and often lack explainability. To address these limitations, we introduce LabanLite, a motion representation developed by adapting and extending the Labanotation system. Unlike black-box text-motion embeddings, LabanLite encodes each atomic body-part action (e.g., a single left-foot step) as a discrete Laban symbol paired with a textual template. This abstraction decomposes complex motions into interpretable symbol sequences and body-part instructions, establishing a symbolic link between high-level language and low-level motion trajectories. Building on LabanLite, we present LaMoGen, a Text-to-LabanLite-to-Motion Generation framework that enables large language models (LLMs) to compose motion sequences through symbolic reasoning. The LLM interprets motion patterns, relates them to textual descriptions, and recombines symbols into executable plans, producing motions that are both interpretable and linguistically grounded. To support rigorous evaluation, we introduce a Labanotation-based benchmark with structured description-motion pairs and three metrics that jointly measure text-motion alignment across symbolic, temporal, and harmony dimensions. Experiments demonstrate that LaMoGen establishes a new baseline for both interpretability and controllability, outperforming prior methods on our benchmark and two public datasets. These results highlight the advantages of symbolic reasoning and agent-based design for language-driven motion synthesis.

**선정 근거**
언어 기반 동생성 기술은 스포츠 동작 분석과 하이라이트 영상 제작에 활용 가능합니다

**활용 인사이트**
LabanLite 기반의 기호적 추론을 통해 스포츠 선수들의 동작을 정밀 분석하고 자연스러운 하이라이트 생성

## 24위: SNAP-V: A RISC-V SoC with Configurable Neuromorphic Acceleration for Small-Scale Spiking Neural Networks

- arXiv: http://arxiv.org/abs/2603.11939v1
- PDF: https://arxiv.org/pdf/2603.11939v1
- 발행일: 2026-03-12
- 카테고리: cs.AR, cs.NE
- 점수: final 84.0 (llm_adjusted:80 = base:70 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Spiking Neural Networks (SNNs) have gained significant attention in edge computing due to their low power consumption and computational efficiency. However, existing implementations either use conventional System on Chip (SoC) architectures that suffer from memory-processor bottlenecks, or large-scale neuromorphic hardware that is inefficient and wasteful for small-scale SNN applications. This work presents SNAP-V, a RISC-V-based neuromorphic SoC with two accelerator variants: Cerebra-S (bus-based) and Cerebra-H (Network-on-Chip (NoC)-based) which are optimized for small-scale SNN inference, integrating a RISC-V core for management tasks, with both accelerators featuring parallel processing nodes and distributed memory. Experimental results show close agreement between software and hardware inference, with an average accuracy deviation of 2.62% across multiple network configurations, and an average synaptic energy of 1.05 pJ per synaptic operation (SOP) in 45 nm CMOS technology. These results show that the proposed solution enables accurate, energy-efficient SNN inference suitable for real-time edge applications.

**선정 근거**
RISC-V 기반의 신경망 가속기가 에지 디바이스의 저전력 실시간 처리에 적합

**활용 인사이트**
스포츠 촬영 장치에 통합하여 자세 분석과 하이라이트 감출 효율 향상

## 25위: BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning

- arXiv: http://arxiv.org/abs/2603.12176v1
- PDF: https://arxiv.org/pdf/2603.12176v1
- 발행일: 2026-03-12
- 카테고리: cs.CV, cs.AI
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Understanding freely moving animal behavior is central to neuroscience, where pose estimation and behavioral understanding form the foundation for linking neural activity to natural actions. Yet both tasks still depend heavily on human annotation or unstable unsupervised pipelines, limiting scalability and reproducibility. We present BehaviorVLM, a unified vision-language framework for pose estimation and behavioral understanding that requires no task-specific finetuning and minimal human labeling by guiding pretrained Vision-Language Models (VLMs) through detailed, explicit, and verifiable reasoning steps. For pose estimation, we leverage quantum-dot-grounded behavioral data and propose a multi-stage pipeline that integrates temporal, spatial, and cross-view reasoning. This design greatly reduces human annotation effort, exposes low-confidence labels through geometric checks such as reprojection error, and produces labels that can later be filtered, corrected, or used to fine-tune downstream pose models. For behavioral understanding, we propose a pipeline that integrates deep embedded clustering for over-segmented behavior discovery, VLM-based per-clip video captioning, and LLM-based reasoning to merge and semantically label behavioral segments. The behavioral pipeline can operate directly from visual information and does not require keypoints to segment behavior. Together, these components enable scalable, interpretable, and label-light analysis of multi-animal behavior.

**선정 근거**
동물 행동 이해 기술로 스포츠 분석에 간접적으로 적용 가능

## 26위: Derain-Agent: A Plug-and-Play Agent Framework for Rainy Image Restoration

- arXiv: http://arxiv.org/abs/2603.11866v1
- PDF: https://arxiv.org/pdf/2603.11866v1
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
While deep learning has advanced single-image deraining, existing models suffer from a fundamental limitation: they employ a static inference paradigm that fails to adapt to the complex, coupled degradations (e.g., noise artifacts, blur, and color deviation) of real-world rain. Consequently, restored images often exhibit residual artifacts and inconsistent perceptual quality. In this work, we present Derain-Agent, a plug-and-play refinement framework that transitions deraining from static processing to dynamic, agent-based restoration. Derain-Agent equips a base deraining model with two core capabilities: 1) a Planning Network that intelligently schedules an optimal sequence of restoration tools for each instance, and 2) a Strength Modulation mechanism that applies these tools with spatially adaptive intensity. This design enables precise, region-specific correction of residual errors without the prohibitive cost of iterative search. Our method demonstrates strong generalization, consistently boosting the performance of state-of-the-art deraining models on both synthetic and real-world benchmarks.

**선정 근거**
스포츠 촬영 시 발생하는 다양한 날씨 조건에서 이미지 품질 향상 가능

**활용 인사이트**
플러그 앤 플레이 프레임워크로 기존 촬영 시스템에 통합하여 실시간 이미지 보정 구현

## 27위: RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset

- arXiv: http://arxiv.org/abs/2603.11811v1
- PDF: https://arxiv.org/pdf/2603.11811v1
- 발행일: 2026-03-12
- 카테고리: cs.RO, cs.AI, cs.CV
- 점수: final 78.4 (llm_adjusted:73 = base:68 + bonus:+5)
- 플래그: 실시간

**개요**
The acquisition of large-scale physical interaction data, a critical prerequisite for modern robot learning, is severely bottlenecked by the prohibitive cost and scalability limits of human-in-the-loop collection paradigms. To break this barrier, we introduce Robust Autonomous Data Acquisition for Robotics (RADAR), a fully autonomous, closed-loop data generation engine that completely removes human intervention from the collection cycle. RADAR elegantly divides the cognitive load into a four-module pipeline. Anchored by 2-5 3D human demonstrations as geometric priors, a Vision-Language Model first orchestrates scene-relevant task generation via precise semantic object grounding and skill retrieval. Next, a Graph Neural Network policy translates these subtasks into physical actions via in-context imitation learning. Following execution, the VLM performs automated success evaluation using a structured Visual Question Answering pipeline. Finally, to shatter the bottleneck of manual resets, a Finite State Machine orchestrates an autonomous environment reset and asymmetric data routing mechanism. Driven by simultaneous forward-reverse planning with a strict Last-In, First-Out causal sequence, the system seamlessly restores unstructured workspaces and robustly recovers from execution failures. This continuous brain-cerebellum synergy transforms data collection into a self-sustaining process. Extensive evaluations highlight RADAR's exceptional versatility. In simulation, our framework achieves up to 90% success rates on complex, long-horizon tasks, effortlessly solving challenges where traditional baselines plummet to near-zero performance. In real-world deployments, the system reliably executes diverse, contact-rich skills (e.g., deformable object manipulation) via few-shot adaptation without domain-specific fine-tuning, providing a highly scalable paradigm for robotic data acquisition.

**선정 근거**
자율 데이터 생성 원리가 자동 촬영에 적용 가능하지만 로봇 데이터 수집에 집중됩니다

## 28위: Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding

- arXiv: http://arxiv.org/abs/2603.11423v1
- PDF: https://arxiv.org/pdf/2603.11423v1
- 발행일: 2026-03-12
- 카테고리: cs.CV
- 점수: final 76.0 (llm_adjusted:70 = base:70 + bonus:+0)

**개요**
Traditional black-box distillation for Large Vision-Language Models (LVLMs) typically relies on a single teacher response per input, which often yields high-variance responses and format inconsistencies in multimodal or temporal scenarios. To mitigate this unreliable supervision, we propose R-MSD (Reliable Multi-Sample Distillation), a framework that explicitly models teacher sampling variance to enhance distillation stability. Rather than relying on a single teacher response, our approach leverages a task-adaptive teacher pool to provide robust supervision tailored to both closed-ended and open-ended reasoning. By integrating quality-aware signal matching with an adversarial distillation objective, our approach effectively filters teacher noise while maximizing knowledge transfer. Extensive evaluations across comprehensive video understanding benchmarks demonstrate that R-MSD consistently outperforms single sample distillation methods. We additionally include an original SFT+RL 4B baseline under the same training budget, which shows only marginal gains, while our method achieves significant improvements. With a 4B student model, our approach delivers gains on VideoMME (+1.5%), Video-MMMU (+3.2%), and MathVerse (+3.6%).

**선정 근거**
스포츠 영상의 다양한 장면과 동작을 정확히 분석하여 하이라이트 편집 정확도 향상

**활용 인사이트**
다중 샘플 디스틸레이션 기법으로 비디오 이해 모델의 안정성과 일관성 개선

## 29위: LLMs can construct powerful representations and streamline sample-efficient supervised learning

- arXiv: http://arxiv.org/abs/2603.11679v1
- PDF: https://arxiv.org/pdf/2603.11679v1
- 발행일: 2026-03-12
- 카테고리: cs.AI
- 점수: final 76.0 (llm_adjusted:70 = base:70 + bonus:+0)

**개요**
As real-world datasets become increasingly complex and heterogeneous, supervised learning is often bottlenecked by input representation design. Modeling multimodal data for downstream tasks, such as time-series, free text, and structured records, often requires non-trivial domain-specific engineering. We propose an agentic pipeline to streamline this process. First, an LLM analyzes a small but diverse subset of text-serialized input examples in-context to synthesize a global rubric, which acts as a programmatic specification for extracting and organizing evidence. This rubric is then used to transform naive text-serializations of inputs into a more standardized format for downstream models. We also describe local rubrics, which are task-conditioned summaries generated by an LLM. Across 15 clinical tasks from the EHRSHOT benchmark, our rubric-based approaches significantly outperform traditional count-feature models, naive text-serialization-based LLM baselines, and a clinical foundation model, which is pretrained on orders of magnitude more data. Beyond performance, rubrics offer several advantages for operational healthcare settings such as being easy to audit, cost-effectiveness to deploy at scale, and they can be converted to tabular representations that unlock a swath of machine learning techniques.

**선정 근거**
복잡한 스포츠 데이터 효과적으로 처리하여 자세 및 전략 분석 성능 향상

**활용 인사이트**
LLM 기반 루브릭 시스템으로 스포츠 데이터 표준화 및 다운스트림 모델 성능 최적화

## 30위: Decentralized Orchestration Architecture for Fluid Computing: A Secure Distributed AI Use Case

- arXiv: http://arxiv.org/abs/2603.12001v1
- PDF: https://arxiv.org/pdf/2603.12001v1
- 발행일: 2026-03-12
- 카테고리: cs.DC, cs.LG
- 점수: final 76.0 (llm_adjusted:70 = base:65 + bonus:+5)
- 플래그: 엣지

**개요**
Distributed AI and IoT applications increasingly execute across heterogeneous resources spanning end devices, edge/fog infrastructure, and cloud platforms, often under different administrative domains. Fluid Computing has emerged as a promising paradigm for enhancing massive resource management across the computing continuum by treating such resources as a unified fabric, enabling optimal service-agnostic deployments driven by application requirements. However, existing solutions remain largely centralized and often do not explicitly address multi-domain considerations. This paper proposes an agnostic multi-domain orchestration architecture for fluid computing environments. The orchestration plane enables decentralized coordination among domains that maintain local autonomy while jointly realizing intent-based deployment requests from tenants, ensuring end-to-end placement and execution. To this end, the architecture elevates domain-side control services as first-class capabilities to support application-level enhancement at runtime. As a representative use case, we consider a multi-domain Decentralized Federated Learning (DFL) deployment under Byzantine threats. We leverage domain-side capabilities to enhance Byzantine security by introducing FU-HST, an SDN-enabled multi-domain anomaly detection mechanism that complements Byzantine-robust aggregation. We validate the approach via simulation in single- and multi-domain settings, evaluating anomaly detection, DFL performance, and computation/communication overhead.

**선정 근거**
분산 AI 및 엣지 인프라 아키텍처 관련 기술로 엣기 디바이스 구현에 간접적 관련

## 31위: Grounding Robot Generalization in Training Data via Retrieval-Augmented VLMs

- arXiv: http://arxiv.org/abs/2603.11426v1 | 2026-03-12 | final 74.4

Recent work on robot manipulation has advanced policy generalization to novel scenarios. However, it is often difficult to characterize how different evaluation settings actually represent generalization from the training distribution of a given policy.

-> 비전-언어 모델이 스포츠 동작 분석에 간접적으로 활용될 수 있으나 로봇 학습에 중점을 둡니다

## 32위: Intelligent 6G Edge Connectivity: A Knowledge Driven Optimization Framework for Small Cell Selection

- arXiv: http://arxiv.org/abs/2603.12086v1 | 2026-03-12 | final 74.4

Sixth-generation (6G) wireless networks are expected to support immersive and mission-critical applications requiring ultra-reliable communication, sub-second responsiveness, and multi-Gbps data rates. Dense small-cell deployments are a key enabler of these capabilities; however, the large number of candidate cells available to mobile users makes efficient user-cell association increasingly complex.

-> 엣지 컴퓨팅 개념은 관련 있지만 네트워크 연결성에 초점을 맞춰 스포츠 분석과는 간접적

## 33위: A Two-Stage Dual-Modality Model for Facial Emotional Expression Recognition

- arXiv: http://arxiv.org/abs/2603.12221v1 | 2026-03-12 | final 72.0

This paper addresses the expression (EXPR) recognition challenge in the 10th Affective Behavior Analysis in-the-Wild (ABAW) workshop and competition, which requires frame-level classification of eight facial emotional expressions from unconstrained videos. This task is challenging due to inaccurate face localization, large pose and scale variations, motion blur, temporal instability, and other confounding factors across adjacent frames.

-> Facial expression recognition techniques could be adapted for sports movement analysis

## 34위: Linking Perception, Confidence and Accuracy in MLLMs

- arXiv: http://arxiv.org/abs/2603.12149v1 | 2026-03-12 | final 72.0

Recent advances in Multi-modal Large Language Models (MLLMs) have predominantly focused on enhancing visual perception to improve accuracy. However, a critical question remains unexplored: Do models know when they do not know?

-> 지각 및 분석 기술을 다루지만 스포츠 특화 분석을 위한 것은 아니어서 간접적 관련성

## 35위: Beyond the Limits of Rigid Arrays: Flexible Intelligent Metasurfaces for Next-Generation Wireless Networks

- arXiv: http://arxiv.org/abs/2603.11886v1 | 2026-03-12 | final 72.0

Following recent advances in flexible electronics and programmable metasurfaces, flexible intelligent metasurfaces (FIMs) have emerged as a promising enabling technology for next-generation wireless networks. A FIM is a morphable electromagnetic surface capable of dynamically adjusting its physical geometry to influence the radiation and propagation of electromagnetic waves.

-> 유연한 무선 네트워크 기술로 엣지 디바이스 통신에 간접적 관련

## 36위: Agentic AI for Embodied-enhanced Beam Prediction in Low-Altitude Economy Networks

- arXiv: http://arxiv.org/abs/2603.11392v1 | 2026-03-12 | final 71.6

Millimeter-wave or terahertz communications can meet demands of low-altitude economy networks for high-throughput sensing and real-time decision making. However, high-frequency characteristics of wireless channels result in severe propagation loss and strong beam directivity, which make beam prediction challenging in highly mobile uncrewed aerial vehicles (UAV) scenarios.

-> 다 에이전트 아키텍처와 멀티모달 융합이 스포츠 분석에 간접적으로 적용 가능

## 37위: Miniaturized microscopes to study neural dynamics in freely-behaving animals

- arXiv: http://arxiv.org/abs/2603.11435v1 | 2026-03-12 | final 70.4

Head-mounted miniaturized microscopes, commonly known as miniscopes, have undergone rapid development and seen widespread adoption over the past two decades, enabling the imaging of neural activity in freely-behaving animals such as rodents, songbirds, and non-human primates. These miniscopes facilitate numerous studies that are not feasible with head-fixed preparations.

-> 소형 카메라 개념이 엣지 디바이스 개발과 관련이 있습니다.

## 38위: High-Precision 6DOF Pose Estimation via Global Phase Retrieval in Fringe Projection Profilometry for 3D Mapping

- arXiv: http://arxiv.org/abs/2603.11389v1 | 2026-03-12 | final 70.0

Digital fringe projection (DFP) enables micrometer-level 3D reconstruction, yet extending it to large-scale mapping remains challenging because six-degree-of-freedom pose estimation often cannot match the reconstruction's precision. Conventional iterative closest point (ICP) registration becomes inefficient on multi-million-point clouds and typically relies on downsampling or feature-based selection, which can reduce local detail and degrade pose precision.

-> 6DOF 포즈 추술 기술이 스포츠 동작 분석에 간접적으로 적용 가능

## 39위: Toward Complex-Valued Neural Networks for Waveform Generation

- arXiv: http://arxiv.org/abs/2603.11589v1 | 2026-03-12 | final 68.8

Neural vocoders have recently advanced waveform generation, yielding natural and expressive audio. Among these approaches, iSTFT-based vocoders have recently gained attention.

-> 복소수 신경망을 이용한 파형 생성 기술로 영상 처리에 간접적 관련

## 40위: Silent Speech Interfaces in the Era of Large Language Models: A Comprehensive Taxonomy and Systematic Review

- arXiv: http://arxiv.org/abs/2603.11877v1 | 2026-03-12 | final 68.0

Human-computer interaction has traditionally relied on the acoustic channel, a dependency that introduces systemic vulnerabilities to environmental noise, privacy constraints, and physiological speech impairments. Silent Speech Interfaces (SSIs) emerge as a transformative paradigm that bypasses the acoustic stage by decoding linguistic intent directly from the neuro-muscular-articulatory continuum.

-> Paper discusses Silent Speech Interfaces transitioning to wearables, which has some relevance to edge devices but not specifically to sports filming.

## 41위: DVD: Deterministic Video Depth Estimation with Generative Priors

- arXiv: http://arxiv.org/abs/2603.12250v1 | 2026-03-12 | final 66.4

Existing video depth estimation faces a fundamental trade-off: generative models suffer from stochastic geometric hallucinations and scale drift, while discriminative models demand massive labeled datasets to resolve semantic ambiguities. To break this impasse, we present DVD, the first framework to deterministically adapt pre-trained video diffusion models into single-pass depth regressors.

-> 비디오 심도 추정 기술은 스포츠 촬영 장치에 간접적으로 적용 가능한 기술이다.

## 42위: ActiveFreq: Integrating Active Learning and Frequency Domain Analysis for Interactive Segmentation

- arXiv: http://arxiv.org/abs/2603.11498v1 | 2026-03-12 | final 66.4

Interactive segmentation is commonly used in medical image analysis to obtain precise, pixel-level labeling, typically involving iterative user input to correct mislabeled regions. However, existing approaches often fail to fully utilize user knowledge from interactive inputs and achieve comprehensive feature extraction.

-> 의료 이미지 분할 기술은 스포츠 이미지 처리에 간접적으로 적용 가능

## 43위: CEI-3D: Collaborative Explicit-Implicit 3D Reconstruction for Realistic and Fine-Grained Object Editing

- arXiv: http://arxiv.org/abs/2603.11810v1 | 2026-03-12 | final 66.4

Existing 3D editing methods often produce unrealistic and unrefined results due to the deeply integrated nature of their reconstruction networks. To address the challenge, this paper introduces CEI-3D, an editing-oriented reconstruction pipeline designed to facilitate realistic and fine-grained editing.

-> 3D 재구성 기술은 스포츠 장면 분석에 간접적으로 적용 가능

## 44위: RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images

- arXiv: http://arxiv.org/abs/2603.12215v1 | 2026-03-12 | final 66.4

Salient object detection (SOD) in remote sensing images faces significant challenges due to large variations in object sizes, the computational cost of self-attention mechanisms, and the limitations of CNN-based extractors in capturing global context and long-range dependencies. Existing methods that rely on fixed convolution kernels often struggle to adapt to diverse object scales, leading to detail loss or irrelevant feature aggregation.

-> 객체 감지 기술이 스포츠 장면 분석에 간접적으로 적용 가능

## 45위: CAETC: Causal Autoencoding and Treatment Conditioning for Counterfactual Estimation over Time

- arXiv: http://arxiv.org/abs/2603.11565v1 | 2026-03-12 | final 66.4

Counterfactual estimation over time is important in various applications, such as personalized medicine. However, time-dependent confounding bias in observational data still poses a significant challenge in achieving accurate and efficient estimation.

-> 시계열 데이터 분석 기술로 스포츠 동작 분석에 간접적으로 적용 가능

## 46위: Adversarial Reinforcement Learning for Detecting False Data Injection Attacks in Vehicular Routing

- arXiv: http://arxiv.org/abs/2603.11433v1 | 2026-03-12 | final 66.4

In modern transportation networks, adversaries can manipulate routing algorithms using false data injection attacks, such as simulating heavy traffic with multiple devices running crowdsourced navigation applications, to mislead vehicles toward suboptimal routes and increase congestion. To address these threats, we formulate a strategically zero-sum game between an attacker, who injects such perturbations, and a defender, who detects anomalies based on the observed travel times of network edges.

-> 강화 학습 기술로 스포츠 전략 분석에 간접적으로 적용 가능

## 47위: COTONET: A custom cotton detection algorithm based on YOLO11 for stage of growth cotton boll detection

- arXiv: http://arxiv.org/abs/2603.11717v1 | 2026-03-12 | final 64.0

Cotton harvesting is a critical phase where cotton capsules are physically manipulated and can lead to fibre degradation. To maintain the highest quality, harvesting methods must emulate delicate manual grasping, to preserve cotton's intrinsic properties.

-> 엣지 최적화 객체 감지 모델이 관련되어 있지만 면화 감지에 특화되어 스포츠 분석과는 거리가 있습니다

## 48위: HELM: Hierarchical and Explicit Label Modeling with Graph Learning for Multi-Label Image Classification

- arXiv: http://arxiv.org/abs/2603.11783v1 | 2026-03-12 | final 64.0

Hierarchical multi-label classification (HMLC) is essential for modeling complex label dependencies in remote sensing. Existing methods, however, struggle with multi-path hierarchies where instances belong to multiple branches, and they rarely exploit unlabeled data.

-> 계층적 다중 레이블 이미지 분류 기술은 스포츠 장면 인식에 적용 가능하지만 스포츠 특화 수정이 필요합니다.

## 49위: Machine Learning-Based Analysis of Critical Process Parameters Influencing Product Quality Defects: A Real-World Case Study in Manufacturing

- arXiv: http://arxiv.org/abs/2603.11666v1 | 2026-03-12 | final 64.0

Quality control is an essential operation in manufacturing, ensuring products meet the necessary standards of quality, safety, and reliability. Traditional methods, such as visual inspections, measurements, and statistical techniques, help meet these standards but are often time-consuming, costly, and reactive.

-> 분석 및 예측 방법론을 다루지만 제조업 품질 관리에 초점을 맞춰 스포츠 분석으로의 적용은 간접적

## 50위: From Pets to Robots: MojiKit as a Data-Informed Toolkit for Affective HRI Design

- arXiv: http://arxiv.org/abs/2603.11632v1 | 2026-03-12 | final 64.0

Designing affective behaviors for animal-inspired social robots often relies on intuition and personal experience, leading to fragmented outcomes. To provide more systematic guidance, we first coded and analyzed human-pet interaction videos, validated insights through literature and interviews, and created structured reference cards that map the design space of pet-inspired affective interactions.

-> 상호작용 비디오 분석 기술로 스포츠 동작 분석에 간접적으로 참조 가능

## 51위: HPC Containers for EBRAINS: Towards Portable Cross-Domain Software Environment

- arXiv: http://arxiv.org/abs/2603.12044v1 | 2026-03-12 | final 64.0

Deploying complex, distributed scientific workflows across diverse HPC sites is often hindered by site-specific dependencies and complex build environments. This paper investigates the design and performance of portable HPC container images capable of encapsulating MPI- and CUDA-enabled software stacks without sacrificing bare-metal performance.

-> 컨테이너화 및 CUDA 지원 기술로 rk3588 장치에서 AI 모델 배포에 간접적으로 참조 가능

## 52위: Affect Decoding in Phonated and Silent Speech Production from Surface EMG

- arXiv: http://arxiv.org/abs/2603.11715v1 | 2026-03-12 | final 64.0

The expression of affect is integral to spoken communication, yet, its link to underlying articulatory execution remains unclear. Measures of articulatory muscle activity such as EMG could reveal how speech production is modulated by emotion alongside acoustic speech analyses.

-> EMG를 이용한 감정 분석 기술로 동작 분석에 간접적 관련

## 53위: Examining Reasoning LLMs-as-Judges in Non-Verifiable LLM Post-Training

- arXiv: http://arxiv.org/abs/2603.12246v1 | 2026-03-12 | final 64.0

Reasoning LLMs-as-Judges, which can benefit from inference-time scaling, provide a promising path for extending the success of reasoning models to non-verifiable domains where the output correctness/quality cannot be directly checked. However, while reasoning judges have shown better performance on static evaluation benchmarks, their effectiveness in actual policy training has not been systematically examined.

-> LLM 개념이 스포츠 전략 분석에 잠재적으로 적용 가능합니다.

## 54위: Preliminary analysis of RGB-NIR Image Registration techniques for off-road forestry environments

- arXiv: http://arxiv.org/abs/2603.11952v1 | 2026-03-12 | final 61.6

RGB-NIR image registration plays an important role in sensor-fusion, image enhancement and off-road autonomy. In this work, we evaluate both classical and Deep Learning (DL) based image registration techniques to access their suitability for off-road forestry applications.

-> RGB-NIR 이미지 등록 기술은 스포츠 촬영 환경에 간접적으로 적용 가능하다.

## 55위: HATS: Hardness-Aware Trajectory Synthesis for GUI Agents

- arXiv: http://arxiv.org/abs/2603.12138v1 | 2026-03-12 | final 56.0

Graphical user interface (GUI) agents powered by large vision-language models (VLMs) have shown remarkable potential in automating digital tasks, highlighting the need for high-quality trajectory data to support effective agent training. Yet existing trajectory synthesis pipelines often yield agents that fail to generalize beyond simple interactions.

-> GUI agent trajectory synthesis with limited applicability to sports analysis

## 56위: Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography

- arXiv: http://arxiv.org/abs/2603.11627v1 | 2026-03-12 | final 56.0

Positron emission tomography (PET) is a key nuclear medicine imaging modality that visualizes radiotracer distributions to quantify in vivo physiological and metabolic processes, playing an irreplaceable role in disease management. Despite its clinical importance, the development of deep learning models for quantitative PET image analysis remains severely limited, driven by both the inherent segmentation challenge from PET's paucity of anatomical contrast and the high costs of data acquisition and annotation.

-> PET 이미지 분할 모델이 스포츠 영상 처리와 약간 관련

## 57위: Efficient Cross-View Localization in 6G Space-Air-Ground Integrated Network

- arXiv: http://arxiv.org/abs/2603.11398v1 | 2026-03-12 | final 55.6

Recently, visual localization has become an important supplement to improve localization reliability, and cross-view approaches can greatly enhance coverage and adaptability. Meanwhile, future 6G will enable a globally covered mobile communication system, with a space-air-ground integrated network (SAGIN) serving as key supporting architecture.

-> 시각적 위치 개념이 카메라 포지셔닝에 적용될 수 있지만 통신 네트워크에 중점을 둡니다

## 58위: QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions

- arXiv: http://arxiv.org/abs/2603.12165v1 | 2026-03-12 | final 54.4

Synthetic data has become essential for training code generation models, yet it introduces significant noise and hallucinations that are difficult to detect with current metrics. Existing data selection methods like Instruction-Following Difficulty (IFD) typically assess how hard a model generates an answer given a query ($A|Q$).

-> Weakly related to data quality assessment which could be tangentially applicable to video content processing

## 59위: Real-time Rendering-based Surgical Instrument Tracking via Evolutionary Optimization

- arXiv: http://arxiv.org/abs/2603.11404v1 | 2026-03-12 | final 52.0

Accurate and efficient tracking of surgical instruments is fundamental for Robot-Assisted Minimally Invasive Surgery. Although vision-based robot pose estimation has enabled markerless calibration without tedious physical setups, reliable tool tracking for surgical robots still remains challenging due to partial visibility and specialized articulation design of surgical instruments.

-> 수술 기구 추적 기술로 스포츠 분석과 직접적인 관련성이 낮음

## 60위: Cascade: Composing Software-Hardware Attack Gadgets for Adversarial Threat Amplification in Compound AI Systems

- arXiv: http://arxiv.org/abs/2603.12023v1 | 2026-03-12 | final 52.0

Rapid progress in generative AI has given rise to Compound AI systems - pipelines comprised of multiple large language models (LLM), software tools and database systems. Compound AI systems are constructed on a layered traditional software stack running on a distributed hardware infrastructure.

-> Discusses AI systems and hardware security which could be tangentially relevant to the AI camera device

## 61위: Evaluation format, not model capability, drives triage failure in the assessment of consumer health AI

- arXiv: http://arxiv.org/abs/2603.11413v1 | 2026-03-12 | final 52.0

Ramaswamy et al. reported in \textit{Nature Medicine} that ChatGPT Health under-triages 51.6\% of emergencies, concluding that consumer-facing AI triage poses safety risks.

-> Weakly related as evaluation methodology could be tangentially relevant to sports AI system evaluation

---

## 다시 보기

### Streaming Autoregressive Video Generation via Diagonal Distillation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09488v1
- 점수: final 100.0

Large pretrained diffusion models have significantly enhanced the quality of generated videos, and yet their use in real-time streaming remains limited. Autoregressive models offer a natural framework for sequential frame synthesis but require heavy computation to achieve high fidelity. Diffusion distillation can compress these models into efficient few-step variants, but existing video distillation approaches largely adapt image-specific methods that neglect temporal dependencies. These techniques often excel in image generation but underperform in video synthesis, exhibiting reduced motion coherence, error accumulation over long sequences, and a latency-quality trade-off. We identify two factors that result in these limitations: insufficient utilization of temporal context during step reduction and implicit prediction of subsequent noise levels in next-chunk prediction (i.e., exposure bias). To address these issues, we propose Diagonal Distillation, which operates orthogonally to existing approaches and better exploits temporal information across both video chunks and denoising steps. Central to our approach is an asymmetric generation strategy: more steps early, fewer steps later. This design allows later chunks to inherit rich appearance information from thoroughly processed early chunks, while using partially denoised chunks as conditional inputs for subsequent synthesis. By aligning the implicit prediction of subsequent noise levels during chunk generation with the actual inference conditions, our approach mitigates error propagation and reduces oversaturation in long-range sequences. We further incorporate implicit optical flow modeling to preserve motion quality under strict step constraints. Our method generates a 5-second video in 2.61 seconds (up to 31 FPS), achieving a 277.3x speedup over the undistilled model.

-> 물리적으로 타당한 스포츠 하이라이트 자동 생성 기술로 경기 장면의 인과적 연결을 유지하며 자연스러운 편집 가능

### AsyncMDE: Real-Time Monocular Depth Estimation via Asynchronous Spatial Memory (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10438v1
- 점수: final 100.0

Foundation-model-based monocular depth estimation offers a viable alternative to active sensors for robot perception, yet its computational cost often prohibits deployment on edge platforms. Existing methods perform independent per-frame inference, wasting the substantial computational redundancy between adjacent viewpoints in continuous robot operation. This paper presents AsyncMDE, an asynchronous depth perception system consisting of a foundation model and a lightweight model that amortizes the foundation model's computational cost over time. The foundation model produces high-quality spatial features in the background, while the lightweight model runs asynchronously in the foreground, fusing cached memory with current observations through complementary fusion, outputting depth estimates, and autoregressively updating the memory. This enables cross-frame feature reuse with bounded accuracy degradation. At a mere 3.83M parameters, it operates at 237 FPS on an RTX 4090, recovering 77% of the accuracy gap to the foundation model while achieving a 25X parameter reduction. Validated across indoor static, dynamic, and synthetic extreme-motion benchmarks, AsyncMDE degrades gracefully between refreshes and achieves 161FPS on a Jetson AGX Orin with TensorRT, clearly demonstrating its feasibility for real-time edge deployment.

-> 에지 디바이스용 실시간 심도 추정 시스템으로 스포츠 장면의 3차원 공간 이해에 필수적이며, rk3588 기반 하드웨어에 최적화되어 있음

### Chain of Event-Centric Causal Thought for Physically Plausible Video Generation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09094v1
- 점수: final 100.0

Physically Plausible Video Generation (PPVG) has emerged as a promising avenue for modeling real-world physical phenomena. PPVG requires an understanding of commonsense knowledge, which remains a challenge for video diffusion models. Current approaches leverage commonsense reasoning capability of large language models to embed physical concepts into prompts. However, generation models often render physical phenomena as a single moment defined by prompts, due to the lack of conditioning mechanisms for modeling causal progression. In this paper, we view PPVG as generating a sequence of causally connected and dynamically evolving events. To realize this paradigm, we design two key modules: (1) Physics-driven Event Chain Reasoning. This module decomposes the physical phenomena described in prompts into multiple elementary event units, leveraging chain-of-thought reasoning. To mitigate causal ambiguity, we embed physical formulas as constraints to impose deterministic causal dependencies during reasoning. (2) Transition-aware Cross-modal Prompting (TCP). To maintain continuity between events, this module transforms causal event units into temporally aligned vision-language prompts. It summarizes discrete event descriptions to obtain causally consistent narratives, while progressively synthesizing visual keyframes of individual events by interactive editing. Comprehensive experiments on PhyGenBench and VideoPhy benchmarks demonstrate that our framework achieves superior performance in generating physically plausible videos across diverse physical domains. Our code will be released soon.

-> 물리적으로 타당한 영상 생성 기술이 스포츠 하이라이트 자동 편집 및 보정에 직접적으로 적용 가능

### CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09418v1
- 점수: final 98.4

State-of-the-art whole-body pose estimators often lack robustness, producing anatomically implausible predictions in challenging scenes. We posit this failure stems from spurious correlations learned from visual context, a problem we formalize using a Structural Causal Model (SCM). The SCM identifies visual context as a confounder that creates a non-causal backdoor path, corrupting the model's reasoning. We introduce the Causal Intervention Graph Pose (CIGPose) framework to address this by approximating the true causal effect between visual evidence and pose. The core of CIGPose is a novel Causal Intervention Module: it first identifies confounded keypoint representations via predictive uncertainty and then replaces them with learned, context-invariant canonical embeddings. These deconfounded embeddings are processed by a hierarchical graph neural network that reasons over the human skeleton at both local and global semantic levels to enforce anatomical plausibility. Extensive experiments show CIGPose achieves a new state-of-the-art on COCO-WholeBody. Notably, our CIGPose-x model achieves 67.0\% AP, surpassing prior methods that rely on extra training data. With the additional UBody dataset, CIGPose-x is further boosted to 67.5\% AP, demonstrating superior robustness and data efficiency. The codes and models are publicly available at https://github.com/53mins/CIGPose.

-> 전신 자세 추정 기술은 스포츠 동작 분석에 직접적으로 활용 가능한 핵심 기술이다

### TrainDeeploy: Hardware-Accelerated Parameter-Efficient Fine-Tuning of Small Transformer Models at the Extreme Edge (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09511v1
- 점수: final 96.0

On-device tuning of deep neural networks enables long-term adaptation at the edge while preserving data privacy. However, the high computational and memory demands of backpropagation pose significant challenges for ultra-low-power, memory-constrained extreme-edge devices. These challenges are further amplified for attention-based models due to their architectural complexity and computational scale. We present TrainDeeploy, a framework that unifies efficient inference and on-device training on heterogeneous ultra-low-power System-on-Chips (SoCs). TrainDeeploy provides the first complete on-device training pipeline for extreme-edge SoCs supporting both Convolutional Neural Networks (CNNs) and Transformer models, together with multiple training strategies such as selective layer-wise fine-tuning and Low-Rank Adaptation (LoRA). On a RISC-V-based heterogeneous SoC, we demonstrate the first end-to-end on-device fine-tuning of a Compact Convolutional Transformer (CCT), achieving up to 11 trained images per second. We show that LoRA reduces dynamic memory usage by 23%, decreases the number of trainable parameters and gradients by 15x, and reduces memory transfer volume by 1.6x compared to full backpropagation. TrainDeeploy achieves up to 4.6 FLOP/cycle on CCT (0.28M parameters, 71-126M FLOPs) and up to 13.4 FLOP/cycle on Deep-AE (0.27M parameters, 0.8M FLOPs), while expanding the scope of prior frameworks to support both CNN and Transformer models with parameter-efficient tuning on extreme-edge platforms.

-> rk3588 기반 edge device에서 AI 모델 효율적 튜닝을 위한 하드웨어 가속 기술

### LCAMV: High-Accuracy 3D Reconstruction of Color-Varying Objects Using LCA Correction and Minimum-Variance Fusion in Structured Light (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10456v1
- 점수: final 96.0

Accurate 3D reconstruction of colored objects with structured light (SL) is hindered by lateral chromatic aberration (LCA) in optical components and uneven noise characteristics across RGB channels. This paper introduces lateral chromatic aberration correction and minimum-variance fusion (LCAMV), a robust 3D reconstruction method that operates with a single projector-camera pair without additional hardware or acquisition constraints. LCAMV analytically models and pixel-wise compensates LCA in both the projector and camera, then adaptively fuses multi-channel phase data using a Poisson-Gaussian noise model and minimum-variance estimation. Unlike existing methods that require extra hardware or multiple exposures, LCAMV enables fast acquisition. Experiments on planar and non-planar colored surfaces show that LCAMV outperforms grayscale conversion and conventional channel-weighting, reducing depth error by up to 43.6\%. These results establish LCAMV as an effective solution for high-precision 3D reconstruction of nonuniformly colored objects.

-> 색상 보정 및 최소 분산 융합 기술로 스포츠 장면의 정밀 3D 재구성 및 영상 보정에 적용 가능

### PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09216v1
- 점수: final 96.0

On-device deployments of large language models (LLMs) are rapidly proliferating across mobile and edge platforms. LLM inference comprises a compute-intensive prefill phase and a memory bandwidth-intensive decode phase, and the decode phase has been widely recognized as well-suited to processing-in-memory (PIM) in both academia and industry. However, practical PIM-enabled systems face two obstacles between these phases, a memory attribute inconsistency in which prefill favors placing weights in a cacheable region for reuse whereas decode requires weights in a non-cacheable region to reliably trigger PIM, and a weight layout inconsistency between host-friendly and PIM-aware layouts. To address these problems, we introduce \textit{PIM-SHERPA}, a software-only method for efficient on-device LLM inference by resolving PIM memory attribute and layout inconsistencies. PIM-SHERPA provides two approaches, DRAM double buffering (DDB), which keeps a single PIM-aware weights in the non-cacheable region while prefetching the swizzled weights of the next layer into small cacheable buffers, and online weight rearrangement with swizzled memory copy (OWR), which performs the on-demand swizzled memory copy immediately before GEMM. Compared to a baseline PIM emulation system, PIM-SHERPA achieves approximately 47.8 - 49.7\% memory capacity savings while maintaining comparable performance to the theoretical maximum on the Llama 3.2 model. To the best of our knowledge, this is the first work to identify the memory attribute inconsistency and propose effective solutions on product-level PIM-enabled systems.

-> rk3588 엣지 디바이스에서의 효율적인 AI 추론을 위한 메모리 최적화 방법으로 실시간 스포츠 분석에 적용 가능

### CycleULM: A unified label-free deep learning framework for ultrasound localisation microscopy (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09840v1
- 점수: final 96.0

Super-resolution ultrasound via microbubble (MB) localisation and tracking, also known as ultrasound localisation microscopy (ULM), can resolve microvasculature beyond the acoustic diffraction limit. However, significant challenges remain in localisation performance and data acquisition and processing time. Deep learning methods for ULM have shown promise to address these challenges, however, they remain limited by in vivo label scarcity and the simulation-to-reality domain gap. We present CycleULM, the first unified label-free deep learning framework for ULM. CycleULM learns a physics-emulating translation between the real contrast-enhanced ultrasound (CEUS) data domain and a simplified MB-only domain, leveraging the power of CycleGAN without requiring paired ground truth data. With this translation, CycleULM removes dependence on high-fidelity simulators or labelled data, and makes MB localisation and tracking substantially easier. Deployed as modular plug-and-play components within existing pipelines or as an end-to-end processing framework, CycleULM delivers substantial performance gains across both in silico and in vivo datasets. Specifically, CycleULM improves image contrast (contrast-to-noise ratio) by up to 15.3 dB and sharpens CEUS resolution with a 2.5{\times} reduction in the full width at half maximum of the point spread function. CycleULM also improves MB localisation performance, with up to +40% recall, +46% precision, and a -14.0 μm mean localisation error, yielding more faithful vascular reconstructions. Importantly, CycleULM achieves real-time processing throughput at 18.3 frames per second with order-of-magnitude speed-ups (up to ~14.5{\times}). By combining label-free learning, performance enhancement, and computational efficiency, CycleULM provides a practical pathway toward robust, real-time ULM and accelerates its translation to clinical applications.

-> 실시간 처리 성능과 물리 모델링 기술이 스포츠 영상 분석에 직접적으로 적용 가능

### HyPER-GAN: Hybrid Patch-Based Image-to-Image Translation for Real-Time Photorealism Enhancement (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10604v1
- 점수: final 96.0

Generative models are widely employed to enhance the photorealism of synthetic data for training computer vision algorithms. However, they often introduce visual artifacts that degrade the accuracy of these algorithms and require high computational resources, limiting their applicability in real-time training or evaluation scenarios. In this paper, we propose Hybrid Patch Enhanced Realism Generative Adversarial Network (HyPER-GAN), a lightweight image-to-image translation method based on a U-Net-style generator designed for real-time inference. The model is trained using paired synthetic and photorealism-enhanced images, complemented by a hybrid training strategy that incorporates matched patches from real-world data to improve visual realism and semantic consistency. Experimental results demonstrate that HyPER-GAN outperforms state-of-the-art paired image-to-image translation methods in terms of inference latency, visual realism, and semantic robustness. Moreover, it is illustrated that the proposed hybrid training strategy indeed improves visual quality and semantic consistency compared to training the model solely with paired synthetic and photorealism-enhanced images. Code and pretrained models are publicly available for download at: https://github.com/stefanos50/HyPER-GAN

-> Lightweight real-time image enhancement technology that could be applied to improve the visual quality of sports footage on edge devices.

### MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09116v1
- 점수: final 93.6

We present MetaSpectra+, a compact multifunctional camera that supports two operating modes: (1) snapshot HDR + hyperspectral or (2) snapshot polarization + hyperspectral imaging. It utilizes a novel metasurface-refractive assembly that splits the incident beam into multiple channels and independently controls each channel's dispersion, exposure, and polarization. Unlike prior multifunctional metasurface imagers restricted to narrow (10-100 nm) bands, MetaSpectra+ operates over nearly the entire visible spectrum (250 nm). Relative to snapshot hyperspectral imagers, it achieves the shortest total track length and the highest reconstruction accuracy on benchmark datasets. The demonstrated prototype reconstructs high-quality hyperspectral datacubes and either an HDR image or two orthogonal polarization channels from a single snapshot.

-> 다중 기능 컴팩트 카메라 기술이 스포츠 촬영 및 분석에 적용 가능

### Decoder-Free Distillation for Quantized Image Restoration (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09624v1
- 점수: final 93.6

Quantization-Aware Training (QAT), combined with Knowledge Distillation (KD), holds immense promise for compressing models for edge deployment. However, joint optimization for precision-sensitive image restoration (IR) to recover visual quality from degraded images remains largely underexplored. Directly adapting QAT-KD to low-level vision reveals three critical bottlenecks: teacher-student capacity mismatch, spatial error amplification during decoder distillation, and an optimization "tug-of-war" between reconstruction and distillation losses caused by quantization noise. To tackle these, we introduce Quantization-aware Distilled Restoration (QDR), a framework for edge-deployed IR. QDR eliminates capacity mismatch via FP32 self-distillation and prevents error amplification through Decoder-Free Distillation (DFD), which corrects quantization errors strictly at the network bottleneck. To stabilize the optimization tug-of-war, we propose a Learnable Magnitude Reweighting (LMR) that dynamically balances competing gradients. Finally, we design an Edge-Friendly Model (EFM) featuring a lightweight Learnable Degradation Gating (LDG) to dynamically modulate spatial degradation localization. Extensive experiments across four IR tasks demonstrate that our Int8 model recovers 96.5% of FP32 performance, achieves 442 frames per second (FPS) on an NVIDIA Jetson Orin, and boosts downstream object detection by 16.3 mAP

-> 실시간 처리가 가능한 엣지 기반 이미지 복원 기술로 스포츠 촬영 장비에 직접 적용 가능

### Safety-critical Control Under Partial Observability: Reach-Avoid POMDP meets Belief Space Control (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10572v1
- 점수: final 93.6

Partially Observable Markov Decision Processes (POMDPs) provide a principled framework for robot decision-making under uncertainty. Solving reach-avoid POMDPs, however, requires coordinating three distinct behaviors: goal reaching, safety, and active information gathering to reduce uncertainty. Existing online POMDP solvers attempt to address all three within a single belief tree search, but this unified approach struggles with the conflicting time scales inherent to these objectives. We propose a layered, certificate-based control architecture that operates directly in belief space, decoupling goal reaching, information gathering, and safety into modular components. We introduce Belief Control Lyapunov Functions (BCLFs) that formalize information gathering as a Lyapunov convergence problem in belief space, and show how they can be learned via reinforcement learning. For safety, we develop Belief Control Barrier Functions (BCBFs) that leverage conformal prediction to provide probabilistic safety guarantees over finite horizons. The resulting control synthesis reduces to lightweight quadratic programs solvable in real time, even for non-Gaussian belief representations with dimension $>10^4$. Experiments in simulation and on a space-robotics platform demonstrate real-time performance and improved safety and task success compared to state-of-the-art constrained POMDP solvers.

-> 실시간 제어 시스템 및 불확실성 상황에서의 의사결정 기술이 스포츠 장면 분석에 적용 가능

### UHD Image Deblurring via Autoregressive Flow with Ill-conditioned Constraints (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10517v1
- 점수: final 93.6

Ultra-high-definition (UHD) image deblurring poses significant challenges for UHD restoration methods, which must balance fine-grained detail recovery and practical inference efficiency. Although prominent discriminative and generative methods have achieved remarkable results, a trade-off persists between computational cost and the ability to generate fine-grained detail for UHD image deblurring tasks. To further alleviate these issues, we propose a novel autoregressive flow method for UHD image deblurring with an ill-conditioned constraint. Our core idea is to decompose UHD restoration into a progressive, coarse-to-fine process: at each scale, the sharp estimate is formed by upsampling the previous-scale result and adding a current-scale residual, enabling stable, stage-wise refinement from low to high resolution. We further introduce Flow Matching to model residual generation as a conditional vector field and perform few-step ODE sampling with efficient Euler/Heun solvers, enriching details while keeping inference affordable. Since multi-step generation at UHD can be numerically unstable, we propose an ill-conditioning suppression scheme by imposing condition-number regularization on a feature-induced attention matrix, improving convergence and cross-scale consistency. Our method demonstrates promising performance on blurred images at 4K (3840$\times$2160) or higher resolutions.

-> 고해상도 이미지 보정 기술이 스포츠 영상 보정에 직접적으로 적용 가능

### Frames2Residual: Spatiotemporal Decoupling for Self-Supervised Video Denoising (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10417v1
- 점수: final 93.6

Self-supervised video denoising methods typically extend image-based frameworks into the temporal dimension, yet they often struggle to integrate inter-frame temporal consistency with intra-frame spatial specificity. Existing Video Blind-Spot Networks (BSNs) require noise independence by masking the center pixel, this constraint prevents the use of spatial evidence for texture recovery, thereby severing spatiotemporal correlations and causing texture loss. To address this, we propose Frames2Residual (F2R), a spatiotemporal decoupling framework that explicitly divides self-supervised training into two distinct stages: blind temporal consistency modeling and non-blind spatial texture recovery. In Stage 1, a blind temporal estimator learns inter-frame consistency using a frame-wise blind strategy, producing a temporally consistent anchor. In Stage 2, a non-blind spatial refiner leverages this anchor to safely reintroduce the center frame and recover intra-frame high-frequency spatial residuals while preserving temporal stability. Extensive experiments demonstrate that our decoupling strategy allows F2R to outperform existing self-supervised methods on both sRGB and raw video benchmarks.

-> 비디오 노이즈 제거 기술로 스포츠 경기 촬영 영상의 품질 향상에 적용 가능

### DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10538v1
- 점수: final 93.6

Scene Graph Generation (SGG) aims to extract a detailed graph structure from an image, a representation that holds significant promise as a robust intermediate step for complex downstream tasks like reasoning for embodied agents. However, practical deployment in real-world applications - especially on resource constrained edge devices - requires speed and resource efficiency, challenges that have received limited attention in existing research. To bridge this gap, we introduce DSFlash, a low-latency model for panoptic scene graph generation designed to overcome these limitations. DSFlash can process a video stream at 56 frames per second on a standard RTX 3090 GPU, without compromising performance against existing state-of-the-art methods. Crucially, unlike prior approaches that often restrict themselves to salient relationships, DSFlash computes comprehensive scene graphs, offering richer contextual information while maintaining its superior latency. Furthermore, DSFlash is light on resources, requiring less than 24 hours to train on a single, nine-year-old GTX 1080 GPU. This accessibility makes DSFlash particularly well-suited for researchers and practitioners operating with limited computational resources, empowering them to adapt and fine-tune SGG models for specialized applications.

-> 실시간 영상 처리 및 에지 디바이스에서의 실행 가능성이 스포츠 자동 촬영 장치와 관련 있음

### From Imitation to Intuition: Intrinsic Reasoning for Open-Instance Video Classification (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10300v1
- 점수: final 92.0

Conventional video classification models, acting as effective imitators, excel in scenarios with homogeneous data distributions. However, real-world applications often present an open-instance challenge, where intra-class variations are vast and complex, beyond existing benchmarks. While traditional video encoder models struggle to fit these diverse distributions, vision-language models (VLMs) offer superior generalization but have not fully leveraged their reasoning capabilities (intuition) for such tasks. In this paper, we bridge this gap with an intrinsic reasoning framework that evolves open-instance video classification from imitation to intuition. Our approach, namely DeepIntuit, begins with a cold-start supervised alignment to initialize reasoning capability, followed by refinement using Group Relative Policy Optimization (GRPO) to enhance reasoning coherence through reinforcement learning. Crucially, to translate this reasoning into accurate classification, DeepIntuit then introduces an intuitive calibration stage. In this stage, a classifier is trained on this intrinsic reasoning traces generated by the refined VLM, ensuring stable knowledge transfer without distribution mismatch. Extensive experiments demonstrate that for open-instance video classification, DeepIntuit benefits significantly from transcending simple feature imitation and evolving toward intrinsic reasoning. Our project is available at https://bwgzk-keke.github.io/DeepIntuit/.

-> 비디오 분류 프레임워크로 스포츠 장면 분석에 적용 가능

### PPGuide: Steering Diffusion Policies with Performance Predictive Guidance (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10980v1
- 점수: final 90.4

Diffusion policies have shown to be very efficient at learning complex, multi-modal behaviors for robotic manipulation. However, errors in generated action sequences can compound over time which can potentially lead to failure. Some approaches mitigate this by augmenting datasets with expert demonstrations or learning predictive world models which might be computationally expensive. We introduce Performance Predictive Guidance (PPGuide), a lightweight, classifier-based framework that steers a pre-trained diffusion policy away from failure modes at inference time. PPGuide makes use of a novel self-supervised process: it uses attention-based multiple instance learning to automatically estimate which observation-action chunks from the policy's rollouts are relevant to success or failure. We then train a performance predictor on this self-labeled data. During inference, this predictor provides a real-time gradient to guide the policy toward more robust actions. We validated our proposed PPGuide across a diverse set of tasks from the Robomimic and MimicGen benchmarks, demonstrating consistent improvements in performance.

-> 경량 확산 정책 프레임워크로 스포츠 하이라이트 생성 및 동작 분석에 적용 가능

### Two Teachers Better Than One: Hardware-Physics Co-Guided Distributed Scientific Machine Learning (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09032v1
- 점수: final 90.0

Scientific machine learning (SciML) is increasingly applied to in-field processing, controlling, and monitoring; however, wide-area sensing, real-time demands, and strict energy and reliability constraints make centralized SciML implementation impractical. Most SciML models assume raw data aggregation at a central node, incurring prohibitively high communication latency and energy costs; yet, distributing models developed for general-purpose ML often breaks essential physical principles, resulting in degraded performance. To address these challenges, we introduce EPIC, a hardware- and physics-co-guided distributed SciML framework, using full-waveform inversion (FWI) as a representative task. EPIC performs lightweight local encoding on end devices and physics-aware decoding at a central node. By transmitting compact latent features rather than high-volume raw data and by using cross-attention to capture inter-receiver wavefield coupling, EPIC significantly reduces communication cost while preserving physical fidelity. Evaluated on a distributed testbed with five end devices and one central node, and across 10 datasets from OpenFWI, EPIC reduces latency by 8.9$\times$ and communication energy by 33.8$\times$, while even improving reconstruction fidelity on 8 out of 10 datasets.

-> 엣지 디바이스용 분산 컴퓨팅 프레임워크로 물리 인식 처리가 스포츠 분석에 적용 가능

### Multi-Person Pose Estimation Evaluation Using Optimal Transportation and Improved Pose Matching (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10398v1
- 점수: final 89.6

In Multi-Person Pose Estimation, many metrics place importance on ranking of pose detection confidence scores. Current metrics tend to disregard false-positive poses with low confidence, focusing primarily on a larger number of high-confidence poses. Consequently, these metrics may yield high scores even when many false-positive poses with low confidence are detected. For fair evaluation taking into account a tradeoff between true-positive and false-positive poses, this paper proposes Optimal Correction Cost for pose (OCpose), which evaluates detected poses against pose annotations as an optimal transportation. For the fair tradeoff between true-positive and false-positive poses, OCpose equally evaluates all the detected poses regardless of their confidence scores. In OCpose, on the other hand, the confidence score of each pose is utilized to improve the reliability of matching scores between the estimated pose and pose annotations. As a result, OCpose provides a different perspective assessment than other confidence ranking-based metrics.

-> Pose estimation evaluation methodology applicable to sports movement analysis

### TemporalDoRA: Temporal PEFT for Robust Surgical Video Question Answering (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09696v1
- 점수: final 88.8

Surgical Video Question Answering (VideoQA) requires accurate temporal grounding while remaining robust to natural variation in how clinicians phrase questions, where linguistic bias can arise. Standard Parameter Efficient Fine Tuning (PEFT) methods adapt pretrained projections without explicitly modeling frame-to-frame interactions within the adaptation pathway, limiting their ability to exploit sparse temporal evidence. We introduce TemporalDoRA, a video-specific PEFT formulation that extends Weight-Decomposed Low-Rank Adaptation by (i) inserting lightweight temporal Multi-Head Attention (MHA) inside the low-rank bottleneck of the vision encoder and (ii) selectively applying weight decomposition only to the trainable low-rank branch rather than the full adapted weight. This design enables temporally-aware updates while preserving a frozen backbone and stable scaling. By mixing information across frames within the adaptation subspace, TemporalDoRA steers updates toward temporally consistent visual cues and improves robustness with minimal parameter overhead. To benchmark this setting, we present REAL-Colon-VQA, a colonoscopy VideoQA dataset with 6,424 clip--question pairs, including paired rephrased Out-of-Template questions to evaluate sensitivity to linguistic variation. TemporalDoRA improves Out-of-Template performance, and ablation studies confirm that temporal mixing inside the low-rank branch is the primary driver of these gains. We also validate on EndoVis18-VQA adapted to short clips and observe consistent improvements on the Out-of-Template split. Code and dataset available at~\href{https://anonymous.4open.science/r/TemporalDoRA-BFC8/}{Anonymous GitHub}.

-> 수술 비디오 QA를 위한 시간적 처리 기술은 스포츠 분석으로 적용될 수 있습니다

### WikiCLIP: An Efficient Contrastive Baseline for Open-domain Visual Entity Recognition (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09921v1
- 점수: final 88.0

Open-domain visual entity recognition (VER) seeks to associate images with entities in encyclopedic knowledge bases such as Wikipedia. Recent generative methods tailored for VER demonstrate strong performance but incur high computational costs, limiting their scalability and practical deployment. In this work, we revisit the contrastive paradigm for VER and introduce WikiCLIP, a simple yet effective framework that establishes a strong and efficient baseline for open-domain VER. WikiCLIP leverages large language model embeddings as knowledge-rich entity representations and enhances them with a Vision-Guided Knowledge Adaptor (VGKA) that aligns textual semantics with visual cues at the patch level. To further encourage fine-grained discrimination, a Hard Negative Synthesis Mechanism generates visually similar but semantically distinct negatives during training. Experimental results on popular open-domain VER benchmarks, such as OVEN, demonstrate that WikiCLIP significantly outperforms strong baselines. Specifically, WikiCLIP achieves a 16% improvement on the challenging OVEN unseen set, while reducing inference latency by nearly 100 times compared with the leading generative model, AutoVER. The project page is available at https://artanic30.github.io/project_pages/WikiCLIP/

-> 효율적인 대비 학습 방식으로 스포츠 장면의 시각적 엔티티 인식에 적용 가능

### Event-based Photometric Stereo via Rotating Illumination and Per-Pixel Learning (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10748v1
- 점수: final 88.0

Photometric stereo is a technique for estimating surface normals using images captured under varying illumination. However, conventional frame-based photometric stereo methods are limited in real-world applications due to their reliance on controlled lighting, and susceptibility to ambient illumination. To address these limitations, we propose an event-based photometric stereo system that leverages an event camera, which is effective in scenarios with continuously varying scene radiance and high dynamic range conditions. Our setup employs a single light source moving along a predefined circular trajectory, eliminating the need for multiple synchronized light sources and enabling a more compact and scalable design. We further introduce a lightweight per-pixel multi-layer neural network that directly predicts surface normals from event signals generated by intensity changes as the light source rotates, without system calibration. Experimental results on benchmark datasets and real-world data collected with our data acquisition system demonstrate the effectiveness of our method, achieving a 7.12\% reduction in mean angular error compared to existing event-based photometric stereo methods. In addition, our method demonstrates robustness in regions with sparse event activity, strong ambient illumination, and scenes affected by specularities.

-> 이벤트 기반 카메라와 경량 학습 방식으로 스포츠 촬영 및 보정에 활용 가능

### GroundCount: Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10978v1
- 점수: final 88.0

Vision Language Models (VLMs) exhibit persistent hallucinations in counting tasks, with accuracy substantially lower than other visual reasoning tasks (excluding sentiment). This phenomenon persists even in state-of-the-art reasoning-capable VLMs. Conversely, CNN-based object detection models (ODMs) such as YOLO excel at spatial localization and instance counting with minimal computational overhead. We propose GroundCount, a framework that augments VLMs with explicit spatial grounding from ODMs to mitigate counting hallucinations. In the best case, our prompt-based augmentation strategy achieves 81.3% counting accuracy on the best-performing model (Ovis2.5-2B) - a 6.6pp improvement - while reducing inference time by 22% through elimination of hallucination-driven reasoning loops for stronger models. We conduct comprehensive ablation studies demonstrating that positional encoding is a critical component, being beneficial for stronger models but detrimental for weaker ones. Confidence scores, by contrast, introduce noise for most architectures and their removal improves performance in four of five evaluated models. We further evaluate feature-level fusion architectures, finding that explicit symbolic grounding via structured prompts outperforms implicit feature fusion despite sophisticated cross-attention mechanisms. Our approach yields consistent improvements across four of five evaluated VLM architectures (6.2--7.5pp), with one architecture exhibiting degraded performance due to incompatibility between its iterative reflection mechanisms and structured prompts. These results suggest that counting failures stem from fundamental spatial-semantic integration limitations rather than architecture-specific deficiencies, while highlighting the importance of architectural compatibility in augmentation strategies.

-> 객체 탐지와 비전-언어 모델 결합으로 스포츠 장면 내 객체 분석에 적용 가능

### M3GCLR: Multi-View Mini-Max Infinite Skeleton-Data Game Contrastive Learning For Skeleton-Based Action Recognition (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09367v1
- 점수: final 88.0

In recent years, contrastive learning has drawn significant attention as an effective approach to reducing reliance on labeled data. However, existing methods for self-supervised skeleton-based action recognition still face three major limitations: insufficient modeling of view discrepancies, lack of effective adversarial mechanisms, and uncontrollable augmentation perturbations. To tackle these issues, we propose the Multi-view Mini-Max infinite skeleton-data Game Contrastive Learning for skeleton-based action Recognition (M3GCLR), a game-theoretic contrastive framework. First, we establish the Infinite Skeleton-data Game (ISG) model and the ISG equilibrium theorem, and further provide a rigorous proof, enabling mini-max optimization based on multi-view mutual information. Then, we generate normal-extreme data pairs through multi-view rotation augmentation and adopt temporally averaged input as a neutral anchor to achieve structural alignment, thereby explicitly characterizing perturbation strength. Next, leveraging the proposed equilibrium theorem, we construct a strongly adversarial mini-max skeleton-data game to encourage the model to mine richer action-discriminative information. Finally, we introduce the dual-loss equilibrium optimizer to optimize the game equilibrium, allowing the learning process to maximize action-relevant information while minimizing encoding redundancy, and we prove the equivalence between the proposed optimizer and the ISG model. Extensive Experiments show that M3GCLR achieves three-stream 82.1%, 85.8% accuracy on NTU RGB+D 60 (X-Sub, X-View) and 72.3%, 75.0% accuracy on NTU RGB+D 120 (X-Sub, X-Set). On PKU-MMD Part I and II, it attains 89.1%, 45.2% in three-stream respectively, all results matching or outperforming state-of-the-art performance. Ablation studies confirm the effectiveness of each component.

-> 골격 기반 동작 인식 기술이 스포츠 동작 분석에 직접적으로 적용 가능하여 선수 기술 분석과 자세 교정에 활용 가능

### NS-VLA: Towards Neuro-Symbolic Vision-Language-Action Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09542v1
- 점수: final 88.0

Vision-Language-Action (VLA) models are formulated to ground instructions in visual context and generate action sequences for robotic manipulation. Despite recent progress, VLA models still face challenges in learning related and reusable primitives, reducing reliance on large-scale data and complex architectures, and enabling exploration beyond demonstrations. To address these challenges, we propose a novel Neuro-Symbolic Vision-Language-Action (NS-VLA) framework via online reinforcement learning (RL). It introduces a symbolic encoder to embedding vision and language features and extract structured primitives, utilizes a symbolic solver for data-efficient action sequencing, and leverages online RL to optimize generation via expansive exploration. Experiments on robotic manipulation benchmarks demonstrate that NS-VLA outperforms previous methods in both one-shot training and data-perturbed settings, while simultaneously exhibiting superior zero-shot generalizability, high data efficiency and expanded exploration space. Our code is available.

-> 멀티모달 파싱 기술이 스포츠 영상에서 의미 있는 정보를 추출하고 구조화된 지식으로 변환하여 전략 분석에 활용 가능

### Logics-Parsing-Omni Technical Report (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09677v1
- 점수: final 86.4

Addressing the challenges of fragmented task definitions and the heterogeneity of unstructured data in multimodal parsing, this paper proposes the Omni Parsing framework. This framework establishes a Unified Taxonomy covering documents, images, and audio-visual streams, introducing a progressive parsing paradigm that bridges perception and cognition. Specifically, the framework integrates three hierarchical levels: 1) Holistic Detection, which achieves precise spatial-temporal grounding of objects or events to establish a geometric baseline for perception; 2) Fine-grained Recognition, which performs symbolization (e.g., OCR/ASR) and attribute extraction on localized objects to complete structured entity parsing; and 3) Multi-level Interpreting, which constructs a reasoning chain from local semantics to global logic. A pivotal advantage of this framework is its evidence anchoring mechanism, which enforces a strict alignment between high-level semantic descriptions and low-level facts. This enables ``evidence-based'' logical induction, transforming unstructured signals into standardized knowledge that is locatable, enumerable, and traceable. Building on this foundation, we constructed a standardized dataset and released the Logics-Parsing-Omni model, which successfully converts complex audio-visual signals into machine-readable structured knowledge. Experiments demonstrate that fine-grained perception and high-level cognition are synergistic, effectively enhancing model reliability. Furthermore, to quantitatively evaluate these capabilities, we introduce OmniParsingBench. Code, models and the benchmark are released at https://github.com/alibaba/Logics-Parsing/tree/master/Logics-Parsing-Omni.

-> Multimodal parsing techniques could be applicable for analyzing sports footage and extracting meaningful information.

### A Multi-Prototype-Guided Federated Knowledge Distillation Approach in AI-RAN Enabled Multi-Access Edge Computing System (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09727v1
- 점수: final 86.4

With the development of wireless network, Multi-Access Edge Computing (MEC) and Artificial Intelligence (AI)-native Radio Access Network (RAN) have attracted significant attention. Particularly, the integration of AI-RAN and MEC is envisioned to transform network efficiency and responsiveness. Therefore, it is valuable to investigate AI-RAN enabled MEC system. Federated learning (FL) nowadays is emerging as a promising approach for AI-RAN enabled MEC system, in which edge devices are enabled to train a global model cooperatively without revealing their raw data. However, conventional FL encounters the challenge in processing the non-independent and identically distributed (non-IID) data. Single prototype obtained by averaging the embedding vectors per class can be employed in FL to handle the data heterogeneity issue. Nevertheless, this may result in the loss of useful information owing to the average operation. Therefore, in this paper, a multi-prototype-guided federated knowledge distillation (MP-FedKD) approach is proposed. Particularly, self-knowledge distillation is integrated into FL to deal with the non-IID issue. To cope with the problem of information loss caused by single prototype-based strategy, multi-prototype strategy is adopted, where we present a conditional hierarchical agglomerative clustering (CHAC) approach and a prototype alignment scheme. Additionally, we design a novel loss function (called LEMGP loss) for each local client, where the relationship between global prototypes and local embedding will be focused. Extensive experiments over multiple datasets with various non-IID settings showcase that the proposed MP-FedKD approach outperforms the considered state-of-the-art baselines regarding accuracy, average accuracy and errors (RMSE and MAE).

-> 엣지 컴퓨팅 시스템을 위한 연합 학습 접근법이 AI 카메라 장치에 직접 적용 가능

### P-GSVC: Layered Progressive 2D Gaussian Splatting for Scalable Image and Video (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10551v1
- 점수: final 86.4

Gaussian splatting has emerged as a competitive explicit representation for image and video reconstruction. In this work, we present P-GSVC, the first layered progressive 2D Gaussian splatting framework that provides a unified solution for scalable Gaussian representation in both images and videos. P-GSVC organizes 2D Gaussian splats into a base layer and successive enhancement layers, enabling coarse-to-fine reconstructions. To effectively optimize this layered representation, we propose a joint training strategy that simultaneously updates Gaussians across layers, aligning their optimization trajectories to ensure inter-layer compatibility and a stable progressive reconstruction. P-GSVC supports scalability in terms of both quality and resolution. Our experiments show that the joint training strategy can gain up to 1.9 dB improvement in PSNR for video and 2.6 dB improvement in PSNR for image when compared to methods that perform sequential layer-wise training. Project page: https://longanwang-cs.github.io/PGSVC-webpage/

-> Image and video processing techniques directly applicable to creating photo-like videos and highlights

### Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10990v1
- 점수: final 86.4

Recent advances in text-to-image (T2I) generation have greatly improved visual quality, yet producing images that appear visually authentic to real-world photography remains challenging. This is partly due to biases in existing evaluation paradigms: human ratings and preference-trained metrics often favor visually vivid images with exaggerated saturation and contrast, which make generations often too vivid to be real even when prompted for realistic-style images. To address this issue, we present Color Fidelity Dataset (CFD) and Color Fidelity Metric (CFM) for objective evaluation of color fidelity in realistic-style generations. CFD contains over 1.3M real and synthetic images with ordered levels of color realism, while CFM employs a multimodal encoder to learn perceptual color fidelity. In addition, we propose a training-free Color Fidelity Refinement (CFR) that adaptively modulates spatial-temporal guidance scale in generation, thereby enhancing color authenticity. Together, CFD supports CFM for assessment, whose learned attention further guides CFR to refine T2I fidelity, forming a progressive framework for assessing and improving color fidelity in realistic-style T2I generation. The dataset and code are available at https://github.com/ZhengyaoFang/CFM.

-> 사실적인 외관을 위한 이미지 향상 기술을 직접적으로 다루어 스포츠 사진 생성에 적용 가능

### Muscle Synergy Priors Enhance Biomechanical Fidelity in Predictive Musculoskeletal Locomotion Simulation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10474v1
- 점수: final 85.6

Human locomotion emerges from high-dimensional neuromuscular control, making predictive musculoskeletal simulation challenging. We present a physiology-informed reinforcement-learning framework that constrains control using muscle synergies. We extracted a low-dimensional synergy basis from inverse musculoskeletal analyses of a small set of overground walking trials and used it as the action space for a muscle-driven three-dimensional model trained across variable speeds, slopes and uneven terrain. The resulting controller generated stable gait from 0.7-1.8 m/s and on $\pm$ 6$^{\circ}$ grades and reproduced condition-dependent modulation of joint angles, joint moments and ground reaction forces. Compared with an unconstrained controller, synergy-constrained control reduced non-physiological knee kinematics and kept knee moment profiles within the experimental envelope. Across conditions, simulated vertical ground reaction forces correlated strongly with human measurements, and muscle-activation timing largely fell within inter-subject variability. These results show that embedding neurophysiological structure into reinforcement learning can improve biomechanical fidelity and generalization in predictive human locomotion simulation with limited experimental data.

-> 근육 시너지 기반 강화 학습으로 스포츠 동작 분석 및 생체역학적 시뮬레이션 구현 가능

### From Ideal to Real: Stable Video Object Removal under Imperfect Conditions (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09283v1
- 점수: final 85.6

Removing objects from videos remains difficult in the presence of real-world imperfections such as shadows, abrupt motion, and defective masks. Existing diffusion-based video inpainting models often struggle to maintain temporal stability and visual consistency under these challenges. We propose Stable Video Object Removal (SVOR), a robust framework that achieves shadow-free, flicker-free, and mask-defect-tolerant removal through three key designs: (1) Mask Union for Stable Erasure (MUSE), a windowed union strategy applied during temporal mask downsampling to preserve all target regions observed within each window, effectively handling abrupt motion and reducing missed removals; (2) Denoising-Aware Segmentation (DA-Seg), a lightweight segmentation head on a decoupled side branch equipped with Denoising-Aware AdaLN and trained with mask degradation to provide an internal diffusion-aware localization prior without affecting content generation; and (3) Curriculum Two-Stage Training: where Stage I performs self-supervised pretraining on unpaired real-background videos with online random masks to learn realistic background and temporal priors, and Stage II refines on synthetic pairs using mask degradation and side-effect-weighted losses, jointly removing objects and their associated shadows/reflections while improving cross-domain robustness. Extensive experiments show that SVOR attains new state-of-the-art results across multiple datasets and degraded-mask benchmarks, advancing video object removal from ideal settings toward real-world applications.

-> 비디오 객체 제거 기술은 스포츠 경기 영상 편집에 적용 가능하여 원치 않는 객체를 제거하거나 특정 선수를 분리하는 데 활용될 수 있습니다.

### Novel Architecture of RPA In Oral Cancer Lesion Detection (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10928v1
- 점수: final 85.6

Accurate and early detection of oral cancer lesions is crucial for effective diagnosis and treatment. This study evaluates two RPA implementations, OC-RPAv1 and OC-RPAv2, using a test set of 31 images. OC-RPAv1 processes one image per prediction in an average of 0.29 seconds, while OCRPAv2 employs a Singleton design pattern and batch processing, reducing prediction time to just 0.06 seconds per image. This represents a 60-100x efficiency improvement over standard RPA methods, showcasing that design patterns and batch processing can enhance scalability and reduce costs in oral cancer detection

-> 효율적인 배치 처리 기술이 엣지 디바이스에서 스포츠 영상 처리에 적용 가능

### When to Lock Attention: Training-Free KV Control in Video Diffusion (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09657v1
- 점수: final 85.6

Maintaining background consistency while enhancing foreground quality remains a core challenge in video editing. Injecting full-image information often leads to background artifacts, whereas rigid background locking severely constrains the model's capacity for foreground generation. To address this issue, we propose KV-Lock, a training-free framework tailored for DiT-based video diffusion models. Our core insight is that the hallucination metric (variance of denoising prediction) directly quantifies generation diversity, which is inherently linked to the classifier-free guidance (CFG) scale. Building upon this, KV-Lock leverages diffusion hallucination detection to dynamically schedule two key components: the fusion ratio between cached background key-values (KVs) and newly generated KVs, and the CFG scale. When hallucination risk is detected, KV-Lock strengthens background KV locking and simultaneously amplifies conditional guidance for foreground generation, thereby mitigating artifacts and improving generation fidelity. As a training-free, plug-and-play module, KV-Lock can be easily integrated into any pre-trained DiT-based models. Extensive experiments validate that our method outperforms existing approaches in improved foreground quality with high background fidelity across various video editing tasks.

-> 비디오 확산 모델을 이용한 영상 편집 기술은 스포츠 영상 보정에 활용 가능하여 배경과 전경을 동시에 개선할 수 있습니다.

### Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09681v1
- 점수: final 85.6

State-of-the-art methods can recover accurate overall 3D human body motion from in-the-wild videos. However, they often fail to capture fine-grained articulations, especially in the feet, which are critical for applications such as gait analysis and animation. This limitation results from training datasets with inaccurate foot annotations and limited foot motion diversity. We address this gap with FootMR, a Foot Motion Refinement method that refines foot motion estimated by an existing human recovery model through lifting 2D foot keypoint sequences to 3D. By avoiding direct image input, FootMR circumvents inaccurate image-3D annotation pairs and can instead leverage large-scale motion capture data. To resolve ambiguities of 2D-to-3D lifting, FootMR incorporates knee and foot motion as context and predicts only residual foot motion. Generalization to extreme foot poses is further improved by representing joints in global rather than parent-relative rotations and applying extensive data augmentation. To support evaluation of foot motion reconstruction, we introduce MOOF, a 2D dataset of complex foot movements. Experiments on MOOF, MOYO, and RICH show that FootMR outperforms state-of-the-art methods, reducing ankle joint angle error on MOYO by up to 30% over the best video-based approach.

-> 멀티모달 인터리브 생성 기술은 스포츠 하이라이트 영상 제작에 적용 가능하여 콘텐츠 생성 효율성을 높일 수 있습니다.

### Fine-grained Motion Retrieval via Joint-Angle Motion Images and Token-Patch Late Interaction (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09930v1
- 점수: final 84.8

Text-motion retrieval aims to learn a semantically aligned latent space between natural language descriptions and 3D human motion skeleton sequences, enabling bidirectional search across the two modalities. Most existing methods use a dual-encoder framework that compresses motion and text into global embeddings, discarding fine-grained local correspondences, and thus reducing accuracy. Additionally, these global-embedding methods offer limited interpretability of the retrieval results. To overcome these limitations, we propose an interpretable, joint-angle-based motion representation that maps joint-level local features into a structured pseudo-image, compatible with pre-trained Vision Transformers. For text-to-motion retrieval, we employ MaxSim, a token-wise late interaction mechanism, and enhance it with Masked Language Modeling regularization to foster robust, interpretable text-motion alignment. Extensive experiments on HumanML3D and KIT-ML show that our method outperforms state-of-the-art text-motion retrieval approaches while offering interpretable fine-grained correspondences between text and motion. The code is available in the supplementary material.

-> 관절 각도 기반 움직임 검색 기술은 스포츠 하이라이트 생성에 적용 가능하여 특정 동작을 기반으로 영상을 검색하고 편집할 수 있습니다.

### A$^2$-Edit: Precise Reference-Guided Image Editing of Arbitrary Objects and Ambiguous Masks (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10685v1
- 점수: final 84.0

We propose \textbf{A$^2$-Edit}, a unified inpainting framework for arbitrary object categories, which allows users to replace any target region with a reference object using only a coarse mask. To address the issues of severe homogenization and limited category coverage in existing datasets, we construct a large-scale, multi-category dataset \textbf{UniEdit-500K}, which includes 8 major categories, 209 fine-grained subcategories, and a total of 500,104 image pairs. Such rich category diversity poses new challenges for the model, requiring it to automatically learn semantic relationships and distinctions across categories. To this end, we introduce the \textbf{Mixture of Transformer} module, which performs differentiated modeling of various object categories through dynamic expert selection, and further enhances cross-category semantic transfer and generalization through collaboration among experts. In addition, we propose a \textbf{Mask Annealing Training Strategy} (MATS) that progressively relaxes mask precision during training, reducing the model's reliance on accurate masks and improving robustness across diverse editing tasks. Extensive experiments on benchmarks such as VITON-HD and AnyInsertion demonstrate that A$^2$-Edit consistently outperforms existing approaches across all metrics, providing a new and efficient solution for arbitrary object editing.

-> 임의 객체 편 프레임워크로 스포츠 하이라이트 영상 및 이미지 정교한 편집 가능

### Variance-Aware Adaptive Weighting for Diffusion Model Training (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10391v1
- 점수: final 82.4

Diffusion models have recently achieved remarkable success in generative modeling, yet their training dynamics across different noise levels remain highly imbalanced, which can lead to inefficient optimization and unstable learning behavior. In this work, we investigate this imbalance from the perspective of loss variance across log-SNR levels and propose a variance-aware adaptive weighting strategy to address it. The proposed approach dynamically adjusts training weights based on the observed variance distribution, encouraging a more balanced optimization process across noise levels. Extensive experiments on CIFAR-10 and CIFAR-100 demonstrate that the proposed method consistently improves generative performance over standard training schemes, achieving lower Fréchet Inception Distance (FID) while also reducing performance variance across random seeds. Additional analysis, including loss-log-SNR visualization, variance heatmaps, and ablation studies, further reveal that the adaptive weighting effectively stabilizes training dynamics. These results highlight the potential of variance-aware training strategies for improving diffusion model optimization.

-> 확산 모델 학습 최적화로 스포츠 영상 보정 및 이미지 생성 품질 향상

### Parallel-in-Time Nonlinear Optimal Control via GPU-native Sequential Convex Programming (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10711v1
- 점수: final 80.0

Real-time trajectory optimization for nonlinear constrained autonomous systems is critical and typically performed by CPU-based sequential solvers. Specifically, reliance on global sparse linear algebra or the serial nature of dynamic programming algorithms restricts the utilization of massively parallel computing architectures like GPUs. To bridge this gap, we introduce a fully GPU-native trajectory optimization framework that combines sequential convex programming with a consensus-based alternating direction method of multipliers. By applying a temporal splitting strategy, our algorithm decouples the optimization horizon into independent, per-node subproblems that execute massively in parallel. The entire process runs fully on the GPU, eliminating costly memory transfers and large-scale sparse factorizations. This architecture naturally scales to multi-trajectory optimization. We validate the solver on a quadrotor agile flight task and a Mars powered descent problem using an on-board edge computing platform. Benchmarks reveal a sustained 4x throughput speedup and a 51% reduction in energy consumption over a heavily optimized 12-core CPU baseline. Crucially, the framework saturates the hardware, maintaining over 96% active GPU utilization to achieve planning rates exceeding 100 Hz. Furthermore, we demonstrate the solver's extensibility to robust Model Predictive Control by jointly optimizing dynamically coupled scenarios under stochastic disturbances, enabling scalable and safe autonomy.

-> GPU 기본 최적화 프레임워크로 실시간 스포츠 촬영 및 처리 성능 향상

### Are Video Reasoning Models Ready to Go Outside? (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10652v1
- 점수: final 80.0

In real-world deployment, vision-language models often encounter disturbances such as weather, occlusion, and camera motion. Under such conditions, their understanding and reasoning degrade substantially, revealing a gap between clean, controlled (i.e., unperturbed) evaluation settings and real-world robustness. To address this limitation, we propose ROVA, a novel training framework that improves robustness by modeling a robustness-aware consistency reward under spatio-temporal corruptions. ROVA introduces a difficulty-aware online training strategy that prioritizes informative samples based on the model's evolving capability. Specifically, it continuously re-estimates sample difficulty via self-reflective evaluation, enabling adaptive training with a robustness-aware consistency reward. We also introduce PVRBench, a new benchmark that injects real-world perturbations into embodied video datasets to assess both accuracy and reasoning quality under realistic disturbances. We evaluate ROVA and baselines on PVRBench, UrbanVideo, and VisBench, where open-source and proprietary models suffer up to 35% and 28% drops in accuracy and reasoning under realistic perturbations. ROVA effectively mitigates performance degradation, boosting relative accuracy by at least 24% and reasoning by over 9% compared with baseline models (QWen2.5/3-VL, InternVL2.5, Embodied-R). These gains transfer to clean standard benchmarks, yielding consistent improvements.

-> 실제 환경에서의 강건성 향상으로 다양한 조건에서 스포츠 촬영 안정성 확보

### An Event-Driven E-Skin System with Dynamic Binary Scanning and real time SNN Classification (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.10537v1
- 점수: final 80.0

This paper presents a novel hardware system for high-speed, event-sparse sampling-based electronic skin (e-skin)that integrates sensing and neuromorphic computing. The system is built around a 16x16 piezoresistive tactile array with front end and introduces a event-based binary scan search strategy to classify the digits. This event-driven strategy achieves a 12.8x reduction in scan counts, a 38.2x data compression rate and a 28.4x equivalent dynamic range, a 99% data sparsity, drastically reducing the data acquisition overhead. The resulting sparse data stream is processed by a multi-layer convolutional spiking neural network (Conv-SNN) implemented on an FPGA, which requires only 65% of the computation and 15.6% of the weight storage relative to a CNN. Despite these significant efficiency gains, the system maintains a high classification accuracy of 92.11% for real-time handwritten digit recognition. Furthermore, a real neuromorphic tactile dataset using Address Event Representation (AER) is constructed. This work demonstrates a fully integrated, event-driven pipeline from analog sensing to neuromorphic classification, offering an efficient solution for robotic perception and human-computer interaction.

-> 이벤트 기반 하드웨어 시스템으로 스포츠 장면 처리에 효율적인 접근 방식 제시. 이 논문은 이벤트 기반 이진 스캔 전략을 제안하며, 핵심은 데이터 효율성을 38.2배 향상시키는 것입니다.

### DCAU-Net: Differential Cross Attention and Channel-Spatial Feature Fusion for Medical Image Segmentation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09530v1
- 점수: final 80.0

Accurate medical image segmentation requires effective modeling of both long-range dependencies and fine-grained boundary details. While transformers mitigate the issue of insufficient semantic information arising from the limited receptive field inherent in convolutional neural networks, they introduce new challenges: standard self-attention incurs quadratic computational complexity and often assigns non-negligible attention weights to irrelevant regions, diluting focus on discriminative structures and ultimately compromising segmentation accuracy. Existing attention variants, although effective in reducing computational complexity, fail to suppress redundant computation and inadvertently impair global context modeling. Furthermore, conventional fusion strategies in encoder-decoder architectures, typically based on simple concatenation or summation, can not adaptively integrate high-level semantic information with low-level spatial details. To address these limitations, we propose DCAU-Net, a novel yet efficient segmentation framework with two key ideas. First, a new Differential Cross Attention (DCA) is designed to compute the difference between two independent softmax attention maps to adaptively highlight discriminative structures. By replacing pixel-wise key and value tokens with window-level summary tokens, DCA dramatically reduces computational complexity without sacrificing precision. Second, a Channel-Spatial Feature Fusion (CSFF) strategy is introduced to adaptively recalibrate features from skip connections and up-sampling paths through using sequential channel and spatial attention, effectively suppressing redundant information and amplifying salient cues. Experiments on two public benchmarks demonstrate that DCAU-Net achieves competitive performance with enhanced segmentation accuracy and robustness.

-> DCAU-Net의 차이 크로스 어텐션과 채널-공간 특징 융합 기술은 스포츠 장면에서 선수와 객체를 식별하고 중요한 순간을 분석하는 데 직접적으로 적용 가능하며, rk3588 엣지 디바이스에서 효율적으로 실행될 수 있는 계산 효율성 개선 방법을 제공합니다.

### Towards Unified Multimodal Interleaved Generation via Group Relative Policy Optimization (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09538v1
- 점수: final 80.0

Unified vision-language models have made significant progress in multimodal understanding and generation, yet they largely fall short in producing multimodal interleaved outputs, which is a crucial capability for tasks like visual storytelling and step-by-step visual reasoning. In this work, we propose a reinforcement learning-based post-training strategy to unlock this capability in existing unified models, without relying on large-scale multimodal interleaved datasets. We begin with a warm-up stage using a hybrid dataset comprising curated interleaved sequences and limited data for multimodal understanding and text-to-image generation, which exposes the model to interleaved generation patterns while preserving its pretrained capabilities. To further refine interleaved generation, we propose a unified policy optimization framework that extends Group Relative Policy Optimization (GRPO) to the multimodal setting. Our approach jointly models text and image generation within a single decoding trajectory and optimizes it with our novel hybrid rewards covering textual relevance, visual-text alignment, and structural fidelity. Additionally, we incorporate process-level rewards to provide step-wise guidance, enhancing training efficiency in complex multimodal tasks. Experiments on MMIE and InterleavedBench demonstrate that our approach significantly enhances the quality and coherence of multimodal interleaved generation.

-> Multimodal generation techniques could be applicable for creating highlight reels from sports footage.

### A Text-Native Interface for Generative Video Authoring (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09072v1
- 점수: final 80.0

Everyone can write their stories in freeform text format -- it's something we all learn in school. Yet storytelling via video requires one to learn specialized and complicated tools. In this paper, we introduce Doki, a text-native interface for generative video authoring, aligning video creation with the natural process of text writing. In Doki, writing text is the primary interaction: within a single document, users define assets, structure scenes, create shots, refine edits, and add audio. We articulate the design principles of this text-first approach and demonstrate Doki's capabilities through a series of examples. To evaluate its real-world use, we conducted a week-long deployment study with participants of varying expertise in video authoring. This work contributes a fundamental shift in generative video interfaces, demonstrating a powerful and accessible new way to craft visual stories.

-> 비디오 제작 인터페이스 기술로 하이라이트 편집과 관련 있으나 스포츠 자동 촬영에 직접적으로 연관되지 않음

### Evolving Prompt Adaptation for Vision-Language Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09493v1
- 점수: final 80.0

The adaptation of large-scale vision-language models (VLMs) to downstream tasks with limited labeled data remains a significant challenge. While parameter-efficient prompt learning methods offer a promising path, they often suffer from catastrophic forgetting of pre-trained knowledge. Toward addressing this limitation, our work is grounded in the insight that governing the evolutionary path of prompts is essential for forgetting-free adaptation. To this end, we propose EvoPrompt, a novel framework designed to explicitly steer the prompt trajectory for stable, knowledge-preserving fine-tuning. Specifically, our approach employs a Modality-Shared Prompt Projector (MPP) to generate hierarchical prompts from a unified embedding space. Critically, an evolutionary training strategy decouples low-rank updates into directional and magnitude components, preserving early-learned semantic directions while only adapting their magnitude, thus enabling prompts to evolve without discarding foundational knowledge. This process is further stabilized by Feature Geometric Regularization (FGR), which enforces feature decorrelation to prevent representation collapse. Extensive experiments demonstrate that EvoPrompt achieves state-of-the-art performance in few-shot learning while robustly preserving the original zero-shot capabilities of pre-trained VLMs.

-> Presents vision-language model adaptation techniques applicable to sports analysis but doesn't address edge computing

### M2Diff: Multi-Modality Multi-Task Enhanced Diffusion Model for MRI-Guided Low-Dose PET Enhancement (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.09075v1
- 점수: final 80.0

Positron emission tomography (PET) scans expose patients to radiation, which can be mitigated by reducing the dose, albeit at the cost of diminished quality. This makes low-dose (LD) PET recovery an active research area. Previous studies have focused on standard-dose (SD) PET recovery from LD PET scans and/or multi-modal scans, e.g., PET/CT or PET/MRI, using deep learning. While these studies incorporate multi-modal information through conditioning in a single-task model, such approaches may limit the capacity to extract modality-specific features, potentially leading to early feature dilution. Although recent studies have begun incorporating pathology-rich data, challenges remain in effectively leveraging multi-modality inputs for reconstructing diverse features, particularly in heterogeneous patient populations. To address these limitations, we introduce a multi-modality multi-task diffusion model (M2Diff) that processes MRI and LD PET scans separately to learn modality-specific features and fuse them via hierarchical feature fusion to reconstruct SD PET. This design enables effective integration of complementary structural and functional information, leading to improved reconstruction fidelity. We have validated the effectiveness of our model on both healthy and Alzheimer's disease brain datasets. The M2Diff achieves superior qualitative and quantitative performance on both datasets.

-> 다중 모달리티 이미지 처리 기술이 스포츠 영상 향상에 활용 가능

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
