# Weekly Paper Report: 20260301

> ISO 2026-W09




## 1. Weekly Briefing


금주 CS.CV 분야 평가 논문이 71건(전주 대비 32건 증가)으로 주목되며 평균 점수도 76.4로 소폭 상승(2.23p)했으나, 전체 논문 수(107건)는 4주 연속 하락세를 이어가며 전반적 감소 우려가 지속되고 있습니다. 특히 4주간 논문 건수 추이([0, 323, 108, 107])에서 금주에도 저조한 수준을 보였고, 키워드 히트 건수는 0건으로 정체 상태를 유지했습니다.



| Metric | Value | WoW |
|--------|-------|-----|
| Evaluated | 107 | - |
| Tier 1 | 79 | - |
| cs.CV | 71 | 32 |
| Keyword Hits | 0 | 0 |
| Avg Score | 76.4 | 2.23 |



### 4-Week Trend (downtrend)

| Week | Papers | cs.CV | Avg Score |
|------|--------|-------|-----------|

| W1 | 0 | 0 | 0.0 |

| W2 | 323 | 172 | 71.3 |

| W3 | 108 | 39 | 74.17 |

| W4 | 107 | 71 | 76.4 |





### Top Categories

| Category | Count | WoW % |
|----------|-------|-------|

| cs.CV | 71 | 82.1 |

| cs.AI | 23 | -28.1 |

| cs.LG | 17 | -48.5 |

| cs.RO | 12 | -40.0 |

| eess.IV | 7 | 40.0 |





- Graduated Reminds: 178
- Active Reminds: 2




### Tech Radar








#### TF-IDF Keywords

`video` (STABLE) `image` (STABLE) `motion` (NEW) `human` (NEW) `detection` (STABLE) `data` (STABLE) `learning` (STABLE) `language` (STABLE) `generation` (NEW) `temporal` (NEW) `multi` (NEW) `multimodal` (NEW) `camera` (NEW) `aware` (NEW) `scale` (NEW) `time` (STABLE) `vision` (NEW) `training` (STABLE) `alignment` (NEW) `real` (STABLE) `social` (DISAPPEARED) `visual` (DISAPPEARED) `knowledge` (DISAPPEARED) `accuracy` (DISAPPEARED) `graph` (DISAPPEARED) `control` (DISAPPEARED) `system` (DISAPPEARED) `object` (DISAPPEARED) `tasks` (DISAPPEARED) `quality` (DISAPPEARED) `classification` (DISAPPEARED) 







## 2. Top Papers


### 1. No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.23141v1](http://arxiv.org/abs/2602.23141v1)

- We propose a new unsupervised framework for online video stabilization. Unlike methods based on deep learning that require paired stable and unstable datasets, our approach instantiates the classical stabilization pipeline with three stages and incorporates a multithreaded buffering mechanism. This design addresses three longstanding challenges in end-to-end learning: limited data, poor controllability, and inefficiency on hardware with constrained resources. Existing benchmarks focus mainly on handheld videos with a forward view in visible light, which restricts the applicability of stabilization to domains such as UAV nighttime remote sensing. To fill this gap, we introduce a new multimodal UAV aerial video dataset (UAV-Test). Experiments show that our method consistently outperforms state-of-the-art online stabilizers in both quantitative metrics and visual quality, while achieving performance comparable to offline methods.
- 실시간 골격 그래프 구성 및 효율적 공간 추론 기술이 스포츠 장면 추적에 적합


### 2. Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device

- **Score**: 98.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.20161v1](http://arxiv.org/abs/2602.20161v1)

- Unified multimodal models can both understand and generate visual content within a single architecture. Existing models, however, remain data-hungry and too heavy for deployment on edge devices. We present Mobile-O, a compact vision-language-diffusion model that brings unified multimodal intelligence to a mobile device. Its core module, the Mobile Conditioning Projector (MCP), fuses vision-language features with a diffusion generator using depthwise-separable convolutions and layerwise alignment. This design enables efficient cross-modal conditioning with minimal computational cost. Trained on only a few million samples and post-trained in a novel quadruplet format (generation prompt, image, question, answer), Mobile-O jointly enhances both visual understanding and generation capabilities. Despite its efficiency, Mobile-O attains competitive or superior performance compared to other unified models, achieving 74% on GenEval and outperforming Show-O and JanusFlow by 5% and 11%, while running 6x and 11x faster, respectively. For visual understanding, Mobile-O surpasses them by 15.3% and 5.1% averaged across seven benchmarks. Running in only ~3s per 512x512 image on an iPhone, Mobile-O establishes the first practical framework for real-time unified multimodal understanding and generation on edge devices. We hope Mobile-O will ease future research in real-time unified multimodal intelligence running entirely on-device with no cloud dependency. Our code, models, datasets, and mobile application are publicly available at https://amshaker.github.io/Mobile-O/
- 에지 디바이스에서 실시간 멀티모달 이해/생성 가능해 영상 보정 및 분석에 직접 적용. Mobile-O의 경량 설계(fps 6~11배 향상)가 rk3588 호환성 높음.


### 3. U-Net-Based Generative Joint Source-Channel Coding for Wireless Image Transmission

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.22691v1](http://arxiv.org/abs/2602.22691v1)

- Deep learning (DL)-based joint source-channel coding (JSCC) methods have achieved remarkable success in wireless image transmission. However, these methods either focus on conventional distortion metrics that do not necessarily yield high perceptual quality or incur high computational complexity. In this paper, we propose two DL-based JSCC (DeepJSCC) methods that leverage deep generative architectures for wireless image transmission. Specifically, we propose G-UNet-JSCC, a scheme comprising an encoder and a U-Net-based generator serving as the decoder. Its skip connections enable multi-scale feature fusion to improve both pixel-level fidelity and perceptual quality of reconstructed images by integrating low- and high-level features. To further enhance pixel-level fidelity, the encoder and the U-Net-based decoder are jointly optimized using a weighted sum of structural similarity and mean-squared error (MSE) losses. Building upon G-UNet-JSCC, we further develop a DeepJSCC method called cGAN-JSCC, where the decoder is enhanced through adversarial training. In this scheme, we retain the encoder of G-UNet-JSCC and adversarially train the decoder's generator against a patch-based discriminator. cGAN-JSCC employs a two-stage training procedure. The outer stage trains the encoder and the decoder end-to-end using an MSE loss, while the inner stage adversarially trains the decoder's generator and the discriminator by minimizing a joint loss combining adversarial and distortion losses. Simulation results demonstrate that the proposed methods achieve superior pixel-level fidelity and perceptual quality on both high- and low-resolution images. For low-resolution images, cGAN-JSCC achieves better reconstruction performance and greater robustness to channel variations than G-UNet-JSCC.
- U-Net 아키텍처와 생성적 방법이 비디오/이미지 처리에 적용 가능


### 4. SCOPE: Skeleton Graph-Based Computation-Efficient Framework for Autonomous UAV Exploration

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.22707v1](http://arxiv.org/abs/2602.22707v1)

- Autonomous exploration in unknown environments is key for mobile robots, helping them perceive, map, and make decisions in complex areas. However, current methods often rely on frequent global optimization, suffering from high computational latency and trajectory oscillation, especially on resource-constrained edge devices. To address these limitations, we propose SCOPE, a novel framework that incrementally constructs a real-time skeletal graph and introduces Implicit Unknown Region Analysis for efficient spatial reasoning. The planning layer adopts a hierarchical on-demand strategy: the Proximal Planner generates smooth, high-frequency local trajectories, while the Region-Sequence Planner is activated only when necessary to optimize global visitation order. Comparative evaluations in simulation demonstrate that SCOPE achieves competitive exploration performance comparable to state-of-the-art global planners, while reducing computational cost by an average of 86.9%. Real-world experiments further validate the system's robustness and low latency in practical scenarios.
- U-Net 아키텍처와 생성적 방법이 비디오/이미지 처리에 적용 가능


### 5. From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.20630v1](http://arxiv.org/abs/2602.20630v1)

- Keypoint-based matching is a fundamental component of modern 3D vision systems, such as Structure-from-Motion (SfM) and SLAM. Most existing learning-based methods are trained on image pairs, a paradigm that fails to explicitly optimize for the long-term trackability of keypoints across sequences under challenging viewpoint and illumination changes. In this paper, we reframe keypoint detection as a sequential decision-making problem. We introduce TraqPoint, a novel, end-to-end Reinforcement Learning (RL) framework designed to optimize the \textbf{Tra}ck-\textbf{q}uality (Traq) of keypoints directly on image sequences. Our core innovation is a track-aware reward mechanism that jointly encourages the consistency and distinctiveness of keypoints across multiple views, guided by a policy gradient method. Extensive evaluations on sparse matching benchmarks, including relative pose estimation and 3D reconstruction, demonstrate that TraqPoint significantly outperforms some state-of-the-art (SOTA) keypoint detection and description methods.
- 실시간 동분할 기술은 스포츠 장면에서 움직임을 효과적으로 분석하여 하이라이트 편집에 적합합니다.


### 6. UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception

- **Score**: 94.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.23224v1](http://arxiv.org/abs/2602.23224v1)

- We present UniScale, a unified, scale-aware multi-view 3D reconstruction framework for robotic applications that flexibly integrates geometric priors through a modular, semantically informed design. In vision-based robotic navigation, the accurate extraction of environmental structure from raw image sequences is critical for downstream tasks. UniScale addresses this challenge with a single feed-forward network that jointly estimates camera intrinsics and extrinsics, scale-invariant depth and point maps, and the metric scale of a scene from multi-view images, while optionally incorporating auxiliary geometric priors when available. By combining global contextual reasoning with camera-aware feature representations, UniScale is able to recover the metric-scale of the scene. In robotic settings where camera intrinsics are known, they can be easily incorporated to improve performance, with additional gains obtained when camera poses are also available. This co-design enables robust, metric-aware 3D reconstruction within a single unified model. Importantly, UniScale does not require training from scratch, and leverages world priors exhibited in pre-existing models without geometric encoding strategies, making it particularly suitable for resource-constrained robotic teams. We evaluate UniScale on multiple benchmarks, demonstrating strong generalization and consistent performance across diverse environments. We will release our implementation upon acceptance.
- 자원 제약이 있는 엣지 디바이스에서 스포츠 촬영을 위한 3D 재구성 기술


### 7. Training Deep Stereo Matching Networks on Tree Branch Imagery: A Benchmark Study for Real-Time UAV Forestry Applications

- **Score**: 93.6 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.19763v1](http://arxiv.org/abs/2602.19763v1)

- Autonomous drone-based tree pruning needs accurate, real-time depth estimation from stereo cameras. Depth is computed from disparity maps using $Z = f B/d$, so even small disparity errors cause noticeable depth mistakes at working distances. Building on our earlier work that identified DEFOM-Stereo as the best reference disparity generator for vegetation scenes, we present the first study to train and test ten deep stereo matching networks on real tree branch images. We use the Canterbury Tree Branches dataset -- 5,313 stereo pairs from a ZED Mini camera at 1080P and 720P -- with DEFOM-generated disparity maps as training targets. The ten methods cover step-by-step refinement, 3D convolution, edge-aware attention, and lightweight designs. Using perceptual metrics (SSIM, LPIPS, ViTScore) and structural metrics (SIFT/ORB feature matching), we find that BANet-3D produces the best overall quality (SSIM = 0.883, LPIPS = 0.157), while RAFT-Stereo scores highest on scene-level understanding (ViTScore = 0.799). Testing on an NVIDIA Jetson Orin Super (16 GB, independently powered) mounted on our drone shows that AnyNet reaches 6.99 FPS at 1080P -- the only near-real-time option -- while BANet-2D gives the best quality-speed balance at 1.21 FPS. We also compare 720P and 1080P processing times to guide resolution choices for forestry drone systems.
- 실시간 스테레오 깊이 추정이 운동 동작 3D 분석에 필수. AnyNet 6.99fps로 rk3588에서 동작 캡처 가능성 높음.


### 8. Real-time Motion Segmentation with Event-based Normal Flow

- **Score**: 93.6 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.20790v1](http://arxiv.org/abs/2602.20790v1)

- Event-based cameras are bio-inspired sensors with pixels that independently and asynchronously respond to brightness changes at microsecond resolution, offering the potential to handle visual tasks in challenging scenarios. However, due to the sparse information content in individual events, directly processing the raw event data to solve vision tasks is highly inefficient, which severely limits the applicability of state-of-the-art methods in real-time tasks, such as motion segmentation, a fundamental task for dynamic scene understanding. Incorporating normal flow as an intermediate representation to compress motion information from event clusters within a localized region provides a more effective solution. In this work, we propose a normal flow-based motion segmentation framework for event-based vision. Leveraging the dense normal flow directly learned from event neighborhoods as input, we formulate the motion segmentation task as an energy minimization problem solved via graph cuts, and optimize it iteratively with normal flow clustering and motion model fitting. By using a normal flow-based motion model initialization and fitting method, the proposed system is able to efficiently estimate the motion models of independently moving objects with only a limited number of candidate models, which significantly reduces the computational complexity and ensures real-time performance, achieving nearly a 800x speedup in comparison to the open-source state-of-the-art method. Extensive evaluations on multiple public datasets fully demonstrate the accuracy and efficiency of our framework.
- 고속 움직임으로 인한 흐림 문제를 해결하여 스포츠 촬영의 품질을 향상시키는 데 중요합니다.


### 9. BRIDGE: Borderless Reconfiguration for Inclusive and Diverse Gameplay Experience via Embodiment Transformation

- **Score**: 93.6 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.23288v1](http://arxiv.org/abs/2602.23288v1)

- Training resources for parasports are limited, reducing opportunities for athletes and coaches to engage with sport-specific movements and tactical coordination. To address this gap, we developed BRIDGE, a system that integrates a reconstruction pipeline, which detects and tracks players from broadcast video to generate 3D play sequences, with an embodiment-aware visualization framework that decomposes head, trunk, and wheelchair base orientations to represent attention, intent, and mobility. We evaluated BRIDGE in two controlled studies with 20 participants (10 national wheelchair basketball team players and 10 amateur players). The results showed that BRIDGE significantly enhanced the perceived naturalness of player postures and made tactical intentions easier to understand. In addition, it supported functional classification by realistically conveying players' capabilities, which in turn improved participants' sense of self-efficacy. This work advances inclusive sports learning and accessible coaching practices, contributing to more equitable access to tactical resources in parasports.
- 스포츠 비디오 분석과 전략 이해를 위한 3D 재구성 시스템


### 10. MovieTeller: Tool-augmented Movie Synopsis with ID Consistent Progressive Abstraction

- **Score**: 93.6 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.23228v1](http://arxiv.org/abs/2602.23228v1)

- With the explosive growth of digital entertainment, automated video summarization has become indispensable for applications such as content indexing, personalized recommendation, and efficient media archiving. Automatic synopsis generation for long-form videos, such as movies and TV series, presents a significant challenge for existing Vision-Language Models (VLMs). While proficient at single-image captioning, these general-purpose models often exhibit critical failures in long-duration contexts, primarily a lack of ID-consistent character identification and a fractured narrative coherence. To overcome these limitations, we propose MovieTeller, a novel framework for generating movie synopses via tool-augmented progressive abstraction. Our core contribution is a training-free, tool-augmented, fact-grounded generation process. Instead of requiring costly model fine-tuning, our framework directly leverages off-the-shelf models in a plug-and-play manner. We first invoke a specialized face recognition model as an external "tool" to establish Factual Groundings--precise character identities and their corresponding bounding boxes. These groundings are then injected into the prompt to steer the VLM's reasoning, ensuring the generated scene descriptions are anchored to verifiable facts. Furthermore, our progressive abstraction pipeline decomposes the summarization of a full-length movie into a multi-stage process, effectively mitigating the context length limitations of current VLMs. Experiments demonstrate that our approach yields significant improvements in factual accuracy, character consistency, and overall narrative coherence compared to end-to-end baselines.
- 비디오 요약 및 캐릭터 일관성 유지 기술이 스포츠 하이라이트 자동 생성에 적용 가능






### Graduated Reminds


- **MING: An Automated CNN-to-Edge MLIR HLS framework** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.11966v1](http://arxiv.org/abs/2602.11966v1)

- **Quantization-Aware Collaborative Inference for Large Embodied AI Models** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.13052v1](http://arxiv.org/abs/2602.13052v1)

- **FlexAM: Flexible Appearance-Motion Decomposition for Versatile Video Generation Control** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.13185v1](http://arxiv.org/abs/2602.13185v1)

- **LAF-YOLOv10 with Partial Convolution Backbone, Attention-Guided Feature Pyramid, Auxiliary P2 Head, and Wise-IoU Loss for Small Object Detection in Drone Aerial Imagery** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.13378v1](http://arxiv.org/abs/2602.13378v1)

- **Floe: Federated Specialization for Real-Time LLM-SLM Inference** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.14302v1](http://arxiv.org/abs/2602.14302v1)

- **YOLO26: A Comprehensive Architecture Overview and Key Improvements** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.14582v1](http://arxiv.org/abs/2602.14582v1)

- **SpecFuse: A Spectral-Temporal Fusion Predictive Control Framework for UAV Landing on Oscillating Marine Platforms** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.15633v1](http://arxiv.org/abs/2602.15633v1)

- **How Reliable is Your Service at the Extreme Edge? Analytical Modeling of Computational Reliability** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.16362v1](http://arxiv.org/abs/2602.16362v1)

- **Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.16545v1](http://arxiv.org/abs/2602.16545v1)

- **Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.17997v1](http://arxiv.org/abs/2602.17997v1)

- **Flexi-NeurA: A Configurable Neuromorphic Accelerator with Adaptive Bit-Precision Exploration for Edge SNNs** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.18140v1](http://arxiv.org/abs/2602.18140v1)

- **A reliability- and latency-driven task allocation framework for workflow applications in the edge-hub-cloud continuum** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.18158v1](http://arxiv.org/abs/2602.18158v1)

- **A Self-Supervised Approach on Motion Calibration for Enhancing Physical Plausibility in Text-to-Motion** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.18199v1](http://arxiv.org/abs/2602.18199v1)

- **Multi-Level Conditioning by Pairing Localized Text and Sketch for Fashion Image Generation** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.18309v1](http://arxiv.org/abs/2602.18309v1)

- **Unifying Color and Lightness Correction with View-Adaptive Curve Adjustment for Robust 3D Novel View Synthesis** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.18322v1](http://arxiv.org/abs/2602.18322v1)

- **How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf** (score: None, graduated: 2026-02-24)
  [http://arxiv.org/abs/2602.18397v1](http://arxiv.org/abs/2602.18397v1)

- **Redefining the Down-Sampling Scheme of U-Net for Precision Biomedical Image Segmentation** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19412v1](http://arxiv.org/abs/2602.19412v1)

- **Laplacian Multi-scale Flow Matching for Generative Modeling** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19461v1](http://arxiv.org/abs/2602.19461v1)

- **Real-time Win Probability and Latent Player Ability via STATS X in Team Sports** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19513v1](http://arxiv.org/abs/2602.19513v1)

- **ORION: ORthonormal Text Encoding for Universal VLM AdaptatION** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19530v1](http://arxiv.org/abs/2602.19530v1)

- **CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19605v1](http://arxiv.org/abs/2602.19605v1)

- **Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19615v1](http://arxiv.org/abs/2602.19615v1)

- **Accurate Planar Tracking With Robust Re-Detection** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19624v1](http://arxiv.org/abs/2602.19624v1)

- **HDR Reconstruction Boosting with Training-Free and Exposure-Consistent Diffusion** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19706v1](http://arxiv.org/abs/2602.19706v1)

- **A Risk-Aware UAV-Edge Service Framework for Wildfire Monitoring and Emergency Response** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19742v1](http://arxiv.org/abs/2602.19742v1)

- **Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19756v1](http://arxiv.org/abs/2602.19756v1)

- **Training Deep Stereo Matching Networks on Tree Branch Imagery: A Benchmark Study for Real-Time UAV Forestry Applications** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19763v1](http://arxiv.org/abs/2602.19763v1)

- **TraceVision: Trajectory-Aware Vision-Language Model for Human-Like Spatial Understanding** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19768v1](http://arxiv.org/abs/2602.19768v1)

- **Using Unsupervised Domain Adaptation Semantic Segmentation for Pulmonary Embolism Detection in Computed Tomography Pulmonary Angiogram (CTPA) Images** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19891v1](http://arxiv.org/abs/2602.19891v1)

- **A Context-Aware Knowledge Graph Platform for Stream Processing in Industrial IoT** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.19990v1](http://arxiv.org/abs/2602.19990v1)

- **Training-Free Generative Modeling via Kernelized Stochastic Interpolants** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.20070v1](http://arxiv.org/abs/2602.20070v1)

- **CQ-CiM: Hardware-Aware Embedding Shaping for Robust CiM-Based Retrieval** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.20083v1](http://arxiv.org/abs/2602.20083v1)

- **StructXLIP: Enhancing Vision-language Models with Multimodal Structural Cues** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.20089v1](http://arxiv.org/abs/2602.20089v1)

- **Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device** (score: None, graduated: 2026-02-25)
  [http://arxiv.org/abs/2602.20161v1](http://arxiv.org/abs/2602.20161v1)

- **Strategy-Supervised Autonomous Laparoscopic Camera Control via Event-Driven Graph Mining** (score: None, graduated: 2026-02-26)
  [http://arxiv.org/abs/2602.20500v1](http://arxiv.org/abs/2602.20500v1)

- **From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection** (score: None, graduated: 2026-02-26)
  [http://arxiv.org/abs/2602.20630v1](http://arxiv.org/abs/2602.20630v1)

- **PyVision-RL: Forging Open Agentic Vision Models via RL** (score: None, graduated: 2026-02-26)
  [http://arxiv.org/abs/2602.20739v1](http://arxiv.org/abs/2602.20739v1)

- **Real-time Motion Segmentation with Event-based Normal Flow** (score: None, graduated: 2026-02-26)
  [http://arxiv.org/abs/2602.20790v1](http://arxiv.org/abs/2602.20790v1)

- **SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking** (score: None, graduated: 2026-02-26)
  [http://arxiv.org/abs/2602.20792v1](http://arxiv.org/abs/2602.20792v1)

- **EKF-Based Depth Camera and Deep Learning Fusion for UAV-Person Distance Estimation and Following in SAR Operations** (score: None, graduated: 2026-02-26)
  [http://arxiv.org/abs/2602.20958v1](http://arxiv.org/abs/2602.20958v1)

- **Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones** (score: None, graduated: 2026-02-26)
  [http://arxiv.org/abs/2602.21101v1](http://arxiv.org/abs/2602.21101v1)

- **Human Video Generation from a Single Image with 3D Pose and View Control** (score: None, graduated: 2026-02-26)
  [http://arxiv.org/abs/2602.21188v1](http://arxiv.org/abs/2602.21188v1)







### Notable Authors

| Author | Papers | Avg Score |
|--------|--------|-----------|

| Yen-Yu Lin | 3 | 84.8 |

| Hao Li | 3 | 82.8 |

| Hao Liu | 3 | 80.8 |

| Feng Gao | 3 | 74.3 |

| Lei Zhang | 3 | 73.6 |

| Qing Li | 3 | 69.5 |

| Liming Liu | 2 | 93.2 |

| Jiangkai Wu | 2 | 93.2 |

| Xinggong Zhang | 2 | 93.2 |

| Zhangcheng Wang | 2 | 87.6 |







## 3. Trends



### cappic-ai

| Date | Avg Score |
|------|-----------|

| 2026-02-09 | 73.12 |

| 2026-02-10 | 75.09 |

| 2026-02-11 | 72.47 |

| 2026-02-11 | 67.83 |

| 2026-02-16 | 78.09 |

| 2026-02-17 | 72.39 |

| 2026-02-18 | 74.52 |

| 2026-02-18 | 75.53 |

| 2026-02-19 | 62.97 |

| 2026-02-23 | 75.83 |

| 2026-02-24 | 85.2 |

| 2026-02-25 | 68.0 |

| 2026-02-26 | 76.58 |



