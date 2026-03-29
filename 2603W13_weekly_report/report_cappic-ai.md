# Weekly Paper Report: 20260329

> ISO 2026-W13




## 1. Weekly Briefing


이번 주 CS.CV 분야 평가 논문이 전주 대비 49건 감소(76건)하며 주목할 만한 하락세를 보였고, 평균 점수도 4.23포인트 하락(72.99)하며 품질 저하 우려가 제기되었습니다.  
4주 연속 논문 평가 건수 감소 추세([117, 181, 200, 161])와 함께 전반적인 성과 지표의 지속적인 약세가 확인되어 원인 분석 및 대응이 시급한 상황입니다.



| Metric | Value | WoW |
|--------|-------|-----|
| Evaluated | 161 | - |
| Tier 1 | 113 | - |
| cs.CV | 76 | -49 |
| Keyword Hits | 0 | 0 |
| Avg Score | 72.99 | -4.23 |



### 4-Week Trend (downtrend)

| Week | Papers | cs.CV | Avg Score |
|------|--------|-------|-----------|

| W1 | 117 | 70 | 76.42 |

| W2 | 181 | 109 | 75.78 |

| W3 | 200 | 125 | 77.22 |

| W4 | 161 | 76 | 72.99 |





### Top Categories

| Category | Count | WoW % |
|----------|-------|-------|

| cs.CV | 76 | -39.2 |

| cs.LG | 41 | 0.0 |

| cs.AI | 27 | -42.6 |

| cs.RO | 15 | 7.1 |

| eess.SP | 7 | 250.0 |





- Graduated Reminds: 450
- Active Reminds: 8




### Tech Radar








#### TF-IDF Keywords

`data` (STABLE) `video` (STABLE) `semantic` (RISING) `detection` (STABLE) `visual` (STABLE) `time` (STABLE) `systems` (NEW) `spatial` (STABLE) `learning` (STABLE) `pose` (NEW) `training` (STABLE) `audio` (NEW) `real` (STABLE) `image` (STABLE) `multi` (STABLE) `multimodal` (RISING) `estimation` (NEW) `language` (STABLE) `text` (STABLE) `aware` (RISING) `quality` (DISAPPEARED) `alignment` (DISAPPEARED) `system` (DISAPPEARED) `motion` (DISAPPEARED) `graph` (DISAPPEARED) `human` (DISAPPEARED) `images` (DISAPPEARED) `segmentation` (DISAPPEARED) `edge` (DISAPPEARED) `knowledge` (DISAPPEARED) `dataset` (DISAPPEARED) `reconstruction` (DISAPPEARED) `scale` (DISAPPEARED) `object` (DISAPPEARED) `classification` (DISAPPEARED) `control` (DISAPPEARED) `temporal` (DISAPPEARED) `accuracy` (DISAPPEARED) `memory` (DISAPPEARED) `social` (DISAPPEARED) `reasoning` (DISAPPEARED) `vision` (DISAPPEARED) `camera` (DISAPPEARED) `generation` (DISAPPEARED) `diffusion` (DISAPPEARED) `tasks` (DISAPPEARED) `task` (DISAPPEARED) `long` (DISAPPEARED) 







## 2. Top Papers


### 1. TRINE: A Token-Aware, Runtime-Adaptive FPGA Inference Engine for Multimodal AI

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.22867v1](http://arxiv.org/abs/2603.22867v1)

- Multimodal stacks that mix ViTs, CNNs, GNNs, and transformer NLP strain embedded platforms because their compute/memory patterns diverge and hard real-time targets leave little slack. TRINE is a single-bitstream FPGA accelerator and compiler that executes end-to-end multimodal inference without reconfiguration. Layers are unified as DDMM/SDDMM/SpMM and mapped to a mode-switchable engine that toggles at runtime among weight/output-stationary systolic, 1xCS SIMD, and a routable adder tree (RADT) on a shared PE array. A width-matched, two-stage top-k unit enables in-stream token pruning, while dependency-aware layer offloading (DALO) overlaps independent kernels across reconfigurable processing units to sustain utilization. Evaluated on Alveo U50 and ZCU104, TRINE reduces latency by up to 22.57x vs. RTX 4090 and 6.86x vs. Jetson Orin Nano at 20-21 W; token pruning alone yields up to 7.8x on ViT-heavy pipelines, and DALO contributes up to 79% throughput improvement. With int8 quantization, accuracy drops remain <2.5% across representative tasks, delivering state-of-the-art latency and energy efficiency for unified vision, language, and graph workloads-in one bitstream.
- TRINE은 다중 모달 AI 가속을 위한 FPGA 엔진으로, rk3588 기반의 에지 디바이스에서 스포츠 촬영 및 분석을 위한 여러 AI 모델을 효율적으로 실행할 수 있게 해줍니다.


### 2. Convolutions Predictable Offloading to an Accelerator: Formalization and Optimization

- **Score**: 98.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.21792v1](http://arxiv.org/abs/2603.21792v1)

- Convolutional neural networks (CNNs) require a large number of multiply-accumulate (MAC) operations. To meet real-time constraints, they often need to be executed on specialized accelerators composed of an on-chip memory and a processing unit. However, the on-chip memory is often insufficient to store all the data required to compute a CNN layer. Thus, the computation must be performed in several offloading steps. We formalise such sequences of steps and apply our formalism to a state of the art decomposition of convolutions. In order to find optimal strategies in terms of duration, we encode the problem with a set of constraints. A Python-based simulator allows to analyse in-depth computed strategies.
- rk3588 엣지 디바이스에서 AI 모델 실행 최적화에 직접적으로 관련된 연구로 실시간 스포츠 촬영 및 분석 성능 향상에 필수적임


### 3. Rateless DeepJSCC for Broadcast Channels: a Rate-Distortion-Complexity Tradeoff

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.21616v1](http://arxiv.org/abs/2603.21616v1)

- In recent years, numerous data-intensive broadcasting applications have emerged at the wireless edge, calling for a flexible tradeoff between distortion, transmission rate, and processing complexity. While deep learning-based joint source-channel coding (DeepJSCC) has been identified as a potential solution to data-intensive communications, most of these schemes are confined to worst-case solutions, lack adaptive complexity, and are inefficient in broadcast settings. To overcome these limitations, this paper introduces nonlinear transform rateless source-channel coding (NTRSCC), a variable-length JSCC framework for broadcast channels based on rateless codes. In particular, we integrate learned source transformations with physical-layer LT codes, develop unequal protection schemes that exploit decoder side information, and devise approximations to enable end-to-end optimization of rateless parameters. Our framework enables heterogeneous receivers to adaptively adjust their received number of rateless symbols and decoding iterations in belief propagation, thereby achieving a controllable tradeoff between distortion, rate, and decoding complexity. Simulation results demonstrate that the proposed method enhances image broadcast quality under stringent communication and processing budgets over heterogeneous edge devices.
- Rateless DeepJSCC framework for broadcast channels could enable efficient streaming of sports content to heterogeneous edge devices.


### 4. No Dense Tensors Needed: Fully Sparse Object Detection on Event-Camera Voxel Grids

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.21638v1](http://arxiv.org/abs/2603.21638v1)

- Event cameras produce asynchronous, high-dynamic-range streams well suited for detecting small, fast-moving drones, yet most event-based detectors convert the sparse event stream into dense tensors, discarding the representational efficiency of neuromorphic sensing. We propose SparseVoxelDet, to our knowledge the first fully sparse object detector for event cameras, in which backbone feature extraction, feature pyramid fusion, and the detection head all operate exclusively on occupied voxel positions through 3D sparse convolutions; no dense feature tensor is instantiated at any stage of the pipeline. On the FRED benchmark (629,832 annotated frames), SparseVoxelDet achieves 83.38% mAP at 50 while processing only 14,900 active voxels per frame (0.23% of the T.H.W grid), compared to 409,600 pixels for the dense YOLOv11 baseline (87.68% mAP at 50). Relaxing the IoU threshold from 0.50 to 0.40 recovers mAP to 89.26%, indicating that the remaining accuracy gap is dominated by box regression precision rather than detection capability. The sparse representation yields 858 times GPU memory compression and 3,670 times storage reduction relative to the equivalent dense 3D voxel tensor, with data-structure size that scales with scene dynamics rather than sensor resolution. Error forensics across 119,459 test frames confirms that 71 percent of failures are localization near-misses rather than missed targets. These results demonstrate that native sparse processing is a viable paradigm for event-camera object detection, exploiting the structural sparsity of neuromorphic sensor data without requiring neuromorphic computing hardware, and providing a framework whose representation cost is governed by scene activity rather than pixel count, a property that becomes increasingly valuable as event cameras scale to higher resolutions.
- 이벤트 카메라를 이용한 효율적인 희소 객체 탐지 기술로 엣지 디바이스에 직접 적용 가능


### 5. LEMMA: Laplacian pyramids for Efficient Marine SeMAntic Segmentation

- **Score**: 94.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.25689v1](http://arxiv.org/abs/2603.25689v1)

- Semantic segmentation in marine environments is crucial for the autonomous navigation of unmanned surface vessels (USVs) and coastal Earth Observation events such as oil spills. However, existing methods, often relying on deep CNNs and transformer-based architectures, face challenges in deployment due to their high computational costs and resource-intensive nature. These limitations hinder the practicality of real-time, low-cost applications in real-world marine settings.   To address this, we propose LEMMA, a lightweight semantic segmentation model designed specifically for accurate remote sensing segmentation under resource constraints. The proposed architecture leverages Laplacian Pyramids to enhance edge recognition, a critical component for effective feature extraction in complex marine environments for disaster response, environmental surveillance, and coastal monitoring. By integrating edge information early in the feature extraction process, LEMMA eliminates the need for computationally expensive feature map computations in deeper network layers, drastically reducing model size, complexity and inference time. LEMMA demonstrates state-of-the-art performance across datasets captured from diverse platforms while reducing trainable parameters and computational requirements by up to 71x, GFLOPs by up to 88.5\%, and inference time by up to 84.65\%, as compared to existing models. Experimental results highlight its effectiveness and real-world applicability, including 93.42\% IoU on the Oil Spill dataset and 98.97\% mIoU on Mastr1325.
- 가벼운 세그멘테이션 모델로 엣지 디바이스에서 실시간 처리에 필수적인 계산 효율성 제공


### 6. Bridging Biological Hearing and Neuromorphic Computing: End-to-End Time-Domain Audio Signal Processing with Reservoir Computing

- **Score**: 93.6 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.24283v1](http://arxiv.org/abs/2603.24283v1)

- Despite the advancements in cutting-edge technologies, audio signal processing continues to pose challenges and lacks the precision of a human speech processing system. To address these challenges, we propose a novel approach to simplify audio signal processing by leveraging time-domain techniques and reservoir computing. Through our research, we have developed a real-time audio signal processing system by simplifying audio signal processing through the utilization of reservoir computers, which are significantly easier to train.   Feature extraction is a fundamental step in speech signal processing, with Mel Frequency Cepstral Coefficients (MFCCs) being a dominant choice due to their perceptual relevance to human hearing. However, conventional MFCC extraction relies on computationally intensive time-frequency transformations, limiting efficiency in real-time applications. To address this, we propose a novel approach that leverages reservoir computing to streamline MFCC extraction. By replacing traditional frequency-domain conversions with convolution operations, we eliminate the need for complex transformations while maintaining feature discriminability. We present an end-to-end audio processing framework that integrates this method, demonstrating its potential for efficient and real-time speech analysis. Our results contribute to the advancement of energy-efficient audio processing technologies, enabling seamless deployment in embedded systems and voice-driven applications. This work bridges the gap between biologically inspired feature extraction and modern neuromorphic computing, offering a scalable solution for next-generation speech recognition systems.
- 실시간 오디오 처리 기술이 엣지 디바이스에 적용 가능


### 7. ANCHOR: Adaptive Network based on Cascaded Harmonic Offset Routing

- **Score**: 93.6 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.21718v1](http://arxiv.org/abs/2603.21718v1)

- Time series analysis plays a foundational role in a wide range of real-world applications, yet accurately modeling complex non-stationary signals remains a shared challenge across downstream tasks. Existing methods attempt to extract features directly from one-dimensional sequences, making it difficult to handle the widely observed dynamic phase drift and discrete quantization error. To address this issue, we decouple temporal evolution into macroscopic physical periods and microscopic phase perturbations, and inject frequency-domain priors derived from the Real Fast Fourier Transform (RFFT) into the underlying spatial sampling process. Based on this idea, we propose a Frequency-Guided Deformable Module (FGDM) to adaptively compensate for microscopic phase deviations. Built upon FGDM, we further develop an Adaptive Network based on Cascaded Harmonic Offset Routing (ANCHOR) as a general-purpose backbone for time-series modeling. Through orthogonal channel partitioning and a progressive residual architecture, ANCHOR efficiently decouples multi-scale harmonic features while substantially suppressing the computational redundancy of multi-branch networks. Extensive experiments demonstrate that ANCHOR achieves the best performance in most short-term forecasting sub-tasks and exhibits strong competitiveness on several specific sub-tasks in anomaly detection and time-series classification, validating its effectiveness as a universal time-series foundation backbone.
- 시계열 분석 기술로 스포츠 동작 분석에 적용 가능


### 8. TorR: Towards Brain-Inspired Task-Oriented Reasoning via Cache-Oriented Algorithm-Architecture Co-design

- **Score**: 93.6 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.22855v1](http://arxiv.org/abs/2603.22855v1)

- Task-oriented object detection (TOOD) atop CLIP offers open-vocabulary, prompt-driven semantics, yet dense per-window computation and heavy memory traffic hinder real-time, power-limited edge deployment. We present \emph{TorR}, a brain-inspired \textbf{algorithm--architecture co-design} that \textbf{replaces CLIP-style dense alignment with a hyperdimensional (HDC) associative reasoner} and turns temporal coherence into reuse. On the \emph{algorithm} side, TorR reformulates alignment as HDC similarity and graph composition, introducing \emph{partial-similarity reuse} via (i) query caching with per-class score accumulation, (ii) exact $δ$-updates when only a small set of hypervector bits change, and (iii) similarity/load-gated bypass under high system load. On the \emph{architecture} side, TorR instantiates a lane-scalable, bit-sliced item memory with bank/precision gating and a lightweight controller that schedules bypass/$δ$/full paths to meet RT-30/RT-60 targets as object counts vary. Synthesized in a TSMC 28\,nm process and exercised with a cycle-accurate simulator, TorR sustains real-time throughput with millijoule-scale energy per window ($\approx$50\,mJ at 60\,FPS; $\approx$113\,mJ at 30\,FPS) and low latency jitter, while delivering competitive AP@0.5 across five task prompts (mean 44.27\%) within a bounded margin to strong VLM baselines, but at orders-of-magnitude lower energy. The design exposes deployment-time configurability (effective dimension $D'$, thresholds, precision) to trade accuracy, latency, and energy for edge budgets.
- TorR는 에지 디바이스에서 실시간 객체 탐지를 위한 뇀 영감 기술로, 스포츠 장면에서 선수 및 공 등의 실시간 추적과 분석에 적합합니다.


### 9. Not All Layers Are Created Equal: Adaptive LoRA Ranks for Personalized Image Generation

- **Score**: 92.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.21884v1](http://arxiv.org/abs/2603.21884v1)

- Low Rank Adaptation (LoRA) is the de facto fine-tuning strategy to generate personalized images from pre-trained diffusion models. Choosing a good rank is extremely critical, since it trades off performance and memory consumption, but today the decision is often left to the community's consensus, regardless of the personalized subject's complexity. The reason is evident: the cost of selecting a good rank for each LoRA component is combinatorial, so we opt for practical shortcuts such as fixing the same rank for all components. In this paper, we take a first step to overcome this challenge. Inspired by variational methods that learn an adaptive width of neural networks, we let the ranks of each layer freely adapt during fine-tuning on a subject. We achieve it by imposing an ordering of importance on the rank's positions, effectively encouraging the creation of higher ranks when strictly needed. Qualitatively and quantitatively, our approach, LoRA$^2$, achieves a competitive trade-off between DINO, CLIP-I, and CLIP-T across 29 subjects while requiring much less memory and lower rank than high rank LoRA versions. Code: https://github.com/donaldssh/NotAllLayersAreCreatedEqual.
- 개인화된 이미지 생성을 위한 적응형 LoRA 기술은 스포츠 영상을 맞춤형 하이라이트 시각물로 변환하는 데 사용될 수 있습니다.


### 10. Short-Form Video Viewing Behavior Analysis and Multi-Step Viewing Time Prediction

- **Score**: 92.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.22663v1](http://arxiv.org/abs/2603.22663v1)

- Short-form videos have become one of the most popular user-generated content formats nowadays. Popular short-video platforms use a simple streaming approach that preloads one or more videos in the recommendation list in advance. However, this approach results in significant data wastage, as a large portion of the downloaded video data is not used due to the user's early skip behavior. To address this problem, the chunk-based preloading approach has been proposed, where videos are divided into chunks, and preloading is performed in a chunk-based manner to reduce data wastage. To optimize chunk-based preloading, it is important to understand the user's viewing behavior in short-form video streaming. In this paper, we conduct a measurement study to construct a user behavior dataset that contains users' viewing times of one hundred short videos of various categories. Using the dataset, we evaluate the performance of standard time-series forecasting algorithms for predicting user viewing time in short-form video streaming. Our evaluation results show that Auto-ARIMA generally achieves the lowest and most stable forecasting errors across most experimental settings. The remaining methods, including AR, LR, SVR, and DTR, tend to produce higher errors and exhibit lower stability in many cases. The dataset is made publicly available at https://nvduc.github.io/shortvideodataset.
- 숏폼 영상 시청 행동 분석 연구는 스포츠 하이라이트 플랫폼의 콘텐츠 추천 알고리즘을 최적화하는 데 직접적으로 활용될 수 있습니다.






### Graduated Reminds


- **On the Cone Effect and Modality Gap in Medical Vision-Language Embeddings** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17246v1](http://arxiv.org/abs/2603.17246v1)

- **FineViT: Progressively Unlocking Fine-Grained Perception with Dense Recaptions** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17326v1](http://arxiv.org/abs/2603.17326v1)

- **Motion-Adaptive Temporal Attention for Lightweight Video Generation with Stable Diffusion** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17398v1](http://arxiv.org/abs/2603.17398v1)

- **Joint Degradation-Aware Arbitrary-Scale Super-Resolution for Variable-Rate Extreme Image Compression** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17408v1](http://arxiv.org/abs/2603.17408v1)

- **From Digital Twins to World Models:Opportunities, Challenges, and Applications for Mobile Edge General Intelligence** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17420v1](http://arxiv.org/abs/2603.17420v1)

- **VLM2Rec: Resolving Modality Collapse in Vision-Language Model Embedders for Multimodal Sequential Recommendation** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17450v1](http://arxiv.org/abs/2603.17450v1)

- **AR-CoPO: Align Autoregressive Video Generation with Contrastive Policy Optimization** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17461v1](http://arxiv.org/abs/2603.17461v1)

- **VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17470v1](http://arxiv.org/abs/2603.17470v1)

- **A spatio-temporal graph-based model for team sports analysis** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17471v1](http://arxiv.org/abs/2603.17471v1)

- **Revisiting Cross-Attention Mechanisms: Leveraging Beneficial Noise for Domain-Adaptive Learning** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17474v1](http://arxiv.org/abs/2603.17474v1)

- **AdapTS: Lightweight Teacher-Student Approach for Multi-Class and Continual Visual Anomaly Detection** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17530v1](http://arxiv.org/abs/2603.17530v1)

- **Prompt-Free Universal Region Proposal Network** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17554v1](http://arxiv.org/abs/2603.17554v1)

- **ReLaGS: Relational Language Gaussian Splatting** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17605v1](http://arxiv.org/abs/2603.17605v1)

- **Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17655v1](http://arxiv.org/abs/2603.17655v1)

- **TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17735v1](http://arxiv.org/abs/2603.17735v1)

- **CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17779v1](http://arxiv.org/abs/2603.17779v1)

- **ChopGrad: Pixel-Wise Losses for Latent Video Diffusion via Truncated Backpropagation** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17812v1](http://arxiv.org/abs/2603.17812v1)

- **Steering Video Diffusion Transformers with Massive Activations** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17825v1](http://arxiv.org/abs/2603.17825v1)

- **TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17828v1](http://arxiv.org/abs/2603.17828v1)

- **Enabling Real-Time Programmability for RAN Functions: A Wasm-Based Approach for Robust and High-Performance dApps** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17880v1](http://arxiv.org/abs/2603.17880v1)

- **A Creative Agent is Worth a 64-Token Template** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17895v1](http://arxiv.org/abs/2603.17895v1)

- **SpiderCam: Low-Power Snapshot Depth from Differential Defocus** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17910v1](http://arxiv.org/abs/2603.17910v1)

- **SegFly: A 2D-3D-2D Paradigm for Aerial RGB-Thermal Semantic Segmentation at Scale** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17920v1](http://arxiv.org/abs/2603.17920v1)

- **Feeling the Space: Egomotion-Aware Video Representation for Efficient and Accurate 3D Scene Understanding** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17980v1](http://arxiv.org/abs/2603.17980v1)

- **The Unreasonable Effectiveness of Text Embedding Interpolation for Continuous Image Steering** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.17998v1](http://arxiv.org/abs/2603.17998v1)

- **EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.18001v1](http://arxiv.org/abs/2603.18001v1)

- **Universal Skeleton Understanding via Differentiable Rendering and MLLMs** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.18003v1](http://arxiv.org/abs/2603.18003v1)

- **Unified Spatio-Temporal Token Scoring for Efficient Video VLMs** (score: None, graduated: 2026-03-23)
  [http://arxiv.org/abs/2603.18004v1](http://arxiv.org/abs/2603.18004v1)

- **R&D: Balancing Reliability and Diversity in Synthetic Data Augmentation for Semantic Segmentation** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18427v1](http://arxiv.org/abs/2603.18427v1)

- **FILT3R: Latent State Adaptive Kalman Filter for Streaming 3D Reconstruction** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18493v1](http://arxiv.org/abs/2603.18493v1)

- **Counting Circuits: Mechanistic Interpretability of Visual Reasoning in Large Vision-Language Models** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18523v1](http://arxiv.org/abs/2603.18523v1)

- **Scaling Sim-to-Real Reinforcement Learning for Robot VLAs with Generative 3D Worlds** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18532v1](http://arxiv.org/abs/2603.18532v1)

- **Modeling the Impacts of Swipe Delay on User Quality of Experience in Short Video Streaming** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18575v1](http://arxiv.org/abs/2603.18575v1)

- **Improving Joint Audio-Video Generation with Cross-Modal Context Learning** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18600v1](http://arxiv.org/abs/2603.18600v1)

- **Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18671v1](http://arxiv.org/abs/2603.18671v1)

- **EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18739v1](http://arxiv.org/abs/2603.18739v1)

- **SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18774v1](http://arxiv.org/abs/2603.18774v1)

- **HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18850v1](http://arxiv.org/abs/2603.18850v1)

- **Through the Looking-Glass: AI-Mediated Video Communication Reduces Interpersonal Trust and Confidence in Judgments** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18868v1](http://arxiv.org/abs/2603.18868v1)

- **Translating MRI to PET through Conditional Diffusion Models with Enhanced Pathology Awareness** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18896v1](http://arxiv.org/abs/2603.18896v1)

- **Balancing Performance and Fairness in Explainable AI for Anomaly Detection in Distributed Power Plants Monitoring** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18954v1](http://arxiv.org/abs/2603.18954v1)

- **CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.18991v1](http://arxiv.org/abs/2603.18991v1)

- **Generalized Hand-Object Pose Estimation with Occlusion Awareness** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.19013v1](http://arxiv.org/abs/2603.19013v1)

- **Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.19026v1](http://arxiv.org/abs/2603.19026v1)

- **A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.19133v1](http://arxiv.org/abs/2603.19133v1)

- **ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.19157v1](http://arxiv.org/abs/2603.19157v1)

- **DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.19172v1](http://arxiv.org/abs/2603.19172v1)

- **GenMFSR: Generative Multi-Frame Image Restoration and Super-Resolution** (score: None, graduated: 2026-03-24)
  [http://arxiv.org/abs/2603.19187v1](http://arxiv.org/abs/2603.19187v1)

- **ALADIN:Attribute-Language Distillation Network for Person Re-Identification** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21482v1](http://arxiv.org/abs/2603.21482v1)

- **A Framework for Closed-Loop Robotic Assembly, Alignment and Self-Recovery of Precision Optical Systems** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21496v1](http://arxiv.org/abs/2603.21496v1)

- **Feature Incremental Clustering with Generalization Bounds** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21590v1](http://arxiv.org/abs/2603.21590v1)

- **In-network Attack Detection with Federated Deep Learning in IoT Networks: Real Implementation and Analysis** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21596v1](http://arxiv.org/abs/2603.21596v1)

- **Benchmarking Message Brokers for IoT Edge Computing: A Comprehensive Performance Study** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21600v1](http://arxiv.org/abs/2603.21600v1)

- **AdaEdit: Adaptive Temporal and Channel Modulation for Flow-Based Image Editing** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21615v1](http://arxiv.org/abs/2603.21615v1)

- **Rateless DeepJSCC for Broadcast Channels: a Rate-Distortion-Complexity Tradeoff** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21616v1](http://arxiv.org/abs/2603.21616v1)

- **No Dense Tensors Needed: Fully Sparse Object Detection on Event-Camera Voxel Grids** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21638v1](http://arxiv.org/abs/2603.21638v1)

- **Optimal Memory Encoding Through Fluctuation-Response Structure** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21666v1](http://arxiv.org/abs/2603.21666v1)

- **ANCHOR: Adaptive Network based on Cascaded Harmonic Offset Routing** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21718v1](http://arxiv.org/abs/2603.21718v1)

- **Mapping Travel Experience in Public Transport: Real-Time Evidence and Spatial Analysis in Hamburg** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21763v1](http://arxiv.org/abs/2603.21763v1)

- **SHARP: Spectrum-aware Highly-dynamic Adaptation for Resolution Promotion in Remote Sensing Synthesis** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21783v1](http://arxiv.org/abs/2603.21783v1)

- **The Universal Normal Embedding** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21786v1](http://arxiv.org/abs/2603.21786v1)

- **Convolutions Predictable Offloading to an Accelerator: Formalization and Optimization** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21792v1](http://arxiv.org/abs/2603.21792v1)

- **Not All Layers Are Created Equal: Adaptive LoRA Ranks for Personalized Image Generation** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21884v1](http://arxiv.org/abs/2603.21884v1)

- **λ-GELU: Learning Gating Hardness for Controlled ReLU-ization in Deep Networks** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.21991v1](http://arxiv.org/abs/2603.21991v1)

- **StreamingClaw Technical Report** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.22120v1](http://arxiv.org/abs/2603.22120v1)

- **One Model, Two Markets: Bid-Aware Generative Recommendation** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.22231v1](http://arxiv.org/abs/2603.22231v1)

- **DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution** (score: None, graduated: 2026-03-25)
  [http://arxiv.org/abs/2603.22271v1](http://arxiv.org/abs/2603.22271v1)

- **Short-Form Video Viewing Behavior Analysis and Multi-Step Viewing Time Prediction** (score: None, graduated: 2026-03-26)
  [http://arxiv.org/abs/2603.22663v1](http://arxiv.org/abs/2603.22663v1)

- **Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis** (score: None, graduated: 2026-03-26)
  [http://arxiv.org/abs/2603.22786v1](http://arxiv.org/abs/2603.22786v1)

- **TorR: Towards Brain-Inspired Task-Oriented Reasoning via Cache-Oriented Algorithm-Architecture Co-design** (score: None, graduated: 2026-03-26)
  [http://arxiv.org/abs/2603.22855v1](http://arxiv.org/abs/2603.22855v1)

- **TRINE: A Token-Aware, Runtime-Adaptive FPGA Inference Engine for Multimodal AI** (score: None, graduated: 2026-03-26)
  [http://arxiv.org/abs/2603.22867v1](http://arxiv.org/abs/2603.22867v1)

- **Dual-Teacher Distillation with Subnetwork Rectification for Black-Box Domain Adaptation** (score: None, graduated: 2026-03-26)
  [http://arxiv.org/abs/2603.22908v1](http://arxiv.org/abs/2603.22908v1)

- **Toward Integrated Sensing, Communications, and Edge Intelligence Networks** (score: None, graduated: 2026-03-26)
  [http://arxiv.org/abs/2603.22958v1](http://arxiv.org/abs/2603.22958v1)

- **VQ-Jarvis: Retrieval-Augmented Video Restoration Agent with Sharp Vision and Fast Thought** (score: None, graduated: 2026-03-26)
  [http://arxiv.org/abs/2603.22998v1](http://arxiv.org/abs/2603.22998v1)

- **GTLR-GS: Geometry-Texture Aware LiDAR-Regularized 3D Gaussian Splatting for Realistic Scene Reconstruction** (score: None, graduated: 2026-03-26)
  [http://arxiv.org/abs/2603.23192v1](http://arxiv.org/abs/2603.23192v1)







### Notable Authors

| Author | Papers | Avg Score |
|--------|--------|-----------|

| Hyunwoo Oh | 4 | 93.4 |

| Nicu Sebe | 4 | 74.8 |

| Cewu Lu | 3 | 86.8 |

| Jae-Sang Hyun | 3 | 84.7 |

| Yi Wang | 3 | 83.3 |

| Yang Li | 3 | 82.9 |

| Wenbin Li | 3 | 82.0 |

| Hao Li | 3 | 80.8 |

| Peng Wang | 3 | 79.6 |

| Lin Liu | 3 | 78.9 |







## 3. Trends



### cappic-ai

| Date | Avg Score |
|------|-----------|

| 2026-03-02 | 76.66 |

| 2026-03-03 | 79.6 |

| 2026-03-04 | 75.8 |

| 2026-03-05 | 77.33 |

| 2026-03-10 | 75.92 |

| 2026-03-11 | 75.21 |

| 2026-03-12 | 76.22 |

| 2026-03-16 | 79.03 |

| 2026-03-17 | 77.66 |

| 2026-03-18 | 76.84 |

| 2026-03-19 | 75.3 |

| 2026-03-23 | 75.73 |

| 2026-03-24 | 69.33 |

| 2026-03-25 | 75.55 |

| 2026-03-26 | 73.49 |



