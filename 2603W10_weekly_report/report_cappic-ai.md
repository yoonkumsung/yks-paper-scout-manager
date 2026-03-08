# Weekly Paper Report: 20260308

> ISO 2026-W10




## 1. Weekly Briefing


금주 평가 논문 총 117편으로 전주(107편) 대비 소폭 증가하며 4주간 상승세가 안정화되고 있습니다. CS.CV 분야 논문(70편, 전주 대비 -1)과 평균 점수(76.42, +0.02)는 미미한 변동에 그쳤으나, 키워드 히트 건수(0건)는 2주 연속 변동 없음이 지속되었습니다. 특히 티어 1 논문 비중(67.5%)이 높게 유지된 점이 주목됩니다.



| Metric | Value | WoW |
|--------|-------|-----|
| Evaluated | 117 | - |
| Tier 1 | 79 | - |
| cs.CV | 70 | -1 |
| Keyword Hits | 0 | 0 |
| Avg Score | 76.42 | 0.02 |



### 4-Week Trend (uptrend)

| Week | Papers | cs.CV | Avg Score |
|------|--------|-------|-----------|

| W1 | 323 | 172 | 71.3 |

| W2 | 108 | 39 | 74.17 |

| W3 | 107 | 71 | 76.4 |

| W4 | 117 | 70 | 76.42 |





### Top Categories

| Category | Count | WoW % |
|----------|-------|-------|

| cs.CV | 70 | -1.4 |

| cs.RO | 24 | 100.0 |

| cs.AI | 22 | -4.3 |

| cs.LG | 20 | 17.6 |

| eess.IV | 7 | 0.0 |





- Graduated Reminds: 220
- Active Reminds: 22




### Tech Radar








#### TF-IDF Keywords

`video` (STABLE) `data` (STABLE) `image` (STABLE) `reasoning` (NEW) `time` (STABLE) `long` (NEW) `visual` (RISING) `memory` (NEW) `semantic` (NEW) `real` (STABLE) `detection` (STABLE) `reconstruction` (NEW) `multi` (STABLE) `generation` (STABLE) `segmentation` (NEW) `images` (NEW) `training` (STABLE) `task` (NEW) `human` (STABLE) `dataset` (NEW) `scale` (DISAPPEARED) `language` (DISAPPEARED) `tasks` (DISAPPEARED) `learning` (DISAPPEARED) `control` (DISAPPEARED) `quality` (DISAPPEARED) `social` (DISAPPEARED) `knowledge` (DISAPPEARED) `temporal` (DISAPPEARED) `object` (DISAPPEARED) `graph` (DISAPPEARED) `classification` (DISAPPEARED) `vision` (DISAPPEARED) `accuracy` (DISAPPEARED) `system` (DISAPPEARED) `camera` (DISAPPEARED) `motion` (DISAPPEARED) `alignment` (DISAPPEARED) `multimodal` (DISAPPEARED) `aware` (DISAPPEARED) 







## 2. Top Papers


### 1. Helios: Real Real-Time Long Video Generation Model

- **Score**: 98.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.04379v1](http://arxiv.org/abs/2603.04379v1)

- We introduce Helios, the first 14B video generation model that runs at 19.5 FPS on a single NVIDIA H100 GPU and supports minute-scale generation while matching the quality of a strong baseline. We make breakthroughs along three key dimensions: (1) robustness to long-video drifting without commonly used anti-drifting heuristics such as self-forcing, error-banks, or keyframe sampling; (2) real-time generation without standard acceleration techniques such as KV-cache, sparse/linear attention, or quantization; and (3) training without parallelism or sharding frameworks, enabling image-diffusion-scale batch sizes while fitting up to four 14B models within 80 GB of GPU memory. Specifically, Helios is a 14B autoregressive diffusion model with a unified input representation that natively supports T2V, I2V, and V2V tasks. To mitigate drifting in long-video generation, we characterize typical failure modes and propose simple yet effective training strategies that explicitly simulate drifting during training, while eliminating repetitive motion at its source. For efficiency, we heavily compress the historical and noisy context and reduce the number of sampling steps, yielding computational costs comparable to -- or lower than -- those of 1.3B video generative models. Moreover, we introduce infrastructure-level optimizations that accelerate both inference and training while reducing memory consumption. Extensive experiments demonstrate that Helios consistently outperforms prior methods on both short- and long-video generation. We plan to release the code, base model, and distilled model to support further development by the community.
- 엣지 컴퓨팅 프레임워크가 AI 카메라 엣지 디바이스 배포에 직접적으로 관련


### 2. Search Multilayer Perceptron-Based Fusion for Efficient and Accurate Siamese Tracking

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.01706v1](http://arxiv.org/abs/2603.01706v1)

- Siamese visual trackers have recently advanced through increasingly sophisticated fusion mechanisms built on convolutional or Transformer architectures. However, both struggle to deliver pixel-level interactions efficiently on resource-constrained hardware, leading to a persistent accuracy-efficiency imbalance. Motivated by this limitation, we redesign the Siamese neck with a simple yet effective Multilayer Perception (MLP)-based fusion module that enables pixel-level interaction with minimal structural overhead. Nevertheless, naively stacking MLP blocks introduces a new challenge: computational cost can scale quadratically with channel width. To overcome this, we construct a hierarchical search space of carefully designed MLP modules and introduce a customized relaxation strategy that enables differentiable neural architecture search (DNAS) to decouple channel-width optimization from other architectural choices. This targeted decoupling automatically balances channel width and depth, yielding a low-complexity architecture. The resulting tracker achieves state-of-the-art accuracy-efficiency trade-offs. It ranks among the top performers on four general-purpose and three aerial tracking benchmarks, while maintaining real-time performance on both resource-constrained Graphics Processing Units (GPUs) and Neural Processing Units (NPUs).
- 자원 제한 하드웨어에서 실시간 추적 성능으로 선수 추적에 최적


### 3. OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.02134v1](http://arxiv.org/abs/2603.02134v1)

- Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent stable state, and then cohesively fuses the information from the former into the latter to achieve both fidelity and stability. Moreover, we jointly model visual appearance and language fields and incorporate an implicit Gaussian fusion module to enhance reconstruction quality. Experiments on mainstream datasets demonstrate that our method consistently outperforms prior work in novel view synthesis and semantic understanding, showcasing robust performance across input sequences of varying lengths with real-time inference speed.
- 실시간 3D 재구성 기술로 스포츠 장면 포착 및 분석에 적합


### 4. Agentic Peer-to-Peer Networks: From Content Distribution to Capability and Action Sharing

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.03753v1](http://arxiv.org/abs/2603.03753v1)

- The ongoing shift of AI models from centralized cloud APIs to local AI agents on edge devices is enabling \textit{Client-Side Autonomous Agents (CSAAs)} -- persistent personal agents that can plan, access local context, and invoke tools on behalf of users. As these agents begin to collaborate by delegating subtasks directly between clients, they naturally form \emph{Agentic Peer-to-Peer (P2P) Networks}. Unlike classic file-sharing overlays where the exchanged object is static, hash-indexed content (e.g., files in BitTorrent), agentic overlays exchange \emph{capabilities and actions} that are heterogeneous, state-dependent, and potentially unsafe if delegated to untrusted peers. This article outlines the networking foundations needed to make such collaboration practical. We propose a plane-based reference architecture that decouples connectivity/identity, semantic discovery, and execution. Besides, we introduce signed, soft-state capability descriptors to support intent- and constraint-aware discovery. To cope with adversarial settings, we further present a \textit{tiered verification} spectrum: Tier~1 relies on reputation signals, Tier~2 applies lightweight canary challenge-response with fallback selection, and Tier~3 requires evidence packages such as signed tool receipts/traces (and, when applicable, attestation). Using a discrete-event simulator that models registry-based discovery, Sybil-style index poisoning, and capability drift, we show that tiered verification substantially improves end-to-end workflow success while keeping discovery latency near-constant and control-plane overhead modest.
- 엣지 디바이스 및 AI 에이전트 기술이 프로젝트의 AI 촬영 장치 및 플랫폼에 직접 적용 가능


### 5. SURE: Semi-dense Uncertainty-REfined Feature Matching

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.04869v1](http://arxiv.org/abs/2603.04869v1)

- Establishing reliable image correspondences is essential for many robotic vision problems. However, existing methods often struggle in challenging scenarios with large viewpoint changes or textureless regions, where incorrect cor- respondences may still receive high similarity scores. This is mainly because conventional models rely solely on fea- ture similarity, lacking an explicit mechanism to estimate the reliability of predicted matches, leading to overconfident errors. To address this issue, we propose SURE, a Semi- dense Uncertainty-REfined matching framework that jointly predicts correspondences and their confidence by modeling both aleatoric and epistemic uncertainties. Our approach in- troduces a novel evidential head for trustworthy coordinate regression, along with a lightweight spatial fusion module that enhances local feature precision with minimal overhead. We evaluated our method on multiple standard benchmarks, where it consistently outperforms existing state-of-the-art semi-dense matching models in both accuracy and efficiency. our code will be available on https://github.com/LSC-ALAN/SURE.
- 시각적 표현 향상 기술은 스포츠 영상 보정 및 하이라이트 편집에 직접적으로 적용 가능하며, 자세 및 동작 분석에도 활용될 수 있습니다.


### 6. Real Eyes Realize Faster: Gaze Stability and Pupil Novelty for Efficient Egocentric Learning

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.04098v1](http://arxiv.org/abs/2603.04098v1)

- Always-on egocentric cameras are increasingly used as demonstrations for embodied robotics, imitation learning, and assistive AR, but the resulting video streams are dominated by redundant and low-quality frames. Under the storage and battery constraints of wearable devices, choosing which frames to keep is as important as how to learn from them. We observe that modern eye-tracking headsets provide a continuous, training-free side channel that decomposes into two complementary axes: gaze fixation captures visual stability (quality), while pupil response captures arousal-linked moments (novelty). We operationalize this insight as a Dual-Criterion Frame Curator that first gates frames by gaze quality and then ranks the survivors by pupil-derived novelty. On the Visual Experience Dataset (VEDB), curated frames at 10% budget match the classification performance of the full stream, and naive signal fusion consistently destroys both contributions. The benefit is task-dependent: pupil ranking improves activity recognition, while gaze-only selection already dominates for scene recognition, confirming that the two signals serve genuinely different roles. Our method requires no model inference and operates at capture time, offering a path toward efficient, always-on egocentric data curation.
- 안와 추적 데이터를 활용한 효율적인 프레임 선택 방식으로 스포츠 하이라이트 자동 추출에 직접 적용 가능


### 7. RIVER: A Real-Time Interaction Benchmark for Video LLMs

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.03985v1](http://arxiv.org/abs/2603.03985v1)

- The rapid advancement of multimodal large language models has demonstrated impressive capabilities, yet nearly all operate in an offline paradigm, hindering real-time interactivity. Addressing this gap, we introduce the Real-tIme Video intERaction Bench (RIVER Bench), designed for evaluating online video comprehension. RIVER Bench introduces a novel framework comprising Retrospective Memory, Live-Perception, and Proactive Anticipation tasks, closely mimicking interactive dialogues rather than responding to entire videos at once. We conducted detailed annotations using videos from diverse sources and varying lengths, and precisely defined the real-time interactive format. Evaluations across various model categories reveal that while offline models perform well in single question-answering tasks, they struggle with real-time processing. Addressing the limitations of existing models in online video interaction, especially their deficiencies in long-term memory and future perception, we proposed a general improvement method that enables models to interact with users more flexibly in real time. We believe this work will significantly advance the development of real-time interactive video understanding models and inspire future research in this emerging field. Datasets and code are publicly available at https://github.com/OpenGVLab/RIVER.
- 실시간 비디오 이해 및 상호작용 기술로 스포츠 경기 분석에 적용 가능


### 8. Lambdas at the Far Edge: a Tale of Flying Lambdas and Lambdas on Wheels

- **Score**: 94.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.04008v1](http://arxiv.org/abs/2603.04008v1)

- Aggregate Programming (AP) is a paradigm for programming the collective behaviour of sets of distributed devices, possibly situated at the network far edge, by relying on asynchronous proximity-based interactions. The eXchange Calculus (XC), a recently proposed foundational model for AP, is essentially a typed lambda calculus extended with an operator (the exchange operator) providing an implicit communication mechanism between neighbour devices. This paper provides a gentle introduction to XC and to its implementation as a C++ library, called FCPP. The FCPP library and toolchain has been mainly developed at the Department of Computer Science of the University of Turin, where Stefano Berardi spent most of his academic career conducting outstanding research about logical foundation of computer science and transmitting his passion for research to students and young researchers, often exploiting typed lambda calculi. An FCCP program is essentially a typed lambda term, and FCPP has been used to write code that has been deployed on devices at the far edge of the network, including rovers and (soon) Uncrewed Aerial Vehicles (UAVs); hence the title of the paper.
- 엣지 컴퓨팅 프레임워크가 AI 카메라 엣지 디바이스 배포에 직접적으로 관련


### 9. Yolo-Key-6D: Single Stage Monocular 6D Pose Estimation with Keypoint Enhancements

- **Score**: 93.6 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.03879v1](http://arxiv.org/abs/2603.03879v1)

- Estimating the 6D pose of objects from a single RGB image is a critical task for robotics and extended reality applications. However, state-of-the-art multi stage methods often suffer from high latency, making them unsuitable for real time use. In this paper, we present Yolo-Key-6D, a novel single stage, end-to-end framework for monocular 6D pose estimation designed for both speed and accuracy. Our approach enhances a YOLO based architecture by integrating an auxiliary head that regresses the 2D projections of an object's 3D bounding box corners. This keypoint detection task significantly improves the network's understanding of 3D geometry. For stable end-to-end training, we directly regress rotation using a continuous 9D representation projected to SO(3) via singular value decomposition. On the LINEMOD and LINEMOD-Occluded benchmarks, YOLO-Key-6D achieves competitive accuracy scores of 96.24% and 69.41%, respectively, with the ADD(-S) 0.1d metric, while proving itself to operate in real time. Our results demonstrate that a carefully designed single stage method can provide a practical and effective balance of performance and efficiency for real world deployment.
- 6D 포즈 추정 기술이 스포츠 선수 자세 분석에 적용 가능하며 실시간 처리 가능


### 10. Trainable Bitwise Soft Quantization for Input Feature Compression

- **Score**: 93.6 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.05172v1](http://arxiv.org/abs/2603.05172v1)

- The growing demand for machine learning applications in the context of the Internet of Things calls for new approaches to optimize the use of limited compute and memory resources. Despite significant progress that has been made w.r.t. reducing model sizes and improving efficiency, many applications still require remote servers to provide the required resources. However, such approaches rely on transmitting data from edge devices to remote servers, which may not always be feasible due to bandwidth, latency, or energy constraints. We propose a task-specific, trainable feature quantization layer that compresses the input features of a neural network. This can significantly reduce the amount of data that needs to be transferred from the device to a remote server. In particular, the layer allows each input feature to be quantized to a user-defined number of bits, enabling a simple on-device compression at the time of data collection. The layer is designed to approximate step functions with sigmoids, enabling trainable quantization thresholds. By concatenating outputs from multiple sigmoids, introduced as bitwise soft quantization, it achieves trainable quantized values when integrated with a neural network. We compare our method to full-precision inference as well as to several quantization baselines. Experiments show that our approach outperforms standard quantization methods, while maintaining accuracy levels close to those of full-precision models. In particular, depending on the dataset, compression factors of $5\times$ to $16\times$ can be achieved compared to $32$-bit input without significant performance loss.
- RK3588 엣지 장치에서 데이터 전송량을 최대 16배 줄여 실시간 처리와 배터리 효율성을 크게 향상시킬 수 있습니다.






### Graduated Reminds


- **From Statics to Dynamics: Physics-Aware Image Editing with Latent Transition Priors** (score: None, graduated: 2026-03-02)
  [http://arxiv.org/abs/2602.21778v1](http://arxiv.org/abs/2602.21778v1)

- **GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry** (score: None, graduated: 2026-03-02)
  [http://arxiv.org/abs/2602.21810v1](http://arxiv.org/abs/2602.21810v1)

- **TEFL: Prediction-Residual-Guided Rolling Forecasting for Multi-Horizon Time Series** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.22520v1](http://arxiv.org/abs/2602.22520v1)

- **LoR-LUT: Learning Compact 3D Lookup Tables via Low-Rank Residuals** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.22607v1](http://arxiv.org/abs/2602.22607v1)

- **U-Net-Based Generative Joint Source-Channel Coding for Wireless Image Transmission** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.22691v1](http://arxiv.org/abs/2602.22691v1)

- **SCOPE: Skeleton Graph-Based Computation-Efficient Framework for Autonomous UAV Exploration** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.22707v1](http://arxiv.org/abs/2602.22707v1)

- **Beyond Detection: Multi-Scale Hidden-Code for Natural Image Deepfake Recovery and Factual Retrieval** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.22759v1](http://arxiv.org/abs/2602.22759v1)

- **Doubly Adaptive Channel and Spatial Attention for Semantic Image Communication by IoT Devices** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.22794v1](http://arxiv.org/abs/2602.22794v1)

- **GSTurb: Gaussian Splatting for Atmospheric Turbulence Mitigation** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.22800v1](http://arxiv.org/abs/2602.22800v1)

- **Velocity and stroke rate reconstruction of canoe sprint team boats based on panned and zoomed video recordings** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.22941v1](http://arxiv.org/abs/2602.22941v1)

- **UCM: Unifying Camera Control and Memory with Time-aware Positional Encoding Warping for World Models** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.22960v1](http://arxiv.org/abs/2602.22960v1)

- **PackUV: Packed Gaussian UV Maps for 4D Volumetric Video** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23040v1](http://arxiv.org/abs/2602.23040v1)

- **Align then Adapt: Rethinking Parameter-Efficient Transfer Learning in 4D Perception** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23069v1](http://arxiv.org/abs/2602.23069v1)

- **Locally Adaptive Decay Surfaces for High-Speed Face and Landmark Detection with Event Cameras** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23101v1](http://arxiv.org/abs/2602.23101v1)

- **FLIGHT: Fibonacci Lattice-based Inference for Geometric Heading in real-Time** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23115v1](http://arxiv.org/abs/2602.23115v1)

- **No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23141v1](http://arxiv.org/abs/2602.23141v1)

- **Learning Continuous Wasserstein Barycenter Space for Generalized All-in-One Image Restoration** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23169v1](http://arxiv.org/abs/2602.23169v1)

- **Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23188v1](http://arxiv.org/abs/2602.23188v1)

- **UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23224v1](http://arxiv.org/abs/2602.23224v1)

- **MovieTeller: Tool-augmented Movie Synopsis with ID Consistent Progressive Abstraction** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23228v1](http://arxiv.org/abs/2602.23228v1)

- **BRIDGE: Borderless Reconfiguration for Inclusive and Diverse Gameplay Experience via Embodiment Transformation** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23288v1](http://arxiv.org/abs/2602.23288v1)

- **Towards Long-Form Spatio-Temporal Video Grounding** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23294v1](http://arxiv.org/abs/2602.23294v1)

- **SOTAlign: Semi-Supervised Alignment of Unimodal Vision and Language Models via Optimal Transport** (score: None, graduated: 2026-03-03)
  [http://arxiv.org/abs/2602.23353v1](http://arxiv.org/abs/2602.23353v1)

- **Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.01400v1](http://arxiv.org/abs/2603.01400v1)

- **SeaVIS: Sound-Enhanced Association for Online Audio-Visual Instance Segmentation** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.01431v1](http://arxiv.org/abs/2603.01431v1)

- **WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.01475v1](http://arxiv.org/abs/2603.01475v1)

- **Boosting AI Reliability with an FSM-Driven Streaming Inference Pipeline: An Industrial Case** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.01528v1](http://arxiv.org/abs/2603.01528v1)

- **InterCoG: Towards Spatially Precise Image Editing with Interleaved Chain-of-Grounding Reasoning** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.01586v1](http://arxiv.org/abs/2603.01586v1)

- **PPEDCRF: Privacy-Preserving Enhanced Dynamic CRF for Location-Privacy Protection for Sequence Videos with Minimal Detection Degradation** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.01593v1](http://arxiv.org/abs/2603.01593v1)

- **MSP-ReID: Hairstyle-Robust Cloth-Changing Person Re-Identification** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.01640v1](http://arxiv.org/abs/2603.01640v1)

- **Search Multilayer Perceptron-Based Fusion for Efficient and Accurate Siamese Tracking** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.01706v1](http://arxiv.org/abs/2603.01706v1)

- **Downstream Task Inspired Underwater Image Enhancement: A Perception-Aware Study from Dataset Construction to Network Design** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.01767v1](http://arxiv.org/abs/2603.01767v1)

- **WorldStereo: Bridging Camera-Guided Video Generation and Scene Reconstruction via 3D Geometric Memories** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.02049v1](http://arxiv.org/abs/2603.02049v1)

- **Orchestrating Multimodal DNN Workloads in Wireless Neural Processing** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.02109v1](http://arxiv.org/abs/2603.02109v1)

- **Stereo-Inertial Poser: Towards Metric-Accurate Shape-Aware Motion Capture Using Sparse IMUs and a Single Stereo Camera** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.02130v1](http://arxiv.org/abs/2603.02130v1)

- **OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.02134v1](http://arxiv.org/abs/2603.02134v1)

- **NextAds: Towards Next-generation Personalized Video Advertising** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.02137v1](http://arxiv.org/abs/2603.02137v1)

- **Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.02139v1](http://arxiv.org/abs/2603.02139v1)

- **Kiwi-Edit: Versatile Video Editing via Instruction and Reference Guidance** (score: None, graduated: 2026-03-04)
  [http://arxiv.org/abs/2603.02175v1](http://arxiv.org/abs/2603.02175v1)

- **Biomechanically Accurate Gait Analysis: A 3d Human Reconstruction Framework for Markerless Estimation of Gait Parameters** (score: None, graduated: 2026-03-05)
  [http://arxiv.org/abs/2603.02499v1](http://arxiv.org/abs/2603.02499v1)

- **Self-supervised Domain Adaptation for Visual 3D Pose Estimation of Nano-drone Racing Gates by Enforcing Geometric Consistency** (score: None, graduated: 2026-03-05)
  [http://arxiv.org/abs/2603.02936v1](http://arxiv.org/abs/2603.02936v1)

- **DLIOS: An LLM-Augmented Real-Time Multi-Modal Interactive Enhancement Overlay System for Douyin Live Streaming** (score: None, graduated: 2026-03-05)
  [http://arxiv.org/abs/2603.03060v1](http://arxiv.org/abs/2603.03060v1)







### Notable Authors

| Author | Papers | Avg Score |
|--------|--------|-----------|

| Feng Gao | 4 | 78.3 |

| Yen-Yu Lin | 3 | 84.8 |

| Hao Li | 3 | 82.8 |

| Hao Liu | 3 | 80.8 |

| Hao Yang | 3 | 76.8 |

| Lei Zhang | 3 | 73.6 |

| Qing Li | 3 | 69.5 |

| Liming Liu | 2 | 93.2 |

| Jiangkai Wu | 2 | 93.2 |

| Xinggong Zhang | 2 | 93.2 |







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

| 2026-03-02 | 76.66 |

| 2026-03-03 | 79.6 |

| 2026-03-04 | 75.8 |

| 2026-03-05 | 77.33 |



