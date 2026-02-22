# Weekly Paper Report: 20260222

> ISO 2026-W08




## 1. Weekly Briefing


이번 주 평가 논문 수가 108건으로 급증하며, 지난 3주간 무활동 상태에서 벗어나 4주 추세가 상승세로 전환되었습니다. 모든 논문이 Tier 1로 분류되었으며, CS.CV 분야 논문이 39건(전체의 36.1%)을 차지한 점이 주목됩니다. 평균 점수 74.17을 기록했으나, 키워드 히트 수는 여전히 0건으로 나타났습니다.



| Metric | Value | WoW |
|--------|-------|-----|
| Evaluated | 108 | - |
| Tier 1 | 108 | - |
| cs.CV | 39 | N/A |
| Keyword Hits | 0 | N/A |
| Avg Score | 74.17 | N/A |



### 4-Week Trend (uptrend)

| Week | Papers | cs.CV | Avg Score |
|------|--------|-------|-----------|

| W1 | 0 | 0 | 0.0 |

| W2 | 0 | 0 | 0.0 |

| W3 | 0 | 0 | 0.0 |

| W4 | 108 | 39 | 74.17 |





### Top Categories

| Category | Count | WoW % |
|----------|-------|-------|

| cs.CV | 39 | N/A |

| cs.LG | 33 | N/A |

| cs.AI | 32 | N/A |

| cs.RO | 20 | N/A |

| cs.CL | 6 | N/A |





- Graduated Reminds: 35
- Active Reminds: 9




### Tech Radar








#### TF-IDF Keywords

`video` (NEW) `visual` (NEW) `learning` (NEW) `detection` (NEW) `data` (NEW) `time` (NEW) `image` (NEW) `system` (NEW) `language` (NEW) `control` (NEW) `training` (NEW) `tasks` (NEW) `real` (NEW) `object` (NEW) `graph` (NEW) `social` (NEW) `classification` (NEW) `accuracy` (NEW) `quality` (NEW) `knowledge` (NEW) 







## 2. Top Papers


### 1. Pareto Optimal Benchmarking of AI Models on ARM Cortex Processors for Sustainable Embedded Systems

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.17508v1](http://arxiv.org/abs/2602.17508v1)

- This work presents a practical benchmarking framework for optimizing artificial intelligence (AI) models on ARM Cortex processors (M0+, M4, M7), focusing on energy efficiency, accuracy, and resource utilization in embedded systems. Through the design of an automated test bench, we provide a systematic approach to evaluate across key performance indicators (KPIs) and identify optimal combinations of processor and AI model. The research highlights a nearlinear correlation between floating-point operations (FLOPs) and inference time, offering a reliable metric for estimating computational demands. Using Pareto analysis, we demonstrate how to balance trade-offs between energy consumption and model accuracy, ensuring that AI applications meet performance requirements without compromising sustainability. Key findings indicate that the M7 processor is ideal for short inference cycles, while the M4 processor offers better energy efficiency for longer inference tasks. The M0+ processor, while less efficient for complex AI models, remains suitable for simpler tasks. This work provides insights for developers, guiding them to design energy-efficient AI systems that deliver high performance in realworld applications.
- 이 논문은 ARM Cortex 프로세서에서 AI 모델의 에너지 효율성과 성능 최적화 방법을 제안합니다. 핵심은 Pareto 분석을 통한 자원 활용 균형입니다. 에지 디바이스(rk3588)의 실시간 영상 처리에 필수적인 AI 모델 최적화 기술이기 때문에 중요합니다.


### 2. HybridPrompt: Bridging Generative Priors and Traditional Codecs for Mobile Streaming

- **Score**: 98.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.17120v1](http://arxiv.org/abs/2602.17120v1)

- In Video on Demand (VoD) scenarios, traditional codecs are the industry standard due to their high decoding efficiency. However, they suffer from severe quality degradation under low bandwidth conditions. While emerging generative neural codecs offer significantly higher perceptual quality, their reliance on heavy frame-by-frame generation makes real-time playback on mobile devices impractical. We ask: is it possible to combine the blazing-fast speed of traditional standards with the superior visual fidelity of neural approaches? We present HybridPrompt, the first generative-based video system capable of achieving real-time 1080p decoding at over 150 FPS on a commercial smartphone. Specifically, we employ a hybrid architecture that encodes Keyframes using a generative model while relying on traditional codecs for the remaining frames. A major challenge is that the two paradigms have conflicting objectives: the "hallucinated" details from generative models often misalign with the rigid prediction mechanisms of traditional codecs, causing bitrate inefficiency. To address this, we demonstrate that the traditional decoding process is differentiable, enabling an end-to-end optimization loop. This allows us to use subsequent frames as additional supervision, forcing the generative model to synthesize keyframes that are not only perceptually high-fidelity but also mathematically optimal references for the traditional codec. By integrating a two-stage generation strategy, our system outperforms pure neural baselines by orders of magnitude in speed while achieving an average LPIPS gain of 8% over traditional codecs at 200kbps.
- 이 논문은 저대역폭 환경에서 고품질 영상 스트리밍을 위한 하이브리드 인코딩 방법을 제안합니다. 핵심은 생성 모델과 전통 코덱의 결합입니다. 에지 디바이스에서 실시간 영상 보정 및 SNS 공유 시 대역폭 효율성을 높일 수 있어 중요합니다.


### 3. Hybrid F' and ROS2 Architecture for Vision-Based Autonomous Flight: Design and Experimental Validation

- **Score**: 93.6 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.15398v1](http://arxiv.org/abs/2602.15398v1)

- Autonomous aerospace systems require architectures that balance deterministic real-time control with advanced perception capabilities. This paper presents an integrated system combining NASA's F' flight software framework with ROS2 middleware via Protocol Buffers bridging. We evaluate the architecture through a 32.25-minute indoor quadrotor flight test using vision-based navigation. The vision system achieved 87.19 Hz position estimation with 99.90\% data continuity and 11.47 ms mean latency, validating real-time performance requirements. All 15 ground commands executed successfully with 100 % success rate, demonstrating robust F'--PX4 integration. System resource utilization remained low (15.19 % CPU, 1,244 MB RAM) with zero stale telemetry messages, confirming efficient operation on embedded platforms. Results validate the feasibility of hybrid flight-software architectures combining certification-grade determinism with flexible autonomy for autonomous aerial vehicles.
- 이 연구는 임베디드 플랫폼에서 고속 비전 처리(87.19Hz, 11.47ms 지연)를 검증했으며, 우리 장치의 실시간 경기 촬영과 선수 위치 추적에 직접 적용 가능합니다.


### 4. Hybrid System Planning using a Mixed-Integer ADMM Heuristic and Hybrid Zonotopes

- **Score**: 92.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.17574v1](http://arxiv.org/abs/2602.17574v1)

- Embedded optimization-based planning for hybrid systems is challenging due to the use of mixed-integer programming, which is computationally intensive and often sensitive to the specific numerical formulation. To address that challenge, this article proposes a framework for motion planning of hybrid systems that pairs hybrid zonotopes - an advanced set representation - with a new alternating direction method of multipliers (ADMM) mixed-integer programming heuristic. A general treatment of piecewise affine (PWA) system reachability analysis using hybrid zonotopes is presented and extended to formulate optimal planning problems. Sets produced using the proposed identities have lower memory complexity and tighter convex relaxations than equivalent sets produced from preexisting techniques. The proposed ADMM heuristic makes efficient use of the hybrid zonotope structure. For planning problems formulated as hybrid zonotopes, the proposed heuristic achieves improved convergence rates as compared to state-of-the-art mixed-integer programming heuristics. The proposed methods for hybrid system planning on embedded hardware are experimentally applied in a combined behavior and motion planning scenario for autonomous driving.
- 이 논문은 내장 하드웨어용 효율적 모션 플래닝 방법을 제안합니다. 핵심은 하이브리드 존토프와 ADMM 휴리스틱의 결합입니다. 자동 촬영 디바이스의 카메라 움직임 최적화 및 선수 추적 알고리즘에 직접 적용 가능해 중요합니다.


### 5. Time-Archival Camera Virtualization for Sports and Visual Performances

- **Score**: 92.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.15181v1](http://arxiv.org/abs/2602.15181v1)

- Camera virtualization -- an emerging solution to novel view synthesis -- holds transformative potential for visual entertainment, live performances, and sports broadcasting by enabling the generation of photorealistic images from novel viewpoints using images from a limited set of calibrated multiple static physical cameras. Despite recent advances, achieving spatially and temporally coherent and photorealistic rendering of dynamic scenes with efficient time-archival capabilities, particularly in fast-paced sports and stage performances, remains challenging for existing approaches. Recent methods based on 3D Gaussian Splatting (3DGS) for dynamic scenes could offer real-time view-synthesis results. Yet, they are hindered by their dependence on accurate 3D point clouds from the structure-from-motion method and their inability to handle large, non-rigid, rapid motions of different subjects (e.g., flips, jumps, articulations, sudden player-to-player transitions). Moreover, independent motions of multiple subjects can break the Gaussian-tracking assumptions commonly used in 4DGS, ST-GS, and other dynamic splatting variants. This paper advocates reconsidering a neural volume rendering formulation for camera virtualization and efficient time-archival capabilities, making it useful for sports broadcasting and related applications. By modeling a dynamic scene as rigid transformations across multiple synchronized camera views at a given time, our method performs neural representation learning, providing enhanced visual rendering quality at test time. A key contribution of our approach is its support for time-archival, i.e., users can revisit any past temporal instance of a dynamic scene and can perform novel view synthesis, enabling retrospective rendering for replay, analysis, and archival of live events, a functionality absent in existing neural rendering approaches and novel view synthesis...
- 이 논문은 스포츠 중계를 위한 카메라 가상화 방법을 제안한다. 핵심은 타임-아카이브 기능이다. 우리 프로젝트에 중요한 이유는 경기 하이라이트 생성과 재생 분석에 직접 적용 가능하기 때문이다.


### 6. Towards Secure and Interoperable Data Spaces for 6G: The 6G-DALI Approach

- **Score**: 92.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.16386v1](http://arxiv.org/abs/2602.16386v1)

- The next generation of mobile networks, 6G, is expected to enable data-driven services at unprecedented scale and complexity, with stringent requirements for trust, interoperability, and automation. Central to this vision is the ability to create, manage, and share high-quality datasets across distributed and heterogeneous environments. This paper presents the data architecture of the 6G-DALI project, which implements a federated dataspace and DataOps infrastructure to support secure, compliant, and scalable data sharing for AI-driven experimentation and service orchestration. Drawing from principles defined by GAIA-X and the International Data Spaces Association (IDSA), the architecture incorporates components such as federated identity management, policy-based data contracts, and automated data pipelines. We detail how the 6G-DALI architecture aligns with and extends GAIA-X and IDSA reference models to meet the unique demands of 6G networks, including low-latency edge processing, dynamic trust management, and cross-domain federation. A comparative analysis highlights both convergence points and necessary innovations.
- 엣지 처리와 연합 데이터 아키텍처가 디바이스 플랫폼 및 SNS 공유 기능에 핵심적입니다. 데이터 파이프라인 자동화로 실시간 영상 처리 시 지연 시간 최적화가 가능합니다.


### 7. A.R.I.S.: Automated Recycling Identification System for E-Waste Classification Using Deep Learning

- **Score**: 92.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.17642v1](http://arxiv.org/abs/2602.17642v1)

- Traditional electronic recycling processes suffer from significant resource loss due to inadequate material separation and identification capabilities, limiting material recovery. We present A.R.I.S. (Automated Recycling Identification System), a low-cost, portable sorter for shredded e-waste that addresses this efficiency gap. The system employs a YOLOx model to classify metals, plastics, and circuit boards in real time, achieving low inference latency with high detection accuracy. Experimental evaluation yielded 90% overall precision, 82.2% mean average precision (mAP), and 84% sortation purity. By integrating deep learning with established sorting methods, A.R.I.S. enhances material recovery efficiency and lowers barriers to advanced recycling adoption. This work complements broader initiatives in extending product life cycles, supporting trade-in and recycling programs, and reducing environmental impact across the supply chain.
- 이 논문의 실시간 객체 인식 기술(YOLOx)은 우리 프로젝트의 핵심인 스포츠 경기에서 선수나 장비를 빠르게 식별하는 데 직접 적용 가능합니다. 낮은 inference latency와 높은 정확도가 경기 중 실시간 하이라이트 생성에 중요합니다.


### 8. EventMemAgent: Hierarchical Event-Centric Memory for Online Video Understanding with Adaptive Tool Use

- **Score**: 90.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.15329v1](http://arxiv.org/abs/2602.15329v1)

- Online video understanding requires models to perform continuous perception and long-range reasoning within potentially infinite visual streams. Its fundamental challenge lies in the conflict between the unbounded nature of streaming media input and the limited context window of Multimodal Large Language Models (MLLMs). Current methods primarily rely on passive processing, which often face a trade-off between maintaining long-range context and capturing the fine-grained details necessary for complex tasks. To address this, we introduce EventMemAgent, an active online video agent framework based on a hierarchical memory module. Our framework employs a dual-layer strategy for online videos: short-term memory detects event boundaries and utilizes event-granular reservoir sampling to process streaming video frames within a fixed-length buffer dynamically; long-term memory structuredly archives past observations on an event-by-event basis. Furthermore, we integrate a multi-granular perception toolkit for active, iterative evidence capture and employ Agentic Reinforcement Learning (Agentic RL) to end-to-end internalize reasoning and tool-use strategies into the agent's intrinsic capabilities. Experiments show that EventMemAgent achieves competitive results on online video benchmarks. The code will be released here: https://github.com/lingcco/EventMemAgent.
- 이벤트 감지 기술이 경기의 자동 하이라이트 편집에 필수적이므로, 실시간 스트리밍 영상에서 골이나 역전 같은 주요 장면을 효율적으로 식별합니다.


### 9. Selective Perception for Robot: Task-Aware Attention in Multimodal VLA

- **Score**: 90.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.15543v1](http://arxiv.org/abs/2602.15543v1)

- In robotics, Vision-Language-Action (VLA) models that integrate diverse multimodal signals from multi-view inputs have emerged as an effective approach. However, most prior work adopts static fusion that processes all visual inputs uniformly, which incurs unnecessary computational overhead and allows task-irrelevant background information to act as noise. Inspired by the principles of human active perception, we propose a dynamic information fusion framework designed to maximize the efficiency and robustness of VLA models. Our approach introduces a lightweight adaptive routing architecture that analyzes the current text prompt and observations from a wrist-mounted camera in real-time to predict the task-relevance of multiple camera views. By conditionally attenuating computations for views with low informational utility and selectively providing only essential visual features to the policy network, Our framework achieves computation efficiency proportional to task relevance. Furthermore, to efficiently secure large-scale annotation data for router training, we established an automated labeling pipeline utilizing Vision-Language Models (VLMs) to minimize data collection and annotation costs. Experimental results in real-world robotic manipulation scenarios demonstrate that the proposed approach achieves significant improvements in both inference efficiency and control performance compared to existing VLA models, validating the effectiveness and practicality of dynamic information fusion in resource-constrained, real-time robot control environments.
- 동적 정보 융합 기술이 계산 효율을 높여 에지 디바이스의 실시간 성능을 개선하므로, 다중 카메라 스포츠 장면 분석 시 불필요한 연산을 줄일 수 있습니다.


### 10. SAM 3D Body: Robust Full-Body Human Mesh Recovery

- **Score**: 90.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2602.15989v1](http://arxiv.org/abs/2602.15989v1)

- We introduce SAM 3D Body (3DB), a promptable model for single-image full-body 3D human mesh recovery (HMR) that demonstrates state-of-the-art performance, with strong generalization and consistent accuracy in diverse in-the-wild conditions. 3DB estimates the human pose of the body, feet, and hands. It is the first model to use a new parametric mesh representation, Momentum Human Rig (MHR), which decouples skeletal structure and surface shape. 3DB employs an encoder-decoder architecture and supports auxiliary prompts, including 2D keypoints and masks, enabling user-guided inference similar to the SAM family of models. We derive high-quality annotations from a multi-stage annotation pipeline that uses various combinations of manual keypoint annotation, differentiable optimization, multi-view geometry, and dense keypoint detection. Our data engine efficiently selects and processes data to ensure data diversity, collecting unusual poses and rare imaging conditions. We present a new evaluation dataset organized by pose and appearance categories, enabling nuanced analysis of model behavior. Our experiments demonstrate superior generalization and substantial improvements over prior methods in both qualitative user preference studies and traditional quantitative analysis. Both 3DB and MHR are open-source.
- 3D 인체 메쉬 복원 기술이 운동 자세 분석의 핵심이므로, 우리 장치의 동작 교정 기능에 즉시 통합해 정확한 골절/관절 데이터를 제공할 수 있습니다.






### Graduated Reminds


- **AdvSynGNN: Structure-Adaptive Graph Neural Nets via Adversarial Synthesis and Self-Corrective Propagation** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17071v1](http://arxiv.org/abs/2602.17071v1)

- **Cross Pseudo Labeling For Weakly Supervised Video Anomaly Detection** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17077v1](http://arxiv.org/abs/2602.17077v1)

- **EditCtrl: Disentangled Local and Global Control for Real-Time Generative Video Editing** (score: None, graduated: 2026-02-17)
  [http://arxiv.org/abs/2602.15031v1](http://arxiv.org/abs/2602.15031v1)

- **Joint Enhancement and Classification using Coupled Diffusion Models of Signals and Logits** (score: None, graduated: 2026-02-19)
  [http://arxiv.org/abs/2602.15405v1](http://arxiv.org/abs/2602.15405v1)

- **WarpRec: Unifying Academic Rigor and Industrial Scale for Responsible, Reproducible, and Efficient Recommendation** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17442v1](http://arxiv.org/abs/2602.17442v1)

- **GOT-JEPA: Generic Object Tracking with Model Adaptation and Occlusion Handling using Joint-Embedding Predictive Architecture** (score: None, graduated: 2026-02-18)
  [http://arxiv.org/abs/2602.14771v1](http://arxiv.org/abs/2602.14771v1)

- **EntropyPrune: Matrix Entropy Guided Visual Token Pruning for Multimodal Large Language Models** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17196v1](http://arxiv.org/abs/2602.17196v1)

- **Pareto Optimal Benchmarking of AI Models on ARM Cortex Processors for Sustainable Embedded Systems** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17508v1](http://arxiv.org/abs/2602.17508v1)

- **Catastrophic Forgetting Resilient One-Shot Incremental Federated Learning** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17625v1](http://arxiv.org/abs/2602.17625v1)

- **GraphThinker: Reinforcing Video Reasoning with Event Graph Thinking** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17555v1](http://arxiv.org/abs/2602.17555v1)

- **A.R.I.S.: Automated Recycling Identification System for E-Waste Classification Using Deep Learning** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17642v1](http://arxiv.org/abs/2602.17642v1)

- **VIPA: Visual Informative Part Attention for Referring Image Segmentation** (score: None, graduated: 2026-02-17)
  [http://arxiv.org/abs/2602.14788v1](http://arxiv.org/abs/2602.14788v1)

- **A Q-Learning Approach for Dynamic Resource Management in Three-Tier Vehicular Fog Computing** (score: None, graduated: 2026-02-18)
  [http://arxiv.org/abs/2602.14390v1](http://arxiv.org/abs/2602.14390v1)

- **Time-Archival Camera Virtualization for Sports and Visual Performances** (score: None, graduated: 2026-02-17)
  [http://arxiv.org/abs/2602.15181v1](http://arxiv.org/abs/2602.15181v1)

- **Adapting VACE for Real-Time Autoregressive Video Diffusion** (score: None, graduated: 2026-02-17)
  [http://arxiv.org/abs/2602.14381v1](http://arxiv.org/abs/2602.14381v1)

- **EventMemAgent: Hierarchical Event-Centric Memory for Online Video Understanding with Adaptive Tool Use** (score: None, graduated: 2026-02-19)
  [http://arxiv.org/abs/2602.15329v1](http://arxiv.org/abs/2602.15329v1)

- **Hybrid F' and ROS2 Architecture for Vision-Based Autonomous Flight: Design and Experimental Validation** (score: None, graduated: 2026-02-19)
  [http://arxiv.org/abs/2602.15398v1](http://arxiv.org/abs/2602.15398v1)

- **ManeuverNet: A Soft Actor-Critic Framework for Precise Maneuvering of Double-Ackermann-Steering Robots with Optimized Reward Functions** (score: None, graduated: 2026-02-18)
  [http://arxiv.org/abs/2602.14726v1](http://arxiv.org/abs/2602.14726v1)

- **Reflecting on 1,000 Social Media Journeys: Generational Patterns in Platform Transition** (score: None, graduated: 2026-02-19)
  [http://arxiv.org/abs/2602.15489v1](http://arxiv.org/abs/2602.15489v1)

- **Selective Training for Large Vision Language Models via Visual Information Gain** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17186v1](http://arxiv.org/abs/2602.17186v1)

- **Universal Image Immunization against Diffusion-based Image Editing via Semantic Injection** (score: None, graduated: 2026-02-18)
  [http://arxiv.org/abs/2602.14679v1](http://arxiv.org/abs/2602.14679v1)

- **Automatic Funny Scene Extraction from Long-form Cinematic Videos** (score: None, graduated: 2026-02-19)
  [http://arxiv.org/abs/2602.15381v1](http://arxiv.org/abs/2602.15381v1)

- **Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition** (score: None, graduated: 2026-02-17)
  [http://arxiv.org/abs/2602.15124v1](http://arxiv.org/abs/2602.15124v1)

- **Efficient Text-Guided Convolutional Adapter for the Diffusion Model** (score: None, graduated: 2026-02-18)
  [http://arxiv.org/abs/2602.14514v1](http://arxiv.org/abs/2602.14514v1)

- **FLoRG: Federated Fine-tuning with Low-rank Gram Matrices and Procrustes Alignment** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17095v1](http://arxiv.org/abs/2602.17095v1)

- **AI Sessions for Network-Exposed AI-as-a-Service** (score: None, graduated: 2026-02-19)
  [http://arxiv.org/abs/2602.15288v2](http://arxiv.org/abs/2602.15288v2)

- **RetouchIQ: MLLM Agents for Instruction-Based Image Retouching with Generalist Reward** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17558v1](http://arxiv.org/abs/2602.17558v1)

- **Selective Perception for Robot: Task-Aware Attention in Multimodal VLA** (score: None, graduated: 2026-02-19)
  [http://arxiv.org/abs/2602.15543v1](http://arxiv.org/abs/2602.15543v1)

- **Hybrid System Planning using a Mixed-Integer ADMM Heuristic and Hybrid Zonotopes** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17574v1](http://arxiv.org/abs/2602.17574v1)

- **Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17155v1](http://arxiv.org/abs/2602.17155v1)

- **IRIS: Learning-Driven Task-Specific Cinema Robot Arm for Visuomotor Motion Control** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17537v1](http://arxiv.org/abs/2602.17537v1)

- **MyoInteract: A Framework for Fast Prototyping of Biomechanical HCI Tasks using Reinforcement Learning** (score: None, graduated: 2026-02-18)
  [http://arxiv.org/abs/2602.15245v1](http://arxiv.org/abs/2602.15245v1)

- **MaS-VQA: A Mask-and-Select Framework for Knowledge-Based Visual Question Answering** (score: None, graduated: 2026-02-19)
  [http://arxiv.org/abs/2602.15915v1](http://arxiv.org/abs/2602.15915v1)

- **SAM 3D Body: Robust Full-Body Human Mesh Recovery** (score: None, graduated: 2026-02-19)
  [http://arxiv.org/abs/2602.15989v1](http://arxiv.org/abs/2602.15989v1)

- **HybridPrompt: Bridging Generative Priors and Traditional Codecs for Mobile Streaming** (score: None, graduated: 2026-02-16)
  [http://arxiv.org/abs/2602.17120v1](http://arxiv.org/abs/2602.17120v1)







### Notable Authors

| Author | Papers | Avg Score |
|--------|--------|-----------|

| Obaidullah Zaland | 2 | 78.8 |

| Monowar Bhuyan | 2 | 78.8 |

| Adel N. Toosi | 2 | 78.0 |

| Alejandro Flores | 2 | 70.0 |

| Konstantinos Ntontin | 2 | 70.0 |

| Symeon Chatzinotas | 2 | 70.0 |







## 3. Trends



### cappic-ai

| Date | Avg Score |
|------|-----------|

| 2026-02-16 | 78.09 |

| 2026-02-17 | 72.39 |

| 2026-02-18 | 74.52 |

| 2026-02-18 | 75.53 |

| 2026-02-19 | 62.97 |



