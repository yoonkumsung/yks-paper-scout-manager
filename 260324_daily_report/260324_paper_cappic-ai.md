# CAPP!C_AI 논문 리포트 (2026-03-24)

> 수집 96 | 필터 93 | 폐기 11 | 평가 73 | 출력 38 | 기준 50점

검색 윈도우: 2026-03-23T00:00:00+00:00 ~ 2026-03-24T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 48

---

## 검색 키워드

active camera, camera control, sports tracking, action recognition, keyframe extraction, highlight generation, video stabilization, image correction, quality enhancement, pose estimation, biomechanics analysis, movement tracking, tactical analysis, sports analytics, game strategy, video summarization, clip generation, short-form video, edge deployment, embedded AI, physical interaction, multi-object tracking, real-time processing, user-generated content, social platform

---

## 1위: Convolutions Predictable Offloading to an Accelerator: Formalization and Optimization

- arXiv: http://arxiv.org/abs/2603.21792v1
- PDF: https://arxiv.org/pdf/2603.21792v1
- 발행일: 2026-03-23
- 카테고리: cs.AR
- 점수: final 98.4 (llm_adjusted:98 = base:85 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Convolutional neural networks (CNNs) require a large number of multiply-accumulate (MAC) operations. To meet real-time constraints, they often need to be executed on specialized accelerators composed of an on-chip memory and a processing unit. However, the on-chip memory is often insufficient to store all the data required to compute a CNN layer. Thus, the computation must be performed in several offloading steps. We formalise such sequences of steps and apply our formalism to a state of the art decomposition of convolutions. In order to find optimal strategies in terms of duration, we encode the problem with a set of constraints. A Python-based simulator allows to analyse in-depth computed strategies.

**선정 근거**
rk3588 엣지 디바이스에서 AI 모델 실행 최적화에 직접적으로 관련된 연구로 실시간 스포츠 촬영 및 분석 성능 향상에 필수적임

**활용 인사이트**
컨볼루션 신경망의 계산 단계를 예측 가능한 오프로딩 전략으로 최적화하여 메모리 제약을 극복하고 실시간 처리 속도를 향상시킬 수 있음

## 2위: No Dense Tensors Needed: Fully Sparse Object Detection on Event-Camera Voxel Grids

- arXiv: http://arxiv.org/abs/2603.21638v1
- PDF: https://arxiv.org/pdf/2603.21638v1
- 발행일: 2026-03-23
- 카테고리: cs.CV
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Event cameras produce asynchronous, high-dynamic-range streams well suited for detecting small, fast-moving drones, yet most event-based detectors convert the sparse event stream into dense tensors, discarding the representational efficiency of neuromorphic sensing. We propose SparseVoxelDet, to our knowledge the first fully sparse object detector for event cameras, in which backbone feature extraction, feature pyramid fusion, and the detection head all operate exclusively on occupied voxel positions through 3D sparse convolutions; no dense feature tensor is instantiated at any stage of the pipeline. On the FRED benchmark (629,832 annotated frames), SparseVoxelDet achieves 83.38% mAP at 50 while processing only 14,900 active voxels per frame (0.23% of the T.H.W grid), compared to 409,600 pixels for the dense YOLOv11 baseline (87.68% mAP at 50). Relaxing the IoU threshold from 0.50 to 0.40 recovers mAP to 89.26%, indicating that the remaining accuracy gap is dominated by box regression precision rather than detection capability. The sparse representation yields 858 times GPU memory compression and 3,670 times storage reduction relative to the equivalent dense 3D voxel tensor, with data-structure size that scales with scene dynamics rather than sensor resolution. Error forensics across 119,459 test frames confirms that 71 percent of failures are localization near-misses rather than missed targets. These results demonstrate that native sparse processing is a viable paradigm for event-camera object detection, exploiting the structural sparsity of neuromorphic sensor data without requiring neuromorphic computing hardware, and providing a framework whose representation cost is governed by scene activity rather than pixel count, a property that becomes increasingly valuable as event cameras scale to higher resolutions.

**선정 근거**
이벤트 카메라를 이용한 효율적인 희소 객체 탐지 기술로 엣지 디바이스에 직접 적용 가능

## 3위: Rateless DeepJSCC for Broadcast Channels: a Rate-Distortion-Complexity Tradeoff

- arXiv: http://arxiv.org/abs/2603.21616v1
- PDF: https://arxiv.org/pdf/2603.21616v1
- 발행일: 2026-03-23
- 카테고리: cs.IT, cs.LG, eess.SP
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
In recent years, numerous data-intensive broadcasting applications have emerged at the wireless edge, calling for a flexible tradeoff between distortion, transmission rate, and processing complexity. While deep learning-based joint source-channel coding (DeepJSCC) has been identified as a potential solution to data-intensive communications, most of these schemes are confined to worst-case solutions, lack adaptive complexity, and are inefficient in broadcast settings. To overcome these limitations, this paper introduces nonlinear transform rateless source-channel coding (NTRSCC), a variable-length JSCC framework for broadcast channels based on rateless codes. In particular, we integrate learned source transformations with physical-layer LT codes, develop unequal protection schemes that exploit decoder side information, and devise approximations to enable end-to-end optimization of rateless parameters. Our framework enables heterogeneous receivers to adaptively adjust their received number of rateless symbols and decoding iterations in belief propagation, thereby achieving a controllable tradeoff between distortion, rate, and decoding complexity. Simulation results demonstrate that the proposed method enhances image broadcast quality under stringent communication and processing budgets over heterogeneous edge devices.

**선정 근거**
Rateless DeepJSCC framework for broadcast channels could enable efficient streaming of sports content to heterogeneous edge devices.

## 4위: ANCHOR: Adaptive Network based on Cascaded Harmonic Offset Routing

- arXiv: http://arxiv.org/abs/2603.21718v1
- PDF: https://arxiv.org/pdf/2603.21718v1
- 발행일: 2026-03-23
- 카테고리: eess.SP
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Time series analysis plays a foundational role in a wide range of real-world applications, yet accurately modeling complex non-stationary signals remains a shared challenge across downstream tasks. Existing methods attempt to extract features directly from one-dimensional sequences, making it difficult to handle the widely observed dynamic phase drift and discrete quantization error. To address this issue, we decouple temporal evolution into macroscopic physical periods and microscopic phase perturbations, and inject frequency-domain priors derived from the Real Fast Fourier Transform (RFFT) into the underlying spatial sampling process. Based on this idea, we propose a Frequency-Guided Deformable Module (FGDM) to adaptively compensate for microscopic phase deviations. Built upon FGDM, we further develop an Adaptive Network based on Cascaded Harmonic Offset Routing (ANCHOR) as a general-purpose backbone for time-series modeling. Through orthogonal channel partitioning and a progressive residual architecture, ANCHOR efficiently decouples multi-scale harmonic features while substantially suppressing the computational redundancy of multi-branch networks. Extensive experiments demonstrate that ANCHOR achieves the best performance in most short-term forecasting sub-tasks and exhibits strong competitiveness on several specific sub-tasks in anomaly detection and time-series classification, validating its effectiveness as a universal time-series foundation backbone.

**선정 근거**
시계열 분석 기술로 스포츠 동작 분석에 적용 가능

## 5위: StreamingClaw Technical Report

- arXiv: http://arxiv.org/abs/2603.22120v1
- PDF: https://arxiv.org/pdf/2603.22120v1
- 발행일: 2026-03-23
- 카테고리: cs.CV
- 점수: final 92.0 (llm_adjusted:90 = base:85 + bonus:+5)
- 플래그: 실시간

**개요**
Applications such as embodied intelligence rely on a real-time perception-decision-action closed loop, posing stringent challenges for streaming video understanding. However, current agents suffer from fragmented capabilities, such as supporting only offline video understanding, lacking long-term multimodal memory mechanisms, or struggling to achieve real-time reasoning and proactive interaction under streaming inputs. These shortcomings have become a key bottleneck for preventing them from sustaining perception, making real-time decisions, and executing actions in real-world environments. To alleviate these issues, we propose StreamingClaw, a unified agent framework for streaming video understanding and embodied intelligence. It is also an OpenClaw-compatible framework that supports real-time, multimodal streaming interaction. StreamingClaw integrates five core capabilities: (1) It supports real-time streaming reasoning. (2) It supports reasoning about future events and proactive interaction under the online evolution of interaction objectives. (3) It supports multimodal long-term storage, hierarchical evolution, and efficient retrieval of shared memory across multiple agents. (4) It supports a closed-loop of perception-decision-action. In addition to conventional tools and skills, it also provides streaming tools and action-centric skills tailored for real-world physical environments. (5) It is compatible with the OpenClaw framework, allowing it to fully leverage the resources and support of the open-source community. With these designs, StreamingClaw integrates online real-time reasoning, multimodal long-term memory, and proactive interaction within a unified framework. Moreover, by translating decisions into executable actions, it enables direct control of the physical world, supporting practical deployment of embodied interaction.

**선정 근거**
실시간 영상 이해 및 의사결정-행동 루프 기술이 스포츠 자동 촬영에 적용 가능

**활용 인사이트**
StreamingClaw 프레임워크를 활용해 실시간으로 운동선수의 동작을 분석하고 최적의 촬영 각도를 자동으로 결정

## 6위: Not All Layers Are Created Equal: Adaptive LoRA Ranks for Personalized Image Generation

- arXiv: http://arxiv.org/abs/2603.21884v1
- PDF: https://arxiv.org/pdf/2603.21884v1
- 코드: https://github.com/donaldssh/NotAllLayersAreCreatedEqual
- 발행일: 2026-03-23
- 카테고리: cs.CV, cs.AI, cs.LG
- 점수: final 92.0 (llm_adjusted:90 = base:82 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Low Rank Adaptation (LoRA) is the de facto fine-tuning strategy to generate personalized images from pre-trained diffusion models. Choosing a good rank is extremely critical, since it trades off performance and memory consumption, but today the decision is often left to the community's consensus, regardless of the personalized subject's complexity. The reason is evident: the cost of selecting a good rank for each LoRA component is combinatorial, so we opt for practical shortcuts such as fixing the same rank for all components. In this paper, we take a first step to overcome this challenge. Inspired by variational methods that learn an adaptive width of neural networks, we let the ranks of each layer freely adapt during fine-tuning on a subject. We achieve it by imposing an ordering of importance on the rank's positions, effectively encouraging the creation of higher ranks when strictly needed. Qualitatively and quantitatively, our approach, LoRA$^2$, achieves a competitive trade-off between DINO, CLIP-I, and CLIP-T across 29 subjects while requiring much less memory and lower rank than high rank LoRA versions. Code: https://github.com/donaldssh/NotAllLayersAreCreatedEqual.

**선정 근거**
개인화된 이미지 생성을 위한 적응형 LoRA 기술은 스포츠 영상을 맞춤형 하이라이트 시각물로 변환하는 데 사용될 수 있습니다.

**활용 인사이트**
LoRA² 기술을 적용하여 스포츠 장면 복잡도에 맞춰 메모리 효율적으로 고품질 하이라이트 영상을 생성할 수 있습니다.

## 7위: Benchmarking Message Brokers for IoT Edge Computing: A Comprehensive Performance Study

- arXiv: http://arxiv.org/abs/2603.21600v1
- PDF: https://arxiv.org/pdf/2603.21600v1
- 발행일: 2026-03-23
- 카테고리: cs.DC
- 점수: final 92.0 (llm_adjusted:90 = base:80 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Asynchronous messaging is a cornerstone of modern distributed systems, enabling decoupled communication for scalable and resilient applications. Today's message queue (MQ) ecosystem spans a wide range of designs, from high-throughput streaming platforms to lightweight protocols tailored for edge and IoT environments. Despite this diversity, choosing an appropriate MQ system remains difficult. Existing evaluations largely focus on throughput and latency on fixed hardware, while overlooking CPU and memory footprint and the effects of resource constraints, factors that are critical for edge and IoT deployments. In this paper, we present a systematic performance study of eight prominent message brokers: Mosquitto, EMQX, HiveMQ, RabbitMQ, ActiveMQ Artemis, NATS Server, Redis (Pub/Sub), and Zenoh Router. We introduce mq-bench, a unified benchmarking framework to evaluate these systems under identical conditions, scaling up to 10,000 concurrent client pairs across three VM configurations representative of edge hardware. This study reveals several interesting and sometimes counter-intuitive insights. Lightweight native brokers achieve sub-millisecond latency, while feature-rich enterprise platforms incur 2-3X higher overhead. Under high connection loads, multi-threaded brokers like NATS and Zenoh scale efficiently, whereas the widely-deployed Mosquitto saturates earlier due to its single-threaded architecture. We also find that Java-based brokers consume significantly more memory than native implementations, which has important implications for memory-constrained edge deployments. Based on these findings, we provide practical deployment guidelines that map workload requirements and resource constraints to appropriate broker choices for telemetry, streaming analytics, and IoT use cases.

**선정 근거**
IoT 엣지 컴퓨팅용 메시지 브로커 벤치마킹은 AI 카메라 디바이스의 내부 통신 시스템 설계에 중요합니다.

**활용 인사이트**
가벼운 네이티브 브로커를 선택하여 저지연(ms 이하) 고성능 통신을 구현하고, 다중 스레딩으로 높은 연결 부하를 효율적으로 처리할 수 있습니다.

## 8위: The Universal Normal Embedding

- arXiv: http://arxiv.org/abs/2603.21786v1
- PDF: https://arxiv.org/pdf/2603.21786v1
- 발행일: 2026-03-23
- 카테고리: cs.CV, eess.IV
- 점수: final 90.4 (llm_adjusted:88 = base:85 + bonus:+3)
- 플래그: 코드 공개

**개요**
Generative models and vision encoders have largely advanced on separate tracks, optimized for different goals and grounded in different mathematical principles. Yet, they share a fundamental property: latent space Gaussianity. Generative models map Gaussian noise to images, while encoders map images to semantic embeddings whose coordinates empirically behave as Gaussian. We hypothesize that both are views of a shared latent source, the Universal Normal Embedding (UNE): an approximately Gaussian latent space from which encoder embeddings and DDIM-inverted noise arise as noisy linear projections. To test our hypothesis, we introduce NoiseZoo, a dataset of per-image latents comprising DDIM-inverted diffusion noise and matching encoder representations (CLIP, DINO). On CelebA, linear probes in both spaces yield strong, aligned attribute predictions, indicating that generative noise encodes meaningful semantics along linear directions. These directions further enable faithful, controllable edits (e.g., smile, gender, age) without architectural changes, where simple orthogonalization mitigates spurious entanglements. Taken together, our results provide empirical support for the UNE hypothesis and reveal a shared Gaussian-like latent geometry that concretely links encoding and generation. Code and data are available https://rbetser.github.io/UNE/

**선정 근거**
유니버설 노말 임베딩 프레임워크는 스포츠 영상 처리 및 분석에 직접적으로 적용 가능하며, 생성 모델과 인코더 간의 잠재 공간 연결을 통해 영상 보정과 하이라이트 생성을 향상시킬 수 있습니다.

**활용 인사이트**
스포츠 영상을 CLIP 및 DINO 임베딩으로 변환하고 선형 탐침을 통해 의미 있는 속성 예측을 수행하여 특정 동작이나 경기 장면을 정확히 식별하고 분류하며, 이를 통해 개인별 하이라이트 자동 편집이 가능해집니다.

## 9위: Feature Incremental Clustering with Generalization Bounds

- arXiv: http://arxiv.org/abs/2603.21590v1
- PDF: https://arxiv.org/pdf/2603.21590v1
- 발행일: 2026-03-23
- 카테고리: math.ST, cs.LG
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
In many learning systems, such as activity recognition systems, as new data collection methods continue to emerge in various dynamic environmental applications, the attributes of instances accumulate incrementally, with data being stored in gradually expanding feature spaces. How to design theoretically guaranteed algorithms to effectively cluster this special type of data stream, commonly referred to as activity recognition, remains unexplored. Compared to traditional scenarios, we will face at least two fundamental questions in this feature incremental scenario. (i) How to design preliminary and effective algorithms to address the feature incremental clustering problem? (ii) How to analyze the generalization bounds for the proposed algorithms and under what conditions do these algorithms provide a strong generalization guarantee? To address these problems, by tailoring the most common clustering algorithm, i.e., $k$-means, as an example, we propose four types of Feature Incremental Clustering (FIC) algorithms corresponding to different situations of data access: Feature Tailoring (FT), Data Reconstruction (DR), Data Adaptation (DA), and Model Reuse (MR), abbreviated as FIC-FT, FIC-DR, FIC-DA, and FIC-MR. Subsequently, we offer a detailed analysis of the generalization error bounds for these four algorithms and highlight the critical factors influencing these bounds, such as the amounts of training data, the complexity of the hypothesis space, the quality of pre-trained models, and the discrepancy of the reconstruction feature distribution. The numerical experiments show the effectiveness of the proposed algorithms, particularly in their application to activity recognition clustering tasks.

**선정 근거**
특성 증분 클러스터링 알고리즘은 스포츠 동작 분석과 하이라이트 감지에 직접 적용 가능합니다.

**활용 인사이트**
FIC 알고리즘을 사용하여 실시간으로 스포츠 동작을 분류하고 중요한 순간을 자동으로 감지하여 하이라이트 영상을 생성할 수 있습니다.

## 10위: Optimal Memory Encoding Through Fluctuation-Response Structure

- arXiv: http://arxiv.org/abs/2603.21666v1
- PDF: https://arxiv.org/pdf/2603.21666v1
- 발행일: 2026-03-23
- 카테고리: cs.NE
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
Physical reservoir computing exploits the intrinsic dynamics of physical systems for information processing, while keeping the internal dynamics fixed and training only linear readouts; yet the role of input encoding remains poorly understood. We show that optimal input encoding is a geometric problem governed by the system's fluctuation-response structure. By measuring steady-state fluctuations and linear response, we derive an analytical criterion for the input direction that maximizes task-specific linear memory under a fixed power constraint, termed Response-based Optimal Memory Encoding (ROME). Backpropagation-based encoder optimization is shown to be equivalent to ROME, revealing a trade-off between task-dependent feature mixing and intrinsic noise. We apply ROME to various reservoir platforms, including spin-wave waveguides and spiking neural networks, demonstrating effective encoder design across physical and neuromorphic reservoirs, even in non-differentiable systems.

**선정 근거**
물리적 리저버 컴퓨팅과 스파이킹 신경망은 엣지 디바이스의 영상 처리 효율성을 향상시키는 데 적용 가능합니다.

**활용 인사이트**
ROME 기법을 적용하여 입력 인코딩을 최적화하고, 고정된 전력 제약 내에서 엣지 디바이스의 선형 메모리를 최대화할 수 있습니다.

## 11위: AdaEdit: Adaptive Temporal and Channel Modulation for Flow-Based Image Editing

- arXiv: http://arxiv.org/abs/2603.21615v1
- PDF: https://arxiv.org/pdf/2603.21615v1
- 코드: https://github.com/leeguandong/AdaEdit
- 발행일: 2026-03-23
- 카테고리: cs.CV
- 점수: final 86.4 (llm_adjusted:83 = base:80 + bonus:+3)
- 플래그: 코드 공개

**개요**
Inversion-based image editing in flow matching models has emerged as a powerful paradigm for training-free, text-guided image manipulation. A central challenge in this paradigm is the injection dilemma: injecting source features during denoising preserves the background of the original image but simultaneously suppresses the model's ability to synthesize edited content. Existing methods address this with fixed injection strategies -- binary on/off temporal schedules, uniform spatial mixing ratios, and channel-agnostic latent perturbation -- that ignore the inherently heterogeneous nature of injection demand across both the temporal and channel dimensions. In this paper, we present AdaEdit, a training-free adaptive editing framework that resolves this dilemma through two complementary innovations. First, we propose a Progressive Injection Schedule that replaces hard binary cutoffs with continuous decay functions (sigmoid, cosine, or linear), enabling a smooth transition from source-feature preservation to target-feature generation and eliminating feature discontinuity artifacts. Second, we introduce Channel-Selective Latent Perturbation, which estimates per-channel importance based on the distributional gap between the inverted and random latents and applies differentiated perturbation strengths accordingly -- strongly perturbing edit-relevant channels while preserving structure-encoding channels. Extensive experiments on the PIE-Bench benchmark (700 images, 10 editing types) demonstrate that AdaEdit achieves an 8.7% reduction in LPIPS, a 2.6% improvement in SSIM, and a 2.3% improvement in PSNR over strong baselines, while maintaining competitive CLIP similarity. AdaEdit is fully plug-and-play and compatible with multiple ODE solvers including Euler, RF-Solver, and FireFlow. Code is available at https://github.com/leeguandong/AdaEdit

**선정 근거**
스포츠 영상 하이라이트 제작에 직접 적용 가능한 이미지 편집 기술

**활용 인사이트**
적응적 시간 및 채널 변조로 경기 장면을 자연스럽게 보정하고 주요 순간 강조

## 12위: In-network Attack Detection with Federated Deep Learning in IoT Networks: Real Implementation and Analysis

- arXiv: http://arxiv.org/abs/2603.21596v1
- PDF: https://arxiv.org/pdf/2603.21596v1
- 발행일: 2026-03-23
- 카테고리: cs.LG, cs.CR
- 점수: final 84.0 (llm_adjusted:80 = base:70 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
The rapid expansion of the Internet of Things (IoT) and its integration with backbone networks have heightened the risk of security breaches. Traditional centralized approaches to anomaly detection, which require transferring large volumes of data to central servers, suffer from privacy, scalability, and latency limitations. This paper proposes a lightweight autoencoder-based anomaly detection framework designed for deployment on resource-constrained edge devices, enabling real-time detection while minimizing data transfer and preserving privacy. Federated learning is employed to train models collaboratively across distributed devices, where local training occurs on edge nodes and only model weights are aggregated at a central server. A real-world IoT testbed using Raspberry Pi sensor nodes was developed to collect normal and attack traffic data. The proposed federated anomaly detection system, implemented and evaluated on the testbed, demonstrates its effectiveness in accurately identifying network attacks. The communication overhead was reduced significantly while achieving comparable performance to the centralized method.

**선정 근거**
Federated learning for IoT edge devices with real-time processing, applicable to edge computing aspects of the project

## 13위: One Model, Two Markets: Bid-Aware Generative Recommendation

- arXiv: http://arxiv.org/abs/2603.22231v1
- PDF: https://arxiv.org/pdf/2603.22231v1
- 발행일: 2026-03-23
- 카테고리: cs.IR, cs.AI, cs.GT, cs.LG
- 점수: final 84.0 (llm_adjusted:80 = base:75 + bonus:+5)
- 플래그: 실시간

**개요**
Generative Recommender Systems using semantic ids, such as TIGER (Rajput et al., 2023), have emerged as a widely adopted competitive paradigm in sequential recommendation. However, existing architectures are designed solely for semantic retrieval and do not address concerns such as monetization via ad revenue and incorporation of bids for commercial retrieval. We propose GEM-Rec, a unified framework that integrates commercial relevance and monetization objectives directly into the generative sequence. We introduce control tokens to decouple the decision of whether to show an ad from which item to show. This allows the model to learn valid placement patterns directly from interaction logs, which inherently reflect past successful ad placements. Complementing this, we devise a Bid-Aware Decoding mechanism that handles real-time pricing, injecting bids directly into the inference process to steer the generation toward high-value items. We prove that this approach guarantees allocation monotonicity, ensuring that higher bids weakly increase an ad's likelihood of being shown without requiring model retraining. Experiments demonstrate that GEM-Rec allows platforms to dynamically optimize for semantic relevance and platform revenue.

**선정 근거**
GEM-Rec 프레임워크는 광고 수익 모델을 통합하여 스포츠 콘텐츠 플랫폼의 수익화 전략에 직접 적용 가능

**활용 인사이트**
스포츠 하이라이트 영상과 광고를 결합한 제어 토큰을 활용해 사용자 참여도와 광고 수익을 동시에 최적화

## 14위: SHARP: Spectrum-aware Highly-dynamic Adaptation for Resolution Promotion in Remote Sensing Synthesis

- arXiv: http://arxiv.org/abs/2603.21783v1
- PDF: https://arxiv.org/pdf/2603.21783v1
- 코드: https://github.com/bxuanz/SHARP
- 발행일: 2026-03-23
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:75 + bonus:+3)
- 플래그: 코드 공개

**개요**
Text-to-image generation powered by Diffusion Transformers (DiTs) has made remarkable strides, yet remote sensing (RS) synthesis lags behind due to two barriers: the absence of a domain-specialized DiT prior and the prohibitive cost of training at the large resolutions that RS applications demand. Training-free resolution promotion via Rotary Position Embedding (RoPE) rescaling offers a practical remedy, but every existing method applies a static positional scaling rule throughout the denoising process. This uniform compression is particularly harmful for RS imagery, whose substantially denser medium- and high-frequency energy encodes the fine structures critical for aerial-scene realism, such as vehicles, building contours, and road markings. Addressing both challenges requires a domain-specialized generative prior coupled with a denoising-aware positional adaptation strategy. To this end, we fine-tune FLUX on over 100,000 curated RS images to build a strong domain prior (RS-FLUX), and propose Spectrum-aware Highly-dynamic Adaptation for Resolution Promotion (SHARP), a training-free method that introduces a rational fractional time schedule k_rs(t) into RoPE. SHARP applies strong positional promotion during the early layout-formation stage and progressively relaxes it during detail recovery, aligning extrapolation strength with the frequency-progressive nature of diffusion denoising. Its resolution-agnostic formulation further enables robust multi-scale generation from a single set of hyperparameters. Extensive experiments across six square and rectangular resolutions show that SHARP consistently outperforms all training-free baselines on CLIP Score, Aesthetic Score, and HPSv2, with widening margins at more aggressive extrapolation factors and negligible computational overhead. Code and weights are available at https://github.com/bxuanz/SHARP.

**선정 근거**
스포츠 영상 품질 향상을 위한 고해상도 기술 적용 가능

**활용 인사이트**
스펙트럼 인지적 적응으로 경기 영상의 세부 사항을 선명하게 복원

## 15위: ALADIN:Attribute-Language Distillation Network for Person Re-Identification

- arXiv: http://arxiv.org/abs/2603.21482v1
- PDF: https://arxiv.org/pdf/2603.21482v1
- 발행일: 2026-03-23
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:70 + bonus:+5)
- 플래그: 엣지

**개요**
Recent vision-language models such as CLIP provide strong cross-modal alignment, but current CLIP-guided ReID pipelines rely on global features and fixed prompts. This limits their ability to capture fine-grained attribute cues and adapt to diverse appearances. We propose ALADIN, an attribute-language distillation network that distills knowledge from a frozen CLIP teacher to a lightweight ReID student. ALADIN introduces fine-grained attribute-local alignment to establish adaptive text-visual correspondence and robust representation learning. A Scene-Aware Prompt Generator produces image-specific soft prompts to facilitate adaptive alignment. Attribute-local distillation enforces consistency between textual attributes and local visual features, significantly enhancing robustness under occlusions. Furthermore, we employ cross-modal contrastive and relation distillation to preserve the inherent structural relationships among attributes. To provide precise supervision, we leverage Multimodal LLMs to generate structured attribute descriptions, which are then converted into localized attention maps via CLIP. At inference, only the student is used. Experiments on Market-1501, DukeMTMC-reID, and MSMT17 show improvements over CNN-, Transformer-, and CLIP-based methods, with better generalization and interpretability.

**선정 근거**
선수 추적 및 분석을 위한 인물 재식별 시스템 직접 적용 가능

**활용 인사이트**
다중 모달 학습으로 선수별 특징을 정확히 파악하고 움직임 추적

## 16위: DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution

- arXiv: http://arxiv.org/abs/2603.22271v1
- PDF: https://arxiv.org/pdf/2603.22271v1
- 발행일: 2026-03-23
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Diffusion-based video super-resolution (VSR) has recently achieved remarkable fidelity but still suffers from prohibitive sampling costs. While distribution matching distillation (DMD) can accelerate diffusion models toward one-step generation, directly applying it to VSR often results in training instability alongside degraded and insufficient supervision. To address these issues, we propose DUO-VSR, a three-stage framework built upon a Dual-Stream Distillation strategy that unifies distribution matching and adversarial supervision for one-step VSR. Firstly, a Progressive Guided Distillation Initialization is employed to stabilize subsequent training through trajectory-preserving distillation. Next, the Dual-Stream Distillation jointly optimizes the DMD and Real-Fake Score Feature GAN (RFS-GAN) streams, with the latter providing complementary adversarial supervision leveraging discriminative features from both real and fake score models. Finally, a Preference-Guided Refinement stage further aligns the student with perceptual quality preferences. Extensive experiments demonstrate that DUO-VSR achieves superior visual quality and efficiency over previous one-step VSR approaches.

**선정 근거**
비디오 슈퍼-해상도를 위한 이중-스트림 증류 프레임워크로 스포츠 영상 보정 및 편집과 직접적인 관련성 있음

**활용 인사이트**
스포츠 경기 영상의 해상도를 향상시키고 실시간으로 주요 장면을 보정하여 고품질 하이라이트 영상 제작 가능

## 17위: A Framework for Closed-Loop Robotic Assembly, Alignment and Self-Recovery of Precision Optical Systems

- arXiv: http://arxiv.org/abs/2603.21496v1
- PDF: https://arxiv.org/pdf/2603.21496v1
- 발행일: 2026-03-23
- 카테고리: cs.RO, cs.AI, physics.optics
- 점수: final 80.0 (llm_adjusted:75 = base:70 + bonus:+5)
- 플래그: 실시간

**개요**
Robotic automation has transformed scientific workflows in domains such as chemistry and materials science, yet free-space optics, which is a high precision domain, remains largely manual. Optical systems impose strict spatial and angular tolerances, and their performance is governed by tightly coupled physical parameters, making generalizable automation particularly challenging. In this work, we present a robotics framework for the autonomous construction, alignment, and maintenance of precision optical systems. Our approach integrates hierarchical computer vision systems, optimization routines, and custom-built tools to achieve this functionality. As a representative demonstration, we perform the fully autonomous construction of a tabletop laser cavity from randomly distributed components. The system performs several tasks such as laser beam centering, spatial alignment of multiple beams, resonator alignment, laser mode selection, and self-recovery from induced misalignment and disturbances. By achieving closed-loop autonomy for highly sensitive optical systems, this work establishes a foundation for autonomous optical experiments for applications across technical domains.

**선정 근거**
Computer vision and optimization methodologies relevant to video analysis but specifically for optical systems

## 18위: Mapping Travel Experience in Public Transport: Real-Time Evidence and Spatial Analysis in Hamburg

- arXiv: http://arxiv.org/abs/2603.21763v1
- PDF: https://arxiv.org/pdf/2603.21763v1
- 발행일: 2026-03-23
- 카테고리: cs.HC
- 점수: final 80.0 (llm_adjusted:75 = base:65 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Shifting travel from private cars to public transport is critical for meeting climate and related mobility goals, yet passengers will only choose transit if it offers a consistently positive experience. Previous studies of passenger satisfaction have largely relied on retrospective surveys, which overlook the dynamic and spatially differentiated nature of travel experience. This paper introduces a novel combination of real-time experience sampling and spatial hot spot analysis to capture and map where public transport users report consistently positive or negative experiences.   Data were collected from 239 participants in Hamburg between March and September 2025. Using a smartphone application, travelers reported their momentary journey experience every five minutes during everyday trips, yielding over 21,000 in-situ evaluations. These geo-referenced data were analyzed with the Getis-Ord $Gi^{*}$ statistic to detect significant clusters of positive and negative travel experience. The analysis identified distinct hot and cold spots of travel experience across the network. Cold spots were shaped by heterogeneous problems, ranging from predominantly delay-dominated to overcrowding or socially stressful locations. In contrast, hot spots emerged through different pathways, including comfort-oriented, time-efficient or context-driven environments.   The findings highlight three contributions. First, cold spots are not uniform but reflect specific local constellations of problems, requiring targeted interventions. Second, hot spots illustrate multiple success models that can serve as benchmarks for replication. Third, this study demonstrates the value of combining dynamic high-resolution sampling with spatial statistics to guide more effective and place-specific improvements in public transport.

**선정 근거**
실시간 데이터 수집 및 공간 분석 방법론이 스포츠 경기 데이터 수집과 분석에 직접적으로 적용 가능하며, 경험 매핑 기법을 통해 선수들의 움직임 패턴과 경기 전략 분석에 활용 가능

**활용 인사이트**
Getis-Ord Gi* 통계 기반의 핫스팟 분석을 적용하여 경기 중 중요한 순간을 자동으로 식별하고, 실시간 경험 수집 방식을 활용하여 선수 및 팬들의 반응 데이터를 수집하여 플랫폼 콘텐츠 개선에 활용

## 19위: λ-GELU: Learning Gating Hardness for Controlled ReLU-ization in Deep Networks

- arXiv: http://arxiv.org/abs/2603.21991v1
- PDF: https://arxiv.org/pdf/2603.21991v1
- 발행일: 2026-03-23
- 카테고리: cs.LG, cs.AI
- 점수: final 80.0 (llm_adjusted:75 = base:70 + bonus:+5)
- 플래그: 엣지

**개요**
Gaussian Error Linear Unit (GELU) is a widely used smooth alternative to Rectifier Linear Unit (ReLU), yet many deployment, compression, and analysis toolchains are most naturally expressed for piecewise-linear (ReLU-type) networks. We study a hardness-parameterized formulation of GELU, f(x;λ)=xΦ(λ x), where Φ is the Gaussian CDF and λ \in [1, infty) controls gate sharpness, with the goal of turning smooth gated training into a controlled path toward ReLU-compatible models. Learning λ is non-trivial: naive updates yield unstable dynamics and effective gradient attenuation, so we introduce a constrained reparameterization and an optimizer-aware update scheme.   Empirically, across a diverse set of model--dataset pairs spanning MLPs, CNNs, and Transformers, we observe structured layerwise hardness profiles and assess their robustness under different initializations. We further study a deterministic ReLU-ization strategy in which the learned gates are progressively hardened toward a principled target, enabling a post-training substitution of λ-GELU by ReLU with reduced disruption. Overall, λ-GELU provides a minimal and interpretable knob to profile and control gating hardness, bridging smooth training with ReLU-centric downstream pipelines.

**선정 근거**
신경망 최적화 기술로 rk3588 같은 엣지 디바이스에서 효율적인 AI 모델 배포에 적용 가능

**활용 인사이트**
경량화된 모델 구조를 통해 실시간 영상 처리 및 자세 분석을 위한 계산 효율성 향상

## 20위: MIHT: A Hoeffding Tree for Time Series Classification using Multiple Instance Learning

- arXiv: http://arxiv.org/abs/2603.22074v1
- PDF: https://arxiv.org/pdf/2603.22074v1
- 발행일: 2026-03-23
- 카테고리: cs.LG
- 점수: final 72.0 (llm_adjusted:65 = base:60 + bonus:+5)
- 플래그: 엣지

**개요**
Due to the prevalence of temporal data and its inherent dependencies in many real-world problems, time series classification is of paramount importance in various domains. However, existing models often struggle with series of variable length or high dimensionality. This paper introduces the MIHT (Multi-instance Hoeffding Tree) algorithm, an efficient model that uses multi-instance learning to classify multivariate and variable-length time series while providing interpretable results. The algorithm uses a novel representation of time series as "bags of subseries," together with an optimization process based on incremental decision trees that distinguish relevant parts of the series from noise. This methodology extracts the underlying concept of series with multiple variables and variable lengths. The generated decision tree is a compact, white-box representation of the series' concept, providing interpretability insights into the most relevant variables and segments of the series. Experimental results demonstrate MIHT's superiority, as it outperforms 11 state-of-the-art time series classification models on 28 public datasets, including high-dimensional ones. MIHT offers enhanced accuracy and interpretability, making it a promising solution for handling complex, dynamic time series data.

**선정 근거**
시계열 분류 알고리즘으로 스포츠 동작 분석에 간접적으로 적용 가능

**활용 인사이트**
다변량 시계열 데이터를 분석하여 선수들의 동작 패턴을 식별하고 전략 분석에 활용 가능

## 21위: FGIM: a Fast Graph-based Indexes Merging Framework for Approximate Nearest Neighbor Search

- arXiv: http://arxiv.org/abs/2603.21710v1
- PDF: https://arxiv.org/pdf/2603.21710v1
- 발행일: 2026-03-23
- 카테고리: cs.DB
- 점수: final 68.0 (llm_adjusted:60 = base:60 + bonus:+0)

**개요**
As the state-of-the-art methods for high-dimensional data retrieval, Approximate Nearest Neighbor Search (ANNS) approaches with graph-based indexes have attracted increasing attention and play a crucial role in many real-world applications, e.g., retrieval-augmented generation (RAG) and recommendation systems. Unlike the extensive works focused on designing efficient graph-based ANNS methods, this paper delves into merging multiple existing graph-based indexes into a single one, which is also crucial in many real-world scenarios (e.g., cluster consolidation in distributed systems and read-write contention in real-time vector databases). We propose a Fast Graph-based Indexes Merging (FGIM) framework with three core techniques: (1) Proximity Graphs (PGs) to $k$ Nearest Neighbor Graph ($k$-NNG) transformation used to extract potential candidate neighbors from input graph-based indexes through cross-querying, (2) $k$-NNG refinement designed to identify overlooked high-quality neighbors and maintain graph connectivity, and (3) $k$-NNG to PG transformation aimed at improving graph navigability and enhancing search performance. Then, we integrate our FGIM framework with the state-of-the-art ANNS method, HNSW, and other existing mainstream graph-based methods to demonstrate its generality and merging efficiency. Extensive experiments on six real-world datasets show that our FGIM framework is applicable to various mainstream graph-based ANNS methods, achieves up to 3.5$\times$ speedup over HNSW's incremental construction and an average of 7.9$\times$ speedup for methods without incremental support, while maintaining comparable or superior search performance.

**선정 근거**
고차원 데이터 검색 프레임워크로 스포츠 동작 패턴 분석에 적용 가능

**활용 인사이트**
경기 영상에서 특정 동작 패턴을 빠르게 검색하고 분류하는 데 활용

## 22위: Mind over Space: Can Multimodal Large Language Models Mentally Navigate?

- arXiv: http://arxiv.org/abs/2603.21577v1
- PDF: https://arxiv.org/pdf/2603.21577v1
- 발행일: 2026-03-23
- 카테고리: cs.AI
- 점수: final 66.4 (llm_adjusted:58 = base:58 + bonus:+0)

**개요**
Despite the widespread adoption of MLLMs in embodied agents, their capabilities remain largely confined to reactive planning from immediate observations, consistently failing in spatial reasoning across extensive spatiotemporal scales. Cognitive science reveals that Biological Intelligence (BI) thrives on "mental navigation": the strategic construction of spatial representations from experience and the subsequent mental simulation of paths prior to action. To bridge the gap between AI and BI, we introduce Video2Mental, a pioneering benchmark for evaluating the mental navigation capabilities of MLLMs. The task requires constructing hierarchical cognitive maps from long egocentric videos and generating landmark-based path plans step by step, with planning accuracy verified through simulator-based physical interaction. Our benchmarking results reveal that mental navigation capability does not naturally emerge from standard pre-training. Frontier MLLMs struggle profoundly with zero-shot structured spatial representation, and their planning accuracy decays precipitously over extended horizons. To overcome this, we propose \textbf{NavMind}, a reasoning model that internalizes mental navigation using explicit, fine-grained cognitive maps as learnable intermediate representations. Through a difficulty-stratified progressive supervised fine-tuning paradigm, NavMind effectively bridges the gap between raw perception and structured planning. Experiments demonstrate that NavMind achieves superior mental navigation capabilities, significantly outperforming frontier commercial and spatial MLLMs.

**선정 근거**
공간 추론 능력을 활용해 선수들의 움직임 패턴과 전략 분석 가능

**활용 인사이트**
경기장 내 선수들의 위치와 움직임을 인지하고 예측하는 데 적용

## 23위: Is AI Ready for Multimodal Hate Speech Detection? A Comprehensive Dataset and Benchmark Evaluation

- arXiv: http://arxiv.org/abs/2603.21686v1
- PDF: https://arxiv.org/pdf/2603.21686v1
- 코드: https://github.com/mira-ai-lab/M3
- 발행일: 2026-03-23
- 카테고리: cs.MA
- 점수: final 66.4 (llm_adjusted:58 = base:55 + bonus:+3)
- 플래그: 코드 공개

**개요**
Hate speech online targets individuals or groups based on identity attributes and spreads rapidly, posing serious social risks. Memes, which combine images and text, have emerged as a nuanced vehicle for disseminating hate speech, often relying on cultural knowledge for interpretation. However, existing multimodal hate speech datasets suffer from coarse-grained labeling and a lack of integration with surrounding discourse, leading to imprecise and incomplete assessments. To bridge this gap, we propose an agentic annotation framework that coordinates seven specialized agents to generate hierarchical labels and rationales. Based on this framework, we construct M^3 (Multi-platform, Multi-lingual, and Multimodal Meme), a dataset of 2,455 memes collected from X, 4chan, and Weibo, featuring fine-grained hate labels and human-verified rationales. Benchmarking state-of-the-art Multimodal Large Language Models reveals that these models struggle to effectively utilize surrounding post context, which often fails to improve or even degrades detection performance. Our finding highlights the challenges these models face in reasoning over memes embedded in real-world discourse and underscores the need for a context-aware multimodal architecture. Our dataset and code are available at https://github.com/mira-ai-lab/M3.

**선정 근거**
다중 모달 분석 기술과 데이터셋 생성 방법론이 스포츠 콘텐츠 분석 시스템에 적용 가능합니다.

**활용 인사이트**
계층적 라벨링 방식을 스포츠 동작 분류에 적용하고, 주변 맥락을 고려한 분석 모델을 개발할 수 있습니다.

## 24위: SynSym: A Synthetic Data Generation Framework for Psychiatric Symptom Identification

- arXiv: http://arxiv.org/abs/2603.21529v1
- PDF: https://arxiv.org/pdf/2603.21529v1
- 발행일: 2026-03-23
- 카테고리: cs.CL
- 점수: final 66.4 (llm_adjusted:58 = base:58 + bonus:+0)

**개요**
Psychiatric symptom identification on social media aims to infer fine-grained mental health symptoms from user-generated posts, allowing a detailed understanding of users' mental states. However, the construction of large-scale symptom-level datasets remains challenging due to the resource-intensive nature of expert labeling and the lack of standardized annotation guidelines, which in turn limits the generalizability of models to identify diverse symptom expressions from user-generated text. To address these issues, we propose SynSym, a synthetic data generation framework for constructing generalizable datasets for symptom identification. Leveraging large language models (LLMs), SynSym constructs high-quality training samples by (1) expanding each symptom into sub-concepts to enhance the diversity of generated expressions, (2) producing synthetic expressions that reflect psychiatric symptoms in diverse linguistic styles, and (3) composing realistic multi-symptom expressions, informed by clinical co-occurrence patterns. We validate SynSym on three benchmark datasets covering different styles of depressive symptom expression. Experimental results demonstrate that models trained solely on the synthetic data generated by SynSym perform comparably to those trained on real data, and benefit further from additional fine-tuning with real data. These findings underscore the potential of synthetic data as an alternative resource to real-world annotations in psychiatric symptom modeling, and SynSym serves as a practical framework for generating clinically relevant and realistic symptom expressions.

**선정 근거**
합성 데이터 생성 기술로 다양한 스포츠 장면 데이터 효율적으로 생성

**활용 인사이트**
실제 경기 데이터 부족 시 시뮬레이션 데이터로 모델 학습 강화

## 25위: Do World Action Models Generalize Better than VLAs? A Robustness Study

- arXiv: http://arxiv.org/abs/2603.22078v1
- PDF: https://arxiv.org/pdf/2603.22078v1
- 발행일: 2026-03-23
- 카테고리: cs.RO
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
Robot action planning in the real world is challenging as it requires not only understanding the current state of the environment but also predicting how it will evolve in response to actions. Vision-language-action (VLA), which repurpose large-scale vision-language models for robot action generation using action experts, have achieved notable success across a variety of robotic tasks. Nevertheless, their performance remains constrained by the scope of their training data, exhibiting limited generalization to unseen scenarios and vulnerability to diverse contextual perturbations. More recently, world models have been revisited as an alternative to VLAs. These models, referred to as world action models (WAMs), are built upon world models that are trained on large corpora of video data to predict future states. With minor adaptations, their latent representation can be decoded into robot actions. It has been suggested that their explicit dynamic prediction capacity, combined with spatiotemporal priors acquired from web-scale video pretraining, enables WAMs to generalize more effectively than VLAs. In this paper, we conduct a comparative study of prominent state-of-the-art VLA policies and recently released WAMs. We evaluate their performance on the LIBERO-Plus and RoboTwin 2.0-Plus benchmarks under various visual and language perturbations. Our results show that WAMs achieve strong robustness, with LingBot-VA reaching 74.2% success rate on RoboTwin 2.0-Plus and Cosmos-Policy achieving 82.2% on LIBERO-Plus. While VLAs such as $π_{0.5}$ can achieve comparable robustness on certain tasks, they typically require extensive training with diverse robotic datasets and varied learning objectives. Hybrid approaches that partially incorporate video-based dynamic learning exhibit intermediate robustness, highlighting the importance of how video priors are integrated.

**선정 근거**
로봇 동작 계획 모델 관련, 스포츠 촬영/분석과 직접적 연관 없음

## 26위: SafePilot: A Framework for Assuring LLM-enabled Cyber-Physical Systems

- arXiv: http://arxiv.org/abs/2603.21523v1
- PDF: https://arxiv.org/pdf/2603.21523v1
- 발행일: 2026-03-23
- 카테고리: cs.RO, cs.AI
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
Large Language Models (LLMs), deep learning architectures with typically over 10 billion parameters, have recently begun to be integrated into various cyber-physical systems (CPS) such as robotics, industrial automation, and autopilot systems. The abstract knowledge and reasoning capabilities of LLMs are employed for tasks like planning and navigation. However, a significant challenge arises from the tendency of LLMs to produce "hallucinations" - outputs that are coherent yet factually incorrect or contextually unsuitable. This characteristic can lead to undesirable or unsafe actions in the CPS. Therefore, our research focuses on assuring the LLM-enabled CPS by enhancing their critical properties. We propose SafePilot, a novel hierarchical neuro-symbolic framework that provides end-to-end assurance for LLM-enabled CPS according to attribute-based and temporal specifications. Given a task and its specification, SafePilot first invokes a hierarchical planner with a discriminator that assesses task complexity. If the task is deemed manageable, it is passed directly to an LLM-based task planner with built-in verification. Otherwise, the hierarchical planner applies a divide-and-conquer strategy, decomposing the task into sub-tasks, each of which is individually planned and later merged into a final solution. The LLM-based task planner translates natural language constraints into formal specifications and verifies the LLM's output against them. If violations are detected, it identifies the flaw, adjusts the prompt accordingly, and re-invokes the LLM. This iterative process continues until a valid plan is produced or a predefined limit is reached. Our framework supports LLM-enabled CPS with both attribute-based and temporal constraints. Its effectiveness and adaptability are demonstrated through two illustrative case studies.

**선정 근거**
AI framework for cyber-physical systems, not specifically for sports video analysis

## 27위: Proximal Policy Optimization in Path Space: A Schrödinger Bridge Perspective

- arXiv: http://arxiv.org/abs/2603.21621v1
- PDF: https://arxiv.org/pdf/2603.21621v1
- 발행일: 2026-03-23
- 카테고리: cs.LG
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
On-policy reinforcement learning with generative policies is promising but remains underexplored. A central challenge is that proximal policy optimization (PPO) is traditionally formulated in terms of action-space probability ratios, whereas diffusion- and flow-based policies are more naturally represented as trajectory-level generative processes. In this work, we propose GSB-PPO, a path-space formulation of generative PPO inspired by the Generalized Schrödinger Bridge (GSB). Our framework lifts PPO-style proximal updates from terminal actions to full generation trajectories, yielding a unified view of on-policy optimization for generative policies. Within this framework, we develop two concrete objectives: a clipping-based objective, GSB-PPO-Clip, and a penalty-based objective, GSB-PPO-Penalty. Experimental results show that while both objectives are compatible with on-policy training, the penalty formulation consistently delivers better stability and performance than the clipping counterpart. Overall, our results highlight path-space proximal regularization as an effective principle for training generative policies with PPO.

**선정 근거**
경기 동작 분석과 하이라이트 영상 생성에 직접적으로 적용 가능한 경로 공간 최적화 접근법을 제시하여 스포츠 영상 처리에 유용합니다.

**활용 인사이트**
GSB-PPO 프레임워크를 활용해 경기 동작의 궤적을 분석하고, 이를 기반으로 최적의 하이라이트 영상을 생성하는 모델을 구축할 수 있습니다.

## 28위: Cluster-Specific Predictive Modeling: A Scalable Solution for Resource-Constrained Wi-Fi Controllers

- arXiv: http://arxiv.org/abs/2603.21778v1
- PDF: https://arxiv.org/pdf/2603.21778v1
- 발행일: 2026-03-23
- 카테고리: eess.SP, cs.LG
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
This manuscript presents a comprehensive analysis of predictive modeling optimization in managed Wi-Fi networks through the integration of clustering algorithms and model evaluation techniques. The study addresses the challenges of deploying forecasting algorithms in large-scale environments managed by a central controller constrained by memory and computational resources. Feature-based clustering, supported by Principal Component Analysis (PCA) and advanced feature engineering, is employed to group time series data based on shared characteristics, enabling the development of cluster-specific predictive models. Comparative evaluations between global models (GMs) and cluster-specific models demonstrate that cluster-specific models consistently achieve superior accuracy in terms of Mean Absolute Error (MAE) values in high-activity clusters. The trade-offs between model complexity (and accuracy) and resource utilization are analyzed, highlighting the scalability of tailored modeling approaches. The findings advocate for adaptive network management strategies that optimize resource allocation through selective model deployment, enhance predictive accuracy, and ensure scalable operations in large-scale, centrally managed Wi-Fi environments.

**선정 근거**
Clustering algorithms for resource-constrained systems could be indirectly applicable to edge device processing for sports analysis

## 29위: A Latent Representation Learning Framework for Hyperspectral Image Emulation in Remote Sensing

- arXiv: http://arxiv.org/abs/2603.21911v1
- PDF: https://arxiv.org/pdf/2603.21911v1
- 발행일: 2026-03-23
- 카테고리: cs.CV, cs.LG, eess.IV
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
Synthetic hyperspectral image (HSI) generation is essential for large-scale simulation, algorithm development, and mission design, yet traditional radiative transfer models remain computationally expensive and often limited to spectrum-level outputs. In this work, we propose a latent representation-based framework for hyperspectral emulation that learns a latent generative representation of hyperspectral data. The proposed approach supports both spectrum-level and spatial-spectral emulation and can be trained either in a direct one-step formulation or in a two-step strategy that couples variational autoencoder (VAE) pretraining with parameter-to-latent interpolation. Experiments on PROSAIL-simulated vegetation data and Sentinel-3 OLCI imagery demonstrate that the method outperforms classical regression-based emulators in reconstruction accuracy, spectral fidelity, and robustness to real-world spatial variability. We further show that emulated HSIs preserve performance in downstream biophysical parameter retrieval, highlighting the practical relevance of emulated data for remote sensing applications.

**선정 근거**
잠재 표현 학습은 스포츠 영상 처리에 적합하며, rk3588 엣지 디바이스에서 실시간으로 하이라이트를 생성하고 보정하는 데 효과적입니다.

**활용 인사이트**
스포츠 영상의 잠재 공간을 학습하여 동작 패턴을 분석하고, 자동으로 하이라이트 장면을 추출하며, 영상 품질을 향상시키는 데 적용할 수 있습니다.

## 30위: IF-CPS: Influence Functions for Cyber-Physical Systems -- A Unified Framework for Diagnosis, Curation, and Safety Attribution

- arXiv: http://arxiv.org/abs/2603.21543v1
- PDF: https://arxiv.org/pdf/2603.21543v1
- 발행일: 2026-03-23
- 카테고리: eess.SY
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
Neural network controllers trained via behavior cloning are increasingly deployed in cyber-physical systems (CPS), yet practitioners lack tools to trace controller failures back to training data. Existing data attribution methods assume i.i.d.\ data and standard loss targets, ignoring CPS-specific properties: closed-loop dynamics, safety constraints, and temporal trajectory structure. We propose IF-CPS, a modular influence function framework with three CPS-adapted variants: safety influence (attributing constraint violations), trajectory influence (temporal discounting over trajectories), and propagated influence (tracing effects through plant dynamics). We evaluate IF-CPS on six benchmarks across diagnosis, curation, and safety attribution tasks. IF-CPS improves over standard influence functions in the majority of settings, achieving AUROC $1.00$ in Pendulum (5-10\% poisoning), $0.92$ vs.\ $0.50$ in HVAC (10\%), and the strongest constraint-boundary correlation (Spearman $ρ= 0.55$ in Pendulum).

**선정 근거**
Cyber-physical system diagnosis framework indirectly applicable to AI camera analysis

## 31위: The Semantic Ladder: A Framework for Progressive Formalization of Natural Language Content for Knowledge Graphs and AI Systems

- arXiv: http://arxiv.org/abs/2603.22136v1 | 2026-03-23 | final 64.0

Semantic data and knowledge infrastructures must reconcile two fundamentally different forms of representation: natural language, in which most knowledge is created and communicated, and formal semantic models, which enable machine-actionable integration, interoperability, and reasoning. Bridging this gap remains a central challenge, particularly when full semantic formalization is required at the point of data entry.

-> Semantic formalization framework potentially applicable to sports knowledge representation

## 32위: Input Convex Encoder-Only Transformer for Fast and Gradient-Stable MPC in Building Demand Response

- arXiv: http://arxiv.org/abs/2603.22095v1 | 2026-03-23 | final 64.0

Learning-based Model Predictive Control (MPC) has emerged as a powerful strategy for building demand response. However, its practical deployment is often hindered by the non-convex optimization problems induced by standard neural network models.

-> Transformer architecture for real-time processing could be adapted for video analysis

## 33위: Efficient Failure Management for Multi-Agent Systems with Reasoning Trace Representation

- arXiv: http://arxiv.org/abs/2603.21522v1 | 2026-03-23 | final 64.0

Large Language Models (LLM)-based Multi-Agent Systems (MASs) have emerged as a new paradigm in software system design, increasingly demonstrating strong reasoning and collaboration capabilities. As these systems become more complex and autonomous, effective failure management is essential to ensure reliability and availability.

-> Multi-agent system approach with real-time processing could coordinate camera systems

## 34위: SARe: Structure-Aware Large-Scale 3D Fragment Reassembly

- arXiv: http://arxiv.org/abs/2603.21611v1 | 2026-03-23 | final 64.0

3D fragment reassembly aims to recover the rigid poses of unordered fragment point clouds or meshes in a common object coordinate system to reconstruct the complete shape. The problem becomes particularly challenging as the number of fragments grows, since the target shape is unknown and fragments provide weak semantic cues.

-> 3D 조각 재조립 기술이 스포츠 분석에 간접적으로 관련

## 35위: Nonlinear Control Synchronization Method for Fractional-order Time Derivatives Chaotic Systems

- arXiv: http://arxiv.org/abs/2603.21747v1 | 2026-03-23 | final 64.0

"Synchronization of two dynamical systems" is the term used to describe the phenomenon when two or more systems gradually change their states or behaviors to become similar or identical. This can happen in a lot of fields, such as physics, engineering, biology, and economics.

-> Theoretical systems analysis that could indirectly relate to sports movement analysis

## 36위: Select, Label, Evaluate: Active Testing in NLP

- arXiv: http://arxiv.org/abs/2603.21840v1 | 2026-03-23 | final 60.0

Human annotation cost and time remain significant bottlenecks in Natural Language Processing (NLP), with test data annotation being particularly expensive due to the stringent requirement for low-error and high-quality labels necessary for reliable model evaluation. Traditional approaches require annotating entire test sets, leading to substantial resource requirements.

-> Paper presents active learning framework for NLP tasks, conceptually related to sample selection but not directly applicable to sports video analysis

## 37위: Kolmogorov Complexity Bounds for LLM Steganography and a Perplexity-Based Detection Proxy

- arXiv: http://arxiv.org/abs/2603.21567v1 | 2026-03-23 | final 56.0

Large language models can rewrite text to embed hidden payloads while preserving surface-level meaning, a capability that opens covert channels between cooperating AI systems and poses challenges for alignment monitoring. We study the information-theoretic cost of such embedding.

-> Kolmogorov complexity bounds and perplexity-based detection could be applicable to video processing and compression

## 38위: Distributionally robust optimization for recommendation selection

- arXiv: http://arxiv.org/abs/2603.22090v1 | 2026-03-23 | final 52.0

Recommender systems play an essential role in online services by providing personalized item lists to support users' decision-making processes. While collaborative filtering methods can achieve high accuracy, it is crucial to consider not only accuracy but also the diversity of recommended items to improve user satisfaction.

-> Recommendation systems could be applicable to the content sharing platform component

---

## 다시 보기

### SpiderCam: Low-Power Snapshot Depth from Differential Defocus (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17910v1
- 점수: final 100.0

We introduce SpiderCam, an FPGA-based snapshot depth-from-defocus camera which produces 480x400 sparse depth maps in real-time at 32.5 FPS over a working range of 52 cm while consuming 624 mW of power in total. SpiderCam comprises a custom camera that simultaneously captures two differently focused images of the same scene, processed with a SystemVerilog implementation of depth from differential defocus (DfDD) on a low-power FPGA. To achieve state-of-the-art power consumption, we present algorithmic improvements to DfDD that overcome challenges caused by low-power sensors, and design a memory-local implementation for streaming depth computation on a device that is too small to store even a single image pair. We report the first sub-Watt total power measurement for passive FPGA-based 3D cameras in the literature.

-> 스포츠 촬영을 위한 저전력 엣지 디바이스에 직접 적용 가능한 깊이 맵 생성 기술

### HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18850v1
- 점수: final 100.0

Video question answering (VQA) with vision-language models (VLMs) depends critically on which frames are selected from the input video, yet most systems rely on uniform or heuristic sampling that cannot be optimized for downstream answering quality. We introduce \textbf{HORNet}, a lightweight frame selection policy trained with Group Relative Policy Optimization (GRPO) to learn which frames a frozen VLM needs to answer questions correctly. With fewer than 1M trainable parameters, HORNet reduces input frames by up to 99\% and VLM processing time by up to 93\%, while improving answer quality on short-form benchmarks (+1.7\% F1 on MSVD-QA) and achieving strong performance on temporal reasoning tasks (+7.3 points over uniform sampling on NExT-QA). We formalize this as Select Any Frames (SAF), a task that decouples visual input curation from VLM reasoning, and show that GRPO-trained selection generalizes better out-of-distribution than supervised and PPO alternatives. HORNet's policy further transfers across VLM answerers without retraining, yielding an additional 8.5\% relative gain when paired with a stronger model. Evaluated across six benchmarks spanning 341,877 QA pairs and 114.2 hours of video, our results demonstrate that optimizing \emph{what} a VLM sees is a practical and complementary alternative to optimizing what it generates while improving efficiency. Code is available at https://github.com/ostadabbas/HORNet.

-> 경량 프레임 선택 기술은 스포츠 영상 하이라이트 생성에 직접적으로 적용 가능합니다

### DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.19172v1
- 점수: final 100.0

Despite the computational efficiency of MoE models, the excessive memory footprint and I/O overhead inherent in multi-expert architectures pose formidable challenges for real-time inference on resource-constrained edge platforms. While existing static methods struggle with a rigid latency-accuracy trade-off, we observe that expert importance is highly skewed and depth-dependent. Motivated by these insights, we propose DyMoE, a dynamic mixed-precision quantization framework designed for high-performance edge inference. Leveraging insights into expert importance skewness and depth-dependent sensitivity, DyMoE introduces: (1) importance-aware prioritization to dynamically quantize experts at runtime; (2) depth-adaptive scheduling to preserve semantic integrity in critical layers; and (3) look-ahead prefetching to overlap I/O stalls. Experimental results on commercial edge hardware show that DyMoE reduces Time-to-First-Token (TTFT) by 3.44x-22.7x and up to a 14.58x speedup in Time-Per-Output-Token (TPOT) compared to state-of-the-art offloading baselines, enabling real-time, accuracy-preserving MoE inference on resource-constrained edge devices.

-> 엣지 디바이스 효율적인 추론 기술이 프로젝트의 엣지 디바이스 구현에 직접적으로 관련됨

### EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18739v1
- 점수: final 98.4

Deploying high-performance dense prediction models on resource-constrained edge devices remains challenging due to strict limits on computation and memory. In practice, lightweight systems for object detection, instance segmentation, and pose estimation are still dominated by CNN-based architectures such as YOLO, while compact Vision Transformers (ViTs) often struggle to achieve similarly strong accuracy efficiency tradeoff, even with large scale pretraining. We argue that this gap is largely due to insufficient task specific representation learning in small scale ViTs, rather than an inherent mismatch between ViTs and edge dense prediction. To address this issue, we introduce EdgeCrafter, a unified compact ViT framework for edge dense prediction centered on ECDet, a detection model built from a distilled compact backbone and an edge-friendly encoder decoder design. On the COCO dataset, ECDet-S achieves 51.7 AP with fewer than 10M parameters using only COCO annotations. For instance segmentation, ECInsSeg achieves performance comparable to RF-DETR while using substantially fewer parameters. For pose estimation, ECPose-X reaches 74.8 AP, significantly outperforming YOLO26Pose-X (71.6 AP) despite the latter's reliance on extensive Objects365 pretraining. These results show that compact ViTs, when paired with task-specialized distillation and edge-aware design, can be a practical and competitive option for edge dense prediction. Code is available at: https://intellindust-ai-lab.github.io/projects/EdgeCrafter/

-> 엣지 기기에서의 포즈 추정 및 객체 탐지 기술이 스포츠 동작 분석에 직접 적용 가능함

### Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.19026v1
- 점수: final 98.4

Recent segmentation methods leveraging Multi-modal Large Language Models (MLLMs) have shown reliable object-level segmentation and enhanced spatial perception. However, almost all previous methods predominantly rely on specialist mask decoders to interpret masks from generated segmentation-related embeddings and visual features, or incorporate multiple additional tokens to assist. This paper aims to investigate whether and how we can unlock segmentation from MLLM itSELF with 1 segmentation Embedding (SELF1E) while achieving competitive results, which eliminates the need for external decoders. To this end, our approach targets the fundamental limitation of resolution reduction in pixel-shuffled image features from MLLMs. First, we retain image features at their original uncompressed resolution, and refill them with residual features extracted from MLLM-processed compressed features, thereby improving feature precision. Subsequently, we integrate pixel-unshuffle operations on image features with and without LLM processing, respectively, to unleash the details of compressed features and amplify the residual features under uncompressed resolution, which further enhances the resolution of refilled features. Moreover, we redesign the attention mask with dual perception pathways, i.e., image-to-image and image-to-segmentation, enabling rich feature interaction between pixels and the segmentation token. Comprehensive experiments across multiple segmentation tasks validate that SELF1E achieves performance competitive with specialist mask decoder-based methods, demonstrating the feasibility of decoder-free segmentation in MLLMs. Project page: https://github.com/ANDYZAQ/SELF1E.

-> 스포츠 영상에서 선수와 객체 식별에 직접적으로 적용 가능한 고급 분할 기술

### Balancing Performance and Fairness in Explainable AI for Anomaly Detection in Distributed Power Plants Monitoring (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18954v1
- 점수: final 96.0

Reliable anomaly detection in distributed power plant monitoring systems is essential for ensuring operational continuity and reducing maintenance costs, particularly in regions where telecom operators heavily rely on diesel generators. However, this task is challenged by extreme class imbalance, lack of interpretability, and potential fairness issues across regional clusters. In this work, we propose a supervised ML framework that integrates ensemble methods (LightGBM, XGBoost, Random Forest, CatBoost, GBDT, AdaBoost) and baseline models (Support Vector Machine, K-Nearrest Neighbors, Multilayer Perceptrons, and Logistic Regression) with advanced resampling techniques (SMOTE with Tomek Links and ENN) to address imbalance in a dataset of diesel generator operations in Cameroon. Interpretability is achieved through SHAP (SHapley Additive exPlanations), while fairness is quantified using the Disparate Impact Ratio (DIR) across operational clusters. We further evaluate model generalization using Maximum Mean Discrepancy (MMD) to capture domain shifts between regions. Experimental results show that ensemble models consistently outperform baselines, with LightGBM achieving an F1-score of 0.99 and minimal bias across clusters (DIR $\approx 0.95$). SHAP analysis highlights fuel consumption rate and runtime per day as dominant predictors, providing actionable insights for operators. Our findings demonstrate that it is possible to balance performance, interpretability, and fairness in anomaly detection, paving the way for more equitable and explainable AI systems in industrial power management. {\color{black} Finally, beyond offline evaluation, we also discuss how the trained models can be deployed in practice for real-time monitoring. We show how containerized services can process in real-time, deliver low-latency predictions, and provide interpretable outputs for operators.

-> 실시간 이상 감지 프레임워크로 엣지 배포 가능성이 있어 스포츠 모니터링 시스템에 적용 가능

### A spatio-temporal graph-based model for team sports analysis (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17471v1
- 점수: final 96.0

Team sports represent complex phenomena characterized by both spatial and temporal dimensions, making their analysis inherently challenging. In this study, we examine team sports as complex systems, specifically focusing on the tactical aspects influenced by external constraints. To this end, we introduce a new generic graph-based model to analyze these phenomena. Specifically, we model a team sport's attacking play as a directed path containing absolute and relative ball carrier-centered spatial information, temporal information, and semantic information. We apply our model to union rugby, aiming to validate two hypotheses regarding the impact of the pedagogy provided by the coach on the one hand, and the effect of the initial positioning of the defensive team on the other hand. Preliminary results from data collected on six-player rugby from several French clubs indicate notable effects of these constraints. The model is intended to be applied to other team sports and to validate additional hypotheses related to team coordination patterns, including upcoming applications in basketball.

-> 팀 스포츠 전략 분석에 직접적으로 적용 가능한 그래프 기반 모델

### Universal Skeleton Understanding via Differentiable Rendering and MLLMs (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18003v1
- 점수: final 94.4

Multimodal large language models (MLLMs) exhibit strong visual-language reasoning, yet remain confined to their native modalities and cannot directly process structured, non-visual data such as human skeletons. Existing methods either compress skeleton dynamics into lossy feature vectors for text alignment, or quantize motion into discrete tokens that generalize poorly across heterogeneous skeleton formats. We present SkeletonLLM, which achieves universal skeleton understanding by translating arbitrary skeleton sequences into the MLLM's native visual modality. At its core is DrAction, a differentiable, format-agnostic renderer that converts skeletal kinematics into compact image sequences. Because the pipeline is end-to-end differentiable, MLLM gradients can directly guide the rendering to produce task-informative visual tokens. To further enhance reasoning capabilities, we introduce a cooperative training strategy: Causal Reasoning Distillation transfers structured, step-by-step reasoning from a teacher model, while Discriminative Finetuning sharpens decision boundaries between confusable actions. SkeletonLLM demonstrates strong generalization on diverse tasks including recognition, captioning, reasoning, and cross-format transfer -- suggesting a viable path for applying MLLMs to non-native modalities. Code will be released upon acceptance.

-> 미분가능 렌더링을 통한 골격 이해로 스포츠 동작 분석에 적합

### Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18671v1
- 점수: final 94.4

Standard deep learning models for image segmentation cannot guarantee topology accuracy, failing to preserve the correct number of connected components or structures. This, in turn, affects the quality of the segmentations and compromises the reliability of the subsequent quantification analyses. Previous works have proposed to enhance topology accuracy with specialized frameworks, architectures, and loss functions. However, these methods are often cumbersome to integrate into existing training pipelines, they are computationally very expensive, or they are restricted to structures with tubular morphology. We present SCNP, an efficient method that improves topology accuracy by penalizing the logits with their poorest-classified neighbor, forcing the model to improve the prediction at the pixels' neighbors before allowing it to improve the pixels themselves. We show the effectiveness of SCNP across 13 datasets, covering different structure morphologies and image modalities, and integrate it into three frameworks for semantic and instance segmentation. Additionally, we show that SCNP can be integrated into several loss functions, making them improve topology accuracy. Our code can be found at https://jmlipman.github.io/SCNP-SameClassNeighborPenalization.

-> 토폴로지 정확도를 개선하는 이미지 분할 방법은 스포츠 분석에서 선수와 객체 구조를 정확하게 유지하는 데 중요합니다.

### Scaling Sim-to-Real Reinforcement Learning for Robot VLAs with Generative 3D Worlds (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18532v1
- 점수: final 93.6

The strong performance of large vision-language models (VLMs) trained with reinforcement learning (RL) has motivated similar approaches for fine-tuning vision-language-action (VLA) models in robotics. Many recent works fine-tune VLAs directly in the real world to avoid addressing the sim-to-real gap. While real-world RL circumvents sim-to-real issues, it inherently limits the generality of the resulting VLA, as scaling scene and object diversity in the physical world is prohibitively difficult. This leads to the paradoxical outcome of transforming a broadly pretrained model into an overfitted, scene-specific policy. Training in simulation can instead provide access to diverse scenes, but designing those scenes is also costly. In this work, we show that VLAs can be RL fine-tuned without sacrificing generality and with reduced labor by leveraging 3D world generative models. Using these models together with a language-driven scene designer, we generate hundreds of diverse interactive scenes containing unique objects and backgrounds, enabling scalable and highly parallel policy learning. Starting from a pretrained imitation baseline, our approach increases simulation success from 9.7% to 79.8% while achieving a 1.25$\times$ speedup in task completion time. We further demonstrate successful sim-to-real transfer enabled by the quality of the generated digital twins together with domain randomization, improving real-world success from 21.7% to 75% and achieving a 1.13$\times$ speedup. Finally, we further highlight the benefits of leveraging the effectively unlimited data from 3D world generative models through an ablation study showing that increasing scene diversity directly improves zero-shot generalization.

-> 비전-언어-액션 모델과 생성적 3D 세계는 스포츠 장면 분석과 시뮬레이션에 적용 가능하여 우리 프로젝트의 핵심 기술이 될 수 있습니다.

### A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.19133v1
- 점수: final 93.6

Recent advancements and widespread adoption of Large Language Models (LLMs) in both industry and academia have catalyzed significant demand for LLM serving. However, traditional cloud services incur high costs, while on-device inference alone faces challenges due to limited resources. Edge-cloud collaboration emerges as a key research direction to combine the strengths of both paradigms, yet efficiently utilizing limited network bandwidth while fully leveraging and balancing the computational capabilities of edge devices and the cloud remains an open problem. To address these challenges, we propose Pipelined Collaborative Speculative Decoding Framework (PicoSpec), a novel, general-purpose, and training-free speculative decoding framework for LLM edge-cloud collaborative inference. We design an asynchronous pipeline that resolves the mutual waiting problem inherent in vanilla speculative decoding within edge collaboration scenarios, which concurrently executes a Small Language Model (SLM) on the edge device and a LLM in the cloud. Meanwhile, to mitigate the significant communication latency caused by transmitting vocabulary distributions, we introduce separate rejection sampling with sparse compression, which completes the rejection sampling with only a one-time cost of transmitting the compressed vocabulary. Experimental results demonstrate that our solution outperforms baseline and existing methods, achieving up to 2.9 speedup.

-> 엣지-클라우드 협력 프레임워크는 rk3588 기반 엣지 디바이스에서 실시간 스포츠 영상 처리에 필수적입니다.

### R&D: Balancing Reliability and Diversity in Synthetic Data Augmentation for Semantic Segmentation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18427v1
- 점수: final 92.0

Collecting and annotating datasets for pixel-level semantic segmentation tasks are highly labor-intensive. Data augmentation provides a viable solution by enhancing model generalization without additional real-world data collection. Traditional augmentation techniques, such as translation, scaling, and color transformations, create geometric variations but fail to generate new structures. While generative models have been employed to extend semantic information of datasets, they often struggle to maintain consistency between the original and generated images, particularly for pixel-level tasks. In this work, we propose a novel synthetic data augmentation pipeline that integrates controllable diffusion models. Our approach balances diversity and reliability data, effectively bridging the gap between synthetic and real data. We utilize class-aware prompting and visual prior blending to improve image quality further, ensuring precise alignment with segmentation labels. By evaluating benchmark datasets such as PASCAL VOC and BDD100K, we demonstrate that our method significantly enhances semantic segmentation performance, especially in data-scarce scenarios, while improving model robustness in real-world applications. Our code is available at \href{https://github.com/chequanghuy/Enhanced-Generative-Data-Augmentation-for-Semantic-Segmentation-via-Stronger-Guidance}{https://github.com/chequanghuy/Enhanced-Generative-Data-Augmentation-for-Semantic-Segmentation-via-Stronger-Guidance}.

-> 합성 데이터 증강 기술은 스포츠 장면 분석을 위한 다양한 학습 데이터 생성에 효과적입니다.

### AdapTS: Lightweight Teacher-Student Approach for Multi-Class and Continual Visual Anomaly Detection (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17530v1
- 점수: final 92.0

Visual Anomaly Detection (VAD) is crucial for industrial inspection, yet most existing methods are limited to single-category scenarios, failing to address the multi-class and continual learning demands of real-world environments. While Teacher-Student (TS) architectures are efficient, they remain unexplored for the Continual Setting. To bridge this gap, we propose AdapTS, a unified TS framework designed for multi-class and continual settings, optimized for edge deployment. AdapTS eliminates the need for two different architectures by utilizing a single shared frozen backbone and injecting lightweight trainable adapters into the student pathway. Training is enhanced via a segmentation-guided objective and synthetic Perlin noise, while a prototype-based task identification mechanism dynamically selects adapters at inference with 99\% accuracy.   Experiments on MVTec AD and VisA demonstrate that AdapTS matches the performance of existing TS methods across multi-class and continual learning scenarios, while drastically reducing memory overhead. Our lightest variant, AdapTS-S, requires only 8 MB of additional memory, 13x less than STFPM (95 MB), 48x less than RD4AD (360 MB), and 149x less than DeSTSeg (1120 MB), making it a highly scalable solution for edge deployment in complex industrial environments.

-> 엣지 기기 최적화된 경량 비정상 탐지 기술로 스포츠 동작 분석에 적합하며, 단 8MB 메모리로 다른 방법 대비 13~149배 더 효율적

### Unified Spatio-Temporal Token Scoring for Efficient Video VLMs (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18004v1
- 점수: final 92.0

Token pruning is essential for enhancing the computational efficiency of vision-language models (VLMs), particularly for video-based tasks where temporal redundancy is prevalent. Prior approaches typically prune tokens either (1) within the vision transformer (ViT) exclusively for unimodal perception tasks such as action recognition and object segmentation, without adapting to downstream vision-language tasks; or (2) only within the LLM while leaving the ViT output intact, often requiring complex text-conditioned token selection mechanisms. In this paper, we introduce Spatio-Temporal Token Scoring (STTS), a simple and lightweight module that prunes vision tokens across both the ViT and the LLM without text conditioning or token merging, and is fully compatible with end-to-end training. By learning how to score temporally via an auxiliary loss and spatially via LLM downstream gradients, aided by our efficient packing algorithm, STTS prunes 50% of vision tokens throughout the entire architecture, resulting in a 62% improvement in efficiency during both training and inference with only a 0.7% drop in average performance across 13 short and long video QA tasks. Efficiency gains increase with more sampled frames per video. Applying test-time scaling for long-video QA further yields performance gains of 0.5-1% compared to the baseline. Overall, STTS represents a novel, simple yet effective technique for unified, architecture-wide vision token pruning.

-> 비디오 VLM의 효율적 처리 기술이 엣지 기기에서 스포츠 영상 분석에 적용 가능함

### SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18774v1
- 점수: final 90.4

Foundational feed-forward visual geometry models enable accurate and efficient camera pose estimation and scene reconstruction by learning strong scene priors from massive RGB datasets. However, their effectiveness drops when applied to mixed sensing modalities, such as RGB-thermal (RGB-T) images. We observe that while a visual geometry grounded transformer pretrained on RGB data generalizes well to thermal-only reconstruction, it struggles to align RGB and thermal modalities when processed jointly. To address this, we propose SEAR, a simple yet efficient fine-tuning strategy that adapts a pretrained geometry transformer to multimodal RGB-T inputs. Despite being trained on a relatively small RGB-T dataset, our approach significantly outperforms state-of-the-art methods for 3D reconstruction and camera pose estimation, achieving significant improvements over all metrics (e.g., over 29\% in AUC@30) and delivering higher detail and consistency between modalities with negligible overhead in inference time compared to the original pretrained model. Notably, SEAR enables reliable multimodal pose estimation and reconstruction even under challenging conditions, such as low lighting and dense smoke. We validate our architecture through extensive ablation studies, demonstrating how the model aligns both modalities. Additionally, we introduce a new dataset featuring RGB and thermal sequences captured at different times, viewpoints, and illumination conditions, providing a robust benchmark for future work in multimodal 3D scene reconstruction. Code and models are publicly available at https://www.github.com/Schindler-EPFL-Lab/SEAR.

-> 다중 모달 3D 복원 기술은 스포츠 장면 분석에 적용 가능할 수 있습니다

### Translating MRI to PET through Conditional Diffusion Models with Enhanced Pathology Awareness (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18896v1
- 점수: final 90.4

Positron emission tomography (PET) is a widely recognized technique for diagnosing neurodegenerative diseases, offering critical functional insights. However, its high costs and radiation exposure hinder its widespread use. In contrast, magnetic resonance imaging (MRI) does not involve such limitations. While MRI also detects neurodegenerative changes, it is less sensitive for diagnosis compared to PET. To overcome such limitations, one approach is to generate synthetic PET from MRI. Recent advances in generative models have paved the way for cross-modality medical image translation; however, existing methods largely emphasize structural preservation while neglecting the critical need for pathology awareness. To address this gap, we propose PASTA, a novel image translation framework built on conditional diffusion models with enhanced pathology awareness. PASTA surpasses state-of-the-art methods by preserving both structural and pathological details through its highly interactive dual-arm architecture and multi-modal condition integration. Additionally, we introduce a novel cycle exchange consistency and volumetric generation strategy that significantly enhances PASTA's ability to produce high-quality 3D PET images. Our qualitative and quantitative results demonstrate the high quality and pathology awareness of the synthesized PET scans. For Alzheimer's diagnosis, the performance of these synthesized scans improves over MRI by 4%, almost reaching the performance of actual PET. Our code is available at https://github.com/ai-med/PASTA.

-> 조건적 확산 모델을 활용한 이미지 변환 및 향상 기술이 프로젝트의 영상/이미지 보정 기능에 직접 적용 가능하며, 코드가 공개되어 있음

### The Unreasonable Effectiveness of Text Embedding Interpolation for Continuous Image Steering (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17998v1
- 점수: final 89.6

We present a training-free framework for continuous and controllable image editing at test time for text-conditioned generative models. In contrast to prior approaches that rely on additional training or manual user intervention, we find that a simple steering in the text-embedding space is sufficient to produce smooth edit control. Given a target concept (e.g., enhancing photorealism or changing facial expression), we use a large language model to automatically construct a small set of debiased contrastive prompt pairs, from which we compute a steering vector in the generator's text-encoder space. We then add this vector directly to the input prompt representation to control generation along the desired semantic axis. To obtain a continuous control, we propose an elastic range search procedure that automatically identifies an effective interval of steering magnitudes, avoiding both under-steering (no-edit) and over-steering (changing other attributes). Adding the scaled versions of the same vector within this interval yields smooth and continuous edits. Since our method modifies only textual representations, it naturally generalizes across text-conditioned modalities, including image and video generation. To quantify the steering continuity, we introduce a new evaluation metric that measures the uniformity of semantic change across edit strengths. We compare the continuous editing behavior across methods and find that, despite its simplicity and lightweight design, our approach is comparable to training-based alternatives, outperforming other training-free methods.

-> 스포츠 사진 및 영상을 보정하고 개선하기 위한 훈련 없는 이미지 편집 기술 제공

### ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.19157v1
- 점수: final 89.6

Generating rare compositional concepts in text-to-image synthesis remains a challenge for diffusion models, particularly for attributes that are uncommon in the training data. While recent approaches, such as R2F, address this challenge by utilizing LLM for prompt scheduling, they suffer from inherent variance due to the randomness of language models and suboptimal guidance from iterative text embedding switching. To address these problems, we propose the ADAPT framework, a training-free framework that deterministically plans and semantically aligns prompt schedules, providing consistent guidance to enhance the composition of rare concepts. By leveraging attention scores and orthogonal components, ADAPT significantly enhances compositional generation of rare concepts in the RareBench benchmark without additional training or fine-tuning. Through comprehensive experiments, we demonstrate that ADAPT achieves superior performance in RareBench and accurately reflects the semantic information of rare attributes, providing deterministic and precise control over the generation of rare compositions without compromising visual integrity.

-> ADAPT 프레임워크는 스포츠 영상의 품질 향상과 보정 기능을 강화하여 우리 AI 촬영 edge 디바이스의 핵심 기술을 보완합니다.

### Joint Degradation-Aware Arbitrary-Scale Super-Resolution for Variable-Rate Extreme Image Compression (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17408v1
- 점수: final 89.6

Recent diffusion-based extreme image compression methods have demonstrated remarkable performance at ultra-low bitrates. However, most approaches require training separate diffusion models for each target bitrate, resulting in substantial computational overhead and hindering practical deployment. Meanwhile, recent studies have shown that joint super-resolution can serve as an effective approach for enhancing low-bitrate reconstruction. However, when moving toward ultra-low bitrate regimes, these methods struggle due to severe information loss, and their reliance on fixed super-resolution scales prevents flexible adaptation across diverse bitrates.   To address these limitations, we propose ASSR-EIC, a novel image compression framework that leverages arbitrary-scale super-resolution (ASSR) to support variable-rate extreme image compression (EIC). An arbitrary-scale downsampling module is introduced at the encoder side to provide controllable rate reduction, while a diffusion-based, joint degradation-aware ASSR decoder enables rate-adaptive reconstruction within a single model. We exploit the compression- and rescaling-aware diffusion prior to guide the reconstruction, yielding high fidelity and high realism restoration across diverse compression and rescaling settings. Specifically, we design a global compression-rescaling adaptor that offers holistic guidance for rate adaptation, and a local compression-rescaling modulator that dynamically balances generative and fidelity-oriented behaviors to achieve fine-grained, bitrate-adaptive detail restoration. To further enhance reconstruction quality, we introduce a dual semantic-enhanced design.   Extensive experiments demonstrate that ASSR-EIC delivers state-of-the-art performance in extreme image compression while simultaneously supporting flexible bitrate control and adaptive rate-dependent reconstruction.

-> 엣지 디바이스에서 저사용량 영상을 고품질로 복원하는 슈퍼해상도 기술 제공

### On the Cone Effect and Modality Gap in Medical Vision-Language Embeddings (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17246v1
- 점수: final 89.6

Vision-Language Models (VLMs) exhibit a characteristic "cone effect" in which nonlinear encoders map embeddings into highly concentrated regions of the representation space, contributing to cross-modal separation known as the modality gap. While this phenomenon has been widely observed, its practical impact on supervised multimodal learning -particularly in medical domains- remains unclear. In this work, we introduce a lightweight post-hoc mechanism that keeps pretrained VLM encoders frozen while continuously controlling cross-modal separation through a single hyperparameter {λ}. This enables systematic analysis of how the modality gap affects downstream multimodal performance without expensive retraining. We evaluate generalist (CLIP, SigLIP) and medically specialized (BioMedCLIP, MedSigLIP) models across diverse medical and natural datasets in a supervised multimodal settings. Results consistently show that reducing excessive modality gap improves downstream performance, with medical datasets exhibiting stronger sensitivity to gap modulation; however, fully collapsing the gap is not always optimal, and intermediate, task-dependent separation yields the best results. These findings position the modality gap as a tunable property of multimodal representations rather than a quantity that should be universally minimized.

-> Lightweight VLM techniques for cross-modal separation could be applicable to sports content analysis on edge devices

### CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18991v1
- 점수: final 89.6

Aligning Diffusion models has achieved remarkable breakthroughs in generating high-quality, human preference-aligned images. Existing techniques, such as supervised fine-tuning (SFT) and DPO-style preference optimization, have become principled tools for fine-tuning diffusion models. However, SFT relies on high-quality images that are costly to obtain, while DPO-style methods depend on large-scale preference datasets, which are often inconsistent in quality. Beyond data dependency, these methods are further constrained by computational inefficiency. To address these two challenges, we propose Composite Reward Assisted Fine-Tuning (CRAFT), a lightweight yet powerful fine-tuning paradigm that requires significantly reduced training data while maintaining computational efficiency. It first leverages a Composite Reward Filtering (CRF) technique to construct a high-quality and consistent training dataset and then perform an enhanced variant of SFT. We also theoretically prove that CRAFT actually optimizes the lower bound of group-based reinforcement learning, establishing a principled connection between SFT with selected data and reinforcement learning. Our extensive empirical results demonstrate that CRAFT with only 100 samples can easily outperform recent SOTA preference optimization methods with thousands of preference-paired samples. Moreover, CRAFT can even achieve 11-220$\times$ faster convergences than the baseline preference optimization methods, highlighting its extremely high efficiency.

-> 경량 확산 모델 미세조정 기술이 엣지 디바이스에서 스포츠 이미지 생성 및 보정에 적용 가능

### Revisiting Cross-Attention Mechanisms: Leveraging Beneficial Noise for Domain-Adaptive Learning (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17474v1
- 점수: final 89.6

Unsupervised Domain Adaptation (UDA) seeks to transfer knowledge from a labeled source domain to an unlabeled target domain but often suffers from severe domain and scale gaps that degrade performance. Existing cross-attention-based transformers can align features across domains, yet they struggle to preserve content semantics under large appearance and scale variations. To explicitly address these challenges, we introduce the concept of beneficial noise, which regularizes cross-attention by injecting controlled perturbations, encouraging the model to ignore style distractions and focus on content. We propose the Domain-Adaptive Cross-Scale Matching (DACSM) framework, which consists of a Domain-Adaptive Transformer (DAT) for disentangling domain-shared content from domain-specific style, and a Cross-Scale Matching (CSM) module that adaptively aligns features across multiple resolutions. DAT incorporates beneficial noise into cross-attention, enabling progressive domain translation with enhanced robustness, yielding content-consistent and style-invariant representations. Meanwhile, CSM ensures semantic consistency under scale changes. Extensive experiments on VisDA-2017, Office-Home, and DomainNet demonstrate that DACSM achieves state-of-the-art performance, with up to +2.3% improvement over CDTrans on VisDA-2017. Notably, DACSM achieves a +5.9% gain on the challenging "truck" class of VisDA, evidencing the strength of beneficial noise in handling scale discrepancies. These results highlight the effectiveness of combining domain translation, beneficial-noise-enhanced attention, and scale-aware alignment for robust cross-domain representation learning.

-> 다양한 환경에서 촬영된 스포츠 영상을 정규화하고 도메인 적응 기술 제공

### Motion-Adaptive Temporal Attention for Lightweight Video Generation with Stable Diffusion (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17398v1
- 점수: final 89.6

We present a motion-adaptive temporal attention mechanism for parameter-efficient video generation built upon frozen Stable Diffusion models. Rather than treating all video content uniformly, our method dynamically adjusts temporal attention receptive fields based on estimated motion content: high-motion sequences attend locally across frames to preserve rapidly changing details, while low-motion sequences attend globally to enforce scene consistency. We inject lightweight temporal attention modules into all UNet transformer blocks via a cascaded strategy -- global attention in down-sampling and middle blocks for semantic stabilization, motion-adaptive attention in up-sampling blocks for fine-grained refinement. Combined with temporally correlated noise initialization and motion-aware gating, the system adds only 25.8M trainable parameters (2.9\% of the base UNet) while achieving competitive results on WebVid validation when trained on 100K videos. We demonstrate that the standard denoising objective alone provides sufficient implicit temporal regularization, outperforming approaches that add explicit temporal consistency losses. Our ablation studies reveal a clear trade-off between noise correlation and motion amplitude, providing a practical inference-time control for diverse generation behaviors.

-> 모션 적응 시간적 주의 메커니즘을 통해 스포츠 하이라이트 생성에 최적화된 가벼운 비디오 생성 기술로, 엣지 디바이스에서 높은 성능을 유지하면서도 파라미터 수가 25.8M(기본 UNet의 2.9%)에 불과해 rk3588에 적합하다.

### Prompt-Free Universal Region Proposal Network (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17554v1
- 점수: final 88.0

Identifying potential objects is critical for object recognition and analysis across various computer vision applications. Existing methods typically localize potential objects by relying on exemplar images, predefined categories, or textual descriptions. However, their reliance on image and text prompts often limits flexibility, restricting adaptability in real-world scenarios. In this paper, we introduce a novel Prompt-Free Universal Region Proposal Network (PF-RPN), which identifies potential objects without relying on external prompts. First, the Sparse Image-Aware Adapter (SIA) module performs initial localization of potential objects using a learnable query embedding dynamically updated with visual features. Next, the Cascade Self-Prompt (CSP) module identifies the remaining potential objects by leveraging the self-prompted learnable embedding, autonomously aggregating informative visual features in a cascading manner. Finally, the Centerness-Guided Query Selection (CG-QS) module facilitates the selection of high-quality query embeddings using a centerness scoring network. Our method can be optimized with limited data (e.g., 5% of MS COCO data) and applied directly to various object detection application domains for identifying potential objects without fine-tuning, such as underwater object detection, industrial defect detection, and remote sensing image object detection. Experimental results across 19 datasets validate the effectiveness of our method. Code is available at https://github.com/tangqh03/PF-RPN.

-> 외부 프롬프트 없이 스포츠 장면 내 객체를 식별해 선수 및 경기 요소 자동 분석에 활용 가능

### ChopGrad: Pixel-Wise Losses for Latent Video Diffusion via Truncated Backpropagation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17812v1
- 점수: final 88.0

Recent video diffusion models achieve high-quality generation through recurrent frame processing where each frame generation depends on previous frames. However, this recurrent mechanism means that training such models in the pixel domain incurs prohibitive memory costs, as activations accumulate across the entire video sequence. This fundamental limitation also makes fine-tuning these models with pixel-wise losses computationally intractable for long or high-resolution videos. This paper introduces ChopGrad, a truncated backpropagation scheme for video decoding, limiting gradient computation to local frame windows while maintaining global consistency. We provide a theoretical analysis of this approximation and show that it enables efficient fine-tuning with frame-wise losses. ChopGrad reduces training memory from scaling linearly with the number of video frames (full backpropagation) to constant memory, and compares favorably to existing state-of-the-art video diffusion models across a suite of conditional video generation tasks with pixel-wise losses, including video super-resolution, video inpainting, video enhancement of neural-rendered scenes, and controlled driving video generation.

-> 스포츠 영상의 메모리 효율적 처리로 실시간 하이라이트 편집 가능하며, 고해상도 영상 보정에 적합한 기술

### A Creative Agent is Worth a 64-Token Template (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17895v1
- 점수: final 88.0

Text-to-image (T2I) models have substantially improved image fidelity and prompt adherence, yet their creativity remains constrained by reliance on discrete natural language prompts. When presented with fuzzy prompts such as ``a creative vinyl record-inspired skyscraper'', these models often fail to infer the underlying creative intent, leaving creative ideation and prompt design largely to human users. Recent reasoning- or agent-driven approaches iteratively augment prompts but incur high computational and monetary costs, as their instance-specific generation makes ``creativity'' costly and non-reusable, requiring repeated queries or reasoning for subsequent generations. To address this, we introduce \textbf{CAT}, a framework for \textbf{C}reative \textbf{A}gent \textbf{T}okenization that encapsulates agents' intrinsic understanding of ``creativity'' through a \textit{Creative Tokenizer}. Given the embeddings of fuzzy prompts, the tokenizer generates a reusable token template that can be directly concatenated with them to inject creative semantics into T2I models without repeated reasoning or prompt augmentation. To enable this, the tokenizer is trained via creative semantic disentanglement, leveraging relations among partially overlapping concept pairs to capture the agent's latent creative representations. Extensive experiments on \textbf{\textit{Architecture Design}}, \textbf{\textit{Furniture Design}}, and \textbf{\textit{Nature Mixture}} tasks demonstrate that CAT provides a scalable and effective paradigm for enhancing creativity in T2I generation, achieving a $3.7\times$ speedup and a $4.8\times$ reduction in computational cost, while producing images with superior human preference and text-image alignment compared to state-of-the-art T2I models and creative generation methods.

-> 관련된 이미지 생성 및 편집 기술을 제공

### Enabling Real-Time Programmability for RAN Functions: A Wasm-Based Approach for Robust and High-Performance dApps (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17880v1
- 점수: final 88.0

While the Open Radio Access Network Alliance (O-RAN) architecture enables third-party applications to optimize radio access networks at multiple timescales, real-time distributed applications (dApps) that demand low latency, high performance, and strong isolation remain underexplored. Existing approaches propose colocating a new RAN Intelligent Controller (RIC) at the edge, or deploying dApps in bare metal along with RAN functions. While the former approach increases network complexity and requires additional edge computing resources, the latter raises serious security concerns due to the lack of native mechanisms to isolate dApps and RAN functions. Meanwhile, WebAssembly (Wasm) has emerged as a lightweight, fast technology for robust execution of external, untrusted code. In this work, we propose a new approach to executing dApps using Wasm to isolate applications in real-time in O-RAN. Results show that our lightweight and robust approach ensures predictable, deterministic performance, strong isolation, and low latency, enabling real-time control loops.

-> 엣지 컴퓨팅 및 실시간 처리 기술은 스포츠 영상 분석에 필수적이며 디바이스의 핵심 기능과 직접 관련됩니다.

### Modeling the Impacts of Swipe Delay on User Quality of Experience in Short Video Streaming (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18575v1
- 점수: final 88.0

Short video streaming platforms have gained immense popularity in recent years, transforming the way users consume video content. A critical aspect of user interaction with these platforms is the swipe gesture, which allows users to navigate through videos seamlessly. However, the delay between a user's swipe action and the subsequent video playback can significantly impact the overall user experience. This paper presents the first systematic study investigating the effects of swipe delay on user Quality of Experience (QoE) in short video streaming. In particular, we conduct a subjective quality assessment containing 132 swipe delay patterns. The obtained results show that user experience is affected not only by the swipe delay duration, but also by the number of delays and their temporal positions. A single delay of eight seconds or longer is likely to lead to user dissatisfaction. Moreover, early-session delays are less harmful to user QoE than late-session delays. Based on the findings, we propose a novel QoE model that accurately predicts user experience based on swipe delay characteristics. The proposed model demonstrates high correlation with subjective ratings, outperforming existing models in short video streaming.

-> User experience modeling for short video streaming relevant to sports content platform

### CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17779v1
- 점수: final 86.4

Single-view 3D human reconstruction has garnered significant attention in recent years. Despite numerous advancements, prior research has concentrated on reconstructing 3D models from clear, close-up images of individual subjects, often yielding subpar results in the more prevalent multi-person scenarios. Reconstructing 3D human crowd models is a highly intricate task, laden with challenges such as: 1) extensive occlusions, 2) low clarity, and 3) numerous and various appearances. To address this task, we propose CrowdGaussian, a unified framework that directly reconstructs multi-person 3D Gaussian Splatting (3DGS) representations from single-image inputs. To handle occlusions, we devise a self-supervised adaptation pipeline that enables the pretrained large human model to reconstruct complete 3D humans with plausible geometry and appearance from heavily occluded inputs. Furthermore, we introduce Self-Calibrated Learning (SCL). This training strategy enables single-step diffusion models to adaptively refine coarse renderings to optimal quality by blending identity-preserving samples with clean/corrupted image pairs. The outputs can be distilled back to enhance the quality of multi-person 3DGS representations. Extensive experiments demonstrate that CrowdGaussian generates photorealistic, geometrically coherent reconstructions of multi-person scenes.

-> 단일 이미지에서 다인물 3D 재구성 기술로 스포츠 장면의 공간적 관계 분석에 적합

### From Digital Twins to World Models:Opportunities, Challenges, and Applications for Mobile Edge General Intelligence (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17420v1
- 점수: final 85.6

The rapid evolution toward 6G and beyond communication systems is accelerating the convergence of digital twins and world models at the network edge. Traditional digital twins provide high-fidelity representations of physical systems and support monitoring, analysis, and offline optimization. However, in highly dynamic edge environments, they face limitations in autonomy, adaptability, and scalability. This paper presents a systematic survey of the transition from digital twins to world models and discusses its role in enabling edge general intelligence (EGI). First, the paper clarifies the conceptual differences between digital twins and world models and highlights the shift from physics-based, centralized, and system-centric replicas to data-driven, decentralized, and agent-centric internal models. This discussion helps readers gain a clear understanding of how this transition enables more adaptive, autonomous, and resource-efficient intelligence at the network edge. The paper reviews the design principles, architectures, and key components of world models, including perception, latent state representation, dynamics learning, imagination-based planning, and memory. In addition, it examines the integration of world models and digital twins in wireless EGI systems and surveys emerging applications in integrated sensing and communications, semantic communication, air-ground networks, and low-altitude wireless networks. Finally, this survey provides a systematic roadmap and practical insights for designing world-model-driven edge intelligence systems in wireless and edge computing environments. It also outlines key research challenges and future directions toward scalable, reliable, and interoperable world models for edge-native agentic AI.

-> 네트워크 엣지에서의 디지털 트윈과 월드 모델의 전환은 스포츠 촬영 및 분석을 위한 AI 엣지 디바이스에 직접 적용 가능합니다.

### SegFly: A 2D-3D-2D Paradigm for Aerial RGB-Thermal Semantic Segmentation at Scale (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17920v1
- 점수: final 84.8

Semantic segmentation for uncrewed aerial vehicles (UAVs) is fundamental for aerial scene understanding, yet existing RGB and RGB-T datasets remain limited in scale, diversity, and annotation efficiency due to the high cost of manual labeling and the difficulties of accurate RGB-T alignment on off-the-shelf UAVs. To address these challenges, we propose a scalable geometry-driven 2D-3D-2D paradigm that leverages multi-view redundancy in high-overlap aerial imagery to automatically propagate labels from a small subset of manually annotated RGB images to both RGB and thermal modalities within a unified framework. By lifting less than 3% of RGB images into a semantic 3D point cloud and reprojecting it into all views, our approach enables dense pseudo ground-truth generation across large image collections, automatically producing 97% of RGB labels and 100% of thermal labels while achieving 91% and 88% annotation accuracy without any 2D manual refinement. We further extend this 2D-3D-2D paradigm to cross-modal image registration, using 3D geometry as an intermediate alignment space to obtain fully automatic, strong pixel-level RGB-T alignment with 87% registration accuracy and no hardware-level synchronization. Applying our framework to existing geo-referenced aerial imagery, we construct SegFly, a large-scale benchmark with over 20,000 high-resolution RGB images and more than 15,000 geometrically aligned RGB-T pairs spanning diverse urban, industrial, and rural environments across multiple altitudes and seasons. On SegFly, we establish the Firefly baseline for RGB and thermal semantic segmentation and show that both conventional architectures and vision foundation models benefit substantially from SegFly supervision, highlighting the potential of geometry-driven 2D-3D-2D pipelines for scalable multi-modal scene understanding. Data and Code available at https://github.com/markus-42/SegFly.

-> 2D-3D-2D 패러다임을 활용한 의미론적 분할 기술로 다양한 스포츠 장면의 효율적 분석 가능

### TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17828v1
- 점수: final 84.8

Although text-to-image diffusion models exhibit remarkable generative power, concept erasure techniques are essential for their safe deployment to prevent the creation of harmful content. This has fostered a dynamic interplay between the development of erasure defenses and the adversarial probes designed to bypass them, and this co-evolution has progressively enhanced the efficacy of erasure methods. However, this adversarial co-evolution has converged on a narrow, text-centric paradigm that equates erasure with severing the text-to-image mapping, ignoring that the underlying visual knowledge related to undesired concepts still persist. To substantiate this claim, we investigate from a visual perspective, leveraging DDIM inversion to probe whether a generative pathway for the erased concept can still be found. However, identifying such a visual generative pathway is challenging because standard text-guided DDIM inversion is actively resisted by text-centric defenses within the erased model. To address this, we introduce TINA, a novel Text-free INversion Attack, which enforces this visual-only probe by operating under a null-text condition, thereby avoiding existing text-centric defenses. Moreover, TINA integrates an optimization procedure to overcome the accumulating approximation errors that arise when standard inversion operates without its usual textual guidance. Our experiments demonstrate that TINA regenerates erased concepts from models treated with state-of-the-art unlearning. The success of TINA proves that current methods merely obscure concepts, highlighting an urgent need for paradigms that operate directly on internal visual knowledge.

-> 이미지 생성 및 편집 기술과 관련

### TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17735v1
- 점수: final 84.0

Automatically generating photorealistic and self-consistent appearances for untextured 3D models is a critical challenge in digital content creation. The advancement of large-scale video generation models offers a natural approach: directly synthesizing 360-degree turntable videos (TTVs), which can serve not only as high-quality dynamic previews but also as an intermediate representation to drive texture synthesis and neural rendering. However, existing general-purpose video diffusion models struggle to maintain strict geometric consistency and appearance stability across the full range of views, making their outputs ill-suited for high-quality 3D reconstruction. To this end, we introduce TAPESTRY, a framework for generating high-fidelity TTVs conditioned on explicit 3D geometry. We reframe the 3D appearance generation task as a geometry-conditioned video diffusion problem: given a 3D mesh, we first render and encode multi-modal geometric features to constrain the video generation process with pixel-level precision, thereby enabling the creation of high-quality and consistent TTVs. Building upon this, we also design a method for downstream reconstruction tasks from the TTV input, featuring a multi-stage pipeline with 3D-Aware Inpainting. By rotating the model and performing a context-aware secondary generation, this pipeline effectively completes self-occluded regions to achieve full surface coverage. The videos generated by TAPESTRY are not only high-quality dynamic previews but also serve as a reliable, 3D-aware intermediate representation that can be seamlessly back-projected into UV textures or used to supervise neural rendering methods like 3DGS. This enables the automated creation of production-ready, complete 3D assets from untextured meshes. Experimental results demonstrate that our method outperforms existing approaches in both video consistency and final reconstruction quality.

-> 기하학 조건 기반 비디오 생성으로 스포츠 하이라이트의 일관성 있는 제작 가능

### FineViT: Progressively Unlocking Fine-Grained Perception with Dense Recaptions (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17326v1
- 점수: final 84.0

While Multimodal Large Language Models (MLLMs) have experienced rapid advancements, their visual encoders frequently remain a performance bottleneck. Conventional CLIP-based encoders struggle with dense spatial tasks due to the loss of visual details caused by low-resolution pretraining and the reliance on noisy, coarse web-crawled image-text pairs. To overcome these limitations, we introduce FineViT, a novel vision encoder specifically designed to unlock fine-grained perception. By replacing coarse web data with dense recaptions, we systematically mitigate information loss through a progressive training paradigm.: first, the encoder is trained from scratch at a high native resolution on billions of global recaptioned image-text pairs, establishing a robust, detail rich semantic foundation. Subsequently, we further enhance its local perception through LLM alignment, utilizing our curated FineCap-450M dataset that comprises over $450$ million high quality local captions. Extensive experiments validate the effectiveness of the progressive strategy. FineViT achieves state-of-the-art zero-shot recognition and retrieval performance, especially in long-context retrieval, and consistently outperforms multimodal visual encoders such as SigLIP2 and Qwen-ViT when integrated into MLLMs. We hope FineViT could serve as a powerful new baseline for fine-grained visual perception.

-> 세밀한 시각 인식에 특화된 비전 인코더로 스포츠 동작 분석에 직접적으로 적용 가능한 최고 성능 기술

### VLM2Rec: Resolving Modality Collapse in Vision-Language Model Embedders for Multimodal Sequential Recommendation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17450v1
- 점수: final 84.0

Sequential Recommendation (SR) in multimodal settings typically relies on small frozen pretrained encoders, which limits semantic capacity and prevents Collaborative Filtering (CF) signals from being fully integrated into item representations. Inspired by the recent success of Large Language Models (LLMs) as high-capacity embedders, we investigate the use of Vision-Language Models (VLMs) as CF-aware multimodal encoders for SR. However, we find that standard contrastive supervised fine-tuning (SFT), which adapts VLMs for embedding generation and injects CF signals, can amplify its inherent modality collapse. In this state, optimization is dominated by a single modality while the other degrades, ultimately undermining recommendation accuracy. To address this, we propose VLM2Rec, a VLM embedder-based framework for multimodal sequential recommendation designed to ensure balanced modality utilization. Specifically, we introduce Weak-modality Penalized Contrastive Learning to rectify gradient imbalance during optimization and Cross-Modal Relational Topology Regularization to preserve geometric consistency between modalities. Extensive experiments demonstrate that VLM2Rec consistently outperforms state-of-the-art baselines in both accuracy and robustness across diverse scenarios.

-> 멀티모달 스포츠 콘텐츠 분석 및 추천 시스템 구축에 적합한 VLM 임베더 기술

### AR-CoPO: Align Autoregressive Video Generation with Contrastive Policy Optimization (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17461v1
- 점수: final 84.0

Streaming autoregressive (AR) video generators combined with few-step distillation achieve low-latency, high-quality synthesis, yet remain difficult to align via reinforcement learning from human feedback (RLHF). Existing SDE-based GRPO methods face challenges in this setting: few-step ODEs and consistency model samplers deviate from standard flow-matching ODEs, and their short, low-stochasticity trajectories are highly sensitive to initialization noise, rendering intermediate SDE exploration ineffective. We propose AR-CoPO (AutoRegressive Contrastive Policy Optimization), a framework that adapts the Neighbor GRPO contrastive perspective to streaming AR generation. AR-CoPO introduces chunk-level alignment via a forking mechanism that constructs neighborhood candidates at a randomly selected chunk, assigns sequence-level rewards, and performs localized GRPO updates. We further propose a semi-on-policy training strategy that complements on-policy exploration with exploitation over a replay buffer of reference rollouts, improving generation quality across domains. Experiments on Self-Forcing demonstrate that AR-CoPO improves both out-of-domain generalization and in-domain human preference alignment over the baseline, providing evidence of genuine alignment rather than reward hacking.

-> 저지연 고품질 스포츠 하이라이트 영상 생성에 적용 가능한 스트리밍 AR 생성 기술

### GenMFSR: Generative Multi-Frame Image Restoration and Super-Resolution (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.19187v1
- 점수: final 84.0

Camera pipelines receive raw Bayer-format frames that need to be denoised, demosaiced, and often super-resolved. Multiple frames are captured to utilize natural hand tremors and enhance resolution. Multi-frame super-resolution is therefore a fundamental problem in camera pipelines. Existing adversarial methods are constrained by the quality of ground truth. We propose GenMFSR, the first Generative Multi-Frame Raw-to-RGB Super Resolution pipeline, that incorporates image priors from foundation models to obtain sub-pixel information for camera ISP applications. GenMFSR can align multiple raw frames, unlike existing single-frame super-resolution methods, and we propose a loss term that restricts generation to high-frequency regions in the raw domain, thus preventing low-frequency artifacts.

-> 다중 프레임 이미지 복원 및 슈퍼해상도 기술이 AI 촬영 디바이스에 직접적으로 적용 가능

### Counting Circuits: Mechanistic Interpretability of Visual Reasoning in Large Vision-Language Models (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18523v1
- 점수: final 84.0

Counting serves as a simple but powerful test of a Large Vision-Language Model's (LVLM's) reasoning; it forces the model to identify each individual object and then add them all up. In this study, we investigate how LVLMs implement counting using controlled synthetic and real-world benchmarks, combined with mechanistic analyses. Our results show that LVLMs display a human-like counting behavior, with precise performance on small numerosities and noisy estimation for larger quantities. We introduce two novel interpretability methods, Visual Activation Patching and HeadLens, and use them to uncover a structured "counting circuit" that is largely shared across a variety of visual reasoning tasks. Building on these insights, we propose a lightweight intervention strategy that exploits simple and abundantly available synthetic images to fine-tune arbitrary pretrained LVLMs exclusively on counting. Despite the narrow scope of this fine-tuning, the intervention not only enhances counting accuracy on in-distribution synthetic data, but also yields an average improvement of +8.36% on out-of-distribution counting benchmarks and an average gain of +1.54% on complex, general visual reasoning tasks for Qwen2.5-VL. These findings highlight the central, influential role of counting in visual reasoning and suggest a potential pathway for improving overall visual reasoning capabilities through targeted enhancement of counting mechanisms.

-> 시각적 추론과 객체 인식 기술이 스포츠 장면 분석에 직접적으로 적용 가능하여 선수 추적 및 경기 전략 분석에 활용될 수 있습니다.

### Through the Looking-Glass: AI-Mediated Video Communication Reduces Interpersonal Trust and Confidence in Judgments (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18868v1
- 점수: final 84.0

AI-based tools that mediate, enhance or generate parts of video communication may interfere with how people evaluate trustworthiness and credibility. In two preregistered online experiments (N = 2,000), we examined whether AI-mediated video retouching, background replacement and avatars affect interpersonal trust, people's ability to detect lies and confidence in their judgments. Participants watched short videos of speakers making truthful or deceptive statements across three conditions with varying levels of AI mediation. We observed that perceived trust and confidence in judgments declined in AI-mediated videos, particularly in settings in which some participants used avatars while others did not. However, participants' actual judgment accuracy remained unchanged, and they were no more inclined to suspect those using AI tools of lying. Our findings provide evidence against concerns that AI mediation undermines people's ability to distinguish truth from lies, and against cue-based accounts of lie detection more generally. They highlight the importance of trustworthy AI mediation tools in contexts where not only truth, but also trust and confidence matter.

-> AI 매개 비디오 처리 및 향상 기술은 프로젝트의 영상 향상 기능에 직접 적용 가능하여 핵심 기술이 될 수 있습니다.

### Feeling the Space: Egomotion-Aware Video Representation for Efficient and Accurate 3D Scene Understanding (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17980v1
- 점수: final 82.4

Recent Multimodal Large Language Models (MLLMs) have shown high potential for spatial reasoning within 3D scenes. However, they typically rely on computationally expensive 3D representations like point clouds or reconstructed Bird's-Eye View (BEV) maps, or lack physical grounding to resolve ambiguities in scale and size. This paper significantly enhances MLLMs with egomotion modality data, captured by Inertial Measurement Units (IMUs) concurrently with the video. In particular, we propose a novel framework, called Motion-MLLM, introducing two key components: (1) a cascaded motion-visual keyframe filtering module that leverages both IMU data and visual features to efficiently select a sparse yet representative set of keyframes, and (2) an asymmetric cross-modal fusion module where motion tokens serve as intermediaries that channel egomotion cues and cross-frame visual context into the visual representation. By grounding visual content in physical egomotion trajectories, Motion-MLLM can reason about absolute scale and spatial relationships across the scene. Our extensive evaluation shows that Motion-MLLM makes significant improvements in various tasks related to 3D scene understanding and spatial reasoning. Compared to state-of-the-art (SOTA) methods based on video frames and explicit 3D data, Motion-MLLM exhibits similar or even higher accuracy with significantly less overhead (i.e., 1.40$\times$ and 1.63$\times$ higher cost-effectiveness, respectively).

-> IMU 데이터를 활용한 3D 스포츠 장면 이해 기술로 공간적 관계 분석 가능

### FILT3R: Latent State Adaptive Kalman Filter for Streaming 3D Reconstruction (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18493v1
- 점수: final 82.4

Streaming 3D reconstruction maintains a persistent latent state that is updated online from incoming frames, enabling constant-memory inference. A key failure mode is the state update rule: aggressive overwrites forget useful history, while conservative updates fail to track new evidence, and both behaviors become unstable beyond the training horizon. To address this challenge, we propose FILT3R, a training-free latent filtering layer that casts recurrent state updates as stochastic state estimation in token space. FILT3R maintains a per-token variance and computes a Kalman-style gain that adaptively balances memory retention against new observations. Process noise -- governing how much the latent state is expected to change between frames -- is estimated online from EMA-normalized temporal drift of candidate tokens. Using extensive experiments, we demonstrate that FILT3R yields an interpretable, plug-in update rule that generalizes common overwrite and gating policies as special cases. Specifically, we show that gains shrink in stable regimes as uncertainty contracts with accumulated evidence, and rise when genuine scene change increases process uncertainty, improving long-horizon stability for depth, pose, and 3D reconstruction, compared to the existing methods. Code will be released at https://github.com/jinotter3/FILT3R.

-> 스트리밍 3D 재구성 기술이 실시간 스포츠 촬영에 적용 가능하여 빠른 움직임과 변화하는 관점에서도 안정적인 3D 재구성을 제공합니다.

### ReLaGS: Relational Language Gaussian Splatting (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17605v1
- 점수: final 80.0

Achieving unified 3D perception and reasoning across tasks such as segmentation, retrieval, and relation understanding remains challenging, as existing methods are either object-centric or rely on costly training for inter-object reasoning. We present a novel framework that constructs a hierarchical language-distilled Gaussian scene and its 3D semantic scene graph without scene-specific training. A Gaussian pruning mechanism refines scene geometry, while a robust multi-view language alignment strategy aggregates noisy 2D features into accurate 3D object embeddings. On top of this hierarchy, we build an open-vocabulary 3D scene graph with Vision Language derived annotations and Graph Neural Network-based relational reasoning. Our approach enables efficient and scalable open-vocabulary 3D reasoning by jointly modeling hierarchical semantics and inter/intra-object relationships, validated across tasks including open-vocabulary segmentation, scene graph generation, and relation-guided retrieval. Project page: https://dfki-av.github.io/ReLaGS/

-> 스포츠 분석에 잠재적으로 유용한 3D 장면 이해 기술로, 선수 추적 및 전략 분석에 필수적입니다.

### VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17470v1
- 점수: final 80.0

Monocular 3D object detection typically relies on pseudo-labeling techniques to reduce dependency on real-world annotations. Recent advances demonstrate that deterministic linguistic cues can serve as effective auxiliary weak supervision signals, providing complementary semantic context. However, hand-crafted textual descriptions struggle to capture the inherent visual diversity of individuals across scenes, limiting the model's ability to learn scene-aware representations. To address this challenge, we propose Visual-referred Probabilistic Prompt Learning (VirPro), an adaptive multi-modal pretraining paradigm that can be seamlessly integrated into diverse weakly supervised monocular 3D detection frameworks. Specifically, we generate a diverse set of learnable, instance-conditioned prompts across scenes and store them in an Adaptive Prompt Bank (APB). Subsequently, we introduce Multi-Gaussian Prompt Modeling (MGPM), which incorporates scene-based visual features into the corresponding textual embeddings, allowing the text prompts to express visual uncertainties. Then, from the fused vision-language embeddings, we decode a prompt-targeted Gaussian, from which we derive a unified object-level prompt embedding for each instance. RoI-level contrastive matching is employed to enforce modality alignment, bringing embeddings of co-occurring objects within the same scene closer in the latent space, thus enhancing semantic coherence. Extensive experiments on the KITTI benchmark demonstrate that integrating our pretraining paradigm consistently yields substantial performance gains, achieving up to a 4.8% average precision improvement than the baseline.

-> Provides applicable 3D detection technology for sports scene analysis

### Steering Video Diffusion Transformers with Massive Activations (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17825v1
- 점수: final 80.0

Despite rapid progress in video diffusion transformers, how their internal model signals can be leveraged with minimal overhead to enhance video generation quality remains underexplored. In this work, we study the role of Massive Activations (MAs), which are rare, high-magnitude hidden state spikes in video diffusion transformers. We observed that MAs emerge consistently across all visual tokens, with a clear magnitude hierarchy: first-frame tokens exhibit the largest MA magnitudes, latent-frame boundary tokens (the head and tail portions of each temporal chunk in the latent space) show elevated but slightly lower MA magnitudes than the first frame, and interior tokens within each latent frame remain elevated, yet are comparatively moderate in magnitude. This structured pattern suggests that the model implicitly prioritizes token positions aligned with the temporal chunking in the latent space. Based on this observation, we propose Structured Activation Steering (STAS), a training-free self-guidance-like method that steers MA values at first-frame and boundary tokens toward a scaled global maximum reference magnitude. STAS achieves consistent improvements in terms of video quality and temporal coherence across different text-to-video models, while introducing negligible computational overhead.

-> 비디오 확산 트랜스포머의 내부 신호를 활용해 스포츠 하이라이트 영상 품질을 향상시키는 기술

### EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18001v1
- 점수: final 80.0

In this work, we present EchoGen, a unified framework for layout-to-image generation and image grounding, capable of generating images with accurate layouts and high fidelity to text descriptions (e.g., spatial relationships), while grounding the image robustly at the same time. We believe that image grounding possesses strong text and layout understanding abilities, which can compensate for the corresponding limitations in layout-to-image generation. At the same time, images generated from layouts exhibit high diversity in content, thereby enhancing the robustness of image grounding. Jointly training both tasks within a unified model can promote performance improvements for each. However, we identify that this joint training paradigm encounters several optimization challenges and results in restricted performance. To address these issues, we propose progressive training strategies. First, the Parallel Multi-Task Pre-training (PMTP) stage equips the model with basic abilities for both tasks, leveraging shared tokens to accelerate training. Next, the Dual Joint Optimization (DJO) stage exploits task duality to sequentially integrate the two tasks, enabling unified optimization. Finally, the Cycle RL stage eliminates reliance on visual supervision by using consistency constraints as rewards, significantly enhancing the model's unified capabilities via the GRPO strategy. Extensive experiments demonstrate state-of-the-art results on both layout-to-image generation and image grounding benchmarks, and reveal clear synergistic gains from optimizing the two tasks together.

-> 이미지 생성 및 이해에 대한 통합 프레임워크로 스포츠 콘텐츠 생성에 적용 가능

### Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.17655v1
- 점수: final 80.0

Cross-Domain Few-Shot Learning (CDFSL) adapts models trained with large-scale general data (source domain) to downstream target domains with only scarce training data, where the research on vision-language models (e.g., CLIP) is still in the early stages. Typical downstream domains, such as medical diagnosis, require fine-grained visual cues for interpretable recognition, but we find that current fine-tuned CLIP models can hardly focus on these cues, albeit they can roughly focus on important regions in source domains. Although current works have demonstrated CLIP's shortcomings in capturing local subtle patterns, in this paper, we find that the domain gap and scarce training data further exacerbate such shortcomings, much more than that of holistic patterns, which we call the local misalignment problem in CLIP-based CDFSL. To address this problem, due to the lack of supervision in aligning local visual features and text semantics, we turn to self-supervision information. Inspired by the translation task, we propose the CC-CDFSL method with cycle consistency, which translates local visual features into text features and then translates them back into visual features (and vice versa), and constrains the original features close to the translated back features. To reduce the noise imported by richer information in the visual modality, we further propose a Semantic Anchor mechanism, which first augments visual features to provide a larger corpus for the text-to-image mapping, and then shrinks the image features to filter out irrelevant image-to-text mapping. Extensive experiments on various benchmarks, backbones, and fine-tuning methods show we can (1) effectively improve the local vision-language alignment, (2) enhance the interpretability of learned patterns and model decisions by visualizing patches, and (3) achieve state-of-the-art performance.

-> 크로스 도메인 적응 기술로 다양한 스포츠 분야에서 적응적인 분석 모델 구현 가능

### Generalized Hand-Object Pose Estimation with Occlusion Awareness (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.19013v1
- 점수: final 80.0

Generalized 3D hand-object pose estimation from a single RGB image remains challenging due to the large variations in object appearances and interaction patterns, especially under heavy occlusion. We propose GenHOI, a framework for generalized hand-object pose estimation with occlusion awareness. GenHOI integrates hierarchical semantic knowledge with hand priors to enhance model generalization under challenging occlusion conditions. Specifically, we introduce a hierarchical semantic prompt that encodes object states, hand configurations, and interaction patterns via textual descriptions. This enables the model to learn abstract high-level representations of hand-object interactions for generalization to unseen objects and novel interactions while compensating for missing or ambiguous visual cues. To enable robust occlusion reasoning, we adopt a multi-modal masked modeling strategy over RGB images, predicted point clouds, and textual descriptions. Moreover, we leverage hand priors as stable spatial references to extract implicit interaction constraints. This allows reliable pose inference even under significant variations in object shapes and interaction patterns. Extensive experiments on the challenging DexYCB and HO3Dv2 benchmarks demonstrate that our method achieves state-of-the-art performance in hand-object pose estimation.

-> 손 자세 추정 기술은 스포츠 동작 분석에 적용 가능하여 선수들의 기술 분석과 전략 개발에 활용될 수 있습니다.

### Improving Joint Audio-Video Generation with Cross-Modal Context Learning (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.18600v1
- 점수: final 80.0

The dual-stream transformer architecture-based joint audio-video generation method has become the dominant paradigm in current research. By incorporating pre-trained video diffusion models and audio diffusion models, along with a cross-modal interaction attention module, high-quality, temporally synchronized audio-video content can be generated with minimal training data. In this paper, we first revisit the dual-stream transformer paradigm and further analyze its limitations, including model manifold variations caused by the gating mechanism controlling cross-modal interactions, biases in multi-modal background regions introduced by cross-modal attention, and the inconsistencies in multi-modal classifier-free guidance (CFG) during training and inference, as well as conflicts between multiple conditions. To alleviate these issues, we propose Cross-Modal Context Learning (CCL), equipped with several carefully designed modules. Temporally Aligned RoPE and Partitioning (TARP) effectively enhances the temporal alignment between audio latent and video latent representations. The Learnable Context Tokens (LCT) and Dynamic Context Routing (DCR) in the Cross-Modal Context Attention (CCA) module provide stable unconditional anchors for cross-modal information, while dynamically routing based on different training tasks, further enhancing the model's convergence speed and generation quality. During inference, Unconditional Context Guidance (UCG) leverages the unconditional support provided by LCT to facilitate different forms of CFG, improving train-inference consistency and further alleviating conflicts. Through comprehensive evaluations, CCL achieves state-of-the-art performance compared with recent academic methods while requiring substantially fewer resources.

-> 오디오-비디오 생성 기술은 스포츠 하이라이트 영상 제작에 적용 가능하여 플랫폼 콘텐츠 생성의 핵심 기술이 될 수 있습니다.

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
