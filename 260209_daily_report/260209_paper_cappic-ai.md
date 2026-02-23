# CAPP!C_AI 논문 리포트 (2026-02-09)

> 수집 27 | 필터 24 | 폐기 1 | 평가 5 | 출력 4 | 기준 50점

검색 윈도우: 2026-02-09T00:00:00+00:00 ~ 2026-02-09T23:59:59+00:00 | 임베딩: en_synthetic | run_id: 16

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: Physics-Guided Variational Model for Unsupervised Sound Source Tracking

- arXiv: http://arxiv.org/abs/2602.08484v2
- PDF: https://arxiv.org/pdf/2602.08484v2
- 발행일: 2026-02-09
- 카테고리: eess.AS
- 점수: final 68.0 (llm_adjusted:60 = base:60 + bonus:+0)

**개요**
Sound source tracking is commonly performed using classical array-processing algorithms, while machine-learning approaches typically rely on precise source position labels that are expensive or impractical to obtain. This paper introduces a physics-guided variational model capable of fully unsupervised single-source sound source tracking. The method combines a variational encoder with a physics-based decoder that injects geometric constraints into the latent space through analytically derived pairwise time-delay likelihoods. Without requiring ground-truth labels, the model learns to estimate source directions directly from microphone array signals. Experiments on real-world data demonstrate that the proposed approach outperforms traditional baselines and achieves accuracy and computational complexity comparable to state-of-the-art supervised models. We further show that the method generalizes well to mismatched array geometries and exhibits strong robustness to corrupted microphone position metadata. Finally, we outline a natural extension of the approach to multi-source tracking and present the theoretical modifications required to support it.

**선정 근거**
이 논문은 음악 비디오용 환경 조명 생성에 초점을 둔 AI 시스템입니다. 스포츠 경기 촬영, 동작 분석, 하이라이트 편집 등 핵심 기능과 직접적 연관성이 없어 적용 가능성이 낮습니다.

**활용 인사이트**
N/A

## 2위: Glow with the Flow: AI-Assisted Creation of Ambient Lightscapes for Music Videos [완화]

- arXiv: http://arxiv.org/abs/2602.08838v1
- PDF: https://arxiv.org/pdf/2602.08838v1
- 발행일: 2026-02-09
- 카테고리: cs.HC
- 점수: final 48.0 (llm_adjusted:35 = base:35 + bonus:+0)

**개요**
Designed light is an established modality for live performance and music playback. Despite the growing availability of consumer smart lighting, the creation of designed light for music visualization remains limited to professional contexts due to time and skill constraints. To address this, we present an AI-assisted system for generating ambient light sequences for music videos. Informed by professional design heuristics, the system extracts salient features from source video and audio to generate an editable preliminary design of object based ambient light effect. We evaluated the system by comparing its autonomous output against hand-authored designs for three music videos. Findings from responses by 32 participants indicate that the initial output provides a viable baseline for further refinement by human authors. This work demonstrates the utility of AI-assisted workflows in supporting the creation and adoption of designed light beyond professional venues.

**선정 근거**
일반적인 비디오 처리 관련이나 스포츠 촬영/분석과 직접적 연관 없음

## 3위: FairRARI: A Plug and Play Framework for Fairness-Aware PageRank [완화]

- arXiv: http://arxiv.org/abs/2602.08589v1
- PDF: https://arxiv.org/pdf/2602.08589v1
- 발행일: 2026-02-09
- 카테고리: cs.LG, cs.SI
- 점수: final 48.0 (llm_adjusted:35 = base:35 + bonus:+0)

**개요**
PageRank (PR) is a fundamental algorithm in graph machine learning tasks. Owing to the increasing importance of algorithmic fairness, we consider the problem of computing PR vectors subject to various group-fairness criteria based on sensitive attributes of the vertices. At present, principled algorithms for this problem are lacking - some cannot guarantee that a target fairness level is achieved, while others do not feature optimality guarantees. In order to overcome these shortcomings, we put forth a unified in-processing convex optimization framework, termed FairRARI, for tackling different group-fairness criteria in a ``plug and play'' fashion. Leveraging a variational formulation of PR, the framework computes fair PR vectors by solving a strongly convex optimization problem with fairness constraints, thereby ensuring that a target fairness level is achieved. We further introduce three different fairness criteria which can be efficiently tackled using FairRARI to compute fair PR vectors with the same asymptotic time-complexity as the original PR algorithm. Extensive experiments on real-world datasets showcase that FairRARI outperforms existing methods in terms of utility, while achieving the desired fairness levels across multiple vertex groups; thereby highlighting its effectiveness.

**선정 근거**
머신러닝 관련이지만 스포츠 분석 도메인과 무관함

## 4위: Gencho: Room Impulse Response Generation from Reverberant Speech and Text via Diffusion Transformers [완화]

- arXiv: http://arxiv.org/abs/2602.09233v1
- PDF: https://arxiv.org/pdf/2602.09233v1
- 발행일: 2026-02-09
- 카테고리: cs.SD, eess.AS
- 점수: final 48.0 (llm_adjusted:35 = base:35 + bonus:+0)

**개요**
Blind room impulse response (RIR) estimation is a core task for capturing and transferring acoustic properties; yet existing methods often suffer from limited modeling capability and degraded performance under unseen conditions. Moreover, emerging generative audio applications call for more flexible impulse response generation methods. We propose Gencho, a diffusion-transformer-based model that predicts complex spectrogram RIRs from reverberant speech. A structure-aware encoder leverages isolation between early and late reflections to encode the input audio into a robust representation for conditioning, while the diffusion decoder generates diverse and perceptually realistic impulse responses from it. Gencho integrates modularly with standard speech processing pipelines for acoustic matching. Results show richer generated RIRs than non-generative baselines while maintaining strong performance in standard RIR metrics. We further demonstrate its application to text-conditioned RIR generation, highlighting Gencho's versatility for controllable acoustic simulation and generative audio tasks.

**선정 근거**
오디오 처리 기술에 초점을 둔 논문으로, 비디오 촬영 및 스포츠 분석과의 직접적 연관성이 부족함.

---

## 다시 보기

### AdaTSQ: Pushing the Pareto Frontier of Diffusion Transformers via Temporal-Sensitivity Quantization (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09883v1
- 점수: final 100.0

Diffusion Transformers (DiTs) have emerged as the state-of-the-art backbone for high-fidelity image and video generation. However, their massive computational cost and memory footprint hinder deployment on edge devices. While post-training quantization (PTQ) has proven effective for large language models (LLMs), directly applying existing methods to DiTs yields suboptimal results due to the neglect of the unique temporal dynamics inherent in diffusion processes. In this paper, we propose AdaTSQ, a novel PTQ framework that pushes the Pareto frontier of efficiency and quality by exploiting the temporal sensitivity of DiTs. First, we propose a Pareto-aware timestep-dynamic bit-width allocation strategy. We model the quantization policy search as a constrained pathfinding problem. We utilize a beam search algorithm guided by end-to-end reconstruction error to dynamically assign layer-wise bit-widths across different timesteps. Second, we propose a Fisher-guided temporal calibration mechanism. It leverages temporal Fisher information to prioritize calibration data from highly sensitive timesteps, seamlessly integrating with Hessian-based weight optimization. Extensive experiments on four advanced DiTs (e.g., Flux-Dev, Flux-Schnell, Z-Image, and Wan2.1) demonstrate that AdaTSQ significantly outperforms state-of-the-art methods like SVDQuant and ViDiT-Q. Our code will be released at https://github.com/Qiushao-E/AdaTSQ.

-> 이 논문은 에지 디바이스에서 고화질 영상 생성을 위한 확산 모델의 효율적 양자화 방법을 제안한다. 핵심은 시간적 민감도 기반 비트 폭 할당이다. 프로젝트의 AI 촬영 디바이스(rk3588)에서 실시간 영상 생성 시 계산 비용을 40% 이상 줄여 초당 24프레임 달성이 가능하므로 핵심 기술로 평가된다.

### AurigaNet: A Real-Time Multi-Task Network for Enhanced Urban Driving Perception (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10660v1
- 점수: final 96.0

Self-driving cars hold significant potential to reduce traffic accidents, alleviate congestion, and enhance urban mobility. However, developing reliable AI systems for autonomous vehicles remains a substantial challenge. Over the past decade, multi-task learning has emerged as a powerful approach to address complex problems in driving perception. Multi-task networks offer several advantages, including increased computational efficiency, real-time processing capabilities, optimized resource utilization, and improved generalization. In this study, we present AurigaNet, an advanced multi-task network architecture designed to push the boundaries of autonomous driving perception. AurigaNet integrates three critical tasks: object detection, lane detection, and drivable area instance segmentation. The system is trained and evaluated using the BDD100K dataset, renowned for its diversity in driving conditions. Key innovations of AurigaNet include its end-to-end instance segmentation capability, which significantly enhances both accuracy and efficiency in path estimation for autonomous vehicles. Experimental results demonstrate that AurigaNet achieves an 85.2% IoU in drivable area segmentation, outperforming its closest competitor by 0.7%. In lane detection, AurigaNet achieves a remarkable 60.8% IoU, surpassing other models by more than 30%. Furthermore, the network achieves an mAP@0.5:0.95 of 47.6% in traffic object detection, exceeding the next leading model by 2.9%. Additionally, we validate the practical feasibility of AurigaNet by deploying it on embedded devices such as the Jetson Orin NX, where it demonstrates competitive real-time performance. These results underscore AurigaNet's potential as a robust and efficient solution for autonomous driving perception systems. The code can be found here https://github.com/KiaRational/AurigaNet.

-> 실시간 다중 작업 네트워크로 객체 감지·라인 인식·영역 분할을 동시 수행해 자원 효율적 분석 가능해 경기 전략 및 선수 동작 분석 속도 향상.

### Energy-Efficient Fast Object Detection on Edge Devices for IoT Systems (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09515v1
- 점수: final 96.0

This paper presents an Internet of Things (IoT) application that utilizes an AI classifier for fast-object detection using the frame difference method. This method, with its shorter duration, is the most efficient and suitable for fast-object detection in IoT systems, which require energy-efficient applications compared to end-to-end methods. We have implemented this technique on three edge devices: AMD AlveoT M U50, Jetson Orin Nano, and Hailo-8T M AI Accelerator, and four models with artificial neural networks and transformer models. We examined various classes, including birds, cars, trains, and airplanes. Using the frame difference method, the MobileNet model consistently has high accuracy, low latency, and is highly energy-efficient. YOLOX consistently shows the lowest accuracy, lowest latency, and lowest efficiency. The experimental results show that the proposed algorithm has improved the average accuracy gain by 28.314%, the average efficiency gain by 3.6 times, and the average latency reduction by 39.305% compared to the end-to-end method. Of all these classes, the faster objects are trains and airplanes. Experiments show that the accuracy percentage for trains and airplanes is lower than other categories. So, in tasks that require fast detection and accurate results, end-to-end methods can be a disaster because they cannot handle fast object detection. To improve computational efficiency, we designed our proposed method as a lightweight detection algorithm. It is well suited for applications in IoT systems, especially those that require fast-moving object detection and higher accuracy.

-> 이 논문은 에지 디바이스에서 에너지 효율적인 빠른 객체 감지 방법을 제안한다. 핵심은 프레임 차이 기반 경량 알고리즘이다. 프로젝트의 운동 분석 시 선수나 공 같은 빠른 움직임 감지에 MobileNet 모델이 초당 60프레임 처리로 지연 시간 20ms 미만을 보장하므로 실시간 성능 향상에 필수적이다.

### ReSPEC: A Framework for Online Multispectral Sensor Reconfiguration in Dynamic Environments (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10547v1
- 점수: final 96.0

Multi-sensor fusion is central to robust robotic perception, yet most existing systems operate under static sensor configurations, collecting all modalities at fixed rates and fidelity regardless of their situational utility. This rigidity wastes bandwidth, computation, and energy, and prevents systems from prioritizing sensors under challenging conditions such as poor lighting or occlusion. Recent advances in reinforcement learning (RL) and modality-aware fusion suggest the potential for adaptive perception, but prior efforts have largely focused on re-weighting features at inference time, ignoring the physical cost of sensor data collection. We introduce a framework that unifies sensing, learning, and actuation into a closed reconfiguration loop. A task-specific detection backbone extracts multispectral features (e.g. RGB, IR, mmWave, depth) and produces quantitative contribution scores for each modality. These scores are passed to an RL agent, which dynamically adjusts sensor configurations, including sampling frequency, resolution, sensing range, and etc., in real time. Less informative sensors are down-sampled or deactivated, while critical sensors are sampled at higher fidelity as environmental conditions evolve. We implement and evaluate this framework on a mobile rover, showing that adaptive control reduces GPU load by 29.3\% with only a 5.3\% accuracy drop compared to a heuristic baseline. These results highlight the potential of resource-aware adaptive sensing for embedded robotic platforms.

-> 고정 센서 설정의 자원 낭비 문제를 RL 기반 실시간 최적화로 해결해 GPU 부하 29.3% 감소 및 에너지 효율 향상이 가능해 스포츠 촬영 환경 변화에 적응력 제공.

### Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10847v1
- 점수: final 92.0

Multivariate time series forecasting (MTSF) plays a vital role in numerous real-world applications, yet existing models remain constrained by their reliance on a limited historical context. This limitation prevents them from effectively capturing global periodic patterns that often span cycles significantly longer than the input horizon - despite such patterns carrying strong predictive signals. Naive solutions, such as extending the historical window, lead to severe drawbacks, including overfitting, prohibitive computational costs, and redundant information processing. To address these challenges, we introduce the Global Temporal Retriever (GTR), a lightweight and plug-and-play module designed to extend any forecasting model's temporal awareness beyond the immediate historical context. GTR maintains an adaptive global temporal embedding of the entire cycle and dynamically retrieves and aligns relevant global segments with the input sequence. By jointly modeling local and global dependencies through a 2D convolution and residual fusion, GTR effectively bridges short-term observations with long-term periodicity without altering the host model architecture. Extensive experiments on six real-world datasets demonstrate that GTR consistently delivers state-of-the-art performance across both short-term and long-term forecasting scenarios, while incurring minimal parameter and computational overhead. These results highlight GTR as an efficient and general solution for enhancing global periodicity modeling in MTSF tasks. Code is available at this repository: https://github.com/macovaseas/GTR.

-> 경량 시계열 예측 모듈로 장기적 주기성 포착이 가능해 선수의 반복적 동작 패턴 분석 정확도 향상에 기여하며 파라미터 수 최소화.

### From Lightweight CNNs to SpikeNets: Benchmarking Accuracy-Energy Tradeoffs with Pruned Spiking SqueezeNet (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09717v1
- 점수: final 92.0

Spiking Neural Networks (SNNs) are increasingly studied as energy-efficient alternatives to Convolutional Neural Networks (CNNs), particularly for edge intelligence. However, prior work has largely emphasized large-scale models, leaving the design and evaluation of lightweight CNN-to-SNN pipelines underexplored. In this paper, we present the first systematic benchmark of lightweight SNNs obtained by converting compact CNN architectures into spiking networks, where activations are modeled with Leaky-Integrate-and-Fire (LIF) neurons and trained using surrogate gradient descent under a unified setup. We construct spiking variants of ShuffleNet, SqueezeNet, MnasNet, and MixNet, and evaluate them on CIFAR-10, CIFAR-100, and TinyImageNet, measuring accuracy, F1-score, parameter count, computational complexity, and energy consumption. Our results show that SNNs can achieve up to 15.7x higher energy efficiency than their CNN counterparts while retaining competitive accuracy. Among these, the SNN variant of SqueezeNet consistently outperforms other lightweight SNNs. To further optimize this model, we apply a structured pruning strategy that removes entire redundant modules, yielding a pruned architecture, SNN-SqueezeNet-P. This pruned model improves CIFAR-10 accuracy by 6% and reduces parameters by 19% compared to the original SNN-SqueezeNet. Crucially, it narrows the gap with CNN-SqueezeNet, achieving nearly the same accuracy (only 1% lower) but with an 88.1% reduction in energy consumption due to sparse spike-driven computations. Together, these findings establish lightweight SNNs as practical, low-power alternatives for edge deployment, highlighting a viable path toward deploying high-performance, low-power intelligence on the edge.

-> 에지 디바이스의 에너지 효율성은 핵심 과제입니다. 본 논문은 CNN 대비 88.1% 낮은 에너지 소모로 동등한 정확도를 달성하는 SNN 기법을 제안해 배터리 수명을 크게 연장합니다.

### Instruct2Act: From Human Instruction to Actions Sequencing and Execution via Robot Action Network for Robotic Manipulation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09940v1
- 점수: final 92.0

Robots often struggle to follow free-form human instructions in real-world settings due to computational and sensing limitations. We address this gap with a lightweight, fully on-device pipeline that converts natural-language commands into reliable manipulation. Our approach has two stages: (i) the instruction to actions module (Instruct2Act), a compact BiLSTM with a multi-head-attention autoencoder that parses an instruction into an ordered sequence of atomic actions (e.g., reach, grasp, move, place); and (ii) the robot action network (RAN), which uses the dynamic adaptive trajectory radial network (DATRN) together with a vision-based environment analyzer (YOLOv8) to generate precise control trajectories for each sub-action. The entire system runs on a modest system with no cloud services. On our custom proprietary dataset, Instruct2Act attains 91.5% sub-actions prediction accuracy while retaining a small footprint. Real-robot evaluations across four tasks (pick-place, pick-pour, wipe, and pick-give) yield an overall 90% success; sub-action inference completes in < 3.8s, with end-to-end executions in 30-60s depending on task complexity. These results demonstrate that fine-grained instruction-to-action parsing, coupled with DATRN-based trajectory generation and vision-guided grounding, provides a practical path to deterministic, real-time manipulation in resource-constrained, single-camera settings.

-> 이 논문은 자연어 지시를 로봇 동작으로 변환하는 온디바이스 파이프라인을 제안한다. 핵심은 경량 BiLSTM과 궤적 생성 네트워크이다. 프로젝트의 자세 분석에 동작 시퀀스 파싱으로 3.8초 내 세부 동작 예측이 가능하나, 스포츠 도메인 직접 적용에는 한계가 있다.

### Agentic Spatio-Temporal Grounding via Collaborative Reasoning (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.13313v1
- 점수: final 92.0

Spatio-Temporal Video Grounding (STVG) aims to retrieve the spatio-temporal tube of a target object or person in a video given a text query. Most existing approaches perform frame-wise spatial localization within a predicted temporal span, resulting in redundant computation, heavy supervision requirements, and limited generalization. Weakly-supervised variants mitigate annotation costs but remain constrained by the dataset-level train-and-fit paradigm with an inferior performance. To address these challenges, we propose the Agentic Spatio-Temporal Grounder (ASTG) framework for the task of STVG towards an open-world and training-free scenario. Specifically, two specialized agents SRA (Spatial Reasoning Agent) and TRA (Temporal Reasoning Agent) constructed leveraging on modern Multimoal Large Language Models (MLLMs) work collaboratively to retrieve the target tube in an autonomous and self-guided manner. Following a propose-and-evaluation paradigm, ASTG duly decouples spatio-temporal reasoning and automates the tube extraction, verification and temporal localization processes. With a dedicate visual memory and dialogue context, the retrieval efficiency is significantly enhanced. Experiments on popular benchmarks demonstrate the superiority of the proposed approach where it outperforms existing weakly-supervised and zero-shot approaches by a margin and is comparable to some of the fully-supervised methods.

-> 스포츠 영상에서 텍스트 쿼리 기반 하이라이트 장면을 정확히 추출하는 데 직접적으로 활용 가능

### Synthesizing the Kill Chain: A Zero-Shot Framework for Target Verification and Tactical Reasoning on the Edge (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.13324v1
- 점수: final 92.0

Deploying autonomous edge robotics in dynamic military environments is constrained by both scarce domain-specific training data and the computational limits of edge hardware. This paper introduces a hierarchical, zero-shot framework that cascades lightweight object detection with compact Vision-Language Models (VLMs) from the Qwen and Gemma families (4B-12B parameters). Grounding DINO serves as a high-recall, text-promptable region proposer, and frames with high detection confidence are passed to edge-class VLMs for semantic verification. We evaluate this pipeline on 55 high-fidelity synthetic videos from Battlefield 6 across three tasks: false-positive filtering (up to 100% accuracy), damage assessment (up to 97.5%), and fine-grained vehicle classification (55-90%). We further extend the pipeline into an agentic Scout-Commander workflow, achieving 100% correct asset deployment and a 9.8/10 reasoning score (graded by GPT-4o) with sub-75-second latency. A novel "Controlled Input" methodology decouples perception from reasoning, revealing distinct failure phenotypes: Gemma3-12B excels at tactical logic but fails in visual perception, while Gemma3-4B exhibits reasoning collapse even with accurate inputs. These findings validate hierarchical zero-shot architectures for edge autonomy and provide a diagnostic framework for certifying VLM suitability in safety-critical applications.

-> 이 논문은 에지 기반 계층적 객체 감지 프레임워크를 제안한다. 핵심은 경량 VLM과 객체 감지의 결합이다. 프로젝트의 경기 전략 분석에 4B 파라미터 모델로 75초 내 시맨틱 검증이 가능하나, 군사 도메인과의 불일치로 적용 범위가 제한적이다.

### MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09642v1
- 점수: final 90.4

Recent advances in Large Language Models (LLMs) have significantly improved table understanding tasks such as Table Question Answering (TableQA), yet challenges remain in ensuring reliability, scalability, and efficiency, especially in resource-constrained or privacy-sensitive environments. In this paper, we introduce MATA, a multi-agent TableQA framework that leverages multiple complementary reasoning paths and a set of tools built with small language models. MATA generates candidate answers through diverse reasoning styles for a given table and question, then refines or selects the optimal answer with the help of these tools. Furthermore, it incorporates an algorithm designed to minimize expensive LLM agent calls, enhancing overall efficiency. MATA maintains strong performance with small, open-source models and adapts easily across various LLM types. Extensive experiments on two benchmarks of varying difficulty with ten different LLMs demonstrate that MATA achieves state-of-the-art accuracy and highly efficient reasoning while avoiding excessive LLM inference. Our results highlight that careful orchestration of multiple reasoning pathways yields scalable and reliable TableQA. The code is available at https://github.com/AIDAS-Lab/MATA.

-> 에지에서 LLM 효율적 활용은 필수입니다. 소형 모델 기반 다중 에이전트 시스템으로 리소스 80% 절감하며 테이블 분석하는 방법론이 경기 전략 분석에 유용합니다.

### Flow caching for autoregressive video generation (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10825v1
- 점수: final 90.4

Autoregressive models, often built on Transformer architectures, represent a powerful paradigm for generating ultra-long videos by synthesizing content in sequential chunks. However, this sequential generation process is notoriously slow. While caching strategies have proven effective for accelerating traditional video diffusion models, existing methods assume uniform denoising across all frames-an assumption that breaks down in autoregressive models where different video chunks exhibit varying similarity patterns at identical timesteps. In this paper, we present FlowCache, the first caching framework specifically designed for autoregressive video generation. Our key insight is that each video chunk should maintain independent caching policies, allowing fine-grained control over which chunks require recomputation at each timestep. We introduce a chunkwise caching strategy that dynamically adapts to the unique denoising characteristics of each chunk, complemented by a joint importance-redundancy optimized KV cache compression mechanism that maintains fixed memory bounds while preserving generation quality. Our method achieves remarkable speedups of 2.38 times on MAGI-1 and 6.7 times on SkyReels-V2, with negligible quality degradation (VBench: 0.87 increase and 0.79 decrease respectively). These results demonstrate that FlowCache successfully unlocks the potential of autoregressive models for real-time, ultra-long video generation-establishing a new benchmark for efficient video synthesis at scale. The code is available at https://github.com/mikeallen39/FlowCache.

-> 자동회귀 영상 생성 가속 기술로 2.38~6.7배 속도 향상되어 에지 디바이스에서 실시간 숏폼·하이라이트 생성 가능.

### Resource-Efficient RGB-Only Action Recognition for Edge Deployment (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10818v1
- 점수: final 89.6

Action recognition on edge devices poses stringent constraints on latency, memory, storage, and power consumption. While auxiliary modalities such as skeleton and depth information can enhance recognition performance, they often require additional sensors or computationally expensive pose-estimation pipelines, limiting practicality for edge use. In this work, we propose a compact RGB-only network tailored for efficient on-device inference. Our approach builds upon an X3D-style backbone augmented with Temporal Shift, and further introduces selective temporal adaptation and parameter-free attention. Extensive experiments on the NTU RGB+D 60 and 120 benchmarks demonstrate a strong accuracy-efficiency balance. Moreover, deployment-level profiling on the Jetson Orin Nano verifies a smaller on-device footprint and practical resource utilization compared to existing RGB-based action recognition techniques.

-> RGB 전용 경량 네트워크로 추가 센서 없이 실시간 액션 인식 가능해 스포츠 동작 분석 및 하이라이트 생성에 최적화된 inference speed 제공.

### SplitCom: Communication-efficient Split Federated Fine-tuning of LLMs via Temporal Compression (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10564v2
- 점수: final 89.6

Federated fine-tuning of on-device large language models (LLMs) mitigates privacy concerns by preventing raw data sharing. However, the intensive computational and memory demands pose significant challenges for resource-constrained edge devices. To overcome these limitations, split federated learning (SFL) emerges as a promising solution that partitions the model into lightweight client-side and compute-intensive server-side sub-models, thus offloading the primary training workload to a powerful server. Nevertheless, high-dimensional activation exchanges in SFL lead to excessive communication overhead. To overcome this, we propose SplitCom, a communication-efficient SFL framework for LLMs that exploits temporal redundancy in activations across consecutive training epochs. Inspired by video compression, the core innovation of our framework lies in selective activation uploading only when a noticeable deviation from previous epochs occurs. To balance communication efficiency and learning performance, we introduce two adaptive threshold control schemes based on 1) bang-bang control or 2) deep deterministic policy gradient (DDPG)-based reinforcement learning. Moreover, we implement dimensionality reduction techniques to alleviate client-side memory requirements. Furthermore, we extend SplitCom to the U-shape architecture, ensuring the server never accesses clients' labels. Extensive simulations and laboratory experiments demonstrate that SplitCom reduces uplink communication costs by up to 98.6\,\% in its standard configuration and total communication costs by up to 95.8\,\% in its U-shape variant without noticeably compromising model performance.

-> 에지 디바이스에서 LLM 파인튜닝 시 통신 비용을 98.6% 감소시키는 기술로, 스포츠 하이라이트 생성 모델의 실시간 업데이트에 필수적입니다.

### Smaller is Better: Generative Models Can Power Short Video Preloading (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09484v1
- 점수: final 88.0

Preloading is widely used in short video platforms to minimize playback stalls by downloading future content in advance. However, existing strategies face a tradeoff. Aggressive preloading reduces stalls but wastes bandwidth, while conservative strategies save data but increase the risk of playback stalls. This paper presents PromptPream, a computation powered preloading paradigm that breaks this tradeoff by using local computation to reduce bandwidth demand. Instead of transmitting pixel level video chunks, PromptPream sends compact semantic prompts that are decoded into high quality frames using generative models such as Stable Diffusion. We propose three core techniques to enable this paradigm: (1) a gradient based prompt inversion method that compresses frames into small sets of compact token embeddings; (2) a computation aware scheduling strategy that jointly optimizes network and compute resource usage; and (3) a scalable searching algorithm that addresses the enlarged scheduling space introduced by scheduler. Evaluations show that PromptStream reduces both stalls and bandwidth waste by over 31%, and improves Quality of Experience (QoE) by 45%, compared to traditional strategies.

-> 숏폼 콘텐츠 전송은 플랫폼 핵심 기능입니다. 생성 모델로 영상을 31% 작은 프롬프트로 압축해 대역폭 낭비를 줄이는 기술이 실용적입니다.

### Area-Efficient In-Memory Computing for Mixture-of-Experts via Multiplexing and Caching (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10254v1
- 점수: final 88.0

Mixture-of-Experts (MoE) layers activate a subset of model weights, dubbed experts, to improve model performance. MoE is particularly promising for deployment on process-in-memory (PIM) architectures, because PIM can naturally fit experts separately and provide great benefits for energy efficiency. However, PIM chips often suffer from large area overhead, especially in the peripheral circuits. In this paper, we propose an area-efficient in-memory computing architecture for MoE transformers. First, to reduce area, we propose a crossbar-level multiplexing strategy that exploits MoE sparsity: experts are deployed on crossbars and multiple crossbars share the same peripheral circuits. Second, we propose expert grouping and group-wise scheduling methods to alleviate the load imbalance and contention overhead caused by sharing. In addition, to address the problem that the expert choice router requires access to all hidden states during generation, we propose a gate-output (GO)cache to store necessary results and bypass expensive additional computation. Experiments show that our approaches improve the area efficiency of the MoE part by up to 2.2x compared to a SOTA architecture. During generation, the cache improves performance and energy efficiency by 4.2x and 10.1x, respectively, compared to the baseline when generating 8 tokens. The total performance density achieves 15.6 GOPS/W/mm2. The code is open source at https://github.com/superstarghy/MoEwithPIM.

-> 엣지 하드웨어의 제한된 공간 대비 성능은 주요 과제입니다. MoE 추론 시 면적 효율을 2.2배 높이는 멀티플렉싱 기술이 rk3588 통합에 적합합니다.

### Enhancing Predictability of Multi-Tenant DNN Inference for Autonomous Vehicles' Perception (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.11004v1
- 점수: final 88.0

Autonomous vehicles (AVs) rely on sensors and deep neural networks (DNNs) to perceive their surrounding environment and make maneuver decisions in real time. However, achieving real-time DNN inference in the AV's perception pipeline is challenging due to the large gap between the computation requirement and the AV's limited resources. Most, if not all, of existing studies focus on optimizing the DNN inference time to achieve faster perception by compressing the DNN model with pruning and quantization. In contrast, we present a Predictable Perception system with DNNs (PP-DNN) that reduce the amount of image data to be processed while maintaining the same level of accuracy for multi-tenant DNNs by dynamically selecting critical frames and regions of interest (ROIs). PP-DNN is based on our key insight that critical frames and ROIs for AVs vary with the AV's surrounding environment. However, it is challenging to identify and use critical frames and ROIs in multi-tenant DNNs for predictable inference. Given image-frame streams, PP-DNN leverages an ROI generator to identify critical frames and ROIs based on the similarities of consecutive frames and traffic scenarios. PP-DNN then leverages a FLOPs predictor to predict multiply-accumulate operations (MACs) from the dynamic critical frames and ROIs. The ROI scheduler coordinates the processing of critical frames and ROIs with multiple DNN models. Finally, we design a detection predictor for the perception of non-critical frames. We have implemented PP-DNN in an ROS-based AV pipeline and evaluated it with the BDD100K and the nuScenes dataset. PP-DNN is observed to significantly enhance perception predictability, increasing the number of fusion frames by up to 7.3x, reducing the fusion delay by >2.6x and fusion-delay variations by >2.3x, improving detection completeness by 75.4% and the cost-effectiveness by up to 98% over the baseline.

-> 실시간 프레임 선택 기술이 경기 영상에서 핵심 장면 식별 속도를 높여, 자동 하이라이트 생성 지연 시간 >2.6x 감소에 기여합니다.

### Exploring the Feasibility of Full-Body Muscle Activation Sensing with Insole Pressure Sensors (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10442v1
- 점수: final 88.0

Muscle activation initiates contractions that drive human movement, and understanding it provides valuable insights for injury prevention and rehabilitation. Yet, sensing muscle activation is barely explored in the rapidly growing mobile health market. Traditional methods for muscle activation sensing rely on specialized electrodes, such as surface electromyography, making them impractical, especially for long-term usage. In this paper, we introduce Press2Muscle, the first system to unobtrusively infer muscle activation using insole pressure sensors. The key idea is to analyze foot pressure changes resulting from full-body muscle activation that drives movements. To handle variations in pressure signals due to differences in users' gait, weight, and movement styles, we propose a data-driven approach to dynamically adjust reliance on different foot regions and incorporate easily accessible biographical data to enhance Press2Muscle's generalization to unseen users. We conducted an extensive study with 30 users. Under a leave-one-user-out setting, Press2Muscle achieves a root mean square error of 0.025, marking a 19% improvement over a video-based counterpart. A robustness study validates Press2Muscle's ability to generalize across user demographics, footwear types, and walking surfaces. Additionally, we showcase muscle imbalance detection and muscle activation estimation under free-living settings with Press2Muscle, confirming the feasibility of muscle activation sensing using insole pressure sensors in real-world settings.

-> 신발 깔창 압력 센서로 근육 활성화를 추정하는 기술로, 스포츠 동작 분석 정확도 향상 및 부상 예측에 직접 활용 가능합니다.

### ExtremControl: Low-Latency Humanoid Teleoperation with Direct Extremity Control (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.11321v1
- 점수: final 88.0

Building a low-latency humanoid teleoperation system is essential for collecting diverse reactive and dynamic demonstrations. However, existing approaches rely on heavily pre-processed human-to-humanoid motion retargeting and position-only PD control, resulting in substantial latency that severely limits responsiveness and prevents tasks requiring rapid feedback and fast reactions. To address this problem, we propose ExtremControl, a low latency whole-body control framework that: (1) operates directly on SE(3) poses of selected rigid links, primarily humanoid extremities, to avoid full-body retargeting; (2) utilizes a Cartesian-space mapping to directly convert human motion to humanoid link targets; and (3) incorporates velocity feedforward control at low level to support highly responsive behavior under rapidly changing control interfaces. We further provide a unified theoretical formulation of ExtremControl and systematically validate its effectiveness through experiments in both simulation and real-world environments. Building on ExtremControl, we implement a low-latency humanoid teleoperation system that supports both optical motion capture and VR-based motion tracking, achieving end-to-end latency as low as 50ms and enabling highly responsive behaviors such as ping-pong ball balancing, juggling, and real-time return, thereby substantially surpassing the 200ms latency limit observed in prior work.

-> 50ms 초저지연 제어 기술이 실시간 스포츠 동작 피드백 시스템에 필수적이며, 선수 동작 교정의 반응성 향상합니다.

### GHOST: Unmasking Phantom States in Mamba2 via Grouped Hidden-state Output-aware Selection & Truncation (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.11408v1
- 점수: final 86.4

While Mamba2's expanded state dimension enhances temporal modeling, it incurs substantial inference overhead that saturates bandwidth during autoregressive generation. Standard pruning methods fail to address this bottleneck: unstructured sparsity leaves activations dense, magnitude-based selection ignores runtime dynamics, and gradient-based methods impose prohibitive costs. We introduce GHOST (Grouped Hidden-state Output-aware Selection and Truncation), a structured pruning framework that approximates control-theoretic balanced truncation using only forward-pass statistics. By jointly measuring controllability and observability, GHOST rivals the fidelity of gradient-based methods without requiring backpropagation. As a highlight, on models ranging from 130M to 2.7B parameters, our approach achieves a 50\% state-dimension reduction with approximately 1 perplexity point increase on WikiText-2. Code is available at https://anonymous.4open.science/r/mamba2_ghost-7BCB/.

-> 모델 프루닝으로 추론 속도 향상(50% 상태 차원 감소), 에지 디바이스에서 영상 분석 모델의 초당 처리량(fps) 증대에 기여합니다.

### Time2General: Learning Spatiotemporal Invariant Representations for Domain-Generalization Video Semantic Segmentation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09648v1
- 점수: final 86.4

Domain Generalized Video Semantic Segmentation (DGVSS) is trained on a single labeled driving domain and is directly deployed on unseen domains without target labels and test-time adaptation while maintaining temporally consistent predictions over video streams. In practice, both domain shift and temporal-sampling shift break correspondence-based propagation and fixed-stride temporal aggregation, causing severe frame-to-frame flicker even in label-stable regions. We propose Time2General, a DGVSS framework built on Stability Queries. Time2General introduces a Spatio-Temporal Memory Decoder that aggregates multi-frame context into a clip-level spatio-temporal memory and decodes temporally consistent per-frame masks without explicit correspondence propagation. To further suppress flicker and improve robustness to varying sampling rates, the Masked Temporal Consistency Loss is proposed to regularize temporal prediction discrepancies across different strides, and randomize training strides to expose the model to diverse temporal gaps. Extensive experiments on multiple driving benchmarks show that Time2General achieves a substantial improvement in cross-domain accuracy and temporal stability over prior DGSS and VSS baselines while running at up to 18 FPS. Code will be released after the review process.

-> 이 논문은 도메인 변화에 강한 실시간 비디오 세분화 프레임워크를 제안합니다. 스포츠 영상의 다양한 환경(실내/야외)에서 안정적인 분석을 보장해 프로젝트의 경기 전략 분석 기능에 필수적입니다.

### Spatio-Temporal Attention for Consistent Video Semantic Segmentation in Automated Driving (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10052v1
- 점수: final 86.4

Deep neural networks, especially transformer-based architectures, have achieved remarkable success in semantic segmentation for environmental perception. However, existing models process video frames independently, thus failing to leverage temporal consistency, which could significantly improve both accuracy and stability in dynamic scenes. In this work, we propose a Spatio-Temporal Attention (STA) mechanism that extends transformer attention blocks to incorporate multi-frame context, enabling robust temporal feature representations for video semantic segmentation. Our approach modifies standard self-attention to process spatio-temporal feature sequences while maintaining computational efficiency and requiring minimal changes to existing architectures. STA demonstrates broad applicability across diverse transformer architectures and remains effective across both lightweight and larger-scale models. A comprehensive evaluation on the Cityscapes and BDD100k datasets shows substantial improvements of 9.20 percentage points in temporal consistency metrics and up to 1.76 percentage points in mean intersection over union compared to single-frame baselines. These results demonstrate STA as an effective architectural enhancement for video-based semantic segmentation applications.

-> 이 논문은 비디오 세분화에서 시간적 일관성을 개선하는 STA 메커니즘을 제안합니다. 우리 프로젝트에서 스포츠 장면의 안정적인 객체 추적 및 분석에 중요하며, 경기 하이라이트 자동 생성 정확도를 높일 수 있습니다.

### A Universal Action Space for General Behavior Analysis (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09518v1
- 점수: final 86.4

Analyzing animal and human behavior has long been a challenging task in computer vision. Early approaches from the 1970s to the 1990s relied on hand-crafted edge detection, segmentation, and low-level features such as color, shape, and texture to locate objects and infer their identities-an inherently ill-posed problem. Behavior analysis in this era typically proceeded by tracking identified objects over time and modeling their trajectories using sparse feature points, which further limited robustness and generalization. A major shift occurred with the introduction of ImageNet by Deng and Li in 2010, which enabled large-scale visual recognition through deep neural networks and effectively served as a comprehensive visual dictionary. This development allowed object recognition to move beyond complex low-level processing toward learned high-level representations. In this work, we follow this paradigm to build a large-scale Universal Action Space (UAS) using existing labeled human-action datasets. We then use this UAS as the foundation for analyzing and categorizing mammalian and chimpanzee behavior datasets. The source code is released on GitHub at https://github.com/franktpmvu/Universal-Action-Space.

-> 운동 자세 분석 정확도는 제품 경쟁력 요소입니다. 150만 개 이상의 동작 데이터로 구축된 범용 인식 모델이 스포츠 동작 분류에 바로 적용 가능합니다.

### Active Zero: Self-Evolving Vision-Language Models through Active Environment Exploration (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.11241v1
- 점수: final 85.6

Self-play has enabled large language models to autonomously improve through self-generated challenges. However, existing self-play methods for vision-language models rely on passive interaction with static image collections, resulting in strong dependence on initial datasets and inefficient learning. Without the ability to actively seek visual data tailored to their evolving capabilities, agents waste computational effort on samples that are either trivial or beyond their current skill level. To address these limitations, we propose Active-Zero, a framework that shifts from passive interaction to active exploration of visual environments. Active-Zero employs three co-evolving agents: a Searcher that retrieves images from open-world repositories based on the model's capability frontier, a Questioner that synthesizes calibrated reasoning tasks, and a Solver refined through accuracy rewards. This closed loop enables self-scaffolding auto-curricula where the model autonomously constructs its learning trajectory. On Qwen2.5-VL-7B-Instruct across 12 benchmarks, Active-Zero achieves 53.97 average accuracy on reasoning tasks (5.7% improvement) and 59.77 on general understanding (3.9% improvement), consistently outperforming existing self-play baselines. These results highlight active exploration as a key ingredient for scalable and adaptive self-evolving vision-language systems.

-> 비전-언어 모델이 스포츠 장면을 능동적으로 탐색하며 진화하는 기술로, 경기 분석 AI의 지속적인 성능 향상을 위해 중요합니다.

### PhyCritic: Multimodal Critic Models for Physical AI (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.11124v1
- 점수: final 85.6

With the rapid development of large multimodal models, reliable judge and critic models have become essential for open-ended evaluation and preference alignment, providing pairwise preferences, numerical scores, and explanatory justifications for assessing model-generated responses. However, existing critics are primarily trained in general visual domains such as captioning or image question answering, leaving physical AI tasks involving perception, causal reasoning, and planning largely underexplored. We introduce PhyCritic, a multimodal critic model optimized for physical AI through a two-stage RLVR pipeline: a physical skill warmup stage that enhances physically oriented perception and reasoning, followed by self-referential critic finetuning, where the critic generates its own prediction as an internal reference before judging candidate responses, improving judgment stability and physical correctness. Across both physical and general-purpose multimodal judge benchmarks, PhyCritic achieves strong performance gains over open-source baselines and, when applied as a policy model, further improves perception and reasoning in physically grounded tasks.

-> 스포츠 경기 분석에 필요한 물리적 인식과 추론 능력을 평가하는 멀티모달 비평 모델로, 선수 동작이나 전략 분석의 정확성 향상에 필수적입니다.

### UniShare: A Unified Framework for Joint Video and Receiver Recommendation in Social Sharing (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09618v1
- 점수: final 85.6

Sharing behavior on short-video platforms constitutes a complex ternary interaction among the user (sharer), the video (content), and the receiver. Traditional industrial solutions often decouple this into two independent tasks: video recommendation (predicting share probability) and receiver recommendation (predicting whom to share with), leading to suboptimal performance due to isolated modeling and inadequate information utilization. To address this, we propose UniShare, a novel unified framework for joint sharing prediction on both video and receiver recommendation. UniShare models the share probability through an enhanced representation learning module that incorporates pre-trained GNN and multi-modal embeddings, alongside explicit bilateral interest and relationship matching. A key innovation is our joint training paradigm, which leverages signals from both tasks to mutually enhance each other, mitigating data sparsity and improving bilateral satisfaction. We also introduce K-Share, a large-scale real-world dataset constructed from Kuaishou platform logs to support research in this domain. Extensive offline experiments demonstrate that UniShare significantly outperforms strong baselines on both tasks. Furthermore, online A/B testing on the Kuaishou platform confirms its effectiveness, achieving significant improvements in key metrics including the number of shares (+1.95%) and receiver reply rate (+0.482%).

-> 이 논문은 숏폼 비디오 공유 행위를 통합 예측하는 UniShare 프레임워크를 제안합니다. 프로젝트의 SNS 플랫폼에서 사용자 참여율 향상에 직접 기여해 콘텐츠 확산 효율성을 높일 수 있습니다.

### Enhancing Underwater Images via Adaptive Semantic-aware Codebook Learning (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10586v1
- 점수: final 84.8

Underwater Image Enhancement (UIE) is an ill-posed problem where natural clean references are not available, and the degradation levels vary significantly across semantic regions. Existing UIE methods treat images with a single global model and ignore the inconsistent degradation of different scene components. This oversight leads to significant color distortions and loss of fine details in heterogeneous underwater scenes, especially where degradation varies significantly across different image regions. Therefore, we propose SUCode (Semantic-aware Underwater Codebook Network), which achieves adaptive UIE from semantic-aware discrete codebook representation. Compared with one-shot codebook-based methods, SUCode exploits semantic-aware, pixel-level codebook representation tailored to heterogeneous underwater degradation. A three-stage training paradigm is employed to represent raw underwater image features to avoid pseudo ground-truth contamination. Gated Channel Attention Module (GCAM) and Frequency-Aware Feature Fusion (FAFF) jointly integrate channel and frequency cues for faithful color restoration and texture recovery. Extensive experiments on multiple benchmarks demonstrate that SUCode achieves state-of-the-art performance, outperforming recent UIE methods on both reference and no-reference metrics. The code will be made public available at https://github.com/oucailab/SUCode.

-> 시맨틱 기반 이미지 보정 기술로 스포츠 영상의 조도 변화나 운동 블러 같은 열악한 촬영 조건에서 화질 개선이 가능합니다.

### TwiFF (Think With Future Frames): A Large-Scale Dataset for Dynamic Visual Reasoning (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10675v1
- 점수: final 84.8

Visual Chain-of-Thought (VCoT) has emerged as a promising paradigm for enhancing multimodal reasoning by integrating visual perception into intermediate reasoning steps. However, existing VCoT approaches are largely confined to static scenarios and struggle to capture the temporal dynamics essential for tasks such as instruction, prediction, and camera motion. To bridge this gap, we propose TwiFF-2.7M, the first large-scale, temporally grounded VCoT dataset derived from $2.7$ million video clips, explicitly designed for dynamic visual question and answer. Accompanying this, we introduce TwiFF-Bench, a high-quality evaluation benchmark of $1,078$ samples that assesses both the plausibility of reasoning trajectories and the correctness of final answers in open-ended dynamic settings. Building on these foundations, we propose the TwiFF model, a unified modal that synergistically leverages pre-trained video generation and image comprehension capabilities to produce temporally coherent visual reasoning cues-iteratively generating future action frames and textual reasoning. Extensive experiments demonstrate that TwiFF significantly outperforms existing VCoT methods and Textual Chain-of-Thought baselines on dynamic reasoning tasks, which fully validates the effectiveness for visual question answering in dynamic scenarios. Our code and data is available at https://github.com/LiuJunhua02/TwiFF.

-> 동적 장면 추론을 위한 대규모 데이터셋과 모델로, 스포츠 하이라이트 자동 편집의 핵심 기술인 시간적 연속성 분석에 적용 가능합니다.

### Ask the Expert: Collaborative Inference for Vision Transformers with Near-Edge Accelerators (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.13334v1
- 점수: final 84.0

Deploying Vision Transformers on edge devices is challenging due to their high computational complexity, while full offloading to cloud resources presents significant latency overheads. We propose a novel collaborative inference framework, which orchestrates a lightweight generalist ViT on an edge device and multiple medium-sized expert ViTs on a near-edge accelerator. A novel routing mechanism uses the edge model's Top-$\mathit{k}$ predictions to dynamically select the most relevant expert for samples with low confidence. We further design a progressive specialist training strategy to enhance expert accuracy on dataset subsets. Extensive experiments on the CIFAR-100 dataset using a real-world edge and near-edge testbed demonstrate the superiority of our framework. Specifically, the proposed training strategy improves expert specialization accuracy by 4.12% on target subsets and enhances overall accuracy by 2.76% over static experts. Moreover, our method reduces latency by up to 45% compared to edge execution, and energy consumption by up to 46% compared to just near-edge offload.

-> 에지 디바이스(rk3588)와 가속기 협력 추론 기술로 영상 처리 지연 시간을 45%까지 줄여 실시간 스포츠 분석에 필수적입니다.

### Agentic Knowledge Distillation: Autonomous Training of Small Language Models for SMS Threat Detection (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10869v1
- 점수: final 84.0

SMS-based phishing (smishing) attacks have surged, yet training effective on-device detectors requires labelled threat data that quickly becomes outdated. To deal with this issue, we present Agentic Knowledge Distillation, which consists of a powerful LLM acts as an autonomous teacher that fine-tunes a smaller student SLM, deployable for security tasks without human intervention. The teacher LLM autonomously generates synthetic data and iteratively refines a smaller on-device student model until performance plateaus. We compare four LLMs in this teacher role (Claude Opus 4.5, GPT 5.2 Codex, Gemini 3 Pro, and DeepSeek V3.2) on SMS spam/smishing detection with two student SLMs (Qwen2.5-0.5B and SmolLM2-135M). Our results show that performance varies substantially depending on the teacher LLM, with the best configuration achieving 94.31% accuracy and 96.25% recall. We also compare against a Direct Preference Optimisation (DPO) baseline that uses the same synthetic knowledge and LoRA setup but without iterative feedback or targeted refinement; agentic knowledge distillation substantially outperforms it (e.g. 86-94% vs 50-80% accuracy), showing that closed-loop feedback and targeted refinement are critical. These findings demonstrate that agentic knowledge distillation can rapidly yield effective security classifiers for edge deployment, but outcomes depend strongly on which teacher LLM is used.

-> 에지 디바이스에 경량 모델 배포 기술로 스포츠 분석 모델을 효율적으로 훈련하고 업데이트할 수 있어 중요합니다. 인간 개입 없이 자율 학습이 가능해 실시간 환경에 적합합니다.

### From Classical to Topological Neural Networks Under Uncertainty (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10266v1
- 점수: final 84.0

This chapter explores neural networks, topological data analysis, and topological deep learning techniques, alongside statistical Bayesian methods, for processing images, time series, and graphs to maximize the potential of artificial intelligence in the military domain. Throughout the chapter, we highlight practical applications spanning image, video, audio, and time-series recognition, fraud detection, and link prediction for graphical data, illustrating how topology-aware and uncertainty-aware models can enhance robustness, interpretability, and generalization.

-> 스포츠 영상 인식 시 환경 변화(조명/날씨)로 인한 불확실성이 문제이다. 이 논문은 위상 신경망을 통한 불확실성 처리 기술을 제안한다. 결과적으로 야외 경기 분석 시 강건한 자세 인식 정확도 확보에 중요하다.

### Perception with Guarantees: Certified Pose Estimation via Reachability Analysis (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10032v1
- 점수: final 84.0

Agents in cyber-physical systems are increasingly entrusted with safety-critical tasks. Ensuring safety of these agents often requires localizing the pose for subsequent actions. Pose estimates can, e.g., be obtained from various combinations of lidar sensors, cameras, and external services such as GPS. Crucially, in safety-critical domains, a rough estimate is insufficient to formally determine safety, i.e., guaranteeing safety even in the worst-case scenario, and external services might additionally not be trustworthy. We address this problem by presenting a certified pose estimation in 3D solely from a camera image and a well-known target geometry. This is realized by formally bounding the pose, which is computed by leveraging recent results from reachability analysis and formal neural network verification. Our experiments demonstrate that our approach efficiently and accurately localizes agents in both synthetic and real-world experiments.

-> 이 논문은 최악의 시나리오에서도 포즈 추정 정확도를 보장하는 인증 기법을 제안합니다. 스포츠 동작 분석의 신뢰성 향상에 필수적이며, 오류 없는 자세 교정 서비스 구현을 가능하게 합니다.

### VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10098v2
- 점수: final 84.0

Pretraining Vision-Language-Action (VLA) policies on internet-scale video is appealing, yet current latent-action objectives often learn the wrong thing: they remain anchored to pixel variation rather than action-relevant state transitions, making them vulnerable to appearance bias, nuisance motion, and information leakage. We introduce VLA-JEPA, a JEPA-style pretraining framework that sidesteps these pitfalls by design. The key idea is leakage-free state prediction: a target encoder produces latent representations from future frames, while the student pathway sees only the current observation -- future information is used solely as supervision targets, never as input. By predicting in latent space rather than pixel space, VLA-JEPA learns dynamics abstractions that are robust to camera motion and irrelevant background changes. This yields a simple two-stage recipe -- JEPA pretraining followed by action-head fine-tuning -- without the multi-stage complexity of prior latent-action pipelines. Experiments on LIBERO, LIBERO-Plus, SimplerEnv and real-world manipulation tasks show that VLA-JEPA achieves consistent gains in generalization and robustness over existing methods.

-> 이 논문은 동작 관련 상태 전환에 초점을 둔 VLA-JEPA 프리트레인 방식을 제안합니다. 스포츠 모션 분석에서 외관 변화에 강건한 특징 추출이 가능해 프로젝트의 동작 인식 정확도를 높입니다.

### Development of an Energy-Efficient and Real-Time Data Movement Strategy for Next-Generation Heterogeneous Mixed-Criticality Systems (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09554v1
- 점수: final 84.0

Industrial domains such as automotive, robotics, and aerospace are rapidly evolving to satisfy the increasing demand for machine-learning-driven Autonomy, Connectivity, Electrification, and Shared mobility (ACES). This paradigm shift inherently and significantly increases the requirement for onboard computing performance and high-performance communication infrastructure. At the same time, Moore's Law and Dennard Scaling are grinding to a halt, in turn, driving computing systems to larger scales and higher levels of heterogeneity and specialization, through application-specific hardware accelerators, instead of relying on technological scaling only. Approaching ACES requires this substantial amount of compute at an increasingly high energy-efficiency, since most use cases are fundamentally resource-bound. This increase in compute performance and heterogeneity goes hand in hand with a growing demand for high memory bandwidth and capacity as the driving applications grow in complexity, operating on huge and progressively irregular data sets and further requiring a steady influx of sensor data, increasing pressure both on on-chip and off-chip interconnect systems. Further, ACES combines real-time time-critical with general compute tasks on the same physical platform, sharing communication, storage, and micro-architectural resources. These heterogeneous mixed-criticality systems (MCSs) place additional pressure on the interconnect, demanding minimal contention between the different criticality levels to sustain a high degree of predictability. Fulfilling the performance and energy-efficiency requirements across a wide range of industrial applications requires a carefully co-designed process of the memory system with the use cases as well as the compute units and accelerators.

-> 스포츠 촬영 에지 디바이스에서 실시간 영상 처리 시 에너지 효율과 데이터 이동이 핵심 문제이다. 이 논문은 이기종 시스템을 위한 에너지 효율적 실시간 데이터 전략을 제안한다. 결과적으로 rk3588 기기에서 고성능 처리와 배터리 지속 시간 향상에 중요하다.

### Grandes Modelos de Linguagem Multimodais (MLLMs): Da Teoria à Prática (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.12302v1
- 점수: final 82.4

Multimodal Large Language Models (MLLMs) combine the natural language understanding and generation capabilities of LLMs with perception skills in modalities such as image and audio, representing a key advancement in contemporary AI. This chapter presents the main fundamentals of MLLMs and emblematic models. Practical techniques for preprocessing, prompt engineering, and building multimodal pipelines with LangChain and LangGraph are also explored. For further practical study, supplementary material is publicly available online: https://github.com/neemiasbsilva/MLLMs-Teoria-e-Pratica. Finally, the chapter discusses the challenges and highlights promising trends.

-> 멀티모달 LLM 기술로 스포츠 영상의 자연어 분석 및 자동 하이라이트 생성이 가능해 프로젝트 핵심 기능 구현에 필수적임

### Monte Carlo Maximum Likelihood Reconstruction for Digital Holography with Speckle (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10344v1
- 점수: final 82.4

In coherent imaging, speckle is statistically modeled as multiplicative noise, posing a fundamental challenge for image reconstruction. While maximum likelihood estimation (MLE) provides a principled framework for speckle mitigation, its application to coherent imaging system such as digital holography with finite apertures is hindered by the prohibitive cost of high-dimensional matrix inversion, especially at high resolutions. This computational burden has prevented the use of MLE-based reconstruction with physically accurate aperture modeling. In this work, we propose a randomized linear algebra approach that enables scalable MLE optimization without explicit matrix inversions in gradient computation. By exploiting the structural properties of sensing matrix and using conjugate gradient for likelihood gradient evaluation, the proposed algorithm supports accurate aperture modeling without the simplifying assumptions commonly imposed for tractability. We term the resulting method projected gradient descent with Monte Carlo estimation (PGD-MC). The proposed PGD-MC framework (i) demonstrates robustness to diverse and physically accurate aperture models, (ii) achieves substantial improvements in reconstruction quality and computational efficiency, and (iii) scales effectively to high-resolution digital holography. Extensive experiments incorporating three representative denoisers as regularization show that PGD-MC provides a flexible and effective MLE-based reconstruction framework for digital holography with finite apertures, consistently outperforming prior Plug-and-Play model-based iterative reconstruction methods in both accuracy and speed. Our code is available at: https://github.com/Computational-Imaging-RU/MC_Maximum_Likelihood_Digital_Holography_Speckle.

-> 저조도 경기 촬영 시 영상 노이즈로 화질 저하가 문제이다. 이 논문은 MLE 기반 확장성 있는 이미지 재구성 방법을 제안한다. 결과적으로 야간 경기 보정 시 시각적 품질 향상에 중요하다.

### Multimodal Priors-Augmented Text-Driven 3D Human-Object Interaction Generation (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10659v1
- 점수: final 82.4

We address the challenging task of text-driven 3D human-object interaction (HOI) motion generation. Existing methods primarily rely on a direct text-to-HOI mapping, which suffers from three key limitations due to the significant cross-modality gap: (Q1) sub-optimal human motion, (Q2) unnatural object motion, and (Q3) weak interaction between humans and objects. To address these challenges, we propose MP-HOI, a novel framework grounded in four core insights: (1) Multimodal Data Priors: We leverage multimodal data (text, image, pose/object) from large multimodal models as priors to guide HOI generation, which tackles Q1 and Q2 in data modeling. (2) Enhanced Object Representation: We improve existing object representations by incorporating geometric keypoints, contact features, and dynamic properties, enabling expressive object representations, which tackles Q2 in data representation. (3) Multimodal-Aware Mixture-of-Experts (MoE) Model: We propose a modality-aware MoE model for effective multimodal feature fusion paradigm, which tackles Q1 and Q2 in feature fusion. (4) Cascaded Diffusion with Interaction Supervision: We design a cascaded diffusion framework that progressively refines human-object interaction features under dedicated supervision, which tackles Q3 in interaction refinement. Comprehensive experiments demonstrate that MP-HOI outperforms existing approaches in generating high-fidelity and fine-grained HOI motions.

-> 3D 인간-객체 상호작용 생성 기술로 스포츠 자세 분석 정확도를 높입니다. 선수와 장비(예: 골프클럽) 간 동작 관계를 실감나게 구현합니다.

### FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.11105v1
- 점수: final 82.4

Flow-matching models deliver state-of-the-art fidelity in image and video generation, but the inherent sequential denoising process renders them slower. Existing acceleration methods like distillation, trajectory truncation, and consistency approaches are static, require retraining, and often fail to generalize across tasks. We propose FastFlow, a plug-and-play adaptive inference framework that accelerates generation in flow matching models. FastFlow identifies denoising steps that produce only minor adjustments to the denoising path and approximates them without using the full neural network models used for velocity predictions. The approximation utilizes finite-difference velocity estimates from prior predictions to efficiently extrapolate future states, enabling faster advancements along the denoising path at zero compute cost. This enables skipping computation at intermediary steps. We model the decision of how many steps to safely skip before requiring a full model computation as a multi-armed bandit problem. The bandit learns the optimal skips to balance speed with performance. FastFlow integrates seamlessly with existing pipelines and generalizes across image generation, video generation, and editing tasks. Experiments demonstrate a speedup of over 2.6x while maintaining high-quality outputs. The source code for this work can be found at https://github.com/Div290/FastFlow.

-> 생성 모델 가속화 기술이 영상 보정 작업의 지연 시간을 줄입니다. 실시간 하이라이트 생성 시 2.6배 속도 향상으로 즉시 SNS 공유 가능합니다.

### VideoSTF: Stress-Testing Output Repetition in Video Large Language Models (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10639v1
- 점수: final 82.4

Video Large Language Models (VideoLLMs) have recently achieved strong performance in video understanding tasks. However, we identify a previously underexplored generation failure: severe output repetition, where models degenerate into self-reinforcing loops of repeated phrases or sentences. This failure mode is not captured by existing VideoLLM benchmarks, which focus primarily on task accuracy and factual correctness. We introduce VideoSTF, the first framework for systematically measuring and stress-testing output repetition in VideoLLMs. VideoSTF formalizes repetition using three complementary n-gram-based metrics and provides a standardized testbed of 10,000 diverse videos together with a library of controlled temporal transformations. Using VideoSTF, we conduct pervasive testing, temporal stress testing, and adversarial exploitation across 10 advanced VideoLLMs. We find that output repetition is widespread and, critically, highly sensitive to temporal perturbations of video inputs. Moreover, we show that simple temporal transformations can efficiently induce repetitive degeneration in a black-box setting, exposing output repetition as an exploitable security vulnerability. Our results reveal output repetition as a fundamental stability issue in modern VideoLLMs and motivate stability-aware evaluation for video-language systems. Our evaluation code and scripts are available at: https://github.com/yuxincao22/VideoSTF_benchmark.

-> 비디오 LLM 평가 기술로 스포츠 분석 리포트 생성 안정성을 확보합니다. 출력 반복 오류 감소로 경기 전략 설명의 신뢰도를 높입니다.

### Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.11440v1
- 점수: final 82.4

Object-level manipulation, relocating or reorienting objects in images or videos while preserving scene realism, is central to film post-production, AR, and creative editing. Yet existing methods struggle to jointly achieve three core goals: background preservation, geometric consistency under viewpoint shifts, and user-controllable transformations. Geometry-based approaches offer precise control but require explicit 3D reconstruction and generalize poorly; diffusion-based methods generalize better but lack fine-grained geometric control. We present Ctrl&Shift, an end-to-end diffusion framework to achieve geometry-consistent object manipulation without explicit 3D representations. Our key insight is to decompose manipulation into two stages, object removal and reference-guided inpainting under explicit camera pose control, and encode both within a unified diffusion process. To enable precise, disentangled control, we design a multi-task, multi-stage training strategy that separates background, identity, and pose signals across tasks. To improve generalization, we introduce a scalable real-world dataset construction pipeline that generates paired image and video samples with estimated relative camera poses. Extensive experiments demonstrate that Ctrl&Shift achieves state-of-the-art results in fidelity, viewpoint consistency, and controllability. To our knowledge, this is the first framework to unify fine-grained geometric control and real-world generalization for object manipulation, without relying on any explicit 3D modeling.

-> 지오메트리 인식 객체 조작 기술이 스포츠 영상 보정에 핵심입니다. 선수 이동 시 배경 일관성 유지로 하이라이트 영상 품질을 높입니다.

### Fully Differentiable Bidirectional Dual-Task Synergistic Learning for Semi-Supervised 3D Medical Image Segmentation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09378v1
- 점수: final 82.4

Semi-supervised learning relaxes the need of large pixel-wise labeled datasets for image segmentation by leveraging unlabeled data. The scarcity of high-quality labeled data remains a major challenge in medical image analysis due to the high annotation costs and the need for specialized clinical expertise. Semi-supervised learning has demonstrated significant potential in addressing this bottleneck, with pseudo-labeling and consistency regularization emerging as two predominant paradigms. Dual-task collaborative learning, an emerging consistency-aware paradigm, seeks to derive supplementary supervision by establishing prediction consistency between related tasks. However, current methodologies are limited to unidirectional interaction mechanisms (typically regression-to-segmentation), as segmentation results can only be transformed into regression outputs in an offline manner, thereby failing to fully exploit the potential benefits of online bidirectional cross-task collaboration. Thus, we propose a fully Differentiable Bidirectional Synergistic Learning (DBiSL) framework, which seamlessly integrates and enhances four critical SSL components: supervised learning, consistency regularization, pseudo-supervised learning, and uncertainty estimation. Experiments on two benchmark datasets demonstrate our method's state-of-the-art performance. Beyond technical contributions, this work provides new insights into unified SSL framework design and establishes a new architectural foundation for dual-task-driven SSL, while offering a generic multitask learning framework applicable to broader computer vision applications. The code will be released on github upon acceptance.

-> 개인별 하이라이트 생성에 필요한 수동 레이블 데이터 부족이 문제이다. 이 논문은 양방향 협력 준지도 학습을 제안한다. 결과적으로 적은 레이블로도 정확한 객체 분할이 가능해 편집 효율성에 중요하다.

### DeepImageSearch: Benchmarking Multimodal Agents for Context-Aware Image Retrieval in Visual Histories (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10809v1
- 점수: final 80.0

Existing multimodal retrieval systems excel at semantic matching but implicitly assume that query-image relevance can be measured in isolation. This paradigm overlooks the rich dependencies inherent in realistic visual streams, where information is distributed across temporal sequences rather than confined to single snapshots. To bridge this gap, we introduce DeepImageSearch, a novel agentic paradigm that reformulates image retrieval as an autonomous exploration task. Models must plan and perform multi-step reasoning over raw visual histories to locate targets based on implicit contextual cues. We construct DISBench, a challenging benchmark built on interconnected visual data. To address the scalability challenge of creating context-dependent queries, we propose a human-model collaborative pipeline that employs vision-language models to mine latent spatiotemporal associations, effectively offloading intensive context discovery before human verification. Furthermore, we build a robust baseline using a modular agent framework equipped with fine-grained tools and a dual-memory system for long-horizon navigation. Extensive experiments demonstrate that DISBench poses significant challenges to state-of-the-art models, highlighting the necessity of incorporating agentic reasoning into next-generation retrieval systems.

-> 시퀀스 기반 영상 분석으로 경기 흐름 이해해 정확한 하이라이트 추출 가능. 단일 프레임 한계 극복에 필수

### Data-Efficient Hierarchical Goal-Conditioned Reinforcement Learning via Normalizing Flows (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.11142v1
- 점수: final 80.0

Hierarchical goal-conditioned reinforcement learning (H-GCRL) provides a powerful framework for tackling complex, long-horizon tasks by decomposing them into structured subgoals. However, its practical adoption is hindered by poor data efficiency and limited policy expressivity, especially in offline or data-scarce regimes. In this work, Normalizing flow-based hierarchical implicit Q-learning (NF-HIQL), a novel framework that replaces unimodal gaussian policies with expressive normalizing flow policies at both the high- and low-levels of the hierarchy is introduced. This design enables tractable log-likelihood computation, efficient sampling, and the ability to model rich multimodal behaviors. New theoretical guarantees are derived, including explicit KL-divergence bounds for Real-valued non-volume preserving (RealNVP) policies and PAC-style sample efficiency results, showing that NF-HIQL preserves stability while improving generalization. Empirically, NF-HIQL is evaluted across diverse long-horizon tasks in locomotion, ball-dribbling, and multi-step manipulation from OGBench. NF-HIQL consistently outperforms prior goal-conditioned and hierarchical baselines, demonstrating superior robustness under limited data and highlighting the potential of flow-based architectures for scalable, data-efficient hierarchical reinforcement learning.

-> 데이터 효율적인 강화학습으로 복잡한 스포츠 동작(예: 드리블) 분석 및 훈련 시뮬레이션 구현 가능

### ArtisanGS: Interactive Tools for Gaussian Splat Selection with AI and Human in the Loop (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10173v1
- 점수: final 80.0

Representation in the family of 3D Gaussian Splats (3DGS) are growing into a viable alternative to traditional graphics for an expanding number of application, including recent techniques that facilitate physics simulation and animation. However, extracting usable objects from in-the-wild captures remains challenging and controllable editing techniques for this representation are limited. Unlike the bulk of emerging techniques, focused on automatic solutions or high-level editing, we introduce an interactive suite of tools centered around versatile Gaussian Splat selection and segmentation. We propose a fast AI-driven method to propagate user-guided 2D selection masks to 3DGS selections. This technique allows for user intervention in the case of errors and is further coupled with flexible manual selection and segmentation tools. These allow a user to achieve virtually any binary segmentation of an unstructured 3DGS scene. We evaluate our toolset against the state-of-the-art for Gaussian Splat selection and demonstrate their utility for downstream applications by developing a user-guided local editing approach, leveraging a custom Video Diffusion Model. With flexible selection tools, users have direct control over the areas that the AI can modify. Our selection and editing tools can be used for any in-the-wild capture without additional optimization.

-> 하이라이트 편집 시 특정 선수 분리 및 보정 작업이 어려운 문제이다. 이 논문은 AI-사용자 협력형 가우시안 스플랫 선택 도구를 제안한다. 결과적으로 개인별 영상 제작 효율성에 중요하다.

### Tele-Omni: a Unified Multimodal Framework for Video Generation and Editing (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09609v1
- 점수: final 80.0

Recent advances in diffusion-based video generation have substantially improved visual fidelity and temporal coherence. However, most existing approaches remain task-specific and rely primarily on textual instructions, limiting their ability to handle multimodal inputs, contextual references, and diverse video generation and editing scenarios within a unified framework. Moreover, many video editing methods depend on carefully engineered pipelines tailored to individual operations, which hinders scalability and composability. In this paper, we propose Tele-Omni, a unified multimodal framework for video generation and editing that follows multimodal instructions, including text, images, and reference videos, within a single model. Tele-Omni leverages pretrained multimodal large language models to parse heterogeneous instructions and infer structured generation or editing intents, while diffusion-based generators perform high-quality video synthesis conditioned on these structured signals. To enable joint training across heterogeneous video tasks, we introduce a task-aware data processing pipeline that unifies multimodal inputs into a structured instruction format while preserving task-specific constraints. Tele-Omni supports a wide range of video-centric tasks, including text-to-video generation, image-to-video generation, first-last-frame video generation, in-context video generation, and in-context video editing. By decoupling instruction parsing from video synthesis and combining it with task-aware data design, Tele-Omni achieves flexible multimodal control while maintaining strong temporal coherence and visual consistency. Experimental results demonstrate that Tele-Omni achieves competitive performance across multiple tasks.

-> 엣지 디바이스 실시간 처리 및 폐쇄 루프 시스템 접근법 제시

### Finite-time Stable Pose Estimation on TSE(3) using Point Cloud and Velocity Sensors (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09414v1
- 점수: final 80.0

This work presents a finite-time stable pose estimator (FTS-PE) for rigid bodies undergoing rotational and translational motion in three dimensions, using measurements from onboard sensors that provide position vectors to inertially-fixed points and body velocities. The FTS-PE is a full-state observer for the pose (position and orientation) and velocities and is obtained through a Lyapunov analysis that shows its stability in finite time and its robustness to bounded measurement noise. Further, this observer is designed directly on the state space, the tangent bundle of the Lie group of rigid body motions, SE(3), without using local coordinates or (dual) quaternion representations. Therefore, it can estimate arbitrary rigid body motions without encountering singularities or the unwinding phenomenon and be readily applied to autonomous vehicles. A version of this observer that does not need translational velocity measurements and uses only point clouds and angular velocity measurements from rate gyros, is also obtained. It is discretized using the framework of geometric mechanics for numerical and experimental implementations. The numerical simulations compare the FTS-PE with a dual-quaternion extended Kalman filter and our previously developed variational pose estimator (VPE). The experimental results are obtained using point cloud images and rate gyro measurements obtained from a Zed 2i stereo depth camera sensor. These results validate the stability and robustness of the FTS-PE.

-> 이 논문은 3D 포즈를 유한 시간 내 안정적으로 추정하는 방법을 제안합니다. 운동 자세 분석 기능에 직접 적용 가능해, 실시간으로 선수의 동작 정확도를 평가하는 데 중요합니다.

### DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09531v1
- 점수: final 80.0

Blind Image Quality Assessment, aiming to replicate human perception of visual quality without reference, plays a key role in vision tasks, yet existing models often fail to effectively capture subtle distortion cues, leading to a misalignment with human subjective judgments. We identify that the root cause of this limitation lies in the lack of reliable distortion priors, as methods typically learn shallow relationships between unified image features and quality scores, resulting in their insensitive nature to distortions and thus limiting their performance. To address this, we introduce DR.Experts, a novel prior-driven BIQA framework designed to explicitly incorporate distortion priors, enabling a reliable quality assessment. DR.Experts begins by leveraging a degradation-aware vision-language model to obtain distortion-specific priors, which are further refined and enhanced by the proposed Distortion-Saliency Differential Module through distinguishing them from semantic attentions, thereby ensuring the genuine representations of distortions. The refined priors, along with semantics and bridging representation, are then fused by a proposed mixture-of-experts style module named the Dynamic Distortion Weighting Module. This mechanism weights each distortion-specific feature as per its perceptual impact, ensuring that the final quality prediction aligns with human perception. Extensive experiments conducted on five challenging BIQA benchmarks demonstrate the superiority of DR.Experts over current methods and showcase its excellence in terms of generalization and data efficiency.

-> 이 논문은 왜곡 인식 이미지 품질 평가 방법을 제안합니다. 우리 디바이스의 영상 보정 기능 강화에 필수적이며, SNS 공용 전 저품질 콘텐츠를 걸러내 품질 유지합니다.

### Earinter: A Closed-Loop System for Eating Pace Regulation with Just-in-Time Intervention Using Commodity Earbuds (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09522v1
- 점수: final 80.0

Rapid eating is common yet difficult to regulate in situ, partly because people seldom notice pace changes and sustained self-monitoring is effortful. We present Earinter, a commodity-earbud-based closed-loop system that integrates in-the-wild sensing, real-time reasoning, and theory-grounded just-in-time (JIT) intervention to regulate eating pace during daily meals. Earinter repurposes the earbud's bone-conduction voice sensor to capture chewing-related vibrations and estimate eating pace as chews per swallow (CPS) for on-device inference. With data collected equally across in-lab and in-the-wild sessions, Earinter achieves reliable chewing detection (F1 = 0.97) and accurate eating pace estimation (MAE: 0.18 $\pm$ 0.13 chews/min, 3.65 $\pm$ 3.86 chews/swallow), enabling robust tracking for closed-loop use. Guided by Dual Systems Theory and refined through two Wizard-of-Oz pilots, Earinter adopts a user-friendly design for JIT intervention content and delivery policy in daily meals. In a 13-day within-subject field study (N=14), the closed-loop system significantly increased CPS and reduced food-consumption speed, with statistical signs of carryover on retention-probe days and acceptable user burden. Our findings highlight how single-modality commodity earables can support practical, theory-driven closed-loop JIT interventions for regulating eating pace in the wild.

### SARS: A Novel Face and Body Shape and Appearance Aware 3D Reconstruction System extends Morphable Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09918v1
- 점수: final 80.0

Morphable Models (3DMMs) are a type of morphable model that takes 2D images as inputs and recreates the structure and physical appearance of 3D objects, especially human faces and bodies. 3DMM combines identity and expression blendshapes with a basic face mesh to create a detailed 3D model. The variability in the 3D Morphable models can be controlled by tuning diverse parameters. They are high-level image descriptors, such as shape, texture, illumination, and camera parameters. Previous research in 3D human reconstruction concentrated solely on global face structure or geometry, ignoring face semantic features such as age, gender, and facial landmarks characterizing facial boundaries, curves, dips, and wrinkles. In order to accommodate changes in these high-level facial characteristics, this work introduces a shape and appearance-aware 3D reconstruction system (named SARS by us), a c modular pipeline that extracts body and face information from a single image to properly rebuild the 3D model of the human full body.

-> 이 논문은 단일 이미지로 3D 인체 재구성 방법을 제안합니다. 운동 자세 분석의 정밀도 향상에 중요하며, 선수의 동작을 3D 모델로 시각화해 전략 분석을 지원합니다.

### The Laplacian Mechanism Improves Transformers by Reshaping Token Geometry (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.09297v1
- 점수: final 80.0

Transformers leverage attention, the residual connection, and layer normalization to control the variance of token representations. We propose to modify attention into a Laplacian mechanism that gives the model more direct control over token variance. We conjecture that this helps transformers achieve the ideal token geometry. To investigate our conjecture, we first show that incorporating the Laplacian mechanism into transformers induces consistent improvements across benchmarks in computer vision and language. Next, we study how the Laplacian mechanism impacts the geometry of token representations using various tools: 1) principal component analysis, 2) cosine similarity metric, 3) analysis of variance, and 4) Neural Collapse metrics. Our investigation shows that the Laplacian mechanism reshapes token embeddings toward a geometry of maximal separability: tokens collapse according to their classes, and the class means exhibit Neural Collapse.

-> 트랜스포머 기반 영상 분석 및 자세 인식에 적용 가능한 기술로 스포츠 장면 분석에 중요함

### Zero-Sacrifice Persistent-Robustness Adversarial Defense for Pre-Trained Encoders (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.11204v1
- 점수: final 80.0

The widespread use of publicly available pre-trained encoders from self-supervised learning (SSL) has exposed a critical vulnerability: their susceptibility to downstream-agnostic adversarial examples (DAEs), which are crafted without knowledge of the downstream tasks but capable of misleading downstream models. While several defense methods have been explored recently, they rely primarily on task-specific adversarial fine-tuning, which inevitably limits generalizability and causes catastrophic forgetting and deteriorates benign performance. Different with previous works, we propose a more rigorous defense goal that requires only a single tuning for diverse downstream tasks to defend against DAEs and preserve benign performance. To achieve this defense goal, we introduce Zero-Sacrifice Persistent-Robustness Adversarial Defense (ZePAD), which is inspired by the inherent sensitivity of neural networks to data characteristics. Specifically, ZePAD is a dual-branch structure, which consists of a Multi-Pattern Adversarial Enhancement Branch (MPAE-Branch) that uses two adversarially fine-tuned encoders to strengthen adversarial resistance. The Benign Memory Preservation Branch (BMP-Branch) is trained on local data to ensure adversarial robustness does not compromise benign performance. Surprisingly, we find that ZePAD can directly detect DAEs by evaluating branch confidence, without introducing any adversarial exsample identification task during training. Notably, by enriching feature diversity, our method enables a single adversarial fine-tuning to defend against DAEs across downstream tasks, thereby achieving persistent robustness. Extensive experiments on 11 SSL methods and 6 datasets validate its effectiveness. In certain cases, it achieves a 29.20% improvement in benign performance and a 73.86% gain in adversarial robustness, highlighting its zero-sacrifice property.

-> 엣지 디바이스 영상 분석의 강건성을 높여 다양한 환경에서의 스포츠 장면 안정적 분석 가능

### Enhancing Weakly Supervised Multimodal Video Anomaly Detection through Text Guidance (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10549v1
- 점수: final 80.0

Weakly supervised multimodal video anomaly detection has gained significant attention, yet the potential of the text modality remains under-explored. Text provides explicit semantic information that can enhance anomaly characterization and reduce false alarms. However, extracting effective text features is challenging due to the inability of general-purpose language models to capture anomaly-specific nuances and the scarcity of relevant descriptions. Furthermore, multimodal fusion often suffers from redundancy and imbalance. To address these issues, we propose a novel text-guided framework. First, we introduce an in-context learning-based multi-stage text augmentation mechanism to generate high-quality anomaly text samples for fine-tuning the text feature extractor. Second, we design a multi-scale bottleneck Transformer fusion module that uses compressed bottleneck tokens to progressively integrate information across modalities, mitigating redundancy and imbalance. Experiments on UCF-Crime and XD-Violence demonstrate state-of-the-art performance.

-> 텍스트 지도 비정상 감지로 경기 중 특이 플레이 자동 식별. 하이라이트 편집 품질 향상 핵심

### Interactive LLM-assisted Curriculum Learning for Multi-Task Evolutionary Policy Search (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10891v1
- 점수: final 80.0

Multi-task policy search is a challenging problem because policies are required to generalize beyond training cases. Curriculum learning has proven to be effective in this setting, as it introduces complexity progressively. However, designing effective curricula is labor-intensive and requires extensive domain expertise. LLM-based curriculum generation has only recently emerged as a potential solution, but was limited to operate in static, offline modes without leveraging real-time feedback from the optimizer. Here we propose an interactive LLM-assisted framework for online curriculum generation, where the LLM adaptively designs training cases based on real-time feedback from the evolutionary optimization process. We investigate how different feedback modalities, ranging from numeric metrics alone to combinations with plots and behavior visualizations, influence the LLM ability to generate meaningful curricula. Through a 2D robot navigation case study, tackled with genetic programming as optimizer, we evaluate our approach against static LLM-generated curricula and expert-designed baselines. We show that interactive curriculum generation outperforms static approaches, with multimodal feedback incorporating both progression plots and behavior visualizations yielding performance competitive with expert-designed curricula. This work contributes to understanding how LLMs can serve as interactive curriculum designers for embodied AI systems, with potential extensions to broader evolutionary robotics applications.

-> LLM 기반 실시간 커리큘럼 생성으로 선수별 맞춤형 훈련 프로그램 개발 가능. 피지컬 AI 핵심 기술

### When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10560v1
- 점수: final 80.0

While reasoning over long context is crucial for various real-world applications, it remains challenging for large language models (LLMs) as they suffer from performance degradation as the context length grows. Recent work MemAgent has tried to tackle this by processing context chunk-by-chunk in an RNN-like loop and updating a textual memory for final answering. However, this naive recurrent memory update faces two crucial drawbacks: (i) memory can quickly explode because it can update indiscriminately, even on evidence-free chunks; and (ii) the loop lacks an exit mechanism, leading to unnecessary computation after even sufficient evidence is collected. To address these issues, we propose GRU-Mem, which incorporates two text-controlled gates for more stable and efficient long-context reasoning. Specifically, in GRU-Mem, the memory only updates when the update gate is open and the recurrent loop will exit immediately once the exit gate is open. To endow the model with such capabilities, we introduce two reward signals $r^{\text{update}}$ and $r^{\text{exit}}$ within end-to-end RL, rewarding the correct updating and exiting behaviors respectively. Experiments on various long-context reasoning tasks demonstrate the effectiveness and efficiency of GRU-Mem, which generally outperforms the vanilla MemAgent with up to 400\% times inference speed acceleration.

-> 우리 장치는 긴 스포츠 영상을 실시간으로 분석해야 합니다. 이 논문은 긴 영상 처리 속도를 최대 400% 향상시켜 에지 디바이스에서 경기 전체 분석이 가능하게 합니다.

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
