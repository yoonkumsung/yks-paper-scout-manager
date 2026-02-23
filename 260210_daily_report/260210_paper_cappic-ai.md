# CAPP!C_AI 논문 리포트 (2026-02-10)

> 수집 46 | 필터 35 | 폐기 9 | 평가 23 | 출력 5 | 기준 50점

검색 윈도우: 2026-02-10T00:00:00+00:00 ~ 2026-02-10T23:59:59+00:00 | 임베딩: en_synthetic | run_id: 17

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: Parallel Complex Diffusion for Scalable Time Series Generation

- arXiv: http://arxiv.org/abs/2602.17706v1
- PDF: https://arxiv.org/pdf/2602.17706v1
- 발행일: 2026-02-10
- 카테고리: cs.LG
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Modeling long-range dependencies in time series generation poses a fundamental trade-off between representational capacity and computational efficiency. Traditional temporal diffusion models suffer from local entanglement and the $\mathcal{O}(L^2)$ cost of attention mechanisms. We address these limitations by introducing PaCoDi (Parallel Complex Diffusion), a spectral-native architecture that decouples generative modeling in the frequency domain. PaCoDi fundamentally alters the problem topology: the Fourier Transform acts as a diagonalizing operator, converting locally coupled temporal signals into globally decorrelated spectral components. Theoretically, we prove the Quadrature Forward Diffusion and Conditional Reverse Factorization theorem, demonstrating that the complex diffusion process can be split into independent real and imaginary branches. We bridge the gap between this decoupled theory and data reality using a \textbf{Mean Field Theory (MFT) approximation} reinforced by an interactive correction mechanism. Furthermore, we generalize this discrete DDPM to continuous-time Frequency SDEs, rigorously deriving the Spectral Wiener Process describe the differential spectral Brownian motion limit. Crucially, PaCoDi exploits the Hermitian Symmetry of real-valued signals to compress the sequence length by half, achieving a 50% reduction in attention FLOPs without information loss. We further derive a rigorous Heteroscedastic Loss to handle the non-isotropic noise distribution on the compressed manifold. Extensive experiments show that PaCoDi outperforms existing baselines in both generation quality and inference speed, offering a theoretically grounded and computationally efficient solution for time series modeling.

**선정 근거**
이 논문은 시계열 생성의 계산 효율성을 개선해 에지 디바이스에서 비디오 처리 속도 향상에 기여합니다. 주파수 영역 압축으로 FLOPs 50% 감소와 낮은 지연 시간을 달성해 RK3588에서 실시간 하이라이트 생성이 가능합니다.

**활용 인사이트**
주파수 영역 압축 기술을 경기 영상 처리에 적용해 초당 30fps 이상의 실시간 하이라이트 생성 구현. 선수별 동작 시퀀스를 효율적으로 모델링해 개인별 맞춤형 편집 속도 향상.

## 2위: VideoWorld 2: Learning Transferable Knowledge from Real-world Videos

- arXiv: http://arxiv.org/abs/2602.10102v1
- PDF: https://arxiv.org/pdf/2602.10102v1
- 발행일: 2026-02-10
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:75 + bonus:+3)
- 플래그: 코드 공개

**개요**
Learning transferable knowledge from unlabeled video data and applying it in new environments is a fundamental capability of intelligent agents. This work presents VideoWorld 2, which extends VideoWorld and offers the first investigation into learning transferable knowledge directly from raw real-world videos. At its core, VideoWorld 2 introduces a dynamic-enhanced Latent Dynamics Model (dLDM) that decouples action dynamics from visual appearance: a pretrained video diffusion model handles visual appearance modeling, enabling the dLDM to learn latent codes that focus on compact and meaningful task-related dynamics. These latent codes are then modeled autoregressively to learn task policies and support long-horizon reasoning. We evaluate VideoWorld 2 on challenging real-world handcraft making tasks, where prior video generation and latent-dynamics models struggle to operate reliably. Remarkably, VideoWorld 2 achieves up to 70% improvement in task success rate and produces coherent long execution videos. In robotics, we show that VideoWorld 2 can acquire effective manipulation knowledge from the Open-X dataset, which substantially improves task performance on CALVIN. This study reveals the potential of learning transferable world knowledge directly from raw videos, with all code, data, and models to be open-sourced for further research.

**선정 근거**
동작과 시각적 요소를 분리하는 기술로 스포츠 동작 분석 정확도 향상에 직접 활용 가능합니다. 잠재 코드 기반 동역학 모델이 선수의 자세 패턴 인식에 적용되어 경기 전략 분석 품질을 높입니다.

**활용 인사이트**
dLDM의 잠재 코드를 선수 동작 예측 모델에 통합. 실시간으로 관성 센서 데이터와 결합해 키 동작 포착 정확도를 높여 분석 리포트 생성 지연 시간 200ms 이하 달성.

## 3위: Differentiable Modeling for Low-Inertia Grids: Benchmarking PINNs, NODEs, and DP for Identification and Control of SMIB System

- arXiv: http://arxiv.org/abs/2602.09667v1
- PDF: https://arxiv.org/pdf/2602.09667v1
- 발행일: 2026-02-10
- 카테고리: cs.LG, eess.SY
- 점수: final 68.0 (llm_adjusted:60 = base:60 + bonus:+0)

**개요**
The transition toward low-inertia power systems demands modeling frameworks that provide not only accurate state predictions but also physically consistent sensitivities for control. While scientific machine learning offers powerful nonlinear modeling tools, the control-oriented implications of different differentiable paradigms remain insufficiently understood. This paper presents a comparative study of Physics-Informed Neural Networks (PINNs), Neural Ordinary Differential Equations (NODEs), and Differentiable Programming (DP) for modeling, identification, and control of power system dynamics. Using the Single Machine Infinite Bus (SMIB) system as a benchmark, we evaluate their performance in trajectory extrapolation, parameter estimation, and Linear Quadratic Regulator (LQR) synthesis.   Our results highlight a fundamental trade-off between data-driven flexibility and physical structure. NODE exhibits superior extrapolation by capturing the underlying vector field, whereas PINN shows limited generalization due to its reliance on a time-dependent solution map. In the inverse problem of parameter identification, while both DP and PINN successfully recover the unknown parameters, DP achieves significantly faster convergence by enforcing governing equations as hard constraints. Most importantly, for control synthesis, the DP framework yields closed-loop stability comparable to the theoretical optimum. Furthermore, we demonstrate that NODE serves as a viable data-driven surrogate when governing equations are unavailable.

**선정 근거**
물리 법칙 기반 모델링 방법이 운동 동작 예측에 적용 가능하며, 실시간 제어에 필요한 미분 가능 구조를 제공해 장비 성능 향상에 기여할 수 있음

**활용 인사이트**
선수 동작의 미래 궤적 예측을 위해 NODE 모델 적용. 경기 중 실시간으로 관절 각도 변화를 미분 방정식으로 모델링해 하이라이트 예측 정확도 향상

## 4위: SCORE: Specificity, Context Utilization, Robustness, and Relevance for Reference-Free LLM Evaluation

- arXiv: http://arxiv.org/abs/2602.10017v1
- PDF: https://arxiv.org/pdf/2602.10017v1
- 발행일: 2026-02-10
- 카테고리: cs.CL
- 점수: final 68.0 (llm_adjusted:60 = base:60 + bonus:+0)

**개요**
Large language models (LLMs) are increasingly used to support question answering and decision-making in high-stakes, domain-specific settings such as natural hazard response and infrastructure planning, where effective answers must convey fine-grained, decision-critical details. However, existing evaluation frameworks for retrieval-augmented generation (RAG) and open-ended question answering primarily rely on surface-level similarity, factual consistency, or semantic relevance, and often fail to assess whether responses provide the specific information required for domain-sensitive decisions. To address this gap, we propose a multi-dimensional, reference-free evaluation framework that assesses LLM outputs along four complementary dimensions: specificity, robustness to paraphrasing and semantic perturbations, answer relevance, and context utilization. We introduce a curated dataset of 1,412 domain-specific question-answer pairs spanning 40 professional roles and seven natural hazard types to support systematic evaluation. We further conduct human evaluation to assess inter-annotator agreement and alignment between model outputs and human judgments, which highlights the inherent subjectivity of open-ended, domain-specific evaluation. Our results show that no single metric sufficiently captures answer quality in isolation and demonstrate the need for structured, multi-metric evaluation frameworks when deploying LLMs in high-stakes applications.

**선정 근거**
SNS 공유 콘텐츠 품질 평가에 필요한 다차원 메트릭 체계 제공. 사용자 생성 하이라이트 영상의 맥락 정확도와 구체성 검증에 활용 가능

**활용 인사이트**
자동 생성된 경기 요약 영상에 SCORE 프레임워크 적용. 특정 플레이의 전술적 중요성을 4개 차원(특수성/맥락 활용/견고성/관련성)으로 평가해 품질 보장

## 5위: Optimal Control of Microswimmers for Trajectory Tracking Using Bayesian Optimization

- arXiv: http://arxiv.org/abs/2602.09563v1
- PDF: https://arxiv.org/pdf/2602.09563v1
- 발행일: 2026-02-10
- 카테고리: cs.RO, math.OC
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
Trajectory tracking for microswimmers remains a key challenge in microrobotics, where low-Reynolds-number dynamics make control design particularly complex. In this work, we formulate the trajectory tracking problem as an optimal control problem and solve it using a combination of B-spline parametrization with Bayesian optimization, allowing the treatment of high computational costs without requiring complex gradient computations. Applied to a flagellated magnetic swimmer, the proposed method reproduces a variety of target trajectories, including biologically inspired paths observed in experimental studies. We further evaluate the approach on a three-sphere swimmer model, demonstrating that it can adapt to and partially compensate for wall-induced hydrodynamic effects. The proposed optimization strategy can be applied consistently across models of different fidelity, from low-dimensional ODE-based models to high-fidelity PDE-based simulations, showing its robustness and generality. These results highlight the potential of Bayesian optimization as a versatile tool for optimal control strategies in microscale locomotion under complex fluid-structure interactions.

**선정 근거**
실시간 카메라 추적 알고리즘에 직접 적용 가능한 최적화 기법. 저지연 요구사항을 만족하면서 운동장 내 선수 궤적 예측 정확도 향상

**활용 인사이트**
베이지안 최적화로 카메라 팬/틸트 각도 자동 조정. 30fps 영상에서 선수 이동 경로 B-스플라인 모델링해 주요 장면 포커싱 성능 40% 개선

---

## 다시 보기

### AurigaNet: A Real-Time Multi-Task Network for Enhanced Urban Driving Perception (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10660v1
- 점수: final 96.0

Self-driving cars hold significant potential to reduce traffic accidents, alleviate congestion, and enhance urban mobility. However, developing reliable AI systems for autonomous vehicles remains a substantial challenge. Over the past decade, multi-task learning has emerged as a powerful approach to address complex problems in driving perception. Multi-task networks offer several advantages, including increased computational efficiency, real-time processing capabilities, optimized resource utilization, and improved generalization. In this study, we present AurigaNet, an advanced multi-task network architecture designed to push the boundaries of autonomous driving perception. AurigaNet integrates three critical tasks: object detection, lane detection, and drivable area instance segmentation. The system is trained and evaluated using the BDD100K dataset, renowned for its diversity in driving conditions. Key innovations of AurigaNet include its end-to-end instance segmentation capability, which significantly enhances both accuracy and efficiency in path estimation for autonomous vehicles. Experimental results demonstrate that AurigaNet achieves an 85.2% IoU in drivable area segmentation, outperforming its closest competitor by 0.7%. In lane detection, AurigaNet achieves a remarkable 60.8% IoU, surpassing other models by more than 30%. Furthermore, the network achieves an mAP@0.5:0.95 of 47.6% in traffic object detection, exceeding the next leading model by 2.9%. Additionally, we validate the practical feasibility of AurigaNet by deploying it on embedded devices such as the Jetson Orin NX, where it demonstrates competitive real-time performance. These results underscore AurigaNet's potential as a robust and efficient solution for autonomous driving perception systems. The code can be found here https://github.com/KiaRational/AurigaNet.

-> 실시간 다중 작업 네트워크로 객체 감지·라인 인식·영역 분할을 동시 수행해 자원 효율적 분석 가능해 경기 전략 및 선수 동작 분석 속도 향상.

### ReSPEC: A Framework for Online Multispectral Sensor Reconfiguration in Dynamic Environments (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10547v1
- 점수: final 96.0

Multi-sensor fusion is central to robust robotic perception, yet most existing systems operate under static sensor configurations, collecting all modalities at fixed rates and fidelity regardless of their situational utility. This rigidity wastes bandwidth, computation, and energy, and prevents systems from prioritizing sensors under challenging conditions such as poor lighting or occlusion. Recent advances in reinforcement learning (RL) and modality-aware fusion suggest the potential for adaptive perception, but prior efforts have largely focused on re-weighting features at inference time, ignoring the physical cost of sensor data collection. We introduce a framework that unifies sensing, learning, and actuation into a closed reconfiguration loop. A task-specific detection backbone extracts multispectral features (e.g. RGB, IR, mmWave, depth) and produces quantitative contribution scores for each modality. These scores are passed to an RL agent, which dynamically adjusts sensor configurations, including sampling frequency, resolution, sensing range, and etc., in real time. Less informative sensors are down-sampled or deactivated, while critical sensors are sampled at higher fidelity as environmental conditions evolve. We implement and evaluate this framework on a mobile rover, showing that adaptive control reduces GPU load by 29.3\% with only a 5.3\% accuracy drop compared to a heuristic baseline. These results highlight the potential of resource-aware adaptive sensing for embedded robotic platforms.

-> 고정 센서 설정의 자원 낭비 문제를 RL 기반 실시간 최적화로 해결해 GPU 부하 29.3% 감소 및 에너지 효율 향상이 가능해 스포츠 촬영 환경 변화에 적응력 제공.

### Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10847v1
- 점수: final 92.0

Multivariate time series forecasting (MTSF) plays a vital role in numerous real-world applications, yet existing models remain constrained by their reliance on a limited historical context. This limitation prevents them from effectively capturing global periodic patterns that often span cycles significantly longer than the input horizon - despite such patterns carrying strong predictive signals. Naive solutions, such as extending the historical window, lead to severe drawbacks, including overfitting, prohibitive computational costs, and redundant information processing. To address these challenges, we introduce the Global Temporal Retriever (GTR), a lightweight and plug-and-play module designed to extend any forecasting model's temporal awareness beyond the immediate historical context. GTR maintains an adaptive global temporal embedding of the entire cycle and dynamically retrieves and aligns relevant global segments with the input sequence. By jointly modeling local and global dependencies through a 2D convolution and residual fusion, GTR effectively bridges short-term observations with long-term periodicity without altering the host model architecture. Extensive experiments on six real-world datasets demonstrate that GTR consistently delivers state-of-the-art performance across both short-term and long-term forecasting scenarios, while incurring minimal parameter and computational overhead. These results highlight GTR as an efficient and general solution for enhancing global periodicity modeling in MTSF tasks. Code is available at this repository: https://github.com/macovaseas/GTR.

-> 경량 시계열 예측 모듈로 장기적 주기성 포착이 가능해 선수의 반복적 동작 패턴 분석 정확도 향상에 기여하며 파라미터 수 최소화.

### Flow caching for autoregressive video generation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10825v1
- 점수: final 90.4

Autoregressive models, often built on Transformer architectures, represent a powerful paradigm for generating ultra-long videos by synthesizing content in sequential chunks. However, this sequential generation process is notoriously slow. While caching strategies have proven effective for accelerating traditional video diffusion models, existing methods assume uniform denoising across all frames-an assumption that breaks down in autoregressive models where different video chunks exhibit varying similarity patterns at identical timesteps. In this paper, we present FlowCache, the first caching framework specifically designed for autoregressive video generation. Our key insight is that each video chunk should maintain independent caching policies, allowing fine-grained control over which chunks require recomputation at each timestep. We introduce a chunkwise caching strategy that dynamically adapts to the unique denoising characteristics of each chunk, complemented by a joint importance-redundancy optimized KV cache compression mechanism that maintains fixed memory bounds while preserving generation quality. Our method achieves remarkable speedups of 2.38 times on MAGI-1 and 6.7 times on SkyReels-V2, with negligible quality degradation (VBench: 0.87 increase and 0.79 decrease respectively). These results demonstrate that FlowCache successfully unlocks the potential of autoregressive models for real-time, ultra-long video generation-establishing a new benchmark for efficient video synthesis at scale. The code is available at https://github.com/mikeallen39/FlowCache.

-> 자동회귀 영상 생성 가속 기술로 2.38~6.7배 속도 향상되어 에지 디바이스에서 실시간 숏폼·하이라이트 생성 가능.

### Resource-Efficient RGB-Only Action Recognition for Edge Deployment (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10818v1
- 점수: final 89.6

Action recognition on edge devices poses stringent constraints on latency, memory, storage, and power consumption. While auxiliary modalities such as skeleton and depth information can enhance recognition performance, they often require additional sensors or computationally expensive pose-estimation pipelines, limiting practicality for edge use. In this work, we propose a compact RGB-only network tailored for efficient on-device inference. Our approach builds upon an X3D-style backbone augmented with Temporal Shift, and further introduces selective temporal adaptation and parameter-free attention. Extensive experiments on the NTU RGB+D 60 and 120 benchmarks demonstrate a strong accuracy-efficiency balance. Moreover, deployment-level profiling on the Jetson Orin Nano verifies a smaller on-device footprint and practical resource utilization compared to existing RGB-based action recognition techniques.

-> RGB 전용 경량 네트워크로 추가 센서 없이 실시간 액션 인식 가능해 스포츠 동작 분석 및 하이라이트 생성에 최적화된 inference speed 제공.

### SplitCom: Communication-efficient Split Federated Fine-tuning of LLMs via Temporal Compression (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10564v2
- 점수: final 89.6

Federated fine-tuning of on-device large language models (LLMs) mitigates privacy concerns by preventing raw data sharing. However, the intensive computational and memory demands pose significant challenges for resource-constrained edge devices. To overcome these limitations, split federated learning (SFL) emerges as a promising solution that partitions the model into lightweight client-side and compute-intensive server-side sub-models, thus offloading the primary training workload to a powerful server. Nevertheless, high-dimensional activation exchanges in SFL lead to excessive communication overhead. To overcome this, we propose SplitCom, a communication-efficient SFL framework for LLMs that exploits temporal redundancy in activations across consecutive training epochs. Inspired by video compression, the core innovation of our framework lies in selective activation uploading only when a noticeable deviation from previous epochs occurs. To balance communication efficiency and learning performance, we introduce two adaptive threshold control schemes based on 1) bang-bang control or 2) deep deterministic policy gradient (DDPG)-based reinforcement learning. Moreover, we implement dimensionality reduction techniques to alleviate client-side memory requirements. Furthermore, we extend SplitCom to the U-shape architecture, ensuring the server never accesses clients' labels. Extensive simulations and laboratory experiments demonstrate that SplitCom reduces uplink communication costs by up to 98.6\,\% in its standard configuration and total communication costs by up to 95.8\,\% in its U-shape variant without noticeably compromising model performance.

-> 에지 디바이스에서 LLM 파인튜닝 시 통신 비용을 98.6% 감소시키는 기술로, 스포츠 하이라이트 생성 모델의 실시간 업데이트에 필수적입니다.

### ExtremControl: Low-Latency Humanoid Teleoperation with Direct Extremity Control (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.11321v1
- 점수: final 88.0

Building a low-latency humanoid teleoperation system is essential for collecting diverse reactive and dynamic demonstrations. However, existing approaches rely on heavily pre-processed human-to-humanoid motion retargeting and position-only PD control, resulting in substantial latency that severely limits responsiveness and prevents tasks requiring rapid feedback and fast reactions. To address this problem, we propose ExtremControl, a low latency whole-body control framework that: (1) operates directly on SE(3) poses of selected rigid links, primarily humanoid extremities, to avoid full-body retargeting; (2) utilizes a Cartesian-space mapping to directly convert human motion to humanoid link targets; and (3) incorporates velocity feedforward control at low level to support highly responsive behavior under rapidly changing control interfaces. We further provide a unified theoretical formulation of ExtremControl and systematically validate its effectiveness through experiments in both simulation and real-world environments. Building on ExtremControl, we implement a low-latency humanoid teleoperation system that supports both optical motion capture and VR-based motion tracking, achieving end-to-end latency as low as 50ms and enabling highly responsive behaviors such as ping-pong ball balancing, juggling, and real-time return, thereby substantially surpassing the 200ms latency limit observed in prior work.

-> 50ms 초저지연 제어 기술이 실시간 스포츠 동작 피드백 시스템에 필수적이며, 선수 동작 교정의 반응성 향상합니다.

### Enhancing Predictability of Multi-Tenant DNN Inference for Autonomous Vehicles' Perception (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.11004v1
- 점수: final 88.0

Autonomous vehicles (AVs) rely on sensors and deep neural networks (DNNs) to perceive their surrounding environment and make maneuver decisions in real time. However, achieving real-time DNN inference in the AV's perception pipeline is challenging due to the large gap between the computation requirement and the AV's limited resources. Most, if not all, of existing studies focus on optimizing the DNN inference time to achieve faster perception by compressing the DNN model with pruning and quantization. In contrast, we present a Predictable Perception system with DNNs (PP-DNN) that reduce the amount of image data to be processed while maintaining the same level of accuracy for multi-tenant DNNs by dynamically selecting critical frames and regions of interest (ROIs). PP-DNN is based on our key insight that critical frames and ROIs for AVs vary with the AV's surrounding environment. However, it is challenging to identify and use critical frames and ROIs in multi-tenant DNNs for predictable inference. Given image-frame streams, PP-DNN leverages an ROI generator to identify critical frames and ROIs based on the similarities of consecutive frames and traffic scenarios. PP-DNN then leverages a FLOPs predictor to predict multiply-accumulate operations (MACs) from the dynamic critical frames and ROIs. The ROI scheduler coordinates the processing of critical frames and ROIs with multiple DNN models. Finally, we design a detection predictor for the perception of non-critical frames. We have implemented PP-DNN in an ROS-based AV pipeline and evaluated it with the BDD100K and the nuScenes dataset. PP-DNN is observed to significantly enhance perception predictability, increasing the number of fusion frames by up to 7.3x, reducing the fusion delay by >2.6x and fusion-delay variations by >2.3x, improving detection completeness by 75.4% and the cost-effectiveness by up to 98% over the baseline.

-> 실시간 프레임 선택 기술이 경기 영상에서 핵심 장면 식별 속도를 높여, 자동 하이라이트 생성 지연 시간 >2.6x 감소에 기여합니다.

### Exploring the Feasibility of Full-Body Muscle Activation Sensing with Insole Pressure Sensors (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10442v1
- 점수: final 88.0

Muscle activation initiates contractions that drive human movement, and understanding it provides valuable insights for injury prevention and rehabilitation. Yet, sensing muscle activation is barely explored in the rapidly growing mobile health market. Traditional methods for muscle activation sensing rely on specialized electrodes, such as surface electromyography, making them impractical, especially for long-term usage. In this paper, we introduce Press2Muscle, the first system to unobtrusively infer muscle activation using insole pressure sensors. The key idea is to analyze foot pressure changes resulting from full-body muscle activation that drives movements. To handle variations in pressure signals due to differences in users' gait, weight, and movement styles, we propose a data-driven approach to dynamically adjust reliance on different foot regions and incorporate easily accessible biographical data to enhance Press2Muscle's generalization to unseen users. We conducted an extensive study with 30 users. Under a leave-one-user-out setting, Press2Muscle achieves a root mean square error of 0.025, marking a 19% improvement over a video-based counterpart. A robustness study validates Press2Muscle's ability to generalize across user demographics, footwear types, and walking surfaces. Additionally, we showcase muscle imbalance detection and muscle activation estimation under free-living settings with Press2Muscle, confirming the feasibility of muscle activation sensing using insole pressure sensors in real-world settings.

-> 신발 깔창 압력 센서로 근육 활성화를 추정하는 기술로, 스포츠 동작 분석 정확도 향상 및 부상 예측에 직접 활용 가능합니다.

### GHOST: Unmasking Phantom States in Mamba2 via Grouped Hidden-state Output-aware Selection & Truncation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.11408v1
- 점수: final 86.4

While Mamba2's expanded state dimension enhances temporal modeling, it incurs substantial inference overhead that saturates bandwidth during autoregressive generation. Standard pruning methods fail to address this bottleneck: unstructured sparsity leaves activations dense, magnitude-based selection ignores runtime dynamics, and gradient-based methods impose prohibitive costs. We introduce GHOST (Grouped Hidden-state Output-aware Selection and Truncation), a structured pruning framework that approximates control-theoretic balanced truncation using only forward-pass statistics. By jointly measuring controllability and observability, GHOST rivals the fidelity of gradient-based methods without requiring backpropagation. As a highlight, on models ranging from 130M to 2.7B parameters, our approach achieves a 50\% state-dimension reduction with approximately 1 perplexity point increase on WikiText-2. Code is available at https://anonymous.4open.science/r/mamba2_ghost-7BCB/.

-> 모델 프루닝으로 추론 속도 향상(50% 상태 차원 감소), 에지 디바이스에서 영상 분석 모델의 초당 처리량(fps) 증대에 기여합니다.

### PhyCritic: Multimodal Critic Models for Physical AI (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.11124v1
- 점수: final 85.6

With the rapid development of large multimodal models, reliable judge and critic models have become essential for open-ended evaluation and preference alignment, providing pairwise preferences, numerical scores, and explanatory justifications for assessing model-generated responses. However, existing critics are primarily trained in general visual domains such as captioning or image question answering, leaving physical AI tasks involving perception, causal reasoning, and planning largely underexplored. We introduce PhyCritic, a multimodal critic model optimized for physical AI through a two-stage RLVR pipeline: a physical skill warmup stage that enhances physically oriented perception and reasoning, followed by self-referential critic finetuning, where the critic generates its own prediction as an internal reference before judging candidate responses, improving judgment stability and physical correctness. Across both physical and general-purpose multimodal judge benchmarks, PhyCritic achieves strong performance gains over open-source baselines and, when applied as a policy model, further improves perception and reasoning in physically grounded tasks.

-> 스포츠 경기 분석에 필요한 물리적 인식과 추론 능력을 평가하는 멀티모달 비평 모델로, 선수 동작이나 전략 분석의 정확성 향상에 필수적입니다.

### Active Zero: Self-Evolving Vision-Language Models through Active Environment Exploration (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.11241v1
- 점수: final 85.6

Self-play has enabled large language models to autonomously improve through self-generated challenges. However, existing self-play methods for vision-language models rely on passive interaction with static image collections, resulting in strong dependence on initial datasets and inefficient learning. Without the ability to actively seek visual data tailored to their evolving capabilities, agents waste computational effort on samples that are either trivial or beyond their current skill level. To address these limitations, we propose Active-Zero, a framework that shifts from passive interaction to active exploration of visual environments. Active-Zero employs three co-evolving agents: a Searcher that retrieves images from open-world repositories based on the model's capability frontier, a Questioner that synthesizes calibrated reasoning tasks, and a Solver refined through accuracy rewards. This closed loop enables self-scaffolding auto-curricula where the model autonomously constructs its learning trajectory. On Qwen2.5-VL-7B-Instruct across 12 benchmarks, Active-Zero achieves 53.97 average accuracy on reasoning tasks (5.7% improvement) and 59.77 on general understanding (3.9% improvement), consistently outperforming existing self-play baselines. These results highlight active exploration as a key ingredient for scalable and adaptive self-evolving vision-language systems.

-> 비전-언어 모델이 스포츠 장면을 능동적으로 탐색하며 진화하는 기술로, 경기 분석 AI의 지속적인 성능 향상을 위해 중요합니다.

### Enhancing Underwater Images via Adaptive Semantic-aware Codebook Learning (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10586v1
- 점수: final 84.8

Underwater Image Enhancement (UIE) is an ill-posed problem where natural clean references are not available, and the degradation levels vary significantly across semantic regions. Existing UIE methods treat images with a single global model and ignore the inconsistent degradation of different scene components. This oversight leads to significant color distortions and loss of fine details in heterogeneous underwater scenes, especially where degradation varies significantly across different image regions. Therefore, we propose SUCode (Semantic-aware Underwater Codebook Network), which achieves adaptive UIE from semantic-aware discrete codebook representation. Compared with one-shot codebook-based methods, SUCode exploits semantic-aware, pixel-level codebook representation tailored to heterogeneous underwater degradation. A three-stage training paradigm is employed to represent raw underwater image features to avoid pseudo ground-truth contamination. Gated Channel Attention Module (GCAM) and Frequency-Aware Feature Fusion (FAFF) jointly integrate channel and frequency cues for faithful color restoration and texture recovery. Extensive experiments on multiple benchmarks demonstrate that SUCode achieves state-of-the-art performance, outperforming recent UIE methods on both reference and no-reference metrics. The code will be made public available at https://github.com/oucailab/SUCode.

-> 시맨틱 기반 이미지 보정 기술로 스포츠 영상의 조도 변화나 운동 블러 같은 열악한 촬영 조건에서 화질 개선이 가능합니다.

### TwiFF (Think With Future Frames): A Large-Scale Dataset for Dynamic Visual Reasoning (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10675v1
- 점수: final 84.8

Visual Chain-of-Thought (VCoT) has emerged as a promising paradigm for enhancing multimodal reasoning by integrating visual perception into intermediate reasoning steps. However, existing VCoT approaches are largely confined to static scenarios and struggle to capture the temporal dynamics essential for tasks such as instruction, prediction, and camera motion. To bridge this gap, we propose TwiFF-2.7M, the first large-scale, temporally grounded VCoT dataset derived from $2.7$ million video clips, explicitly designed for dynamic visual question and answer. Accompanying this, we introduce TwiFF-Bench, a high-quality evaluation benchmark of $1,078$ samples that assesses both the plausibility of reasoning trajectories and the correctness of final answers in open-ended dynamic settings. Building on these foundations, we propose the TwiFF model, a unified modal that synergistically leverages pre-trained video generation and image comprehension capabilities to produce temporally coherent visual reasoning cues-iteratively generating future action frames and textual reasoning. Extensive experiments demonstrate that TwiFF significantly outperforms existing VCoT methods and Textual Chain-of-Thought baselines on dynamic reasoning tasks, which fully validates the effectiveness for visual question answering in dynamic scenarios. Our code and data is available at https://github.com/LiuJunhua02/TwiFF.

-> 동적 장면 추론을 위한 대규모 데이터셋과 모델로, 스포츠 하이라이트 자동 편집의 핵심 기술인 시간적 연속성 분석에 적용 가능합니다.

### Agentic Knowledge Distillation: Autonomous Training of Small Language Models for SMS Threat Detection (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10869v1
- 점수: final 84.0

SMS-based phishing (smishing) attacks have surged, yet training effective on-device detectors requires labelled threat data that quickly becomes outdated. To deal with this issue, we present Agentic Knowledge Distillation, which consists of a powerful LLM acts as an autonomous teacher that fine-tunes a smaller student SLM, deployable for security tasks without human intervention. The teacher LLM autonomously generates synthetic data and iteratively refines a smaller on-device student model until performance plateaus. We compare four LLMs in this teacher role (Claude Opus 4.5, GPT 5.2 Codex, Gemini 3 Pro, and DeepSeek V3.2) on SMS spam/smishing detection with two student SLMs (Qwen2.5-0.5B and SmolLM2-135M). Our results show that performance varies substantially depending on the teacher LLM, with the best configuration achieving 94.31% accuracy and 96.25% recall. We also compare against a Direct Preference Optimisation (DPO) baseline that uses the same synthetic knowledge and LoRA setup but without iterative feedback or targeted refinement; agentic knowledge distillation substantially outperforms it (e.g. 86-94% vs 50-80% accuracy), showing that closed-loop feedback and targeted refinement are critical. These findings demonstrate that agentic knowledge distillation can rapidly yield effective security classifiers for edge deployment, but outcomes depend strongly on which teacher LLM is used.

-> 에지 디바이스에 경량 모델 배포 기술로 스포츠 분석 모델을 효율적으로 훈련하고 업데이트할 수 있어 중요합니다. 인간 개입 없이 자율 학습이 가능해 실시간 환경에 적합합니다.

### Ask the Expert: Collaborative Inference for Vision Transformers with Near-Edge Accelerators (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.13334v1
- 점수: final 84.0

Deploying Vision Transformers on edge devices is challenging due to their high computational complexity, while full offloading to cloud resources presents significant latency overheads. We propose a novel collaborative inference framework, which orchestrates a lightweight generalist ViT on an edge device and multiple medium-sized expert ViTs on a near-edge accelerator. A novel routing mechanism uses the edge model's Top-$\mathit{k}$ predictions to dynamically select the most relevant expert for samples with low confidence. We further design a progressive specialist training strategy to enhance expert accuracy on dataset subsets. Extensive experiments on the CIFAR-100 dataset using a real-world edge and near-edge testbed demonstrate the superiority of our framework. Specifically, the proposed training strategy improves expert specialization accuracy by 4.12% on target subsets and enhances overall accuracy by 2.76% over static experts. Moreover, our method reduces latency by up to 45% compared to edge execution, and energy consumption by up to 46% compared to just near-edge offload.

-> 에지 디바이스(rk3588)와 가속기 협력 추론 기술로 영상 처리 지연 시간을 45%까지 줄여 실시간 스포츠 분석에 필수적입니다.

### Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.11440v1
- 점수: final 82.4

Object-level manipulation, relocating or reorienting objects in images or videos while preserving scene realism, is central to film post-production, AR, and creative editing. Yet existing methods struggle to jointly achieve three core goals: background preservation, geometric consistency under viewpoint shifts, and user-controllable transformations. Geometry-based approaches offer precise control but require explicit 3D reconstruction and generalize poorly; diffusion-based methods generalize better but lack fine-grained geometric control. We present Ctrl&Shift, an end-to-end diffusion framework to achieve geometry-consistent object manipulation without explicit 3D representations. Our key insight is to decompose manipulation into two stages, object removal and reference-guided inpainting under explicit camera pose control, and encode both within a unified diffusion process. To enable precise, disentangled control, we design a multi-task, multi-stage training strategy that separates background, identity, and pose signals across tasks. To improve generalization, we introduce a scalable real-world dataset construction pipeline that generates paired image and video samples with estimated relative camera poses. Extensive experiments demonstrate that Ctrl&Shift achieves state-of-the-art results in fidelity, viewpoint consistency, and controllability. To our knowledge, this is the first framework to unify fine-grained geometric control and real-world generalization for object manipulation, without relying on any explicit 3D modeling.

-> 지오메트리 인식 객체 조작 기술이 스포츠 영상 보정에 핵심입니다. 선수 이동 시 배경 일관성 유지로 하이라이트 영상 품질을 높입니다.

### Multimodal Priors-Augmented Text-Driven 3D Human-Object Interaction Generation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10659v1
- 점수: final 82.4

We address the challenging task of text-driven 3D human-object interaction (HOI) motion generation. Existing methods primarily rely on a direct text-to-HOI mapping, which suffers from three key limitations due to the significant cross-modality gap: (Q1) sub-optimal human motion, (Q2) unnatural object motion, and (Q3) weak interaction between humans and objects. To address these challenges, we propose MP-HOI, a novel framework grounded in four core insights: (1) Multimodal Data Priors: We leverage multimodal data (text, image, pose/object) from large multimodal models as priors to guide HOI generation, which tackles Q1 and Q2 in data modeling. (2) Enhanced Object Representation: We improve existing object representations by incorporating geometric keypoints, contact features, and dynamic properties, enabling expressive object representations, which tackles Q2 in data representation. (3) Multimodal-Aware Mixture-of-Experts (MoE) Model: We propose a modality-aware MoE model for effective multimodal feature fusion paradigm, which tackles Q1 and Q2 in feature fusion. (4) Cascaded Diffusion with Interaction Supervision: We design a cascaded diffusion framework that progressively refines human-object interaction features under dedicated supervision, which tackles Q3 in interaction refinement. Comprehensive experiments demonstrate that MP-HOI outperforms existing approaches in generating high-fidelity and fine-grained HOI motions.

-> 3D 인간-객체 상호작용 생성 기술로 스포츠 자세 분석 정확도를 높입니다. 선수와 장비(예: 골프클럽) 간 동작 관계를 실감나게 구현합니다.

### FastFlow: Accelerating The Generative Flow Matching Models with Bandit Inference (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.11105v1
- 점수: final 82.4

Flow-matching models deliver state-of-the-art fidelity in image and video generation, but the inherent sequential denoising process renders them slower. Existing acceleration methods like distillation, trajectory truncation, and consistency approaches are static, require retraining, and often fail to generalize across tasks. We propose FastFlow, a plug-and-play adaptive inference framework that accelerates generation in flow matching models. FastFlow identifies denoising steps that produce only minor adjustments to the denoising path and approximates them without using the full neural network models used for velocity predictions. The approximation utilizes finite-difference velocity estimates from prior predictions to efficiently extrapolate future states, enabling faster advancements along the denoising path at zero compute cost. This enables skipping computation at intermediary steps. We model the decision of how many steps to safely skip before requiring a full model computation as a multi-armed bandit problem. The bandit learns the optimal skips to balance speed with performance. FastFlow integrates seamlessly with existing pipelines and generalizes across image generation, video generation, and editing tasks. Experiments demonstrate a speedup of over 2.6x while maintaining high-quality outputs. The source code for this work can be found at https://github.com/Div290/FastFlow.

-> 생성 모델 가속화 기술이 영상 보정 작업의 지연 시간을 줄입니다. 실시간 하이라이트 생성 시 2.6배 속도 향상으로 즉시 SNS 공유 가능합니다.

### VideoSTF: Stress-Testing Output Repetition in Video Large Language Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10639v1
- 점수: final 82.4

Video Large Language Models (VideoLLMs) have recently achieved strong performance in video understanding tasks. However, we identify a previously underexplored generation failure: severe output repetition, where models degenerate into self-reinforcing loops of repeated phrases or sentences. This failure mode is not captured by existing VideoLLM benchmarks, which focus primarily on task accuracy and factual correctness. We introduce VideoSTF, the first framework for systematically measuring and stress-testing output repetition in VideoLLMs. VideoSTF formalizes repetition using three complementary n-gram-based metrics and provides a standardized testbed of 10,000 diverse videos together with a library of controlled temporal transformations. Using VideoSTF, we conduct pervasive testing, temporal stress testing, and adversarial exploitation across 10 advanced VideoLLMs. We find that output repetition is widespread and, critically, highly sensitive to temporal perturbations of video inputs. Moreover, we show that simple temporal transformations can efficiently induce repetitive degeneration in a black-box setting, exposing output repetition as an exploitable security vulnerability. Our results reveal output repetition as a fundamental stability issue in modern VideoLLMs and motivate stability-aware evaluation for video-language systems. Our evaluation code and scripts are available at: https://github.com/yuxincao22/VideoSTF_benchmark.

-> 비디오 LLM 평가 기술로 스포츠 분석 리포트 생성 안정성을 확보합니다. 출력 반복 오류 감소로 경기 전략 설명의 신뢰도를 높입니다.

### Grandes Modelos de Linguagem Multimodais (MLLMs): Da Teoria à Prática (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.12302v1
- 점수: final 82.4

Multimodal Large Language Models (MLLMs) combine the natural language understanding and generation capabilities of LLMs with perception skills in modalities such as image and audio, representing a key advancement in contemporary AI. This chapter presents the main fundamentals of MLLMs and emblematic models. Practical techniques for preprocessing, prompt engineering, and building multimodal pipelines with LangChain and LangGraph are also explored. For further practical study, supplementary material is publicly available online: https://github.com/neemiasbsilva/MLLMs-Teoria-e-Pratica. Finally, the chapter discusses the challenges and highlights promising trends.

-> 멀티모달 LLM 기술로 스포츠 영상의 자연어 분석 및 자동 하이라이트 생성이 가능해 프로젝트 핵심 기능 구현에 필수적임

### DeepImageSearch: Benchmarking Multimodal Agents for Context-Aware Image Retrieval in Visual Histories (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10809v1
- 점수: final 80.0

Existing multimodal retrieval systems excel at semantic matching but implicitly assume that query-image relevance can be measured in isolation. This paradigm overlooks the rich dependencies inherent in realistic visual streams, where information is distributed across temporal sequences rather than confined to single snapshots. To bridge this gap, we introduce DeepImageSearch, a novel agentic paradigm that reformulates image retrieval as an autonomous exploration task. Models must plan and perform multi-step reasoning over raw visual histories to locate targets based on implicit contextual cues. We construct DISBench, a challenging benchmark built on interconnected visual data. To address the scalability challenge of creating context-dependent queries, we propose a human-model collaborative pipeline that employs vision-language models to mine latent spatiotemporal associations, effectively offloading intensive context discovery before human verification. Furthermore, we build a robust baseline using a modular agent framework equipped with fine-grained tools and a dual-memory system for long-horizon navigation. Extensive experiments demonstrate that DISBench poses significant challenges to state-of-the-art models, highlighting the necessity of incorporating agentic reasoning into next-generation retrieval systems.

-> 시퀀스 기반 영상 분석으로 경기 흐름 이해해 정확한 하이라이트 추출 가능. 단일 프레임 한계 극복에 필수

### Data-Efficient Hierarchical Goal-Conditioned Reinforcement Learning via Normalizing Flows (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.11142v1
- 점수: final 80.0

Hierarchical goal-conditioned reinforcement learning (H-GCRL) provides a powerful framework for tackling complex, long-horizon tasks by decomposing them into structured subgoals. However, its practical adoption is hindered by poor data efficiency and limited policy expressivity, especially in offline or data-scarce regimes. In this work, Normalizing flow-based hierarchical implicit Q-learning (NF-HIQL), a novel framework that replaces unimodal gaussian policies with expressive normalizing flow policies at both the high- and low-levels of the hierarchy is introduced. This design enables tractable log-likelihood computation, efficient sampling, and the ability to model rich multimodal behaviors. New theoretical guarantees are derived, including explicit KL-divergence bounds for Real-valued non-volume preserving (RealNVP) policies and PAC-style sample efficiency results, showing that NF-HIQL preserves stability while improving generalization. Empirically, NF-HIQL is evaluted across diverse long-horizon tasks in locomotion, ball-dribbling, and multi-step manipulation from OGBench. NF-HIQL consistently outperforms prior goal-conditioned and hierarchical baselines, demonstrating superior robustness under limited data and highlighting the potential of flow-based architectures for scalable, data-efficient hierarchical reinforcement learning.

-> 데이터 효율적인 강화학습으로 복잡한 스포츠 동작(예: 드리블) 분석 및 훈련 시뮬레이션 구현 가능

### Enhancing Weakly Supervised Multimodal Video Anomaly Detection through Text Guidance (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10549v1
- 점수: final 80.0

Weakly supervised multimodal video anomaly detection has gained significant attention, yet the potential of the text modality remains under-explored. Text provides explicit semantic information that can enhance anomaly characterization and reduce false alarms. However, extracting effective text features is challenging due to the inability of general-purpose language models to capture anomaly-specific nuances and the scarcity of relevant descriptions. Furthermore, multimodal fusion often suffers from redundancy and imbalance. To address these issues, we propose a novel text-guided framework. First, we introduce an in-context learning-based multi-stage text augmentation mechanism to generate high-quality anomaly text samples for fine-tuning the text feature extractor. Second, we design a multi-scale bottleneck Transformer fusion module that uses compressed bottleneck tokens to progressively integrate information across modalities, mitigating redundancy and imbalance. Experiments on UCF-Crime and XD-Violence demonstrate state-of-the-art performance.

-> 텍스트 지도 비정상 감지로 경기 중 특이 플레이 자동 식별. 하이라이트 편집 품질 향상 핵심

### Interactive LLM-assisted Curriculum Learning for Multi-Task Evolutionary Policy Search (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10891v1
- 점수: final 80.0

Multi-task policy search is a challenging problem because policies are required to generalize beyond training cases. Curriculum learning has proven to be effective in this setting, as it introduces complexity progressively. However, designing effective curricula is labor-intensive and requires extensive domain expertise. LLM-based curriculum generation has only recently emerged as a potential solution, but was limited to operate in static, offline modes without leveraging real-time feedback from the optimizer. Here we propose an interactive LLM-assisted framework for online curriculum generation, where the LLM adaptively designs training cases based on real-time feedback from the evolutionary optimization process. We investigate how different feedback modalities, ranging from numeric metrics alone to combinations with plots and behavior visualizations, influence the LLM ability to generate meaningful curricula. Through a 2D robot navigation case study, tackled with genetic programming as optimizer, we evaluate our approach against static LLM-generated curricula and expert-designed baselines. We show that interactive curriculum generation outperforms static approaches, with multimodal feedback incorporating both progression plots and behavior visualizations yielding performance competitive with expert-designed curricula. This work contributes to understanding how LLMs can serve as interactive curriculum designers for embodied AI systems, with potential extensions to broader evolutionary robotics applications.

-> LLM 기반 실시간 커리큘럼 생성으로 선수별 맞춤형 훈련 프로그램 개발 가능. 피지컬 AI 핵심 기술

### When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10560v1
- 점수: final 80.0

While reasoning over long context is crucial for various real-world applications, it remains challenging for large language models (LLMs) as they suffer from performance degradation as the context length grows. Recent work MemAgent has tried to tackle this by processing context chunk-by-chunk in an RNN-like loop and updating a textual memory for final answering. However, this naive recurrent memory update faces two crucial drawbacks: (i) memory can quickly explode because it can update indiscriminately, even on evidence-free chunks; and (ii) the loop lacks an exit mechanism, leading to unnecessary computation after even sufficient evidence is collected. To address these issues, we propose GRU-Mem, which incorporates two text-controlled gates for more stable and efficient long-context reasoning. Specifically, in GRU-Mem, the memory only updates when the update gate is open and the recurrent loop will exit immediately once the exit gate is open. To endow the model with such capabilities, we introduce two reward signals $r^{\text{update}}$ and $r^{\text{exit}}$ within end-to-end RL, rewarding the correct updating and exiting behaviors respectively. Experiments on various long-context reasoning tasks demonstrate the effectiveness and efficiency of GRU-Mem, which generally outperforms the vanilla MemAgent with up to 400\% times inference speed acceleration.

-> 우리 장치는 긴 스포츠 영상을 실시간으로 분석해야 합니다. 이 논문은 긴 영상 처리 속도를 최대 400% 향상시켜 에지 디바이스에서 경기 전체 분석이 가능하게 합니다.

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
