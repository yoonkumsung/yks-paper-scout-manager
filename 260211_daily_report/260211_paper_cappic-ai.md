# CAPP!C_AI 논문 리포트 (2026-02-11)

> 수집 36 | 필터 31 | 폐기 9 | 평가 15 | 출력 6 | 기준 50점

검색 윈도우: 2026-02-11T00:00:00+00:00 ~ 2026-02-11T23:59:59+00:00 | 임베딩: en_synthetic | run_id: 18

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: Developing Neural Network-Based Gaze Control Systems for Social Robots

- arXiv: http://arxiv.org/abs/2602.10946v1
- PDF: https://arxiv.org/pdf/2602.10946v1
- 발행일: 2026-02-11
- 카테고리: cs.RO
- 점수: final 74.4 (llm_adjusted:68 = base:58 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
During multi-party interactions, gaze direction is a key indicator of interest and intent, making it essential for social robots to direct their attention appropriately. Understanding the social context is crucial for robots to engage effectively, predict human intentions, and navigate interactions smoothly. This study aims to develop an empirical motion-time pattern for human gaze behavior in various social situations (e.g., entering, leaving, waving, talking, and pointing) using deep neural networks based on participants' data. We created two video clips-one for a computer screen and another for a virtual reality headset-depicting different social scenarios. Data were collected from 30 participants: 15 using an eye-tracker and 15 using an Oculus Quest 1 headset. Deep learning models, specifically Long Short-Term Memory (LSTM) and Transformers, were used to analyze and predict gaze patterns. Our models achieved 60% accuracy in predicting gaze direction in a 2D animation and 65% accuracy in a 3D animation. Then, the best model was implemented onto the Nao robot; and 36 new participants evaluated its performance. The feedback indicated overall satisfaction, with those experienced in robotics rating the models more favorably.

**선정 근거**
스포츠 경기에서 선수들의 시선 방향은 전략적 의도를 나타내므로, 본 연구의 LSTM/트랜스포머 기반 시선 분석 기술이 팀 전략 해석에 간접적 참고 가능

**활용 인사이트**
RK3588 디바이스에 실시간 시선 추적 모델 적용. 경기 영상에서 공 위치나 상대 팀 주시 패턴을 분석해 자동 하이라이트 생성 및 전략 리포트 지원

## 2위: SecCodePRM: A Process Reward Model for Code Security

- arXiv: http://arxiv.org/abs/2602.10418v1
- PDF: https://arxiv.org/pdf/2602.10418v1
- 발행일: 2026-02-11
- 카테고리: cs.CR, cs.SE
- 점수: final 52.0 (llm_adjusted:40 = base:35 + bonus:+5)
- 플래그: 실시간

**개요**
Large Language Models are rapidly becoming core components of modern software development workflows, yet ensuring code security remains challenging. Existing vulnerability detection pipelines either rely on static analyzers or use LLM/GNN-based detectors trained with coarse program-level supervision. Both families often require complete context, provide sparse end-of-completion feedback, and can degrade as code length grows, making them ill-suited for real-time, prefix-level assessment during interactive coding and streaming generation. We propose SecCodePRM, a security-oriented process reward model that assigns a context-aware, step-level security score along a code trajectory. To train the model, we derive step-level supervision labels from static analyzers and expert annotations, allowing the model to attend more precisely to fine-grained regions associated with inter-procedural vulnerabilities. SecCodePRM has three applications: full-code vulnerability detection (VD), partial-code VD, and secure code generation (CG). For VD, SecCodePRM uses risk-sensitive aggregation that emphasizes high-risk steps; for CG, SecCodePRM supports inference-time scaling by ranking candidate continuations and favoring higher cumulative reward. This design yields dense, real-time feedback that scales to long-horizon generation. Empirically, SecCodePRM outperforms prior approaches in all three settings, while preserving code functional correctness, suggesting improved security without a safety-utility tradeoff.

**선정 근거**
Weakly related: Real-time code security scoring, indirect link to AI processing but not sports/visual domain.

## 3위: Towards Affordable, Non-Invasive Real-Time Hypoglycemia Detection Using Wearable Sensor Signals [완화]

- arXiv: http://arxiv.org/abs/2602.10407v1
- PDF: https://arxiv.org/pdf/2602.10407v1
- 발행일: 2026-02-11
- 카테고리: cs.HC, cs.LG
- 점수: final 48.0 (llm_adjusted:35 = base:25 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Accurately detecting hypoglycemia without invasive glucose sensors remains a critical challenge in diabetes management, particularly in regions where continuous glucose monitoring (CGM) is prohibitively expensive or clinically inaccessible. This extended study introduces a comprehensive, multimodal physiological framework for non-invasive hypoglycemia detection using wearable sensor signals. Unlike prior work limited to single-signal analysis, this chapter evaluates three physiological modalities, galvanic skin response (GSR), heart rate (HR), and their combined fusion, using the OhioT1DM 2018 dataset. We develop an end-to-end pipeline that integrates advanced preprocessing, temporal windowing, handcrafted and sequence-based feature extraction, early and late fusion strategies, and a broad spectrum of machine learning and deep temporal models, including CNNs, LSTMs, GRUs, and TCNs. Our results demonstrate that physiological signals exhibit distinct autonomic patterns preceding hypoglycemia and that combining GSR with HR consistently enhances detection sensitivity and stability compared to single-signal models. Multimodal deep learning architectures achieve the most reliable performance, particularly in recall, the most clinically urgent metric. Ablation studies further highlight the complementary contributions of each modality, strengthening the case for affordable, sensor-based glycemic monitoring. The findings show that real-time hypoglycemia detection is achievable using only inexpensive, non-invasive wearable sensors, offering a pathway toward accessible glucose monitoring in underserved communities and low-resource healthcare environments.

**선정 근거**
의료용 혈당 모니터링 연구로 스포츠 영상 분석과 직접적 관련성이 낮음

## 4위: Resilient Voltage Estimation for Battery Packs Using Self-Learning Koopman Operator [완화]

- arXiv: http://arxiv.org/abs/2602.10397v1
- PDF: https://arxiv.org/pdf/2602.10397v1
- 발행일: 2026-02-11
- 카테고리: eess.SY
- 점수: final 44.0 (llm_adjusted:30 = base:25 + bonus:+5)
- 플래그: 실시간

**개요**
Cloud-based battery management systems (BMSs) rely on real-time voltage measurement data to ensure coordinated bi-directional charging of electric vehicles (EVs) with vehicle-to-grid technology. Unfortunately, an adversary can corrupt the measurement data during transmission from the local-BMS to the cloud-BMS, leading to disrupted EV charging. Therefore, to ensure reliable voltage data under such sensor attacks, this paper proposes a two-stage error-corrected self-learning Koopman operator-based secure voltage estimation scheme for large-format battery packs. The first stage of correction compensates for the Koopman approximation error. The second stage aims to recover the error amassing from the lack of higher-order battery dynamics information in the self-learning feedback, using two alternative methods: an adaptable empirical strategy that uses cell-level knowledge of open circuit voltage to state-of-charge mapping for pack-level estimation, and a Gaussian process regression-based data-driven method that leverages minimal data-training. During our comprehensive case studies using the high-fidelity battery simulation package 'PyBaMM-liionpack', our proposed secure estimator reliably generated real-time voltage estimation with high accuracy under varying pack topologies, charging settings, battery age-levels, and attack policies. Thus, the scalable and adaptable algorithm can be easily employed to diverse battery configurations and operating conditions, without requiring significant modifications, excessive data or sensor redundancy, to ensure optimum charging of EVs under compromised sensing.

**선정 근거**
Weakly related: Real-time estimation for batteries, tangential to edge processing but not sports-focused.

## 5위: Normalized Surveillance in the Datafied Car: How Autonomous Vehicle Users Rationalize Privacy Trade-offs [완화]

- arXiv: http://arxiv.org/abs/2602.11026v1
- PDF: https://arxiv.org/pdf/2602.11026v1
- 발행일: 2026-02-11
- 카테고리: cs.HC
- 점수: final 40.0 (llm_adjusted:25 = base:25 + bonus:+0)

**개요**
Autonomous vehicles (AVs) are characterized by pervasive datafication and surveillance through sensors like in-cabin cameras, LIDAR, and GPS. Drawing on 16 semi-structured interviews with AV drivers analyzed using constructivist grounded theory, this study examines how users make sense of vehicular surveillance within everyday datafication. Findings reveal drivers demonstrate few AV-specific privacy concerns, instead normalizing monitoring through comparisons with established digital platforms. We theorize this indifference by situating AV surveillance within the `surveillance ecology' of platform environments, arguing the datafied car functions as a mobile extension of the `leaky home' -- private spaces rendered permeable through connected technologies continuously transmitting behavioral data.   The study contributes to scholarship on surveillance beliefs, datafication, and platform governance by demonstrating how users who have accepted comprehensive smartphone and smart home monitoring encounter AV datafication as just another node in normalized data extraction. We highlight how geographic restrictions on data access -- currently limiting driver log access to California -- create asymmetries that impede informed privacy deliberation, exemplifying `tertiary digital divides.' Finally, we examine how machine learning's reliance on data-intensive approaches creates structural pressure for surveillance that transcends individual manufacturer choices. We propose governance interventions to democratize social learning, including universal data access rights, binding transparency requirements, and data minimization standards to prevent race-to-the-bottom dynamics in automotive datafication.

**선정 근거**
자동차 내 감시 시스템 관련 연구로 스포츠 촬영과 약한 연관

## 6위: DiSCoKit: An Open-Source Toolkit for Deploying Live LLM Experiences in Survey Research [완화]

- arXiv: http://arxiv.org/abs/2602.11230v1
- PDF: https://arxiv.org/pdf/2602.11230v1
- 발행일: 2026-02-11
- 카테고리: cs.HC
- 점수: final 40.0 (llm_adjusted:25 = base:25 + bonus:+0)

**개요**
Advancing social-scientific research of human-AI interaction dynamics and outcomes often requires researchers to deliver experiences with live large-language models (LLMs) to participants through online survey platforms. However, technical and practical challenges (from logging chat data to manipulating AI behaviors for experimental designs) often inhibit survey-based deployment of AI stimuli. We developed DiSCoKit--an open-source toolkit for deploying live LLM experiences (e.g., ones based on models delivered through Microsoft Azure portal) through JavaScript-enabled survey platforms (e.g., Qualtrics). This paper introduces that toolkit, explaining its scientific impetus, describes its architecture and operation, as well as its deployment possibilities and limitations.

**선정 근거**
이 논문은 설문조사 플랫폼에서 실시간 LLM 경험을 배포하는 소프트웨어 툴킷을 제안합니다. 우리 프로젝트의 핵심인 스포츠 영상 촬영, 실시간 동작 분석, 엣지 디바이스 최적화와 직접적인 기술 연관성이 없습니다.

**활용 인사이트**
스포츠 영상 분석에 적용 가능한 구체적인 방법이 제시되지 않았습니다. AI 행동 조작이나 채팅 데이터 로깅 기능이 운동 경기 전략 분석이나 개인 하이라이트 생성에 활용되기 어렵습니다.

---

## 다시 보기

### Parallel Complex Diffusion for Scalable Time Series Generation (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.17706v1
- 점수: final 85.6

Modeling long-range dependencies in time series generation poses a fundamental trade-off between representational capacity and computational efficiency. Traditional temporal diffusion models suffer from local entanglement and the $\mathcal{O}(L^2)$ cost of attention mechanisms. We address these limitations by introducing PaCoDi (Parallel Complex Diffusion), a spectral-native architecture that decouples generative modeling in the frequency domain. PaCoDi fundamentally alters the problem topology: the Fourier Transform acts as a diagonalizing operator, converting locally coupled temporal signals into globally decorrelated spectral components. Theoretically, we prove the Quadrature Forward Diffusion and Conditional Reverse Factorization theorem, demonstrating that the complex diffusion process can be split into independent real and imaginary branches. We bridge the gap between this decoupled theory and data reality using a \textbf{Mean Field Theory (MFT) approximation} reinforced by an interactive correction mechanism. Furthermore, we generalize this discrete DDPM to continuous-time Frequency SDEs, rigorously deriving the Spectral Wiener Process describe the differential spectral Brownian motion limit. Crucially, PaCoDi exploits the Hermitian Symmetry of real-valued signals to compress the sequence length by half, achieving a 50% reduction in attention FLOPs without information loss. We further derive a rigorous Heteroscedastic Loss to handle the non-isotropic noise distribution on the compressed manifold. Extensive experiments show that PaCoDi outperforms existing baselines in both generation quality and inference speed, offering a theoretically grounded and computationally efficient solution for time series modeling.

-> 이 논문은 시계열 생성의 계산 효율성을 개선해 에지 디바이스에서 비디오 처리 속도 향상에 기여합니다. 주파수 영역 압축으로 FLOPs 50% 감소와 낮은 지연 시간을 달성해 RK3588에서 실시간 하이라이트 생성이 가능합니다.

### VideoWorld 2: Learning Transferable Knowledge from Real-world Videos (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.10102v1
- 점수: final 82.4

Learning transferable knowledge from unlabeled video data and applying it in new environments is a fundamental capability of intelligent agents. This work presents VideoWorld 2, which extends VideoWorld and offers the first investigation into learning transferable knowledge directly from raw real-world videos. At its core, VideoWorld 2 introduces a dynamic-enhanced Latent Dynamics Model (dLDM) that decouples action dynamics from visual appearance: a pretrained video diffusion model handles visual appearance modeling, enabling the dLDM to learn latent codes that focus on compact and meaningful task-related dynamics. These latent codes are then modeled autoregressively to learn task policies and support long-horizon reasoning. We evaluate VideoWorld 2 on challenging real-world handcraft making tasks, where prior video generation and latent-dynamics models struggle to operate reliably. Remarkably, VideoWorld 2 achieves up to 70% improvement in task success rate and produces coherent long execution videos. In robotics, we show that VideoWorld 2 can acquire effective manipulation knowledge from the Open-X dataset, which substantially improves task performance on CALVIN. This study reveals the potential of learning transferable world knowledge directly from raw videos, with all code, data, and models to be open-sourced for further research.

-> 동작과 시각적 요소를 분리하는 기술로 스포츠 동작 분석 정확도 향상에 직접 활용 가능합니다. 잠재 코드 기반 동역학 모델이 선수의 자세 패턴 인식에 적용되어 경기 전략 분석 품질을 높입니다.

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
