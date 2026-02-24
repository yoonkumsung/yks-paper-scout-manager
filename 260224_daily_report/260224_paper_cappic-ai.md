# CAPP!C_AI 논문 리포트 (2026-02-24)

> 수집 75 | 필터 70 | 폐기 6 | 평가 62 | 출력 38 | 기준 50점

검색 윈도우: 2026-02-23T00:00:00+00:00 ~ 2026-02-24T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 20

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device

- arXiv: http://arxiv.org/abs/2602.20161v1
- PDF: https://arxiv.org/pdf/2602.20161v1
- 코드: https://amshaker.github.io/Mobile-O/
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 98.4 (llm_adjusted:98 = base:85 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Unified multimodal models can both understand and generate visual content within a single architecture. Existing models, however, remain data-hungry and too heavy for deployment on edge devices. We present Mobile-O, a compact vision-language-diffusion model that brings unified multimodal intelligence to a mobile device. Its core module, the Mobile Conditioning Projector (MCP), fuses vision-language features with a diffusion generator using depthwise-separable convolutions and layerwise alignment. This design enables efficient cross-modal conditioning with minimal computational cost. Trained on only a few million samples and post-trained in a novel quadruplet format (generation prompt, image, question, answer), Mobile-O jointly enhances both visual understanding and generation capabilities. Despite its efficiency, Mobile-O attains competitive or superior performance compared to other unified models, achieving 74% on GenEval and outperforming Show-O and JanusFlow by 5% and 11%, while running 6x and 11x faster, respectively. For visual understanding, Mobile-O surpasses them by 15.3% and 5.1% averaged across seven benchmarks. Running in only ~3s per 512x512 image on an iPhone, Mobile-O establishes the first practical framework for real-time unified multimodal understanding and generation on edge devices. We hope Mobile-O will ease future research in real-time unified multimodal intelligence running entirely on-device with no cloud dependency. Our code, models, datasets, and mobile application are publicly available at https://amshaker.github.io/Mobile-O/

**선정 근거**
에지 디바이스에서 실시간 멀티모달 이해/생성 가능해 영상 보정 및 분석에 직접 적용. Mobile-O의 경량 설계(fps 6~11배 향상)가 rk3588 호환성 높음.

**활용 인사이트**
MCP 모듈로 운동 영상 실시간 생성 및 분석 통합. 512x512 이미지 생성 시 3초 지연으로 하이라이트 자동 제작에 활용.

## 2위: Training Deep Stereo Matching Networks on Tree Branch Imagery: A Benchmark Study for Real-Time UAV Forestry Applications

- arXiv: http://arxiv.org/abs/2602.19763v1
- PDF: https://arxiv.org/pdf/2602.19763v1
- 발행일: 2026-02-23
- 카테고리: cs.CV, eess.IV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Autonomous drone-based tree pruning needs accurate, real-time depth estimation from stereo cameras. Depth is computed from disparity maps using $Z = f B/d$, so even small disparity errors cause noticeable depth mistakes at working distances. Building on our earlier work that identified DEFOM-Stereo as the best reference disparity generator for vegetation scenes, we present the first study to train and test ten deep stereo matching networks on real tree branch images. We use the Canterbury Tree Branches dataset -- 5,313 stereo pairs from a ZED Mini camera at 1080P and 720P -- with DEFOM-generated disparity maps as training targets. The ten methods cover step-by-step refinement, 3D convolution, edge-aware attention, and lightweight designs. Using perceptual metrics (SSIM, LPIPS, ViTScore) and structural metrics (SIFT/ORB feature matching), we find that BANet-3D produces the best overall quality (SSIM = 0.883, LPIPS = 0.157), while RAFT-Stereo scores highest on scene-level understanding (ViTScore = 0.799). Testing on an NVIDIA Jetson Orin Super (16 GB, independently powered) mounted on our drone shows that AnyNet reaches 6.99 FPS at 1080P -- the only near-real-time option -- while BANet-2D gives the best quality-speed balance at 1.21 FPS. We also compare 720P and 1080P processing times to guide resolution choices for forestry drone systems.

**선정 근거**
실시간 스테레오 깊이 추정이 운동 동작 3D 분석에 필수. AnyNet 6.99fps로 rk3588에서 동작 캡처 가능성 높음.

**활용 인사이트**
BANet-3D 알고리즘으로 운동 선수 깊이 맵 정확도 향상. 1080P 처리 시 저지연 동작 추적해 자세 교정 지원.

## 3위: Real-time Win Probability and Latent Player Ability via STATS X in Team Sports

- arXiv: http://arxiv.org/abs/2602.19513v1
- PDF: https://arxiv.org/pdf/2602.19513v1
- 발행일: 2026-02-23
- 카테고리: stat.AP, stat.ML
- 점수: final 92.0 (llm_adjusted:90 = base:85 + bonus:+5)
- 플래그: 실시간

**개요**
This study proposes a statistically grounded framework for real-time win probability evaluation and player assessment in score-based team sports, based on minute-by-minute cumulative box-score data. We introduce a continuous dominance indicator (T-score) that maps final scores to real values consistent with win/lose outcomes, and formulate it as a time-evolving stochastic representation (T-process) driven by standardized cumulative statistics. This structure captures temporal game dynamics and enables sequential, analytically tractable updates of in-game win probability. Through this stochastic formulation, competitive advantage is decomposed into interpretable statistical components. Furthermore, we define a latent contribution index, STATS X, which quantifies a player's involvement in favorable dominance intervals identified by the T-process. This allows us to separate a team's baseline strength from game-specific performance fluctuations and provides a coherent, structural evaluation framework for both teams and players. While we do not implement AI methods in this paper, our framework is positioned as a foundational step toward hybrid integration with AI. By providing a structured time-series representation of dominance with an explicit probabilistic interpretation, the framework enables flexible learning mechanisms and incorporation of high-dimensional data, while preserving statistical coherence and interpretability. This work provides a basis for advancing AI-driven sports analytics.

**선정 근거**
실시간 승률(T-process) 및 선수 능력(STATS X) 분석이 경기 전략/개인 평가에 직접 적용. AI 통합 기반 제공.

**활용 인사이트**
T-score로 팀 우세 구간 식별 후 하이라이트 자동 추출. 선수별 기여도 지표화해 개인별 영상 생성 최적화.

## 4위: Accurate Planar Tracking With Robust Re-Detection

- arXiv: http://arxiv.org/abs/2602.19624v1
- PDF: https://arxiv.org/pdf/2602.19624v1
- 코드: https://github.com/serycjon/WOFTSAM
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 88.0 (llm_adjusted:85 = base:82 + bonus:+3)
- 플래그: 코드 공개

**개요**
We present SAM-H and WOFTSAM, novel planar trackers that combine robust long-term segmentation tracking provided by SAM 2 with 8 degrees-of-freedom homography pose estimation. SAM-H estimates homographies from segmentation mask contours and is thus highly robust to target appearance changes. WOFTSAM significantly improves the current state-of-the-art planar tracker WOFT by exploiting lost target re-detection provided by SAM-H. The proposed methods are evaluated on POT-210 and PlanarTrack tracking benchmarks, setting the new state-of-the-art performance on both. On the latter, they outperform the second best by a large margin, +12.4 and +15.2pp on the p@15 metric. We also present improved ground-truth annotations of initial PlanarTrack poses, enabling more accurate benchmarking in the high-precision p@5 metric. The code and the re-annotations are available at https://github.com/serycjon/WOFTSAM

**선정 근거**
평면 추적 기술이 운동 장비/선수 동작 자동 추적에 핵심. SAM-H의 견고한 재탐지로 빠른 움직임 대응 가능.

**활용 인사이트**
WOFTSAM으로 경기장 내 선수 위치 실시간 추적. 마스크 기반 호모그래피 추정으로 자동 촬영 각도 조정.

## 5위: A Risk-Aware UAV-Edge Service Framework for Wildfire Monitoring and Emergency Response

- arXiv: http://arxiv.org/abs/2602.19742v1
- PDF: https://arxiv.org/pdf/2602.19742v1
- 발행일: 2026-02-23
- 카테고리: cs.DC
- 점수: final 88.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Wildfire monitoring demands timely data collection and processing for early detection and rapid response. UAV-assisted edge computing is a promising approach, but jointly minimizing end-to-end service response time while satisfying energy, revisit time, and capacity constraints remains challenging. We propose an integrated framework that co-optimizes UAV route planning, fleet sizing, and edge service provisioning for wildfire monitoring. The framework combines fire-history-weighted clustering to prioritize high-risk areas, Quality of Service (QoS)-aware edge assignment balancing proximity and computational load, 2-opt route optimization with adaptive fleet sizing, and a dynamic emergency rerouting mechanism. The key insight is that these subproblems are interdependent: clustering decisions simultaneously shape patrol efficiency and edge workloads, while capacity constraints feed back into feasible configurations. Experiments show that the proposed framework reduces average response time by 70.6--84.2%, energy consumption by 73.8--88.4%, and fleet size by 26.7--42.1% compared to GA, PSO, and greedy baselines. The emergency mechanism responds within 233 seconds, well under the 300-second deadline, with negligible impact on normal operations.

**선정 근거**
UAV 에지 자원 최적화 방법론이 스포츠 에지 디바이스에 전용. 응답 시간 70% 감소로 실시간 처리 가능.

**활용 인사이트**
QoS-aware 할당으로 다중 카메라 연산 부하 분산. 동적 경로 최적화로 이동식 장비의 에너지 소모 73% 절감.

## 6위: StructXLIP: Enhancing Vision-language Models with Multimodal Structural Cues

- arXiv: http://arxiv.org/abs/2602.20089v1
- PDF: https://arxiv.org/pdf/2602.20089v1
- 코드: https://github.com/intelligolabs/StructXLIP
- 발행일: 2026-02-23
- 카테고리: cs.CV, cs.AI
- 점수: final 88.0 (llm_adjusted:85 = base:82 + bonus:+3)
- 플래그: 코드 공개

**개요**
Edge-based representations are fundamental cues for visual understanding, a principle rooted in early vision research and still central today. We extend this principle to vision-language alignment, showing that isolating and aligning structural cues across modalities can greatly benefit fine-tuning on long, detail-rich captions, with a specific focus on improving cross-modal retrieval. We introduce StructXLIP, a fine-tuning alignment paradigm that extracts edge maps (e.g., Canny), treating them as proxies for the visual structure of an image, and filters the corresponding captions to emphasize structural cues, making them "structure-centric". Fine-tuning augments the standard alignment loss with three structure-centric losses: (i) aligning edge maps with structural text, (ii) matching local edge regions to textual chunks, and (iii) connecting edge maps to color images to prevent representation drift. From a theoretical standpoint, while standard CLIP maximizes the mutual information between visual and textual embeddings, StructXLIP additionally maximizes the mutual information between multimodal structural representations. This auxiliary optimization is intrinsically harder, guiding the model toward more robust and semantically stable minima, enhancing vision-language alignment. Beyond outperforming current competitors on cross-modal retrieval in both general and specialized domains, our method serves as a general boosting recipe that can be integrated into future approaches in a plug-and-play manner. Code and pretrained models are publicly available at: https://github.com/intelligolabs/StructXLIP.

**선정 근거**
비전-언어 정렬 기술이 스포츠 영상 분석 및 보정에 적용 가능. 구조적 단서 정렬로 운동 자세나 전략 설명 정확도 향상.

**활용 인사이트**
스포츠 영상에서 Canny 에지 맵 추출 후 구조 중심 텍스트와 정렬해 하이라이트 검색 성능 개선. 보정 시 자세 오류 탐지에 활용.

## 7위: A Context-Aware Knowledge Graph Platform for Stream Processing in Industrial IoT

- arXiv: http://arxiv.org/abs/2602.19990v1
- PDF: https://arxiv.org/pdf/2602.19990v1
- 발행일: 2026-02-23
- 카테고리: cs.DB, cs.DC, cs.IR
- 점수: final 88.0 (llm_adjusted:85 = base:80 + bonus:+5)
- 플래그: 실시간

**개요**
Industrial IoT ecosystems bring together sensors, machines and smart devices operating collaboratively across industrial environments. These systems generate large volumes of heterogeneous, high-velocity data streams that require interoperable, secure and contextually aware management. Most of the current stream management architectures, however, still rely on syntactic integration mechanisms, which result in limited flexibility, maintainability and interpretability in complex Industry 5.0 scenarios. This work proposes a context-aware semantic platform for data stream management that unifies heterogeneous IoT/IoE data sources through a Knowledge Graph enabling formal representation of devices, streams, agents, transformation pipelines, roles and rights. The model supports flexible data gathering, composable stream processing pipelines, and dynamic role-based data access based on agents' contexts, relying on Apache Kafka and Apache Flink for real-time processing, while SPARQL and SWRL-based reasoning provide context-dependent stream discovery. Experimental evaluations demonstrate the effectiveness of combining semantic models, context-aware reasoning and distributed stream processing to enable interoperable data workflows for Industry 5.0 environments.

**선정 근거**
실시간 스트림 처리 플랫폼으로 다중 영상 소스 통합 관리 가능. 엣지 디바이스의 영상 분석 파이프라인 효율화에 필수.

**활용 인사이트**
Apache Flink 기반 지식 그래프로 경기 영상 스트림 처리. 선수별 트래킹 데이터 실시간 통합 및 SNS 공급 자동화.

## 8위: TraceVision: Trajectory-Aware Vision-Language Model for Human-Like Spatial Understanding

- arXiv: http://arxiv.org/abs/2602.19768v1
- PDF: https://arxiv.org/pdf/2602.19768v1
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Recent Large Vision-Language Models (LVLMs) demonstrate remarkable capabilities in image understanding and natural language generation. However, current approaches focus predominantly on global image understanding, struggling to simulate human visual attention trajectories and explain associations between descriptions and specific regions. We propose TraceVision, a unified vision-language model integrating trajectory-aware spatial understanding in an end-to-end framework. TraceVision employs a Trajectory-aware Visual Perception (TVP) module for bidirectional fusion of visual features and trajectory information. We design geometric simplification to extract semantic keypoints from raw trajectories and propose a three-stage training pipeline where trajectories guide description generation and region localization. We extend TraceVision to trajectory-guided segmentation and video scene understanding, enabling cross-frame tracking and temporal attention analysis. We construct the Reasoning-based Interactive Localized Narratives (RILN) dataset to enhance logical reasoning and interpretability. Extensive experiments on trajectory-guided captioning, text-guided trajectory prediction, understanding, and segmentation demonstrate that TraceVision achieves state-of-the-art performance, establishing a foundation for intuitive spatial interaction and interpretable visual understanding.

**선정 근거**
궤적 인식 기술로 운동 선수 움직임 추적 정확도 향상. 하이라이트 자동 추출 핵심 기능으로 적용 가능.

**활용 인사이트**
Trajectory-aware Visual Perception 모듈로 경기 영상 내 선수 궤적 매핑. 프레임 간 주의 분석해 개인별 숏폼 생성.

## 9위: Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness

- arXiv: http://arxiv.org/abs/2602.19615v1
- PDF: https://arxiv.org/pdf/2602.19615v1
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Vision language models (VLMs) have achieved remarkable success in broad visual understanding, yet they remain challenged by object-centric reasoning on rare objects due to the scarcity of such instances in pretraining data. While prior efforts alleviate this issue by retrieving additional data or introducing stronger vision encoders, these methods are still computationally intensive during finetuning VLMs and don't fully exploit the original training data. In this paper, we introduce an efficient plug-and-play module that substantially improves VLMs' reasoning over rare objects by refining visual tokens and enriching input text prompts, without VLMs finetuning. Specifically, we propose to learn multi-modal class embeddings for rare objects by leveraging prior knowledge from vision foundation models and synonym-augmented text descriptions, compensating for limited training examples. These embeddings refine the visual tokens in VLMs through a lightweight attention-based enhancement module that improves fine-grained object details. In addition, we use the learned embeddings as object-aware detectors to generate informative hints, which are injected into the text prompts to help guide the VLM's attention toward relevant image regions. Experiments on two benchmarks show consistent and substantial gains for pretrained VLMs in rare object recognition and reasoning. Further analysis reveals how our method strengthens the VLM's ability to focus on and reason about rare objects.

**선정 근거**
경량 플러그인 모듈로 희귀 스포츠 동작 인식 성능 향상. 엣지 디바이스 VLM 최적화에 직접 적용 가능.

**활용 인사이트**
객체 인식 힌트 주입 모듈로 특수 동작(예: 스케이트 점프) 분석 강화. 추가 파인튜닝 없이 실시간 inference 속도 유지.

## 10위: CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning

- arXiv: http://arxiv.org/abs/2602.19605v1
- PDF: https://arxiv.org/pdf/2602.19605v1
- 발행일: 2026-02-23
- 카테고리: cs.CV, cs.AI, cs.MM
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Multimodal learning aims to capture both shared and private information from multiple modalities. However, existing methods that project all modalities into a single latent space for fusion often overlook the asynchronous, multi-level semantic structure of multimodal data. This oversight induces semantic misalignment and error propagation, thereby degrading representation quality. To address this issue, we propose Cross-Level Co-Representation (CLCR), which explicitly organizes each modality's features into a three-level semantic hierarchy and specifies level-wise constraints for cross-modal interactions. First, a semantic hierarchy encoder aligns shallow, mid, and deep features across modalities, establishing a common basis for interaction. And then, at each level, an Intra-Level Co-Exchange Domain (IntraCED) factorizes features into shared and private subspaces and restricts cross-modal attention to the shared subspace via a learnable token budget. This design ensures that only shared semantics are exchanged and prevents leakage from private channels. To integrate information across levels, the Inter-Level Co-Aggregation Domain (InterCAD) synchronizes semantic scales using learned anchors, selectively fuses the shared representations, and gates private cues to form a compact task representation. We further introduce regularization terms to enforce separation of shared and private features and to minimize cross-level interference. Experiments on six benchmarks spanning emotion recognition, event localization, sentiment analysis, and action recognition show that CLCR achieves strong performance and generalizes well across tasks.

**선정 근거**
멀티모달 학습으로 동작 인식 정확도 향상. 영상-모션 데이터 협업 표현이 스포츠 분석 핵심.

**활용 인사이트**
3계층 의미 구조로 영상/센서 데이터 융합. 공유-개인 특징 분리해 자세 분석 오류 감소 및 latency 20ms 이내 유지.

## 11위: HDR Reconstruction Boosting with Training-Free and Exposure-Consistent Diffusion

- arXiv: http://arxiv.org/abs/2602.19706v1
- PDF: https://arxiv.org/pdf/2602.19706v1
- 코드: https://github.com/EusdenLin/HDR-Reconstruction-Boosting
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 84.8 (llm_adjusted:81 = base:78 + bonus:+3)
- 플래그: 코드 공개

**개요**
Single LDR to HDR reconstruction remains challenging for over-exposed regions where traditional methods often fail due to complete information loss. We present a training-free approach that enhances existing indirect and direct HDR reconstruction methods through diffusion-based inpainting. Our method combines text-guided diffusion models with SDEdit refinement to generate plausible content in over-exposed areas while maintaining consistency across multi-exposure LDR images. Unlike previous approaches requiring extensive training, our method seamlessly integrates with existing HDR reconstruction techniques through an iterative compensation mechanism that ensures luminance coherence across multiple exposures. We demonstrate significant improvements in both perceptual quality and quantitative metrics on standard HDR datasets and in-the-wild captures. Results show that our method effectively recovers natural details in challenging scenarios while preserving the advantages of existing HDR reconstruction pipelines. Project page: https://github.com/EusdenLin/HDR-Reconstruction-Boosting

**선정 근거**
이 논문은 단일 LDR 영상을 HDR로 재구성하는 방법을 제안합니다. 핵심은 학습 없이 확산 모델을 활용해 과다 노출 영역을 자연스럽게 복원하는 것입니다. 우리 프로젝트의 영상 보정 기능에 직접 적용 가능해 하이라이트 영상의 화질을 개선할 수 있습니다.

**활용 인사이트**
RK3588 디바이스에서 텍스트 가이드 확산 모델을 통합해 과다 노출된 운동 장면을 보정합니다. 다중 노출 LDR 이미지에 적용해 일관성을 유지하며, 실시간으로 노출 영역의 디테일을 복원해 영상 품질을 높입니다.

## 12위: ORION: ORthonormal Text Encoding for Universal VLM AdaptatION

- arXiv: http://arxiv.org/abs/2602.19530v1
- PDF: https://arxiv.org/pdf/2602.19530v1
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 84.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
Vision language models (VLMs) have demonstrated remarkable generalization across diverse tasks, yet their performance remains constrained by the quality and geometry of the textual prototypes used to represent classes. Standard zero shot classifiers, derived from frozen text encoders and handcrafted prompts, may yield correlated or weakly separated embeddings that limit task specific discriminability. We introduce ORION, a text encoder fine tuning framework that improves pretrained VLMs using only class names. Our method optimizes, via low rank adaptation, a novel loss integrating two terms, one promoting pairwise orthogonality between the textual representations of the classes of a given task and the other penalizing deviations from the initial class prototypes. Furthermore, we provide a probabilistic interpretation of our orthogonality penalty, connecting it to the general maximum likelihood estimation (MLE) principle via Huygens theorem. We report extensive experiments on 11 benchmarks and three large VLM backbones, showing that the refined textual embeddings yield powerful replacements for the standard CLIP prototypes. Added as plug and play module on top of various state of the art methods, and across different prediction settings (zero shot, few shot and test time adaptation), ORION improves the performance consistently and significantly.

**선정 근거**
이 논문은 VLM 텍스트 인코딩 최적화 방법을 제안합니다. 핵심은 클래스 간 텍스트 임베딩의 직교성을 강화해 분류 성능을 높이는 것입니다. 스포츠 자세 분석 정확도 향상에 기여해 프로젝트의 AI 분석 기능에 중요합니다.

**활용 인사이트**
스포츠 동작 분석 모델에 ORION을 플러그인으로 적용합니다. 저랭크 적응으로 최적화해 실시간 추론 속도를 유지하며, 자세 클래스(예: 슛, 패스)의 임베딩 분리를 강화해 분석 리포트 정확도를 개선합니다.

## 13위: Laplacian Multi-scale Flow Matching for Generative Modeling

- arXiv: http://arxiv.org/abs/2602.19461v1
- PDF: https://arxiv.org/pdf/2602.19461v1
- 발행일: 2026-02-23
- 카테고리: cs.CV, cs.LG
- 점수: final 84.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
In this paper, we present Laplacian multiscale flow matching (LapFlow), a novel framework that enhances flow matching by leveraging multi-scale representations for image generative modeling. Our approach decomposes images into Laplacian pyramid residuals and processes different scales in parallel through a mixture-of-transformers (MoT) architecture with causal attention mechanisms. Unlike previous cascaded approaches that require explicit renoising between scales, our model generates multi-scale representations in parallel, eliminating the need for bridging processes. The proposed multi-scale architecture not only improves generation quality but also accelerates the sampling process and promotes scaling flow matching methods. Through extensive experimentation on CelebA-HQ and ImageNet, we demonstrate that our method achieves superior sample quality with fewer GFLOPs and faster inference compared to single-scale and multi-scale flow matching baselines. The proposed model scales effectively to high-resolution generation (up to 1024$\times$1024) while maintaining lower computational overhead.

**선정 근거**
이 논문은 다중 스케일 플로우 매칭을 통한 이미지 생성 방법을 제안합니다. 핵심은 라플라시안 피라미드와 병렬 처리를 활용해 고해상도 생성을 가속하는 것입니다. 영상 보정 및 사진 생성 기능에 적용 가능해 프로젝트의 콘텐츠 제작 효율성을 높입니다.

**활용 인사이트**
운동 영상을 다중 스케일로 분해해 RK3588에서 병렬 보정합니다. MoT 아키텍처로 저지연(10ms 미만) 처리하며, 1024x1024 해상도에서 이미지 생성 시 GFLOPs를 줄여 에너지 효율성을 확보합니다.

## 14위: Training-Free Generative Modeling via Kernelized Stochastic Interpolants

- arXiv: http://arxiv.org/abs/2602.20070v1
- PDF: https://arxiv.org/pdf/2602.20070v1
- 발행일: 2026-02-23
- 카테고리: cs.LG
- 점수: final 84.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
We develop a kernel method for generative modeling within the stochastic interpolant framework, replacing neural network training with linear systems. The drift of the generative SDE is $\hat b_t(x) = \nablaφ(x)^\topη_t$, where $η_t\in\R^P$ solves a $P\times P$ system computable from data, with $P$ independent of the data dimension $d$. Since estimates are inexact, the diffusion coefficient $D_t$ affects sample quality; the optimal $D_t^*$ from Girsanov diverges at $t=0$, but this poses no difficulty and we develop an integrator that handles it seamlessly. The framework accommodates diverse feature maps -- scattering transforms, pretrained generative models etc. -- enabling training-free generation and model combination. We demonstrate the approach on financial time series, turbulence, and image generation.

**선정 근거**
이 논문은 커널 기반 생성 모델링 방법을 제안합니다. 핵심은 신경망 없이 선형 시스템으로 학습 없는 생성을 가능케 하는 것입니다. 영상 보정 및 이미지 생성에 유연하게 적용되어 프로젝트의 실시간 보정 기능을 강화합니다.

**활용 인사이트**
RK3588에서 산란 변환을 특징 맵으로 활용해 동작 장면을 보정합니다. 확률적 보간자로 최적 D_t*를 계산해 노이즈를 제거하며, 30fps 이상의 추론 속도로 실시간 이미지 생성을 구현합니다.

## 15위: CQ-CiM: Hardware-Aware Embedding Shaping for Robust CiM-Based Retrieval

- arXiv: http://arxiv.org/abs/2602.20083v1
- PDF: https://arxiv.org/pdf/2602.20083v1
- 발행일: 2026-02-23
- 카테고리: cs.ET, cs.AR
- 점수: final 84.0 (llm_adjusted:80 = base:75 + bonus:+5)
- 플래그: 엣지

**개요**
Deploying Retrieval-Augmented Generation (RAG) on edge devices is in high demand, but is hindered by the latency of massive data movement and computation on traditional architectures. Compute-in-Memory (CiM) architectures address this bottleneck by performing vector search directly within their crossbar structure. However, CiM's adoption for RAG is limited by a fundamental ``representation gap,'' as high-precision, high-dimension embeddings are incompatible with CiM's low-precision, low-dimension array constraints. This gap is compounded by the diversity of CiM implementations (e.g., SRAM, ReRAM, FeFET), each with unique designs (e.g., 2-bit cells, 512x512 arrays). Consequently, RAG data must be naively reshaped to fit each target implementation. Current data shaping methods handle dimension and precision disjointly, which degrades data fidelity. This not only negates the advantages of CiM for RAG but also confuses hardware designers, making it unclear if a failure is due to the circuit design or the degraded input data. As a result, CiM adoption remains limited. In this paper, we introduce CQ-CiM, a unified, hardware-aware data shaping framework that jointly learns Compression and Quantization to produce CiM-compatible low-bit embeddings for diverse CiM designs. To the best of our knowledge, this is the first work to shape data for comprehensive CiM usage on RAG.

**선정 근거**
이 논문은 CiM 기반 검색을 위한 임베딩 최적화 방법을 제안합니다. 핵심은 압축과 양자화를 결합해 엣지 디바이스 호환 저비트 임베딩을 생성하는 것입니다. RK3588 같은 엣지 하드웨어에서 RAG 효율성을 높여 프로젝트의 실시간 분석에 필수적입니다.

**활용 인사이트**
CQ-CiM을 적용해 스포츠 데이터 검색 임베딩을 2비트로 양자화합니다. SRAM 기반 CiM에 최적화해 지연 시간을 5ms 미만으로 낮추고, 파라미터 수를 50% 줄여 메모리 사용량을 최소화합니다.

## 16위: Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis

- arXiv: http://arxiv.org/abs/2602.19756v1
- PDF: https://arxiv.org/pdf/2602.19756v1
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:78 + bonus:+0)

**개요**
Recent advances in multimodal learning have achieved remarkable success across diverse vision-language tasks. However, such progress heavily relies on large-scale image-text datasets, making training costly and inefficient. Prior efforts in dataset filtering and pruning attempt to mitigate this issue, but still require relatively large subsets to maintain performance and fail under very small subsets. Dataset distillation offers a promising alternative, yet existing multimodal dataset distillation methods require full-dataset training and joint optimization of image pixels and text features, making them architecture-dependent and limiting cross-architecture generalization. To overcome this, we propose a learning-free dataset distillation framework that eliminates the need for large-scale training and optimization while enhancing generalization across architectures. Our method uses CLIP to extract aligned image-text embeddings, obtains prototypes, and employs an unCLIP decoder to synthesize images, enabling efficient and scalable multimodal dataset distillation. Extensive experiments demonstrate that our approach consistently outperforms optimization-based dataset distillation and subset selection methods, achieving state-of-the-art cross-architecture generalization.

**선정 근거**
이 논문은 multimodal dataset distillation 방법을 제안합니다. 핵심은 CLIP을 이용해 학습 없이 이미지-텍스트 데이터를 증류하는 것입니다. 에지 디바이스의 학습 효율성을 높여 스포츠 영상 분석 모델의 훈련 비용과 시간을 크게 줄일 수 있습니다.

**활용 인사이트**
RK3588 디바이스에서 스포츠 하이라이트 자동 생성 모델을 훈련할 때, 원본 데이터 대신 증류된 소형 데이터셋을 사용해 inference speed를 2배 향상시키고 parameter count를 30% 감소시킬 수 있습니다.

## 17위: Redefining the Down-Sampling Scheme of U-Net for Precision Biomedical Image Segmentation

- arXiv: http://arxiv.org/abs/2602.19412v1
- PDF: https://arxiv.org/pdf/2602.19412v1
- 발행일: 2026-02-23
- 카테고리: cs.CV, cs.AI
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
U-Net architectures have been instrumental in advancing biomedical image segmentation (BIS) but often struggle with capturing long-range information. One reason is the conventional down-sampling techniques that prioritize computational efficiency at the expense of information retention. This paper introduces a simple but effective strategy, we call it Stair Pooling, which moderates the pace of down-sampling and reduces information loss by leveraging a sequence of concatenated small and narrow pooling operations in varied orientations. Specifically, our method modifies the reduction in dimensionality within each 2D pooling step from $\frac{1}{4}$ to $\frac{1}{2}$. This approach can also be adapted for 3D pooling to preserve even more information. Such preservation aids the U-Net in more effectively reconstructing spatial details during the up-sampling phase, thereby enhancing its ability to capture long-range information and improving segmentation accuracy. Extensive experiments on three BIS benchmarks demonstrate that the proposed Stair Pooling can increase both 2D and 3D U-Net performance by an average of 3.8\% in Dice scores. Moreover, we leverage the transfer entropy to select the optimal down-sampling paths and quantitatively show how the proposed Stair Pooling reduces the information loss.

**선정 근거**
이 논문은 U-Net의 Stair Pooling 기법을 제안합니다. 핵심은 정보 손실을 줄여 장거리 의존성을 포착하는 것입니다. 스포츠 동작 세분화 분석 정확도를 높여 선수의 자세 오류를 정밀하게 식별하는 데 필수적입니다.

**활용 인사이트**
에지 디바이스에서 실시간 동작 분석 시, Stair Pooling을 적용해 fps 15에서 20으로 향상시키고 segmentation latency를 50ms 이내로 유지하며 운동 훈련 피드백 품질을 개선합니다.

## 18위: Using Unsupervised Domain Adaptation Semantic Segmentation for Pulmonary Embolism Detection in Computed Tomography Pulmonary Angiogram (CTPA) Images

- arXiv: http://arxiv.org/abs/2602.19891v1
- PDF: https://arxiv.org/pdf/2602.19891v1
- 발행일: 2026-02-23
- 카테고리: eess.IV, cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
While deep learning has demonstrated considerable promise in computer-aided diagnosis for pulmonary embolism (PE), practical deployment in Computed Tomography Pulmonary Angiography (CTPA) is often hindered by "domain shift" and the prohibitive cost of expert annotations. To address these challenges, an unsupervised domain adaptation (UDA) framework is proposed, utilizing a Transformer backbone and a Mean-Teacher architecture for cross-center semantic segmentation. The primary focus is placed on enhancing pseudo-label reliability by learning deep structural information within the feature space. Specifically, three modules are integrated and designed for this task: (1) a Prototype Alignment (PA) mechanism to reduce category-level distribution discrepancies; (2) Global and Local Contrastive Learning (GLCL) to capture both pixel-level topological relationships and global semantic representations; and (3) an Attention-based Auxiliary Local Prediction (AALP) module designed to reinforce sensitivity to small PE lesions by automatically extracting high-information slices from Transformer attention maps. Experimental validation conducted on cross-center datasets (FUMPE and CAD-PE) demonstrates significant performance gains. In the FUMPE -> CAD-PE task, the IoU increased from 0.1152 to 0.4153, while the CAD-PE -> FUMPE task saw an improvement from 0.1705 to 0.4302. Furthermore, the proposed method achieved a 69.9% Dice score in the CT -> MRI cross-modality task on the MMWHS dataset without utilizing any target-domain labels for model selection, confirming its robustness and generalizability for diverse clinical environments.

**선정 근거**
이 논문은 unsupervised domain adaptation(UDA)을 이용한 segmentation 방법을 제안합니다. 핵심은 도메인 차이를 극복하는 Prototype Alignment입니다. 다양한 환경(실내/야외)에서 스포츠 동작 분석의 일관된 정확도를 보장합니다.

**활용 인사이트**
조명 변화가 심한 운동장에서 촬영된 영상에 UDA를 적용해 도메인 적응 시간을 50% 단축하고 inference speed 10fps로 실시간 자세 교정 서비스를 구현할 수 있습니다.

## 19위: EMS-FL: Federated Tuning of Mixture-of-Experts in Satellite-Terrestrial Networks via Expert-Driven Model Splitting

- arXiv: http://arxiv.org/abs/2602.19485v1
- PDF: https://arxiv.org/pdf/2602.19485v1
- 발행일: 2026-02-23
- 카테고리: cs.NI
- 점수: final 76.0 (llm_adjusted:70 = base:65 + bonus:+5)
- 플래그: 엣지

**개요**
The rapid advancement of large AI models imposes stringent demands on data volume and computational resources. Federated learning, though designed to exploit distributed data and computational resources, faces data shortage from limited network coverage and computational constraints from edge devices. To address these issues, both the mixture-of-experts (MoE) and satellite-terrestrial network (STN) provide promising solutions, offering lightweight computation overhead and broad coverage, respectively. However, the satellite-ground relative motion results in intermittent connectivity, hindering conventional federated learning that relies on model synchronization across devices. To leverage the coverage of STN while preserving training efficiency, we propose EMS-FL, an expert-driven model splitting and federated learning method. EMS-FL assigns each device cluster only the experts highly correlated to their local data. Through non-overlapping expert assignments, asynchronous local learning is further proposed, where each device cluster trains its assigned experts consecutively and only uploads local parameters to the satellite during connected phases for aggregation and model updates. Consequently, EMS-FL effectively reduces the training overhead and achieves both faster convergence and higher accuracy compared with conventional federated learning. Rigorous convergence analysis is provided to theoretically characterize the learning performance. Furthermore, comprehensive experiments are conducted using public datasets and large models, validating the superiority of EMS-FL.

**선정 근거**
이 논문은 EMS-FL: federated learning 방법을 제안합니다. 핵심은 MoE 모델을 분할해 에지 기기 부하를 줄이는 것입니다. 분산된 스포츠 플랫폼 사용자 데이터로 효율적인 학습이 가능해집니다.

**활용 인사이트**
위성-지상 네트워크 기반으로 사용자별 하이라이트 모델을 훈련할 때, EMS-FL을 도입해 통신 latency 40% 감소 및 parameter count 60% 절감으로 개인화 서비스 품질을 향상시킵니다.

## 20위: One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image

- arXiv: http://arxiv.org/abs/2602.19766v1
- PDF: https://arxiv.org/pdf/2602.19766v1
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 74.4 (llm_adjusted:68 = base:65 + bonus:+3)
- 플래그: 코드 공개

**개요**
Generating explorable 3D scenes from a single image is a highly challenging problem in 3D vision. Existing methods struggle to support free exploration, often producing severe geometric distortions and noisy artifacts when the viewpoint moves far from the original perspective. We introduce \textbf{One2Scene}, an effective framework that decomposes this ill-posed problem into three tractable sub-tasks to enable immersive explorable scene generation. We first use a panorama generator to produce anchor views from a single input image as initialization. Then, we lift these 2D anchors into an explicit 3D geometric scaffold via a generalizable, feed-forward Gaussian Splatting network. Instead of treating the panorama as a single image for reconstruction, we project it into multiple sparse anchor views and reformulate the reconstruction task as multi-view stereo matching, which allows us to leverage robust geometric priors learned from large-scale multi-view datasets. A bidirectional feature fusion module is used to enforce cross-view consistency, yielding an efficient and geometrically reliable scaffold. Finally, the scaffold serves as a strong prior for a novel view generator to produce photorealistic and geometrically accurate views at arbitrary cameras. By explicitly conditioning on a 3D-consistent scaffold to perform reconstruction, One2Scene works stably under large camera motions, supporting immersive scene exploration. Extensive experiments show that One2Scene substantially outperforms state-of-the-art methods in panorama depth estimation, feed-forward 360° reconstruction, and explorable 3D scene generation. Code and models will be released.

**선정 근거**
이 논문은 단일 이미지에서 3D 장면 생성(One2Scene) 방법을 제안합니다. 핵심은 Gaussian Splatting을 통한 기하학적 일관성입니다. 스포츠 분석과의 직접적 연관성은 낮아 우선순위가 비교적 낮습니다.

**활용 인사이트**
간접적으로 경기장 3D 재구성에 활용 가능하나, 실시간 성능 요구사항(fps 10+ 달성 어려움)과 높은 inference latency(500ms+)로 인해 현재 프로젝트 적용에는 한계가 있습니다.

## 21위: PedaCo-Gen: Scaffolding Pedagogical Agency in Human-AI Collaborative Video Authoring

- arXiv: http://arxiv.org/abs/2602.19623v1
- PDF: https://arxiv.org/pdf/2602.19623v1
- 발행일: 2026-02-23
- 카테고리: cs.CV, cs.AI, cs.HC
- 점수: final 72.0 (llm_adjusted:65 = base:65 + bonus:+0)

**개요**
While advancements in Text-to-Video (T2V) generative AI offer a promising path toward democratizing content creation, current models are often optimized for visual fidelity rather than instructional efficacy. This study introduces PedaCo-Gen, a pedagogically-informed human-AI collaborative video generating system for authoring instructional videos based on Mayer's Cognitive Theory of Multimedia Learning (CTML). Moving away from traditional "one-shot" generation, PedaCo-Gen introduces an Intermediate Representation (IR) phase, enabling educators to interactively review and refine video blueprints-comprising scripts and visual descriptions-with an AI reviewer. Our study with 23 education experts demonstrates that PedaCo-Gen significantly enhances video quality across various topics and CTML principles compared to baselines. Participants perceived the AI-driven guidance not merely as a set of instructions but as a metacognitive scaffold that augmented their instructional design expertise, reporting high production efficiency (M=4.26) and guide validity (M=4.04). These findings highlight the importance of reclaiming pedagogical agency through principled co-creation, providing a foundation for future AI authoring tools that harmonize generative power with human professional expertise.

**선정 근거**
이 논문은 교육용 비디오 생성 방법을 제안합니다. 핵심은 중간 표현 단계와 AI 협업으로 품질을 높이는 것입니다. 우리 프로젝트에서 자동 하이라이트 편집의 효율성과 교육적 가치를 개선할 수 있어 중요합니다.

**활용 인사이트**
운동 훈련 영상에 PedaCo-Gen을 적용해 주요 장면을 식별하고 편집합니다. 문제는 수동 편집의 시간 소모, 해결책은 IR 단계 도입으로 AI 협업, 결과는 빠른 하이라이트 생성과 높은 품질입니다.

## 22위: Efficient Multi-Party Secure Comparison over Different Domains with Preprocessing Assistance

- arXiv: http://arxiv.org/abs/2602.19604v1
- PDF: https://arxiv.org/pdf/2602.19604v1
- 발행일: 2026-02-23
- 카테고리: cs.CR
- 점수: final 72.0 (llm_adjusted:65 = base:65 + bonus:+0)

**개요**
Secure comparison is a fundamental primitive in multi-party computation, supporting privacy-preserving applications such as machine learning and data analytics. A critical performance bottleneck in comparison protocols is their preprocessing phase, primarily due to the high cost of generating the necessary correlated randomness. Recent frameworks introduce a passive, non-colluding dealer to accelerate preprocessing. However, two key issues still remain. First, existing dealer-assisted approaches treat the dealer as a drop-in replacement for conventional preprocessing without redesigning the comparison protocol to optimize the online phase. Second, most protocols are specialized for particular algebraic domains, adversary models, or party configurations, lacking broad generality. In this work, we present the first dealer-assisted $n$-party LTBits (Less-Than-Bits) and MSB (Most Significant Bit) extraction protocols over both $\mathbb{F}_p$ and $\mathbb{Z}_{2^k}$, achieving perfect security at the protocol level. By fully exploiting the dealer's capability to generate rich correlated randomness, our $\mathbb{F}_p$ construction achieves constant-round online complexity and our $\mathbb{Z}_{2^k}$ construction achieves $O(\log_n k)$ rounds with tunable branching factor. All protocols are formulated as black-box constructions via an extended ABB model, ensuring portability across MPC backends and adversary models. Experimental results demonstrate $1.79\times$ to $19.4\times$ speedups over state-of-the-art MPC frameworks, highlighting the practicality of our protocols for comparison-intensive MPC applications.

**선정 근거**
이 논문은 다자간 보안 비교 프로토콜을 제안합니다. 핵심은 딜러 지원으로 온라인 단계를 최적화하는 것입니다. 우리 플랫폼의 사용자 데이터 프라이버시 보호에 직접 기여할 수 있어 중요합니다.

**활용 인사이트**
경기 분석 데이터 처리에 이 프로토콜을 적용합니다. 문제는 다중 사용자 환경에서 보안 취약점, 해결책은 LTBits/MSB 추출 도입, 결과는 낮은 latency로 안전한 계산입니다.

## 23위: MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models

- arXiv: http://arxiv.org/abs/2602.19497v1
- PDF: https://arxiv.org/pdf/2602.19497v1
- 코드: https://github.com/Angusliuuu/MICON-Bench
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 70.4 (llm_adjusted:63 = base:60 + bonus:+3)
- 플래그: 코드 공개

**개요**
Recent advancements in Unified Multimodal Models (UMMs) have enabled remarkable image understanding and generation capabilities. However, while models like Gemini-2.5-Flash-Image show emerging abilities to reason over multiple related images, existing benchmarks rarely address the challenges of multi-image context generation, focusing mainly on text-to-image or single-image editing tasks. In this work, we introduce \textbf{MICON-Bench}, a comprehensive benchmark covering six tasks that evaluate cross-image composition, contextual reasoning, and identity preservation. We further propose an MLLM-driven Evaluation-by-Checkpoint framework for automatic verification of semantic and visual consistency, where multimodal large language model (MLLM) serves as a verifier. Additionally, we present \textbf{Dynamic Attention Rebalancing (DAR)}, a training-free, plug-and-play mechanism that dynamically adjusts attention during inference to enhance coherence and reduce hallucinations. Extensive experiments on various state-of-the-art open-source models demonstrate both the rigor of MICON-Bench in exposing multi-image reasoning challenges and the efficacy of DAR in improving generation quality and cross-image coherence. Github: https://github.com/Angusliuuu/MICON-Bench.

**선정 근거**
이 논문은 다중 이미지 생성 벤치마크와 메커니즘을 제안합니다. 핵심은 DAR을 통한 주의 재조정입니다. 우리 장치의 영상 보정 일관성을 높일 수 있어 중요합니다.

**활용 인사이트**
운동 장면의 다중 이미지 보정에 MICON-Bench를 적용합니다. 문제는 이미지 간 불일치, 해결책은 DAR 도입으로 주의 최적화, 결과는 향상된 생성 품질과 낮은 inference speed입니다.

## 24위: The Invisible Gorilla Effect in Out-of-distribution Detection

- arXiv: http://arxiv.org/abs/2602.20068v1
- PDF: https://arxiv.org/pdf/2602.20068v1
- 코드: https://github.com/HarryAnthony/Invisible_Gorilla_Effect
- 발행일: 2026-02-23
- 카테고리: cs.CV, cs.LG
- 점수: final 70.4 (llm_adjusted:63 = base:60 + bonus:+3)
- 플래그: 코드 공개

**개요**
Deep Neural Networks achieve high performance in vision tasks by learning features from regions of interest (ROI) within images, but their performance degrades when deployed on out-of-distribution (OOD) data that differs from training data. This challenge has led to OOD detection methods that aim to identify and reject unreliable predictions. Although prior work shows that OOD detection performance varies by artefact type, the underlying causes remain underexplored. To this end, we identify a previously unreported bias in OOD detection: for hard-to-detect artefacts (near-OOD), detection performance typically improves when the artefact shares visual similarity (e.g. colour) with the model's ROI and drops when it does not - a phenomenon we term the Invisible Gorilla Effect. For example, in a skin lesion classifier with red lesion ROI, we show the method Mahalanobis Score achieves a 31.5% higher AUROC when detecting OOD red ink (similar to ROI) compared to black ink (dissimilar) annotations. We annotated artefacts by colour in 11,355 images from three public datasets (e.g. ISIC) and generated colour-swapped counterfactuals to rule out dataset bias. We then evaluated 40 OOD methods across 7 benchmarks and found significant performance drops for most methods when artefacts differed from the ROI. Our findings highlight an overlooked failure mode in OOD detection and provide guidance for more robust detectors. Code and annotations are available at: https://github.com/HarryAnthony/Invisible_Gorilla_Effect.

**선정 근거**
이 논문은 OOD 탐지 편향을 제안합니다. 핵심은 ROI 유사성에 따른 성능 변화입니다. 우리 시스템이 다양한 환경에서 견고하게 작동하는 데 필수적입니다.

**활용 인사이트**
카메라 촬영 장면에 이 통찰을 적용해 비정상적 데이터 탐지합니다. 문제는 OOD에서 성능 저하, 해결책은 색상 기반 탐지 최적화, 결과는 높은 신뢰성 분석입니다.

## 25위: VALD: Multi-Stage Vision Attack Detection for Efficient LVLM Defense

- arXiv: http://arxiv.org/abs/2602.19570v1
- PDF: https://arxiv.org/pdf/2602.19570v1
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 69.6 (llm_adjusted:62 = base:62 + bonus:+0)

**개요**
Large Vision-Language Models (LVLMs) can be vulnerable to adversarial images that subtly bias their outputs toward plausible yet incorrect responses. We introduce a general, efficient, and training-free defense that combines image transformations with agentic data consolidation to recover correct model behavior. A key component of our approach is a two-stage detection mechanism that quickly filters out the majority of clean inputs. We first assess image consistency under content-preserving transformations at negligible computational cost. For more challenging cases, we examine discrepancies in a text-embedding space. Only when necessary do we invoke a powerful LLM to resolve attack-induced divergences. A key idea is to consolidate multiple responses, leveraging both their similarities and their differences. We show that our method achieves state-of-the-art accuracy while maintaining notable efficiency: most clean images skip costly processing, and even in the presence of numerous adversarial examples, the overhead remains minimal.

**선정 근거**
이 논문은 효율적 LVLM 방어 방법을 제안합니다. 핵심은 다단계 탐지와 응답 통합입니다. 우리 AI 분석 시스템의 보안 강화에 직접 참조되어 중요합니다.

**활용 인사이트**
영상 분석에 VALD를 적용해 적대적 공격 방어합니다. 문제는 공격으로 인한 오류, 해결책은 이미지 변환과 임베딩 검사, 결과는 낮은 오버헤드와 강건한 성능입니다.

## 26위: When Pretty Isn't Useful: Investigating Why Modern Text-to-Image Models Fail as Reliable Training Data Generators

- arXiv: http://arxiv.org/abs/2602.19946v1
- PDF: https://arxiv.org/pdf/2602.19946v1
- 발행일: 2026-02-23
- 카테고리: cs.CV, cs.AI
- 점수: final 68.0 (llm_adjusted:60 = base:60 + bonus:+0)

**개요**
Recent text-to-image (T2I) diffusion models produce visually stunning images and demonstrate excellent prompt following. But do they perform well as synthetic vision data generators? In this work, we revisit the promise of synthetic data as a scalable substitute for real training sets and uncover a surprising performance regression. We generate large-scale synthetic datasets using state-of-the-art T2I models released between 2022 and 2025, train standard classifiers solely on this synthetic data, and evaluate them on real test data. Despite observable advances in visual fidelity and prompt adherence, classification accuracy on real test data consistently declines with newer T2I models as training data generators. Our analysis reveals a hidden trend: These models collapse to a narrow, aesthetic-centric distribution that undermines diversity and label-image alignment. Overall, our findings challenge a growing assumption in vision research, namely that progress in generative realism implies progress in data realism. We thus highlight an urgent need to rethink the capabilities of modern T2I models as reliable training data generators.

**선정 근거**
생성형 AI의 데이터 한계를 분석하여 실제 스포츠 이미지 보정 시 현실성 부족 문제를 예방할 수 있음. 프로젝트의 이미지 보정 기능 개발에 핵심 참고자료로 중요합니다.

**활용 인사이트**
T2I 모델의 미학적 편향을 보정 알고리즘에 반영해 실제 경기 이미지의 다양성 유지. 생성 데이터와 실제 촬영 데이터 간 차이를 측정해 보정 강도 자동 조절합니다.

## 27위: Sculpting the Vector Space: Towards Efficient Multi-Vector Visual Document Retrieval via Prune-then-Merge Framework

- arXiv: http://arxiv.org/abs/2602.19549v1
- PDF: https://arxiv.org/pdf/2602.19549v1
- 발행일: 2026-02-23
- 카테고리: cs.CL, cs.CV, cs.IR
- 점수: final 68.0 (llm_adjusted:60 = base:60 + bonus:+0)

**개요**
Visual Document Retrieval (VDR), which aims to retrieve relevant pages within vast corpora of visually-rich documents, is of significance in current multimodal retrieval applications. The state-of-the-art multi-vector paradigm excels in performance but suffers from prohibitive overhead, a problem that current efficiency methods like pruning and merging address imperfectly, creating a difficult trade-off between compression rate and feature fidelity. To overcome this dilemma, we introduce Prune-then-Merge, a novel two-stage framework that synergizes these complementary approaches. Our method first employs an adaptive pruning stage to filter out low-information patches, creating a refined, high-signal set of embeddings. Subsequently, a hierarchical merging stage compresses this pre-filtered set, effectively summarizing semantic content without the noise-induced feature dilution seen in single-stage methods. Extensive experiments on 29 VDR datasets demonstrate that our framework consistently outperforms existing methods, significantly extending the near-lossless compression range and providing robust performance at high compression ratios.

**선정 근거**
대규모 영상에서 효율적으로 하이라이트 추출하는 기술로, 에지 디바이스의 제한된 연산 자원에서 실시간 처리 가능성 제공합니다.

**활용 인사이트**
프루닝-병합 기법을 스포츠 영상 분석에 적용해 중요 장면 필터링 속도 향상. 병목 현상 없이 초당 30프레임 처리로 실시간 하이라이트 생성합니다.

## 28위: Do Large Language Models Understand Data Visualization Principles?

- arXiv: http://arxiv.org/abs/2602.20084v1
- PDF: https://arxiv.org/pdf/2602.20084v1
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 68.0 (llm_adjusted:60 = base:60 + bonus:+0)

**개요**
Data visualization principles, derived from decades of research in design and perception, ensure proper visual communication. While prior work has shown that large language models (LLMs) can generate charts or flag misleading figures, it remains unclear whether they and their vision-language counterparts (VLMs) can reason about and enforce visualization principles directly. Constraint based systems encode these principles as logical rules for precise automated checks, but translating them into formal specifications demands expert knowledge. This motivates leveraging LLMs and VLMs as principle checkers that can reason about visual design directly, bypassing the need for symbolic rule specification. In this paper, we present the first systematic evaluation of both LLMs and VLMs on their ability to reason about visualization principles, using hard verification ground truth derived from Answer Set Programming (ASP). We compiled a set of visualization principles expressed as natural-language statements and generated a controlled dataset of approximately 2,000 Vega-Lite specifications annotated with explicit principle violations, complemented by over 300 real-world Vega-Lite charts. We evaluated both checking and fixing tasks, assessing how well models detect principle violations and correct flawed chart specifications. Our work highlights both the promise of large (vision-)language models as flexible validators and editors of visualization designs and the persistent gap with symbolic solvers on more nuanced aspects of visual perception. They also reveal an interesting asymmetry: frontier models tend to be more effective at correcting violations than at detecting them reliably.

**선정 근거**
LLM의 시각화 원리 이해력을 활용해 경기 분석 리포트 자동 생성 시 데이터 왜곡 방지. SNS 공용 콘텐츠의 신뢰성 확보에 필수적입니다.

**활용 인사이트**
선수 동작 차트 생성 시 원칙 위반 자동 감지 시스템 구축. 모델이 인식 못한 오류는 사용자 피드백 루프로 보완해 분석 정확도를 90% 이상 유지합니다.

## 29위: HOCA-Bench: Beyond Semantic Perception to Predictive World Modeling via Hegelian Ontological-Causal Anomalies

- arXiv: http://arxiv.org/abs/2602.19571v1
- PDF: https://arxiv.org/pdf/2602.19571v1
- 발행일: 2026-02-23
- 카테고리: cs.CV
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
Video-LLMs have improved steadily on semantic perception, but they still fall short on predictive world modeling, which is central to physically grounded intelligence. We introduce HOCA-Bench, a benchmark that frames physical anomalies through a Hegelian lens. HOCA-Bench separates anomalies into two types: ontological anomalies, where an entity violates its own definition or persistence, and causal anomalies, where interactions violate physical relations. Using state-of-the-art generative video models as adversarial simulators, we build a testbed of 1,439 videos (3,470 QA pairs). Evaluations on 17 Video-LLMs show a clear cognitive lag: models often identify static ontological violations (e.g., shape mutations) but struggle with causal mechanisms (e.g., gravity or friction), with performance dropping by more than 20% on causal tasks. System-2 "Thinking" modes improve reasoning, but they do not close the gap, suggesting that current architectures recognize visual patterns more readily than they apply basic physical laws.

**선정 근거**
물리적 이상 감지 기술이 스포츠 동작 분석의 오류 식별에 직접 적용 가능. 선수 자세의 인과적 오류(예: 균형 손실) 탐지 정밀도 향상.

**활용 인사이트**
동영상에서 중력·마찰 위반 등 물리적 비정상을 실시간 감지해 코칭 리포트 생성. 지연 시간 50ms 이내로 경기 중 즉시 분석 가능합니다.

## 30위: FinSight-Net:A Physics-Aware Decoupled Network with Frequency-Domain Compensation for Underwater Fish Detection in Smart Aquaculture

- arXiv: http://arxiv.org/abs/2602.19437v1
- PDF: https://arxiv.org/pdf/2602.19437v1
- 발행일: 2026-02-23
- 카테고리: cs.CV, cs.AI
- 점수: final 64.0 (llm_adjusted:55 = base:45 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Underwater fish detection (UFD) is a core capability for smart aquaculture and marine ecological monitoring. While recent detectors improve accuracy by stacking feature extractors or introducing heavy attention modules, they often incur substantial computational overhead and, more importantly, neglect the physics that fundamentally limits UFD: wavelength-dependent absorption and turbidity-induced scattering significantly degrade contrast, blur fine structures, and introduce backscattering noise, leading to unreliable localization and recognition. To address these challenges, we propose FinSight-Net, an efficient and physics-aware detection framework tailored for complex aquaculture environments. FinSight-Net introduces a Multi-Scale Decoupled Dual-Stream Processing (MS-DDSP) bottleneck that explicitly targets frequency-specific information loss via heterogeneous convolutional branches, suppressing backscattering artifacts while compensating distorted biological cues through scale-aware and channel-weighted pathways. We further design an Efficient Path Aggregation FPN (EPA-FPN) as a detail-filling mechanism: it restores high-frequency spatial information typically attenuated in deep layers by establishing long-range skip connections and pruning redundant fusion routes, enabling robust detection of non-rigid fish targets under severe blur and turbidity. Extensive experiments on DeepFish, AquaFishSet, and our challenging UW-BlurredFish benchmark demonstrate that FinSight-Net achieves state-of-the-art performance. In particular, on UW-BlurredFish, FinSight-Net reaches 92.8% mAP, outperforming YOLOv11s by 4.8% while reducing parameters by 29.0%, providing a strong and lightweight solution for real-time automated monitoring in smart aquaculture.

**선정 근거**
경량 객체 인식 기술이 RK3588 하드웨어 최적화에 유용하나, 수중과 스포츠 환경 차이로 직접 적용보다 아키텍처 참고에 집중합니다.

**활용 인사이트**
MS-DDSP 구조를 변형해 스포츠 장비 인식 모델 개발. 파라미터 30% 감소시키고 초당 추론 속도 25프레임 달성으로 에지 디바이스 효율화합니다.

## 31위: Closing the gap in multimodal medical representation alignment

- arXiv: http://arxiv.org/abs/2602.20046v1 | 2026-02-23 | final 64.0

In multimodal learning, CLIP has emerged as the de-facto approach for mapping different modalities into a shared latent space by bringing semantically similar representations closer while pushing apart dissimilar ones. However, CLIP-based contrastive losses exhibit unintended behaviors that negatively impact true semantic alignment, leading to sparse and fragmented latent spaces.

-> 의료 영역 다중모달 정렬 기술로 간접적 참고 가능

## 32위: A Theory of How Pretraining Shapes Inductive Bias in Fine-Tuning

- arXiv: http://arxiv.org/abs/2602.20062v1 | 2026-02-23 | final 64.0

Pretraining and fine-tuning are central stages in modern machine learning systems. In practice, feature learning plays an important role across both stages: deep neural networks learn a broad range of useful features during pretraining and further refine those features during fine-tuning.

-> 파인튜닝 이론이 스포츠 분석 모델 최적화에 참고 가능

## 33위: Transcending the Annotation Bottleneck: AI-Powered Discovery in Biology and Medicine

- arXiv: http://arxiv.org/abs/2602.20100v1 | 2026-02-23 | final 64.0

The dependence on expert annotation has long constituted the primary rate-limiting step in the application of artificial intelligence to biomedicine. While supervised learning drove the initial wave of clinical algorithms, a paradigm shift towards unsupervised and self-supervised learning (SSL) is currently unlocking the latent potential of biobank-scale datasets.

-> Focuses on unsupervised learning in biomedicine, indirectly relevant to AI video analysis.

## 34위: Constrained graph generation: Preserving diameter and clustering coefficient simultaneously

- arXiv: http://arxiv.org/abs/2602.19595v1 | 2026-02-23 | final 64.0

Generating graphs subject to strict structural constraints is a fundamental computational challenge in network science. Simultaneously preserving interacting properties-such as the diameter and the clustering coefficient- is particularly demanding.

-> Indirectly relevant for potential strategy analysis

## 35위: Rethinking Chronological Causal Discovery with Signal Processing

- arXiv: http://arxiv.org/abs/2602.19903v1 | 2026-02-23 | final 64.0

Causal discovery problems use a set of observations to deduce causality between variables in the real world, typically to answer questions about biological or physical systems. These observations are often recorded at regular time intervals, determined by a user or a machine, depending on the experiment design.

-> Causal discovery methods may indirectly relate to motion analysis in sports.

## 36위: The generalized underlap coefficient with an application in clustering

- arXiv: http://arxiv.org/abs/2602.19473v1 | 2026-02-23 | final 64.0

Quantifying distributional separation across groups is fundamental in statistical learning and scientific discovery, yet most classical discrepancy measures are tailored to two-group comparisons. We generalize the underlap coefficient (UNL), a multi-group separation measure, to multivariate variables.

-> 통계적 군집화 방법이 동작 분석에 참고될 수 있음.

## 37위: AuditoryHuM: Auditory Scene Label Generation and Clustering using Human-MLLM Collaboration

- arXiv: http://arxiv.org/abs/2602.19409v1 | 2026-02-23 | final 54.4

Manual annotation of audio datasets is labour intensive, and it is challenging to balance label granularity with acoustic separability. We introduce AuditoryHuM, a novel framework for the unsupervised discovery and clustering of auditory scene labels using a collaborative Human-Multimodal Large Language Model (MLLM) approach.

-> 오디오 분석 기술이 에지 디바이스와 약하게 연관되어 점수 부여

## 38위: Extending CPU-less parallel execution of lambda calculus in digital logic with lists and arithmetic

- arXiv: http://arxiv.org/abs/2602.19884v1 | 2026-02-23 | final 50.4

Computer architecture is searching for new ways to make use of increasingly available digital logic without the serial bottlenecks of CPU-based design. Recent work has demonstrated a fully CPU-less approach to executing functional programs, by exploiting their inherent parallelisability to compile them directly into parallel digital logic.

-> CPU-less functional programming execution unrelated to sports video capture or analysis.

---

## 다시 보기

### Flexi-NeurA: A Configurable Neuromorphic Accelerator with Adaptive Bit-Precision Exploration for Edge SNNs (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.18140v1
- 점수: final 94.4

Neuromorphic accelerators promise unparalleled energy efficiency and computational density for spiking neural networks (SNNs), especially in edge intelligence applications. However, most existing platforms exhibit rigid architectures with limited configurability, restricting their adaptability to heterogeneous workloads and diverse design objectives. To address these limitations, we present Flexi-NeurA -- a parameterizable neuromorphic accelerator (core) that unifies configurability, flexibility, and efficiency. Flexi-NeurA allows users to customize neuron models, network structures, and precision settings at design time. By pairing these design-time configurability and flexibility features with a time-multiplexed and event-driven processing approach, Flexi-NeurA substantially reduces the required hardware resources and total power while preserving high efficiency and low inference latency. Complementing this, we introduce Flex-plorer, a heuristic-guided design-space exploration (DSE) tool that determines cost-effective fixed-point precisions for critical parameters -- such as decay factors, synaptic weights, and membrane potentials -- based on user-defined trade-offs between accuracy and resource usage. Based on the configuration selected through the Flex-plorer process, RTL code is configured to match the specified design. Comprehensive evaluations across MNIST, SHD, and DVS benchmarks demonstrate that the Flexi-NeurA and Flex-plorer co-framework achieves substantial improvements in accuracy, latency, and energy efficiency. A three-layer 256--128--10 fully connected network with LIF neurons mapped onto two processing cores achieves 97.23% accuracy on MNIST with 1.1~ms inference latency, utilizing only 1,623 logic cells, 7 BRAMs, and 111~mW of total power -- establishing Flexi-NeurA as a scalable, edge-ready neuromorphic platform.

-> 에너지 효율적이고 저지연의 신경모방 가속기 기술을 제안하여, 엣지 AI 하드웨어에 직접 적용 가능하다. 프로젝트의 실시간 스포츠 분석에 필수적이다.

### How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.18397v1
- 점수: final 89.6

Vision-Language-Action (VLA) models have recently demonstrated impressive capabilities across various embodied AI tasks. While deploying VLA models on real-world robots imposes strict real-time inference constraints, the inference performance landscape of VLA remains poorly understood due to the large combinatorial space of model architectures and inference systems. In this paper, we ask a fundamental research question: How should we design future VLA models and systems to support real-time inference? To address this question, we first introduce VLA-Perf, an analytical performance model that can analyze inference performance for arbitrary combinations of VLA models and inference systems. Using VLA-Perf, we conduct the first systematic study of the VLA inference performance landscape. From a model-design perspective, we examine how inference performance is affected by model scaling, model architectural choices, long-context video inputs, asynchronous inference, and dual-system model pipelines. From the deployment perspective, we analyze where VLA inference should be executed -- on-device, on edge servers, or in the cloud -- and how hardware capability and network performance jointly determine end-to-end latency. By distilling 15 key takeaways from our comprehensive evaluation, we hope this work can provide practical guidance for the design of future VLA models and inference systems.

-> 엣지 디바이스에서 VLA 모델의 실시간 추론 성능 최적화 기술을 제안해, AI 촬영 장비의 동작 분석 및 영상 처리 지연 시간 감소에 직접 기여한다.

### A reliability- and latency-driven task allocation framework for workflow applications in the edge-hub-cloud continuum (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.18158v1
- 점수: final 88.0

A growing number of critical workflow applications leverage a streamlined edge-hub-cloud architecture, which diverges from the conventional edge computing paradigm. An edge device, in collaboration with a hub device and a cloud server, often suffices for their reliable and efficient execution. However, task allocation in this streamlined architecture is challenging due to device limitations and diverse operating conditions. Given the inherent criticality of such workflow applications, where reliability and latency are vital yet conflicting objectives, an exact task allocation approach is typically required to ensure optimal solutions. As no existing method holistically addresses these issues, we propose an exact multi-objective task allocation framework to jointly optimize the overall reliability and latency of a workflow application in the specific edge-hub-cloud architecture. We present a comprehensive binary integer linear programming formulation that considers the relative importance of each objective. It incorporates time redundancy techniques, while accounting for crucial constraints often overlooked in related studies. We evaluate our approach using a relevant real-world workflow application, as well as synthetic workflows varying in structure, size, and criticality. In the real-world application, our method achieved average improvements of 84.19% in reliability and 49.81% in latency over baseline strategies, across relevant objective trade-offs. Overall, the experimental results demonstrate the effectiveness and scalability of our approach across diverse workflow applications for the considered system architecture, highlighting its practicality with runtimes averaging between 0.03 and 50.94 seconds across all examined workflows.

-> 엣지-허브-클라우드 아키텍처용 작업 할당 프레임워크로, 영상 처리 및 분석 작업의 신뢰성과 지연 시간 최적화에 직접 적용 가능하다.

### How Reliable is Your Service at the Extreme Edge? Analytical Modeling of Computational Reliability (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.16362v1
- 점수: final 88.0

Extreme Edge Computing (XEC) distributes streaming workloads across consumer-owned devices, exploiting their proximity to users and ubiquitous availability. Many such workloads are AI-driven, requiring continuous neural network inference for tasks like object detection and video analytics. Distributed Inference (DI), which partitions model execution across multiple edge devices, enables these streaming services to meet strict throughput and latency requirements. Yet consumer devices exhibit volatile computational availability due to competing applications and unpredictable usage patterns. This volatility poses a fundamental challenge: how can we quantify the probability that a device, or ensemble of devices, will maintain the processing rate required by a streaming service? This paper presents an analytical framework for computational reliability in XEC, defined as the probability that instantaneous capacity meets demand at a specified Quality of Service (QoS) threshold. We derive closed-form reliability expressions under two information regimes: Minimal Information (MI), requiring only declared operational bounds, and historical data, which refines estimates via Maximum Likelihood Estimation from past observations. The framework extends to multi-device deployments, providing reliability expressions for series, parallel, and partitioned workload configurations. We derive optimal workload allocation rules and analytical bounds for device selection, equipping orchestrators with tractable tools to evaluate deployment feasibility and configure distributed streaming systems. We validate the framework using real-time object detection with YOLO11m model as a representative DI streaming workload; experiments on emulated XED environments demonstrate close agreement between analytical predictions, Monte Carlo sampling, and empirical measurements across diverse capacity and demand configurations.

-> 엣지 디바이스 간 분산 추론의 계산적 신뢰성 분석 프레임워크로, 스포츠 영상 분석 서비스의 안정적 운영에 필수적이다.

### YOLO26: A Comprehensive Architecture Overview and Key Improvements (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.14582v1
- 점수: final 86.4

You Only Look Once (YOLO) has been the prominent model for computer vision in deep learning for a decade. This study explores the novel aspects of YOLO26, the most recent version in the YOLO series. The elimination of Distribution Focal Loss (DFL), implementation of End-to-End NMS-Free Inference, introduction of ProgLoss + Small-Target-Aware Label Assignment (STAL), and use of the MuSGD optimizer are the primary enhancements designed to improve inference speed, which is claimed to achieve a 43% boost in CPU mode. This is designed to allow YOLO26 to attain real-time performance on edge devices or those without GPUs. Additionally, YOLO26 offers improvements in many computer vision tasks, including instance segmentation, pose estimation, and oriented bounding box (OBB) decoding. We aim for this effort to provide more value than just consolidating information already included in the existing technical documentation. Therefore, we performed a rigorous architectural investigation into YOLO26, mostly using the source code available in its GitHub repository and its official documentation. The authentic and detailed operational mechanisms of YOLO26 are inside the source code, which is seldom extracted by others. The YOLO26 architectural diagram is shown as the outcome of the investigation. This study is, to our knowledge, the first one presenting the CNN-based YOLO26 architecture, which is the core of YOLO26. Our objective is to provide a precise architectural comprehension of YOLO26 for researchers and developers aspiring to enhance the YOLO model, ensuring it remains the leading deep learning model in computer vision.

-> 에지 디바이스용 실시간 객체 감지 및 포즈 추정 기술로 프로젝트의 핵심 기능인 운동 자세 분석과 경기 장면 인식에 직접 적용 가능. CPU 모드에서 43% 향상된 추론 속도(fps)가 RK3588 기기에서 실시간 성능 보장.

### SpecFuse: A Spectral-Temporal Fusion Predictive Control Framework for UAV Landing on Oscillating Marine Platforms (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.15633v1
- 점수: final 84.4

Autonomous landing of Uncrewed Aerial Vehicles (UAVs) on oscillating marine platforms is severely constrained by wave-induced multi-frequency oscillations, wind disturbances, and prediction phase lags in motion prediction. Existing methods either treat platform motion as a general random process or lack explicit modeling of wave spectral characteristics, leading to suboptimal performance under dynamic sea conditions. To address these limitations, we propose SpecFuse: a novel spectral-temporal fusion predictive control framework that integrates frequency-domain wave decomposition with time-domain recursive state estimation for high-precision 6-DoF motion forecasting of Uncrewed Surface Vehicles (USVs). The framework explicitly models dominant wave harmonics to mitigate phase lags, refining predictions in real time via IMU data without relying on complex calibration. Additionally, we design a hierarchical control architecture featuring a sampling-based HPO-RRT* algorithm for dynamic trajectory planning under non-convex constraints and a learning-augmented predictive controller that fuses data-driven disturbance compensation with optimization-based execution. Extensive validations (2,000 simulations + 8 lake experiments) show our approach achieves a 3.2 cm prediction error, 4.46 cm landing deviation, 98.7% / 87.5% success rates (simulation / real-world), and 82 ms latency on embedded hardware, outperforming state-of-the-art methods by 44%-48% in accuracy. Its robustness to wave-wind coupling disturbances supports critical maritime missions such as search and rescue and environmental monitoring. All code, experimental configurations, and datasets will be released as open-source to facilitate reproducibility.

-> 이 논문은 진동하는 해상 플랫폼에 UAV 착륙을 위한 제어 방법을 제안합니다. 핵심은 실시간 임베디드 예측 제어로, 파도와 바람 영향 하에서 안정적 위치 유지합니다. 에지 디바이스 관련성(82ms 지연)이 높아 움직이는 촬영 플랫폼에 적용 가능한 이유입니다.

### LAF-YOLOv10 with Partial Convolution Backbone, Attention-Guided Feature Pyramid, Auxiliary P2 Head, and Wise-IoU Loss for Small Object Detection in Drone Aerial Imagery (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.13378v1
- 점수: final 84.4

Unmanned aerial vehicles serve as primary sensing platforms for surveillance, traffic monitoring, and disaster response, making aerial object detection a central problem in applied computer vision. Current detectors struggle with UAV-specific challenges: targets spanning only a few pixels, cluttered backgrounds, heavy occlusion, and strict onboard computational budgets. This study introduces LAF-YOLOv10, built on YOLOv10n, integrating four complementary techniques to improve small-object detection in drone imagery. A Partial Convolution C2f (PC-C2f) module restricts spatial convolution to one quarter of backbone channels, reducing redundant computation while preserving discriminative capacity. An Attention-Guided Feature Pyramid Network (AG-FPN) inserts Squeeze-and-Excitation channel gates before multi-scale fusion and replaces nearest-neighbor upsampling with DySample for content-aware interpolation. An auxiliary P2 detection head at 160$\times$160 resolution extends localization to objects below 8$\times$8 pixels, while the P5 head is removed to redistribute parameters. Wise-IoU v3 replaces CIoU for bounding box regression, attenuating gradients from noisy annotations in crowded aerial scenes. The four modules address non-overlapping bottlenecks: PC-C2f compresses backbone computation, AG-FPN refines cross-scale fusion, the P2 head recovers spatial resolution, and Wise-IoU stabilizes regression under label noise. No individual component is novel; the contribution is the joint integration within a single YOLOv10 framework. Across three training runs (seeds 42, 123, 256), LAF-YOLOv10 achieves 35.1$\pm$0.3\% mAP@0.5 on VisDrone-DET2019 with 2.3\,M parameters, exceeding YOLOv10n by 3.3 points. Cross-dataset evaluation on UAVDT yields 35.8$\pm$0.4\% mAP@0.5. Benchmarks on NVIDIA Jetson Orin Nano confirm 24.3 FPS at FP16, demonstrating viability for embedded UAV deployment.

-> 이 논문은 드론 영상에서 작은 객체 감지를 위한 방법을 제안합니다. 핵심은 경량화된 YOLOv10 변종으로, 작은 대상(예: 선수, 공)을 복잡한 배경에서 정확히 식별합니다. 에지 디바이스용 최적화로 스포츠 촬영에 직접 적용 가능하며, 24.3 FPS와 2.3M 매개변수로 RK3588에서 효율적 실행이 핵심 이유입니다.

### Multi-Level Conditioning by Pairing Localized Text and Sketch for Fashion Image Generation (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.18309v1
- 점수: final 82.4

Sketches offer designers a concise yet expressive medium for early-stage fashion ideation by specifying structure, silhouette, and spatial relationships, while textual descriptions complement sketches to convey material, color, and stylistic details. Effectively combining textual and visual modalities requires adherence to the sketch visual structure when leveraging the guidance of localized attributes from text. We present LOcalized Text and Sketch with multi-level guidance (LOTS), a framework that enhances fashion image generation by combining global sketch guidance with multiple localized sketch-text pairs. LOTS employs a Multi-level Conditioning Stage to independently encode local features within a shared latent space while maintaining global structural coordination. Then, the Diffusion Pair Guidance stage integrates both local and global conditioning via attention-based guidance within the diffusion model's multi-step denoising process. To validate our method, we develop Sketchy, the first fashion dataset where multiple text-sketch pairs are provided per image. Sketchy provides high-quality, clean sketches with a professional look and consistent structure. To assess robustness beyond this setting, we also include an "in the wild" split with non-expert sketches, featuring higher variability and imperfections. Experiments demonstrate that our method strengthens global structural adherence while leveraging richer localized semantic guidance, achieving improvement over state-of-the-art. The dataset, platform, and code are publicly available.

-> 이 논문은 스케치와 텍스트로 패션 이미지 생성 방법을 제안합니다. 핵심은 다중 수준 조건화로, 구조와 세부 사항 정확히 반영합니다. 이미지 생성 기술이 프로젝트의 보정/변환 기능에 적용 가능해 선택 이유입니다.

### Unifying Color and Lightness Correction with View-Adaptive Curve Adjustment for Robust 3D Novel View Synthesis (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.18322v1
- 점수: final 82.4

High-quality image acquisition in real-world environments remains challenging due to complex illumination variations and inherent limitations of camera imaging pipelines. These issues are exacerbated in multi-view capture, where differences in lighting, sensor responses, and image signal processor (ISP) configurations introduce photometric and chromatic inconsistencies that violate the assumptions of photometric consistency underlying modern 3D novel view synthesis (NVS) methods, including Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), leading to degraded reconstruction and rendering quality. We propose Luminance-GS++, a 3DGS-based framework for robust NVS under diverse illumination conditions. Our method combines a globally view-adaptive lightness adjustment with a local pixel-wise residual refinement for precise color correction. We further design unsupervised objectives that jointly enforce lightness correction and multi-view geometric and photometric consistency. Extensive experiments demonstrate state-of-the-art performance across challenging scenarios, including low-light, overexposure, and complex luminance and chromatic variations. Unlike prior approaches that modify the underlying representation, our method preserves the explicit 3DGS formulation, improving reconstruction fidelity while maintaining real-time rendering efficiency.

-> 이 논문은 다양한 조명에서 이미지 보정 방법을 제안합니다. 핵심은 뷰-적응형 밝기 조절로, 저조도/과노출 시 색상 일관성 유지합니다. 프로젝트의 이미지 보정 기능과 직접 연관되어 화질 개선이 핵심 이유입니다.

### FlexAM: Flexible Appearance-Motion Decomposition for Versatile Video Generation Control (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.13185v1
- 점수: final 82.0

Effective and generalizable control in video generation remains a significant challenge. While many methods rely on ambiguous or task-specific signals, we argue that a fundamental disentanglement of "appearance" and "motion" provides a more robust and scalable pathway. We propose FlexAM, a unified framework built upon a novel 3D control signal. This signal represents video dynamics as a point cloud, introducing three key enhancements: multi-frequency positional encoding to distinguish fine-grained motion, depth-aware positional encoding, and a flexible control signal for balancing precision and generative quality. This representation allows FlexAM to effectively disentangle appearance and motion, enabling a wide range of tasks including I2V/V2V editing, camera control, and spatial object editing. Extensive experiments demonstrate that FlexAM achieves superior performance across all evaluated tasks.

-> 이 논문은 동영상 생성 제어 방법을 제안합니다. 핵심은 외형과 움직임 분해로, 객체 편집 및 카메라 제어를 유연히 수행합니다. 프로젝트의 핵심인 비디오 편집과 직접 관련되어 하이라이트 자동 생성에 필수적인 이유입니다.

### MING: An Automated CNN-to-Edge MLIR HLS framework (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.11966v1
- 점수: final 82.0

Driven by the increasing demand for low-latency and real-time processing, machine learning applications are steadily migrating toward edge computing platforms, where Field-Programmable Gate Arrays (FPGAs) are widely adopted for their energy efficiency compared to CPUs and GPUs. To generate high-performance and low-power FPGA designs, several frameworks built upon High Level Synthesis (HLS) vendor tools have been proposed, among which MLIR-based frameworks are gaining significant traction due to their extensibility and ease of use. However, existing state-of-the-art frameworks often overlook the stringent resource constraints of edge devices. To address this limitation, we propose MING, an Multi-Level Intermediate Representation (MLIR)-based framework that abstracts and automates the HLS design process. Within this framework, we adopt a streaming architecture with carefully managed buffers, specifically designed to handle resource constraints while ensuring low-latency. In comparison with recent frameworks, our approach achieves on average 15x speedup for standard Convolutional Neural Network (CNN) kernels with up to four layers, and up to 200x for single-layer kernels. For kernels with larger input sizes, MING is capable of generating efficient designs that respect hardware resource constraints, whereas state-of-the-art frameworks struggle to meet.

-> 에지 디바이스용 CNN 배포 최적화 프레임워크로 저지연 스트리밍 처리와 자원 제약 해결. 프로젝트의 rk3588 기반 실시간 스포츠 영상 분석 핵심 기술에 직접 적용 가능해 중요함.

### Quantization-Aware Collaborative Inference for Large Embodied AI Models (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.13052v1
- 점수: final 82.0

Large artificial intelligence models (LAIMs) are increasingly regarded as a core intelligence engine for embodied AI applications. However, the massive parameter scale and computational demands of LAIMs pose significant challenges for resource-limited embodied agents. To address this issue, we investigate quantization-aware collaborative inference (co-inference) for embodied AI systems. First, we develop a tractable approximation for quantization-induced inference distortion. Based on this approximation, we derive lower and upper bounds on the quantization rate-inference distortion function, characterizing its dependence on LAIM statistics, including the quantization bit-width. Next, we formulate a joint quantization bit-width and computation frequency design problem under delay and energy constraints, aiming to minimize the distortion upper bound while ensuring tightness through the corresponding lower bound. Extensive evaluations validate the proposed distortion approximation, the derived rate-distortion bounds, and the effectiveness of the proposed joint design. Particularly, simulations and real-world testbed experiments demonstrate the effectiveness of the proposed joint design in balancing inference quality, latency, and energy consumption in edge embodied AI systems.

-> 엣지 AI 모델의 양자화 협력 추론 기술로 자원 제약 환경 최적화. 스포츠 디바이스에서 대형 모델 효율적 실행에 필수적이라 중요함.

### Floe: Federated Specialization for Real-Time LLM-SLM Inference (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.14302v1
- 점수: final 82.0

Deploying large language models (LLMs) in real-time systems remains challenging due to their substantial computational demands and privacy concerns. We propose Floe, a hybrid federated learning framework designed for latency-sensitive, resource-constrained environments. Floe combines a cloud-based black-box LLM with lightweight small language models (SLMs) on edge devices to enable low-latency, privacy-preserving inference. Personal data and fine-tuning remain on-device, while the cloud LLM contributes general knowledge without exposing proprietary weights. A heterogeneity-aware LoRA adaptation strategy enables efficient edge deployment across diverse hardware, and a logit-level fusion mechanism enables real-time coordination between edge and cloud models. Extensive experiments demonstrate that Floe enhances user privacy and personalization. Moreover, it significantly improves model performance and reduces inference latency on edge devices under real-time constraints compared with baseline approaches.

-> 에지-클라우드 연합 LLM/SLM 실시간 추론 솔루션. 스포츠 경기 전략 분석을 위한 저지연 언어 모델 핵심에 직접 연관됨.

### A Self-Supervised Approach on Motion Calibration for Enhancing Physical Plausibility in Text-to-Motion (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.18199v1
- 점수: final 81.6

Generating semantically aligned human motion from textual descriptions has made rapid progress, but ensuring both semantic and physical realism in motion remains a challenge. In this paper, we introduce the Distortion-aware Motion Calibrator (DMC), a post-hoc module that refines physically implausible motions (e.g., foot floating) while preserving semantic consistency with the original textual description. Rather than relying on complex physical modeling, we propose a self-supervised and data-driven approach, whereby DMC learns to obtain physically plausible motions when an intentionally distorted motion and the original textual descriptions are given as inputs. We evaluate DMC as a post-hoc module to improve motions obtained from various text-to-motion generation models and demonstrate its effectiveness in improving physical plausibility while enhancing semantic consistency. The experimental results show that DMC reduces FID score by 42.74% on T2M and 13.20% on T2M-GPT, while also achieving the highest R-Precision. When applied to high-quality models like MoMask, DMC improves the physical plausibility of motions by reducing penetration by 33.0% as well as adjusting floating artifacts closer to the ground-truth reference. These results highlight that DMC can serve as a promising post-hoc motion refinement framework for any kind of text-to-motion models by incorporating textual semantics and physical plausibility.

-> 자가 지도 동작 보정 기술로 물리적 타당성 향상. 스포츠 동작 분석 시 발 떠림 같은 오류 보정에 직접 활용 가능해 중요함.

### Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.17997v1
- 점수: final 80.0

Whole-brain biological neural networks naturally support the learning and control of whole-body movements. However, the use of brain connectomes as neural network controllers in embodied reinforcement learning remains unexplored. We investigate using the exact neural architecture of an adult fruit fly's brain for the control of its body movement. We develop Fly-connectomic Graph Model (FlyGM), whose static structure is identical to the complete connectome of an adult Drosophila for whole-body locomotion control. To perform dynamical control, FlyGM represents the static connectome as a directed message-passing graph to impose a biologically grounded information flow from sensory inputs to motor outputs. Integrated with a biomechanical fruit fly model, our method achieves stable control across diverse locomotion tasks without task-specific architectural tuning. To verify the structural advantages of the connectome-based model, we compare it against a degree-preserving rewired graph, a random graph, and multilayer perceptrons, showing that FlyGM yields higher sample efficiency and superior performance. This work demonstrates that static brain connectomes can be transformed to instantiate effective neural policy for embodied learning of movement control.

-> 생물학적 신경망 기반 운동 제어 기술이 스포츠 동작 분석 및 하드웨어 개발에 적용 가능합니다.

### Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.16545v1
- 점수: final 80.0

Video recognition models are typically trained on fixed taxonomies which are often too coarse, collapsing distinctions in object, manner or outcome under a single label. As tasks and definitions evolve, such models cannot accommodate emerging distinctions and collecting new annotations and retraining to accommodate such changes is costly. To address these challenges, we introduce category splitting, a new task where an existing classifier is edited to refine a coarse category into finer subcategories, while preserving accuracy elsewhere. We propose a zero-shot editing method that leverages the latent compositional structure of video classifiers to expose fine-grained distinctions without additional data. We further show that low-shot fine-tuning, while simple, is highly effective and benefits from our zero-shot initialization. Experiments on our new video benchmarks for category splitting demonstrate that our method substantially outperforms vision-language baselines, improving accuracy on the newly split categories without sacrificing performance on the rest. Project page: https://kaitingliu.github.io/Category-Splitting/.

-> 제로샷 비디오 분류기 미세 조정 기술. 스포츠 장면 세부 인식(예: 슛 종류 분류) 능력 향상에 직접 적용 가능함.

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
