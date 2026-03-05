# CAPP!C_AI 논문 리포트 (2026-03-05)

> 수집 88 | 필터 84 | 폐기 14 | 평가 84 | 출력 58 | 기준 50점

검색 윈도우: 2026-03-04T00:00:00+00:00 ~ 2026-03-05T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 29

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: Helios: Real Real-Time Long Video Generation Model

- arXiv: http://arxiv.org/abs/2603.04379v1
- PDF: https://arxiv.org/pdf/2603.04379v1
- 발행일: 2026-03-04
- 카테고리: cs.CV
- 점수: final 98.4 (llm_adjusted:98 = base:85 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
We introduce Helios, the first 14B video generation model that runs at 19.5 FPS on a single NVIDIA H100 GPU and supports minute-scale generation while matching the quality of a strong baseline. We make breakthroughs along three key dimensions: (1) robustness to long-video drifting without commonly used anti-drifting heuristics such as self-forcing, error-banks, or keyframe sampling; (2) real-time generation without standard acceleration techniques such as KV-cache, sparse/linear attention, or quantization; and (3) training without parallelism or sharding frameworks, enabling image-diffusion-scale batch sizes while fitting up to four 14B models within 80 GB of GPU memory. Specifically, Helios is a 14B autoregressive diffusion model with a unified input representation that natively supports T2V, I2V, and V2V tasks. To mitigate drifting in long-video generation, we characterize typical failure modes and propose simple yet effective training strategies that explicitly simulate drifting during training, while eliminating repetitive motion at its source. For efficiency, we heavily compress the historical and noisy context and reduce the number of sampling steps, yielding computational costs comparable to -- or lower than -- those of 1.3B video generative models. Moreover, we introduce infrastructure-level optimizations that accelerate both inference and training while reducing memory consumption. Extensive experiments demonstrate that Helios consistently outperforms prior methods on both short- and long-video generation. We plan to release the code, base model, and distilled model to support further development by the community.

**선정 근거**
엣지 컴퓨팅 프레임워크가 AI 카메라 엣지 디바이스 배포에 직접적으로 관련

**활용 인사이트**
FCPP 라이브러리를 활용해 분산된 엣지 디바이스 간 상호작용 구현 가능, 실시간 영상 처리 지연 최소화

## 2위: Real Eyes Realize Faster: Gaze Stability and Pupil Novelty for Efficient Egocentric Learning

- arXiv: http://arxiv.org/abs/2603.04098v1
- PDF: https://arxiv.org/pdf/2603.04098v1
- 발행일: 2026-03-04
- 카테고리: cs.CV, cs.HC
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Always-on egocentric cameras are increasingly used as demonstrations for embodied robotics, imitation learning, and assistive AR, but the resulting video streams are dominated by redundant and low-quality frames. Under the storage and battery constraints of wearable devices, choosing which frames to keep is as important as how to learn from them. We observe that modern eye-tracking headsets provide a continuous, training-free side channel that decomposes into two complementary axes: gaze fixation captures visual stability (quality), while pupil response captures arousal-linked moments (novelty). We operationalize this insight as a Dual-Criterion Frame Curator that first gates frames by gaze quality and then ranks the survivors by pupil-derived novelty. On the Visual Experience Dataset (VEDB), curated frames at 10% budget match the classification performance of the full stream, and naive signal fusion consistently destroys both contributions. The benefit is task-dependent: pupil ranking improves activity recognition, while gaze-only selection already dominates for scene recognition, confirming that the two signals serve genuinely different roles. Our method requires no model inference and operates at capture time, offering a path toward efficient, always-on egocentric data curation.

**선정 근거**
안와 추적 데이터를 활용한 효율적인 프레임 선택 방식으로 스포츠 하이라이트 자동 추출에 직접 적용 가능

**활용 인사이트**
시선 응시 및 동공 반응 데이터를 분석해 중요한 순간을 자동으로 감지하고 하이라이트 영상을 생성함

## 3위: RIVER: A Real-Time Interaction Benchmark for Video LLMs

- arXiv: http://arxiv.org/abs/2603.03985v1
- PDF: https://arxiv.org/pdf/2603.03985v1
- 코드: https://github.com/OpenGVLab/RIVER
- 발행일: 2026-03-04
- 카테고리: cs.CV
- 점수: final 96.0 (llm_adjusted:95 = base:82 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
The rapid advancement of multimodal large language models has demonstrated impressive capabilities, yet nearly all operate in an offline paradigm, hindering real-time interactivity. Addressing this gap, we introduce the Real-tIme Video intERaction Bench (RIVER Bench), designed for evaluating online video comprehension. RIVER Bench introduces a novel framework comprising Retrospective Memory, Live-Perception, and Proactive Anticipation tasks, closely mimicking interactive dialogues rather than responding to entire videos at once. We conducted detailed annotations using videos from diverse sources and varying lengths, and precisely defined the real-time interactive format. Evaluations across various model categories reveal that while offline models perform well in single question-answering tasks, they struggle with real-time processing. Addressing the limitations of existing models in online video interaction, especially their deficiencies in long-term memory and future perception, we proposed a general improvement method that enables models to interact with users more flexibly in real time. We believe this work will significantly advance the development of real-time interactive video understanding models and inspire future research in this emerging field. Datasets and code are publicly available at https://github.com/OpenGVLab/RIVER.

**선정 근거**
실시간 비디오 이해 및 상호작용 기술로 스포츠 경기 분석에 적용 가능

## 4위: Agentic Peer-to-Peer Networks: From Content Distribution to Capability and Action Sharing

- arXiv: http://arxiv.org/abs/2603.03753v1
- PDF: https://arxiv.org/pdf/2603.03753v1
- 발행일: 2026-03-04
- 카테고리: cs.NI, cs.AI
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
The ongoing shift of AI models from centralized cloud APIs to local AI agents on edge devices is enabling \textit{Client-Side Autonomous Agents (CSAAs)} -- persistent personal agents that can plan, access local context, and invoke tools on behalf of users. As these agents begin to collaborate by delegating subtasks directly between clients, they naturally form \emph{Agentic Peer-to-Peer (P2P) Networks}. Unlike classic file-sharing overlays where the exchanged object is static, hash-indexed content (e.g., files in BitTorrent), agentic overlays exchange \emph{capabilities and actions} that are heterogeneous, state-dependent, and potentially unsafe if delegated to untrusted peers. This article outlines the networking foundations needed to make such collaboration practical. We propose a plane-based reference architecture that decouples connectivity/identity, semantic discovery, and execution. Besides, we introduce signed, soft-state capability descriptors to support intent- and constraint-aware discovery. To cope with adversarial settings, we further present a \textit{tiered verification} spectrum: Tier~1 relies on reputation signals, Tier~2 applies lightweight canary challenge-response with fallback selection, and Tier~3 requires evidence packages such as signed tool receipts/traces (and, when applicable, attestation). Using a discrete-event simulator that models registry-based discovery, Sybil-style index poisoning, and capability drift, we show that tiered verification substantially improves end-to-end workflow success while keeping discovery latency near-constant and control-plane overhead modest.

**선정 근거**
엣지 디바이스 및 AI 에이전트 기술이 프로젝트의 AI 촬영 장치 및 플랫폼에 직접 적용 가능

**활용 인사이트**
계층화된 검증 시스템을 통해 여러 AI 카메라 장치가 안전하게 연결되고 콘텐츠를 공유하는 플랫폼 구현 가능

## 5위: Lambdas at the Far Edge: a Tale of Flying Lambdas and Lambdas on Wheels

- arXiv: http://arxiv.org/abs/2603.04008v1
- PDF: https://arxiv.org/pdf/2603.04008v1
- 발행일: 2026-03-04
- 카테고리: cs.DC, cs.PL, cs.RO
- 점수: final 94.4 (llm_adjusted:93 = base:85 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Aggregate Programming (AP) is a paradigm for programming the collective behaviour of sets of distributed devices, possibly situated at the network far edge, by relying on asynchronous proximity-based interactions. The eXchange Calculus (XC), a recently proposed foundational model for AP, is essentially a typed lambda calculus extended with an operator (the exchange operator) providing an implicit communication mechanism between neighbour devices. This paper provides a gentle introduction to XC and to its implementation as a C++ library, called FCPP. The FCPP library and toolchain has been mainly developed at the Department of Computer Science of the University of Turin, where Stefano Berardi spent most of his academic career conducting outstanding research about logical foundation of computer science and transmitting his passion for research to students and young researchers, often exploiting typed lambda calculi. An FCCP program is essentially a typed lambda term, and FCPP has been used to write code that has been deployed on devices at the far edge of the network, including rovers and (soon) Uncrewed Aerial Vehicles (UAVs); hence the title of the paper.

**선정 근거**
엣지 컴퓨팅 프레임워크가 AI 카메라 엣지 디바이스 배포에 직접적으로 관련

## 6위: Architecture and evaluation protocol for transformer-based visual object tracking in UAV applications

- arXiv: http://arxiv.org/abs/2603.03904v1
- PDF: https://arxiv.org/pdf/2603.03904v1
- 발행일: 2026-03-04
- 카테고리: cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Object tracking from Unmanned Aerial Vehicles (UAVs) is challenged by platform dynamics, camera motion, and limited onboard resources. Existing visual trackers either lack robustness in complex scenarios or are too computationally demanding for real-time embedded use. We propose an Modular Asynchronous Tracking Architecture (MATA) that combines a transformer-based tracker with an Extended Kalman Filter, integrating ego-motion compensation from sparse optical flow and an object trajectory model. We further introduce a hardware-independent, embedded oriented evaluation protocol and a new metric called Normalized time to Failure (NT2F) to quantify how long a tracker can sustain a tracking sequence without external help. Experiments on UAV benchmarks, including an augmented UAV123 dataset with synthetic occlusions, show consistent improvements in Success and NT2F metrics across multiple tracking processing frequency. A ROS 2 implementation on a Nvidia Jetson AGX Orin confirms that the evaluation protocol more closely matches real-time performance on embedded systems.

**선정 근거**
객체 추적 기술이 스포츠 경기 자동 촬영에 적용 가능하며, 임베디드 시스템에서 실시간 작동 가능

**활용 인사이트**
모듈식 비동기 추적 아키텍처로 복잡한 경기 장면에서도 안정적 객체 추적 가능

## 7위: EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR

- arXiv: http://arxiv.org/abs/2603.04090v1
- PDF: https://arxiv.org/pdf/2603.04090v1
- 발행일: 2026-03-04
- 카테고리: cs.CV, cs.GR, cs.HC
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Egocentric human motion estimation is essential for AR/VR experiences, yet remains challenging due to limited body coverage from the egocentric viewpoint, frequent occlusions, and scarce labeled data. We present EgoPoseFormer v2, a method that addresses these challenges through two key contributions: (1) a transformer-based model for temporally consistent and spatially grounded body pose estimation, and (2) an auto-labeling system that enables the use of large unlabeled datasets for training. Our model is fully differentiable, introduces identity-conditioned queries, multi-view spatial refinement, causal temporal attention, and supports both keypoints and parametric body representations under a constant compute budget. The auto-labeling system scales learning to tens of millions of unlabeled frames via uncertainty-aware semi-supervised training. The system follows a teacher-student schema to generate pseudo-labels and guide training with uncertainty distillation, enabling the model to generalize to different environments. On the EgoBody3M benchmark, with a 0.8 ms latency on GPU, our model outperforms two state-of-the-art methods by 12.2% and 19.4% in accuracy, and reduces temporal jitter by 22.2% and 51.7%. Furthermore, our auto-labeling system further improves the wrist MPJPE by 13.1%.

**선정 근거**
인간 동작 추정 기술이 스포츠 동작 및 자세 분석에 직접적으로 적용 가능하며 실시간 처리 가능

**활용 인사이트**
0.8ms 지연 시간으로 실시간 동작 분석 가능, 자동 라벨링 시스템으로 대규모 데이터 학습 지원

## 8위: Yolo-Key-6D: Single Stage Monocular 6D Pose Estimation with Keypoint Enhancements

- arXiv: http://arxiv.org/abs/2603.03879v1
- PDF: https://arxiv.org/pdf/2603.03879v1
- 발행일: 2026-03-04
- 카테고리: cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Estimating the 6D pose of objects from a single RGB image is a critical task for robotics and extended reality applications. However, state-of-the-art multi stage methods often suffer from high latency, making them unsuitable for real time use. In this paper, we present Yolo-Key-6D, a novel single stage, end-to-end framework for monocular 6D pose estimation designed for both speed and accuracy. Our approach enhances a YOLO based architecture by integrating an auxiliary head that regresses the 2D projections of an object's 3D bounding box corners. This keypoint detection task significantly improves the network's understanding of 3D geometry. For stable end-to-end training, we directly regress rotation using a continuous 9D representation projected to SO(3) via singular value decomposition. On the LINEMOD and LINEMOD-Occluded benchmarks, YOLO-Key-6D achieves competitive accuracy scores of 96.24% and 69.41%, respectively, with the ADD(-S) 0.1d metric, while proving itself to operate in real time. Our results demonstrate that a carefully designed single stage method can provide a practical and effective balance of performance and efficiency for real world deployment.

**선정 근거**
6D 포즈 추정 기술이 스포츠 선수 자세 분석에 적용 가능하며 실시간 처리 가능

**활용 인사이트**
단일 RGB 이미지에서 3D 자세 추정, YOLO 기반 아키텍처로 실시간 성능과 정밀도 균형

## 9위: Toward Native ISAC Support in O-RAN Architectures for 6G

- arXiv: http://arxiv.org/abs/2603.03607v1
- PDF: https://arxiv.org/pdf/2603.03607v1
- 발행일: 2026-03-04
- 카테고리: cs.NI
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
ISAC is an emerging paradigm in 6G networks that enables environmental sensing using wireless communication infrastructure. Current O-RAN specifications lack the architectural primitives for sensing integration: no service models expose physical-layer observables, no execution frameworks support sub-millisecond sensing tasks, and fronthaul interfaces cannot correlate transmitted waveforms with their reflections.   This article proposes three extensions to O-RAN for monostatic sensing, where transmission and reception are co-located at the base station. First, we specify sensing dApps at the O-DU that process IQ samples to extract delay, Doppler, and angular features. Second, we define E2SM-SENS, a service model enabling xApps to subscribe to sensing telemetry with configurable periodicity. Third, we identify required Open Fronthaul metadata for waveform-echo association. We validate the architecture through a prototype implementation using beamforming and Full-Duplex operation, demonstrating closed-loop control with median end-to-end latency suitable for near-real-time sensing applications. While focused on monostatic configurations, the proposed interfaces extend to bistatic and cooperative sensing scenarios.

**선정 근거**
저지연 센싱 기술이 실시간 스포츠 분석에 적용 가능

## 10위: Adaptive Enhancement and Dual-Pooling Sequential Attention for Lightweight Underwater Object Detection with YOLOv10

- arXiv: http://arxiv.org/abs/2603.03807v1
- PDF: https://arxiv.org/pdf/2603.03807v1
- 발행일: 2026-03-04
- 카테고리: cs.CV
- 점수: final 92.0 (llm_adjusted:90 = base:80 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Underwater object detection constitutes a pivotal endeavor within the realms of marine surveillance and autonomous underwater systems; however, it presents significant challenges due to pronounced visual impairments arising from phenomena such as light absorption, scattering, and diminished contrast. In response to these formidable challenges, this manuscript introduces a streamlined yet robust framework for underwater object detection, grounded in the YOLOv10 architecture. The proposed method integrates a Multi-Stage Adaptive Enhancement module to improve image quality, a Dual-Pooling Sequential Attention (DPSA) mechanism embedded into the backbone to strengthen multi-scale feature representation, and a Focal Generalized IoU Objectness (FGIoU) loss to jointly improve localization accuracy and objectness prediction under class imbalance. Comprehensive experimental evaluations conducted on the RUOD and DUO benchmark datasets substantiate that the proposed DPSA_FGIoU_YOLOv10n attains exceptional performance, achieving mean Average Precision (mAP) scores of 88.9% and 88.0% at IoU threshold 0.5, respectively. In comparison to the baseline YOLOv10n, this represents enhancements of 6.7% for RUOD and 6.2% for DUO, all while preserving a compact model architecture comprising merely 2.8M parameters. These findings validate that the proposed framework establishes an efficacious equilibrium among accuracy, robustness, and real-time operational efficiency, making it suitable for deployment in resource-constrained underwater settings.

**선정 근거**
가벼운 모델(2.8M 파라미터)과 이미지 보정 기술이 스포츠 캠용 엣지 디바이스에 직접 적용 가능

**활용 인사이트**
다중 단계 적응형 향상 모듈과 이중 풀링 순차적 주의 메커니즘을 rk3588 기반 AI 카메라에 통합하여 실시간 객체 탐지 및 영상 향상 구현

## 11위: Exploring Challenges in Developing Edge-Cloud-Native Applications Across Multiple Business Domains

- arXiv: http://arxiv.org/abs/2603.03738v1
- PDF: https://arxiv.org/pdf/2603.03738v1
- 발행일: 2026-03-04
- 카테고리: cs.DC
- 점수: final 92.0 (llm_adjusted:90 = base:85 + bonus:+5)
- 플래그: 엣지

**개요**
As the convergence of cloud computing and advanced networking continues to reshape modern software development, edge-cloud-native paradigms have become essential for enabling scalable, resilient, and agile digital services that depend on high-performance, low-latency, and reliable communication. This study investigates the practical challenges of developing, deploying, and maintaining edge-cloud-native applications through in-depth interviews with professionals from diverse domains, including IT, finance, healthcare, education, and industry. Despite significant advancements in cloud technologies, practitioners, particularly those from non-technical backgrounds-continue to encounter substantial complexity stemming from fragmented toolchains, steep learning curves, and operational overhead of managing distributed networking and computing, ensuring consistent performance across hybrid environments, and navigating steep learning curves at the cloud-network boundary. Across sectors, participants consistently prioritized productivity, Quality of Service, and usability over conventional concerns such as cost or migration. These findings highlight the need for operationally simplified, SLA-aware, and developer-friendly platforms that streamline the full application lifecycle. This study contributes a practice-informed perspective to support the alignment of edge-cloud-native systems with the realities and needs of modern enterprises, offering critical insights for the advancement of seamless cloud-network convergence.

**선정 근거**
Edge-클라우드 개발 과제가 AI 카메라 디바이스에 직접 적용 가능

**활용 인사이트**
rk3588 기반 에지 디바이스와 클라우드 간 협업을 위한 단순화된 개발 플랫폼 적용

## 12위: DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation

- arXiv: http://arxiv.org/abs/2603.03744v1
- PDF: https://arxiv.org/pdf/2603.03744v1
- 발행일: 2026-03-04
- 카테고리: cs.CV
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
Estimating accurate, view-consistent geometry and camera poses from uncalibrated multi-view/video inputs remains challenging - especially at high spatial resolutions and over long sequences. We present DAGE, a dual-stream transformer whose main novelty is to disentangle global coherence from fine detail. A low-resolution stream operates on aggressively downsampled frames with alternating frame/global attention to build a view-consistent representation and estimate cameras efficiently, while a high-resolution stream processes the original images per-frame to preserve sharp boundaries and small structures. A lightweight adapter fuses these streams via cross-attention, injecting global context without disturbing the pretrained single-frame pathway. This design scales resolution and clip length independently, supports inputs up to 2K, and maintains practical inference cost. DAGE delivers sharp depth/pointmaps, strong cross-view consistency, and accurate poses, establishing new state-of-the-art results for video geometry estimation and multi-view reconstruction.

**선정 근거**
듀얼 스트림 아키텍처로 스포츠 장면의 기하학적 분석 가능

**활용 인사이트**
저해상도 스트림으로 전체 장면 파악, 고해상도 스트림으로 세부 동작 분석하여 선수 기술 분석에 적용

## 13위: Scaling Dense Event-Stream Pretraining from Visual Foundation Models

- arXiv: http://arxiv.org/abs/2603.03969v1
- PDF: https://arxiv.org/pdf/2603.03969v1
- 발행일: 2026-03-04
- 카테고리: cs.CV
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
Learning versatile, fine-grained representations from irregular event streams is pivotal yet nontrivial, primarily due to the heavy annotation that hinders scalability in dataset size, semantic richness, and application scope. To mitigate this dilemma, we launch a novel self-supervised pretraining method that distills visual foundation models (VFMs) to push the boundaries of event representation at scale. Specifically, we curate an extensive synchronized image-event collection to amplify cross-modal alignment. Nevertheless, due to inherent mismatches in sparsity and granularity between image-event domains, existing distillation paradigms are prone to semantic collapse in event representations, particularly at high resolutions. To bridge this gap, we propose to extend the alignment objective to semantic structures provided off-the-shelf by VFMs, indicating a broader receptive field and stronger supervision. The key ingredient of our method is a structure-aware distillation loss that grounds higher-quality image-event correspondences for alignment, optimizing dense event representations. Extensive experiments demonstrate that our approach takes a great leap in downstream benchmarks, significantly surpassing traditional methods and existing pretraining techniques. This breakthrough manifests in enhanced generalization, superior data efficiency and elevated transferability.

**선정 근거**
시각 기반 모델과 이벤트 스트림 처리 기술이 스포츠 영상 분석에 직접적으로 적용 가능하여 하이라이트 장면 추출에 효과적

**활용 인사이트**
구조 인식 증류 손실을 활용해 스포츠 경기의 중요 순간을 정교하게 포착하고 다양한 스포츠에 대한 일반화 성능 향상

## 14위: Optimal Short Video Ordering and Transmission Scheduling for Reducing Video Delivery Cost in Peer-to-Peer CDNs

- arXiv: http://arxiv.org/abs/2603.03938v1
- PDF: https://arxiv.org/pdf/2603.03938v1
- 발행일: 2026-03-04
- 카테고리: cs.NI, cs.MM, eess.IV
- 점수: final 88.0 (llm_adjusted:85 = base:80 + bonus:+5)
- 플래그: 엣지

**개요**
The explosive growth of short video platforms has generated a massive surge in global traffic, imposing heavy financial burdens on content providers. While Peer-to-Peer Content Delivery Networks (PCDNs) offer a cost-effective alternative by leveraging resource-constrained edge nodes, the limited storage and concurrent service capacities of these peers struggle to absorb the intense temporal demand spikes characteristic of short video consumption. In this paper, we propose to minimize transmission costs by exploiting a novel degree of freedom, the inherent flexibility of server-driven playback sequences. We formulate the Optimal Video Ordering and Transmission Scheduling (OVOTS) problem as an Integer Linear Program to jointly optimize personalized video ordering and transmission scheduling. By strategically permuting playlists, our approach proactively smooths temporal traffic peaks, maximizing the offloading of requests to low-cost peer nodes. To solve the OVOTS problem, we provide a rigorous theoretical reduction of the OVOTS problem to an auxiliary Minimum Cost Maximum Flow (MCMF) formulation. Leveraging König's Edge Coloring Theorem, we prove the strict equivalence of these formulations and develop the Minimum-cost Maximum-flow with Edge Coloring (MMEC) algorithm, a globally optimal, polynomial-time solution. Extensive simulations demonstrate that MMEC significantly outperforms baseline strategies, achieving cost reductions of up to 67% compared to random scheduling and 36% compared to a simulated annealing approach. Our results establish playback sequence flexibility as a robust and highly effective paradigm for cost optimization in PCDN architectures.

**선정 근거**
단편 비디오 전송 최적화 기술이 콘텐츠 공유 플랫폼에 적용 가능

**활용 인사이트**
동영상 재생 순서 최적화로 PCDN 비용 최대 67% 절감 및 트래픽 피크 완화

## 15위: Point Cloud Feature Coding for Object Detection over an Error-Prone Cloud-Edge Collaborative System

- arXiv: http://arxiv.org/abs/2603.03890v1
- PDF: https://arxiv.org/pdf/2603.03890v1
- 코드: https://github.com/yuanhui0325/T-PCFC
- 발행일: 2026-03-04
- 카테고리: eess.IV
- 점수: final 86.4 (llm_adjusted:83 = base:70 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Cloud-edge collaboration enhances machine perception by combining the strengths of edge and cloud computing. Edge devices capture raw data (e.g., 3D point clouds) and extract salient features, which are sent to the cloud for deeper analysis and data fusion. However, efficiently and reliably transmitting features between cloud and edge devices remains a challenging problem. We focus on point cloud-based object detection and propose a task-driven point cloud compression and reliable transmission framework based on source and channel coding. To meet the low-latency and low-power requirements of edge devices, we design a lightweight yet effective feature compaction module that compresses the deepest feature among multi-scale representations by removing task-irrelevant regions and applying channel-wise dimensionality reduction to task-relevant areas. Then, a signal-to-noise ratio (SNR)-adaptive channel encoder dynamically encodes the attribute information of the compacted features, while a Low-Density Parity-Check (LDPC) encoder ensures reliable transmission of geometric information. At the cloud side, an SNR-adaptive channel decoder guides the decoding of attribute information, and the LDPC decoder corrects geometry errors. Finally, a feature decompaction module restores the channel-wise dimensionality, and a diffusion-based feature upsampling module reconstructs shallow-layer features, enabling multi-scale feature reconstruction. On the KITTI dataset, our method achieved a 172-fold reduction in feature size with 3D average precision scores of 93.17%, 86.96%, and 77.25% for easy, moderate, and hard objects, respectively, over a 0 dB SNR wireless channel. Our source code will be released on GitHub at: https://github.com/yuanhui0325/T-PCFC.

**선정 근거**
에지-클라우드 협업 아키텍처에 특징 압축 및 전송 기술이 적용 가능하여 실시간 스포츠 분석 시스템 구축에 유리

**활용 인사이트**
가벼운 특징 압축 모듈을 통해 rk3588 에지 디바이스의 성능을 최적화하고 노이즈 환경에서도 안정적인 데이터 전송 보장

## 16위: Semantic Bridging Domains: Pseudo-Source as Test-Time Connector

- arXiv: http://arxiv.org/abs/2603.03844v1
- PDF: https://arxiv.org/pdf/2603.03844v1
- 발행일: 2026-03-04
- 카테고리: cs.CL
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Distribution shifts between training and testing data are a critical bottleneck limiting the practical utility of models, especially in real-world test-time scenarios. To adapt models when the source domain is unknown and the target domain is unlabeled, previous works constructed pseudo-source domains via data generation and translation, then aligned the target domain with them. However, significant discrepancies exist between the pseudo-source and the original source domain, leading to potential divergence when correcting the target directly. From this perspective, we propose a Stepwise Semantic Alignment (SSA) method, viewing the pseudo-source as a semantic bridge connecting the source and target, rather than a direct substitute for the source. Specifically, we leverage easily accessible universal semantics to rectify the semantic features of the pseudo-source, and then align the target domain using the corrected pseudo-source semantics. Additionally, we introduce a Hierarchical Feature Aggregation (HFA) module and a Confidence-Aware Complementary Learning (CACL) strategy to enhance the semantic quality of the SSA process in the absence of source and ground truth of target domains. We evaluated our approach on tasks like semantic segmentation and image classification, achieving a 5.2% performance boost on GTA2Cityscapes over the state-of-the-art.

**선정 근거**
도메인 적응 기술이 엣지 디바이스에서의 스포츠 장면 분석에 적용 가능하여 다양한 환경에서 안정적인 성능 보장

**활용 인사이트**
SSA 방식을 적용해 훈련 데이터와 실제 스포츠 환경 간의 격차를 줄이고, HFA 모듈로 실시간 분석 정도 향상

## 17위: InfinityStory: Unlimited Video Generation with World Consistency and Character-Aware Shot Transitions

- arXiv: http://arxiv.org/abs/2603.03646v1
- PDF: https://arxiv.org/pdf/2603.03646v1
- 발행일: 2026-03-04
- 카테고리: cs.CV
- 점수: final 84.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
Generating long-form storytelling videos with consistent visual narratives remains a significant challenge in video synthesis. We present a novel framework, dataset, and a model that address three critical limitations: background consistency across shots, seamless multi-subject shot-to-shot transitions, and scalability to hour-long narratives. Our approach introduces a background-consistent generation pipeline that maintains visual coherence across scenes while preserving character identity and spatial relationships. We further propose a transition-aware video synthesis module that generates smooth shot transitions for complex scenarios involving multiple subjects entering or exiting frames, going beyond the single-subject limitations of prior work. To support this, we contribute with a synthetic dataset of 10,000 multi-subject transition sequences covering underrepresented dynamic scene compositions. On VBench, InfinityStory achieves the highest Background Consistency (88.94), highest Subject Consistency (82.11), and the best overall average rank (2.80), showing improved stability, smoother transitions, and better temporal coherence.

**선정 근거**
하이라이트 영상 생성 및 편집 기술로 직접적으로 관련되어 경기 주요 장면 자동 편집에 필수적

**활용 인사이트**
배경 일관성 유지와 캐릭터 인식 전환 기술을 적용해 스포츠 선수들의 움직임을 자연스럽게 연결

## 18위: Separators in Enhancing Autoregressive Pretraining for Vision Mamba

- arXiv: http://arxiv.org/abs/2603.03806v1
- PDF: https://arxiv.org/pdf/2603.03806v1
- 발행일: 2026-03-04
- 카테고리: cs.CV, cs.AI
- 점수: final 84.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
The state space model Mamba has recently emerged as a promising paradigm in computer vision, attracting significant attention due to its efficient processing of long sequence tasks. Mamba's inherent causal mechanism renders it particularly suitable for autoregressive pretraining. However, current autoregressive pretraining methods are constrained to short sequence tasks, failing to fully exploit Mamba's prowess in handling extended sequences. To address this limitation, we introduce an innovative autoregressive pretraining method for Vision Mamba that substantially extends the input sequence length. We introduce new \textbf{S}epara\textbf{T}ors for \textbf{A}uto\textbf{R}egressive pretraining to demarcate and differentiate between different images, known as \textbf{STAR}. Specifically, we insert identical separators before each image to demarcate its inception. This strategy enables us to quadruple the input sequence length of Vision Mamba while preserving the original dimensions of the dataset images. Employing this long sequence pretraining technique, our STAR-B model achieved an impressive accuracy of 83.5\% on ImageNet-1k, which is highly competitive in Vision Mamba. These results underscore the potential of our method in enhancing the performance of vision models through improved leveraging of long-range dependencies.

**선정 근거**
Vision Mamba for autoregressive pretraining is directly applicable to sports video processing and analysis.

**활용 인사이트**
STAR 방식으로 입력 시퀀스 길이를 확장하며 스포츠 경기 전체를 한 번에 처리해 분석 속도 향상

## 19위: A Baseline Study and Benchmark for Few-Shot Open-Set Action Recognition with Feature Residual Discrimination

- arXiv: http://arxiv.org/abs/2603.04125v1
- PDF: https://arxiv.org/pdf/2603.04125v1
- 발행일: 2026-03-04
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:75 + bonus:+3)
- 플래그: 코드 공개

**개요**
Few-Shot Action Recognition (FS-AR) has shown promising results but is often limited by a closed-set assumption that fails in real-world open-set scenarios. While Few-Shot Open-Set (FSOS) recognition is well-established for images, its extension to spatio-temporal video data remains underexplored. To address this, we propose an architectural extension based on a Feature-Residual Discriminator (FR-Disc), adapting previous work on skeletal data to the more complex video domain. Extensive experiments on five datasets demonstrate that while common open-set techniques provide only marginal gains, our FR-Disc significantly enhances unknown rejection capabilities without compromising closed-set accuracy, setting a new state-of-the-art for FSOS-AR. The project website, code, and benchmark are available at: https://hsp-iit.github.io/fsosar/.

**선정 근거**
오픈셋 액션 인식 기술이 스포츠 경기 전략 분석에 적용 가능하며 코드 공개됨

**활용 인사이트**
FR-Disc를 적용해 알려지지 않은 동작도 식별하고 경기 전략 분석 정확도를 높일 수 있음

## 20위: HE-VPR: Height Estimation Enabled Aerial Visual Place Recognition Against Scale Variance

- arXiv: http://arxiv.org/abs/2603.04050v1
- PDF: https://arxiv.org/pdf/2603.04050v1
- 코드: https://github.com/hmf21/HE-VPR
- 발행일: 2026-03-04
- 카테고리: cs.RO
- 점수: final 82.4 (llm_adjusted:78 = base:65 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
In this work, we propose HE-VPR, a visual place recognition (VPR) framework that incorporates height estimation. Our system decouples height inference from place recognition, allowing both modules to share a frozen DINOv2 backbone. Two lightweight bypass adapter branches are integrated into our system. The first estimates the height partition of the query image via retrieval from a compact height database, and the second performs VPR within the corresponding height-specific sub-database. The adaptation design reduces training cost and significantly decreases the search space of the database. We also adopt a center-weighted masking strategy to further enhance the robustness against scale differences. Experiments on two self-collected challenging multi-altitude datasets demonstrate that HE-VPR achieves up to 6.1\% Recall@1 improvement over state-of-the-art ViT-based baselines and reduces memory usage by up to 90\%. These results indicate that HE-VPR offers a scalable and efficient solution for height-aware aerial VPR, enabling practical deployment in GNSS-denied environments. All the code and datasets for this work have been released on https://github.com/hmf21/HE-VPR.

**선정 근거**
경공학 설계와 고도 추정 기술이 스포츠 장면 분석에 간접적으로 적용 가능

**활용 인사이트**
다중 고도 데이터베이스 검색 기술을 활용해 다양한 각도에서 촬영된 스포츠 장면 분석 가능

## 21위: Semi-Supervised Generative Learning via Latent Space Distribution Matching

- arXiv: http://arxiv.org/abs/2603.04223v1
- PDF: https://arxiv.org/pdf/2603.04223v1
- 발행일: 2026-03-04
- 카테고리: stat.ML, cs.LG
- 점수: final 82.4 (llm_adjusted:78 = base:78 + bonus:+0)

**개요**
We introduce Latent Space Distribution Matching (LSDM), a novel framework for semi-supervised generative modeling of conditional distributions. LSDM operates in two stages: (i) learning a low-dimensional latent space from both paired and unpaired data, and (ii) performing joint distribution matching in this space via the 1-Wasserstein distance, using only paired data. This two-step approach minimizes an upper bound on the 1-Wasserstein distance between joint distributions, reducing reliance on scarce paired samples while enabling fast one-step generation. Theoretically, we establish non-asymptotic error bounds and demonstrate a key benefit of unpaired data: enhanced geometric fidelity in generated outputs. Furthermore, by extending the scope of its two core steps, LSDM provides a coherent statistical perspective that connects to a broad class of latent-space approaches. Notably, Latent Diffusion Models (LDMs) can be viewed as a variant of LSDM, in which joint distribution matching is achieved indirectly via score matching. Consequently, our results also provide theoretical insights into the consistency of LDMs. Empirical evaluations on real-world image tasks, including class-conditional generation and image super-resolution, demonstrate the effectiveness of LSDM in leveraging unpaired data to enhance generation quality.

**선정 근거**
생성적 학습 기술이 스포츠 콘텐츠의 영상/이미지 보정에 적용 가능

**활용 인사이트**
잠재 공간 분포 매칭을 통한 반지도 학습으로 스포츠 영상의 화질 향상 및 스타일 변환 가능

## 22위: SSR: A Generic Framework for Text-Aided Map Compression for Localization

- arXiv: http://arxiv.org/abs/2603.04272v1
- PDF: https://arxiv.org/pdf/2603.04272v1
- 발행일: 2026-03-04
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Mapping is crucial in robotics for localization and downstream decision-making. As robots are deployed in ever-broader settings, the maps they rely on continue to increase in size. However, storing these maps indefinitely (cold storage), transferring them across networks, or sending localization queries to cloud-hosted maps imposes prohibitive memory and bandwidth costs. We propose a text-enhanced compression framework that reduces both memory and bandwidth footprints while retaining high-fidelity localization. The key idea is to treat text as an alternative modality: one that can be losslessly compressed with large language models. We propose leveraging lightweight text descriptions combined with very small image feature vectors, which capture "complementary information" as a compact representation for the mapping task. Building on this, our novel technique, Similarity Space Replication (SSR), learns an adaptive image embedding in one shot that captures only the information "complementary" to the text descriptions. We validate our compression framework on multiple downstream localization tasks, including Visual Place Recognition as well as object-centric Monte Carlo localization in both indoor and outdoor settings. SSR achieves 2 times better compression than competing baselines on state-of-the-art datasets, including TokyoVal, Pittsburgh30k, Replica, and KITTI.

**선정 근거**
SSR 프레임워크는 스포츠 영상 압축 및 저장에 직접적으로 적용 가능하며, 텍스트와 시각 정보의 보완적 접근법이 하이라이트 장면 식별에 유용합니다.

**활용 인사이트**
경기 영상을 텍스트 메타데이터와 함께 압축하여 저장 공간을 절반으로 줄이고, 경기 중요 순간을 자동으로 식별하여 하이라이트 영상 생성 효율을 높일 수 있습니다.

## 23위: DISC: Dense Integrated Semantic Context for Large-Scale Open-Set Semantic Mapping

- arXiv: http://arxiv.org/abs/2603.03935v1
- PDF: https://arxiv.org/pdf/2603.03935v1
- 코드: https://github.com/DFKI-NI/DISC
- 발행일: 2026-03-04
- 카테고리: cs.CV, cs.RO
- 점수: final 78.4 (llm_adjusted:73 = base:60 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Open-set semantic mapping enables language-driven robotic perception, but current instance-centric approaches are bottlenecked by context-depriving and computationally expensive crop-based feature extraction. To overcome this fundamental limitation, we introduce DISC (Dense Integrated Semantic Context), featuring a novel single-pass, distance-weighted extraction mechanism. By deriving high-fidelity CLIP embeddings directly from the vision transformer's intermediate layers, our approach eliminates the latency and domain-shift artifacts of traditional image cropping, yielding pure, mask-aligned semantic representations. To fully leverage these features in large-scale continuous mapping, DISC is built upon a fully GPU-accelerated architecture that replaces periodic offline processing with precise, on-the-fly voxel-level instance refinement. We evaluate our approach on standard benchmarks (Replica, ScanNet) and a newly generated large-scale-mapping dataset based on Habitat-Matterport 3D (HM3DSEM) to assess scalability across complex scenes in multi-story buildings. Extensive evaluations demonstrate that DISC significantly surpasses current state-of-the-art zero-shot methods in both semantic accuracy and query retrieval, providing a robust, real-time capable framework for robotic deployment. The full source code, data generation and evaluation pipelines will be made available at https://github.com/DFKI-NI/DISC.

**선정 근거**
스포츠 장면의 의미론적 분석을 위한 기술로 간접적으로 관련

## 24위: Detection and Identification of Penguins Using Appearance and Motion Features

- arXiv: http://arxiv.org/abs/2603.03603v1
- PDF: https://arxiv.org/pdf/2603.03603v1
- 발행일: 2026-03-04
- 카테고리: cs.CV, q-bio.QM
- 점수: final 78.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
In animal facilities, continuous surveillance of penguins is essential yet technically challenging due to their homogeneous visual characteristics, rapid and frequent posture changes, and substantial environmental noise such as water reflections. In this study, we propose a framework that enhances both detection and identification performance by integrating appearance and motion features. For detection, we adapted YOLO11 to process consecutive frames to overcome the lack of temporal consistency in single-frame detectors. This approach leverages motion cues to detect targets even when distinct visual features are obscured. Our evaluation shows that fine-tuning the model with two-frame inputs improves mAP@0.5 from 0.922 to 0.933, outperforming the baseline, and successfully recovers individuals that are indistinguishable in static images. For identification, we introduce a tracklet-based contrastive learning approach applied after tracking. Through qualitative visualization, we demonstrate that the method produces coherent feature embeddings, bringing samples from the same individual closer in the feature space, suggesting the potential for mitigating ID switching.

**선정 근거**
움직임 특징을 활용한 객체 검출 기술이 스포츠 장면 자동 촬영에 적용 가능

**활용 인사이트**
연속 프레임 처리 기술로 빠르게 움직이는 선수들의 자동 촬영 및 개별 식별 가능

## 25위: MEM: Multi-Scale Embodied Memory for Vision Language Action Models

- arXiv: http://arxiv.org/abs/2603.03596v1
- PDF: https://arxiv.org/pdf/2603.03596v1
- 발행일: 2026-03-04
- 카테고리: cs.RO, cs.LG
- 점수: final 78.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Conventionally, memory in end-to-end robotic learning involves inputting a sequence of past observations into the learned policy. However, in complex multi-stage real-world tasks, the robot's memory must represent past events at multiple levels of granularity: from long-term memory that captures abstracted semantic concepts (e.g., a robot cooking dinner should remember which stages of the recipe are already done) to short-term memory that captures recent events and compensates for occlusions (e.g., a robot remembering the object it wants to pick up once its arm occludes it). In this work, our main insight is that an effective memory architecture for long-horizon robotic control should combine multiple modalities to capture these different levels of abstraction. We introduce Multi-Scale Embodied Memory (MEM), an approach for mixed-modal long-horizon memory in robot policies. MEM combines video-based short-horizon memory, compressed via a video encoder, with text-based long-horizon memory. Together, they enable robot policies to perform tasks that span up to fifteen minutes, like cleaning up a kitchen, or preparing a grilled cheese sandwich. Additionally, we find that memory enables MEM policies to intelligently adapt manipulation strategies in-context.

**선정 근거**
다중 규모 몽거 구조로 장기간 스포츠 경기 분석 및 전략 수립에 활용 가능한 기술

**활용 인사이트**
단기/장기 기억 결합으로 전체 경기 흐름 분석 및 전략적 인사이트 도출 가능

## 26위: Learning Hip Exoskeleton Control Policy via Predictive Neuromusculoskeletal Simulation

- arXiv: http://arxiv.org/abs/2603.04166v1
- PDF: https://arxiv.org/pdf/2603.04166v1
- 발행일: 2026-03-04
- 카테고리: cs.RO, cs.LG
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Developing exoskeleton controllers that generalize across diverse locomotor conditions typically requires extensive motion-capture data and biomechanical labeling, limiting scalability beyond instrumented laboratory settings. Here, we present a physics-based neuromusculoskeletal learning framework that trains a hip-exoskeleton control policy entirely in simulation, without motion-capture demonstrations, and deploys it on hardware via policy distillation. A reinforcement learning teacher policy is trained using a muscle-synergy action prior over a wide range of walking speeds and slopes through a two-stage curriculum, enabling direct comparison between assisted and no-exoskeleton conditions. In simulation, exoskeleton assistance reduces mean muscle activation by up to 3.4% and mean positive joint power by up to 7.0% on level ground and ramp ascent, with benefits increasing systematically with walking speed. On hardware, the assistance profiles learned in simulation are preserved across matched speed-slope conditions (r: 0.82, RMSE: 0.03 Nm/kg), providing quantitative evidence of sim-to-real transfer without additional hardware tuning. These results demonstrate that physics-based neuromusculoskeletal simulation can serve as a practical and scalable foundation for exoskeleton controller development, substantially reducing experimental burden during the design phase.

**선정 근거**
신경 근골격 시뮬레이션을 통한 신체 활동 분석 기술

**활용 인사이트**
시뮬레이션을 통한 제어 정책 학습이 스포츠 동작 분석 및 보조에 적용 가능

## 27위: Data-Aware Random Feature Kernel for Transformers

- arXiv: http://arxiv.org/abs/2603.04127v1
- PDF: https://arxiv.org/pdf/2603.04127v1
- 발행일: 2026-03-04
- 카테고리: cs.LG, cs.AI
- 점수: final 76.0 (llm_adjusted:70 = base:65 + bonus:+5)
- 플래그: 엣지

**개요**
Transformers excel across domains, yet their quadratic attention complexity poses a barrier to scaling. Random-feature attention, as in Performers, can reduce this cost to linear in the sequence length by approximating the softmax kernel with positive random features drawn from an isotropic distribution. In pretrained models, however, queries and keys are typically anisotropic. This induces high Monte Carlo variance in isotropic sampling schemes unless one retrains the model or uses a large feature budget. Importance sampling can address this by adapting the sampling distribution to the input geometry, but complex data-dependent proposal distributions are often intractable. We show that by data aligning the softmax kernel, we obtain an attention mechanism which can both admit a tractable minimal-variance proposal distribution for importance sampling, and exhibits better training stability. Motivated by this finding, we introduce DARKFormer, a Data-Aware Random-feature Kernel transformer that features a data-aligned kernel geometry. DARKFormer learns the random-projection covariance, efficiently realizing an importance-sampled positive random-feature estimator for its data-aligned kernel. Empirically, DARKFormer narrows the performance gap with exact softmax attention, particularly in finetuning regimes where pretrained representations are anisotropic. By combining random-feature efficiency with data-aware kernels, DARKFormer advances kernel-based attention in resource-constrained settings.

**선정 근거**
트랜스포머 효율성 개선으로 엣지 기기에서의 스포츠 영상 처리에 관련

**활용 인사이트**
DARKFormer을 적용하면 rk3588 엣지 디바이스에서 복잡한 스포츠 영상 분석을 실시간으로 처리하면서도 정확도를 유지할 수 있어 하이라이트 자동 편집 및 동작 분석 성능 향상

## 28위: ORION: Intent-Aware Orchestration in Open RAN for SLA-Driven Network Management

- arXiv: http://arxiv.org/abs/2603.03667v1
- PDF: https://arxiv.org/pdf/2603.03667v1
- 발행일: 2026-03-04
- 카테고리: cs.NI
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
The disaggregation of the Radio Access Network (RAN) introduces unprecedented flexibility but significant operational complexity, necessitating automated management frameworks. However, current Open RAN (O-RAN) orchestration relies on fragmented manual policies, lacking end-to-end intent assurance from high-level requirements to low-level configurations. In this paper, we propose ORION, an O-RAN compliant intent orchestration framework that integrates Large Language Models (LLMs) via the Model Context Protocol (MCP) to translate natural language intents into enforceable network policies. ORION leverages a hierarchical agent architecture, combining an MCP-based Service Management and Orchestration (SMO) layer for semantic translation with a Non-Real-Time RIC rApp and Near-Real-Time RIC xApp for closed-loop enforcement. Extensive evaluations using GPT-5, Gemini 3 Pro, and Claude Opus demonstrate a 100% policy generation success rate for high-capacity models, highlighting significant trade-offs in reasoning efficiency. We show that ORION reduces provisioning complexity by automating the complete intent lifecycle, from ingestion to E2-level enforcement, paving the way for autonomous 6G networks.

**선정 근거**
의도 인식 오케스트레이션이 스포츠 분석 엣지 디바이스 구성에 적용 가능

## 29위: Learning Surgical Robotic Manipulation with 3D Spatial Priors

- arXiv: http://arxiv.org/abs/2603.03798v1
- PDF: https://arxiv.org/pdf/2603.03798v1
- 발행일: 2026-03-04
- 카테고리: cs.RO
- 점수: final 74.4 (llm_adjusted:68 = base:55 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Achieving 3D spatial awareness is crucial for surgical robotic manipulation, where precise and delicate operations are required. Existing methods either explicitly reconstruct the surgical scene prior to manipulation, or enhance multi-view features by adding wrist-mounted cameras to supplement the default stereo endoscopes. However, both paradigms suffer from notable limitations: the former easily leads to error accumulation and prevents end-to-end optimization due to its multi-stage nature, while the latter is rarely adopted in clinical practice since wrist-mounted cameras can interfere with the motion of surgical robot arms. In this work, we introduce the Spatial Surgical Transformer (SST), an end-to-end visuomotor policy that empowers surgical robots with 3D spatial awareness by directly exploring 3D spatial cues embedded in endoscopic images. First, we build Surgical3D, a large-scale photorealistic dataset containing 30K stereo endoscopic image pairs with accurate 3D geometry, addressing the scarcity of 3D data in surgical scenes. Based on Surgical3D, we finetune a powerful geometric transformer to extract robust 3D latent representations from stereo endoscopes images. These representations are then seamlessly aligned with the robot's action space via a lightweight multi-level spatial feature connector (MSFC), all within an endoscope-centric coordinate frame. Extensive real-robot experiments demonstrate that SST achieves state-of-the-art performance and strong spatial generalization on complex surgical tasks such as knot tying and ex-vivo organ dissection, representing a significant step toward practical clinical deployment. The dataset and code will be released.

**선정 근거**
3D 공간 인식 기술로 스포츠 동작 분석에 간접적으로 적용 가능

## 30위: Lightweight Visual Reasoning for Socially-Aware Robots

- arXiv: http://arxiv.org/abs/2603.03942v1
- PDF: https://arxiv.org/pdf/2603.03942v1
- 코드: https://github.com/alessioGalatolo/VLM-Reasoning-for-Robotics
- 발행일: 2026-03-04
- 카테고리: cs.RO
- 점수: final 74.4 (llm_adjusted:68 = base:60 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Robots operating in shared human environments must not only navigate, interact, and detect their surroundings, they must also interpret and respond to dynamic, and often unpredictable, human behaviours. Although recent advances have shown promise in enhancing robotic perception and instruction-following using Vision-Language Models (VLMs), they remain limited in addressing the complexities of multimodal human-robot interactions (HRI). Motivated by this challenge, we introduce a lightweight language-to-vision feedback module that closes the loop between an LLM and the vision encoder in VLMs. The module projects image-token hidden states through a gated Multi-Layer Perceptron (MLP) back into the encoder input, prompting a second pass that reinterprets the scene under text context. We evaluate this approach on three robotics-centred tasks: navigation in a simulated environment (Habitat), sequential scene description (Mementos-Robotics), and human-intention recognition (our HRI dataset). Results show that our method improves Qwen 2.5 (7B) by $3.3\%$ (less distance), $+0.057$ description score, and $+2.93\%$ accuracy, with less than $3\%$ extra parameters; Gemma 3 (4B) and LLaVA OV 1.5 (4B) show mixed navigation results but gains $+0.111,+0.055$ and $+10.81\%,+4.79\%$ on the latter two tasks. Code is available at https://github.com/alessioGalatolo/VLM-Reasoning-for-Robotics

**선정 근거**
가벼운 시각 추론 기술로 스포츠 장면 분석에 간접적으로 적용 가능

**활용 인사이트**
LLM과 비전 인코더 간의 피드백 루프를 통해 스포츠 장면의 의도와 행동을 정확히 분석할 수 있음

## 31위: Cross-Modal Mapping and Dual-Branch Reconstruction for 2D-3D Multimodal Industrial Anomaly Detection

- arXiv: http://arxiv.org/abs/2603.03939v1 | 2026-03-04 | final 72.8

Multimodal industrial anomaly detection benefits from integrating RGB appearance with 3D surface geometry, yet existing \emph{unsupervised} approaches commonly rely on memory banks, teacher-student architectures, or fragile fusion schemes, limiting robustness under noisy depth, weak texture, or missing modalities. This paper introduces \textbf{CMDR-IAD}, a lightweight and modality-flexible unsupervised framework for reliable anomaly detection in 2D+3D multimodal as well as single-modality (2D-only or 3D-only) settings.

-> 멀티모달 처리 기술이 스포츠 데이터 분석에 간접적으로 적용 가능하지만 산업 이상 탐지에 특화

## 32위: Activation Outliers in Transformer Quantization: Reproduction, Statistical Analysis, and Deployment Tradeoffs

- arXiv: http://arxiv.org/abs/2603.04308v1 | 2026-03-04 | final 72.8

Post-training quantization (PTQ) of transformers is known to suffer from severe accuracy degradation due to structured activation outliers, as originally analyzed by Bondarenko et al. (EMNLP 2021) in work associated with Qualcomm AI Research.

-> Discusses model quantization for edge deployment but focused on NLP transformers rather than computer vision for sports analysis

## 33위: Perception-Aware Time-Optimal Planning for Quadrotor Waypoint Flight

- arXiv: http://arxiv.org/abs/2603.04305v1 | 2026-03-04 | final 72.0

Agile quadrotor flight pushes the limits of control, actuation, and onboard perception. While time-optimal trajectory planning has been extensively studied, existing approaches typically neglect the tight coupling between vehicle dynamics, environmental geometry, and the visual requirements of onboard state estimation.

-> 자동 촬영을 위한 드론 카메라 제어 기술로 간접적으로 관련

## 34위: A multi-center analysis of deep learning methods for video polyp detection and segmentation

- arXiv: http://arxiv.org/abs/2603.04288v1 | 2026-03-04 | final 72.0

Colonic polyps are well-recognized precursors to colorectal cancer (CRC), typically detected during colonoscopy. However, the variability in appearance, location, and size of these polyps complicates their detection and removal, leading to challenges in effective surveillance, intervention, and subsequently CRC prevention.

-> Video analysis with temporal information for medical applications, not sports

## 35위: DQE-CIR: Distinctive Query Embeddings through Learnable Attribute Weights and Target Relative Negative Sampling in Composed Image Retrieval

- arXiv: http://arxiv.org/abs/2603.04037v1 | 2026-03-04 | final 72.0

Composed image retrieval (CIR) addresses the task of retrieving a target image by jointly interpreting a reference image and a modification text that specifies the intended change. Most existing methods are still built upon contrastive learning frameworks that treat the ground truth image as the only positive instance and all remaining images as negatives.

-> 이미지 검색 기술로 스포츠 영상 분석에 관련

## 36위: A Constrained RL Approach for Cost-Efficient Delivery of Latency-Sensitive Applications

- arXiv: http://arxiv.org/abs/2603.04353v1 | 2026-03-04 | final 72.0

Next-generation networks aim to provide performance guarantees to real-time interactive services that require timely and cost-efficient packet delivery. In this context, the goal is to reliably deliver packets with strict deadlines imposed by the application while minimizing overall resource allocation cost.

-> 실시간 처리 기술이 스포츠 분석에 적용 가능

## 37위: LifeBench: A Benchmark for Long-Horizon Multi-Source Memory

- arXiv: http://arxiv.org/abs/2603.03781v1 | 2026-03-04 | final 68.8

Long-term memory is fundamental for personalized agents capable of accumulating knowledge, reasoning over user experiences, and adapting across time. However, existing memory benchmarks primarily target declarative memory, specifically semantic and episodic types, where all information is explicitly presented in dialogues.

-> 장기 기억 벤치마크가 시간에 따른 스포츠 활동 분석에 적용 가능

## 38위: Large-Language-Model-Guided State Estimation for Partially Observable Task and Motion Planning

- arXiv: http://arxiv.org/abs/2603.03704v1 | 2026-03-04 | final 68.0

Robot planning in partially observable environments, where not all objects are known or visible, is a challenging problem, as it requires reasoning under uncertainty through partially observable Markov decision processes. During the execution of a computed plan, a robot may unexpectedly observe task-irrelevant objects, which are typically ignored by naive planners.

-> LLM-guided state estimation has some indirect relevance to sports strategy analysis but not directly applicable.

## 39위: CAMMSR: Category-Guided Attentive Mixture of Experts for Multimodal Sequential Recommendation

- arXiv: http://arxiv.org/abs/2603.04320v1 | 2026-03-04 | final 68.0

The explosion of multimedia data in information-rich environments has intensified the challenges of personalized content discovery, positioning recommendation systems as an essential form of passive data management. Multimodal sequential recommendation, which leverages diverse item information such as text and images, has shown great promise in enriching item representations and deepening the understanding of user interests.

-> Multimodal recommendation system for content discovery, partially relevant to sharing platform

## 40위: TextBoost: Boosting Scene Text Fidelity in Ultra-low Bitrate Image Compression

- arXiv: http://arxiv.org/abs/2603.04115v1 | 2026-03-04 | final 68.0

Ultra-low bitrate image compression faces a critical challenge: preserving small-font scene text while maintaining overall visual quality. Region-of-interest (ROI) bit allocation can prioritize text but often degrades global fidelity, leading to a trade-off between local accuracy and overall image quality.

-> Image compression technique that could be applied to enhance sports photos on edge devices

## 41위: Bridging Human Evaluation to Infrared and Visible Image Fusion

- arXiv: http://arxiv.org/abs/2603.03871v1 | 2026-03-04 | final 66.4

Infrared and visible image fusion (IVIF) integrates complementary modalities to enhance scene perception. Current methods predominantly focus on optimizing handcrafted losses and objective metrics, often resulting in fusion outcomes that do not align with human visual preferences.

-> 적외선 및 가시광선 이미지 융합 기술로 스포츠 영상 향상에 간접적으로 적용 가능한 방법론

## 42위: Behind the Prompt: The Agent-User Problem in Information Retrieval

- arXiv: http://arxiv.org/abs/2603.03630v1 | 2026-03-04 | final 66.4

User models in information retrieval rest on a foundational assumption that observed behavior reveals intent. This assumption collapses when the user is an AI agent privately configured by a human operator.

-> 에이전트-사용자 상호작용 인사이트가 플랫폼 콘텐츠 관리 시스템에 관련

## 43위: Reckless Designs and Broken Promises: Privacy Implications of Targeted Interactive Advertisements on Social Media Platforms

- arXiv: http://arxiv.org/abs/2603.03659v1 | 2026-03-04 | final 66.4

Popular social media platforms TikTok, Facebook and Instagram allow third-parties to run targeted advertising campaigns on sensitive attributes in-platform. These ads are interactive by default, meaning users can comment or ``react'' (e.g., ``like'', ``love'') to them.

-> Relates to social media platforms and advertising which aligns with project's sharing platform goal, but doesn't address core AI filming technology

## 44위: Scalable Evaluation of the Realism of Synthetic Environmental Augmentations in Images

- arXiv: http://arxiv.org/abs/2603.04325v1 | 2026-03-04 | final 64.0

Evaluation of AI systems often requires synthetic test cases, particularly for rare or safety-critical conditions that are difficult to observe in operational data. Generative AI offers a promising approach for producing such data through controllable image editing, but its usefulness depends on whether the resulting images are sufficiently realistic to support meaningful evaluation.

-> 이미지 현실성 평가 프레임워크가 이미지 보정 기술 개발에 간접적으로 참고 가능

## 45위: Long-Term Visual Localization in Dynamic Benthic Environments: A Dataset, Footprint-Based Ground Truth, and Visual Place Recognition Benchmark

- arXiv: http://arxiv.org/abs/2603.04056v1 | 2026-03-04 | final 64.0

Long-term visual localization has the potential to reduce cost and improve mapping quality in optical benthic monitoring with autonomous underwater vehicles (AUVs). Despite this potential, long-term visual localization in benthic environments remains understudied, primarily due to the lack of curated datasets for benchmarking.

-> Underwater visual localization techniques could be indirectly applicable to sports video analysis but not directly relevant.

## 46위: TreeLoc++: Robust 6-DoF LiDAR Localization in Forests with a Compact Digital Forest Inventory

- arXiv: http://arxiv.org/abs/2603.03695v1 | 2026-03-04 | final 64.0

Reliable localization is essential for sustainable forest management, as it allows robots or sensor systems to revisit and monitor the status of individual trees over long periods. In modern forestry, this management is structured around Digital Forest Inventories (DFIs), which encode stems using compact geometric attributes rather than raw data.

-> Forest localization using LiDAR has some indirect relevance to sports tracking but not directly applicable.

## 47위: Plug-and-Play blind super-resolution of real MRI images for improved multiple sclerosis diagnosis

- arXiv: http://arxiv.org/abs/2603.03876v1 | 2026-03-04 | final 64.0

Magnetic resonance imaging (MRI) is central to the diagnosis of multiple sclerosis, where the identification of biomarkers such as the central vein sign benefits from high-resolution images. However, most clinical brain MRI scans are performed using 1.5 T scanners, which provide lower sensitivity compared to higher-field systems.

-> Image enhancement techniques for medical MRI, not sports applications

## 48위: Enhancing Authorship Attribution with Synthetic Paintings

- arXiv: http://arxiv.org/abs/2603.04343v1 | 2026-03-04 | final 64.0

Attributing authorship to paintings is a historically complex task, and one of its main challenges is the limited availability of real artworks for training computational models. This study investigates whether synthetic images, generated through DreamBooth fine-tuning of Stable Diffusion, can improve the performance of classification models in this context.

-> Image generation technique that could be applied to enhance sports photos

## 49위: Weakly Supervised Patch Annotation for Improved Screening of Diabetic Retinopathy

- arXiv: http://arxiv.org/abs/2603.03991v1 | 2026-03-04 | final 64.0

Diabetic Retinopathy (DR) requires timely screening to prevent irreversible vision loss. However, its early detection remains a significant challenge since often the subtle pathological manifestations (lesions) get overlooked due to insufficient annotation.

-> Image analysis technique that could be adapted for sports movement analysis

## 50위: Polyp Segmentation Using Wavelet-Based Cross-Band Integration for Enhanced Boundary Representation

- arXiv: http://arxiv.org/abs/2603.03682v1 | 2026-03-04 | final 64.0

Accurate polyp segmentation is essential for early colorectal cancer detection, yet achieving reliable boundary localization remains challenging due to low mucosal contrast, uneven illumination, and color similarity between polyps and surrounding tissue. Conventional methods relying solely on RGB information often struggle to delineate precise boundaries due to weak contrast and ambiguous structures between polyps and surrounding mucosa.

-> 이미지 처리 기술로 스포츠 영상 분석에 응용 가능성 있음

## 51위: IntroductionDMD-augmented Unpaired Neural Schrödinger Bridge for Ultra-Low Field MRI Enhancement

- arXiv: http://arxiv.org/abs/2603.03769v1 | 2026-03-04 | final 64.0

Ultra Low Field (64 mT) brain MRI improves accessibility but suffers from reduced image quality compared to 3 T. As paired 64 mT - 3 T scans are scarce, we propose an unpaired 64 mT $\rightarrow$ 3 T translation framework that enhances realism while preserving anatomy.

-> 이미지 향상 기술이 스포츠 비디오 처리에 적용 가능

## 52위: Are You Comfortable Sharing It?: Leveraging Image Obfuscation Techniques to Enhance Sharing Privacy for Blind and Visually Impaired Users

- arXiv: http://arxiv.org/abs/2603.03606v1 | 2026-03-04 | final 62.0

People with Blind Visual Impairments (BVI) face unique challenges when sharing images, as these may accidentally contain sensitive or inappropriate content. In many instances, they are unaware of the potential risks associated with sharing such content, which can compromise their privacy and interpersonal relationships.

-> 프라이버시 보호를 위한 이미지 처리 기술이 이미지 향상과 간접적으로 관련

## 53위: Turning Trust to Transactions: Tracking Affiliate Marketing and FTC Compliance in YouTube's Influencer Economy

- arXiv: http://arxiv.org/abs/2603.04383v1 | 2026-03-04 | final 61.6

YouTube has evolved into a powerful platform that where creators monetize their influence through affiliate marketing, raising concerns about transparency and ethics, especially when creators fail to disclose their affiliate relationships. Although regulatory agencies like the US Federal Trade Commission (FTC) have issued guidelines to address these issues, non-compliance and consumer harm persist, and the extent of these problems remains unclear.

-> 플랫폼 규정 준수 인사이트가 스포츠 콘텐츠 공유 플랫폼에 관련

## 54위: CLIP-Guided Multi-Task Regression for Multi-View Plant Phenotyping

- arXiv: http://arxiv.org/abs/2603.04091v1 | 2026-03-04 | final 58.4

Modeling plant growth dynamics plays a central role in modern agricultural research. However, learning robust predictors from multi-view plant imagery remains challenging due to strong viewpoint redundancy and viewpoint-dependent appearance changes.

-> 다중 뷰 식물 분석 프레임워크로 스포츠 영상 분석과 약간 유사한 다중 뷰 처리 기술 포함

## 55위: Scalable and Convergent Generalized Power Iteration Precoding for Massive MIMO Systems

- arXiv: http://arxiv.org/abs/2603.03708v1 | 2026-03-04 | final 56.0

In massive multiple-input multiple-output (MIMO) systems, achieving high spectral efficiency (SE) often requires advanced precoding algorithms whose complexity scales rapidly with the number of antennas, limiting practical deployment. In this paper, we develop a scalable and computationally efficient generalized power iteration precoding (GPIP) framework for massive MIMO systems under both perfect and imperfect channel state information at the transmitter (CSIT).

-> 대규모 MIMO 시스템을 위한 확장 가능한 알고리즘으로 엣지 디바이스에서의 실시간 처리 가능성이 있으나 스포츠 촬영과 직접적 관련은 없음

## 56위: Low-Altitude Agentic Networks for Optical Wireless Communication and Sensing: An Oceanic Scenario

- arXiv: http://arxiv.org/abs/2603.04042v1 | 2026-03-04 | final 53.6

The cross-domain oceanic connectivity ranging from underwater to the sky has become increasingly indispensable for a plethora of data-consuming maritime applications, such as maritime meteorological monitoring and offshore exploration. However, broadband implementations can be severely hindered by the isolation from terrestrial networks, limited satellite resources, and the fundamental inability of radio waves to bridge the water-air interface at high rates.

-> 통신 및 센싱 기술이 일부 관련성 있지만 해양 시나리오에 특화

## 57위: Harmonic Dataset Distillation for Time Series Forecasting

- arXiv: http://arxiv.org/abs/2603.03760v1 | 2026-03-04 | final 52.0

Time Series forecasting (TSF) in the modern era faces significant computational and storage cost challenges due to the massive scale of real-world data. Dataset Distillation (DD), a paradigm that synthesizes a small, compact dataset to achieve training performance comparable to that of the original dataset, has emerged as a promising solution.

-> 시계열 예측을 위한 데이터셋 증류 방법으로 스포츠 영상 처리에 간접적으로 적용 가능할 수 있음

## 58위: UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization

- arXiv: http://arxiv.org/abs/2603.03967v1 | 2026-03-04 | final 50.4

Despite significant progress has been made in image deraining, we note that most existing methods are often developed for only specific types of rain degradation and fail to generalize across diverse real-world rainy scenes. How to effectively model different rain degradations within a universal framework is important for real-world image deraining.

-> 약간 관련: 이미지 처리 기술이나 보정 방법과 관련 있으나 스포츠 특화되지 않음

---

## 다시 보기

### Search Multilayer Perceptron-Based Fusion for Efficient and Accurate Siamese Tracking (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.01706v1
- 점수: final 96.0

Siamese visual trackers have recently advanced through increasingly sophisticated fusion mechanisms built on convolutional or Transformer architectures. However, both struggle to deliver pixel-level interactions efficiently on resource-constrained hardware, leading to a persistent accuracy-efficiency imbalance. Motivated by this limitation, we redesign the Siamese neck with a simple yet effective Multilayer Perception (MLP)-based fusion module that enables pixel-level interaction with minimal structural overhead. Nevertheless, naively stacking MLP blocks introduces a new challenge: computational cost can scale quadratically with channel width. To overcome this, we construct a hierarchical search space of carefully designed MLP modules and introduce a customized relaxation strategy that enables differentiable neural architecture search (DNAS) to decouple channel-width optimization from other architectural choices. This targeted decoupling automatically balances channel width and depth, yielding a low-complexity architecture. The resulting tracker achieves state-of-the-art accuracy-efficiency trade-offs. It ranks among the top performers on four general-purpose and three aerial tracking benchmarks, while maintaining real-time performance on both resource-constrained Graphics Processing Units (GPUs) and Neural Processing Units (NPUs).

-> 자원 제한 하드웨어에서 실시간 추적 성능으로 선수 추적에 최적

### OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.02134v1
- 점수: final 96.0

Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent stable state, and then cohesively fuses the information from the former into the latter to achieve both fidelity and stability. Moreover, we jointly model visual appearance and language fields and incorporate an implicit Gaussian fusion module to enhance reconstruction quality. Experiments on mainstream datasets demonstrate that our method consistently outperforms prior work in novel view synthesis and semantic understanding, showcasing robust performance across input sequences of varying lengths with real-time inference speed.

-> 실시간 3D 재구성 기술로 스포츠 장면 포착 및 분석에 적합

### Stereo-Inertial Poser: Towards Metric-Accurate Shape-Aware Motion Capture Using Sparse IMUs and a Single Stereo Camera (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.02130v1
- 점수: final 93.6

Recent advancements in visual-inertial motion capture systems have demonstrated the potential of combining monocular cameras with sparse inertial measurement units (IMUs) as cost-effective solutions, which effectively mitigate occlusion and drift issues inherent in single-modality systems. However, they are still limited by metric inaccuracies in global translations stemming from monocular depth ambiguity, and shape-agnostic local motion estimations that ignore anthropometric variations. We present Stereo-Inertial Poser, a real-time motion capture system that leverages a single stereo camera and six IMUs to estimate metric-accurate and shape-aware 3D human motion. By replacing the monocular RGB with stereo vision, our system resolves depth ambiguity through calibrated baseline geometry, enabling direct 3D keypoint extraction and body shape parameter estimation. IMU data and visual cues are fused for predicting drift-compensated joint positions and root movements, while a novel shape-aware fusion module dynamically harmonizes anthropometry variations with global translations. Our end-to-end pipeline achieves over 200 FPS without optimization-based post-processing, enabling real-time deployment. Quantitative evaluations across various datasets demonstrate state-of-the-art performance. Qualitative results show our method produces drift-free global translation under a long recording time and reduces foot-skating effects.

-> 200fps 초고속 동작 캡처로 스포츠 동작 정밀 분석 가능

### PPEDCRF: Privacy-Preserving Enhanced Dynamic CRF for Location-Privacy Protection for Sequence Videos with Minimal Detection Degradation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.01593v1
- 점수: final 92.8

Dashcam videos collected by autonomous or assisted-driving systems are increasingly shared for safety auditing and model improvement. Even when explicit GPS metadata are removed, an attacker can still infer the recording location by matching background visual cues (e.g., buildings and road layouts) against large-scale street-view imagery. This paper studies location-privacy leakage under a background-based retrieval attacker, and proposes PPEDCRF, a privacy-preserving enhanced dynamic conditional random field framework that injects calibrated perturbations only into inferred location-sensitive background regions while preserving foreground detection utility. PPEDCRF consists of three components: (i) a dynamic CRF that enforces temporal consistency to discover and track location sensitive regions across frames, (ii) a normalized control penalty (NCP) that allocates perturbation strength according to a hierarchical sensitivity model, and (iii) a utility-preserving noise injection module that minimizes interference to object detection and segmentation. Experiments on public driving datasets demonstrate that PPEDCRF significantly reduces location-retrieval attack success (e.g., Top-k retrieval accuracy) while maintaining competitive detection performance (e.g., mAP and segmentation metrics) compared with common baselines such as global noise, white-noise masking, and feature-based anonymization. The source code is in https://github.com/mabo1215/PPEDCRF.git

-> 영상 처리 및 프라이버시 보호 기술로 사용자 데이터 보강

### Downstream Task Inspired Underwater Image Enhancement: A Perception-Aware Study from Dataset Construction to Network Design (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.01767v1
- 점수: final 90.4

In real underwater environments, downstream image recognition tasks such as semantic segmentation and object detection often face challenges posed by problems like blurring and color inconsistencies. Underwater image enhancement (UIE) has emerged as a promising preprocessing approach, aiming to improve the recognizability of targets in underwater images. However, most existing UIE methods mainly focus on enhancing images for human visual perception, frequently failing to reconstruct high-frequency details that are critical for task-specific recognition. To address this issue, we propose a Downstream Task-Inspired Underwater Image Enhancement (DTI-UIE) framework, which leverages human visual perception model to enhance images effectively for underwater vision tasks. Specifically, we design an efficient two-branch network with task-aware attention module for feature mixing. The network benefits from a multi-stage training framework and a task-driven perceptual loss. Additionally, inspired by human perception, we automatically construct a Task-Inspired UIE Dataset (TI-UIED) using various task-specific networks. Experimental results demonstrate that DTI-UIE significantly improves task performance by generating preprocessed images that are beneficial for downstream tasks such as semantic segmentation, object detection, and instance segmentation. The codes are publicly available at https://github.com/oucailab/DTIUIE.

-> Task-aware 이미지 향상 기술이 스포츠 영상 품질 개선에 직접 적용 가능하여 분석 정확도 향상

### Kiwi-Edit: Versatile Video Editing via Instruction and Reference Guidance (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.02175v1
- 점수: final 90.4

Instruction-based video editing has witnessed rapid progress, yet current methods often struggle with precise visual control, as natural language is inherently limited in describing complex visual nuances. Although reference-guided editing offers a robust solution, its potential is currently bottlenecked by the scarcity of high-quality paired training data. To bridge this gap, we introduce a scalable data generation pipeline that transforms existing video editing pairs into high-fidelity training quadruplets, leveraging image generative models to create synthesized reference scaffolds. Using this pipeline, we construct RefVIE, a large-scale dataset tailored for instruction-reference-following tasks, and establish RefVIE-Bench for comprehensive evaluation. Furthermore, we propose a unified editing architecture, Kiwi-Edit, that synergizes learnable queries and latent visual features for reference semantic guidance. Our model achieves significant gains in instruction following and reference fidelity via a progressive multi-stage training curriculum. Extensive experiments demonstrate that our data and architecture establish a new state-of-the-art in controllable video editing. All datasets, models, and code is released at https://github.com/showlab/Kiwi-Edit.

-> 지침 기반 비디오 편집으로 스포츠 하이라이트 제작 효율화

### WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.01475v1
- 점수: final 88.0

Recent years have seen a significant increase in demand for robotic solutions in unstructured natural environments, alongside growing interest in bridging 2D and 3D scene understanding. However, existing robotics datasets are predominantly captured in structured urban environments, making them inadequate for addressing the challenges posed by complex, unstructured natural settings. To address this gap, we propose WildCross, a cross-modal benchmark for place recognition and metric depth estimation in large-scale natural environments. WildCross comprises over 476K sequential RGB frames with semi-dense depth and surface normal annotations, each aligned with accurate 6DoF poses and synchronized dense lidar submaps. We conduct comprehensive experiments on visual, lidar, and cross-modal place recognition, as well as metric depth estimation, demonstrating the value of WildCross as a challenging benchmark for multi-modal robotic perception tasks. We provide access to the code repository and dataset at https://csiro-robotics.github.io/WildCross.

-> 교차 모달 환경 인식 기술이 스포츠 장면 분석에 적용 가능

### SeaVIS: Sound-Enhanced Association for Online Audio-Visual Instance Segmentation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.01431v1
- 점수: final 88.0

Recently, an audio-visual instance segmentation (AVIS) task has been introduced, aiming to identify, segment and track individual sounding instances in videos. However, prevailing methods primarily adopt the offline paradigm, that cannot associate detected instances across consecutive clips, making them unsuitable for real-world scenarios that involve continuous video streams. To address this limitation, we introduce SeaVIS, the first online framework designed for audio-visual instance segmentation. SeaVIS leverages the Causal Cross Attention Fusion (CCAF) module to enable efficient online processing, which integrates visual features from the current frame with the entire audio history under strict causal constraints. A major challenge for conventional VIS methods is that appearance-based instance association fails to distinguish between an object's sounding and silent states, resulting in the incorrect segmentation of silent objects. To tackle this, we employ an Audio-Guided Contrastive Learning (AGCL) strategy to generate instance prototypes that encode not only visual appearance but also sounding activity. In this way, instances preserved during per-frame prediction that do not emit sound can be effectively suppressed during instance association process, thereby significantly enhancing the audio-following capability of SeaVIS. Extensive experiments conducted on the AVISeg dataset demonstrate that SeaVIS surpasses existing state-of-the-art models across multiple evaluation metrics while maintaining a competitive inference speed suitable for real-time processing.

-> 실시간 오디오-비전 인스턴스 분할 기술로 스포츠 영상에서 선수 추적 및 동작 분석 정확도 향상

### Orchestrating Multimodal DNN Workloads in Wireless Neural Processing (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.02109v1
- 점수: final 88.0

In edge inference, wireless resource allocation and accelerator-level deep neural network (DNN) scheduling have yet to be co-optimized in an end-to-end manner. The lack of coordination between wireless transmission and accelerator-level DNN execution prevents efficient overlap, leading to higher end-to-end inference latency. To address this issue, this paper investigates multimodal DNN workload orchestration in wireless neural processing (WNP), a paradigm that integrates wireless transmission and multi-core accelerator execution into a unified end-to-end pipeline. First, we develop a unified communication-computation model for multimodal DNN execution and formulate the corresponding optimization problem. Second, we propose O-WiN, a framework that orchestrates DNN workloads in WNP through two tightly coupled stages: simulation-based optimization and runtime execution. Third, we develop two algorithms, RTFS and PACS. RTFS schedules communication and computation sequentially, whereas PACS interleaves them to enable pipeline parallelism by overlapping wireless data transfer with accelerator-level DNN execution. Simulation results demonstrate that PACS significantly outperforms RTFS under high modality heterogeneity by better masking wireless latency through communication-computation overlap, thereby highlighting the effectiveness of communication-computation pipelining in accelerating multimodal DNN execution in WNP.

-> 다중 모달 DNN 워크로드 오케스트레이션 기술은 rk3588 엣지 디바이스에서 실시간 스포츠 콘텐츠 처리를 최적화하는 데 직접적으로 적용 가능하며, 지연 시간을 줄이고 계산 자원을 효율적으로 사용하는 데 기여합니다.

### NextAds: Towards Next-generation Personalized Video Advertising (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.02137v1
- 점수: final 88.0

With the rapid growth of online video consumption, video advertising has become increasingly dominant in the digital advertising landscape. Yet diverse users and viewing contexts makes one-size-fits-all ad creatives insufficient for consistent effectiveness, underlining the importance of personalization. In practice, most personalized video advertising systems follow a retrieval-based paradigm, selecting the optimal one from a small set of professionally pre-produced creatives for each user. Such static and finite inventories limits both the granularity and the timeliness of personalization, and prevents the creatives from being continuously refined based on online user feedback. Recent advances in generative AI make it possible to move beyond retrieval toward optimizing video creatives in a continuous space at serving time.   In this light, we propose NextAds, a generation-based paradigm for next-generation personalized video advertising, and conceptualize NextAds with four core components. To enable comparable research progress, we formulate two representative tasks: personalized creative generation and personalized creative integration, and introduce corresponding lightweight benchmarks. To assess feasibility, we instantiate end-to-end pipelines for both tasks and conduct initial exploratory experiments, demonstrating that GenAI can generate and integrate personalized creatives with encouraging performance. Moreover, we discuss the key challenges and opportunities under this paradigm, aiming to provide actionable insights for both researchers and practitioners and to catalyze progress in personalized video advertising.

-> 생성 기반 개인화 동영상 광고 기술이 플랫폼의 광고 기능에 직접 적용 가능하여 수익 모델 다각화

### Self-supervised Domain Adaptation for Visual 3D Pose Estimation of Nano-drone Racing Gates by Enforcing Geometric Consistency (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.02936v1
- 점수: final 88.0

We consider the task of visually estimating the relative pose of a drone racing gate in front of a nano-quadrotor, using a convolutional neural network pre-trained on simulated data to regress the gate's pose. Due to the sim-to-real gap, the pre-trained model underperforms in the real world and must be adapted to the target domain. We propose an unsupervised domain adaptation (UDA) approach using only real image sequences collected by the drone flying an arbitrary trajectory in front of a gate; sequences are annotated in a self-supervised fashion with the drone's odometry as measured by its onboard sensors. On this dataset, a state consistency loss enforces that two images acquired at different times yield pose predictions that are consistent with the drone's odometry. Results indicate that our approach outperforms other SoA UDA approaches, has a low mean absolute error in position (x=26, y=28, z=10 cm) and orientation ($ψ$=13${^{\circ}}$), an improvement of 40% in position and 37% in orientation over a baseline. The approach's effectiveness is appreciable with as few as 10 minutes of real-world flight data and yields models with an inference time of 30.4ms (33 fps) when deployed aboard the Crazyflie 2.1 Brushless nano-drone.

-> 드론 레이싱 게이트 포즈 추정 기술로 스포츠 분석에 부분적으로 적용 가능하나 매우 구체적 사례

### Boosting AI Reliability with an FSM-Driven Streaming Inference Pipeline: An Industrial Case (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.01528v1
- 점수: final 86.4

The widespread adoption of AI in industry is often hampered by its limited robustness when faced with scenarios absent from training data, leading to prediction bias and vulnerabilities. To address this, we propose a novel streaming inference pipeline that enhances data-driven models by explicitly incorporating prior knowledge. This paper presents the work on an industrial AI application that automatically counts excavator workloads from surveillance videos. Our approach integrates an object detection model with a Finite State Machine (FSM), which encodes knowledge of operational scenarios to guide and correct the AI's predictions on streaming data. In experiments on a real-world dataset of over 7,000 images from 12 site videos, encompassing more than 300 excavator workloads, our method demonstrates superior performance and greater robustness compared to the original solution based on manual heuristic rules. We will release the code at https://github.com/thulab/video-streamling-inference-pipeline.

-> 어안렌즈의 넓은 시야각은 스포츠 장면 촬영에 유리하며, 환경 다양성 확보를 통해 일반화 성능 향상 가능

### Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.02139v1
- 점수: final 86.4

The adoption of fisheye cameras in robotic manipulation, driven by their exceptionally wide Field of View (FoV), is rapidly outpacing a systematic understanding of their downstream effects on policy learning. This paper presents the first comprehensive empirical study to bridge this gap, rigorously analyzing the properties of wrist-mounted fisheye cameras for imitation learning. Through extensive experiments in both simulation and the real world, we investigate three critical research questions: spatial localization, scene generalization, and hardware generalization. Our investigation reveals that: (1) The wide FoV significantly enhances spatial localization, but this benefit is critically contingent on the visual complexity of the environment. (2) Fisheye-trained policies, while prone to overfitting in simple scenes, unlock superior scene generalization when trained with sufficient environmental diversity. (3) While naive cross-camera transfer leads to failures, we identify the root cause as scale overfitting and demonstrate that hardware generalization performance can be improved with a simple Random Scale Augmentation (RSA) strategy. Collectively, our findings provide concrete, actionable guidance for the large-scale collection and effective use of fisheye datasets in robotic learning. More results and videos are available on https://robo-fisheye.github.io/

-> 비디오 토큰 감소 기술은 스포츠 영상 분석 효율성을 높이고 하이라이트 추출 속도 개선 가능

### WorldStereo: Bridging Camera-Guided Video Generation and Scene Reconstruction via 3D Geometric Memories (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.02049v1
- 점수: final 86.4

Recent advances in foundational Video Diffusion Models (VDMs) have yielded significant progress. Yet, despite the remarkable visual quality of generated videos, reconstructing consistent 3D scenes from these outputs remains challenging, due to limited camera controllability and inconsistent generated content when viewed from distinct camera trajectories. In this paper, we propose WorldStereo, a novel framework that bridges camera-guided video generation and 3D reconstruction via two dedicated geometric memory modules. Formally, the global-geometric memory enables precise camera control while injecting coarse structural priors through incrementally updated point clouds. Moreover, the spatial-stereo memory constrains the model's attention receptive fields with 3D correspondence to focus on fine-grained details from the memory bank. These components enable WorldStereo to generate multi-view-consistent videos under precise camera control, facilitating high-quality 3D reconstruction. Furthermore, the flexible control branch-based WorldStereo shows impressive efficiency, benefiting from the distribution matching distilled VDM backbone without joint training. Extensive experiments across both camera-guided video generation and 3D reconstruction benchmarks demonstrate the effectiveness of our approach. Notably, we show that WorldStereo acts as a powerful world model, tackling diverse scene generation tasks (whether starting from perspective or panoramic images) with high-fidelity 3D results. Models will be released.

-> 카메라 가이드 비디오 생성과 3D 복원을 위한 프레임워크가 스포츠 콘텐츠 제작에 적용 가능

### Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.01400v1
- 점수: final 86.4

Video Large Language Models (VLLMs) demonstrate strong video understanding but suffer from inefficiency due to redundant visual tokens. Existing pruning primary targets intra-frame spatial redundancy or prunes inside the LLM with shallow-layer overhead, yielding suboptimal spatiotemporal reduction and underutilizing long-context compressibility. All of them often discard subtle yet informative context from merged or pruned tokens. In this paper, we propose a new perspective that elaborates token \textbf{A}nchors within intra-frame and inter-frame to comprehensively aggregate the informative contexts via local-global \textbf{O}ptimal \textbf{T}ransport (\textbf{AOT}). Specifically, we first establish local- and global-aware token anchors within each frame under the attention guidance, which then optimal transport aggregates the informative contexts from pruned tokens, constructing intra-frame token anchors. Then, building on the temporal frame clips, the first frame within each clip will be considered as the keyframe anchors to ensemble similar information from consecutive frames through optimal transport, while keeping distinct tokens to represent temporal dynamics, leading to efficient token reduction in a training-free manner. Extensive evaluations show that our proposed AOT obtains competitive performances across various short- and long-video benchmarks on leading video LLMs, obtaining substantial computational efficiency while preserving temporal and visual fidelity. Project webpage: \href{https://tyroneli.github.io/AOT}{AOT}.

-> Token reduction method for efficient video understanding applicable to sports content analysis

### MSP-ReID: Hairstyle-Robust Cloth-Changing Person Re-Identification (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.01640v1
- 점수: final 85.6

Cloth-Changing Person Re-Identification (CC-ReID) aims to match the same individual across cameras under varying clothing conditions. Existing approaches often remove apparel and focus on the head region to reduce clothing bias. However, treating the head holistically without distinguishing between face and hair leads to over-reliance on volatile hairstyle cues, causing performance degradation under hairstyle changes. To address this issue, we propose the Mitigating Hairstyle Distraction and Structural Preservation (MSP) framework. Specifically, MSP introduces Hairstyle-Oriented Augmentation (HSOA), which generates intra-identity hairstyle diversity to reduce hairstyle dependence and enhance attention to stable facial and body cues. To prevent the loss of structural information, we design Cloth-Preserved Random Erasing (CPRE), which performs ratio-controlled erasing within clothing regions to suppress texture bias while retaining body shape and context. Furthermore, we employ Region-based Parsing Attention (RPA) to incorporate parsing-guided priors that highlight face and limb regions while suppressing hair features. Extensive experiments on multiple CC-ReID benchmarks demonstrate that MSP achieves state-of-the-art performance, providing a robust and practical solution for long-term person re-identification.

-> 선수 추적 기술이 스포츠 장면에서 선수 추적에 적용 가능

### InterCoG: Towards Spatially Precise Image Editing with Interleaved Chain-of-Grounding Reasoning (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.01586v1
- 점수: final 84.0

Emerging unified editing models have demonstrated strong capabilities in general object editing tasks. However, it remains a significant challenge to perform fine-grained editing in complex multi-entity scenes, particularly those where targets are not visually salient and require spatial reasoning. To this end, we propose InterCoG, a novel text-vision Interleaved Chain-of-Grounding reasoning framework for fine-grained image editing in complex real-world scenes. The key insight of InterCoG is to first perform object position reasoning solely within text that includes spatial relation details to explicitly deduce the location and identity of the edited target. It then conducts visual grounding via highlighting the editing targets with generated bounding boxes and masks in pixel space, and finally rewrites the editing description to specify the intended outcomes. To further facilitate this paradigm, we propose two auxiliary training modules: multimodal grounding reconstruction supervision and multimodal grounding reasoning alignment to enforce spatial localization accuracy and reasoning interpretability, respectively. We also construct GroundEdit-45K, a dataset comprising 45K grounding-oriented editing samples with detailed reasoning annotations, and GroundEdit-Bench for grounding-aware editing evaluation. Extensive experiments substantiate the superiority of our approach in highly precise edits under spatially intricate and multi-entity scenes.

-> 복잡한 스포츠 장면에서 정밀한 이미지 편집 기술로 영상 보정 및 하이라이트 생성에 적용 가능

### Biomechanically Accurate Gait Analysis: A 3d Human Reconstruction Framework for Markerless Estimation of Gait Parameters (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.02499v1
- 점수: final 82.4

This paper presents a biomechanically interpretable framework for gait analysis using 3D human reconstruction from video data. Unlike conventional keypoint based approaches, the proposed method extracts biomechanically meaningful markers analogous to motion capture systems and integrates them within OpenSim for joint kinematic estimation. To evaluate performance, both spatiotemporal and kinematic gait parameters were analysed against reference marker-based data. Results indicate strong agreement with marker-based measurements, with considerable improvements when compared with pose-estimation methods alone. The proposed framework offers a scalable, markerless, and interpretable approach for accurate gait assessment, supporting broader clinical and real world deployment of vision based biomechanics

-> 이 논문은 마커리스 3D 인간 재구성 프레임워크를 제공하여 스포츠 자세 분석에 직접 적용 가능하며, 특수 장비 없이 운동 동작을 분석할 수 있다는 점에서 프로젝트 목표와 정확히 일치한다.

### DLIOS: An LLM-Augmented Real-Time Multi-Modal Interactive Enhancement Overlay System for Douyin Live Streaming (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03060v1
- 점수: final 81.6

We present DLIOS, a Large Language Model (LLM)-augmented real-time multi-modal interactive enhancement overlay system for Douyin (TikTok) live streaming. DLIOS employs a three-layer transparent window architecture for independent rendering of danmaku (scrolling text), gift and like particle effects, and VIP entrance animations, built around an event-driven WebView2 capture pipeline and a thread-safe event bus. On top of this foundation we contribute an LLM broadcast automation framework comprising: (1) a per-song four-segment prompt scheduling system (T1 opening/transition, T2 empathy, T3 era story/production notes, T4 closing) that generates emotionally coherent radio-style commentary from lyric metadata; (2) a JSON-serializable RadioPersonaConfig schema supporting hot-swap multi-persona broadcasting; (3) a real-time danmaku quick-reaction engine with keyword routing to static urgent speech or LLM-generated empathetic responses; and (4) the Suwan Li AI singer-songwriter persona case study -- over 100 AI-generated songs produced with Suno. A 36-hour stress test demonstrates: zero danmaku overlap, zero deadlock crashes, gift effect P95 latency <= 180 ms, LLM-to-TTS segment P95 latency <= 2.1 s, and TTS integrated loudness gain of 9.5 LUFS. live streaming; danmaku; large language model; prompt engineering; virtual persona; WebView2; WINMM; TTS; Suno; loudness normalization; real-time scheduling

-> 실시간 라이브 스트리밍 시스템이지만 스포츠 촬영 및 분석과 직접적인 연관성은 낮음

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
