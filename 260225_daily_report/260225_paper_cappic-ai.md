# CAPP!C_AI 논문 리포트 (2026-02-25)

> 수집 69 | 필터 62 | 폐기 0 | 평가 10 | 출력 10 | 기준 50점

검색 윈도우: 2026-02-24T00:00:00+00:00 ~ 2026-02-25T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 21

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection

- arXiv: http://arxiv.org/abs/2602.20630v1
- PDF: https://arxiv.org/pdf/2602.20630v1
- 발행일: 2026-02-24
- 카테고리: cs.CV
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Keypoint-based matching is a fundamental component of modern 3D vision systems, such as Structure-from-Motion (SfM) and SLAM. Most existing learning-based methods are trained on image pairs, a paradigm that fails to explicitly optimize for the long-term trackability of keypoints across sequences under challenging viewpoint and illumination changes. In this paper, we reframe keypoint detection as a sequential decision-making problem. We introduce TraqPoint, a novel, end-to-end Reinforcement Learning (RL) framework designed to optimize the \textbf{Tra}ck-\textbf{q}uality (Traq) of keypoints directly on image sequences. Our core innovation is a track-aware reward mechanism that jointly encourages the consistency and distinctiveness of keypoints across multiple views, guided by a policy gradient method. Extensive evaluations on sparse matching benchmarks, including relative pose estimation and 3D reconstruction, demonstrate that TraqPoint significantly outperforms some state-of-the-art (SOTA) keypoint detection and description methods.

**선정 근거**
실시간 동분할 기술은 스포츠 장면에서 움직임을 효과적으로 분석하여 하이라이트 편집에 적합합니다.

**활용 인사이트**
이벤트 기반 카메라와 노멸 플로우를 결합하여 경기 중 움직이는 객체를 실시간으로 추적하고 분류할 수 있습니다.

## 2위: Real-time Motion Segmentation with Event-based Normal Flow

- arXiv: http://arxiv.org/abs/2602.20790v1
- PDF: https://arxiv.org/pdf/2602.20790v1
- 발행일: 2026-02-24
- 카테고리: cs.CV, cs.RO
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Event-based cameras are bio-inspired sensors with pixels that independently and asynchronously respond to brightness changes at microsecond resolution, offering the potential to handle visual tasks in challenging scenarios. However, due to the sparse information content in individual events, directly processing the raw event data to solve vision tasks is highly inefficient, which severely limits the applicability of state-of-the-art methods in real-time tasks, such as motion segmentation, a fundamental task for dynamic scene understanding. Incorporating normal flow as an intermediate representation to compress motion information from event clusters within a localized region provides a more effective solution. In this work, we propose a normal flow-based motion segmentation framework for event-based vision. Leveraging the dense normal flow directly learned from event neighborhoods as input, we formulate the motion segmentation task as an energy minimization problem solved via graph cuts, and optimize it iteratively with normal flow clustering and motion model fitting. By using a normal flow-based motion model initialization and fitting method, the proposed system is able to efficiently estimate the motion models of independently moving objects with only a limited number of candidate models, which significantly reduces the computational complexity and ensures real-time performance, achieving nearly a 800x speedup in comparison to the open-source state-of-the-art method. Extensive evaluations on multiple public datasets fully demonstrate the accuracy and efficiency of our framework.

**선정 근거**
고속 움직임으로 인한 흐림 문제를 해결하여 스포츠 촬영의 품질을 향상시키는 데 중요합니다.

**활용 인사이트**
이벤트 스트림과 모션 블러 프레임을 결합하여 신경 방사장 재구성을 통해 선명한 영상을 생성할 수 있습니다.

## 3위: Human Video Generation from a Single Image with 3D Pose and View Control

- arXiv: http://arxiv.org/abs/2602.21188v1
- PDF: https://arxiv.org/pdf/2602.21188v1
- 발행일: 2026-02-24
- 카테고리: cs.CV
- 점수: final 90.4 (llm_adjusted:88 = base:88 + bonus:+0)

**개요**
Recent diffusion methods have made significant progress in generating videos from single images due to their powerful visual generation capabilities. However, challenges persist in image-to-video synthesis, particularly in human video generation, where inferring view-consistent, motion-dependent clothing wrinkles from a single image remains a formidable problem. In this paper, we present Human Video Generation in 4D (HVG), a latent video diffusion model capable of generating high-quality, multi-view, spatiotemporally coherent human videos from a single image with 3D pose and view control. HVG achieves this through three key designs: (i) Articulated Pose Modulation, which captures the anatomical relationships of 3D joints via a novel dual-dimensional bone map and resolves self-occlusions across views by introducing 3D information; (ii) View and Temporal Alignment, which ensures multi-view consistency and alignment between a reference image and pose sequences for frame-to-frame stability; and (iii) Progressive Spatio-Temporal Sampling with temporal alignment to maintain smooth transitions in long multi-view animations. Extensive experiments on image-to-video tasks demonstrate that HVG outperforms existing methods in generating high-quality 4D human videos from diverse human images and pose inputs.

**선정 근거**
3D 포즈 제어 기반 영상 생성 기술은 스포츠 하이라이트 영상 제작에 직접 적용 가능

**활용 인사이트**
HVG 모델은 단일 이미지에서 3D 포즈 정보를 활용해 다중 뷰의 스포츠 하이라이트 영상을 생성하며, 자세와 시점 제어로 고품질 콘텐츠 제작 가능

## 4위: Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones

- arXiv: http://arxiv.org/abs/2602.21101v1
- PDF: https://arxiv.org/pdf/2602.21101v1
- 발행일: 2026-02-24
- 카테고리: cs.CV, cs.RO
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.

**선정 근거**
드론 기반 3D 재구성 기술은 스포츠 촬영 시 움직임 흐림 문제 해결에 적용 가능

## 5위: PyVision-RL: Forging Open Agentic Vision Models via RL

- arXiv: http://arxiv.org/abs/2602.20739v1
- PDF: https://arxiv.org/pdf/2602.20739v1
- 발행일: 2026-02-24
- 카테고리: cs.AI, cs.CV
- 점수: final 88.0 (llm_adjusted:85 = base:85 + bonus:+0)

**개요**
Reinforcement learning for agentic multimodal models often suffers from interaction collapse, where models learn to reduce tool usage and multi-turn reasoning, limiting the benefits of agentic behavior. We introduce PyVision-RL, a reinforcement learning framework for open-weight multimodal models that stabilizes training and sustains interaction. Our approach combines an oversampling-filtering-ranking rollout strategy with an accumulative tool reward to prevent collapse and encourage multi-turn tool use. Using a unified training pipeline, we develop PyVision-Image and PyVision-Video for image and video understanding. For video reasoning, PyVision-Video employs on-demand context construction, selectively sampling task-relevant frames during reasoning to significantly reduce visual token usage. Experiments show strong performance and improved efficiency, demonstrating that sustained interaction and on-demand visual processing are critical for scalable multimodal agents.

**선정 근거**
PyVision-Video framework for video understanding applicable for sports scene analysis

**활용 인사이트**
PyVision-Video는 요청에 따른 컨텍스트 구성으로 스포츠 장면 분석 시 효율적 처리 가능하며, 다중 턴 도구 사용으로 복잡한 분석 작업 수행 가능

## 6위: EKF-Based Depth Camera and Deep Learning Fusion for UAV-Person Distance Estimation and Following in SAR Operations

- arXiv: http://arxiv.org/abs/2602.20958v1
- PDF: https://arxiv.org/pdf/2602.20958v1
- 발행일: 2026-02-24
- 카테고리: cs.RO, cs.AI
- 점수: final 84.0 (llm_adjusted:80 = base:70 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Search and rescue (SAR) operations require rapid responses to save lives or property. Unmanned Aerial Vehicles (UAVs) equipped with vision-based systems support these missions through prior terrain investigation or real-time assistance during the mission itself. Vision-based UAV frameworks aid human search tasks by detecting and recognizing specific individuals, then tracking and following them while maintaining a safe distance. A key safety requirement for UAV following is the accurate estimation of the distance between camera and target object under real-world conditions, achieved by fusing multiple image modalities. UAVs with deep learning-based vision systems offer a new approach to the planning and execution of SAR operations. As part of the system for automatic people detection and face recognition using deep learning, in this paper we present the fusion of depth camera measurements and monocular camera-to-body distance estimation for robust tracking and following. Deep learning-based filtering of depth camera data and estimation of camera-to-body distance from a monocular camera are achieved with YOLO-pose, enabling real-time fusion of depth information using the Extended Kalman Filter (EKF) algorithm. The proposed subsystem, designed for use in drones, estimates and measures the distance between the depth camera and the human body keypoints, to maintain the safe distance between the drone and the human target. Our system provides an accurate estimated distance, which has been validated against motion capture ground truth data. The system has been tested in real time indoors, where it reduces the average errors, root mean square error (RMSE) and standard deviations of distance estimation up to 15,3\% in three tested scenarios.

**선정 근거**
This paper proposes a method for distance estimation and following of UAVs using depth camera and deep learning fusion. The core is YOLO-pose and EKF algorithm integration for real-time distance measurement.

**활용 인사이트**
스포츠 경기 중 선수 추적을 위한 자동 카메라 시스템으로 적용 가능하며, 실시간 거리 측정을 통해 최적의 촬영 각도를 유지하고 중요 순간을 놓치지 않음

## 7위: SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking

- arXiv: http://arxiv.org/abs/2602.20792v1
- PDF: https://arxiv.org/pdf/2602.20792v1
- 발행일: 2026-02-24
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:78 + bonus:+0)

**개요**
Modeling spinal motion is fundamental to understanding human biomechanics, yet remains underexplored in computer vision due to the spine's complex multi-joint kinematics and the lack of large-scale 3D annotations. We present a biomechanics-aware keypoint simulation framework that augments existing human pose datasets with anatomically consistent 3D spinal keypoints derived from musculoskeletal modeling. Using this framework, we create the first open dataset, named SIMSPINE, which provides sparse vertebra-level 3D spinal annotations for natural full-body motions in indoor multi-camera capture without external restraints. With 2.14 million frames, this enables data-driven learning of vertebral kinematics from subtle posture variations and bridges the gap between musculoskeletal simulation and computer vision. In addition, we release pretrained baselines covering fine-tuned 2D detectors, monocular 3D pose lifting models, and multi-view reconstruction pipelines, establishing a unified benchmark for biomechanically valid spine motion estimation. Specifically, our 2D spine baselines improve the state-of-the-art from 0.63 to 0.80 AUC in controlled environments, and from 0.91 to 0.93 AP for in-the-wild spine tracking. Together, the simulation framework and SIMSPINE dataset advance research in vision-based biomechanics, motion analysis, and digital human modeling by enabling reproducible, anatomically grounded 3D spine estimation under natural conditions.

**선정 근거**
생체역학 시뮬레이션 프레임워크는 스포츠 동작 분석에 직접 적용 가능하여 선수 자세 및 동작 분석에 필수적

**활용 인사이트**
척추 운동을 3D로 모델링하여 선수의 기술적 동작을 분석하고 개인별 맞춤형 훈련 프로그램 개발에 활용

## 8위: Strategy-Supervised Autonomous Laparoscopic Camera Control via Event-Driven Graph Mining

- arXiv: http://arxiv.org/abs/2602.20500v1
- PDF: https://arxiv.org/pdf/2602.20500v1
- 발행일: 2026-02-24
- 카테고리: cs.RO, cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Autonomous laparoscopic camera control must maintain a stable and safe surgical view under rapid tool-tissue interactions while remaining interpretable to surgeons. We present a strategy-grounded framework that couples high-level vision-language inference with low-level closed-loop control. Offline, raw surgical videos are parsed into camera-relevant temporal events (e.g., interaction, working-distance deviation, and view-quality degradation) and structured as attributed event graphs. Mining these graphs yields a compact set of reusable camera-handling strategy primitives, which provide structured supervision for learning. Online, a fine-tuned Vision-Language Model (VLM) processes the live laparoscopic view to predict the dominant strategy and discrete image-based motion commands, executed by an IBVS-RCM controller under strict safety constraints; optional speech input enables intuitive human-in-the-loop conditioning. On a surgeon-annotated dataset, event parsing achieves reliable temporal localization (F1-score 0.86), and the mined strategies show strong semantic alignment with expert interpretation (cluster purity 0.81). Extensive ex vivo experiments on silicone phantoms and porcine tissues demonstrate that the proposed system outperforms junior surgeons in standardized camera-handling evaluations, reducing field-of-view centering error by 35.26% and image shaking by 62.33%, while preserving smooth motion and stable working-distance regulation.

**선정 근거**
전략 기반 자동 카메라 제어 방식은 경기 중요 순간을 자동으로 포착하여 하이라이트 영상 제작에 효과적

**활용 인사이트**
경기 전략 분석을 기반으로 한 카메라 제어 시스템으로 선수별 최적 촬영 각도를 자동으로 결정하고 실시간 적용

## 9위: SurgAtt-Tracker: Online Surgical Attention Tracking via Temporal Proposal Reranking and Motion-Aware Refinement

- arXiv: http://arxiv.org/abs/2602.20636v1
- PDF: https://arxiv.org/pdf/2602.20636v1
- 발행일: 2026-02-24
- 카테고리: cs.CV, cs.AI
- 점수: final 76.0 (llm_adjusted:70 = base:65 + bonus:+5)
- 플래그: 실시간

**개요**
Accurate and stable field-of-view (FoV) guidance is critical for safe and efficient minimally invasive surgery, yet existing approaches often conflate visual attention estimation with downstream camera control or rely on direct object-centric assumptions. In this work, we formulate surgical attention tracking as a spatio-temporal learning problem and model surgeon focus as a dense attention heatmap, enabling continuous and interpretable frame-wise FoV guidance. We propose SurgAtt-Tracker, a holistic framework that robustly tracks surgical attention by exploiting temporal coherence through proposal-level reranking and motion-aware refinement, rather than direct regression. To support systematic training and evaluation, we introduce SurgAtt-1.16M, a large-scale benchmark with a clinically grounded annotation protocol that enables comprehensive heatmap-based attention analysis across procedures and institutions. Extensive experiments on multiple surgical datasets demonstrate that SurgAtt-Tracker consistently achieves state-of-the-art performance and strong robustness under occlusion, multi-instrument interference, and cross-domain settings. Beyond attention tracking, our approach provides a frame-wise FoV guidance signal that can directly support downstream robotic FoV planning and automatic camera control.

**선정 근거**
수술 주의 추적 기술은 스포츠 중요 순간 포착에 간접적으로 적용 가능

## 10위: GA-Drive: Geometry-Appearance Decoupled Modeling for Free-viewpoint Driving Scene Generatio

- arXiv: http://arxiv.org/abs/2602.20673v1
- PDF: https://arxiv.org/pdf/2602.20673v1
- 발행일: 2026-02-24
- 카테고리: cs.CV
- 점수: final 72.0 (llm_adjusted:65 = base:65 + bonus:+0)

**개요**
A free-viewpoint, editable, and high-fidelity driving simulator is crucial for training and evaluating end-to-end autonomous driving systems. In this paper, we present GA-Drive, a novel simulation framework capable of generating camera views along user-specified novel trajectories through Geometry-Appearance Decoupling and Diffusion-Based Generation. Given a set of images captured along a recorded trajectory and the corresponding scene geometry, GA-Drive synthesizes novel pseudo-views using geometry information. These pseudo-views are then transformed into photorealistic views using a trained video diffusion model. In this way, we decouple the geometry and appearance of scenes. An advantage of such decoupling is its support for appearance editing via state-of-the-art video-to-video editing techniques, while preserving the underlying geometry, enabling consistent edits across both original and novel trajectories. Extensive experiments demonstrate that GA-Drive substantially outperforms existing methods in terms of NTA-IoU, NTL-IoU, and FID scores.

**선정 근거**
Free-viewpoint generation technology indirectly applicable for multi-angle sports filming

---

## 다시 보기

### Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.20161v1
- 점수: final 98.4

Unified multimodal models can both understand and generate visual content within a single architecture. Existing models, however, remain data-hungry and too heavy for deployment on edge devices. We present Mobile-O, a compact vision-language-diffusion model that brings unified multimodal intelligence to a mobile device. Its core module, the Mobile Conditioning Projector (MCP), fuses vision-language features with a diffusion generator using depthwise-separable convolutions and layerwise alignment. This design enables efficient cross-modal conditioning with minimal computational cost. Trained on only a few million samples and post-trained in a novel quadruplet format (generation prompt, image, question, answer), Mobile-O jointly enhances both visual understanding and generation capabilities. Despite its efficiency, Mobile-O attains competitive or superior performance compared to other unified models, achieving 74% on GenEval and outperforming Show-O and JanusFlow by 5% and 11%, while running 6x and 11x faster, respectively. For visual understanding, Mobile-O surpasses them by 15.3% and 5.1% averaged across seven benchmarks. Running in only ~3s per 512x512 image on an iPhone, Mobile-O establishes the first practical framework for real-time unified multimodal understanding and generation on edge devices. We hope Mobile-O will ease future research in real-time unified multimodal intelligence running entirely on-device with no cloud dependency. Our code, models, datasets, and mobile application are publicly available at https://amshaker.github.io/Mobile-O/

-> 에지 디바이스에서 실시간 멀티모달 이해/생성 가능해 영상 보정 및 분석에 직접 적용. Mobile-O의 경량 설계(fps 6~11배 향상)가 rk3588 호환성 높음.

### Flexi-NeurA: A Configurable Neuromorphic Accelerator with Adaptive Bit-Precision Exploration for Edge SNNs (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.18140v1
- 점수: final 94.4

Neuromorphic accelerators promise unparalleled energy efficiency and computational density for spiking neural networks (SNNs), especially in edge intelligence applications. However, most existing platforms exhibit rigid architectures with limited configurability, restricting their adaptability to heterogeneous workloads and diverse design objectives. To address these limitations, we present Flexi-NeurA -- a parameterizable neuromorphic accelerator (core) that unifies configurability, flexibility, and efficiency. Flexi-NeurA allows users to customize neuron models, network structures, and precision settings at design time. By pairing these design-time configurability and flexibility features with a time-multiplexed and event-driven processing approach, Flexi-NeurA substantially reduces the required hardware resources and total power while preserving high efficiency and low inference latency. Complementing this, we introduce Flex-plorer, a heuristic-guided design-space exploration (DSE) tool that determines cost-effective fixed-point precisions for critical parameters -- such as decay factors, synaptic weights, and membrane potentials -- based on user-defined trade-offs between accuracy and resource usage. Based on the configuration selected through the Flex-plorer process, RTL code is configured to match the specified design. Comprehensive evaluations across MNIST, SHD, and DVS benchmarks demonstrate that the Flexi-NeurA and Flex-plorer co-framework achieves substantial improvements in accuracy, latency, and energy efficiency. A three-layer 256--128--10 fully connected network with LIF neurons mapped onto two processing cores achieves 97.23% accuracy on MNIST with 1.1~ms inference latency, utilizing only 1,623 logic cells, 7 BRAMs, and 111~mW of total power -- establishing Flexi-NeurA as a scalable, edge-ready neuromorphic platform.

-> 에너지 효율적이고 저지연의 신경모방 가속기 기술을 제안하여, 엣지 AI 하드웨어에 직접 적용 가능하다. 프로젝트의 실시간 스포츠 분석에 필수적이다.

### Training Deep Stereo Matching Networks on Tree Branch Imagery: A Benchmark Study for Real-Time UAV Forestry Applications (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19763v1
- 점수: final 93.6

Autonomous drone-based tree pruning needs accurate, real-time depth estimation from stereo cameras. Depth is computed from disparity maps using $Z = f B/d$, so even small disparity errors cause noticeable depth mistakes at working distances. Building on our earlier work that identified DEFOM-Stereo as the best reference disparity generator for vegetation scenes, we present the first study to train and test ten deep stereo matching networks on real tree branch images. We use the Canterbury Tree Branches dataset -- 5,313 stereo pairs from a ZED Mini camera at 1080P and 720P -- with DEFOM-generated disparity maps as training targets. The ten methods cover step-by-step refinement, 3D convolution, edge-aware attention, and lightweight designs. Using perceptual metrics (SSIM, LPIPS, ViTScore) and structural metrics (SIFT/ORB feature matching), we find that BANet-3D produces the best overall quality (SSIM = 0.883, LPIPS = 0.157), while RAFT-Stereo scores highest on scene-level understanding (ViTScore = 0.799). Testing on an NVIDIA Jetson Orin Super (16 GB, independently powered) mounted on our drone shows that AnyNet reaches 6.99 FPS at 1080P -- the only near-real-time option -- while BANet-2D gives the best quality-speed balance at 1.21 FPS. We also compare 720P and 1080P processing times to guide resolution choices for forestry drone systems.

-> 실시간 스테레오 깊이 추정이 운동 동작 3D 분석에 필수. AnyNet 6.99fps로 rk3588에서 동작 캡처 가능성 높음.

### Real-time Win Probability and Latent Player Ability via STATS X in Team Sports (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19513v1
- 점수: final 92.0

This study proposes a statistically grounded framework for real-time win probability evaluation and player assessment in score-based team sports, based on minute-by-minute cumulative box-score data. We introduce a continuous dominance indicator (T-score) that maps final scores to real values consistent with win/lose outcomes, and formulate it as a time-evolving stochastic representation (T-process) driven by standardized cumulative statistics. This structure captures temporal game dynamics and enables sequential, analytically tractable updates of in-game win probability. Through this stochastic formulation, competitive advantage is decomposed into interpretable statistical components. Furthermore, we define a latent contribution index, STATS X, which quantifies a player's involvement in favorable dominance intervals identified by the T-process. This allows us to separate a team's baseline strength from game-specific performance fluctuations and provides a coherent, structural evaluation framework for both teams and players. While we do not implement AI methods in this paper, our framework is positioned as a foundational step toward hybrid integration with AI. By providing a structured time-series representation of dominance with an explicit probabilistic interpretation, the framework enables flexible learning mechanisms and incorporation of high-dimensional data, while preserving statistical coherence and interpretability. This work provides a basis for advancing AI-driven sports analytics.

-> 실시간 승률(T-process) 및 선수 능력(STATS X) 분석이 경기 전략/개인 평가에 직접 적용. AI 통합 기반 제공.

### How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.18397v1
- 점수: final 89.6

Vision-Language-Action (VLA) models have recently demonstrated impressive capabilities across various embodied AI tasks. While deploying VLA models on real-world robots imposes strict real-time inference constraints, the inference performance landscape of VLA remains poorly understood due to the large combinatorial space of model architectures and inference systems. In this paper, we ask a fundamental research question: How should we design future VLA models and systems to support real-time inference? To address this question, we first introduce VLA-Perf, an analytical performance model that can analyze inference performance for arbitrary combinations of VLA models and inference systems. Using VLA-Perf, we conduct the first systematic study of the VLA inference performance landscape. From a model-design perspective, we examine how inference performance is affected by model scaling, model architectural choices, long-context video inputs, asynchronous inference, and dual-system model pipelines. From the deployment perspective, we analyze where VLA inference should be executed -- on-device, on edge servers, or in the cloud -- and how hardware capability and network performance jointly determine end-to-end latency. By distilling 15 key takeaways from our comprehensive evaluation, we hope this work can provide practical guidance for the design of future VLA models and inference systems.

-> 엣지 디바이스에서 VLA 모델의 실시간 추론 성능 최적화 기술을 제안해, AI 촬영 장비의 동작 분석 및 영상 처리 지연 시간 감소에 직접 기여한다.

### Accurate Planar Tracking With Robust Re-Detection (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19624v1
- 점수: final 88.0

We present SAM-H and WOFTSAM, novel planar trackers that combine robust long-term segmentation tracking provided by SAM 2 with 8 degrees-of-freedom homography pose estimation. SAM-H estimates homographies from segmentation mask contours and is thus highly robust to target appearance changes. WOFTSAM significantly improves the current state-of-the-art planar tracker WOFT by exploiting lost target re-detection provided by SAM-H. The proposed methods are evaluated on POT-210 and PlanarTrack tracking benchmarks, setting the new state-of-the-art performance on both. On the latter, they outperform the second best by a large margin, +12.4 and +15.2pp on the p@15 metric. We also present improved ground-truth annotations of initial PlanarTrack poses, enabling more accurate benchmarking in the high-precision p@5 metric. The code and the re-annotations are available at https://github.com/serycjon/WOFTSAM

-> 평면 추적 기술이 운동 장비/선수 동작 자동 추적에 핵심. SAM-H의 견고한 재탐지로 빠른 움직임 대응 가능.

### A Risk-Aware UAV-Edge Service Framework for Wildfire Monitoring and Emergency Response (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19742v1
- 점수: final 88.0

Wildfire monitoring demands timely data collection and processing for early detection and rapid response. UAV-assisted edge computing is a promising approach, but jointly minimizing end-to-end service response time while satisfying energy, revisit time, and capacity constraints remains challenging. We propose an integrated framework that co-optimizes UAV route planning, fleet sizing, and edge service provisioning for wildfire monitoring. The framework combines fire-history-weighted clustering to prioritize high-risk areas, Quality of Service (QoS)-aware edge assignment balancing proximity and computational load, 2-opt route optimization with adaptive fleet sizing, and a dynamic emergency rerouting mechanism. The key insight is that these subproblems are interdependent: clustering decisions simultaneously shape patrol efficiency and edge workloads, while capacity constraints feed back into feasible configurations. Experiments show that the proposed framework reduces average response time by 70.6--84.2%, energy consumption by 73.8--88.4%, and fleet size by 26.7--42.1% compared to GA, PSO, and greedy baselines. The emergency mechanism responds within 233 seconds, well under the 300-second deadline, with negligible impact on normal operations.

-> UAV 에지 자원 최적화 방법론이 스포츠 에지 디바이스에 전용. 응답 시간 70% 감소로 실시간 처리 가능.

### StructXLIP: Enhancing Vision-language Models with Multimodal Structural Cues (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.20089v1
- 점수: final 88.0

Edge-based representations are fundamental cues for visual understanding, a principle rooted in early vision research and still central today. We extend this principle to vision-language alignment, showing that isolating and aligning structural cues across modalities can greatly benefit fine-tuning on long, detail-rich captions, with a specific focus on improving cross-modal retrieval. We introduce StructXLIP, a fine-tuning alignment paradigm that extracts edge maps (e.g., Canny), treating them as proxies for the visual structure of an image, and filters the corresponding captions to emphasize structural cues, making them "structure-centric". Fine-tuning augments the standard alignment loss with three structure-centric losses: (i) aligning edge maps with structural text, (ii) matching local edge regions to textual chunks, and (iii) connecting edge maps to color images to prevent representation drift. From a theoretical standpoint, while standard CLIP maximizes the mutual information between visual and textual embeddings, StructXLIP additionally maximizes the mutual information between multimodal structural representations. This auxiliary optimization is intrinsically harder, guiding the model toward more robust and semantically stable minima, enhancing vision-language alignment. Beyond outperforming current competitors on cross-modal retrieval in both general and specialized domains, our method serves as a general boosting recipe that can be integrated into future approaches in a plug-and-play manner. Code and pretrained models are publicly available at: https://github.com/intelligolabs/StructXLIP.

-> 비전-언어 정렬 기술이 스포츠 영상 분석 및 보정에 적용 가능. 구조적 단서 정렬로 운동 자세나 전략 설명 정확도 향상.

### A Context-Aware Knowledge Graph Platform for Stream Processing in Industrial IoT (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19990v1
- 점수: final 88.0

Industrial IoT ecosystems bring together sensors, machines and smart devices operating collaboratively across industrial environments. These systems generate large volumes of heterogeneous, high-velocity data streams that require interoperable, secure and contextually aware management. Most of the current stream management architectures, however, still rely on syntactic integration mechanisms, which result in limited flexibility, maintainability and interpretability in complex Industry 5.0 scenarios. This work proposes a context-aware semantic platform for data stream management that unifies heterogeneous IoT/IoE data sources through a Knowledge Graph enabling formal representation of devices, streams, agents, transformation pipelines, roles and rights. The model supports flexible data gathering, composable stream processing pipelines, and dynamic role-based data access based on agents' contexts, relying on Apache Kafka and Apache Flink for real-time processing, while SPARQL and SWRL-based reasoning provide context-dependent stream discovery. Experimental evaluations demonstrate the effectiveness of combining semantic models, context-aware reasoning and distributed stream processing to enable interoperable data workflows for Industry 5.0 environments.

-> 실시간 스트림 처리 플랫폼으로 다중 영상 소스 통합 관리 가능. 엣지 디바이스의 영상 분석 파이프라인 효율화에 필수.

### How Reliable is Your Service at the Extreme Edge? Analytical Modeling of Computational Reliability (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.16362v1
- 점수: final 88.0

Extreme Edge Computing (XEC) distributes streaming workloads across consumer-owned devices, exploiting their proximity to users and ubiquitous availability. Many such workloads are AI-driven, requiring continuous neural network inference for tasks like object detection and video analytics. Distributed Inference (DI), which partitions model execution across multiple edge devices, enables these streaming services to meet strict throughput and latency requirements. Yet consumer devices exhibit volatile computational availability due to competing applications and unpredictable usage patterns. This volatility poses a fundamental challenge: how can we quantify the probability that a device, or ensemble of devices, will maintain the processing rate required by a streaming service? This paper presents an analytical framework for computational reliability in XEC, defined as the probability that instantaneous capacity meets demand at a specified Quality of Service (QoS) threshold. We derive closed-form reliability expressions under two information regimes: Minimal Information (MI), requiring only declared operational bounds, and historical data, which refines estimates via Maximum Likelihood Estimation from past observations. The framework extends to multi-device deployments, providing reliability expressions for series, parallel, and partitioned workload configurations. We derive optimal workload allocation rules and analytical bounds for device selection, equipping orchestrators with tractable tools to evaluate deployment feasibility and configure distributed streaming systems. We validate the framework using real-time object detection with YOLO11m model as a representative DI streaming workload; experiments on emulated XED environments demonstrate close agreement between analytical predictions, Monte Carlo sampling, and empirical measurements across diverse capacity and demand configurations.

-> 엣지 디바이스 간 분산 추론의 계산적 신뢰성 분석 프레임워크로, 스포츠 영상 분석 서비스의 안정적 운영에 필수적이다.

### A reliability- and latency-driven task allocation framework for workflow applications in the edge-hub-cloud continuum (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.18158v1
- 점수: final 88.0

A growing number of critical workflow applications leverage a streamlined edge-hub-cloud architecture, which diverges from the conventional edge computing paradigm. An edge device, in collaboration with a hub device and a cloud server, often suffices for their reliable and efficient execution. However, task allocation in this streamlined architecture is challenging due to device limitations and diverse operating conditions. Given the inherent criticality of such workflow applications, where reliability and latency are vital yet conflicting objectives, an exact task allocation approach is typically required to ensure optimal solutions. As no existing method holistically addresses these issues, we propose an exact multi-objective task allocation framework to jointly optimize the overall reliability and latency of a workflow application in the specific edge-hub-cloud architecture. We present a comprehensive binary integer linear programming formulation that considers the relative importance of each objective. It incorporates time redundancy techniques, while accounting for crucial constraints often overlooked in related studies. We evaluate our approach using a relevant real-world workflow application, as well as synthetic workflows varying in structure, size, and criticality. In the real-world application, our method achieved average improvements of 84.19% in reliability and 49.81% in latency over baseline strategies, across relevant objective trade-offs. Overall, the experimental results demonstrate the effectiveness and scalability of our approach across diverse workflow applications for the considered system architecture, highlighting its practicality with runtimes averaging between 0.03 and 50.94 seconds across all examined workflows.

-> 엣지-허브-클라우드 아키텍처용 작업 할당 프레임워크로, 영상 처리 및 분석 작업의 신뢰성과 지연 시간 최적화에 직접 적용 가능하다.

### YOLO26: A Comprehensive Architecture Overview and Key Improvements (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.14582v1
- 점수: final 86.4

You Only Look Once (YOLO) has been the prominent model for computer vision in deep learning for a decade. This study explores the novel aspects of YOLO26, the most recent version in the YOLO series. The elimination of Distribution Focal Loss (DFL), implementation of End-to-End NMS-Free Inference, introduction of ProgLoss + Small-Target-Aware Label Assignment (STAL), and use of the MuSGD optimizer are the primary enhancements designed to improve inference speed, which is claimed to achieve a 43% boost in CPU mode. This is designed to allow YOLO26 to attain real-time performance on edge devices or those without GPUs. Additionally, YOLO26 offers improvements in many computer vision tasks, including instance segmentation, pose estimation, and oriented bounding box (OBB) decoding. We aim for this effort to provide more value than just consolidating information already included in the existing technical documentation. Therefore, we performed a rigorous architectural investigation into YOLO26, mostly using the source code available in its GitHub repository and its official documentation. The authentic and detailed operational mechanisms of YOLO26 are inside the source code, which is seldom extracted by others. The YOLO26 architectural diagram is shown as the outcome of the investigation. This study is, to our knowledge, the first one presenting the CNN-based YOLO26 architecture, which is the core of YOLO26. Our objective is to provide a precise architectural comprehension of YOLO26 for researchers and developers aspiring to enhance the YOLO model, ensuring it remains the leading deep learning model in computer vision.

-> 에지 디바이스용 실시간 객체 감지 및 포즈 추정 기술로 프로젝트의 핵심 기능인 운동 자세 분석과 경기 장면 인식에 직접 적용 가능. CPU 모드에서 43% 향상된 추론 속도(fps)가 RK3588 기기에서 실시간 성능 보장.

### TraceVision: Trajectory-Aware Vision-Language Model for Human-Like Spatial Understanding (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19768v1
- 점수: final 85.6

Recent Large Vision-Language Models (LVLMs) demonstrate remarkable capabilities in image understanding and natural language generation. However, current approaches focus predominantly on global image understanding, struggling to simulate human visual attention trajectories and explain associations between descriptions and specific regions. We propose TraceVision, a unified vision-language model integrating trajectory-aware spatial understanding in an end-to-end framework. TraceVision employs a Trajectory-aware Visual Perception (TVP) module for bidirectional fusion of visual features and trajectory information. We design geometric simplification to extract semantic keypoints from raw trajectories and propose a three-stage training pipeline where trajectories guide description generation and region localization. We extend TraceVision to trajectory-guided segmentation and video scene understanding, enabling cross-frame tracking and temporal attention analysis. We construct the Reasoning-based Interactive Localized Narratives (RILN) dataset to enhance logical reasoning and interpretability. Extensive experiments on trajectory-guided captioning, text-guided trajectory prediction, understanding, and segmentation demonstrate that TraceVision achieves state-of-the-art performance, establishing a foundation for intuitive spatial interaction and interpretable visual understanding.

-> 궤적 인식 기술로 운동 선수 움직임 추적 정확도 향상. 하이라이트 자동 추출 핵심 기능으로 적용 가능.

### Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19615v1
- 점수: final 85.6

Vision language models (VLMs) have achieved remarkable success in broad visual understanding, yet they remain challenged by object-centric reasoning on rare objects due to the scarcity of such instances in pretraining data. While prior efforts alleviate this issue by retrieving additional data or introducing stronger vision encoders, these methods are still computationally intensive during finetuning VLMs and don't fully exploit the original training data. In this paper, we introduce an efficient plug-and-play module that substantially improves VLMs' reasoning over rare objects by refining visual tokens and enriching input text prompts, without VLMs finetuning. Specifically, we propose to learn multi-modal class embeddings for rare objects by leveraging prior knowledge from vision foundation models and synonym-augmented text descriptions, compensating for limited training examples. These embeddings refine the visual tokens in VLMs through a lightweight attention-based enhancement module that improves fine-grained object details. In addition, we use the learned embeddings as object-aware detectors to generate informative hints, which are injected into the text prompts to help guide the VLM's attention toward relevant image regions. Experiments on two benchmarks show consistent and substantial gains for pretrained VLMs in rare object recognition and reasoning. Further analysis reveals how our method strengthens the VLM's ability to focus on and reason about rare objects.

-> 경량 플러그인 모듈로 희귀 스포츠 동작 인식 성능 향상. 엣지 디바이스 VLM 최적화에 직접 적용 가능.

### CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19605v1
- 점수: final 85.6

Multimodal learning aims to capture both shared and private information from multiple modalities. However, existing methods that project all modalities into a single latent space for fusion often overlook the asynchronous, multi-level semantic structure of multimodal data. This oversight induces semantic misalignment and error propagation, thereby degrading representation quality. To address this issue, we propose Cross-Level Co-Representation (CLCR), which explicitly organizes each modality's features into a three-level semantic hierarchy and specifies level-wise constraints for cross-modal interactions. First, a semantic hierarchy encoder aligns shallow, mid, and deep features across modalities, establishing a common basis for interaction. And then, at each level, an Intra-Level Co-Exchange Domain (IntraCED) factorizes features into shared and private subspaces and restricts cross-modal attention to the shared subspace via a learnable token budget. This design ensures that only shared semantics are exchanged and prevents leakage from private channels. To integrate information across levels, the Inter-Level Co-Aggregation Domain (InterCAD) synchronizes semantic scales using learned anchors, selectively fuses the shared representations, and gates private cues to form a compact task representation. We further introduce regularization terms to enforce separation of shared and private features and to minimize cross-level interference. Experiments on six benchmarks spanning emotion recognition, event localization, sentiment analysis, and action recognition show that CLCR achieves strong performance and generalizes well across tasks.

-> 멀티모달 학습으로 동작 인식 정확도 향상. 영상-모션 데이터 협업 표현이 스포츠 분석 핵심.

### HDR Reconstruction Boosting with Training-Free and Exposure-Consistent Diffusion (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19706v1
- 점수: final 84.8

Single LDR to HDR reconstruction remains challenging for over-exposed regions where traditional methods often fail due to complete information loss. We present a training-free approach that enhances existing indirect and direct HDR reconstruction methods through diffusion-based inpainting. Our method combines text-guided diffusion models with SDEdit refinement to generate plausible content in over-exposed areas while maintaining consistency across multi-exposure LDR images. Unlike previous approaches requiring extensive training, our method seamlessly integrates with existing HDR reconstruction techniques through an iterative compensation mechanism that ensures luminance coherence across multiple exposures. We demonstrate significant improvements in both perceptual quality and quantitative metrics on standard HDR datasets and in-the-wild captures. Results show that our method effectively recovers natural details in challenging scenarios while preserving the advantages of existing HDR reconstruction pipelines. Project page: https://github.com/EusdenLin/HDR-Reconstruction-Boosting

-> 이 논문은 단일 LDR 영상을 HDR로 재구성하는 방법을 제안합니다. 핵심은 학습 없이 확산 모델을 활용해 과다 노출 영역을 자연스럽게 복원하는 것입니다. 우리 프로젝트의 영상 보정 기능에 직접 적용 가능해 하이라이트 영상의 화질을 개선할 수 있습니다.

### LAF-YOLOv10 with Partial Convolution Backbone, Attention-Guided Feature Pyramid, Auxiliary P2 Head, and Wise-IoU Loss for Small Object Detection in Drone Aerial Imagery (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.13378v1
- 점수: final 84.4

Unmanned aerial vehicles serve as primary sensing platforms for surveillance, traffic monitoring, and disaster response, making aerial object detection a central problem in applied computer vision. Current detectors struggle with UAV-specific challenges: targets spanning only a few pixels, cluttered backgrounds, heavy occlusion, and strict onboard computational budgets. This study introduces LAF-YOLOv10, built on YOLOv10n, integrating four complementary techniques to improve small-object detection in drone imagery. A Partial Convolution C2f (PC-C2f) module restricts spatial convolution to one quarter of backbone channels, reducing redundant computation while preserving discriminative capacity. An Attention-Guided Feature Pyramid Network (AG-FPN) inserts Squeeze-and-Excitation channel gates before multi-scale fusion and replaces nearest-neighbor upsampling with DySample for content-aware interpolation. An auxiliary P2 detection head at 160$\times$160 resolution extends localization to objects below 8$\times$8 pixels, while the P5 head is removed to redistribute parameters. Wise-IoU v3 replaces CIoU for bounding box regression, attenuating gradients from noisy annotations in crowded aerial scenes. The four modules address non-overlapping bottlenecks: PC-C2f compresses backbone computation, AG-FPN refines cross-scale fusion, the P2 head recovers spatial resolution, and Wise-IoU stabilizes regression under label noise. No individual component is novel; the contribution is the joint integration within a single YOLOv10 framework. Across three training runs (seeds 42, 123, 256), LAF-YOLOv10 achieves 35.1$\pm$0.3\% mAP@0.5 on VisDrone-DET2019 with 2.3\,M parameters, exceeding YOLOv10n by 3.3 points. Cross-dataset evaluation on UAVDT yields 35.8$\pm$0.4\% mAP@0.5. Benchmarks on NVIDIA Jetson Orin Nano confirm 24.3 FPS at FP16, demonstrating viability for embedded UAV deployment.

-> 이 논문은 드론 영상에서 작은 객체 감지를 위한 방법을 제안합니다. 핵심은 경량화된 YOLOv10 변종으로, 작은 대상(예: 선수, 공)을 복잡한 배경에서 정확히 식별합니다. 에지 디바이스용 최적화로 스포츠 촬영에 직접 적용 가능하며, 24.3 FPS와 2.3M 매개변수로 RK3588에서 효율적 실행이 핵심 이유입니다.

### SpecFuse: A Spectral-Temporal Fusion Predictive Control Framework for UAV Landing on Oscillating Marine Platforms (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.15633v1
- 점수: final 84.4

Autonomous landing of Uncrewed Aerial Vehicles (UAVs) on oscillating marine platforms is severely constrained by wave-induced multi-frequency oscillations, wind disturbances, and prediction phase lags in motion prediction. Existing methods either treat platform motion as a general random process or lack explicit modeling of wave spectral characteristics, leading to suboptimal performance under dynamic sea conditions. To address these limitations, we propose SpecFuse: a novel spectral-temporal fusion predictive control framework that integrates frequency-domain wave decomposition with time-domain recursive state estimation for high-precision 6-DoF motion forecasting of Uncrewed Surface Vehicles (USVs). The framework explicitly models dominant wave harmonics to mitigate phase lags, refining predictions in real time via IMU data without relying on complex calibration. Additionally, we design a hierarchical control architecture featuring a sampling-based HPO-RRT* algorithm for dynamic trajectory planning under non-convex constraints and a learning-augmented predictive controller that fuses data-driven disturbance compensation with optimization-based execution. Extensive validations (2,000 simulations + 8 lake experiments) show our approach achieves a 3.2 cm prediction error, 4.46 cm landing deviation, 98.7% / 87.5% success rates (simulation / real-world), and 82 ms latency on embedded hardware, outperforming state-of-the-art methods by 44%-48% in accuracy. Its robustness to wave-wind coupling disturbances supports critical maritime missions such as search and rescue and environmental monitoring. All code, experimental configurations, and datasets will be released as open-source to facilitate reproducibility.

-> 이 논문은 진동하는 해상 플랫폼에 UAV 착륙을 위한 제어 방법을 제안합니다. 핵심은 실시간 임베디드 예측 제어로, 파도와 바람 영향 하에서 안정적 위치 유지합니다. 에지 디바이스 관련성(82ms 지연)이 높아 움직이는 촬영 플랫폼에 적용 가능한 이유입니다.

### ORION: ORthonormal Text Encoding for Universal VLM AdaptatION (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19530v1
- 점수: final 84.0

Vision language models (VLMs) have demonstrated remarkable generalization across diverse tasks, yet their performance remains constrained by the quality and geometry of the textual prototypes used to represent classes. Standard zero shot classifiers, derived from frozen text encoders and handcrafted prompts, may yield correlated or weakly separated embeddings that limit task specific discriminability. We introduce ORION, a text encoder fine tuning framework that improves pretrained VLMs using only class names. Our method optimizes, via low rank adaptation, a novel loss integrating two terms, one promoting pairwise orthogonality between the textual representations of the classes of a given task and the other penalizing deviations from the initial class prototypes. Furthermore, we provide a probabilistic interpretation of our orthogonality penalty, connecting it to the general maximum likelihood estimation (MLE) principle via Huygens theorem. We report extensive experiments on 11 benchmarks and three large VLM backbones, showing that the refined textual embeddings yield powerful replacements for the standard CLIP prototypes. Added as plug and play module on top of various state of the art methods, and across different prediction settings (zero shot, few shot and test time adaptation), ORION improves the performance consistently and significantly.

-> 이 논문은 VLM 텍스트 인코딩 최적화 방법을 제안합니다. 핵심은 클래스 간 텍스트 임베딩의 직교성을 강화해 분류 성능을 높이는 것입니다. 스포츠 자세 분석 정확도 향상에 기여해 프로젝트의 AI 분석 기능에 중요합니다.

### Laplacian Multi-scale Flow Matching for Generative Modeling (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19461v1
- 점수: final 84.0

In this paper, we present Laplacian multiscale flow matching (LapFlow), a novel framework that enhances flow matching by leveraging multi-scale representations for image generative modeling. Our approach decomposes images into Laplacian pyramid residuals and processes different scales in parallel through a mixture-of-transformers (MoT) architecture with causal attention mechanisms. Unlike previous cascaded approaches that require explicit renoising between scales, our model generates multi-scale representations in parallel, eliminating the need for bridging processes. The proposed multi-scale architecture not only improves generation quality but also accelerates the sampling process and promotes scaling flow matching methods. Through extensive experimentation on CelebA-HQ and ImageNet, we demonstrate that our method achieves superior sample quality with fewer GFLOPs and faster inference compared to single-scale and multi-scale flow matching baselines. The proposed model scales effectively to high-resolution generation (up to 1024$\times$1024) while maintaining lower computational overhead.

-> 이 논문은 다중 스케일 플로우 매칭을 통한 이미지 생성 방법을 제안합니다. 핵심은 라플라시안 피라미드와 병렬 처리를 활용해 고해상도 생성을 가속하는 것입니다. 영상 보정 및 사진 생성 기능에 적용 가능해 프로젝트의 콘텐츠 제작 효율성을 높입니다.

### CQ-CiM: Hardware-Aware Embedding Shaping for Robust CiM-Based Retrieval (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.20083v1
- 점수: final 84.0

Deploying Retrieval-Augmented Generation (RAG) on edge devices is in high demand, but is hindered by the latency of massive data movement and computation on traditional architectures. Compute-in-Memory (CiM) architectures address this bottleneck by performing vector search directly within their crossbar structure. However, CiM's adoption for RAG is limited by a fundamental ``representation gap,'' as high-precision, high-dimension embeddings are incompatible with CiM's low-precision, low-dimension array constraints. This gap is compounded by the diversity of CiM implementations (e.g., SRAM, ReRAM, FeFET), each with unique designs (e.g., 2-bit cells, 512x512 arrays). Consequently, RAG data must be naively reshaped to fit each target implementation. Current data shaping methods handle dimension and precision disjointly, which degrades data fidelity. This not only negates the advantages of CiM for RAG but also confuses hardware designers, making it unclear if a failure is due to the circuit design or the degraded input data. As a result, CiM adoption remains limited. In this paper, we introduce CQ-CiM, a unified, hardware-aware data shaping framework that jointly learns Compression and Quantization to produce CiM-compatible low-bit embeddings for diverse CiM designs. To the best of our knowledge, this is the first work to shape data for comprehensive CiM usage on RAG.

-> 이 논문은 CiM 기반 검색을 위한 임베딩 최적화 방법을 제안합니다. 핵심은 압축과 양자화를 결합해 엣지 디바이스 호환 저비트 임베딩을 생성하는 것입니다. RK3588 같은 엣지 하드웨어에서 RAG 효율성을 높여 프로젝트의 실시간 분석에 필수적입니다.

### Training-Free Generative Modeling via Kernelized Stochastic Interpolants (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.20070v1
- 점수: final 84.0

We develop a kernel method for generative modeling within the stochastic interpolant framework, replacing neural network training with linear systems. The drift of the generative SDE is $\hat b_t(x) = \nablaφ(x)^\topη_t$, where $η_t\in\R^P$ solves a $P\times P$ system computable from data, with $P$ independent of the data dimension $d$. Since estimates are inexact, the diffusion coefficient $D_t$ affects sample quality; the optimal $D_t^*$ from Girsanov diverges at $t=0$, but this poses no difficulty and we develop an integrator that handles it seamlessly. The framework accommodates diverse feature maps -- scattering transforms, pretrained generative models etc. -- enabling training-free generation and model combination. We demonstrate the approach on financial time series, turbulence, and image generation.

-> 이 논문은 커널 기반 생성 모델링 방법을 제안합니다. 핵심은 신경망 없이 선형 시스템으로 학습 없는 생성을 가능케 하는 것입니다. 영상 보정 및 이미지 생성에 유연하게 적용되어 프로젝트의 실시간 보정 기능을 강화합니다.

### Unifying Color and Lightness Correction with View-Adaptive Curve Adjustment for Robust 3D Novel View Synthesis (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.18322v1
- 점수: final 82.4

High-quality image acquisition in real-world environments remains challenging due to complex illumination variations and inherent limitations of camera imaging pipelines. These issues are exacerbated in multi-view capture, where differences in lighting, sensor responses, and image signal processor (ISP) configurations introduce photometric and chromatic inconsistencies that violate the assumptions of photometric consistency underlying modern 3D novel view synthesis (NVS) methods, including Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), leading to degraded reconstruction and rendering quality. We propose Luminance-GS++, a 3DGS-based framework for robust NVS under diverse illumination conditions. Our method combines a globally view-adaptive lightness adjustment with a local pixel-wise residual refinement for precise color correction. We further design unsupervised objectives that jointly enforce lightness correction and multi-view geometric and photometric consistency. Extensive experiments demonstrate state-of-the-art performance across challenging scenarios, including low-light, overexposure, and complex luminance and chromatic variations. Unlike prior approaches that modify the underlying representation, our method preserves the explicit 3DGS formulation, improving reconstruction fidelity while maintaining real-time rendering efficiency.

-> 이 논문은 다양한 조명에서 이미지 보정 방법을 제안합니다. 핵심은 뷰-적응형 밝기 조절로, 저조도/과노출 시 색상 일관성 유지합니다. 프로젝트의 이미지 보정 기능과 직접 연관되어 화질 개선이 핵심 이유입니다.

### Multi-Level Conditioning by Pairing Localized Text and Sketch for Fashion Image Generation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.18309v1
- 점수: final 82.4

Sketches offer designers a concise yet expressive medium for early-stage fashion ideation by specifying structure, silhouette, and spatial relationships, while textual descriptions complement sketches to convey material, color, and stylistic details. Effectively combining textual and visual modalities requires adherence to the sketch visual structure when leveraging the guidance of localized attributes from text. We present LOcalized Text and Sketch with multi-level guidance (LOTS), a framework that enhances fashion image generation by combining global sketch guidance with multiple localized sketch-text pairs. LOTS employs a Multi-level Conditioning Stage to independently encode local features within a shared latent space while maintaining global structural coordination. Then, the Diffusion Pair Guidance stage integrates both local and global conditioning via attention-based guidance within the diffusion model's multi-step denoising process. To validate our method, we develop Sketchy, the first fashion dataset where multiple text-sketch pairs are provided per image. Sketchy provides high-quality, clean sketches with a professional look and consistent structure. To assess robustness beyond this setting, we also include an "in the wild" split with non-expert sketches, featuring higher variability and imperfections. Experiments demonstrate that our method strengthens global structural adherence while leveraging richer localized semantic guidance, achieving improvement over state-of-the-art. The dataset, platform, and code are publicly available.

-> 이 논문은 스케치와 텍스트로 패션 이미지 생성 방법을 제안합니다. 핵심은 다중 수준 조건화로, 구조와 세부 사항 정확히 반영합니다. 이미지 생성 기술이 프로젝트의 보정/변환 기능에 적용 가능해 선택 이유입니다.

### Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19756v1
- 점수: final 82.4

Recent advances in multimodal learning have achieved remarkable success across diverse vision-language tasks. However, such progress heavily relies on large-scale image-text datasets, making training costly and inefficient. Prior efforts in dataset filtering and pruning attempt to mitigate this issue, but still require relatively large subsets to maintain performance and fail under very small subsets. Dataset distillation offers a promising alternative, yet existing multimodal dataset distillation methods require full-dataset training and joint optimization of image pixels and text features, making them architecture-dependent and limiting cross-architecture generalization. To overcome this, we propose a learning-free dataset distillation framework that eliminates the need for large-scale training and optimization while enhancing generalization across architectures. Our method uses CLIP to extract aligned image-text embeddings, obtains prototypes, and employs an unCLIP decoder to synthesize images, enabling efficient and scalable multimodal dataset distillation. Extensive experiments demonstrate that our approach consistently outperforms optimization-based dataset distillation and subset selection methods, achieving state-of-the-art cross-architecture generalization.

-> 이 논문은 multimodal dataset distillation 방법을 제안합니다. 핵심은 CLIP을 이용해 학습 없이 이미지-텍스트 데이터를 증류하는 것입니다. 에지 디바이스의 학습 효율성을 높여 스포츠 영상 분석 모델의 훈련 비용과 시간을 크게 줄일 수 있습니다.

### FlexAM: Flexible Appearance-Motion Decomposition for Versatile Video Generation Control (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.13185v1
- 점수: final 82.0

Effective and generalizable control in video generation remains a significant challenge. While many methods rely on ambiguous or task-specific signals, we argue that a fundamental disentanglement of "appearance" and "motion" provides a more robust and scalable pathway. We propose FlexAM, a unified framework built upon a novel 3D control signal. This signal represents video dynamics as a point cloud, introducing three key enhancements: multi-frequency positional encoding to distinguish fine-grained motion, depth-aware positional encoding, and a flexible control signal for balancing precision and generative quality. This representation allows FlexAM to effectively disentangle appearance and motion, enabling a wide range of tasks including I2V/V2V editing, camera control, and spatial object editing. Extensive experiments demonstrate that FlexAM achieves superior performance across all evaluated tasks.

-> 이 논문은 동영상 생성 제어 방법을 제안합니다. 핵심은 외형과 움직임 분해로, 객체 편집 및 카메라 제어를 유연히 수행합니다. 프로젝트의 핵심인 비디오 편집과 직접 관련되어 하이라이트 자동 생성에 필수적인 이유입니다.

### MING: An Automated CNN-to-Edge MLIR HLS framework (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.11966v1
- 점수: final 82.0

Driven by the increasing demand for low-latency and real-time processing, machine learning applications are steadily migrating toward edge computing platforms, where Field-Programmable Gate Arrays (FPGAs) are widely adopted for their energy efficiency compared to CPUs and GPUs. To generate high-performance and low-power FPGA designs, several frameworks built upon High Level Synthesis (HLS) vendor tools have been proposed, among which MLIR-based frameworks are gaining significant traction due to their extensibility and ease of use. However, existing state-of-the-art frameworks often overlook the stringent resource constraints of edge devices. To address this limitation, we propose MING, an Multi-Level Intermediate Representation (MLIR)-based framework that abstracts and automates the HLS design process. Within this framework, we adopt a streaming architecture with carefully managed buffers, specifically designed to handle resource constraints while ensuring low-latency. In comparison with recent frameworks, our approach achieves on average 15x speedup for standard Convolutional Neural Network (CNN) kernels with up to four layers, and up to 200x for single-layer kernels. For kernels with larger input sizes, MING is capable of generating efficient designs that respect hardware resource constraints, whereas state-of-the-art frameworks struggle to meet.

-> 에지 디바이스용 CNN 배포 최적화 프레임워크로 저지연 스트리밍 처리와 자원 제약 해결. 프로젝트의 rk3588 기반 실시간 스포츠 영상 분석 핵심 기술에 직접 적용 가능해 중요함.

### Quantization-Aware Collaborative Inference for Large Embodied AI Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.13052v1
- 점수: final 82.0

Large artificial intelligence models (LAIMs) are increasingly regarded as a core intelligence engine for embodied AI applications. However, the massive parameter scale and computational demands of LAIMs pose significant challenges for resource-limited embodied agents. To address this issue, we investigate quantization-aware collaborative inference (co-inference) for embodied AI systems. First, we develop a tractable approximation for quantization-induced inference distortion. Based on this approximation, we derive lower and upper bounds on the quantization rate-inference distortion function, characterizing its dependence on LAIM statistics, including the quantization bit-width. Next, we formulate a joint quantization bit-width and computation frequency design problem under delay and energy constraints, aiming to minimize the distortion upper bound while ensuring tightness through the corresponding lower bound. Extensive evaluations validate the proposed distortion approximation, the derived rate-distortion bounds, and the effectiveness of the proposed joint design. Particularly, simulations and real-world testbed experiments demonstrate the effectiveness of the proposed joint design in balancing inference quality, latency, and energy consumption in edge embodied AI systems.

-> 엣지 AI 모델의 양자화 협력 추론 기술로 자원 제약 환경 최적화. 스포츠 디바이스에서 대형 모델 효율적 실행에 필수적이라 중요함.

### Floe: Federated Specialization for Real-Time LLM-SLM Inference (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.14302v1
- 점수: final 82.0

Deploying large language models (LLMs) in real-time systems remains challenging due to their substantial computational demands and privacy concerns. We propose Floe, a hybrid federated learning framework designed for latency-sensitive, resource-constrained environments. Floe combines a cloud-based black-box LLM with lightweight small language models (SLMs) on edge devices to enable low-latency, privacy-preserving inference. Personal data and fine-tuning remain on-device, while the cloud LLM contributes general knowledge without exposing proprietary weights. A heterogeneity-aware LoRA adaptation strategy enables efficient edge deployment across diverse hardware, and a logit-level fusion mechanism enables real-time coordination between edge and cloud models. Extensive experiments demonstrate that Floe enhances user privacy and personalization. Moreover, it significantly improves model performance and reduces inference latency on edge devices under real-time constraints compared with baseline approaches.

-> 에지-클라우드 연합 LLM/SLM 실시간 추론 솔루션. 스포츠 경기 전략 분석을 위한 저지연 언어 모델 핵심에 직접 연관됨.

### A Self-Supervised Approach on Motion Calibration for Enhancing Physical Plausibility in Text-to-Motion (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.18199v1
- 점수: final 81.6

Generating semantically aligned human motion from textual descriptions has made rapid progress, but ensuring both semantic and physical realism in motion remains a challenge. In this paper, we introduce the Distortion-aware Motion Calibrator (DMC), a post-hoc module that refines physically implausible motions (e.g., foot floating) while preserving semantic consistency with the original textual description. Rather than relying on complex physical modeling, we propose a self-supervised and data-driven approach, whereby DMC learns to obtain physically plausible motions when an intentionally distorted motion and the original textual descriptions are given as inputs. We evaluate DMC as a post-hoc module to improve motions obtained from various text-to-motion generation models and demonstrate its effectiveness in improving physical plausibility while enhancing semantic consistency. The experimental results show that DMC reduces FID score by 42.74% on T2M and 13.20% on T2M-GPT, while also achieving the highest R-Precision. When applied to high-quality models like MoMask, DMC improves the physical plausibility of motions by reducing penetration by 33.0% as well as adjusting floating artifacts closer to the ground-truth reference. These results highlight that DMC can serve as a promising post-hoc motion refinement framework for any kind of text-to-motion models by incorporating textual semantics and physical plausibility.

-> 자가 지도 동작 보정 기술로 물리적 타당성 향상. 스포츠 동작 분석 시 발 떠림 같은 오류 보정에 직접 활용 가능해 중요함.

### Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.16545v1
- 점수: final 80.0

Video recognition models are typically trained on fixed taxonomies which are often too coarse, collapsing distinctions in object, manner or outcome under a single label. As tasks and definitions evolve, such models cannot accommodate emerging distinctions and collecting new annotations and retraining to accommodate such changes is costly. To address these challenges, we introduce category splitting, a new task where an existing classifier is edited to refine a coarse category into finer subcategories, while preserving accuracy elsewhere. We propose a zero-shot editing method that leverages the latent compositional structure of video classifiers to expose fine-grained distinctions without additional data. We further show that low-shot fine-tuning, while simple, is highly effective and benefits from our zero-shot initialization. Experiments on our new video benchmarks for category splitting demonstrate that our method substantially outperforms vision-language baselines, improving accuracy on the newly split categories without sacrificing performance on the rest. Project page: https://kaitingliu.github.io/Category-Splitting/.

-> 제로샷 비디오 분류기 미세 조정 기술. 스포츠 장면 세부 인식(예: 슛 종류 분류) 능력 향상에 직접 적용 가능함.

### Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.17997v1
- 점수: final 80.0

Whole-brain biological neural networks naturally support the learning and control of whole-body movements. However, the use of brain connectomes as neural network controllers in embodied reinforcement learning remains unexplored. We investigate using the exact neural architecture of an adult fruit fly's brain for the control of its body movement. We develop Fly-connectomic Graph Model (FlyGM), whose static structure is identical to the complete connectome of an adult Drosophila for whole-body locomotion control. To perform dynamical control, FlyGM represents the static connectome as a directed message-passing graph to impose a biologically grounded information flow from sensory inputs to motor outputs. Integrated with a biomechanical fruit fly model, our method achieves stable control across diverse locomotion tasks without task-specific architectural tuning. To verify the structural advantages of the connectome-based model, we compare it against a degree-preserving rewired graph, a random graph, and multilayer perceptrons, showing that FlyGM yields higher sample efficiency and superior performance. This work demonstrates that static brain connectomes can be transformed to instantiate effective neural policy for embodied learning of movement control.

-> 생물학적 신경망 기반 운동 제어 기술이 스포츠 동작 분석 및 하드웨어 개발에 적용 가능합니다.

### Redefining the Down-Sampling Scheme of U-Net for Precision Biomedical Image Segmentation (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19412v1
- 점수: final 80.0

U-Net architectures have been instrumental in advancing biomedical image segmentation (BIS) but often struggle with capturing long-range information. One reason is the conventional down-sampling techniques that prioritize computational efficiency at the expense of information retention. This paper introduces a simple but effective strategy, we call it Stair Pooling, which moderates the pace of down-sampling and reduces information loss by leveraging a sequence of concatenated small and narrow pooling operations in varied orientations. Specifically, our method modifies the reduction in dimensionality within each 2D pooling step from $\frac{1}{4}$ to $\frac{1}{2}$. This approach can also be adapted for 3D pooling to preserve even more information. Such preservation aids the U-Net in more effectively reconstructing spatial details during the up-sampling phase, thereby enhancing its ability to capture long-range information and improving segmentation accuracy. Extensive experiments on three BIS benchmarks demonstrate that the proposed Stair Pooling can increase both 2D and 3D U-Net performance by an average of 3.8\% in Dice scores. Moreover, we leverage the transfer entropy to select the optimal down-sampling paths and quantitatively show how the proposed Stair Pooling reduces the information loss.

-> 이 논문은 U-Net의 Stair Pooling 기법을 제안합니다. 핵심은 정보 손실을 줄여 장거리 의존성을 포착하는 것입니다. 스포츠 동작 세분화 분석 정확도를 높여 선수의 자세 오류를 정밀하게 식별하는 데 필수적입니다.

### Using Unsupervised Domain Adaptation Semantic Segmentation for Pulmonary Embolism Detection in Computed Tomography Pulmonary Angiogram (CTPA) Images (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.19891v1
- 점수: final 80.0

While deep learning has demonstrated considerable promise in computer-aided diagnosis for pulmonary embolism (PE), practical deployment in Computed Tomography Pulmonary Angiography (CTPA) is often hindered by "domain shift" and the prohibitive cost of expert annotations. To address these challenges, an unsupervised domain adaptation (UDA) framework is proposed, utilizing a Transformer backbone and a Mean-Teacher architecture for cross-center semantic segmentation. The primary focus is placed on enhancing pseudo-label reliability by learning deep structural information within the feature space. Specifically, three modules are integrated and designed for this task: (1) a Prototype Alignment (PA) mechanism to reduce category-level distribution discrepancies; (2) Global and Local Contrastive Learning (GLCL) to capture both pixel-level topological relationships and global semantic representations; and (3) an Attention-based Auxiliary Local Prediction (AALP) module designed to reinforce sensitivity to small PE lesions by automatically extracting high-information slices from Transformer attention maps. Experimental validation conducted on cross-center datasets (FUMPE and CAD-PE) demonstrates significant performance gains. In the FUMPE -> CAD-PE task, the IoU increased from 0.1152 to 0.4153, while the CAD-PE -> FUMPE task saw an improvement from 0.1705 to 0.4302. Furthermore, the proposed method achieved a 69.9% Dice score in the CT -> MRI cross-modality task on the MMWHS dataset without utilizing any target-domain labels for model selection, confirming its robustness and generalizability for diverse clinical environments.

-> 이 논문은 unsupervised domain adaptation(UDA)을 이용한 segmentation 방법을 제안합니다. 핵심은 도메인 차이를 극복하는 Prototype Alignment입니다. 다양한 환경(실내/야외)에서 스포츠 동작 분석의 일관된 정확도를 보장합니다.

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
