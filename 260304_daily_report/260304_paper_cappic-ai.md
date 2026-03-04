# CAPP!C_AI 논문 리포트 (2026-03-04)

> 수집 80 | 필터 74 | 폐기 0 | 평가 4 | 출력 4 | 기준 50점

검색 윈도우: 2026-03-03T00:00:00+00:00 ~ 2026-03-04T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 28

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: Self-supervised Domain Adaptation for Visual 3D Pose Estimation of Nano-drone Racing Gates by Enforcing Geometric Consistency

- arXiv: http://arxiv.org/abs/2603.02936v1
- PDF: https://arxiv.org/pdf/2603.02936v1
- 발행일: 2026-03-03
- 카테고리: cs.RO
- 점수: final 88.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
We consider the task of visually estimating the relative pose of a drone racing gate in front of a nano-quadrotor, using a convolutional neural network pre-trained on simulated data to regress the gate's pose. Due to the sim-to-real gap, the pre-trained model underperforms in the real world and must be adapted to the target domain. We propose an unsupervised domain adaptation (UDA) approach using only real image sequences collected by the drone flying an arbitrary trajectory in front of a gate; sequences are annotated in a self-supervised fashion with the drone's odometry as measured by its onboard sensors. On this dataset, a state consistency loss enforces that two images acquired at different times yield pose predictions that are consistent with the drone's odometry. Results indicate that our approach outperforms other SoA UDA approaches, has a low mean absolute error in position (x=26, y=28, z=10 cm) and orientation ($ψ$=13${^{\circ}}$), an improvement of 40% in position and 37% in orientation over a baseline. The approach's effectiveness is appreciable with as few as 10 minutes of real-world flight data and yields models with an inference time of 30.4ms (33 fps) when deployed aboard the Crazyflie 2.1 Brushless nano-drone.

**선정 근거**
드론 레이싱 게이트 포즈 추정 기술로 스포츠 분석에 부분적으로 적용 가능하나 매우 구체적 사례

**활용 인사이트**
자율 학습 도메인 적용 기술을 통해 실제 환경에서의 포즈 추정 정확도를 향상시키며, 33fps의 실시간 처리로 스포츠 장면 분석에 적합함

## 2위: Biomechanically Accurate Gait Analysis: A 3d Human Reconstruction Framework for Markerless Estimation of Gait Parameters

- arXiv: http://arxiv.org/abs/2603.02499v1
- PDF: https://arxiv.org/pdf/2603.02499v1
- 발행일: 2026-03-03
- 카테고리: eess.IV, cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:78 + bonus:+0)

**개요**
This paper presents a biomechanically interpretable framework for gait analysis using 3D human reconstruction from video data. Unlike conventional keypoint based approaches, the proposed method extracts biomechanically meaningful markers analogous to motion capture systems and integrates them within OpenSim for joint kinematic estimation. To evaluate performance, both spatiotemporal and kinematic gait parameters were analysed against reference marker-based data. Results indicate strong agreement with marker-based measurements, with considerable improvements when compared with pose-estimation methods alone. The proposed framework offers a scalable, markerless, and interpretable approach for accurate gait assessment, supporting broader clinical and real world deployment of vision based biomechanics

**선정 근거**
이 논문은 마커리스 3D 인간 재구성 프레임워크를 제공하여 스포츠 자세 분석에 직접 적용 가능하며, 특수 장비 없이 운동 동작을 분석할 수 있다는 점에서 프로젝트 목표와 정확히 일치한다.

**활용 인사이트**
OpenSim 통합 프레임워크를 활용해 선수들의 관절 운동학적 데이터를 실시간으로 추출하고, 이를 바탕으로 기술 개선점을 분석하며 하이라이트 장면을 자동 생성할 수 있다.

## 3위: DLIOS: An LLM-Augmented Real-Time Multi-Modal Interactive Enhancement Overlay System for Douyin Live Streaming

- arXiv: http://arxiv.org/abs/2603.03060v1
- PDF: https://arxiv.org/pdf/2603.03060v1
- 발행일: 2026-03-03
- 카테고리: eess.IV, eess.AS
- 점수: final 81.6 (llm_adjusted:77 = base:72 + bonus:+5)
- 플래그: 실시간

**개요**
We present DLIOS, a Large Language Model (LLM)-augmented real-time multi-modal interactive enhancement overlay system for Douyin (TikTok) live streaming. DLIOS employs a three-layer transparent window architecture for independent rendering of danmaku (scrolling text), gift and like particle effects, and VIP entrance animations, built around an event-driven WebView2 capture pipeline and a thread-safe event bus. On top of this foundation we contribute an LLM broadcast automation framework comprising: (1) a per-song four-segment prompt scheduling system (T1 opening/transition, T2 empathy, T3 era story/production notes, T4 closing) that generates emotionally coherent radio-style commentary from lyric metadata; (2) a JSON-serializable RadioPersonaConfig schema supporting hot-swap multi-persona broadcasting; (3) a real-time danmaku quick-reaction engine with keyword routing to static urgent speech or LLM-generated empathetic responses; and (4) the Suwan Li AI singer-songwriter persona case study -- over 100 AI-generated songs produced with Suno. A 36-hour stress test demonstrates: zero danmaku overlap, zero deadlock crashes, gift effect P95 latency <= 180 ms, LLM-to-TTS segment P95 latency <= 2.1 s, and TTS integrated loudness gain of 9.5 LUFS. live streaming; danmaku; large language model; prompt engineering; virtual persona; WebView2; WINMM; TTS; Suno; loudness normalization; real-time scheduling

**선정 근거**
실시간 라이브 스트리밍 시스템이지만 스포츠 촬영 및 분석과 직접적인 연관성은 낮음

## 4위: Compositional Visual Planning via Inference-Time Diffusion Scaling

- arXiv: http://arxiv.org/abs/2603.02646v1
- PDF: https://arxiv.org/pdf/2603.02646v1
- 발행일: 2026-03-03
- 카테고리: cs.RO
- 점수: final 66.4 (llm_adjusted:58 = base:55 + bonus:+3)
- 플래그: 코드 공개

**개요**
Diffusion models excel at short-horizon robot planning, yet scaling them to long-horizon tasks remains challenging due to computational constraints and limited training data. Existing compositional approaches stitch together short segments by separately denoising each component and averaging overlapping regions. However, this suffers from instability as the factorization assumption breaks down in noisy data space, leading to inconsistent global plans. We propose that the key to stable compositional generation lies in enforcing boundary agreement on the estimated clean data (Tweedie estimates) rather than on noisy intermediate states. Our method formulates long-horizon planning as inference over a chain-structured factor graph of overlapping video chunks, where pretrained short-horizon video diffusion models provide local priors. At inference time, we enforce boundary agreement through a novel combination of synchronous and asynchronous message passing that operates on Tweedie estimates, producing globally consistent guidance without requiring additional training. Our training-free framework demonstrates significant improvements over existing baselines, effectively generalizing to unseen start-goal combinations that were not present in the original training data. Project website: https://comp-visual-planning.github.io/

**선정 근거**
로봇 계획을 위한 비디오 분할 기술로 스포츠 분석에 간접적 적용 가능

---

## 다시 보기

### No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23141v1
- 점수: final 100.0

We propose a new unsupervised framework for online video stabilization. Unlike methods based on deep learning that require paired stable and unstable datasets, our approach instantiates the classical stabilization pipeline with three stages and incorporates a multithreaded buffering mechanism. This design addresses three longstanding challenges in end-to-end learning: limited data, poor controllability, and inefficiency on hardware with constrained resources. Existing benchmarks focus mainly on handheld videos with a forward view in visible light, which restricts the applicability of stabilization to domains such as UAV nighttime remote sensing. To fill this gap, we introduce a new multimodal UAV aerial video dataset (UAV-Test). Experiments show that our method consistently outperforms state-of-the-art online stabilizers in both quantitative metrics and visual quality, while achieving performance comparable to offline methods.

-> 실시간 골격 그래프 구성 및 효율적 공간 추론 기술이 스포츠 장면 추적에 적합

### OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.02134v1
- 점수: final 96.0

Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent stable state, and then cohesively fuses the information from the former into the latter to achieve both fidelity and stability. Moreover, we jointly model visual appearance and language fields and incorporate an implicit Gaussian fusion module to enhance reconstruction quality. Experiments on mainstream datasets demonstrate that our method consistently outperforms prior work in novel view synthesis and semantic understanding, showcasing robust performance across input sequences of varying lengths with real-time inference speed.

-> 실시간 3D 재구성 기술로 스포츠 장면 포착 및 분석에 적합

### U-Net-Based Generative Joint Source-Channel Coding for Wireless Image Transmission (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.22691v1
- 점수: final 96.0

Deep learning (DL)-based joint source-channel coding (JSCC) methods have achieved remarkable success in wireless image transmission. However, these methods either focus on conventional distortion metrics that do not necessarily yield high perceptual quality or incur high computational complexity. In this paper, we propose two DL-based JSCC (DeepJSCC) methods that leverage deep generative architectures for wireless image transmission. Specifically, we propose G-UNet-JSCC, a scheme comprising an encoder and a U-Net-based generator serving as the decoder. Its skip connections enable multi-scale feature fusion to improve both pixel-level fidelity and perceptual quality of reconstructed images by integrating low- and high-level features. To further enhance pixel-level fidelity, the encoder and the U-Net-based decoder are jointly optimized using a weighted sum of structural similarity and mean-squared error (MSE) losses. Building upon G-UNet-JSCC, we further develop a DeepJSCC method called cGAN-JSCC, where the decoder is enhanced through adversarial training. In this scheme, we retain the encoder of G-UNet-JSCC and adversarially train the decoder's generator against a patch-based discriminator. cGAN-JSCC employs a two-stage training procedure. The outer stage trains the encoder and the decoder end-to-end using an MSE loss, while the inner stage adversarially trains the decoder's generator and the discriminator by minimizing a joint loss combining adversarial and distortion losses. Simulation results demonstrate that the proposed methods achieve superior pixel-level fidelity and perceptual quality on both high- and low-resolution images. For low-resolution images, cGAN-JSCC achieves better reconstruction performance and greater robustness to channel variations than G-UNet-JSCC.

-> U-Net 아키텍처와 생성적 방법이 비디오/이미지 처리에 적용 가능

### SCOPE: Skeleton Graph-Based Computation-Efficient Framework for Autonomous UAV Exploration (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.22707v1
- 점수: final 96.0

Autonomous exploration in unknown environments is key for mobile robots, helping them perceive, map, and make decisions in complex areas. However, current methods often rely on frequent global optimization, suffering from high computational latency and trajectory oscillation, especially on resource-constrained edge devices. To address these limitations, we propose SCOPE, a novel framework that incrementally constructs a real-time skeletal graph and introduces Implicit Unknown Region Analysis for efficient spatial reasoning. The planning layer adopts a hierarchical on-demand strategy: the Proximal Planner generates smooth, high-frequency local trajectories, while the Region-Sequence Planner is activated only when necessary to optimize global visitation order. Comparative evaluations in simulation demonstrate that SCOPE achieves competitive exploration performance comparable to state-of-the-art global planners, while reducing computational cost by an average of 86.9%. Real-world experiments further validate the system's robustness and low latency in practical scenarios.

-> U-Net 아키텍처와 생성적 방법이 비디오/이미지 처리에 적용 가능

### Search Multilayer Perceptron-Based Fusion for Efficient and Accurate Siamese Tracking (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.01706v1
- 점수: final 96.0

Siamese visual trackers have recently advanced through increasingly sophisticated fusion mechanisms built on convolutional or Transformer architectures. However, both struggle to deliver pixel-level interactions efficiently on resource-constrained hardware, leading to a persistent accuracy-efficiency imbalance. Motivated by this limitation, we redesign the Siamese neck with a simple yet effective Multilayer Perception (MLP)-based fusion module that enables pixel-level interaction with minimal structural overhead. Nevertheless, naively stacking MLP blocks introduces a new challenge: computational cost can scale quadratically with channel width. To overcome this, we construct a hierarchical search space of carefully designed MLP modules and introduce a customized relaxation strategy that enables differentiable neural architecture search (DNAS) to decouple channel-width optimization from other architectural choices. This targeted decoupling automatically balances channel width and depth, yielding a low-complexity architecture. The resulting tracker achieves state-of-the-art accuracy-efficiency trade-offs. It ranks among the top performers on four general-purpose and three aerial tracking benchmarks, while maintaining real-time performance on both resource-constrained Graphics Processing Units (GPUs) and Neural Processing Units (NPUs).

-> 자원 제한 하드웨어에서 실시간 추적 성능으로 선수 추적에 최적

### UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23224v1
- 점수: final 94.4

We present UniScale, a unified, scale-aware multi-view 3D reconstruction framework for robotic applications that flexibly integrates geometric priors through a modular, semantically informed design. In vision-based robotic navigation, the accurate extraction of environmental structure from raw image sequences is critical for downstream tasks. UniScale addresses this challenge with a single feed-forward network that jointly estimates camera intrinsics and extrinsics, scale-invariant depth and point maps, and the metric scale of a scene from multi-view images, while optionally incorporating auxiliary geometric priors when available. By combining global contextual reasoning with camera-aware feature representations, UniScale is able to recover the metric-scale of the scene. In robotic settings where camera intrinsics are known, they can be easily incorporated to improve performance, with additional gains obtained when camera poses are also available. This co-design enables robust, metric-aware 3D reconstruction within a single unified model. Importantly, UniScale does not require training from scratch, and leverages world priors exhibited in pre-existing models without geometric encoding strategies, making it particularly suitable for resource-constrained robotic teams. We evaluate UniScale on multiple benchmarks, demonstrating strong generalization and consistent performance across diverse environments. We will release our implementation upon acceptance.

-> 자원 제약이 있는 엣지 디바이스에서 스포츠 촬영을 위한 3D 재구성 기술

### BRIDGE: Borderless Reconfiguration for Inclusive and Diverse Gameplay Experience via Embodiment Transformation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23288v1
- 점수: final 93.6

Training resources for parasports are limited, reducing opportunities for athletes and coaches to engage with sport-specific movements and tactical coordination. To address this gap, we developed BRIDGE, a system that integrates a reconstruction pipeline, which detects and tracks players from broadcast video to generate 3D play sequences, with an embodiment-aware visualization framework that decomposes head, trunk, and wheelchair base orientations to represent attention, intent, and mobility. We evaluated BRIDGE in two controlled studies with 20 participants (10 national wheelchair basketball team players and 10 amateur players). The results showed that BRIDGE significantly enhanced the perceived naturalness of player postures and made tactical intentions easier to understand. In addition, it supported functional classification by realistically conveying players' capabilities, which in turn improved participants' sense of self-efficacy. This work advances inclusive sports learning and accessible coaching practices, contributing to more equitable access to tactical resources in parasports.

-> 스포츠 비디오 분석과 전략 이해를 위한 3D 재구성 시스템

### MovieTeller: Tool-augmented Movie Synopsis with ID Consistent Progressive Abstraction (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23228v1
- 점수: final 93.6

With the explosive growth of digital entertainment, automated video summarization has become indispensable for applications such as content indexing, personalized recommendation, and efficient media archiving. Automatic synopsis generation for long-form videos, such as movies and TV series, presents a significant challenge for existing Vision-Language Models (VLMs). While proficient at single-image captioning, these general-purpose models often exhibit critical failures in long-duration contexts, primarily a lack of ID-consistent character identification and a fractured narrative coherence. To overcome these limitations, we propose MovieTeller, a novel framework for generating movie synopses via tool-augmented progressive abstraction. Our core contribution is a training-free, tool-augmented, fact-grounded generation process. Instead of requiring costly model fine-tuning, our framework directly leverages off-the-shelf models in a plug-and-play manner. We first invoke a specialized face recognition model as an external "tool" to establish Factual Groundings--precise character identities and their corresponding bounding boxes. These groundings are then injected into the prompt to steer the VLM's reasoning, ensuring the generated scene descriptions are anchored to verifiable facts. Furthermore, our progressive abstraction pipeline decomposes the summarization of a full-length movie into a multi-stage process, effectively mitigating the context length limitations of current VLMs. Experiments demonstrate that our approach yields significant improvements in factual accuracy, character consistency, and overall narrative coherence compared to end-to-end baselines.

-> 비디오 요약 및 캐릭터 일관성 유지 기술이 스포츠 하이라이트 자동 생성에 적용 가능

### Stereo-Inertial Poser: Towards Metric-Accurate Shape-Aware Motion Capture Using Sparse IMUs and a Single Stereo Camera (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.02130v1
- 점수: final 93.6

Recent advancements in visual-inertial motion capture systems have demonstrated the potential of combining monocular cameras with sparse inertial measurement units (IMUs) as cost-effective solutions, which effectively mitigate occlusion and drift issues inherent in single-modality systems. However, they are still limited by metric inaccuracies in global translations stemming from monocular depth ambiguity, and shape-agnostic local motion estimations that ignore anthropometric variations. We present Stereo-Inertial Poser, a real-time motion capture system that leverages a single stereo camera and six IMUs to estimate metric-accurate and shape-aware 3D human motion. By replacing the monocular RGB with stereo vision, our system resolves depth ambiguity through calibrated baseline geometry, enabling direct 3D keypoint extraction and body shape parameter estimation. IMU data and visual cues are fused for predicting drift-compensated joint positions and root movements, while a novel shape-aware fusion module dynamically harmonizes anthropometry variations with global translations. Our end-to-end pipeline achieves over 200 FPS without optimization-based post-processing, enabling real-time deployment. Quantitative evaluations across various datasets demonstrate state-of-the-art performance. Qualitative results show our method produces drift-free global translation under a long recording time and reduces foot-skating effects.

-> 200fps 초고속 동작 캡처로 스포츠 동작 정밀 분석 가능

### PPEDCRF: Privacy-Preserving Enhanced Dynamic CRF for Location-Privacy Protection for Sequence Videos with Minimal Detection Degradation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.01593v1
- 점수: final 92.8

Dashcam videos collected by autonomous or assisted-driving systems are increasingly shared for safety auditing and model improvement. Even when explicit GPS metadata are removed, an attacker can still infer the recording location by matching background visual cues (e.g., buildings and road layouts) against large-scale street-view imagery. This paper studies location-privacy leakage under a background-based retrieval attacker, and proposes PPEDCRF, a privacy-preserving enhanced dynamic conditional random field framework that injects calibrated perturbations only into inferred location-sensitive background regions while preserving foreground detection utility. PPEDCRF consists of three components: (i) a dynamic CRF that enforces temporal consistency to discover and track location sensitive regions across frames, (ii) a normalized control penalty (NCP) that allocates perturbation strength according to a hierarchical sensitivity model, and (iii) a utility-preserving noise injection module that minimizes interference to object detection and segmentation. Experiments on public driving datasets demonstrate that PPEDCRF significantly reduces location-retrieval attack success (e.g., Top-k retrieval accuracy) while maintaining competitive detection performance (e.g., mAP and segmentation metrics) compared with common baselines such as global noise, white-noise masking, and feature-based anonymization. The source code is in https://github.com/mabo1215/PPEDCRF.git

-> 영상 처리 및 프라이버시 보호 기술로 사용자 데이터 보강

### Align then Adapt: Rethinking Parameter-Efficient Transfer Learning in 4D Perception (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23069v1
- 점수: final 92.0

Point cloud video understanding is critical for robotics as it accurately encodes motion and scene interaction. We recognize that 4D datasets are far scarcer than 3D ones, which hampers the scalability of self-supervised 4D models. A promising alternative is to transfer 3D pre-trained models to 4D perception tasks. However, rigorous empirical analysis reveals two critical limitations that impede transfer capability: overfitting and the modality gap. To overcome these challenges, we develop a novel "Align then Adapt" (PointATA) paradigm that decomposes parameter-efficient transfer learning into two sequential stages. Optimal-transport theory is employed to quantify the distributional discrepancy between 3D and 4D datasets, enabling our proposed point align embedder to be trained in Stage 1 to alleviate the underlying modality gap. To mitigate overfitting, an efficient point-video adapter and a spatial-context encoder are integrated into the frozen 3D backbone to enhance temporal modeling capacity in Stage 2. Notably, with the above engineering-oriented designs, PointATA enables a pre-trained 3D model without temporal knowledge to reason about dynamic video content at a smaller parameter cost compared to previous work. Extensive experiments show that PointATA can match or even outperform strong full fine-tuning models, whilst enjoying the advantage of parameter efficiency, e.g. 97.21 \% accuracy on 3D action recognition, $+8.7 \%$ on 4 D action segmentation, and 84.06\% on 4D semantic segmentation.

-> 4D perception technology with parameter-efficient transfer learning applicable to edge devices for sports video analysis

### Velocity and stroke rate reconstruction of canoe sprint team boats based on panned and zoomed video recordings (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.22941v1
- 점수: final 92.0

Pacing strategies, defined by velocity and stroke rate profiles, are essential for peak performance in canoe sprint. While GPS is the gold standard for analysis, its limited availability necessitates automated video-based solutions. This paper presents an extended framework for reconstructing performance metrics from panned and zoomed video recordings across all sprint disciplines (K1-K4, C1-C2) and distances (200m-500m). Our method utilizes YOLOv8 for buoy and athlete detection, leveraging the known buoy grid to estimate homographies. We generalized the estimation of the boat position by means of learning a boat-specific athlete offset using a U-net based boat tip calibration. Further, we implement a robust tracking scheme using optical flow to adapt to multi-athlete boat types. Finally, we introduce methods to extract stroke rate information from either pose estimations or the athlete bounding boxes themselves. Evaluation against GPS data from elite competitions yields a velocity RRMSE of 0.020 +- 0.011 (rho = 0.956) and a stroke rate RRMSE of 0.022 +- 0.024 (rho = 0.932). The methods provide coaches with highly accurate, automated feedback without requiring on-boat sensors or manual annotation.

-> 스포츠 비디오 분석을 위한 컴퓨터 비전 기술로 프로젝트와 직접 관련 있으나 특정 스포츠(카누)에 국한됨

### UCM: Unifying Camera Control and Memory with Time-aware Positional Encoding Warping for World Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.22960v1
- 점수: final 90.4

World models based on video generation demonstrate remarkable potential for simulating interactive environments but face persistent difficulties in two key areas: maintaining long-term content consistency when scenes are revisited and enabling precise camera control from user-provided inputs. Existing methods based on explicit 3D reconstruction often compromise flexibility in unbounded scenarios and fine-grained structures. Alternative methods rely directly on previously generated frames without establishing explicit spatial correspondence, thereby constraining controllability and consistency. To address these limitations, we present UCM, a novel framework that unifies long-term memory and precise camera control via a time-aware positional encoding warping mechanism. To reduce computational overhead, we design an efficient dual-stream diffusion transformer for high-fidelity generation. Moreover, we introduce a scalable data curation strategy utilizing point-cloud-based rendering to simulate scene revisiting, facilitating training on over 500K monocular videos. Extensive experiments on real-world and synthetic benchmarks demonstrate that UCM significantly outperforms state-of-the-art methods in long-term scene consistency, while also achieving precise camera controllability in high-fidelity video generation.

-> 카메라 제어 및 일관성 유지 기술이 스포츠 경기 자동 촬영에 활용 가능

### Kiwi-Edit: Versatile Video Editing via Instruction and Reference Guidance (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.02175v1
- 점수: final 90.4

Instruction-based video editing has witnessed rapid progress, yet current methods often struggle with precise visual control, as natural language is inherently limited in describing complex visual nuances. Although reference-guided editing offers a robust solution, its potential is currently bottlenecked by the scarcity of high-quality paired training data. To bridge this gap, we introduce a scalable data generation pipeline that transforms existing video editing pairs into high-fidelity training quadruplets, leveraging image generative models to create synthesized reference scaffolds. Using this pipeline, we construct RefVIE, a large-scale dataset tailored for instruction-reference-following tasks, and establish RefVIE-Bench for comprehensive evaluation. Furthermore, we propose a unified editing architecture, Kiwi-Edit, that synergizes learnable queries and latent visual features for reference semantic guidance. Our model achieves significant gains in instruction following and reference fidelity via a progressive multi-stage training curriculum. Extensive experiments demonstrate that our data and architecture establish a new state-of-the-art in controllable video editing. All datasets, models, and code is released at https://github.com/showlab/Kiwi-Edit.

-> 지침 기반 비디오 편집으로 스포츠 하이라이트 제작 효율화

### Downstream Task Inspired Underwater Image Enhancement: A Perception-Aware Study from Dataset Construction to Network Design (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.01767v1
- 점수: final 90.4

In real underwater environments, downstream image recognition tasks such as semantic segmentation and object detection often face challenges posed by problems like blurring and color inconsistencies. Underwater image enhancement (UIE) has emerged as a promising preprocessing approach, aiming to improve the recognizability of targets in underwater images. However, most existing UIE methods mainly focus on enhancing images for human visual perception, frequently failing to reconstruct high-frequency details that are critical for task-specific recognition. To address this issue, we propose a Downstream Task-Inspired Underwater Image Enhancement (DTI-UIE) framework, which leverages human visual perception model to enhance images effectively for underwater vision tasks. Specifically, we design an efficient two-branch network with task-aware attention module for feature mixing. The network benefits from a multi-stage training framework and a task-driven perceptual loss. Additionally, inspired by human perception, we automatically construct a Task-Inspired UIE Dataset (TI-UIED) using various task-specific networks. Experimental results demonstrate that DTI-UIE significantly improves task performance by generating preprocessed images that are beneficial for downstream tasks such as semantic segmentation, object detection, and instance segmentation. The codes are publicly available at https://github.com/oucailab/DTIUIE.

-> Task-aware 이미지 향상 기술이 스포츠 영상 품질 개선에 직접 적용 가능하여 분석 정확도 향상

### Learning Continuous Wasserstein Barycenter Space for Generalized All-in-One Image Restoration (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23169v1
- 점수: final 89.6

Despite substantial advances in all-in-one image restoration for addressing diverse degradations within a unified model, existing methods remain vulnerable to out-of-distribution degradations, thereby limiting their generalization in real-world scenarios. To tackle the challenge, this work is motivated by the intuition that multisource degraded feature distributions are induced by different degradation-specific shifts from an underlying degradation-agnostic distribution, and recovering such a shared distribution is thus crucial for achieving generalization across degradations. With this insight, we propose BaryIR, a representation learning framework that aligns multisource degraded features in the Wasserstein barycenter (WB) space, which models a degradation-agnostic distribution by minimizing the average of Wasserstein distances to multisource degraded distributions. We further introduce residual subspaces, whose embeddings are mutually contrasted while remaining orthogonal to the WB embeddings. Consequently, BaryIR explicitly decouples two orthogonal spaces: a WB space that encodes the degradation-agnostic invariant contents shared across degradations, and residual subspaces that adaptively preserve the degradation-specific knowledge. This disentanglement mitigates overfitting to in-distribution degradations and enables adaptive restoration grounded on the degradation-agnostic shared invariance. Extensive experiments demonstrate that BaryIR performs competitively against state-of-the-art all-in-one methods. Notably, BaryIR generalizes well to unseen degradations (\textit{e.g.,} types and levels) and shows remarkable robustness in learning generalized features, even when trained on limited degradation types and evaluated on real-world data with mixed degradations.

-> 이미지 복원 및 향상 기술로 스포츠 영상 보정에 직접 적용 가능

### NextAds: Towards Next-generation Personalized Video Advertising (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.02137v1
- 점수: final 88.0

With the rapid growth of online video consumption, video advertising has become increasingly dominant in the digital advertising landscape. Yet diverse users and viewing contexts makes one-size-fits-all ad creatives insufficient for consistent effectiveness, underlining the importance of personalization. In practice, most personalized video advertising systems follow a retrieval-based paradigm, selecting the optimal one from a small set of professionally pre-produced creatives for each user. Such static and finite inventories limits both the granularity and the timeliness of personalization, and prevents the creatives from being continuously refined based on online user feedback. Recent advances in generative AI make it possible to move beyond retrieval toward optimizing video creatives in a continuous space at serving time.   In this light, we propose NextAds, a generation-based paradigm for next-generation personalized video advertising, and conceptualize NextAds with four core components. To enable comparable research progress, we formulate two representative tasks: personalized creative generation and personalized creative integration, and introduce corresponding lightweight benchmarks. To assess feasibility, we instantiate end-to-end pipelines for both tasks and conduct initial exploratory experiments, demonstrating that GenAI can generate and integrate personalized creatives with encouraging performance. Moreover, we discuss the key challenges and opportunities under this paradigm, aiming to provide actionable insights for both researchers and practitioners and to catalyze progress in personalized video advertising.

-> 생성 기반 개인화 동영상 광고 기술이 플랫폼의 광고 기능에 직접 적용 가능하여 수익 모델 다각화

### Orchestrating Multimodal DNN Workloads in Wireless Neural Processing (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.02109v1
- 점수: final 88.0

In edge inference, wireless resource allocation and accelerator-level deep neural network (DNN) scheduling have yet to be co-optimized in an end-to-end manner. The lack of coordination between wireless transmission and accelerator-level DNN execution prevents efficient overlap, leading to higher end-to-end inference latency. To address this issue, this paper investigates multimodal DNN workload orchestration in wireless neural processing (WNP), a paradigm that integrates wireless transmission and multi-core accelerator execution into a unified end-to-end pipeline. First, we develop a unified communication-computation model for multimodal DNN execution and formulate the corresponding optimization problem. Second, we propose O-WiN, a framework that orchestrates DNN workloads in WNP through two tightly coupled stages: simulation-based optimization and runtime execution. Third, we develop two algorithms, RTFS and PACS. RTFS schedules communication and computation sequentially, whereas PACS interleaves them to enable pipeline parallelism by overlapping wireless data transfer with accelerator-level DNN execution. Simulation results demonstrate that PACS significantly outperforms RTFS under high modality heterogeneity by better masking wireless latency through communication-computation overlap, thereby highlighting the effectiveness of communication-computation pipelining in accelerating multimodal DNN execution in WNP.

-> 다중 모달 DNN 워크로드 오케스트레이션 기술은 rk3588 엣지 디바이스에서 실시간 스포츠 콘텐츠 처리를 최적화하는 데 직접적으로 적용 가능하며, 지연 시간을 줄이고 계산 자원을 효율적으로 사용하는 데 기여합니다.

### SeaVIS: Sound-Enhanced Association for Online Audio-Visual Instance Segmentation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.01431v1
- 점수: final 88.0

Recently, an audio-visual instance segmentation (AVIS) task has been introduced, aiming to identify, segment and track individual sounding instances in videos. However, prevailing methods primarily adopt the offline paradigm, that cannot associate detected instances across consecutive clips, making them unsuitable for real-world scenarios that involve continuous video streams. To address this limitation, we introduce SeaVIS, the first online framework designed for audio-visual instance segmentation. SeaVIS leverages the Causal Cross Attention Fusion (CCAF) module to enable efficient online processing, which integrates visual features from the current frame with the entire audio history under strict causal constraints. A major challenge for conventional VIS methods is that appearance-based instance association fails to distinguish between an object's sounding and silent states, resulting in the incorrect segmentation of silent objects. To tackle this, we employ an Audio-Guided Contrastive Learning (AGCL) strategy to generate instance prototypes that encode not only visual appearance but also sounding activity. In this way, instances preserved during per-frame prediction that do not emit sound can be effectively suppressed during instance association process, thereby significantly enhancing the audio-following capability of SeaVIS. Extensive experiments conducted on the AVISeg dataset demonstrate that SeaVIS surpasses existing state-of-the-art models across multiple evaluation metrics while maintaining a competitive inference speed suitable for real-time processing.

-> 실시간 오디오-비전 인스턴스 분할 기술로 스포츠 영상에서 선수 추적 및 동작 분석 정확도 향상

### WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.01475v1
- 점수: final 88.0

Recent years have seen a significant increase in demand for robotic solutions in unstructured natural environments, alongside growing interest in bridging 2D and 3D scene understanding. However, existing robotics datasets are predominantly captured in structured urban environments, making them inadequate for addressing the challenges posed by complex, unstructured natural settings. To address this gap, we propose WildCross, a cross-modal benchmark for place recognition and metric depth estimation in large-scale natural environments. WildCross comprises over 476K sequential RGB frames with semi-dense depth and surface normal annotations, each aligned with accurate 6DoF poses and synchronized dense lidar submaps. We conduct comprehensive experiments on visual, lidar, and cross-modal place recognition, as well as metric depth estimation, demonstrating the value of WildCross as a challenging benchmark for multi-modal robotic perception tasks. We provide access to the code repository and dataset at https://csiro-robotics.github.io/WildCross.

-> 교차 모달 환경 인식 기술이 스포츠 장면 분석에 적용 가능

### Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23188v1
- 점수: final 88.0

We propose an efficient retraining strategy for a parameterized Reduced Order Model (ROM) that attains accuracy comparable to full retraining while requiring only a fraction of the computational time and relying solely on sparse observations of the full system. The architecture employs an encode-process-decode structure: a Variational Autoencoder (VAE) to perform dimensionality reduction, and a transformer network to evolve the latent states and model the dynamics. The ROM is parameterized by an external control variable, the Reynolds number in the Navier-Stokes setting, with the transformer exploiting attention mechanisms to capture both temporal dependencies and parameter effects. The probabilistic VAE enables stochastic sampling of trajectory ensembles, providing predictive means and uncertainty quantification through the first two moments. After initial training on a limited set of dynamical regimes, the model is adapted to out-of-sample parameter regions using only sparse data. Its probabilistic formulation naturally supports ensemble generation, which we employ within an ensemble Kalman filtering framework to assimilate data and reconstruct full-state trajectories from minimal observations. We further show that, for the dynamical system considered, the dominant source of error in out-of-sample forecasts stems from distortions of the latent manifold rather than changes in the latent dynamics. Consequently, retraining can be limited to the autoencoder, allowing for a lightweight, computationally efficient, real-time adaptation procedure with very sparse fine-tuning data.

-> Real-time adaptation techniques applicable to edge device AI processing

### Doubly Adaptive Channel and Spatial Attention for Semantic Image Communication by IoT Devices (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.22794v1
- 점수: final 88.0

Internet of Things (IoT) networks face significant challenges such as limited communication bandwidth, constrained computational and energy resources, and highly dynamic wireless channel conditions. Utilization of deep neural networks (DNNs) combined with semantic communication has emerged as a promising paradigm to address these limitations. Deep joint source-channel coding (DJSCC) has recently been proposed to enable semantic communication of images. Building upon the original DJSCC formulation, low-complexity attention-style architectures has been added to the DNNs for further performance enhancement. As a main hurdle, training these DNNs separately for various signal-to-noise ratios (SNRs) will amount to excessive storage or communication overhead, which can not be maintained by small IoT devices. SNR Adaptive DJSCC (ADJSCC), has been proposed to train the DNNs once but feed the current SNR as part of the data to the channel-wise attention mechanism. We improve upon ADJSCC by a simultaneous utilization of doubly adaptive channel-wise and spatial attention modules at both transmitter and receiver. These modules dynamically adjust to varying channel conditions and spatial feature importance, enabling robust and efficient feature extraction and semantic information recovery. Simulation results corroborate that our proposed doubly adaptive DJSCC (DA-DJSCC) significantly improves upon ADJSCC in several performance criteria, while incurring a mild increase in complexity. These facts render DA-DJSCC a desirable choice for semantic communication in performance demanding but low-complexity IoT networks.

-> IoT device image communication with adaptive attention, applicable to edge device video processing

### LoR-LUT: Learning Compact 3D Lookup Tables via Low-Rank Residuals (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.22607v1
- 점수: final 88.0

We present LoR-LUT, a unified low-rank formulation for compact and interpretable 3D lookup table (LUT) generation. Unlike conventional 3D-LUT-based techniques that rely on fusion of basis LUTs, which are usually dense tensors, our unified approach extends the current framework by jointly using residual corrections, which are in fact low-rank tensors, together with a set of basis LUTs. The approach described here improves the existing perceptual quality of an image, which is primarily due to the technique's novel use of residual corrections. At the same time, we achieve the same level of trilinear interpolation complexity, using a significantly smaller number of network, residual corrections, and LUT parameters. The experimental results obtained from LoR-LUT, which is trained on the MIT-Adobe FiveK dataset, reproduce expert-level retouching characteristics with high perceptual fidelity and a sub-megabyte model size. Furthermore, we introduce an interactive visualization tool, termed LoR-LUT Viewer, which transforms an input image into the LUT-adjusted output image, via a number of slidebars that control different parameters. The tool provides an effective way to enhance interpretability and user confidence in the visual results. Overall, our proposed formulation offers a compact, interpretable, and efficient direction for future LUT-based image enhancement and style transfer.

-> 컴팩트 3D 룩업 테이블 기술로 스포츠 영상 및 이미지 보정에 직접적으로 적용 가능

### Towards Long-Form Spatio-Temporal Video Grounding (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23294v1
- 점수: final 88.0

In real scenarios, videos can span several minutes or even hours. However, existing research on spatio-temporal video grounding (STVG), given a textual query, mainly focuses on localizing targets in short videos of tens of seconds, typically less than one minute, which limits real-world applications. In this paper, we explore Long-Form STVG (LF-STVG), which aims to locate targets in long-term videos. Compared with short videos, long-term videos contain much longer temporal spans and more irrelevant information, making it difficult for existing STVG methods that process all frames at once. To address this challenge, we propose an AutoRegressive Transformer architecture for LF-STVG, termed ART-STVG. Unlike conventional STVG methods that require the entire video sequence to make predictions at once, ART-STVG treats the video as streaming input and processes frames sequentially, enabling efficient handling of long videos. To model spatio-temporal context, we design spatial and temporal memory banks and apply them to the decoders. Since memories from different moments are not always relevant to the current frame, we introduce simple yet effective memory selection strategies to provide more relevant information to the decoders, significantly improving performance. Furthermore, instead of parallel spatial and temporal localization, we propose a cascaded spatio-temporal design that connects the spatial decoder to the temporal decoder, allowing fine-grained spatial cues to assist complex temporal localization in long videos. Experiments on newly extended LF-STVG datasets show that ART-STVG significantly outperforms state-of-the-art methods, while achieving competitive performance on conventional short-form STVG.

-> Long-form video processing technology applicable to sports game analysis

### Boosting AI Reliability with an FSM-Driven Streaming Inference Pipeline: An Industrial Case (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.01528v1
- 점수: final 86.4

The widespread adoption of AI in industry is often hampered by its limited robustness when faced with scenarios absent from training data, leading to prediction bias and vulnerabilities. To address this, we propose a novel streaming inference pipeline that enhances data-driven models by explicitly incorporating prior knowledge. This paper presents the work on an industrial AI application that automatically counts excavator workloads from surveillance videos. Our approach integrates an object detection model with a Finite State Machine (FSM), which encodes knowledge of operational scenarios to guide and correct the AI's predictions on streaming data. In experiments on a real-world dataset of over 7,000 images from 12 site videos, encompassing more than 300 excavator workloads, our method demonstrates superior performance and greater robustness compared to the original solution based on manual heuristic rules. We will release the code at https://github.com/thulab/video-streamling-inference-pipeline.

-> 어안렌즈의 넓은 시야각은 스포츠 장면 촬영에 유리하며, 환경 다양성 확보를 통해 일반화 성능 향상 가능

### GSTurb: Gaussian Splatting for Atmospheric Turbulence Mitigation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.22800v1
- 점수: final 86.4

Atmospheric turbulence causes significant image degradation due to pixel displacement (tilt) and blur, particularly in long-range imaging applications. In this paper, we propose a novel framework for atmospheric turbulence mitigation, GSTurb, which integrates optical flow-guided tilt correction and Gaussian splatting for modeling non-isoplanatic blur. The framework employs Gaussian parameters to represent tilt and blur, and optimizes them across multiple frames to enhance restoration. Experimental results on the ATSyn-static dataset demonstrate the effectiveness of our method, achieving a peak PSNR of 27.67 dB and SSIM of 0.8735. Compared to the state-of-the-art method, GSTurb improves PSNR by 1.3 dB (a 4.5% increase) and SSIM by 0.048 (a 5.8% increase). Additionally, on real datasets, including the TSRWGAN Real-World and CLEAR datasets, GSTurb outperforms existing methods, showing significant improvements in both qualitative and quantitative performance. These results highlight that combining optical flow-guided tilt correction with Gaussian splatting effectively enhances image restoration under both synthetic and real-world turbulence conditions. The code for this method will be available at https://github.com/DuhlLiamz/3DGS_turbulence/tree/main.

-> 대기 왜곡으로 인한 스포츠 영상 저하 문제 해결에 필수적인 기술

### FLIGHT: Fibonacci Lattice-based Inference for Geometric Heading in real-Time (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23115v1
- 점수: final 86.4

Estimating camera motion from monocular video is a fundamental problem in computer vision, central to tasks such as SLAM, visual odometry, and structure-from-motion. Existing methods that recover the camera's heading under known rotation, whether from an IMU or an optimization algorithm, tend to perform well in low-noise, low-outlier conditions, but often decrease in accuracy or become computationally expensive as noise and outlier levels increase. To address these limitations, we propose a novel generalization of the Hough transform on the unit sphere (S(2)) to estimate the camera's heading. First, the method extracts correspondences between two frames and generates a great circle of directions compatible with each pair of correspondences. Then, by discretizing the unit sphere using a Fibonacci lattice as bin centers, each great circle casts votes for a range of directions, ensuring that features unaffected by noise or dynamic objects vote consistently for the correct motion direction. Experimental results on three datasets demonstrate that the proposed method is on the Pareto frontier of accuracy versus efficiency. Additionally, experiments on SLAM show that the proposed method reduces RMSE by correcting the heading during camera pose initialization.

-> 실시간 카메라 모션 추정으로 자동 촬영 정확도 향상, 노이즈 많은 스포츠 환경에서도 효과적

### Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.01400v1
- 점수: final 86.4

Video Large Language Models (VLLMs) demonstrate strong video understanding but suffer from inefficiency due to redundant visual tokens. Existing pruning primary targets intra-frame spatial redundancy or prunes inside the LLM with shallow-layer overhead, yielding suboptimal spatiotemporal reduction and underutilizing long-context compressibility. All of them often discard subtle yet informative context from merged or pruned tokens. In this paper, we propose a new perspective that elaborates token \textbf{A}nchors within intra-frame and inter-frame to comprehensively aggregate the informative contexts via local-global \textbf{O}ptimal \textbf{T}ransport (\textbf{AOT}). Specifically, we first establish local- and global-aware token anchors within each frame under the attention guidance, which then optimal transport aggregates the informative contexts from pruned tokens, constructing intra-frame token anchors. Then, building on the temporal frame clips, the first frame within each clip will be considered as the keyframe anchors to ensemble similar information from consecutive frames through optimal transport, while keeping distinct tokens to represent temporal dynamics, leading to efficient token reduction in a training-free manner. Extensive evaluations show that our proposed AOT obtains competitive performances across various short- and long-video benchmarks on leading video LLMs, obtaining substantial computational efficiency while preserving temporal and visual fidelity. Project webpage: \href{https://tyroneli.github.io/AOT}{AOT}.

-> Token reduction method for efficient video understanding applicable to sports content analysis

### WorldStereo: Bridging Camera-Guided Video Generation and Scene Reconstruction via 3D Geometric Memories (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.02049v1
- 점수: final 86.4

Recent advances in foundational Video Diffusion Models (VDMs) have yielded significant progress. Yet, despite the remarkable visual quality of generated videos, reconstructing consistent 3D scenes from these outputs remains challenging, due to limited camera controllability and inconsistent generated content when viewed from distinct camera trajectories. In this paper, we propose WorldStereo, a novel framework that bridges camera-guided video generation and 3D reconstruction via two dedicated geometric memory modules. Formally, the global-geometric memory enables precise camera control while injecting coarse structural priors through incrementally updated point clouds. Moreover, the spatial-stereo memory constrains the model's attention receptive fields with 3D correspondence to focus on fine-grained details from the memory bank. These components enable WorldStereo to generate multi-view-consistent videos under precise camera control, facilitating high-quality 3D reconstruction. Furthermore, the flexible control branch-based WorldStereo shows impressive efficiency, benefiting from the distribution matching distilled VDM backbone without joint training. Extensive experiments across both camera-guided video generation and 3D reconstruction benchmarks demonstrate the effectiveness of our approach. Notably, we show that WorldStereo acts as a powerful world model, tackling diverse scene generation tasks (whether starting from perspective or panoramic images) with high-fidelity 3D results. Models will be released.

-> 카메라 가이드 비디오 생성과 3D 복원을 위한 프레임워크가 스포츠 콘텐츠 제작에 적용 가능

### Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.02139v1
- 점수: final 86.4

The adoption of fisheye cameras in robotic manipulation, driven by their exceptionally wide Field of View (FoV), is rapidly outpacing a systematic understanding of their downstream effects on policy learning. This paper presents the first comprehensive empirical study to bridge this gap, rigorously analyzing the properties of wrist-mounted fisheye cameras for imitation learning. Through extensive experiments in both simulation and the real world, we investigate three critical research questions: spatial localization, scene generalization, and hardware generalization. Our investigation reveals that: (1) The wide FoV significantly enhances spatial localization, but this benefit is critically contingent on the visual complexity of the environment. (2) Fisheye-trained policies, while prone to overfitting in simple scenes, unlock superior scene generalization when trained with sufficient environmental diversity. (3) While naive cross-camera transfer leads to failures, we identify the root cause as scale overfitting and demonstrate that hardware generalization performance can be improved with a simple Random Scale Augmentation (RSA) strategy. Collectively, our findings provide concrete, actionable guidance for the large-scale collection and effective use of fisheye datasets in robotic learning. More results and videos are available on https://robo-fisheye.github.io/

-> 비디오 토큰 감소 기술은 스포츠 영상 분석 효율성을 높이고 하이라이트 추출 속도 개선 가능

### MSP-ReID: Hairstyle-Robust Cloth-Changing Person Re-Identification (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.01640v1
- 점수: final 85.6

Cloth-Changing Person Re-Identification (CC-ReID) aims to match the same individual across cameras under varying clothing conditions. Existing approaches often remove apparel and focus on the head region to reduce clothing bias. However, treating the head holistically without distinguishing between face and hair leads to over-reliance on volatile hairstyle cues, causing performance degradation under hairstyle changes. To address this issue, we propose the Mitigating Hairstyle Distraction and Structural Preservation (MSP) framework. Specifically, MSP introduces Hairstyle-Oriented Augmentation (HSOA), which generates intra-identity hairstyle diversity to reduce hairstyle dependence and enhance attention to stable facial and body cues. To prevent the loss of structural information, we design Cloth-Preserved Random Erasing (CPRE), which performs ratio-controlled erasing within clothing regions to suppress texture bias while retaining body shape and context. Furthermore, we employ Region-based Parsing Attention (RPA) to incorporate parsing-guided priors that highlight face and limb regions while suppressing hair features. Extensive experiments on multiple CC-ReID benchmarks demonstrate that MSP achieves state-of-the-art performance, providing a robust and practical solution for long-term person re-identification.

-> 선수 추적 기술이 스포츠 장면에서 선수 추적에 적용 가능

### InterCoG: Towards Spatially Precise Image Editing with Interleaved Chain-of-Grounding Reasoning (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.01586v1
- 점수: final 84.0

Emerging unified editing models have demonstrated strong capabilities in general object editing tasks. However, it remains a significant challenge to perform fine-grained editing in complex multi-entity scenes, particularly those where targets are not visually salient and require spatial reasoning. To this end, we propose InterCoG, a novel text-vision Interleaved Chain-of-Grounding reasoning framework for fine-grained image editing in complex real-world scenes. The key insight of InterCoG is to first perform object position reasoning solely within text that includes spatial relation details to explicitly deduce the location and identity of the edited target. It then conducts visual grounding via highlighting the editing targets with generated bounding boxes and masks in pixel space, and finally rewrites the editing description to specify the intended outcomes. To further facilitate this paradigm, we propose two auxiliary training modules: multimodal grounding reconstruction supervision and multimodal grounding reasoning alignment to enforce spatial localization accuracy and reasoning interpretability, respectively. We also construct GroundEdit-45K, a dataset comprising 45K grounding-oriented editing samples with detailed reasoning annotations, and GroundEdit-Bench for grounding-aware editing evaluation. Extensive experiments substantiate the superiority of our approach in highly precise edits under spatially intricate and multi-entity scenes.

-> 복잡한 스포츠 장면에서 정밀한 이미지 편집 기술로 영상 보정 및 하이라이트 생성에 적용 가능

### Locally Adaptive Decay Surfaces for High-Speed Face and Landmark Detection with Event Cameras (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23101v1
- 점수: final 84.0

Event cameras record luminance changes with microsecond resolution, but converting their sparse, asynchronous output into dense tensors that neural networks can exploit remains a core challenge. Conventional histograms or globally-decayed time-surface representations apply fixed temporal parameters across the entire image plane, which in practice creates a trade-off between preserving spatial structure during still periods and retaining sharp edges during rapid motion. We introduce Locally Adaptive Decay Surfaces (LADS), a family of event representations in which the temporal decay at each location is modulated according to local signal dynamics. Three strategies are explored, based on event rate, Laplacian-of-Gaussian response, and high-frequency spectral energy. These adaptive schemes preserve detail in quiescent regions while reducing blur in regions of dense activity. Extensive experiments on the public data show that LADS consistently improves both face detection and facial landmark accuracy compared to standard non-adaptive representations. At 30 Hz, LADS achieves higher detection accuracy and lower landmark error than either baseline, and at 240 Hz it mitigates the accuracy decline typically observed at higher frequencies, sustaining 2.44 % normalized mean error for landmarks and 0.966 mAP50 in face detection. These high-frequency results even surpass the accuracy reported in prior works operating at 30 Hz, setting new benchmarks for event-based face analysis. Moreover, by preserving spatial structure at the representation stage, LADS supports the use of much lighter network architectures while still retaining real-time performance. These results highlight the importance of context-aware temporal integration for neuromorphic vision and point toward real-time, high-frequency human-computer interaction systems that exploit the unique advantages of event cameras.

-> 고속 스포츠 동작 캡처에 적용 가능한 이벤트 카메라 기술로 빠른 움직임을 정확히 포착할 수 있습니다.

### TEFL: Prediction-Residual-Guided Rolling Forecasting for Multi-Horizon Time Series (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.22520v1
- 점수: final 84.0

Time series forecasting plays a critical role in domains such as transportation, energy, and meteorology. Despite their success, modern deep forecasting models are typically trained to minimize point-wise prediction loss without leveraging the rich information contained in past prediction residuals from rolling forecasts - residuals that reflect persistent biases, unmodeled patterns, or evolving dynamics. We propose TEFL (Temporal Error Feedback Learning), a unified learning framework that explicitly incorporates these historical residuals into the forecasting pipeline during both training and evaluation. To make this practical in deep multi-step settings, we address three key challenges: (1) selecting observable multi-step residuals under the partial observability of rolling forecasts, (2) integrating them through a lightweight low-rank adapter to preserve efficiency and prevent overfitting, and (3) designing a two-stage training procedure that jointly optimizes the base forecaster and error module. Extensive experiments across 10 real-world datasets and 5 backbone architectures show that TEFL consistently improves accuracy, reducing MAE by 5-10% on average. Moreover, it demonstrates strong robustness under abrupt changes and distribution shifts, with error reductions exceeding 10% (up to 19.5%) in challenging scenarios. By embedding residual-based feedback directly into the learning process, TEFL offers a simple, general, and effective enhancement to modern deep forecasting systems.

-> 스포츠 선수의 동작 패턴과 성적 예측을 위한 정확한 시계열 분석 필요

### PackUV: Packed Gaussian UV Maps for 4D Volumetric Video (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23040v1
- 점수: final 82.4

Volumetric videos offer immersive 4D experiences, but remain difficult to reconstruct, store, and stream at scale. Existing Gaussian Splatting based methods achieve high-quality reconstruction but break down on long sequences, temporal inconsistency, and fail under large motions and disocclusions. Moreover, their outputs are typically incompatible with conventional video coding pipelines, preventing practical applications.   We introduce PackUV, a novel 4D Gaussian representation that maps all Gaussian attributes into a sequence of structured, multi-scale UV atlas, enabling compact, image-native storage. To fit this representation from multi-view videos, we propose PackUV-GS, a temporally consistent fitting method that directly optimizes Gaussian parameters in the UV domain. A flow-guided Gaussian labeling and video keyframing module identifies dynamic Gaussians, stabilizes static regions, and preserves temporal coherence even under large motions and disocclusions. The resulting UV atlas format is the first unified volumetric video representation compatible with standard video codecs (e.g., FFV1) without losing quality, enabling efficient streaming within existing multimedia infrastructure.   To evaluate long-duration volumetric capture, we present PackUV-2B, the largest multi-view video dataset to date, featuring more than 50 synchronized cameras, substantial motion, and frequent disocclusions across 100 sequences and 2B (billion) frames. Extensive experiments demonstrate that our method surpasses existing baselines in rendering fidelity while scaling to sequences up to 30 minutes with consistent quality.

-> 대용량 스포츠 영상 효율적 저장 및 공플랫폼에서의 실시간 스트리밍 필요

### Beyond Detection: Multi-Scale Hidden-Code for Natural Image Deepfake Recovery and Factual Retrieval (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.22759v1
- 점수: final 80.0

Recent advances in image authenticity have primarily focused on deepfake detection and localization, leaving recovery of tampered contents for factual retrieval relatively underexplored. We propose a unified hidden-code recovery framework that enables both retrieval and restoration from post-hoc and in-generation watermarking paradigms. Our method encodes semantic and perceptual information into a compact hidden-code representation, refined through multi-scale vector quantization, and enhances contextual reasoning via conditional Transformer modules. To enable systematic evaluation for natural images, we construct ImageNet-S, a benchmark that provides paired image-label factual retrieval tasks. Extensive experiments on ImageNet-S demonstrate that our method exhibits promising retrieval and reconstruction performance while remaining fully compatible with diverse watermarking pipelines. This framework establishes a foundation for general-purpose image recovery beyond detection and localization.

-> 이미지 복구 및 사실 검색 기술로 스포츠 영상 보정에 적용 가능하며 저조나 저품질 촬영 영상을 향상시킬 수 있습니다.

### SOTAlign: Semi-Supervised Alignment of Unimodal Vision and Language Models via Optimal Transport (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.23353v1
- 점수: final 80.0

The Platonic Representation Hypothesis posits that neural networks trained on different modalities converge toward a shared statistical model of the world. Recent work exploits this convergence by aligning frozen pretrained vision and language models with lightweight alignment layers, but typically relies on contrastive losses and millions of paired samples. In this work, we ask whether meaningful alignment can be achieved with substantially less supervision. We introduce a semi-supervised setting in which pretrained unimodal encoders are aligned using a small number of image-text pairs together with large amounts of unpaired data. To address this challenge, we propose SOTAlign, a two-stage framework that first recovers a coarse shared geometry from limited paired data using a linear teacher, then refines the alignment on unpaired samples via an optimal-transport-based divergence that transfers relational structure without overconstraining the target space. Unlike existing semi-supervised methods, SOTAlign effectively leverages unpaired images and text, learning robust joint embeddings across datasets and encoder pairs, and significantly outperforming supervised and semi-supervised baselines.

-> 생물학적 조직 세포 계산 알고리즘으로 스포츠 분석에 직접 적용 불가

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
