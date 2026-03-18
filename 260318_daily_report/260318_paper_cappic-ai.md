# CAPP!C_AI 논문 리포트 (2026-03-18)

> 수집 79 | 필터 73 | 폐기 3 | 평가 60 | 출력 46 | 기준 50점

검색 윈도우: 2026-03-17T00:00:00+00:00 ~ 2026-03-18T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 42

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: Advancing Visual Reliability: Color-Accurate Underwater Image Enhancement for Real-Time Underwater Missions

- arXiv: http://arxiv.org/abs/2603.16363v1
- PDF: https://arxiv.org/pdf/2603.16363v1
- 발행일: 2026-03-17
- 카테고리: cs.CV
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Underwater image enhancement plays a crucial role in providing reliable visual information for underwater platforms, since strong absorption and scattering in water-related environments generally lead to image quality degradation. Existing high-performance methods often rely on complex architectures, which hinder deployment on underwater devices. Lightweight methods often sacrifice quality for speed and struggle to handle severely degraded underwater images. To address this limitation, we present a real-time underwater image enhancement framework with accurate color restoration. First, an Adaptive Weighted Channel Compensation module is introduced to achieve dynamic color recovery of the red and blue channels using the green channel as a reference anchor. Second, we design a Multi-branch Re-parameterized Dilated Convolution that employs multi-branch fusion during training and structural re-parameterization during inference, enabling large receptive field representation with low computational overhead. Finally, a Statistical Global Color Adjustment module is employed to optimize overall color performance based on statistical priors. Extensive experiments on eight datasets demonstrate that the proposed method achieves state-of-the-art performance across seven evaluation metrics. The model contains only 3,880 inference parameters and achieves an inference speed of 409 FPS. Our method improves the UCIQE score by 29.7% under diverse environmental conditions, and the deployment on ROV platforms and performance gains in downstream tasks further validate its superiority for real-time underwater missions.

**선정 근거**
실시간 이미지 보정 및 향상 기술은 스포츠 영상 처리에 직접적으로 적용 가능하며 경량화 모델 제공

**활용 인사이트**
적응 가중 채널 보상 및 다중 분리 확장 컨볼루션을 rk3588 엣지 디바이스에 구현하여 409fps로 실시간 스포츠 영상 보정 가능

## 2위: Deep Reinforcement Learning-driven Edge Offloading for Latency-constrained XR pipelines

- arXiv: http://arxiv.org/abs/2603.16823v1
- PDF: https://arxiv.org/pdf/2603.16823v1
- 발행일: 2026-03-17
- 카테고리: cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Immersive extended reality (XR) applications introduce latency-critical workloads that must satisfy stringent real-time responsiveness while operating on energy- and battery-constrained devices, making execution placement between end devices and nearby edge servers a fundamental systems challenge. Existing approaches to adaptive execution and computation offloading typically optimize average performance metrics and do not fully capture the sustained interaction between real-time latency requirements and device battery lifetime in closed-loop XR workloads. In this paper, we present a battery-aware execution management framework for edge-assisted XR systems that jointly considers execution placement, workload quality, latency requirements, and battery dynamics. We design an online decision mechanism based on a lightweight deep reinforcement learning policy that continuously adapts execution decisions under dynamic network conditions while maintaining high motion-to-photon latency compliance. Experimental results show that the proposed approach extends the projected device battery lifetime by up to 163% compared to latency-optimal local execution while maintaining over 90% motion-to-photon latency compliance under stable network conditions. Such compliance does not fall below 80% even under significantly limited network bandwidth availability, thereby demonstrating the effectiveness of explicitly managing latency-energy trade-offs in immersive XR systems.

**선정 근거**
엣지 디바이스에서 실시간으로 작동하는 강화 학습 프레임워크로 프로젝트의 rk3588 기반 AI 촬영 장치 및 실시간 처리 요구사항과 직접적으로 관련되어 있습니다. 저지연-배터리 수명 트레이드오프 관리는 모바일 스포츠 촬영 장치에 필수적입니다.

**활용 인사이트**
강화 학습 기반 실행 관리 프레임워크를 적용하여 스포츠 장면 촬영 시 네트워크 상태에 따라 처리 위치를 동적으로 최적화하고, 실시간 하이라이트 편집 및 동작 분석을 위한 지연 시간을 90% 이상 유지하면서 배터리 수명을 최대 163% 연장할 수 있습니다.

## 3위: SpikeCLR: Contrastive Self-Supervised Learning for Few-Shot Event-Based Vision using Spiking Neural Networks

- arXiv: http://arxiv.org/abs/2603.16338v1
- PDF: https://arxiv.org/pdf/2603.16338v1
- 발행일: 2026-03-17
- 카테고리: cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Event-based vision sensors provide significant advantages for high-speed perception, including microsecond temporal resolution, high dynamic range, and low power consumption. When combined with Spiking Neural Networks (SNNs), they can be deployed on neuromorphic hardware, enabling energy-efficient applications on embedded systems. However, this potential is severely limited by the scarcity of large-scale labeled datasets required to effectively train such models. In this work, we introduce SpikeCLR, a contrastive self-supervised learning framework that enables SNNs to learn robust visual representations from unlabeled event data. We adapt prior frame-based methods to the spiking domain using surrogate gradient training and introduce a suite of event-specific augmentations that leverage spatial, temporal, and polarity transformations. Through extensive experiments on CIFAR10-DVS, N-Caltech101, N-MNIST, and DVS-Gesture benchmarks, we demonstrate that self-supervised pretraining with subsequent fine-tuning outperforms supervised learning in low-data regimes, achieving consistent gains in few-shot and semi-supervised settings. Our ablation studies reveal that combining spatial and temporal augmentations is critical for learning effective spatio-temporal invariances in event data. We further show that learned representations transfer across datasets, contributing to efforts for powerful event-based models in label-scarce settings.

**선정 근거**
Event-based vision with SNNs provides efficient processing for edge sports camera devices

## 4위: Emotion-Aware Classroom Quality Assessment Leveraging IoT-Based Real-Time Student Monitoring

- arXiv: http://arxiv.org/abs/2603.16719v1
- PDF: https://arxiv.org/pdf/2603.16719v1
- 발행일: 2026-03-17
- 카테고리: cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
This study presents high-throughput, real-time multi-agent affective computing framework designed to enhance classroom learning through emotional state monitoring. As large classroom sizes and limited teacher student interaction increasingly challenge educators, there is a growing need for scalable, data-driven tools capable of capturing students' emotional and engagement patterns in real time. The system was evaluated using the Classroom Emotion Dataset, consisting of 1,500 labeled images and 300 classroom detection videos. Tailored for IoT devices, the system addresses load balancing and latency challenges through efficient real-time processing. Field testing was conducted across three educational institutions in a large metropolitan area: a primary school (hereafter school A), a secondary school (school B), and a high school (school C). The system demonstrated robust performance, detecting up to 50 faces at 25 FPS and achieving 88% overall accuracy in classifying classroom engagement states. Implementation results showed positive outcomes, with favorable feedback from students, teachers, and parents regarding improved classroom interaction and teaching adaptation. Key contributions of this research include establishing a practical, IoT-based framework for emotion-aware learning environments and introducing the 'Classroom Emotion Dataset' to facilitate further validation and research.

**선정 근거**
실시간 비디오 분석 및 IoT 장치 배포 기술은 스포츠 촬영 장치에 적용 가능하나 교실 환경에 특화됨

## 5위: Efficient Reasoning on the Edge

- arXiv: http://arxiv.org/abs/2603.16867v1
- PDF: https://arxiv.org/pdf/2603.16867v1
- 발행일: 2026-03-17
- 카테고리: cs.LG, cs.CL
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Large language models (LLMs) with chain-of-thought reasoning achieve state-of-the-art performance across complex problem-solving tasks, but their verbose reasoning traces and large context requirements make them impractical for edge deployment. These challenges include high token generation costs, large KV-cache footprints, and inefficiencies when distilling reasoning capabilities into smaller models for mobile devices. Existing approaches often rely on distilling reasoning traces from larger models into smaller models, which are verbose and stylistically redundant, undesirable for on-device inference. In this work, we propose a lightweight approach to enable reasoning in small LLMs using LoRA adapters combined with supervised fine-tuning. We further introduce budget forcing via reinforcement learning on these adapters, significantly reducing response length with minimal accuracy loss. To address memory-bound decoding, we exploit parallel test-time scaling, improving accuracy at minor latency increase. Finally, we present a dynamic adapter-switching mechanism that activates reasoning only when needed and a KV-cache sharing strategy during prompt encoding, reducing time-to-first-token for on-device inference. Experiments on Qwen2.5-7B demonstrate that our method achieves efficient, accurate reasoning under strict resource constraints, making LLM reasoning practical for mobile scenarios. Videos demonstrating our solution running on mobile devices are available on our project page.

**선정 근거**
Edge computing techniques for efficient AI reasoning applicable to sports analysis on rk3588 device

## 6위: PA-LVIO: Real-Time LiDAR-Visual-Inertial Odometry and Mapping with Pose-Only Bundle Adjustment

- arXiv: http://arxiv.org/abs/2603.16228v1
- PDF: https://arxiv.org/pdf/2603.16228v1
- 발행일: 2026-03-17
- 카테고리: cs.RO
- 점수: final 90.4 (llm_adjusted:88 = base:78 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Real-time LiDAR-visual-inertial odometry and mapping is crucial for navigation and planning tasks in intelligent transportation systems. This study presents a pose-only bundle adjustment (PA) LiDAR-visual-inertial odometry (LVIO), named PA-LVIO, to meet the urgent need for real-time navigation and mapping. The proposed PA framework for LiDAR and visual measurements is highly accurate and efficient, and it can derive reliable frame-to-frame constraints within multiple frames. A marginalization-free and frame-to-map (F2M) LiDAR measurement model is integrated into the state estimator to eliminate odometry drifts. Meanwhile, an IMU-centric online spatial-temporal calibration is employed to obtain a pixel-wise LiDAR-camera alignment. With accurate estimated odometry and extrinsics, a high-quality and RGB-rendered point-cloud map can be built. Comprehensive experiments are conducted on both public and private datasets collected by wheeled robot, unmanned aerial vehicle (UAV), and handheld devices with 28 sequences and more than 50 km trajectories. Sufficient results demonstrate that the proposed PA-LVIO yields superior or comparable performance to state-of-the-art LVIO methods, in terms of the odometry accuracy and mapping quality. Besides, PA-LVIO can run in real-time on both the desktop PC and the onboard ARM computer.

**선정 근거**
실시간 시각-관성 항법으로 스포츠 장면의 정확한 움직임 추적이 가능해 자동 촬영 및 하이라이트 편집의 기반이 됩니다.

**활용 인사이트**
rk3588 기반 장치에 PA-LVIO를 적용해 실시간으로 선수 움직임을 추적하며, 주요 장면을 자동으로 감지해 하이라이트 영상을 생성합니다.

## 7위: SE(3)-LIO: Smooth IMU Propagation With Jointly Distributed Poses on SE(3) Manifold for Accurate and Robust LiDAR-Inertial Odometry

- arXiv: http://arxiv.org/abs/2603.16118v1
- PDF: https://arxiv.org/pdf/2603.16118v1
- 코드: https://se3-lio.github.io/
- 발행일: 2026-03-17
- 카테고리: cs.RO
- 점수: final 90.4 (llm_adjusted:88 = base:80 + bonus:+8)
- 플래그: 실시간, 코드 공개

**개요**
In estimating odometry accurately, an inertial measurement unit (IMU) is widely used owing to its high-rate measurements, which can be utilized to obtain motion information through IMU propagation. In this paper, we address the limitations of existing IMU propagation methods in terms of motion prediction and motion compensation. In motion prediction, the existing methods typically represent a 6-DoF pose by separating rotation and translation and propagate them on their respective manifold, so that the rotational variation is not effectively incorporated into translation propagation. During motion compensation, the relative transformation between predicted poses is used to compensate motion-induced distortion in other measurements, while inherent errors in the predicted poses introduce uncertainty in the relative transformation. To tackle these challenges, we represent and propagate the pose on SE(3) manifold, where propagated translation properly accounts for rotational variation. Furthermore, we precisely characterize the relative transformation uncertainty by considering the correlation between predicted poses, and incorporate this uncertainty into the measurement noise during motion compensation. To this end, we propose a LiDAR-inertial odometry (LIO), referred to as SE(3)-LIO, that integrates the proposed IMU propagation and uncertainty-aware motion compensation (UAMC). We validate the effectiveness of SE(3)-LIO on diverse datasets. Our source code and additional material are available at: https://se3-lio.github.io/.

**선정 근거**
SE(3) 다양체를 이용한 정밀한 움직임 추정은 스포츠 동작 분석의 정확도를 높여 전략 분석에 기여합니다.

**활용 인사이트**
SE(3)-LIO를 활용해 선수들의 정밀한 동작을 분석하고, 자세 교정 및 경기 전략 개선을 위한 실시간 피드백을 제공합니다.

## 8위: Collaborative Temporal Feature Generation via Critic-Free Reinforcement Learning for Cross-User Sensor-Based Activity Recognition

- arXiv: http://arxiv.org/abs/2603.16043v1
- PDF: https://arxiv.org/pdf/2603.16043v1
- 발행일: 2026-03-17
- 카테고리: cs.LG, cs.AI, cs.CV
- 점수: final 90.4 (llm_adjusted:88 = base:78 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Human Activity Recognition using wearable inertial sensors is foundational to healthcare monitoring, fitness analytics, and context-aware computing, yet its deployment is hindered by cross-user variability arising from heterogeneous physiological traits, motor habits, and sensor placements. Existing domain generalization approaches either neglect temporal dependencies in sensor streams or depend on impractical target-domain annotations. We propose a different paradigm: modeling generalizable feature extraction as a collaborative sequential generation process governed by reinforcement learning. Our framework, CTFG (Collaborative Temporal Feature Generation), employs a Transformer-based autoregressive generator that incrementally constructs feature token sequences, each conditioned on prior context and the encoded sensor input. The generator is optimized via Group-Relative Policy Optimization, a critic-free algorithm that evaluates each generated sequence against a cohort of alternatives sampled from the same input, deriving advantages through intra-group normalization rather than learned value estimation. This design eliminates the distribution-dependent bias inherent in critic-based methods and provides self-calibrating optimization signals that remain stable across heterogeneous user distributions. A tri-objective reward comprising class discrimination, cross-user invariance, and temporal fidelity jointly shapes the feature space to separate activities, align user distributions, and preserve fine-grained temporal content. Evaluations on the DSADS and PAMAP2 benchmarks demonstrate state-of-the-art cross-user accuracy (88.53\% and 75.22\%), substantial reduction in inter-task training variance, accelerated convergence, and robust generalization under varying action-space dimensionalities.

**선정 근거**
크로스-유저 활동 인식 기술은 다양한 사용자의 스포츠 동작을 정확히 분석해 개인별 맞춤형 훈련을 가능하게 합니다.

**활용 인사이트**
CTFG 프레임워크를 적용해 각 사용자의 특성에 맞춰 동작을 분석하고, 개인별 맞춤형 훈련 프로그램과 피드백을 생성합니다.

## 9위: DST-Net: A Dual-Stream Transformer with Illumination-Independent Feature Guidance and Multi-Scale Spatial Convolution for Low-Light Image Enhancement

- arXiv: http://arxiv.org/abs/2603.16482v1
- PDF: https://arxiv.org/pdf/2603.16482v1
- 발행일: 2026-03-17
- 카테고리: cs.CV, cs.AI
- 점수: final 90.4 (llm_adjusted:88 = base:78 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Low-light image enhancement aims to restore the visibility of images captured by visual sensors in dim environments by addressing their inherent signal degradations, such as luminance attenuation and structural corruption. Although numerous algorithms attempt to improve image quality, existing methods often cause a severe loss of intrinsic signal priors. To overcome these challenges, we propose a Dual-Stream Transformer Network (DST-Net) based on illumination-agnostic signal prior guidance and multi-scale spatial convolutions. First, to address the loss of critical signal features under low-light conditions, we design a feature extraction module. This module integrates Difference of Gaussians (DoG), LAB color space transformations, and VGG-16 for texture extraction, utilizing decoupled illumination-agnostic features as signal priors to continuously guide the enhancement process. Second, we construct a dual-stream interaction architecture. By employing a cross-modal attention mechanism, the network leverages the extracted priors to dynamically rectify the deteriorated signal representation of the enhanced image, ultimately achieving iterative enhancement through differentiable curve estimation. Furthermore, to overcome the inability of existing methods to preserve fine structures and textures, we propose a Multi-Scale Spatial Fusion Block (MSFB) featuring pseudo-3D and 3D gradient operator convolutions. This module integrates explicit gradient operators to recover high-frequency edges while capturing inter-channel spatial correlations via multi-scale spatial convolutions. Extensive evaluations and ablation studies demonstrate that DST-Net achieves superior performance in subjective visual quality and objective metrics. Specifically, our method achieves a PSNR of 25.64 dB on the LOL dataset. Subsequent validation on the LSRW dataset further confirms its robust cross-scene generalization.

**선정 근거**
저조도 환경에서의 이미지 향상 기술은 다양한 조건에서 스포츠 영상의 품질을 보장해 콘텐츠 가치를 높입니다.

**활용 인사이트**
DST-Net을 통합해 야외 스포츠 경기나 저조도 환경에서 촬영된 영상을 자동으로 보정하고 사진처럼 고품질 이미지를 생성합니다.

## 10위: Knowledge Distillation for Collaborative Learning in Distributed Communications and Sensing

- arXiv: http://arxiv.org/abs/2603.16116v1
- PDF: https://arxiv.org/pdf/2603.16116v1
- 발행일: 2026-03-17
- 카테고리: eess.SP
- 점수: final 90.4 (llm_adjusted:88 = base:78 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
The rise of sixth generation (6G) wireless networks promises to deliver ultra-reliable, low-latency, and energy-efficient communications, sensing, and computing. However, traditional centralized artificial intelligence (AI) paradigms are ill-suited to the decentralized, resource-constrained, and dynamic nature of 6G ecosystems. This paper explores knowledge distillation (KD) and collaborative learning as promising techniques that enable the efficient and scalable deployment of lightweight AI models across distributed communications and sensing (C&S) nodes. We begin by providing an overview of KD and highlight the key strengths that make it particularly effective in distributed scenarios characterized by device heterogeneity, task diversity, and constrained resources. We then examine its role in fostering collective intelligence through collaborative learning between the central and distributed nodes via various knowledge distilling and deployment strategies. Finally, we present a systematic numerical study demonstrating that KD-empowered collaborative learning can effectively support lightweight AI models for multi-modal sensing-assisted beam tracking applications with substantial performance gains and complexity reduction.

**선정 근거**
지식 증류 기술은 경량화 AI 모델을 효율적으로 배포해 rk3588 기반 장치의 성능을 최적화합니다.

**활용 인사이트**
지식 증류를 통해 대형 모델의 성능을 유지하면서도 경량화해 실시간 영상 처리 및 분석을 가능하게 합니다.

## 11위: MessyKitchens: Contact-rich object-level 3D scene reconstruction

- arXiv: http://arxiv.org/abs/2603.16868v1
- PDF: https://arxiv.org/pdf/2603.16868v1
- 발행일: 2026-03-17
- 카테고리: cs.CV, cs.AI, cs.RO
- 점수: final 88.0 (llm_adjusted:85 = base:82 + bonus:+3)
- 플래그: 코드 공개

**개요**
Monocular 3D scene reconstruction has recently seen significant progress. Powered by the modern neural architectures and large-scale data, recent methods achieve high performance in depth estimation from a single image. Meanwhile, reconstructing and decomposing common scenes into individual 3D objects remains a hard challenge due to the large variety of objects, frequent occlusions and complex object relations. Notably, beyond shape and pose estimation of individual objects, applications in robotics and animation require physically-plausible scene reconstruction where objects obey physical principles of non-penetration and realistic contacts. In this work we advance object-level scene reconstruction along two directions. First, we introduceMessyKitchens, a new dataset with real-world scenes featuring cluttered environments and providing high-fidelity object-level ground truth in terms of 3D object shapes, poses and accurate object contacts. Second, we build on the recent SAM 3D approach for single-object reconstruction and extend it with Multi-Object Decoder (MOD) for joint object-level scene reconstruction. To validate our contributions, we demonstrate MessyKitchens to significantly improve previous datasets in registration accuracy and inter-object penetration. We also compare our multi-object reconstruction approach on three datasets and demonstrate consistent and significant improvements of MOD over the state of the art. Our new benchmark, code and pre-trained models will become publicly available on our project website: https://messykitchens.github.io/.

**선정 근거**
3D scene reconstruction technology applicable to sports analysis but not specifically designed for it

## 12위: Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty

- arXiv: http://arxiv.org/abs/2603.16538v1
- PDF: https://arxiv.org/pdf/2603.16538v1
- 발행일: 2026-03-17
- 카테고리: cs.CV
- 점수: final 88.0 (llm_adjusted:85 = base:85 + bonus:+0)

**개요**
3D Gaussian Splatting (3DGS) has recently emerged as a powerful scene representation and is increasingly used for visual localization and pose refinement. However, despite its high-quality differentiable rendering, the robustness of 3DGS-based pose refinement remains highly sensitive to both the initial camera pose and the reconstructed geometry. In this work, we take a closer look at these limitations and identify two major sources of uncertainty: (i) pose prior uncertainty, which often arises from regression or retrieval models that output a single deterministic estimate, and (ii) geometric uncertainty, caused by imperfections in the 3DGS reconstruction that propagate errors into PnP solvers. Such uncertainties can distort reprojection geometry and destabilize optimization, even when the rendered appearance still looks plausible. To address these uncertainties, we introduce a relocalization framework that combines Monte Carlo pose sampling with Fisher Information-based PnP optimization. Our method explicitly accounts for both pose and geometric uncertainty and requires no retraining or additional supervision. Across diverse indoor and outdoor benchmarks, our approach consistently improves localization accuracy and significantly increases stability under pose and depth noise.

**선정 근거**
4D 포인트 클라우드 비디오 이해 기술은 스포츠 장면의 동적 분석에 직접적으로 적용 가능하며, 다양한 프레임 레이트 처리와 움직임 패턴 인식에 유리합니다.

**활용 인사이트**
GATS 모델을 스포츠 영상 분석에 적용하여 선수들의 움직임을 4D로 분석하고, 경기 전략을 자동으로 추출하며, 실시간 하이라이트 장면을 식별할 수 있습니다.

## 13위: BLADE: Adaptive Wi-Fi Contention Control for Next-Generation Real-Time Communication

- arXiv: http://arxiv.org/abs/2603.16119v1
- PDF: https://arxiv.org/pdf/2603.16119v1
- 발행일: 2026-03-17
- 카테고리: cs.NI
- 점수: final 88.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Next-generation real-time communication (NGRTC) applications, such as cloud gaming and XR, demand consistently ultra-low latency. However, through our first large-scale measurement, we find that despite the deployment of edge servers, dedicated congestion control, and loss recovery mechanisms, cloud gaming users still experience long-tail latency in Wi-Fi networks. We further identify that Wi-Fi last-mile access points (APs) serve as the primary latency bottleneck. Specifically, short-term packet delivery droughts, caused by fundamental limitations in Wi-Fi contention control standards, are the root cause. To address this issue, we propose BLADE, an adaptive contention control algorithm that dynamically adjusts the contention windows (CW) of all Wi-Fi transmitters based on the channel contention level in a fully distributed manner. Our NS3 simulations and real-world evaluations with commercial Wi-Fi APs demonstrate that, compared to standard contention control, BLADE reduces Wi-Fi packet transmission tail latency by over 5X under heavy channel contention and significantly stabilizes MAC throughput while ensuring fast and fair convergence. Consequently, BLADE reduces the video stall rate in cloud gaming by over 90%.

**선정 근거**
Wi-Fi 경쟁 제어 알고리즘은 스포츠 콘텐츠 플랫폼의 실시간 공유 기능에 직접적으로 적용 가능하며, 지연 시간을 크게 줄여 사용자 경험을 향상시킵니다.

**활용 인사이트**
BLADE 알고리즘을 적용하여 스포츠 경기 영상과 하이라이트 콘텐츠의 전송 지연 시간을 5배 이상 줄이고, 비디오 중단률을 90% 이상 감소시켜 실시간 스트리밍과 공유 기능을 최적화할 수 있습니다.

## 14위: GATS: Gaussian Aware Temporal Scaling Transformer for Invariant 4D Spatio-Temporal Point Cloud Representation

- arXiv: http://arxiv.org/abs/2603.16154v1
- PDF: https://arxiv.org/pdf/2603.16154v1
- 발행일: 2026-03-17
- 카테고리: cs.CV, cs.AI
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Understanding 4D point cloud videos is essential for enabling intelligent agents to perceive dynamic environments. However, temporal scale bias across varying frame rates and distributional uncertainty in irregular point clouds make it highly challenging to design a unified and robust 4D backbone. Existing CNN or Transformer based methods are constrained either by limited receptive fields or by quadratic computational complexity, while neglecting these implicit distortions. To address this problem, we propose a novel dual invariant framework, termed \textbf{Gaussian Aware Temporal Scaling (GATS)}, which explicitly resolves both distributional inconsistencies and temporal. The proposed \emph{Uncertainty Guided Gaussian Convolution (UGGC)} incorporates local Gaussian statistics and uncertainty aware gating into point convolution, thereby achieving robust neighborhood aggregation under density variation, noise, and occlusion. In parallel, the \emph{Temporal Scaling Attention (TSA)} introduces a learnable scaling factor to normalize temporal distances, ensuring frame partition invariance and consistent velocity estimation across different frame rates. These two modules are complementary: temporal scaling normalizes time intervals prior to Gaussian estimation, while Gaussian modeling enhances robustness to irregular distributions. Our experiments on mainstream benchmarks MSR-Action3D (\textbf{+6.62\%} accuracy), NTU RGBD (\textbf{+1.4\%} accuracy), and Synthia4D (\textbf{+1.8\%} mIoU) demonstrate significant performance gains, offering a more efficient and principled paradigm for invariant 4D point cloud video understanding with superior accuracy, robustness, and scalability compared to Transformer based counterparts.

**선정 근거**
포인트 클라우드 데이터 처리 기술은 스포츠 장면 분석에 간접적으로 적용 가능

## 15위: LICA: Layered Image Composition Annotations for Graphic Design Research

- arXiv: http://arxiv.org/abs/2603.16098v1
- PDF: https://arxiv.org/pdf/2603.16098v1
- 발행일: 2026-03-17
- 카테고리: cs.CV, cs.AI
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
We introduce LICA (Layered Image Composition Annotations), a large-scale dataset of 1,550,244 multi-layer graphic design compositions designed to advance structured understanding and generation of graphic layouts1. In addition to ren- dered PNG images, LICA represents each design as a hierarchical composition of typed components including text, image, vector, and group elements, each paired with rich per-element metadata such as spatial geometry, typographic attributes, opacity, and visibility. The dataset spans 20 design categories and 971,850 unique templates, providing broad coverage of real-world design structures. We further introduce graphic design video as a new and largely unexplored challenge for current vision-language models through 27,261 animated layouts annotated with per-component keyframes and motion parameters. Beyond scale, LICA establishes a new paradigm of research tasks for graphic design, enabling structured investiga- tions into problems such as layer-aware inpainting, structured layout generation, controlled design editing, and temporally-aware generative modeling. By repre- senting design as a system of compositional layers and relationships, the dataset supports research on models that operate directly on design structure rather than pixels alone.

**선정 근거**
계층적 이미지 합성 기술은 스포츠 영상 편집 및 보정에 적용 가능하며, 다양한 디자인 요소를 구조화하여 전문적인 하이라이트 영상 제작을 지원합니다.

**활용 인사이트**
LICA 데이터셋과 기술을 활용하여 스포츠 영상을 계층적으로 분석하고 텍스트, 이미지, 벡터 요소를 조합하여 사진처럼 보정된 전문적인 스포츠 하이라이트 영상을 자동으로 생성할 수 있습니다.

## 16위: Change is Hard: Consistent Player Behavior Across Games with Conflicting Incentives

- arXiv: http://arxiv.org/abs/2603.16136v1
- PDF: https://arxiv.org/pdf/2603.16136v1
- 발행일: 2026-03-17
- 카테고리: cs.HC
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
This paper examines how player flexibility -- a player's willingness to engage in a breadth of options or specialize -- manifests across two gaming environments: League of Legends (League) and Teamfight Tactics (TFT). We analyze the gameplay decisions of 4,830 players who have played at least 50 competitive games in both titles and explore cross-game dynamics of behavior retention and consistency. Our work introduces a novel cross-game analysis that tracks the same players' behavior across two different environments, reducing self-selection bias. Our findings reveal that while games incentivize different behaviors (specialization in League versus flexibility in TFT) for performance-based success, players exhibit consistent behavior across platforms. This study contributes to long-standing debate about agency versus structure, showing individual agency may be more predictive of cross-platform behavior than game-imposed structure in competitive settings. These insights offer implications for game developers, designers and researchers interested in building systems to promote behavior change.

**선정 근거**
스포츠 촬영 시 다양한 조명 조건에서 발생하는 과도노출 문제를 해결하여 영상 품질을 향상시키는 데 직접적으로 적용 가능합니다.

**활용 인사이트**
적외선과 가시광 영상 융합 기술을 활용하여 과도노출 영역의 세부 정보를 보존하면서도 자연스러운 영상을 생성하여 스포츠 하이라이트 영상의 시각적 품질을 개선할 수 있습니다.

## 17위: SparkVSR: Interactive Video Super-Resolution via Sparse Keyframe Propagation

- arXiv: http://arxiv.org/abs/2603.16864v1
- PDF: https://arxiv.org/pdf/2603.16864v1
- 발행일: 2026-03-17
- 카테고리: cs.CV, cs.AI
- 점수: final 84.8 (llm_adjusted:81 = base:78 + bonus:+3)
- 플래그: 코드 공개

**개요**
Video Super-Resolution (VSR) aims to restore high-quality video frames from low-resolution (LR) estimates, yet most existing VSR approaches behave like black boxes at inference time: users cannot reliably correct unexpected artifacts, but instead can only accept whatever the model produces. In this paper, we propose a novel interactive VSR framework dubbed SparkVSR that makes sparse keyframes a simple and expressive control signal. Specifically, users can first super-resolve or optionally a small set of keyframes using any off-the-shelf image super-resolution (ISR) model, then SparkVSR propagates the keyframe priors to the entire video sequence while remaining grounded by the original LR video motion. Concretely, we introduce a keyframe-conditioned latent-pixel two-stage training pipeline that fuses LR video latents with sparsely encoded HR keyframe latents to learn robust cross-space propagation and refine perceptual details. At inference time, SparkVSR supports flexible keyframe selection (manual specification, codec I-frame extraction, or random sampling) and a reference-free guidance mechanism that continuously balances keyframe adherence and blind restoration, ensuring robust performance even when reference keyframes are absent or imperfect. Experiments on multiple VSR benchmarks demonstrate improved temporal consistency and strong restoration quality, surpassing baselines by up to 24.6%, 21.8%, and 5.6% on CLIP-IQA, DOVER, and MUSIQ, respectively, enabling controllable, keyframe-driven video super-resolution. Moreover, we demonstrate that SparkVSR is a generic interactive, keyframe-conditioned video processing framework as it can be applied out of the box to unseen tasks such as old-film restoration and video style transfer. Our project page is available at: https://sparkvsr.github.io/

**선정 근거**
비디오 슈퍼리졸루션 기술은 스포츠 영상 품질 개선에 적용 가능하나, 스포츠 촬영 및 분석에 직접적으로 관련되지는 않음

## 18위: EPOFusion: Exposure aware Progressive Optimization Method for Infrared and Visible Image Fusion

- arXiv: http://arxiv.org/abs/2603.16130v1
- PDF: https://arxiv.org/pdf/2603.16130v1
- 코드: https://github.com/warren-wzw/EPOFusion.git
- 발행일: 2026-03-17
- 카테고리: cs.CV
- 점수: final 84.8 (llm_adjusted:81 = base:78 + bonus:+3)
- 플래그: 코드 공개

**개요**
Overexposure frequently occurs in practical scenarios, causing the loss of critical visual information. However, existing infrared and visible fusion methods still exhibit unsatisfactory performance in highly bright regions. To address this, we propose EPOFusion, an exposure-aware fusion model. Specifically, a guidance module is introduced to facilitate the encoder in extracting fine-grained infrared features from overexposed regions. Meanwhile, an iterative decoder incorporating a multiscale context fusion module is designed to progressively enhance the fused image, ensuring consistent details and superior visual quality. Finally, an adaptive loss function dynamically constrains the fusion process, enabling an effective balance between the modalities under varying exposure conditions. To achieve better exposure awareness, we construct the first infrared and visible overexposure dataset (IVOE) with high quality infrared guided annotations for overexposed regions. Extensive experiments show that EPOFusion outperforms existing methods. It maintains infrared cues in overexposed regions while achieving visually faithful fusion in non-overexposed areas, thereby enhancing both visual fidelity and downstream task performance. Code, fusion results and IVOE dataset will be made available at https://github.com/warren-wzw/EPOFusion.git.

**선정 근거**
Image fusion technology directly applicable to improving sports video quality under various lighting conditions

## 19위: M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM

- arXiv: http://arxiv.org/abs/2603.16844v1
- PDF: https://arxiv.org/pdf/2603.16844v1
- 발행일: 2026-03-17
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:78 + bonus:+0)

**개요**
Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments. While coupling 3D foundation models with SLAM frameworks is a promising paradigm, a critical bottleneck persists: most multi-view foundation models estimate poses in a feed-forward manner, yielding pixel-level correspondences that lack the requisite precision for rigorous geometric optimization. To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM. M^3 further enhances tracking stability by incorporating dynamic area suppression and cross-inference intrinsic alignment. Extensive experiments on diverse indoor and outdoor benchmarks demonstrate state-of-the-art accuracy in both pose estimation and scene reconstruction. Notably, M^3 reduces ATE RMSE by 64.3% compared to VGGT-SLAM 2.0 and outperforms ARTDECO by 2.11 dB in PSNR on the ScanNet++ dataset.

**선정 근거**
동적 환경에서의 정밀 자세 추정과 영상 재구성 기술은 스포츠 경기 촬영 및 하이라이트 자동 편집 시 실시간 처리와 정확도 향상에 필수적

**활용 인사이트**
다중 뷰 기반 모델을 슬램 시스템에 통합하여 스포츠 경기장의 복잡한 환경에서도 안정적인 영상 촬영과 3D 재구성을 실현하며, 하이라이트 장면 자동 추출에 활용

## 20위: Learning Human-Object Interaction for 3D Human Pose Estimation from LiDAR Point Clouds

- arXiv: http://arxiv.org/abs/2603.16343v1
- PDF: https://arxiv.org/pdf/2603.16343v1
- 발행일: 2026-03-17
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:75 + bonus:+3)
- 플래그: 코드 공개

**개요**
Understanding humans from LiDAR point clouds is one of the most critical tasks in autonomous driving due to its close relationships with pedestrian safety, yet it remains challenging in the presence of diverse human-object interactions and cluttered backgrounds. Nevertheless, existing methods largely overlook the potential of leveraging human-object interactions to build robust 3D human pose estimation frameworks. There are two major challenges that motivate the incorporation of human-object interaction. First, human-object interactions introduce spatial ambiguity between human and object points, which often leads to erroneous 3D human keypoint predictions in interaction regions. Second, there exists severe class imbalance in the number of points between interacting and non-interacting body parts, with the interaction-frequent regions such as hand and foot being sparsely observed in LiDAR data. To address these challenges, we propose a Human-Object Interaction Learning (HOIL) framework for robust 3D human pose estimation from LiDAR point clouds. To mitigate the spatial ambiguity issue, we present human-object interaction-aware contrastive learning (HOICL) that effectively enhances feature discrimination between human and object points, particularly in interaction regions. To alleviate the class imbalance issue, we introduce contact-aware part-guided pooling (CPPool) that adaptively reallocates representational capacity by compressing overrepresented points while preserving informative points from interacting body parts. In addition, we present an optional contact-based temporal refinement that refines erroneous per-frame keypoint estimates using contact cues over time. As a result, our HOIL effectively leverages human-object interaction to resolve spatial ambiguity and class imbalance in interaction regions. Codes will be released.

**선정 근거**
인간-객체 상호작용 학습 기술은 스포츠에서의 선수 자세 분석과 동작 패턴 인식에 적용 가능하며, 다양한 상황에서의 정확한 추적이 필요

**활용 인사이트**
리더 포인트 클라우드 기반의 인간 자세 추정을 스포츠 분석에 적용하여 선수의 동작 패턴과 상호작용을 분석하고, 개인별 기술 평가에 활용

## 21위: When Thinking Hurts: Mitigating Visual Forgetting in Video Reasoning via Frame Repetition

- arXiv: http://arxiv.org/abs/2603.16256v1
- PDF: https://arxiv.org/pdf/2603.16256v1
- 발행일: 2026-03-17
- 카테고리: cs.CV
- 점수: final 80.8 (llm_adjusted:76 = base:76 + bonus:+0)

**개요**
Recently, Multimodal Large Language Models (MLLMs) have demonstrated significant potential in complex visual tasks through the integration of Chain-of-Thought (CoT) reasoning. However, in Video Question Answering, extended thinking processes do not consistently yield performance gains and may even lead to degradation due to ``visual anchor drifting'', where models increasingly rely on self-generated text, sidelining visual inputs and causing hallucinations. While existing mitigations typically introduce specific mechanisms for the model to re-attend to visual inputs during inference, these approaches often incur prohibitive training costs and suffer from poor generalizability across different architectures. To address this, we propose FrameRepeat, an automated enhancement framework which features a lightweight repeat scoring module that enables Video-LLMs to autonomously identify which frames should be reinforced. We introduce a novel training strategy, Add-One-In (AOI), that uses MLLM output probabilities to generate supervision signals representing repeat gain. This can be used to train a frame scoring network, which guides the frame repetition behavior. Experimental results across multiple models and datasets demonstrate that FrameRepeat is both effective and generalizable in strengthening important visual cues during the reasoning process.

**선정 근거**
비디오 추론 시 중요한 프레임을 식별하는 기술로 스포츠 하이라이트 장면 자동 식별에 직접 적용 가능합니다.

**활용 인사이트**
FrameRepeat 프레임 반복 점수 모듈을 스포츠 장면에 적용하여 자동으로 하이라이트 장면을 강화하고 정확도를 높일 수 있습니다.

## 22위: TinyGLASS: Real-Time Self-Supervised In-Sensor Anomaly Detection

- arXiv: http://arxiv.org/abs/2603.16451v1
- PDF: https://arxiv.org/pdf/2603.16451v1
- 발행일: 2026-03-17
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:65 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Anomaly detection plays a key role in industrial quality control, where defects must be identified despite the scarcity of labeled faulty samples. Recent self-supervised approaches, such as GLASS, learn normal visual patterns using only defect-free data and have shown strong performance on industrial benchmarks. However, their computational requirements limit deployment on resource-constrained edge platforms.   This work introduces TinyGLASS, a lightweight adaptation of the GLASS framework designed for real-time in-sensor anomaly detection on the Sony IMX500 intelligent vision sensor. The proposed architecture replaces the original WideResNet-50 backbone with a compact ResNet-18 and introduces deployment-oriented modifications that enable static graph tracing and INT8 quantization using Sony's Model Compression Toolkit.   In addition to evaluating performance on the MVTec-AD benchmark, we investigate robustness to contaminated training data and introduce a custom industrial dataset, named MMS Dataset, for cross-device evaluation. Experimental results show that TinyGLASS achieves 8.7x parameter compression while maintaining competitive detection performance, reaching 94.2% image-level AUROC on MVTec-AD and operating at 20 FPS within the 8 MB memory constraints of the IMX500 platform.   System profiling demonstrates low power consumption (4.0 mJ per inference), real-time end-to-end latency (20 FPS), and high energy efficiency (470 GMAC/J). Furthermore, the model maintains stable performance under moderate levels of training data contamination.

**선정 근거**
엣지 디바이스에서 실시간 이상 탐지 기술은 프로젝트의 rk3588 기반 엣지 디바이스 개발에 기술적 참고가 됩니다.

**활용 인사이트**
경량화된 ResNet-18 아키텍처와 INT8 양자화 기술을 적용하여 저전력 고성능 스포츠 촬영 시스템을 구축할 수 있습니다.

## 23위: Large Reward Models: Generalizable Online Robot Reward Generation with Vision-Language Models

- arXiv: http://arxiv.org/abs/2603.16065v1
- PDF: https://arxiv.org/pdf/2603.16065v1
- 발행일: 2026-03-17
- 카테고리: cs.RO, cs.AI
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Reinforcement Learning (RL) has shown great potential in refining robotic manipulation policies, yet its efficacy remains strongly bottlenecked by the difficulty of designing generalizable reward functions. In this paper, we propose a framework for online policy refinement by adapting foundation VLMs into online reward generators. We develop a robust, scalable reward model based on a state-of-the-art VLM, trained on a large-scale, multi-source dataset encompassing real-world robot trajectories, human-object interactions, and diverse simulated environments. Unlike prior approaches that evaluate entire trajectories post-hoc, our method leverages the VLM to formulate a multifaceted reward signal comprising process, completion, and temporal contrastive rewards based on current visual observations. Initializing with a base policy trained via Imitation Learning (IL), we employ these VLM rewards to guide the model to correct sub-optimal behaviors in a closed-loop manner. We evaluate our framework on challenging long-horizon manipulation benchmarks requiring sequential execution and precise control. Crucially, our reward model operates in a purely zero-shot manner within these test environments. Experimental results demonstrate that our method significantly improves the success rate of the initial IL policy within just 30 RL iterations, demonstrating remarkable sample efficiency. This empirical evidence highlights that VLM-generated signals can provide reliable feedback to resolve execution errors, effectively eliminating the need for manual reward engineering and facilitating efficient online refinement for robot learning.

**선정 근거**
비전-언어 모델을 활용한 보상 생성 시스템은 스포츠 자세 및 동작 분석에 적용 가능한 핵심 기술입니다.

**활용 인사이트**
VLM 기반 보상 모델을 스포츠 동작 분석에 적용하여 정밀한 자세 교정 및 전략 개선을 위한 실시간 피드백 시스템을 구축할 수 있습니다.

## 24위: Demystifing Video Reasoning

- arXiv: http://arxiv.org/abs/2603.16870v1
- PDF: https://arxiv.org/pdf/2603.16870v1
- 발행일: 2026-03-17
- 카테고리: cs.CV, cs.AI
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Recent advances in video generation have revealed an unexpected phenomenon: diffusion-based video models exhibit non-trivial reasoning capabilities. Prior work attributes this to a Chain-of-Frames (CoF) mechanism, where reasoning is assumed to unfold sequentially across video frames. In this work, we challenge this assumption and uncover a fundamentally different mechanism. We show that reasoning in video models instead primarily emerges along the diffusion denoising steps. Through qualitative analysis and targeted probing experiments, we find that models explore multiple candidate solutions in early denoising steps and progressively converge to a final answer, a process we term Chain-of-Steps (CoS). Beyond this core mechanism, we identify several emergent reasoning behaviors critical to model performance: (1) working memory, enabling persistent reference; (2) self-correction and enhancement, allowing recovery from incorrect intermediate solutions; and (3) perception before action, where early steps establish semantic grounding and later steps perform structured manipulation. During a diffusion step, we further uncover self-evolved functional specialization within Diffusion Transformers, where early layers encode dense perceptual structure, middle layers execute reasoning, and later layers consolidate latent representations. Motivated by these insights, we present a simple training-free strategy as a proof-of-concept, demonstrating how reasoning can be improved by ensembling latent trajectories from identical models with different random seeds. Overall, our work provides a systematic understanding of how reasoning emerges in video generation models, offering a foundation to guide future research in better exploiting the inherent reasoning dynamics of video models as a new substrate for intelligence.

**선정 근거**
비디오 모델의 추론 메커니즘 이해는 스포츠 하이라이트 감지 및 전략 분석에 중요한 기술적 기반을 제공합니다.

**활용 인사이트**
Chain-of-Steps(CoS) 추론 원리를 적용하여 다중 후보 솔루션 탐색 및 점진적 수렴 과정을 스포츠 장면 분석에 활용할 수 있습니다.

## 25위: Optimal uncertainty bounds for multivariate kernel regression under bounded noise: A Gaussian process-based dual function

- arXiv: http://arxiv.org/abs/2603.16481v1
- PDF: https://arxiv.org/pdf/2603.16481v1
- 발행일: 2026-03-17
- 카테고리: cs.LG, eess.SY, math.OC
- 점수: final 80.0 (llm_adjusted:75 = base:70 + bonus:+5)
- 플래그: 실시간

**개요**
Non-conservative uncertainty bounds are essential for making reliable predictions about latent functions from noisy data--and thus, a key enabler for safe learning-based control. In this domain, kernel methods such as Gaussian process regression are established techniques, thanks to their inherent uncertainty quantification mechanism. Still, existing bounds either pose strong assumptions on the underlying noise distribution, are conservative, do not scale well in the multi-output case, or are difficult to integrate into downstream tasks. This paper addresses these limitations by presenting a tight, distribution-free bound for multi-output kernel-based estimates. It is obtained through an unconstrained, duality-based formulation, which shares the same structure of classic Gaussian process confidence bounds and can thus be straightforwardly integrated into downstream optimization pipelines. We show that the proposed bound generalizes many existing results and illustrate its application using an example inspired by quadrotor dynamics learning.

**선정 근거**
가우시안 프로세스 불확정성 정량화 기술은 스포츠 동작 및 전략 분석의 신뢰성을 높이는 데 직접 적용 가능합니다.

**활용 인사이트**
다변량 커널 회귀 기반 불확정성 경계 계산을 스포츠 동작 분석에 적용하여 예측의 신뢰도를 평가하고 정밀도를 향상시킬 수 있습니다.

## 26위: Learning Whole-Body Control for a Salamander Robot

- arXiv: http://arxiv.org/abs/2603.16683v1
- PDF: https://arxiv.org/pdf/2603.16683v1
- 발행일: 2026-03-17
- 카테고리: cs.RO
- 점수: final 80.0 (llm_adjusted:75 = base:70 + bonus:+5)
- 플래그: 실시간

**개요**
Amphibious legged robots inspired by salamanders are promising in applications in complex amphibious environments. However, despite the significant success of training controllers that achieve diverse locomotion behaviors in conventional quadrupedal robots, most salamander robots relied on central-pattern-generator (CPG)-based and model-based coordination strategies for locomotion control. Learning unified joint-level whole-body control that reliably transfers from simulation to highly articulated physical salamander robots remains relatively underexplored. In addition, few legged robots have tried learning-based controllers in amphibious environments. In this work, we employ Reinforcement Learning to map proprioceptive observations and commanded velocities to joint-level actions, allowing coordinated locomotor behaviors to emerge. To deploy these policies on hardware, we adopt a system-level real-to-sim matching and sim-to-real transfer strategy. The learned controller achieves stable and coordinated walking on both flat and uneven terrains in the real world. Beyond terrestrial locomotion, the framework enables transitions between walking and swimming in simulation, highlighting a phenomenon of interest for understanding locomotion across distinct physical modes.

**선정 근거**
로봇 동작 학습 기술은 스포츠 동작 분석에 직접 적용 가능

**활용 인사이트**
강화학습으로 학습된 제어 알고리즘으로 선수 동작 패턴 분석 및 최적 동작 추천

## 27위: Early Pre-Stroke Detection via Wearable IMU-Based Gait Variability and Postural Drift Analysis

- arXiv: http://arxiv.org/abs/2603.16178v1
- PDF: https://arxiv.org/pdf/2603.16178v1
- 발행일: 2026-03-17
- 카테고리: q-bio.NC
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Early identification of individuals at risk of stroke remains a major clinical challenge, as prodromal motor im- pairments are often subtle and transient. In this pilot study, a wearable sensor-based framework is proposed for early pre- stroke risk screening using a single inertial measurement unit mounted on the sacral region to capture pelvic motion during gait and standing tasks. The pelvis is treated as a biomechanical proxy for global motor control, enabling the quantification of gait variability and postural drift as digital biomarkers of neurological instability. Raw inertial signals are processed using a sensor fusion pipeline to estimate pelvic kinematics, from which variability and nonlinear dynamic features are extracted. These features are subsequently used to train a machine learning model for risk stratification across control, pre-stroke, and stroke groups. Progressive increases in pelvic angular variability and postural instability are observed from the control to stroke groups, with the pre-stroke cohort exhibiting intermediate char- acteristics. As a proof-of-concept investigation, the proposed framework demonstrates the feasibility of using a minimal wearable configuration to capture pelvic micro-instability associ- ated with early cerebrovascular motor adaptation. The classifier achieves a macro-averaged area under the curve of 0.785, indicating preliminary discriminative capability between risk categories. While not intended for clinical diagnosis, the proposed approach provides a low-cost, non-invasive, and scalable solution for continuous community-level screening, supporting proactive intervention prior to the onset of major stroke events.

**선정 근거**
웨어러블 IMU 센서를 이용한 보행 변이도 분석 기술은 스포츠 선수의 동작 패턴을 분석하고 개인별 특징을 파악하는 데 직접적으로 활용될 수 있습니다.

**활용 인사이트**
선수의 움직임 데이터를 실시간으로 분석하여 개인별 최적의 동작 패턴을 추출하고, 이를 바탕으로 맞춤형 훈련 프로그램과 하이라이트 영상을 생성할 수 있습니다.

## 28위: KidsNanny: A Two-Stage Multimodal Content Moderation Pipeline Integrating Visual Classification, Object Detection, OCR, and Contextual Reasoning for Child Safety

- arXiv: http://arxiv.org/abs/2603.16181v1
- PDF: https://arxiv.org/pdf/2603.16181v1
- 발행일: 2026-03-17
- 카테고리: cs.CV, cs.CR
- 점수: final 74.4 (llm_adjusted:68 = base:58 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
We present KidsNanny, a two-stage multimodal content moderation architecture for child safety. Stage 1 combines a vision transformer (ViT) with an object detector for visual screening (11.7 ms); outputs are routed as text not raw pixels to Stage 2, which applies OCR and a text based 7B language model for contextual reasoning (120 ms total pipeline). We evaluate on the UnsafeBench Sexual category (1,054 images) under two regimes: vision-only, isolating Stage 1, and multimodal, evaluating the full Stage 1+2 pipeline. Stage 1 achieves 80.27% accuracy and 85.39% F1 at 11.7 ms; vision-only baselines range from 59.01% to 77.04% accuracy. The full pipeline achieves 81.40% accuracy and 86.16% F1 at 120 ms, compared to ShieldGemma-2 (64.80% accuracy, 1,136 ms) and LlavaGuard (80.36% accuracy, 4,138 ms). To evaluate text-awareness, we filter two subsets: a text+visual subset (257 images) and a text-only subset (44 images where safety depends primarily on embedded text). On text-only images, KidsNanny achieves 100% recall (25/25 positives; small sample) and 75.76% precision; ShieldGemma-2 achieves 84% recall and 60% precision at 1,136 ms. Results suggest that dedicated OCR-based reasoning may offer recall-precision advantages on text-embedded threats at lower latency, though the small text-only subset limits generalizability. By documenting this architecture and evaluation methodology, we aim to contribute to the broader research effort on efficient multimodal content moderation for child safety.

**선정 근거**
Multimodal processing approach conceptually similar but application domain differs significantly

## 29위: On the Transfer of Collinearity to Computer Vision

- arXiv: http://arxiv.org/abs/2603.16592v1
- PDF: https://arxiv.org/pdf/2603.16592v1
- 발행일: 2026-03-17
- 카테고리: cs.CV
- 점수: final 72.0 (llm_adjusted:65 = base:65 + bonus:+0)

**개요**
Collinearity is a visual perception phenomenon in the human brain that amplifies spatially aligned edges arranged along a straight line. However, it is vague for which purpose humans might have this principle in the real-world, and its utilization in computer vision and engineering applications even is a largely unexplored field. In this work, our goal is to transfer the collinearity principle to computer vision, and we explore the potential usages of this novel principle for computer vision applications. We developed a prototype model to exemplify the principle, then tested it systematically, and benchmarked it in the context of four use cases. Our cases are selected to spawn a broad range of potential applications and scenarios: sketching the combination of collinearity with deep learning (case I and II), using collinearity with saliency models (case II), and as a feature detector (case I). In the first use case, we found that collinearity is able to improve the fault detection of wafers and obtain a performance increase by a factor 1.24 via collinearity (decrease of the error rate from 6.5% to 5.26%). In the second use case, we test the defect recognition in nanotechnology materials and achieve a performance increase by 3.2x via collinearity (deep learning, error from 21.65% to 6.64%), and also explore saliency models. As third experiment, we cover occlusions; while as fourth experiment, we test ImageNet and observe that it might not be very beneficial for ImageNet. Therefore, we can assemble a list of scenarios for which collinearity is beneficial (wafers, nanotechnology, occlusions), and for what is not beneficial (ImageNet). Hence, we infer collinearity might be suitable for industry applications as it helps if the image structures of interest are man-made because they often consist of lines. Our work provides another tool for CV, hope to capture the power of human processing.

**선정 근거**
공간 정렬된 가장자리를 분석하는 공선성 원리는 스포츠 동작 분석에 직접 적용 가능하며, 웨어러블 센서와 결합하여 선수의 움직임 패턴을 더 정확하게 분석할 수 있습니다.

**활용 인사이트**
경기 중 움직임 패턴을 분석하여 공선성 원리를 적용하면, 특정 동작의 오류를 감지하고 개인별 하이라이트 장면을 자동으로 식별하여 편집 효율을 크게 향상시킬 수 있습니다.

## 30위: Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights

- arXiv: http://arxiv.org/abs/2603.16817v1
- PDF: https://arxiv.org/pdf/2603.16817v1
- 발행일: 2026-03-17
- 카테고리: cs.AI, cs.CL, cs.LG
- 점수: final 70.4 (llm_adjusted:63 = base:58 + bonus:+5)
- 플래그: 엣지

**개요**
Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged as potential ways to address this limitation. While RAG aims to ground responses in retrieved evidence, it provides no statistical guarantee that the final output is correct. Conformal factuality filtering offers distribution-free statistical reliability by scoring and filtering atomic claims using a threshold calibrated on held-out data, however, the informativeness of the final output is not guaranteed. We systematically analyze the reliability and usefulness of conformal factuality for RAG-based LLMs across generation, scoring, calibration, robustness, and efficiency. We propose novel informativeness-aware metrics that better reflect task utility under conformal filtering. Across three benchmarks and multiple model families, we find that (i) conformal filtering suffers from low usefulness at high factuality levels due to vacuous outputs, (ii) conformal factuality guarantee is not robust to distribution shifts and distractors, highlighting the limitation that requires calibration data to closely match deployment conditions, and (iii) lightweight entailment-based verifiers match or outperform LLM-based model confidence scorers while requiring over $100\times$ fewer FLOPs. Overall, our results expose factuality-informativeness trade-offs and fragility of conformal filtering framework under distribution shifts and distractors, highlighting the need for new approaches for reliability with robustness and usefulness as key metrics, and provide actionable guidance for building RAG pipelines that are both reliable and computationally efficient.

**선정 근거**
Some relevance to AI systems and computational efficiency

## 31위: 360° Image Perception with MLLMs: A Comprehensive Benchmark and a Training-Free Method

- arXiv: http://arxiv.org/abs/2603.16179v1 | 2026-03-17 | final 68.8

Multimodal Large Language Models (MLLMs) have shown impressive abilities in understanding and reasoning over conventional images. However, their perception of 360° images remains largely underexplored.

-> 360도 이미지 인식 기술은 특정 스포츠 촬영 시나리오에 간접적으로 적용 가능

## 32위: MLLM-based Textual Explanations for Face Comparison

- arXiv: http://arxiv.org/abs/2603.16629v1 | 2026-03-17 | final 68.8

Multimodal Large Language Models (MLLMs) have recently been proposed as a means to generate natural-language explanations for face recognition decisions. While such explanations facilitate human interpretability, their reliability on unconstrained face images remains underexplored.

-> Face recognition techniques could partially support athlete identification in sports videos

## 33위: Visual Prompt Discovery via Semantic Exploration

- arXiv: http://arxiv.org/abs/2603.16250v1 | 2026-03-17 | final 68.0

LVLMs encounter significant challenges in image understanding and visual reasoning, leading to critical perception failures. Visual prompts, which incorporate image manipulation code, have shown promising potential in mitigating these issues.

-> 이미지 이해와 시각적 추론에 초점을 맞춰 스포츠 동작 분석에 간접적으로 적용 가능할 수 있으나, 스포츠 특화 솔루션은 아닙니다.

## 34위: pADAM: A Plug-and-Play All-in-One Diffusion Architecture for Multi-Physics Learning

- arXiv: http://arxiv.org/abs/2603.16757v1 | 2026-03-17 | final 68.0

Generalizing across disparate physical laws remains a fundamental challenge for artificial intelligence in science. Existing deep-learning solvers are largely confined to single-equation settings, limiting transfer across physical regimes and inference tasks.

-> 다중 물리학 학습을 위한 확산 아키텍처는 비디오 처리에 간접적으로 적용 가능할 수 있음

## 35위: Spectral Property-Driven Data Augmentation for Hyperspectral Single-Source Domain Generalization

- arXiv: http://arxiv.org/abs/2603.16662v1 | 2026-03-17 | final 68.0

While hyperspectral images (HSI) benefit from numerous spectral channels that provide rich information for classification, the increased dimensionality and sensor variability make them more sensitive to distributional discrepancies across domains, which in turn can affect classification performance. To tackle this issue, hyperspectral single-source domain generalization (SDG) typically employs data augmentation to simulate potential domain shifts and enhance model robustness under the condition of single-source domain training data availability.

-> 분광 데이터 증강 기술은 스포츠 분석을 위한 비디오 처리에 적용 가능할 수 있음

## 36위: HGP-Mamba: Integrating Histology and Generated Protein Features for Mamba-based Multimodal Survival Risk Prediction

- arXiv: http://arxiv.org/abs/2603.16421v1 | 2026-03-17 | final 66.4

Recent advances in multimodal learning have significantly improved cancer survival risk prediction. However, the joint prognostic potential of protein markers and histopathology images remains underexplored, largely due to the high cost and limited availability of protein expression profiling.

-> Multimodal learning techniques could be adapted for sports video analysis

## 37위: ExpressMind: A Multimodal Pretrained Large Language Model for Expressway Operation

- arXiv: http://arxiv.org/abs/2603.16495v1 | 2026-03-17 | final 66.4

The current expressway operation relies on rule-based and isolated models, which limits the ability to jointly analyze knowledge across different systems. Meanwhile, Large Language Models (LLMs) are increasingly applied in intelligent transportation, advancing traffic models from algorithmic to cognitive intelligence.

-> Multimodal video analysis techniques from traffic domain could be adapted for sports analysis

## 38위: Micro-AU CLIP: Fine-Grained Contrastive Learning from Local Independence to Global Dependency for Micro-Expression Action Unit Detection

- arXiv: http://arxiv.org/abs/2603.16302v1 | 2026-03-17 | final 66.4

Micro-expression (ME) action units (Micro-AUs) provide objective clues for fine-grained genuine emotion analysis. Most existing Micro-AU detection methods learn AU features from the whole facial image/video, which conflicts with the inherent locality of AU, resulting in insufficient perception of AU regions.

-> 미세 표정 분석 기술은 스포츠 동작 분석에 간접적으로 적용 가능할 수 있으나 스포츠 특화되지 않음

## 39위: Speak, Segment, Track, Navigate: An Interactive System for Video-Guided Skull-Base Surgery

- arXiv: http://arxiv.org/abs/2603.16024v1 | 2026-03-17 | final 66.0

We introduce a speech-guided embodied agent framework for video-guided skull base surgery that dynamically executes perception and image-guidance tasks in response to surgeon queries. The proposed system integrates natural language interaction with real-time visual perception directly on live intraoperative video streams, thereby enabling surgeons to request computational assistance without disengaging from operative tasks.

-> 수술 영상 분석 기술로 스포츠 촬영과 간접적 연관성 있음

## 40위: FlowComposer: Composable Flows for Compositional Zero-Shot Learning

- arXiv: http://arxiv.org/abs/2603.16641v1 | 2026-03-17 | final 64.0

Compositional zero-shot learning (CZSL) aims to recognize unseen attribute-object compositions by recombining primitives learned from seen pairs. Recent CZSL methods built on vision-language models (VLMs) typically adopt parameter-efficient fine-tuning (PEFT).

-> Compositional learning framework potentially applicable to sports movement analysis but not directly related

## 41위: Exploring different approaches to customize language models for domain-specific text-to-code generation

- arXiv: http://arxiv.org/abs/2603.16526v1 | 2026-03-17 | final 62.4

Large language models (LLMs) have demonstrated strong capabilities in generating executable code from natural language descriptions. However, general-purpose models often struggle in specialized programming contexts where domain-specific libraries, APIs, or conventions must be used.

-> 컴퓨터 비전 모델 커스터마이징, 스포츠 비디오 분석에 잠재적 적용 가능

## 42위: Reliable Reasoning in SVG-LLMs via Multi-Task Multi-Reward Reinforcement Learning

- arXiv: http://arxiv.org/abs/2603.16189v1 | 2026-03-17 | final 61.6

With the rapid advancement of vision-language models, an increasing number of studies have explored their potential for SVG generation tasks. Although existing approaches improve performance by constructing large-scale SVG datasets and introducing SVG-specific tokens, they still suffer from limited generalization, redundant paths in code outputs, and a lack of explicit reasoning.

-> 비전-언어 모델은 스포츠 분석에 간접적으로 관련될 수 있으나 SVG 생성에 특화됨

## 43위: GAP-MLLM: Geometry-Aligned Pre-training for Activating 3D Spatial Perception in Multimodal Large Language Models

- arXiv: http://arxiv.org/abs/2603.16461v1 | 2026-03-17 | final 60.0

Multimodal Large Language Models (MLLMs) demonstrate exceptional semantic reasoning but struggle with 3D spatial perception when restricted to pure RGB inputs. Despite leveraging implicit geometric priors from 3D reconstruction models, image-based methods still exhibit a notable performance gap compared to methods using explicit 3D data.

-> 다중모달 AI를 위한 3D 공간 지각, 스포츠 동작 분석에 잠재적 적용 가능

## 44위: ETM2: Empowering Traditional Memory Bandwidth Regulation using ETM

- arXiv: http://arxiv.org/abs/2603.16490v1 | 2026-03-17 | final 60.0

The Embedded Trace Macrocell (ETM) is a standard component of Arm's CoreSight architecture, present in a wide range of platforms and primarily designed for tracing and debugging. In this work, we demonstrate that it can be repurposed to implement a novel hardware-assisted memory bandwidth regulator, providing a portable and effective solution to mitigate memory interference in real-time multicore systems.

-> Memory bandwidth regulation for multicore systems with potential optimization for rk3588 edge device

## 45위: Fine-Grained Network Traffic Classification with Contextual QoS Profiling

- arXiv: http://arxiv.org/abs/2603.16748v1 | 2026-03-17 | final 56.0

Accurate network traffic classification is vital for managing modern applications with strict Quality of Service (QoS) demands, such as edge computing, real-time XR, and autonomous systems. While recent advances in application-level classification show high accuracy, they often miss fine-grained in-app QoS variations critical for service differentiation.

-> 네트워크 트래픽 분류는 스포츠 영상 분석과 직접적인 관련이 없지만, 엣지 컴퓨팅 언급이 프로젝트의 하드웨어 측면과 약간의 연관성이 있습니다.

## 46위: REFORGE: Multi-modal Attacks Reveal Vulnerable Concept Unlearning in Image Generation Models

- arXiv: http://arxiv.org/abs/2603.16576v1 | 2026-03-17 | final 50.4

Recent progress in image generation models (IGMs) enables high-fidelity content creation but also amplifies risks, including the reproduction of copyrighted content and the generation of offensive content. Image Generation Model Unlearning (IGMU) mitigates these risks by removing harmful concepts without full retraining.

-> 이미지 생성 모델을 다루지만 보안 테스트에 초점을 맞춰 스포츠 영상 분석이나 하이라이트 생성과 직접적인 연관성이 부족합니다.

---

## 다시 보기

### Mobile-GS: Real-time Gaussian Splatting for Mobile Devices (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11531v1
- 점수: final 100.0

3D Gaussian Splatting (3DGS) has emerged as a powerful representation for high-quality rendering across a wide range of applications.However, its high computational demands and large storage costs pose significant challenges for deployment on mobile devices. In this work, we propose a mobile-tailored real-time Gaussian Splatting method, dubbed Mobile-GS, enabling efficient inference of Gaussian Splatting on edge devices. Specifically, we first identify alpha blending as the primary computational bottleneck, since it relies on the time-consuming Gaussian depth sorting process. To solve this issue, we propose a depth-aware order-independent rendering scheme that eliminates the need for sorting, thereby substantially accelerating rendering. Although this order-independent rendering improves rendering speed, it may introduce transparency artifacts in regions with overlapping geometry due to the scarcity of rendering order. To address this problem, we propose a neural view-dependent enhancement strategy, enabling more accurate modeling of view-dependent effects conditioned on viewing direction, 3D Gaussian geometry, and appearance attributes. In this way, Mobile-GS can achieve both high-quality and real-time rendering. Furthermore, to facilitate deployment on memory-constrained mobile platforms, we also introduce first-order spherical harmonics distillation, a neural vector quantization technique, and a contribution-based pruning strategy to reduce the number of Gaussian primitives and compress the 3D Gaussian representation with the assistance of neural networks. Extensive experiments demonstrate that our proposed Mobile-GS achieves real-time rendering and compact model size while preserving high visual quality, making it well-suited for mobile applications.

-> 모바일 기용 실시간 가우시안 스플래팅으로 엣지 디바이스에서 고품질 3D 재구현 가능

### HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.12222v1
- 점수: final 100.0

Vision Transformers require significant computational resources and memory bandwidth, severely limiting their deployment on edge devices. While recent structured pruning methods successfully reduce theoretical FLOPs, they typically operate at a single structural granularity and rely on complex, multi-stage pipelines with post-hoc thresholding to satisfy sparsity budgets. In this paper, we propose Hierarchical Auto-Pruning (HiAP), a continuous relaxation framework that discovers optimal sub-networks in a single end-to-end training phase without requiring manual importance heuristics or predefined per-layer sparsity targets. HiAP introduces stochastic Gumbel-Sigmoid gates at multiple granularities: macro-gates to prune entire attention heads and FFN blocks, and micro-gates to selectively prune intra-head dimensions and FFN neurons. By optimizing both levels simultaneously, HiAP addresses both the memory-bound overhead of loading large matrices and the compute-bound mathematical operations. HiAP naturally converges to stable sub-networks using a loss function that incorporates both structural feasibility penalties and analytical FLOPs. Extensive experiments on ImageNet demonstrate that HiAP organically discovers highly efficient architectures, and achieves a competitive accuracy-efficiency Pareto frontier for models like DeiT-Small, matching the performance of sophisticated multi-stage methods while significantly simplifying the deployment pipeline.

-> 엣지 디바이스에서 AI 모델을 효율적으로 실행하여 실시간 처리 성능 극대화

### MONET: Modeling and Optimization of neural NEtwork Training from Edge to Data Centers (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15002v1
- 점수: final 100.0

While hardware-software co-design has significantly improved the efficiency of neural network inference, modeling the training phase remains a critical yet underexplored challenge. Training workloads impose distinct constraints, particularly regarding memory footprint and backpropagation complexity, which existing inference-focused tools fail to capture. This paper introduces MONET, a framework designed to model the training of neural networks on heterogeneous dataflow accelerators. MONET builds upon Stream, an experimentally verified framework that that models the inference of neural networks on heterogeneous dataflow accelerators with layer fusion. Using MONET, we explore the design space of ResNet-18 and a small GPT-2, demonstrating the framework's capability to model training workflows and find better hardware architectures. We then further examine problems that become more complex in neural network training due to the larger design space, such as determining the best layer-fusion configuration. Additionally, we use our framework to find interesting trade-offs in activation checkpointing, with the help of a genetic algorithm. Our findings highlight the importance of a holistic approach to hardware-software co-design for scalable and efficient deep learning deployment.

-> 에지 디바이스를 위한 신경망 최적화 기술이 rk3588 기반 AI 촬영 장비의 핵심 기술로 직접 적용 가능

### Detect Anything in Real Time: From Single-Prompt Segmentation to Multi-Class Detection (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11441v1
- 점수: final 100.0

Recent advances in vision-language modeling have produced promptable detection and segmentation systems that accept arbitrary natural language queries at inference time. Among these, SAM3 achieves state-of-the-art accuracy by combining a ViT-H/14 backbone with cross-modal transformer decoding and learned object queries. However, SAM3 processes a single text prompt per forward pass. Detecting N categories requires N independent executions, each dominated by the 439M-parameter backbone. We present Detect Anything in Real Time (DART), a training-free framework that converts SAM3 into a real-time multi-class detector by exploiting a structural invariant: the visual backbone is class-agnostic, producing image features independent of the text prompt. This allows the backbone computation to be shared between all classes, reducing its cost from O(N) to O(1). Combined with batched multi-class decoding, detection-only inference, and TensorRT FP16 deployment, these optimizations yield 5.6x cumulative speedup at 3 classes, scaling to 25x at 80 classes, without modifying any model weight. On COCO val2017 (5,000 images, 80 classes), DART achieves 55.8 AP at 15.8 FPS (4 classes, 1008x1008) on a single RTX 4080, surpassing purpose-built open-vocabulary detectors trained on millions of box annotations. For extreme latency targets, adapter distillation with a frozen encoder-decoder achieves 38.7 AP with a 13.9 ms backbone. Code and models are available at https://github.com/mkturkcan/DART.

-> 실시간 다중 클래스 검색 기술로 스포츠 장면에서 선수와 객체를 즉시 식별하여 하이라이트 자동 생성 가능

### Multi-Objective Load Balancing for Heterogeneous Edge-Based Object Detection Systems (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15400v1
- 점수: final 98.4

The rapid proliferation of the Internet of Things (IoT) and smart applications has led to a surge in data generated by distributed sensing devices. Edge computing is a mainstream approach to managing this data by pushing computation closer to the data source, typically onto resource-constrained devices such as single-board computers (SBCs). In such environments, the unavoidable heterogeneity of hardware and software makes effective load balancing particularly challenging. In this paper, we propose a multi-objective load balancing method tailored to heterogeneous, edge-based object detection systems. We study a setting in which multiple device-model pairs expose distinct accuracy, latency, and energy profiles, while both request intensity and scene complexity fluctuate over time. To handle this dynamically varying environment, our approach uses a two-stage decision mechanism: it first performs accuracy-aware filtering to identify suitable device-model candidates that provide accuracy within the acceptable range, and then applies a weighted-sum scoring function over expected latency and energy consumption to select the final execution target. We evaluate the proposed load balancer through extensive experiments on real-world datasets, comparing against widely used baseline strategies. The results indicate that the proposed multi-objective load balancing method halves energy consumption and achieves an 80% reduction in end-to-end latency, while incurring only a modest, up to 10%, decrease in detection accuracy relative to an accuracy-centric baseline.

-> 이기종 엣지 기반 객체 탐지 시스템의 다목적 부하 분산이 스포츠 촬영 시스템 자원 제약에 직접 적용 가능

### Face-to-Face: A Video Dataset for Multi-Person Interaction Modeling (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.14794v1
- 점수: final 96.0

Modeling the reactive tempo of human conversation remains difficult because most audio-visual datasets portray isolated speakers delivering short monologues. We introduce \textbf{Face-to-Face with Jimmy Fallon (F2F-JF)}, a 70-hour, 14k-clip dataset of two-person talk-show exchanges that preserves the sequential dependency between a guest turn and the host's response. A semi-automatic pipeline combines multi-person tracking, speech diarization, and lightweight human verification to extract temporally aligned host/guest tracks with tight crops and metadata that are ready for downstream modeling. We showcase the dataset with a reactive, speech-driven digital avatar task in which the host video during $[t_1,t_2]$ is generated from their audio plus the guest's preceding video during $[t_0,t_1]$. Conditioning a MultiTalk-style diffusion model on this cross-person visual context yields small but consistent Emotion-FID and FVD gains while preserving lip-sync quality relative to an audio-only baseline. The dataset, preprocessing recipe, and baseline together provide an end-to-end blueprint for studying dyadic, sequential behavior, which we expand upon throughout the paper. Dataset and code will be made publicly available.

-> 다중 인물 추적 기술이 스포츠 촬영에서 여러 선수 동시 추적에 적용 가능

### Low-light Image Enhancement with Retinex Decomposition in Latent Space (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15131v1
- 점수: final 96.0

Retinex theory provides a principled foundation for low-light image enhancement, inspiring numerous learning-based methods that integrate its principles. However, existing methods exhibits limitations in accurately decomposing reflectance and illumination components. To address this, we propose a Retinex-Guided Transformer~(RGT) model, which is a two-stage model consisting of decomposition and enhancement phases. First, we propose a latent space decomposition strategy to separate reflectance and illumination components. By incorporating the log transformation and 1-pixel offset, we convert the intrinsically multiplicative relationship into an additive formulation, enhancing decomposition stability and precision. Subsequently, we construct a U-shaped component refiner incorporating the proposed guidance fusion transformer block. The component refiner refines reflectance component to preserve texture details and optimize illumination distribution, effectively transforming low-light inputs to normal-light counterparts. Experimental evaluations across four benchmark datasets validate that our method achieves competitive performance in low-light enhancement and a more stable training process.

-> 잔광 이미지 보정 기술이 다양한 조건에서 스포츠 영상 품질 향상에 적합

### Lightweight User-Personalization Method for Closed Split Computing (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.14958v1
- 점수: final 96.0

Split Computing enables collaborative inference between edge devices and the cloud by partitioning a deep neural network into an edge-side head and a server-side tail, reducing latency and limiting exposure of raw input data. However, inference performance often degrades in practical deployments due to user-specific data distribution shifts, unreliable communication, and privacy-oriented perturbations, especially in closed environments where model architectures and parameters are inaccessible. To address this challenge, we propose SALT (Split-Adaptive Lightweight Tuning), a lightweight adaptation framework for closed Split Computing systems. SALT introduces a compact client-side adapter that refines intermediate representations produced by a frozen head network, enabling effective model adaptation without modifying the head or tail networks or increasing communication overhead. By modifying only the training conditions, SALT supports multiple adaptation objectives, including user personalization, communication robustness, and privacy-aware inference. Experiments using ResNet-18 on CIFAR-10 and CIFAR-100 show that SALT achieves higher accuracy than conventional retraining and fine-tuning while significantly reducing training cost. On CIFAR-10, SALT improves personalized accuracy from 88.1% to 93.8% while reducing training latency by more than 60%. SALT also maintains over 90% accuracy under 75% packet loss and preserves high accuracy (about 88% at sigma = 1.0) under noise injection. These results demonstrate that SALT provides an efficient and practical adaptation framework for real-world Split Computing systems.

-> 경량 분산 컴퓨팅 프레임워크가 rk3588 엣지 디바이스에 최적화

### OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.12265v1
- 점수: final 96.0

Modern visual agents require representations that are general, causal, and physically structured to operate in real-time streaming environments. However, current vision foundation models remain fragmented, specializing narrowly in image semantic perception, offline temporal modeling, or spatial geometry. This paper introduces OmniStream, a unified streaming visual backbone that effectively perceives, reconstructs, and acts from diverse visual inputs. By incorporating causal spatiotemporal attention and 3D rotary positional embeddings (3D-RoPE), our model supports efficient, frame-by-frame online processing of video streams via a persistent KV-cache. We pre-train OmniStream using a synergistic multi-task framework coupling static and temporal representation learning, streaming geometric reconstruction, and vision-language alignment on 29 datasets. Extensive evaluations show that, even with a strictly frozen backbone, OmniStream achieves consistently competitive performance with specialized experts across image and video probing, streaming geometric reconstruction, complex video and spatial reasoning, as well as robotic manipulation (unseen at training). Rather than pursuing benchmark-specific dominance, our work demonstrates the viability of training a single, versatile vision backbone that generalizes across semantic, spatial, and temporal reasoning, i.e., a more meaningful step toward general-purpose visual understanding for interactive and embodied agents.

-> 연속 스트림에서 지각, 재구성 및 행동을 통합하는 실시간 비전 백본 기술

### Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.12255v1
- 점수: final 96.0

Humans perceive and understand real-world spaces through a stream of visual observations. Therefore, the ability to streamingly maintain and update spatial evidence from potentially unbounded video streams is essential for spatial intelligence. The core challenge is not simply longer context windows but how spatial information is selected, organized, and retained over time. In this paper, we propose Spatial-TTT towards streaming visual-based spatial intelligence with test-time training (TTT), which adapts a subset of parameters (fast weights) to capture and organize spatial evidence over long-horizon scene videos. Specifically, we design a hybrid architecture and adopt large-chunk updates parallel with sliding-window attention for efficient spatial video processing. To further promote spatial awareness, we introduce a spatial-predictive mechanism applied to TTT layers with 3D spatiotemporal convolution, which encourages the model to capture geometric correspondence and temporal continuity across frames. Beyond architecture design, we construct a dataset with dense 3D spatial descriptions, which guides the model to update its fast weights to memorize and organize global 3D spatial signals in a structured manner. Extensive experiments demonstrate that Spatial-TTT improves long-horizon spatial understanding and achieves state-of-the-art performance on video spatial benchmarks. Project page: https://liuff19.github.io/Spatial-TTT.

-> 스트리밍 비전 처리 기술이 스포츠 촬영 및 공간 이해에 직접적으로 적용 가능합니다

### BiGain: Unified Token Compression for Joint Generation and Classification (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.12240v1
- 점수: final 96.0

Acceleration methods for diffusion models (e.g., token merging or downsampling) typically optimize synthesis quality under reduced compute, yet often ignore discriminative capacity. We revisit token compression with a joint objective and present BiGain, a training-free, plug-and-play framework that preserves generation quality while improving classification in accelerated diffusion models. Our key insight is frequency separation: mapping feature-space signals into a frequency-aware representation disentangles fine detail from global semantics, enabling compression that respects both generative fidelity and discriminative utility. BiGain reflects this principle with two frequency-aware operators: (1) Laplacian-gated token merging, which encourages merges among spectrally smooth tokens while discouraging merges of high-contrast tokens, thereby retaining edges and textures; and (2) Interpolate-Extrapolate KV Downsampling, which downsamples keys/values via a controllable interextrapolation between nearest and average pooling while keeping queries intact, thereby conserving attention precision. Across DiT- and U-Net-based backbones and ImageNet-1K, ImageNet-100, Oxford-IIIT Pets, and COCO-2017, our operators consistently improve the speed-accuracy trade-off for diffusion-based classification, while maintaining or enhancing generation quality under comparable acceleration. For instance, on ImageNet-1K, with 70% token merging on Stable Diffusion 2.0, BiGain increases classification accuracy by 7.15% while improving FID by 0.34 (1.85%). Our analyses indicate that balanced spectral retention, preserving high-frequency detail and low/mid-frequency semantics, is a reliable design rule for token compression in diffusion models. To our knowledge, BiGain is the first framework to jointly study and advance both generation and classification under accelerated diffusion, supporting lower-cost deployment.

-> 엣지 장치에서 실시간 영역 분할 기술은 스포츠 촬영 및 분석에 직접적으로 적용 가능하다.

### InSpatio-WorldFM: An Open-Source Real-Time Generative Frame Model (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11911v1
- 점수: final 94.4

We present InSpatio-WorldFM, an open-source real-time frame model for spatial intelligence. Unlike video-based world models that rely on sequential frame generation and incur substantial latency due to window-level processing, InSpatio-WorldFM adopts a frame-based paradigm that generates each frame independently, enabling low-latency real-time spatial inference. By enforcing multi-view spatial consistency through explicit 3D anchors and implicit spatial memory, the model preserves global scene geometry while maintaining fine-grained visual details across viewpoint changes. We further introduce a progressive three-stage training pipeline that transforms a pretrained image diffusion model into a controllable frame model and finally into a real-time generator through few-step distillation. Experimental results show that InSpatio-WorldFM achieves strong multi-view consistency while supporting interactive exploration on consumer-grade GPUs, providing an efficient alternative to traditional video-based world models for real-time world simulation.

-> Efficient video tokenization technology applicable for sports highlight extraction on edge devices

### Generative Video Compression with One-Dimensional Latent Representation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15302v1
- 점수: final 94.4

Recent advancements in generative video codec (GVC) typically encode video into a 2D latent grid and employ high-capacity generative decoders for reconstruction. However, this paradigm still leaves two key challenges in fully exploiting spatial-temporal redundancy: Spatially, the 2D latent grid inevitably preserves intra-frame redundancy due to its rigid structure, where adjacent patches remain highly similar, thereby necessitating a higher bitrate. Temporally, the 2D latent grid is less effective for modeling long-term correlations in a compact and semantically coherent manner, as it hinders the aggregation of common contents across frames. To address these limitations, we introduce Generative Video Compression with One-Dimensional (1D) Latent Representation (GVC1D). GVC1D encodes the video data into extreme compact 1D latent tokens conditioned on both short- and long-term contexts. Without the rigid 2D spatial correspondence, these 1D latent tokens can adaptively attend to semantic regions and naturally facilitate token reduction, thereby reducing spatial redundancy. Furthermore, the proposed 1D memory provides semantically rich long-term context while maintaining low computational cost, thereby further reducing temporal redundancy. Experimental results indicate that GVC1D attains superior compression efficiency, where it achieves bitrate reductions of 60.4\% under LPIPS and 68.8\% under DISTS on the HEVC Class B dataset, surpassing the previous video compression methods.Project: https://gvc1d.github.io/

-> 이 논문은 스포츠 촬영 엣지 디바이스를 위한 카메라 보정 방법을 제안한다. 핵심은 레이저 트래커와 카메라 결합 기술이다.

### PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11917v1
- 점수: final 93.6

Real-time, on-device segmentation is critical for latency-sensitive and privacy-aware applications such as smart glasses and Internet-of-Things devices. We introduce PicoSAM3, a lightweight promptable visual segmentation model optimized for edge and in-sensor execution, including deployment on the Sony IMX500 vision sensor. PicoSAM3 has 1.3 M parameters and combines a dense CNN architecture with region of interest prompt encoding, Efficient Channel Attention, and knowledge distillation from SAM2 and SAM3. On COCO and LVIS, PicoSAM3 achieves 65.45% and 64.01% mIoU, respectively, outperforming existing SAM-based and edge-oriented baselines at similar or lower complexity. The INT8 quantized model preserves accuracy with negligible degradation while enabling real-time in-sensor inference at 11.82 ms latency on the IMX500, fully complying with its memory and operator constraints. Ablation studies show that distillation from large SAM models yields up to +14.5% mIoU improvement over supervised training and demonstrate that high-quality, spatially flexible promptable segmentation is feasible directly at the sensor level.

-> PicoSAM3의 실시간 영역 분할 기술은 스포츠 장면에서 선수나 중요 객체를 식별하는 데 직접적으로 활용 가능하며, 엣지 장치에서의 저지연 처리가 실시간 하이라이트 생성에 필수적입니다.

### SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.12193v1
- 점수: final 93.6

Active perception and manipulation are crucial for robots to interact with complex scenes. Existing methods struggle to unify semantic-driven active perception with robust, viewpoint-invariant execution. We propose SaPaVe, an end-to-end framework that jointly learns these capabilities in a data-efficient manner. Our approach decouples camera and manipulation actions rather than placing them in a shared action space, and follows a bottom-up training strategy: we first train semantic camera control on a large-scale dataset, then jointly optimize both action types using hybrid data. To support this framework, we introduce ActiveViewPose-200K, a dataset of 200k image-language-camera movement pairs for semantic camera movement learning, and a 3D geometry-aware module that improves execution robustness under dynamic viewpoints. We also present ActiveManip-Bench, the first benchmark for evaluating active manipulation beyond fixed-view settings. Extensive experiments in both simulation and real-world environments show that SaPaVe outperforms recent vision-language-action models such as GR00T N1 and \(π_0\), achieving up to 31.25\% higher success rates in real-world tasks. These results show that tightly coupled perception and execution, when trained with decoupled yet coordinated strategies, enable efficient and generalizable active manipulation. Project page: https://lmzpai.github.io/SaPaVe

-> 가속화된 확산 모델 기술이 엣지 디바이스에서 스포츠 영상 처리에 적용 가능

### EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.12267v1
- 점수: final 93.6

Autoregressive (AR) video generative models rely on video tokenizers that compress pixels into discrete token sequences. The length of these token sequences is crucial for balancing reconstruction quality against downstream generation computational cost. Traditional video tokenizers apply a uniform token assignment across temporal blocks of different videos, often wasting tokens on simple, static, or repetitive segments while underserving dynamic or complex ones. To address this inefficiency, we introduce $\textbf{EVATok}$, a framework to produce $\textbf{E}$fficient $\textbf{V}$ideo $\textbf{A}$daptive $\textbf{Tok}$enizers. Our framework estimates optimal token assignments for each video to achieve the best quality-cost trade-off, develops lightweight routers for fast prediction of these optimal assignments, and trains adaptive tokenizers that encode videos based on the assignments predicted by routers. We demonstrate that EVATok delivers substantial improvements in efficiency and overall quality for video reconstruction and downstream AR generation. Enhanced by our advanced training recipe that integrates video semantic encoders, EVATok achieves superior reconstruction and state-of-the-art class-to-video generation on UCF-101, with at least 24.4% savings in average token usage compared to the prior state-of-the-art LARP and our fixed-length baseline.

-> Efficient video tokenization technology applicable for sports highlight extraction on edge devices

### Multimodal Emotion Recognition via Bi-directional Cross-Attention and Temporal Modeling (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11971v1
- 점수: final 93.6

Emotion recognition in in-the-wild video data remains a challenging problem due to large variations in facial appearance, head pose, illumination, background noise, and the inherently dynamic nature of human affect. Relying on a single modality, such as facial expressions or speech, is often insufficient to capture these complex emotional cues. To address this issue, we propose a multimodal emotion recognition framework for the Expression (EXPR) Recognition task in the 10th Affective Behavior Analysis in-the-wild (ABAW) Challenge.   Our approach leverages large-scale pre-trained models, namely CLIP for visual encoding and Wav2Vec 2.0 for audio representation learning, as frozen backbone networks. To model temporal dependencies in facial expression sequences, we employ a Temporal Convolutional Network (TCN) over fixed-length video windows. In addition, we introduce a bi-directional cross-attention fusion module, in which visual and audio features interact symmetrically to enhance cross-modal contextualization and capture complementary emotional information. A lightweight classification head is then used for final emotion prediction. We further incorporate a text-guided contrastive objective based on CLIP text features to encourage semantically aligned visual representations.   Experimental results on the ABAW 10th EXPR benchmark show that the proposed framework provides a strong multimodal baseline and achieves improved performance over unimodal modeling. These results demonstrate the effectiveness of combining temporal visual modeling, audio representation learning, and cross-modal fusion for robust emotion recognition in unconstrained real-world environments.

-> 다중 모달 접근법과 시간적 모델링이 스포츠 선수의 감정 반응과 경기 상황 분석에 적용 가능하여 경기 전략 수립에 도움을 줄 수 있습니다.

### Affordable Precision Agriculture: A Deployment-Oriented Review of Low-Cost, Low-Power Edge AI and TinyML for Resource-Constrained Farming Systems (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15085v1
- 점수: final 93.6

Precision agriculture increasingly integrates artificial intelligence to enhance crop monitoring, irrigation management, and resource efficiency. Nevertheless, the vast majority of the current systems are still mostly cloud-based and require reliable connectivity, which hampers the adoption to smaller scale, smallholder farming and underdeveloped country systems. Using recent literature reviews, ranging from 2023 to 2026, this review covers deployments of Edge AI, focused on the evolution and acceptance of Tiny Machine Learning, in low-cost and low-powered agriculture. A hardware-targeted deployment-oriented study has shown pronounced variation in architecture with microcontroller-class platforms i.e. ESP32, STM32, ATMega dominating the inference options, in parallel with single-board computers and UAV-assisted solutions. Quantitative synthesis shows quantization is the dominant optimization strategy; the approach in many works identified: around 50% of such works are quantized, while structured pruning, multi-objective compression and hardware aware neural architecture search are relatively under-researched. Also, resource profiling practices are not uniform: while model size is occasionally reported, explicit flash, RAM, MAC, latency and millijoule level energy metrics are not well documented, hampering reproducibility and cross-system comparison. Moreoever, to bridge the gap between research prototypes and deployment-ready systems, the review also presents a literature-informed deployment perspective in the form of a privacy-preserving layered Edge AI architecture for agriculture, synthesizing the key system-level design insights emerging from the surveyed works. Overall, the findings demonstrate a clear architectural shift toward localized inference with centralized training asymmetry.

-> 이 논문은 다양한 입력 시나리오 처리를 위한 OOD 검출 방법을 제안한다. 핵심은 이중 프로토타이프 추적 기술이다.

### A Novel Camera-to-Robot Calibration Method for Vision-Based Floor Measurements (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15126v1
- 점수: final 93.6

A novel hand-eye calibration method for ground-observing mobile robots is proposed. While cameras on mobile robots are com- mon, they are rarely used for ground-observing measurement tasks. Laser trackers are increasingly used in robotics for precise localization. A referencing plate is designed to combine the two measurement modalities of laser-tracker 3D metrology and camera- based 2D imaging. It incorporates reflector nests for pose acquisition using a laser tracker and a camera calibration target that is observed by the robot-mounted camera. The procedure comprises estimating the plate pose, the plate-camera pose, and the robot pose, followed by computing the robot-camera transformation. Experiments indicate sub-millimeter repeatability.

-> Camera calibration techniques applicable to sports filming edge device

### Federated Learning of Binary Neural Networks: Enabling Low-Cost Inference (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15507v1
- 점수: final 93.6

Federated Learning (FL) preserves privacy by distributing training across devices. However, using DNNs is computationally intensive at the low-powered edge during inference. Edge deployment demands models that simultaneously optimize memory footprint and computational efficiency, a dilemma where conventional DNNs fail by exceeding resource limits. Traditional post-training binarization reduces model size but suffers from severe accuracy loss due to quantization errors. To address these challenges, we propose FedBNN, a rotation-aware binary neural network framework that learns binary representations directly during local training. By encoding each weight as a single bit $\{+1, -1\}$ instead of a $32$-bit float, FedBNN shrinks the model footprint, significantly reducing runtime (during inference) FLOPs and memory requirements in comparison to federated methods using real models. Evaluations across multiple benchmark datasets demonstrate that FedBNN significantly reduces resource consumption while performing similarly to existing federated methods using real-valued models.

-> 이진 신경망 기술이 엣지 디바이스에서의 효율적인 AI 처리에 직접 적용 가능

### Tracking the Discriminative Axis: Dual Prototypes for Test-Time OOD Detection Under Covariate Shift (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15213v1
- 점수: final 93.6

For reliable deployment of deep-learning systems, out-of-distribution (OOD) detection is indispensable. In the real world, where test-time inputs often arrive as streaming mixtures of in-distribution (ID) and OOD samples under evolving covariate shifts, OOD samples are domain-constrained and bounded by the environment, and both ID and OOD are jointly affected by the same covariate factors. Existing methods typically assume a stationary ID distribution, but this assumption breaks down in such settings, leading to severe performance degradation. We empirically discover that, even under covariate shift, covariate-shifted ID (csID) and OOD (csOOD) samples remain separable along a discriminative axis in feature space. Building on this observation, we propose DART, a test-time, online OOD detection method that dynamically tracks dual prototypes -- one for ID and the other for OOD -- to recover the drifting discriminative axis, augmented with multi-layer fusion and flip correction for robustness. Extensive experiments on a wide range of challenging benchmarks, where all datasets are subjected to 15 common corruption types at severity level 5, demonstrate that our method significantly improves performance, yielding 15.32 percentage points (pp) AUROC gain and 49.15 pp FPR@95TPR reduction on ImageNet-C vs. Textures-C compared to established baselines. These results highlight the potential of the test-time discriminative axis tracking for dependable OOD detection in dynamically changing environments.

-> OOD detection technology applicable for sports filming edge device handling various input scenarios

### Nova: Scalable Streaming Join Placement and Parallelization in Resource-Constrained Geo-Distributed Environments (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15453v1
- 점수: final 93.6

Real-time data processing in large geo-distributed applications, like the Internet of Things (IoT), increasingly shifts computation from the cloud to the network edge to reduce latency and mitigate network congestion. In this setting, minimizing latency while avoiding node overload requires jointly optimizing operator replication and placement of operator instances, a challenge known as the Operator Placement and Replication (OPR) problem. OPR is NP-hard and particularly difficult to solve in large-scale, heterogeneous, and dynamic geo-distributed networks, where solutions must be scalable, resource-aware, and adaptive to changes like node failures. Existing work on OPR has primarily focused on single-stream operators, such as filters and aggregations. However, many latency-sensitive applications, like environmental monitoring and anomaly detection, require efficient regional stream joins near data sources.   This paper introduces Nova, an optimization approach designed to address OPR for join operators that are computable on resource-constrained edge devices. Nova relaxes the NP-hard OPR into a convex optimization problem by embedding cost metrics into a Euclidean space and partitioning joins into smaller sub-joins. This new formulation enables linear scalability and efficient adaptation to topological changes through partial re-optimizations. We evaluate Nova through simulations on real-world topologies and on a local testbed, demonstrating up to 39x latency reduction and 4.5x increase in throughput compared to existing edge-centered solutions, while also preventing node overload and maintaining near-constant re-optimization times regardless of topology size.

-> Edit2Interp은 이미지 모델을 적은 데이터로 비디오 처리에 적용하여 스포츠 영상 향상에 효과적입니다. 이는 에지 디바이스에서의 실시간 영상 처리에 적합합니다.

### Efficient Event Camera Volume System (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.14738v1
- 점수: final 90.4

Event cameras promise low latency and high dynamic range, yet their sparse output challenges integration into standard robotic pipelines. We introduce \nameframew (Efficient Event Camera Volume System), a novel framework that models event streams as continuous-time Dirac impulse trains, enabling artifact-free compression through direct transform evaluation at event timestamps. Our key innovation combines density-driven adaptive selection among DCT, DTFT, and DWT transforms with transform-specific coefficient pruning strategies tailored to each domain's sparsity characteristics. The framework eliminates temporal binning artifacts while automatically adapting compression strategies based on real-time event density analysis. On EHPT-XC and MVSEC datasets, our framework achieves superior reconstruction fidelity with DTFT delivering the lowest earth mover distance. In downstream segmentation tasks, EECVS demonstrates robust generalization. Notably, our approach demonstrates exceptional cross-dataset generalization: when evaluated with EventSAM segmentation, EECVS achieves mean IoU 0.87 on MVSEC versus 0.44 for voxel grids at 24 channels, while remaining competitive on EHPT-XC. Our ROS2 implementation provides real-time deployment with DCT processing achieving 1.5 ms latency and 2.7X higher throughput than alternative transforms, establishing the first adaptive event compression framework that maintains both computational efficiency and superior generalization across diverse robotic scenarios.

-> 자원 제약된 edge 환경에서 실시간 데이터 처리를 최적화하여 지연 시간을 최대 39배까지 줄일 수 있습니다.

### M2IR: Proactive All-in-One Image Restoration via Mamba-style Modulation and Mixture-of-Experts (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.14816v1
- 점수: final 90.4

While Transformer-based architectures have dominated recent advances in all-in-one image restoration, they remain fundamentally reactive: propagating degradations rather than proactively suppressing them. In the absence of explicit suppression mechanisms, degraded signals interfere with feature learning, compelling the decoder to balance artifact removal and detail preservation, thereby increasing model complexity and limiting adaptability. To address these challenges, we propose M2IR, a novel restoration framework that proactively regulates degradation propagation during the encoding stage and efficiently eliminates residual degradations during decoding. Specifically, the Mamba-Style Transformer (MST) block performs pixel-wise selective state modulation to mitigate degradations while preserving structural integrity. In parallel, the Adaptive Degradation Expert Collaboration (ADEC) module utilizes degradation-specific experts guided by a DA-CLIP-driven router and complemented by a shared expert to eliminate residual degradations through targeted and cooperative restoration. By integrating the MST block and ADEC module, M2IR transitions from passive reaction to active degradation control, effectively harnessing learned representations to achieve superior generalization, enhanced adaptability, and refined recovery of fine-grained details across diverse all-in-one image restoration benchmarks. Our source codes are available at https://github.com/Im34v/M2IR.

-> 이미지 편집 모델을 영상 처리로 확장하여 적은 데이터로(64-256 샘플) 스포츠 영상 향상이 가능합니다.

### Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11460v1
- 점수: final 90.4

Existing retrieval-augmented approaches for Dense Video Captioning (DVC) often fail to achieve accurate temporal segmentation aligned with true event boundaries, as they rely on heuristic strategies that overlook ground truth event boundaries. The proposed framework, \textbf{STaRC}, overcomes this limitation by supervising frame-level saliency through a highlight detection module. Note that the highlight detection module is trained on binary labels derived directly from DVC ground truth annotations without the need for additional annotation. We also propose to utilize the saliency scores as a unified temporal signal that drives retrieval via saliency-guided segmentation and informs caption generation through explicit Saliency Prompts injected into the decoder. By enforcing saliency-constrained segmentation, our method produces temporally coherent segments that align closely with actual event transitions, leading to more accurate retrieval and contextually grounded caption generation. We conduct comprehensive evaluations on the YouCook2 and ViTT benchmarks, where STaRC achieves state-of-the-art performance across most of the metrics. Our code is available at https://github.com/ermitaju1/STaRC

-> 시각적 중요도를 기반으로 한 하이라이트 감지 기술은 스포츠 하이라이트 편집에 직접적으로 적용 가능하여 자동으로 주요 장면을 추출하고 편집할 수 있습니다.

### Edit2Interp: Adapting Image Foundation Models from Spatial Editing to Video Frame Interpolation with Few-Shot Learning (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15003v1
- 점수: final 89.6

Pre-trained image editing models exhibit strong spatial reasoning and object-aware transformation capabilities acquired from billions of image-text pairs, yet they possess no explicit temporal modeling. This paper demonstrates that these spatial priors can be repurposed to unlock temporal synthesis capabilities through minimal adaptation - without introducing any video-specific architecture or motion estimation modules. We show that a large image editing model (Qwen-Image-Edit), originally designed solely for static instruction-based edits, can be adapted for Video Frame Interpolation (VFI) using only 64-256 training samples via Low-Rank Adaptation (LoRA). Our core contribution is revealing that the model's inherent understanding of "how objects transform" in static scenes contains latent temporal reasoning that can be activated through few-shot fine-tuning. While the baseline model completely fails at producing coherent intermediate frames, our parameter-efficient adaptation successfully unlocks its interpolation capability. Rather than competing with task-specific VFI methods trained from scratch on massive datasets, our work establishes that foundation image editing models possess untapped potential for temporal tasks, offering a data-efficient pathway for video synthesis in resource-constrained scenarios. This bridges the gap between image manipulation and video understanding, suggesting that spatial and temporal reasoning may be more intertwined in foundation models than previously recognized

-> Adapting image models for video processing with few-shot learning applicable to sports video enhancement

### Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.14726v1
- 점수: final 89.6

Accurately recovering hand poses within the body context remains a major challenge in 3D whole-body pose estimation. This difficulty arises from a fundamental supervision gap: whole-body pose estimators are trained on full-body datasets with limited hand diversity, while hand-only estimators, trained on hand-centric datasets, excel at detailed finger articulation but lack global body awareness. To address this, we propose Hand4Whole++, a modular framework that leverages the strengths of both pre-trained whole-body and hand pose estimators. We introduce CHAM (Conditional Hands Modulator), a lightweight module that modulates the whole-body feature stream using hand-specific features extracted from a pre-trained hand pose estimator. This modulation enables the whole-body model to predict wrist orientations that are both accurate and coherent with the upper-body kinematic structure, without retraining the full-body model. In parallel, we directly incorporate finger articulations and hand shapes predicted by the hand pose estimator, aligning them to the full-body mesh via differentiable rigid alignment. This design allows Hand4Whole++ to combine globally consistent body reasoning with fine-grained hand detail. Extensive experiments demonstrate that Hand4Whole++ substantially improves hand accuracy and enhances overall full-body pose quality.

-> 3D 전신 자세 추정 프레임워크로 스포츠 동작 분석에 적합하며, 손 동작 정밀도 향상으로 다양한 스포츠 기술 분석 가능

### CyCLeGen: Cycle-Consistent Layout Prediction and Image Generation in Vision Foundation Models (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.14957v1
- 점수: final 89.6

We present CyCLeGen, a unified vision-language foundation model capable of both image understanding and image generation within a single autoregressive framework. Unlike existing vision models that depend on separate modules for perception and synthesis, CyCLeGen adopts a fully integrated architecture that enforces cycle-consistent learning through image->layout->image and layout->image->layout generation loops. This unified formulation introduces two key advantages: introspection, enabling the model to reason about its own generations, and data efficiency, allowing self-improvement via synthetic supervision under a reinforcement learning objective guided by cycle consistency. Extensive experiments show that CyCLeGen achieves significant gains across diverse image understanding and generation benchmarks, highlighting the potential of unified vision-language foundation models.

-> 이미지 생성 및 이해 기술을 통한 스포츠 영상의 사진처럼 보정 기구 구현 가능하며, 통합 아키텍처로 효율적인 처리

### Riemannian Motion Generation: A Unified Framework for Human Motion Representation and Generation via Riemannian Flow Matching (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15016v1
- 점수: final 89.6

Human motion generation is often learned in Euclidean spaces, although valid motions follow structured non-Euclidean geometry. We present Riemannian Motion Generation (RMG), a unified framework that represents motion on a product manifold and learns dynamics via Riemannian flow matching. RMG factorizes motion into several manifold factors, yielding a scale-free representation with intrinsic normalization, and uses geodesic interpolation, tangent-space supervision, and manifold-preserving ODE integration for training and sampling. On HumanML3D, RMG achieves state-of-the-art FID in the HumanML3D format (0.043) and ranks first on all reported metrics under the MotionStreamer format. On MotionMillion, it also surpasses strong baselines (FID 5.6, R@1 0.86). Ablations show that the compact $\mathscr{T}+\mathscr{R}$ (translation + rotations) representation is the most stable and effective, highlighting geometry-aware modeling as a practical and scalable route to high-fidelity motion generation.

-> Human motion generation framework applicable for sports movement analysis

### $Ψ_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.12263v1
- 점수: final 88.8

We introduce $Ψ_0$ (Psi-Zero), an open foundation model to address challenging humanoid loco-manipulation tasks. While existing approaches often attempt to address this fundamental problem by co-training on large and diverse human and humanoid data, we argue that this strategy is suboptimal due to the fundamental kinematic and motion disparities between humans and humanoid robots. Therefore, data efficiency and model performance remain unsatisfactory despite the considerable data volume. To address this challenge, \ours\;decouples the learning process to maximize the utility of heterogeneous data sources. Specifically, we propose a staged training paradigm with different learning objectives: First, we autoregressively pre-train a VLM backbone on large-scale egocentric human videos to acquire generalizable visual-action representations. Then, we post-train a flow-based action expert on high-quality humanoid robot data to learn precise robot joint control. Our research further identifies a critical yet often overlooked data recipe: in contrast to approaches that scale with noisy Internet clips or heterogeneous cross-embodiment robot datasets, we demonstrate that pre-training on high-quality egocentric human manipulation data followed by post-training on domain-specific real-world humanoid trajectories yields superior performance. Extensive real-world experiments demonstrate that \ours\ achieves the best performance using only about 800 hours of human video data and 30 hours of real-world robot data, outperforming baselines pre-trained on more than 10$\times$ as much data by over 40\% in overall success rate across multiple tasks. We will open-source the entire ecosystem to the community, including a data processing and training pipeline, a humanoid foundation model, and a real-time action inference engine.

-> 인간형 로봇의 움직임 및 조작을 위한 오픈소스 기반 모델은 향후 스포츠 활동과 상호작용하는 피지컬 AI 하드웨어 개발에 기반이 될 수 있습니다.

### Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11556v1
- 점수: final 88.0

Image aesthetic enhancement aims to perceive aesthetic deficiencies in images and perform corresponding editing operations, which is highly challenging and requires the model to possess creativity and aesthetic perception capabilities. Although recent advancements in image editing models have significantly enhanced their controllability and flexibility, they struggle with enhancing image aesthetic. The primary challenges are twofold: first, following editing instructions with aesthetic perception is difficult, and second, there is a scarcity of "perfectly-paired" images that have consistent content but distinct aesthetic qualities. In this paper, we propose Dual-supervised Image Aesthetic Enhancement (DIAE), a diffusion-based generative model with multimodal aesthetic perception. First, DIAE incorporates Multimodal Aesthetic Perception (MAP) to convert the ambiguous aesthetic instruction into explicit guidance by (i) employing detailed, standardized aesthetic instructions across multiple aesthetic attributes, and (ii) utilizing multimodal control signals derived from text-image pairs that maintain consistency within the same aesthetic attribute. Second, to mitigate the lack of "perfectly-paired" images, we collect "imperfectly-paired" dataset called IIAEData, consisting of images with varying aesthetic qualities while sharing identical semantics. To better leverage the weak matching characteristics of IIAEData during training, a dual-branch supervision framework is also introduced for weakly supervised image aesthetic enhancement. Experimental results demonstrate that DIAE outperforms the baselines and obtains superior image aesthetic scores and image content consistency scores.

-> 이미지 미적 향상 기술이 스포츠 영상 보정에 직접적으로 적용 가능하며 촬영된 영상의 품질을 향상시켜 시각적 효과를 극대화할 수 있습니다.

### Bridging Discrete Marks and Continuous Dynamics: Dual-Path Cross-Interaction for Marked Temporal Point Processes (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11462v1
- 점수: final 88.0

Predicting irregularly spaced event sequences with discrete marks poses significant challenges due to the complex, asynchronous dependencies embedded within continuous-time data streams.Existing sequential approaches capture dependencies among event tokens but ignore the continuous evolution between events, while Neural Ordinary Differential Equation (Neural ODE) methods model smooth dynamics yet fail to account for how event types influence future timing.To overcome these limitations, we propose NEXTPP, a dual-channel framework that unifies discrete and continuous representations via Event-granular Neural Evolution with Cross-Interaction for Marked Temporal Point Processes. Specifically, NEXTPP encodes discrete event marks via a self-attention mechanism, simultaneously evolving a latent continuous-time state using a Neural ODE. These parallel streams are then fused through a crossattention module to enable explicit bidirectional interaction between continuous and discrete representations. The fused representations drive the conditional intensity function of the neural Hawkes process, while an iterative thinning sampler is employed to generate future events. Extensive evaluations on five real-world datasets demonstrate that NEXTPP consistently outperforms state-of-the-art models. The source code can be found at https://github.com/AONE-NLP/NEXTPP.

-> 이 논문은 스포츠 경기의 주요 순간을 자동으로 식별하는 데 적용 가능한 시간적 이벤트 예측 기술을 제안합니다. 이는 우리 AI 촬영 에지 디바이스가 자동으로 중요 장면을 포착하고 편집하는 핵심 기술이 될 수 있습니다.

### CFD-HAR: User-controllable Privacy through Conditional Feature Disentanglement (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11526v1
- 점수: final 88.0

Modern wearable and mobile devices are equipped with inertial measurement units (IMUs). Human Activity Recognition (HAR) applications running on such devices use machine-learning-based, data-driven techniques that leverage such sensor data. However, sensor-data-driven HAR deployments face two critical challenges: protecting sensitive user information embedded in sensor data in accordance with users' privacy preferences and maintaining high recognition performance with limited labeled samples. This paper proposes a technique for user-controllable privacy through feature disentanglement-based representation learning at the granular level for dynamic privacy filtering. We also compare the efficacy of our technique against few-shot HAR using autoencoder-based representation learning. We analyze their architectural designs, learning objectives, privacy guarantees, data efficiency, and suitability for edge Internet of Things (IoT) deployment. Our study shows that CFD-based HAR provides explicit, tunable privacy protection controls by separating activity and sensitive attributes in the latent space, whereas autoencoder-based few-shot HAR offers superior label efficiency and lightweight adaptability but lacks inherent privacy safeguards. We further examine the security implications of both approaches in continual IoT settings, highlighting differences in susceptibility to representation leakage and embedding-level attacks. The analysis reveals that neither paradigm alone fully satisfies the emerging requirements of next-generation IoT HAR systems. We conclude by outlining research directions toward unified frameworks that jointly optimize privacy preservation, few-shot adaptability, and robustness for trustworthy IoT intelligence.

-> 사용자 프라이버시 보호 기능을 갖춘 인간 활동 인식 기술은 스포츠 동작 분석에 직접적으로 적용 가능하며 개인정보 보호와 분석 성능을 동시에 만족시킵니다.

### PAKAN: Pixel Adaptive Kolmogorov-Arnold Network Modules for Pansharpening (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15109v1
- 점수: final 86.4

Pansharpening aims to fuse high-resolution spatial details from panchromatic images with the rich spectral information of multispectral images. Existing deep neural networks for this task typically rely on static activation functions, which limit their ability to dynamically model the complex, non-linear mappings required for optimal spatial-spectral fusion. While the recently introduced Kolmogorov-Arnold Network (KAN) utilizes learnable activation functions, traditional KANs lack dynamic adaptability during inference. To address this limitation, we propose a Pixel Adaptive Kolmogorov-Arnold Network framework. Starting from KAN, we design two adaptive variants: a 2D Adaptive KAN that generates spline summation weights across spatial dimensions and a 1D Adaptive KAN that generates them across spectral channels. These two components are then assembled into PAKAN 2to1 for feature fusion and PAKAN 1to1 for feature refinement. Extensive experiments demonstrate that our proposed modules significantly enhance network performance, proving the effectiveness and superiority of pixel-adaptive activation in pansharpening tasks.

-> 이미지 보정 기술로 스포츠 사진을 전문가 수준으로 향상시킬 수 있어 플랫폼 콘텐츠 퀄리티 향상에 필수적

### SoulX-LiveAct: Towards Hour-Scale Real-Time Human Animation with Neighbor Forcing and ConvKV Memory (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11746v1
- 점수: final 86.4

Autoregressive (AR) diffusion models offer a promising framework for sequential generation tasks such as video synthesis by combining diffusion modeling with causal inference. Although they support streaming generation, existing AR diffusion methods struggle to scale efficiently. In this paper, we identify two key challenges in hour-scale real-time human animation. First, most forcing strategies propagate sample-level representations with mismatched diffusion states, causing inconsistent learning signals and unstable convergence. Second, historical representations grow unbounded and lack structure, preventing effective reuse of cached states and severely limiting inference efficiency. To address these challenges, we propose Neighbor Forcing, a diffusion-step-consistent AR formulation that propagates temporally adjacent frames as latent neighbors under the same noise condition. This design provides a distribution-aligned and stable learning signal while preserving drifting throughout the AR chain. Building upon this, we introduce a structured ConvKV memory mechanism that compresses the keys and values in causal attention into a fixed-length representation, enabling constant-memory inference and truly infinite video generation without relying on short-term motion-frame memory. Extensive experiments demonstrate that our approach significantly improves training convergence, hour-scale generation quality, and inference efficiency compared to existing AR diffusion methods. Numerically, LiveAct enables hour-scale real-time human animation and supports 20 FPS real-time streaming inference on as few as two NVIDIA H100 or H200 GPUs. Quantitative results demonstrate that our method attains state-of-the-art performance in lip-sync accuracy, human animation quality, and emotional expressiveness, with the lowest inference cost.

-> 실시간 인간 애니메이션 기술은 스포츠 동작 분석에 적용 가능하다.

### MV-SAM3D: Adaptive Multi-View Fusion for Layout-Aware 3D Generation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11633v1
- 점수: final 86.4

Recent unified 3D generation models have made remarkable progress in producing high-quality 3D assets from a single image. Notably, layout-aware approaches such as SAM3D can reconstruct multiple objects while preserving their spatial arrangement, opening the door to practical scene-level 3D generation. However, current methods are limited to single-view input and cannot leverage complementary multi-view observations, while independently estimated object poses often lead to physically implausible layouts such as interpenetration and floating artifacts.   We present MV-SAM3D, a training-free framework that extends layout-aware 3D generation with multi-view consistency and physical plausibility. We formulate multi-view fusion as a Multi-Diffusion process in 3D latent space and propose two adaptive weighting strategies -- attention-entropy weighting and visibility weighting -- that enable confidence-aware fusion, ensuring each viewpoint contributes according to its local observation reliability. For multi-object composition, we introduce physics-aware optimization that injects collision and contact constraints both during and after generation, yielding physically plausible object arrangements. Experiments on standard benchmarks and real-world multi-object scenes demonstrate significant improvements in reconstruction fidelity and layout plausibility, all without any additional training. Code is available at https://github.com/devinli123/MV-SAM3D.

-> Multi-view 3D generation technology applicable for sports camera systems

### Resource-Efficient Iterative LLM-Based NAS with Feedback Memory (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.12091v1
- 점수: final 86.4

Neural Architecture Search (NAS) automates network design, but conventional methods demand substantial computational resources. We propose a closed-loop pipeline leveraging large language models (LLMs) to iteratively generate, evaluate, and refine convolutional neural network architectures for image classification on a single consumer-grade GPU without LLM fine-tuning. Central to our approach is a historical feedback memory inspired by Markov chains: a sliding window of $K{=}5$ recent improvement attempts keeps context size constant while providing sufficient signal for iterative learning. Unlike prior LLM optimizers that discard failure trajectories, each history entry is a structured diagnostic triple -- recording the identified problem, suggested modification, and resulting outcome -- treating code execution failures as first-class learning signals. A dual-LLM specialization reduces per-call cognitive load: a Code Generator produces executable PyTorch architectures while a Prompt Improver handles diagnostic reasoning. Since both the LLM and architecture training share limited VRAM, the search implicitly favors compact, hardware-efficient models suited to edge deployment. We evaluate three frozen instruction-tuned LLMs (${\leq}7$B parameters) across up to 2000 iterations in an unconstrained open code space, using one-epoch proxy accuracy on CIFAR-10, CIFAR-100, and ImageNette as a fast ranking signal. On CIFAR-10, DeepSeek-Coder-6.7B improves from 28.2% to 69.2%, Qwen2.5-7B from 50.0% to 71.5%, and GLM-5 from 43.2% to 62.0%. A full 2000-iteration search completes in ${\approx}18$ GPU hours on a single RTX~4090, establishing a low-budget, reproducible, and hardware-aware paradigm for LLM-driven NAS without cloud infrastructure.

-> 에지 장치용 효율적인 모델 설계가 스포츠 촬영 에지 디바이스에 적용 가능한 기술

### Spatio-temporal probabilistic forecast using MMAF-guided learning (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15055v1
- 점수: final 85.6

We employ stochastic feed-forward neural networks with Gaussian-distributed weights to determine a probabilistic forecast for spatio-temporal raster datasets. The networks are trained using MMAF-guided learning, a generalized Bayesian methodology in which the observed data are preprocessed using an embedding designed to produce a low-dimensional representation that captures their dependence and causal structure. The design of the embedding is theory-guided by the assumption that a spatio-temporal Ornstein-Uhlenbeck process with finite second-order moments generates the observed data. The trained networks, in inference mode, are then used to generate ensemble forecasts by applying different initial conditions at different horizons. Experiments conducted on both synthetic and real data demonstrate that our forecasts remain calibrated across multiple time horizons. Moreover, we show that on such data, simple feed-forward architectures can achieve performance comparable to, and in some cases better than, convolutional or diffusion deep learning architectures used in probabilistic forecasting tasks.

-> Spatio-temporal forecasting techniques could be applicable to sports game analysis and strategy prediction

### Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11439v1
- 점수: final 85.6

Dense Video Captioning (DVC) is a challenging multimodal task that involves temporally localizing multiple events within a video and describing them with natural language. While query-based frameworks enable the simultaneous, end-to-end processing of localization and captioning, their reliance on shared queries often leads to significant multi-task interference between the two tasks, as well as temporal redundancy in localization. In this paper, we propose utilizing role-specific queries that separate localization and captioning into independent components, allowing each to exclusively learn its role. We then employ contrastive alignment to enforce semantic consistency between the corresponding outputs, ensuring coherent behavior across the separated queries. Furthermore, we design a novel suppression mechanism in which mutual temporal overlaps across queries are penalized to tackle temporal redundancy, supervising the model to learn distinct, non-overlapping event regions for more precise localization. Additionally, we introduce a lightweight module that captures core event concepts to further enhance semantic richness in captions through concept-level representations. We demonstrate the effectiveness of our method through extensive experiments on major DVC benchmarks YouCook2 and ActivityNet Captions.

-> 비디오 캡셔닝 기술은 스포츠 영상 분석에 적용 가능

### INFACT: A Diagnostic Benchmark for Induced Faithfulness and Factuality Hallucinations in Video-LLMs (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11481v1
- 점수: final 85.6

Despite rapid progress, Video Large Language Models (Video-LLMs) remain unreliable due to hallucinations, which are outputs that contradict either video evidence (faithfulness) or verifiable world knowledge (factuality). Existing benchmarks provide limited coverage of factuality hallucinations and predominantly evaluate models only in clean settings. We introduce \textsc{INFACT}, a diagnostic benchmark comprising 9{,}800 QA instances with fine-grained taxonomies for faithfulness and factuality, spanning real and synthetic videos. \textsc{INFACT} evaluates models in four modes: Base (clean), Visual Degradation, Evidence Corruption, and Temporal Intervention for order-sensitive items. Reliability under induced modes is quantified using Resist Rate (RR) and Temporal Sensitivity Score (TSS). Experiments on 14 representative Video-LLMs reveal that higher Base-mode accuracy does not reliably translate to higher reliability in the induced modes, with evidence corruption reducing stability and temporal intervention yielding the largest degradation. Notably, many open-source baselines exhibit near-zero TSS on factuality, indicating pronounced temporal inertia on order-sensitive questions.

-> 비디오 LLM 기술이 스포츠 비디오 분석에 적용 가능합니다

### Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.12247v1
- 점수: final 84.8

Reinforcement learning (RL) has emerged as a promising paradigm for enhancing image editing and text-to-image (T2I) generation. However, current reward models, which act as critics during RL, often suffer from hallucinations and assign noisy scores, inherently misguiding the optimization process. In this paper, we present FIRM (Faithful Image Reward Modeling), a comprehensive framework that develops robust reward models to provide accurate and reliable guidance for faithful image generation and editing. First, we design tailored data curation pipelines to construct high-quality scoring datasets. Specifically, we evaluate editing using both execution and consistency, while generation is primarily assessed via instruction following. Using these pipelines, we collect the FIRM-Edit-370K and FIRM-Gen-293K datasets, and train specialized reward models (FIRM-Edit-8B and FIRM-Gen-8B) that accurately reflect these criteria. Second, we introduce FIRM-Bench, a comprehensive benchmark specifically designed for editing and generation critics. Evaluations demonstrate that our models achieve superior alignment with human judgment compared to existing metrics. Furthermore, to seamlessly integrate these critics into the RL pipeline, we formulate a novel "Base-and-Bonus" reward strategy that balances competing objectives: Consistency-Modulated Execution (CME) for editing and Quality-Modulated Alignment (QMA) for generation. Empowered by this framework, our resulting models FIRM-Qwen-Edit and FIRM-SD3.5 achieve substantial performance breakthroughs. Comprehensive experiments demonstrate that FIRM mitigates hallucinations, establishing a new standard for fidelity and instruction adherence over existing general models. All of our datasets, models, and code have been publicly available at https://firm-reward.github.io.

-> 이미지 편집 및 생성 기술이 프로젝트의 영상 보정 및 편집 부분에 직접 적용 가능

### SNAP-V: A RISC-V SoC with Configurable Neuromorphic Acceleration for Small-Scale Spiking Neural Networks (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11939v1
- 점수: final 84.0

Spiking Neural Networks (SNNs) have gained significant attention in edge computing due to their low power consumption and computational efficiency. However, existing implementations either use conventional System on Chip (SoC) architectures that suffer from memory-processor bottlenecks, or large-scale neuromorphic hardware that is inefficient and wasteful for small-scale SNN applications. This work presents SNAP-V, a RISC-V-based neuromorphic SoC with two accelerator variants: Cerebra-S (bus-based) and Cerebra-H (Network-on-Chip (NoC)-based) which are optimized for small-scale SNN inference, integrating a RISC-V core for management tasks, with both accelerators featuring parallel processing nodes and distributed memory. Experimental results show close agreement between software and hardware inference, with an average accuracy deviation of 2.62% across multiple network configurations, and an average synaptic energy of 1.05 pJ per synaptic operation (SOP) in 45 nm CMOS technology. These results show that the proposed solution enables accurate, energy-efficient SNN inference suitable for real-time edge applications.

-> RISC-V 기반의 신경망 가속기가 에지 디바이스의 저전력 실시간 처리에 적합

### SimCert: Probabilistic Certification for Behavioral Similarity in Deep Neural Network Compression (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.14818v1
- 점수: final 84.0

Deploying Deep Neural Networks (DNNs) on resource-constrained embedded systems requires aggressive model compression techniques like quantization and pruning. However, ensuring that the compressed model preserves the behavioral fidelity of the original design is a critical challenge in the safety-critical system design flow. Existing verification methods often lack scalability or fail to handle the architectural heterogeneity introduced by pruning. In this work, we propose SimCert, a probabilistic certification framework for verifying the behavioral similarity of compressed neural networks. Unlike worst-case analysis, SimCert provides quantitative safety guarantees with adjustable confidence levels. Our framework features: (1) A dual-network symbolic propagation method supporting both quantization and pruning; (2) A variance-aware bounding technique using Bernstein's inequality to tighten safety certificates; and (3) An automated verification toolchain. Experimental results on ACAS Xu and computer vision benchmarks demonstrate that SimCert outperforms state-of-the-art baselines.

-> rk3588 같은 리소스 제약이 있는 엣지 디바이스에 AI 모델을 효율적으로 배포하고 검증하는 데 필수적

### LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11605v1
- 점수: final 84.0

Human motion is highly expressive and naturally aligned with language, yet prevailing methods relying heavily on joint text-motion embeddings struggle to synthesize temporally accurate, detailed motions and often lack explainability. To address these limitations, we introduce LabanLite, a motion representation developed by adapting and extending the Labanotation system. Unlike black-box text-motion embeddings, LabanLite encodes each atomic body-part action (e.g., a single left-foot step) as a discrete Laban symbol paired with a textual template. This abstraction decomposes complex motions into interpretable symbol sequences and body-part instructions, establishing a symbolic link between high-level language and low-level motion trajectories. Building on LabanLite, we present LaMoGen, a Text-to-LabanLite-to-Motion Generation framework that enables large language models (LLMs) to compose motion sequences through symbolic reasoning. The LLM interprets motion patterns, relates them to textual descriptions, and recombines symbols into executable plans, producing motions that are both interpretable and linguistically grounded. To support rigorous evaluation, we introduce a Labanotation-based benchmark with structured description-motion pairs and three metrics that jointly measure text-motion alignment across symbolic, temporal, and harmony dimensions. Experiments demonstrate that LaMoGen establishes a new baseline for both interpretability and controllability, outperforming prior methods on our benchmark and two public datasets. These results highlight the advantages of symbolic reasoning and agent-based design for language-driven motion synthesis.

-> 언어 기반 동생성 기술은 스포츠 동작 분석과 하이라이트 영상 제작에 활용 가능합니다

### Anchor then Polish for Low-light Enhancement (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15472v1
- 점수: final 82.4

Low-light image enhancement is challenging due to entangled degradations, mainly including poor illumination, color shifts, and texture interference. Existing methods often rely on complex architectures to address these issues jointly but may overfit simple physical constraints, leading to global distortions. This work proposes a novel anchor-then-polish (ATP) framework to fundamentally decouple global energy alignment from local detail refinement. First, macro anchoring is customized to (greatly) stabilize luminance distribution and correct color by learning a scene-adaptive projection matrix with merely 12 degrees of freedom, revealing that a simple linear operator can effectively align global energy. The macro anchoring then reduces the task to micro polishing, which further refines details in the wavelet domain and chrominance space under matrix guidance. A constrained luminance update strategy is designed to ensure global consistency while directing the network to concentrate on fine-grained polishing. Extensive experiments on multiple benchmarks show that our method achieves state-of-the-art performance, producing visually natural and quantitatively superior low-light enhancements.

-> Low-light enhancement technology directly applicable for improving sports footage in challenging lighting conditions

### GATE-AD: Graph Attention Network Encoding For Few-Shot Industrial Visual Anomaly Detection (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15300v1
- 점수: final 82.4

Few-Shot Industrial Visual Anomaly Detection (FS-IVAD) comprises a critical task in modern manufacturing settings, where automated product inspection systems need to identify rare defects using only a handful of normal/defect-free training samples. In this context, the current study introduces a novel reconstruction-based approach termed GATE-AD. In particular, the proposed framework relies on the employment of a masked, representation-aligned Graph Attention Network (GAT) encoding scheme to learn robust appearance patterns of normal samples. By leveraging dense, patch-level, visual feature tokens as graph nodes, the model employs stacked self-attentional layers to adaptively encode complex, irregular, non-Euclidean, local relations. The graph is enhanced with a representation alignment component grounded on a learnable, latent space, where high reconstruction residual areas (i.e., defects) are assessed using a Scaled Cosine Error (SCE) objective function. Extensive comparative evaluation on the MVTec AD, VisA, and MPDD industrial defect detection benchmarks demonstrates that GATE-AD achieves state-of-the-art performance across the $1$- to $8$-shot settings, combining the highest detection accuracy (increase up to $1.8\%$ in image AUROC in the 8-shot case in MPDD) with the lowest per-image inference latency (at least $25.05\%$ faster), compared to the best-performing literature methods. In order to facilitate reproducibility and further research, the source code of GATE-AD is available at https://github.com/gthpapadopoulos/GATE-AD.

-> Visual analysis techniques could be adapted for sports movement analysis

### Multi-turn Physics-informed Vision-language Model for Physics-grounded Anomaly Detection (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15237v1
- 점수: final 82.4

Vision-Language Models (VLMs) demonstrate strong general-purpose reasoning but remain limited in physics-grounded anomaly detection, where causal understanding of dynamics is essential. Existing VLMs, trained predominantly on appearance-centric correlations, fail to capture kinematic constraints, leading to poor performance on anomalies such as irregular rotations or violated mechanical motions. We introduce a physics-informed instruction tuning framework that explicitly encodes object properties, motion paradigms, and dynamic constraints into structured prompts. By delivering these physical priors through multi-turn dialogues, our method decomposes causal reasoning into incremental steps, enabling robust internal representations of normal and abnormal dynamics. Evaluated on the Phys-AD benchmark, our approach achieves 96.7% AUROC in video-level detection--substantially outperforming prior SOTA (66.9%)--and yields superior causal explanations (0.777 LLM score). This work highlights how structured physics priors can transform VLMs into reliable detectors of dynamic anomalies.

-> 물리 기반 비디오 이상 감지 기술로 스포츠 동작 및 전략 분석에 적용 가능

### Video Detector: A Dual-Phase Vision-Based System for Real-Time Traffic Intersection Control and Intelligent Transportation Analysis (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.14861v1
- 점수: final 82.4

Urban traffic management increasingly requires intelligent sensing systems capable of adapting to dynamic traffic conditions without costly infrastructure modifications. Vision-based vehicle detection has therefore become a key technology for modern intelligent transportation systems. This study presents Video Detector (VD), a dual-phase vision-based traffic intersection management system designed as a flexible and cost-effective alternative to traditional inductive loop detectors. The framework integrates a real-time module (VD-RT) for intersection control with an offline analytical module (VD-Offline) for detailed traffic behavior analysis. Three system configurations were implemented using SSD Inception v2, Faster R-CNN Inception v2, and CenterNet ResNet-50 V1 FPN, trained on datasets totaling 108,000 annotated images across 6-10 vehicle classes. Experimental results show detection performance of up to 90% test accuracy and 29.5 mAP@0.5, while maintaining real-time throughput of 37 FPS on HD video streams. Field deployments conducted in collaboration with Istanbul IT and Smart City Technologies Inc. (ISBAK) demonstrate stable operation under diverse environmental conditions. The system supports virtual loop detection, vehicle counting, multi-object tracking, queue estimation, speed analysis, and multiclass vehicle classification, enabling comprehensive intersection monitoring without the need for embedded road sensors. The annotated dataset and training pipeline are publicly released to support reproducibility. These results indicate that the proposed framework provides a scalable and deployable vision-based solution for intelligent transportation systems and smart-city traffic management.

-> 실시간 비전 처리 시스템으로 스포츠 영상 분석에 직접 적용 가능하며, 객체 탐지 및 추적 기술은 선수 추적에 활용될 수 있습니다.

### IRIS: Intersection-aware Ray-based Implicit Editable Scenes (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15368v1
- 점수: final 80.0

Neural Radiance Fields achieve high-fidelity scene representation but suffer from costly training and rendering, while 3D Gaussian splatting offers real-time performance with strong empirical results. Recently, solutions that harness the best of both worlds by using Gaussians as proxies to guide neural field evaluations, still suffer from significant computational inefficiencies. They typically rely on stochastic volumetric sampling to aggregate features, which severely limits rendering performance. To address this issue, a novel framework named IRIS (Intersection-aware Ray-based Implicit Editable Scenes) is introduced as a method designed for efficient and interactive scene editing. To overcome the limitations of standard ray marching, an analytical sampling strategy is employed that precisely identifies interaction points between rays and scene primitives, effectively eliminating empty space processing. Furthermore, to address the computational bottleneck of spatial neighbor lookups, a continuous feature aggregation mechanism is introduced that operates directly along the ray. By interpolating latent attributes from sorted intersections, costly 3D searches are bypassed, ensuring geometric consistency, enabling high-fidelity, real-time rendering, and flexible shape editing. Code can be found at https://github.com/gwilczynski95/iris.

-> 스포츠 하이라이트 영상 제작에 효율적인 장면 편집과 실시간 렌더링 기술 적용 가능

### Exposing Cross-Modal Consistency for Fake News Detection in Short-Form Videos (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.14992v1
- 점수: final 80.0

Short-form video platforms are major channels for news but also fertile ground for multimodal misinformation where each modality appears plausible alone yet cross-modal relationships are subtly inconsistent, like mismatched visuals and captions. On two benchmark datasets, FakeSV (Chinese) and FakeTT (English), we observe a clear asymmetry: real videos exhibit high text-visual but moderate text-audio consistency, while fake videos show the opposite pattern. Moreover, a single global consistency score forms an interpretable axis along which fake probability and prediction errors vary smoothly. Motivated by these observations, we present MAGIC3 (Modal-Adversarial Gated Interaction and Consistency-Centric Classifier), a detector that explicitly models and exposes cross-tri-modal consistency signals at multiple granularities. MAGIC3 combines explicit pairwise and global consistency modeling with token- and frame-level consistency signals derived from cross-modal attention, incorporates multi-style LLM rewrites to obtain style-robust text representations, and employs an uncertainty-aware classifier for selective VLM routing. Using pre-extracted features, MAGIC3 consistently outperforms the strongest non-VLM baselines on FakeSV and FakeTT. While matching VLM-level accuracy, the two-stage system achieves 18-27x higher throughput and 93% VRAM savings, offering a strong cost-performance tradeoff.

-> 숏폼 콘텐츠의 진위 여부 판단 기술로 스포츠 영상의 신뢰성 검증에 활용 가능

### Learning Latent Proxies for Controllable Single-Image Relighting (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.15555v1
- 점수: final 80.0

Single-image relighting is highly under-constrained: small illumination changes can produce large, nonlinear variations in shading, shadows, and specularities, while geometry and materials remain unobserved. Existing diffusion-based approaches either rely on intrinsic or G-buffer pipelines that require dense and fragile supervision, or operate purely in latent space without physical grounding, making fine-grained control of direction, intensity, and color unreliable. We observe that a full intrinsic decomposition is unnecessary and redundant for accurate relighting. Instead, sparse but physically meaningful cues, indicating where illumination should change and how materials should respond, are sufficient to guide a diffusion model. Based on this insight, we introduce LightCtrl that integrates physical priors at two levels: a few-shot latent proxy encoder that extracts compact material-geometry cues from limited PBR supervision, and a lighting-aware mask that identifies sensitive illumination regions and steers the denoiser toward shading relevant pixels. To compensate for scarce PBR data, we refine the proxy branch using a DPO-based objective that enforces physical consistency in the predicted cues. We also present ScaLight, a large-scale object-level dataset with systematically varied illumination and complete camera-light metadata, enabling physically consistent and controllable training. Across object and scene level benchmarks, our method achieves photometrically faithful relighting with accurate continuous control, surpassing prior diffusion and intrinsic-based baselines, including gains of up to +2.4 dB PSNR and 35% lower RMSE under controlled lighting shifts.

-> 이미지 보정 기술로 촬영된 스포츠 영상의 품질 향상에 직접 적용 가능

### BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.12176v1
- 점수: final 80.0

Understanding freely moving animal behavior is central to neuroscience, where pose estimation and behavioral understanding form the foundation for linking neural activity to natural actions. Yet both tasks still depend heavily on human annotation or unstable unsupervised pipelines, limiting scalability and reproducibility. We present BehaviorVLM, a unified vision-language framework for pose estimation and behavioral understanding that requires no task-specific finetuning and minimal human labeling by guiding pretrained Vision-Language Models (VLMs) through detailed, explicit, and verifiable reasoning steps. For pose estimation, we leverage quantum-dot-grounded behavioral data and propose a multi-stage pipeline that integrates temporal, spatial, and cross-view reasoning. This design greatly reduces human annotation effort, exposes low-confidence labels through geometric checks such as reprojection error, and produces labels that can later be filtered, corrected, or used to fine-tune downstream pose models. For behavioral understanding, we propose a pipeline that integrates deep embedded clustering for over-segmented behavior discovery, VLM-based per-clip video captioning, and LLM-based reasoning to merge and semantically label behavioral segments. The behavioral pipeline can operate directly from visual information and does not require keypoints to segment behavior. Together, these components enable scalable, interpretable, and label-light analysis of multi-animal behavior.

-> 동물 행동 이해 기술로 스포츠 분석에 간접적으로 적용 가능

### Derain-Agent: A Plug-and-Play Agent Framework for Rainy Image Restoration (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.11866v1
- 점수: final 80.0

While deep learning has advanced single-image deraining, existing models suffer from a fundamental limitation: they employ a static inference paradigm that fails to adapt to the complex, coupled degradations (e.g., noise artifacts, blur, and color deviation) of real-world rain. Consequently, restored images often exhibit residual artifacts and inconsistent perceptual quality. In this work, we present Derain-Agent, a plug-and-play refinement framework that transitions deraining from static processing to dynamic, agent-based restoration. Derain-Agent equips a base deraining model with two core capabilities: 1) a Planning Network that intelligently schedules an optimal sequence of restoration tools for each instance, and 2) a Strength Modulation mechanism that applies these tools with spatially adaptive intensity. This design enables precise, region-specific correction of residual errors without the prohibitive cost of iterative search. Our method demonstrates strong generalization, consistently boosting the performance of state-of-the-art deraining models on both synthetic and real-world benchmarks.

-> 스포츠 촬영 시 발생하는 다양한 날씨 조건에서 이미지 품질 향상 가능

### GeoNVS: Geometry Grounded Video Diffusion for Novel View Synthesis (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.14965v1
- 점수: final 80.0

Novel view synthesis requires strong 3D geometric consistency and the ability to generate visually coherent images across diverse viewpoints. While recent camera-controlled video diffusion models show promising results, they often suffer from geometric distortions and limited camera controllability. To overcome these challenges, we introduce GeoNVS, a geometry-grounded novel-view synthesizer that enhances both geometric fidelity and camera controllability through explicit 3D geometric guidance. Our key innovation is the Gaussian Splat Feature Adapter (GS-Adapter), which lifts input-view diffusion features into 3D Gaussian representations, renders geometry-constrained novel-view features, and adaptively fuses them with diffusion features to correct geometrically inconsistent representations. Unlike prior methods that inject geometry at the input level, GS-Adapter operates in feature space, avoiding view-dependent color noise that degrades structural consistency. Its plug-and-play design enables zero-shot compatibility with diverse feed-forward geometry models without additional training, and can be adapted to other video diffusion backbones. Experiments across 9 scenes and 18 settings demonstrate state-of-the-art performance, achieving 11.3% and 14.9% improvements over SEVA and CameraCtrl, with up to 2x reduction in translation error and 7x in Chamfer Distance.

-> 영상의 새로운 각도 생성 기술로 스포츠 촬영에 간접적 적용 가능하나 스포츠/엣지 특화 아님

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
