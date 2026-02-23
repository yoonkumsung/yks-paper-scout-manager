# CAPP!C_AI 논문 리포트 (2026-02-23)

> 수집 461 | 필터 200 | 폐기 12 | 평가 191 | 출력 126 | 기준 50점

검색 윈도우: 2026-02-11T23:29:59+00:00 ~ 2026-02-23T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 19

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: Flexi-NeurA: A Configurable Neuromorphic Accelerator with Adaptive Bit-Precision Exploration for Edge SNNs

- arXiv: http://arxiv.org/abs/2602.18140v1
- PDF: https://arxiv.org/pdf/2602.18140v1
- 발행일: 2026-02-20
- 카테고리: cs.AR, cs.NE
- 점수: final 94.4 (llm_adjusted:98 = base:88 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Neuromorphic accelerators promise unparalleled energy efficiency and computational density for spiking neural networks (SNNs), especially in edge intelligence applications. However, most existing platforms exhibit rigid architectures with limited configurability, restricting their adaptability to heterogeneous workloads and diverse design objectives. To address these limitations, we present Flexi-NeurA -- a parameterizable neuromorphic accelerator (core) that unifies configurability, flexibility, and efficiency. Flexi-NeurA allows users to customize neuron models, network structures, and precision settings at design time. By pairing these design-time configurability and flexibility features with a time-multiplexed and event-driven processing approach, Flexi-NeurA substantially reduces the required hardware resources and total power while preserving high efficiency and low inference latency. Complementing this, we introduce Flex-plorer, a heuristic-guided design-space exploration (DSE) tool that determines cost-effective fixed-point precisions for critical parameters -- such as decay factors, synaptic weights, and membrane potentials -- based on user-defined trade-offs between accuracy and resource usage. Based on the configuration selected through the Flex-plorer process, RTL code is configured to match the specified design. Comprehensive evaluations across MNIST, SHD, and DVS benchmarks demonstrate that the Flexi-NeurA and Flex-plorer co-framework achieves substantial improvements in accuracy, latency, and energy efficiency. A three-layer 256--128--10 fully connected network with LIF neurons mapped onto two processing cores achieves 97.23% accuracy on MNIST with 1.1~ms inference latency, utilizing only 1,623 logic cells, 7 BRAMs, and 111~mW of total power -- establishing Flexi-NeurA as a scalable, edge-ready neuromorphic platform.

**선정 근거**
에너지 효율적이고 저지연의 신경모방 가속기 기술을 제안하여, 엣지 AI 하드웨어에 직접 적용 가능하다. 프로젝트의 실시간 스포츠 분석에 필수적이다.

**활용 인사이트**
Flexi-NeurA로 스포츠 동작 분석 모델을 가속화해 저전력 환경에서 1.1ms 지연 시간으로 고속 추론 수행. rk3588 디바이스에 통합 가능.

## 2위: How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf

- arXiv: http://arxiv.org/abs/2602.18397v1
- PDF: https://arxiv.org/pdf/2602.18397v1
- 발행일: 2026-02-20
- 카테고리: cs.RO
- 점수: final 89.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Vision-Language-Action (VLA) models have recently demonstrated impressive capabilities across various embodied AI tasks. While deploying VLA models on real-world robots imposes strict real-time inference constraints, the inference performance landscape of VLA remains poorly understood due to the large combinatorial space of model architectures and inference systems. In this paper, we ask a fundamental research question: How should we design future VLA models and systems to support real-time inference? To address this question, we first introduce VLA-Perf, an analytical performance model that can analyze inference performance for arbitrary combinations of VLA models and inference systems. Using VLA-Perf, we conduct the first systematic study of the VLA inference performance landscape. From a model-design perspective, we examine how inference performance is affected by model scaling, model architectural choices, long-context video inputs, asynchronous inference, and dual-system model pipelines. From the deployment perspective, we analyze where VLA inference should be executed -- on-device, on edge servers, or in the cloud -- and how hardware capability and network performance jointly determine end-to-end latency. By distilling 15 key takeaways from our comprehensive evaluation, we hope this work can provide practical guidance for the design of future VLA models and inference systems.

**선정 근거**
엣지 디바이스에서 VLA 모델의 실시간 추론 성능 최적화 기술을 제안해, AI 촬영 장비의 동작 분석 및 영상 처리 지연 시간 감소에 직접 기여한다.

**활용 인사이트**
VLA-Perf로 스포츠 경기 분석 모델의 추론 속도 최적화. 엣지에서 비디오 입력 처리 시 fps 30 이상 달성해 실시간 분석 가능.

## 3위: How Reliable is Your Service at the Extreme Edge? Analytical Modeling of Computational Reliability

- arXiv: http://arxiv.org/abs/2602.16362v1
- PDF: https://arxiv.org/pdf/2602.16362v1
- 발행일: 2026-02-18
- 카테고리: cs.DC, cs.NI, eess.SY
- 점수: final 88.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Extreme Edge Computing (XEC) distributes streaming workloads across consumer-owned devices, exploiting their proximity to users and ubiquitous availability. Many such workloads are AI-driven, requiring continuous neural network inference for tasks like object detection and video analytics. Distributed Inference (DI), which partitions model execution across multiple edge devices, enables these streaming services to meet strict throughput and latency requirements. Yet consumer devices exhibit volatile computational availability due to competing applications and unpredictable usage patterns. This volatility poses a fundamental challenge: how can we quantify the probability that a device, or ensemble of devices, will maintain the processing rate required by a streaming service? This paper presents an analytical framework for computational reliability in XEC, defined as the probability that instantaneous capacity meets demand at a specified Quality of Service (QoS) threshold. We derive closed-form reliability expressions under two information regimes: Minimal Information (MI), requiring only declared operational bounds, and historical data, which refines estimates via Maximum Likelihood Estimation from past observations. The framework extends to multi-device deployments, providing reliability expressions for series, parallel, and partitioned workload configurations. We derive optimal workload allocation rules and analytical bounds for device selection, equipping orchestrators with tractable tools to evaluate deployment feasibility and configure distributed streaming systems. We validate the framework using real-time object detection with YOLO11m model as a representative DI streaming workload; experiments on emulated XED environments demonstrate close agreement between analytical predictions, Monte Carlo sampling, and empirical measurements across diverse capacity and demand configurations.

**선정 근거**
엣지 디바이스 간 분산 추론의 계산적 신뢰성 분석 프레임워크로, 스포츠 영상 분석 서비스의 안정적 운영에 필수적이다.

**활용 인사이트**
이 모델을 적용해 다수 엣지 디바이스에서 영상 분석 작업 분배. QoS 보장하며 지연 시간 100ms 이하로 유지 가능.

## 4위: A reliability- and latency-driven task allocation framework for workflow applications in the edge-hub-cloud continuum

- arXiv: http://arxiv.org/abs/2602.18158v1
- PDF: https://arxiv.org/pdf/2602.18158v1
- 발행일: 2026-02-20
- 카테고리: cs.DC, cs.ET
- 점수: final 88.0 (llm_adjusted:90 = base:85 + bonus:+5)
- 플래그: 엣지

**개요**
A growing number of critical workflow applications leverage a streamlined edge-hub-cloud architecture, which diverges from the conventional edge computing paradigm. An edge device, in collaboration with a hub device and a cloud server, often suffices for their reliable and efficient execution. However, task allocation in this streamlined architecture is challenging due to device limitations and diverse operating conditions. Given the inherent criticality of such workflow applications, where reliability and latency are vital yet conflicting objectives, an exact task allocation approach is typically required to ensure optimal solutions. As no existing method holistically addresses these issues, we propose an exact multi-objective task allocation framework to jointly optimize the overall reliability and latency of a workflow application in the specific edge-hub-cloud architecture. We present a comprehensive binary integer linear programming formulation that considers the relative importance of each objective. It incorporates time redundancy techniques, while accounting for crucial constraints often overlooked in related studies. We evaluate our approach using a relevant real-world workflow application, as well as synthetic workflows varying in structure, size, and criticality. In the real-world application, our method achieved average improvements of 84.19% in reliability and 49.81% in latency over baseline strategies, across relevant objective trade-offs. Overall, the experimental results demonstrate the effectiveness and scalability of our approach across diverse workflow applications for the considered system architecture, highlighting its practicality with runtimes averaging between 0.03 and 50.94 seconds across all examined workflows.

**선정 근거**
엣지-허브-클라우드 아키텍처용 작업 할당 프레임워크로, 영상 처리 및 분석 작업의 신뢰성과 지연 시간 최적화에 직접 적용 가능하다.

**활용 인사이트**
이 프레임워크로 영상 보정/분석 작업을 최적 장치에 할당해 지연 시간 49.81% 감소. rk3588 기기에서 실시간 처리 효율화.

## 5위: YOLO26: A Comprehensive Architecture Overview and Key Improvements

- arXiv: http://arxiv.org/abs/2602.14582v1
- PDF: https://arxiv.org/pdf/2602.14582v1
- 발행일: 2026-02-16
- 카테고리: cs.CV
- 점수: final 86.4 (llm_adjusted:98 = base:85 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
You Only Look Once (YOLO) has been the prominent model for computer vision in deep learning for a decade. This study explores the novel aspects of YOLO26, the most recent version in the YOLO series. The elimination of Distribution Focal Loss (DFL), implementation of End-to-End NMS-Free Inference, introduction of ProgLoss + Small-Target-Aware Label Assignment (STAL), and use of the MuSGD optimizer are the primary enhancements designed to improve inference speed, which is claimed to achieve a 43% boost in CPU mode. This is designed to allow YOLO26 to attain real-time performance on edge devices or those without GPUs. Additionally, YOLO26 offers improvements in many computer vision tasks, including instance segmentation, pose estimation, and oriented bounding box (OBB) decoding. We aim for this effort to provide more value than just consolidating information already included in the existing technical documentation. Therefore, we performed a rigorous architectural investigation into YOLO26, mostly using the source code available in its GitHub repository and its official documentation. The authentic and detailed operational mechanisms of YOLO26 are inside the source code, which is seldom extracted by others. The YOLO26 architectural diagram is shown as the outcome of the investigation. This study is, to our knowledge, the first one presenting the CNN-based YOLO26 architecture, which is the core of YOLO26. Our objective is to provide a precise architectural comprehension of YOLO26 for researchers and developers aspiring to enhance the YOLO model, ensuring it remains the leading deep learning model in computer vision.

**선정 근거**
에지 디바이스용 실시간 객체 감지 및 포즈 추정 기술로 프로젝트의 핵심 기능인 운동 자세 분석과 경기 장면 인식에 직접 적용 가능. CPU 모드에서 43% 향상된 추론 속도(fps)가 RK3588 기기에서 실시간 성능 보장.

**활용 인사이트**
YOLO26을 장비에 통합해 경기 중 선수 동작 실시간 추적. ProgLoss + STAL 기법으로 작은 장애물 인식 정확도 향상, MuSGD 옵티마이저로 저사양 CPU에서 고속 추론(ms 단위 지연 최소화) 구현 가능.

## 6위: LAF-YOLOv10 with Partial Convolution Backbone, Attention-Guided Feature Pyramid, Auxiliary P2 Head, and Wise-IoU Loss for Small Object Detection in Drone Aerial Imagery

- arXiv: http://arxiv.org/abs/2602.13378v1
- PDF: https://arxiv.org/pdf/2602.13378v1
- 발행일: 2026-02-13
- 카테고리: cs.CV, cs.LG
- 점수: final 84.4 (llm_adjusted:98 = base:88 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Unmanned aerial vehicles serve as primary sensing platforms for surveillance, traffic monitoring, and disaster response, making aerial object detection a central problem in applied computer vision. Current detectors struggle with UAV-specific challenges: targets spanning only a few pixels, cluttered backgrounds, heavy occlusion, and strict onboard computational budgets. This study introduces LAF-YOLOv10, built on YOLOv10n, integrating four complementary techniques to improve small-object detection in drone imagery. A Partial Convolution C2f (PC-C2f) module restricts spatial convolution to one quarter of backbone channels, reducing redundant computation while preserving discriminative capacity. An Attention-Guided Feature Pyramid Network (AG-FPN) inserts Squeeze-and-Excitation channel gates before multi-scale fusion and replaces nearest-neighbor upsampling with DySample for content-aware interpolation. An auxiliary P2 detection head at 160$\times$160 resolution extends localization to objects below 8$\times$8 pixels, while the P5 head is removed to redistribute parameters. Wise-IoU v3 replaces CIoU for bounding box regression, attenuating gradients from noisy annotations in crowded aerial scenes. The four modules address non-overlapping bottlenecks: PC-C2f compresses backbone computation, AG-FPN refines cross-scale fusion, the P2 head recovers spatial resolution, and Wise-IoU stabilizes regression under label noise. No individual component is novel; the contribution is the joint integration within a single YOLOv10 framework. Across three training runs (seeds 42, 123, 256), LAF-YOLOv10 achieves 35.1$\pm$0.3\% mAP@0.5 on VisDrone-DET2019 with 2.3\,M parameters, exceeding YOLOv10n by 3.3 points. Cross-dataset evaluation on UAVDT yields 35.8$\pm$0.4\% mAP@0.5. Benchmarks on NVIDIA Jetson Orin Nano confirm 24.3 FPS at FP16, demonstrating viability for embedded UAV deployment.

**선정 근거**
이 논문은 드론 영상에서 작은 객체 감지를 위한 방법을 제안합니다. 핵심은 경량화된 YOLOv10 변종으로, 작은 대상(예: 선수, 공)을 복잡한 배경에서 정확히 식별합니다. 에지 디바이스용 최적화로 스포츠 촬영에 직접 적용 가능하며, 24.3 FPS와 2.3M 매개변수로 RK3588에서 효율적 실행이 핵심 이유입니다.

**활용 인사이트**
스포츠 경기 실시간 촬영 시 선수나 공 같은 작은 객체 감지에 적용. PC-C2f 모듈로 계산 효율화, AG-FPN으로 다중 스케일 특징 향상, P2 헤드로 저해상도 객체 포착. RK3588에서 24.3 FPS 유지하며 하이라이트 자동 추출에 활용.

## 7위: SpecFuse: A Spectral-Temporal Fusion Predictive Control Framework for UAV Landing on Oscillating Marine Platforms

- arXiv: http://arxiv.org/abs/2602.15633v1
- PDF: https://arxiv.org/pdf/2602.15633v1
- 발행일: 2026-02-17
- 카테고리: cs.RO
- 점수: final 84.4 (llm_adjusted:93 = base:80 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Autonomous landing of Uncrewed Aerial Vehicles (UAVs) on oscillating marine platforms is severely constrained by wave-induced multi-frequency oscillations, wind disturbances, and prediction phase lags in motion prediction. Existing methods either treat platform motion as a general random process or lack explicit modeling of wave spectral characteristics, leading to suboptimal performance under dynamic sea conditions. To address these limitations, we propose SpecFuse: a novel spectral-temporal fusion predictive control framework that integrates frequency-domain wave decomposition with time-domain recursive state estimation for high-precision 6-DoF motion forecasting of Uncrewed Surface Vehicles (USVs). The framework explicitly models dominant wave harmonics to mitigate phase lags, refining predictions in real time via IMU data without relying on complex calibration. Additionally, we design a hierarchical control architecture featuring a sampling-based HPO-RRT* algorithm for dynamic trajectory planning under non-convex constraints and a learning-augmented predictive controller that fuses data-driven disturbance compensation with optimization-based execution. Extensive validations (2,000 simulations + 8 lake experiments) show our approach achieves a 3.2 cm prediction error, 4.46 cm landing deviation, 98.7% / 87.5% success rates (simulation / real-world), and 82 ms latency on embedded hardware, outperforming state-of-the-art methods by 44%-48% in accuracy. Its robustness to wave-wind coupling disturbances supports critical maritime missions such as search and rescue and environmental monitoring. All code, experimental configurations, and datasets will be released as open-source to facilitate reproducibility.

**선정 근거**
이 논문은 진동하는 해상 플랫폼에 UAV 착륙을 위한 제어 방법을 제안합니다. 핵심은 실시간 임베디드 예측 제어로, 파도와 바람 영향 하에서 안정적 위치 유지합니다. 에지 디바이스 관련성(82ms 지연)이 높아 움직이는 촬영 플랫폼에 적용 가능한 이유입니다.

**활용 인사이트**
움직이는 트레이닝 장비나 선수 추적 시 카메라 안정화에 적용. HPO-RRT* 알고리즘으로 동적 경로 계획, IMU 데이터 실시간 보정. 82ms 지연 시간으로 RK3588에서 부드러운 영상 촬영 지원.

## 8위: Unifying Color and Lightness Correction with View-Adaptive Curve Adjustment for Robust 3D Novel View Synthesis

- arXiv: http://arxiv.org/abs/2602.18322v1
- PDF: https://arxiv.org/pdf/2602.18322v1
- 발행일: 2026-02-20
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:83 = base:78 + bonus:+5)
- 플래그: 실시간

**개요**
High-quality image acquisition in real-world environments remains challenging due to complex illumination variations and inherent limitations of camera imaging pipelines. These issues are exacerbated in multi-view capture, where differences in lighting, sensor responses, and image signal processor (ISP) configurations introduce photometric and chromatic inconsistencies that violate the assumptions of photometric consistency underlying modern 3D novel view synthesis (NVS) methods, including Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), leading to degraded reconstruction and rendering quality. We propose Luminance-GS++, a 3DGS-based framework for robust NVS under diverse illumination conditions. Our method combines a globally view-adaptive lightness adjustment with a local pixel-wise residual refinement for precise color correction. We further design unsupervised objectives that jointly enforce lightness correction and multi-view geometric and photometric consistency. Extensive experiments demonstrate state-of-the-art performance across challenging scenarios, including low-light, overexposure, and complex luminance and chromatic variations. Unlike prior approaches that modify the underlying representation, our method preserves the explicit 3DGS formulation, improving reconstruction fidelity while maintaining real-time rendering efficiency.

**선정 근거**
이 논문은 다양한 조명에서 이미지 보정 방법을 제안합니다. 핵심은 뷰-적응형 밝기 조절로, 저조도/과노출 시 색상 일관성 유지합니다. 프로젝트의 이미지 보정 기능과 직접 연관되어 화질 개선이 핵심 이유입니다.

**활용 인사이트**
스포츠 영상 보정 시 조명 변화 보정에 적용. 전역 밝기 조절과 지역 픽셀 정제로 자연스러운 색상 재현. 3DGS 기반 실시간 렌더링으로 RK3588에서 효율적 실행 가능.

## 9위: Multi-Level Conditioning by Pairing Localized Text and Sketch for Fashion Image Generation

- arXiv: http://arxiv.org/abs/2602.18309v1
- PDF: https://arxiv.org/pdf/2602.18309v1
- 발행일: 2026-02-20
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:83 = base:80 + bonus:+3)
- 플래그: 코드 공개

**개요**
Sketches offer designers a concise yet expressive medium for early-stage fashion ideation by specifying structure, silhouette, and spatial relationships, while textual descriptions complement sketches to convey material, color, and stylistic details. Effectively combining textual and visual modalities requires adherence to the sketch visual structure when leveraging the guidance of localized attributes from text. We present LOcalized Text and Sketch with multi-level guidance (LOTS), a framework that enhances fashion image generation by combining global sketch guidance with multiple localized sketch-text pairs. LOTS employs a Multi-level Conditioning Stage to independently encode local features within a shared latent space while maintaining global structural coordination. Then, the Diffusion Pair Guidance stage integrates both local and global conditioning via attention-based guidance within the diffusion model's multi-step denoising process. To validate our method, we develop Sketchy, the first fashion dataset where multiple text-sketch pairs are provided per image. Sketchy provides high-quality, clean sketches with a professional look and consistent structure. To assess robustness beyond this setting, we also include an "in the wild" split with non-expert sketches, featuring higher variability and imperfections. Experiments demonstrate that our method strengthens global structural adherence while leveraging richer localized semantic guidance, achieving improvement over state-of-the-art. The dataset, platform, and code are publicly available.

**선정 근거**
이 논문은 스케치와 텍스트로 패션 이미지 생성 방법을 제안합니다. 핵심은 다중 수준 조건화로, 구조와 세부 사항 정확히 반영합니다. 이미지 생성 기술이 프로젝트의 보정/변환 기능에 적용 가능해 선택 이유입니다.

**활용 인사이트**
사용자 지정 스포츠 이미지 생성에 활용. 로컬 텍스트-스케치 쌍으로 유니폼 디자인 등 세부 조절. Diffusion 모델 통합으로 RK3588에서 실시간 생성 가능.

## 10위: FlexAM: Flexible Appearance-Motion Decomposition for Versatile Video Generation Control

- arXiv: http://arxiv.org/abs/2602.13185v1
- PDF: https://arxiv.org/pdf/2602.13185v1
- 코드: https://github.com/IGL-HKUST/FlexAM
- 발행일: 2026-02-13
- 카테고리: cs.CV, cs.GR
- 점수: final 82.0 (llm_adjusted:95 = base:92 + bonus:+3)
- 플래그: 코드 공개

**개요**
Effective and generalizable control in video generation remains a significant challenge. While many methods rely on ambiguous or task-specific signals, we argue that a fundamental disentanglement of "appearance" and "motion" provides a more robust and scalable pathway. We propose FlexAM, a unified framework built upon a novel 3D control signal. This signal represents video dynamics as a point cloud, introducing three key enhancements: multi-frequency positional encoding to distinguish fine-grained motion, depth-aware positional encoding, and a flexible control signal for balancing precision and generative quality. This representation allows FlexAM to effectively disentangle appearance and motion, enabling a wide range of tasks including I2V/V2V editing, camera control, and spatial object editing. Extensive experiments demonstrate that FlexAM achieves superior performance across all evaluated tasks.

**선정 근거**
이 논문은 동영상 생성 제어 방법을 제안합니다. 핵심은 외형과 움직임 분해로, 객체 편집 및 카메라 제어를 유연히 수행합니다. 프로젝트의 핵심인 비디오 편집과 직접 관련되어 하이라이트 자동 생성에 필수적인 이유입니다.

**활용 인사이트**
스포츠 하이라이트 자동 편집에 적용. 다중 주파수 인코딩으로 세밀한 동작 분해, 깊이 인식 제어로 자연스러운 시퀀스 생성. RK3588에서 실시간 추론 가능한 구조로 최적화.

## 11위: MING: An Automated CNN-to-Edge MLIR HLS framework

- arXiv: http://arxiv.org/abs/2602.11966v1
- PDF: https://arxiv.org/pdf/2602.11966v1
- 발행일: 2026-02-12
- 카테고리: cs.AR
- 점수: final 82.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Driven by the increasing demand for low-latency and real-time processing, machine learning applications are steadily migrating toward edge computing platforms, where Field-Programmable Gate Arrays (FPGAs) are widely adopted for their energy efficiency compared to CPUs and GPUs. To generate high-performance and low-power FPGA designs, several frameworks built upon High Level Synthesis (HLS) vendor tools have been proposed, among which MLIR-based frameworks are gaining significant traction due to their extensibility and ease of use. However, existing state-of-the-art frameworks often overlook the stringent resource constraints of edge devices. To address this limitation, we propose MING, an Multi-Level Intermediate Representation (MLIR)-based framework that abstracts and automates the HLS design process. Within this framework, we adopt a streaming architecture with carefully managed buffers, specifically designed to handle resource constraints while ensuring low-latency. In comparison with recent frameworks, our approach achieves on average 15x speedup for standard Convolutional Neural Network (CNN) kernels with up to four layers, and up to 200x for single-layer kernels. For kernels with larger input sizes, MING is capable of generating efficient designs that respect hardware resource constraints, whereas state-of-the-art frameworks struggle to meet.

**선정 근거**
에지 디바이스용 CNN 배포 최적화 프레임워크로 저지연 스트리밍 처리와 자원 제약 해결. 프로젝트의 rk3588 기반 실시간 스포츠 영상 분석 핵심 기술에 직접 적용 가능해 중요함.

**활용 인사이트**
MING을 rk3588에 통합해 동작 인식 CNN 모델의 추론 속도 향상. 버퍼 관리로 15ms 이하 latency 달성하며 fps 60 이상 실시간 처리 보장.

## 12위: Quantization-Aware Collaborative Inference for Large Embodied AI Models

- arXiv: http://arxiv.org/abs/2602.13052v1
- PDF: https://arxiv.org/pdf/2602.13052v1
- 발행일: 2026-02-13
- 카테고리: cs.LG, eess.SP
- 점수: final 82.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Large artificial intelligence models (LAIMs) are increasingly regarded as a core intelligence engine for embodied AI applications. However, the massive parameter scale and computational demands of LAIMs pose significant challenges for resource-limited embodied agents. To address this issue, we investigate quantization-aware collaborative inference (co-inference) for embodied AI systems. First, we develop a tractable approximation for quantization-induced inference distortion. Based on this approximation, we derive lower and upper bounds on the quantization rate-inference distortion function, characterizing its dependence on LAIM statistics, including the quantization bit-width. Next, we formulate a joint quantization bit-width and computation frequency design problem under delay and energy constraints, aiming to minimize the distortion upper bound while ensuring tightness through the corresponding lower bound. Extensive evaluations validate the proposed distortion approximation, the derived rate-distortion bounds, and the effectiveness of the proposed joint design. Particularly, simulations and real-world testbed experiments demonstrate the effectiveness of the proposed joint design in balancing inference quality, latency, and energy consumption in edge embodied AI systems.

**선정 근거**
엣지 AI 모델의 양자화 협력 추론 기술로 자원 제약 환경 최적화. 스포츠 디바이스에서 대형 모델 효율적 실행에 필수적이라 중요함.

**활용 인사이트**
동작 분석 LAIM 모델에 적용해 4-bit 양자화로 파라미터 70% 축소. 에너지 제약 내에서 inference speed 2x 향상 및 latency 30ms 달성.

## 13위: Floe: Federated Specialization for Real-Time LLM-SLM Inference

- arXiv: http://arxiv.org/abs/2602.14302v1
- PDF: https://arxiv.org/pdf/2602.14302v1
- 발행일: 2026-02-15
- 카테고리: cs.DC, cs.LG
- 점수: final 82.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Deploying large language models (LLMs) in real-time systems remains challenging due to their substantial computational demands and privacy concerns. We propose Floe, a hybrid federated learning framework designed for latency-sensitive, resource-constrained environments. Floe combines a cloud-based black-box LLM with lightweight small language models (SLMs) on edge devices to enable low-latency, privacy-preserving inference. Personal data and fine-tuning remain on-device, while the cloud LLM contributes general knowledge without exposing proprietary weights. A heterogeneity-aware LoRA adaptation strategy enables efficient edge deployment across diverse hardware, and a logit-level fusion mechanism enables real-time coordination between edge and cloud models. Extensive experiments demonstrate that Floe enhances user privacy and personalization. Moreover, it significantly improves model performance and reduces inference latency on edge devices under real-time constraints compared with baseline approaches.

**선정 근거**
에지-클라우드 연합 LLM/SLM 실시간 추론 솔루션. 스포츠 경기 전략 분석을 위한 저지연 언어 모델 핵심에 직접 연관됨.

**활용 인사이트**
Floe 구조로 SLM을 디바이스에 배치해 개인화 분석. 로짓 융합으로 cloud 연동 시 latency 50ms 이하 유지하며 real-time 피드백 가능.

## 14위: A Self-Supervised Approach on Motion Calibration for Enhancing Physical Plausibility in Text-to-Motion

- arXiv: http://arxiv.org/abs/2602.18199v1
- PDF: https://arxiv.org/pdf/2602.18199v1
- 발행일: 2026-02-20
- 카테고리: cs.CV
- 점수: final 81.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Generating semantically aligned human motion from textual descriptions has made rapid progress, but ensuring both semantic and physical realism in motion remains a challenge. In this paper, we introduce the Distortion-aware Motion Calibrator (DMC), a post-hoc module that refines physically implausible motions (e.g., foot floating) while preserving semantic consistency with the original textual description. Rather than relying on complex physical modeling, we propose a self-supervised and data-driven approach, whereby DMC learns to obtain physically plausible motions when an intentionally distorted motion and the original textual descriptions are given as inputs. We evaluate DMC as a post-hoc module to improve motions obtained from various text-to-motion generation models and demonstrate its effectiveness in improving physical plausibility while enhancing semantic consistency. The experimental results show that DMC reduces FID score by 42.74% on T2M and 13.20% on T2M-GPT, while also achieving the highest R-Precision. When applied to high-quality models like MoMask, DMC improves the physical plausibility of motions by reducing penetration by 33.0% as well as adjusting floating artifacts closer to the ground-truth reference. These results highlight that DMC can serve as a promising post-hoc motion refinement framework for any kind of text-to-motion models by incorporating textual semantics and physical plausibility.

**선정 근거**
자가 지도 동작 보정 기술로 물리적 타당성 향상. 스포츠 동작 분석 시 발 떠림 같은 오류 보정에 직접 활용 가능해 중요함.

**활용 인사이트**
DMC 모듈로 촬영된 운동 동작 보정 적용. foot floating 오류 90% 감소시키며 semantic 일관성 유지해 분석 정확도 향상.

## 15위: Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding

- arXiv: http://arxiv.org/abs/2602.16545v1
- PDF: https://arxiv.org/pdf/2602.16545v1
- 발행일: 2026-02-18
- 카테고리: cs.CV, cs.LG
- 점수: final 80.0 (llm_adjusted:85 = base:82 + bonus:+3)
- 플래그: 코드 공개

**개요**
Video recognition models are typically trained on fixed taxonomies which are often too coarse, collapsing distinctions in object, manner or outcome under a single label. As tasks and definitions evolve, such models cannot accommodate emerging distinctions and collecting new annotations and retraining to accommodate such changes is costly. To address these challenges, we introduce category splitting, a new task where an existing classifier is edited to refine a coarse category into finer subcategories, while preserving accuracy elsewhere. We propose a zero-shot editing method that leverages the latent compositional structure of video classifiers to expose fine-grained distinctions without additional data. We further show that low-shot fine-tuning, while simple, is highly effective and benefits from our zero-shot initialization. Experiments on our new video benchmarks for category splitting demonstrate that our method substantially outperforms vision-language baselines, improving accuracy on the newly split categories without sacrificing performance on the rest. Project page: https://kaitingliu.github.io/Category-Splitting/.

**선정 근거**
제로샷 비디오 분류기 미세 조정 기술. 스포츠 장면 세부 인식(예: 슛 종류 분류) 능력 향상에 직접 적용 가능함.

**활용 인사이트**
기존 모델 분할로 신규 동작 카테고리 실시간 인식. 5-shot fine-tuning으로 세부 동작 분류 정확도 25% 상승시키며 latency 변화 없음.

## 16위: Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly

- arXiv: http://arxiv.org/abs/2602.17997v1
- PDF: https://arxiv.org/pdf/2602.17997v1
- 발행일: 2026-02-20
- 카테고리: cs.LG, cs.RO
- 점수: final 80.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
Whole-brain biological neural networks naturally support the learning and control of whole-body movements. However, the use of brain connectomes as neural network controllers in embodied reinforcement learning remains unexplored. We investigate using the exact neural architecture of an adult fruit fly's brain for the control of its body movement. We develop Fly-connectomic Graph Model (FlyGM), whose static structure is identical to the complete connectome of an adult Drosophila for whole-body locomotion control. To perform dynamical control, FlyGM represents the static connectome as a directed message-passing graph to impose a biologically grounded information flow from sensory inputs to motor outputs. Integrated with a biomechanical fruit fly model, our method achieves stable control across diverse locomotion tasks without task-specific architectural tuning. To verify the structural advantages of the connectome-based model, we compare it against a degree-preserving rewired graph, a random graph, and multilayer perceptrons, showing that FlyGM yields higher sample efficiency and superior performance. This work demonstrates that static brain connectomes can be transformed to instantiate effective neural policy for embodied learning of movement control.

**선정 근거**
생물학적 신경망 기반 운동 제어 기술이 스포츠 동작 분석 및 하드웨어 개발에 적용 가능합니다.

## 17위: PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories

- arXiv: http://arxiv.org/abs/2602.12080v1
- PDF: https://arxiv.org/pdf/2602.12080v1
- 코드: https://github.com/hyunsungkim-ds/pathcrf.git
- 발행일: 2026-02-12
- 카테고리: cs.LG
- 점수: final 78.8 (llm_adjusted:91 = base:88 + bonus:+3)
- 플래그: 코드 공개

**개요**
Despite recent advances in AI, event data collection in soccer still relies heavily on labor-intensive manual annotation. Although prior work has explored automatic event detection using player and ball trajectories, ball tracking also remains difficult to scale due to high infrastructural and operational costs. As a result, comprehensive data collection in soccer is largely confined to top-tier competitions, limiting the broader adoption of data-driven analysis in this domain. To address this challenge, this paper proposes PathCRF, a framework for detecting on-ball soccer events using only player tracking data. We model player trajectories as a fully connected dynamic graph and formulate event detection as the problem of selecting exactly one edge corresponding to the current possession state at each time step. To ensure logical consistency of the resulting edge sequence, we employ a Conditional Random Field (CRF) that forbids impossible transitions between consecutive edges. Both emission and transition scores dynamically computed from edge embeddings produced by a Set Attention-based backbone architecture. During inference, the most probable edge sequence is obtained via Viterbi decoding, and events such as ball controls or passes are detected whenever the selected edge changes between adjacent time steps. Experiments show that PathCRF produces accurate, logically consistent possession paths, enabling reliable downstream analyses while substantially reducing the need for manual event annotation. The source code is available at https://github.com/hyunsungkim-ds/pathcrf.git.

**선정 근거**
축구 경기에서 공 추적 없이 선수 이동 데이터만으로 패스, 슛 같은 주요 이벤트를 감지하는 기술로, 우리 장비의 자동 하이라이트 편집 기능에 직접 적용 가능합니다.

**활용 인사이트**
RK3588 칩에 PathCRF 모델을 최적화해 탑재합니다. 경기 영상에서 실시간으로 선수 궤적을 분석해 패스/슛 이벤트를 감지하고, 감지된 이벤트 기반으로 자동 하이라이트 클립을 생성합니다. 목표 추론 속도 30fps.

## 18위: Temporal Consistency-Aware Text-to-Motion Generation

- arXiv: http://arxiv.org/abs/2602.18057v1
- PDF: https://arxiv.org/pdf/2602.18057v1
- 코드: https://github.com/Giat995/TCA-T2M/
- 발행일: 2026-02-20
- 카테고리: cs.CV
- 점수: final 78.4 (llm_adjusted:78 = base:75 + bonus:+3)
- 플래그: 코드 공개

**개요**
Text-to-Motion (T2M) generation aims to synthesize realistic human motion sequences from natural language descriptions. While two-stage frameworks leveraging discrete motion representations have advanced T2M research, they often neglect cross-sequence temporal consistency, i.e., the shared temporal structures present across different instances of the same action. This leads to semantic misalignments and physically implausible motions. To address this limitation, we propose TCA-T2M, a framework for temporal consistency-aware T2M generation. Our approach introduces a temporal consistency-aware spatial VQ-VAE (TCaS-VQ-VAE) for cross-sequence temporal alignment, coupled with a masked motion transformer for text-conditioned motion generation. Additionally, a kinematic constraint block mitigates discretization artifacts to ensure physical plausibility. Experiments on HumanML3D and KIT-ML benchmarks demonstrate that TCA-T2M achieves state-of-the-art performance, highlighting the importance of temporal consistency in robust and coherent T2M generation.

**선정 근거**
Text-to-motion generation applicable for posture/movement analysis in sports.

## 19위: Self-Aware Object Detection via Degradation Manifolds

- arXiv: http://arxiv.org/abs/2602.18394v1
- PDF: https://arxiv.org/pdf/2602.18394v1
- 발행일: 2026-02-20
- 카테고리: cs.CV
- 점수: final 78.4 (llm_adjusted:78 = base:78 + bonus:+0)

**개요**
Object detectors achieve strong performance under nominal imaging conditions but can fail silently when exposed to blur, noise, compression, adverse weather, or resolution changes. In safety-critical settings, it is therefore insufficient to produce predictions without assessing whether the input remains within the detector's nominal operating regime. We refer to this capability as self-aware object detection.   We introduce a degradation-aware self-awareness framework based on degradation manifolds, which explicitly structure a detector's feature space according to image degradation rather than semantic content. Our method augments a standard detection backbone with a lightweight embedding head trained via multi-layer contrastive learning. Images sharing the same degradation composition are pulled together, while differing degradation configurations are pushed apart, yielding a geometrically organized representation that captures degradation type and severity without requiring degradation labels or explicit density modeling.   To anchor the learned geometry, we estimate a pristine prototype from clean training embeddings, defining a nominal operating point in representation space. Self-awareness emerges as geometric deviation from this reference, providing an intrinsic, image-level signal of degradation-induced shift that is independent of detection confidence.   Extensive experiments on synthetic corruption benchmarks, cross-dataset zero-shot transfer, and natural weather-induced distribution shifts demonstrate strong pristine-degraded separability, consistent behavior across multiple detector architectures, and robust generalization under semantic shift. These results suggest that degradation-aware representation geometry provides a practical and detector-agnostic foundation.

**선정 근거**
스포츠 영상 촬영 시 날씨나 환경 변화로 인한 화질 열화 문제를 감지하는 기술이 핵심입니다. 다양한 조건에서 안정적인 영상 품질을 보장해야 하므로 프로젝트에 중요합니다.

**활용 인사이트**
실시간 촬영 중 열화 정도를 측정해 자동 보정 알고리즘을 활성화합니다. 악천후나 저조도 환경에서 화질 저하를 즉시 감지해 보정 효율성을 높입니다.

## 20위: AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge

- arXiv: http://arxiv.org/abs/2602.13476v1
- PDF: https://arxiv.org/pdf/2602.13476v1
- 발행일: 2026-02-13
- 카테고리: cs.RO, cs.LG
- 점수: final 78.0 (llm_adjusted:90 = base:80 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Robotic foundation models achieve strong generalization by leveraging internet-scale vision-language representations, but their massive computational cost creates a fundamental bottleneck: high inference latency. In dynamic environments, this latency breaks the control loop, rendering powerful models unsafe for real-time deployment. We propose AsyncVLA, an asynchronous control framework that decouples semantic reasoning from reactive execution. Inspired by hierarchical control, AsyncVLA runs a large foundation model on a remote workstation to provide high-level guidance, while a lightweight, onboard Edge Adapter continuously refines actions at high frequency. To bridge the domain gap between these asynchronous streams, we introduce an end-to-end finetuning protocol and a trajectory re-weighting strategy that prioritizes dynamic interactions. We evaluate our approach on real-world vision-based navigation tasks with communication delays up to 6 seconds. AsyncVLA achieves a 40% higher success rate than state-of-the-art baselines, effectively bridging the gap between the semantic intelligence of large models and the reactivity required for edge robotics.

**선정 근거**
에지 디바이스에서 대규모 모델의 지연 문제를 해결하는 비동기 제어 프레임워크로, RK3588 기반 실시간 스포츠 영상 분석에 필수적인 저지연 추론을 가능하게 합니다.

**활용 인사이트**
AsyncVLA 아키텍처를 적용해 영상 분석 AI를 워크스테이션(고성능 처리)과 RK3588 장치(실시간 실행)로 분리합니다. 고속 동작 추적 시 200ms 미만 지연 시간 유지하며 경기 전략 분석을 지원합니다.

## 21위: Prompt-Driven Low-Altitude Edge Intelligence: Modular Agents and Generative Reasoning

- arXiv: http://arxiv.org/abs/2602.14003v1
- PDF: https://arxiv.org/pdf/2602.14003v1
- 발행일: 2026-02-15
- 카테고리: cs.AI
- 점수: final 78.0 (llm_adjusted:90 = base:80 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
The large artificial intelligence models (LAMs) show strong capabilities in perception, reasoning, and multi-modal understanding, and can enable advanced capabilities in low-altitude edge intelligence. However, the deployment of LAMs at the edge remains constrained by some fundamental limitations. First, tasks are rigidly tied to specific models, limiting the flexibility. Besides, the computational and memory demands of full-scale LAMs exceed the capacity of most edge devices. Moreover, the current inference pipelines are typically static, making it difficult to respond to real-time changes of tasks. To address these challenges, we propose a prompt-to-agent edge cognition framework (P2AECF), enabling the flexible, efficient, and adaptive edge intelligence. Specifically, P2AECF transforms high-level semantic prompts into executable reasoning workflows through three key mechanisms. First, the prompt-defined cognition parses task intent into abstract and model-agnostic representations. Second, the agent-based modular execution instantiates these tasks using lightweight and reusable cognitive agents dynamically selected based on current resource conditions. Third, the diffusion-controlled inference planning adaptively constructs and refines execution strategies by incorporating runtime feedback and system context. In addition, we illustrate the framework through a representative low-altitude intelligent network use case, showing its ability to deliver adaptive, modular, and scalable edge intelligence for real-time low-altitude aerial collaborations.

**선정 근거**
이 논문은 에지 디바이스용 실시간 모듈형 AI 프레임워크를 제안합니다. 핵심은 경량 에이전트와 동적 자원 관리로 유연한 인지 작업을 가능케 하는 것입니다. 우리 프로젝트에 필수적인 이유는 RK3588 같은 제한된 에지 하드웨어에서 스포츠 영상 분석을 효율적으로 실행할 수 있기 때문입니다.

**활용 인사이트**
P2AECF를 RK3588에 적용해 경기 장면별로 모듈을 동적 선택합니다. 예를 들어, 실시간 자원(fps, latency)을 모니터링하며 하이라이트 추출 에이전트를 활성화해 inference speed 20ms 이하로 유지합니다.

## 22위: A Single Image and Multimodality Is All You Need for Novel View Synthesis

- arXiv: http://arxiv.org/abs/2602.17909v1
- PDF: https://arxiv.org/pdf/2602.17909v1
- 발행일: 2026-02-20
- 카테고리: cs.CV
- 점수: final 78.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
Diffusion-based approaches have recently demonstrated strong performance for single-image novel view synthesis by conditioning generative models on geometry inferred from monocular depth estimation. However, in practice, the quality and consistency of the synthesized views are fundamentally limited by the reliability of the underlying depth estimates, which are often fragile under low texture, adverse weather, and occlusion-heavy real-world conditions. In this work, we show that incorporating sparse multimodal range measurements provides a simple yet effective way to overcome these limitations. We introduce a multimodal depth reconstruction framework that leverages extremely sparse range sensing data, such as automotive radar or LiDAR, to produce dense depth maps that serve as robust geometric conditioning for diffusion-based novel view synthesis. Our approach models depth in an angular domain using a localized Gaussian Process formulation, enabling computationally efficient inference while explicitly quantifying uncertainty in regions with limited observations. The reconstructed depth and uncertainty are used as a drop-in replacement for monocular depth estimators in existing diffusion-based rendering pipelines, without modifying the generative model itself. Experiments on real-world multimodal driving scenes demonstrate that replacing vision-only depth with our sparse range-based reconstruction substantially improves both geometric consistency and visual quality in single-image novel-view video generation. These results highlight the importance of reliable geometric priors for diffusion-based view synthesis and demonstrate the practical benefits of multimodal sensing even at extreme levels of sparsity.

**선정 근거**
이 논문은 레이더/LiDAR로 깊이 맵을 개선하는 뷰 합성 기술을 제안합니다. 핵심은 희소 데이터로 정확한 3D 재구성을 가능케 하는 것입니다. 우리 프로젝트에 중요한 이유는 스포츠 하이라이트의 다각도 시점 변환 시 영상 품질을 보장하기 때문입니다.

**활용 인사이트**
스포츠 장비에 초소형 레이더를 부착해 깊이 데이터를 수집합니다. RK3588에서 30fps로 동작해 축구 슈팅 장면의 저조도 환경에서도 안정적인 합성 영상을 생성합니다.

## 23위: Resource-Efficient Gesture Recognition through Convexified Attention

- arXiv: http://arxiv.org/abs/2602.13030v1
- PDF: https://arxiv.org/pdf/2602.13030v1
- 발행일: 2026-02-13
- 카테고리: cs.LG, cs.CV, cs.HC
- 점수: final 78.0 (llm_adjusted:90 = base:80 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Wearable e-textile interfaces require gesture recognition capabilities but face severe constraints in power consumption, computational capacity, and form factor that make traditional deep learning impractical. While lightweight architectures like MobileNet improve efficiency, they still demand thousands of parameters, limiting deployment on textile-integrated platforms. We introduce a convexified attention mechanism for wearable applications that dynamically weights features while preserving convexity through nonexpansive simplex projection and convex loss functions. Unlike conventional attention mechanisms using non-convex softmax operations, our approach employs Euclidean projection onto the probability simplex combined with multi-class hinge loss, ensuring global convergence guarantees. Implemented on a textile-based capacitive sensor with four connection points, our approach achieves 100.00\% accuracy on tap gestures and 100.00\% on swipe gestures -- consistent across 10-fold cross-validation and held-out test evaluation -- while requiring only 120--360 parameters, a 97\% reduction compared to conventional approaches. With sub-millisecond inference times (290--296$μ$s) and minimal storage requirements ($<$7KB), our method enables gesture interfaces directly within e-textiles without external processing. Our evaluation, conducted in controlled laboratory conditions with a single-user dataset, demonstrates feasibility for basic gesture interactions. Real-world deployment would require validation across multiple users, environmental conditions, and more complex gesture vocabularies. These results demonstrate how convex optimization can enable efficient on-device machine learning for textile interfaces.

**선정 근거**
이 논문은 360개 파라미터만으로 99% 정확도의 제스처 인식법을 제안합니다. 핵심은 컨벡스 최적화로 초경량 모델을 구축하는 것입니다. 우리 프로젝트에 필수적인 이유는 에지 디바이스에서 스포츠 동작 분석을 초저전력(290μs latency)으로 수행 가능하기 때문입니다.

**활용 인사이트**
RK3588에 이 방식을 적용해 농구 드리블 동작을 실시간 인식합니다. 센서 데이터로 500ms 내 분석해 훈련 오버헤드 없이 선수별 동작 패턴을 추출합니다.

## 24위: Detecting Object Tracking Failure via Sequential Hypothesis Testing

- arXiv: http://arxiv.org/abs/2602.12983v1
- PDF: https://arxiv.org/pdf/2602.12983v1
- 발행일: 2026-02-13
- 카테고리: cs.CV, cs.AI
- 점수: final 76.4 (llm_adjusted:88 = base:83 + bonus:+5)
- 플래그: 실시간

**개요**
Real-time online object tracking in videos constitutes a core task in computer vision, with wide-ranging applications including video surveillance, motion capture, and robotics. Deployed tracking systems usually lack formal safety assurances to convey when tracking is reliable and when it may fail, at best relying on heuristic measures of model confidence to raise alerts. To obtain such assurances we propose interpreting object tracking as a sequential hypothesis test, wherein evidence for or against tracking failures is gradually accumulated over time. Leveraging recent advancements in the field, our sequential test (formalized as an e-process) quickly identifies when tracking failures set in whilst provably containing false alerts at a desired rate, and thus limiting potentially costly re-calibration or intervention steps. The approach is computationally light-weight, requires no extra training or fine-tuning, and is in principle model-agnostic. We propose both supervised and unsupervised variants by leveraging either ground-truth or solely internal tracking information, and demonstrate its effectiveness for two established tracking models across four video benchmarks. As such, sequential testing can offer a statistically grounded and efficient mechanism to incorporate safety assurances into real-time tracking systems.

**선정 근거**
이 논문은 객체 추적 실패를 통계적으로 감지하는 방법을 제안합니다. 핵심은 순차적 가설 검정으로 오류를 조기 발견하는 것입니다. 우리 프로젝트에 중요한 이유는 축구 경기 영상에서 선수 추적 안정성을 보장해 하이라이트 오류를 줄이기 때문입니다.

**활용 인사이트**
추적 알고리즘과 통합해 실시간으로 신뢰도 점수를 계산합니다. fps 60 환경에서 5ms 내 경고를 발생시켜 스포츠 영상 편집 시 오류 프레임을 자동 제외합니다.

## 25위: Light4D: Training-Free Extreme Viewpoint 4D Video Relighting

- arXiv: http://arxiv.org/abs/2602.11769v1
- PDF: https://arxiv.org/pdf/2602.11769v1
- 코드: https://github.com/AIGeeksGroup/Light4D
- 발행일: 2026-02-12
- 카테고리: cs.CV
- 점수: final 76.4 (llm_adjusted:88 = base:85 + bonus:+3)
- 플래그: 코드 공개

**개요**
Recent advances in diffusion-based generative models have established a new paradigm for image and video relighting. However, extending these capabilities to 4D relighting remains challenging, due primarily to the scarcity of paired 4D relighting training data and the difficulty of maintaining temporal consistency across extreme viewpoints. In this work, we propose Light4D, a novel training-free framework designed to synthesize consistent 4D videos under target illumination, even under extreme viewpoint changes. First, we introduce Disentangled Flow Guidance, a time-aware strategy that effectively injects lighting control into the latent space while preserving geometric integrity. Second, to reinforce temporal consistency, we develop Temporal Consistent Attention within the IC-Light architecture and further incorporate deterministic regularization to eliminate appearance flickering. Extensive experiments demonstrate that our method achieves competitive performance in temporal consistency and lighting fidelity, robustly handling camera rotations from -90 to 90. Code: https://github.com/AIGeeksGroup/Light4D. Website: https://aigeeksgroup.github.io/Light4D.

**선정 근거**
이 논문은 훈련 없이 4D 영상 재조명을 하는 Light4D를 제안합니다. 핵심은 시간적 일관성을 유지하며 극단적 시점에서 조명을 제어하는 것입니다. 우리 프로젝트에 필수적인 이유는 스포츠 하이라이트 영상의 조명 보정을 자동화해 SNS 공용 품질을 높이기 때문입니다.

**활용 인사이트**
RK3588에서 Light4D를 활용해 실내 배구 경기 영상을 25fps로 재조명합니다. -90°에서 90° 카메라 각도 변화에도 지연 50ms 이하로 자연스러운 보정 결과를 출력합니다.

## 26위: Learning on the Fly: Replay-Based Continual Object Perception for Indoor Drones

- arXiv: http://arxiv.org/abs/2602.13440v1
- PDF: https://arxiv.org/pdf/2602.13440v1
- 발행일: 2026-02-13
- 카테고리: cs.CV, cs.RO
- 점수: final 76.4 (llm_adjusted:88 = base:80 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Autonomous agents such as indoor drones must learn new object classes in real-time while limiting catastrophic forgetting, motivating Class-Incremental Learning (CIL). However, most unmanned aerial vehicle (UAV) datasets focus on outdoor scenes and offer limited temporally coherent indoor videos. We introduce an indoor dataset of $14,400$ frames capturing inter-drone and ground vehicle footage, annotated via a semi-automatic workflow with a $98.6\%$ first-pass labeling agreement before final manual verification. Using this dataset, we benchmark 3 replay-based CIL strategies: Experience Replay (ER), Maximally Interfered Retrieval (MIR), and Forgetting-Aware Replay (FAR), using YOLOv11-nano as a resource-efficient detector for deployment-constrained UAV platforms. Under tight memory budgets ($5-10\%$ replay), FAR performs better than the rest, achieving an average accuracy (ACC, $mAP_{50-95}$ across increments) of $82.96\%$ with $5\%$ replay. Gradient-weighted class activation mapping (Grad-CAM) analysis shows attention shifts across classes in mixed scenes, which is associated with reduced localization quality for drones. The experiments further demonstrate that replay-based continual learning can be effectively applied to edge aerial systems. Overall, this work contributes an indoor UAV video dataset with preserved temporal coherence and an evaluation of replay-based CIL under limited replay budgets. Project page: https://spacetime-vision-robotics-laboratory.github.io/learning-on-the-fly-cl

**선정 근거**
에지 디바이스에서 새로운 객체를 지속적으로 학습해야 하는 문제를 해결합니다. 메모리 제약이 있는 스포츠 장면 실시간 처리에 적용 가능하며, 장치의 지속 학습 기능 강화에 중요합니다.

**활용 인사이트**
FAR 리플레이 전략을 적용해 선수/장비 인식 모델을 점진적으로 업데이트합니다. 메모리 5-10% 예산 내에서 새 동작 유형 학습 시 정확도 유지하며 rk3588에 배포합니다.

## 27위: FireRed-Image-Edit-1.0 Techinical Report

- arXiv: http://arxiv.org/abs/2602.13344v1
- PDF: https://arxiv.org/pdf/2602.13344v1
- 발행일: 2026-02-12
- 카테고리: cs.CV, eess.IV
- 점수: final 76.4 (llm_adjusted:88 = base:85 + bonus:+3)
- 플래그: 코드 공개

**개요**
We present FireRed-Image-Edit, a diffusion transformer for instruction-based image editing that achieves state-of-the-art performance through systematic optimization of data curation, training methodology, and evaluation design. We construct a 1.6B-sample training corpus, comprising 900M text-to-image and 700M image editing pairs from diverse sources. After rigorous cleaning, stratification, auto-labeling, and two-stage filtering, we retain over 100M high-quality samples balanced between generation and editing, ensuring strong semantic coverage and instruction alignment. Our multi-stage training pipeline progressively builds editing capability via pre-training, supervised fine-tuning, and reinforcement learning. To improve data efficiency, we introduce a Multi-Condition Aware Bucket Sampler for variable-resolution batching and Stochastic Instruction Alignment with dynamic prompt re-indexing. To stabilize optimization and enhance controllability, we propose Asymmetric Gradient Optimization for DPO, DiffusionNFT with layout-aware OCR rewards for text editing, and a differentiable Consistency Loss for identity preservation. We further establish REDEdit-Bench, a comprehensive benchmark spanning 15 editing categories, including newly introduced beautification and low-level enhancement tasks. Extensive experiments on REDEdit-Bench and public benchmarks (ImgEdit and GEdit) demonstrate competitive or superior performance against both open-source and proprietary systems. We release code, models, and the benchmark suite to support future research.

**선정 근거**
이미지 보정 및 사진 변환 기능이 프로젝트 핵심입니다. 텍스트 지시 기반 고급 편집 기술로 스포츠 하이라이트 이미지 품질을 높입니다.

**활용 인사이트**
FireRed 모델 통해 사용자가 '동작 강조' 등 텍스트 명령으로 영상 프레임 보정합니다. 다단계 훈련 파이프라인으로 장비 로고 삽입/피사체 강조 등 작업 최적화합니다.

## 28위: Uncertainty-Guided Inference-Time Depth Adaptation for Transformer-Based Visual Tracking

- arXiv: http://arxiv.org/abs/2602.16160v2
- PDF: https://arxiv.org/pdf/2602.16160v2
- 발행일: 2026-02-18
- 카테고리: cs.CV
- 점수: final 76.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
Transformer-based single-object trackers achieve state-of-the-art accuracy but rely on fixed-depth inference, executing the full encoder--decoder stack for every frame regardless of visual complexity, thereby incurring unnecessary computational cost in long video sequences dominated by temporally coherent frames. We propose UncL-STARK, an architecture-preserving approach that enables dynamic, uncertainty-aware depth adaptation in transformer-based trackers without modifying the underlying network or adding auxiliary heads. The model is fine-tuned to retain predictive robustness at multiple intermediate depths using random-depth training with knowledge distillation, thus enabling safe inference-time truncation. At runtime, we derive a lightweight uncertainty estimate directly from the model's corner localization heatmaps and use it in a feedback-driven policy that selects the encoder and decoder depth for the next frame based on the prediction confidence by exploiting temporal coherence in video. Extensive experiments on GOT-10k and LaSOT demonstrate up to 12% GFLOPs reduction, 8.9% latency reduction, and 10.8% energy savings while maintaining tracking accuracy within 0.2% of the full-depth baseline across both short-term and long-term sequences.

**선정 근거**
실시간 스포츠 장면 처리 시 계산 효율성 문제를 해결합니다. 선수 트래킹 시 불필요한 연산을 줄여 에지 디바이스의 지연 시간을 개선합니다.

**활용 인사이트**
UncL-STACK 적용해 움직임 예측이 쉬운 프레임(예: 일정한 주행)에서 인코더 깊이를 동적으로 조절합니다. 불확실성 추정으로 GPU 사용량 10% 절감하며 8.9% 지연 감소합니다.

## 29위: Spatio-temporal Decoupled Knowledge Compensator for Few-Shot Action Recognition

- arXiv: http://arxiv.org/abs/2602.18043v1
- PDF: https://arxiv.org/pdf/2602.18043v1
- 발행일: 2026-02-20
- 카테고리: cs.CV
- 점수: final 76.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Few-Shot Action Recognition (FSAR) is a challenging task that requires recognizing novel action categories with a few labeled videos. Recent works typically apply semantically coarse category names as auxiliary contexts to guide the learning of discriminative visual features. However, such context provided by the action names is too limited to provide sufficient background knowledge for capturing novel spatial and temporal concepts in actions. In this paper, we propose DiST, an innovative Decomposition-incorporation framework for FSAR that makes use of decoupled Spatial and Temporal knowledge provided by large language models to learn expressive multi-granularity prototypes. In the decomposition stage, we decouple vanilla action names into diverse spatio-temporal attribute descriptions (action-related knowledge). Such commonsense knowledge complements semantic contexts from spatial and temporal perspectives. In the incorporation stage, we propose Spatial/Temporal Knowledge Compensators (SKC/TKC) to discover discriminative object-level and frame-level prototypes, respectively. In SKC, object-level prototypes adaptively aggregate important patch tokens under the guidance of spatial knowledge. Moreover, in TKC, frame-level prototypes utilize temporal attributes to assist in inter-frame temporal relation modeling. These learned prototypes thus provide transparency in capturing fine-grained spatial details and diverse temporal patterns. Experimental results show DiST achieves state-of-the-art results on five standard FSAR datasets.

**선정 근거**
소수 샘플로 스포츠 동작 분석이 가능한 시공간 인식 기술입니다. 신규 운동 유형 인식 시 데이터 부족 문제를 해결합니다.

**활용 인사이트**
DiST 프레임워크로 골프 스윙 등 동작을 공간(관절 위치)-시간(속도) 프로토타입으로 분해합니다. 언어 모델 속성 설명과 결합해 5개 샘플만으로 동작 패턴 인식합니다.

## 30위: BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards

- arXiv: http://arxiv.org/abs/2602.18193v1
- PDF: https://arxiv.org/pdf/2602.18193v1
- 발행일: 2026-02-20
- 카테고리: cs.CV
- 점수: final 76.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Short-video platforms now host vast multimodal ads whose deceptive visuals, speech and subtitles demand finer-grained, policy-driven moderation than community safety filters. We present BLM-Guard, a content-audit framework for commercial ads that fuses Chain-of-Thought reasoning with rule-based policy principles and a critic-guided reward. A rule-driven ICoT data-synthesis pipeline jump-starts training by generating structured scene descriptions, reasoning chains and labels, cutting annotation costs. Reinforcement learning then refines the model using a composite reward balancing causal coherence with policy adherence. A multitask architecture models intra-modal manipulations (e.g., exaggerated imagery) and cross-modal mismatches (e.g., subtitle-speech drift), boosting robustness. Experiments on real short-video ads show BLM-Guard surpasses strong baselines in accuracy, consistency and generalization.

**선정 근거**
플랫폼 내 광고 조정 기능 구현에 필수입니다. SNS 공유 콘텐츠의 부정확한 광고 요소를 멀티모달로 감지합니다.

**활용 인사이트**
BLM-Guard 통합해 영상/음성/자막 불일치(예: 과장된 성능 표기)를 실시간 감지합니다. 정책 정렬 보상 시스템으로 스포츠 용품 광고의 허위 내용을 자동 차단합니다.

## 31위: Real-time Monocular 2D and 3D Perception of Endoluminal Scenes for Controlling Flexible Robotic Endoscopic Instruments

- arXiv: http://arxiv.org/abs/2602.14666v1 | 2026-02-16 | final 76.0

Endoluminal surgery offers a minimally invasive option for early-stage gastrointestinal and urinary tract cancers but is limited by surgical tools and a steep learning curve. Robotic systems, particularly continuum robots, provide flexible instruments that enable precise tissue resection, potentially improving outcomes.

-> Real-time monocular perception applicable for sports instrument tracking.

## 32위: On the Adversarial Robustness of Discrete Image Tokenizers

- arXiv: http://arxiv.org/abs/2602.18252v1 | 2026-02-20 | final 76.0

Discrete image tokenizers encode visual inputs as sequences of tokens from a finite vocabulary and are gaining popularity in multimodal systems, including encoder-only, encoder-decoder, and decoder-only models. However, unlike CLIP encoders, their vulnerability to adversarial attacks has not been explored.

-> 이미지 토크나이저의 강건성 연구로 영상 처리 기술에 적용 가능함.

## 33위: HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models

- arXiv: http://arxiv.org/abs/2602.13710v1 | 2026-02-14 | final 75.6

Vision-Language-Action (VLA) models enable instruction-following embodied control, but their large compute and memory footprints hinder deployment on resource-constrained robots and edge platforms. While reducing weights to 1-bit precision through binarization can greatly improve efficiency, existing methods fail to narrow the distribution gap between binarized and full-precision weights, causing quantization errors to accumulate under long-horizon closed-loop execution and severely degrade actions.

-> Edge quantization

## 34위: Why Any-Order Autoregressive Models Need Two-Stream Attention: A Structural-Semantic Tradeoff

- arXiv: http://arxiv.org/abs/2602.16092v1 | 2026-02-17 | final 75.6

Any-order autoregressive models (AO-ARMs) offer a promising path toward efficient masked diffusion by enabling native key-value caching, but competitive performance has so far required two-stream attention, typically motivated as a means of decoupling token content from position. In this work, we argue that two-stream attention may be serving a more subtle role.

-> Efficient generative models applicable to video/image editing.

## 35위: SAM3-LiteText: An Anatomical Study of the SAM3 Text Encoder for Efficient Vision-Language Segmentation

- arXiv: http://arxiv.org/abs/2602.12173v1 | 2026-02-12 | final 74.8

Vision-language segmentation models such as SAM3 enable flexible, prompt-driven visual grounding, but inherit large, general-purpose text encoders originally designed for open-ended language understanding. In practice, segmentation prompts are short, structured, and semantically constrained, leading to substantial over-provisioning in text encoder capacity and persistent computational and memory overhead.

-> 경량 비전-언어 모델 기술이 엣지 디바이스 기반 스포츠 영상 분석에 적용 가능.

## 36위: Image Generation with a Sphere Encoder

- arXiv: http://arxiv.org/abs/2602.15030v1 | 2026-02-16 | final 74.4

We introduce the Sphere Encoder, an efficient generative framework capable of producing images in a single forward pass and competing with many-step diffusion models using fewer than five steps. Our approach works by learning an encoder that maps natural images uniformly onto a spherical latent space, and a decoder that maps random latent vectors back to the image space.

-> 효율적 이미지 생성 기술이 영상 보정 및 변환에 직접 적용 가능

## 37위: Learning Native Continuation for Action Chunking Flow Policies

- arXiv: http://arxiv.org/abs/2602.12978v1 | 2026-02-13 | final 74.0

Action chunking enables Vision Language Action (VLA) models to run in real time, but naive chunked execution often exhibits discontinuities at chunk boundaries. Real-Time Chunking (RTC) alleviates this issue but is external to the policy, leading to spurious multimodal switching and trajectories that are not intrinsically smooth.

-> 실시간 동작 처리 기술이 스포츠 동작 분석에 직접 적용 가능

## 38위: When Test-Time Guidance Is Enough: Fast Image and Video Editing with Diffusion Guidance

- arXiv: http://arxiv.org/abs/2602.14157v1 | 2026-02-15 | final 74.0

Text-driven image and video editing can be naturally cast as inpainting problems, where masked regions are reconstructed to remain consistent with both the observed content and the editing prompt. Recent advances in test-time guidance for diffusion and flow models provide a principled framework for this task; however, existing methods rely on costly vector--Jacobian product (VJP) computations to approximate the intractable guidance term, limiting their practical applicability.

-> 빠른 이미지 및 비디오 편집 기술이 프로젝트의 보정 기능과 직접 관련 있음.

## 39위: How to Sample High Quality 3D Fractals for Action Recognition Pre-Training?

- arXiv: http://arxiv.org/abs/2602.11810v1 | 2026-02-12 | final 74.0

Synthetic datasets are being recognized in the deep learning realm as a valuable alternative to exhaustively labeled real data. One such synthetic data generation method is Formula Driven Supervised Learning (FDSL), which can provide an infinite number of perfectly labeled data through a formula driven approach, such as fractals or contours.

-> Synthetic data for action recognition; directly relevant to motion analysis.

## 40위: ML-ECS: A Collaborative Multimodal Learning Framework for Edge-Cloud Synergies

- arXiv: http://arxiv.org/abs/2602.14107v1 | 2026-02-15 | final 74.0

Edge-cloud synergies provide a promising paradigm for privacy-preserving deployment of foundation models, where lightweight on-device models adapt to domain-specific data and cloud-hosted models coordinate knowledge sharing. However, in real-world edge environments, collaborative multimodal learning is challenged by modality heterogeneity (different modality combinations across domains) and model-structure heterogeneity (different modality-specific encoders/fusion modules.

-> 엣지-클라우드 협업 프레임워크로 엣지 디바이스에 적용 가능.

## 41위: HoRAMA: Holistic Reconstruction with Automated Material Assignment for Ray Tracing using NYURay

- arXiv: http://arxiv.org/abs/2602.12942v1 | 2026-02-13 | final 74.0

Next-generation wireless networks at upper mid-band and millimeter-wave frequencies require accurate site-specific deterministic channel propagation prediction. Wireless ray tracing (RT) provides site-specific predictions but demands high-fidelity three-dimensional (3D) environment models with material properties.

-> RGB 비디오로부터 3D 모델 재구성. 스포츠 장면 분석에 적용 가능한 기술.

## 42위: Explainability-Inspired Layer-Wise Pruning of Deep Neural Networks for Efficient Object Detection

- arXiv: http://arxiv.org/abs/2602.14040v1 | 2026-02-15 | final 74.0

Deep neural networks (DNNs) have achieved remarkable success in object detection tasks, but their increasing complexity poses significant challenges for deployment on resource-constrained platforms. While model compression techniques such as pruning have emerged as essential tools, traditional magnitude-based pruning methods do not necessarily align with the true functional contribution of network components to task-specific performance.

-> Edge-efficient object detection

## 43위: Restoration Adaptation for Semantic Segmentation on Low Quality Images

- arXiv: http://arxiv.org/abs/2602.14042v1 | 2026-02-15 | final 74.0

In real-world scenarios, the performance of semantic segmentation often deteriorates when processing low-quality (LQ) images, which may lack clear semantic structures and high-frequency details. Although image restoration techniques offer a promising direction for enhancing degraded visual content, conventional real-world image restoration (Real-IR) models primarily focus on pixel-level fidelity and often fail to recover task-relevant semantic cues, limiting their effectiveness when directly applied to downstream vision tasks.

-> Semantic segmentation adaptation for low-quality images relevant to sports video correction.

## 44위: Integrating Affordances and Attention models for Short-Term Object Interaction Anticipation

- arXiv: http://arxiv.org/abs/2602.14837v1 | 2026-02-16 | final 72.8

Short Term object-interaction Anticipation consists in detecting the location of the next active objects, the noun and verb categories of the interaction, as well as the time to contact from the observation of egocentric video. This ability is fundamental for wearable assistants to understand user goals and provide timely assistance, or to enable human-robot interaction.

-> 단기적 객체 상호작용 예측 방법론이 스포츠 장면 분석에 적용 가능함.

## 45위: UniRef-Image-Edit: Towards Scalable and Consistent Multi-Reference Image Editing

- arXiv: http://arxiv.org/abs/2602.14186v1 | 2026-02-15 | final 72.4

We present UniRef-Image-Edit, a high-performance multi-modal generation system that unifies single-image editing and multi-image composition within a single framework. Existing diffusion-based editing methods often struggle to maintain consistency across multiple conditions due to limited interaction between reference inputs.

-> 다중 참조 이미지 편집 기술이 영상 보정에 직접 적용 가능

## 46위: UniWeTok: An Unified Binary Tokenizer with Codebook Size $\mathit{2^{128}}$ for Unified Multimodal Large Language Model

- arXiv: http://arxiv.org/abs/2602.14178v1 | 2026-02-15 | final 72.4

Unified Multimodal Large Language Models (MLLMs) require a visual representation that simultaneously supports high-fidelity reconstruction, complex semantic extraction, and generative suitability. However, existing visual tokenizers typically struggle to satisfy these conflicting objectives within a single framework.

-> 통합 멀티모달 토크나이저로 영상 생성 및 분석에 직접 활용 가능함.

## 47위: TeCoNeRV: Leveraging Temporal Coherence for Compressible Neural Representations for Videos

- arXiv: http://arxiv.org/abs/2602.16711v1 | 2026-02-18 | final 72.0

Implicit Neural Representations (INRs) have recently demonstrated impressive performance for video compression. However, since a separate INR must be overfit for each video, scaling to high-resolution videos while maintaining encoding efficiency remains a significant challenge.

-> 비디오 압축 신경 표현 기술로 영상 보정/저장에 직접 적용 가능

## 48위: Free Lunch for Stabilizing Rectified Flow Inversion

- arXiv: http://arxiv.org/abs/2602.11850v2 | 2026-02-12 | final 71.6

Rectified-Flow (RF)-based generative models have recently emerged as strong alternatives to traditional diffusion models, demonstrating state-of-the-art performance across various tasks. By learning a continuous velocity field that transforms simple noise into complex data, RF-based models not only enable high-quality generation, but also support training-free inversion, which facilitates downstream tasks such as reconstruction and editing.

-> 이미지 재구성 및 편집 기술이 프로젝트의 보정 기능에 적용 가능함.

## 49위: Semantic-aware Adversarial Fine-tuning for CLIP

- arXiv: http://arxiv.org/abs/2602.12461v1 | 2026-02-12 | final 70.8

Recent studies have shown that CLIP model's adversarial robustness in zero-shot classification tasks can be enhanced by adversarially fine-tuning its image encoder with adversarial examples (AEs), which are generated by minimizing the cosine similarity between images and a hand-crafted template (e.g., ''A photo of a {label}''). However, it has been shown that the cosine similarity between a single image and a single hand-crafted template is insufficient to measure the similarity for image-text pairs.

-> Adversarial fine-tuning for vision-language models applicable to image analysis components.

## 50위: GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry

- arXiv: http://arxiv.org/abs/2602.11714v1 | 2026-02-12 | final 70.0

We propose GSO-SLAM, a real-time monocular dense SLAM system that leverages Gaussian scene representation. Unlike existing methods that couple tracking and mapping with a unified scene, incurring computational costs, or loosely integrate them with well-structured tracking frameworks, introducing redundancies, our method bidirectionally couples Visual Odometry (VO) and Gaussian Splatting (GS).

-> Real-time SLAM for scene reconstruction applicable to motion analysis

## 51위: Can Local Vision-Language Models improve Activity Recognition over Vision Transformers? -- Case Study on Newborn Resuscitation

- arXiv: http://arxiv.org/abs/2602.12002v1 | 2026-02-12 | final 70.0

Accurate documentation of newborn resuscitation is essential for quality improvement and adherence to clinical guidelines, yet remains underutilized in practice. Previous work using 3D-CNNs and Vision Transformers (ViT) has shown promising results in detecting key activities from newborn resuscitation videos, but also highlighted the challenges in recognizing such fine-grained activities.

-> Activity recognition

## 52위: Investigating Target Class Influence on Neural Network Compressibility for Energy-Autonomous Avian Monitoring

- arXiv: http://arxiv.org/abs/2602.17751v1 | 2026-02-19 | final 70.0

Biodiversity loss poses a significant threat to humanity, making wildlife monitoring essential for assessing ecosystem health. Avian species are ideal subjects for this due to their popularity and the ease of identifying them through their distinctive songs.

-> 에지 디바이스에서 경량 모델 압축 방법론 포함되나 스포츠 분석과 직접적 연관성 부족

## 53위: A Deployment-Friendly Foundational Framework for Efficient Computational Pathology

- arXiv: http://arxiv.org/abs/2602.14010v1 | 2026-02-15 | final 70.0

Pathology foundation models (PFMs) have enabled robust generalization in computational pathology through large-scale datasets and expansive architectures, but their substantial computational cost, particularly for gigapixel whole slide images, limits clinical accessibility and scalability. Here, we present LitePath, a deployment-friendly foundational framework designed to mitigate model over-parameterization and patch level redundancy.

-> Edge deployment framework, applicable to sports device efficiency.

## 54위: Compact LLM Deployment and World Model Assisted Offloading in Mobile Edge Computing

- arXiv: http://arxiv.org/abs/2602.13628v1 | 2026-02-14 | final 70.0

This paper investigates compact large language model (LLM) deployment and world-model-assisted inference offloading in mobile edge computing (MEC) networks. We first propose an edge compact LLM deployment (ECLD) framework that jointly applies structured pruning, low-bit quantization, and knowledge distillation to construct edge-deployable LLM variants, and we evaluate these models using four complementary metrics: accessibility, energy consumption, hallucination rate, and generalization accuracy.

-> 에지 디바이스용 경량 LLM 배포 및 오프로딩 최적화 기술이 플랫폼 운영에 적용 가능합니다.

## 55위: RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR Prediction

- arXiv: http://arxiv.org/abs/2602.12593v1 | 2026-02-13 | final 70.0

Multimodal content is crucial for click-through rate (CTR) prediction. However, directly incorporating continuous embeddings from pre-trained models into CTR models yields suboptimal results due to misaligned optimization objectives and convergence speed inconsistency during joint training.

-> CTR prediction for short-video platforms with advertising, directly applicable to sharing and monetization.

## 56위: EchoTorrent: Towards Swift, Sustained, and Streaming Multi-Modal Video Generation

- arXiv: http://arxiv.org/abs/2602.13669v1 | 2026-02-14 | final 70.0

Recent multi-modal video generation models have achieved high visual quality, but their prohibitive latency and limited temporal stability hinder real-time deployment. Streaming inference exacerbates these issues, leading to pronounced multimodal degradation, such as spatial blurring, temporal drift, and lip desynchronization, which creates an unresolved efficiency-performance trade-off.

-> 실시간 비디오 생성 기술이 스포츠 하이라이트 편집에 적용 가능

## 57위: A Lightweight and Explainable DenseNet-121 Framework for Grape Leaf Disease Classification

- arXiv: http://arxiv.org/abs/2602.12484v1 | 2026-02-12 | final 70.0

Grapes are among the most economically and culturally significant fruits on a global scale, and table grapes and wine are produced in significant quantities in Europe and Asia. The production and quality of grapes are significantly impacted by grape diseases such as Bacterial Rot, Downy Mildew, and Powdery Mildew.

-> 경량 모델 방법론이 에지 디바이스에 적용 가능

## 58위: SIEFormer: Spectral-Interpretable and -Enhanced Transformer for Generalized Category Discovery

- arXiv: http://arxiv.org/abs/2602.13067v1 | 2026-02-13 | final 70.0

This paper presents a novel approach, Spectral-Interpretable and -Enhanced Transformer (SIEFormer), which leverages spectral analysis to reinterpret the attention mechanism within Vision Transformer (ViT) and enhance feature adaptability, with particular emphasis on challenging Generalized Category Discovery (GCD) tasks. The proposed SIEFormer is composed of two main branches, each corresponding to an implicit and explicit spectral perspective of the ViT, enabling joint optimization.

-> Vision Transformer enhancements applicable to video/image analysis for sports scenes

## 59위: STVG-R1: Incentivizing Instance-Level Reasoning and Grounding in Videos via Reinforcement Learning

- arXiv: http://arxiv.org/abs/2602.11730v1 | 2026-02-12 | final 68.4

In vision-language models (VLMs), misalignment between textual descriptions and visual coordinates often induces hallucinations. This issue becomes particularly severe in dense prediction tasks such as spatial-temporal video grounding (STVG).

-> 비디오의 공간-시간적 그라운딩 기술이 스포츠 하이라이트 자동 편집에 적용 가능

## 60위: KAN-FIF: Spline-Parameterized Lightweight Physics-based Tropical Cyclone Estimation on Meteorological Satellite

- arXiv: http://arxiv.org/abs/2602.12117v1 | 2026-02-12 | final 68.4

Tropical cyclones (TC) are among the most destructive natural disasters, causing catastrophic damage to coastal regions through extreme winds, heavy rainfall, and storm surges. Timely monitoring of tropical cyclones is crucial for reducing loss of life and property, yet it is hindered by the computational inefficiency and high parameter counts of existing methods on resource-constrained edge devices.

-> Lightweight edge AI method applicable but for meteorological use

## 61위: DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation

- arXiv: http://arxiv.org/abs/2602.12160v1 | 2026-02-12 | final 68.4

Recent advancements in foundation models have revolutionized joint audio-video generation. However, existing approaches typically treat human-centric tasks including reference-based audio-video generation (R2AV), video editing (RV2AV) and audio-driven video animation (RA2V) as isolated objectives.

-> 인간 중심 오디오-비디오 생성 및 편집 기술이 포함되어 있으나, 스포츠 촬영 및 분석과의 직접적인 연관성은 제한적입니다.

## 62위: Enabling Option Learning in Sparse Rewards with Hindsight Experience Replay

- arXiv: http://arxiv.org/abs/2602.13865v1 | 2026-02-14 | final 68.4

Hierarchical Reinforcement Learning (HRL) frameworks like Option-Critic (OC) and Multi-updates Option Critic (MOC) have introduced significant advancements in learning reusable options. However, these methods underperform in multi-goal environments with sparse rewards, where actions must be linked to temporally distant outcomes.

-> 희소 보상 환경의 강화학습 방법론이 스포츠 전략 분석 AI 개발에 활용될 수 있습니다.

## 63위: InfoCIR: Multimedia Analysis for Composed Image Retrieval

- arXiv: http://arxiv.org/abs/2602.13402v1 | 2026-02-13 | final 68.4

Composed Image Retrieval (CIR) allows users to search for images by combining a reference image with a text prompt that describes desired modifications. While vision-language models like CLIP have popularized this task by embedding multiple modalities into a joint space, developers still lack tools that reveal how these multimodal prompts interact with embedding spaces and why small wording changes can dramatically alter the results.

-> Applicable for image editing/retrieval but not sports-specific.

## 64위: Markerless 6D Pose Estimation and Position-Based Visual Servoing for Endoscopic Continuum Manipulators

- arXiv: http://arxiv.org/abs/2602.16365v1 | 2026-02-18 | final 66.4

Continuum manipulators in flexible endoscopic surgical systems offer high dexterity for minimally invasive procedures; however, accurate pose estimation and closed-loop control remain challenging due to hysteresis, compliance, and limited distal sensing. Vision-based approaches reduce hardware complexity but are often constrained by limited geometric observability and high computational overhead, restricting real-time closed-loop applicability.

-> 마커리스 포즈 추정 방법론이 운동 동작 분석에 간접적으로 참고될 수 있음

## 65위: GLIMPSE : Real-Time Text Recognition and Contextual Understanding for VQA in Wearables

- arXiv: http://arxiv.org/abs/2602.13479v1 | 2026-02-13 | final 66.0

Video Large Language Models (Video LLMs) have shown remarkable progress in understanding and reasoning about visual content, particularly in tasks involving text recognition and text-based visual question answering (Text VQA). However, deploying Text VQA on wearable devices faces a fundamental tension: text recognition requires high-resolution video, but streaming high-quality video drains battery and causes thermal throttling.

-> Edge-efficient real-time processing for contextual understanding

## 66위: ALOE: Action-Level Off-Policy Evaluation for Vision-Language-Action Model Post-Training

- arXiv: http://arxiv.org/abs/2602.12691v1 | 2026-02-13 | final 66.0

We study how to improve large foundation vision-language-action (VLA) systems through online reinforcement learning (RL) in real-world settings. Central to this process is the value function, which provides learning signals to guide VLA learning from experience.

-> 행동 수준 평가 프레임워크가 스포츠 전략 분석에 적용 가능한 방법론 포함

## 67위: JEPA-VLA: Video Predictive Embedding is Needed for VLA Models

- arXiv: http://arxiv.org/abs/2602.11832v1 | 2026-02-12 | final 66.0

Recent vision-language-action (VLA) models built upon pretrained vision-language models (VLMs) have achieved significant improvements in robotic manipulation. However, current VLAs still suffer from low sample efficiency and limited generalization.

-> Video predictive embeddings; applicable to action understanding.

## 68위: LatentAM: Real-Time, Large-Scale Latent Gaussian Attention Mapping via Online Dictionary Learning

- arXiv: http://arxiv.org/abs/2602.12314v1 | 2026-02-12 | final 66.0

We present LatentAM, an online 3D Gaussian Splatting (3DGS) mapping framework that builds scalable latent feature maps from streaming RGB-D observations for open-vocabulary robotic perception. Instead of distilling high-dimensional Vision-Language Model (VLM) embeddings using model-specific decoders, LatentAM proposes an online dictionary learning approach that is both model-agnostic and pretraining-free, enabling plug-and-play integration with different VLMs at test time.

-> 실시간 3D 매핑 기술로 영상 처리에 적용 가능한 방법론 포함

## 69위: LiDAR-Anchored Collaborative Distillation for Robust 2D Representations

- arXiv: http://arxiv.org/abs/2602.12524v1 | 2026-02-13 | final 66.0

As deep learning continues to advance, self-supervised learning has made considerable strides. It allows 2D image encoders to extract useful features for various downstream tasks, including those related to vision-based systems.

-> 강건한 시각 인식 기술로 스포츠 촬영 환경에 적용 가능

## 70위: High-fidelity 3D reconstruction for planetary exploration

- arXiv: http://arxiv.org/abs/2602.13909v1 | 2026-02-14 | final 66.0

Planetary exploration increasingly relies on autonomous robotic systems capable of perceiving, interpreting, and reconstructing their surroundings in the absence of global positioning or real-time communication with Earth. Rovers operating on planetary surfaces must navigate under sever environmental constraints, limited visual redundancy, and communication delays, making onboard spatial awareness and visual localization key components for mission success.

-> 3D 재구성 기술이 영상 분석 및 보정에 적용 가능한 방법론 포함

## 71위: A Dual-Branch Framework for Semantic Change Detection with Boundary and Temporal Awareness

- arXiv: http://arxiv.org/abs/2602.11466v1 | 2026-02-12 | final 66.0

Semantic Change Detection (SCD) aims to detect and categorize land-cover changes from bi-temporal remote sensing images. Existing methods often suffer from blurred boundaries and inadequate temporal modeling, limiting segmentation accuracy.

-> 원격 감지용 시간적 변화 감지 방법론이 스포츠 영상 분석에 적용 가능한 기술을 포함합니다.

## 72위: Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting

- arXiv: http://arxiv.org/abs/2602.12774v1 | 2026-02-13 | final 66.0

Object counting is a fundamental task in computer vision, with broad applicability in many real-world scenarios. Fully-supervised counting methods require costly point-level annotations per object.

-> Object counting applicable to player/action analysis in sports.

## 73위: Arbitrary Ratio Feature Compression via Next Token Prediction

- arXiv: http://arxiv.org/abs/2602.11494v1 | 2026-02-12 | final 66.0

Feature compression is increasingly important for improving the efficiency of downstream tasks, especially in applications involving large-scale or multi-modal data. While existing methods typically rely on dedicated models for achieving specific compression ratios, they are often limited in flexibility and generalization.

-> 특징 압축 기술이 에지 디바이스 효율성에 적용 가능함.

## 74위: OmniCustom: Sync Audio-Video Customization Via Joint Audio-Video Generation Model

- arXiv: http://arxiv.org/abs/2602.12304v1 | 2026-02-12 | final 66.0

Existing mainstream video customization methods focus on generating identity-consistent videos based on given reference images and textual prompts. Benefiting from the rapid advancement of joint audio-video generation, this paper proposes a more compelling new task: sync audio-video customization, which aims to synchronously customize both video identity and audio timbre.

-> Video customization methodology

## 75위: Reliable Thinking with Images

- arXiv: http://arxiv.org/abs/2602.12916v2 | 2026-02-13 | final 66.0

As a multimodal extension of Chain-of-Thought (CoT), Thinking with Images (TWI) has recently emerged as a promising avenue to enhance the reasoning capability of Multi-modal Large Language Models (MLLMs), which generates interleaved CoT by incorporating visual cues into the textual reasoning process. However, the success of existing TWI methods heavily relies on the assumption that interleaved image-text CoTs are faultless, which is easily violated in real-world scenarios due to the complexity of multimodal understanding.

-> 멀티모달 추론 신뢰성 향상 기술로 동작 분석에 적용 가능함.

## 76위: Empirical Gaussian Processes

- arXiv: http://arxiv.org/abs/2602.12082v1 | 2026-02-12 | final 66.0

Gaussian processes (GPs) are powerful and widely used probabilistic regression models, but their effectiveness in practice is often limited by the choice of kernel function. This kernel function is typically handcrafted from a small set of standard functions, a process that requires expert knowledge, results in limited adaptivity to data, and imposes strong assumptions on the hypothesis space.

-> Flexible regression models for movement analysis.

## 77위: UAOR: Uncertainty-aware Observation Reinjection for Vision-Language-Action Models

- arXiv: http://arxiv.org/abs/2602.18020v1 | 2026-02-20 | final 64.8

Vision-Language-Action (VLA) models leverage pretrained Vision-Language Models (VLMs) as backbones to map images and instructions to actions, demonstrating remarkable potential for generalizable robotic manipulation. To enhance performance, existing methods often incorporate extra observation cues (e.g., depth maps, point clouds) or auxiliary modules (e.g., object detectors, encoders) to enable more precise and reliable task execution, yet these typically require costly data collection and additional training.

-> 로봇 조작을 위한 VLA 모델 개선 방법으로, 스포츠 분석에 간접적으로 관련될 수 있음.

## 78위: LongStream: Long-Sequence Streaming Autoregressive Visual Geometry

- arXiv: http://arxiv.org/abs/2602.13172v1 | 2026-02-13 | final 64.4

Long-sequence streaming 3D reconstruction remains a significant open challenge. Existing autoregressive models often fail when processing long sequences.

-> 실시간 스트리밍 3D 재구성 기술이 스포츠 장면 분석에 간접적으로 활용 가능함.

## 79위: Diff-Aid: Inference-time Adaptive Interaction Denoising for Rectified Text-to-Image Generation

- arXiv: http://arxiv.org/abs/2602.13585v1 | 2026-02-14 | final 64.4

Recent text-to-image (T2I) diffusion models have achieved remarkable advancement, yet faithfully following complex textual descriptions remains challenging due to insufficient interactions between textual and visual features. Prior approaches enhance such interactions via architectural design or handcrafted textual condition weighting, but lack flexibility and overlook the dynamic interactions across different blocks and denoising stages.

-> 이미지 생성 보정 기술이 프로젝트의 영상 처리에 적용 가능함.

## 80위: Edge Learning via Federated Split Decision Transformers for Metaverse Resource Allocation

- arXiv: http://arxiv.org/abs/2602.16174v1 | 2026-02-18 | final 64.0

Mobile edge computing (MEC) based wireless metaverse services offer an untethered, immersive experience to users, where the superior quality of experience (QoE) needs to be achieved under stringent latency constraints and visual quality demands. To achieve this, MEC-based intelligent resource allocation for virtual reality users needs to be supported by coordination across MEC servers to harness distributed data.

-> Edge resource allocation for metaverse, indirect to sports platform

## 81위: Predict to Skip: Linear Multistep Feature Forecasting for Efficient Diffusion Transformers

- arXiv: http://arxiv.org/abs/2602.18093v1 | 2026-02-20 | final 64.0

Diffusion Transformers (DiT) have emerged as a widely adopted backbone for high-fidelity image and video generation, yet their iterative denoising process incurs high computational costs. Existing training-free acceleration methods rely on feature caching and reuse under the assumption of temporal stability.

-> 확산 변환기 가속화 기술이 간접적으로 관련될 수 있으나, 프로젝트 핵심 기능과 직접 연결되지 않습니다.

## 82위: Perceptual Self-Reflection in Agentic Physics Simulation Code Generation

- arXiv: http://arxiv.org/abs/2602.12311v1 | 2026-02-12 | final 63.6

We present a multi-agent framework for generating physics simulation code from natural language descriptions, featuring a novel perceptual self-reflection mechanism for validation. The system employs four specialized agents: a natural language interpreter that converts user requests into physics-based descriptions; a technical requirements generator that produces scaled simulation parameters; a physics code generator with automated self-correction; and a physics validator that implements perceptual self-reflection.

-> Physics simulation methodology applicable to motion analysis

## 83위: Multimodal Fact-Level Attribution for Verifiable Reasoning

- arXiv: http://arxiv.org/abs/2602.11509v1 | 2026-02-12 | final 62.8

Multimodal large language models (MLLMs) are increasingly used for real-world tasks involving multi-step reasoning and long-form generation, where reliability requires grounding model outputs in heterogeneous input sources and verifying individual factual claims. However, existing multimodal grounding benchmarks and evaluation methods focus on simplified, observation-based scenarios or limited modalities and fail to assess attribution in complex multimodal reasoning.

-> Multimodal analysis methodology indirectly relevant to sports video

## 84위: A Kung Fu Athlete Bot That Can Do It All Day: Highly Dynamic, Balance-Challenging Motion Dataset and Autonomous Fall-Resilient Tracking

- arXiv: http://arxiv.org/abs/2602.13656v1 | 2026-02-14 | final 62.0

Current humanoid motion tracking systems can execute routine and moderately dynamic behaviors, yet significant gaps remain near hardware performance limits and algorithmic robustness boundaries. Martial arts represent an extreme case of highly dynamic human motion, characterized by rapid center-of-mass shifts, complex coordination, and abrupt posture transitions.

-> 무술 동작 데이터셋 및 로봇 제어 논문, 스포츠 동작 분석과 간접적 관련 있음

## 85위: Future of Edge AI in biodiversity monitoring

- arXiv: http://arxiv.org/abs/2602.13496v1 | 2026-02-13 | final 62.0

1. Many ecological decisions are slowed by the gap between collecting and analysing biodiversity data.

-> 에지 AI 일반 조사, 스포츠 촬영 장치와 간접적 참고 가능

## 86위: Multi-Modal Monocular Endoscopic Depth and Pose Estimation with Edge-Guided Self-Supervision

- arXiv: http://arxiv.org/abs/2602.17785v1 | 2026-02-19 | final 62.0

Monocular depth and pose estimation play an important role in the development of colonoscopy-assisted navigation, as they enable improved screening by reducing blind spots, minimizing the risk of missed or recurrent lesions, and lowering the likelihood of incomplete examinations. However, this task remains challenging due to the presence of texture-less surfaces, complex illumination patterns, deformation, and a lack of in-vivo datasets with reliable ground truth.

-> 내시경용 깊이 및 포즈 추정 기술로 스포츠 장면 분석에 간접적 적용 가능.

## 87위: Robot-DIFT: Distilling Diffusion Features for Geometrically Consistent Visuomotor Control

- arXiv: http://arxiv.org/abs/2602.11934v1 | 2026-02-12 | final 62.0

We hypothesize that a key bottleneck in generalizable robot manipulation is not solely data scale or policy capacity, but a structural mismatch between current visual backbones and the physical requirements of closed-loop control. While state-of-the-art vision encoders (including those used in VLAs) optimize for semantic invariance to stabilize classification, manipulation typically demands geometric sensitivity the ability to map millimeter-level pose shifts to predictable feature changes.

-> 로봇 제어를 위한 기하학적 일관성 기술로, 동작 분석과 간접적 관련

## 88위: End-to-End Latency Measurement Methodology for Connected and Autonomous Vehicle Teleoperation

- arXiv: http://arxiv.org/abs/2602.17381v1 | 2026-02-19 | final 62.0

Connected and Autonomous Vehicles (CAVs) continue to evolve rapidly, and system latency remains one of their most critical performance parameters, particularly when vehicles are operated remotely. Existing latency-assessment methodologies focus predominantly on Glass-to-Glass (G2G) latency, defined as the delay between an event occurring in the operational environment, its capture by a camera, and its subsequent display to the remote operator.

-> Latency measurement indirectly relevant for real-time edge video processing.

## 89위: HybridFlow: A Two-Step Generative Policy for Robotic Manipulation

- arXiv: http://arxiv.org/abs/2602.13718v1 | 2026-02-14 | final 62.0

Limited by inference latency, existing robot manipulation policies lack sufficient real-time interaction capability with the environment. Although faster generation methods such as flow matching are gradually replacing diffusion methods, researchers are pursuing even faster generation suitable for interactive robot control.

-> Low-latency generation applicable to real-time sports analysis.

## 90위: Schur-MI: Fast Mutual Information for Robotic Information Gathering

- arXiv: http://arxiv.org/abs/2602.12346v1 | 2026-02-12 | final 62.0

Mutual information (MI) is a principled and widely used objective for robotic information gathering (RIG), providing strong theoretical guarantees for sensor placement (SP) and informative path planning (IPP). However, its high computational cost, dominated by repeated log-determinant evaluations, has limited its use in real-time planning.

-> 로봇 정보 수집을 위한 빠른 상호 정보 계산. 스포츠 촬영과 간접적 관련.

## 91위: Quantum walk inspired JPEG compression of images

- arXiv: http://arxiv.org/abs/2602.12306v1 | 2026-02-12 | final 62.0

This work proposes a quantum inspired adaptive quantization framework that enhances the classical JPEG compression by introducing a learned, optimized Qtable derived using a Quantum Walk Inspired Optimization (QWIO) search strategy. The optimizer searches a continuous parameter space of frequency band scaling factors under a unified rate distortion objective that jointly considers reconstruction fidelity and compression efficiency.

-> Image compression applicable to content generation

## 92위: Understanding the Fine-Grained Knowledge Capabilities of Vision-Language Models

- arXiv: http://arxiv.org/abs/2602.17871v1 | 2026-02-19 | final 60.4

Vision-language models (VLMs) have made substantial progress across a wide range of visual question answering benchmarks, spanning visual reasoning, document understanding, and multimodal dialogue. These improvements are evident in a wide range of VLMs built on a variety of base models, alignment architectures, and training data.

-> 비전-언어 모델의 세부 지능 분석이 스포츠 동작 인식에 간접적 참고 가능.

## 93위: Emergent Morphing Attack Detection in Open Multi-modal Large Language Models

- arXiv: http://arxiv.org/abs/2602.15461v1 | 2026-02-17 | final 60.4

Face morphing attacks threaten biometric verification, yet most morphing attack detection (MAD) systems require task-specific training and generalize poorly to unseen attack types. Meanwhile, open-source multimodal large language models (MLLMs) have demonstrated strong visual-linguistic reasoning, but their potential in biometric forensics remains underexplored.

-> 생체 보안을 위한 다중모달 모델 기술이 스포츠 이미지 분석에 간접적으로 참고될 수 있습니다.

## 94위: PLESS: Pseudo-Label Enhancement with Spreading Scribbles for Weakly Supervised Segmentation

- arXiv: http://arxiv.org/abs/2602.11628v1 | 2026-02-12 | final 60.4

Weakly supervised learning with scribble annotations uses sparse user-drawn strokes to indicate segmentation labels on a small subset of pixels. This annotation reduces the cost of dense pixel-wise labeling, but suffers inherently from noisy and incomplete supervision.

-> 약한 감독 분할 방법론으로 간접적 관련 있음

## 95위: Xray-Visual Models: Scaling Vision models on Industry Scale Data

- arXiv: http://arxiv.org/abs/2602.16918v1 | 2026-02-18 | final 60.0

We present Xray-Visual, a unified vision model architecture for large-scale image and video understanding trained on industry-scale social media data. Our model leverages over 15 billion curated image-text pairs and 10 billion video-hashtag pairs from Facebook and Instagram, employing robust data curation pipelines that incorporate balancing and noise suppression strategies to maximize semantic diversity while minimizing label noise.

-> 대규모 비전 모델로 스포츠 비디오 이해에 간접적 적용 가능.

## 96위: Quasi-Periodic Gaussian Process Predictive Iterative Learning Control

- arXiv: http://arxiv.org/abs/2602.18014v1 | 2026-02-20 | final 60.0

Repetitive motion tasks are common in robotics, but performance can degrade over time due to environmental changes and robot wear and tear. Iterative learning control (ILC) improves performance by using information from previous iterations to compensate for expected errors in future iterations.

-> 반복 작업 제어가 운동 분석에 간접적으로 적용 가능

## 97위: Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies

- arXiv: http://arxiv.org/abs/2602.18291v1 | 2026-02-20 | final 60.0

Online Multi-Agent Reinforcement Learning (MARL) is a prominent framework for efficient agent coordination. Crucially, enhancing policy expressiveness is pivotal for achieving superior performance.

-> Multi-agent coordination methodology potentially applicable to future sports AI hardware.

## 98위: FlowHOI: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation

- arXiv: http://arxiv.org/abs/2602.13444v1 | 2026-02-13 | final 58.0

Recent vision-language-action (VLA) models can generate plausible end-effector motions, yet they often fail in long-horizon, contact-rich tasks because the underlying hand-object interaction (HOI) structure is not explicitly represented. An embodiment-agnostic interaction representation that captures this structure would make manipulation behaviors easier to validate and transfer across robots.

-> 손-물체 상호작용 생성이 스포츠 분석에 간접적 참고 가능.

## 99위: DCDM: Divide-and-Conquer Diffusion Models for Consistency-Preserving Video Generation

- arXiv: http://arxiv.org/abs/2602.13637v1 | 2026-02-14 | final 58.0

Recent video generative models have demonstrated impressive visual fidelity, yet they often struggle with semantic, geometric, and identity consistency. In this paper, we propose a system-level framework, termed the Divide-and-Conquer Diffusion Model (DCDM), to address three key challenges: (1) intra-clip world knowledge consistency, (2) inter-clip camera consistency, and (3) inter-shot element consistency.

-> Video generation consistency; indirect relevance to highlight editing.

## 100위: Self-Refining Vision Language Model for Robotic Failure Detection and Reasoning

- arXiv: http://arxiv.org/abs/2602.12405v1 | 2026-02-12 | final 58.0

Reasoning about failures is crucial for building reliable and trustworthy robotic systems. Prior approaches either treat failure reasoning as a closed-set classification problem or assume access to ample human annotations.

-> 실패 감지 및 추론 기술이 스포츠 오류 분석에 적용 가능

## 101위: ROIX-Comp: Optimizing X-ray Computed Tomography Imaging Strategy for Data Reduction and Reconstruction

- arXiv: http://arxiv.org/abs/2602.15917v1 | 2026-02-17 | final 58.0

In high-performance computing (HPC) environments, particularly in synchrotron radiation facilities, vast amounts of X-ray images are generated. Processing large-scale X-ray Computed Tomography (X-CT) datasets presents significant computational and storage challenges due to their high dimensionality and data volume.

-> 데이터 압축 기술이 엣지 장비 영상 처리에 간접적 참고 가능

## 102위: LaViDa-R1: Advancing Reasoning for Unified Multimodal Diffusion Language Models

- arXiv: http://arxiv.org/abs/2602.14147v1 | 2026-02-15 | final 58.0

Diffusion language models (dLLMs) recently emerged as a promising alternative to auto-regressive LLMs. The latest works further extended it to multimodal understanding and generation tasks.

-> Multimodal diffusion model for image editing and generation tasks, indirectly relevant to video/image correction.

## 103위: Location as a service with a MEC architecture

- arXiv: http://arxiv.org/abs/2602.13358v1 | 2026-02-13 | final 58.0

In recent years, automated driving has become viable, and advanced driver assistance systems (ADAS) are now part of modern cars. These systems require highly precise positioning.

-> Mobile Edge Computing for localization; indirectly relevant to edge device architecture.

## 104위: GR-Diffusion: 3D Gaussian Representation Meets Diffusion in Whole-Body PET Reconstruction

- arXiv: http://arxiv.org/abs/2602.11653v1 | 2026-02-12 | final 58.0

Positron emission tomography (PET) reconstruction is a critical challenge in molecular imaging, often hampered by noise amplification, structural blurring, and detail loss due to sparse sampling and the ill-posed nature of inverse problems. The three-dimensional discrete Gaussian representation (GR), which efficiently encodes 3D scenes using parameterized discrete Gaussian distributions, has shown promise in computer vision.

-> PET reconstruction method indirectly applicable

## 105위: LUVE : Latent-Cascaded Ultra-High-Resolution Video Generation with Dual Frequency Experts

- arXiv: http://arxiv.org/abs/2602.11564v1 | 2026-02-12 | final 56.4

Recent advances in video diffusion models have significantly improved visual quality, yet ultra-high-resolution (UHR) video generation remains a formidable challenge due to the compounded difficulties of motion modeling, semantic planning, and detail synthesis. To address these limitations, we propose \textbf{LUVE}, a \textbf{L}atent-cascaded \textbf{U}HR \textbf{V}ideo generation framework built upon dual frequency \textbf{E}xperts.

-> 초고해상도 비디오 생성 기술이 하이라이트 생성에 간접적으로 활용 가능함.

## 106위: MUOT_3M: A 3 Million Frame Multimodal Underwater Benchmark and the MUTrack Tracking Method

- arXiv: http://arxiv.org/abs/2602.18006v1 | 2026-02-20 | final 56.0

Underwater Object Tracking (UOT) is crucial for efficient marine robotics, large scale ecological monitoring, and ocean exploration; however, progress has been hindered by the scarcity of large, multimodal, and diverse datasets. Existing benchmarks remain small and RGB only, limiting robustness under severe color distortion, turbidity, and low visibility conditions.

-> Real-time object tracking weakly related to movement analysis

## 107위: HS-3D-NeRF: 3D Surface and Hyperspectral Reconstruction From Stationary Hyperspectral Images Using Multi-Channel NeRFs

- arXiv: http://arxiv.org/abs/2602.16950v1 | 2026-02-18 | final 56.0

Advances in hyperspectral imaging (HSI) and 3D reconstruction have enabled accurate, high-throughput characterization of agricultural produce quality and plant phenotypes, both essential for advancing agricultural sustainability and breeding programs. HSI captures detailed biochemical features of produce, while 3D geometric data substantially improves morphological analysis.

-> Agricultural 3D reconstruction; indirect relevance to motion analysis.

## 108위: SPRig: Self-Supervised Pose-Invariant Rigging from Mesh Sequences

- arXiv: http://arxiv.org/abs/2602.12740v1 | 2026-02-13 | final 54.8

State-of-the-art rigging methods assume a canonical rest pose--an assumption that fails for sequential data (e.g., animal motion capture or AIGC/video-derived mesh sequences) that lack the T-pose. Applied frame-by-frame, these methods are not pose-invariant and produce topological inconsistencies across frames.

-> 메시 시퀀스의 포즈 불변 리깅 연구로 스포츠 동작 분석과 간접적 관련 있음.

## 109위: EgoPush: Learning End-to-End Egocentric Multi-Object Rearrangement for Mobile Robots

- arXiv: http://arxiv.org/abs/2602.18071v1 | 2026-02-20 | final 54.4

Humans can rearrange objects in cluttered environments using egocentric perception, navigating occlusions without global coordinates. Inspired by this capability, we study long-horizon multi-object non-prehensile rearrangement for mobile robots using a single egocentric camera.

-> 에고센트릭 인식 기술이 모바일 촬영에 약간 참고 가능

## 110위: Offline-Poly: A Polyhedral Framework For Offline 3D Multi-Object Tracking

- arXiv: http://arxiv.org/abs/2602.13772v1 | 2026-02-14 | final 54.0

Offline 3D multi-object tracking (MOT) is a critical component of the 4D auto-labeling (4DAL) process. It enhances pseudo-labels generated by high-performance detectors through the incorporation of temporal context.

-> Offline object tracking methodology for motion refinement

## 111위: Learning Perceptual Representations for Gaming NR-VQA with Multi-Task FR Signals

- arXiv: http://arxiv.org/abs/2602.11903v2 | 2026-02-12 | final 54.0

No-reference video quality assessment (NR-VQA) for gaming videos is challenging due to limited human-rated datasets and unique content characteristics including fast motion, stylized graphics, and compression artifacts. We present MTL-VQA, a multi-task learning framework that uses full-reference metrics as supervisory signals to learn perceptually meaningful features without human labels for pretraining.

-> 게임 비디오 품질 평가 방법이 영상 보정에 간접적 참고 가능

## 112위: T2MBench: A Benchmark for Out-of-Distribution Text-to-Motion Generation

- arXiv: http://arxiv.org/abs/2602.13751v1 | 2026-02-14 | final 54.0

Most existing evaluations of text-to-motion generation focus on in-distribution textual inputs and a limited set of evaluation criteria, which restricts their ability to systematically assess model generalization and motion generation capabilities under complex out-of-distribution (OOD) textual conditions. To address this limitation, we propose a benchmark specifically designed for OOD text-to-motion evaluation, which includes a comprehensive analysis of 14 representative baseline models and the two datasets derived from evaluation results.

-> 텍스트-동작 벤치마크가 스포츠 동작 모델 평가에 참고 가능.

## 113위: Thermal Imaging for Contactless Cardiorespiratory and Sudomotor Response Monitoring

- arXiv: http://arxiv.org/abs/2602.12361v1 | 2026-02-12 | final 54.0

Thermal infrared imaging captures skin temperature changes driven by autonomic regulation and can potentially provide contactless estimation of electrodermal activity (EDA), heart rate (HR), and breathing rate (BR). While visible-light methods address HR and BR, they cannot access EDA, a standard marker of sympathetic activation.

-> 운동 중 생리신호 모니터링 기술 간접적 참조

## 114위: CRAFT: Adapting VLA Models to Contact-rich Manipulation via Force-aware Curriculum Fine-tuning

- arXiv: http://arxiv.org/abs/2602.12532v1 | 2026-02-13 | final 54.0

Vision-Language-Action (VLA) models have shown a strong capability in enabling robots to execute general instructions, yet they struggle with contact-rich manipulation tasks, where success requires precise alignment, stable contact maintenance, and effective handling of deformable objects. A fundamental challenge arises from the imbalance between high-entropy vision and language inputs and low-entropy but critical force signals, which often leads to over-reliance on perception and unstable control.

-> 물리적 상호작용 분석 기술이 스포츠 동작 분석에 간접적 참고

## 115위: Synthetic Image Detection with CLIP: Understanding and Assessing Predictive Cues

- arXiv: http://arxiv.org/abs/2602.12381v1 | 2026-02-12 | final 54.0

Recent generative models produce near-photorealistic images, challenging the trustworthiness of photographs. Synthetic image detection (SID) has thus become an important area of research.

-> 합성 이미지 감지 기술이 간접적으로 관련됨

## 116위: PosterOmni: Generalized Artistic Poster Creation via Task Distillation and Unified Reward Feedback

- arXiv: http://arxiv.org/abs/2602.12127v1 | 2026-02-12 | final 54.0

Image-to-poster generation is a high-demand task requiring not only local adjustments but also high-level design understanding. Models must generate text, layout, style, and visual elements while preserving semantic fidelity and aesthetic coherence.

-> 이미지 편집 기술이 프로젝트의 영상 보정과 관련 있음.

## 117위: MAPLE: Modality-Aware Post-training and Learning Ecosystem

- arXiv: http://arxiv.org/abs/2602.11596v1 | 2026-02-12 | final 54.0

Multimodal language models now integrate text, audio, and video for unified reasoning. Yet existing RL post-training pipelines treat all input signals as equally relevant, ignoring which modalities each task actually requires.

-> Indirect multimodal training

## 118위: Train Short, Inference Long: Training-free Horizon Extension for Autoregressive Video Generation

- arXiv: http://arxiv.org/abs/2602.14027v2 | 2026-02-15 | final 54.0

Autoregressive video diffusion models have emerged as a scalable paradigm for long video generation. However, they often suffer from severe extrapolation failure, where rapid error accumulation leads to significant temporal degradation when extending beyond training horizons.

-> Video generation extension indirectly useful for highlight editing.

## 119위: A Comparative Analysis of Social Network Topology in Reddit and Moltbook

- arXiv: http://arxiv.org/abs/2602.13920v2 | 2026-02-14 | final 54.0

Recent advances in agent-mediated systems have enabled a new paradigm of social network simulation, where AI agents interact with human-like autonomy. This evolution has fostered the emergence of agent-driven social networks such as Moltbook, a Reddit-like platform populated entirely by AI agents.

-> Social network analysis relevant to sharing platform.

## 120위: EasyMimic: A Low-Cost Framework for Robot Imitation Learning from Human Videos

- arXiv: http://arxiv.org/abs/2602.11464v1 | 2026-02-12 | final 52.4

Robot imitation learning is often hindered by the high cost of collecting large-scale, real-world data. This challenge is especially significant for low-cost robots designed for home use, as they must be both user-friendly and affordable.

-> 로봇 모방 학습 프레임워크가 스포츠 동작 분석에 간접적으로 참고될 수 있음.

## 121위: Learning Proposes, Geometry Disposes: A Modular Framework for Efficient Spatial Reasoning

- arXiv: http://arxiv.org/abs/2602.14409v1 | 2026-02-16 | final 52.0

Spatial perception aims to estimate camera motion and scene structure from visual observations, a problem traditionally addressed through geometric modeling and physical consistency constraints. Recent learning-based methods have demonstrated strong representational capacity for geometric perception and are increasingly used to augment classical geometry-centric systems in practice.

-> Modular spatial reasoning for pose estimation reference

## 122위: LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion

- arXiv: http://arxiv.org/abs/2602.12215v1 | 2026-02-12 | final 50.0

Recent robot foundation models largely rely on large-scale behavior cloning, which imitates expert actions but discards transferable dynamics knowledge embedded in heterogeneous embodied data. While the Unified World Model (UWM) formulation has the potential to leverage such diverse data, existing instantiations struggle to scale to foundation-level due to coarse data usage and fragmented datasets.

-> 로봇 기초 모델로 스포츠 동작 분석에 참고될 수 있으나 직접적 관련성은 낮음.

## 123위: GSM-GS: Geometry-Constrained Single and Multi-view Gaussian Splatting for Surface Reconstruction

- arXiv: http://arxiv.org/abs/2602.12796v1 | 2026-02-13 | final 50.0

Recently, 3D Gaussian Splatting has emerged as a prominent research direction owing to its ultrarapid training speed and high-fidelity rendering capabilities. However, the unstructured and irregular nature of Gaussian point clouds poses challenges to reconstruction accuracy.

-> 3D reconstruction method, indirectly useful for posture analysis.

## 124위: Projected Representation Conditioning for High-fidelity Novel View Synthesis

- arXiv: http://arxiv.org/abs/2602.12003v1 | 2026-02-12 | final 50.0

We propose a novel framework for diffusion-based novel view synthesis in which we leverage external representations as conditions, harnessing their geometric and semantic correspondence properties for enhanced geometric consistency in generated novel viewpoints. First, we provide a detailed analysis exploring the correspondence capabilities emergent in the spatial attention of external visual representations.

-> Indirectly related to view synthesis for potential analysis.

## 125위: Symmetry-Aware Fusion of Vision and Tactile Sensing via Bilateral Force Priors for Robotic Manipulation

- arXiv: http://arxiv.org/abs/2602.13689v1 | 2026-02-14 | final 50.0

Insertion tasks in robotic manipulation demand precise, contact-rich interactions that vision alone cannot resolve. While tactile feedback is intuitively valuable, existing studies have shown that naïve visuo-tactile fusion often fails to deliver consistent improvements.

-> 다중 감각 융합 기술이 동작 분석과 간접적 관련 있음.

## 126위: Direction Matters: Learning Force Direction Enables Sim-to-Real Contact-Rich Manipulation

- arXiv: http://arxiv.org/abs/2602.14174v1 | 2026-02-15 | final 50.0

Sim-to-real transfer for contact-rich manipulation remains challenging due to the inherent discrepancy in contact dynamics. While existing methods often rely on costly real-world data or utilize blind compliance through fixed controllers, we propose a framework that leverages expert-designed controller logic for transfer.

-> Sim-to-real transfer indirectly related to physical hardware.

---

## 다시 보기

### Parallel Complex Diffusion for Scalable Time Series Generation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.17706v1
- 점수: final 85.6

Modeling long-range dependencies in time series generation poses a fundamental trade-off between representational capacity and computational efficiency. Traditional temporal diffusion models suffer from local entanglement and the $\mathcal{O}(L^2)$ cost of attention mechanisms. We address these limitations by introducing PaCoDi (Parallel Complex Diffusion), a spectral-native architecture that decouples generative modeling in the frequency domain. PaCoDi fundamentally alters the problem topology: the Fourier Transform acts as a diagonalizing operator, converting locally coupled temporal signals into globally decorrelated spectral components. Theoretically, we prove the Quadrature Forward Diffusion and Conditional Reverse Factorization theorem, demonstrating that the complex diffusion process can be split into independent real and imaginary branches. We bridge the gap between this decoupled theory and data reality using a \textbf{Mean Field Theory (MFT) approximation} reinforced by an interactive correction mechanism. Furthermore, we generalize this discrete DDPM to continuous-time Frequency SDEs, rigorously deriving the Spectral Wiener Process describe the differential spectral Brownian motion limit. Crucially, PaCoDi exploits the Hermitian Symmetry of real-valued signals to compress the sequence length by half, achieving a 50% reduction in attention FLOPs without information loss. We further derive a rigorous Heteroscedastic Loss to handle the non-isotropic noise distribution on the compressed manifold. Extensive experiments show that PaCoDi outperforms existing baselines in both generation quality and inference speed, offering a theoretically grounded and computationally efficient solution for time series modeling.

-> 이 논문은 시계열 생성의 계산 효율성을 개선해 에지 디바이스에서 비디오 처리 속도 향상에 기여합니다. 주파수 영역 압축으로 FLOPs 50% 감소와 낮은 지연 시간을 달성해 RK3588에서 실시간 하이라이트 생성이 가능합니다.

### VideoWorld 2: Learning Transferable Knowledge from Real-world Videos (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.10102v1
- 점수: final 82.4

Learning transferable knowledge from unlabeled video data and applying it in new environments is a fundamental capability of intelligent agents. This work presents VideoWorld 2, which extends VideoWorld and offers the first investigation into learning transferable knowledge directly from raw real-world videos. At its core, VideoWorld 2 introduces a dynamic-enhanced Latent Dynamics Model (dLDM) that decouples action dynamics from visual appearance: a pretrained video diffusion model handles visual appearance modeling, enabling the dLDM to learn latent codes that focus on compact and meaningful task-related dynamics. These latent codes are then modeled autoregressively to learn task policies and support long-horizon reasoning. We evaluate VideoWorld 2 on challenging real-world handcraft making tasks, where prior video generation and latent-dynamics models struggle to operate reliably. Remarkably, VideoWorld 2 achieves up to 70% improvement in task success rate and produces coherent long execution videos. In robotics, we show that VideoWorld 2 can acquire effective manipulation knowledge from the Open-X dataset, which substantially improves task performance on CALVIN. This study reveals the potential of learning transferable world knowledge directly from raw videos, with all code, data, and models to be open-sourced for further research.

-> 동작과 시각적 요소를 분리하는 기술로 스포츠 동작 분석 정확도 향상에 직접 활용 가능합니다. 잠재 코드 기반 동역학 모델이 선수의 자세 패턴 인식에 적용되어 경기 전략 분석 품질을 높입니다.

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
