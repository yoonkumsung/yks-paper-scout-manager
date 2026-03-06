# CAPP!C_AI 논문 리포트 (2026-03-06)

> 수집 89 | 필터 81 | 폐기 0 | 평가 17 | 출력 15 | 기준 50점

검색 윈도우: 2026-03-05T00:00:00+00:00 ~ 2026-03-06T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 30

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: SURE: Semi-dense Uncertainty-REfined Feature Matching

- arXiv: http://arxiv.org/abs/2603.04869v1
- PDF: https://arxiv.org/pdf/2603.04869v1
- 코드: https://github.com/LSC-ALAN/SURE
- 발행일: 2026-03-05
- 카테고리: cs.CV
- 점수: final 96.0 (llm_adjusted:95 = base:82 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Establishing reliable image correspondences is essential for many robotic vision problems. However, existing methods often struggle in challenging scenarios with large viewpoint changes or textureless regions, where incorrect cor- respondences may still receive high similarity scores. This is mainly because conventional models rely solely on fea- ture similarity, lacking an explicit mechanism to estimate the reliability of predicted matches, leading to overconfident errors. To address this issue, we propose SURE, a Semi- dense Uncertainty-REfined matching framework that jointly predicts correspondences and their confidence by modeling both aleatoric and epistemic uncertainties. Our approach in- troduces a novel evidential head for trustworthy coordinate regression, along with a lightweight spatial fusion module that enhances local feature precision with minimal overhead. We evaluated our method on multiple standard benchmarks, where it consistently outperforms existing state-of-the-art semi-dense matching models in both accuracy and efficiency. our code will be available on https://github.com/LSC-ALAN/SURE.

**선정 근거**
시각적 표현 향상 기술은 스포츠 영상 보정 및 하이라이트 편집에 직접적으로 적용 가능하며, 자세 및 동작 분석에도 활용될 수 있습니다.

**활용 인사이트**
DCR 방식을 도입하여 영상 생성 시 대비 신호를 활용하면 사실적인 스포츠 장면 구현과 동시에 개인별 특징을 강조하는 하이라이트 영상 제작이 가능해집니다.

## 2위: Trainable Bitwise Soft Quantization for Input Feature Compression

- arXiv: http://arxiv.org/abs/2603.05172v1
- PDF: https://arxiv.org/pdf/2603.05172v1
- 발행일: 2026-03-05
- 카테고리: cs.LG
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
The growing demand for machine learning applications in the context of the Internet of Things calls for new approaches to optimize the use of limited compute and memory resources. Despite significant progress that has been made w.r.t. reducing model sizes and improving efficiency, many applications still require remote servers to provide the required resources. However, such approaches rely on transmitting data from edge devices to remote servers, which may not always be feasible due to bandwidth, latency, or energy constraints. We propose a task-specific, trainable feature quantization layer that compresses the input features of a neural network. This can significantly reduce the amount of data that needs to be transferred from the device to a remote server. In particular, the layer allows each input feature to be quantized to a user-defined number of bits, enabling a simple on-device compression at the time of data collection. The layer is designed to approximate step functions with sigmoids, enabling trainable quantization thresholds. By concatenating outputs from multiple sigmoids, introduced as bitwise soft quantization, it achieves trainable quantized values when integrated with a neural network. We compare our method to full-precision inference as well as to several quantization baselines. Experiments show that our approach outperforms standard quantization methods, while maintaining accuracy levels close to those of full-precision models. In particular, depending on the dataset, compression factors of $5\times$ to $16\times$ can be achieved compared to $32$-bit input without significant performance loss.

**선정 근거**
RK3588 엣지 장치에서 데이터 전송량을 최대 16배 줄여 실시간 처리와 배터리 효율성을 크게 향상시킬 수 있습니다.

**활용 인사이트**
비트 단위 소프트 양자화를 통해 스포츠 영상을 장치 내에서 압축 처리하여 클라우드 전송 없이도 고성능 분석 가능.

## 3위: Person Detection and Tracking from an Overhead Crane LiDAR

- arXiv: http://arxiv.org/abs/2603.04938v1
- PDF: https://arxiv.org/pdf/2603.04938v1
- 코드: https://github.com/nilushacj/O-LiPeDeT-Overhead-LiDAR-Person-Detection-and-Tracking
- 발행일: 2026-03-05
- 카테고리: cs.CV, cs.LG, cs.RO
- 점수: final 90.4 (llm_adjusted:88 = base:75 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
This paper investigates person detection and tracking in an industrial indoor workspace using a LiDAR mounted on an overhead crane. The overhead viewpoint introduces a strong domain shift from common vehicle-centric LiDAR benchmarks, and limited availability of suitable public training data. Henceforth, we curate a site-specific overhead LiDAR dataset with 3D human bounding-box annotations and adapt selected candidate 3D detectors under a unified training and evaluation protocol. We further integrate lightweight tracking-by-detection using AB3DMOT and SimpleTrack to maintain person identities over time. Detection performance is reported with distance-sliced evaluation to quantify the practical operating envelope of the sensing setup. The best adapted detector configurations achieve average precision (AP) up to 0.84 within a 5.0 m horizontal radius, increasing to 0.97 at 1.0 m, with VoxelNeXt and SECOND emerging as the most reliable backbones across this range. The acquired results contribute in bridging the domain gap between standard driving datasets and overhead sensing for person detection and tracking. We also report latency measurements, highlighting practical real-time feasibility. Finally, we release our dataset and implementations in GitHub to support further research

**선정 근거**
산업 현장의 LiDAR 기반 사람 감지 및 추적 기술로 스포츠 촬영과 유사한 추적 기술이 적용 가능하나 도메인이 상이함

## 4위: VisionPangu: A Compact and Fine-Grained Multimodal Assistant with 1.7B Parameters

- arXiv: http://arxiv.org/abs/2603.04957v1
- PDF: https://arxiv.org/pdf/2603.04957v1
- 발행일: 2026-03-05
- 카테고리: cs.CV, cs.CL
- 점수: final 90.4 (llm_adjusted:88 = base:80 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Large Multimodal Models (LMMs) have achieved strong performance in vision-language understanding, yet many existing approaches rely on large-scale architectures and coarse supervision, which limits their ability to generate detailed image captions. In this work, we present VisionPangu, a compact 1.7B-parameter multimodal model designed to improve detailed image captioning through efficient multimodal alignment and high-quality supervision. Our model combines an InternVL-derived vision encoder with the OpenPangu-Embedded language backbone via a lightweight MLP projector and adopts an instruction-tuning pipeline inspired by LLaVA. By incorporating dense human-authored descriptions from the DOCCI dataset, VisionPangu improves semantic coherence and descriptive richness without relying on aggressive model scaling. Experimental results demonstrate that compact multimodal models can achieve competitive performance while producing more structured and detailed captions. The code and model weights will be publicly available at https://www.modelscope.cn/models/asdfgh007/visionpangu.

**선정 근거**
1.7B 파라미터로 경량화된 멀티모달 모델이 엣지 장치에서 스포츠 장면의 상세한 이해를 가능하게 합니다.

**활용 인사이트**
InternVL 비전 인코더와 OpenPangu 언어 모델을 결합하여 스포츠 영상의 세부적인 동작 분석과 자연스러운 설명 생성이 가능.

## 5위: Guiding Diffusion-based Reconstruction with Contrastive Signals for Balanced Visual Representation

- arXiv: http://arxiv.org/abs/2603.04803v1
- PDF: https://arxiv.org/pdf/2603.04803v1
- 코드: https://github.com/boyuh/DCR
- 발행일: 2026-03-05
- 카테고리: cs.CV, cs.AI, cs.LG
- 점수: final 86.4 (llm_adjusted:83 = base:80 + bonus:+3)
- 플래그: 코드 공개

**개요**
The limited understanding capacity of the visual encoder in Contrastive Language-Image Pre-training (CLIP) has become a key bottleneck for downstream performance. This capacity includes both Discriminative Ability (D-Ability), which reflects class separability, and Detail Perceptual Ability (P-Ability), which focuses on fine-grained visual cues. Recent solutions use diffusion models to enhance representations by conditioning image reconstruction on CLIP visual tokens. We argue that such paradigms may compromise D-Ability and therefore fail to effectively address CLIP's representation limitations. To address this, we integrate contrastive signals into diffusion-based reconstruction to pursue more comprehensive visual representations. We begin with a straightforward design that augments the diffusion process with contrastive learning on input images. However, empirical results show that the naive combination suffers from gradient conflict and yields suboptimal performance. To balance the optimization, we introduce the Diffusion Contrastive Reconstruction (DCR), which unifies the learning objective. The key idea is to inject contrastive signals derived from each reconstructed image, rather than from the original input, into the diffusion process. Our theoretical analysis shows that the DCR loss can jointly optimize D-Ability and P-Ability. Extensive experiments across various benchmarks and multi-modal large language models validate the effectiveness of our method. The code is available at https://github.com/boyuh/DCR.

**선정 근거**
일반적인 시각적 표현 기술이 스포츠 영상 분석에 적용 가능

## 6위: Scalable Injury-Risk Screening in Baseball Pitching From Broadcast Video

- arXiv: http://arxiv.org/abs/2603.04864v1
- PDF: https://arxiv.org/pdf/2603.04864v1
- 발행일: 2026-03-05
- 카테고리: cs.CV
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Injury prediction in pitching depends on precise biomechanical signals, yet gold-standard measurements come from expensive, stadium-installed multi-camera systems that are unavailable outside professional venues. We present a monocular video pipeline that recovers 18 clinically relevant biomechanics metrics from broadcast footage, positioning pose-derived kinematics as a scalable source for injury-risk modeling. Built on DreamPose3D, our approach introduces a drift-controlled global lifting module that recovers pelvis trajectory via velocity-based parameterization and sliding-window inference, lifting pelvis-rooted poses into global space. To address motion blur, compression artifacts, and extreme pitching poses, we incorporate a kinematics refinement pipeline with bone-length constraints, joint-limited inverse kinematics, smoothing, and symmetry constraints to ensure temporally stable and physically plausible kinematics. On 13 professional pitchers (156 paired pitches), 16/18 metrics achieve sub-degree agreement (MAE $< 1^{\circ}$). Using these metrics for injury prediction, an automated screening model achieves AUC 0.811 for Tommy John surgery and 0.825 for significant arm injuries on 7,348 pitchers. The resulting pose-derived metrics support scalable injury-risk screening, establishing monocular broadcast video as a viable alternative to stadium-scale motion capture for biomechanics.

**선정 근거**
야구 투구 자세 분석과 부상 예측에 관한 연구로 스포츠 동작 분석 기술이 적용 가능하나, 특정 스포츠(야구)와 부상 예측에 초점이 맞춰져 있어 일반적인 스포츠 촬영 및 하이라이트 편집과는 직접적 연관성은 떨어집니다.

**활용 인사이트**
모노큘러 비디오를 통한 생역학적 지표 추출 기술은 우리 프로젝트의 자세 분석 및 동작 분석 기능에 직접 적용 가능하며, 특히 부상 위험 예측 모델링에 활용될 수 있습니다.

## 7위: Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding

- arXiv: http://arxiv.org/abs/2603.04977v1
- PDF: https://arxiv.org/pdf/2603.04977v1
- 코드: https://github.com/Haorane/VideoHV-Agent
- 발행일: 2026-03-05
- 카테고리: cs.CV
- 점수: final 84.8 (llm_adjusted:81 = base:78 + bonus:+3)
- 플래그: 코드 공개

**개요**
Long video understanding is challenging due to dense visual redundancy, long-range temporal dependencies, and the tendency of chain-of-thought and retrieval-based agents to accumulate semantic drift and correlation-driven errors. We argue that long-video reasoning should begin not with reactive retrieval, but with deliberate task formulation: the model must first articulate what must be true in the video for each candidate answer to hold. This thinking-before-finding principle motivates VideoHV-Agent, a framework that reformulates video question answering as a structured hypothesis-verification process. Based on video summaries, a Thinker rewrites answer candidates into testable hypotheses, a Judge derives a discriminative clue specifying what evidence must be checked, a Verifier grounds and tests the clue using localized, fine-grained video content, and an Answer agent integrates validated evidence to produce the final answer. Experiments on three long-video understanding benchmarks show that VideoHV-Agent achieves state-of-the-art accuracy while providing enhanced interpretability, improved logical soundness, and lower computational cost. We make our code publicly available at: https://github.com/Haorane/VideoHV-Agent.

**선정 근거**
장기 동영상 이해를 위한 가설-검증 다중 에이전트 프레임워크로, 스포츠 영상 분석에 적용 가능하지만 스포츠 특화나 실시간 처리는 다루지 않음.

**활용 인사이트**
Thinker-Judge-Verifier-Answer 에이전트 구조는 스포츠 경기 전략 분석 및 하이라이트 장면 식별에 활용될 수 있으며, 장기간의 경기 영상 분석에 효과적일 것입니다.

## 8위: AI+HW 2035: Shaping the Next Decade

- arXiv: http://arxiv.org/abs/2603.05225v1
- PDF: https://arxiv.org/pdf/2603.05225v1
- 발행일: 2026-03-05
- 카테고리: cs.AI, cs.AR
- 점수: final 84.0 (llm_adjusted:80 = base:75 + bonus:+5)
- 플래그: 엣지

**개요**
Artificial intelligence (AI) and hardware (HW) are advancing at unprecedented rates, yet their trajectories have become inseparably intertwined. The global research community lacks a cohesive, long-term vision to strategically coordinate the development of AI and HW. This fragmentation constrains progress toward holistic, sustainable, and adaptive AI systems capable of learning, reasoning, and operating efficiently across cloud, edge, and physical environments. The future of AI depends not only on scaling intelligence, but on scaling efficiency, achieving exponential gains in intelligence per joule, rather than unbounded compute consumption. Addressing this grand challenge requires rethinking the entire computing stack. This vision paper lays out a 10-year roadmap for AI+HW co-design and co-development, spanning algorithms, architectures, systems, and sustainability. We articulate key insights that redefine scaling around energy efficiency, system-level integration, and cross-layer optimization. We identify key challenges and opportunities, candidly assess potential obstacles and pitfalls, and propose integrated solutions grounded in algorithmic innovation, hardware advances, and software abstraction. Looking ahead, we define what success means in 10 years: achieving a 1000x improvement in efficiency for AI training and inference; enabling energy-aware, self-optimizing systems that seamlessly span cloud, edge, and physical AI; democratizing access to advanced AI infrastructure; and embedding human-centric principles into the design of intelligent systems. Finally, we outline concrete action items for academia, industry, government, and the broader community, calling for coordinated national initiatives, shared infrastructure, workforce development, cross-agency collaboration, and sustained public-private partnerships to ensure that AI+HW co-design becomes a unifying long-term mission.

**선정 근거**
AI+HW 공설계 접근법은 스포츠 촬영 및 분석을 위한 에지 디바이스의 효율성과 성능을 극대화하는 데 핵심적입니다. 특히 에너지 효율성 향상과 시스템 통합은 실시간 스포츠 영상 처리에 필수적입니다.

**활용 인사이트**
rk3588 기반의 AI 촬영 디바이스에 AI+HW 공설계 원리를 적용하여 1000x 효율성 향상을 달성하고, 클라우드-에지-물리적 환경에서의 실시간 스포츠 분석을 위한 자기 최적화 시스템을 구축할 수 있습니다.

## 9위: FluxSieve: Unifying Streaming and Analytical Data Planes for Scalable Cloud Observability

- arXiv: http://arxiv.org/abs/2603.04937v1
- PDF: https://arxiv.org/pdf/2603.04937v1
- 발행일: 2026-03-05
- 카테고리: cs.DB, cs.DC, cs.PF
- 점수: final 76.0 (llm_adjusted:70 = base:65 + bonus:+5)
- 플래그: 실시간

**개요**
Despite many advances in query optimization, indexing techniques, and data storage, modern data platforms still face difficulties in delivering robust query performance under high concurrency and computationally intensive queries. This challenge is particularly pronounced in large-scale observability platforms handling high-volume, high-velocity data records. For instance, recurrent, expensive filtering queries at query time impose substantial computational and storage overheads in the analytical data plane. In this paper, we propose FluxSieve, a unified architecture that reconciles traditional pull-based query processing with push-based stream processing by embedding a lightweight in-stream precomputation and filtering layer directly into the data ingestion path. This avoids the complexity and operational burden of running queries in dedicated stream processing frameworks. Concretely, this work (i) introduces a foundational architecture that unifies streaming and analytical data planes via in-stream filtering and records enrichment, (ii) designs a scalable multi-pattern matching mechanism that supports concurrent evaluation and on-the-fly updates of filtering rules with minimal per-record overhead, (iii) demonstrates how to integrate this ingestion-time processing with two open-source analytical systems -- Apache Pinot as a Real-Time Online Analytical Processing (RTOLAP) engine and DuckDB as an embedded analytical database, and (iv) performs comprehensive experimental evaluation of our approach. Our evaluation across different systems, query types, and performance metrics shows up to orders-of-magnitude improvements in query performance at the cost of negligible additional storage and very low computational overhead.

**선정 근거**
스트리밍 및 분석 데이터 플레인을 통합한 아키텍처로 실시간 비디오 처리에 간접적으로 적용 가능

## 10위: Can LLMs Synthesize Court-Ready Statistical Evidence? Evaluating AI-Assisted Sentencing Bias Analysis for California Racial Justice Act Claims

- arXiv: http://arxiv.org/abs/2603.04804v1
- PDF: https://arxiv.org/pdf/2603.04804v1
- 발행일: 2026-03-05
- 카테고리: cs.HC
- 점수: final 70.4 (llm_adjusted:63 = base:60 + bonus:+3)
- 플래그: 코드 공개

**개요**
Resentencing in California remains a complex legal challenge despite legislative reforms like the Racial Justice Act (2020), which allows defendants to challenge convictions based on statistical evidence of racial disparities in sentencing and charging. Policy implementation lags behind legislative intent, creating a 'second-chance gap' where hundreds of resentencing opportunities remain unidentified. We present Redo.io, an open-source platform that processes 95,000 prison records acquired under the California Public Records Act (CPRA) and generates court-ready statistical evidence of racial bias in sentencing for prima facie and discovery motions. We explore the design of an LLM-powered interpretive layer that synthesizes results from statistical methods like Odds Ratio, Relative Risk, and Chi-Square Tests into cohesive narratives contextualized with confidence intervals, sample sizes, and data limitations. Our evaluations comparing LLM performance to statisticians using the LLM-as-a-Judge framework suggest that AI can serve as a powerful descriptive assistant for real-time evidence generation when ethically incorporated in the analysis pipeline.

**선정 근거**
LLM을 통한 통계적 결과 해석에 초점을 맞춘 연구로 스포츠 전략 분석에 간접적으로 적용 가능

## 11위: Safe-SAGE: Social-Semantic Adaptive Guidance for Safe Engagement through Laplace-Modulated Poisson Safety Functions

- arXiv: http://arxiv.org/abs/2603.05497v1
- PDF: https://arxiv.org/pdf/2603.05497v1
- 발행일: 2026-03-05
- 카테고리: cs.RO
- 점수: final 68.0 (llm_adjusted:60 = base:55 + bonus:+5)
- 플래그: 실시간

**개요**
Traditional safety-critical control methods, such as control barrier functions, suffer from semantic blindness, exhibiting the same behavior around obstacles regardless of contextual significance. This limitation leads to the uniform treatment of all obstacles, despite their differing semantic meanings. We present Safe-SAGE (Social-Semantic Adaptive Guidance for Safe Engagement), a unified framework that bridges the gap between high-level semantic understanding and low-level safety-critical control through a Poisson safety function (PSF) modulated using a Laplace guidance field. Our approach perceives the environment by fusing multi-sensor point clouds with vision-based instance segmentation and persistent object tracking to maintain up-to-date semantics beyond the camera's field of view. A multi-layer safety filter is then used to modulate system inputs to achieve safe navigation using this semantic understanding of the environment. This safety filter consists of both a model predictive control layer and a control barrier function layer. Both layers utilize the PSF and flux modulation of the guidance field to introduce varying levels of conservatism and multi-agent passing norms for different obstacles in the environment. Our framework enables legged robots to navigate semantically rich, dynamic environments with context-dependent safety margins while maintaining rigorous safety guarantees.

**선정 근거**
스포츠 촬영 및 분석 플랫폼 설계에 디자인, AI, 도메인 지식 통합 프레임워크를 적용하여 사용자 경험을 향상시키고, 코드 없이 AI 시스템을 구축할 수 있는 방법을 제공함

**활용 인사이트**
스포츠 분석을 위한 사용자 맞춤형 GPT 시스템을 개발하여 경기 영상 자동 분석, 하이라이트 추출, 전략 분석 등을 코드 없이 구현하고, 디자인을 통해 사용자와 AI의 상호작용을 최적화할 수 있음

## 12위: Exploiting Intermediate Reconstructions in Optical Coherence Tomography for Test-Time Adaption of Medical Image Segmentation

- arXiv: http://arxiv.org/abs/2603.05041v1
- PDF: https://arxiv.org/pdf/2603.05041v1
- 발행일: 2026-03-05
- 카테고리: cs.CV
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
Primary health care frequently relies on low-cost imaging devices, which are commonly used for screening purposes. To ensure accurate diagnosis, these systems depend on advanced reconstruction algorithms designed to approximate the performance of high-quality counterparts. Such algorithms typically employ iterative reconstruction methods that incorporate domain-specific prior knowledge. However, downstream task performance is generally assessed using only the final reconstructed image, thereby disregarding the informative intermediate representations generated throughout the reconstruction process. In this work, we propose IRTTA to exploit these intermediate representations at test-time by adapting the normalization-layer parameters of a frozen downstream network via a modulator network that conditions on the current reconstruction timescale. The modulator network is learned during test-time using an averaged entropy loss across all individual timesteps. Variation among the timestep-wise segmentations additionally provides uncertainty estimates at no extra cost. This approach enhances segmentation performance and enables semantically meaningful uncertainty estimation, all without modifying either the reconstruction process or the downstream model.

**선정 근거**
의료 영상 분할 관련 기술로 스포츠 촬영과 간접적 연관성 있음

## 13위: The Trilingual Triad Framework: Integrating Design, AI, and Domain Knowledge in No-code AI Smart City Course

- arXiv: http://arxiv.org/abs/2603.05036v1
- PDF: https://arxiv.org/pdf/2603.05036v1
- 발행일: 2026-03-05
- 카테고리: cs.AI
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
This paper introduces the "Trilingual Triad" framework, a model that explains how students learn to design with generative artificial intelligence (AI) through the integration of Design, AI, and Domain Knowledge. As generative AI rapidly enters higher education, students often engage with these systems as passive users of generated outputs rather than active creators of AI-enabled knowledge tools. This study investigates how students can transition from using AI as a tool to designing AI as a collaborative teammate. The research examines a graduate course, Creating the Frontier of No-code Smart Cities at the Singapore University of Technology and Design (SUTD), in which students developed domain-specific custom GPT systems without coding. Using a qualitative multi-case study approach, three projects - the Interview Companion GPT, the Urban Observer GPT, and Buddy Buddy - were analyzed across three dimensions: design, AI architecture, and domain expertise. The findings show that effective human-AI collaboration emerges when these three "languages" are orchestrated together: domain knowledge structures the AI's logic, design mediates human-AI interaction, and AI extends learners' cognitive capacity. The Trilingual Triad framework highlights how building AI systems can serve as a constructionist learning process that strengthens AI literacy, metacognition, and learner agency.

**선정 근거**
디자인, AI 및 도메인 지식을 통합한 프레임워크로 스포츠 분석 시스템 설계에 간접적으로 적용 가능

## 14위: VinePT-Map: Pole-Trunk Semantic Mapping for Resilient Autonomous Robotics in Vineyards

- arXiv: http://arxiv.org/abs/2603.05070v1
- PDF: https://arxiv.org/pdf/2603.05070v1
- 발행일: 2026-03-05
- 카테고리: cs.RO
- 점수: final 56.0 (llm_adjusted:45 = base:35 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Reliable long-term deployment of autonomous robots in agricultural environments remains challenging due to perceptual aliasing, seasonal variability, and the dynamic nature of crop canopies. Vineyards, characterized by repetitive row structures and significant visual changes across phenological stages, represent a pivotal field challenge, limiting the robustness of conventional feature-based localization and mapping approaches. This paper introduces VinePT-Map, a semantic mapping framework that leverages vine trunks and support poles as persistent structural landmarks to enable season-agnostic and resilient robot localization. The proposed method formulates the mapping problem as a factor graph, integrating GPS, IMU, and RGB-D observations through robust geometrical constraints that exploit vineyard structure. An efficient perception pipeline based on instance segmentation and tracking, combined with a clustering filter for outlier rejection and pose refinement, enables accurate landmark detection using low-cost sensors and onboard computation. To validate the pipeline, we present a multi-season dataset for trunk and pole segmentation and tracking. Extensive field experiments conducted across diverse seasons demonstrate the robustness and accuracy of the proposed approach, highlighting its suitability for long-term autonomous operation in agricultural environments.

**선정 근거**
농업 로봇ics 기술로 스포츠 촬영과 간접적 관련

## 15위: Judge Reliability Harness: Stress Testing the Reliability of LLM Judges

- arXiv: http://arxiv.org/abs/2603.05399v1
- PDF: https://arxiv.org/pdf/2603.05399v1
- 코드: https://github.com/RANDCorporation/judge-reliability-harness
- 발행일: 2026-03-05
- 카테고리: cs.AI
- 점수: final 50.4 (llm_adjusted:38 = base:35 + bonus:+3)
- 플래그: 코드 공개

**개요**
We present the Judge Reliability Harness, an open source library for constructing validation suites that test the reliability of LLM judges. As LLM based scoring is widely deployed in AI benchmarks, more tooling is needed to efficiently assess the reliability of these methods. Given a benchmark dataset and an LLM judge configuration, the harness generates reliability tests that evaluate both binary judgment accuracy and ordinal grading performance for free-response and agentic task formats. We evaluate four state-of-the-art judges across four benchmarks spanning safety, persuasion, misuse, and agentic behavior, and find meaningful variation in performance across models and perturbation types, highlighting opportunities to improve the robustness of LLM judges. No judge that we evaluated is uniformly reliable across benchmarks using our harness. For example, our preliminary experiments on judges revealed consistency issues as measured by accuracy in judging another LLM's ability to complete a task due to simple text formatting changes, paraphrasing, changes in verbosity, and flipping the ground truth label in LLM-produced responses. The code for this tool is available at: https://github.com/RANDCorporation/judge-reliability-harness

**선정 근거**
LLM 판사의 신뢰성 테스트에 초점을 맞춘 도구로 스포츠 영상 분석과 직접적인 연관성이 부족함

---

## 다시 보기

### Helios: Real Real-Time Long Video Generation Model (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04379v1
- 점수: final 98.4

We introduce Helios, the first 14B video generation model that runs at 19.5 FPS on a single NVIDIA H100 GPU and supports minute-scale generation while matching the quality of a strong baseline. We make breakthroughs along three key dimensions: (1) robustness to long-video drifting without commonly used anti-drifting heuristics such as self-forcing, error-banks, or keyframe sampling; (2) real-time generation without standard acceleration techniques such as KV-cache, sparse/linear attention, or quantization; and (3) training without parallelism or sharding frameworks, enabling image-diffusion-scale batch sizes while fitting up to four 14B models within 80 GB of GPU memory. Specifically, Helios is a 14B autoregressive diffusion model with a unified input representation that natively supports T2V, I2V, and V2V tasks. To mitigate drifting in long-video generation, we characterize typical failure modes and propose simple yet effective training strategies that explicitly simulate drifting during training, while eliminating repetitive motion at its source. For efficiency, we heavily compress the historical and noisy context and reduce the number of sampling steps, yielding computational costs comparable to -- or lower than -- those of 1.3B video generative models. Moreover, we introduce infrastructure-level optimizations that accelerate both inference and training while reducing memory consumption. Extensive experiments demonstrate that Helios consistently outperforms prior methods on both short- and long-video generation. We plan to release the code, base model, and distilled model to support further development by the community.

-> 엣지 컴퓨팅 프레임워크가 AI 카메라 엣지 디바이스 배포에 직접적으로 관련

### Agentic Peer-to-Peer Networks: From Content Distribution to Capability and Action Sharing (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03753v1
- 점수: final 96.0

The ongoing shift of AI models from centralized cloud APIs to local AI agents on edge devices is enabling \textit{Client-Side Autonomous Agents (CSAAs)} -- persistent personal agents that can plan, access local context, and invoke tools on behalf of users. As these agents begin to collaborate by delegating subtasks directly between clients, they naturally form \emph{Agentic Peer-to-Peer (P2P) Networks}. Unlike classic file-sharing overlays where the exchanged object is static, hash-indexed content (e.g., files in BitTorrent), agentic overlays exchange \emph{capabilities and actions} that are heterogeneous, state-dependent, and potentially unsafe if delegated to untrusted peers. This article outlines the networking foundations needed to make such collaboration practical. We propose a plane-based reference architecture that decouples connectivity/identity, semantic discovery, and execution. Besides, we introduce signed, soft-state capability descriptors to support intent- and constraint-aware discovery. To cope with adversarial settings, we further present a \textit{tiered verification} spectrum: Tier~1 relies on reputation signals, Tier~2 applies lightweight canary challenge-response with fallback selection, and Tier~3 requires evidence packages such as signed tool receipts/traces (and, when applicable, attestation). Using a discrete-event simulator that models registry-based discovery, Sybil-style index poisoning, and capability drift, we show that tiered verification substantially improves end-to-end workflow success while keeping discovery latency near-constant and control-plane overhead modest.

-> 엣지 디바이스 및 AI 에이전트 기술이 프로젝트의 AI 촬영 장치 및 플랫폼에 직접 적용 가능

### RIVER: A Real-Time Interaction Benchmark for Video LLMs (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03985v1
- 점수: final 96.0

The rapid advancement of multimodal large language models has demonstrated impressive capabilities, yet nearly all operate in an offline paradigm, hindering real-time interactivity. Addressing this gap, we introduce the Real-tIme Video intERaction Bench (RIVER Bench), designed for evaluating online video comprehension. RIVER Bench introduces a novel framework comprising Retrospective Memory, Live-Perception, and Proactive Anticipation tasks, closely mimicking interactive dialogues rather than responding to entire videos at once. We conducted detailed annotations using videos from diverse sources and varying lengths, and precisely defined the real-time interactive format. Evaluations across various model categories reveal that while offline models perform well in single question-answering tasks, they struggle with real-time processing. Addressing the limitations of existing models in online video interaction, especially their deficiencies in long-term memory and future perception, we proposed a general improvement method that enables models to interact with users more flexibly in real time. We believe this work will significantly advance the development of real-time interactive video understanding models and inspire future research in this emerging field. Datasets and code are publicly available at https://github.com/OpenGVLab/RIVER.

-> 실시간 비디오 이해 및 상호작용 기술로 스포츠 경기 분석에 적용 가능

### Real Eyes Realize Faster: Gaze Stability and Pupil Novelty for Efficient Egocentric Learning (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04098v1
- 점수: final 96.0

Always-on egocentric cameras are increasingly used as demonstrations for embodied robotics, imitation learning, and assistive AR, but the resulting video streams are dominated by redundant and low-quality frames. Under the storage and battery constraints of wearable devices, choosing which frames to keep is as important as how to learn from them. We observe that modern eye-tracking headsets provide a continuous, training-free side channel that decomposes into two complementary axes: gaze fixation captures visual stability (quality), while pupil response captures arousal-linked moments (novelty). We operationalize this insight as a Dual-Criterion Frame Curator that first gates frames by gaze quality and then ranks the survivors by pupil-derived novelty. On the Visual Experience Dataset (VEDB), curated frames at 10% budget match the classification performance of the full stream, and naive signal fusion consistently destroys both contributions. The benefit is task-dependent: pupil ranking improves activity recognition, while gaze-only selection already dominates for scene recognition, confirming that the two signals serve genuinely different roles. Our method requires no model inference and operates at capture time, offering a path toward efficient, always-on egocentric data curation.

-> 안와 추적 데이터를 활용한 효율적인 프레임 선택 방식으로 스포츠 하이라이트 자동 추출에 직접 적용 가능

### Lambdas at the Far Edge: a Tale of Flying Lambdas and Lambdas on Wheels (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04008v1
- 점수: final 94.4

Aggregate Programming (AP) is a paradigm for programming the collective behaviour of sets of distributed devices, possibly situated at the network far edge, by relying on asynchronous proximity-based interactions. The eXchange Calculus (XC), a recently proposed foundational model for AP, is essentially a typed lambda calculus extended with an operator (the exchange operator) providing an implicit communication mechanism between neighbour devices. This paper provides a gentle introduction to XC and to its implementation as a C++ library, called FCPP. The FCPP library and toolchain has been mainly developed at the Department of Computer Science of the University of Turin, where Stefano Berardi spent most of his academic career conducting outstanding research about logical foundation of computer science and transmitting his passion for research to students and young researchers, often exploiting typed lambda calculi. An FCCP program is essentially a typed lambda term, and FCPP has been used to write code that has been deployed on devices at the far edge of the network, including rovers and (soon) Uncrewed Aerial Vehicles (UAVs); hence the title of the paper.

-> 엣지 컴퓨팅 프레임워크가 AI 카메라 엣지 디바이스 배포에 직접적으로 관련

### EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04090v1
- 점수: final 93.6

Egocentric human motion estimation is essential for AR/VR experiences, yet remains challenging due to limited body coverage from the egocentric viewpoint, frequent occlusions, and scarce labeled data. We present EgoPoseFormer v2, a method that addresses these challenges through two key contributions: (1) a transformer-based model for temporally consistent and spatially grounded body pose estimation, and (2) an auto-labeling system that enables the use of large unlabeled datasets for training. Our model is fully differentiable, introduces identity-conditioned queries, multi-view spatial refinement, causal temporal attention, and supports both keypoints and parametric body representations under a constant compute budget. The auto-labeling system scales learning to tens of millions of unlabeled frames via uncertainty-aware semi-supervised training. The system follows a teacher-student schema to generate pseudo-labels and guide training with uncertainty distillation, enabling the model to generalize to different environments. On the EgoBody3M benchmark, with a 0.8 ms latency on GPU, our model outperforms two state-of-the-art methods by 12.2% and 19.4% in accuracy, and reduces temporal jitter by 22.2% and 51.7%. Furthermore, our auto-labeling system further improves the wrist MPJPE by 13.1%.

-> 인간 동작 추정 기술이 스포츠 동작 및 자세 분석에 직접적으로 적용 가능하며 실시간 처리 가능

### Yolo-Key-6D: Single Stage Monocular 6D Pose Estimation with Keypoint Enhancements (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03879v1
- 점수: final 93.6

Estimating the 6D pose of objects from a single RGB image is a critical task for robotics and extended reality applications. However, state-of-the-art multi stage methods often suffer from high latency, making them unsuitable for real time use. In this paper, we present Yolo-Key-6D, a novel single stage, end-to-end framework for monocular 6D pose estimation designed for both speed and accuracy. Our approach enhances a YOLO based architecture by integrating an auxiliary head that regresses the 2D projections of an object's 3D bounding box corners. This keypoint detection task significantly improves the network's understanding of 3D geometry. For stable end-to-end training, we directly regress rotation using a continuous 9D representation projected to SO(3) via singular value decomposition. On the LINEMOD and LINEMOD-Occluded benchmarks, YOLO-Key-6D achieves competitive accuracy scores of 96.24% and 69.41%, respectively, with the ADD(-S) 0.1d metric, while proving itself to operate in real time. Our results demonstrate that a carefully designed single stage method can provide a practical and effective balance of performance and efficiency for real world deployment.

-> 6D 포즈 추정 기술이 스포츠 선수 자세 분석에 적용 가능하며 실시간 처리 가능

### Architecture and evaluation protocol for transformer-based visual object tracking in UAV applications (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03904v1
- 점수: final 93.6

Object tracking from Unmanned Aerial Vehicles (UAVs) is challenged by platform dynamics, camera motion, and limited onboard resources. Existing visual trackers either lack robustness in complex scenarios or are too computationally demanding for real-time embedded use. We propose an Modular Asynchronous Tracking Architecture (MATA) that combines a transformer-based tracker with an Extended Kalman Filter, integrating ego-motion compensation from sparse optical flow and an object trajectory model. We further introduce a hardware-independent, embedded oriented evaluation protocol and a new metric called Normalized time to Failure (NT2F) to quantify how long a tracker can sustain a tracking sequence without external help. Experiments on UAV benchmarks, including an augmented UAV123 dataset with synthetic occlusions, show consistent improvements in Success and NT2F metrics across multiple tracking processing frequency. A ROS 2 implementation on a Nvidia Jetson AGX Orin confirms that the evaluation protocol more closely matches real-time performance on embedded systems.

-> 객체 추적 기술이 스포츠 경기 자동 촬영에 적용 가능하며, 임베디드 시스템에서 실시간 작동 가능

### Toward Native ISAC Support in O-RAN Architectures for 6G (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03607v1
- 점수: final 93.6

ISAC is an emerging paradigm in 6G networks that enables environmental sensing using wireless communication infrastructure. Current O-RAN specifications lack the architectural primitives for sensing integration: no service models expose physical-layer observables, no execution frameworks support sub-millisecond sensing tasks, and fronthaul interfaces cannot correlate transmitted waveforms with their reflections.   This article proposes three extensions to O-RAN for monostatic sensing, where transmission and reception are co-located at the base station. First, we specify sensing dApps at the O-DU that process IQ samples to extract delay, Doppler, and angular features. Second, we define E2SM-SENS, a service model enabling xApps to subscribe to sensing telemetry with configurable periodicity. Third, we identify required Open Fronthaul metadata for waveform-echo association. We validate the architecture through a prototype implementation using beamforming and Full-Duplex operation, demonstrating closed-loop control with median end-to-end latency suitable for near-real-time sensing applications. While focused on monostatic configurations, the proposed interfaces extend to bistatic and cooperative sensing scenarios.

-> 저지연 센싱 기술이 실시간 스포츠 분석에 적용 가능

### Exploring Challenges in Developing Edge-Cloud-Native Applications Across Multiple Business Domains (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03738v1
- 점수: final 92.0

As the convergence of cloud computing and advanced networking continues to reshape modern software development, edge-cloud-native paradigms have become essential for enabling scalable, resilient, and agile digital services that depend on high-performance, low-latency, and reliable communication. This study investigates the practical challenges of developing, deploying, and maintaining edge-cloud-native applications through in-depth interviews with professionals from diverse domains, including IT, finance, healthcare, education, and industry. Despite significant advancements in cloud technologies, practitioners, particularly those from non-technical backgrounds-continue to encounter substantial complexity stemming from fragmented toolchains, steep learning curves, and operational overhead of managing distributed networking and computing, ensuring consistent performance across hybrid environments, and navigating steep learning curves at the cloud-network boundary. Across sectors, participants consistently prioritized productivity, Quality of Service, and usability over conventional concerns such as cost or migration. These findings highlight the need for operationally simplified, SLA-aware, and developer-friendly platforms that streamline the full application lifecycle. This study contributes a practice-informed perspective to support the alignment of edge-cloud-native systems with the realities and needs of modern enterprises, offering critical insights for the advancement of seamless cloud-network convergence.

-> Edge-클라우드 개발 과제가 AI 카메라 디바이스에 직접 적용 가능

### Adaptive Enhancement and Dual-Pooling Sequential Attention for Lightweight Underwater Object Detection with YOLOv10 (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03807v1
- 점수: final 92.0

Underwater object detection constitutes a pivotal endeavor within the realms of marine surveillance and autonomous underwater systems; however, it presents significant challenges due to pronounced visual impairments arising from phenomena such as light absorption, scattering, and diminished contrast. In response to these formidable challenges, this manuscript introduces a streamlined yet robust framework for underwater object detection, grounded in the YOLOv10 architecture. The proposed method integrates a Multi-Stage Adaptive Enhancement module to improve image quality, a Dual-Pooling Sequential Attention (DPSA) mechanism embedded into the backbone to strengthen multi-scale feature representation, and a Focal Generalized IoU Objectness (FGIoU) loss to jointly improve localization accuracy and objectness prediction under class imbalance. Comprehensive experimental evaluations conducted on the RUOD and DUO benchmark datasets substantiate that the proposed DPSA_FGIoU_YOLOv10n attains exceptional performance, achieving mean Average Precision (mAP) scores of 88.9% and 88.0% at IoU threshold 0.5, respectively. In comparison to the baseline YOLOv10n, this represents enhancements of 6.7% for RUOD and 6.2% for DUO, all while preserving a compact model architecture comprising merely 2.8M parameters. These findings validate that the proposed framework establishes an efficacious equilibrium among accuracy, robustness, and real-time operational efficiency, making it suitable for deployment in resource-constrained underwater settings.

-> 가벼운 모델(2.8M 파라미터)과 이미지 보정 기술이 스포츠 캠용 엣지 디바이스에 직접 적용 가능

### Scaling Dense Event-Stream Pretraining from Visual Foundation Models (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03969v1
- 점수: final 89.6

Learning versatile, fine-grained representations from irregular event streams is pivotal yet nontrivial, primarily due to the heavy annotation that hinders scalability in dataset size, semantic richness, and application scope. To mitigate this dilemma, we launch a novel self-supervised pretraining method that distills visual foundation models (VFMs) to push the boundaries of event representation at scale. Specifically, we curate an extensive synchronized image-event collection to amplify cross-modal alignment. Nevertheless, due to inherent mismatches in sparsity and granularity between image-event domains, existing distillation paradigms are prone to semantic collapse in event representations, particularly at high resolutions. To bridge this gap, we propose to extend the alignment objective to semantic structures provided off-the-shelf by VFMs, indicating a broader receptive field and stronger supervision. The key ingredient of our method is a structure-aware distillation loss that grounds higher-quality image-event correspondences for alignment, optimizing dense event representations. Extensive experiments demonstrate that our approach takes a great leap in downstream benchmarks, significantly surpassing traditional methods and existing pretraining techniques. This breakthrough manifests in enhanced generalization, superior data efficiency and elevated transferability.

-> 시각 기반 모델과 이벤트 스트림 처리 기술이 스포츠 영상 분석에 직접적으로 적용 가능하여 하이라이트 장면 추출에 효과적

### DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03744v1
- 점수: final 89.6

Estimating accurate, view-consistent geometry and camera poses from uncalibrated multi-view/video inputs remains challenging - especially at high spatial resolutions and over long sequences. We present DAGE, a dual-stream transformer whose main novelty is to disentangle global coherence from fine detail. A low-resolution stream operates on aggressively downsampled frames with alternating frame/global attention to build a view-consistent representation and estimate cameras efficiently, while a high-resolution stream processes the original images per-frame to preserve sharp boundaries and small structures. A lightweight adapter fuses these streams via cross-attention, injecting global context without disturbing the pretrained single-frame pathway. This design scales resolution and clip length independently, supports inputs up to 2K, and maintains practical inference cost. DAGE delivers sharp depth/pointmaps, strong cross-view consistency, and accurate poses, establishing new state-of-the-art results for video geometry estimation and multi-view reconstruction.

-> 듀얼 스트림 아키텍처로 스포츠 장면의 기하학적 분석 가능

### Optimal Short Video Ordering and Transmission Scheduling for Reducing Video Delivery Cost in Peer-to-Peer CDNs (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03938v1
- 점수: final 88.0

The explosive growth of short video platforms has generated a massive surge in global traffic, imposing heavy financial burdens on content providers. While Peer-to-Peer Content Delivery Networks (PCDNs) offer a cost-effective alternative by leveraging resource-constrained edge nodes, the limited storage and concurrent service capacities of these peers struggle to absorb the intense temporal demand spikes characteristic of short video consumption. In this paper, we propose to minimize transmission costs by exploiting a novel degree of freedom, the inherent flexibility of server-driven playback sequences. We formulate the Optimal Video Ordering and Transmission Scheduling (OVOTS) problem as an Integer Linear Program to jointly optimize personalized video ordering and transmission scheduling. By strategically permuting playlists, our approach proactively smooths temporal traffic peaks, maximizing the offloading of requests to low-cost peer nodes. To solve the OVOTS problem, we provide a rigorous theoretical reduction of the OVOTS problem to an auxiliary Minimum Cost Maximum Flow (MCMF) formulation. Leveraging König's Edge Coloring Theorem, we prove the strict equivalence of these formulations and develop the Minimum-cost Maximum-flow with Edge Coloring (MMEC) algorithm, a globally optimal, polynomial-time solution. Extensive simulations demonstrate that MMEC significantly outperforms baseline strategies, achieving cost reductions of up to 67% compared to random scheduling and 36% compared to a simulated annealing approach. Our results establish playback sequence flexibility as a robust and highly effective paradigm for cost optimization in PCDN architectures.

-> 단편 비디오 전송 최적화 기술이 콘텐츠 공유 플랫폼에 적용 가능

### Self-supervised Domain Adaptation for Visual 3D Pose Estimation of Nano-drone Racing Gates by Enforcing Geometric Consistency (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.02936v1
- 점수: final 88.0

We consider the task of visually estimating the relative pose of a drone racing gate in front of a nano-quadrotor, using a convolutional neural network pre-trained on simulated data to regress the gate's pose. Due to the sim-to-real gap, the pre-trained model underperforms in the real world and must be adapted to the target domain. We propose an unsupervised domain adaptation (UDA) approach using only real image sequences collected by the drone flying an arbitrary trajectory in front of a gate; sequences are annotated in a self-supervised fashion with the drone's odometry as measured by its onboard sensors. On this dataset, a state consistency loss enforces that two images acquired at different times yield pose predictions that are consistent with the drone's odometry. Results indicate that our approach outperforms other SoA UDA approaches, has a low mean absolute error in position (x=26, y=28, z=10 cm) and orientation ($ψ$=13${^{\circ}}$), an improvement of 40% in position and 37% in orientation over a baseline. The approach's effectiveness is appreciable with as few as 10 minutes of real-world flight data and yields models with an inference time of 30.4ms (33 fps) when deployed aboard the Crazyflie 2.1 Brushless nano-drone.

-> 드론 레이싱 게이트 포즈 추정 기술로 스포츠 분석에 부분적으로 적용 가능하나 매우 구체적 사례

### Point Cloud Feature Coding for Object Detection over an Error-Prone Cloud-Edge Collaborative System (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03890v1
- 점수: final 86.4

Cloud-edge collaboration enhances machine perception by combining the strengths of edge and cloud computing. Edge devices capture raw data (e.g., 3D point clouds) and extract salient features, which are sent to the cloud for deeper analysis and data fusion. However, efficiently and reliably transmitting features between cloud and edge devices remains a challenging problem. We focus on point cloud-based object detection and propose a task-driven point cloud compression and reliable transmission framework based on source and channel coding. To meet the low-latency and low-power requirements of edge devices, we design a lightweight yet effective feature compaction module that compresses the deepest feature among multi-scale representations by removing task-irrelevant regions and applying channel-wise dimensionality reduction to task-relevant areas. Then, a signal-to-noise ratio (SNR)-adaptive channel encoder dynamically encodes the attribute information of the compacted features, while a Low-Density Parity-Check (LDPC) encoder ensures reliable transmission of geometric information. At the cloud side, an SNR-adaptive channel decoder guides the decoding of attribute information, and the LDPC decoder corrects geometry errors. Finally, a feature decompaction module restores the channel-wise dimensionality, and a diffusion-based feature upsampling module reconstructs shallow-layer features, enabling multi-scale feature reconstruction. On the KITTI dataset, our method achieved a 172-fold reduction in feature size with 3D average precision scores of 93.17%, 86.96%, and 77.25% for easy, moderate, and hard objects, respectively, over a 0 dB SNR wireless channel. Our source code will be released on GitHub at: https://github.com/yuanhui0325/T-PCFC.

-> 에지-클라우드 협업 아키텍처에 특징 압축 및 전송 기술이 적용 가능하여 실시간 스포츠 분석 시스템 구축에 유리

### Semantic Bridging Domains: Pseudo-Source as Test-Time Connector (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03844v1
- 점수: final 85.6

Distribution shifts between training and testing data are a critical bottleneck limiting the practical utility of models, especially in real-world test-time scenarios. To adapt models when the source domain is unknown and the target domain is unlabeled, previous works constructed pseudo-source domains via data generation and translation, then aligned the target domain with them. However, significant discrepancies exist between the pseudo-source and the original source domain, leading to potential divergence when correcting the target directly. From this perspective, we propose a Stepwise Semantic Alignment (SSA) method, viewing the pseudo-source as a semantic bridge connecting the source and target, rather than a direct substitute for the source. Specifically, we leverage easily accessible universal semantics to rectify the semantic features of the pseudo-source, and then align the target domain using the corrected pseudo-source semantics. Additionally, we introduce a Hierarchical Feature Aggregation (HFA) module and a Confidence-Aware Complementary Learning (CACL) strategy to enhance the semantic quality of the SSA process in the absence of source and ground truth of target domains. We evaluated our approach on tasks like semantic segmentation and image classification, achieving a 5.2% performance boost on GTA2Cityscapes over the state-of-the-art.

-> 도메인 적응 기술이 엣지 디바이스에서의 스포츠 장면 분석에 적용 가능하여 다양한 환경에서 안정적인 성능 보장

### InfinityStory: Unlimited Video Generation with World Consistency and Character-Aware Shot Transitions (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03646v1
- 점수: final 84.0

Generating long-form storytelling videos with consistent visual narratives remains a significant challenge in video synthesis. We present a novel framework, dataset, and a model that address three critical limitations: background consistency across shots, seamless multi-subject shot-to-shot transitions, and scalability to hour-long narratives. Our approach introduces a background-consistent generation pipeline that maintains visual coherence across scenes while preserving character identity and spatial relationships. We further propose a transition-aware video synthesis module that generates smooth shot transitions for complex scenarios involving multiple subjects entering or exiting frames, going beyond the single-subject limitations of prior work. To support this, we contribute with a synthetic dataset of 10,000 multi-subject transition sequences covering underrepresented dynamic scene compositions. On VBench, InfinityStory achieves the highest Background Consistency (88.94), highest Subject Consistency (82.11), and the best overall average rank (2.80), showing improved stability, smoother transitions, and better temporal coherence.

-> 하이라이트 영상 생성 및 편집 기술로 직접적으로 관련되어 경기 주요 장면 자동 편집에 필수적

### Separators in Enhancing Autoregressive Pretraining for Vision Mamba (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.03806v1
- 점수: final 84.0

The state space model Mamba has recently emerged as a promising paradigm in computer vision, attracting significant attention due to its efficient processing of long sequence tasks. Mamba's inherent causal mechanism renders it particularly suitable for autoregressive pretraining. However, current autoregressive pretraining methods are constrained to short sequence tasks, failing to fully exploit Mamba's prowess in handling extended sequences. To address this limitation, we introduce an innovative autoregressive pretraining method for Vision Mamba that substantially extends the input sequence length. We introduce new \textbf{S}epara\textbf{T}ors for \textbf{A}uto\textbf{R}egressive pretraining to demarcate and differentiate between different images, known as \textbf{STAR}. Specifically, we insert identical separators before each image to demarcate its inception. This strategy enables us to quadruple the input sequence length of Vision Mamba while preserving the original dimensions of the dataset images. Employing this long sequence pretraining technique, our STAR-B model achieved an impressive accuracy of 83.5\% on ImageNet-1k, which is highly competitive in Vision Mamba. These results underscore the potential of our method in enhancing the performance of vision models through improved leveraging of long-range dependencies.

-> Vision Mamba for autoregressive pretraining is directly applicable to sports video processing and analysis.

### HE-VPR: Height Estimation Enabled Aerial Visual Place Recognition Against Scale Variance (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04050v1
- 점수: final 82.4

In this work, we propose HE-VPR, a visual place recognition (VPR) framework that incorporates height estimation. Our system decouples height inference from place recognition, allowing both modules to share a frozen DINOv2 backbone. Two lightweight bypass adapter branches are integrated into our system. The first estimates the height partition of the query image via retrieval from a compact height database, and the second performs VPR within the corresponding height-specific sub-database. The adaptation design reduces training cost and significantly decreases the search space of the database. We also adopt a center-weighted masking strategy to further enhance the robustness against scale differences. Experiments on two self-collected challenging multi-altitude datasets demonstrate that HE-VPR achieves up to 6.1\% Recall@1 improvement over state-of-the-art ViT-based baselines and reduces memory usage by up to 90\%. These results indicate that HE-VPR offers a scalable and efficient solution for height-aware aerial VPR, enabling practical deployment in GNSS-denied environments. All the code and datasets for this work have been released on https://github.com/hmf21/HE-VPR.

-> 경공학 설계와 고도 추정 기술이 스포츠 장면 분석에 간접적으로 적용 가능

### Semi-Supervised Generative Learning via Latent Space Distribution Matching (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04223v1
- 점수: final 82.4

We introduce Latent Space Distribution Matching (LSDM), a novel framework for semi-supervised generative modeling of conditional distributions. LSDM operates in two stages: (i) learning a low-dimensional latent space from both paired and unpaired data, and (ii) performing joint distribution matching in this space via the 1-Wasserstein distance, using only paired data. This two-step approach minimizes an upper bound on the 1-Wasserstein distance between joint distributions, reducing reliance on scarce paired samples while enabling fast one-step generation. Theoretically, we establish non-asymptotic error bounds and demonstrate a key benefit of unpaired data: enhanced geometric fidelity in generated outputs. Furthermore, by extending the scope of its two core steps, LSDM provides a coherent statistical perspective that connects to a broad class of latent-space approaches. Notably, Latent Diffusion Models (LDMs) can be viewed as a variant of LSDM, in which joint distribution matching is achieved indirectly via score matching. Consequently, our results also provide theoretical insights into the consistency of LDMs. Empirical evaluations on real-world image tasks, including class-conditional generation and image super-resolution, demonstrate the effectiveness of LSDM in leveraging unpaired data to enhance generation quality.

-> 생성적 학습 기술이 스포츠 콘텐츠의 영상/이미지 보정에 적용 가능

### A Baseline Study and Benchmark for Few-Shot Open-Set Action Recognition with Feature Residual Discrimination (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04125v1
- 점수: final 82.4

Few-Shot Action Recognition (FS-AR) has shown promising results but is often limited by a closed-set assumption that fails in real-world open-set scenarios. While Few-Shot Open-Set (FSOS) recognition is well-established for images, its extension to spatio-temporal video data remains underexplored. To address this, we propose an architectural extension based on a Feature-Residual Discriminator (FR-Disc), adapting previous work on skeletal data to the more complex video domain. Extensive experiments on five datasets demonstrate that while common open-set techniques provide only marginal gains, our FR-Disc significantly enhances unknown rejection capabilities without compromising closed-set accuracy, setting a new state-of-the-art for FSOS-AR. The project website, code, and benchmark are available at: https://hsp-iit.github.io/fsosar/.

-> 오픈셋 액션 인식 기술이 스포츠 경기 전략 분석에 적용 가능하며 코드 공개됨

### Biomechanically Accurate Gait Analysis: A 3d Human Reconstruction Framework for Markerless Estimation of Gait Parameters (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.02499v1
- 점수: final 82.4

This paper presents a biomechanically interpretable framework for gait analysis using 3D human reconstruction from video data. Unlike conventional keypoint based approaches, the proposed method extracts biomechanically meaningful markers analogous to motion capture systems and integrates them within OpenSim for joint kinematic estimation. To evaluate performance, both spatiotemporal and kinematic gait parameters were analysed against reference marker-based data. Results indicate strong agreement with marker-based measurements, with considerable improvements when compared with pose-estimation methods alone. The proposed framework offers a scalable, markerless, and interpretable approach for accurate gait assessment, supporting broader clinical and real world deployment of vision based biomechanics

-> 이 논문은 마커리스 3D 인간 재구성 프레임워크를 제공하여 스포츠 자세 분석에 직접 적용 가능하며, 특수 장비 없이 운동 동작을 분석할 수 있다는 점에서 프로젝트 목표와 정확히 일치한다.

### DLIOS: An LLM-Augmented Real-Time Multi-Modal Interactive Enhancement Overlay System for Douyin Live Streaming (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03060v1
- 점수: final 81.6

We present DLIOS, a Large Language Model (LLM)-augmented real-time multi-modal interactive enhancement overlay system for Douyin (TikTok) live streaming. DLIOS employs a three-layer transparent window architecture for independent rendering of danmaku (scrolling text), gift and like particle effects, and VIP entrance animations, built around an event-driven WebView2 capture pipeline and a thread-safe event bus. On top of this foundation we contribute an LLM broadcast automation framework comprising: (1) a per-song four-segment prompt scheduling system (T1 opening/transition, T2 empathy, T3 era story/production notes, T4 closing) that generates emotionally coherent radio-style commentary from lyric metadata; (2) a JSON-serializable RadioPersonaConfig schema supporting hot-swap multi-persona broadcasting; (3) a real-time danmaku quick-reaction engine with keyword routing to static urgent speech or LLM-generated empathetic responses; and (4) the Suwan Li AI singer-songwriter persona case study -- over 100 AI-generated songs produced with Suno. A 36-hour stress test demonstrates: zero danmaku overlap, zero deadlock crashes, gift effect P95 latency <= 180 ms, LLM-to-TTS segment P95 latency <= 2.1 s, and TTS integrated loudness gain of 9.5 LUFS. live streaming; danmaku; large language model; prompt engineering; virtual persona; WebView2; WINMM; TTS; Suno; loudness normalization; real-time scheduling

-> 실시간 라이브 스트리밍 시스템이지만 스포츠 촬영 및 분석과 직접적인 연관성은 낮음

### SSR: A Generic Framework for Text-Aided Map Compression for Localization (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04272v1
- 점수: final 80.0

Mapping is crucial in robotics for localization and downstream decision-making. As robots are deployed in ever-broader settings, the maps they rely on continue to increase in size. However, storing these maps indefinitely (cold storage), transferring them across networks, or sending localization queries to cloud-hosted maps imposes prohibitive memory and bandwidth costs. We propose a text-enhanced compression framework that reduces both memory and bandwidth footprints while retaining high-fidelity localization. The key idea is to treat text as an alternative modality: one that can be losslessly compressed with large language models. We propose leveraging lightweight text descriptions combined with very small image feature vectors, which capture "complementary information" as a compact representation for the mapping task. Building on this, our novel technique, Similarity Space Replication (SSR), learns an adaptive image embedding in one shot that captures only the information "complementary" to the text descriptions. We validate our compression framework on multiple downstream localization tasks, including Visual Place Recognition as well as object-centric Monte Carlo localization in both indoor and outdoor settings. SSR achieves 2 times better compression than competing baselines on state-of-the-art datasets, including TokyoVal, Pittsburgh30k, Replica, and KITTI.

-> SSR 프레임워크는 스포츠 영상 압축 및 저장에 직접적으로 적용 가능하며, 텍스트와 시각 정보의 보완적 접근법이 하이라이트 장면 식별에 유용합니다.

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
