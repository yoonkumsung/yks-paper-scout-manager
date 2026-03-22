# Weekly Paper Report: 20260322

> ISO 2026-W12




## 1. Weekly Briefing


이번 주 평가 논문 수가 200건으로 사상 최대를 기록하며 4주 연속 상승세(107→117→181→200)를 이어갔습니다. 특히 CS.CV 분야 논문이 전주 대비 16건 증가한 125건으로 집중 성장했으며, 평균 평가 점수도 77.22점(전주 대비 +1.44점)으로 개선세를 보였습니다. 키워드 히트 수는 변동 없이 0건을 유지한 가운데, 핵심 분야의 질적 성장이 두드러진 주간으로 분석됩니다.



| Metric | Value | WoW |
|--------|-------|-----|
| Evaluated | 200 | - |
| Tier 1 | 120 | - |
| cs.CV | 125 | 16 |
| Keyword Hits | 0 | 0 |
| Avg Score | 77.22 | 1.44 |



### 4-Week Trend (uptrend)

| Week | Papers | cs.CV | Avg Score |
|------|--------|-------|-----------|

| W1 | 107 | 71 | 76.4 |

| W2 | 117 | 70 | 76.42 |

| W3 | 181 | 109 | 75.78 |

| W4 | 200 | 125 | 77.22 |





### Top Categories

| Category | Count | WoW % |
|----------|-------|-------|

| cs.CV | 125 | 14.7 |

| cs.AI | 47 | 17.5 |

| cs.LG | 41 | 70.8 |

| cs.RO | 14 | -36.4 |

| cs.CL | 8 | 0.0 |





- Graduated Reminds: 375
- Active Reminds: 28




### Tech Radar








#### TF-IDF Keywords

`video` (STABLE) `reasoning` (STABLE) `image` (STABLE) `visual` (STABLE) `generation` (STABLE) `multi` (STABLE) `data` (STABLE) `time` (STABLE) `learning` (STABLE) `object` (RISING) `vision` (RISING) `training` (STABLE) `text` (NEW) `temporal` (RISING) `motion` (STABLE) `real` (STABLE) `detection` (STABLE) `edge` (NEW) `spatial` (STABLE) `language` (STABLE) `classification` (DISAPPEARED) `task` (DISAPPEARED) `scale` (DISAPPEARED) `camera` (DISAPPEARED) `knowledge` (DISAPPEARED) `system` (DISAPPEARED) `reconstruction` (DISAPPEARED) `alignment` (DISAPPEARED) `images` (DISAPPEARED) `aware` (DISAPPEARED) `accuracy` (DISAPPEARED) `segmentation` (DISAPPEARED) `graph` (DISAPPEARED) `long` (DISAPPEARED) `tasks` (DISAPPEARED) `dataset` (DISAPPEARED) `semantic` (DISAPPEARED) `memory` (DISAPPEARED) `multimodal` (DISAPPEARED) `diffusion` (DISAPPEARED) `control` (DISAPPEARED) `human` (DISAPPEARED) `social` (DISAPPEARED) `quality` (DISAPPEARED) 







## 2. Top Papers


### 1. MONET: Modeling and Optimization of neural NEtwork Training from Edge to Data Centers

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.15002v1](http://arxiv.org/abs/2603.15002v1)

- While hardware-software co-design has significantly improved the efficiency of neural network inference, modeling the training phase remains a critical yet underexplored challenge. Training workloads impose distinct constraints, particularly regarding memory footprint and backpropagation complexity, which existing inference-focused tools fail to capture. This paper introduces MONET, a framework designed to model the training of neural networks on heterogeneous dataflow accelerators. MONET builds upon Stream, an experimentally verified framework that that models the inference of neural networks on heterogeneous dataflow accelerators with layer fusion. Using MONET, we explore the design space of ResNet-18 and a small GPT-2, demonstrating the framework's capability to model training workflows and find better hardware architectures. We then further examine problems that become more complex in neural network training due to the larger design space, such as determining the best layer-fusion configuration. Additionally, we use our framework to find interesting trade-offs in activation checkpointing, with the help of a genetic algorithm. Our findings highlight the importance of a holistic approach to hardware-software co-design for scalable and efficient deep learning deployment.
- 에지 디바이스를 위한 신경망 최적화 기술이 rk3588 기반 AI 촬영 장비의 핵심 기술로 직접 적용 가능


### 2. HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.18850v1](http://arxiv.org/abs/2603.18850v1)

- Video question answering (VQA) with vision-language models (VLMs) depends critically on which frames are selected from the input video, yet most systems rely on uniform or heuristic sampling that cannot be optimized for downstream answering quality. We introduce \textbf{HORNet}, a lightweight frame selection policy trained with Group Relative Policy Optimization (GRPO) to learn which frames a frozen VLM needs to answer questions correctly. With fewer than 1M trainable parameters, HORNet reduces input frames by up to 99\% and VLM processing time by up to 93\%, while improving answer quality on short-form benchmarks (+1.7\% F1 on MSVD-QA) and achieving strong performance on temporal reasoning tasks (+7.3 points over uniform sampling on NExT-QA). We formalize this as Select Any Frames (SAF), a task that decouples visual input curation from VLM reasoning, and show that GRPO-trained selection generalizes better out-of-distribution than supervised and PPO alternatives. HORNet's policy further transfers across VLM answerers without retraining, yielding an additional 8.5\% relative gain when paired with a stronger model. Evaluated across six benchmarks spanning 341,877 QA pairs and 114.2 hours of video, our results demonstrate that optimizing \emph{what} a VLM sees is a practical and complementary alternative to optimizing what it generates while improving efficiency. Code is available at https://github.com/ostadabbas/HORNet.
- 경량 프레임 선택 기술은 스포츠 영상 하이라이트 생성에 직접적으로 적용 가능합니다


### 3. SpiderCam: Low-Power Snapshot Depth from Differential Defocus

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.17910v1](http://arxiv.org/abs/2603.17910v1)

- We introduce SpiderCam, an FPGA-based snapshot depth-from-defocus camera which produces 480x400 sparse depth maps in real-time at 32.5 FPS over a working range of 52 cm while consuming 624 mW of power in total. SpiderCam comprises a custom camera that simultaneously captures two differently focused images of the same scene, processed with a SystemVerilog implementation of depth from differential defocus (DfDD) on a low-power FPGA. To achieve state-of-the-art power consumption, we present algorithmic improvements to DfDD that overcome challenges caused by low-power sensors, and design a memory-local implementation for streaming depth computation on a device that is too small to store even a single image pair. We report the first sub-Watt total power measurement for passive FPGA-based 3D cameras in the literature.
- 스포츠 촬영을 위한 저전력 엣지 디바이스에 직접 적용 가능한 깊이 맵 생성 기술


### 4. DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge

- **Score**: 100.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.19172v1](http://arxiv.org/abs/2603.19172v1)

- Despite the computational efficiency of MoE models, the excessive memory footprint and I/O overhead inherent in multi-expert architectures pose formidable challenges for real-time inference on resource-constrained edge platforms. While existing static methods struggle with a rigid latency-accuracy trade-off, we observe that expert importance is highly skewed and depth-dependent. Motivated by these insights, we propose DyMoE, a dynamic mixed-precision quantization framework designed for high-performance edge inference. Leveraging insights into expert importance skewness and depth-dependent sensitivity, DyMoE introduces: (1) importance-aware prioritization to dynamically quantize experts at runtime; (2) depth-adaptive scheduling to preserve semantic integrity in critical layers; and (3) look-ahead prefetching to overlap I/O stalls. Experimental results on commercial edge hardware show that DyMoE reduces Time-to-First-Token (TTFT) by 3.44x-22.7x and up to a 14.58x speedup in Time-Per-Output-Token (TPOT) compared to state-of-the-art offloading baselines, enabling real-time, accuracy-preserving MoE inference on resource-constrained edge devices.
- 엣지 디바이스 효율적인 추론 기술이 프로젝트의 엣지 디바이스 구현에 직접적으로 관련됨


### 5. Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token

- **Score**: 98.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.19026v1](http://arxiv.org/abs/2603.19026v1)

- Recent segmentation methods leveraging Multi-modal Large Language Models (MLLMs) have shown reliable object-level segmentation and enhanced spatial perception. However, almost all previous methods predominantly rely on specialist mask decoders to interpret masks from generated segmentation-related embeddings and visual features, or incorporate multiple additional tokens to assist. This paper aims to investigate whether and how we can unlock segmentation from MLLM itSELF with 1 segmentation Embedding (SELF1E) while achieving competitive results, which eliminates the need for external decoders. To this end, our approach targets the fundamental limitation of resolution reduction in pixel-shuffled image features from MLLMs. First, we retain image features at their original uncompressed resolution, and refill them with residual features extracted from MLLM-processed compressed features, thereby improving feature precision. Subsequently, we integrate pixel-unshuffle operations on image features with and without LLM processing, respectively, to unleash the details of compressed features and amplify the residual features under uncompressed resolution, which further enhances the resolution of refilled features. Moreover, we redesign the attention mask with dual perception pathways, i.e., image-to-image and image-to-segmentation, enabling rich feature interaction between pixels and the segmentation token. Comprehensive experiments across multiple segmentation tasks validate that SELF1E achieves performance competitive with specialist mask decoder-based methods, demonstrating the feasibility of decoder-free segmentation in MLLMs. Project page: https://github.com/ANDYZAQ/SELF1E.
- 스포츠 영상에서 선수와 객체 식별에 직접적으로 적용 가능한 고급 분할 기술


### 6. EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation

- **Score**: 98.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.18739v1](http://arxiv.org/abs/2603.18739v1)

- Deploying high-performance dense prediction models on resource-constrained edge devices remains challenging due to strict limits on computation and memory. In practice, lightweight systems for object detection, instance segmentation, and pose estimation are still dominated by CNN-based architectures such as YOLO, while compact Vision Transformers (ViTs) often struggle to achieve similarly strong accuracy efficiency tradeoff, even with large scale pretraining. We argue that this gap is largely due to insufficient task specific representation learning in small scale ViTs, rather than an inherent mismatch between ViTs and edge dense prediction. To address this issue, we introduce EdgeCrafter, a unified compact ViT framework for edge dense prediction centered on ECDet, a detection model built from a distilled compact backbone and an edge-friendly encoder decoder design. On the COCO dataset, ECDet-S achieves 51.7 AP with fewer than 10M parameters using only COCO annotations. For instance segmentation, ECInsSeg achieves performance comparable to RF-DETR while using substantially fewer parameters. For pose estimation, ECPose-X reaches 74.8 AP, significantly outperforming YOLO26Pose-X (71.6 AP) despite the latter's reliance on extensive Objects365 pretraining. These results show that compact ViTs, when paired with task-specialized distillation and edge-aware design, can be a practical and competitive option for edge dense prediction. Code is available at: https://intellindust-ai-lab.github.io/projects/EdgeCrafter/
- 엣지 기기에서의 포즈 추정 및 객체 탐지 기술이 스포츠 동작 분석에 직접 적용 가능함


### 7. Multi-Objective Load Balancing for Heterogeneous Edge-Based Object Detection Systems

- **Score**: 98.4 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.15400v1](http://arxiv.org/abs/2603.15400v1)

- The rapid proliferation of the Internet of Things (IoT) and smart applications has led to a surge in data generated by distributed sensing devices. Edge computing is a mainstream approach to managing this data by pushing computation closer to the data source, typically onto resource-constrained devices such as single-board computers (SBCs). In such environments, the unavoidable heterogeneity of hardware and software makes effective load balancing particularly challenging. In this paper, we propose a multi-objective load balancing method tailored to heterogeneous, edge-based object detection systems. We study a setting in which multiple device-model pairs expose distinct accuracy, latency, and energy profiles, while both request intensity and scene complexity fluctuate over time. To handle this dynamically varying environment, our approach uses a two-stage decision mechanism: it first performs accuracy-aware filtering to identify suitable device-model candidates that provide accuracy within the acceptable range, and then applies a weighted-sum scoring function over expected latency and energy consumption to select the final execution target. We evaluate the proposed load balancer through extensive experiments on real-world datasets, comparing against widely used baseline strategies. The results indicate that the proposed multi-objective load balancing method halves energy consumption and achieves an 80% reduction in end-to-end latency, while incurring only a modest, up to 10%, decrease in detection accuracy relative to an accuracy-centric baseline.
- 이기종 엣지 기반 객체 탐지 시스템의 다목적 부하 분산이 스포츠 촬영 시스템 자원 제약에 직접 적용 가능


### 8. Balancing Performance and Fairness in Explainable AI for Anomaly Detection in Distributed Power Plants Monitoring

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.18954v1](http://arxiv.org/abs/2603.18954v1)

- Reliable anomaly detection in distributed power plant monitoring systems is essential for ensuring operational continuity and reducing maintenance costs, particularly in regions where telecom operators heavily rely on diesel generators. However, this task is challenged by extreme class imbalance, lack of interpretability, and potential fairness issues across regional clusters. In this work, we propose a supervised ML framework that integrates ensemble methods (LightGBM, XGBoost, Random Forest, CatBoost, GBDT, AdaBoost) and baseline models (Support Vector Machine, K-Nearrest Neighbors, Multilayer Perceptrons, and Logistic Regression) with advanced resampling techniques (SMOTE with Tomek Links and ENN) to address imbalance in a dataset of diesel generator operations in Cameroon. Interpretability is achieved through SHAP (SHapley Additive exPlanations), while fairness is quantified using the Disparate Impact Ratio (DIR) across operational clusters. We further evaluate model generalization using Maximum Mean Discrepancy (MMD) to capture domain shifts between regions. Experimental results show that ensemble models consistently outperform baselines, with LightGBM achieving an F1-score of 0.99 and minimal bias across clusters (DIR $\approx 0.95$). SHAP analysis highlights fuel consumption rate and runtime per day as dominant predictors, providing actionable insights for operators. Our findings demonstrate that it is possible to balance performance, interpretability, and fairness in anomaly detection, paving the way for more equitable and explainable AI systems in industrial power management. {\color{black} Finally, beyond offline evaluation, we also discuss how the trained models can be deployed in practice for real-time monitoring. We show how containerized services can process in real-time, deliver low-latency predictions, and provide interpretable outputs for operators.
- 실시간 이상 감지 프레임워크로 엣지 배포 가능성이 있어 스포츠 모니터링 시스템에 적용 가능


### 9. A spatio-temporal graph-based model for team sports analysis

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.17471v1](http://arxiv.org/abs/2603.17471v1)

- Team sports represent complex phenomena characterized by both spatial and temporal dimensions, making their analysis inherently challenging. In this study, we examine team sports as complex systems, specifically focusing on the tactical aspects influenced by external constraints. To this end, we introduce a new generic graph-based model to analyze these phenomena. Specifically, we model a team sport's attacking play as a directed path containing absolute and relative ball carrier-centered spatial information, temporal information, and semantic information. We apply our model to union rugby, aiming to validate two hypotheses regarding the impact of the pedagogy provided by the coach on the one hand, and the effect of the initial positioning of the defensive team on the other hand. Preliminary results from data collected on six-player rugby from several French clubs indicate notable effects of these constraints. The model is intended to be applied to other team sports and to validate additional hypotheses related to team coordination patterns, including upcoming applications in basketball.
- 팀 스포츠 전략 분석에 직접적으로 적용 가능한 그래프 기반 모델


### 10. Face-to-Face: A Video Dataset for Multi-Person Interaction Modeling

- **Score**: 96.0 | **Topic**: cappic-ai
- **URL**: [http://arxiv.org/abs/2603.14794v1](http://arxiv.org/abs/2603.14794v1)

- Modeling the reactive tempo of human conversation remains difficult because most audio-visual datasets portray isolated speakers delivering short monologues. We introduce \textbf{Face-to-Face with Jimmy Fallon (F2F-JF)}, a 70-hour, 14k-clip dataset of two-person talk-show exchanges that preserves the sequential dependency between a guest turn and the host's response. A semi-automatic pipeline combines multi-person tracking, speech diarization, and lightweight human verification to extract temporally aligned host/guest tracks with tight crops and metadata that are ready for downstream modeling. We showcase the dataset with a reactive, speech-driven digital avatar task in which the host video during $[t_1,t_2]$ is generated from their audio plus the guest's preceding video during $[t_0,t_1]$. Conditioning a MultiTalk-style diffusion model on this cross-person visual context yields small but consistent Emotion-FID and FVD gains while preserving lip-sync quality relative to an audio-only baseline. The dataset, preprocessing recipe, and baseline together provide an end-to-end blueprint for studying dyadic, sequential behavior, which we expand upon throughout the paper. Dataset and code will be made publicly available.
- 다중 인물 추적 기술이 스포츠 촬영에서 여러 선수 동시 추적에 적용 가능






### Graduated Reminds


- **From Imitation to Intuition: Intrinsic Reasoning for Open-Instance Video Classification** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10300v1](http://arxiv.org/abs/2603.10300v1)

- **Variance-Aware Adaptive Weighting for Diffusion Model Training** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10391v1](http://arxiv.org/abs/2603.10391v1)

- **Multi-Person Pose Estimation Evaluation Using Optimal Transportation and Improved Pose Matching** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10398v1](http://arxiv.org/abs/2603.10398v1)

- **Frames2Residual: Spatiotemporal Decoupling for Self-Supervised Video Denoising** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10417v1](http://arxiv.org/abs/2603.10417v1)

- **AsyncMDE: Real-Time Monocular Depth Estimation via Asynchronous Spatial Memory** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10438v1](http://arxiv.org/abs/2603.10438v1)

- **LCAMV: High-Accuracy 3D Reconstruction of Color-Varying Objects Using LCA Correction and Minimum-Variance Fusion in Structured Light** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10456v1](http://arxiv.org/abs/2603.10456v1)

- **Muscle Synergy Priors Enhance Biomechanical Fidelity in Predictive Musculoskeletal Locomotion Simulation** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10474v1](http://arxiv.org/abs/2603.10474v1)

- **UHD Image Deblurring via Autoregressive Flow with Ill-conditioned Constraints** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10517v1](http://arxiv.org/abs/2603.10517v1)

- **An Event-Driven E-Skin System with Dynamic Binary Scanning and real time SNN Classification** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10537v1](http://arxiv.org/abs/2603.10537v1)

- **DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10538v1](http://arxiv.org/abs/2603.10538v1)

- **P-GSVC: Layered Progressive 2D Gaussian Splatting for Scalable Image and Video** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10551v1](http://arxiv.org/abs/2603.10551v1)

- **Safety-critical Control Under Partial Observability: Reach-Avoid POMDP meets Belief Space Control** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10572v1](http://arxiv.org/abs/2603.10572v1)

- **HyPER-GAN: Hybrid Patch-Based Image-to-Image Translation for Real-Time Photorealism Enhancement** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10604v1](http://arxiv.org/abs/2603.10604v1)

- **Are Video Reasoning Models Ready to Go Outside?** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10652v1](http://arxiv.org/abs/2603.10652v1)

- **A$^2$-Edit: Precise Reference-Guided Image Editing of Arbitrary Objects and Ambiguous Masks** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10685v1](http://arxiv.org/abs/2603.10685v1)

- **Parallel-in-Time Nonlinear Optimal Control via GPU-native Sequential Convex Programming** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10711v1](http://arxiv.org/abs/2603.10711v1)

- **Event-based Photometric Stereo via Rotating Illumination and Per-Pixel Learning** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10748v1](http://arxiv.org/abs/2603.10748v1)

- **Novel Architecture of RPA In Oral Cancer Lesion Detection** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10928v1](http://arxiv.org/abs/2603.10928v1)

- **GroundCount: Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10978v1](http://arxiv.org/abs/2603.10978v1)

- **PPGuide: Steering Diffusion Policies with Performance Predictive Guidance** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10980v1](http://arxiv.org/abs/2603.10980v1)

- **Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity** (score: None, graduated: 2026-03-16)
  [http://arxiv.org/abs/2603.10990v1](http://arxiv.org/abs/2603.10990v1)

- **Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11439v1](http://arxiv.org/abs/2603.11439v1)

- **Detect Anything in Real Time: From Single-Prompt Segmentation to Multi-Class Detection** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11441v1](http://arxiv.org/abs/2603.11441v1)

- **Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11460v1](http://arxiv.org/abs/2603.11460v1)

- **Bridging Discrete Marks and Continuous Dynamics: Dual-Path Cross-Interaction for Marked Temporal Point Processes** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11462v1](http://arxiv.org/abs/2603.11462v1)

- **INFACT: A Diagnostic Benchmark for Induced Faithfulness and Factuality Hallucinations in Video-LLMs** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11481v1](http://arxiv.org/abs/2603.11481v1)

- **CFD-HAR: User-controllable Privacy through Conditional Feature Disentanglement** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11526v1](http://arxiv.org/abs/2603.11526v1)

- **Mobile-GS: Real-time Gaussian Splatting for Mobile Devices** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11531v1](http://arxiv.org/abs/2603.11531v1)

- **Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11556v1](http://arxiv.org/abs/2603.11556v1)

- **LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11605v1](http://arxiv.org/abs/2603.11605v1)

- **MV-SAM3D: Adaptive Multi-View Fusion for Layout-Aware 3D Generation** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11633v1](http://arxiv.org/abs/2603.11633v1)

- **SoulX-LiveAct: Towards Hour-Scale Real-Time Human Animation with Neighbor Forcing and ConvKV Memory** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11746v1](http://arxiv.org/abs/2603.11746v1)

- **Derain-Agent: A Plug-and-Play Agent Framework for Rainy Image Restoration** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11866v1](http://arxiv.org/abs/2603.11866v1)

- **InSpatio-WorldFM: An Open-Source Real-Time Generative Frame Model** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11911v1](http://arxiv.org/abs/2603.11911v1)

- **PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11917v1](http://arxiv.org/abs/2603.11917v1)

- **SNAP-V: A RISC-V SoC with Configurable Neuromorphic Acceleration for Small-Scale Spiking Neural Networks** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11939v1](http://arxiv.org/abs/2603.11939v1)

- **Multimodal Emotion Recognition via Bi-directional Cross-Attention and Temporal Modeling** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.11971v1](http://arxiv.org/abs/2603.11971v1)

- **Resource-Efficient Iterative LLM-Based NAS with Feedback Memory** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.12091v1](http://arxiv.org/abs/2603.12091v1)

- **BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.12176v1](http://arxiv.org/abs/2603.12176v1)

- **SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.12193v1](http://arxiv.org/abs/2603.12193v1)

- **HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.12222v1](http://arxiv.org/abs/2603.12222v1)

- **BiGain: Unified Token Compression for Joint Generation and Classification** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.12240v1](http://arxiv.org/abs/2603.12240v1)

- **Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.12247v1](http://arxiv.org/abs/2603.12247v1)

- **Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.12255v1](http://arxiv.org/abs/2603.12255v1)

- **$Ψ_0$: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.12263v1](http://arxiv.org/abs/2603.12263v1)

- **OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.12265v1](http://arxiv.org/abs/2603.12265v1)

- **EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation** (score: None, graduated: 2026-03-17)
  [http://arxiv.org/abs/2603.12267v1](http://arxiv.org/abs/2603.12267v1)

- **Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.14726v1](http://arxiv.org/abs/2603.14726v1)

- **Efficient Event Camera Volume System** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.14738v1](http://arxiv.org/abs/2603.14738v1)

- **Face-to-Face: A Video Dataset for Multi-Person Interaction Modeling** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.14794v1](http://arxiv.org/abs/2603.14794v1)

- **M2IR: Proactive All-in-One Image Restoration via Mamba-style Modulation and Mixture-of-Experts** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.14816v1](http://arxiv.org/abs/2603.14816v1)

- **SimCert: Probabilistic Certification for Behavioral Similarity in Deep Neural Network Compression** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.14818v1](http://arxiv.org/abs/2603.14818v1)

- **Video Detector: A Dual-Phase Vision-Based System for Real-Time Traffic Intersection Control and Intelligent Transportation Analysis** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.14861v1](http://arxiv.org/abs/2603.14861v1)

- **CyCLeGen: Cycle-Consistent Layout Prediction and Image Generation in Vision Foundation Models** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.14957v1](http://arxiv.org/abs/2603.14957v1)

- **Lightweight User-Personalization Method for Closed Split Computing** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.14958v1](http://arxiv.org/abs/2603.14958v1)

- **GeoNVS: Geometry Grounded Video Diffusion for Novel View Synthesis** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.14965v1](http://arxiv.org/abs/2603.14965v1)

- **Exposing Cross-Modal Consistency for Fake News Detection in Short-Form Videos** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.14992v1](http://arxiv.org/abs/2603.14992v1)

- **MONET: Modeling and Optimization of neural NEtwork Training from Edge to Data Centers** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15002v1](http://arxiv.org/abs/2603.15002v1)

- **Edit2Interp: Adapting Image Foundation Models from Spatial Editing to Video Frame Interpolation with Few-Shot Learning** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15003v1](http://arxiv.org/abs/2603.15003v1)

- **Riemannian Motion Generation: A Unified Framework for Human Motion Representation and Generation via Riemannian Flow Matching** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15016v1](http://arxiv.org/abs/2603.15016v1)

- **Spatio-temporal probabilistic forecast using MMAF-guided learning** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15055v1](http://arxiv.org/abs/2603.15055v1)

- **Affordable Precision Agriculture: A Deployment-Oriented Review of Low-Cost, Low-Power Edge AI and TinyML for Resource-Constrained Farming Systems** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15085v1](http://arxiv.org/abs/2603.15085v1)

- **PAKAN: Pixel Adaptive Kolmogorov-Arnold Network Modules for Pansharpening** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15109v1](http://arxiv.org/abs/2603.15109v1)

- **A Novel Camera-to-Robot Calibration Method for Vision-Based Floor Measurements** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15126v1](http://arxiv.org/abs/2603.15126v1)

- **Low-light Image Enhancement with Retinex Decomposition in Latent Space** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15131v1](http://arxiv.org/abs/2603.15131v1)

- **Tracking the Discriminative Axis: Dual Prototypes for Test-Time OOD Detection Under Covariate Shift** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15213v1](http://arxiv.org/abs/2603.15213v1)

- **Multi-turn Physics-informed Vision-language Model for Physics-grounded Anomaly Detection** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15237v1](http://arxiv.org/abs/2603.15237v1)

- **GATE-AD: Graph Attention Network Encoding For Few-Shot Industrial Visual Anomaly Detection** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15300v1](http://arxiv.org/abs/2603.15300v1)

- **Generative Video Compression with One-Dimensional Latent Representation** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15302v1](http://arxiv.org/abs/2603.15302v1)

- **IRIS: Intersection-aware Ray-based Implicit Editable Scenes** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15368v1](http://arxiv.org/abs/2603.15368v1)

- **Multi-Objective Load Balancing for Heterogeneous Edge-Based Object Detection Systems** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15400v1](http://arxiv.org/abs/2603.15400v1)

- **Nova: Scalable Streaming Join Placement and Parallelization in Resource-Constrained Geo-Distributed Environments** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15453v1](http://arxiv.org/abs/2603.15453v1)

- **Anchor then Polish for Low-light Enhancement** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15472v1](http://arxiv.org/abs/2603.15472v1)

- **Federated Learning of Binary Neural Networks: Enabling Low-Cost Inference** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15507v1](http://arxiv.org/abs/2603.15507v1)

- **Learning Latent Proxies for Controllable Single-Image Relighting** (score: None, graduated: 2026-03-18)
  [http://arxiv.org/abs/2603.15555v1](http://arxiv.org/abs/2603.15555v1)

- **Collaborative Temporal Feature Generation via Critic-Free Reinforcement Learning for Cross-User Sensor-Based Activity Recognition** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16043v1](http://arxiv.org/abs/2603.16043v1)

- **Large Reward Models: Generalizable Online Robot Reward Generation with Vision-Language Models** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16065v1](http://arxiv.org/abs/2603.16065v1)

- **LICA: Layered Image Composition Annotations for Graphic Design Research** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16098v1](http://arxiv.org/abs/2603.16098v1)

- **Knowledge Distillation for Collaborative Learning in Distributed Communications and Sensing** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16116v1](http://arxiv.org/abs/2603.16116v1)

- **SE(3)-LIO: Smooth IMU Propagation With Jointly Distributed Poses on SE(3) Manifold for Accurate and Robust LiDAR-Inertial Odometry** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16118v1](http://arxiv.org/abs/2603.16118v1)

- **BLADE: Adaptive Wi-Fi Contention Control for Next-Generation Real-Time Communication** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16119v1](http://arxiv.org/abs/2603.16119v1)

- **EPOFusion: Exposure aware Progressive Optimization Method for Infrared and Visible Image Fusion** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16130v1](http://arxiv.org/abs/2603.16130v1)

- **Change is Hard: Consistent Player Behavior Across Games with Conflicting Incentives** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16136v1](http://arxiv.org/abs/2603.16136v1)

- **GATS: Gaussian Aware Temporal Scaling Transformer for Invariant 4D Spatio-Temporal Point Cloud Representation** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16154v1](http://arxiv.org/abs/2603.16154v1)

- **PA-LVIO: Real-Time LiDAR-Visual-Inertial Odometry and Mapping with Pose-Only Bundle Adjustment** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16228v1](http://arxiv.org/abs/2603.16228v1)

- **When Thinking Hurts: Mitigating Visual Forgetting in Video Reasoning via Frame Repetition** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16256v1](http://arxiv.org/abs/2603.16256v1)

- **SpikeCLR: Contrastive Self-Supervised Learning for Few-Shot Event-Based Vision using Spiking Neural Networks** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16338v1](http://arxiv.org/abs/2603.16338v1)

- **Learning Human-Object Interaction for 3D Human Pose Estimation from LiDAR Point Clouds** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16343v1](http://arxiv.org/abs/2603.16343v1)

- **Advancing Visual Reliability: Color-Accurate Underwater Image Enhancement for Real-Time Underwater Missions** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16363v1](http://arxiv.org/abs/2603.16363v1)

- **TinyGLASS: Real-Time Self-Supervised In-Sensor Anomaly Detection** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16451v1](http://arxiv.org/abs/2603.16451v1)

- **Optimal uncertainty bounds for multivariate kernel regression under bounded noise: A Gaussian process-based dual function** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16481v1](http://arxiv.org/abs/2603.16481v1)

- **DST-Net: A Dual-Stream Transformer with Illumination-Independent Feature Guidance and Multi-Scale Spatial Convolution for Low-Light Image Enhancement** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16482v1](http://arxiv.org/abs/2603.16482v1)

- **Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16538v1](http://arxiv.org/abs/2603.16538v1)

- **Learning Whole-Body Control for a Salamander Robot** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16683v1](http://arxiv.org/abs/2603.16683v1)

- **Emotion-Aware Classroom Quality Assessment Leveraging IoT-Based Real-Time Student Monitoring** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16719v1](http://arxiv.org/abs/2603.16719v1)

- **Deep Reinforcement Learning-driven Edge Offloading for Latency-constrained XR pipelines** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16823v1](http://arxiv.org/abs/2603.16823v1)

- **M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16844v1](http://arxiv.org/abs/2603.16844v1)

- **SparkVSR: Interactive Video Super-Resolution via Sparse Keyframe Propagation** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16864v1](http://arxiv.org/abs/2603.16864v1)

- **Efficient Reasoning on the Edge** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16867v1](http://arxiv.org/abs/2603.16867v1)

- **MessyKitchens: Contact-rich object-level 3D scene reconstruction** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16868v1](http://arxiv.org/abs/2603.16868v1)

- **Demystifing Video Reasoning** (score: None, graduated: 2026-03-19)
  [http://arxiv.org/abs/2603.16870v1](http://arxiv.org/abs/2603.16870v1)







### Notable Authors

| Author | Papers | Avg Score |
|--------|--------|-----------|

| Hao Li | 4 | 84.6 |

| Zixuan Wang | 4 | 81.2 |

| Nicu Sebe | 4 | 74.8 |

| Wei Xu | 3 | 85.1 |

| Jae-Sang Hyun | 3 | 84.7 |

| Yang Li | 3 | 82.9 |

| Lei Yang | 3 | 80.7 |

| Lin Liu | 3 | 78.9 |

| Hongsheng Li | 3 | 78.7 |

| Yi Zhang | 3 | 76.0 |







## 3. Trends



### cappic-ai

| Date | Avg Score |
|------|-----------|

| 2026-02-23 | 75.83 |

| 2026-02-24 | 85.2 |

| 2026-02-25 | 68.0 |

| 2026-02-26 | 76.58 |

| 2026-03-02 | 76.66 |

| 2026-03-03 | 79.6 |

| 2026-03-04 | 75.8 |

| 2026-03-05 | 77.33 |

| 2026-03-10 | 75.92 |

| 2026-03-11 | 75.21 |

| 2026-03-12 | 76.22 |

| 2026-03-16 | 79.03 |

| 2026-03-17 | 77.66 |

| 2026-03-18 | 76.84 |

| 2026-03-19 | 75.3 |



