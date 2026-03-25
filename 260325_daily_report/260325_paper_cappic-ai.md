# CAPP!C_AI 논문 리포트 (2026-03-25)

> 수집 81 | 필터 78 | 폐기 10 | 평가 66 | 출력 51 | 기준 50점

검색 윈도우: 2026-03-24T00:00:00+00:00 ~ 2026-03-25T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 49

---

## 검색 키워드

autonomous cinematography, sports tracking, active camera, highlight detection, sports summarization, video summarization, video enhancement, image restoration, video stabilization, pose estimation, action recognition, movement analysis, tactical analysis, sports strategy, game analytics, edge computing, embedded vision, real-time processing, sports video, deep learning, real-time

---

## 1위: TRINE: A Token-Aware, Runtime-Adaptive FPGA Inference Engine for Multimodal AI

- arXiv: http://arxiv.org/abs/2603.22867v1
- PDF: https://arxiv.org/pdf/2603.22867v1
- 발행일: 2026-03-24
- 카테고리: cs.AR
- 점수: final 100.0 (llm_adjusted:100 = base:95 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Multimodal stacks that mix ViTs, CNNs, GNNs, and transformer NLP strain embedded platforms because their compute/memory patterns diverge and hard real-time targets leave little slack. TRINE is a single-bitstream FPGA accelerator and compiler that executes end-to-end multimodal inference without reconfiguration. Layers are unified as DDMM/SDDMM/SpMM and mapped to a mode-switchable engine that toggles at runtime among weight/output-stationary systolic, 1xCS SIMD, and a routable adder tree (RADT) on a shared PE array. A width-matched, two-stage top-k unit enables in-stream token pruning, while dependency-aware layer offloading (DALO) overlaps independent kernels across reconfigurable processing units to sustain utilization. Evaluated on Alveo U50 and ZCU104, TRINE reduces latency by up to 22.57x vs. RTX 4090 and 6.86x vs. Jetson Orin Nano at 20-21 W; token pruning alone yields up to 7.8x on ViT-heavy pipelines, and DALO contributes up to 79% throughput improvement. With int8 quantization, accuracy drops remain <2.5% across representative tasks, delivering state-of-the-art latency and energy efficiency for unified vision, language, and graph workloads-in one bitstream.

**선정 근거**
TRINE은 다중 모달 AI 가속을 위한 FPGA 엔진으로, rk3588 기반의 에지 디바이스에서 스포츠 촬영 및 분석을 위한 여러 AI 모델을 효율적으로 실행할 수 있게 해줍니다.

**활용 인사이트**
TRINE의 토큰 프루닝 기술을 활용하면 ViT 기반 스포츠 장면 분석 모델의 처리 속도를 최대 7.8배 향상시켜 실시간 하이라이트 감지를 가능하게 합니다.

## 2위: TorR: Towards Brain-Inspired Task-Oriented Reasoning via Cache-Oriented Algorithm-Architecture Co-design

- arXiv: http://arxiv.org/abs/2603.22855v1
- PDF: https://arxiv.org/pdf/2603.22855v1
- 발행일: 2026-03-24
- 카테고리: cs.AR, cs.LG
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Task-oriented object detection (TOOD) atop CLIP offers open-vocabulary, prompt-driven semantics, yet dense per-window computation and heavy memory traffic hinder real-time, power-limited edge deployment. We present \emph{TorR}, a brain-inspired \textbf{algorithm--architecture co-design} that \textbf{replaces CLIP-style dense alignment with a hyperdimensional (HDC) associative reasoner} and turns temporal coherence into reuse. On the \emph{algorithm} side, TorR reformulates alignment as HDC similarity and graph composition, introducing \emph{partial-similarity reuse} via (i) query caching with per-class score accumulation, (ii) exact $δ$-updates when only a small set of hypervector bits change, and (iii) similarity/load-gated bypass under high system load. On the \emph{architecture} side, TorR instantiates a lane-scalable, bit-sliced item memory with bank/precision gating and a lightweight controller that schedules bypass/$δ$/full paths to meet RT-30/RT-60 targets as object counts vary. Synthesized in a TSMC 28\,nm process and exercised with a cycle-accurate simulator, TorR sustains real-time throughput with millijoule-scale energy per window ($\approx$50\,mJ at 60\,FPS; $\approx$113\,mJ at 30\,FPS) and low latency jitter, while delivering competitive AP@0.5 across five task prompts (mean 44.27\%) within a bounded margin to strong VLM baselines, but at orders-of-magnitude lower energy. The design exposes deployment-time configurability (effective dimension $D'$, thresholds, precision) to trade accuracy, latency, and energy for edge budgets.

**선정 근거**
TorR는 에지 디바이스에서 실시간 객체 탐지를 위한 뇀 영감 기술로, 스포츠 장면에서 선수 및 공 등의 실시간 추적과 분석에 적합합니다.

**활용 인사이트**
TorR의 하이퍼차원 연관 추론 기술을 스포츠 분석에 적용하면, 경기 전략 분석 및 개인별 동작 분석의 정확도를 크게 향상시킬 수 있습니다.

## 3위: Short-Form Video Viewing Behavior Analysis and Multi-Step Viewing Time Prediction

- arXiv: http://arxiv.org/abs/2603.22663v1
- PDF: https://arxiv.org/pdf/2603.22663v1
- 발행일: 2026-03-24
- 카테고리: cs.MM
- 점수: final 92.0 (llm_adjusted:90 = base:80 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Short-form videos have become one of the most popular user-generated content formats nowadays. Popular short-video platforms use a simple streaming approach that preloads one or more videos in the recommendation list in advance. However, this approach results in significant data wastage, as a large portion of the downloaded video data is not used due to the user's early skip behavior. To address this problem, the chunk-based preloading approach has been proposed, where videos are divided into chunks, and preloading is performed in a chunk-based manner to reduce data wastage. To optimize chunk-based preloading, it is important to understand the user's viewing behavior in short-form video streaming. In this paper, we conduct a measurement study to construct a user behavior dataset that contains users' viewing times of one hundred short videos of various categories. Using the dataset, we evaluate the performance of standard time-series forecasting algorithms for predicting user viewing time in short-form video streaming. Our evaluation results show that Auto-ARIMA generally achieves the lowest and most stable forecasting errors across most experimental settings. The remaining methods, including AR, LR, SVR, and DTR, tend to produce higher errors and exhibit lower stability in many cases. The dataset is made publicly available at https://nvduc.github.io/shortvideodataset.

**선정 근거**
숏폼 영상 시청 행동 분석 연구는 스포츠 하이라이트 플랫폼의 콘텐츠 추천 알고리즘을 최적화하는 데 직접적으로 활용될 수 있습니다.

**활용 인사이트**
Auto-ARIMA 모델을 기반으로 한 시청 시간 예측 시스템을 구축하면 사용자의 선호에 맞는 스포츠 하이라이트를 자동으로 추천하여 플랫폼 이용률을 높일 수 있습니다.

## 4위: Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis

- arXiv: http://arxiv.org/abs/2603.22786v1
- PDF: https://arxiv.org/pdf/2603.22786v1
- 발행일: 2026-03-24
- 카테고리: cs.CV
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
Recent advances in 3D Gaussian Splatting have enabled impressive photorealistic novel view synthesis. However, to transition from a pure rendering engine to a reliable spatial map for autonomous agents and safety-critical applications, knowing where the representation is uncertain is as important as the rendering fidelity itself. We bridge this critical gap by introducing a lightweight, plug-and-play framework for pixel-wise, view-dependent predictive uncertainty estimation. Our post-hoc method formulates uncertainty as a Bayesian-regularized linear least-squares optimization over reconstruction residuals. This architecture-agnostic approach extracts a per-primitive uncertainty channel without modifying the underlying scene representation or degrading baseline visual fidelity. Crucially, we demonstrate that providing this actionable reliability signal successfully translates 3D Gaussian splatting into a trustworthy spatial map, further improving state-of-the-art performance across three critical downstream perception tasks: active view selection, pose-agnostic scene change detection, and pose-agnostic anomaly detection.

**선정 근거**
3D 가우시안 스플래팅의 불확실성 예측 기술은 스포츠 장면의 3D 재구성 및 중요 순간 정확히 식별 가능

**활용 인사이트**
픽셀 단위 불확실성 측정을 통해 스포츠 장면의 핵심 동작과 하이라이트 장면을 더 정확하게 포착하고 분석

## 5위: Toward Integrated Sensing, Communications, and Edge Intelligence Networks

- arXiv: http://arxiv.org/abs/2603.22958v1
- PDF: https://arxiv.org/pdf/2603.22958v1
- 발행일: 2026-03-24
- 카테고리: eess.SP
- 점수: final 88.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Wireless systems are expanding their purposes, from merely connecting humans and things to connecting intelligence and opportunistically sensing of the environment through radio-frequency signals. In this paper, we introduce the concept of triple-functional networks in which the same infrastructure and resources are shared for integrated sensing, communications, and (edge) Artificial Intelligence (AI) inference. This concept opens up several opportunities, such as devising non-orthogonal resource deployment and power consumption to concurrently update multiple services, but also challenges related to resource management and signaling cross-talk, among others. The core idea of this work is that computation-related aspects, including computing resources and AI models availability, should be explicitly considered when taking resource allocation decisions, to address the conflicting goals of the services coexistence. After showing the natural coupling between theoretical performance bounds of the three services, we formulate a service coexistence optimization problem that is solved optimally, and showcase the advantages against a disjoint allocation strategy.

**선정 근거**
통합 센싱, 통신, 에지 AI 네트워크로 스포츠 촬영 및 분석 시스템에 활용 가능

## 6위: Dual-Teacher Distillation with Subnetwork Rectification for Black-Box Domain Adaptation

- arXiv: http://arxiv.org/abs/2603.22908v1
- PDF: https://arxiv.org/pdf/2603.22908v1
- 발행일: 2026-03-24
- 카테고리: cs.CV, cs.LG
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Assuming that neither source data nor the source model is accessible, black box domain adaptation represents a highly practical yet extremely challenging setting, as transferable information is restricted to the predictions of the black box source model, which can only be queried using target samples. Existing approaches attempt to extract transferable knowledge through pseudo label refinement or by leveraging external vision language models (ViLs), but they often suffer from noisy supervision or insufficient utilization of the semantic priors provided by ViLs, which ultimately hinder adaptation performance. To overcome these limitations, we propose a dual teacher distillation with subnetwork rectification (DDSR) model that jointly exploits the specific knowledge embedded in black box source models and the general semantic information of a ViL. DDSR adaptively integrates their complementary predictions to generate reliable pseudo labels for the target domain and introduces a subnetwork driven regularization strategy to mitigate overfitting caused by noisy supervision. Furthermore, the refined target predictions iteratively enhance both the pseudo labels and ViL prompts, enabling more accurate and semantically consistent adaptation. Finally, the target model is further optimized through self training with classwise prototypes. Extensive experiments on multiple benchmark datasets validate the effectiveness of our approach, demonstrating consistent improvements over state of the art methods, including those using source data or models.

**선정 근거**
도메인 적응 기술은 다양한 스포츠 환경에서 AI 모델을 효과적으로 적용시키는 데 필수적입니다.

**활용 인사이트**
다양한 스포츠 종목별로 모델을 재학습하지 않고도 경기 분석 정확도를 크게 향상시킬 수 있습니다.

## 7위: GTLR-GS: Geometry-Texture Aware LiDAR-Regularized 3D Gaussian Splatting for Realistic Scene Reconstruction

- arXiv: http://arxiv.org/abs/2603.23192v1
- PDF: https://arxiv.org/pdf/2603.23192v1
- 발행일: 2026-03-24
- 카테고리: cs.GR, cs.MM
- 점수: final 84.0 (llm_adjusted:80 = base:75 + bonus:+5)
- 플래그: 실시간

**개요**
Recent advances in 3D Gaussian Splatting (3DGS) have enabled real-time, photorealistic scene reconstruction. However, conventional 3DGS frameworks typically rely on sparse point clouds derived from Structure-from-Motion (SfM), which inherently suffer from scale ambiguity, limited geometric consistency, and strong view dependency due to the lack of geometric priors. In this work, a LiDAR-centric 3D Gaussian Splatting framework is proposed that explicitly incorporates metric geometric priors into the entire Gaussian optimization process. Instead of treating LiDAR data as a passive initialization source, 3DGS optimization is reformulated as a geometry-conditioned allocation and refinement problem under a fixed representational budget. Specifically, this work introduces (i) a geometry-texture-aware allocation strategy that selectively assigns Gaussian primitives to regions with high structural or appearance complexity, (ii) a curvature-adaptive refinement mechanism that dynamically guides Gaussian splitting toward geometrically complex areas during training, and (iii) a confidence-aware metric depth regularization that anchors the reconstructed geometry to absolute scale using LiDAR measurements while maintaining optimization stability. Extensive experiments on the ScanNet++ dataset and a custom real-world dataset validate the proposed approach. The results demonstrate state-of-the-art performance in metric-scale reconstruction with high geometric fidelity.

**선정 근거**
LiDAR 기반 3D 가우시안 스플래팅은 스포츠 장면의 실시간 3D 재구성을 통해 다각도 분석을 가능하게 합니다.

**활용 인사이트**
선수들의 움직임을 3D 공간에서 정밀하게 분석하여 동작 개선점을 실시간으로 제공할 수 있습니다.

## 8위: VQ-Jarvis: Retrieval-Augmented Video Restoration Agent with Sharp Vision and Fast Thought

- arXiv: http://arxiv.org/abs/2603.22998v1
- PDF: https://arxiv.org/pdf/2603.22998v1
- 발행일: 2026-03-24
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:78 + bonus:+0)

**개요**
Video restoration in real-world scenarios is challenged by heterogeneous degradations, where static architectures and fixed inference pipelines often fail to generalize. Recent agent-based approaches offer dynamic decision making, yet existing video restoration agents remain limited by insufficient quality perception and inefficient search strategies. We propose VQ-Jarvis, a retrieval-augmented, all-in-one intelligent video restoration agent with sharper vision and faster thought. VQ-Jarvis is designed to accurately perceive degradations and subtle differences among paired restoration results, while efficiently discovering optimal restoration trajectories. To enable sharp vision, we construct VSR-Compare, the first large-scale video paired enhancement dataset with 20K comparison pairs covering 7 degradation types, 11 enhancement operators, and diverse content domains. Based on this dataset, we train a multiple operator judge model and a degradation perception model to guide agent decisions. To achieve fast thought, we introduce a hierarchical operator scheduling strategy that adapts to video difficulty: for easy cases, optimal restoration trajectories are retrieved in a one-step manner from a retrieval-augmented generation (RAG) library; for harder cases, a step-by-step greedy search is performed to balance efficiency and accuracy. Extensive experiments demonstrate that VQ-Jarvis consistently outperforms existing methods on complex degraded videos.

**선정 근거**
VQ-Jarvis는 실시간 비디오 복원 기술로 스포츠 촬영 품질 향상에 직접 적용 가능하며, 다양한 품질 저하 상황에 대응할 수 있는 강력한 복원 능력을 제공합니다.

**활용 인사이트**
경기 촬영 시 발생하는 화질 저하 문제를 해결하고, 하이라이트 영상의 품질을 향상시켜 SNS 공용에 적합한 고품질 콘텐츠를 생성할 수 있습니다.

## 9위: Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors

- arXiv: http://arxiv.org/abs/2603.23324v1
- PDF: https://arxiv.org/pdf/2603.23324v1
- 코드: https://github.com/zcq15/PFGS360
- 발행일: 2026-03-24
- 카테고리: cs.CV
- 점수: final 78.4 (llm_adjusted:73 = base:70 + bonus:+3)
- 플래그: 코드 공개

**개요**
Omnidirectional 3D Gaussian Splatting with panoramas is a key technique for 3D scene representation, and existing methods typically rely on slow SfM to provide camera poses and sparse points priors. In this work, we propose a pose-free omnidirectional 3DGS method, named PFGS360, that reconstructs 3D Gaussians from unposed omnidirectional videos. To achieve accurate camera pose estimation, we first construct a spherical consistency-aware pose estimation module, which recovers poses by establishing consistent 2D-3D correspondences between the reconstructed Gaussians and the unposed images using Gaussians' internal depth priors. Besides, to enhance the fidelity of novel view synthesis, we introduce a depth-inlier-aware densification module to extract depth inliers and Gaussian outliers with consistent monocular depth priors, enabling efficient Gaussian densification and achieving photorealistic novel view synthesis. The experiments show significant outperformance over existing pose-free and pose-aware 3DGS methods on both real-world and synthetic 360-degree videos. Code is available at https://github.com/zcq15/PFGS360.

**선정 근거**
360도 촬영 기술은 스포츠 경기 전체를 포괄적으로 기록하고 분석하는 데 중요합니다.

**활용 인사이트**
경기 전체를 360도로 촬영하여 선수들의 위치 관계와 전략을 다각도에서 분석할 수 있습니다.

## 10위: Harnessing Lightweight Transformer with Contextual Synergic Enhancement for Efficient 3D Medical Image Segmentation

- arXiv: http://arxiv.org/abs/2603.23390v1
- PDF: https://arxiv.org/pdf/2603.23390v1
- 코드: https://github.com/CUHK-AIM-Group/Light-UNETR
- 발행일: 2026-03-24
- 카테고리: cs.CV, eess.IV
- 점수: final 78.4 (llm_adjusted:73 = base:65 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Transformers have shown remarkable performance in 3D medical image segmentation, but their high computational requirements and need for large amounts of labeled data limit their applicability. To address these challenges, we consider two crucial aspects: model efficiency and data efficiency. Specifically, we propose Light-UNETR, a lightweight transformer designed to achieve model efficiency. Light-UNETR features a Lightweight Dimension Reductive Attention (LIDR) module, which reduces spatial and channel dimensions while capturing both global and local features via multi-branch attention. Additionally, we introduce a Compact Gated Linear Unit (CGLU) to selectively control channel interaction with minimal parameters. Furthermore, we introduce a Contextual Synergic Enhancement (CSE) learning strategy, which aims to boost the data efficiency of Transformers. It first leverages the extrinsic contextual information to support the learning of unlabeled data with Attention-Guided Replacement, then applies Spatial Masking Consistency that utilizes intrinsic contextual information to enhance the spatial context reasoning for unlabeled data. Extensive experiments on various benchmarks demonstrate the superiority of our approach in both performance and efficiency. For example, with only 10% labeled data on the Left Atrial Segmentation dataset, our method surpasses BCP by 1.43% Jaccard while drastically reducing the FLOPs by 90.8% and parameters by 85.8%. Code is released at https://github.com/CUHK-AIM-Group/Light-UNETR.

**선정 근거**
경량 트랜스포머를 사용한 의료 영상 분석으로 스포츠 영상 처리에 간접적으로 적용 가능

## 11위: Concept-based explanations of Segmentation and Detection models in Natural Disaster Management

- arXiv: http://arxiv.org/abs/2603.23020v1
- PDF: https://arxiv.org/pdf/2603.23020v1
- 발행일: 2026-03-24
- 카테고리: cs.CV, cs.AI
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Deep learning models for flood and wildfire segmentation and object detection enable precise, real-time disaster localization when deployed on embedded drone platforms. However, in natural disaster management, the lack of transparency in their decision-making process hinders human trust required for emergency response. To address this, we present an explainability framework for understanding flood segmentation and car detection predictions on the widely used PIDNet and YOLO architectures. More specifically, we introduce a novel redistribution strategy that extends Layer-wise Relevance Propagation (LRP) explanations for sigmoid-gated element-wise fusion layers. This extension allows LRP relevances to flow through the fusion modules of PIDNet, covering the entire computation graph back to the input image. Furthermore, we apply Prototypical Concept-based Explanations (PCX) to provide both local and global explanations at the concept level, revealing which learned features drive the segmentation and detection of specific disaster semantic classes. Experiments on a publicly available flood dataset show that our framework provides reliable and interpretable explanations while maintaining near real-time inference capabilities, rendering it suitable for deployment on resource-constrained platforms, such as Unmanned Aerial Vehicles (UAVs).

**선정 근거**
엣지 디바이스용 분할 및 탐지 기술을 스포츠 동작 감지에 적용 가능

**활용 인사이트**
AI의 동작 감지 결정을 투명하게 만들어 신뢰도 향상시킬 수 있음

## 12위: Object Pose Transformer: Unifying Unseen Object Pose Estimation

- arXiv: http://arxiv.org/abs/2603.23370v1
- PDF: https://arxiv.org/pdf/2603.23370v1
- 발행일: 2026-03-24
- 카테고리: cs.CV
- 점수: final 76.0 (llm_adjusted:70 = base:70 + bonus:+0)

**개요**
Learning model-free object pose estimation for unseen instances remains a fundamental challenge in 3D vision. Existing methods typically fall into two disjoint paradigms: category-level approaches predict absolute poses in a canonical space but rely on predefined taxonomies, while relative pose methods estimate cross-view transformations but cannot recover single-view absolute pose. In this work, we propose Object Pose Transformer (\ours{}), a unified feed-forward framework that bridges these paradigms through task factorization within a single model. \ours{} jointly predicts depth, point maps, camera parameters, and normalized object coordinates (NOCS) from RGB inputs, enabling both category-level absolute SA(3) pose and unseen-object relative SE(3) pose. Our approach leverages contrastive object-centric latent embeddings for canonicalization without requiring semantic labels at inference time, and uses point maps as a camera-space representation to enable multi-view relative geometric reasoning. Through cross-frame feature interaction and shared object embeddings, our model leverages relative geometric consistency across views to improve absolute pose estimation, reducing ambiguity in single-view predictions. Furthermore, \ours{} is camera-agnostic, learning camera intrinsics on-the-fly and supporting optional depth input for metric-scale recovery, while remaining fully functional in RGB-only settings. Extensive experiments on diverse benchmarks (NOCS, HouseCat6D, Omni6DPose, Toyota-Light) demonstrate state-of-the-art performance in both absolute and relative pose estimation tasks within a single unified architecture.

**선정 근거**
객체 자세 추정 기술로 스포츠 동작 분석에 직접 적용 가능

**활용 인사이트**
실시간 선수 자세 분석을 통해 훈련 및 성능 개선에 유용한 인사이트 제공

## 13위: MsFormer: Enabling Robust Predictive Maintenance Services for Industrial Devices

- arXiv: http://arxiv.org/abs/2603.23076v1
- PDF: https://arxiv.org/pdf/2603.23076v1
- 발행일: 2026-03-24
- 카테고리: cs.LG
- 점수: final 76.0 (llm_adjusted:70 = base:65 + bonus:+5)
- 플래그: 엣지

**개요**
Providing reliable predictive maintenance is a critical industrial AI service essential for ensuring the high availability of manufacturing devices. Existing deep-learning methods present competitive results on such tasks but lack a general service-oriented framework to capture complex dependencies in industrial IoT sensor data. While Transformer-based models show strong sequence modeling capabilities, their direct deployment as robust AI services faces significant bottlenecks. Specifically, streaming sensor data collected in real-world service environments often exhibits multi-scale temporal correlations driven by machine working principles. Besides, the datasets available for training time-to-failure predictive services are typically limited in size. These issues pose significant challenges for directly applying existing models as robust predictive services. To address these challenges, we propose MsFormer, a lightweight Multi-scale Transformer designed as a unified AI service model for reliable industrial predictive maintenance. MsFormer incorporates a Multi-scale Sampling (MS) module and a tailored position encoding mechanism to capture sequential correlations across multi-streaming service data. Additionally, to accommodate data-scarce service environments, MsFormer adopts a lightweight attention mechanism with straightforward pooling operations instead of self-attention. Extensive experiments on real-world datasets demonstrate that the proposed framework achieves significant performance improvements over state-of-the-art methods. Furthermore, MsFormer outperforms across industrial devices and operating conditions, demonstrating strong generalizability while maintaining a highly reliable Quality of Service (QoS).

**선정 근거**
경량 다중 스케일 트랜스포머 기술이 스포츠 동작 분석에 간접적으로 적용 가능

**활용 인사이트**
다중 스케일 접근법으로 미세 동작부터 전략까지 동시 분석 가능

## 14위: FHAvatar: Fast and High-Fidelity Reconstruction of Face-and-Hair Composable 3D Head Avatar from Few Casual Captures

- arXiv: http://arxiv.org/abs/2603.23345v1
- PDF: https://arxiv.org/pdf/2603.23345v1
- 발행일: 2026-03-24
- 카테고리: cs.CV
- 점수: final 72.0 (llm_adjusted:65 = base:60 + bonus:+5)
- 플래그: 실시간

**개요**
We present FHAvatar, a novel framework for reconstructing 3D Gaussian avatars with composable face and hair components from an arbitrary number of views. Unlike previous approaches that couple facial and hair representations within a unified modeling process, we explicitly decouple two components in texture space by representing the face with planar Gaussians and the hair with strand-based Gaussians. To overcome the limitations of existing methods that rely on dense multi-view captures or costly per-identity optimization, we propose an aggregated transformer backbone to learn geometry-aware cross-view priors and head-hair structural coherence from multi-view datasets, enabling effective and efficient feature extraction and fusion from few casual captures. Extensive quantitative and qualitative experiments demonstrate that FHAvatar achieves state-of-the-art reconstruction quality from only a few observations of new identities within minutes, while supporting real-time animation, convenient hairstyle transfer, and stylized editing, broadening the accessibility and applicability of digital avatar creation.

**선정 근거**
3D 아바타 재구성 기술로 스포츠 선수의 3D 모델링에 잠재적 응용 가능성 있음

**활용 인사이트**
제한된 카메라 각도에서 선수의 3D 모델 생성으로 영상 분석 기능 향상

## 15위: A Latency Coding Framework for Deep Spiking Neural Networks with Ultra-Low Latency

- arXiv: http://arxiv.org/abs/2603.23206v1
- PDF: https://arxiv.org/pdf/2603.23206v1
- 발행일: 2026-03-24
- 카테고리: cs.NE
- 점수: final 72.0 (llm_adjusted:65 = base:55 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Spiking neural networks (SNNs) offer a biologically inspired computing paradigm with significant potential for energy-efficient neural processing. Among neural coding schemes of SNNs, Time-To-First-Spike (TTFS) coding, which encodes information through the precise timing of a neuron's first spike, provides exceptional energy efficiency and biological plausibility. Despite its theoretical advantages, existing TTFS models lack efficient training methods, suffering from high inference latency and limited performance. In this work, we present a comprehensive framework, which enables the efficient training of deep TTFS-coded SNNs by employing backpropagation throuh time (BPTT) algorithm. We name the generalized TTFS coding method in our framework as latency coding. The framework includes: (1) a latency encoding (LE) module with feature extraction and straight-through estimators to address severe information loss in direct intensity-to-latency mapping and ensure smooth gradient flow; (2) relaxation of the strict single-spike constraint of traditional TTFS, allowing neurons of intermediate layers to fire multiple times to mitigating gradient vanishing in deep networks; (3) a temporal adaptive decision (TAD) loss function that dynamically weights supervision signals based on sample-dependent confidence, resolving the incompatibility between latency coding and standard cross-entropy loss. Experimental results demonstrate that our method achieves state-of-the-art accuracy in comparison to existing TTFS-coded SNNs with ultra-low inference latency and superior energy efficiency. The framework also demonstrates improved robustness against input corruptions. Our study investigates the characteristics and potential of latency coding in scenarios demanding rapid response, providing valuable insights for further exploiting the temporal learning capabilities of SNNs.

**선정 근거**
초저 지연 처리 기술로 실시간 스포츠 분석에 간접적으로 활용 가능

**활용 인사이트**
초저 지연 처리로 실시간 피드백을 통한 훈련 및 경기 중 즉각적 개선 가능

## 16위: Double Coupling Architecture and Training Method for Optimization Problems of Differential Algebraic Equations with Parameters

- arXiv: http://arxiv.org/abs/2603.22724v1
- PDF: https://arxiv.org/pdf/2603.22724v1
- 발행일: 2026-03-24
- 카테고리: cs.LG, math.AP
- 점수: final 71.2 (llm_adjusted:64 = base:59 + bonus:+5)
- 플래그: 실시간

**개요**
Simulation and modeling are essential in product development, integrated into the design and manufacturing process to enhance efficiency and quality. They are typically represented as complex nonlinear differential algebraic equations. The growing diversity of product requirements demands multi-task optimization, a key challenge in simulation modeling research. A dual physics-informed neural network architecture has been proposed to decouple constraints and objective functions in parametric differential algebraic equation optimization problems. Theoretical analysis shows that introducing a relaxation variable with a global error bound ensures solution equivalence between the network and optimization problem. A genetic algorithm-enhanced training framework for physics-informed neural networks improves training precision and efficiency, avoiding redundant solving of differential algebraic equations. This approach enables generalization for multi-task objectives with a single, training maintaining real-time responsiveness to product requirements.

**선정 근거**
이 논문은 실시간 처리 기능을 갖춘 강화 학습 접근 방식을 제안합니다. 핵심은 자원 할당 최적화 기법입니다. 스포츠 촬영 시스템에서 중요한 순간을 놓치지 않는 동적 촬영 전략 수립에 적용 가능하며, 실시간 성능과 해석 가능성을 동시에 확보하는 데 중요합니다.

**활용 인사이트**
인덱스 기반 정책을 스포츠 촬영에 적용하여 각 카메라에 중요도 점수를 부여하고, 경기 상황에 따라 실시간으로 촬영 각도와 초점을 최적화하여 하이라이트 장면을 효과적으로 포착할 수 있습니다. 이를 통해 fps와 지연 시간을 최적화할 수 있습니다.

## 17위: Minimizing Material Waste in Additive Manufacturing through Online Reel Assignment

- arXiv: http://arxiv.org/abs/2603.23042v1
- PDF: https://arxiv.org/pdf/2603.23042v1
- 발행일: 2026-03-24
- 카테고리: math.OC
- 점수: final 71.2 (llm_adjusted:64 = base:59 + bonus:+5)
- 플래그: 실시간

**개요**
We study a variant of the online bin packing problem that arises in filament-based 3D printing systems operating in make-to-order settings, where only a limited number of filament reels of finite capacity can be handled at once. Components are assigned to reels upon arrival and insufficient reels are discarded to be replaced with new ones, resulting in material waste. To minimize the long-run average discarded filament through an online assignment policy, we formulate this problem as an infinite-horizon average-cost Markov Decision Process and analyze the structure of policies under stochastic, sequential demand. We first show that under a random allocation policy, the system decomposes into a collection of identical single-reel processes, allowing us to derive a closed-form expression for the average waste and enabling a tractable baseline analysis. Building on this decomposition, we construct a theoretically grounded index policy that assigns each reel a score reflecting the marginal cost of assignment and prove that it constitutes a one-step policy improvement over random allocation. We embed the index-based structure within a Deep Reinforcement Learning framework using approximate policy iteration. The resulting method achieves near-optimal performance across a range of simulated and real-world scenarios. Our results demonstrate that Reinforcement Learning policy significantly reduces material waste while maintaining real-time feasibility and interpretability.

**선정 근거**
이 논문은 음성-시청각 인식의 강건성 문제를 다룹니다. 핵심은 다양한 조건에서의 모델 성능 유지 방법입니다. 스포츠 촬영 시 발생할 수 있는 다양한 환경적 요인(조명, 거리, 움직임 등)에 대한 모델의 강건성 향상에 참고할 수 있습니다.

**활용 인사이트**
논문의 데이터셋 구축 방식을 차용하여 다양한 스포츠 환경에서의 촬영 데이터를 수집하고, 이를 바탕으로 AI 모델이 다양한 조건에서도 일관된 추론 속도를 유지하도록 훈련시킬 수 있습니다. 파라미터 수를 최적화하여 실시간 처리를 가능하게 합니다.

## 18위: It Takes Two: A Duet of Periodicity and Directionality for Burst Flicker Removal

- arXiv: http://arxiv.org/abs/2603.22794v1
- PDF: https://arxiv.org/pdf/2603.22794v1
- 코드: https://github.com/qulishen/Flickerformer
- 발행일: 2026-03-24
- 카테고리: cs.CV
- 점수: final 70.4 (llm_adjusted:63 = base:60 + bonus:+3)
- 플래그: 코드 공개

**개요**
Flicker artifacts, arising from unstable illumination and row-wise exposure inconsistencies, pose a significant challenge in short-exposure photography, severely degrading image quality. Unlike typical artifacts, e.g., noise and low-light, flicker is a structured degradation with specific spatial-temporal patterns, which are not accounted for in current generic restoration frameworks, leading to suboptimal flicker suppression and ghosting artifacts. In this work, we reveal that flicker artifacts exhibit two intrinsic characteristics, periodicity and directionality, and propose Flickerformer, a transformer-based architecture that effectively removes flicker without introducing ghosting. Specifically, Flickerformer comprises three key components: a phase-based fusion module (PFM), an autocorrelation feed-forward network (AFFN), and a wavelet-based directional attention module (WDAM). Based on the periodicity, PFM performs inter-frame phase correlation to adaptively aggregate burst features, while AFFN exploits intra-frame structural regularities through autocorrelation, jointly enhancing the network's ability to perceive spatially recurring patterns. Moreover, motivated by the directionality of flicker artifacts, WDAM leverages high-frequency variations in the wavelet domain to guide the restoration of low-frequency dark regions, yielding precise localization of flicker artifacts. Extensive experiments demonstrate that Flickerformer outperforms state-of-the-art approaches in both quantitative metrics and visual quality. The source code is available at https://github.com/qulishen/Flickerformer.

**선정 근거**
이미지 보정 기술로 스포츠 영상 향상에 직접 적용 가능

**활용 인사이트**
주기성과 방향성을 활용한 플리커 제거 기술로 영상 품질 개선

## 19위: From Feature Learning to Spectral Basis Learning: A Unifying and Flexible Framework for Efficient and Robust Shape Matching

- arXiv: http://arxiv.org/abs/2603.23383v1
- PDF: https://arxiv.org/pdf/2603.23383v1
- 코드: https://github.com/LuoFeifan77/Unsupervised-Spectral-Basis-Learning
- 발행일: 2026-03-24
- 카테고리: cs.CV
- 점수: final 70.4 (llm_adjusted:63 = base:60 + bonus:+3)
- 플래그: 코드 공개

**개요**
Shape matching is a fundamental task in computer graphics and vision, with deep functional maps becoming a prominent paradigm. However, existing methods primarily focus on learning informative feature representations by constraining pointwise and functional maps, while neglecting the optimization of the spectral basis-a critical component of the functional map pipeline. This oversight often leads to suboptimal matching results. Furthermore, many current approaches rely on conventional, time-consuming functional map solvers, incurring significant computational overhead. To bridge these gaps, we introduce Advanced Functional Maps, a framework that generalizes standard functional maps by replacing fixed basis functions with learnable ones, supported by rigorous theoretical guarantees. Specifically, the spectral basis is optimized through a set of learned inhibition functions. Building on this, we propose the first unsupervised spectral basis learning method for robust non-rigid 3D shape matching, enabling the joint, end-to-end optimization of feature extraction and basis functions. Our approach incorporates a novel heat diffusion module and an unsupervised loss function, alongside a streamlined architecture that bypasses expensive solvers and auxiliary losses. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art feature-learning approaches, particularly in challenging non-isometric and topological noise scenarios, while maintaining high efficiency. Finally, we reveal that optimizing basis functions is equivalent to spectral convolution, where inhibition functions act as filters. This insight enables enhanced representations inspired by spectral graph networks, opening new avenues for future research. Our code is available at https://github.com/LuoFeifan77/Unsupervised-Spectral-Basis-Learning.

**선정 근거**
형상 일치 기술로 스포츠 동작 분석에 간접적으로 적용 가능

**활용 인사이트**
스펙트럼 기반 학습을 통한 효율적이고 강건한 형상 일치로 동작 분석 정확도 향상

## 20위: When AVSR Meets Video Conferencing: Dataset, Degradation, and the Hidden Mechanism Behind Performance Collapse

- arXiv: http://arxiv.org/abs/2603.22915v1
- PDF: https://arxiv.org/pdf/2603.22915v1
- 발행일: 2026-03-24
- 카테고리: cs.CV
- 점수: final 70.4 (llm_adjusted:63 = base:55 + bonus:+8)
- 플래그: 실시간, 코드 공개

**개요**
Audio-Visual Speech Recognition (AVSR) has achieved remarkable progress in offline conditions, yet its robustness in real-world video conferencing (VC) remains largely unexplored. This paper presents the first systematic evaluation of state-of-the-art AVSR models across mainstream VC platforms, revealing severe performance degradation caused by transmission distortions and spontaneous human hyper-expression. To address this gap, we construct \textbf{MLD-VC}, the first multimodal dataset tailored for VC, comprising 31 speakers, 22.79 hours of audio-visual data, and explicit use of the Lombard effect to enhance human hyper-expression. Through comprehensive analysis, we find that speech enhancement algorithms are the primary source of distribution shift, which alters the first and second formants of audio. Interestingly, we find that the distribution shift induced by the Lombard effect closely resembles that introduced by speech enhancement, which explains why models trained on Lombard data exhibit greater robustness in VC. Fine-tuning AVSR models on MLD-VC mitigates this issue, achieving an average 17.5% reduction in CER across several VC platforms. Our findings and dataset provide a foundation for developing more robust and generalizable AVSR systems in real-world video conferencing. MLD-VC is available at https://huggingface.co/datasets/nccm2p2/MLD-VC.

**선정 근거**
Focuses on audio-visual speech recognition in video conferencing, not directly related to sports filming or analysis

## 21위: Curriculum-Driven 3D CT Report Generation via Language-Free Visual Grafting and Zone-Constrained Compression

- arXiv: http://arxiv.org/abs/2603.23308v1
- PDF: https://arxiv.org/pdf/2603.23308v1
- 발행일: 2026-03-24
- 카테고리: cs.CV, cs.AI
- 점수: final 70.4 (llm_adjusted:63 = base:55 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Automated radiology report generation from 3D computed tomography (CT) volumes is challenging due to extreme sequence lengths, severe class imbalance, and the tendency of large language models (LLMs) to ignore visual tokens in favor of linguistic priors. We present Ker-VLJEPA-3B, a four-phase curriculum learning framework for free-text report generation from thoracic CT volumes. A phased training curriculum progressively adapts a Llama 3.2 3B decoder to ground its output in visual features from a frozen, self-supervised encoder. Our visual backbone (LeJEPA ViT-Large) is trained via self-supervised joint-embedding prediction on unlabeled CTs, without text supervision. Unlike contrastive models (CLIP, BiomedCLIP), this language-free backbone yields modality-pure representations. Vision-language alignment is deferred to the curriculum's bridge and generation phases. This modality-agnostic design can integrate any self-supervised encoder into an LLM without paired text during foundation training. Methodological innovations include: (1) zone-constrained cross-attention compressing slice embeddings into 32 spatially-grounded visual tokens; (2) PCA whitening of anisotropic LLM embeddings; (3) a positive-findings-only strategy eliminating posterior collapse; (4) warm bridge initialization transferring projection weights; and (5) selective cross-attention freezing with elastic weight consolidation to prevent catastrophic forgetting. Evaluated on the CT-RATE benchmark (2,984 validation volumes, 18 classes), Ker-VLJEPA-3B achieves a macro F1 of 0.429, surpassing the state-of-the-art (U-VLM, macro F1 = 0.414) by 3.6%, and reaching 0.448 (+8.2%) with threshold optimization. Ablation studies confirm 56.6% of generation quality derives from patient-specific visual content. Code and weights are available.

**선정 근거**
의료 영상 처리 기술로서 비주얼 처리 방식은 스포츠 영상 분석에 간접적으로 적용 가능

**활용 인사이트**
비전-언어 정렬 기법을 활용한 시각적 특징 추출로 선수별 장면 분류 및 하이라이트 생성

## 22위: VoDaSuRe: A Large-Scale Dataset Revealing Domain Shift in Volumetric Super-Resolution

- arXiv: http://arxiv.org/abs/2603.23153v1
- PDF: https://arxiv.org/pdf/2603.23153v1
- 발행일: 2026-03-24
- 카테고리: cs.CV
- 점수: final 70.4 (llm_adjusted:63 = base:60 + bonus:+3)
- 플래그: 코드 공개

**개요**
Recent advances in volumetric super-resolution (SR) have demonstrated strong performance in medical and scientific imaging, with transformer- and CNN-based approaches achieving impressive results even at extreme scaling factors. In this work, we show that much of this performance stems from training on downsampled data rather than real low-resolution scans. This reliance on downsampling is partly driven by the scarcity of paired high- and low-resolution 3D datasets. To address this, we introduce VoDaSuRe, a large-scale volumetric dataset containing paired high- and low-resolution scans. When training models on VoDaSuRe, we reveal a significant discrepancy: SR models trained on downsampled data produce substantially sharper predictions than those trained on real low-resolution scans, which smooth fine structures. Conversely, applying models trained on downsampled data to real scans preserves more structure but is inaccurate. Our findings suggest that current SR methods are overstated - when applied to real data, they do not recover structures lost in low-resolution scans and instead predict a smoothed average. We argue that progress in deep learning-based volumetric SR requires datasets with paired real scans of high complexity, such as VoDaSuRe. Our dataset and code are publicly available through: https://augusthoeg.github.io/VoDaSuRe/

**선정 근거**
3D 슈퍼 해상도 기술로 영상 보정 기술에 간접적으로 적용 가능

**활용 인사이트**
실제 저해상도 영상 처리 기술로 스포츠 영상의 화질 보정 및 향상 가능

## 23위: PNap: Lifecycle-aware Edge Multi-state sleep for Energy Efficient MEC

- arXiv: http://arxiv.org/abs/2603.23323v1
- PDF: https://arxiv.org/pdf/2603.23323v1
- 발행일: 2026-03-24
- 카테고리: cs.NI
- 점수: final 70.4 (llm_adjusted:63 = base:58 + bonus:+5)
- 플래그: 엣지

**개요**
Multi-access Edge Computings (MECs) enables low-latency services by executing applications at the network edge. To fulfill low-latency requirements of mobile users, providers have to keep multiple edge servers running at multiple locations, even when, in low-load phases, their capacity is not needed. This significantly increases energy consumption. Multi-state sleep mechanisms mitigate this issue by allowing servers to enter progressively deeper sleep states, trading energy savings for longer wake-up delays. At the same time, service execution depends on non-instantaneous lifecycle operations that cannot be performed while servers are asleep, tightly coupling energy management with service continuity. This paper introduces PowerNap (PNap), a lifecycle-aware orchestration framework that jointly manages server sleep states and service lifecycle states. By leveraging traffic forecasting, PNap jointly minimizes the number of active edge servers and service disruptions. We compare PNap against baselines approaches and a state-of-the-art approach. Results validate PNap, showing how it can reduce energy consumption by up to 14.9% with respect to a state-of-the-art solution while matching its service availability results.

**선정 근거**
에지 컴퓨팅의 에너지 효율화 기술로 스포츠 AI 장치의 전력 관리에 활용 가능

**활용 인사이트**
트래픽 예측을 활용한 서버 상태 관리로 전력 소비 최대 14.9% 절감 가능

## 24위: Digital Twin Enabled Simultaneous Learning and Modeling for UAV-assisted Secure Communications with Eavesdropping Attacks

- arXiv: http://arxiv.org/abs/2603.22753v1
- PDF: https://arxiv.org/pdf/2603.22753v1
- 발행일: 2026-03-24
- 카테고리: cs.NI, cs.CR
- 점수: final 69.6 (llm_adjusted:62 = base:62 + bonus:+0)

**개요**
This paper focuses on secure communications in UAV-assisted wireless networks, which comprise multiple legitimate UAVs (LE-UAVs) and an intelligent eavesdropping UAV (EA-UAV). The intelligent EA-UAV can observe the LE-UAVs'transmission strategies and adaptively adjust its trajectory to maximize information interception. To counter this threat, we propose a mode-switching scheme that enables LE-UAVs to dynamically switch between the data transmission and jamming modes, thereby balancing data collection efficiency and communication security. However, acquiring full global network state information for LE-UAVs' decision-making incurs significant overhead, as the network state is highly dynamic and time-varying. To address this challenge, we propose a digital twin-enabled simultaneous learning and modeling (DT-SLAM) framework that allows LE-UAVs to learn policies efficiently within the DT, thereby avoiding frequent interactions with the real environment. To capture the competitive relationship between the EA-UAV and the LE-UAVs, we model their interactions as a multi-stage Stackelberg game and jointly optimize the GUs' transmission control, UAVs' trajectory planning, mode selection, and network formation to maximize overall secure throughput. Considering potential model mismatch between the DT and the real environment, we propose a robust proximal policy optimization (RPPO) algorithm that encourages LE-UAVs to explore service regions with higher uncertainty. Numerical results demonstrate that the proposed DT-SLAM framework effectively supports the learning process. Meanwhile, the RPPO algorithm converges about 12% faster and the secure throughput can be increased by 8.6% compared to benchmark methods.

**선정 근거**
UAV 및 디지털 트윈 개념이 스포츠 촬영 및 모델링에 잠재적으로 적용 가능

**활용 인사이트**
디지털 트윈을 통한 학습으로 실제 환경과의 상호작용 최소화하며 모델링 효율성 향상

## 25위: Rao-Blackwellized Stein Gradient Descent for Joint State-Parameter Estimation

- arXiv: http://arxiv.org/abs/2603.23039v1
- PDF: https://arxiv.org/pdf/2603.23039v1
- 발행일: 2026-03-24
- 카테고리: eess.SY
- 점수: final 68.0 (llm_adjusted:60 = base:55 + bonus:+5)
- 플래그: 실시간

**개요**
We present a filtering framework for online joint state estimation and parameter identification in nonlinear, time-varying systems. The algorithm uses Rao-Blackwellization technique to infer joint state-parameter posteriors efficiently. In particular, conditional state distributions are computed analytically via Kalman filtering, while model parameters including process and measurement noise covariances are approximated using particle-based Stein Variational Gradient Descent (SVGD), enabling stable real-time inference. We prove a theoretical consistency result by bounding the impact of the SVGD approximated parameter posterior on state estimates, relating the divergence between the true and approximate parameter posteriors to the total variation distance between the resulting state marginals. Performance of the proposed filter is validated on two case studies: a bioreactor with Haldane kinetics and a neural-network-augmented dynamic system. The latter demonstrates the filter's capacity for online neural network training within a dynamical model, showcasing its potential for fully adaptive, data-driven system identification.

**선정 근거**
상태 추정 기술이 스포츠 동작 분석에 간접적으로 참조 가능

**활용 인사이트**
실시간 상태-파라미터 추정으로 선수의 동작 패턴 분석 및 전략 수립에 활용

## 26위: InterDyad: Interactive Dyadic Speech-to-Video Generation by Querying Intermediate Visual Guidance

- arXiv: http://arxiv.org/abs/2603.23132v1
- PDF: https://arxiv.org/pdf/2603.23132v1
- 발행일: 2026-03-24
- 카테고리: cs.CV
- 점수: final 68.0 (llm_adjusted:60 = base:60 + bonus:+0)

**개요**
Despite progress in speech-to-video synthesis, existing methods often struggle to capture cross-individual dependencies and provide fine-grained control over reactive behaviors in dyadic settings. To address these challenges, we propose InterDyad, a framework that enables naturalistic interactive dynamics synthesis via querying structural motion guidance. Specifically, we first design an Interactivity Injector that achieves video reenactment based on identity-agnostic motion priors extracted from reference videos. Building upon this, we introduce a MetaQuery-based modality alignment mechanism to bridge the gap between conversational audio and these motion priors. By leveraging a Multimodal Large Language Model (MLLM), our framework is able to distill linguistic intent from audio to dictate the precise timing and appropriateness of reactions. To further improve lip-sync quality under extreme head poses, we propose Role-aware Dyadic Gaussian Guidance (RoDG) for enhanced lip-synchronization and spatial consistency. Finally, we introduce a dedicated evaluation suite with novelly designed metrics to quantify dyadic interaction. Comprehensive experiments demonstrate that InterDyad significantly outperforms state-of-the-art methods in producing natural and contextually grounded two-person interactions. Please refer to our project page for demo videos: https://interdyad.github.io/.

**선정 근거**
상호작용 동합성 기술이 스포츠 상호작용 분석에 간접적으로 적용 가능하여 경기 전략 분석에 활용

**활용 인사이트**
InterDyad의 상호작용 인젝터와 메타쿼리 기반 모달리티 정렬로 선수 간 상호작용 자연스럽게 합성 분석

## 27위: TimeWeaver: Age-Consistent Reference-Based Face Restoration with Identity Preservation

- arXiv: http://arxiv.org/abs/2603.22701v1
- PDF: https://arxiv.org/pdf/2603.22701v1
- 발행일: 2026-03-24
- 카테고리: cs.CV
- 점수: final 68.0 (llm_adjusted:60 = base:60 + bonus:+0)

**개요**
Recent progress in face restoration has shifted from visual fidelity to identity fidelity, driving a transition from reference-free to reference-based paradigms that condition restoration on reference images of the same person. However, these methods assume the reference and degraded input are age-aligned. When only cross-age references are available, as in historical restoration or missing-person retrieval, they fail to maintain age fidelity. To address this limitation, we propose TimeWeaver, the first reference-based face restoration framework supporting cross-age references. Given arbitrary reference images and a target-age prompt, TimeWeaver produces restorations with both identity fidelity and age consistency. Specifically, we decouple identity and age conditioning across training and inference. During training, the model learns an age-robust identity representation by fusing a global identity embedding with age-suppressed facial tokens via a transformer-based ID-Fusion module. During inference, two training-free techniques, Age-Aware Gradient Guidance and Token-Targeted Attention Boost, steer sampling toward desired age semantics, enabling precise adherence to the target-age prompt. Extensive experiments show that TimeWeaver surpasses existing methods in visual quality, identity preservation, and age consistency.

**선정 근거**
이 논문은 크로스 연령 참조 이미지를 기반으로 한 얼굴 복원 방법을 제안합니다. 핵심은 정체성과 연령 조건을 분리하여 처리하는 것입니다.

**활용 인사이트**
TimeWeaver의 ID-Fusion 모듈과 연령 인식 기법을 스포츠 영상의 인물 향상에 적용하여 연령에 맞는 자연스러운 복원을 구현할 수 있습니다.

## 28위: Accelerating Maximum Common Subgraph Computation by Exploiting Symmetries

- arXiv: http://arxiv.org/abs/2603.23031v1
- PDF: https://arxiv.org/pdf/2603.23031v1
- 발행일: 2026-03-24
- 카테고리: cs.DS
- 점수: final 67.2 (llm_adjusted:59 = base:59 + bonus:+0)

**개요**
The Maximum Common Subgraph (MCS) problem plays a key role in many applications, including cheminformatics, bioinformatics, and pattern recognition, where it is used to identify the largest shared substructure between two graphs. Although symmetry exploitation is a powerful means of reducing search space in combinatorial optimization, its potential in MCS algorithms has remained largely underexplored due to the challenges of detecting and integrating symmetries effectively. Existing approaches, such as RRSplit, partially address symmetry through vertex-equivalence reasoning on the variable graph, but symmetries in the value graph remain unexploited. In this work, we introduce a complete dual-symmetry breaking framework that simultaneously handles symmetries in both variable and value graphs. Our method identifies and exploits modular symmetries based on local neighborhood structures, allowing the algorithm to prune isomorphic subtrees during search while rigorously preserving optimality. Extensive experiments on standard MCS benchmarks show that our approach substantially outperforms the state-of-the-art RRSplit algorithm, solving more instances with significant reductions in both computation time and search space. These results highlight the practical effectiveness of comprehensive symmetry-aware pruning for accelerating exact MCS computation.

**선정 근거**
비디오 분석에 잠재적 응용이 가능한 패턴 인용 그래프 알고리즘

## 29위: Detecting outliers of pursuit eye movements: a preliminary analysis of autism spectrum disorder

- arXiv: http://arxiv.org/abs/2603.22705v1
- PDF: https://arxiv.org/pdf/2603.22705v1
- 발행일: 2026-03-24
- 카테고리: q-bio.NC, q-bio.PE
- 점수: final 67.2 (llm_adjusted:59 = base:59 + bonus:+0)

**개요**
Background: Autism spectrum disorder (ASD) is characterized by significant clinical and biological heterogeneity. Conventional group-mean analyses of eye movements often mask individual atypicalities, potentially overlooking critical pathological signatures. This study aimed to identify idiosyncratic oculomotor patterns in ASD using an "outlier analysis" of smooth pursuit eye movement (SPEM).   Methods: We recorded SPEM during a slow Lissajous pursuit task in 18 adults with ASD and 39 typically developed (TD) individuals. To quantify individual deviations, we derived an "outlier score" based on the Mahalanobis distance. This score was calculated from a feature vector, optimized via Principal Component Analysis (PCA), comprising the temporal lag ($Δ$t) and the spatial deviation ($Δ$s). An outlier was statistically defined as a score exceeding $\sqrt{10}$ (approximately 3.16$σ$) relative to the TD normative distribution.   Results: While the TD group exhibited a low outlier rate of 5.1\%, the ASD group demonstrated a significantly higher prevalence of 38.9\% (7/18) (binomial P = 0.0034). Furthermore, the mean outlier score was significantly elevated in the ASD group (3.00 $\pm$ 2.62) compared to the TD group (1.52 $\pm$ 0.80; P = 0.002). Notably, these extreme deviations were captured even when conventional mean-based comparisons showed limited sensitivity.   Conclusions: Our outlier analysis successfully visualized the high degree of idiosyncratic atypicality in ASD oculomotor control. By shifting the focus from group averages to individual deviations, this approach provides a sensitive metric for capturing the inherent heterogeneity of ASD, offering a potential baseline for identifying clinical subtypes.

**선정 근거**
움직임 분석 기술이 스포츠 동작 분석에 직접 적용 가능하여 개인별 기술 평가에 유용

**활용 인사이트**
이상치 분석 방식과 마할라노비스 거리 기반 특징 추출로 선수 동작 패턴 개인별 특이점 식별

## 30위: Avoiding Over-smoothing in Social Media Rumor Detection with Pre-trained Propagation Tree Transformer

- arXiv: http://arxiv.org/abs/2603.22854v1
- PDF: https://arxiv.org/pdf/2603.22854v1
- 발행일: 2026-03-24
- 카테고리: cs.CL, cs.AI
- 점수: final 66.4 (llm_adjusted:58 = base:58 + bonus:+0)

**개요**
Deep learning techniques for rumor detection typically utilize Graph Neural Networks (GNNs) to analyze post relations. These methods, however, falter due to over-smoothing issues when processing rumor propagation structures, leading to declining performance. Our investigation into this issue reveals that over-smoothing is intrinsically tied to the structural characteristics of rumor propagation trees, in which the majority of nodes are 1-level nodes. Furthermore, GNNs struggle to capture long-range dependencies within these trees. To circumvent these challenges, we propose a Pre-Trained Propagation Tree Transformer (P2T3) method based on pure Transformer architecture. It extracts all conversation chains from a tree structure following the propagation direction of replies, utilizes token-wise embedding to infuse connection information and introduces necessary inductive bias, and pre-trains on large-scale unlabeled datasets. Experiments indicate that P2T3 surpasses previous state-of-the-art methods in multiple benchmark datasets and performs well under few-shot conditions. P2T3 not only avoids the over-smoothing issue inherent in GNNs but also potentially offers a large model or unified multi-modal scheme for future social media research.

**선정 근거**
소셜 미디어 분석이 콘텐츠 공유 플랫폼과 관련

## 31위: A Feature Shuffling and Restoration Strategy for Universal Unsupervised Anomaly Detection

- arXiv: http://arxiv.org/abs/2603.22861v1 | 2026-03-24 | final 66.4

Unsupervised anomaly detection is vital in industrial fields, with reconstruction-based methods favored for their simplicity and effectiveness. However, reconstruction methods often encounter an identical shortcut issue, where both normal and anomalous regions can be well reconstructed and fail to identify outliers.

-> 이상 탐지 기술로 스포츠 영상 분석에 간접적으로 적용 가능

## 32위: 3rd Place of MeViS-Audio Track of the 5th PVUW: VIRST-Audio

- arXiv: http://arxiv.org/abs/2603.23126v1 | 2026-03-24 | final 64.0

Audio-based Referring Video Object Segmentation (ARVOS) requires grounding audio queries into pixel-level object masks over time, posing challenges in bridging acoustic signals with spatio-temporal visual representations. In this report, we present VIRST-Audio, a practical framework built upon a pretrained RVOS model integrated with a vision-language architecture.

-> 오디오 기반 비디오 객체 분할 기술로 스포츠 촬영과 간접적 연관성 있음

## 33위: Instrument-Splatting++: Towards Controllable Surgical Instrument Digital Twin Using Gaussian Splatting

- arXiv: http://arxiv.org/abs/2603.22792v1 | 2026-03-24 | final 64.0

High-quality and controllable digital twins of surgical instruments are critical for Real2Sim in robot-assisted surgery, as they enable realistic simulation, synthetic data generation, and perception learning under novel poses. We present Instrument-Splatting++, a monocular 3D Gaussian Splatting (3DGS) framework that reconstructs surgical instruments as a fully controllable Gaussian asset with high fidelity.

-> Pose estimation techniques could be adapted for sports movement analysis but focused on surgical instruments

## 34위: ARGENT: Adaptive Hierarchical Image-Text Representations

- arXiv: http://arxiv.org/abs/2603.23311v1 | 2026-03-24 | final 64.0

Large-scale Vision-Language Models (VLMs) such as CLIP learn powerful semantic representations but operate in Euclidean space, which fails to capture the inherent hierarchical structure of visual and linguistic concepts. Hyperbolic geometry, with its exponential volume growth, offers a principled alternative for embedding such hierarchies with low distortion.

-> 이미지 이해 및 분석 기술을 다루지만 스포츠나 운동 분석과 직접적으로 관련되지 않음

## 35위: Conformal Cross-Modal Active Learning

- arXiv: http://arxiv.org/abs/2603.23159v1 | 2026-03-24 | final 64.0

Foundation models for vision have transformed visual recognition with powerful pretrained representations and strong zero-shot capabilities, yet their potential for data-efficient learning remains largely untapped. Active Learning (AL) aims to minimize annotation costs by strategically selecting the most informative samples for labeling, but existing methods largely overlook the rich multimodal knowledge embedded in modern vision-language models (VLMs).

-> 크로스 모달 액티브 러닝으로 스포츠 영상 분석에 간접적으로 적용 가능

## 36위: AgentFoX: LLM Agent-Guided Fusion with eXplainability for AI-Generated Image Detection

- arXiv: http://arxiv.org/abs/2603.23115v1 | 2026-03-24 | final 64.0

The increasing realism of AI-Generated Images (AIGI) has created an urgent need for forensic tools capable of reliably distinguishing synthetic content from authentic imagery. Existing detectors are typically tailored to specific forgery artifacts--such as frequency-domain patterns or semantic inconsistencies--leading to specialized performance and, at times, conflicting judgments.

-> AI 생성 이미지 탐지로 스포츠 영상 분석에 간접적으로 적용 가능

## 37위: URA-Net: Uncertainty-Integrated Anomaly Perception and Restoration Attention Network for Unsupervised Anomaly Detection

- arXiv: http://arxiv.org/abs/2603.22840v1 | 2026-03-24 | final 64.0

Unsupervised anomaly detection plays a pivotal role in industrial defect inspection and medical image analysis, with most methods relying on the reconstruction framework. However, these methods may suffer from over-generalization, enabling them to reconstruct anomalies well, which leads to poor detection performance.

-> Focuses on anomaly detection and restoration in industrial/medical images, not directly applicable to sports analysis

## 38위: Conditionally Identifiable Latent Representation for Multivariate Time Series with Structural Dynamics

- arXiv: http://arxiv.org/abs/2603.22886v1 | 2026-03-24 | final 64.0

We propose the Identifiable Variational Dynamic Factor Model (iVDFM), which learns latent factors from multivariate time series with identifiability guarantees. By applying iVAE-style conditioning to the innovation process driving the dynamics rather than to the latent states, we show that factors are identifiable up to permutation and component-wise affine (or monotone invertible) transformations.

-> Time series analysis could be applicable to sports performance data but not directly related to video analysis

## 39위: Stable Inversion of Discrete-Time Linear Periodically Time-Varying Systems via Cyclic Reformulation

- arXiv: http://arxiv.org/abs/2603.23147v1 | 2026-03-24 | final 64.0

Stable inverse systems for periodically time-varying plants are essential for feedforward control and iterative learning control of multirate and periodic systems, yet existing approaches either require complex-valued Floquet factors and noncausal processing or operate on a block time scale via lifting. This paper proposes a systematic method for constructing stable inverse systems for discrete-time linear periodically time-varying (LPTV) systems that avoids these limitations.

-> Control systems could be tangentially related to movement analysis but not specifically for sports video

## 40위: Spatial navigation in preclinical Alzheimer's disease: A review

- arXiv: http://arxiv.org/abs/2603.23082v1 | 2026-03-24 | final 64.0

Alzheimer's disease (AD) develops over a prolonged preclinical phase, during which neuropathological changes accumulate long before cognitive symptoms appear. Identifying cognitive functions affected at early stages is critical for the preclinical detection of asymptomatic individuals at-risk of AD.

-> Spatial navigation analysis could be applicable to sports but focused on medical diagnosis

## 41위: Good for the Planet, Bad for Me? Intended and Unintended Consequences of AI Energy Consumption Disclosure

- arXiv: http://arxiv.org/abs/2603.23075v1 | 2026-03-24 | final 64.0

To address the high energy consumption of artificial intelligence, energy consumption disclosure (ECD) has been proposed to steer users toward more sustainable practices, such as choosing efficient small language models (SLMs) over large language models (LLMs). This presents a performance-sustainability trade-off for users.

-> Discusses energy efficiency considerations directly relevant to edge devices

## 42위: L-UNet: An LSTM Network for Remote Sensing Image Change Detection

- arXiv: http://arxiv.org/abs/2603.22842v1 | 2026-03-24 | final 60.0

Change detection of high-resolution remote sensing images is an important task in earth observation and was extensively investigated. Recently, deep learning has shown to be very successful in plenty of remote sensing tasks.

-> 원격 감지 이미지 변화 검색에 대한 LSTM 네트워크 사용으로 스포츠 영상 분석과 간접적으로만 관련

## 43위: GSwap: Realistic Head Swapping with Dynamic Neural Gaussian Field

- arXiv: http://arxiv.org/abs/2603.23168v1 | 2026-03-24 | final 60.0

We present GSwap, a novel consistent and realistic video head-swapping system empowered by dynamic neural Gaussian portrait priors, which significantly advances the state of the art in face and head replacement. Unlike previous methods that rely primarily on 2D generative models or 3D Morphable Face Models (3DMM), our approach overcomes their inherent limitations, including poor 3D consistency, unnatural facial expressions, and restricted synthesis quality.

-> 동적 신경 가우시안 필드를 이용한 헤드 스와핑 기술로 스포츠 영상 보정과 간접적으로만 관련

## 44위: Vision-based Deep Learning Analysis of Unordered Biomedical Tabular Datasets via Optimal Spatial Cartography

- arXiv: http://arxiv.org/abs/2603.22675v1 | 2026-03-24 | final 60.0

Tabular data are central to biomedical research, from liquid biopsy and bulk and single-cell transcriptomics to electronic health records and phenotypic profiling. Unlike images or sequences, however, tabular datasets lack intrinsic spatial organization: features are treated as unordered dimensions, and their relationships must be inferred implicitly by the model.

-> 비정형 생물 의학 데이터 분석을 위한 비전 기반 딥러닝 기술로 스포츠 영상 분석과 간접적으로만 관련

## 45위: Is AI Catching Up to Human Expression? Exploring Emotion, Personality, Authorship, and Linguistic Style in English and Arabic with Six Large Language Models

- arXiv: http://arxiv.org/abs/2603.23251v1 | 2026-03-24 | final 56.0

The advancing fluency of LLMs raises important questions about their ability to emulate complex human traits, including emotional expression and personality, across diverse linguistic and cultural contexts. This study investigates whether LLMs can convincingly mimic emotional nuance in English and personality markers in Arabic, a critical under-resourced language with unique linguistic and cultural characteristics.

-> Investigates AI capabilities for text analysis, weakly related to sports video analysis

## 46위: Typography-Based Monocular Distance Estimation Framework for Vehicle Safety Systems

- arXiv: http://arxiv.org/abs/2603.22781v1 | 2026-03-24 | final 52.0

Accurate inter-vehicle distance estimation is a cornerstone of advanced driver assistance systems and autonomous driving. While LiDAR and radar provide high precision, their cost prohibits widespread adoption in mass-market vehicles.

-> Distance estimation techniques could be useful for sports analysis but not a core component

## 47위: PinPoint: Monocular Needle Pose Estimation for Robotic Suturing via Stein Variational Newton and Geometric Residuals

- arXiv: http://arxiv.org/abs/2603.23365v1 | 2026-03-24 | final 52.0

Reliable estimation of surgical needle 3D position and orientation is essential for autonomous robotic suturing, yet existing methods operate almost exclusively under stereoscopic vision. In monocular endoscopic settings, common in transendoscopic and intraluminal procedures, depth ambiguity and rotational symmetry render needle pose estimation inherently ill-posed, producing a multimodal distribution over feasible configurations, rather than a single, well-grounded estimate.

-> Surgical needle pose estimation not directly applicable to sports movement analysis

## 48위: LiZIP: An Auto-Regressive Compression Framework for LiDAR Point Clouds

- arXiv: http://arxiv.org/abs/2603.23162v1 | 2026-03-24 | final 52.0

The massive volume of data generated by LiDAR sensors in autonomous vehicles creates a bottleneck for real-time processing and vehicle-to-everything (V2X) transmission. Existing lossless compression methods often force a trade-off: industry standard algorithms (e.g., LASzip) lack adaptability, while deep learning approaches suffer from prohibitive computational costs.

-> LiDAR compression not directly applicable to sports video processing

## 49위: Spiking Personalized Federated Learning for Brain-Computer Interface-Enabled Immersive Communication

- arXiv: http://arxiv.org/abs/2603.22727v1 | 2026-03-24 | final 52.0

This work proposes a novel immersive communication framework that leverages brain-computer interface (BCI) to acquire brain signals for inferring user-centric states (e.g., intention and perception-related discomfort), thereby enabling more personalized and robust immersive adaptation under strong individual variability. Specifically, we develop a personalized federated learning (PFL) model to analyze and process the collected brain signals, which not only accommodates neurodiverse brain-signal data but also prevents the leakage of sensitive brain-signal information.

-> Focuses on brain-computer interfaces and immersive communication, with only tangential relevance to edge device efficiency

## 50위: Fault-Tolerant Design and Multi-Objective Model Checking for Real-Time Deep Reinforcement Learning Systems

- arXiv: http://arxiv.org/abs/2603.23113v1 | 2026-03-24 | final 52.0

Deep reinforcement learning (DRL) has emerged as a powerful paradigm for solving complex decision-making problems. However, DRL-based systems still face significant dependability challenges particularly in real-time environments due to the simulation-to-reality gap, out-of-distribution observations, and the critical impact of latency.

-> Addresses fault tolerance in real-time DRL systems, which could be tangentially relevant if project uses reinforcement learning

## 51위: FDIF: Formula-Driven supervised Learning with Implicit Functions for 3D Medical Image Segmentation

- arXiv: http://arxiv.org/abs/2603.23199v1 | 2026-03-24 | final 50.4

Deep learning-based 3D medical image segmentation methods relies on large-scale labeled datasets, yet acquiring such data is difficult due to privacy constraints and the high cost of expert annotation. Formula-Driven Supervised Learning (FDSL) offers an appealing alternative by generating training data and labels directly from mathematical formulas.

-> Medical image segmentation not directly applicable to sports video analysis

---

## 다시 보기

### HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18850v1
- 점수: final 100.0

Video question answering (VQA) with vision-language models (VLMs) depends critically on which frames are selected from the input video, yet most systems rely on uniform or heuristic sampling that cannot be optimized for downstream answering quality. We introduce \textbf{HORNet}, a lightweight frame selection policy trained with Group Relative Policy Optimization (GRPO) to learn which frames a frozen VLM needs to answer questions correctly. With fewer than 1M trainable parameters, HORNet reduces input frames by up to 99\% and VLM processing time by up to 93\%, while improving answer quality on short-form benchmarks (+1.7\% F1 on MSVD-QA) and achieving strong performance on temporal reasoning tasks (+7.3 points over uniform sampling on NExT-QA). We formalize this as Select Any Frames (SAF), a task that decouples visual input curation from VLM reasoning, and show that GRPO-trained selection generalizes better out-of-distribution than supervised and PPO alternatives. HORNet's policy further transfers across VLM answerers without retraining, yielding an additional 8.5\% relative gain when paired with a stronger model. Evaluated across six benchmarks spanning 341,877 QA pairs and 114.2 hours of video, our results demonstrate that optimizing \emph{what} a VLM sees is a practical and complementary alternative to optimizing what it generates while improving efficiency. Code is available at https://github.com/ostadabbas/HORNet.

-> 경량 프레임 선택 기술은 스포츠 영상 하이라이트 생성에 직접적으로 적용 가능합니다

### DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.19172v1
- 점수: final 100.0

Despite the computational efficiency of MoE models, the excessive memory footprint and I/O overhead inherent in multi-expert architectures pose formidable challenges for real-time inference on resource-constrained edge platforms. While existing static methods struggle with a rigid latency-accuracy trade-off, we observe that expert importance is highly skewed and depth-dependent. Motivated by these insights, we propose DyMoE, a dynamic mixed-precision quantization framework designed for high-performance edge inference. Leveraging insights into expert importance skewness and depth-dependent sensitivity, DyMoE introduces: (1) importance-aware prioritization to dynamically quantize experts at runtime; (2) depth-adaptive scheduling to preserve semantic integrity in critical layers; and (3) look-ahead prefetching to overlap I/O stalls. Experimental results on commercial edge hardware show that DyMoE reduces Time-to-First-Token (TTFT) by 3.44x-22.7x and up to a 14.58x speedup in Time-Per-Output-Token (TPOT) compared to state-of-the-art offloading baselines, enabling real-time, accuracy-preserving MoE inference on resource-constrained edge devices.

-> 엣지 디바이스 효율적인 추론 기술이 프로젝트의 엣지 디바이스 구현에 직접적으로 관련됨

### Convolutions Predictable Offloading to an Accelerator: Formalization and Optimization (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21792v1
- 점수: final 98.4

Convolutional neural networks (CNNs) require a large number of multiply-accumulate (MAC) operations. To meet real-time constraints, they often need to be executed on specialized accelerators composed of an on-chip memory and a processing unit. However, the on-chip memory is often insufficient to store all the data required to compute a CNN layer. Thus, the computation must be performed in several offloading steps. We formalise such sequences of steps and apply our formalism to a state of the art decomposition of convolutions. In order to find optimal strategies in terms of duration, we encode the problem with a set of constraints. A Python-based simulator allows to analyse in-depth computed strategies.

-> rk3588 엣지 디바이스에서 AI 모델 실행 최적화에 직접적으로 관련된 연구로 실시간 스포츠 촬영 및 분석 성능 향상에 필수적임

### Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.19026v1
- 점수: final 98.4

Recent segmentation methods leveraging Multi-modal Large Language Models (MLLMs) have shown reliable object-level segmentation and enhanced spatial perception. However, almost all previous methods predominantly rely on specialist mask decoders to interpret masks from generated segmentation-related embeddings and visual features, or incorporate multiple additional tokens to assist. This paper aims to investigate whether and how we can unlock segmentation from MLLM itSELF with 1 segmentation Embedding (SELF1E) while achieving competitive results, which eliminates the need for external decoders. To this end, our approach targets the fundamental limitation of resolution reduction in pixel-shuffled image features from MLLMs. First, we retain image features at their original uncompressed resolution, and refill them with residual features extracted from MLLM-processed compressed features, thereby improving feature precision. Subsequently, we integrate pixel-unshuffle operations on image features with and without LLM processing, respectively, to unleash the details of compressed features and amplify the residual features under uncompressed resolution, which further enhances the resolution of refilled features. Moreover, we redesign the attention mask with dual perception pathways, i.e., image-to-image and image-to-segmentation, enabling rich feature interaction between pixels and the segmentation token. Comprehensive experiments across multiple segmentation tasks validate that SELF1E achieves performance competitive with specialist mask decoder-based methods, demonstrating the feasibility of decoder-free segmentation in MLLMs. Project page: https://github.com/ANDYZAQ/SELF1E.

-> 스포츠 영상에서 선수와 객체 식별에 직접적으로 적용 가능한 고급 분할 기술

### EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18739v1
- 점수: final 98.4

Deploying high-performance dense prediction models on resource-constrained edge devices remains challenging due to strict limits on computation and memory. In practice, lightweight systems for object detection, instance segmentation, and pose estimation are still dominated by CNN-based architectures such as YOLO, while compact Vision Transformers (ViTs) often struggle to achieve similarly strong accuracy efficiency tradeoff, even with large scale pretraining. We argue that this gap is largely due to insufficient task specific representation learning in small scale ViTs, rather than an inherent mismatch between ViTs and edge dense prediction. To address this issue, we introduce EdgeCrafter, a unified compact ViT framework for edge dense prediction centered on ECDet, a detection model built from a distilled compact backbone and an edge-friendly encoder decoder design. On the COCO dataset, ECDet-S achieves 51.7 AP with fewer than 10M parameters using only COCO annotations. For instance segmentation, ECInsSeg achieves performance comparable to RF-DETR while using substantially fewer parameters. For pose estimation, ECPose-X reaches 74.8 AP, significantly outperforming YOLO26Pose-X (71.6 AP) despite the latter's reliance on extensive Objects365 pretraining. These results show that compact ViTs, when paired with task-specialized distillation and edge-aware design, can be a practical and competitive option for edge dense prediction. Code is available at: https://intellindust-ai-lab.github.io/projects/EdgeCrafter/

-> 엣지 기기에서의 포즈 추정 및 객체 탐지 기술이 스포츠 동작 분석에 직접 적용 가능함

### Rateless DeepJSCC for Broadcast Channels: a Rate-Distortion-Complexity Tradeoff (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21616v1
- 점수: final 96.0

In recent years, numerous data-intensive broadcasting applications have emerged at the wireless edge, calling for a flexible tradeoff between distortion, transmission rate, and processing complexity. While deep learning-based joint source-channel coding (DeepJSCC) has been identified as a potential solution to data-intensive communications, most of these schemes are confined to worst-case solutions, lack adaptive complexity, and are inefficient in broadcast settings. To overcome these limitations, this paper introduces nonlinear transform rateless source-channel coding (NTRSCC), a variable-length JSCC framework for broadcast channels based on rateless codes. In particular, we integrate learned source transformations with physical-layer LT codes, develop unequal protection schemes that exploit decoder side information, and devise approximations to enable end-to-end optimization of rateless parameters. Our framework enables heterogeneous receivers to adaptively adjust their received number of rateless symbols and decoding iterations in belief propagation, thereby achieving a controllable tradeoff between distortion, rate, and decoding complexity. Simulation results demonstrate that the proposed method enhances image broadcast quality under stringent communication and processing budgets over heterogeneous edge devices.

-> Rateless DeepJSCC framework for broadcast channels could enable efficient streaming of sports content to heterogeneous edge devices.

### No Dense Tensors Needed: Fully Sparse Object Detection on Event-Camera Voxel Grids (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21638v1
- 점수: final 96.0

Event cameras produce asynchronous, high-dynamic-range streams well suited for detecting small, fast-moving drones, yet most event-based detectors convert the sparse event stream into dense tensors, discarding the representational efficiency of neuromorphic sensing. We propose SparseVoxelDet, to our knowledge the first fully sparse object detector for event cameras, in which backbone feature extraction, feature pyramid fusion, and the detection head all operate exclusively on occupied voxel positions through 3D sparse convolutions; no dense feature tensor is instantiated at any stage of the pipeline. On the FRED benchmark (629,832 annotated frames), SparseVoxelDet achieves 83.38% mAP at 50 while processing only 14,900 active voxels per frame (0.23% of the T.H.W grid), compared to 409,600 pixels for the dense YOLOv11 baseline (87.68% mAP at 50). Relaxing the IoU threshold from 0.50 to 0.40 recovers mAP to 89.26%, indicating that the remaining accuracy gap is dominated by box regression precision rather than detection capability. The sparse representation yields 858 times GPU memory compression and 3,670 times storage reduction relative to the equivalent dense 3D voxel tensor, with data-structure size that scales with scene dynamics rather than sensor resolution. Error forensics across 119,459 test frames confirms that 71 percent of failures are localization near-misses rather than missed targets. These results demonstrate that native sparse processing is a viable paradigm for event-camera object detection, exploiting the structural sparsity of neuromorphic sensor data without requiring neuromorphic computing hardware, and providing a framework whose representation cost is governed by scene activity rather than pixel count, a property that becomes increasingly valuable as event cameras scale to higher resolutions.

-> 이벤트 카메라를 이용한 효율적인 희소 객체 탐지 기술로 엣지 디바이스에 직접 적용 가능

### Balancing Performance and Fairness in Explainable AI for Anomaly Detection in Distributed Power Plants Monitoring (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18954v1
- 점수: final 96.0

Reliable anomaly detection in distributed power plant monitoring systems is essential for ensuring operational continuity and reducing maintenance costs, particularly in regions where telecom operators heavily rely on diesel generators. However, this task is challenged by extreme class imbalance, lack of interpretability, and potential fairness issues across regional clusters. In this work, we propose a supervised ML framework that integrates ensemble methods (LightGBM, XGBoost, Random Forest, CatBoost, GBDT, AdaBoost) and baseline models (Support Vector Machine, K-Nearrest Neighbors, Multilayer Perceptrons, and Logistic Regression) with advanced resampling techniques (SMOTE with Tomek Links and ENN) to address imbalance in a dataset of diesel generator operations in Cameroon. Interpretability is achieved through SHAP (SHapley Additive exPlanations), while fairness is quantified using the Disparate Impact Ratio (DIR) across operational clusters. We further evaluate model generalization using Maximum Mean Discrepancy (MMD) to capture domain shifts between regions. Experimental results show that ensemble models consistently outperform baselines, with LightGBM achieving an F1-score of 0.99 and minimal bias across clusters (DIR $\approx 0.95$). SHAP analysis highlights fuel consumption rate and runtime per day as dominant predictors, providing actionable insights for operators. Our findings demonstrate that it is possible to balance performance, interpretability, and fairness in anomaly detection, paving the way for more equitable and explainable AI systems in industrial power management. {\color{black} Finally, beyond offline evaluation, we also discuss how the trained models can be deployed in practice for real-time monitoring. We show how containerized services can process in real-time, deliver low-latency predictions, and provide interpretable outputs for operators.

-> 실시간 이상 감지 프레임워크로 엣지 배포 가능성이 있어 스포츠 모니터링 시스템에 적용 가능

### Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18671v1
- 점수: final 94.4

Standard deep learning models for image segmentation cannot guarantee topology accuracy, failing to preserve the correct number of connected components or structures. This, in turn, affects the quality of the segmentations and compromises the reliability of the subsequent quantification analyses. Previous works have proposed to enhance topology accuracy with specialized frameworks, architectures, and loss functions. However, these methods are often cumbersome to integrate into existing training pipelines, they are computationally very expensive, or they are restricted to structures with tubular morphology. We present SCNP, an efficient method that improves topology accuracy by penalizing the logits with their poorest-classified neighbor, forcing the model to improve the prediction at the pixels' neighbors before allowing it to improve the pixels themselves. We show the effectiveness of SCNP across 13 datasets, covering different structure morphologies and image modalities, and integrate it into three frameworks for semantic and instance segmentation. Additionally, we show that SCNP can be integrated into several loss functions, making them improve topology accuracy. Our code can be found at https://jmlipman.github.io/SCNP-SameClassNeighborPenalization.

-> 토폴로지 정확도를 개선하는 이미지 분할 방법은 스포츠 분석에서 선수와 객체 구조를 정확하게 유지하는 데 중요합니다.

### Scaling Sim-to-Real Reinforcement Learning for Robot VLAs with Generative 3D Worlds (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18532v1
- 점수: final 93.6

The strong performance of large vision-language models (VLMs) trained with reinforcement learning (RL) has motivated similar approaches for fine-tuning vision-language-action (VLA) models in robotics. Many recent works fine-tune VLAs directly in the real world to avoid addressing the sim-to-real gap. While real-world RL circumvents sim-to-real issues, it inherently limits the generality of the resulting VLA, as scaling scene and object diversity in the physical world is prohibitively difficult. This leads to the paradoxical outcome of transforming a broadly pretrained model into an overfitted, scene-specific policy. Training in simulation can instead provide access to diverse scenes, but designing those scenes is also costly. In this work, we show that VLAs can be RL fine-tuned without sacrificing generality and with reduced labor by leveraging 3D world generative models. Using these models together with a language-driven scene designer, we generate hundreds of diverse interactive scenes containing unique objects and backgrounds, enabling scalable and highly parallel policy learning. Starting from a pretrained imitation baseline, our approach increases simulation success from 9.7% to 79.8% while achieving a 1.25$\times$ speedup in task completion time. We further demonstrate successful sim-to-real transfer enabled by the quality of the generated digital twins together with domain randomization, improving real-world success from 21.7% to 75% and achieving a 1.13$\times$ speedup. Finally, we further highlight the benefits of leveraging the effectively unlimited data from 3D world generative models through an ablation study showing that increasing scene diversity directly improves zero-shot generalization.

-> 비전-언어-액션 모델과 생성적 3D 세계는 스포츠 장면 분석과 시뮬레이션에 적용 가능하여 우리 프로젝트의 핵심 기술이 될 수 있습니다.

### A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.19133v1
- 점수: final 93.6

Recent advancements and widespread adoption of Large Language Models (LLMs) in both industry and academia have catalyzed significant demand for LLM serving. However, traditional cloud services incur high costs, while on-device inference alone faces challenges due to limited resources. Edge-cloud collaboration emerges as a key research direction to combine the strengths of both paradigms, yet efficiently utilizing limited network bandwidth while fully leveraging and balancing the computational capabilities of edge devices and the cloud remains an open problem. To address these challenges, we propose Pipelined Collaborative Speculative Decoding Framework (PicoSpec), a novel, general-purpose, and training-free speculative decoding framework for LLM edge-cloud collaborative inference. We design an asynchronous pipeline that resolves the mutual waiting problem inherent in vanilla speculative decoding within edge collaboration scenarios, which concurrently executes a Small Language Model (SLM) on the edge device and a LLM in the cloud. Meanwhile, to mitigate the significant communication latency caused by transmitting vocabulary distributions, we introduce separate rejection sampling with sparse compression, which completes the rejection sampling with only a one-time cost of transmitting the compressed vocabulary. Experimental results demonstrate that our solution outperforms baseline and existing methods, achieving up to 2.9 speedup.

-> 엣지-클라우드 협력 프레임워크는 rk3588 기반 엣지 디바이스에서 실시간 스포츠 영상 처리에 필수적입니다.

### ANCHOR: Adaptive Network based on Cascaded Harmonic Offset Routing (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21718v1
- 점수: final 93.6

Time series analysis plays a foundational role in a wide range of real-world applications, yet accurately modeling complex non-stationary signals remains a shared challenge across downstream tasks. Existing methods attempt to extract features directly from one-dimensional sequences, making it difficult to handle the widely observed dynamic phase drift and discrete quantization error. To address this issue, we decouple temporal evolution into macroscopic physical periods and microscopic phase perturbations, and inject frequency-domain priors derived from the Real Fast Fourier Transform (RFFT) into the underlying spatial sampling process. Based on this idea, we propose a Frequency-Guided Deformable Module (FGDM) to adaptively compensate for microscopic phase deviations. Built upon FGDM, we further develop an Adaptive Network based on Cascaded Harmonic Offset Routing (ANCHOR) as a general-purpose backbone for time-series modeling. Through orthogonal channel partitioning and a progressive residual architecture, ANCHOR efficiently decouples multi-scale harmonic features while substantially suppressing the computational redundancy of multi-branch networks. Extensive experiments demonstrate that ANCHOR achieves the best performance in most short-term forecasting sub-tasks and exhibits strong competitiveness on several specific sub-tasks in anomaly detection and time-series classification, validating its effectiveness as a universal time-series foundation backbone.

-> 시계열 분석 기술로 스포츠 동작 분석에 적용 가능

### R&D: Balancing Reliability and Diversity in Synthetic Data Augmentation for Semantic Segmentation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18427v1
- 점수: final 92.0

Collecting and annotating datasets for pixel-level semantic segmentation tasks are highly labor-intensive. Data augmentation provides a viable solution by enhancing model generalization without additional real-world data collection. Traditional augmentation techniques, such as translation, scaling, and color transformations, create geometric variations but fail to generate new structures. While generative models have been employed to extend semantic information of datasets, they often struggle to maintain consistency between the original and generated images, particularly for pixel-level tasks. In this work, we propose a novel synthetic data augmentation pipeline that integrates controllable diffusion models. Our approach balances diversity and reliability data, effectively bridging the gap between synthetic and real data. We utilize class-aware prompting and visual prior blending to improve image quality further, ensuring precise alignment with segmentation labels. By evaluating benchmark datasets such as PASCAL VOC and BDD100K, we demonstrate that our method significantly enhances semantic segmentation performance, especially in data-scarce scenarios, while improving model robustness in real-world applications. Our code is available at \href{https://github.com/chequanghuy/Enhanced-Generative-Data-Augmentation-for-Semantic-Segmentation-via-Stronger-Guidance}{https://github.com/chequanghuy/Enhanced-Generative-Data-Augmentation-for-Semantic-Segmentation-via-Stronger-Guidance}.

-> 합성 데이터 증강 기술은 스포츠 장면 분석을 위한 다양한 학습 데이터 생성에 효과적입니다.

### StreamingClaw Technical Report (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.22120v1
- 점수: final 92.0

Applications such as embodied intelligence rely on a real-time perception-decision-action closed loop, posing stringent challenges for streaming video understanding. However, current agents suffer from fragmented capabilities, such as supporting only offline video understanding, lacking long-term multimodal memory mechanisms, or struggling to achieve real-time reasoning and proactive interaction under streaming inputs. These shortcomings have become a key bottleneck for preventing them from sustaining perception, making real-time decisions, and executing actions in real-world environments. To alleviate these issues, we propose StreamingClaw, a unified agent framework for streaming video understanding and embodied intelligence. It is also an OpenClaw-compatible framework that supports real-time, multimodal streaming interaction. StreamingClaw integrates five core capabilities: (1) It supports real-time streaming reasoning. (2) It supports reasoning about future events and proactive interaction under the online evolution of interaction objectives. (3) It supports multimodal long-term storage, hierarchical evolution, and efficient retrieval of shared memory across multiple agents. (4) It supports a closed-loop of perception-decision-action. In addition to conventional tools and skills, it also provides streaming tools and action-centric skills tailored for real-world physical environments. (5) It is compatible with the OpenClaw framework, allowing it to fully leverage the resources and support of the open-source community. With these designs, StreamingClaw integrates online real-time reasoning, multimodal long-term memory, and proactive interaction within a unified framework. Moreover, by translating decisions into executable actions, it enables direct control of the physical world, supporting practical deployment of embodied interaction.

-> 실시간 영상 이해 및 의사결정-행동 루프 기술이 스포츠 자동 촬영에 적용 가능

### Not All Layers Are Created Equal: Adaptive LoRA Ranks for Personalized Image Generation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21884v1
- 점수: final 92.0

Low Rank Adaptation (LoRA) is the de facto fine-tuning strategy to generate personalized images from pre-trained diffusion models. Choosing a good rank is extremely critical, since it trades off performance and memory consumption, but today the decision is often left to the community's consensus, regardless of the personalized subject's complexity. The reason is evident: the cost of selecting a good rank for each LoRA component is combinatorial, so we opt for practical shortcuts such as fixing the same rank for all components. In this paper, we take a first step to overcome this challenge. Inspired by variational methods that learn an adaptive width of neural networks, we let the ranks of each layer freely adapt during fine-tuning on a subject. We achieve it by imposing an ordering of importance on the rank's positions, effectively encouraging the creation of higher ranks when strictly needed. Qualitatively and quantitatively, our approach, LoRA$^2$, achieves a competitive trade-off between DINO, CLIP-I, and CLIP-T across 29 subjects while requiring much less memory and lower rank than high rank LoRA versions. Code: https://github.com/donaldssh/NotAllLayersAreCreatedEqual.

-> 개인화된 이미지 생성을 위한 적응형 LoRA 기술은 스포츠 영상을 맞춤형 하이라이트 시각물로 변환하는 데 사용될 수 있습니다.

### Benchmarking Message Brokers for IoT Edge Computing: A Comprehensive Performance Study (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21600v1
- 점수: final 92.0

Asynchronous messaging is a cornerstone of modern distributed systems, enabling decoupled communication for scalable and resilient applications. Today's message queue (MQ) ecosystem spans a wide range of designs, from high-throughput streaming platforms to lightweight protocols tailored for edge and IoT environments. Despite this diversity, choosing an appropriate MQ system remains difficult. Existing evaluations largely focus on throughput and latency on fixed hardware, while overlooking CPU and memory footprint and the effects of resource constraints, factors that are critical for edge and IoT deployments. In this paper, we present a systematic performance study of eight prominent message brokers: Mosquitto, EMQX, HiveMQ, RabbitMQ, ActiveMQ Artemis, NATS Server, Redis (Pub/Sub), and Zenoh Router. We introduce mq-bench, a unified benchmarking framework to evaluate these systems under identical conditions, scaling up to 10,000 concurrent client pairs across three VM configurations representative of edge hardware. This study reveals several interesting and sometimes counter-intuitive insights. Lightweight native brokers achieve sub-millisecond latency, while feature-rich enterprise platforms incur 2-3X higher overhead. Under high connection loads, multi-threaded brokers like NATS and Zenoh scale efficiently, whereas the widely-deployed Mosquitto saturates earlier due to its single-threaded architecture. We also find that Java-based brokers consume significantly more memory than native implementations, which has important implications for memory-constrained edge deployments. Based on these findings, we provide practical deployment guidelines that map workload requirements and resource constraints to appropriate broker choices for telemetry, streaming analytics, and IoT use cases.

-> IoT 엣지 컴퓨팅용 메시지 브로커 벤치마킹은 AI 카메라 디바이스의 내부 통신 시스템 설계에 중요합니다.

### SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18774v1
- 점수: final 90.4

Foundational feed-forward visual geometry models enable accurate and efficient camera pose estimation and scene reconstruction by learning strong scene priors from massive RGB datasets. However, their effectiveness drops when applied to mixed sensing modalities, such as RGB-thermal (RGB-T) images. We observe that while a visual geometry grounded transformer pretrained on RGB data generalizes well to thermal-only reconstruction, it struggles to align RGB and thermal modalities when processed jointly. To address this, we propose SEAR, a simple yet efficient fine-tuning strategy that adapts a pretrained geometry transformer to multimodal RGB-T inputs. Despite being trained on a relatively small RGB-T dataset, our approach significantly outperforms state-of-the-art methods for 3D reconstruction and camera pose estimation, achieving significant improvements over all metrics (e.g., over 29\% in AUC@30) and delivering higher detail and consistency between modalities with negligible overhead in inference time compared to the original pretrained model. Notably, SEAR enables reliable multimodal pose estimation and reconstruction even under challenging conditions, such as low lighting and dense smoke. We validate our architecture through extensive ablation studies, demonstrating how the model aligns both modalities. Additionally, we introduce a new dataset featuring RGB and thermal sequences captured at different times, viewpoints, and illumination conditions, providing a robust benchmark for future work in multimodal 3D scene reconstruction. Code and models are publicly available at https://www.github.com/Schindler-EPFL-Lab/SEAR.

-> 다중 모달 3D 복원 기술은 스포츠 장면 분석에 적용 가능할 수 있습니다

### Translating MRI to PET through Conditional Diffusion Models with Enhanced Pathology Awareness (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18896v1
- 점수: final 90.4

Positron emission tomography (PET) is a widely recognized technique for diagnosing neurodegenerative diseases, offering critical functional insights. However, its high costs and radiation exposure hinder its widespread use. In contrast, magnetic resonance imaging (MRI) does not involve such limitations. While MRI also detects neurodegenerative changes, it is less sensitive for diagnosis compared to PET. To overcome such limitations, one approach is to generate synthetic PET from MRI. Recent advances in generative models have paved the way for cross-modality medical image translation; however, existing methods largely emphasize structural preservation while neglecting the critical need for pathology awareness. To address this gap, we propose PASTA, a novel image translation framework built on conditional diffusion models with enhanced pathology awareness. PASTA surpasses state-of-the-art methods by preserving both structural and pathological details through its highly interactive dual-arm architecture and multi-modal condition integration. Additionally, we introduce a novel cycle exchange consistency and volumetric generation strategy that significantly enhances PASTA's ability to produce high-quality 3D PET images. Our qualitative and quantitative results demonstrate the high quality and pathology awareness of the synthesized PET scans. For Alzheimer's diagnosis, the performance of these synthesized scans improves over MRI by 4%, almost reaching the performance of actual PET. Our code is available at https://github.com/ai-med/PASTA.

-> 조건적 확산 모델을 활용한 이미지 변환 및 향상 기술이 프로젝트의 영상/이미지 보정 기능에 직접 적용 가능하며, 코드가 공개되어 있음

### The Universal Normal Embedding (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21786v1
- 점수: final 90.4

Generative models and vision encoders have largely advanced on separate tracks, optimized for different goals and grounded in different mathematical principles. Yet, they share a fundamental property: latent space Gaussianity. Generative models map Gaussian noise to images, while encoders map images to semantic embeddings whose coordinates empirically behave as Gaussian. We hypothesize that both are views of a shared latent source, the Universal Normal Embedding (UNE): an approximately Gaussian latent space from which encoder embeddings and DDIM-inverted noise arise as noisy linear projections. To test our hypothesis, we introduce NoiseZoo, a dataset of per-image latents comprising DDIM-inverted diffusion noise and matching encoder representations (CLIP, DINO). On CelebA, linear probes in both spaces yield strong, aligned attribute predictions, indicating that generative noise encodes meaningful semantics along linear directions. These directions further enable faithful, controllable edits (e.g., smile, gender, age) without architectural changes, where simple orthogonalization mitigates spurious entanglements. Taken together, our results provide empirical support for the UNE hypothesis and reveal a shared Gaussian-like latent geometry that concretely links encoding and generation. Code and data are available https://rbetser.github.io/UNE/

-> 유니버설 노말 임베딩 프레임워크는 스포츠 영상 처리 및 분석에 직접적으로 적용 가능하며, 생성 모델과 인코더 간의 잠재 공간 연결을 통해 영상 보정과 하이라이트 생성을 향상시킬 수 있습니다.

### Optimal Memory Encoding Through Fluctuation-Response Structure (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21666v1
- 점수: final 89.6

Physical reservoir computing exploits the intrinsic dynamics of physical systems for information processing, while keeping the internal dynamics fixed and training only linear readouts; yet the role of input encoding remains poorly understood. We show that optimal input encoding is a geometric problem governed by the system's fluctuation-response structure. By measuring steady-state fluctuations and linear response, we derive an analytical criterion for the input direction that maximizes task-specific linear memory under a fixed power constraint, termed Response-based Optimal Memory Encoding (ROME). Backpropagation-based encoder optimization is shown to be equivalent to ROME, revealing a trade-off between task-dependent feature mixing and intrinsic noise. We apply ROME to various reservoir platforms, including spin-wave waveguides and spiking neural networks, demonstrating effective encoder design across physical and neuromorphic reservoirs, even in non-differentiable systems.

-> 물리적 리저버 컴퓨팅과 스파이킹 신경망은 엣지 디바이스의 영상 처리 효율성을 향상시키는 데 적용 가능합니다.

### Feature Incremental Clustering with Generalization Bounds (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21590v1
- 점수: final 89.6

In many learning systems, such as activity recognition systems, as new data collection methods continue to emerge in various dynamic environmental applications, the attributes of instances accumulate incrementally, with data being stored in gradually expanding feature spaces. How to design theoretically guaranteed algorithms to effectively cluster this special type of data stream, commonly referred to as activity recognition, remains unexplored. Compared to traditional scenarios, we will face at least two fundamental questions in this feature incremental scenario. (i) How to design preliminary and effective algorithms to address the feature incremental clustering problem? (ii) How to analyze the generalization bounds for the proposed algorithms and under what conditions do these algorithms provide a strong generalization guarantee? To address these problems, by tailoring the most common clustering algorithm, i.e., $k$-means, as an example, we propose four types of Feature Incremental Clustering (FIC) algorithms corresponding to different situations of data access: Feature Tailoring (FT), Data Reconstruction (DR), Data Adaptation (DA), and Model Reuse (MR), abbreviated as FIC-FT, FIC-DR, FIC-DA, and FIC-MR. Subsequently, we offer a detailed analysis of the generalization error bounds for these four algorithms and highlight the critical factors influencing these bounds, such as the amounts of training data, the complexity of the hypothesis space, the quality of pre-trained models, and the discrepancy of the reconstruction feature distribution. The numerical experiments show the effectiveness of the proposed algorithms, particularly in their application to activity recognition clustering tasks.

-> 특성 증분 클러스터링 알고리즘은 스포츠 동작 분석과 하이라이트 감지에 직접 적용 가능합니다.

### CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18991v1
- 점수: final 89.6

Aligning Diffusion models has achieved remarkable breakthroughs in generating high-quality, human preference-aligned images. Existing techniques, such as supervised fine-tuning (SFT) and DPO-style preference optimization, have become principled tools for fine-tuning diffusion models. However, SFT relies on high-quality images that are costly to obtain, while DPO-style methods depend on large-scale preference datasets, which are often inconsistent in quality. Beyond data dependency, these methods are further constrained by computational inefficiency. To address these two challenges, we propose Composite Reward Assisted Fine-Tuning (CRAFT), a lightweight yet powerful fine-tuning paradigm that requires significantly reduced training data while maintaining computational efficiency. It first leverages a Composite Reward Filtering (CRF) technique to construct a high-quality and consistent training dataset and then perform an enhanced variant of SFT. We also theoretically prove that CRAFT actually optimizes the lower bound of group-based reinforcement learning, establishing a principled connection between SFT with selected data and reinforcement learning. Our extensive empirical results demonstrate that CRAFT with only 100 samples can easily outperform recent SOTA preference optimization methods with thousands of preference-paired samples. Moreover, CRAFT can even achieve 11-220$\times$ faster convergences than the baseline preference optimization methods, highlighting its extremely high efficiency.

-> 경량 확산 모델 미세조정 기술이 엣지 디바이스에서 스포츠 이미지 생성 및 보정에 적용 가능

### ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.19157v1
- 점수: final 89.6

Generating rare compositional concepts in text-to-image synthesis remains a challenge for diffusion models, particularly for attributes that are uncommon in the training data. While recent approaches, such as R2F, address this challenge by utilizing LLM for prompt scheduling, they suffer from inherent variance due to the randomness of language models and suboptimal guidance from iterative text embedding switching. To address these problems, we propose the ADAPT framework, a training-free framework that deterministically plans and semantically aligns prompt schedules, providing consistent guidance to enhance the composition of rare concepts. By leveraging attention scores and orthogonal components, ADAPT significantly enhances compositional generation of rare concepts in the RareBench benchmark without additional training or fine-tuning. Through comprehensive experiments, we demonstrate that ADAPT achieves superior performance in RareBench and accurately reflects the semantic information of rare attributes, providing deterministic and precise control over the generation of rare compositions without compromising visual integrity.

-> ADAPT 프레임워크는 스포츠 영상의 품질 향상과 보정 기능을 강화하여 우리 AI 촬영 edge 디바이스의 핵심 기술을 보완합니다.

### Modeling the Impacts of Swipe Delay on User Quality of Experience in Short Video Streaming (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18575v1
- 점수: final 88.0

Short video streaming platforms have gained immense popularity in recent years, transforming the way users consume video content. A critical aspect of user interaction with these platforms is the swipe gesture, which allows users to navigate through videos seamlessly. However, the delay between a user's swipe action and the subsequent video playback can significantly impact the overall user experience. This paper presents the first systematic study investigating the effects of swipe delay on user Quality of Experience (QoE) in short video streaming. In particular, we conduct a subjective quality assessment containing 132 swipe delay patterns. The obtained results show that user experience is affected not only by the swipe delay duration, but also by the number of delays and their temporal positions. A single delay of eight seconds or longer is likely to lead to user dissatisfaction. Moreover, early-session delays are less harmful to user QoE than late-session delays. Based on the findings, we propose a novel QoE model that accurately predicts user experience based on swipe delay characteristics. The proposed model demonstrates high correlation with subjective ratings, outperforming existing models in short video streaming.

-> User experience modeling for short video streaming relevant to sports content platform

### AdaEdit: Adaptive Temporal and Channel Modulation for Flow-Based Image Editing (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21615v1
- 점수: final 86.4

Inversion-based image editing in flow matching models has emerged as a powerful paradigm for training-free, text-guided image manipulation. A central challenge in this paradigm is the injection dilemma: injecting source features during denoising preserves the background of the original image but simultaneously suppresses the model's ability to synthesize edited content. Existing methods address this with fixed injection strategies -- binary on/off temporal schedules, uniform spatial mixing ratios, and channel-agnostic latent perturbation -- that ignore the inherently heterogeneous nature of injection demand across both the temporal and channel dimensions. In this paper, we present AdaEdit, a training-free adaptive editing framework that resolves this dilemma through two complementary innovations. First, we propose a Progressive Injection Schedule that replaces hard binary cutoffs with continuous decay functions (sigmoid, cosine, or linear), enabling a smooth transition from source-feature preservation to target-feature generation and eliminating feature discontinuity artifacts. Second, we introduce Channel-Selective Latent Perturbation, which estimates per-channel importance based on the distributional gap between the inverted and random latents and applies differentiated perturbation strengths accordingly -- strongly perturbing edit-relevant channels while preserving structure-encoding channels. Extensive experiments on the PIE-Bench benchmark (700 images, 10 editing types) demonstrate that AdaEdit achieves an 8.7% reduction in LPIPS, a 2.6% improvement in SSIM, and a 2.3% improvement in PSNR over strong baselines, while maintaining competitive CLIP similarity. AdaEdit is fully plug-and-play and compatible with multiple ODE solvers including Euler, RF-Solver, and FireFlow. Code is available at https://github.com/leeguandong/AdaEdit

-> 스포츠 영상 하이라이트 제작에 직접 적용 가능한 이미지 편집 기술

### In-network Attack Detection with Federated Deep Learning in IoT Networks: Real Implementation and Analysis (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21596v1
- 점수: final 84.0

The rapid expansion of the Internet of Things (IoT) and its integration with backbone networks have heightened the risk of security breaches. Traditional centralized approaches to anomaly detection, which require transferring large volumes of data to central servers, suffer from privacy, scalability, and latency limitations. This paper proposes a lightweight autoencoder-based anomaly detection framework designed for deployment on resource-constrained edge devices, enabling real-time detection while minimizing data transfer and preserving privacy. Federated learning is employed to train models collaboratively across distributed devices, where local training occurs on edge nodes and only model weights are aggregated at a central server. A real-world IoT testbed using Raspberry Pi sensor nodes was developed to collect normal and attack traffic data. The proposed federated anomaly detection system, implemented and evaluated on the testbed, demonstrates its effectiveness in accurately identifying network attacks. The communication overhead was reduced significantly while achieving comparable performance to the centralized method.

-> Federated learning for IoT edge devices with real-time processing, applicable to edge computing aspects of the project

### GenMFSR: Generative Multi-Frame Image Restoration and Super-Resolution (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.19187v1
- 점수: final 84.0

Camera pipelines receive raw Bayer-format frames that need to be denoised, demosaiced, and often super-resolved. Multiple frames are captured to utilize natural hand tremors and enhance resolution. Multi-frame super-resolution is therefore a fundamental problem in camera pipelines. Existing adversarial methods are constrained by the quality of ground truth. We propose GenMFSR, the first Generative Multi-Frame Raw-to-RGB Super Resolution pipeline, that incorporates image priors from foundation models to obtain sub-pixel information for camera ISP applications. GenMFSR can align multiple raw frames, unlike existing single-frame super-resolution methods, and we propose a loss term that restricts generation to high-frequency regions in the raw domain, thus preventing low-frequency artifacts.

-> 다중 프레임 이미지 복원 및 슈퍼해상도 기술이 AI 촬영 디바이스에 직접적으로 적용 가능

### Counting Circuits: Mechanistic Interpretability of Visual Reasoning in Large Vision-Language Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18523v1
- 점수: final 84.0

Counting serves as a simple but powerful test of a Large Vision-Language Model's (LVLM's) reasoning; it forces the model to identify each individual object and then add them all up. In this study, we investigate how LVLMs implement counting using controlled synthetic and real-world benchmarks, combined with mechanistic analyses. Our results show that LVLMs display a human-like counting behavior, with precise performance on small numerosities and noisy estimation for larger quantities. We introduce two novel interpretability methods, Visual Activation Patching and HeadLens, and use them to uncover a structured "counting circuit" that is largely shared across a variety of visual reasoning tasks. Building on these insights, we propose a lightweight intervention strategy that exploits simple and abundantly available synthetic images to fine-tune arbitrary pretrained LVLMs exclusively on counting. Despite the narrow scope of this fine-tuning, the intervention not only enhances counting accuracy on in-distribution synthetic data, but also yields an average improvement of +8.36% on out-of-distribution counting benchmarks and an average gain of +1.54% on complex, general visual reasoning tasks for Qwen2.5-VL. These findings highlight the central, influential role of counting in visual reasoning and suggest a potential pathway for improving overall visual reasoning capabilities through targeted enhancement of counting mechanisms.

-> 시각적 추론과 객체 인식 기술이 스포츠 장면 분석에 직접적으로 적용 가능하여 선수 추적 및 경기 전략 분석에 활용될 수 있습니다.

### Through the Looking-Glass: AI-Mediated Video Communication Reduces Interpersonal Trust and Confidence in Judgments (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18868v1
- 점수: final 84.0

AI-based tools that mediate, enhance or generate parts of video communication may interfere with how people evaluate trustworthiness and credibility. In two preregistered online experiments (N = 2,000), we examined whether AI-mediated video retouching, background replacement and avatars affect interpersonal trust, people's ability to detect lies and confidence in their judgments. Participants watched short videos of speakers making truthful or deceptive statements across three conditions with varying levels of AI mediation. We observed that perceived trust and confidence in judgments declined in AI-mediated videos, particularly in settings in which some participants used avatars while others did not. However, participants' actual judgment accuracy remained unchanged, and they were no more inclined to suspect those using AI tools of lying. Our findings provide evidence against concerns that AI mediation undermines people's ability to distinguish truth from lies, and against cue-based accounts of lie detection more generally. They highlight the importance of trustworthy AI mediation tools in contexts where not only truth, but also trust and confidence matter.

-> AI 매개 비디오 처리 및 향상 기술은 프로젝트의 영상 향상 기능에 직접 적용 가능하여 핵심 기술이 될 수 있습니다.

### One Model, Two Markets: Bid-Aware Generative Recommendation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.22231v1
- 점수: final 84.0

Generative Recommender Systems using semantic ids, such as TIGER (Rajput et al., 2023), have emerged as a widely adopted competitive paradigm in sequential recommendation. However, existing architectures are designed solely for semantic retrieval and do not address concerns such as monetization via ad revenue and incorporation of bids for commercial retrieval. We propose GEM-Rec, a unified framework that integrates commercial relevance and monetization objectives directly into the generative sequence. We introduce control tokens to decouple the decision of whether to show an ad from which item to show. This allows the model to learn valid placement patterns directly from interaction logs, which inherently reflect past successful ad placements. Complementing this, we devise a Bid-Aware Decoding mechanism that handles real-time pricing, injecting bids directly into the inference process to steer the generation toward high-value items. We prove that this approach guarantees allocation monotonicity, ensuring that higher bids weakly increase an ad's likelihood of being shown without requiring model retraining. Experiments demonstrate that GEM-Rec allows platforms to dynamically optimize for semantic relevance and platform revenue.

-> GEM-Rec 프레임워크는 광고 수익 모델을 통합하여 스포츠 콘텐츠 플랫폼의 수익화 전략에 직접 적용 가능

### FILT3R: Latent State Adaptive Kalman Filter for Streaming 3D Reconstruction (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18493v1
- 점수: final 82.4

Streaming 3D reconstruction maintains a persistent latent state that is updated online from incoming frames, enabling constant-memory inference. A key failure mode is the state update rule: aggressive overwrites forget useful history, while conservative updates fail to track new evidence, and both behaviors become unstable beyond the training horizon. To address this challenge, we propose FILT3R, a training-free latent filtering layer that casts recurrent state updates as stochastic state estimation in token space. FILT3R maintains a per-token variance and computes a Kalman-style gain that adaptively balances memory retention against new observations. Process noise -- governing how much the latent state is expected to change between frames -- is estimated online from EMA-normalized temporal drift of candidate tokens. Using extensive experiments, we demonstrate that FILT3R yields an interpretable, plug-in update rule that generalizes common overwrite and gating policies as special cases. Specifically, we show that gains shrink in stable regimes as uncertainty contracts with accumulated evidence, and rise when genuine scene change increases process uncertainty, improving long-horizon stability for depth, pose, and 3D reconstruction, compared to the existing methods. Code will be released at https://github.com/jinotter3/FILT3R.

-> 스트리밍 3D 재구성 기술이 실시간 스포츠 촬영에 적용 가능하여 빠른 움직임과 변화하는 관점에서도 안정적인 3D 재구성을 제공합니다.

### SHARP: Spectrum-aware Highly-dynamic Adaptation for Resolution Promotion in Remote Sensing Synthesis (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21783v1
- 점수: final 82.4

Text-to-image generation powered by Diffusion Transformers (DiTs) has made remarkable strides, yet remote sensing (RS) synthesis lags behind due to two barriers: the absence of a domain-specialized DiT prior and the prohibitive cost of training at the large resolutions that RS applications demand. Training-free resolution promotion via Rotary Position Embedding (RoPE) rescaling offers a practical remedy, but every existing method applies a static positional scaling rule throughout the denoising process. This uniform compression is particularly harmful for RS imagery, whose substantially denser medium- and high-frequency energy encodes the fine structures critical for aerial-scene realism, such as vehicles, building contours, and road markings. Addressing both challenges requires a domain-specialized generative prior coupled with a denoising-aware positional adaptation strategy. To this end, we fine-tune FLUX on over 100,000 curated RS images to build a strong domain prior (RS-FLUX), and propose Spectrum-aware Highly-dynamic Adaptation for Resolution Promotion (SHARP), a training-free method that introduces a rational fractional time schedule k_rs(t) into RoPE. SHARP applies strong positional promotion during the early layout-formation stage and progressively relaxes it during detail recovery, aligning extrapolation strength with the frequency-progressive nature of diffusion denoising. Its resolution-agnostic formulation further enables robust multi-scale generation from a single set of hyperparameters. Extensive experiments across six square and rectangular resolutions show that SHARP consistently outperforms all training-free baselines on CLIP Score, Aesthetic Score, and HPSv2, with widening margins at more aggressive extrapolation factors and negligible computational overhead. Code and weights are available at https://github.com/bxuanz/SHARP.

-> 스포츠 영상 품질 향상을 위한 고해상도 기술 적용 가능

### ALADIN:Attribute-Language Distillation Network for Person Re-Identification (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21482v1
- 점수: final 80.0

Recent vision-language models such as CLIP provide strong cross-modal alignment, but current CLIP-guided ReID pipelines rely on global features and fixed prompts. This limits their ability to capture fine-grained attribute cues and adapt to diverse appearances. We propose ALADIN, an attribute-language distillation network that distills knowledge from a frozen CLIP teacher to a lightweight ReID student. ALADIN introduces fine-grained attribute-local alignment to establish adaptive text-visual correspondence and robust representation learning. A Scene-Aware Prompt Generator produces image-specific soft prompts to facilitate adaptive alignment. Attribute-local distillation enforces consistency between textual attributes and local visual features, significantly enhancing robustness under occlusions. Furthermore, we employ cross-modal contrastive and relation distillation to preserve the inherent structural relationships among attributes. To provide precise supervision, we leverage Multimodal LLMs to generate structured attribute descriptions, which are then converted into localized attention maps via CLIP. At inference, only the student is used. Experiments on Market-1501, DukeMTMC-reID, and MSMT17 show improvements over CNN-, Transformer-, and CLIP-based methods, with better generalization and interpretability.

-> 선수 추적 및 분석을 위한 인물 재식별 시스템 직접 적용 가능

### DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.22271v1
- 점수: final 80.0

Diffusion-based video super-resolution (VSR) has recently achieved remarkable fidelity but still suffers from prohibitive sampling costs. While distribution matching distillation (DMD) can accelerate diffusion models toward one-step generation, directly applying it to VSR often results in training instability alongside degraded and insufficient supervision. To address these issues, we propose DUO-VSR, a three-stage framework built upon a Dual-Stream Distillation strategy that unifies distribution matching and adversarial supervision for one-step VSR. Firstly, a Progressive Guided Distillation Initialization is employed to stabilize subsequent training through trajectory-preserving distillation. Next, the Dual-Stream Distillation jointly optimizes the DMD and Real-Fake Score Feature GAN (RFS-GAN) streams, with the latter providing complementary adversarial supervision leveraging discriminative features from both real and fake score models. Finally, a Preference-Guided Refinement stage further aligns the student with perceptual quality preferences. Extensive experiments demonstrate that DUO-VSR achieves superior visual quality and efficiency over previous one-step VSR approaches.

-> 비디오 슈퍼-해상도를 위한 이중-스트림 증류 프레임워크로 스포츠 영상 보정 및 편집과 직접적인 관련성 있음

### A Framework for Closed-Loop Robotic Assembly, Alignment and Self-Recovery of Precision Optical Systems (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21496v1
- 점수: final 80.0

Robotic automation has transformed scientific workflows in domains such as chemistry and materials science, yet free-space optics, which is a high precision domain, remains largely manual. Optical systems impose strict spatial and angular tolerances, and their performance is governed by tightly coupled physical parameters, making generalizable automation particularly challenging. In this work, we present a robotics framework for the autonomous construction, alignment, and maintenance of precision optical systems. Our approach integrates hierarchical computer vision systems, optimization routines, and custom-built tools to achieve this functionality. As a representative demonstration, we perform the fully autonomous construction of a tabletop laser cavity from randomly distributed components. The system performs several tasks such as laser beam centering, spatial alignment of multiple beams, resonator alignment, laser mode selection, and self-recovery from induced misalignment and disturbances. By achieving closed-loop autonomy for highly sensitive optical systems, this work establishes a foundation for autonomous optical experiments for applications across technical domains.

-> Computer vision and optimization methodologies relevant to video analysis but specifically for optical systems

### Mapping Travel Experience in Public Transport: Real-Time Evidence and Spatial Analysis in Hamburg (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21763v1
- 점수: final 80.0

Shifting travel from private cars to public transport is critical for meeting climate and related mobility goals, yet passengers will only choose transit if it offers a consistently positive experience. Previous studies of passenger satisfaction have largely relied on retrospective surveys, which overlook the dynamic and spatially differentiated nature of travel experience. This paper introduces a novel combination of real-time experience sampling and spatial hot spot analysis to capture and map where public transport users report consistently positive or negative experiences.   Data were collected from 239 participants in Hamburg between March and September 2025. Using a smartphone application, travelers reported their momentary journey experience every five minutes during everyday trips, yielding over 21,000 in-situ evaluations. These geo-referenced data were analyzed with the Getis-Ord $Gi^{*}$ statistic to detect significant clusters of positive and negative travel experience. The analysis identified distinct hot and cold spots of travel experience across the network. Cold spots were shaped by heterogeneous problems, ranging from predominantly delay-dominated to overcrowding or socially stressful locations. In contrast, hot spots emerged through different pathways, including comfort-oriented, time-efficient or context-driven environments.   The findings highlight three contributions. First, cold spots are not uniform but reflect specific local constellations of problems, requiring targeted interventions. Second, hot spots illustrate multiple success models that can serve as benchmarks for replication. Third, this study demonstrates the value of combining dynamic high-resolution sampling with spatial statistics to guide more effective and place-specific improvements in public transport.

-> 실시간 데이터 수집 및 공간 분석 방법론이 스포츠 경기 데이터 수집과 분석에 직접적으로 적용 가능하며, 경험 매핑 기법을 통해 선수들의 움직임 패턴과 경기 전략 분석에 활용 가능

### λ-GELU: Learning Gating Hardness for Controlled ReLU-ization in Deep Networks (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.21991v1
- 점수: final 80.0

Gaussian Error Linear Unit (GELU) is a widely used smooth alternative to Rectifier Linear Unit (ReLU), yet many deployment, compression, and analysis toolchains are most naturally expressed for piecewise-linear (ReLU-type) networks. We study a hardness-parameterized formulation of GELU, f(x;λ)=xΦ(λ x), where Φ is the Gaussian CDF and λ \in [1, infty) controls gate sharpness, with the goal of turning smooth gated training into a controlled path toward ReLU-compatible models. Learning λ is non-trivial: naive updates yield unstable dynamics and effective gradient attenuation, so we introduce a constrained reparameterization and an optimizer-aware update scheme.   Empirically, across a diverse set of model--dataset pairs spanning MLPs, CNNs, and Transformers, we observe structured layerwise hardness profiles and assess their robustness under different initializations. We further study a deterministic ReLU-ization strategy in which the learned gates are progressively hardened toward a principled target, enabling a post-training substitution of λ-GELU by ReLU with reduced disruption. Overall, λ-GELU provides a minimal and interpretable knob to profile and control gating hardness, bridging smooth training with ReLU-centric downstream pipelines.

-> 신경망 최적화 기술로 rk3588 같은 엣지 디바이스에서 효율적인 AI 모델 배포에 적용 가능

### Generalized Hand-Object Pose Estimation with Occlusion Awareness (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.19013v1
- 점수: final 80.0

Generalized 3D hand-object pose estimation from a single RGB image remains challenging due to the large variations in object appearances and interaction patterns, especially under heavy occlusion. We propose GenHOI, a framework for generalized hand-object pose estimation with occlusion awareness. GenHOI integrates hierarchical semantic knowledge with hand priors to enhance model generalization under challenging occlusion conditions. Specifically, we introduce a hierarchical semantic prompt that encodes object states, hand configurations, and interaction patterns via textual descriptions. This enables the model to learn abstract high-level representations of hand-object interactions for generalization to unseen objects and novel interactions while compensating for missing or ambiguous visual cues. To enable robust occlusion reasoning, we adopt a multi-modal masked modeling strategy over RGB images, predicted point clouds, and textual descriptions. Moreover, we leverage hand priors as stable spatial references to extract implicit interaction constraints. This allows reliable pose inference even under significant variations in object shapes and interaction patterns. Extensive experiments on the challenging DexYCB and HO3Dv2 benchmarks demonstrate that our method achieves state-of-the-art performance in hand-object pose estimation.

-> 손 자세 추정 기술은 스포츠 동작 분석에 적용 가능하여 선수들의 기술 분석과 전략 개발에 활용될 수 있습니다.

### Improving Joint Audio-Video Generation with Cross-Modal Context Learning (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.18600v1
- 점수: final 80.0

The dual-stream transformer architecture-based joint audio-video generation method has become the dominant paradigm in current research. By incorporating pre-trained video diffusion models and audio diffusion models, along with a cross-modal interaction attention module, high-quality, temporally synchronized audio-video content can be generated with minimal training data. In this paper, we first revisit the dual-stream transformer paradigm and further analyze its limitations, including model manifold variations caused by the gating mechanism controlling cross-modal interactions, biases in multi-modal background regions introduced by cross-modal attention, and the inconsistencies in multi-modal classifier-free guidance (CFG) during training and inference, as well as conflicts between multiple conditions. To alleviate these issues, we propose Cross-Modal Context Learning (CCL), equipped with several carefully designed modules. Temporally Aligned RoPE and Partitioning (TARP) effectively enhances the temporal alignment between audio latent and video latent representations. The Learnable Context Tokens (LCT) and Dynamic Context Routing (DCR) in the Cross-Modal Context Attention (CCA) module provide stable unconditional anchors for cross-modal information, while dynamically routing based on different training tasks, further enhancing the model's convergence speed and generation quality. During inference, Unconditional Context Guidance (UCG) leverages the unconditional support provided by LCT to facilitate different forms of CFG, improving train-inference consistency and further alleviating conflicts. Through comprehensive evaluations, CCL achieves state-of-the-art performance compared with recent academic methods while requiring substantially fewer resources.

-> 오디오-비디오 생성 기술은 스포츠 하이라이트 영상 제작에 적용 가능하여 플랫폼 콘텐츠 생성의 핵심 기술이 될 수 있습니다.

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
