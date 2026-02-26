# CAPP!C_AI 논문 리포트 (2026-02-26)

> 수집 44 | 필터 39 | 폐기 0 | 평가 11 | 출력 9 | 기준 50점

검색 윈도우: 2026-02-25T00:00:00+00:00 ~ 2026-02-26T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 22

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry

- arXiv: http://arxiv.org/abs/2602.21810v1
- PDF: https://arxiv.org/pdf/2602.21810v1
- 코드: https://github.com/zjutcvg/GeoMotion
- 발행일: 2026-02-25
- 카테고리: cs.CV
- 점수: final 88.0 (llm_adjusted:85 = base:82 + bonus:+3)
- 플래그: 코드 공개

**개요**
Motion segmentation in dynamic scenes is highly challenging, as conventional methods heavily rely on estimating camera poses and point correspondences from inherently noisy motion cues. Existing statistical inference or iterative optimization techniques that struggle to mitigate the cumulative errors in multi-stage pipelines often lead to limited performance or high computational cost. In contrast, we propose a fully learning-based approach that directly infers moving objects from latent feature representations via attention mechanisms, thus enabling end-to-end feed-forward motion segmentation. Our key insight is to bypass explicit correspondence estimation and instead let the model learn to implicitly disentangle object and camera motion. Supported by recent advances in 4D scene geometry reconstruction (e.g., $π^3$), the proposed method leverages reliable camera poses and rich spatial-temporal priors, which ensure stable training and robust inference for the model. Extensive experiments demonstrate that by eliminating complex pre-processing and iterative refinement, our approach achieves state-of-the-art motion segmentation performance with high efficiency. The code is available at:https://github.com/zjutcvg/GeoMotion.

**선정 근거**
동적 장면에서의 움직임 분할 기술로 스포츠 촬영 시 선수와 공 등 움직이는 객체를 정확히 식별 가능

**활용 인사이트**
어텐션 메커니즘을 활용한 엔드투엔드 움직임 분할로 실시간 하이라이트 자동 생성 가능

## 2위: From Statics to Dynamics: Physics-Aware Image Editing with Latent Transition Priors

- arXiv: http://arxiv.org/abs/2602.21778v1
- PDF: https://arxiv.org/pdf/2602.21778v1
- 발행일: 2026-02-25
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Instruction-based image editing has achieved remarkable success in semantic alignment, yet state-of-the-art models frequently fail to render physically plausible results when editing involves complex causal dynamics, such as refraction or material deformation. We attribute this limitation to the dominant paradigm that treats editing as a discrete mapping between image pairs, which provides only boundary conditions and leaves transition dynamics underspecified. To address this, we reformulate physics-aware editing as predictive physical state transitions and introduce PhysicTran38K, a large-scale video-based dataset comprising 38K transition trajectories across five physical domains, constructed via a two-stage filtering and constraint-aware annotation pipeline. Building on this supervision, we propose PhysicEdit, an end-to-end framework equipped with a textual-visual dual-thinking mechanism. It combines a frozen Qwen2.5-VL for physically grounded reasoning with learnable transition queries that provide timestep-adaptive visual guidance to a diffusion backbone. Experiments show that PhysicEdit improves over Qwen-Image-Edit by 5.9% in physical realism and 10.1% in knowledge-grounded editing, setting a new state-of-the-art for open-source methods, while remaining competitive with leading proprietary models.

**선정 근거**
물리적 사실성을 고려한 이미지 편집 기술로 스포츠 영상의 시각적 품질 향상 가능

**활용 인사이트**
물리적 상태 전이 예측 프레임워크를 통해 스포츠 장면의 사실적인 재구현 및 보정 가능

## 3위: UniVBench: Towards Unified Evaluation for Video Foundation Models

- arXiv: http://arxiv.org/abs/2602.21835v1
- PDF: https://arxiv.org/pdf/2602.21835v1
- 발행일: 2026-02-25
- 카테고리: cs.CV
- 점수: final 68.0 (llm_adjusted:60 = base:60 + bonus:+0)

**개요**
Video foundation models aim to integrate video understanding, generation, editing, and instruction following within a single framework, making them a central direction for next-generation multimodal systems. However, existing evaluation benchmarks remain fragmented and limited in scope, as they each target a single task, rely on task-specific metrics, and typically use short or simple video clips. As a result, they do not capture the unified capabilities that these models are designed to deliver. To address this gap, we introduce UniVBench, a benchmark purpose-built for evaluating video foundation models across four core abilities: video understanding, video generation, video editing, and a newly proposed task, video reconstruction, which assesses how faithfully a model can reproduce video content it has encountered. Our benchmark substantially expands the complexity of evaluation by incorporating 200 high-quality, diverse and multi-shot videos, each paired with detailed captions, multi-format editing instructions, and reference images. All videos are human-created and carefully validated, offering richer cinematic information than prior benchmarks. In addition, we develop a unified agentic evaluation system (UniV-Eval) that standardizes prompting, instruction parsing, and scoring across all tasks, enabling fair, scalable, and reproducible comparisons of unified video models. By grounding evaluation in instruction-based multi-shot video tasks, UniVBench provides the first framework for measuring the integrated capabilities that video foundation models aim to achieve. Extensive human annotations ensure our evaluation aligns with human judgment, enabling rigorous assessment and accelerating progress toward robust video intelligence.

**선정 근거**
UniVBench는 비디오 이해, 편집, 생성 통합 평가 프레임워크로 우리의 스포츠 하이라이트 자동 편집 및 분석 시스템 개발에 적합한 평가 기준 제공

**활용 인사이트**
UniVBench의 다중 샷 비디오 평가 방식을 채택해 스포츠 장면 복잡성에 맞춘 AI 모델 훈련 및 성능 검증 시스템 구축 가능

## 4위: Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels

- arXiv: http://arxiv.org/abs/2602.22140v1
- PDF: https://arxiv.org/pdf/2602.22140v1
- 발행일: 2026-02-25
- 카테고리: eess.IV, cs.CV
- 점수: final 68.0 (llm_adjusted:60 = base:55 + bonus:+5)
- 플래그: 실시간

**개요**
We present Lumosaic, a compact active hyperspectral video system designed for real-time capture of dynamic scenes. Our approach combines a narrowband LED array with a coded-exposure-pixel (CEP) camera capable of high-speed, per-pixel exposure control, enabling joint encoding of scene information across space, time, and wavelength within each video frame. Unlike passive snapshot systems that divide light across multiple spectral channels simultaneously and assume no motion during a frame's exposure, Lumosaic actively synchronizes illumination and pixel-wise exposure, improving photon utilization and preserving spectral fidelity under motion. A learning-based reconstruction pipeline then recovers 31-channel hyperspectral (400-700 nm) video at 30 fps and VGA resolution, producing temporally coherent and spectrally accurate reconstructions. Experiments on synthetic and real data demonstrate that Lumosaic significantly improves reconstruction fidelity and temporal stability over existing snapshot hyperspectral imaging systems, enabling robust hyperspectral video across diverse materials and motion conditions.

**선정 근거**
초분광 영상 촬영 기술은 스포츠 촬영과 간접적으로 관련 있으나 전문 과학 응용에 특화되어 직접 적용은 제한적이다.

## 5위: StoryComposerAI: Supporting Human-AI Story Co-Creation Through Decomposition and Linking

- arXiv: http://arxiv.org/abs/2602.21486v1
- PDF: https://arxiv.org/pdf/2602.21486v1
- 발행일: 2026-02-25
- 카테고리: cs.HC
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
GenAI's ability to produce text and images is increasingly incorporated into human-AI co-creation tasks such as storytelling and video editing. However, integrating GenAI into these tasks requires enabling users to retain control over editing individual story elements while ensuring that generated visuals remain coherent with the storyline and consistent across multiple AI-generated outputs. This work examines a paradigm of creative decomposition and linking, which allows creators to clearly communicate creative intent by prompting GenAI to tailor specific story elements, such as storylines, personas, locations, and scenes, while maintaining coherence among them. We implement and evaluate StoryComposerAI, a system that exemplifies this paradigm for enhancing users' sense of control and content consistency in human-AI co-creation of digital stories.

**선정 근거**
이야기 창작을 위한 AI 비디오 편집 시스템으로 스포츠 분석과는 간접적으로 관련이 있습니다.

## 6위: Non-Extreme Individual Minima for Improved Pareto Front Sampling Efficiency and Decision-Making

- arXiv: http://arxiv.org/abs/2602.21883v1
- PDF: https://arxiv.org/pdf/2602.21883v1
- 발행일: 2026-02-25
- 카테고리: math.OC
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
In multi-objective optimization, the set of optimal trade-offs -- the Pareto front -- often contains regions that are extremely steep or flat. The Pareto optimal points in these regions are typically of limited interest for decision-making, as the marginal rate of substitution is extreme: a marginal improvement in one objective necessitates a significant deterioration in at least one other objective. These unfavorable trade-offs frequently occur near the individual minima, where single objectives attain their minimum values without considering the remaining criteria.   To address this, we propose the concept of \emph{non-extreme individual minima} that relies on the notion of $L$-practical proper efficiency. These points can serve as a less sensitive replacement for \emph{standard} individual minima in subsequent related methods. Specifically, they allow for a more practical restriction of the Pareto front sampling within a refined utopia-nadir hyperbox, provide a meaningful basis for image space normalization, and can enhance decision-making techniques, such as knee-point methods, by focusing on regions with acceptable trade-offs.   We provide a computationally efficient algorithm to determine these non-extreme individual minima by solving at most $2n_J$ standard weighted-sum scalarizations, where $n_J$ is the number of objectives. To ensure robustness across varying objective scales, the method incorporates an integrated image space normalization strategy. Numerical examples, specifically a convex academic case and a non-convex real-world application, demonstrate that the method successfully excludes practically irrelevant regions in the image space.

**선정 근거**
다목적 최적화 이론 연구로 스포츠 촬영/영상 처리와 직접적 연관 없음

## 7위: Learning Unknown Interdependencies for Decentralized Root Cause Analysis in Nonlinear Dynamical Systems

- arXiv: http://arxiv.org/abs/2602.21928v1
- PDF: https://arxiv.org/pdf/2602.21928v1
- 발행일: 2026-02-25
- 카테고리: cs.LG, stat.ML
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
Root cause analysis (RCA) in networked industrial systems, such as supply chains and power networks, is notoriously difficult due to unknown and dynamically evolving interdependencies among geographically distributed clients. These clients represent heterogeneous physical processes and industrial assets equipped with sensors that generate large volumes of nonlinear, high-dimensional, and heterogeneous IoT data. Classical RCA methods require partial or full knowledge of the system's dependency graph, which is rarely available in these complex networks. While federated learning (FL) offers a natural framework for decentralized settings, most existing FL methods assume homogeneous feature spaces and retrainable client models. These assumptions are not compatible with our problem setting. Different clients have different data features and often run fixed, proprietary models that cannot be modified. This paper presents a federated cross-client interdependency learning methodology for feature-partitioned, nonlinear time-series data, without requiring access to raw sensor streams or modifying proprietary client models. Each proprietary local client model is augmented with a Machine Learning (ML) model that encodes cross-client interdependencies. These ML models are coordinated via a global server that enforces representation consistency while preserving privacy through calibrated differential privacy noise. RCA is performed using model residuals and anomaly flags. We establish theoretical convergence guarantees and validate our approach on extensive simulations and a real-world industrial cybersecurity dataset.

**선정 근거**
이 논문은 OCR 기반 이미지 분석 기술을 제안하며, 스포츠 영상에서 객체 인식과 장면 분석에 활용될 수 있음

**활용 인사이트**
경기장 내 텍스트 정보(점수, 선수명 등) 자동 추출을 통한 실시간 데이터 처리로 영상 품질 향상 가능

## 8위: The Silent Spill: Measuring Sensitive Data Leaks Across Public URL Repositories

- arXiv: http://arxiv.org/abs/2602.21826v1
- PDF: https://arxiv.org/pdf/2602.21826v1
- 발행일: 2026-02-25
- 카테고리: cs.CR
- 점수: final 64.0 (llm_adjusted:55 = base:55 + bonus:+0)

**개요**
A large number of URLs are made public by various platforms for security analysis, archiving, and paste sharing -- such as VirusTotal, URLScan.io, Hybrid Analysis, the Wayback Machine, and RedHunt. These services may unintentionally expose links containing sensitive information, as reported in some news articles and blog posts. However, no large-scale measurement has quantified the extent of such exposures. We present an automated system that detects and analyzes potential sensitive information leaked through publicly accessible URLs. The system combines lexical URL filtering, dynamic rendering, OCR-based extraction, and content classification to identify potential leaks. We apply it to 6,094,475 URLs collected from public scanning platforms, paste sites, and web archives, identifying 12,331 potential exposures across authentication, financial, personal, and document-related domains. These findings show that sensitive information remains exposed, underscoring the importance of automated detection to identify accidental leaks.

**선정 근거**
영상 분석 시 노이즈 처리를 위한 강인한 통계 방법 간접 관련

## 9위: "Without AI, I Would Never Share This Online": Unpacking How LLMs Catalyze Women's Sharing of Gendered Experiences on Social Media

- arXiv: http://arxiv.org/abs/2602.21686v1
- PDF: https://arxiv.org/pdf/2602.21686v1
- 발행일: 2026-02-25
- 카테고리: cs.HC
- 점수: final 52.0 (llm_adjusted:40 = base:40 + bonus:+0)

**개요**
Sharing gendered experiences on social media has been widely recognized as supporting women's personal sense-making and contributing to digital feminism. However, there are known concerns, such as fear of judgment and backlash, that may discourage women from posting online. In this study, we examine a recurring practice on Xiaohongshu, a popular Chinese social media platform, in which women share their gendered experiences alongside screenshots of conversations with LLMs. We conducted semi-structured interviews with 20 women to investigate whether and how interactions with LLMs might support women in articulating and sharing gendered experiences. Our findings reveal that, beyond those external concerns, women also hold self-imposed standards regarding what feels appropriate and worthwhile to share publicly. We further show how interactions with LLMs help women meet these standards and navigate such concerns. We conclude by discussing how LLMs might be carefully and critically leveraged to support women's everyday expression online.

**선정 근거**
OCR 기반 이미지 분석 기술이 간접적으로 관련될 수 있으나, 보안 분야에 특화됨

---

## 다시 보기

### Mobile-O: Unified Multimodal Understanding and Generation on Mobile Device (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.20161v1
- 점수: final 98.4

Unified multimodal models can both understand and generate visual content within a single architecture. Existing models, however, remain data-hungry and too heavy for deployment on edge devices. We present Mobile-O, a compact vision-language-diffusion model that brings unified multimodal intelligence to a mobile device. Its core module, the Mobile Conditioning Projector (MCP), fuses vision-language features with a diffusion generator using depthwise-separable convolutions and layerwise alignment. This design enables efficient cross-modal conditioning with minimal computational cost. Trained on only a few million samples and post-trained in a novel quadruplet format (generation prompt, image, question, answer), Mobile-O jointly enhances both visual understanding and generation capabilities. Despite its efficiency, Mobile-O attains competitive or superior performance compared to other unified models, achieving 74% on GenEval and outperforming Show-O and JanusFlow by 5% and 11%, while running 6x and 11x faster, respectively. For visual understanding, Mobile-O surpasses them by 15.3% and 5.1% averaged across seven benchmarks. Running in only ~3s per 512x512 image on an iPhone, Mobile-O establishes the first practical framework for real-time unified multimodal understanding and generation on edge devices. We hope Mobile-O will ease future research in real-time unified multimodal intelligence running entirely on-device with no cloud dependency. Our code, models, datasets, and mobile application are publicly available at https://amshaker.github.io/Mobile-O/

-> 에지 디바이스에서 실시간 멀티모달 이해/생성 가능해 영상 보정 및 분석에 직접 적용. Mobile-O의 경량 설계(fps 6~11배 향상)가 rk3588 호환성 높음.

### From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.20630v1
- 점수: final 96.0

Keypoint-based matching is a fundamental component of modern 3D vision systems, such as Structure-from-Motion (SfM) and SLAM. Most existing learning-based methods are trained on image pairs, a paradigm that fails to explicitly optimize for the long-term trackability of keypoints across sequences under challenging viewpoint and illumination changes. In this paper, we reframe keypoint detection as a sequential decision-making problem. We introduce TraqPoint, a novel, end-to-end Reinforcement Learning (RL) framework designed to optimize the \textbf{Tra}ck-\textbf{q}uality (Traq) of keypoints directly on image sequences. Our core innovation is a track-aware reward mechanism that jointly encourages the consistency and distinctiveness of keypoints across multiple views, guided by a policy gradient method. Extensive evaluations on sparse matching benchmarks, including relative pose estimation and 3D reconstruction, demonstrate that TraqPoint significantly outperforms some state-of-the-art (SOTA) keypoint detection and description methods.

-> 실시간 동분할 기술은 스포츠 장면에서 움직임을 효과적으로 분석하여 하이라이트 편집에 적합합니다.

### Real-time Motion Segmentation with Event-based Normal Flow (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.20790v1
- 점수: final 93.6

Event-based cameras are bio-inspired sensors with pixels that independently and asynchronously respond to brightness changes at microsecond resolution, offering the potential to handle visual tasks in challenging scenarios. However, due to the sparse information content in individual events, directly processing the raw event data to solve vision tasks is highly inefficient, which severely limits the applicability of state-of-the-art methods in real-time tasks, such as motion segmentation, a fundamental task for dynamic scene understanding. Incorporating normal flow as an intermediate representation to compress motion information from event clusters within a localized region provides a more effective solution. In this work, we propose a normal flow-based motion segmentation framework for event-based vision. Leveraging the dense normal flow directly learned from event neighborhoods as input, we formulate the motion segmentation task as an energy minimization problem solved via graph cuts, and optimize it iteratively with normal flow clustering and motion model fitting. By using a normal flow-based motion model initialization and fitting method, the proposed system is able to efficiently estimate the motion models of independently moving objects with only a limited number of candidate models, which significantly reduces the computational complexity and ensures real-time performance, achieving nearly a 800x speedup in comparison to the open-source state-of-the-art method. Extensive evaluations on multiple public datasets fully demonstrate the accuracy and efficiency of our framework.

-> 고속 움직임으로 인한 흐림 문제를 해결하여 스포츠 촬영의 품질을 향상시키는 데 중요합니다.

### Training Deep Stereo Matching Networks on Tree Branch Imagery: A Benchmark Study for Real-Time UAV Forestry Applications (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19763v1
- 점수: final 93.6

Autonomous drone-based tree pruning needs accurate, real-time depth estimation from stereo cameras. Depth is computed from disparity maps using $Z = f B/d$, so even small disparity errors cause noticeable depth mistakes at working distances. Building on our earlier work that identified DEFOM-Stereo as the best reference disparity generator for vegetation scenes, we present the first study to train and test ten deep stereo matching networks on real tree branch images. We use the Canterbury Tree Branches dataset -- 5,313 stereo pairs from a ZED Mini camera at 1080P and 720P -- with DEFOM-generated disparity maps as training targets. The ten methods cover step-by-step refinement, 3D convolution, edge-aware attention, and lightweight designs. Using perceptual metrics (SSIM, LPIPS, ViTScore) and structural metrics (SIFT/ORB feature matching), we find that BANet-3D produces the best overall quality (SSIM = 0.883, LPIPS = 0.157), while RAFT-Stereo scores highest on scene-level understanding (ViTScore = 0.799). Testing on an NVIDIA Jetson Orin Super (16 GB, independently powered) mounted on our drone shows that AnyNet reaches 6.99 FPS at 1080P -- the only near-real-time option -- while BANet-2D gives the best quality-speed balance at 1.21 FPS. We also compare 720P and 1080P processing times to guide resolution choices for forestry drone systems.

-> 실시간 스테레오 깊이 추정이 운동 동작 3D 분석에 필수. AnyNet 6.99fps로 rk3588에서 동작 캡처 가능성 높음.

### Real-time Win Probability and Latent Player Ability via STATS X in Team Sports (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19513v1
- 점수: final 92.0

This study proposes a statistically grounded framework for real-time win probability evaluation and player assessment in score-based team sports, based on minute-by-minute cumulative box-score data. We introduce a continuous dominance indicator (T-score) that maps final scores to real values consistent with win/lose outcomes, and formulate it as a time-evolving stochastic representation (T-process) driven by standardized cumulative statistics. This structure captures temporal game dynamics and enables sequential, analytically tractable updates of in-game win probability. Through this stochastic formulation, competitive advantage is decomposed into interpretable statistical components. Furthermore, we define a latent contribution index, STATS X, which quantifies a player's involvement in favorable dominance intervals identified by the T-process. This allows us to separate a team's baseline strength from game-specific performance fluctuations and provides a coherent, structural evaluation framework for both teams and players. While we do not implement AI methods in this paper, our framework is positioned as a foundational step toward hybrid integration with AI. By providing a structured time-series representation of dominance with an explicit probabilistic interpretation, the framework enables flexible learning mechanisms and incorporation of high-dimensional data, while preserving statistical coherence and interpretability. This work provides a basis for advancing AI-driven sports analytics.

-> 실시간 승률(T-process) 및 선수 능력(STATS X) 분석이 경기 전략/개인 평가에 직접 적용. AI 통합 기반 제공.

### Human Video Generation from a Single Image with 3D Pose and View Control (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.21188v1
- 점수: final 90.4

Recent diffusion methods have made significant progress in generating videos from single images due to their powerful visual generation capabilities. However, challenges persist in image-to-video synthesis, particularly in human video generation, where inferring view-consistent, motion-dependent clothing wrinkles from a single image remains a formidable problem. In this paper, we present Human Video Generation in 4D (HVG), a latent video diffusion model capable of generating high-quality, multi-view, spatiotemporally coherent human videos from a single image with 3D pose and view control. HVG achieves this through three key designs: (i) Articulated Pose Modulation, which captures the anatomical relationships of 3D joints via a novel dual-dimensional bone map and resolves self-occlusions across views by introducing 3D information; (ii) View and Temporal Alignment, which ensures multi-view consistency and alignment between a reference image and pose sequences for frame-to-frame stability; and (iii) Progressive Spatio-Temporal Sampling with temporal alignment to maintain smooth transitions in long multi-view animations. Extensive experiments on image-to-video tasks demonstrate that HVG outperforms existing methods in generating high-quality 4D human videos from diverse human images and pose inputs.

-> 3D 포즈 제어 기반 영상 생성 기술은 스포츠 하이라이트 영상 제작에 직접 적용 가능

### Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.21101v1
- 점수: final 89.6

Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.

-> 드론 기반 3D 재구성 기술은 스포츠 촬영 시 움직임 흐림 문제 해결에 적용 가능

### Accurate Planar Tracking With Robust Re-Detection (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19624v1
- 점수: final 88.0

We present SAM-H and WOFTSAM, novel planar trackers that combine robust long-term segmentation tracking provided by SAM 2 with 8 degrees-of-freedom homography pose estimation. SAM-H estimates homographies from segmentation mask contours and is thus highly robust to target appearance changes. WOFTSAM significantly improves the current state-of-the-art planar tracker WOFT by exploiting lost target re-detection provided by SAM-H. The proposed methods are evaluated on POT-210 and PlanarTrack tracking benchmarks, setting the new state-of-the-art performance on both. On the latter, they outperform the second best by a large margin, +12.4 and +15.2pp on the p@15 metric. We also present improved ground-truth annotations of initial PlanarTrack poses, enabling more accurate benchmarking in the high-precision p@5 metric. The code and the re-annotations are available at https://github.com/serycjon/WOFTSAM

-> 평면 추적 기술이 운동 장비/선수 동작 자동 추적에 핵심. SAM-H의 견고한 재탐지로 빠른 움직임 대응 가능.

### StructXLIP: Enhancing Vision-language Models with Multimodal Structural Cues (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.20089v1
- 점수: final 88.0

Edge-based representations are fundamental cues for visual understanding, a principle rooted in early vision research and still central today. We extend this principle to vision-language alignment, showing that isolating and aligning structural cues across modalities can greatly benefit fine-tuning on long, detail-rich captions, with a specific focus on improving cross-modal retrieval. We introduce StructXLIP, a fine-tuning alignment paradigm that extracts edge maps (e.g., Canny), treating them as proxies for the visual structure of an image, and filters the corresponding captions to emphasize structural cues, making them "structure-centric". Fine-tuning augments the standard alignment loss with three structure-centric losses: (i) aligning edge maps with structural text, (ii) matching local edge regions to textual chunks, and (iii) connecting edge maps to color images to prevent representation drift. From a theoretical standpoint, while standard CLIP maximizes the mutual information between visual and textual embeddings, StructXLIP additionally maximizes the mutual information between multimodal structural representations. This auxiliary optimization is intrinsically harder, guiding the model toward more robust and semantically stable minima, enhancing vision-language alignment. Beyond outperforming current competitors on cross-modal retrieval in both general and specialized domains, our method serves as a general boosting recipe that can be integrated into future approaches in a plug-and-play manner. Code and pretrained models are publicly available at: https://github.com/intelligolabs/StructXLIP.

-> 비전-언어 정렬 기술이 스포츠 영상 분석 및 보정에 적용 가능. 구조적 단서 정렬로 운동 자세나 전략 설명 정확도 향상.

### A Context-Aware Knowledge Graph Platform for Stream Processing in Industrial IoT (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19990v1
- 점수: final 88.0

Industrial IoT ecosystems bring together sensors, machines and smart devices operating collaboratively across industrial environments. These systems generate large volumes of heterogeneous, high-velocity data streams that require interoperable, secure and contextually aware management. Most of the current stream management architectures, however, still rely on syntactic integration mechanisms, which result in limited flexibility, maintainability and interpretability in complex Industry 5.0 scenarios. This work proposes a context-aware semantic platform for data stream management that unifies heterogeneous IoT/IoE data sources through a Knowledge Graph enabling formal representation of devices, streams, agents, transformation pipelines, roles and rights. The model supports flexible data gathering, composable stream processing pipelines, and dynamic role-based data access based on agents' contexts, relying on Apache Kafka and Apache Flink for real-time processing, while SPARQL and SWRL-based reasoning provide context-dependent stream discovery. Experimental evaluations demonstrate the effectiveness of combining semantic models, context-aware reasoning and distributed stream processing to enable interoperable data workflows for Industry 5.0 environments.

-> 실시간 스트림 처리 플랫폼으로 다중 영상 소스 통합 관리 가능. 엣지 디바이스의 영상 분석 파이프라인 효율화에 필수.

### PyVision-RL: Forging Open Agentic Vision Models via RL (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.20739v1
- 점수: final 88.0

Reinforcement learning for agentic multimodal models often suffers from interaction collapse, where models learn to reduce tool usage and multi-turn reasoning, limiting the benefits of agentic behavior. We introduce PyVision-RL, a reinforcement learning framework for open-weight multimodal models that stabilizes training and sustains interaction. Our approach combines an oversampling-filtering-ranking rollout strategy with an accumulative tool reward to prevent collapse and encourage multi-turn tool use. Using a unified training pipeline, we develop PyVision-Image and PyVision-Video for image and video understanding. For video reasoning, PyVision-Video employs on-demand context construction, selectively sampling task-relevant frames during reasoning to significantly reduce visual token usage. Experiments show strong performance and improved efficiency, demonstrating that sustained interaction and on-demand visual processing are critical for scalable multimodal agents.

-> PyVision-Video framework for video understanding applicable for sports scene analysis

### A Risk-Aware UAV-Edge Service Framework for Wildfire Monitoring and Emergency Response (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19742v1
- 점수: final 88.0

Wildfire monitoring demands timely data collection and processing for early detection and rapid response. UAV-assisted edge computing is a promising approach, but jointly minimizing end-to-end service response time while satisfying energy, revisit time, and capacity constraints remains challenging. We propose an integrated framework that co-optimizes UAV route planning, fleet sizing, and edge service provisioning for wildfire monitoring. The framework combines fire-history-weighted clustering to prioritize high-risk areas, Quality of Service (QoS)-aware edge assignment balancing proximity and computational load, 2-opt route optimization with adaptive fleet sizing, and a dynamic emergency rerouting mechanism. The key insight is that these subproblems are interdependent: clustering decisions simultaneously shape patrol efficiency and edge workloads, while capacity constraints feed back into feasible configurations. Experiments show that the proposed framework reduces average response time by 70.6--84.2%, energy consumption by 73.8--88.4%, and fleet size by 26.7--42.1% compared to GA, PSO, and greedy baselines. The emergency mechanism responds within 233 seconds, well under the 300-second deadline, with negligible impact on normal operations.

-> UAV 에지 자원 최적화 방법론이 스포츠 에지 디바이스에 전용. 응답 시간 70% 감소로 실시간 처리 가능.

### TraceVision: Trajectory-Aware Vision-Language Model for Human-Like Spatial Understanding (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19768v1
- 점수: final 85.6

Recent Large Vision-Language Models (LVLMs) demonstrate remarkable capabilities in image understanding and natural language generation. However, current approaches focus predominantly on global image understanding, struggling to simulate human visual attention trajectories and explain associations between descriptions and specific regions. We propose TraceVision, a unified vision-language model integrating trajectory-aware spatial understanding in an end-to-end framework. TraceVision employs a Trajectory-aware Visual Perception (TVP) module for bidirectional fusion of visual features and trajectory information. We design geometric simplification to extract semantic keypoints from raw trajectories and propose a three-stage training pipeline where trajectories guide description generation and region localization. We extend TraceVision to trajectory-guided segmentation and video scene understanding, enabling cross-frame tracking and temporal attention analysis. We construct the Reasoning-based Interactive Localized Narratives (RILN) dataset to enhance logical reasoning and interpretability. Extensive experiments on trajectory-guided captioning, text-guided trajectory prediction, understanding, and segmentation demonstrate that TraceVision achieves state-of-the-art performance, establishing a foundation for intuitive spatial interaction and interpretable visual understanding.

-> 궤적 인식 기술로 운동 선수 움직임 추적 정확도 향상. 하이라이트 자동 추출 핵심 기능으로 적용 가능.

### Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19615v1
- 점수: final 85.6

Vision language models (VLMs) have achieved remarkable success in broad visual understanding, yet they remain challenged by object-centric reasoning on rare objects due to the scarcity of such instances in pretraining data. While prior efforts alleviate this issue by retrieving additional data or introducing stronger vision encoders, these methods are still computationally intensive during finetuning VLMs and don't fully exploit the original training data. In this paper, we introduce an efficient plug-and-play module that substantially improves VLMs' reasoning over rare objects by refining visual tokens and enriching input text prompts, without VLMs finetuning. Specifically, we propose to learn multi-modal class embeddings for rare objects by leveraging prior knowledge from vision foundation models and synonym-augmented text descriptions, compensating for limited training examples. These embeddings refine the visual tokens in VLMs through a lightweight attention-based enhancement module that improves fine-grained object details. In addition, we use the learned embeddings as object-aware detectors to generate informative hints, which are injected into the text prompts to help guide the VLM's attention toward relevant image regions. Experiments on two benchmarks show consistent and substantial gains for pretrained VLMs in rare object recognition and reasoning. Further analysis reveals how our method strengthens the VLM's ability to focus on and reason about rare objects.

-> 경량 플러그인 모듈로 희귀 스포츠 동작 인식 성능 향상. 엣지 디바이스 VLM 최적화에 직접 적용 가능.

### CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19605v1
- 점수: final 85.6

Multimodal learning aims to capture both shared and private information from multiple modalities. However, existing methods that project all modalities into a single latent space for fusion often overlook the asynchronous, multi-level semantic structure of multimodal data. This oversight induces semantic misalignment and error propagation, thereby degrading representation quality. To address this issue, we propose Cross-Level Co-Representation (CLCR), which explicitly organizes each modality's features into a three-level semantic hierarchy and specifies level-wise constraints for cross-modal interactions. First, a semantic hierarchy encoder aligns shallow, mid, and deep features across modalities, establishing a common basis for interaction. And then, at each level, an Intra-Level Co-Exchange Domain (IntraCED) factorizes features into shared and private subspaces and restricts cross-modal attention to the shared subspace via a learnable token budget. This design ensures that only shared semantics are exchanged and prevents leakage from private channels. To integrate information across levels, the Inter-Level Co-Aggregation Domain (InterCAD) synchronizes semantic scales using learned anchors, selectively fuses the shared representations, and gates private cues to form a compact task representation. We further introduce regularization terms to enforce separation of shared and private features and to minimize cross-level interference. Experiments on six benchmarks spanning emotion recognition, event localization, sentiment analysis, and action recognition show that CLCR achieves strong performance and generalizes well across tasks.

-> 멀티모달 학습으로 동작 인식 정확도 향상. 영상-모션 데이터 협업 표현이 스포츠 분석 핵심.

### HDR Reconstruction Boosting with Training-Free and Exposure-Consistent Diffusion (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19706v1
- 점수: final 84.8

Single LDR to HDR reconstruction remains challenging for over-exposed regions where traditional methods often fail due to complete information loss. We present a training-free approach that enhances existing indirect and direct HDR reconstruction methods through diffusion-based inpainting. Our method combines text-guided diffusion models with SDEdit refinement to generate plausible content in over-exposed areas while maintaining consistency across multi-exposure LDR images. Unlike previous approaches requiring extensive training, our method seamlessly integrates with existing HDR reconstruction techniques through an iterative compensation mechanism that ensures luminance coherence across multiple exposures. We demonstrate significant improvements in both perceptual quality and quantitative metrics on standard HDR datasets and in-the-wild captures. Results show that our method effectively recovers natural details in challenging scenarios while preserving the advantages of existing HDR reconstruction pipelines. Project page: https://github.com/EusdenLin/HDR-Reconstruction-Boosting

-> 이 논문은 단일 LDR 영상을 HDR로 재구성하는 방법을 제안합니다. 핵심은 학습 없이 확산 모델을 활용해 과다 노출 영역을 자연스럽게 복원하는 것입니다. 우리 프로젝트의 영상 보정 기능에 직접 적용 가능해 하이라이트 영상의 화질을 개선할 수 있습니다.

### ORION: ORthonormal Text Encoding for Universal VLM AdaptatION (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19530v1
- 점수: final 84.0

Vision language models (VLMs) have demonstrated remarkable generalization across diverse tasks, yet their performance remains constrained by the quality and geometry of the textual prototypes used to represent classes. Standard zero shot classifiers, derived from frozen text encoders and handcrafted prompts, may yield correlated or weakly separated embeddings that limit task specific discriminability. We introduce ORION, a text encoder fine tuning framework that improves pretrained VLMs using only class names. Our method optimizes, via low rank adaptation, a novel loss integrating two terms, one promoting pairwise orthogonality between the textual representations of the classes of a given task and the other penalizing deviations from the initial class prototypes. Furthermore, we provide a probabilistic interpretation of our orthogonality penalty, connecting it to the general maximum likelihood estimation (MLE) principle via Huygens theorem. We report extensive experiments on 11 benchmarks and three large VLM backbones, showing that the refined textual embeddings yield powerful replacements for the standard CLIP prototypes. Added as plug and play module on top of various state of the art methods, and across different prediction settings (zero shot, few shot and test time adaptation), ORION improves the performance consistently and significantly.

-> 이 논문은 VLM 텍스트 인코딩 최적화 방법을 제안합니다. 핵심은 클래스 간 텍스트 임베딩의 직교성을 강화해 분류 성능을 높이는 것입니다. 스포츠 자세 분석 정확도 향상에 기여해 프로젝트의 AI 분석 기능에 중요합니다.

### Laplacian Multi-scale Flow Matching for Generative Modeling (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19461v1
- 점수: final 84.0

In this paper, we present Laplacian multiscale flow matching (LapFlow), a novel framework that enhances flow matching by leveraging multi-scale representations for image generative modeling. Our approach decomposes images into Laplacian pyramid residuals and processes different scales in parallel through a mixture-of-transformers (MoT) architecture with causal attention mechanisms. Unlike previous cascaded approaches that require explicit renoising between scales, our model generates multi-scale representations in parallel, eliminating the need for bridging processes. The proposed multi-scale architecture not only improves generation quality but also accelerates the sampling process and promotes scaling flow matching methods. Through extensive experimentation on CelebA-HQ and ImageNet, we demonstrate that our method achieves superior sample quality with fewer GFLOPs and faster inference compared to single-scale and multi-scale flow matching baselines. The proposed model scales effectively to high-resolution generation (up to 1024$\times$1024) while maintaining lower computational overhead.

-> 이 논문은 다중 스케일 플로우 매칭을 통한 이미지 생성 방법을 제안합니다. 핵심은 라플라시안 피라미드와 병렬 처리를 활용해 고해상도 생성을 가속하는 것입니다. 영상 보정 및 사진 생성 기능에 적용 가능해 프로젝트의 콘텐츠 제작 효율성을 높입니다.

### Training-Free Generative Modeling via Kernelized Stochastic Interpolants (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.20070v1
- 점수: final 84.0

We develop a kernel method for generative modeling within the stochastic interpolant framework, replacing neural network training with linear systems. The drift of the generative SDE is $\hat b_t(x) = \nablaφ(x)^\topη_t$, where $η_t\in\R^P$ solves a $P\times P$ system computable from data, with $P$ independent of the data dimension $d$. Since estimates are inexact, the diffusion coefficient $D_t$ affects sample quality; the optimal $D_t^*$ from Girsanov diverges at $t=0$, but this poses no difficulty and we develop an integrator that handles it seamlessly. The framework accommodates diverse feature maps -- scattering transforms, pretrained generative models etc. -- enabling training-free generation and model combination. We demonstrate the approach on financial time series, turbulence, and image generation.

-> 이 논문은 커널 기반 생성 모델링 방법을 제안합니다. 핵심은 신경망 없이 선형 시스템으로 학습 없는 생성을 가능케 하는 것입니다. 영상 보정 및 이미지 생성에 유연하게 적용되어 프로젝트의 실시간 보정 기능을 강화합니다.

### CQ-CiM: Hardware-Aware Embedding Shaping for Robust CiM-Based Retrieval (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.20083v1
- 점수: final 84.0

Deploying Retrieval-Augmented Generation (RAG) on edge devices is in high demand, but is hindered by the latency of massive data movement and computation on traditional architectures. Compute-in-Memory (CiM) architectures address this bottleneck by performing vector search directly within their crossbar structure. However, CiM's adoption for RAG is limited by a fundamental ``representation gap,'' as high-precision, high-dimension embeddings are incompatible with CiM's low-precision, low-dimension array constraints. This gap is compounded by the diversity of CiM implementations (e.g., SRAM, ReRAM, FeFET), each with unique designs (e.g., 2-bit cells, 512x512 arrays). Consequently, RAG data must be naively reshaped to fit each target implementation. Current data shaping methods handle dimension and precision disjointly, which degrades data fidelity. This not only negates the advantages of CiM for RAG but also confuses hardware designers, making it unclear if a failure is due to the circuit design or the degraded input data. As a result, CiM adoption remains limited. In this paper, we introduce CQ-CiM, a unified, hardware-aware data shaping framework that jointly learns Compression and Quantization to produce CiM-compatible low-bit embeddings for diverse CiM designs. To the best of our knowledge, this is the first work to shape data for comprehensive CiM usage on RAG.

-> 이 논문은 CiM 기반 검색을 위한 임베딩 최적화 방법을 제안합니다. 핵심은 압축과 양자화를 결합해 엣지 디바이스 호환 저비트 임베딩을 생성하는 것입니다. RK3588 같은 엣지 하드웨어에서 RAG 효율성을 높여 프로젝트의 실시간 분석에 필수적입니다.

### EKF-Based Depth Camera and Deep Learning Fusion for UAV-Person Distance Estimation and Following in SAR Operations (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.20958v1
- 점수: final 84.0

Search and rescue (SAR) operations require rapid responses to save lives or property. Unmanned Aerial Vehicles (UAVs) equipped with vision-based systems support these missions through prior terrain investigation or real-time assistance during the mission itself. Vision-based UAV frameworks aid human search tasks by detecting and recognizing specific individuals, then tracking and following them while maintaining a safe distance. A key safety requirement for UAV following is the accurate estimation of the distance between camera and target object under real-world conditions, achieved by fusing multiple image modalities. UAVs with deep learning-based vision systems offer a new approach to the planning and execution of SAR operations. As part of the system for automatic people detection and face recognition using deep learning, in this paper we present the fusion of depth camera measurements and monocular camera-to-body distance estimation for robust tracking and following. Deep learning-based filtering of depth camera data and estimation of camera-to-body distance from a monocular camera are achieved with YOLO-pose, enabling real-time fusion of depth information using the Extended Kalman Filter (EKF) algorithm. The proposed subsystem, designed for use in drones, estimates and measures the distance between the depth camera and the human body keypoints, to maintain the safe distance between the drone and the human target. Our system provides an accurate estimated distance, which has been validated against motion capture ground truth data. The system has been tested in real time indoors, where it reduces the average errors, root mean square error (RMSE) and standard deviations of distance estimation up to 15,3\% in three tested scenarios.

-> This paper proposes a method for distance estimation and following of UAVs using depth camera and deep learning fusion. The core is YOLO-pose and EKF algorithm integration for real-time distance measurement.

### Multimodal Dataset Distillation Made Simple by Prototype-Guided Data Synthesis (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19756v1
- 점수: final 82.4

Recent advances in multimodal learning have achieved remarkable success across diverse vision-language tasks. However, such progress heavily relies on large-scale image-text datasets, making training costly and inefficient. Prior efforts in dataset filtering and pruning attempt to mitigate this issue, but still require relatively large subsets to maintain performance and fail under very small subsets. Dataset distillation offers a promising alternative, yet existing multimodal dataset distillation methods require full-dataset training and joint optimization of image pixels and text features, making them architecture-dependent and limiting cross-architecture generalization. To overcome this, we propose a learning-free dataset distillation framework that eliminates the need for large-scale training and optimization while enhancing generalization across architectures. Our method uses CLIP to extract aligned image-text embeddings, obtains prototypes, and employs an unCLIP decoder to synthesize images, enabling efficient and scalable multimodal dataset distillation. Extensive experiments demonstrate that our approach consistently outperforms optimization-based dataset distillation and subset selection methods, achieving state-of-the-art cross-architecture generalization.

-> 이 논문은 multimodal dataset distillation 방법을 제안합니다. 핵심은 CLIP을 이용해 학습 없이 이미지-텍스트 데이터를 증류하는 것입니다. 에지 디바이스의 학습 효율성을 높여 스포츠 영상 분석 모델의 훈련 비용과 시간을 크게 줄일 수 있습니다.

### SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.20792v1
- 점수: final 82.4

Modeling spinal motion is fundamental to understanding human biomechanics, yet remains underexplored in computer vision due to the spine's complex multi-joint kinematics and the lack of large-scale 3D annotations. We present a biomechanics-aware keypoint simulation framework that augments existing human pose datasets with anatomically consistent 3D spinal keypoints derived from musculoskeletal modeling. Using this framework, we create the first open dataset, named SIMSPINE, which provides sparse vertebra-level 3D spinal annotations for natural full-body motions in indoor multi-camera capture without external restraints. With 2.14 million frames, this enables data-driven learning of vertebral kinematics from subtle posture variations and bridges the gap between musculoskeletal simulation and computer vision. In addition, we release pretrained baselines covering fine-tuned 2D detectors, monocular 3D pose lifting models, and multi-view reconstruction pipelines, establishing a unified benchmark for biomechanically valid spine motion estimation. Specifically, our 2D spine baselines improve the state-of-the-art from 0.63 to 0.80 AUC in controlled environments, and from 0.91 to 0.93 AP for in-the-wild spine tracking. Together, the simulation framework and SIMSPINE dataset advance research in vision-based biomechanics, motion analysis, and digital human modeling by enabling reproducible, anatomically grounded 3D spine estimation under natural conditions.

-> 생체역학 시뮬레이션 프레임워크는 스포츠 동작 분석에 직접 적용 가능하여 선수 자세 및 동작 분석에 필수적

### Strategy-Supervised Autonomous Laparoscopic Camera Control via Event-Driven Graph Mining (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.20500v1
- 점수: final 80.0

Autonomous laparoscopic camera control must maintain a stable and safe surgical view under rapid tool-tissue interactions while remaining interpretable to surgeons. We present a strategy-grounded framework that couples high-level vision-language inference with low-level closed-loop control. Offline, raw surgical videos are parsed into camera-relevant temporal events (e.g., interaction, working-distance deviation, and view-quality degradation) and structured as attributed event graphs. Mining these graphs yields a compact set of reusable camera-handling strategy primitives, which provide structured supervision for learning. Online, a fine-tuned Vision-Language Model (VLM) processes the live laparoscopic view to predict the dominant strategy and discrete image-based motion commands, executed by an IBVS-RCM controller under strict safety constraints; optional speech input enables intuitive human-in-the-loop conditioning. On a surgeon-annotated dataset, event parsing achieves reliable temporal localization (F1-score 0.86), and the mined strategies show strong semantic alignment with expert interpretation (cluster purity 0.81). Extensive ex vivo experiments on silicone phantoms and porcine tissues demonstrate that the proposed system outperforms junior surgeons in standardized camera-handling evaluations, reducing field-of-view centering error by 35.26% and image shaking by 62.33%, while preserving smooth motion and stable working-distance regulation.

-> 전략 기반 자동 카메라 제어 방식은 경기 중요 순간을 자동으로 포착하여 하이라이트 영상 제작에 효과적

### Redefining the Down-Sampling Scheme of U-Net for Precision Biomedical Image Segmentation (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19412v1
- 점수: final 80.0

U-Net architectures have been instrumental in advancing biomedical image segmentation (BIS) but often struggle with capturing long-range information. One reason is the conventional down-sampling techniques that prioritize computational efficiency at the expense of information retention. This paper introduces a simple but effective strategy, we call it Stair Pooling, which moderates the pace of down-sampling and reduces information loss by leveraging a sequence of concatenated small and narrow pooling operations in varied orientations. Specifically, our method modifies the reduction in dimensionality within each 2D pooling step from $\frac{1}{4}$ to $\frac{1}{2}$. This approach can also be adapted for 3D pooling to preserve even more information. Such preservation aids the U-Net in more effectively reconstructing spatial details during the up-sampling phase, thereby enhancing its ability to capture long-range information and improving segmentation accuracy. Extensive experiments on three BIS benchmarks demonstrate that the proposed Stair Pooling can increase both 2D and 3D U-Net performance by an average of 3.8\% in Dice scores. Moreover, we leverage the transfer entropy to select the optimal down-sampling paths and quantitatively show how the proposed Stair Pooling reduces the information loss.

-> 이 논문은 U-Net의 Stair Pooling 기법을 제안합니다. 핵심은 정보 손실을 줄여 장거리 의존성을 포착하는 것입니다. 스포츠 동작 세분화 분석 정확도를 높여 선수의 자세 오류를 정밀하게 식별하는 데 필수적입니다.

### Using Unsupervised Domain Adaptation Semantic Segmentation for Pulmonary Embolism Detection in Computed Tomography Pulmonary Angiogram (CTPA) Images (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.19891v1
- 점수: final 80.0

While deep learning has demonstrated considerable promise in computer-aided diagnosis for pulmonary embolism (PE), practical deployment in Computed Tomography Pulmonary Angiography (CTPA) is often hindered by "domain shift" and the prohibitive cost of expert annotations. To address these challenges, an unsupervised domain adaptation (UDA) framework is proposed, utilizing a Transformer backbone and a Mean-Teacher architecture for cross-center semantic segmentation. The primary focus is placed on enhancing pseudo-label reliability by learning deep structural information within the feature space. Specifically, three modules are integrated and designed for this task: (1) a Prototype Alignment (PA) mechanism to reduce category-level distribution discrepancies; (2) Global and Local Contrastive Learning (GLCL) to capture both pixel-level topological relationships and global semantic representations; and (3) an Attention-based Auxiliary Local Prediction (AALP) module designed to reinforce sensitivity to small PE lesions by automatically extracting high-information slices from Transformer attention maps. Experimental validation conducted on cross-center datasets (FUMPE and CAD-PE) demonstrates significant performance gains. In the FUMPE -> CAD-PE task, the IoU increased from 0.1152 to 0.4153, while the CAD-PE -> FUMPE task saw an improvement from 0.1705 to 0.4302. Furthermore, the proposed method achieved a 69.9% Dice score in the CT -> MRI cross-modality task on the MMWHS dataset without utilizing any target-domain labels for model selection, confirming its robustness and generalizability for diverse clinical environments.

-> 이 논문은 unsupervised domain adaptation(UDA)을 이용한 segmentation 방법을 제안합니다. 핵심은 도메인 차이를 극복하는 Prototype Alignment입니다. 다양한 환경(실내/야외)에서 스포츠 동작 분석의 일관된 정확도를 보장합니다.

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
