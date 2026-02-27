# CAPP!C_AI 논문 리포트 (2026-02-27)

> 수집 68 | 필터 67 | 폐기 10 | 평가 67 | 출력 50 | 기준 50점

검색 윈도우: 2026-02-26T00:00:00+00:00 ~ 2026-02-27T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 23

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors

- arXiv: http://arxiv.org/abs/2602.23141v1
- PDF: https://arxiv.org/pdf/2602.23141v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 100.0 (llm_adjusted:100 = base:92 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
We propose a new unsupervised framework for online video stabilization. Unlike methods based on deep learning that require paired stable and unstable datasets, our approach instantiates the classical stabilization pipeline with three stages and incorporates a multithreaded buffering mechanism. This design addresses three longstanding challenges in end-to-end learning: limited data, poor controllability, and inefficiency on hardware with constrained resources. Existing benchmarks focus mainly on handheld videos with a forward view in visible light, which restricts the applicability of stabilization to domains such as UAV nighttime remote sensing. To fill this gap, we introduce a new multimodal UAV aerial video dataset (UAV-Test). Experiments show that our method consistently outperforms state-of-the-art online stabilizers in both quantitative metrics and visual quality, while achieving performance comparable to offline methods.

**선정 근거**
실시간 골격 그래프 구성 및 효율적 공간 추론 기술이 스포츠 장면 추적에 적합

**활용 인사이트**
계층적 온디맨드 전략을 적용해 선수 위치 추적 및 경기 전략 분석에 활용 가능

## 2위: SCOPE: Skeleton Graph-Based Computation-Efficient Framework for Autonomous UAV Exploration

- arXiv: http://arxiv.org/abs/2602.22707v1
- PDF: https://arxiv.org/pdf/2602.22707v1
- 발행일: 2026-02-26
- 카테고리: cs.RO
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Autonomous exploration in unknown environments is key for mobile robots, helping them perceive, map, and make decisions in complex areas. However, current methods often rely on frequent global optimization, suffering from high computational latency and trajectory oscillation, especially on resource-constrained edge devices. To address these limitations, we propose SCOPE, a novel framework that incrementally constructs a real-time skeletal graph and introduces Implicit Unknown Region Analysis for efficient spatial reasoning. The planning layer adopts a hierarchical on-demand strategy: the Proximal Planner generates smooth, high-frequency local trajectories, while the Region-Sequence Planner is activated only when necessary to optimize global visitation order. Comparative evaluations in simulation demonstrate that SCOPE achieves competitive exploration performance comparable to state-of-the-art global planners, while reducing computational cost by an average of 86.9%. Real-world experiments further validate the system's robustness and low latency in practical scenarios.

**선정 근거**
U-Net 아키텍처와 생성적 방법이 비디오/이미지 처리에 적용 가능

**활용 인사이트**
다중 스케일 특성 융합으로 영상 품질 향상 및 실시간 보정에 효과적

## 3위: U-Net-Based Generative Joint Source-Channel Coding for Wireless Image Transmission

- arXiv: http://arxiv.org/abs/2602.22691v1
- PDF: https://arxiv.org/pdf/2602.22691v1
- 발행일: 2026-02-26
- 카테고리: eess.IV
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Deep learning (DL)-based joint source-channel coding (JSCC) methods have achieved remarkable success in wireless image transmission. However, these methods either focus on conventional distortion metrics that do not necessarily yield high perceptual quality or incur high computational complexity. In this paper, we propose two DL-based JSCC (DeepJSCC) methods that leverage deep generative architectures for wireless image transmission. Specifically, we propose G-UNet-JSCC, a scheme comprising an encoder and a U-Net-based generator serving as the decoder. Its skip connections enable multi-scale feature fusion to improve both pixel-level fidelity and perceptual quality of reconstructed images by integrating low- and high-level features. To further enhance pixel-level fidelity, the encoder and the U-Net-based decoder are jointly optimized using a weighted sum of structural similarity and mean-squared error (MSE) losses. Building upon G-UNet-JSCC, we further develop a DeepJSCC method called cGAN-JSCC, where the decoder is enhanced through adversarial training. In this scheme, we retain the encoder of G-UNet-JSCC and adversarially train the decoder's generator against a patch-based discriminator. cGAN-JSCC employs a two-stage training procedure. The outer stage trains the encoder and the decoder end-to-end using an MSE loss, while the inner stage adversarially trains the decoder's generator and the discriminator by minimizing a joint loss combining adversarial and distortion losses. Simulation results demonstrate that the proposed methods achieve superior pixel-level fidelity and perceptual quality on both high- and low-resolution images. For low-resolution images, cGAN-JSCC achieves better reconstruction performance and greater robustness to channel variations than G-UNet-JSCC.

**선정 근거**
U-Net 아키텍처와 생성적 방법이 비디오/이미지 처리에 적용 가능

## 4위: UniScale: Unified Scale-Aware 3D Reconstruction for Multi-View Understanding via Prior Injection for Robotic Perception

- arXiv: http://arxiv.org/abs/2602.23224v1
- PDF: https://arxiv.org/pdf/2602.23224v1
- 발행일: 2026-02-26
- 카테고리: cs.CV, cs.RO
- 점수: final 94.4 (llm_adjusted:93 = base:85 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
We present UniScale, a unified, scale-aware multi-view 3D reconstruction framework for robotic applications that flexibly integrates geometric priors through a modular, semantically informed design. In vision-based robotic navigation, the accurate extraction of environmental structure from raw image sequences is critical for downstream tasks. UniScale addresses this challenge with a single feed-forward network that jointly estimates camera intrinsics and extrinsics, scale-invariant depth and point maps, and the metric scale of a scene from multi-view images, while optionally incorporating auxiliary geometric priors when available. By combining global contextual reasoning with camera-aware feature representations, UniScale is able to recover the metric-scale of the scene. In robotic settings where camera intrinsics are known, they can be easily incorporated to improve performance, with additional gains obtained when camera poses are also available. This co-design enables robust, metric-aware 3D reconstruction within a single unified model. Importantly, UniScale does not require training from scratch, and leverages world priors exhibited in pre-existing models without geometric encoding strategies, making it particularly suitable for resource-constrained robotic teams. We evaluate UniScale on multiple benchmarks, demonstrating strong generalization and consistent performance across diverse environments. We will release our implementation upon acceptance.

**선정 근거**
자원 제약이 있는 엣지 디바이스에서 스포츠 촬영을 위한 3D 재구성 기술

**활용 인사이트**
다중 뷰 이미지로부터 장치의 내부/외부 파라미터와 깊이 맵을 동시 추출해 정확한 3D 모델링

## 5위: BRIDGE: Borderless Reconfiguration for Inclusive and Diverse Gameplay Experience via Embodiment Transformation

- arXiv: http://arxiv.org/abs/2602.23288v1
- PDF: https://arxiv.org/pdf/2602.23288v1
- 발행일: 2026-02-26
- 카테고리: cs.HC
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Training resources for parasports are limited, reducing opportunities for athletes and coaches to engage with sport-specific movements and tactical coordination. To address this gap, we developed BRIDGE, a system that integrates a reconstruction pipeline, which detects and tracks players from broadcast video to generate 3D play sequences, with an embodiment-aware visualization framework that decomposes head, trunk, and wheelchair base orientations to represent attention, intent, and mobility. We evaluated BRIDGE in two controlled studies with 20 participants (10 national wheelchair basketball team players and 10 amateur players). The results showed that BRIDGE significantly enhanced the perceived naturalness of player postures and made tactical intentions easier to understand. In addition, it supported functional classification by realistically conveying players' capabilities, which in turn improved participants' sense of self-efficacy. This work advances inclusive sports learning and accessible coaching practices, contributing to more equitable access to tactical resources in parasports.

**선정 근거**
스포츠 비디오 분석과 전략 이해를 위한 3D 재구성 시스템

**활용 인사이트**
선수들의 움직임을 3D 시퀀스로 변환해 자세와 전략 분석을 통해 하이라이트 영상 생성

## 6위: MovieTeller: Tool-augmented Movie Synopsis with ID Consistent Progressive Abstraction

- arXiv: http://arxiv.org/abs/2602.23228v1
- PDF: https://arxiv.org/pdf/2602.23228v1
- 발행일: 2026-02-26
- 카테고리: cs.CV, cs.AI
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
With the explosive growth of digital entertainment, automated video summarization has become indispensable for applications such as content indexing, personalized recommendation, and efficient media archiving. Automatic synopsis generation for long-form videos, such as movies and TV series, presents a significant challenge for existing Vision-Language Models (VLMs). While proficient at single-image captioning, these general-purpose models often exhibit critical failures in long-duration contexts, primarily a lack of ID-consistent character identification and a fractured narrative coherence. To overcome these limitations, we propose MovieTeller, a novel framework for generating movie synopses via tool-augmented progressive abstraction. Our core contribution is a training-free, tool-augmented, fact-grounded generation process. Instead of requiring costly model fine-tuning, our framework directly leverages off-the-shelf models in a plug-and-play manner. We first invoke a specialized face recognition model as an external "tool" to establish Factual Groundings--precise character identities and their corresponding bounding boxes. These groundings are then injected into the prompt to steer the VLM's reasoning, ensuring the generated scene descriptions are anchored to verifiable facts. Furthermore, our progressive abstraction pipeline decomposes the summarization of a full-length movie into a multi-stage process, effectively mitigating the context length limitations of current VLMs. Experiments demonstrate that our approach yields significant improvements in factual accuracy, character consistency, and overall narrative coherence compared to end-to-end baselines.

**선정 근거**
비디오 요약 및 캐릭터 일관성 유지 기술이 스포츠 하이라이트 자동 생성에 적용 가능

**활용 인사이트**
MovieTeller의 툴 증강 프로그레시브 추상화 기법을 스포츠 영상 요약에 적용하여 선수별 주요 장면 자동 생성 가능

## 7위: Velocity and stroke rate reconstruction of canoe sprint team boats based on panned and zoomed video recordings

- arXiv: http://arxiv.org/abs/2602.22941v1
- PDF: https://arxiv.org/pdf/2602.22941v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 92.0 (llm_adjusted:90 = base:90 + bonus:+0)

**개요**
Pacing strategies, defined by velocity and stroke rate profiles, are essential for peak performance in canoe sprint. While GPS is the gold standard for analysis, its limited availability necessitates automated video-based solutions. This paper presents an extended framework for reconstructing performance metrics from panned and zoomed video recordings across all sprint disciplines (K1-K4, C1-C2) and distances (200m-500m). Our method utilizes YOLOv8 for buoy and athlete detection, leveraging the known buoy grid to estimate homographies. We generalized the estimation of the boat position by means of learning a boat-specific athlete offset using a U-net based boat tip calibration. Further, we implement a robust tracking scheme using optical flow to adapt to multi-athlete boat types. Finally, we introduce methods to extract stroke rate information from either pose estimations or the athlete bounding boxes themselves. Evaluation against GPS data from elite competitions yields a velocity RRMSE of 0.020 +- 0.011 (rho = 0.956) and a stroke rate RRMSE of 0.022 +- 0.024 (rho = 0.932). The methods provide coaches with highly accurate, automated feedback without requiring on-boat sensors or manual annotation.

**선정 근거**
스포츠 비디오 분석을 위한 컴퓨터 비전 기술로 프로젝트와 직접 관련 있으나 특정 스포츠(카누)에 국한됨

**활용 인사이트**
YOLOv8과 광학 흐름 기반 추적 기술을 다양한 스포츠로 확장하여 선수 속도 및 동작 패턴 분석 가능

## 8위: Align then Adapt: Rethinking Parameter-Efficient Transfer Learning in 4D Perception

- arXiv: http://arxiv.org/abs/2602.23069v1
- PDF: https://arxiv.org/pdf/2602.23069v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 92.0 (llm_adjusted:90 = base:82 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Point cloud video understanding is critical for robotics as it accurately encodes motion and scene interaction. We recognize that 4D datasets are far scarcer than 3D ones, which hampers the scalability of self-supervised 4D models. A promising alternative is to transfer 3D pre-trained models to 4D perception tasks. However, rigorous empirical analysis reveals two critical limitations that impede transfer capability: overfitting and the modality gap. To overcome these challenges, we develop a novel "Align then Adapt" (PointATA) paradigm that decomposes parameter-efficient transfer learning into two sequential stages. Optimal-transport theory is employed to quantify the distributional discrepancy between 3D and 4D datasets, enabling our proposed point align embedder to be trained in Stage 1 to alleviate the underlying modality gap. To mitigate overfitting, an efficient point-video adapter and a spatial-context encoder are integrated into the frozen 3D backbone to enhance temporal modeling capacity in Stage 2. Notably, with the above engineering-oriented designs, PointATA enables a pre-trained 3D model without temporal knowledge to reason about dynamic video content at a smaller parameter cost compared to previous work. Extensive experiments show that PointATA can match or even outperform strong full fine-tuning models, whilst enjoying the advantage of parameter efficiency, e.g. 97.21 \% accuracy on 3D action recognition, $+8.7 \%$ on 4 D action segmentation, and 84.06\% on 4D semantic segmentation.

**선정 근거**
4D perception technology with parameter-efficient transfer learning applicable to edge devices for sports video analysis

**활용 인사이트**
3D 모델을 4D 스포츠 영상 분석으로 전이 학습하여 에지 디바이스에서 실시간 동작 분석 가능

## 9위: UCM: Unifying Camera Control and Memory with Time-aware Positional Encoding Warping for World Models

- arXiv: http://arxiv.org/abs/2602.22960v1
- PDF: https://arxiv.org/pdf/2602.22960v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 90.4 (llm_adjusted:88 = base:78 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
World models based on video generation demonstrate remarkable potential for simulating interactive environments but face persistent difficulties in two key areas: maintaining long-term content consistency when scenes are revisited and enabling precise camera control from user-provided inputs. Existing methods based on explicit 3D reconstruction often compromise flexibility in unbounded scenarios and fine-grained structures. Alternative methods rely directly on previously generated frames without establishing explicit spatial correspondence, thereby constraining controllability and consistency. To address these limitations, we present UCM, a novel framework that unifies long-term memory and precise camera control via a time-aware positional encoding warping mechanism. To reduce computational overhead, we design an efficient dual-stream diffusion transformer for high-fidelity generation. Moreover, we introduce a scalable data curation strategy utilizing point-cloud-based rendering to simulate scene revisiting, facilitating training on over 500K monocular videos. Extensive experiments on real-world and synthetic benchmarks demonstrate that UCM significantly outperforms state-of-the-art methods in long-term scene consistency, while also achieving precise camera controllability in high-fidelity video generation.

**선정 근거**
카메라 제어 및 일관성 유지 기술이 스포츠 경기 자동 촬영에 활용 가능

**활용 인사이트**
UCM의 시간 인코딩 왜핑 메커니즘을 활용해 스포츠 장면의 일관성 유지 및 카메라 제어 정밀도 향상

## 10위: Learning Continuous Wasserstein Barycenter Space for Generalized All-in-One Image Restoration

- arXiv: http://arxiv.org/abs/2602.23169v1
- PDF: https://arxiv.org/pdf/2602.23169v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
Despite substantial advances in all-in-one image restoration for addressing diverse degradations within a unified model, existing methods remain vulnerable to out-of-distribution degradations, thereby limiting their generalization in real-world scenarios. To tackle the challenge, this work is motivated by the intuition that multisource degraded feature distributions are induced by different degradation-specific shifts from an underlying degradation-agnostic distribution, and recovering such a shared distribution is thus crucial for achieving generalization across degradations. With this insight, we propose BaryIR, a representation learning framework that aligns multisource degraded features in the Wasserstein barycenter (WB) space, which models a degradation-agnostic distribution by minimizing the average of Wasserstein distances to multisource degraded distributions. We further introduce residual subspaces, whose embeddings are mutually contrasted while remaining orthogonal to the WB embeddings. Consequently, BaryIR explicitly decouples two orthogonal spaces: a WB space that encodes the degradation-agnostic invariant contents shared across degradations, and residual subspaces that adaptively preserve the degradation-specific knowledge. This disentanglement mitigates overfitting to in-distribution degradations and enables adaptive restoration grounded on the degradation-agnostic shared invariance. Extensive experiments demonstrate that BaryIR performs competitively against state-of-the-art all-in-one methods. Notably, BaryIR generalizes well to unseen degradations (\textit{e.g.,} types and levels) and shows remarkable robustness in learning generalized features, even when trained on limited degradation types and evaluated on real-world data with mixed degradations.

**선정 근거**
이미지 복원 및 향상 기술로 스포츠 영상 보정에 직접 적용 가능

**활용 인사이트**
다양한 품질 저하된 스포츠 영상을 통합 복원하여 사진처럼 고품질 콘텐츠 생성

## 11위: Towards Long-Form Spatio-Temporal Video Grounding

- arXiv: http://arxiv.org/abs/2602.23294v1
- PDF: https://arxiv.org/pdf/2602.23294v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 88.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
In real scenarios, videos can span several minutes or even hours. However, existing research on spatio-temporal video grounding (STVG), given a textual query, mainly focuses on localizing targets in short videos of tens of seconds, typically less than one minute, which limits real-world applications. In this paper, we explore Long-Form STVG (LF-STVG), which aims to locate targets in long-term videos. Compared with short videos, long-term videos contain much longer temporal spans and more irrelevant information, making it difficult for existing STVG methods that process all frames at once. To address this challenge, we propose an AutoRegressive Transformer architecture for LF-STVG, termed ART-STVG. Unlike conventional STVG methods that require the entire video sequence to make predictions at once, ART-STVG treats the video as streaming input and processes frames sequentially, enabling efficient handling of long videos. To model spatio-temporal context, we design spatial and temporal memory banks and apply them to the decoders. Since memories from different moments are not always relevant to the current frame, we introduce simple yet effective memory selection strategies to provide more relevant information to the decoders, significantly improving performance. Furthermore, instead of parallel spatial and temporal localization, we propose a cascaded spatio-temporal design that connects the spatial decoder to the temporal decoder, allowing fine-grained spatial cues to assist complex temporal localization in long videos. Experiments on newly extended LF-STVG datasets show that ART-STVG significantly outperforms state-of-the-art methods, while achieving competitive performance on conventional short-form STVG.

**선정 근거**
Long-form video processing technology applicable to sports game analysis

**활용 인사이트**
긴 경기 영상에서 주요 장면을 자동으로 식별하여 하이라이트 편집 기능 구현 가능

## 12위: LoR-LUT: Learning Compact 3D Lookup Tables via Low-Rank Residuals

- arXiv: http://arxiv.org/abs/2602.22607v1
- PDF: https://arxiv.org/pdf/2602.22607v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 88.0 (llm_adjusted:85 = base:85 + bonus:+0)

**개요**
We present LoR-LUT, a unified low-rank formulation for compact and interpretable 3D lookup table (LUT) generation. Unlike conventional 3D-LUT-based techniques that rely on fusion of basis LUTs, which are usually dense tensors, our unified approach extends the current framework by jointly using residual corrections, which are in fact low-rank tensors, together with a set of basis LUTs. The approach described here improves the existing perceptual quality of an image, which is primarily due to the technique's novel use of residual corrections. At the same time, we achieve the same level of trilinear interpolation complexity, using a significantly smaller number of network, residual corrections, and LUT parameters. The experimental results obtained from LoR-LUT, which is trained on the MIT-Adobe FiveK dataset, reproduce expert-level retouching characteristics with high perceptual fidelity and a sub-megabyte model size. Furthermore, we introduce an interactive visualization tool, termed LoR-LUT Viewer, which transforms an input image into the LUT-adjusted output image, via a number of slidebars that control different parameters. The tool provides an effective way to enhance interpretability and user confidence in the visual results. Overall, our proposed formulation offers a compact, interpretable, and efficient direction for future LUT-based image enhancement and style transfer.

**선정 근거**
컴팩트 3D 룩업 테이블 기술로 스포츠 영상 및 이미지 보정에 직접적으로 적용 가능

**활용 인사이트**
스포츠 영상과 이미지의 품질을 향상시키면서도 적은 파라미터로 edge 디바이스에서 실시간 처리 가능

## 13위: Doubly Adaptive Channel and Spatial Attention for Semantic Image Communication by IoT Devices

- arXiv: http://arxiv.org/abs/2602.22794v1
- PDF: https://arxiv.org/pdf/2602.22794v1
- 발행일: 2026-02-26
- 카테고리: cs.LG
- 점수: final 88.0 (llm_adjusted:85 = base:80 + bonus:+5)
- 플래그: 엣지

**개요**
Internet of Things (IoT) networks face significant challenges such as limited communication bandwidth, constrained computational and energy resources, and highly dynamic wireless channel conditions. Utilization of deep neural networks (DNNs) combined with semantic communication has emerged as a promising paradigm to address these limitations. Deep joint source-channel coding (DJSCC) has recently been proposed to enable semantic communication of images. Building upon the original DJSCC formulation, low-complexity attention-style architectures has been added to the DNNs for further performance enhancement. As a main hurdle, training these DNNs separately for various signal-to-noise ratios (SNRs) will amount to excessive storage or communication overhead, which can not be maintained by small IoT devices. SNR Adaptive DJSCC (ADJSCC), has been proposed to train the DNNs once but feed the current SNR as part of the data to the channel-wise attention mechanism. We improve upon ADJSCC by a simultaneous utilization of doubly adaptive channel-wise and spatial attention modules at both transmitter and receiver. These modules dynamically adjust to varying channel conditions and spatial feature importance, enabling robust and efficient feature extraction and semantic information recovery. Simulation results corroborate that our proposed doubly adaptive DJSCC (DA-DJSCC) significantly improves upon ADJSCC in several performance criteria, while incurring a mild increase in complexity. These facts render DA-DJSCC a desirable choice for semantic communication in performance demanding but low-complexity IoT networks.

**선정 근거**
IoT device image communication with adaptive attention, applicable to edge device video processing

**활용 인사이트**
제한된 리소스를 가진 edge 디바이스에서도 다양한 통신 환경에 적응하여 효율적인 영상 처리 가능

## 14위: Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation

- arXiv: http://arxiv.org/abs/2602.23188v1
- PDF: https://arxiv.org/pdf/2602.23188v1
- 발행일: 2026-02-26
- 카테고리: cs.LG, physics.flu-dyn
- 점수: final 88.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
We propose an efficient retraining strategy for a parameterized Reduced Order Model (ROM) that attains accuracy comparable to full retraining while requiring only a fraction of the computational time and relying solely on sparse observations of the full system. The architecture employs an encode-process-decode structure: a Variational Autoencoder (VAE) to perform dimensionality reduction, and a transformer network to evolve the latent states and model the dynamics. The ROM is parameterized by an external control variable, the Reynolds number in the Navier-Stokes setting, with the transformer exploiting attention mechanisms to capture both temporal dependencies and parameter effects. The probabilistic VAE enables stochastic sampling of trajectory ensembles, providing predictive means and uncertainty quantification through the first two moments. After initial training on a limited set of dynamical regimes, the model is adapted to out-of-sample parameter regions using only sparse data. Its probabilistic formulation naturally supports ensemble generation, which we employ within an ensemble Kalman filtering framework to assimilate data and reconstruct full-state trajectories from minimal observations. We further show that, for the dynamical system considered, the dominant source of error in out-of-sample forecasts stems from distortions of the latent manifold rather than changes in the latent dynamics. Consequently, retraining can be limited to the autoencoder, allowing for a lightweight, computationally efficient, real-time adaptation procedure with very sparse fine-tuning data.

**선정 근거**
Real-time adaptation techniques applicable to edge device AI processing

## 15위: FLIGHT: Fibonacci Lattice-based Inference for Geometric Heading in real-Time

- arXiv: http://arxiv.org/abs/2602.23115v1
- PDF: https://arxiv.org/pdf/2602.23115v1
- 발행일: 2026-02-26
- 카테고리: cs.CV, cs.CG, cs.RO
- 점수: final 86.4 (llm_adjusted:83 = base:78 + bonus:+5)
- 플래그: 실시간

**개요**
Estimating camera motion from monocular video is a fundamental problem in computer vision, central to tasks such as SLAM, visual odometry, and structure-from-motion. Existing methods that recover the camera's heading under known rotation, whether from an IMU or an optimization algorithm, tend to perform well in low-noise, low-outlier conditions, but often decrease in accuracy or become computationally expensive as noise and outlier levels increase. To address these limitations, we propose a novel generalization of the Hough transform on the unit sphere (S(2)) to estimate the camera's heading. First, the method extracts correspondences between two frames and generates a great circle of directions compatible with each pair of correspondences. Then, by discretizing the unit sphere using a Fibonacci lattice as bin centers, each great circle casts votes for a range of directions, ensuring that features unaffected by noise or dynamic objects vote consistently for the correct motion direction. Experimental results on three datasets demonstrate that the proposed method is on the Pareto frontier of accuracy versus efficiency. Additionally, experiments on SLAM show that the proposed method reduces RMSE by correcting the heading during camera pose initialization.

**선정 근거**
실시간 카메라 모션 추정으로 자동 촬영 정확도 향상, 노이즈 많은 스포츠 환경에서도 효과적

**활용 인사이트**
피보나치 격자 기반 헤딩 추정을 슬로모션 촬영 및 자세 분석에 적용하여 하이라이트 자동 생성

## 16위: GSTurb: Gaussian Splatting for Atmospheric Turbulence Mitigation

- arXiv: http://arxiv.org/abs/2602.22800v1
- PDF: https://arxiv.org/pdf/2602.22800v1
- 코드: https://github.com/DuhlLiamz/3DGS_turbulence/tree/main
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 86.4 (llm_adjusted:83 = base:80 + bonus:+3)
- 플래그: 코드 공개

**개요**
Atmospheric turbulence causes significant image degradation due to pixel displacement (tilt) and blur, particularly in long-range imaging applications. In this paper, we propose a novel framework for atmospheric turbulence mitigation, GSTurb, which integrates optical flow-guided tilt correction and Gaussian splatting for modeling non-isoplanatic blur. The framework employs Gaussian parameters to represent tilt and blur, and optimizes them across multiple frames to enhance restoration. Experimental results on the ATSyn-static dataset demonstrate the effectiveness of our method, achieving a peak PSNR of 27.67 dB and SSIM of 0.8735. Compared to the state-of-the-art method, GSTurb improves PSNR by 1.3 dB (a 4.5% increase) and SSIM by 0.048 (a 5.8% increase). Additionally, on real datasets, including the TSRWGAN Real-World and CLEAR datasets, GSTurb outperforms existing methods, showing significant improvements in both qualitative and quantitative performance. These results highlight that combining optical flow-guided tilt correction with Gaussian splatting effectively enhances image restoration under both synthetic and real-world turbulence conditions. The code for this method will be available at https://github.com/DuhlLiamz/3DGS_turbulence/tree/main.

**선정 근거**
대기 왜곡으로 인한 스포츠 영상 저하 문제 해결에 필수적인 기술

**활용 인사이트**
가우시안 스플래팅 기술로 실시간 영상 보정 가능, 경기장 장거리 촬영 시 품질 향상

## 17위: Locally Adaptive Decay Surfaces for High-Speed Face and Landmark Detection with Event Cameras

- arXiv: http://arxiv.org/abs/2602.23101v1
- PDF: https://arxiv.org/pdf/2602.23101v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 84.0 (llm_adjusted:80 = base:70 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Event cameras record luminance changes with microsecond resolution, but converting their sparse, asynchronous output into dense tensors that neural networks can exploit remains a core challenge. Conventional histograms or globally-decayed time-surface representations apply fixed temporal parameters across the entire image plane, which in practice creates a trade-off between preserving spatial structure during still periods and retaining sharp edges during rapid motion. We introduce Locally Adaptive Decay Surfaces (LADS), a family of event representations in which the temporal decay at each location is modulated according to local signal dynamics. Three strategies are explored, based on event rate, Laplacian-of-Gaussian response, and high-frequency spectral energy. These adaptive schemes preserve detail in quiescent regions while reducing blur in regions of dense activity. Extensive experiments on the public data show that LADS consistently improves both face detection and facial landmark accuracy compared to standard non-adaptive representations. At 30 Hz, LADS achieves higher detection accuracy and lower landmark error than either baseline, and at 240 Hz it mitigates the accuracy decline typically observed at higher frequencies, sustaining 2.44 % normalized mean error for landmarks and 0.966 mAP50 in face detection. These high-frequency results even surpass the accuracy reported in prior works operating at 30 Hz, setting new benchmarks for event-based face analysis. Moreover, by preserving spatial structure at the representation stage, LADS supports the use of much lighter network architectures while still retaining real-time performance. These results highlight the importance of context-aware temporal integration for neuromorphic vision and point toward real-time, high-frequency human-computer interaction systems that exploit the unique advantages of event cameras.

**선정 근거**
고속 스포츠 동작 캡처에 적용 가능한 이벤트 카메라 기술로 빠른 움직임을 정확히 포착할 수 있습니다.

**활용 인사이트**
LADS 기술을 rk3588 에지 디바이스에 통합하여 실시간으로 선수 추적 및 빠른 동작 분석이 가능하며, 240Hz 고주파 성능으로 빠른 스포츠 장면을 명확하게 캡처할 수 있습니다.

## 18위: TEFL: Prediction-Residual-Guided Rolling Forecasting for Multi-Horizon Time Series

- arXiv: http://arxiv.org/abs/2602.22520v1
- PDF: https://arxiv.org/pdf/2602.22520v1
- 발행일: 2026-02-26
- 카테고리: cs.LG
- 점수: final 84.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
Time series forecasting plays a critical role in domains such as transportation, energy, and meteorology. Despite their success, modern deep forecasting models are typically trained to minimize point-wise prediction loss without leveraging the rich information contained in past prediction residuals from rolling forecasts - residuals that reflect persistent biases, unmodeled patterns, or evolving dynamics. We propose TEFL (Temporal Error Feedback Learning), a unified learning framework that explicitly incorporates these historical residuals into the forecasting pipeline during both training and evaluation. To make this practical in deep multi-step settings, we address three key challenges: (1) selecting observable multi-step residuals under the partial observability of rolling forecasts, (2) integrating them through a lightweight low-rank adapter to preserve efficiency and prevent overfitting, and (3) designing a two-stage training procedure that jointly optimizes the base forecaster and error module. Extensive experiments across 10 real-world datasets and 5 backbone architectures show that TEFL consistently improves accuracy, reducing MAE by 5-10% on average. Moreover, it demonstrates strong robustness under abrupt changes and distribution shifts, with error reductions exceeding 10% (up to 19.5%) in challenging scenarios. By embedding residual-based feedback directly into the learning process, TEFL offers a simple, general, and effective enhancement to modern deep forecasting systems.

**선정 근거**
스포츠 선수의 동작 패턴과 성적 예측을 위한 정확한 시계열 분석 필요

**활용 인사이트**
예측 오차 기반 학습으로 경기 전략 분석 및 선수 성과 예측 정확도 향상

## 19위: PackUV: Packed Gaussian UV Maps for 4D Volumetric Video

- arXiv: http://arxiv.org/abs/2602.23040v1
- PDF: https://arxiv.org/pdf/2602.23040v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:78 + bonus:+0)

**개요**
Volumetric videos offer immersive 4D experiences, but remain difficult to reconstruct, store, and stream at scale. Existing Gaussian Splatting based methods achieve high-quality reconstruction but break down on long sequences, temporal inconsistency, and fail under large motions and disocclusions. Moreover, their outputs are typically incompatible with conventional video coding pipelines, preventing practical applications.   We introduce PackUV, a novel 4D Gaussian representation that maps all Gaussian attributes into a sequence of structured, multi-scale UV atlas, enabling compact, image-native storage. To fit this representation from multi-view videos, we propose PackUV-GS, a temporally consistent fitting method that directly optimizes Gaussian parameters in the UV domain. A flow-guided Gaussian labeling and video keyframing module identifies dynamic Gaussians, stabilizes static regions, and preserves temporal coherence even under large motions and disocclusions. The resulting UV atlas format is the first unified volumetric video representation compatible with standard video codecs (e.g., FFV1) without losing quality, enabling efficient streaming within existing multimedia infrastructure.   To evaluate long-duration volumetric capture, we present PackUV-2B, the largest multi-view video dataset to date, featuring more than 50 synchronized cameras, substantial motion, and frequent disocclusions across 100 sequences and 2B (billion) frames. Extensive experiments demonstrate that our method surpasses existing baselines in rendering fidelity while scaling to sequences up to 30 minutes with consistent quality.

**선정 근거**
대용량 스포츠 영상 효율적 저장 및 공플랫폼에서의 실시간 스트리밍 필요

**활용 인사이트**
표준 비디오 코덱과 호환되는 4D 볼륨 비디오 포맷으로 콘텐츠 제작 및 공유 효율화

## 20위: Beyond Detection: Multi-Scale Hidden-Code for Natural Image Deepfake Recovery and Factual Retrieval

- arXiv: http://arxiv.org/abs/2602.22759v1
- PDF: https://arxiv.org/pdf/2602.22759v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Recent advances in image authenticity have primarily focused on deepfake detection and localization, leaving recovery of tampered contents for factual retrieval relatively underexplored. We propose a unified hidden-code recovery framework that enables both retrieval and restoration from post-hoc and in-generation watermarking paradigms. Our method encodes semantic and perceptual information into a compact hidden-code representation, refined through multi-scale vector quantization, and enhances contextual reasoning via conditional Transformer modules. To enable systematic evaluation for natural images, we construct ImageNet-S, a benchmark that provides paired image-label factual retrieval tasks. Extensive experiments on ImageNet-S demonstrate that our method exhibits promising retrieval and reconstruction performance while remaining fully compatible with diverse watermarking pipelines. This framework establishes a foundation for general-purpose image recovery beyond detection and localization.

**선정 근거**
이미지 복구 및 사실 검색 기술로 스포츠 영상 보정에 적용 가능하며 저조나 저품질 촬영 영상을 향상시킬 수 있습니다.

**활용 인사이트**
다중 벡터 양자화 기술을 활용하여 스포츠 영상의 품질을 향상시키고, 조건부 Transformer 모듈로 경기 장면의 맥락적 추론을 강화하여 하이라이트 편집의 정확도를 높일 수 있습니다.

## 21위: SOTAlign: Semi-Supervised Alignment of Unimodal Vision and Language Models via Optimal Transport

- arXiv: http://arxiv.org/abs/2602.23353v1
- PDF: https://arxiv.org/pdf/2602.23353v1
- 발행일: 2026-02-26
- 카테고리: cs.LG, cs.AI
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
The Platonic Representation Hypothesis posits that neural networks trained on different modalities converge toward a shared statistical model of the world. Recent work exploits this convergence by aligning frozen pretrained vision and language models with lightweight alignment layers, but typically relies on contrastive losses and millions of paired samples. In this work, we ask whether meaningful alignment can be achieved with substantially less supervision. We introduce a semi-supervised setting in which pretrained unimodal encoders are aligned using a small number of image-text pairs together with large amounts of unpaired data. To address this challenge, we propose SOTAlign, a two-stage framework that first recovers a coarse shared geometry from limited paired data using a linear teacher, then refines the alignment on unpaired samples via an optimal-transport-based divergence that transfers relational structure without overconstraining the target space. Unlike existing semi-supervised methods, SOTAlign effectively leverages unpaired images and text, learning robust joint embeddings across datasets and encoder pairs, and significantly outperforming supervised and semi-supervised baselines.

**선정 근거**
생물학적 조직 세포 계산 알고리즘으로 스포츠 분석에 직접 적용 불가

**활용 인사이트**
커널 카운터 개념은 스포츠 장면 객체 인식에 간접적 참고 가능성

## 22위: SUPERGLASSES: Benchmarking Vision Language Models as Intelligent Agents for AI Smart Glasses

- arXiv: http://arxiv.org/abs/2602.22683v1
- PDF: https://arxiv.org/pdf/2602.22683v1
- 발행일: 2026-02-26
- 카테고리: cs.CV, cs.AI
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
The rapid advancement of AI-powered smart glasses, one of the hottest wearable devices, has unlocked new frontiers for multimodal interaction, with Visual Question Answering (VQA) over external knowledge sources emerging as a core application. Existing Vision Language Models (VLMs) adapted to smart glasses are typically trained and evaluated on traditional multimodal datasets; however, these datasets lack the variety and realism needed to reflect smart glasses usage scenarios and diverge from their specific challenges, where accurately identifying the object of interest must precede any external knowledge retrieval. To bridge this gap, we introduce SUPERGLASSES, the first comprehensive VQA benchmark built on real-world data entirely collected by smart glasses devices. SUPERGLASSES comprises 2,422 egocentric image-question pairs spanning 14 image domains and 8 query categories, enriched with full search trajectories and reasoning annotations. We evaluate 26 representative VLMs on this benchmark, revealing significant performance gaps. To address the limitations of existing models, we further propose SUPERLENS, a multimodal smart glasses agent that enables retrieval-augmented answer generation by integrating automatic object detection, query decoupling, and multimodal web search. Our agent achieves state-of-the-art performance, surpassing GPT-4o by 2.19 percent, and highlights the need for task-specific solutions in smart glasses VQA scenarios.

**선정 근거**
AI 스마트 글래스 기술을 엣지 디바이스에 적용하여 실시간 스포츠 분석 가능

**활용 인사이트**
시각 질의 응답 기술을 활용하여 스포츠 장면에 대한 실시간 질의 응답 시스템 구축 가능

## 23위: Small Object Detection Model with Spatial Laplacian Pyramid Attention and Multi-Scale Features Enhancement in Aerial Images

- arXiv: http://arxiv.org/abs/2602.23031v1
- PDF: https://arxiv.org/pdf/2602.23031v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 76.0 (llm_adjusted:70 = base:70 + bonus:+0)

**개요**
Detecting objects in aerial images confronts some significant challenges, including small size, dense and non-uniform distribution of objects over high-resolution images, which makes detection inefficient. Thus, in this paper, we proposed a small object detection algorithm based on a Spatial Laplacian Pyramid Attention and Multi-Scale Feature Enhancement in aerial images. Firstly, in order to improve the feature representation of ResNet-50 on small objects, we presented a novel Spatial Laplacian Pyramid Attention (SLPA) module, which is integrated after each stage of ResNet-50 to identify and emphasize important local regions. Secondly, to enhance the model's semantic understanding and features representation, we designed a Multi-Scale Feature Enhancement Module (MSFEM), which is incorporated into the lateral connections of C5 layer for building Feature Pyramid Network (FPN). Finally, the features representation quality of traditional feature pyramid network will be affected because the features are not aligned when the upper and lower layers are fused. In order to handle it, we utilized deformable convolutions to align the features in the fusion processing of the upper and lower levels of the Feature Pyramid Network, which can help enhance the model's ability to detect and recognize small objects. The extensive experimental results on two benchmark datasets: VisDrone and DOTA demonstrate that our improved model performs better for small object detection in aerial images compared to the original algorithm.

**선정 근거**
공중 영상에서 작은 객체 검출 기술로 스포츠 선수나 장비 정확히 식별 가능

**활용 인사이트**
공간 라플라시안 피라미드 주의력 모듈로 멀티 스케일 특징 향상, 스포츠 장면에서 작은 객체 검출 성능 향상

## 24위: pQuant: Towards Effective Low-Bit Language Models via Decoupled Linear Quantization-Aware Training

- arXiv: http://arxiv.org/abs/2602.22592v1
- PDF: https://arxiv.org/pdf/2602.22592v1
- 발행일: 2026-02-26
- 카테고리: cs.LG, cs.CL
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Quantization-Aware Training from scratch has emerged as a promising approach for building efficient large language models (LLMs) with extremely low-bit weights (sub 2-bit), which can offer substantial advantages for edge deployment. However, existing methods still fail to achieve satisfactory accuracy and scalability. In this work, we identify a parameter democratization effect as a key bottleneck: the sensitivity of all parameters becomes homogenized, severely limiting expressivity. To address this, we propose pQuant, a method that decouples parameters by splitting linear layers into two specialized branches: a dominant 1-bit branch for efficient computation and a compact high-precision branch dedicated to preserving the most sensitive parameters. Through tailored feature scaling, we explicitly guide the model to allocate sensitive parameters to the high-precision branch. Furthermore, we extend this branch into multiple, sparsely-activated experts, enabling efficient capacity scaling. Extensive experiments indicate our pQuant achieves state-of-the-art performance in extremely low-bit quantization.

**선정 근거**
양자화 기술로 엣지 디바이스에서 AI 모델 효율적으로 실행 가능

**활용 인사이트**
pQuant 기법으로 rk3588 같은 엣지 디바이스에서 저비트 언어 모델 효율적으로 실행

## 25위: An automatic counting algorithm for the quantification and uncertainty analysis of the number of microglial cells trainable in small and heterogeneous datasets

- arXiv: http://arxiv.org/abs/2602.22974v1
- PDF: https://arxiv.org/pdf/2602.22974v1
- 발행일: 2026-02-26
- 카테고리: cs.CE, cs.CV, eess.IV, eess.SP, stat.ML
- 점수: final 74.4 (llm_adjusted:68 = base:60 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Counting immunopositive cells on biological tissues generally requires either manual annotation or (when available) automatic rough systems, for scanning signal surface and intensity in whole slide imaging. In this work, we tackle the problem of counting microglial cells in lumbar spinal cord cross-sections of rats by omitting cell detection and focusing only on the counting task. Manual cell counting is, however, a time-consuming task and additionally entails extensive personnel training. The classic automatic color-based methods roughly inform about the total labeled area and intensity (protein quantification) but do not specifically provide information on cell number. Since the images to be analyzed have a high resolution but a huge amount of pixels contain just noise or artifacts, we first perform a pre-processing generating several filtered images {(providing a tailored, efficient feature extraction)}. Then, we design an automatic kernel counter that is a non-parametric and non-linear method. The proposed scheme can be easily trained in small datasets since, in its basic version, it relies only on one hyper-parameter. However, being non-parametric and non-linear, the proposed algorithm is flexible enough to express all the information contained in rich and heterogeneous datasets as well (providing the maximum overfit if required). Furthermore, the proposed kernel counter also provides uncertainty estimation of the given prediction, and can directly tackle the case of receiving several expert opinions over the same image. Different numerical experiments with artificial and real datasets show very promising results. Related Matlab code is also provided.

**선정 근거**
Image counting algorithm for biological tissues, not directly applicable to sports analysis

## 26위: EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents

- arXiv: http://arxiv.org/abs/2602.23205v1
- PDF: https://arxiv.org/pdf/2602.23205v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 72.0 (llm_adjusted:65 = base:65 + bonus:+0)

**개요**
Human behaviors in the real world naturally encode rich, long-term contextual information that can be leveraged to train embodied agents for perception, understanding, and acting. However, existing capture systems typically rely on costly studio setups and wearable devices, limiting the large-scale collection of scene-conditioned human motion data in the wild. To address this, we propose EmbodMocap, a portable and affordable data collection pipeline using two moving iPhones. Our key idea is to jointly calibrate dual RGB-D sequences to reconstruct both humans and scenes within a unified metric world coordinate frame. The proposed method allows metric-scale and scene-consistent capture in everyday environments without static cameras or markers, bridging human motion and scene geometry seamlessly. Compared with optical capture ground truth, we demonstrate that the dual-view setting exhibits a remarkable ability to mitigate depth ambiguity, achieving superior alignment and reconstruction performance over single iphone or monocular models. Based on the collected data, we empower three embodied AI tasks: monocular human-scene-reconstruction, where we fine-tune on feedforward models that output metric-scale, world-space aligned humans and scenes; physics-based character animation, where we prove our data could be used to scale human-object interaction skills and scene-aware motion tracking; and robot motion control, where we train a humanoid robot via sim-to-real RL to replicate human motions depicted in videos. Experimental results validate the effectiveness of our pipeline and its contributions towards advancing embodied AI research.

**선정 근거**
휴대용 인간-장면 재구성 파이프라인으로 스포츠 촬영에 간접적으로 관련 있으나 연구 목적에 초점

**활용 인사이트**
두 대의 iPhone을 사용한 실외 인간 동작 캡처 기술은 스포츠 장면의 3D 재구성과 분석에 적용 가능

## 27위: OmniGAIA: Towards Native Omni-Modal AI Agents

- arXiv: http://arxiv.org/abs/2602.22897v1
- PDF: https://arxiv.org/pdf/2602.22897v1
- 발행일: 2026-02-26
- 카테고리: cs.AI, cs.CL, cs.CV, cs.LG, cs.MM
- 점수: final 72.0 (llm_adjusted:65 = base:65 + bonus:+0)

**개요**
Human intelligence naturally intertwines omni-modal perception -- spanning vision, audio, and language -- with complex reasoning and tool usage to interact with the world. However, current multi-modal LLMs are primarily confined to bi-modal interactions (e.g., vision-language), lacking the unified cognitive capabilities required for general AI assistants. To bridge this gap, we introduce OmniGAIA, a comprehensive benchmark designed to evaluate omni-modal agents on tasks necessitating deep reasoning and multi-turn tool execution across video, audio, and image modalities. Constructed via a novel omni-modal event graph approach, OmniGAIA synthesizes complex, multi-hop queries derived from real-world data that require cross-modal reasoning and external tool integration. Furthermore, we propose OmniAtlas, a native omni-modal foundation agent under tool-integrated reasoning paradigm with active omni-modal perception. Trained on trajectories synthesized via a hindsight-guided tree exploration strategy and OmniDPO for fine-grained error correction, OmniAtlas effectively enhances the tool-use capabilities of existing open-source models. This work marks a step towards next-generation native omni-modal AI assistants for real-world scenarios.

**선정 근거**
다중 모달 AI 에이전트 기술이 스포츠 영상 분석에 간접적으로 적용 가능

**활용 인사이트**
영상, 오디오, 텍스트를 통합해 경기 전략과 선수 동작 심층 분석 가능

## 28위: SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation

- arXiv: http://arxiv.org/abs/2602.23359v1
- PDF: https://arxiv.org/pdf/2602.23359v1
- 발행일: 2026-02-26
- 카테고리: cs.CV, cs.AI
- 점수: final 72.0 (llm_adjusted:65 = base:65 + bonus:+0)

**개요**
We identify occlusion reasoning as a fundamental yet overlooked aspect for 3D layout-conditioned generation. It is essential for synthesizing partially occluded objects with depth-consistent geometry and scale. While existing methods can generate realistic scenes that follow input layouts, they often fail to model precise inter-object occlusions. We propose SeeThrough3D, a model for 3D layout conditioned generation that explicitly models occlusions. We introduce an occlusion-aware 3D scene representation (OSCR), where objects are depicted as translucent 3D boxes placed within a virtual environment and rendered from desired camera viewpoint. The transparency encodes hidden object regions, enabling the model to reason about occlusions, while the rendered viewpoint provides explicit camera control during generation. We condition a pretrained flow based text-to-image image generation model by introducing a set of visual tokens derived from our rendered 3D representation. Furthermore, we apply masked self-attention to accurately bind each object bounding box to its corresponding textual description, enabling accurate generation of multiple objects without object attribute mixing. To train the model, we construct a synthetic dataset with diverse multi-object scenes with strong inter-object occlusions. SeeThrough3D generalizes effectively to unseen object categories and enables precise 3D layout control with realistic occlusions and consistent camera control.

**선정 근거**
Occlusion-aware 3D generation with potential applications in sports scene analysis

## 29위: SignVLA: A Gloss-Free Vision-Language-Action Framework for Real-Time Sign Language-Guided Robotic Manipulation

- arXiv: http://arxiv.org/abs/2602.22514v1
- PDF: https://arxiv.org/pdf/2602.22514v1
- 발행일: 2026-02-26
- 카테고리: cs.RO, cs.AI, eess.SY
- 점수: final 72.0 (llm_adjusted:65 = base:55 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
We present, to our knowledge, the first sign language-driven Vision-Language-Action (VLA) framework for intuitive and inclusive human-robot interaction. Unlike conventional approaches that rely on gloss annotations as intermediate supervision, the proposed system adopts a gloss-free paradigm and directly maps visual sign gestures to semantic instructions. This design reduces annotation cost and avoids the information loss introduced by gloss representations, enabling more natural and scalable multimodal interaction.   In this work, we focus on a real-time alphabet-level finger-spelling interface that provides a robust and low-latency communication channel for robotic control. Compared with large-scale continuous sign language recognition, alphabet-level interaction offers improved reliability, interpretability, and deployment feasibility in safety-critical embodied environments. The proposed pipeline transforms continuous gesture streams into coherent language commands through geometric normalization, temporal smoothing, and lexical refinement, ensuring stable and consistent interaction.   Furthermore, the framework is designed to support future integration of transformer-based gloss-free sign language models, enabling scalable word-level and sentence-level semantic understanding. Experimental results demonstrate the effectiveness of the proposed system in grounding sign-derived instructions into precise robotic actions under diverse interaction scenarios. These results highlight the potential of the framework to advance accessible, scalable, and multimodal embodied intelligence.

**선정 근거**
실시간 시각-언어-행동 프레임워크로 스포츠 동작 이해에 간접적으로 관련

## 30위: UFO-DETR: Frequency-Guided End-to-End Detector for UAV Tiny Objects

- arXiv: http://arxiv.org/abs/2602.22712v1
- PDF: https://arxiv.org/pdf/2602.22712v1
- 발행일: 2026-02-26
- 카테고리: cs.CV
- 점수: final 72.0 (llm_adjusted:65 = base:60 + bonus:+5)
- 플래그: 엣지

**개요**
Small target detection in UAV imagery faces significant challenges such as scale variations, dense distribution, and the dominance of small targets. Existing algorithms rely on manually designed components, and general-purpose detectors are not optimized for UAV images, making it difficult to balance accuracy and complexity. To address these challenges, this paper proposes an end-to-end object detection framework, UFO-DETR, which integrates an LSKNet-based backbone network to optimize the receptive field and reduce the number of parameters. By combining the DAttention and AIFI modules, the model flexibly models multi-scale spatial relationships, improving multi-scale target detection performance. Additionally, the DynFreq-C3 module is proposed to enhance small target detection capability through cross-space frequency feature enhancement. Experimental results show that, compared to RT-DETR-L, the proposed method offers significant advantages in both detection performance and computational efficiency, providing an efficient solution for UAV edge computing.

**선정 근거**
UAV 영상 작은 객체 탐지로 스포츠 장면 분석에 간접적으로 관련

**활용 인사이트**
다중 스케일 객체 탐지 기술은 스포츠 영상에서 선수와 공 등 작은 객체 추적에 활용 가능

## 31위: Make It Hard to Hear, Easy to Learn: Long-Form Bengali ASR and Speaker Diarization via Extreme Augmentation and Perfect Alignment

- arXiv: http://arxiv.org/abs/2602.23070v1 | 2026-02-26 | final 72.0

Although Automatic Speech Recognition (ASR) in Bengali has seen significant progress, processing long-duration audio and performing robust speaker diarization remain critical research gaps. To address the severe scarcity of joint ASR and diarization resources for this language, we introduce Lipi-Ghor-882, a comprehensive 882-hour multi-speaker Bengali dataset.

-> Audio processing techniques applicable to comprehensive sports analysis system, mentions real-time processing

## 32위: CRAG: Can 3D Generative Models Help 3D Assembly?

- arXiv: http://arxiv.org/abs/2602.22629v1 | 2026-02-26 | final 70.4

Most existing 3D assembly methods treat the problem as pure pose estimation, rearranging observed parts via rigid transformations. In contrast, human assembly naturally couples structural reasoning with holistic shape inference.

-> 3D 조립 및 생성 모델로 스포츠 장면의 3D 재구성에 간접적으로 적용 가능

## 33위: MediX-R1: Open Ended Medical Reinforcement Learning

- arXiv: http://arxiv.org/abs/2602.23363v1 | 2026-02-26 | final 70.4

We introduce MediX-R1, an open-ended Reinforcement Learning (RL) framework for medical multimodal large language models (MLLMs) that enables clinically grounded, free-form answers beyond multiple-choice formats. MediX-R1 fine-tunes a baseline vision-language backbone with Group Based RL and a composite reward tailored for medical reasoning: an LLM-based accuracy reward that judges semantic correctness with a strict YES/NO decision, a medical embedding-based semantic reward to capture paraphrases and terminology variants, and lightweight format and modality rewards that enforce interpretable reasoning and modality recognition.

-> Medical reinforcement learning framework, not directly applicable to sports analysis

## 34위: HARU-Net: Hybrid Attention Residual U-Net for Edge-Preserving Denoising in Cone-Beam Computed Tomography

- arXiv: http://arxiv.org/abs/2602.22544v1 | 2026-02-26 | final 70.4

Cone-beam computed tomography (CBCT) is widely used in dental and maxillofacial imaging, but low-dose acquisition introduces strong, spatially varying noise that degrades soft-tissue visibility and obscures fine anatomical structures. Classical denoising methods struggle to suppress noise in CBCT while preserving edges.

-> 의료 영상 노이즈 제거 기술로 스포츠 영상 보정에 간접적으로 적용 가능

## 35위: Skarimva: Skeleton-based Action Recognition is a Multi-view Application

- arXiv: http://arxiv.org/abs/2602.23231v1 | 2026-02-26 | final 68.0

Human action recognition plays an important role when developing intelligent interactions between humans and machines. While there is a lot of active research on improving the machine learning algorithms for skeleton-based action recognition, not much attention has been given to the quality of the input skeleton data itself.

-> 다중 카메라 뷰를 이용한 액션 인식 연구로 스포츠 분석에 간접적으로 관련 있으나 구체적 스포츠 적용 없음

## 36위: A Perspective on Open Challenges in Deformable Object Manipulation

- arXiv: http://arxiv.org/abs/2602.22998v1 | 2026-02-26 | final 68.0

Deformable object manipulation (DOM) represents a critical challenge in robotics, with applications spanning healthcare, manufacturing, food processing, and beyond. Unlike rigid objects, deformable objects exhibit infinite dimensionality, dynamic shape changes, and complex interactions with their environment, posing significant hurdles for perception, modeling, and control.

-> Indirectly related through multimodal perception systems for robotics

## 37위: SoPE: Spherical Coordinate-Based Positional Embedding for Enhancing Spatial Perception of 3D LVLMs

- arXiv: http://arxiv.org/abs/2602.22716v1 | 2026-02-26 | final 68.0

3D Large Vision-Language Models (3D LVLMs) built upon Large Language Models (LLMs) have achieved remarkable progress across various multimodal tasks. However, their inherited position-dependent modeling mechanism, Rotary Position Embedding (RoPE), remains suboptimal for 3D multimodal understanding.

-> 3D 공간 인식 강화로 스포츠 공간 분석에 간접적으로 관련

## 38위: A Mathematical Theory of Agency and Intelligence

- arXiv: http://arxiv.org/abs/2602.22519v1 | 2026-02-26 | final 68.0

To operate reliably under changing conditions, complex systems require feedback on how effectively they use resources, not just whether objectives are met. Current AI systems process vast information to produce sophisticated predictions, yet predictions can appear successful while the underlying interaction with the environment degrades.

-> AI 시스템의 에이전시와 지능에 대한 이론으로 프로젝트와 간접적으로 관련

## 39위: Uni-Animator: Towards Unified Visual Colorization

- arXiv: http://arxiv.org/abs/2602.23191v1 | 2026-02-26 | final 66.4

We propose Uni-Animator, a novel Diffusion Transformer (DiT)-based framework for unified image and video sketch colorization. Existing sketch colorization methods struggle to unify image and video tasks, suffering from imprecise color transfer with single or multiple references, inadequate preservation of high-frequency physical details, and compromised temporal coherence with motion artifacts in large-motion scenes.

-> 스케치 컬러라이제이션 기술이 스포츠 영상 보정에 간접적으로 활용 가능

## 40위: Physics Informed Viscous Value Representations

- arXiv: http://arxiv.org/abs/2602.23280v1 | 2026-02-26 | final 66.4

Offline goal-conditioned reinforcement learning (GCRL) learns goal-conditioned policies from static pre-collected datasets. However, accurate value estimation remains a challenge due to the limited coverage of the state-action space.

-> 물리 정보 강화 강화학습으로 스포츠 전략 분석에 간접적으로 적용 가능

## 41위: No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings

- arXiv: http://arxiv.org/abs/2602.22689v1 | 2026-02-26 | final 66.4

Latent diffusion models have achieved remarkable success in high-fidelity text-to-image generation, but their tendency to memorize training data raises critical privacy and intellectual property concerns. Membership inference attacks (MIAs) provide a principled way to audit such memorization by determining whether a given sample was included in training.

-> Latent diffusion models for image processing could enhance video correction

## 42위: GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion

- arXiv: http://arxiv.org/abs/2602.22862v1 | 2026-02-26 | final 64.0

This paper focuses on enhancing the grasping precision and generalization of manipulation policies learned via imitation learning. Diffusion-based policy learning methods have recently become the mainstream approach for robotic manipulation tasks.

-> 확산 모델 기반 그래핑 정책으로 스포츠 동작 분석에 간접적으로 관련

## 43위: AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation

- arXiv: http://arxiv.org/abs/2602.22740v1 | 2026-02-26 | final 64.0

Referring Image Segmentation (RIS) aims to segment an object in an image identified by a natural language expression. The paper introduces Alignment-Aware Masked Learning (AML), a training strategy to enhance RIS by explicitly estimating pixel-level vision-language alignment, filtering out poorly aligned regions during optimization, and focusing on trustworthy cues.

-> 자연어 표현을 통한 이미지 분할 기술로 특정 선수나 객체 식별에 간접적으로 적용 가능

## 44위: Bayesian Preference Elicitation: Human-In-The-Loop Optimization of An Active Prosthesis

- arXiv: http://arxiv.org/abs/2602.22922v1 | 2026-02-26 | final 64.0

Tuning active prostheses for people with amputation is time-consuming and relies on metrics that may not fully reflect user needs. We introduce a human-in-the-loop optimization (HILO) approach that leverages direct user preferences to personalize a standard four-parameter prosthesis controller efficiently.

-> Prosthesis optimization using Bayesian methods, tangentially related to performance optimization

## 45위: Benchmarking Temporal Web3 Intelligence: Lessons from the FinSurvival 2025 Challenge

- arXiv: http://arxiv.org/abs/2602.23159v1 | 2026-02-26 | final 64.0

Temporal Web analytics increasingly relies on large-scale, longitudinal data to understand how users, content, and systems evolve over time. A rapidly growing frontier is the \emph{Temporal Web3}: decentralized platforms whose behavior is recorded as immutable, time-stamped event streams.

-> Temporal analysis techniques could be applicable to sports activity tracking but domain is Web3 transactions

## 46위: Physics-informed neural particle flow for the Bayesian update step

- arXiv: http://arxiv.org/abs/2602.23089v1 | 2026-02-26 | final 64.0

The Bayesian update step poses significant computational challenges in high-dimensional nonlinear estimation. While log-homotopy particle flow filters offer an alternative to stochastic sampling, existing formulations usually yield stiff differential equations.

-> Neural network techniques for estimation could be adapted for sports analytics but application domain is different

## 47위: ViCLIP-OT: The First Foundation Vision-Language Model for Vietnamese Image-Text Retrieval with Optimal Transport

- arXiv: http://arxiv.org/abs/2602.22678v1 | 2026-02-26 | final 60.0

Image-text retrieval has become a fundamental component in intelligent multimedia systems; however, most existing vision-language models are optimized for highresource languages and remain suboptimal for low-resource settings such as Vietnamese. This work introduces ViCLIP-OT, a foundation vision-language model specifically designed for Vietnamese image-text retrieval.

-> 비전-언어 모델로 스포츠 콘텐츠 이해에 간접적으로 관련

## 48위: Efficient Parallel Algorithms for Hypergraph Matching

- arXiv: http://arxiv.org/abs/2602.22976v1 | 2026-02-26 | final 58.4

We present efficient parallel algorithms for computing maximal matchings in hypergraphs. Our algorithm finds locally maximal edges in the hypergraph and adds them in parallel to the matching.

-> GPU 알고리즘은 엣지 디바이스에 유용하지만 스포츠 분석과 직접 관련 없음

## 49위: HELMLAB: An Analytical, Data-Driven Color Space for Perceptual Distance in UI Design Systems

- arXiv: http://arxiv.org/abs/2602.23010v1 | 2026-02-26 | final 50.4

We present HELMLAB, a 72-parameter analytical color space for UI design systems. The forward transform maps CIE XYZ to a perceptually-organized Lab representation through learned matrices, per-channel power compression, Fourier hue correction, and embedded Helmholtz-Kohlrausch lightness adjustment.

-> UI 디자인을 위한 색 공간 기술로 스포츠 이미지 보정에 약간의 관련성

## 50위: Devling into Adversarial Transferability on Image Classification: Review, Benchmark, and Evaluation

- arXiv: http://arxiv.org/abs/2602.23117v1 | 2026-02-26 | final 50.4

Adversarial transferability refers to the capacity of adversarial examples generated on the surrogate model to deceive alternate, unexposed victim models. This property eliminates the need for direct access to the victim model during an attack, thereby raising considerable security concerns in practical applications and attracting substantial research attention recently.

-> 이미지 분류에 대한 적대적 예제 연구로 프로젝트와 약간 관련

---

## 다시 보기

### From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.20630v1
- 점수: final 96.0

Keypoint-based matching is a fundamental component of modern 3D vision systems, such as Structure-from-Motion (SfM) and SLAM. Most existing learning-based methods are trained on image pairs, a paradigm that fails to explicitly optimize for the long-term trackability of keypoints across sequences under challenging viewpoint and illumination changes. In this paper, we reframe keypoint detection as a sequential decision-making problem. We introduce TraqPoint, a novel, end-to-end Reinforcement Learning (RL) framework designed to optimize the \textbf{Tra}ck-\textbf{q}uality (Traq) of keypoints directly on image sequences. Our core innovation is a track-aware reward mechanism that jointly encourages the consistency and distinctiveness of keypoints across multiple views, guided by a policy gradient method. Extensive evaluations on sparse matching benchmarks, including relative pose estimation and 3D reconstruction, demonstrate that TraqPoint significantly outperforms some state-of-the-art (SOTA) keypoint detection and description methods.

-> 실시간 동분할 기술은 스포츠 장면에서 움직임을 효과적으로 분석하여 하이라이트 편집에 적합합니다.

### Real-time Motion Segmentation with Event-based Normal Flow (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.20790v1
- 점수: final 93.6

Event-based cameras are bio-inspired sensors with pixels that independently and asynchronously respond to brightness changes at microsecond resolution, offering the potential to handle visual tasks in challenging scenarios. However, due to the sparse information content in individual events, directly processing the raw event data to solve vision tasks is highly inefficient, which severely limits the applicability of state-of-the-art methods in real-time tasks, such as motion segmentation, a fundamental task for dynamic scene understanding. Incorporating normal flow as an intermediate representation to compress motion information from event clusters within a localized region provides a more effective solution. In this work, we propose a normal flow-based motion segmentation framework for event-based vision. Leveraging the dense normal flow directly learned from event neighborhoods as input, we formulate the motion segmentation task as an energy minimization problem solved via graph cuts, and optimize it iteratively with normal flow clustering and motion model fitting. By using a normal flow-based motion model initialization and fitting method, the proposed system is able to efficiently estimate the motion models of independently moving objects with only a limited number of candidate models, which significantly reduces the computational complexity and ensures real-time performance, achieving nearly a 800x speedup in comparison to the open-source state-of-the-art method. Extensive evaluations on multiple public datasets fully demonstrate the accuracy and efficiency of our framework.

-> 고속 움직임으로 인한 흐림 문제를 해결하여 스포츠 촬영의 품질을 향상시키는 데 중요합니다.

### Human Video Generation from a Single Image with 3D Pose and View Control (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.21188v1
- 점수: final 90.4

Recent diffusion methods have made significant progress in generating videos from single images due to their powerful visual generation capabilities. However, challenges persist in image-to-video synthesis, particularly in human video generation, where inferring view-consistent, motion-dependent clothing wrinkles from a single image remains a formidable problem. In this paper, we present Human Video Generation in 4D (HVG), a latent video diffusion model capable of generating high-quality, multi-view, spatiotemporally coherent human videos from a single image with 3D pose and view control. HVG achieves this through three key designs: (i) Articulated Pose Modulation, which captures the anatomical relationships of 3D joints via a novel dual-dimensional bone map and resolves self-occlusions across views by introducing 3D information; (ii) View and Temporal Alignment, which ensures multi-view consistency and alignment between a reference image and pose sequences for frame-to-frame stability; and (iii) Progressive Spatio-Temporal Sampling with temporal alignment to maintain smooth transitions in long multi-view animations. Extensive experiments on image-to-video tasks demonstrate that HVG outperforms existing methods in generating high-quality 4D human videos from diverse human images and pose inputs.

-> 3D 포즈 제어 기반 영상 생성 기술은 스포츠 하이라이트 영상 제작에 직접 적용 가능

### Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.21101v1
- 점수: final 89.6

Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.

-> 드론 기반 3D 재구성 기술은 스포츠 촬영 시 움직임 흐림 문제 해결에 적용 가능

### GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.21810v1
- 점수: final 88.0

Motion segmentation in dynamic scenes is highly challenging, as conventional methods heavily rely on estimating camera poses and point correspondences from inherently noisy motion cues. Existing statistical inference or iterative optimization techniques that struggle to mitigate the cumulative errors in multi-stage pipelines often lead to limited performance or high computational cost. In contrast, we propose a fully learning-based approach that directly infers moving objects from latent feature representations via attention mechanisms, thus enabling end-to-end feed-forward motion segmentation. Our key insight is to bypass explicit correspondence estimation and instead let the model learn to implicitly disentangle object and camera motion. Supported by recent advances in 4D scene geometry reconstruction (e.g., $π^3$), the proposed method leverages reliable camera poses and rich spatial-temporal priors, which ensure stable training and robust inference for the model. Extensive experiments demonstrate that by eliminating complex pre-processing and iterative refinement, our approach achieves state-of-the-art motion segmentation performance with high efficiency. The code is available at:https://github.com/zjutcvg/GeoMotion.

-> 동적 장면에서의 움직임 분할 기술로 스포츠 촬영 시 선수와 공 등 움직이는 객체를 정확히 식별 가능

### PyVision-RL: Forging Open Agentic Vision Models via RL (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.20739v1
- 점수: final 88.0

Reinforcement learning for agentic multimodal models often suffers from interaction collapse, where models learn to reduce tool usage and multi-turn reasoning, limiting the benefits of agentic behavior. We introduce PyVision-RL, a reinforcement learning framework for open-weight multimodal models that stabilizes training and sustains interaction. Our approach combines an oversampling-filtering-ranking rollout strategy with an accumulative tool reward to prevent collapse and encourage multi-turn tool use. Using a unified training pipeline, we develop PyVision-Image and PyVision-Video for image and video understanding. For video reasoning, PyVision-Video employs on-demand context construction, selectively sampling task-relevant frames during reasoning to significantly reduce visual token usage. Experiments show strong performance and improved efficiency, demonstrating that sustained interaction and on-demand visual processing are critical for scalable multimodal agents.

-> PyVision-Video framework for video understanding applicable for sports scene analysis

### EKF-Based Depth Camera and Deep Learning Fusion for UAV-Person Distance Estimation and Following in SAR Operations (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.20958v1
- 점수: final 84.0

Search and rescue (SAR) operations require rapid responses to save lives or property. Unmanned Aerial Vehicles (UAVs) equipped with vision-based systems support these missions through prior terrain investigation or real-time assistance during the mission itself. Vision-based UAV frameworks aid human search tasks by detecting and recognizing specific individuals, then tracking and following them while maintaining a safe distance. A key safety requirement for UAV following is the accurate estimation of the distance between camera and target object under real-world conditions, achieved by fusing multiple image modalities. UAVs with deep learning-based vision systems offer a new approach to the planning and execution of SAR operations. As part of the system for automatic people detection and face recognition using deep learning, in this paper we present the fusion of depth camera measurements and monocular camera-to-body distance estimation for robust tracking and following. Deep learning-based filtering of depth camera data and estimation of camera-to-body distance from a monocular camera are achieved with YOLO-pose, enabling real-time fusion of depth information using the Extended Kalman Filter (EKF) algorithm. The proposed subsystem, designed for use in drones, estimates and measures the distance between the depth camera and the human body keypoints, to maintain the safe distance between the drone and the human target. Our system provides an accurate estimated distance, which has been validated against motion capture ground truth data. The system has been tested in real time indoors, where it reduces the average errors, root mean square error (RMSE) and standard deviations of distance estimation up to 15,3\% in three tested scenarios.

-> This paper proposes a method for distance estimation and following of UAVs using depth camera and deep learning fusion. The core is YOLO-pose and EKF algorithm integration for real-time distance measurement.

### SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.20792v1
- 점수: final 82.4

Modeling spinal motion is fundamental to understanding human biomechanics, yet remains underexplored in computer vision due to the spine's complex multi-joint kinematics and the lack of large-scale 3D annotations. We present a biomechanics-aware keypoint simulation framework that augments existing human pose datasets with anatomically consistent 3D spinal keypoints derived from musculoskeletal modeling. Using this framework, we create the first open dataset, named SIMSPINE, which provides sparse vertebra-level 3D spinal annotations for natural full-body motions in indoor multi-camera capture without external restraints. With 2.14 million frames, this enables data-driven learning of vertebral kinematics from subtle posture variations and bridges the gap between musculoskeletal simulation and computer vision. In addition, we release pretrained baselines covering fine-tuned 2D detectors, monocular 3D pose lifting models, and multi-view reconstruction pipelines, establishing a unified benchmark for biomechanically valid spine motion estimation. Specifically, our 2D spine baselines improve the state-of-the-art from 0.63 to 0.80 AUC in controlled environments, and from 0.91 to 0.93 AP for in-the-wild spine tracking. Together, the simulation framework and SIMSPINE dataset advance research in vision-based biomechanics, motion analysis, and digital human modeling by enabling reproducible, anatomically grounded 3D spine estimation under natural conditions.

-> 생체역학 시뮬레이션 프레임워크는 스포츠 동작 분석에 직접 적용 가능하여 선수 자세 및 동작 분석에 필수적

### Strategy-Supervised Autonomous Laparoscopic Camera Control via Event-Driven Graph Mining (2회째 추천)

- arXiv: http://arxiv.org/abs/2602.20500v1
- 점수: final 80.0

Autonomous laparoscopic camera control must maintain a stable and safe surgical view under rapid tool-tissue interactions while remaining interpretable to surgeons. We present a strategy-grounded framework that couples high-level vision-language inference with low-level closed-loop control. Offline, raw surgical videos are parsed into camera-relevant temporal events (e.g., interaction, working-distance deviation, and view-quality degradation) and structured as attributed event graphs. Mining these graphs yields a compact set of reusable camera-handling strategy primitives, which provide structured supervision for learning. Online, a fine-tuned Vision-Language Model (VLM) processes the live laparoscopic view to predict the dominant strategy and discrete image-based motion commands, executed by an IBVS-RCM controller under strict safety constraints; optional speech input enables intuitive human-in-the-loop conditioning. On a surgeon-annotated dataset, event parsing achieves reliable temporal localization (F1-score 0.86), and the mined strategies show strong semantic alignment with expert interpretation (cluster purity 0.81). Extensive ex vivo experiments on silicone phantoms and porcine tissues demonstrate that the proposed system outperforms junior surgeons in standardized camera-handling evaluations, reducing field-of-view centering error by 35.26% and image shaking by 62.33%, while preserving smooth motion and stable working-distance regulation.

-> 전략 기반 자동 카메라 제어 방식은 경기 중요 순간을 자동으로 포착하여 하이라이트 영상 제작에 효과적

### From Statics to Dynamics: Physics-Aware Image Editing with Latent Transition Priors (1회째 추천)

- arXiv: http://arxiv.org/abs/2602.21778v1
- 점수: final 80.0

Instruction-based image editing has achieved remarkable success in semantic alignment, yet state-of-the-art models frequently fail to render physically plausible results when editing involves complex causal dynamics, such as refraction or material deformation. We attribute this limitation to the dominant paradigm that treats editing as a discrete mapping between image pairs, which provides only boundary conditions and leaves transition dynamics underspecified. To address this, we reformulate physics-aware editing as predictive physical state transitions and introduce PhysicTran38K, a large-scale video-based dataset comprising 38K transition trajectories across five physical domains, constructed via a two-stage filtering and constraint-aware annotation pipeline. Building on this supervision, we propose PhysicEdit, an end-to-end framework equipped with a textual-visual dual-thinking mechanism. It combines a frozen Qwen2.5-VL for physically grounded reasoning with learnable transition queries that provide timestep-adaptive visual guidance to a diffusion backbone. Experiments show that PhysicEdit improves over Qwen-Image-Edit by 5.9% in physical realism and 10.1% in knowledge-grounded editing, setting a new state-of-the-art for open-source methods, while remaining competitive with leading proprietary models.

-> 물리적 사실성을 고려한 이미지 편집 기술로 스포츠 영상의 시각적 품질 향상 가능

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
