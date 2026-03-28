# CAPP!C_AI 논문 리포트 (2026-03-28)

> 수집 75 | 필터 71 | 폐기 8 | 평가 62 | 출력 49 | 기준 50점

검색 윈도우: 2026-03-26T00:00:00+00:00 ~ 2026-03-28T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 52

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action summarization, keyframe extraction, video stabilization, image enhancement, quality assessment, pose estimation, motion analysis, biomechanical analysis, tactical analysis, game strategy, sports analytics, edge computing, embedded vision, real-time processing, social video sharing, content distribution, multimedia platforms, physical AI systems, embodied interaction, sensor fusion, biomechanical analysis, tactical analysis, embodied interaction

---

## 1위: LEMMA: Laplacian pyramids for Efficient Marine SeMAntic Segmentation

- arXiv: http://arxiv.org/abs/2603.25689v1
- PDF: https://arxiv.org/pdf/2603.25689v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 94.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Semantic segmentation in marine environments is crucial for the autonomous navigation of unmanned surface vessels (USVs) and coastal Earth Observation events such as oil spills. However, existing methods, often relying on deep CNNs and transformer-based architectures, face challenges in deployment due to their high computational costs and resource-intensive nature. These limitations hinder the practicality of real-time, low-cost applications in real-world marine settings.   To address this, we propose LEMMA, a lightweight semantic segmentation model designed specifically for accurate remote sensing segmentation under resource constraints. The proposed architecture leverages Laplacian Pyramids to enhance edge recognition, a critical component for effective feature extraction in complex marine environments for disaster response, environmental surveillance, and coastal monitoring. By integrating edge information early in the feature extraction process, LEMMA eliminates the need for computationally expensive feature map computations in deeper network layers, drastically reducing model size, complexity and inference time. LEMMA demonstrates state-of-the-art performance across datasets captured from diverse platforms while reducing trainable parameters and computational requirements by up to 71x, GFLOPs by up to 88.5\%, and inference time by up to 84.65\%, as compared to existing models. Experimental results highlight its effectiveness and real-world applicability, including 93.42\% IoU on the Oil Spill dataset and 98.97\% mIoU on Mastr1325.

**선정 근거**
가벼운 세그멘테이션 모델로 엣지 디바이스에서 실시간 처리에 필수적인 계산 효율성 제공

**활용 인사이트**
해양 환경 설계를 스포츠 장면으로 적용해 선수와 객체를 정확히 식별하며 71x 적은 파라미터로 동작

## 2위: Once-for-All Channel Mixers (HYPERTINYPW): Generative Compression for TinyML

- arXiv: http://arxiv.org/abs/2603.24916v1
- PDF: https://arxiv.org/pdf/2603.24916v1
- 발행일: 2026-03-26
- 카테고리: cs.LG, stat.ML
- 점수: final 91.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Deploying neural networks on microcontrollers is constrained by kilobytes of flash and SRAM, where 1x1 pointwise (PW) mixers often dominate memory even after INT8 quantization across vision, audio, and wearable sensing. We present HYPER-TINYPW, a compression-as-generation approach that replaces most stored PW weights with generated weights: a shared micro-MLP synthesizes PW kernels once at load time from tiny per-layer codes, caches them, and executes them with standard integer operators. This preserves commodity MCU runtimes and adds only a one-off synthesis cost; steady-state latency and energy match INT8 separable CNN baselines. Enforcing a shared latent basis across layers removes cross-layer redundancy, while keeping PW1 in INT8 stabilizes early, morphology-sensitive mixing. We contribute (i) TinyML-faithful packed-byte accounting covering generator, heads/factorization, codes, kept PW1, and backbone; (ii) a unified evaluation with validation-tuned t* and bootstrap confidence intervals; and (iii) a deployability analysis covering integer-only inference and boot versus lazy synthesis. On three ECG benchmarks (Apnea-ECG, PTB-XL, MIT-BIH), HYPER-TINYPW shifts the macro-F1 versus flash Pareto frontier: at about 225 kB it matches a roughly 1.4 MB CNN while being 6.31x smaller (84.15% fewer bytes), retaining at least 95% of large-model macro-F1. Under 32-64 kB budgets it sustains balanced detection where compact baselines degrade. The mechanism applies broadly to other 1D biosignals, on-device speech, and embedded sensing tasks where per-layer redundancy dominates, indicating a wider role for compression-as-generation in resource-constrained ML systems. Beyond ECG, HYPER-TINYPW transfers to TinyML audio: on Speech Commands it reaches 96.2% test accuracy (98.2% best validation), supporting broader applicability to embedded sensing workloads where repeated linear mixers dominate memory.

**선정 근거**
리소스 제약이 있는 엣지 디바이스에서 여러 AI 모델을 동시에 실행하기 위한 모델 압축 기술

**활용 인사이트**
84.15% 적은 메모리로 스포츠 분석 모델들을 동시에 실행해 실시간 다중 작업 처리 가능

## 3위: Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models

- arXiv: http://arxiv.org/abs/2603.25250v1
- PDF: https://arxiv.org/pdf/2603.25250v1
- 코드: https://github.com/YBZh/OpenOOD-VLM}{YBZh/OpenOOD-VLM}
- 발행일: 2026-03-26
- 카테고리: cs.CV, cs.AI, cs.LG
- 점수: final 90.8 (llm_adjusted:91 = base:78 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Out-of-distribution (OOD) detection aims to identify samples that deviate from in-distribution (ID). One popular pipeline addresses this by introducing negative labels distant from ID classes and detecting OOD based on their distance to these labels. However, such labels may present poor activation on OOD samples, failing to capture the OOD characteristics. To address this, we propose \underline{T}est-time \underline{A}ctivated \underline{N}egative \underline{L}abels (TANL) by dynamically evaluating activation levels across the corpus dataset and mining candidate labels with high activation responses during the testing process. Specifically, TANL identifies high-confidence test images online and accumulates their assignment probabilities over the corpus to construct a label activation metric. Such a metric leverages historical test samples to adaptively align with the test distribution, enabling the selection of distribution-adaptive activated negative labels. By further exploring the activation information within the current testing batch, we introduce a more fine-grained, batch-adaptive variant. To fully utilize label activation knowledge, we propose an activation-aware score function that emphasizes negative labels with stronger activations, boosting performance and enhancing its robustness to the label number. Our TANL is training-free, test-efficient, and grounded in theoretical justification. Experiments on diverse backbones and wide task settings validate its effectiveness. Notably, on the large-scale ImageNet benchmark, TANL significantly reduces the FPR95 from 17.5\% to 9.8\%. Codes are available at \href{https://github.com/YBZh/OpenOOD-VLM}{YBZh/OpenOOD-VLM}.

**선정 근거**
Vision-language OOD detection techniques applicable to sports scene analysis

## 4위: Pixelis: Reasoning in Pixels, from Seeing to Acting

- arXiv: http://arxiv.org/abs/2603.25091v1
- PDF: https://arxiv.org/pdf/2603.25091v1
- 발행일: 2026-03-26
- 카테고리: cs.CV, cs.AI
- 점수: final 90.0 (llm_adjusted:90 = base:80 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Most vision-language systems are static observers: they describe pixels, do not act, and cannot safely improve under shift. This passivity limits generalizable, physically grounded visual intelligence. Learning through action, not static description, is essential beyond curated data. We present Pixelis, a pixel-space agent that operates directly on images and videos via a compact set of executable operations (zoom/crop, segment, track, OCR, temporal localization) and learns from its consequences. Pixelis trains in three phases: (1) Supervised Fine-Tuning learns a pixel-tool grammar from Chain-of-Thought-Action traces with a masked imitation loss that upweights operation/argument tokens and auxiliary heads to stabilize pixel-grounded arguments; (2) Curiosity-Coherence Reward Fine-Tuning optimizes a dual-drive objective marrying prediction-error curiosity with adjacent-step coherence and a mild efficiency prior under a KL anchor, yielding short, valid, structured toolchains; (3) Pixel Test-Time RL performs label-free adaptation by retrieving neighbors, voting over complete trajectories rather than answers, and updating toward short, high-fidelity exemplars while constraining drift with a KL-to-EMA safety control. Across six public image and video benchmarks, Pixelis yields consistent improvements: the average relative gain is +4.08% over the same 8B baseline (peaking at +6.03% on VSI-Bench), computed as (ours-baseline)/baseline, while producing shorter, auditable toolchains and maintaining in-corridor KL during test-time learning. Acting within pixels, rather than abstract tokens, grounds multimodal perception in the physical world, linking visual reasoning with actionable outcomes, and enables embodied adaptation without external feedback.

**선정 근거**
픽셀 기반 비디오 분석 기술이 스포츠 하이라이트 자동 편집에 직접 적용 가능

**활용 인사이트**
추적, 분할, OCR 등의 작업으로 중요한 순간을 자동 감지하고 편집해 실시간 하이라이트 생성 가능

## 5위: CIAR: Interval-based Collaborative Decoding for Image Generation Acceleration

- arXiv: http://arxiv.org/abs/2603.25463v1
- PDF: https://arxiv.org/pdf/2603.25463v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 88.4 (llm_adjusted:88 = base:78 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Auto-regressive (AR) models have recently made notable progress in image generation, achieving performance comparable to diffusion-based approaches. However, their computational intensity and sequential nature impede on-device deployment, causing disruptive latency. We address this via a cloud-device collaboration framework \textbf{CIAR}, which utilizes on-device self-verification to handle two key properties of visual synthesis: \textit{the vast token vocabulary} required for high-fidelity images and \textit{inherent spatial redundancy} which leads to extreme predictability in homogeneous regions, while object boundaries exhibit high uncertainty. Uniform verification wastes resources on such redundant tokens. Our solution centers on an on-device token uncertainty quantifier, which adopts continuous probability intervals to accelerate processing and make it feasible for large visual vocabularies instead of conventional discrete solution sets. Additionally, we incorporate a Interval-enhanced decoding module to further speed up decoding while maintaining visual fidelity and semantic consistency via a distribution alignment training strategy. Extensive experiments demonstrate that CIAR achieves a 2.18x speed-up and reduces cloud requests by 70\%, while preserving image quality compared to existing methods.

**선정 근거**
CIAR는 엣지 디바이스의 이미지 처리 속도를 2.18배 향상시키며 클라우드 요청을 70% 감소시켜 실시간 스포츠 촬영에 최적화된 솔루션입니다.

**활용 인사이트**
CIAR의 클라우드-디바이스 협력 프레임워크를 rk3588 엣지 디바이스에 적용하면 스포츠 장면 촬영 및 보정의 실시간성을 유지하면서 고품질 영상 생성이 가능해집니다.

## 6위: MuRF: Unlocking the Multi-Scale Potential of Vision Foundation Models

- arXiv: http://arxiv.org/abs/2603.25744v1
- PDF: https://arxiv.org/pdf/2603.25744v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 88.4 (llm_adjusted:88 = base:78 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Vision Foundation Models (VFMs) have become the cornerstone of modern computer vision, offering robust representations across a wide array of tasks. While recent advances allow these models to handle varying input sizes during training, inference typically remains restricted to a single, fixed scale. This prevalent single-scale paradigm overlooks a fundamental property of visual perception: varying resolutions offer complementary inductive biases, where low-resolution views excel at global semantic recognition and high-resolution views are essential for fine-grained refinement. In this work, we propose Multi-Resolution Fusion (MuRF), a simple yet universally effective strategy to harness this synergy at inference time. Instead of relying on a single view, MuRF constructs a unified representation by processing an image at multiple resolutions through a frozen VFM and fusing the resulting features. The universality of MuRF is its most compelling attribute. It is not tied to a specific architecture, serving instead as a fundamental, training-free enhancement to visual representation. We empirically validate this by applying MuRF to a broad spectrum of critical computer vision tasks across multiple distinct VFM families - primarily DINOv2, but also demonstrating successful generalization to contrastive models like SigLIP2.

**선정 근거**
저조도 이미지 보정 기술이 스포츠 영상 보정에 직접적으로 적용 가능하여 다양한 조건에서 고품질 콘텐츠 생성 가능

**활용 인사이트**
HVI 색공간의 노이즈-디커플링 감전략과 S2D 전략을 rk3588 edge 디바이스에 구현하여 실시간 저조도 보정 및 스타일 조절 가능

## 7위: EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval

- arXiv: http://arxiv.org/abs/2603.25267v1
- PDF: https://arxiv.org/pdf/2603.25267v1
- 코드: https://github.com/draym28/EagleNet
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 88.4 (llm_adjusted:88 = base:85 + bonus:+3)
- 플래그: 코드 공개

**개요**
Text-video retrieval tasks have seen significant improvements due to the recent development of large-scale vision-language pre-trained models. Traditional methods primarily focus on video representations or cross-modal alignment, while recent works shift toward enriching text expressiveness to better match the rich semantics in videos. However, these methods use only interactions between text and frames/video, and ignore rich interactions among the internal frames within a video, so the final expanded text cannot capture frame contextual information, leading to disparities between text and video. In response, we introduce Energy-Aware Fine-Grained Relationship Learning Network (EagleNet) to generate accurate and context-aware enriched text embeddings. Specifically, the proposed Fine-Grained Relationship Learning mechanism (FRL) first constructs a text-frame graph by the generated text candidates and frames, then learns relationships among texts and frames, which are finally used to aggregate text candidates into an enriched text embedding that incorporates frame contextual information. To further improve fine-grained relationship learning in FRL, we design Energy-Aware Matching (EAM) to model the energy of text-frame interactions and thus accurately capture the distribution of real text-video pairs. Moreover, for more effective cross-modal alignment and stable training, we replace the conventional softmax-based contrastive loss with the sigmoid loss. Extensive experiments have demonstrated the superiority of EagleNet across MSRVTT, DiDeMo, MSVD, and VATEX. Codes are available at https://github.com/draym28/EagleNet.

**선정 근거**
동물 개체 식별 연구에서 얻은 시각적 마킹 설계 원리는 선수 추적 시스템에 응용 가능

**활용 인사이트**
ResNet-50 기반 개체 식별 모델을 활용하여 다양한 각도와 움직임 blur에 강한 선수 식별 시스템 구축 가능

## 8위: Towards Controllable Low-Light Image Enhancement: A Continuous Multi-illumination Dataset and Efficient State Space Framework

- arXiv: http://arxiv.org/abs/2603.25296v1
- PDF: https://arxiv.org/pdf/2603.25296v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 87.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
Low-light image enhancement (LLIE) has traditionally been formulated as a deterministic mapping. However, this paradigm often struggles to account for the ill-posed nature of the task, where unknown ambient conditions and sensor parameters create a multimodal solution space. Consequently, state-of-the-art methods frequently encounter luminance discrepancies between predictions and labels, often necessitating "gt-mean" post-processing to align output luminance for evaluation. To address this fundamental limitation, we propose a transition toward Controllable Low-light Enhancement (CLE), explicitly reformulating the task as a well-posed conditional problem. To this end, we introduce CLE-RWKV, a holistic framework supported by Light100, a new benchmark featuring continuous real-world illumination transitions. To resolve the conflict between luminance control and chromatic fidelity, a noise-decoupled supervision strategy in the HVI color space is employed, effectively separating illumination modulation from texture restoration. Architecturally, to adapt efficient State Space Models (SSMs) for dense prediction, we leverage a Space-to-Depth (S2D) strategy. By folding spatial neighborhoods into channel dimensions, this design allows the model to recover local inductive biases and effectively bridge the "scanning gap" inherent in flattened visual sequences without sacrificing linear complexity. Experiments across seven benchmarks demonstrate that our approach achieves competitive performance and robust controllability, providing a real-world multi-illumination alternative that significantly reduces the reliance on gt-mean post-processing.

**선정 근거**
저조도 이미지 보정 기술이 스포츠 영상 보정에 적용 가능

## 9위: Insights on back marking for the automated identification of animals

- arXiv: http://arxiv.org/abs/2603.25535v1
- PDF: https://arxiv.org/pdf/2603.25535v1
- 발행일: 2026-03-26
- 카테고리: cs.CV, cs.LG
- 점수: final 87.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
To date, there is little research on how to design back marks to best support individual-level monitoring of uniform looking species like pigs. With the recent surge of machine learning-based monitoring solutions, there is a particular need for guidelines on the design of marks that can be effectively recognised by such algorithms. This study provides valuable insights on effective back mark design, based on the analysis of a machine learning model, trained to distinguish pigs via their back marks. Specifically, a neural network of type ResNet-50 was trained to classify ten pigs with unique back marks. The analysis of the model's predictions highlights the significance of certain design choices, even in controlled settings. Most importantly, the set of back marks must be designed such that each mark remains unambiguous under conditions of motion blur, diverse view angles and occlusions, caused by animal behaviour. Further, the back mark design must consider data augmentation strategies commonly employed during model training, like colour, flip and crop augmentations. The generated insights can support individual-level monitoring in future studies and real-world applications by optimizing back mark design.

**선정 근거**
Provides insights on individual identification using visual marks, relevant for player tracking in sports

## 10위: Towards Video Anomaly Detection from Event Streams: A Baseline and Benchmark Datasets

- arXiv: http://arxiv.org/abs/2603.24991v1
- PDF: https://arxiv.org/pdf/2603.24991v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 86.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Event-based vision, characterized by low redundancy, focus on dynamic motion, and inherent privacy-preserving properties, naturally fits the demands of video anomaly detection (VAD). However, the absence of dedicated event-stream anomaly detection datasets and effective modeling strategies has significantly hindered progress in this field. In this work, we take the first major step toward establishing event-based VAD as a unified research direction. We first construct multiple event-stream based benchmarks for video anomaly detection, featuring synchronized event and RGB recordings. Leveraging the unique properties of events, we then propose an EVent-centric spatiotemporal Video Anomaly Detection framework, namely EWAD, with three key innovations: an event density aware dynamic sampling strategy to select temporally informative segments; a density-modulated temporal modeling approach that captures contextual relations from sparse event streams; and an RGB-to-event knowledge distillation mechanism to enhance event-based representations under weak supervision. Extensive experiments on three benchmarks demonstrate that our EWAD achieves significant improvements over existing approaches, highlighting the potential and effectiveness of event-driven modeling for video anomaly detection. The benchmark datasets will be made publicly available.

**선정 근거**
이벤트 기반 비디오 이상 탐지 기술을 스포츠 중요 장면 감지에 직접 적용 가능하여 실시간으로 경기의 결정적 순간 포착

**활용 인사이트**
이벤트 중심의 시공간 이상 탐지 프레임워크를 통해 경기 중 중요한 순간을 실시간으로 감지하고 하이라이트 자동 생성

## 11위: AnyHand: A Large-Scale Synthetic Dataset for RGB(-D) Hand Pose Estimation

- arXiv: http://arxiv.org/abs/2603.25726v1
- PDF: https://arxiv.org/pdf/2603.25726v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 86.0 (llm_adjusted:85 = base:80 + bonus:+5)
- 플래그: 엣지

**개요**
We present AnyHand, a large-scale synthetic dataset designed to advance the state of the art in 3D hand pose estimation from both RGB-only and RGB-D inputs. While recent works with foundation approaches have shown that an increase in the quantity and diversity of training data can markedly improve performance and robustness in hand pose estimation, existing real-world-collected datasets on this task are limited in coverage, and prior synthetic datasets rarely provide occlusions, arm details, and aligned depth together at scale. To address this bottleneck, our AnyHand contains 2.5M single-hand and 4.1M hand-object interaction RGB-D images, with rich geometric annotations. In the RGB-only setting, we show that extending the original training sets of existing baselines with AnyHand yields significant gains on multiple benchmarks (FreiHAND and HO-3D), even when keeping the architecture and training scheme fixed. More impressively, the model trained with AnyHand shows stronger generalization to the out-of-domain HO-Cap dataset, without any fine-tuning. We also contribute a lightweight depth fusion module that can be easily integrated into existing RGB-based models. Trained with AnyHand, the resulting RGB-D model achieves superior performance on the HO-3D benchmark, showing the benefits of depth integration and the effectiveness of our synthetic data.

**선정 근거**
스포츠 동작 분석에 필수적인 손 제스처 인식을 위한 대규모 데이터셋 제공

**활용 인사이트**
RK3588 기기에서 실시간으로 손 동작을 추적하여 스포츠 기술 분석 가능

## 12위: Improving Infinitely Deep Bayesian Neural Networks with Nesterov's Accelerated Gradient Method

- arXiv: http://arxiv.org/abs/2603.25024v1
- PDF: https://arxiv.org/pdf/2603.25024v1
- 발행일: 2026-03-26
- 카테고리: stat.ML, cs.LG
- 점수: final 86.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
As a representative continuous-depth neural network approach, stochastic differential equation (SDE)-based Bayesian neural networks (BNNs) have attracted considerable attention due to their solid theoretical foundations and strong potential for real-world applications. However, their reliance on numerical SDE solvers inevitably incurs a large number of function evaluations (NFEs), resulting in high computational cost and occasional convergence instability. To address these challenges, we propose a Nesterov-accelerated gradient (NAG) enhanced SDE-BNN model. By integrating NAG into the SDE-BNN framework along with an NFE-dependent residual skip connection, our method accelerates convergence and substantially reduces NFEs during both training and testing. Extensive empirical results show that our model consistently outperforms conventional SDE-BNNs across various tasks, including image classification and sequence modeling, achieving lower NFEs and improved predictive accuracy.

**선정 근거**
Efficient Bayesian neural networks applicable to sports movement analysis

## 13위: Multimodal Dataset Distillation via Phased Teacher Models

- arXiv: http://arxiv.org/abs/2603.25388v1
- PDF: https://arxiv.org/pdf/2603.25388v1
- 코드: https://github.com/Previsior/PTM-ST
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 84.4 (llm_adjusted:83 = base:75 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Multimodal dataset distillation aims to construct compact synthetic datasets that enable efficient compression and knowledge transfer from large-scale image-text data. However, existing approaches often fail to capture the complex, dynamically evolving knowledge embedded in the later training stages of teacher models. This limitation leads to degraded student performance and compromises the quality of the distilled data. To address critical challenges such as pronounced cross-stage performance gaps and unstable teacher trajectories, we propose Phased Teacher Model with Shortcut Trajectory (PTM-ST) -- a novel phased distillation framework. PTM-ST leverages stage-aware teacher modeling and a shortcut-based trajectory construction strategy to accurately fit the teacher's learning dynamics across distinct training phases. This enhances both the stability and expressiveness of the distillation process. Through theoretical analysis and comprehensive experiments, we show that PTM-ST significantly mitigates optimization oscillations and inter-phase knowledge gaps, while also reducing storage overhead. Our method consistently surpasses state-of-the-art baselines on Flickr30k and COCO, achieving up to 13.5% absolute improvement and an average gain of 9.53% on Flickr30k. Code: https://github.com/Previsior/PTM-ST.

**선정 근거**
대규모 데이터를 압축하여 엣지 디바이스에서 효율적으로 스포츠 영상 처리 가능

**활용 인사이트**
저장 공간을 최소화하면서도 정확도를 유지하는 스포츠 분석 모델 구현 가능

## 14위: VideoTIR: Accurate Understanding for Long Videos with Efficient Tool-Integrated Reasoning

- arXiv: http://arxiv.org/abs/2603.25021v1
- PDF: https://arxiv.org/pdf/2603.25021v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 83.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Existing Multimodal Large Language Models (MLLMs) often suffer from hallucinations in long video understanding (LVU), primarily due to the imbalance between textual and visual tokens. Observing that MLLMs handle short visual inputs well, recent LVU works alleviate hallucinations by automatically parsing the vast visual data into manageable segments that can be effectively processed by MLLMs. SFT-based tool-calling methods can serve this purpose, but they typically require vast amounts of fine-grained, high-quality data and suffer from constrained tool-calling trajectories. We propose a novel VideoTIR that leverages Reinforcement Learning (RL) to encourage proper usage of comprehensive multi-level toolkits for efficient long video understanding. VideoTIR explores both Zero-RL and SFT cold-starting to enable MLLMs to retrieve and focus on meaningful video segments/images/regions, enhancing long video understanding both accurately and efficiently. To reduce redundant tool-calling, we propose Toolkit Action Grouped Policy Optimization (TAGPO), which enhances the efficiency of the calling process through stepwise reward assignment and reuse of failed rollouts. Additionally, we develop a sandbox-based trajectory synthesis framework to generate high-quality trajectories data. Extensive experiments on three long-video QA benchmarks demonstrate the effectiveness and efficiency of our method.

**선정 근거**
이 논문은 장기간 비디오 이해와 도구 통합 추론을 효율적으로 처리하는 방법을 제안합니다. 스포츠 경기 전체를 분석하며 하이라이트 장면을 정확하게 추출하는 데 직접적으로 적용 가능합니다.

**활용 인사이트**
VideoTIR을 사용하여 스포츠 경기 영상을 세그먼트로 분할하고 의미 있는 장면을 자동으로 식별하면 실시간으로 하이라이트를 생성할 수 있습니다. 이는 rk3588 기반 엣지 디바이스에서 낮은 지연 시간으로 작동할 수 있습니다.

## 15위: RefAlign: Representation Alignment for Reference-to-Video Generation

- arXiv: http://arxiv.org/abs/2603.25743v1
- PDF: https://arxiv.org/pdf/2603.25743v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 82.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
Reference-to-video (R2V) generation is a controllable video synthesis paradigm that constrains the generation process using both text prompts and reference images, enabling applications such as personalized advertising and virtual try-on. In practice, existing R2V methods typically introduce additional high-level semantic or cross-modal features alongside the VAE latent representation of the reference image and jointly feed them into the diffusion Transformer (DiT). These auxiliary representations provide semantic guidance and act as implicit alignment signals, which can partially alleviate pixel-level information leakage in the VAE latent space. However, they may still struggle to address copy--paste artifacts and multi-subject confusion caused by modality mismatch across heterogeneous encoder features. In this paper, we propose RefAlign, a representation alignment framework that explicitly aligns DiT reference-branch features to the semantic space of a visual foundation model (VFM). The core of RefAlign is a reference alignment loss that pulls the reference features and VFM features of the same subject closer to improve identity consistency, while pushing apart the corresponding features of different subjects to enhance semantic discriminability. This simple yet effective strategy is applied only during training, incurring no inference-time overhead, and achieves a better balance between text controllability and reference fidelity. Extensive experiments on the OpenS2V-Eval benchmark demonstrate that RefAlign outperforms current state-of-the-art methods in TotalScore, validating the effectiveness of explicit reference alignment for R2V tasks.

**선정 근거**
참조 영상 기반 생성 기술로 스포츠 하이라이트 영상 제작에 직접 적용 가능

**활용 인사이트**
선수 개인별 특징을 유지하면서 하이라이트 영상을 자동으로 생성할 수 있음

## 16위: BFMD: A Full-Match Badminton Dense Dataset for Dense Shot Captioning

- arXiv: http://arxiv.org/abs/2603.25533v1
- PDF: https://arxiv.org/pdf/2603.25533v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 80.4 (llm_adjusted:78 = base:78 + bonus:+0)

**개요**
Understanding tactical dynamics in badminton requires analyzing entire matches rather than isolated clips. However, existing badminton datasets mainly focus on short clips or task-specific annotations and rarely provide full-match data with dense multimodal annotations. This limitation makes it difficult to generate accurate shot captions and perform match-level analysis. To address this limitation, we introduce the first Badminton Full Match Dense (BFMD) dataset, with 19 broadcast matches (including both singles and doubles) covering over 20 hours of play, comprising 1,687 rallies and 16,751 hit events, each annotated with a shot caption. The dataset provides hierarchical annotations including match segments, rally events, and dense rally-level multimodal annotations such as shot types, shuttle trajectories, player pose keypoints, and shot captions. We develop a VideoMAE-based multimodal captioning framework with a Semantic Feedback mechanism that leverages shot semantics to guide caption generation and improve semantic consistency. Experimental results demonstrate that multimodal modeling and semantic feedback improve shot caption quality over RGB-only baselines. We further showcase the potential of BFMD by analyzing the temporal evolution of tactical patterns across full matches.

**선정 근거**
스포츠 분석 및 포즈 키포인트 관련, 하지만 배드민턴 전용 및 엣지 디바이스 촬영/편집 직접적 연관성 낮음

## 17위: LaMP: Learning Vision-Language-Action Policies with 3D Scene Flow as Latent Motion Prior

- arXiv: http://arxiv.org/abs/2603.25399v1
- PDF: https://arxiv.org/pdf/2603.25399v1
- 발행일: 2026-03-26
- 카테고리: cs.CV, cs.RO
- 점수: final 80.4 (llm_adjusted:78 = base:75 + bonus:+3)
- 플래그: 코드 공개

**개요**
We introduce \textbf{LaMP}, a dual-expert Vision-Language-Action framework that embeds dense 3D scene flow as a latent motion prior for robotic manipulation. Existing VLA models regress actions directly from 2D semantic visual features, forcing them to learn complex 3D physical interactions implicitly. This implicit learning strategy degrades under unfamiliar spatial dynamics. LaMP addresses this limitation by aligning a flow-matching \emph{Motion Expert} with a policy-predicting \emph{Action Expert} through gated cross-attention. Specifically, the Motion Expert generates a one-step partially denoised 3D scene flow, and its hidden states condition the Action Expert without full multi-step reconstruction. We evaluate LaMP on the LIBERO, LIBERO-Plus, and SimplerEnv-WidowX simulation benchmarks as well as real-world experiments. LaMP consistently outperforms evaluated VLA baselines across LIBERO, LIBERO-Plus, and SimplerEnv-WidowX benchmarks, achieving the highest reported average success rates under the same training budgets. On LIBERO-Plus OOD perturbations, LaMP shows improved robustness with an average 9.7% gain over the strongest prior baseline. Our project page is available at https://summerwxk.github.io/lamp-project-page/.

**선정 근거**
3D scene flow 기술이 스포츠 동작 분석에 직접 적용 가능하며, 실시간 분석을 위한 효율적인 모델 제공

**활용 인사이트**
Motion Expert와 Action Expert의 게이티드 크로스 어텐션을 스포츠 카메라 시스템에 통합하여 선수들의 움직임을 실시간으로 분석하고 전략을 생성

## 18위: No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models

- arXiv: http://arxiv.org/abs/2603.25722v1
- PDF: https://arxiv.org/pdf/2603.25722v1
- 코드: https://github.com/SamsungLabs/concept_centric_clip
- 발행일: 2026-03-26
- 카테고리: cs.CV, cs.LG
- 점수: final 80.4 (llm_adjusted:78 = base:75 + bonus:+3)
- 플래그: 코드 공개

**개요**
Contrastive vision-language (V&L) models remain a popular choice for various applications. However, several limitations have emerged, most notably the limited ability of V&L models to learn compositional representations. Prior methods often addressed this limitation by generating custom training data to obtain hard negative samples. Hard negatives have been shown to improve performance on compositionality tasks, but are often specific to a single benchmark, do not generalize, and can cause substantial degradation of basic V&L capabilities such as zero-shot or retrieval performance, rendering them impractical. In this work we follow a different approach. We identify two root causes that limit compositionality performance of V&Ls: 1) Long training captions do not require a compositional representation; and 2) The final global pooling in the text and image encoders lead to a complete loss of the necessary information to learn binding in the first place. As a remedy, we propose two simple solutions: 1) We obtain short concept centric caption parts using standard NLP software and align those with the image; and 2) We introduce a parameter-free cross-modal attention-pooling to obtain concept centric visual embeddings from the image encoder. With these two changes and simple auxiliary contrastive losses, we obtain SOTA performance on standard compositionality benchmarks, while maintaining or improving strong zero-shot and retrieval capabilities. This is achieved without increasing inference cost. We release the code for this work at https://github.com/SamsungLabs/concept_centric_clip.

**선정 근거**
Concept-centric learning for vision-language models could enhance sports movement and strategy analysis

## 19위: HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models

- arXiv: http://arxiv.org/abs/2603.25411v1
- PDF: https://arxiv.org/pdf/2603.25411v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 78.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Achieving human-like spatial intelligence for vision-language models (VLMs) requires inferring 3D structures from 2D observations, recognizing object properties and relations in 3D space, and performing high-level spatial reasoning. In this paper, we propose a principled hierarchical framework that decomposes the learning of 3D spatial understanding in VLMs into four progressively complex levels, from geometric perception to abstract spatial reasoning. Guided by this framework, we construct an automated pipeline that processes approximately 5M images with over 45M objects to generate 3D spatial VQA pairs across diverse tasks and scenes for VLM supervised fine-tuning. We also develop an RGB-D VLM incorporating metric-scale point maps as auxiliary inputs to further enhance spatial understanding. Extensive experiments demonstrate that our approach achieves state-of-the-art performance on multiple spatial understanding and reasoning benchmarks, surpassing specialized spatial models and large proprietary systems such as Gemini-2.5-pro and GPT-5. Moreover, our analysis reveals clear dependencies among hierarchical task levels, offering new insights into how multi-level task design facilitates the emergence of 3D spatial intelligence.

**선정 근거**
계층적 3D 공간 이해 프레임워크가 스포츠 동작 및 자세 분석에 직접 적용 가능하여 플랫폼의 핵심 분석 기술로 활용 가능합니다.

**활용 인사이트**
RGB-D VLM과 포인트 맵을 결합해 실제 시간으로 스포츠 선수의 3D 동작을 분석하고, 계층적 프레임워크를 통해 복잡한 경기 전략을 이해할 수 있습니다.

## 20위: Vision Hopfield Memory Networks

- arXiv: http://arxiv.org/abs/2603.25157v1
- PDF: https://arxiv.org/pdf/2603.25157v1
- 발행일: 2026-03-26
- 카테고리: cs.LG, cs.AI, cs.CV, stat.ML
- 점수: final 78.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Recent vision and multimodal foundation backbones, such as Transformer families and state-space models like Mamba, have achieved remarkable progress, enabling unified modeling across images, text, and beyond. Despite their empirical success, these architectures remain far from the computational principles of the human brain, often demanding enormous amounts of training data while offering limited interpretability. In this work, we propose the Vision Hopfield Memory Network (V-HMN), a brain-inspired foundation backbone that integrates hierarchical memory mechanisms with iterative refinement updates. Specifically, V-HMN incorporates local Hopfield modules that provide associative memory dynamics at the image patch level, global Hopfield modules that function as episodic memory for contextual modulation, and a predictive-coding-inspired refinement rule for iterative error correction. By organizing these memory-based modules hierarchically, V-HMN captures both local and global dynamics in a unified framework. Memory retrieval exposes the relationship between inputs and stored patterns, making decisions more interpretable, while the reuse of stored patterns improves data efficiency. This brain-inspired design therefore enhances interpretability and data efficiency beyond existing self-attention- or state-space-based approaches. We conducted extensive experiments on public computer vision benchmarks, and V-HMN achieved competitive results against widely adopted backbone architectures, while offering better interpretability, higher data efficiency, and stronger biological plausibility. These findings highlight the potential of V-HMN to serve as a next-generation vision foundation model, while also providing a generalizable blueprint for multimodal backbones in domains such as text and audio, thereby bridging brain-inspired computation with large-scale machine learning.

**선정 근거**
Brain-inspired vision backbone with improved data efficiency applicable to sports analysis

## 21위: Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale

- arXiv: http://arxiv.org/abs/2603.25544v1
- PDF: https://arxiv.org/pdf/2603.25544v1
- 코드: https://github.com/amathislab/musclemimic
- 발행일: 2026-03-26
- 카테고리: cs.RO
- 점수: final 76.4 (llm_adjusted:73 = base:70 + bonus:+3)
- 플래그: 코드 공개

**개요**
Learning motor control for muscle-driven musculoskeletal models is hindered by the computational cost of biomechanically accurate simulation and the scarcity of validated, open full-body models. Here we present MuscleMimic, an open-source framework for scalable motion imitation learning with physiologically realistic, muscle-actuated humanoids. MuscleMimic provides two validated musculoskeletal embodiments - a fixed-root upper-body model (126 muscles) for bimanual manipulation and a full-body model (416 muscles) for locomotion - together with a retargeting pipeline that maps SMPL-format motion capture data onto musculoskeletal structures while preserving kinematic and dynamic consistency. Leveraging massively parallel GPU simulation, the framework achieves order-of-magnitude training speedups over prior CPU-based approaches while maintaining comprehensive collision handling, enabling a single generalist policy to be trained on hundreds of diverse motions within days. The resulting policy faithfully reproduces a broad repertoire of human movements under full muscular control and can be fine-tuned to novel motions within hours. Biomechanical validation against experimental walking and running data demonstrates strong agreement in joint kinematics (mean correlation r = 0.90), while muscle activation analysis reveals both the promise and fundamental challenges of achieving physiological fidelity through kinematic imitation alone. By lowering the computational and data barriers to musculoskeletal simulation, MuscleMimic enables systematic model validation across diverse dynamic movements and broader participation in neuromuscular control research. Code, models, checkpoints, and retargeted datasets are available at: https://github.com/amathislab/musclemimic

**선정 근거**
근육 기반 인간 움직임 시뮬레이션으로 선수들의 동작과 자세 분석에 적용 가능

**활용 인사이트**
MuscleMimic 프레임워크로 스포츠 선수의 움직임을 정밀 분석하고 개선점 도출 가능

## 22위: Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training

- arXiv: http://arxiv.org/abs/2603.25527v1
- PDF: https://arxiv.org/pdf/2603.25527v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 74.0 (llm_adjusted:70 = base:70 + bonus:+0)

**개요**
Recent advances in video generation models have achieved impressive results. However, these models heavily rely on the use of high-quality data that combines both high visual quality and high motion quality. In this paper, we identify a key challenge in video data curation: the Motion-Vision Quality Dilemma. We discovered that visual quality and motion intensity inherently exhibit a negative correlation, making it hard to obtain golden data that excels in both aspects. To address this challenge, we first examine the hierarchical learning dynamics of video diffusion models and conduct gradient-based analysis on quality-degraded samples. We discover that quality-imbalanced data can produce gradients similar to golden data at appropriate timesteps. Based on this, we introduce the novel concept of Timestep selection in Training Process. We propose Timestep-aware Quality Decoupling (TQD), which modifies the data sampling distribution to better match the model's learning process. For certain types of data, the sampling distribution is skewed toward higher timesteps for motion-rich data, while high visual quality data is more likely to be sampled during lower timesteps. Through extensive experiments, we demonstrate that TQD enables training exclusively on separated imbalanced data to achieve performance surpassing conventional training with better data, challenging the necessity of perfect data in video generation. Moreover, our method also boosts model performance when trained on high-quality data, showcasing its effectiveness across different data scenarios.

**선정 근거**
비디오 품질 개선 기술로 촬영한 경기 영상의 화질 향상과 하이라이트 편집에 적용 가능

**활용 인사이트**
Timestep-aware Quality Decoupling 기법으로 저사양 장비에서도 고품질 영상 생성 가능

## 23위: Training-free Detection and 6D Pose Estimation of Unseen Surgical Instruments

- arXiv: http://arxiv.org/abs/2603.25228v1
- PDF: https://arxiv.org/pdf/2603.25228v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 74.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Purpose: Accurate detection and 6D pose estimation of surgical instruments are crucial for many computer-assisted interventions. However, supervised methods lack flexibility for new or unseen tools and require extensive annotated data. This work introduces a training-free pipeline for accurate multi-view 6D pose estimation of unseen surgical instruments, which only requires a textured CAD model as prior knowledge. Methods: Our pipeline consists of two main stages. First, for detection, we generate object mask proposals in each view and score their similarity to rendered templates using a pre-trained feature extractor. Detections are matched across views, triangulated into 3D instance candidates, and filtered using multi-view geometric consistency. Second, for pose estimation, a set of pose hypotheses is iteratively refined and scored using feature-metric scores with cross-view attention. The best hypothesis undergoes a final refinement using a novel multi-view, occlusion-aware contour registration, which minimizes reprojection errors of unoccluded contour points. Results: The proposed method was rigorously evaluated on real-world surgical data from the MVPSP dataset. The method achieves millimeter-accurate pose estimates that are on par with supervised methods under controlled conditions, while maintaining full generalization to unseen instruments. These results demonstrate the feasibility of training-free, marker-less detection and tracking in surgical scenes, and highlight the unique challenges in surgical environments. Conclusion: We present a novel and flexible pipeline that effectively combines state-of-the-art foundational models, multi-view geometry, and contour-based refinement for high-accuracy 6D pose estimation of surgical instruments without task-specific training. This approach enables robust instrument tracking and scene understanding in dynamic clinical environments.

**선정 근거**
수술 도구 포즈 추적 기술은 스포츠 선수의 정밀 동작 분석에 직접 적용 가능하며, 학습 없이 새로운 도구에 대응하는 유연성이 우리 프로젝트의 핵심 요구사항과 일치합니다.

**활용 인사이트**
다중 뷰 기하학과 윤곽선 기반 정제 기술을 선수 동작 분석에 적용하여 실시간으로 자세 분석을 제공하고, 이를 바탕으로 개인별 하이라이트 영상을 자동 생성할 수 있습니다.

## 24위: Intelligent Navigation and Obstacle-Aware Fabrication for Mobile Additive Manufacturing Systems

- arXiv: http://arxiv.org/abs/2603.25688v1
- PDF: https://arxiv.org/pdf/2603.25688v1
- 발행일: 2026-03-26
- 카테고리: cs.RO
- 점수: final 74.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
As the demand for mass customization increases, manufacturing systems must become more flexible and adaptable to produce personalized products efficiently. Additive manufacturing (AM) enhances production adaptability by enabling on-demand fabrication of customized components directly from digital models, but its flexibility remains constrained by fixed equipment layouts. Integrating mobile robots addresses this limitation by allowing manufacturing resources to move and adapt to changing production requirements. Mobile AM Robots (MAMbots) combine AM with mobile robotics to produce and transport components within dynamic manufacturing environments. However, the dynamic manufacturing environments introduce challenges for MAMbots. Disturbances such as obstacles and uneven terrain can disrupt navigation stability, which in turn affects printing accuracy and surface quality. This work proposes a universal mobile printing-and-delivery platform that couples navigation and material deposition, addressing the limitations of earlier frameworks that treated these processes separately. A real-time control framework is developed to plan and control the robot's navigation, ensuring safe motion, obstacle avoidance, and path stability while maintaining print quality. The closed-loop integration of sensing, mobility, and manufacturing provides real-time feedback for motion and process control, enabling MAMbots to make autonomous decisions in dynamic environments. The framework is validated through simulations and real-world experiments that test its adaptability to trajectory variations and external disturbances. Coupled navigation and printing together enable MAMbots to plan safe, adaptive trajectories, improving flexibility and adaptability in manufacturing.

**선정 근거**
Mobile robot technology with real-time navigation could be adapted for sports camera positioning systems

## 25위: Interpretable PM2.5 Forecasting for Urban Air Quality: A Comparative Study of Operational Time-Series Models

- arXiv: http://arxiv.org/abs/2603.25495v1
- PDF: https://arxiv.org/pdf/2603.25495v1
- 발행일: 2026-03-26
- 카테고리: cs.LG, cs.AI
- 점수: final 74.0 (llm_adjusted:70 = base:65 + bonus:+5)
- 플래그: 엣지

**개요**
Accurate short-term air-quality forecasting is essential for public health protection and urban management, yet many recent forecasting frameworks rely on complex, data-intensive, and computationally demanding models. This study investigates whether lightweight and interpretable forecasting approaches can provide competitive performance for hourly PM2.5 prediction in Beijing, China. Using multi-year pollutant and meteorological time-series data, we developed a leakage-aware forecasting workflow that combined chronological data partitioning, preprocessing, feature selection, and exogenous-driver modeling under the Perfect Prognosis setting. Three forecasting families were evaluated: SARIMAX, Facebook Prophet, and NeuralProphet. To assess practical deployment behavior, the models were tested under two adaptive regimes: weekly walk-forward refitting and frozen forecasting with online residual correction. Results showed clear differences in both predictive accuracy and computational efficiency. Under walk-forward refitting, Facebook Prophet achieved the strongest completed performance, with an MAE of $37.61$ and an RMSE of $50.10$, while also requiring substantially less execution time than NeuralProphet. In the frozen-model regime, online residual correction improved Facebook Prophet and SARIMAX, with corrected SARIMAX yielding the lowest overall error (MAE $32.50$; RMSE $46.85$). NeuralProphet remained less accurate and less stable across both regimes, and residual correction did not improve its forecasts. Notably, corrected Facebook Prophet reached nearly the same error as its walk-forward counterpart while reducing runtime from $15$ min $21.91$ sec to $46.60$ sec. These findings show that lightweight additive forecasting strategies can remain highly competitive for urban air-quality prediction, offering a practical balance between accuracy, interpretability, ...

**선정 근거**
Lightweight forecasting models could be applicable to edge devices for sports analysis

## 26위: Efficient compressive sensing for machinery vibration signals

- arXiv: http://arxiv.org/abs/2603.25166v1
- PDF: https://arxiv.org/pdf/2603.25166v1
- 발행일: 2026-03-26
- 카테고리: eess.SP
- 점수: final 72.4 (llm_adjusted:68 = base:58 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Mechanical vibration monitoring often requires high sampling rates and generates large data volumes, posing challenges for storage, transmission, and power efficiency. Compressive Sensing (CS) offers a promising approach to overcome these constraints by exploiting signal sparsity to enable sub-Nyquist acquisition and efficient reconstruction. This study presents a comprehensive comparative analysis of the key components of the CS framework: sparse basis, measurement matrix, and reconstruction algorithm for machinery vibration signals. In addition, a hardware-efficient measurement matrix, the Wang matrix, originally developed for image compression, is introduced and evaluated for the first time in this context. Experimental assessment using the HUMS2023 and the CETIM gearbox datasets demonstrates that this matrix achieves superior reconstruction quality, with higher SNR, compared to conventional Gaussian and Bernoulli matrices, especially at high compression ratios.

**선정 근거**
압축 센싱 기술은 스포츠 데이터 효율적 캡처에 적용 가능하여 데이터 저장 및 전송 효율성 향상

**활용 인사이트**
Wang 행렬을 활용한 고압축 비율에서의 우수한 재구성 품질로 실시간 스포츠 영상 처리 가능

## 27위: Modernising Reinforcement Learning-Based Navigation for Embodied Semantic Scene Graph Generation

- arXiv: http://arxiv.org/abs/2603.25415v1
- PDF: https://arxiv.org/pdf/2603.25415v1
- 발행일: 2026-03-26
- 카테고리: cs.AI, cs.RO
- 점수: final 70.0 (llm_adjusted:65 = base:55 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Semantic world models enable embodied agents to reason about objects, relations, and spatial context beyond purely geometric representations. In Organic Computing, such models are a key enabler for objective-driven self-adaptation under uncertainty and resource constraints. The core challenge is to acquire observations maximising model quality and downstream usefulness within a limited action budget.   Semantic scene graphs (SSGs) provide a structured and compact representation for this purpose. However, constructing them within a finite action horizon requires exploration strategies that trade off information gain against navigation cost and decide when additional actions yield diminishing returns.   This work presents a modular navigation component for Embodied Semantic Scene Graph Generation and modernises its decision-making by replacing the policy-optimisation method and revisiting the discrete action formulation. We study compact and finer-grained, larger discrete motion sets and compare a single-head policy over atomic actions with a factorised multi-head policy over action components. We evaluate curriculum learning and optional depth-based collision supervision, and assess SSG completeness, execution safety, and navigation behaviour.   Results show that replacing the optimisation algorithm alone improves SSG completeness by 21\% relative to the baseline under identical reward shaping. Depth mainly affects execution safety (collision-free motion), while completeness remains largely unchanged. Combining modern optimisation with a finer-grained, factorised action representation yields the strongest overall completeness--efficiency trade-off.

**선정 근거**
강화 학습 기반의 장면 이해 기술을 스포츠 경기 분석에 간접적으로 적용 가능

## 28위: MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models

- arXiv: http://arxiv.org/abs/2603.24984v1
- PDF: https://arxiv.org/pdf/2603.24984v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 70.0 (llm_adjusted:65 = base:60 + bonus:+5)
- 플래그: 엣지

**개요**
Mixture-of-Experts (MoE) has emerged as an effective approach to reduce the computational overhead of Transformer architectures by sparsely activating a subset of parameters for each token while preserving high model capacity. This paradigm has recently been extended to Vision-Language Models (VLMs), enabling scalable multi-modal understanding with reduced computational cost. However, the widely adopted deterministic top-K routing mechanism may overlook more optimal expert combinations and lead to expert overfitting. To address this limitation and improve the diversity of expert selection, we propose MoE-GRPO, a reinforcement learning (RL)-based framework for optimizing expert routing in MoE-based VLMs. Specifically, we formulate expert selection as a sequential decision-making problem and optimize it using Group Relative Policy Optimization (GRPO), allowing the model to learn adaptive expert routing policies through exploration and reward-based feedback. Furthermore, we introduce a modality-aware router guidance that enhances training stability and efficiency by discouraging the router from exploring experts that are infrequently activated for a given modality. Extensive experiments on multi-modal image and video benchmarks show that MoE-GRPO consistently outperforms standard top-K routing and its variants by promoting more diverse expert selection, thereby mitigating expert overfitting and enabling a task-level expert specialization.

**선정 근거**
비전-언어 모델의 전문가 라우팅 최적화는 다중 모달 스포츠 분석에 간접적으로 적용 가능

**활용 인사이트**
강화 학습 기반의 MoE-GRPO는 전문가 과적합을 완화하고 작업 수준의 전문화를 가능하게 함

## 29위: Quantum Inspired Vehicular Network Optimization for Intelligent Decision Making in Smart Cities

- arXiv: http://arxiv.org/abs/2603.24971v1
- PDF: https://arxiv.org/pdf/2603.24971v1
- 발행일: 2026-03-26
- 카테고리: cs.NI, quant-ph
- 점수: final 70.0 (llm_adjusted:65 = base:55 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Connected and automated vehicles require city-scale coordination under strict latency and reliability constraints. However, many existing approaches optimize communication and mobility separately, which can degrade performance during network outages and under compute contention. This paper presents QIVNOM, a quantum-inspired framework that jointly optimizes vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) communication together with urban traffic control on classical edge--cloud hardware, without requiring a quantum processor. QIVNOM encodes candidate routing--signal plans as probabilistic superpositions and updates them using sphere-projected gradients with annealed sampling to minimize a regularized objective. An entanglement-style regularizer couples networking and mobility decisions, while Tchebycheff multi-objective scalarization with feasibility projection enforces constraints on latency and reliability.   The proposed framework is evaluated in METR-LA--calibrated SUMO--OMNeT++/Veins simulations over a $5\times5$~km urban map with IEEE 802.11p and 5G NR sidelink. Results show that QIVNOM reduces mean end-to-end latency to 57.3~ms, approximately $20\%$ lower than the best baseline. Under incident conditions, latency decreases from 79~ms to 62~ms ($-21.5\%$), while under roadside unit (RSU) outages, it decreases from 86~ms to 67~ms ($-22.1\%$). Packet delivery reaches $96.7\%$ (an improvement of $+2.3$ percentage points), and reliability remains $96.7\%$ overall, including $96.8\%$ under RSU outages versus $94.1\%$ for the baseline. In corridor-closure scenarios, travel performance also improves, with average travel time reduced to 12.8~min and congestion lowered to $33\%$, compared with 14.5~min and $37\%$ for the baseline.

**선정 근거**
엣지 컴퓨팅 최적화 기술은 실시간 스포츠 영상 처리에 직접적으로 적용 가능

**활용 인사이트**
지연 시간 20% 감소 및 패킷 전달률 96.7% 달성으로 실시간 스포츠 분석 성능 향상

## 30위: Towards Comprehensive Real-Time Scene Understanding in Ophthalmic Surgery through Multimodal Image Fusion

- arXiv: http://arxiv.org/abs/2603.25555v1
- PDF: https://arxiv.org/pdf/2603.25555v1
- 발행일: 2026-03-26
- 카테고리: cs.CV
- 점수: final 66.0 (llm_adjusted:60 = base:55 + bonus:+5)
- 플래그: 실시간

**개요**
Purpose: The integration of multimodal imaging into operating rooms paves the way for comprehensive surgical scene understanding. In ophthalmic surgery, by now, two complementary imaging modalities are available: operating microscope (OPMI) imaging and real-time intraoperative optical coherence tomography (iOCT). This first work toward temporal OPMI and iOCT feature fusion demonstrates the potential of multimodal image processing for multi-head prediction through the example of precise instrument tracking in vitreoretinal surgery.   Methods: We propose a multimodal, temporal, real-time capable network architecture to perform joint instrument detection, keypoint localization, and tool-tissue distance estimation. Our network design integrates a cross-attention fusion module to merge OPMI and iOCT image features, which are efficiently extracted via a YoloNAS and a CNN encoder, respectively. Furthermore, a region-based recurrent module leverages temporal coherence.   Results: Our experiments demonstrate reliable instrument localization and keypoint detection (95.79% mAP50) and show that the incorporation of iOCT significantly improves tool-tissue distance estimation, while achieving real-time processing rates of 22.5 ms per frame. Especially for close distances to the retina (below 1 mm), the distance estimation accuracy improved from 284 $μm$ (OPMI only) to 33 $μm$ (multimodal).   Conclusion: Feature fusion of multimodal imaging can enhance multi-task prediction accuracy compared to single-modality processing and real-time processing performance can be achieved through tailored network design. While our results demonstrate the potential of multi-modal processing for image-guided vitreoretinal surgery, they also underline key challenges that motivate future research toward more reliable, consistent, and comprehensive surgical scene understanding.

**선정 근거**
다중 모달 이미지 융합 기술은 실시간 스포츠 장면 이해에 적용 가능하며, 다양한 각도의 영상을 결합하여 하이라이트 장면을 자동으로 식별하고 선수의 움직임을 추적하는 데 효과적입니다.

**활용 인사이트**
의료 영상 분석에서 사용된 크로스 어텐션 융합 모듈을 스포츠 카메라 시스템에 적용하여 다중 각도에서의 실시간 영상 분석을 구현하고, 22.5ms의 처리 지연 시간으로 경기 중 중요 순간을 놓치지 않고 캡처할 수 있습니다.

## 31위: AirSplat: Alignment and Rating for Robust Feed-Forward 3D Gaussian Splatting

- arXiv: http://arxiv.org/abs/2603.25129v1 | 2026-03-26 | final 66.0

While 3D Vision Foundation Models (3DVFMs) have demonstrated remarkable zero-shot capabilities in visual geometry estimation, their direct application to generalizable novel view synthesis (NVS) remains challenging. In this paper, we propose AirSplat, a novel training framework that effectively adapts the robust geometric priors of 3DVFMs into high-fidelity, pose-free NVS.

-> 3D 기하학적 기반의 뷰 합성 기술로 스포츠 장면의 3D 재구성에 간접적으로 적용 가능

## 32위: A Minimum-Energy Control Approach for Redundant Mobile Manipulators in Physical Human-Robot Interaction Applications

- arXiv: http://arxiv.org/abs/2603.25259v1 | 2026-03-26 | final 66.0

Research on mobile manipulation systems that physically interact with humans has expanded rapidly in recent years, opening the way to tasks which could not be performed using fixed-base manipulators. Within this context, developing suitable control methodologies is essential since mobile manipulators introduce additional degrees of freedom, making the design of control approaches more challenging and more prone to performance optimization.

-> 인간-로봇 상호작용이지만 제어 시스템용으로 스포츠 촬영과 직접 관련 없음

## 33위: Longitudinal Digital Phenotyping for Early Cognitive-Motor Screening

- arXiv: http://arxiv.org/abs/2603.25673v1 | 2026-03-26 | final 66.0

Early detection of atypical cognitive-motor development is critical for timely intervention, yet traditional assessments rely heavily on subjective, static evaluations. The integration of digital devices offers an opportunity for continuous, objective monitoring through digital biomarkers.

-> 동작 분석이지만 아동 발달 선별용으로 스포츠 분석과 간접적 관련만 있음

## 34위: Learning to Rank Caption Chains for Video-Text Alignment

- arXiv: http://arxiv.org/abs/2603.25145v1 | 2026-03-26 | final 66.0

Direct preference optimization (DPO) is an effective technique to train language models to generate preferred over dispreferred responses. However, this binary "winner-takes-all" approach is suboptimal for vision-language models whose response quality is highly dependent on visual content.

-> 비디오-텍스트 정렬 기술이 스포츠 영상 분석 및 하이라이트 생성에 간접적으로 적용 가능

## 35위: Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale

- arXiv: http://arxiv.org/abs/2603.25040v1 | 2026-03-26 | final 66.0

We introduce Intern-S1-Pro, the first one-trillion-parameter scientific multimodal foundation model. Scaling to this unprecedented size, the model delivers a comprehensive enhancement across both general and scientific domains.

-> 대규모 멀티모달 모델이 스포츠 비디오 분석에 적용 가능하며 다양한 전문 작업 처리 가능

## 36위: LanteRn: Latent Visual Structured Reasoning

- arXiv: http://arxiv.org/abs/2603.25629v1 | 2026-03-26 | final 66.0

While language reasoning models excel in many tasks, visual reasoning remains challenging for current large multimodal models (LMMs). As a result, most LMMs default to verbalizing perceptual content into text, a strong limitation for tasks requiring fine-grained spatial and visual understanding.

-> 시각적 추론 프레임워크가 스포츠 분석에 간접적으로 적용 가능하며 잠재 공간 접근법이 효율적인 비디오 처리에 유용

## 37위: Few-Shot Left Atrial Wall Segmentation in 3D LGE MRI via Meta-Learning

- arXiv: http://arxiv.org/abs/2603.24985v1 | 2026-03-26 | final 66.0

Segmenting the left atrial wall from late gadolinium enhancement magnetic resonance images (MRI) is challenging due to the wall's thin geometry, low contrast, and the scarcity of expert annotations. We propose a Model-Agnostic Meta-Learning (MAML) framework for K-shot (K = 5, 10, 20) 3D left atrial wall segmentation that is meta-trained on the wall task together with auxiliary left atrial and right atrial cavity tasks and uses a boundary-aware composite loss to emphasize thin-structure accuracy.

-> Meta-learning approach for image segmentation could be applicable to sports video analysis

## 38위: $π$, But Make It Fly: Physics-Guided Transfer of VLA Models to Aerial Manipulation

- arXiv: http://arxiv.org/abs/2603.25038v1 | 2026-03-26 | final 64.4

Vision-Language-Action (VLA) models such as $π_0$ have demonstrated remarkable generalization across diverse fixed-base manipulators. However, transferring these foundation models to aerial platforms remains an open challenge due to the fundamental mismatch between the quasi-static dynamics of fixed-base arms and the underactuated, highly dynamic nature of flight.

-> 비전-언어-행동 모델이 항공 조작에 간접적으로 스포츠 동작 분석과 관련

## 39위: GeoHeight-Bench: Towards Height-Aware Multimodal Reasoning in Remote Sensing

- arXiv: http://arxiv.org/abs/2603.25565v1 | 2026-03-26 | final 64.4

Current Large Multimodal Models (LMMs) in Earth Observation typically neglect the critical "vertical" dimension, limiting their reasoning capabilities in complex remote sensing geometries and disaster scenarios where physical spatial structures often outweigh planar visual textures. To bridge this gap, we introduce a comprehensive evaluation framework dedicated to height-aware remote sensing understanding.

-> 원격 감시의 다중 모델 추론 기술이 스포츠 분석에 간접적으로 적용 가능

## 40위: Infinite Gaze Generation for Videos with Autoregressive Diffusion

- arXiv: http://arxiv.org/abs/2603.24938v1 | 2026-03-26 | final 62.0

Predicting human gaze in video is fundamental to advancing scene understanding and multimodal interaction. While traditional saliency maps provide spatial probability distributions and scanpaths offer ordered fixations, both abstractions often collapse the fine-grained temporal dynamics of raw gaze.

-> 비디오 내 시선 예측 기술이 스포츠 선수 추적에 간접적으로 적용 가능

## 41위: SAVe: Self-Supervised Audio-visual Deepfake Detection Exploiting Visual Artifacts and Audio-visual Misalignment

- arXiv: http://arxiv.org/abs/2603.25140v1 | 2026-03-26 | final 62.0

Multimodal deepfakes can exhibit subtle visual artifacts and cross-modal inconsistencies, which remain challenging to detect, especially when detectors are trained primarily on curated synthetic forgeries. Such synthetic dependence can introduce dataset and generator bias, limiting scalability and robustness to unseen manipulations.

-> 오디오-비주얼 정렬 기술이 스포츠 영상 분석에 간접적으로 적용 가능

## 42위: Belief-Driven Multi-Agent Collaboration via Approximate Perfect Bayesian Equilibrium for Social Simulation

- arXiv: http://arxiv.org/abs/2603.24973v1 | 2026-03-26 | final 60.4

High-fidelity social simulation is pivotal for addressing complex Web societal challenges, yet it demands agents capable of authentically replicating the dynamic spectrum of human interaction. Current LLM-based multi-agent frameworks, however, predominantly adhere to static interaction topologies, failing to capture the fluid oscillation between cooperative knowledge synthesis and competitive critical reasoning seen in real-world scenarios.

-> Multi-agent collaboration framework could be partially applicable to game strategy analysis in sports

## 43위: Sovereign AI at the Front Door of Care: A Physically Unidirectional Architecture for Secure Clinical Intelligence

- arXiv: http://arxiv.org/abs/2603.24898v1 | 2026-03-26 | final 60.0

We present a Sovereign AI architecture for clinical triage in which all inference is performed on-device and inbound data is delivered via a physically unidirectional channel, implemented using receive-only broadcast infrastructure or certified hardware data diodes, with no return path to any external network. This design removes the network-mediated attack surface by construction, rather than attempting to secure it through software controls.

-> 엣지 컴퓨팅 아키텍처이지만 의료용으로 스포츠 플랫폼과 무관

## 44위: Synergistic Event-SVE Imaging for Quantitative Propellant Combustion Diagnostics

- arXiv: http://arxiv.org/abs/2603.25054v1 | 2026-03-26 | final 58.0

Real-time monitoring of high-energy propellant combustion is difficult. Extreme high dynamic range (HDR), microsecond-scale particle motion, and heavy smoke often occur together.

-> 실시간 이미징 시스템이지만 연소 진단용으로 스포츠와 직접 관련 없음

## 45위: The Competence Shadow: Theory and Bounds of AI Assistance in Safety Engineering

- arXiv: http://arxiv.org/abs/2603.25197v1 | 2026-03-26 | final 58.0

As AI assistants become integrated into safety engineering workflows for Physical AI systems, a critical question emerges: does AI assistance improve safety analysis quality, or introduce systematic blind spots that surface only through post-deployment incidents? This paper develops a formal framework for AI assistance in safety analysis.

-> Safety engineering frameworks for AI could inform the development of reliable sports analysis systems

## 46위: VolDiT: Controllable Volumetric Medical Image Synthesis with Diffusion Transformers

- arXiv: http://arxiv.org/abs/2603.25181v1 | 2026-03-26 | final 56.4

Diffusion models have become a leading approach for high-fidelity medical image synthesis. However, most existing methods for 3D medical image generation rely on convolutional U-Net backbones within latent diffusion frameworks.

-> 이미지 합성 기술이지만 의료 분야용으로 스포츠 콘텐츠 제작과 무관

## 47위: Subject-Specific Low-Field MRI Synthesis via a Neural Operator

- arXiv: http://arxiv.org/abs/2603.24968v1 | 2026-03-26 | final 54.0

Low-field (LF) magnetic resonance imaging (MRI) improves accessibility and reduces costs but generally has lower signal-to-noise ratios and degraded contrast compared to high field (HF) MRI, limiting its clinical utility. Simulating LF MRI from HF MRI enables virtual evaluation of novel imaging devices and development of LF algorithms.

-> MRI image synthesis techniques could be partially applicable to enhancing sports videos/images

## 48위: A CDF-First Framework for Free-Form Density Estimation

- arXiv: http://arxiv.org/abs/2603.25204v1 | 2026-03-26 | final 54.0

Conditional density estimation (CDE) is a fundamental task in machine learning that aims to model the full conditional law $\mathbb{P}(\mathbf{y} \mid \mathbf{x})$, beyond mere point prediction (e.g., mean, mode). A core challenge is free-form density estimation, capturing distributions that exhibit multimodality, asymmetry, or topological complexity without restrictive assumptions.

-> Conditional density estimation could be partially applicable to analyzing sports performance data

## 49위: FSGNet: A Frequency-Aware and Semantic Guidance Network for Infrared Small Target Detection

- arXiv: http://arxiv.org/abs/2603.25389v1 | 2026-03-26 | final 52.4

Infrared small target detection (IRSTD) aims to identify and distinguish small targets from complex backgrounds. Leveraging the powerful multi-scale feature fusion capability of the U-Net architecture, IRSTD has achieved significant progress.

-> 경량화 적외선 탐지 프레임워크가 일반 컴퓨터 비전 요구사항과 약간 관련

---

## 다시 보기

### TRINE: A Token-Aware, Runtime-Adaptive FPGA Inference Engine for Multimodal AI (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.22867v1
- 점수: final 100.0

Multimodal stacks that mix ViTs, CNNs, GNNs, and transformer NLP strain embedded platforms because their compute/memory patterns diverge and hard real-time targets leave little slack. TRINE is a single-bitstream FPGA accelerator and compiler that executes end-to-end multimodal inference without reconfiguration. Layers are unified as DDMM/SDDMM/SpMM and mapped to a mode-switchable engine that toggles at runtime among weight/output-stationary systolic, 1xCS SIMD, and a routable adder tree (RADT) on a shared PE array. A width-matched, two-stage top-k unit enables in-stream token pruning, while dependency-aware layer offloading (DALO) overlaps independent kernels across reconfigurable processing units to sustain utilization. Evaluated on Alveo U50 and ZCU104, TRINE reduces latency by up to 22.57x vs. RTX 4090 and 6.86x vs. Jetson Orin Nano at 20-21 W; token pruning alone yields up to 7.8x on ViT-heavy pipelines, and DALO contributes up to 79% throughput improvement. With int8 quantization, accuracy drops remain <2.5% across representative tasks, delivering state-of-the-art latency and energy efficiency for unified vision, language, and graph workloads-in one bitstream.

-> TRINE은 다중 모달 AI 가속을 위한 FPGA 엔진으로, rk3588 기반의 에지 디바이스에서 스포츠 촬영 및 분석을 위한 여러 AI 모델을 효율적으로 실행할 수 있게 해줍니다.

### Bridging Biological Hearing and Neuromorphic Computing: End-to-End Time-Domain Audio Signal Processing with Reservoir Computing (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.24283v1
- 점수: final 93.6

Despite the advancements in cutting-edge technologies, audio signal processing continues to pose challenges and lacks the precision of a human speech processing system. To address these challenges, we propose a novel approach to simplify audio signal processing by leveraging time-domain techniques and reservoir computing. Through our research, we have developed a real-time audio signal processing system by simplifying audio signal processing through the utilization of reservoir computers, which are significantly easier to train.   Feature extraction is a fundamental step in speech signal processing, with Mel Frequency Cepstral Coefficients (MFCCs) being a dominant choice due to their perceptual relevance to human hearing. However, conventional MFCC extraction relies on computationally intensive time-frequency transformations, limiting efficiency in real-time applications. To address this, we propose a novel approach that leverages reservoir computing to streamline MFCC extraction. By replacing traditional frequency-domain conversions with convolution operations, we eliminate the need for complex transformations while maintaining feature discriminability. We present an end-to-end audio processing framework that integrates this method, demonstrating its potential for efficient and real-time speech analysis. Our results contribute to the advancement of energy-efficient audio processing technologies, enabling seamless deployment in embedded systems and voice-driven applications. This work bridges the gap between biologically inspired feature extraction and modern neuromorphic computing, offering a scalable solution for next-generation speech recognition systems.

-> 실시간 오디오 처리 기술이 엣지 디바이스에 적용 가능

### TorR: Towards Brain-Inspired Task-Oriented Reasoning via Cache-Oriented Algorithm-Architecture Co-design (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.22855v1
- 점수: final 93.6

Task-oriented object detection (TOOD) atop CLIP offers open-vocabulary, prompt-driven semantics, yet dense per-window computation and heavy memory traffic hinder real-time, power-limited edge deployment. We present \emph{TorR}, a brain-inspired \textbf{algorithm--architecture co-design} that \textbf{replaces CLIP-style dense alignment with a hyperdimensional (HDC) associative reasoner} and turns temporal coherence into reuse. On the \emph{algorithm} side, TorR reformulates alignment as HDC similarity and graph composition, introducing \emph{partial-similarity reuse} via (i) query caching with per-class score accumulation, (ii) exact $δ$-updates when only a small set of hypervector bits change, and (iii) similarity/load-gated bypass under high system load. On the \emph{architecture} side, TorR instantiates a lane-scalable, bit-sliced item memory with bank/precision gating and a lightweight controller that schedules bypass/$δ$/full paths to meet RT-30/RT-60 targets as object counts vary. Synthesized in a TSMC 28\,nm process and exercised with a cycle-accurate simulator, TorR sustains real-time throughput with millijoule-scale energy per window ($\approx$50\,mJ at 60\,FPS; $\approx$113\,mJ at 30\,FPS) and low latency jitter, while delivering competitive AP@0.5 across five task prompts (mean 44.27\%) within a bounded margin to strong VLM baselines, but at orders-of-magnitude lower energy. The design exposes deployment-time configurability (effective dimension $D'$, thresholds, precision) to trade accuracy, latency, and energy for edge budgets.

-> TorR는 에지 디바이스에서 실시간 객체 탐지를 위한 뇀 영감 기술로, 스포츠 장면에서 선수 및 공 등의 실시간 추적과 분석에 적합합니다.

### Short-Term Turbulence Prediction for Seeing Using Machine Learning (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.24466v1
- 점수: final 92.0

Optical turbulence, driven by fluctuations of the atmospheric refractive index, poses a significant challenge to ground-based optical systems, as it distorts the propagation of light. This degradation affects both astronomical observations and free-space optical communications. While adaptive optics systems correct turbulence effects in real-time, their reactive nature limits their effectiveness under rapidly changing conditions, underscoring the need for predictive solutions. In this study, we address the problem of short-term turbulence forecasting by leveraging machine learning models to predict the atmospheric seeing parameter up to two hours in advance. We compare statistical and deep learning approaches, with a particular focus on probabilistic models that not only produce accurate forecasts but also quantify predictive uncertainty, crucial for robust decision-making in dynamic environments. Our evaluation includes Gaussian processes (GP) for statistical modeling, recurrent neural networks (RNNs) and long short-term memory networks (LSTMs) as deterministic baselines, and our novel implementation of a normalizing flow for time series (FloTS) as a flexible probabilistic deep learning method. All models are trained exclusively on historical seeing data, allowing for a fair performance comparison. We show that FloTS achieves the best overall balance between predictive accuracy and well-calibrated uncertainty.

-> 카메라 안정화 및 이미지 향상 기술은 스포츠 촬영의 품질을 결정하는 중요 요소입니다.

### Towards Safe Learning-Based Non-Linear Model Predictive Control through Recurrent Neural Network Modeling (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.24503v1
- 점수: final 92.0

The practical deployment of nonlinear model predictive control (NMPC) is often limited by online computation: solving a nonlinear program at high control rates can be expensive on embedded hardware, especially when models are complex or horizons are long. Learning-based NMPC approximations shift this computation offline but typically demand large expert datasets and costly training. We propose Sequential-AMPC, a sequential neural policy that generates MPC candidate control sequences by sharing parameters across the prediction horizon. For deployment, we wrap the policy in a safety-augmented online evaluation and fallback mechanism, yielding Safe Sequential-AMPC. Compared to a naive feedforward policy baseline across several benchmarks, Sequential-AMPC requires substantially fewer expert MPC rollouts and yields candidate sequences with higher feasibility rates and improved closed-loop safety. On high-dimensional systems, it also exhibits better learning dynamics and performance in fewer epochs while maintaining stable validation improvement where the feedforward baseline can stagnate.

-> 실시간 제어 시스템이 스포츠 카메라 작동에 필수적이며, 안정적인 추적을 보장합니다.

### Mixed-signal implementation of feedback-control optimizer for single-layer Spiking Neural Networks (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.24113v1
- 점수: final 92.0

On-chip learning is key to scalable and adaptive neuromorphic systems, yet existing training methods are either difficult to implement in hardware or overly restrictive. However, recent studies show that feedback-control optimizers can enable expressive, on-chip training of neuromorphic devices. In this work, we present a proof-of-concept implementation of such feedback-control optimizers on a mixed-signal neuromorphic processor. We assess the proposed approach in an In-The-Loop(ITL) training setup on both a binary classification task and the nonlinear Yin-Yang problem, demonstrating on-chip training that matches the performance of numerical simulations and gradient-based baselines. Our results highlight the feasibility of feedback-driven, online learning under realistic mixed-signal constraints, and represent a co-design approach toward embedding such rules directly in silicon for autonomous and adaptive neuromorphic computing.

-> 뉴로모픽 컴퓨팅은 에지 디바이스의 효율성을 높여 실시간 처리를 가능하게 합니다.

### Short-Form Video Viewing Behavior Analysis and Multi-Step Viewing Time Prediction (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.22663v1
- 점수: final 92.0

Short-form videos have become one of the most popular user-generated content formats nowadays. Popular short-video platforms use a simple streaming approach that preloads one or more videos in the recommendation list in advance. However, this approach results in significant data wastage, as a large portion of the downloaded video data is not used due to the user's early skip behavior. To address this problem, the chunk-based preloading approach has been proposed, where videos are divided into chunks, and preloading is performed in a chunk-based manner to reduce data wastage. To optimize chunk-based preloading, it is important to understand the user's viewing behavior in short-form video streaming. In this paper, we conduct a measurement study to construct a user behavior dataset that contains users' viewing times of one hundred short videos of various categories. Using the dataset, we evaluate the performance of standard time-series forecasting algorithms for predicting user viewing time in short-form video streaming. Our evaluation results show that Auto-ARIMA generally achieves the lowest and most stable forecasting errors across most experimental settings. The remaining methods, including AR, LR, SVR, and DTR, tend to produce higher errors and exhibit lower stability in many cases. The dataset is made publicly available at https://nvduc.github.io/shortvideodataset.

-> 숏폼 영상 시청 행동 분석 연구는 스포츠 하이라이트 플랫폼의 콘텐츠 추천 알고리즘을 최적화하는 데 직접적으로 활용될 수 있습니다.

### RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.24295v1
- 점수: final 90.4

Recently, state space models have demonstrated efficient video segmentation through linear-complexity state space compression. However, Video Semantic Segmentation (VSS) requires pixel-level spatiotemporal modeling capabilities to maintain temporal consistency in segmentation of semantic objects. While state space models can preserve common semantic information during state space compression, the fixed-size state space inevitably forgets specific information, which limits the models' capability for pixel-level segmentation. To tackle the above issue, we proposed a Refining Specifics State Space Model approach (RS-SSM) for video semantic segmentation, which performs complementary refining of forgotten spatiotemporal specifics. Specifically, a Channel-wise Amplitude Perceptron (CwAP) is designed to extract and align the distribution characteristics of specific information in the state space. Besides, a Forgetting Gate Information Refiner (FGIR) is proposed to adaptively invert and refine the forgetting gate matrix in the state space model based on the specific information distribution. Consequently, our RS-SSM leverages the inverted forgetting gate to complementarily refine the specific information forgotten during state space compression, thereby enhancing the model's capability for spatiotemporal pixel-level segmentation. Extensive experiments on four VSS benchmarks demonstrate that our RS-SSM achieves state-of-the-art performance while maintaining high computational efficiency. The code is available at https://github.com/zhoujiahuan1991/CVPR2026-RS-SSM.

-> 비디오 의미 분할 기술이 스포츠 촬영, 하이라이트 추출, 전략 분석에 직접적으로 적용 가능합니다.

### Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.22786v1
- 점수: final 89.6

Recent advances in 3D Gaussian Splatting have enabled impressive photorealistic novel view synthesis. However, to transition from a pure rendering engine to a reliable spatial map for autonomous agents and safety-critical applications, knowing where the representation is uncertain is as important as the rendering fidelity itself. We bridge this critical gap by introducing a lightweight, plug-and-play framework for pixel-wise, view-dependent predictive uncertainty estimation. Our post-hoc method formulates uncertainty as a Bayesian-regularized linear least-squares optimization over reconstruction residuals. This architecture-agnostic approach extracts a per-primitive uncertainty channel without modifying the underlying scene representation or degrading baseline visual fidelity. Crucially, we demonstrate that providing this actionable reliability signal successfully translates 3D Gaussian splatting into a trustworthy spatial map, further improving state-of-the-art performance across three critical downstream perception tasks: active view selection, pose-agnostic scene change detection, and pose-agnostic anomaly detection.

-> 3D 가우시안 스플래팅의 불확실성 예측 기술은 스포츠 장면의 3D 재구성 및 중요 순간 정확히 식별 가능

### Toward Integrated Sensing, Communications, and Edge Intelligence Networks (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.22958v1
- 점수: final 88.0

Wireless systems are expanding their purposes, from merely connecting humans and things to connecting intelligence and opportunistically sensing of the environment through radio-frequency signals. In this paper, we introduce the concept of triple-functional networks in which the same infrastructure and resources are shared for integrated sensing, communications, and (edge) Artificial Intelligence (AI) inference. This concept opens up several opportunities, such as devising non-orthogonal resource deployment and power consumption to concurrently update multiple services, but also challenges related to resource management and signaling cross-talk, among others. The core idea of this work is that computation-related aspects, including computing resources and AI models availability, should be explicitly considered when taking resource allocation decisions, to address the conflicting goals of the services coexistence. After showing the natural coupling between theoretical performance bounds of the three services, we formulate a service coexistence optimization problem that is solved optimally, and showcase the advantages against a disjoint allocation strategy.

-> 통합 센싱, 통신, 에지 AI 네트워크로 스포츠 촬영 및 분석 시스템에 활용 가능

### B-MoE: A Body-Part-Aware Mixture-of-Experts "All Parts Matter" Approach to Micro-Action Recognition (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.24245v1
- 점수: final 86.4

Micro-actions, fleeting and low-amplitude motions, such as glances, nods, or minor posture shifts, carry rich social meaning but remain difficult for current action recognition models to recognize due to their subtlety, short duration, and high inter-class ambiguity. In this paper, we introduce B-MoE, a Body-part-aware Mixture-of-Experts framework designed to explicitly model the structured nature of human motion. In B-MoE, each expert specializes in a distinct body region (head, body, upper limbs, lower limbs), and is based on the lightweight Macro-Micro Motion Encoder (M3E) that captures long-range contextual structure and fine-grained local motion. A cross-attention routing mechanism learns inter-region relationships and dynamically selects the most informative regions for each micro-action. B-MoE uses a dual-stream encoder that fuses these region-specific semantic cues with global motion features to jointly capture spatially localized cues and temporally subtle variations that characterize micro-actions. Experiments on three challenging benchmarks (MA-52, SocialGesture, and MPII-GroupInteraction) show consistent state-of-theart gains, with improvements in ambiguous, underrepresented, and low amplitude classes.

-> 신체 부위별 전문가 모델을 활용한 미세 동작 인식은 스포츠 자세 분석과 하이라이트 장면 자동 추출에 적용 가능하나 스포츠 특화 연구는 아님

### How Vulnerable Are Edge LLMs? (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.23822v1
- 점수: final 86.4

Large language models (LLMs) are increasingly deployed on edge devices under strict computation and quantization constraints, yet their security implications remain unclear. We study query-based knowledge extraction from quantized edge-deployed LLMs under realistic query budgets and show that, although quantization introduces noise, it does not remove the underlying semantic knowledge, allowing substantial behavioral recovery through carefully designed queries. To systematically analyze this risk, we propose \textbf{CLIQ} (\textbf{Cl}ustered \textbf{I}nstruction \textbf{Q}uerying), a structured query construction framework that improves semantic coverage while reducing redundancy. Experiments on quantized Qwen models (INT8/INT4) demonstrate that CLIQ consistently outperforms original queries across BERTScore, BLEU, and ROUGE, enabling more efficient extraction under limited budgets. These results indicate that quantization alone does not provide effective protection against query-based extraction, highlighting a previously underexplored security risk in edge-deployed LLMs.

-> Directly addresses edge device deployment challenges and capabilities, relevant to the project's edge device focus.

### Dual-Teacher Distillation with Subnetwork Rectification for Black-Box Domain Adaptation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.22908v1
- 점수: final 85.6

Assuming that neither source data nor the source model is accessible, black box domain adaptation represents a highly practical yet extremely challenging setting, as transferable information is restricted to the predictions of the black box source model, which can only be queried using target samples. Existing approaches attempt to extract transferable knowledge through pseudo label refinement or by leveraging external vision language models (ViLs), but they often suffer from noisy supervision or insufficient utilization of the semantic priors provided by ViLs, which ultimately hinder adaptation performance. To overcome these limitations, we propose a dual teacher distillation with subnetwork rectification (DDSR) model that jointly exploits the specific knowledge embedded in black box source models and the general semantic information of a ViL. DDSR adaptively integrates their complementary predictions to generate reliable pseudo labels for the target domain and introduces a subnetwork driven regularization strategy to mitigate overfitting caused by noisy supervision. Furthermore, the refined target predictions iteratively enhance both the pseudo labels and ViL prompts, enabling more accurate and semantically consistent adaptation. Finally, the target model is further optimized through self training with classwise prototypes. Extensive experiments on multiple benchmark datasets validate the effectiveness of our approach, demonstrating consistent improvements over state of the art methods, including those using source data or models.

-> 도메인 적응 기술은 다양한 스포츠 환경에서 AI 모델을 효과적으로 적용시키는 데 필수적입니다.

### GTLR-GS: Geometry-Texture Aware LiDAR-Regularized 3D Gaussian Splatting for Realistic Scene Reconstruction (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.23192v1
- 점수: final 84.0

Recent advances in 3D Gaussian Splatting (3DGS) have enabled real-time, photorealistic scene reconstruction. However, conventional 3DGS frameworks typically rely on sparse point clouds derived from Structure-from-Motion (SfM), which inherently suffer from scale ambiguity, limited geometric consistency, and strong view dependency due to the lack of geometric priors. In this work, a LiDAR-centric 3D Gaussian Splatting framework is proposed that explicitly incorporates metric geometric priors into the entire Gaussian optimization process. Instead of treating LiDAR data as a passive initialization source, 3DGS optimization is reformulated as a geometry-conditioned allocation and refinement problem under a fixed representational budget. Specifically, this work introduces (i) a geometry-texture-aware allocation strategy that selectively assigns Gaussian primitives to regions with high structural or appearance complexity, (ii) a curvature-adaptive refinement mechanism that dynamically guides Gaussian splitting toward geometrically complex areas during training, and (iii) a confidence-aware metric depth regularization that anchors the reconstructed geometry to absolute scale using LiDAR measurements while maintaining optimization stability. Extensive experiments on the ScanNet++ dataset and a custom real-world dataset validate the proposed approach. The results demonstrate state-of-the-art performance in metric-scale reconstruction with high geometric fidelity.

-> LiDAR 기반 3D 가우시안 스플래팅은 스포츠 장면의 실시간 3D 재구성을 통해 다각도 분석을 가능하게 합니다.

### VQ-Jarvis: Retrieval-Augmented Video Restoration Agent with Sharp Vision and Fast Thought (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.22998v1
- 점수: final 82.4

Video restoration in real-world scenarios is challenged by heterogeneous degradations, where static architectures and fixed inference pipelines often fail to generalize. Recent agent-based approaches offer dynamic decision making, yet existing video restoration agents remain limited by insufficient quality perception and inefficient search strategies. We propose VQ-Jarvis, a retrieval-augmented, all-in-one intelligent video restoration agent with sharper vision and faster thought. VQ-Jarvis is designed to accurately perceive degradations and subtle differences among paired restoration results, while efficiently discovering optimal restoration trajectories. To enable sharp vision, we construct VSR-Compare, the first large-scale video paired enhancement dataset with 20K comparison pairs covering 7 degradation types, 11 enhancement operators, and diverse content domains. Based on this dataset, we train a multiple operator judge model and a degradation perception model to guide agent decisions. To achieve fast thought, we introduce a hierarchical operator scheduling strategy that adapts to video difficulty: for easy cases, optimal restoration trajectories are retrieved in a one-step manner from a retrieval-augmented generation (RAG) library; for harder cases, a step-by-step greedy search is performed to balance efficiency and accuracy. Extensive experiments demonstrate that VQ-Jarvis consistently outperforms existing methods on complex degraded videos.

-> VQ-Jarvis는 실시간 비디오 복원 기술로 스포츠 촬영 품질 향상에 직접 적용 가능하며, 다양한 품질 저하 상황에 대응할 수 있는 강력한 복원 능력을 제공합니다.

### ViHOI: Human-Object Interaction Synthesis with Visual Priors (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.24383v1
- 점수: final 80.0

Generating realistic and physically plausible 3D Human-Object Interactions (HOI) remains a key challenge in motion generation. One primary reason is that describing these physical constraints with words alone is difficult. To address this limitation, we propose a new paradigm: extracting rich interaction priors from easily accessible 2D images. Specifically, we introduce ViHOI, a novel framework that enables diffusion-based generative models to leverage rich, task-specific priors from 2D images to enhance generation quality. We utilize a large Vision-Language Model (VLM) as a powerful prior-extraction engine and adopt a layer-decoupled strategy to obtain visual and textual priors. Concurrently, we design a Q-Former-based adapter that compresses the VLM's high-dimensional features into compact prior tokens, which significantly facilitates the conditional training of our diffusion model. Our framework is trained on motion-rendered images from the dataset to ensure strict semantic alignment between visual inputs and motion sequences. During inference, it leverages reference images synthesized by a text-to-image generation model to improve generalization to unseen objects and interaction categories. Experimental results demonstrate that ViHOI achieves state-of-the-art performance, outperforming existing methods across multiple benchmarks and demonstrating superior generalization.

-> 인간-물체 상호작용 생성에 초점을 맞춰 스포츠 동작 분석 및 하이라이트 영상 생성에 직접적으로 적용 가능

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
