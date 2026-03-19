# CAPP!C_AI 논문 리포트 (2026-03-19)

> 수집 83 | 필터 79 | 폐기 7 | 평가 63 | 출력 55 | 기준 50점

검색 윈도우: 2026-03-18T00:00:00+00:00 ~ 2026-03-19T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 43

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: SpiderCam: Low-Power Snapshot Depth from Differential Defocus

- arXiv: http://arxiv.org/abs/2603.17910v1
- PDF: https://arxiv.org/pdf/2603.17910v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 100.0 (llm_adjusted:100 = base:95 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
We introduce SpiderCam, an FPGA-based snapshot depth-from-defocus camera which produces 480x400 sparse depth maps in real-time at 32.5 FPS over a working range of 52 cm while consuming 624 mW of power in total. SpiderCam comprises a custom camera that simultaneously captures two differently focused images of the same scene, processed with a SystemVerilog implementation of depth from differential defocus (DfDD) on a low-power FPGA. To achieve state-of-the-art power consumption, we present algorithmic improvements to DfDD that overcome challenges caused by low-power sensors, and design a memory-local implementation for streaming depth computation on a device that is too small to store even a single image pair. We report the first sub-Watt total power measurement for passive FPGA-based 3D cameras in the literature.

**선정 근거**
스포츠 촬영을 위한 저전력 엣지 디바이스에 직접 적용 가능한 깊이 맵 생성 기술

**활용 인사이트**
32.5fps로 실시간 깊이 정보를 제공하며 624mW의 저전력으로 rk3588 기반 장치에 통합 가능

## 2위: A spatio-temporal graph-based model for team sports analysis

- arXiv: http://arxiv.org/abs/2603.17471v1
- PDF: https://arxiv.org/pdf/2603.17471v1
- 발행일: 2026-03-18
- 카테고리: cs.SI
- 점수: final 96.0 (llm_adjusted:95 = base:95 + bonus:+0)

**개요**
Team sports represent complex phenomena characterized by both spatial and temporal dimensions, making their analysis inherently challenging. In this study, we examine team sports as complex systems, specifically focusing on the tactical aspects influenced by external constraints. To this end, we introduce a new generic graph-based model to analyze these phenomena. Specifically, we model a team sport's attacking play as a directed path containing absolute and relative ball carrier-centered spatial information, temporal information, and semantic information. We apply our model to union rugby, aiming to validate two hypotheses regarding the impact of the pedagogy provided by the coach on the one hand, and the effect of the initial positioning of the defensive team on the other hand. Preliminary results from data collected on six-player rugby from several French clubs indicate notable effects of these constraints. The model is intended to be applied to other team sports and to validate additional hypotheses related to team coordination patterns, including upcoming applications in basketball.

**선정 근거**
팀 스포츠 전략 분석에 직접적으로 적용 가능한 그래프 기반 모델

**활용 인사이트**
공간적, 시간적 정보를 결합해 공격 전략을 분석하며 다른 스포츠로 확장 가능

## 3위: Universal Skeleton Understanding via Differentiable Rendering and MLLMs

- arXiv: http://arxiv.org/abs/2603.18003v1
- PDF: https://arxiv.org/pdf/2603.18003v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 94.4 (llm_adjusted:93 = base:85 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Multimodal large language models (MLLMs) exhibit strong visual-language reasoning, yet remain confined to their native modalities and cannot directly process structured, non-visual data such as human skeletons. Existing methods either compress skeleton dynamics into lossy feature vectors for text alignment, or quantize motion into discrete tokens that generalize poorly across heterogeneous skeleton formats. We present SkeletonLLM, which achieves universal skeleton understanding by translating arbitrary skeleton sequences into the MLLM's native visual modality. At its core is DrAction, a differentiable, format-agnostic renderer that converts skeletal kinematics into compact image sequences. Because the pipeline is end-to-end differentiable, MLLM gradients can directly guide the rendering to produce task-informative visual tokens. To further enhance reasoning capabilities, we introduce a cooperative training strategy: Causal Reasoning Distillation transfers structured, step-by-step reasoning from a teacher model, while Discriminative Finetuning sharpens decision boundaries between confusable actions. SkeletonLLM demonstrates strong generalization on diverse tasks including recognition, captioning, reasoning, and cross-format transfer -- suggesting a viable path for applying MLLMs to non-native modalities. Code will be released upon acceptance.

**선정 근거**
미분가능 렌더링을 통한 골격 이해로 스포츠 동작 분석에 적합

**활용 인사이트**
다양한 형식의 골격 데이터를 시각 토큰으로 변환해 동작 인식과 분석 가능

## 4위: Unified Spatio-Temporal Token Scoring for Efficient Video VLMs

- arXiv: http://arxiv.org/abs/2603.18004v1
- PDF: https://arxiv.org/pdf/2603.18004v1
- 발행일: 2026-03-18
- 카테고리: cs.CV, cs.AI, cs.LG
- 점수: final 92.0 (llm_adjusted:90 = base:80 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Token pruning is essential for enhancing the computational efficiency of vision-language models (VLMs), particularly for video-based tasks where temporal redundancy is prevalent. Prior approaches typically prune tokens either (1) within the vision transformer (ViT) exclusively for unimodal perception tasks such as action recognition and object segmentation, without adapting to downstream vision-language tasks; or (2) only within the LLM while leaving the ViT output intact, often requiring complex text-conditioned token selection mechanisms. In this paper, we introduce Spatio-Temporal Token Scoring (STTS), a simple and lightweight module that prunes vision tokens across both the ViT and the LLM without text conditioning or token merging, and is fully compatible with end-to-end training. By learning how to score temporally via an auxiliary loss and spatially via LLM downstream gradients, aided by our efficient packing algorithm, STTS prunes 50% of vision tokens throughout the entire architecture, resulting in a 62% improvement in efficiency during both training and inference with only a 0.7% drop in average performance across 13 short and long video QA tasks. Efficiency gains increase with more sampled frames per video. Applying test-time scaling for long-video QA further yields performance gains of 0.5-1% compared to the baseline. Overall, STTS represents a novel, simple yet effective technique for unified, architecture-wide vision token pruning.

**선정 근거**
비디오 VLM의 효율적 처리 기술이 엣지 기기에서 스포츠 영상 분석에 적용 가능함

## 5위: AdapTS: Lightweight Teacher-Student Approach for Multi-Class and Continual Visual Anomaly Detection

- arXiv: http://arxiv.org/abs/2603.17530v1
- PDF: https://arxiv.org/pdf/2603.17530v1
- 발행일: 2026-03-18
- 카테고리: cs.CV, cs.AI
- 점수: final 92.0 (llm_adjusted:90 = base:85 + bonus:+5)
- 플래그: 엣지

**개요**
Visual Anomaly Detection (VAD) is crucial for industrial inspection, yet most existing methods are limited to single-category scenarios, failing to address the multi-class and continual learning demands of real-world environments. While Teacher-Student (TS) architectures are efficient, they remain unexplored for the Continual Setting. To bridge this gap, we propose AdapTS, a unified TS framework designed for multi-class and continual settings, optimized for edge deployment. AdapTS eliminates the need for two different architectures by utilizing a single shared frozen backbone and injecting lightweight trainable adapters into the student pathway. Training is enhanced via a segmentation-guided objective and synthetic Perlin noise, while a prototype-based task identification mechanism dynamically selects adapters at inference with 99\% accuracy.   Experiments on MVTec AD and VisA demonstrate that AdapTS matches the performance of existing TS methods across multi-class and continual learning scenarios, while drastically reducing memory overhead. Our lightest variant, AdapTS-S, requires only 8 MB of additional memory, 13x less than STFPM (95 MB), 48x less than RD4AD (360 MB), and 149x less than DeSTSeg (1120 MB), making it a highly scalable solution for edge deployment in complex industrial environments.

**선정 근거**
엣지 기기 최적화된 경량 비정상 탐지 기술로 스포츠 동작 분석에 적합하며, 단 8MB 메모리로 다른 방법 대비 13~149배 더 효율적

**활용 인사이트**
공유된 백본에 가벼운 어댑터를 주입하여 다양한 스포츠 종목별로 실시간으로 동작 패턴을 학습하고 분석할 수 있음

## 6위: Motion-Adaptive Temporal Attention for Lightweight Video Generation with Stable Diffusion

- arXiv: http://arxiv.org/abs/2603.17398v1
- PDF: https://arxiv.org/pdf/2603.17398v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
We present a motion-adaptive temporal attention mechanism for parameter-efficient video generation built upon frozen Stable Diffusion models. Rather than treating all video content uniformly, our method dynamically adjusts temporal attention receptive fields based on estimated motion content: high-motion sequences attend locally across frames to preserve rapidly changing details, while low-motion sequences attend globally to enforce scene consistency. We inject lightweight temporal attention modules into all UNet transformer blocks via a cascaded strategy -- global attention in down-sampling and middle blocks for semantic stabilization, motion-adaptive attention in up-sampling blocks for fine-grained refinement. Combined with temporally correlated noise initialization and motion-aware gating, the system adds only 25.8M trainable parameters (2.9\% of the base UNet) while achieving competitive results on WebVid validation when trained on 100K videos. We demonstrate that the standard denoising objective alone provides sufficient implicit temporal regularization, outperforming approaches that add explicit temporal consistency losses. Our ablation studies reveal a clear trade-off between noise correlation and motion amplitude, providing a practical inference-time control for diverse generation behaviors.

**선정 근거**
모션 적응 시간적 주의 메커니즘을 통해 스포츠 하이라이트 생성에 최적화된 가벼운 비디오 생성 기술로, 엣지 디바이스에서 높은 성능을 유지하면서도 파라미터 수가 25.8M(기본 UNet의 2.9%)에 불과해 rk3588에 적합하다.

**활용 인사이트**
rk3588 엣지 디바이스에 배포하여 스포츠 영상의 모션 레벨에 따라 동적으로 시간적 주의를 조정하며 자동으로 하이라이트 리얼을 생성하고, 움직임이 많은 순간과 적은 순간을 효과적으로 구분 처리할 수 있다.

## 7위: The Unreasonable Effectiveness of Text Embedding Interpolation for Continuous Image Steering

- arXiv: http://arxiv.org/abs/2603.17998v1
- PDF: https://arxiv.org/pdf/2603.17998v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
We present a training-free framework for continuous and controllable image editing at test time for text-conditioned generative models. In contrast to prior approaches that rely on additional training or manual user intervention, we find that a simple steering in the text-embedding space is sufficient to produce smooth edit control. Given a target concept (e.g., enhancing photorealism or changing facial expression), we use a large language model to automatically construct a small set of debiased contrastive prompt pairs, from which we compute a steering vector in the generator's text-encoder space. We then add this vector directly to the input prompt representation to control generation along the desired semantic axis. To obtain a continuous control, we propose an elastic range search procedure that automatically identifies an effective interval of steering magnitudes, avoiding both under-steering (no-edit) and over-steering (changing other attributes). Adding the scaled versions of the same vector within this interval yields smooth and continuous edits. Since our method modifies only textual representations, it naturally generalizes across text-conditioned modalities, including image and video generation. To quantify the steering continuity, we introduce a new evaluation metric that measures the uniformity of semantic change across edit strengths. We compare the continuous editing behavior across methods and find that, despite its simplicity and lightweight design, our approach is comparable to training-based alternatives, outperforming other training-free methods.

**선정 근거**
스포츠 사진 및 영상을 보정하고 개선하기 위한 훈련 없는 이미지 편집 기술 제공

**활용 인사이트**
텍스트 임베딩 공간에서의 조작을 통해 경기 장면의 품질을 향상시키고 스타일을 조절할 수 있음

## 8위: Joint Degradation-Aware Arbitrary-Scale Super-Resolution for Variable-Rate Extreme Image Compression

- arXiv: http://arxiv.org/abs/2603.17408v1
- PDF: https://arxiv.org/pdf/2603.17408v1
- 발행일: 2026-03-18
- 카테고리: cs.CV, cs.AI
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
Recent diffusion-based extreme image compression methods have demonstrated remarkable performance at ultra-low bitrates. However, most approaches require training separate diffusion models for each target bitrate, resulting in substantial computational overhead and hindering practical deployment. Meanwhile, recent studies have shown that joint super-resolution can serve as an effective approach for enhancing low-bitrate reconstruction. However, when moving toward ultra-low bitrate regimes, these methods struggle due to severe information loss, and their reliance on fixed super-resolution scales prevents flexible adaptation across diverse bitrates.   To address these limitations, we propose ASSR-EIC, a novel image compression framework that leverages arbitrary-scale super-resolution (ASSR) to support variable-rate extreme image compression (EIC). An arbitrary-scale downsampling module is introduced at the encoder side to provide controllable rate reduction, while a diffusion-based, joint degradation-aware ASSR decoder enables rate-adaptive reconstruction within a single model. We exploit the compression- and rescaling-aware diffusion prior to guide the reconstruction, yielding high fidelity and high realism restoration across diverse compression and rescaling settings. Specifically, we design a global compression-rescaling adaptor that offers holistic guidance for rate adaptation, and a local compression-rescaling modulator that dynamically balances generative and fidelity-oriented behaviors to achieve fine-grained, bitrate-adaptive detail restoration. To further enhance reconstruction quality, we introduce a dual semantic-enhanced design.   Extensive experiments demonstrate that ASSR-EIC delivers state-of-the-art performance in extreme image compression while simultaneously supporting flexible bitrate control and adaptive rate-dependent reconstruction.

**선정 근거**
엣지 디바이스에서 저사용량 영상을 고품질로 복원하는 슈퍼해상도 기술 제공

**활용 인사이트**
변동 비트율 지원으로 다양한 압축률의 스포츠 영상을 실시간으로 복원하고 품질을 향상시킬 수 있음

## 9위: On the Cone Effect and Modality Gap in Medical Vision-Language Embeddings

- arXiv: http://arxiv.org/abs/2603.17246v1
- PDF: https://arxiv.org/pdf/2603.17246v1
- 발행일: 2026-03-18
- 카테고리: cs.LG
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
Vision-Language Models (VLMs) exhibit a characteristic "cone effect" in which nonlinear encoders map embeddings into highly concentrated regions of the representation space, contributing to cross-modal separation known as the modality gap. While this phenomenon has been widely observed, its practical impact on supervised multimodal learning -particularly in medical domains- remains unclear. In this work, we introduce a lightweight post-hoc mechanism that keeps pretrained VLM encoders frozen while continuously controlling cross-modal separation through a single hyperparameter {λ}. This enables systematic analysis of how the modality gap affects downstream multimodal performance without expensive retraining. We evaluate generalist (CLIP, SigLIP) and medically specialized (BioMedCLIP, MedSigLIP) models across diverse medical and natural datasets in a supervised multimodal settings. Results consistently show that reducing excessive modality gap improves downstream performance, with medical datasets exhibiting stronger sensitivity to gap modulation; however, fully collapsing the gap is not always optimal, and intermediate, task-dependent separation yields the best results. These findings position the modality gap as a tunable property of multimodal representations rather than a quantity that should be universally minimized.

**선정 근거**
Lightweight VLM techniques for cross-modal separation could be applicable to sports content analysis on edge devices

## 10위: Revisiting Cross-Attention Mechanisms: Leveraging Beneficial Noise for Domain-Adaptive Learning

- arXiv: http://arxiv.org/abs/2603.17474v1
- PDF: https://arxiv.org/pdf/2603.17474v1
- 발행일: 2026-03-18
- 카테고리: cs.CV, cs.AI
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
Unsupervised Domain Adaptation (UDA) seeks to transfer knowledge from a labeled source domain to an unlabeled target domain but often suffers from severe domain and scale gaps that degrade performance. Existing cross-attention-based transformers can align features across domains, yet they struggle to preserve content semantics under large appearance and scale variations. To explicitly address these challenges, we introduce the concept of beneficial noise, which regularizes cross-attention by injecting controlled perturbations, encouraging the model to ignore style distractions and focus on content. We propose the Domain-Adaptive Cross-Scale Matching (DACSM) framework, which consists of a Domain-Adaptive Transformer (DAT) for disentangling domain-shared content from domain-specific style, and a Cross-Scale Matching (CSM) module that adaptively aligns features across multiple resolutions. DAT incorporates beneficial noise into cross-attention, enabling progressive domain translation with enhanced robustness, yielding content-consistent and style-invariant representations. Meanwhile, CSM ensures semantic consistency under scale changes. Extensive experiments on VisDA-2017, Office-Home, and DomainNet demonstrate that DACSM achieves state-of-the-art performance, with up to +2.3% improvement over CDTrans on VisDA-2017. Notably, DACSM achieves a +5.9% gain on the challenging "truck" class of VisDA, evidencing the strength of beneficial noise in handling scale discrepancies. These results highlight the effectiveness of combining domain translation, beneficial-noise-enhanced attention, and scale-aware alignment for robust cross-domain representation learning.

**선정 근거**
다양한 환경에서 촬영된 스포츠 영상을 정규화하고 도메인 적응 기술 제공

**활용 인사이트**
크로스 어텐션 메커니즘을 통해 다양한 조건에서도 일관된 영상 품질을 유지하며 하이라이트 감지에 활용 가능

## 11위: ChopGrad: Pixel-Wise Losses for Latent Video Diffusion via Truncated Backpropagation

- arXiv: http://arxiv.org/abs/2603.17812v1
- PDF: https://arxiv.org/pdf/2603.17812v1
- 발행일: 2026-03-18
- 카테고리: cs.CV, cs.AI, cs.LG
- 점수: final 88.0 (llm_adjusted:85 = base:80 + bonus:+5)
- 플래그: 엣지

**개요**
Recent video diffusion models achieve high-quality generation through recurrent frame processing where each frame generation depends on previous frames. However, this recurrent mechanism means that training such models in the pixel domain incurs prohibitive memory costs, as activations accumulate across the entire video sequence. This fundamental limitation also makes fine-tuning these models with pixel-wise losses computationally intractable for long or high-resolution videos. This paper introduces ChopGrad, a truncated backpropagation scheme for video decoding, limiting gradient computation to local frame windows while maintaining global consistency. We provide a theoretical analysis of this approximation and show that it enables efficient fine-tuning with frame-wise losses. ChopGrad reduces training memory from scaling linearly with the number of video frames (full backpropagation) to constant memory, and compares favorably to existing state-of-the-art video diffusion models across a suite of conditional video generation tasks with pixel-wise losses, including video super-resolution, video inpainting, video enhancement of neural-rendered scenes, and controlled driving video generation.

**선정 근거**
스포츠 영상의 메모리 효율적 처리로 실시간 하이라이트 편집 가능하며, 고해상도 영상 보정에 적합한 기술

**활용 인사이트**
ChopGrad 적용으로 영상 확산 모델의 훈련 메모리를 선형에서 상수로 줄여 장시간 스포츠 경기 처리 가능

## 12위: Prompt-Free Universal Region Proposal Network

- arXiv: http://arxiv.org/abs/2603.17554v1
- PDF: https://arxiv.org/pdf/2603.17554v1
- 코드: https://github.com/tangqh03/PF-RPN
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 88.0 (llm_adjusted:85 = base:82 + bonus:+3)
- 플래그: 코드 공개

**개요**
Identifying potential objects is critical for object recognition and analysis across various computer vision applications. Existing methods typically localize potential objects by relying on exemplar images, predefined categories, or textual descriptions. However, their reliance on image and text prompts often limits flexibility, restricting adaptability in real-world scenarios. In this paper, we introduce a novel Prompt-Free Universal Region Proposal Network (PF-RPN), which identifies potential objects without relying on external prompts. First, the Sparse Image-Aware Adapter (SIA) module performs initial localization of potential objects using a learnable query embedding dynamically updated with visual features. Next, the Cascade Self-Prompt (CSP) module identifies the remaining potential objects by leveraging the self-prompted learnable embedding, autonomously aggregating informative visual features in a cascading manner. Finally, the Centerness-Guided Query Selection (CG-QS) module facilitates the selection of high-quality query embeddings using a centerness scoring network. Our method can be optimized with limited data (e.g., 5% of MS COCO data) and applied directly to various object detection application domains for identifying potential objects without fine-tuning, such as underwater object detection, industrial defect detection, and remote sensing image object detection. Experimental results across 19 datasets validate the effectiveness of our method. Code is available at https://github.com/tangqh03/PF-RPN.

**선정 근거**
외부 프롬프트 없이 스포츠 장면 내 객체를 식별해 선수 및 경기 요소 자동 분석에 활용 가능

**활용 인사이트**
SIA, CSP, CG-QS 모듈로 스포츠 장면 내 선수, 장비, 경기 상황을 실시간으로 식별하고 분류

## 13위: A Creative Agent is Worth a 64-Token Template

- arXiv: http://arxiv.org/abs/2603.17895v1
- PDF: https://arxiv.org/pdf/2603.17895v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 88.0 (llm_adjusted:85 = base:82 + bonus:+3)
- 플래그: 코드 공개

**개요**
Text-to-image (T2I) models have substantially improved image fidelity and prompt adherence, yet their creativity remains constrained by reliance on discrete natural language prompts. When presented with fuzzy prompts such as ``a creative vinyl record-inspired skyscraper'', these models often fail to infer the underlying creative intent, leaving creative ideation and prompt design largely to human users. Recent reasoning- or agent-driven approaches iteratively augment prompts but incur high computational and monetary costs, as their instance-specific generation makes ``creativity'' costly and non-reusable, requiring repeated queries or reasoning for subsequent generations. To address this, we introduce \textbf{CAT}, a framework for \textbf{C}reative \textbf{A}gent \textbf{T}okenization that encapsulates agents' intrinsic understanding of ``creativity'' through a \textit{Creative Tokenizer}. Given the embeddings of fuzzy prompts, the tokenizer generates a reusable token template that can be directly concatenated with them to inject creative semantics into T2I models without repeated reasoning or prompt augmentation. To enable this, the tokenizer is trained via creative semantic disentanglement, leveraging relations among partially overlapping concept pairs to capture the agent's latent creative representations. Extensive experiments on \textbf{\textit{Architecture Design}}, \textbf{\textit{Furniture Design}}, and \textbf{\textit{Nature Mixture}} tasks demonstrate that CAT provides a scalable and effective paradigm for enhancing creativity in T2I generation, achieving a $3.7\times$ speedup and a $4.8\times$ reduction in computational cost, while producing images with superior human preference and text-image alignment compared to state-of-the-art T2I models and creative generation methods.

**선정 근거**
관련된 이미지 생성 및 편집 기술을 제공

## 14위: Enabling Real-Time Programmability for RAN Functions: A Wasm-Based Approach for Robust and High-Performance dApps

- arXiv: http://arxiv.org/abs/2603.17880v1
- PDF: https://arxiv.org/pdf/2603.17880v1
- 발행일: 2026-03-18
- 카테고리: cs.NI
- 점수: final 88.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
While the Open Radio Access Network Alliance (O-RAN) architecture enables third-party applications to optimize radio access networks at multiple timescales, real-time distributed applications (dApps) that demand low latency, high performance, and strong isolation remain underexplored. Existing approaches propose colocating a new RAN Intelligent Controller (RIC) at the edge, or deploying dApps in bare metal along with RAN functions. While the former approach increases network complexity and requires additional edge computing resources, the latter raises serious security concerns due to the lack of native mechanisms to isolate dApps and RAN functions. Meanwhile, WebAssembly (Wasm) has emerged as a lightweight, fast technology for robust execution of external, untrusted code. In this work, we propose a new approach to executing dApps using Wasm to isolate applications in real-time in O-RAN. Results show that our lightweight and robust approach ensures predictable, deterministic performance, strong isolation, and low latency, enabling real-time control loops.

**선정 근거**
엣지 컴퓨팅 및 실시간 처리 기술은 스포츠 영상 분석에 필수적이며 디바이스의 핵심 기능과 직접 관련됩니다.

**활용 인사이트**
Wasm을 처리 파이프라인에 구현하여 예측 가능한 성능으로 격리된 분석 기능 실행이 가능해져 실시간 스포츠 분석이 향상됩니다.

## 15위: CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image

- arXiv: http://arxiv.org/abs/2603.17779v1
- PDF: https://arxiv.org/pdf/2603.17779v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 86.4 (llm_adjusted:83 = base:78 + bonus:+5)
- 플래그: 엣지

**개요**
Single-view 3D human reconstruction has garnered significant attention in recent years. Despite numerous advancements, prior research has concentrated on reconstructing 3D models from clear, close-up images of individual subjects, often yielding subpar results in the more prevalent multi-person scenarios. Reconstructing 3D human crowd models is a highly intricate task, laden with challenges such as: 1) extensive occlusions, 2) low clarity, and 3) numerous and various appearances. To address this task, we propose CrowdGaussian, a unified framework that directly reconstructs multi-person 3D Gaussian Splatting (3DGS) representations from single-image inputs. To handle occlusions, we devise a self-supervised adaptation pipeline that enables the pretrained large human model to reconstruct complete 3D humans with plausible geometry and appearance from heavily occluded inputs. Furthermore, we introduce Self-Calibrated Learning (SCL). This training strategy enables single-step diffusion models to adaptively refine coarse renderings to optimal quality by blending identity-preserving samples with clean/corrupted image pairs. The outputs can be distilled back to enhance the quality of multi-person 3DGS representations. Extensive experiments demonstrate that CrowdGaussian generates photorealistic, geometrically coherent reconstructions of multi-person scenes.

**선정 근거**
단일 이미지에서 다인물 3D 재구성 기술로 스포츠 장면의 공간적 관계 분석에 적합

**활용 인사이트**
자체 감독 적응 파이프라인으로 가려진 선수들도 3D 모델로 재구성해 경기 전략 분석에 활용

## 16위: From Digital Twins to World Models:Opportunities, Challenges, and Applications for Mobile Edge General Intelligence

- arXiv: http://arxiv.org/abs/2603.17420v1
- PDF: https://arxiv.org/pdf/2603.17420v1
- 발행일: 2026-03-18
- 카테고리: cs.AI
- 점수: final 85.6 (llm_adjusted:82 = base:72 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
The rapid evolution toward 6G and beyond communication systems is accelerating the convergence of digital twins and world models at the network edge. Traditional digital twins provide high-fidelity representations of physical systems and support monitoring, analysis, and offline optimization. However, in highly dynamic edge environments, they face limitations in autonomy, adaptability, and scalability. This paper presents a systematic survey of the transition from digital twins to world models and discusses its role in enabling edge general intelligence (EGI). First, the paper clarifies the conceptual differences between digital twins and world models and highlights the shift from physics-based, centralized, and system-centric replicas to data-driven, decentralized, and agent-centric internal models. This discussion helps readers gain a clear understanding of how this transition enables more adaptive, autonomous, and resource-efficient intelligence at the network edge. The paper reviews the design principles, architectures, and key components of world models, including perception, latent state representation, dynamics learning, imagination-based planning, and memory. In addition, it examines the integration of world models and digital twins in wireless EGI systems and surveys emerging applications in integrated sensing and communications, semantic communication, air-ground networks, and low-altitude wireless networks. Finally, this survey provides a systematic roadmap and practical insights for designing world-model-driven edge intelligence systems in wireless and edge computing environments. It also outlines key research challenges and future directions toward scalable, reliable, and interoperable world models for edge-native agentic AI.

**선정 근거**
네트워크 엣지에서의 디지털 트윈과 월드 모델의 전환은 스포츠 촬영 및 분석을 위한 AI 엣지 디바이스에 직접 적용 가능합니다.

**활용 인사이트**
월드 모델을 활용한 인지 및 동적 학습 기술을 통해 스포츠 경기의 전략 분석과 개인별 동작 최적화가 가능해집니다.

## 17위: SegFly: A 2D-3D-2D Paradigm for Aerial RGB-Thermal Semantic Segmentation at Scale

- arXiv: http://arxiv.org/abs/2603.17920v1
- PDF: https://arxiv.org/pdf/2603.17920v1
- 코드: https://github.com/markus-42/SegFly
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 84.8 (llm_adjusted:81 = base:78 + bonus:+3)
- 플래그: 코드 공개

**개요**
Semantic segmentation for uncrewed aerial vehicles (UAVs) is fundamental for aerial scene understanding, yet existing RGB and RGB-T datasets remain limited in scale, diversity, and annotation efficiency due to the high cost of manual labeling and the difficulties of accurate RGB-T alignment on off-the-shelf UAVs. To address these challenges, we propose a scalable geometry-driven 2D-3D-2D paradigm that leverages multi-view redundancy in high-overlap aerial imagery to automatically propagate labels from a small subset of manually annotated RGB images to both RGB and thermal modalities within a unified framework. By lifting less than 3% of RGB images into a semantic 3D point cloud and reprojecting it into all views, our approach enables dense pseudo ground-truth generation across large image collections, automatically producing 97% of RGB labels and 100% of thermal labels while achieving 91% and 88% annotation accuracy without any 2D manual refinement. We further extend this 2D-3D-2D paradigm to cross-modal image registration, using 3D geometry as an intermediate alignment space to obtain fully automatic, strong pixel-level RGB-T alignment with 87% registration accuracy and no hardware-level synchronization. Applying our framework to existing geo-referenced aerial imagery, we construct SegFly, a large-scale benchmark with over 20,000 high-resolution RGB images and more than 15,000 geometrically aligned RGB-T pairs spanning diverse urban, industrial, and rural environments across multiple altitudes and seasons. On SegFly, we establish the Firefly baseline for RGB and thermal semantic segmentation and show that both conventional architectures and vision foundation models benefit substantially from SegFly supervision, highlighting the potential of geometry-driven 2D-3D-2D pipelines for scalable multi-modal scene understanding. Data and Code available at https://github.com/markus-42/SegFly.

**선정 근거**
2D-3D-2D 패러다임을 활용한 의미론적 분할 기술로 다양한 스포츠 장면의 효율적 분석 가능

**활용 인사이트**
소수의 라벨링 데이터로 대규모 스포츠 장치의 자동 분석 가능하며, 열화상 데이터와 결합해 다양한 환경에서의 스포츠 분석 강화

## 18위: TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models

- arXiv: http://arxiv.org/abs/2603.17828v1
- PDF: https://arxiv.org/pdf/2603.17828v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 84.8 (llm_adjusted:81 = base:78 + bonus:+3)
- 플래그: 코드 공개

**개요**
Although text-to-image diffusion models exhibit remarkable generative power, concept erasure techniques are essential for their safe deployment to prevent the creation of harmful content. This has fostered a dynamic interplay between the development of erasure defenses and the adversarial probes designed to bypass them, and this co-evolution has progressively enhanced the efficacy of erasure methods. However, this adversarial co-evolution has converged on a narrow, text-centric paradigm that equates erasure with severing the text-to-image mapping, ignoring that the underlying visual knowledge related to undesired concepts still persist. To substantiate this claim, we investigate from a visual perspective, leveraging DDIM inversion to probe whether a generative pathway for the erased concept can still be found. However, identifying such a visual generative pathway is challenging because standard text-guided DDIM inversion is actively resisted by text-centric defenses within the erased model. To address this, we introduce TINA, a novel Text-free INversion Attack, which enforces this visual-only probe by operating under a null-text condition, thereby avoiding existing text-centric defenses. Moreover, TINA integrates an optimization procedure to overcome the accumulating approximation errors that arise when standard inversion operates without its usual textual guidance. Our experiments demonstrate that TINA regenerates erased concepts from models treated with state-of-the-art unlearning. The success of TINA proves that current methods merely obscure concepts, highlighting an urgent need for paradigms that operate directly on internal visual knowledge.

**선정 근거**
이미지 생성 및 편집 기술과 관련

## 19위: TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos

- arXiv: http://arxiv.org/abs/2603.17735v1
- PDF: https://arxiv.org/pdf/2603.17735v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 84.0 (llm_adjusted:80 = base:75 + bonus:+5)
- 플래그: 엣지

**개요**
Automatically generating photorealistic and self-consistent appearances for untextured 3D models is a critical challenge in digital content creation. The advancement of large-scale video generation models offers a natural approach: directly synthesizing 360-degree turntable videos (TTVs), which can serve not only as high-quality dynamic previews but also as an intermediate representation to drive texture synthesis and neural rendering. However, existing general-purpose video diffusion models struggle to maintain strict geometric consistency and appearance stability across the full range of views, making their outputs ill-suited for high-quality 3D reconstruction. To this end, we introduce TAPESTRY, a framework for generating high-fidelity TTVs conditioned on explicit 3D geometry. We reframe the 3D appearance generation task as a geometry-conditioned video diffusion problem: given a 3D mesh, we first render and encode multi-modal geometric features to constrain the video generation process with pixel-level precision, thereby enabling the creation of high-quality and consistent TTVs. Building upon this, we also design a method for downstream reconstruction tasks from the TTV input, featuring a multi-stage pipeline with 3D-Aware Inpainting. By rotating the model and performing a context-aware secondary generation, this pipeline effectively completes self-occluded regions to achieve full surface coverage. The videos generated by TAPESTRY are not only high-quality dynamic previews but also serve as a reliable, 3D-aware intermediate representation that can be seamlessly back-projected into UV textures or used to supervise neural rendering methods like 3DGS. This enables the automated creation of production-ready, complete 3D assets from untextured meshes. Experimental results demonstrate that our method outperforms existing approaches in both video consistency and final reconstruction quality.

**선정 근거**
기하학 조건 기반 비디오 생성으로 스포츠 하이라이트의 일관성 있는 제작 가능

**활용 인사이트**
3D 메시 기반 비디오 생성으로 스포츠 장면의 시각적 일관성 유지하며, 자가 가림 영역 보완으로 완전한 하이라이트 영상 생성

## 20위: FineViT: Progressively Unlocking Fine-Grained Perception with Dense Recaptions

- arXiv: http://arxiv.org/abs/2603.17326v1
- PDF: https://arxiv.org/pdf/2603.17326v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 84.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
While Multimodal Large Language Models (MLLMs) have experienced rapid advancements, their visual encoders frequently remain a performance bottleneck. Conventional CLIP-based encoders struggle with dense spatial tasks due to the loss of visual details caused by low-resolution pretraining and the reliance on noisy, coarse web-crawled image-text pairs. To overcome these limitations, we introduce FineViT, a novel vision encoder specifically designed to unlock fine-grained perception. By replacing coarse web data with dense recaptions, we systematically mitigate information loss through a progressive training paradigm.: first, the encoder is trained from scratch at a high native resolution on billions of global recaptioned image-text pairs, establishing a robust, detail rich semantic foundation. Subsequently, we further enhance its local perception through LLM alignment, utilizing our curated FineCap-450M dataset that comprises over $450$ million high quality local captions. Extensive experiments validate the effectiveness of the progressive strategy. FineViT achieves state-of-the-art zero-shot recognition and retrieval performance, especially in long-context retrieval, and consistently outperforms multimodal visual encoders such as SigLIP2 and Qwen-ViT when integrated into MLLMs. We hope FineViT could serve as a powerful new baseline for fine-grained visual perception.

**선정 근거**
세밀한 시각 인식에 특화된 비전 인코더로 스포츠 동작 분석에 직접적으로 적용 가능한 최고 성능 기술

**활용 인사이트**
FineViT의 밀집 재캡션 기법을 활용해 스포츠 동작의 미세한 차이를 포착하고, 다단계 학습으로 정밀한 동작 분석 가능

## 21위: VLM2Rec: Resolving Modality Collapse in Vision-Language Model Embedders for Multimodal Sequential Recommendation

- arXiv: http://arxiv.org/abs/2603.17450v1
- PDF: https://arxiv.org/pdf/2603.17450v1
- 발행일: 2026-03-18
- 카테고리: cs.IR, cs.AI
- 점수: final 84.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
Sequential Recommendation (SR) in multimodal settings typically relies on small frozen pretrained encoders, which limits semantic capacity and prevents Collaborative Filtering (CF) signals from being fully integrated into item representations. Inspired by the recent success of Large Language Models (LLMs) as high-capacity embedders, we investigate the use of Vision-Language Models (VLMs) as CF-aware multimodal encoders for SR. However, we find that standard contrastive supervised fine-tuning (SFT), which adapts VLMs for embedding generation and injects CF signals, can amplify its inherent modality collapse. In this state, optimization is dominated by a single modality while the other degrades, ultimately undermining recommendation accuracy. To address this, we propose VLM2Rec, a VLM embedder-based framework for multimodal sequential recommendation designed to ensure balanced modality utilization. Specifically, we introduce Weak-modality Penalized Contrastive Learning to rectify gradient imbalance during optimization and Cross-Modal Relational Topology Regularization to preserve geometric consistency between modalities. Extensive experiments demonstrate that VLM2Rec consistently outperforms state-of-the-art baselines in both accuracy and robustness across diverse scenarios.

**선정 근거**
멀티모달 스포츠 콘텐츠 분석 및 추천 시스템 구축에 적합한 VLM 임베더 기술

**활용 인사이트**
경기 영상과 선수 데이터를 결합한 추천 알고리즘으로 개인별 하이라이트 자동 생성 가능

## 22위: AR-CoPO: Align Autoregressive Video Generation with Contrastive Policy Optimization

- arXiv: http://arxiv.org/abs/2603.17461v1
- PDF: https://arxiv.org/pdf/2603.17461v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 84.0 (llm_adjusted:80 = base:75 + bonus:+5)
- 플래그: 실시간

**개요**
Streaming autoregressive (AR) video generators combined with few-step distillation achieve low-latency, high-quality synthesis, yet remain difficult to align via reinforcement learning from human feedback (RLHF). Existing SDE-based GRPO methods face challenges in this setting: few-step ODEs and consistency model samplers deviate from standard flow-matching ODEs, and their short, low-stochasticity trajectories are highly sensitive to initialization noise, rendering intermediate SDE exploration ineffective. We propose AR-CoPO (AutoRegressive Contrastive Policy Optimization), a framework that adapts the Neighbor GRPO contrastive perspective to streaming AR generation. AR-CoPO introduces chunk-level alignment via a forking mechanism that constructs neighborhood candidates at a randomly selected chunk, assigns sequence-level rewards, and performs localized GRPO updates. We further propose a semi-on-policy training strategy that complements on-policy exploration with exploitation over a replay buffer of reference rollouts, improving generation quality across domains. Experiments on Self-Forcing demonstrate that AR-CoPO improves both out-of-domain generalization and in-domain human preference alignment over the baseline, providing evidence of genuine alignment rather than reward hacking.

**선정 근거**
저지연 고품질 스포츠 하이라이트 영상 생성에 적용 가능한 스트리밍 AR 생성 기술

**활용 인사이트**
실시간 경기 영상에서 자동으로 주요 장면을 추출하고 고품질 하이라이트 영상 생성

## 23위: Feeling the Space: Egomotion-Aware Video Representation for Efficient and Accurate 3D Scene Understanding

- arXiv: http://arxiv.org/abs/2603.17980v1
- PDF: https://arxiv.org/pdf/2603.17980v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:78 + bonus:+0)

**개요**
Recent Multimodal Large Language Models (MLLMs) have shown high potential for spatial reasoning within 3D scenes. However, they typically rely on computationally expensive 3D representations like point clouds or reconstructed Bird's-Eye View (BEV) maps, or lack physical grounding to resolve ambiguities in scale and size. This paper significantly enhances MLLMs with egomotion modality data, captured by Inertial Measurement Units (IMUs) concurrently with the video. In particular, we propose a novel framework, called Motion-MLLM, introducing two key components: (1) a cascaded motion-visual keyframe filtering module that leverages both IMU data and visual features to efficiently select a sparse yet representative set of keyframes, and (2) an asymmetric cross-modal fusion module where motion tokens serve as intermediaries that channel egomotion cues and cross-frame visual context into the visual representation. By grounding visual content in physical egomotion trajectories, Motion-MLLM can reason about absolute scale and spatial relationships across the scene. Our extensive evaluation shows that Motion-MLLM makes significant improvements in various tasks related to 3D scene understanding and spatial reasoning. Compared to state-of-the-art (SOTA) methods based on video frames and explicit 3D data, Motion-MLLM exhibits similar or even higher accuracy with significantly less overhead (i.e., 1.40$\times$ and 1.63$\times$ higher cost-effectiveness, respectively).

**선정 근거**
IMU 데이터를 활용한 3D 스포츠 장면 이해 기술로 공간적 관계 분석 가능

**활용 인사이트**
카메라 움직임 데이터와 영상을 결합해 선수 위치 및 경기 전략 정확히 분석

## 24위: ReLaGS: Relational Language Gaussian Splatting

- arXiv: http://arxiv.org/abs/2603.17605v1
- PDF: https://arxiv.org/pdf/2603.17605v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Achieving unified 3D perception and reasoning across tasks such as segmentation, retrieval, and relation understanding remains challenging, as existing methods are either object-centric or rely on costly training for inter-object reasoning. We present a novel framework that constructs a hierarchical language-distilled Gaussian scene and its 3D semantic scene graph without scene-specific training. A Gaussian pruning mechanism refines scene geometry, while a robust multi-view language alignment strategy aggregates noisy 2D features into accurate 3D object embeddings. On top of this hierarchy, we build an open-vocabulary 3D scene graph with Vision Language derived annotations and Graph Neural Network-based relational reasoning. Our approach enables efficient and scalable open-vocabulary 3D reasoning by jointly modeling hierarchical semantics and inter/intra-object relationships, validated across tasks including open-vocabulary segmentation, scene graph generation, and relation-guided retrieval. Project page: https://dfki-av.github.io/ReLaGS/

**선정 근거**
스포츠 분석에 잠재적으로 유용한 3D 장면 이해 기술로, 선수 추적 및 전략 분석에 필수적입니다.

**활용 인사이트**
계층적 언어-증류 가우시안 장면을 구성하여 스포츠 장면의 3D 표현을 생성하고 선수 상호작용을 분석할 수 있습니다.

## 25위: VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection

- arXiv: http://arxiv.org/abs/2603.17470v1
- PDF: https://arxiv.org/pdf/2603.17470v1
- 발행일: 2026-03-18
- 카테고리: cs.CV, cs.AI
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Monocular 3D object detection typically relies on pseudo-labeling techniques to reduce dependency on real-world annotations. Recent advances demonstrate that deterministic linguistic cues can serve as effective auxiliary weak supervision signals, providing complementary semantic context. However, hand-crafted textual descriptions struggle to capture the inherent visual diversity of individuals across scenes, limiting the model's ability to learn scene-aware representations. To address this challenge, we propose Visual-referred Probabilistic Prompt Learning (VirPro), an adaptive multi-modal pretraining paradigm that can be seamlessly integrated into diverse weakly supervised monocular 3D detection frameworks. Specifically, we generate a diverse set of learnable, instance-conditioned prompts across scenes and store them in an Adaptive Prompt Bank (APB). Subsequently, we introduce Multi-Gaussian Prompt Modeling (MGPM), which incorporates scene-based visual features into the corresponding textual embeddings, allowing the text prompts to express visual uncertainties. Then, from the fused vision-language embeddings, we decode a prompt-targeted Gaussian, from which we derive a unified object-level prompt embedding for each instance. RoI-level contrastive matching is employed to enforce modality alignment, bringing embeddings of co-occurring objects within the same scene closer in the latent space, thus enhancing semantic coherence. Extensive experiments on the KITTI benchmark demonstrate that integrating our pretraining paradigm consistently yields substantial performance gains, achieving up to a 4.8% average precision improvement than the baseline.

**선정 근거**
Provides applicable 3D detection technology for sports scene analysis

## 26위: Steering Video Diffusion Transformers with Massive Activations

- arXiv: http://arxiv.org/abs/2603.17825v1
- PDF: https://arxiv.org/pdf/2603.17825v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Despite rapid progress in video diffusion transformers, how their internal model signals can be leveraged with minimal overhead to enhance video generation quality remains underexplored. In this work, we study the role of Massive Activations (MAs), which are rare, high-magnitude hidden state spikes in video diffusion transformers. We observed that MAs emerge consistently across all visual tokens, with a clear magnitude hierarchy: first-frame tokens exhibit the largest MA magnitudes, latent-frame boundary tokens (the head and tail portions of each temporal chunk in the latent space) show elevated but slightly lower MA magnitudes than the first frame, and interior tokens within each latent frame remain elevated, yet are comparatively moderate in magnitude. This structured pattern suggests that the model implicitly prioritizes token positions aligned with the temporal chunking in the latent space. Based on this observation, we propose Structured Activation Steering (STAS), a training-free self-guidance-like method that steers MA values at first-frame and boundary tokens toward a scaled global maximum reference magnitude. STAS achieves consistent improvements in terms of video quality and temporal coherence across different text-to-video models, while introducing negligible computational overhead.

**선정 근거**
비디오 확산 트랜스포머의 내부 신호를 활용해 스포츠 하이라이트 영상 품질을 향상시키는 기술

**활용 인사이트**
STAS 방식으로 첫 프레임과 경계 토큰의 활성값을 조정해 영상 품질과 시간적 일관성 개선

## 27위: EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding

- arXiv: http://arxiv.org/abs/2603.18001v1
- PDF: https://arxiv.org/pdf/2603.18001v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
In this work, we present EchoGen, a unified framework for layout-to-image generation and image grounding, capable of generating images with accurate layouts and high fidelity to text descriptions (e.g., spatial relationships), while grounding the image robustly at the same time. We believe that image grounding possesses strong text and layout understanding abilities, which can compensate for the corresponding limitations in layout-to-image generation. At the same time, images generated from layouts exhibit high diversity in content, thereby enhancing the robustness of image grounding. Jointly training both tasks within a unified model can promote performance improvements for each. However, we identify that this joint training paradigm encounters several optimization challenges and results in restricted performance. To address these issues, we propose progressive training strategies. First, the Parallel Multi-Task Pre-training (PMTP) stage equips the model with basic abilities for both tasks, leveraging shared tokens to accelerate training. Next, the Dual Joint Optimization (DJO) stage exploits task duality to sequentially integrate the two tasks, enabling unified optimization. Finally, the Cycle RL stage eliminates reliance on visual supervision by using consistency constraints as rewards, significantly enhancing the model's unified capabilities via the GRPO strategy. Extensive experiments demonstrate state-of-the-art results on both layout-to-image generation and image grounding benchmarks, and reveal clear synergistic gains from optimizing the two tasks together.

**선정 근거**
이미지 생성 및 이해에 대한 통합 프레임워크로 스포츠 콘텐츠 생성에 적용 가능

**활용 인사이트**
스포츠 장면의 레이아웃과 설명을 기반으로 정확한 이미지를 생성하여 하이라이트 영상 제작에 활용할 수 있습니다

## 28위: Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment

- arXiv: http://arxiv.org/abs/2603.17655v1
- PDF: https://arxiv.org/pdf/2603.17655v1
- 발행일: 2026-03-18
- 카테고리: cs.CV, cs.AI
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Cross-Domain Few-Shot Learning (CDFSL) adapts models trained with large-scale general data (source domain) to downstream target domains with only scarce training data, where the research on vision-language models (e.g., CLIP) is still in the early stages. Typical downstream domains, such as medical diagnosis, require fine-grained visual cues for interpretable recognition, but we find that current fine-tuned CLIP models can hardly focus on these cues, albeit they can roughly focus on important regions in source domains. Although current works have demonstrated CLIP's shortcomings in capturing local subtle patterns, in this paper, we find that the domain gap and scarce training data further exacerbate such shortcomings, much more than that of holistic patterns, which we call the local misalignment problem in CLIP-based CDFSL. To address this problem, due to the lack of supervision in aligning local visual features and text semantics, we turn to self-supervision information. Inspired by the translation task, we propose the CC-CDFSL method with cycle consistency, which translates local visual features into text features and then translates them back into visual features (and vice versa), and constrains the original features close to the translated back features. To reduce the noise imported by richer information in the visual modality, we further propose a Semantic Anchor mechanism, which first augments visual features to provide a larger corpus for the text-to-image mapping, and then shrinks the image features to filter out irrelevant image-to-text mapping. Extensive experiments on various benchmarks, backbones, and fine-tuning methods show we can (1) effectively improve the local vision-language alignment, (2) enhance the interpretability of learned patterns and model decisions by visualizing patches, and (3) achieve state-of-the-art performance.

**선정 근거**
크로스 도메인 적응 기술로 다양한 스포츠 분야에서 적응적인 분석 모델 구현 가능

**활용 인사이트**
사이클 일관성 학습을 통해 시각-언어 정렬을 개선해 스포츠 전략 분석 정확도 향상

## 29위: Face anonymization preserving facial expressions and photometric realism

- arXiv: http://arxiv.org/abs/2603.17567v1
- PDF: https://arxiv.org/pdf/2603.17567v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 76.0 (llm_adjusted:70 = base:65 + bonus:+5)
- 플래그: 엣지

**개요**
The widespread sharing of face images on social media platforms and in large-scale datasets raises pressing privacy concerns, as biometric identifiers can be exploited without consent. Face anonymization seeks to generate realistic facial images that irreversibly conceal the subject's identity while preserving their usefulness for downstream tasks. However, most existing generative approaches focus on identity removal and image realism, often neglecting facial expressions as well as photometric consistency -- specifically attributes such as illumination and skin tone -- that are critical for applications like relighting, color constancy, and medical or affective analysis. In this work, we propose a feature-preserving anonymization framework that extends DeepPrivacy by incorporating dense facial landmarks to better retain expressions, and by introducing lightweight post-processing modules that ensure consistency in lighting direction and skin color. We further establish evaluation metrics specifically designed to quantify expression fidelity, lighting consistency, and color preservation, complementing standard measures of image realism, pose accuracy, and re-identification resistance. Experiments on the CelebA-HQ dataset demonstrate that our method produces anonymized faces with improved realism and significantly higher fidelity in expression, illumination, and skin tone compared to state-of-the-art baselines. These results underscore the importance of feature-aware anonymization as a step toward more useful, fair, and trustworthy privacy-preserving facial data.

**선정 근거**
이미지 처리 기술로 표정과 사실성을 보존하면서 익명화하는 방법은 스포츠 영상 품질 향상에 적용 가능

## 30위: FACE-net: Factual Calibration and Emotion Augmentation for Retrieval-enhanced Emotional Video Captioning

- arXiv: http://arxiv.org/abs/2603.17455v1
- PDF: https://arxiv.org/pdf/2603.17455v1
- 발행일: 2026-03-18
- 카테고리: cs.CV
- 점수: final 76.0 (llm_adjusted:70 = base:70 + bonus:+0)

**개요**
Emotional Video Captioning (EVC) is an emerging task, which aims to describe factual content with the intrinsic emotions expressed in videos. Existing works perceive global emotional cues and then combine with video content to generate descriptions. However, insufficient factual and emotional cues mining and coordination during generation make their methods difficult to deal with the factual-emotional bias, which refers to the factual and emotional requirements being different in different samples on generation. To this end, we propose a retrieval-enhanced framework with FActual Calibration and Emotion augmentation (FACE-net), which through a unified architecture collaboratively mines factual-emotional semantics and provides adaptive and accurate guidance for generation, breaking through the compromising tendency of factual-emotional descriptions in all sample learning. Technically, we firstly introduces an external repository and retrieves the most relevant sentences with the video content to augment the semantic information. Subsequently, our factual calibration via uncertainty estimation module splits the retrieved information into subject-predicate-object triplets, and self-refines and cross-refines different components through video content to effectively mine the factual semantics; while our progressive visual emotion augmentation module leverages the calibrated factual semantics as experts, interacts with the video content and emotion dictionary to generate visual queries and candidate emotions, and then aggregates them to adaptively augment emotions to each factual semantics. Moreover, to alleviate the factual-emotional bias, we design a dynamic bias adjustment routing module to predict and adjust the degree of bias of a sample.

**선정 근거**
비디오 캡셔닝과 감정 분석 기술은 스포츠 콘텐츠 분석에 적용 가능

**활용 인사이트**
스포츠 영상의 사실적 내용과 표현된 감정을 동시에 분석하여 선수들의 감정 상태와 경기 전략을 파악하는 데 활용할 수 있습니다

## 31위: PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation

- arXiv: http://arxiv.org/abs/2603.17520v1 | 2026-03-18 | final 76.0

Recent advances in vision-language models (VLMs) have garnered substantial attention in open-vocabulary semantic and part segmentation (OSPS). However, existing methods extract image-text alignment cues from cost volumes through a serial structure of spatial and class aggregations, leading to knowledge interference between class-level semantics and spatial context.

-> 시맨틱 분할과 비전-언어 모델은 스포츠 장면 분석에 적용 가능

## 32위: Multi-stage Flow Scheduling for LLM Serving

- arXiv: http://arxiv.org/abs/2603.17456v1 | 2026-03-18 | final 76.0

Meeting stringent Time-To-First-Token (TTFT) requirements is crucial for LLM applications. To improve efficiency, modern LLM serving systems adopt disaggregated architectures with diverse parallelisms, introducing complex multi-stage workflows involving reusable KV-block retrieval, collective communication, and P2D transfer.

-> Efficient scheduling techniques could improve real-time video processing on edge devices

## 33위: MedSAD-CLIP: Supervised CLIP with Token-Patch Cross-Attention for Medical Anomaly Detection and Segmentation

- arXiv: http://arxiv.org/abs/2603.17325v1 | 2026-03-18 | final 74.4

Medical anomaly detection (MAD) and segmentation play a critical role in assisting clinical diagnosis by identifying abnormal regions in medical images and localizing pathological regions. Recent CLIP-based studies are promising for anomaly detection in zero-/few-shot settings, and typically rely on global representations and weak supervision, often producing coarse localization and limited segmentation quality.

-> Anomaly detection techniques could be adapted for sports movement analysis

## 34위: Uncertainty Quantification and Risk Control for Multi-Speaker Sound Source Localization

- arXiv: http://arxiv.org/abs/2603.17377v1 | 2026-03-18 | final 74.4

Reliable Sound Source Localization (SSL) plays an essential role in many downstream tasks, where informed decision making depends not only on accurate localization but also on the confidence in each estimate. This need for reliability becomes even more pronounced in challenging conditions, such as reverberant environments and multi-source scenarios.

-> Audio source localization techniques could be applied to sports video analysis on edge devices

## 35위: Multi-Source Human-in-the-Loop Digital Twin Testbed for Connected and Autonomous Vehicles in Mixed Traffic Flow

- arXiv: http://arxiv.org/abs/2603.17751v1 | 2026-03-18 | final 70.4

In the emerging mixed traffic environments, Connected and Autonomous Vehicles (CAVs) have to interact with surrounding human-driven vehicles (HDVs). This paper introduces MSH-MCCT (Multi-Source Human-in-the-Loop Mixed Cloud Control Testbed), a novel CAV testbed that captures complex interactions between various CAVs and HDVs.

-> 관련성 낮음 - 차량 상호작용에 초점, 스포츠 촬영/분석과 직접적 연결 부족

## 36위: MCoT-MVS: Multi-level Vision Selection by Multi-modal Chain-of-Thought Reasoning for Composed Image Retrieval

- arXiv: http://arxiv.org/abs/2603.17360v1 | 2026-03-18 | final 70.4

Composed Image Retrieval (CIR) aims to retrieve target images based on a reference image and modified texts. However, existing methods often struggle to extract the correct semantic cues from the reference image that best reflect the user's intent under textual modification prompts, resulting in interference from irrelevant visual noise.

-> Multi-modal reasoning approach could be adapted for sports content analysis

## 37위: Real-Time Online Learning for Model Predictive Control using a Spatio-Temporal Gaussian Process Approximation

- arXiv: http://arxiv.org/abs/2603.17632v1 | 2026-03-18 | final 68.0

Learning-based model predictive control (MPC) can enhance control performance by correcting for model inaccuracies, enabling more precise state trajectory predictions than traditional MPC. A common approach is to model unknown residual dynamics as a Gaussian process (GP), which leverages data and also provides an estimate of the associated uncertainty.

-> 자동 주행 레이싱을 위한 제어 시스템 관련 논문으로, 스포츠 영상 촬영 및 분석과는 간접적으로만 관련이 있다.

## 38위: UniSAFE: A Comprehensive Benchmark for Safety Evaluation of Unified Multimodal Models

- arXiv: http://arxiv.org/abs/2603.17476v1 | 2026-03-18 | final 66.4

Unified Multimodal Models (UMMs) offer powerful cross-modality capabilities but introduce new safety risks not observed in single-task models. Despite their emergence, existing safety benchmarks remain fragmented across tasks and modalities, limiting the comprehensive evaluation of complex system-level vulnerabilities.

-> 다중 모달 모델의 안전 평가로 인해 콘텐츠 공유 플랫폼과 간접적으로 관련됨

## 39위: UniSem: Generalizable Semantic 3D Reconstruction from Sparse Unposed Images

- arXiv: http://arxiv.org/abs/2603.17519v1 | 2026-03-18 | final 66.4

Semantic-aware 3D reconstruction from sparse, unposed images remains challenging for feed-forward 3D Gaussian Splatting (3DGS). Existing methods often predict an over-complete set of Gaussian primitives under sparse-view supervision, leading to unstable geometry and inferior depth quality.

-> 3D reconstruction techniques could be used for sports scene representation

## 40위: ECHO: Towards Emotionally Appropriate and Contextually Aware Interactive Head Generation

- arXiv: http://arxiv.org/abs/2603.17427v1 | 2026-03-18 | final 66.4

In natural face-to-face interaction, participants seamlessly alternate between speaking and listening, producing facial behaviors (FBs) that are finely informed by long-range context and naturally exhibit contextual appropriateness and emotional rationality. Interactive Head Generation (IHG) aims to synthesize lifelike avatar head video emulating such capabilities.

-> Facial behavior analysis techniques could be indirectly applicable to athlete expression analysis

## 41위: Governed Memory: A Production Architecture for Multi-Agent Workflows

- arXiv: http://arxiv.org/abs/2603.17787v1 | 2026-03-18 | final 66.4

Enterprise AI deploys dozens of autonomous agent nodes across workflows, each acting on the same entities with no shared memory and no common governance. We identify five structural challenges arising from this memory governance gap: memory silos across agent workflows; governance fragmentation across teams and tools; unstructured memories unusable by downstream systems; redundant context delivery in autonomous multi-step executions; and silent quality degradation without feedback loops.

-> Multi-agent workflow concepts could be applicable to coordinating different AI components in sports device

## 42위: WINFlowNets: Warm-up Integrated Networks Training of Generative Flow Networks for Robotics and Machine Fault Adaptation

- arXiv: http://arxiv.org/abs/2603.17301v1 | 2026-03-18 | final 64.0

Generative Flow Networks for continuous scenarios (CFlowNets) have shown promise in solving sequential decision-making tasks by learning stochastic policies using a flow and a retrieval network. Despite their demonstrated efficiency compared to state-of-the-art Reinforcement Learning (RL) algorithms, their practical application in robotic control tasks is constrained by the reliance on pre-training the retrieval network.

-> 로봇 적응 개념은 스포츠 시나리오 분석에 간접적으로 적용 가능

## 43위: Structured SIR: Efficient and Expressive Importance-Weighted Inference for High-Dimensional Image Registration

- arXiv: http://arxiv.org/abs/2603.17415v1 | 2026-03-18 | final 64.0

Image registration is an ill-posed dense vision task, where multiple solutions achieve similar loss values, motivating probabilistic inference. Variational inference has previously been employed to capture these distributions, however restrictive assumptions about the posterior form can lead to poor characterisation, overconfidence and low-quality samples.

-> 의료 영상 등록에 특화되어 스포츠 촬영/분석과 간접적 관련성만 존재

## 44위: Toward Phonology-Guided Sign Language Motion Generation: A Diffusion Baseline and Conditioning Analysis

- arXiv: http://arxiv.org/abs/2603.17388v1 | 2026-03-18 | final 64.0

Generating natural, correct, and visually smooth 3D avatar sign language motion conditioned on the text inputs continues to be very challenging. In this work, we train a generative model of 3D body motion and explore the role of phonological attribute conditioning for sign language motion generation, using ASL-LEX 2.0 annotations such as hand shape, hand location and movement.

-> Motion generation techniques could be indirectly applicable to sports movement analysis

## 45위: The Program Hypergraph: Multi-Way Relational Structure for Geometric Algebra, Spatial Compute, and Physics-Aware Compilation

- arXiv: http://arxiv.org/abs/2603.17627v1 | 2026-03-18 | final 64.0

The Program Semantic Graph (PSG) introduced in prior work on Dimensional Type Systems and Deterministic Memory Management encodes compilation-relevant properties as binary edge relations between computation nodes. This representation is adequate for scalar and tensor computations, but becomes structurally insufficient for two classes of problems central to heterogeneous compute: tile co-location and routing constraints in spatial dataflow architectures, which are inherently multi-way; and geometric algebra computation, where graded multi-way products cannot be faithfully represented as sequences of binary operations without loss of algebraic identity.

-> Geometric algebra concepts could be indirectly applicable to motion analysis in sports

## 46위: Causal Representation Learning on High-Dimensional Data: Benchmarks, Reproducibility, and Evaluation Metrics

- arXiv: http://arxiv.org/abs/2603.17405v1 | 2026-03-18 | final 64.0

Causal representation learning (CRL) models aim to transform high-dimensional data into a latent space, enabling interventions to generate counterfactual samples or modify existing data based on the causal relationships among latent variables. To facilitate the development and evaluation of these models, a variety of synthetic and real-world datasets have been proposed, each with distinct advantages and limitations.

-> Causal representation learning could be applicable to analyzing sports movements and strategies

## 47위: Proof-of-Authorship for Diffusion-based AI Generated Content

- arXiv: http://arxiv.org/abs/2603.17513v1 | 2026-03-18 | final 64.0

Recent advancements in AI-generated content (AIGC) have introduced new challenges in intellectual property protection and the authentication of generated objects. We focus on scenarios in which an author seeks to assert authorship of an object generated using latent diffusion models (LDMs), in the presence of adversaries who attempt to falsely claim authorship of objects they did not create.

-> Proof-of-authorship concepts could be applicable to protecting sports content created by device

## 48위: Harnessing the Power of Foundation Models for Accurate Material Classification

- arXiv: http://arxiv.org/abs/2603.17390v1 | 2026-03-18 | final 62.4

Material classification has emerged as a critical task in computer vision and graphics, supporting the assignment of accurate material properties to a wide range of digital and real-world applications. While traditionally framed as an image classification task, this domain faces significant challenges due to the scarcity of annotated data, limiting the accuracy and generalizability of trained models.

-> 이미지 분류 기술은 스포츠 촬영에 적용 가능하나 직접적인 연관성은 낮음

## 49위: Unified Policy Value Decomposition for Rapid Adaptation

- arXiv: http://arxiv.org/abs/2603.17947v1 | 2026-03-18 | final 61.6

Rapid adaptation in complex control systems remains a central challenge in reinforcement learning. We introduce a framework in which policy and value functions share a low-dimensional coefficient vector - a goal embedding - that captures task identity and enables immediate adaptation to novel tasks without retraining representations.

-> Reinforcement learning adaptation techniques could be indirectly applicable to sports strategy analysis

## 50위: Temporal Narrative Monitoring in Dynamic Information Environments

- arXiv: http://arxiv.org/abs/2603.17617v1 | 2026-03-18 | final 60.0

Comprehending the information environment (IE) during crisis events is challenging due to the rapid change and abstract nature of the domain. Many approaches focus on snapshots via classification methods or network approaches to describe the IE in crisis, ignoring the temporal nature of how information changed over time.

-> Temporal analysis techniques could potentially be adapted for analyzing sports game progressions and identifying key moments.

## 51위: Translation Invariance of Neural Operators for the FitzHugh-Nagumo Model

- arXiv: http://arxiv.org/abs/2603.17523v1 | 2026-03-18 | final 60.0

Neural Operators (NOs) are a powerful deep learning framework designed to learn the solution operator that arise from partial differential equations. This study investigates NOs ability to capture the stiff spatio-temporal dynamics of the FitzHugh-Nagumo model, which describes excitable cells.

-> Translation invariance concepts could be relevant for tracking players moving across a sports field and neural operator architectures might apply to sports video analysis.

## 52위: ARES: Scalable and Practical Gradient Inversion Attack in Federated Learning through Activation Recovery

- arXiv: http://arxiv.org/abs/2603.17623v1 | 2026-03-18 | final 60.0

Federated Learning (FL) enables collaborative model training by sharing model updates instead of raw data, aiming to protect user privacy. However, recent studies reveal that these shared updates can inadvertently leak sensitive training data through gradient inversion attacks (GIAs).

-> Activation recovery techniques might be relevant for feature extraction from sports videos and sparse recovery could potentially detect key features in sports footage.

## 53위: Conditional Inverse Learning of Time-Varying Reproduction Numbers Inference

- arXiv: http://arxiv.org/abs/2603.17549v1 | 2026-03-18 | final 60.0

Estimating time-varying reproduction numbers from epidemic incidence data is a central task in infectious disease surveillance, yet it poses an inherently ill-posed inverse problem. Existing approaches often rely on strong structural assumptions derived from epidemiological models, which can limit their ability to adapt to non-stationary transmission dynamics induced by interventions or behavioral changes, leading to delayed detection of regime shifts and degraded estimation accuracy.

-> Time-series analysis techniques could potentially be adapted for analyzing sports game progressions and conditional mapping might identify patterns in sports gameplay.

## 54위: AirDDE: Multifactor Neural Delay Differential Equations for Air Quality Forecasting

- arXiv: http://arxiv.org/abs/2603.17529v1 | 2026-03-18 | final 58.4

Accurate air quality forecasting is essential for public health and environmental sustainability, but remains challenging due to the complex pollutant dynamics. Existing deep learning methods often model pollutant dynamics as an instantaneous process, overlooking the intrinsic delays in pollutant propagation.

-> 신경망 측면에서만 간접적으로 관련

## 55위: rSDNet: Unified Robust Neural Learning against Label Noise and Adversarial Attacks

- arXiv: http://arxiv.org/abs/2603.17628v1 | 2026-03-18 | final 58.4

Neural networks are central to modern artificial intelligence, yet their training remains highly sensitive to data contamination. Standard neural classifiers are trained by minimizing the categorical cross-entropy loss, corresponding to maximum likelihood estimation under a multinomial model.

-> 신경망 측면에서만 약간 관련

---

## 다시 보기

### MONET: Modeling and Optimization of neural NEtwork Training from Edge to Data Centers (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15002v1
- 점수: final 100.0

While hardware-software co-design has significantly improved the efficiency of neural network inference, modeling the training phase remains a critical yet underexplored challenge. Training workloads impose distinct constraints, particularly regarding memory footprint and backpropagation complexity, which existing inference-focused tools fail to capture. This paper introduces MONET, a framework designed to model the training of neural networks on heterogeneous dataflow accelerators. MONET builds upon Stream, an experimentally verified framework that that models the inference of neural networks on heterogeneous dataflow accelerators with layer fusion. Using MONET, we explore the design space of ResNet-18 and a small GPT-2, demonstrating the framework's capability to model training workflows and find better hardware architectures. We then further examine problems that become more complex in neural network training due to the larger design space, such as determining the best layer-fusion configuration. Additionally, we use our framework to find interesting trade-offs in activation checkpointing, with the help of a genetic algorithm. Our findings highlight the importance of a holistic approach to hardware-software co-design for scalable and efficient deep learning deployment.

-> 에지 디바이스를 위한 신경망 최적화 기술이 rk3588 기반 AI 촬영 장비의 핵심 기술로 직접 적용 가능

### Multi-Objective Load Balancing for Heterogeneous Edge-Based Object Detection Systems (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15400v1
- 점수: final 98.4

The rapid proliferation of the Internet of Things (IoT) and smart applications has led to a surge in data generated by distributed sensing devices. Edge computing is a mainstream approach to managing this data by pushing computation closer to the data source, typically onto resource-constrained devices such as single-board computers (SBCs). In such environments, the unavoidable heterogeneity of hardware and software makes effective load balancing particularly challenging. In this paper, we propose a multi-objective load balancing method tailored to heterogeneous, edge-based object detection systems. We study a setting in which multiple device-model pairs expose distinct accuracy, latency, and energy profiles, while both request intensity and scene complexity fluctuate over time. To handle this dynamically varying environment, our approach uses a two-stage decision mechanism: it first performs accuracy-aware filtering to identify suitable device-model candidates that provide accuracy within the acceptable range, and then applies a weighted-sum scoring function over expected latency and energy consumption to select the final execution target. We evaluate the proposed load balancer through extensive experiments on real-world datasets, comparing against widely used baseline strategies. The results indicate that the proposed multi-objective load balancing method halves energy consumption and achieves an 80% reduction in end-to-end latency, while incurring only a modest, up to 10%, decrease in detection accuracy relative to an accuracy-centric baseline.

-> 이기종 엣지 기반 객체 탐지 시스템의 다목적 부하 분산이 스포츠 촬영 시스템 자원 제약에 직접 적용 가능

### Advancing Visual Reliability: Color-Accurate Underwater Image Enhancement for Real-Time Underwater Missions (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16363v1
- 점수: final 96.0

Underwater image enhancement plays a crucial role in providing reliable visual information for underwater platforms, since strong absorption and scattering in water-related environments generally lead to image quality degradation. Existing high-performance methods often rely on complex architectures, which hinder deployment on underwater devices. Lightweight methods often sacrifice quality for speed and struggle to handle severely degraded underwater images. To address this limitation, we present a real-time underwater image enhancement framework with accurate color restoration. First, an Adaptive Weighted Channel Compensation module is introduced to achieve dynamic color recovery of the red and blue channels using the green channel as a reference anchor. Second, we design a Multi-branch Re-parameterized Dilated Convolution that employs multi-branch fusion during training and structural re-parameterization during inference, enabling large receptive field representation with low computational overhead. Finally, a Statistical Global Color Adjustment module is employed to optimize overall color performance based on statistical priors. Extensive experiments on eight datasets demonstrate that the proposed method achieves state-of-the-art performance across seven evaluation metrics. The model contains only 3,880 inference parameters and achieves an inference speed of 409 FPS. Our method improves the UCIQE score by 29.7% under diverse environmental conditions, and the deployment on ROV platforms and performance gains in downstream tasks further validate its superiority for real-time underwater missions.

-> 실시간 이미지 보정 및 향상 기술은 스포츠 영상 처리에 직접적으로 적용 가능하며 경량화 모델 제공

### Lightweight User-Personalization Method for Closed Split Computing (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.14958v1
- 점수: final 96.0

Split Computing enables collaborative inference between edge devices and the cloud by partitioning a deep neural network into an edge-side head and a server-side tail, reducing latency and limiting exposure of raw input data. However, inference performance often degrades in practical deployments due to user-specific data distribution shifts, unreliable communication, and privacy-oriented perturbations, especially in closed environments where model architectures and parameters are inaccessible. To address this challenge, we propose SALT (Split-Adaptive Lightweight Tuning), a lightweight adaptation framework for closed Split Computing systems. SALT introduces a compact client-side adapter that refines intermediate representations produced by a frozen head network, enabling effective model adaptation without modifying the head or tail networks or increasing communication overhead. By modifying only the training conditions, SALT supports multiple adaptation objectives, including user personalization, communication robustness, and privacy-aware inference. Experiments using ResNet-18 on CIFAR-10 and CIFAR-100 show that SALT achieves higher accuracy than conventional retraining and fine-tuning while significantly reducing training cost. On CIFAR-10, SALT improves personalized accuracy from 88.1% to 93.8% while reducing training latency by more than 60%. SALT also maintains over 90% accuracy under 75% packet loss and preserves high accuracy (about 88% at sigma = 1.0) under noise injection. These results demonstrate that SALT provides an efficient and practical adaptation framework for real-world Split Computing systems.

-> 경량 분산 컴퓨팅 프레임워크가 rk3588 엣지 디바이스에 최적화

### Low-light Image Enhancement with Retinex Decomposition in Latent Space (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15131v1
- 점수: final 96.0

Retinex theory provides a principled foundation for low-light image enhancement, inspiring numerous learning-based methods that integrate its principles. However, existing methods exhibits limitations in accurately decomposing reflectance and illumination components. To address this, we propose a Retinex-Guided Transformer~(RGT) model, which is a two-stage model consisting of decomposition and enhancement phases. First, we propose a latent space decomposition strategy to separate reflectance and illumination components. By incorporating the log transformation and 1-pixel offset, we convert the intrinsically multiplicative relationship into an additive formulation, enhancing decomposition stability and precision. Subsequently, we construct a U-shaped component refiner incorporating the proposed guidance fusion transformer block. The component refiner refines reflectance component to preserve texture details and optimize illumination distribution, effectively transforming low-light inputs to normal-light counterparts. Experimental evaluations across four benchmark datasets validate that our method achieves competitive performance in low-light enhancement and a more stable training process.

-> 잔광 이미지 보정 기술이 다양한 조건에서 스포츠 영상 품질 향상에 적합

### Face-to-Face: A Video Dataset for Multi-Person Interaction Modeling (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.14794v1
- 점수: final 96.0

Modeling the reactive tempo of human conversation remains difficult because most audio-visual datasets portray isolated speakers delivering short monologues. We introduce \textbf{Face-to-Face with Jimmy Fallon (F2F-JF)}, a 70-hour, 14k-clip dataset of two-person talk-show exchanges that preserves the sequential dependency between a guest turn and the host's response. A semi-automatic pipeline combines multi-person tracking, speech diarization, and lightweight human verification to extract temporally aligned host/guest tracks with tight crops and metadata that are ready for downstream modeling. We showcase the dataset with a reactive, speech-driven digital avatar task in which the host video during $[t_1,t_2]$ is generated from their audio plus the guest's preceding video during $[t_0,t_1]$. Conditioning a MultiTalk-style diffusion model on this cross-person visual context yields small but consistent Emotion-FID and FVD gains while preserving lip-sync quality relative to an audio-only baseline. The dataset, preprocessing recipe, and baseline together provide an end-to-end blueprint for studying dyadic, sequential behavior, which we expand upon throughout the paper. Dataset and code will be made publicly available.

-> 다중 인물 추적 기술이 스포츠 촬영에서 여러 선수 동시 추적에 적용 가능

### Generative Video Compression with One-Dimensional Latent Representation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15302v1
- 점수: final 94.4

Recent advancements in generative video codec (GVC) typically encode video into a 2D latent grid and employ high-capacity generative decoders for reconstruction. However, this paradigm still leaves two key challenges in fully exploiting spatial-temporal redundancy: Spatially, the 2D latent grid inevitably preserves intra-frame redundancy due to its rigid structure, where adjacent patches remain highly similar, thereby necessitating a higher bitrate. Temporally, the 2D latent grid is less effective for modeling long-term correlations in a compact and semantically coherent manner, as it hinders the aggregation of common contents across frames. To address these limitations, we introduce Generative Video Compression with One-Dimensional (1D) Latent Representation (GVC1D). GVC1D encodes the video data into extreme compact 1D latent tokens conditioned on both short- and long-term contexts. Without the rigid 2D spatial correspondence, these 1D latent tokens can adaptively attend to semantic regions and naturally facilitate token reduction, thereby reducing spatial redundancy. Furthermore, the proposed 1D memory provides semantically rich long-term context while maintaining low computational cost, thereby further reducing temporal redundancy. Experimental results indicate that GVC1D attains superior compression efficiency, where it achieves bitrate reductions of 60.4\% under LPIPS and 68.8\% under DISTS on the HEVC Class B dataset, surpassing the previous video compression methods.Project: https://gvc1d.github.io/

-> 이 논문은 스포츠 촬영 엣지 디바이스를 위한 카메라 보정 방법을 제안한다. 핵심은 레이저 트래커와 카메라 결합 기술이다.

### Affordable Precision Agriculture: A Deployment-Oriented Review of Low-Cost, Low-Power Edge AI and TinyML for Resource-Constrained Farming Systems (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15085v1
- 점수: final 93.6

Precision agriculture increasingly integrates artificial intelligence to enhance crop monitoring, irrigation management, and resource efficiency. Nevertheless, the vast majority of the current systems are still mostly cloud-based and require reliable connectivity, which hampers the adoption to smaller scale, smallholder farming and underdeveloped country systems. Using recent literature reviews, ranging from 2023 to 2026, this review covers deployments of Edge AI, focused on the evolution and acceptance of Tiny Machine Learning, in low-cost and low-powered agriculture. A hardware-targeted deployment-oriented study has shown pronounced variation in architecture with microcontroller-class platforms i.e. ESP32, STM32, ATMega dominating the inference options, in parallel with single-board computers and UAV-assisted solutions. Quantitative synthesis shows quantization is the dominant optimization strategy; the approach in many works identified: around 50% of such works are quantized, while structured pruning, multi-objective compression and hardware aware neural architecture search are relatively under-researched. Also, resource profiling practices are not uniform: while model size is occasionally reported, explicit flash, RAM, MAC, latency and millijoule level energy metrics are not well documented, hampering reproducibility and cross-system comparison. Moreoever, to bridge the gap between research prototypes and deployment-ready systems, the review also presents a literature-informed deployment perspective in the form of a privacy-preserving layered Edge AI architecture for agriculture, synthesizing the key system-level design insights emerging from the surveyed works. Overall, the findings demonstrate a clear architectural shift toward localized inference with centralized training asymmetry.

-> 이 논문은 다양한 입력 시나리오 처리를 위한 OOD 검출 방법을 제안한다. 핵심은 이중 프로토타이프 추적 기술이다.

### Emotion-Aware Classroom Quality Assessment Leveraging IoT-Based Real-Time Student Monitoring (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16719v1
- 점수: final 93.6

This study presents high-throughput, real-time multi-agent affective computing framework designed to enhance classroom learning through emotional state monitoring. As large classroom sizes and limited teacher student interaction increasingly challenge educators, there is a growing need for scalable, data-driven tools capable of capturing students' emotional and engagement patterns in real time. The system was evaluated using the Classroom Emotion Dataset, consisting of 1,500 labeled images and 300 classroom detection videos. Tailored for IoT devices, the system addresses load balancing and latency challenges through efficient real-time processing. Field testing was conducted across three educational institutions in a large metropolitan area: a primary school (hereafter school A), a secondary school (school B), and a high school (school C). The system demonstrated robust performance, detecting up to 50 faces at 25 FPS and achieving 88% overall accuracy in classifying classroom engagement states. Implementation results showed positive outcomes, with favorable feedback from students, teachers, and parents regarding improved classroom interaction and teaching adaptation. Key contributions of this research include establishing a practical, IoT-based framework for emotion-aware learning environments and introducing the 'Classroom Emotion Dataset' to facilitate further validation and research.

-> 실시간 비디오 분석 및 IoT 장치 배포 기술은 스포츠 촬영 장치에 적용 가능하나 교실 환경에 특화됨

### Efficient Reasoning on the Edge (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16867v1
- 점수: final 93.6

Large language models (LLMs) with chain-of-thought reasoning achieve state-of-the-art performance across complex problem-solving tasks, but their verbose reasoning traces and large context requirements make them impractical for edge deployment. These challenges include high token generation costs, large KV-cache footprints, and inefficiencies when distilling reasoning capabilities into smaller models for mobile devices. Existing approaches often rely on distilling reasoning traces from larger models into smaller models, which are verbose and stylistically redundant, undesirable for on-device inference. In this work, we propose a lightweight approach to enable reasoning in small LLMs using LoRA adapters combined with supervised fine-tuning. We further introduce budget forcing via reinforcement learning on these adapters, significantly reducing response length with minimal accuracy loss. To address memory-bound decoding, we exploit parallel test-time scaling, improving accuracy at minor latency increase. Finally, we present a dynamic adapter-switching mechanism that activates reasoning only when needed and a KV-cache sharing strategy during prompt encoding, reducing time-to-first-token for on-device inference. Experiments on Qwen2.5-7B demonstrate that our method achieves efficient, accurate reasoning under strict resource constraints, making LLM reasoning practical for mobile scenarios. Videos demonstrating our solution running on mobile devices are available on our project page.

-> Edge computing techniques for efficient AI reasoning applicable to sports analysis on rk3588 device

### SpikeCLR: Contrastive Self-Supervised Learning for Few-Shot Event-Based Vision using Spiking Neural Networks (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16338v1
- 점수: final 93.6

Event-based vision sensors provide significant advantages for high-speed perception, including microsecond temporal resolution, high dynamic range, and low power consumption. When combined with Spiking Neural Networks (SNNs), they can be deployed on neuromorphic hardware, enabling energy-efficient applications on embedded systems. However, this potential is severely limited by the scarcity of large-scale labeled datasets required to effectively train such models. In this work, we introduce SpikeCLR, a contrastive self-supervised learning framework that enables SNNs to learn robust visual representations from unlabeled event data. We adapt prior frame-based methods to the spiking domain using surrogate gradient training and introduce a suite of event-specific augmentations that leverage spatial, temporal, and polarity transformations. Through extensive experiments on CIFAR10-DVS, N-Caltech101, N-MNIST, and DVS-Gesture benchmarks, we demonstrate that self-supervised pretraining with subsequent fine-tuning outperforms supervised learning in low-data regimes, achieving consistent gains in few-shot and semi-supervised settings. Our ablation studies reveal that combining spatial and temporal augmentations is critical for learning effective spatio-temporal invariances in event data. We further show that learned representations transfer across datasets, contributing to efforts for powerful event-based models in label-scarce settings.

-> Event-based vision with SNNs provides efficient processing for edge sports camera devices

### Deep Reinforcement Learning-driven Edge Offloading for Latency-constrained XR pipelines (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16823v1
- 점수: final 93.6

Immersive extended reality (XR) applications introduce latency-critical workloads that must satisfy stringent real-time responsiveness while operating on energy- and battery-constrained devices, making execution placement between end devices and nearby edge servers a fundamental systems challenge. Existing approaches to adaptive execution and computation offloading typically optimize average performance metrics and do not fully capture the sustained interaction between real-time latency requirements and device battery lifetime in closed-loop XR workloads. In this paper, we present a battery-aware execution management framework for edge-assisted XR systems that jointly considers execution placement, workload quality, latency requirements, and battery dynamics. We design an online decision mechanism based on a lightweight deep reinforcement learning policy that continuously adapts execution decisions under dynamic network conditions while maintaining high motion-to-photon latency compliance. Experimental results show that the proposed approach extends the projected device battery lifetime by up to 163% compared to latency-optimal local execution while maintaining over 90% motion-to-photon latency compliance under stable network conditions. Such compliance does not fall below 80% even under significantly limited network bandwidth availability, thereby demonstrating the effectiveness of explicitly managing latency-energy trade-offs in immersive XR systems.

-> 엣지 디바이스에서 실시간으로 작동하는 강화 학습 프레임워크로 프로젝트의 rk3588 기반 AI 촬영 장치 및 실시간 처리 요구사항과 직접적으로 관련되어 있습니다. 저지연-배터리 수명 트레이드오프 관리는 모바일 스포츠 촬영 장치에 필수적입니다.

### Nova: Scalable Streaming Join Placement and Parallelization in Resource-Constrained Geo-Distributed Environments (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15453v1
- 점수: final 93.6

Real-time data processing in large geo-distributed applications, like the Internet of Things (IoT), increasingly shifts computation from the cloud to the network edge to reduce latency and mitigate network congestion. In this setting, minimizing latency while avoiding node overload requires jointly optimizing operator replication and placement of operator instances, a challenge known as the Operator Placement and Replication (OPR) problem. OPR is NP-hard and particularly difficult to solve in large-scale, heterogeneous, and dynamic geo-distributed networks, where solutions must be scalable, resource-aware, and adaptive to changes like node failures. Existing work on OPR has primarily focused on single-stream operators, such as filters and aggregations. However, many latency-sensitive applications, like environmental monitoring and anomaly detection, require efficient regional stream joins near data sources.   This paper introduces Nova, an optimization approach designed to address OPR for join operators that are computable on resource-constrained edge devices. Nova relaxes the NP-hard OPR into a convex optimization problem by embedding cost metrics into a Euclidean space and partitioning joins into smaller sub-joins. This new formulation enables linear scalability and efficient adaptation to topological changes through partial re-optimizations. We evaluate Nova through simulations on real-world topologies and on a local testbed, demonstrating up to 39x latency reduction and 4.5x increase in throughput compared to existing edge-centered solutions, while also preventing node overload and maintaining near-constant re-optimization times regardless of topology size.

-> Edit2Interp은 이미지 모델을 적은 데이터로 비디오 처리에 적용하여 스포츠 영상 향상에 효과적입니다. 이는 에지 디바이스에서의 실시간 영상 처리에 적합합니다.

### Tracking the Discriminative Axis: Dual Prototypes for Test-Time OOD Detection Under Covariate Shift (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15213v1
- 점수: final 93.6

For reliable deployment of deep-learning systems, out-of-distribution (OOD) detection is indispensable. In the real world, where test-time inputs often arrive as streaming mixtures of in-distribution (ID) and OOD samples under evolving covariate shifts, OOD samples are domain-constrained and bounded by the environment, and both ID and OOD are jointly affected by the same covariate factors. Existing methods typically assume a stationary ID distribution, but this assumption breaks down in such settings, leading to severe performance degradation. We empirically discover that, even under covariate shift, covariate-shifted ID (csID) and OOD (csOOD) samples remain separable along a discriminative axis in feature space. Building on this observation, we propose DART, a test-time, online OOD detection method that dynamically tracks dual prototypes -- one for ID and the other for OOD -- to recover the drifting discriminative axis, augmented with multi-layer fusion and flip correction for robustness. Extensive experiments on a wide range of challenging benchmarks, where all datasets are subjected to 15 common corruption types at severity level 5, demonstrate that our method significantly improves performance, yielding 15.32 percentage points (pp) AUROC gain and 49.15 pp FPR@95TPR reduction on ImageNet-C vs. Textures-C compared to established baselines. These results highlight the potential of the test-time discriminative axis tracking for dependable OOD detection in dynamically changing environments.

-> OOD detection technology applicable for sports filming edge device handling various input scenarios

### Federated Learning of Binary Neural Networks: Enabling Low-Cost Inference (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15507v1
- 점수: final 93.6

Federated Learning (FL) preserves privacy by distributing training across devices. However, using DNNs is computationally intensive at the low-powered edge during inference. Edge deployment demands models that simultaneously optimize memory footprint and computational efficiency, a dilemma where conventional DNNs fail by exceeding resource limits. Traditional post-training binarization reduces model size but suffers from severe accuracy loss due to quantization errors. To address these challenges, we propose FedBNN, a rotation-aware binary neural network framework that learns binary representations directly during local training. By encoding each weight as a single bit $\{+1, -1\}$ instead of a $32$-bit float, FedBNN shrinks the model footprint, significantly reducing runtime (during inference) FLOPs and memory requirements in comparison to federated methods using real models. Evaluations across multiple benchmark datasets demonstrate that FedBNN significantly reduces resource consumption while performing similarly to existing federated methods using real-valued models.

-> 이진 신경망 기술이 엣지 디바이스에서의 효율적인 AI 처리에 직접 적용 가능

### A Novel Camera-to-Robot Calibration Method for Vision-Based Floor Measurements (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15126v1
- 점수: final 93.6

A novel hand-eye calibration method for ground-observing mobile robots is proposed. While cameras on mobile robots are com- mon, they are rarely used for ground-observing measurement tasks. Laser trackers are increasingly used in robotics for precise localization. A referencing plate is designed to combine the two measurement modalities of laser-tracker 3D metrology and camera- based 2D imaging. It incorporates reflector nests for pose acquisition using a laser tracker and a camera calibration target that is observed by the robot-mounted camera. The procedure comprises estimating the plate pose, the plate-camera pose, and the robot pose, followed by computing the robot-camera transformation. Experiments indicate sub-millimeter repeatability.

-> Camera calibration techniques applicable to sports filming edge device

### M2IR: Proactive All-in-One Image Restoration via Mamba-style Modulation and Mixture-of-Experts (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.14816v1
- 점수: final 90.4

While Transformer-based architectures have dominated recent advances in all-in-one image restoration, they remain fundamentally reactive: propagating degradations rather than proactively suppressing them. In the absence of explicit suppression mechanisms, degraded signals interfere with feature learning, compelling the decoder to balance artifact removal and detail preservation, thereby increasing model complexity and limiting adaptability. To address these challenges, we propose M2IR, a novel restoration framework that proactively regulates degradation propagation during the encoding stage and efficiently eliminates residual degradations during decoding. Specifically, the Mamba-Style Transformer (MST) block performs pixel-wise selective state modulation to mitigate degradations while preserving structural integrity. In parallel, the Adaptive Degradation Expert Collaboration (ADEC) module utilizes degradation-specific experts guided by a DA-CLIP-driven router and complemented by a shared expert to eliminate residual degradations through targeted and cooperative restoration. By integrating the MST block and ADEC module, M2IR transitions from passive reaction to active degradation control, effectively harnessing learned representations to achieve superior generalization, enhanced adaptability, and refined recovery of fine-grained details across diverse all-in-one image restoration benchmarks. Our source codes are available at https://github.com/Im34v/M2IR.

-> 이미지 편집 모델을 영상 처리로 확장하여 적은 데이터로(64-256 샘플) 스포츠 영상 향상이 가능합니다.

### Knowledge Distillation for Collaborative Learning in Distributed Communications and Sensing (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16116v1
- 점수: final 90.4

The rise of sixth generation (6G) wireless networks promises to deliver ultra-reliable, low-latency, and energy-efficient communications, sensing, and computing. However, traditional centralized artificial intelligence (AI) paradigms are ill-suited to the decentralized, resource-constrained, and dynamic nature of 6G ecosystems. This paper explores knowledge distillation (KD) and collaborative learning as promising techniques that enable the efficient and scalable deployment of lightweight AI models across distributed communications and sensing (C&S) nodes. We begin by providing an overview of KD and highlight the key strengths that make it particularly effective in distributed scenarios characterized by device heterogeneity, task diversity, and constrained resources. We then examine its role in fostering collective intelligence through collaborative learning between the central and distributed nodes via various knowledge distilling and deployment strategies. Finally, we present a systematic numerical study demonstrating that KD-empowered collaborative learning can effectively support lightweight AI models for multi-modal sensing-assisted beam tracking applications with substantial performance gains and complexity reduction.

-> 지식 증류 기술은 경량화 AI 모델을 효율적으로 배포해 rk3588 기반 장치의 성능을 최적화합니다.

### DST-Net: A Dual-Stream Transformer with Illumination-Independent Feature Guidance and Multi-Scale Spatial Convolution for Low-Light Image Enhancement (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16482v1
- 점수: final 90.4

Low-light image enhancement aims to restore the visibility of images captured by visual sensors in dim environments by addressing their inherent signal degradations, such as luminance attenuation and structural corruption. Although numerous algorithms attempt to improve image quality, existing methods often cause a severe loss of intrinsic signal priors. To overcome these challenges, we propose a Dual-Stream Transformer Network (DST-Net) based on illumination-agnostic signal prior guidance and multi-scale spatial convolutions. First, to address the loss of critical signal features under low-light conditions, we design a feature extraction module. This module integrates Difference of Gaussians (DoG), LAB color space transformations, and VGG-16 for texture extraction, utilizing decoupled illumination-agnostic features as signal priors to continuously guide the enhancement process. Second, we construct a dual-stream interaction architecture. By employing a cross-modal attention mechanism, the network leverages the extracted priors to dynamically rectify the deteriorated signal representation of the enhanced image, ultimately achieving iterative enhancement through differentiable curve estimation. Furthermore, to overcome the inability of existing methods to preserve fine structures and textures, we propose a Multi-Scale Spatial Fusion Block (MSFB) featuring pseudo-3D and 3D gradient operator convolutions. This module integrates explicit gradient operators to recover high-frequency edges while capturing inter-channel spatial correlations via multi-scale spatial convolutions. Extensive evaluations and ablation studies demonstrate that DST-Net achieves superior performance in subjective visual quality and objective metrics. Specifically, our method achieves a PSNR of 25.64 dB on the LOL dataset. Subsequent validation on the LSRW dataset further confirms its robust cross-scene generalization.

-> 저조도 환경에서의 이미지 향상 기술은 다양한 조건에서 스포츠 영상의 품질을 보장해 콘텐츠 가치를 높입니다.

### Collaborative Temporal Feature Generation via Critic-Free Reinforcement Learning for Cross-User Sensor-Based Activity Recognition (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16043v1
- 점수: final 90.4

Human Activity Recognition using wearable inertial sensors is foundational to healthcare monitoring, fitness analytics, and context-aware computing, yet its deployment is hindered by cross-user variability arising from heterogeneous physiological traits, motor habits, and sensor placements. Existing domain generalization approaches either neglect temporal dependencies in sensor streams or depend on impractical target-domain annotations. We propose a different paradigm: modeling generalizable feature extraction as a collaborative sequential generation process governed by reinforcement learning. Our framework, CTFG (Collaborative Temporal Feature Generation), employs a Transformer-based autoregressive generator that incrementally constructs feature token sequences, each conditioned on prior context and the encoded sensor input. The generator is optimized via Group-Relative Policy Optimization, a critic-free algorithm that evaluates each generated sequence against a cohort of alternatives sampled from the same input, deriving advantages through intra-group normalization rather than learned value estimation. This design eliminates the distribution-dependent bias inherent in critic-based methods and provides self-calibrating optimization signals that remain stable across heterogeneous user distributions. A tri-objective reward comprising class discrimination, cross-user invariance, and temporal fidelity jointly shapes the feature space to separate activities, align user distributions, and preserve fine-grained temporal content. Evaluations on the DSADS and PAMAP2 benchmarks demonstrate state-of-the-art cross-user accuracy (88.53\% and 75.22\%), substantial reduction in inter-task training variance, accelerated convergence, and robust generalization under varying action-space dimensionalities.

-> 크로스-유저 활동 인식 기술은 다양한 사용자의 스포츠 동작을 정확히 분석해 개인별 맞춤형 훈련을 가능하게 합니다.

### SE(3)-LIO: Smooth IMU Propagation With Jointly Distributed Poses on SE(3) Manifold for Accurate and Robust LiDAR-Inertial Odometry (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16118v1
- 점수: final 90.4

In estimating odometry accurately, an inertial measurement unit (IMU) is widely used owing to its high-rate measurements, which can be utilized to obtain motion information through IMU propagation. In this paper, we address the limitations of existing IMU propagation methods in terms of motion prediction and motion compensation. In motion prediction, the existing methods typically represent a 6-DoF pose by separating rotation and translation and propagate them on their respective manifold, so that the rotational variation is not effectively incorporated into translation propagation. During motion compensation, the relative transformation between predicted poses is used to compensate motion-induced distortion in other measurements, while inherent errors in the predicted poses introduce uncertainty in the relative transformation. To tackle these challenges, we represent and propagate the pose on SE(3) manifold, where propagated translation properly accounts for rotational variation. Furthermore, we precisely characterize the relative transformation uncertainty by considering the correlation between predicted poses, and incorporate this uncertainty into the measurement noise during motion compensation. To this end, we propose a LiDAR-inertial odometry (LIO), referred to as SE(3)-LIO, that integrates the proposed IMU propagation and uncertainty-aware motion compensation (UAMC). We validate the effectiveness of SE(3)-LIO on diverse datasets. Our source code and additional material are available at: https://se3-lio.github.io/.

-> SE(3) 다양체를 이용한 정밀한 움직임 추정은 스포츠 동작 분석의 정확도를 높여 전략 분석에 기여합니다.

### PA-LVIO: Real-Time LiDAR-Visual-Inertial Odometry and Mapping with Pose-Only Bundle Adjustment (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16228v1
- 점수: final 90.4

Real-time LiDAR-visual-inertial odometry and mapping is crucial for navigation and planning tasks in intelligent transportation systems. This study presents a pose-only bundle adjustment (PA) LiDAR-visual-inertial odometry (LVIO), named PA-LVIO, to meet the urgent need for real-time navigation and mapping. The proposed PA framework for LiDAR and visual measurements is highly accurate and efficient, and it can derive reliable frame-to-frame constraints within multiple frames. A marginalization-free and frame-to-map (F2M) LiDAR measurement model is integrated into the state estimator to eliminate odometry drifts. Meanwhile, an IMU-centric online spatial-temporal calibration is employed to obtain a pixel-wise LiDAR-camera alignment. With accurate estimated odometry and extrinsics, a high-quality and RGB-rendered point-cloud map can be built. Comprehensive experiments are conducted on both public and private datasets collected by wheeled robot, unmanned aerial vehicle (UAV), and handheld devices with 28 sequences and more than 50 km trajectories. Sufficient results demonstrate that the proposed PA-LVIO yields superior or comparable performance to state-of-the-art LVIO methods, in terms of the odometry accuracy and mapping quality. Besides, PA-LVIO can run in real-time on both the desktop PC and the onboard ARM computer.

-> 실시간 시각-관성 항법으로 스포츠 장면의 정확한 움직임 추적이 가능해 자동 촬영 및 하이라이트 편집의 기반이 됩니다.

### Efficient Event Camera Volume System (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.14738v1
- 점수: final 90.4

Event cameras promise low latency and high dynamic range, yet their sparse output challenges integration into standard robotic pipelines. We introduce \nameframew (Efficient Event Camera Volume System), a novel framework that models event streams as continuous-time Dirac impulse trains, enabling artifact-free compression through direct transform evaluation at event timestamps. Our key innovation combines density-driven adaptive selection among DCT, DTFT, and DWT transforms with transform-specific coefficient pruning strategies tailored to each domain's sparsity characteristics. The framework eliminates temporal binning artifacts while automatically adapting compression strategies based on real-time event density analysis. On EHPT-XC and MVSEC datasets, our framework achieves superior reconstruction fidelity with DTFT delivering the lowest earth mover distance. In downstream segmentation tasks, EECVS demonstrates robust generalization. Notably, our approach demonstrates exceptional cross-dataset generalization: when evaluated with EventSAM segmentation, EECVS achieves mean IoU 0.87 on MVSEC versus 0.44 for voxel grids at 24 channels, while remaining competitive on EHPT-XC. Our ROS2 implementation provides real-time deployment with DCT processing achieving 1.5 ms latency and 2.7X higher throughput than alternative transforms, establishing the first adaptive event compression framework that maintains both computational efficiency and superior generalization across diverse robotic scenarios.

-> 자원 제약된 edge 환경에서 실시간 데이터 처리를 최적화하여 지연 시간을 최대 39배까지 줄일 수 있습니다.

### Riemannian Motion Generation: A Unified Framework for Human Motion Representation and Generation via Riemannian Flow Matching (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15016v1
- 점수: final 89.6

Human motion generation is often learned in Euclidean spaces, although valid motions follow structured non-Euclidean geometry. We present Riemannian Motion Generation (RMG), a unified framework that represents motion on a product manifold and learns dynamics via Riemannian flow matching. RMG factorizes motion into several manifold factors, yielding a scale-free representation with intrinsic normalization, and uses geodesic interpolation, tangent-space supervision, and manifold-preserving ODE integration for training and sampling. On HumanML3D, RMG achieves state-of-the-art FID in the HumanML3D format (0.043) and ranks first on all reported metrics under the MotionStreamer format. On MotionMillion, it also surpasses strong baselines (FID 5.6, R@1 0.86). Ablations show that the compact $\mathscr{T}+\mathscr{R}$ (translation + rotations) representation is the most stable and effective, highlighting geometry-aware modeling as a practical and scalable route to high-fidelity motion generation.

-> Human motion generation framework applicable for sports movement analysis

### Edit2Interp: Adapting Image Foundation Models from Spatial Editing to Video Frame Interpolation with Few-Shot Learning (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15003v1
- 점수: final 89.6

Pre-trained image editing models exhibit strong spatial reasoning and object-aware transformation capabilities acquired from billions of image-text pairs, yet they possess no explicit temporal modeling. This paper demonstrates that these spatial priors can be repurposed to unlock temporal synthesis capabilities through minimal adaptation - without introducing any video-specific architecture or motion estimation modules. We show that a large image editing model (Qwen-Image-Edit), originally designed solely for static instruction-based edits, can be adapted for Video Frame Interpolation (VFI) using only 64-256 training samples via Low-Rank Adaptation (LoRA). Our core contribution is revealing that the model's inherent understanding of "how objects transform" in static scenes contains latent temporal reasoning that can be activated through few-shot fine-tuning. While the baseline model completely fails at producing coherent intermediate frames, our parameter-efficient adaptation successfully unlocks its interpolation capability. Rather than competing with task-specific VFI methods trained from scratch on massive datasets, our work establishes that foundation image editing models possess untapped potential for temporal tasks, offering a data-efficient pathway for video synthesis in resource-constrained scenarios. This bridges the gap between image manipulation and video understanding, suggesting that spatial and temporal reasoning may be more intertwined in foundation models than previously recognized

-> Adapting image models for video processing with few-shot learning applicable to sports video enhancement

### Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.14726v1
- 점수: final 89.6

Accurately recovering hand poses within the body context remains a major challenge in 3D whole-body pose estimation. This difficulty arises from a fundamental supervision gap: whole-body pose estimators are trained on full-body datasets with limited hand diversity, while hand-only estimators, trained on hand-centric datasets, excel at detailed finger articulation but lack global body awareness. To address this, we propose Hand4Whole++, a modular framework that leverages the strengths of both pre-trained whole-body and hand pose estimators. We introduce CHAM (Conditional Hands Modulator), a lightweight module that modulates the whole-body feature stream using hand-specific features extracted from a pre-trained hand pose estimator. This modulation enables the whole-body model to predict wrist orientations that are both accurate and coherent with the upper-body kinematic structure, without retraining the full-body model. In parallel, we directly incorporate finger articulations and hand shapes predicted by the hand pose estimator, aligning them to the full-body mesh via differentiable rigid alignment. This design allows Hand4Whole++ to combine globally consistent body reasoning with fine-grained hand detail. Extensive experiments demonstrate that Hand4Whole++ substantially improves hand accuracy and enhances overall full-body pose quality.

-> 3D 전신 자세 추정 프레임워크로 스포츠 동작 분석에 적합하며, 손 동작 정밀도 향상으로 다양한 스포츠 기술 분석 가능

### CyCLeGen: Cycle-Consistent Layout Prediction and Image Generation in Vision Foundation Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.14957v1
- 점수: final 89.6

We present CyCLeGen, a unified vision-language foundation model capable of both image understanding and image generation within a single autoregressive framework. Unlike existing vision models that depend on separate modules for perception and synthesis, CyCLeGen adopts a fully integrated architecture that enforces cycle-consistent learning through image->layout->image and layout->image->layout generation loops. This unified formulation introduces two key advantages: introspection, enabling the model to reason about its own generations, and data efficiency, allowing self-improvement via synthetic supervision under a reinforcement learning objective guided by cycle consistency. Extensive experiments show that CyCLeGen achieves significant gains across diverse image understanding and generation benchmarks, highlighting the potential of unified vision-language foundation models.

-> 이미지 생성 및 이해 기술을 통한 스포츠 영상의 사진처럼 보정 기구 구현 가능하며, 통합 아키텍처로 효율적인 처리

### BLADE: Adaptive Wi-Fi Contention Control for Next-Generation Real-Time Communication (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16119v1
- 점수: final 88.0

Next-generation real-time communication (NGRTC) applications, such as cloud gaming and XR, demand consistently ultra-low latency. However, through our first large-scale measurement, we find that despite the deployment of edge servers, dedicated congestion control, and loss recovery mechanisms, cloud gaming users still experience long-tail latency in Wi-Fi networks. We further identify that Wi-Fi last-mile access points (APs) serve as the primary latency bottleneck. Specifically, short-term packet delivery droughts, caused by fundamental limitations in Wi-Fi contention control standards, are the root cause. To address this issue, we propose BLADE, an adaptive contention control algorithm that dynamically adjusts the contention windows (CW) of all Wi-Fi transmitters based on the channel contention level in a fully distributed manner. Our NS3 simulations and real-world evaluations with commercial Wi-Fi APs demonstrate that, compared to standard contention control, BLADE reduces Wi-Fi packet transmission tail latency by over 5X under heavy channel contention and significantly stabilizes MAC throughput while ensuring fast and fair convergence. Consequently, BLADE reduces the video stall rate in cloud gaming by over 90%.

-> Wi-Fi 경쟁 제어 알고리즘은 스포츠 콘텐츠 플랫폼의 실시간 공유 기능에 직접적으로 적용 가능하며, 지연 시간을 크게 줄여 사용자 경험을 향상시킵니다.

### Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16538v1
- 점수: final 88.0

3D Gaussian Splatting (3DGS) has recently emerged as a powerful scene representation and is increasingly used for visual localization and pose refinement. However, despite its high-quality differentiable rendering, the robustness of 3DGS-based pose refinement remains highly sensitive to both the initial camera pose and the reconstructed geometry. In this work, we take a closer look at these limitations and identify two major sources of uncertainty: (i) pose prior uncertainty, which often arises from regression or retrieval models that output a single deterministic estimate, and (ii) geometric uncertainty, caused by imperfections in the 3DGS reconstruction that propagate errors into PnP solvers. Such uncertainties can distort reprojection geometry and destabilize optimization, even when the rendered appearance still looks plausible. To address these uncertainties, we introduce a relocalization framework that combines Monte Carlo pose sampling with Fisher Information-based PnP optimization. Our method explicitly accounts for both pose and geometric uncertainty and requires no retraining or additional supervision. Across diverse indoor and outdoor benchmarks, our approach consistently improves localization accuracy and significantly increases stability under pose and depth noise.

-> 4D 포인트 클라우드 비디오 이해 기술은 스포츠 장면의 동적 분석에 직접적으로 적용 가능하며, 다양한 프레임 레이트 처리와 움직임 패턴 인식에 유리합니다.

### MessyKitchens: Contact-rich object-level 3D scene reconstruction (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16868v1
- 점수: final 88.0

Monocular 3D scene reconstruction has recently seen significant progress. Powered by the modern neural architectures and large-scale data, recent methods achieve high performance in depth estimation from a single image. Meanwhile, reconstructing and decomposing common scenes into individual 3D objects remains a hard challenge due to the large variety of objects, frequent occlusions and complex object relations. Notably, beyond shape and pose estimation of individual objects, applications in robotics and animation require physically-plausible scene reconstruction where objects obey physical principles of non-penetration and realistic contacts. In this work we advance object-level scene reconstruction along two directions. First, we introduceMessyKitchens, a new dataset with real-world scenes featuring cluttered environments and providing high-fidelity object-level ground truth in terms of 3D object shapes, poses and accurate object contacts. Second, we build on the recent SAM 3D approach for single-object reconstruction and extend it with Multi-Object Decoder (MOD) for joint object-level scene reconstruction. To validate our contributions, we demonstrate MessyKitchens to significantly improve previous datasets in registration accuracy and inter-object penetration. We also compare our multi-object reconstruction approach on three datasets and demonstrate consistent and significant improvements of MOD over the state of the art. Our new benchmark, code and pre-trained models will become publicly available on our project website: https://messykitchens.github.io/.

-> 3D scene reconstruction technology applicable to sports analysis but not specifically designed for it

### PAKAN: Pixel Adaptive Kolmogorov-Arnold Network Modules for Pansharpening (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15109v1
- 점수: final 86.4

Pansharpening aims to fuse high-resolution spatial details from panchromatic images with the rich spectral information of multispectral images. Existing deep neural networks for this task typically rely on static activation functions, which limit their ability to dynamically model the complex, non-linear mappings required for optimal spatial-spectral fusion. While the recently introduced Kolmogorov-Arnold Network (KAN) utilizes learnable activation functions, traditional KANs lack dynamic adaptability during inference. To address this limitation, we propose a Pixel Adaptive Kolmogorov-Arnold Network framework. Starting from KAN, we design two adaptive variants: a 2D Adaptive KAN that generates spline summation weights across spatial dimensions and a 1D Adaptive KAN that generates them across spectral channels. These two components are then assembled into PAKAN 2to1 for feature fusion and PAKAN 1to1 for feature refinement. Extensive experiments demonstrate that our proposed modules significantly enhance network performance, proving the effectiveness and superiority of pixel-adaptive activation in pansharpening tasks.

-> 이미지 보정 기술로 스포츠 사진을 전문가 수준으로 향상시킬 수 있어 플랫폼 콘텐츠 퀄리티 향상에 필수적

### Spatio-temporal probabilistic forecast using MMAF-guided learning (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15055v1
- 점수: final 85.6

We employ stochastic feed-forward neural networks with Gaussian-distributed weights to determine a probabilistic forecast for spatio-temporal raster datasets. The networks are trained using MMAF-guided learning, a generalized Bayesian methodology in which the observed data are preprocessed using an embedding designed to produce a low-dimensional representation that captures their dependence and causal structure. The design of the embedding is theory-guided by the assumption that a spatio-temporal Ornstein-Uhlenbeck process with finite second-order moments generates the observed data. The trained networks, in inference mode, are then used to generate ensemble forecasts by applying different initial conditions at different horizons. Experiments conducted on both synthetic and real data demonstrate that our forecasts remain calibrated across multiple time horizons. Moreover, we show that on such data, simple feed-forward architectures can achieve performance comparable to, and in some cases better than, convolutional or diffusion deep learning architectures used in probabilistic forecasting tasks.

-> Spatio-temporal forecasting techniques could be applicable to sports game analysis and strategy prediction

### Change is Hard: Consistent Player Behavior Across Games with Conflicting Incentives (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16136v1
- 점수: final 85.6

This paper examines how player flexibility -- a player's willingness to engage in a breadth of options or specialize -- manifests across two gaming environments: League of Legends (League) and Teamfight Tactics (TFT). We analyze the gameplay decisions of 4,830 players who have played at least 50 competitive games in both titles and explore cross-game dynamics of behavior retention and consistency. Our work introduces a novel cross-game analysis that tracks the same players' behavior across two different environments, reducing self-selection bias. Our findings reveal that while games incentivize different behaviors (specialization in League versus flexibility in TFT) for performance-based success, players exhibit consistent behavior across platforms. This study contributes to long-standing debate about agency versus structure, showing individual agency may be more predictive of cross-platform behavior than game-imposed structure in competitive settings. These insights offer implications for game developers, designers and researchers interested in building systems to promote behavior change.

-> 스포츠 촬영 시 다양한 조명 조건에서 발생하는 과도노출 문제를 해결하여 영상 품질을 향상시키는 데 직접적으로 적용 가능합니다.

### LICA: Layered Image Composition Annotations for Graphic Design Research (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16098v1
- 점수: final 85.6

We introduce LICA (Layered Image Composition Annotations), a large-scale dataset of 1,550,244 multi-layer graphic design compositions designed to advance structured understanding and generation of graphic layouts1. In addition to ren- dered PNG images, LICA represents each design as a hierarchical composition of typed components including text, image, vector, and group elements, each paired with rich per-element metadata such as spatial geometry, typographic attributes, opacity, and visibility. The dataset spans 20 design categories and 971,850 unique templates, providing broad coverage of real-world design structures. We further introduce graphic design video as a new and largely unexplored challenge for current vision-language models through 27,261 animated layouts annotated with per-component keyframes and motion parameters. Beyond scale, LICA establishes a new paradigm of research tasks for graphic design, enabling structured investiga- tions into problems such as layer-aware inpainting, structured layout generation, controlled design editing, and temporally-aware generative modeling. By repre- senting design as a system of compositional layers and relationships, the dataset supports research on models that operate directly on design structure rather than pixels alone.

-> 계층적 이미지 합성 기술은 스포츠 영상 편집 및 보정에 적용 가능하며, 다양한 디자인 요소를 구조화하여 전문적인 하이라이트 영상 제작을 지원합니다.

### GATS: Gaussian Aware Temporal Scaling Transformer for Invariant 4D Spatio-Temporal Point Cloud Representation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16154v1
- 점수: final 85.6

Understanding 4D point cloud videos is essential for enabling intelligent agents to perceive dynamic environments. However, temporal scale bias across varying frame rates and distributional uncertainty in irregular point clouds make it highly challenging to design a unified and robust 4D backbone. Existing CNN or Transformer based methods are constrained either by limited receptive fields or by quadratic computational complexity, while neglecting these implicit distortions. To address this problem, we propose a novel dual invariant framework, termed \textbf{Gaussian Aware Temporal Scaling (GATS)}, which explicitly resolves both distributional inconsistencies and temporal. The proposed \emph{Uncertainty Guided Gaussian Convolution (UGGC)} incorporates local Gaussian statistics and uncertainty aware gating into point convolution, thereby achieving robust neighborhood aggregation under density variation, noise, and occlusion. In parallel, the \emph{Temporal Scaling Attention (TSA)} introduces a learnable scaling factor to normalize temporal distances, ensuring frame partition invariance and consistent velocity estimation across different frame rates. These two modules are complementary: temporal scaling normalizes time intervals prior to Gaussian estimation, while Gaussian modeling enhances robustness to irregular distributions. Our experiments on mainstream benchmarks MSR-Action3D (\textbf{+6.62\%} accuracy), NTU RGBD (\textbf{+1.4\%} accuracy), and Synthia4D (\textbf{+1.8\%} mIoU) demonstrate significant performance gains, offering a more efficient and principled paradigm for invariant 4D point cloud video understanding with superior accuracy, robustness, and scalability compared to Transformer based counterparts.

-> 포인트 클라우드 데이터 처리 기술은 스포츠 장면 분석에 간접적으로 적용 가능

### SparkVSR: Interactive Video Super-Resolution via Sparse Keyframe Propagation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16864v1
- 점수: final 84.8

Video Super-Resolution (VSR) aims to restore high-quality video frames from low-resolution (LR) estimates, yet most existing VSR approaches behave like black boxes at inference time: users cannot reliably correct unexpected artifacts, but instead can only accept whatever the model produces. In this paper, we propose a novel interactive VSR framework dubbed SparkVSR that makes sparse keyframes a simple and expressive control signal. Specifically, users can first super-resolve or optionally a small set of keyframes using any off-the-shelf image super-resolution (ISR) model, then SparkVSR propagates the keyframe priors to the entire video sequence while remaining grounded by the original LR video motion. Concretely, we introduce a keyframe-conditioned latent-pixel two-stage training pipeline that fuses LR video latents with sparsely encoded HR keyframe latents to learn robust cross-space propagation and refine perceptual details. At inference time, SparkVSR supports flexible keyframe selection (manual specification, codec I-frame extraction, or random sampling) and a reference-free guidance mechanism that continuously balances keyframe adherence and blind restoration, ensuring robust performance even when reference keyframes are absent or imperfect. Experiments on multiple VSR benchmarks demonstrate improved temporal consistency and strong restoration quality, surpassing baselines by up to 24.6%, 21.8%, and 5.6% on CLIP-IQA, DOVER, and MUSIQ, respectively, enabling controllable, keyframe-driven video super-resolution. Moreover, we demonstrate that SparkVSR is a generic interactive, keyframe-conditioned video processing framework as it can be applied out of the box to unseen tasks such as old-film restoration and video style transfer. Our project page is available at: https://sparkvsr.github.io/

-> 비디오 슈퍼리졸루션 기술은 스포츠 영상 품질 개선에 적용 가능하나, 스포츠 촬영 및 분석에 직접적으로 관련되지는 않음

### EPOFusion: Exposure aware Progressive Optimization Method for Infrared and Visible Image Fusion (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16130v1
- 점수: final 84.8

Overexposure frequently occurs in practical scenarios, causing the loss of critical visual information. However, existing infrared and visible fusion methods still exhibit unsatisfactory performance in highly bright regions. To address this, we propose EPOFusion, an exposure-aware fusion model. Specifically, a guidance module is introduced to facilitate the encoder in extracting fine-grained infrared features from overexposed regions. Meanwhile, an iterative decoder incorporating a multiscale context fusion module is designed to progressively enhance the fused image, ensuring consistent details and superior visual quality. Finally, an adaptive loss function dynamically constrains the fusion process, enabling an effective balance between the modalities under varying exposure conditions. To achieve better exposure awareness, we construct the first infrared and visible overexposure dataset (IVOE) with high quality infrared guided annotations for overexposed regions. Extensive experiments show that EPOFusion outperforms existing methods. It maintains infrared cues in overexposed regions while achieving visually faithful fusion in non-overexposed areas, thereby enhancing both visual fidelity and downstream task performance. Code, fusion results and IVOE dataset will be made available at https://github.com/warren-wzw/EPOFusion.git.

-> Image fusion technology directly applicable to improving sports video quality under various lighting conditions

### SimCert: Probabilistic Certification for Behavioral Similarity in Deep Neural Network Compression (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.14818v1
- 점수: final 84.0

Deploying Deep Neural Networks (DNNs) on resource-constrained embedded systems requires aggressive model compression techniques like quantization and pruning. However, ensuring that the compressed model preserves the behavioral fidelity of the original design is a critical challenge in the safety-critical system design flow. Existing verification methods often lack scalability or fail to handle the architectural heterogeneity introduced by pruning. In this work, we propose SimCert, a probabilistic certification framework for verifying the behavioral similarity of compressed neural networks. Unlike worst-case analysis, SimCert provides quantitative safety guarantees with adjustable confidence levels. Our framework features: (1) A dual-network symbolic propagation method supporting both quantization and pruning; (2) A variance-aware bounding technique using Bernstein's inequality to tighten safety certificates; and (3) An automated verification toolchain. Experimental results on ACAS Xu and computer vision benchmarks demonstrate that SimCert outperforms state-of-the-art baselines.

-> rk3588 같은 리소스 제약이 있는 엣지 디바이스에 AI 모델을 효율적으로 배포하고 검증하는 데 필수적

### M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16844v1
- 점수: final 82.4

Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments. While coupling 3D foundation models with SLAM frameworks is a promising paradigm, a critical bottleneck persists: most multi-view foundation models estimate poses in a feed-forward manner, yielding pixel-level correspondences that lack the requisite precision for rigorous geometric optimization. To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM. M^3 further enhances tracking stability by incorporating dynamic area suppression and cross-inference intrinsic alignment. Extensive experiments on diverse indoor and outdoor benchmarks demonstrate state-of-the-art accuracy in both pose estimation and scene reconstruction. Notably, M^3 reduces ATE RMSE by 64.3% compared to VGGT-SLAM 2.0 and outperforms ARTDECO by 2.11 dB in PSNR on the ScanNet++ dataset.

-> 동적 환경에서의 정밀 자세 추정과 영상 재구성 기술은 스포츠 경기 촬영 및 하이라이트 자동 편집 시 실시간 처리와 정확도 향상에 필수적

### Learning Human-Object Interaction for 3D Human Pose Estimation from LiDAR Point Clouds (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16343v1
- 점수: final 82.4

Understanding humans from LiDAR point clouds is one of the most critical tasks in autonomous driving due to its close relationships with pedestrian safety, yet it remains challenging in the presence of diverse human-object interactions and cluttered backgrounds. Nevertheless, existing methods largely overlook the potential of leveraging human-object interactions to build robust 3D human pose estimation frameworks. There are two major challenges that motivate the incorporation of human-object interaction. First, human-object interactions introduce spatial ambiguity between human and object points, which often leads to erroneous 3D human keypoint predictions in interaction regions. Second, there exists severe class imbalance in the number of points between interacting and non-interacting body parts, with the interaction-frequent regions such as hand and foot being sparsely observed in LiDAR data. To address these challenges, we propose a Human-Object Interaction Learning (HOIL) framework for robust 3D human pose estimation from LiDAR point clouds. To mitigate the spatial ambiguity issue, we present human-object interaction-aware contrastive learning (HOICL) that effectively enhances feature discrimination between human and object points, particularly in interaction regions. To alleviate the class imbalance issue, we introduce contact-aware part-guided pooling (CPPool) that adaptively reallocates representational capacity by compressing overrepresented points while preserving informative points from interacting body parts. In addition, we present an optional contact-based temporal refinement that refines erroneous per-frame keypoint estimates using contact cues over time. As a result, our HOIL effectively leverages human-object interaction to resolve spatial ambiguity and class imbalance in interaction regions. Codes will be released.

-> 인간-객체 상호작용 학습 기술은 스포츠에서의 선수 자세 분석과 동작 패턴 인식에 적용 가능하며, 다양한 상황에서의 정확한 추적이 필요

### Video Detector: A Dual-Phase Vision-Based System for Real-Time Traffic Intersection Control and Intelligent Transportation Analysis (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.14861v1
- 점수: final 82.4

Urban traffic management increasingly requires intelligent sensing systems capable of adapting to dynamic traffic conditions without costly infrastructure modifications. Vision-based vehicle detection has therefore become a key technology for modern intelligent transportation systems. This study presents Video Detector (VD), a dual-phase vision-based traffic intersection management system designed as a flexible and cost-effective alternative to traditional inductive loop detectors. The framework integrates a real-time module (VD-RT) for intersection control with an offline analytical module (VD-Offline) for detailed traffic behavior analysis. Three system configurations were implemented using SSD Inception v2, Faster R-CNN Inception v2, and CenterNet ResNet-50 V1 FPN, trained on datasets totaling 108,000 annotated images across 6-10 vehicle classes. Experimental results show detection performance of up to 90% test accuracy and 29.5 mAP@0.5, while maintaining real-time throughput of 37 FPS on HD video streams. Field deployments conducted in collaboration with Istanbul IT and Smart City Technologies Inc. (ISBAK) demonstrate stable operation under diverse environmental conditions. The system supports virtual loop detection, vehicle counting, multi-object tracking, queue estimation, speed analysis, and multiclass vehicle classification, enabling comprehensive intersection monitoring without the need for embedded road sensors. The annotated dataset and training pipeline are publicly released to support reproducibility. These results indicate that the proposed framework provides a scalable and deployable vision-based solution for intelligent transportation systems and smart-city traffic management.

-> 실시간 비전 처리 시스템으로 스포츠 영상 분석에 직접 적용 가능하며, 객체 탐지 및 추적 기술은 선수 추적에 활용될 수 있습니다.

### Multi-turn Physics-informed Vision-language Model for Physics-grounded Anomaly Detection (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15237v1
- 점수: final 82.4

Vision-Language Models (VLMs) demonstrate strong general-purpose reasoning but remain limited in physics-grounded anomaly detection, where causal understanding of dynamics is essential. Existing VLMs, trained predominantly on appearance-centric correlations, fail to capture kinematic constraints, leading to poor performance on anomalies such as irregular rotations or violated mechanical motions. We introduce a physics-informed instruction tuning framework that explicitly encodes object properties, motion paradigms, and dynamic constraints into structured prompts. By delivering these physical priors through multi-turn dialogues, our method decomposes causal reasoning into incremental steps, enabling robust internal representations of normal and abnormal dynamics. Evaluated on the Phys-AD benchmark, our approach achieves 96.7% AUROC in video-level detection--substantially outperforming prior SOTA (66.9%)--and yields superior causal explanations (0.777 LLM score). This work highlights how structured physics priors can transform VLMs into reliable detectors of dynamic anomalies.

-> 물리 기반 비디오 이상 감지 기술로 스포츠 동작 및 전략 분석에 적용 가능

### GATE-AD: Graph Attention Network Encoding For Few-Shot Industrial Visual Anomaly Detection (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15300v1
- 점수: final 82.4

Few-Shot Industrial Visual Anomaly Detection (FS-IVAD) comprises a critical task in modern manufacturing settings, where automated product inspection systems need to identify rare defects using only a handful of normal/defect-free training samples. In this context, the current study introduces a novel reconstruction-based approach termed GATE-AD. In particular, the proposed framework relies on the employment of a masked, representation-aligned Graph Attention Network (GAT) encoding scheme to learn robust appearance patterns of normal samples. By leveraging dense, patch-level, visual feature tokens as graph nodes, the model employs stacked self-attentional layers to adaptively encode complex, irregular, non-Euclidean, local relations. The graph is enhanced with a representation alignment component grounded on a learnable, latent space, where high reconstruction residual areas (i.e., defects) are assessed using a Scaled Cosine Error (SCE) objective function. Extensive comparative evaluation on the MVTec AD, VisA, and MPDD industrial defect detection benchmarks demonstrates that GATE-AD achieves state-of-the-art performance across the $1$- to $8$-shot settings, combining the highest detection accuracy (increase up to $1.8\%$ in image AUROC in the 8-shot case in MPDD) with the lowest per-image inference latency (at least $25.05\%$ faster), compared to the best-performing literature methods. In order to facilitate reproducibility and further research, the source code of GATE-AD is available at https://github.com/gthpapadopoulos/GATE-AD.

-> Visual analysis techniques could be adapted for sports movement analysis

### Anchor then Polish for Low-light Enhancement (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15472v1
- 점수: final 82.4

Low-light image enhancement is challenging due to entangled degradations, mainly including poor illumination, color shifts, and texture interference. Existing methods often rely on complex architectures to address these issues jointly but may overfit simple physical constraints, leading to global distortions. This work proposes a novel anchor-then-polish (ATP) framework to fundamentally decouple global energy alignment from local detail refinement. First, macro anchoring is customized to (greatly) stabilize luminance distribution and correct color by learning a scene-adaptive projection matrix with merely 12 degrees of freedom, revealing that a simple linear operator can effectively align global energy. The macro anchoring then reduces the task to micro polishing, which further refines details in the wavelet domain and chrominance space under matrix guidance. A constrained luminance update strategy is designed to ensure global consistency while directing the network to concentrate on fine-grained polishing. Extensive experiments on multiple benchmarks show that our method achieves state-of-the-art performance, producing visually natural and quantitatively superior low-light enhancements.

-> Low-light enhancement technology directly applicable for improving sports footage in challenging lighting conditions

### When Thinking Hurts: Mitigating Visual Forgetting in Video Reasoning via Frame Repetition (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16256v1
- 점수: final 80.8

Recently, Multimodal Large Language Models (MLLMs) have demonstrated significant potential in complex visual tasks through the integration of Chain-of-Thought (CoT) reasoning. However, in Video Question Answering, extended thinking processes do not consistently yield performance gains and may even lead to degradation due to ``visual anchor drifting'', where models increasingly rely on self-generated text, sidelining visual inputs and causing hallucinations. While existing mitigations typically introduce specific mechanisms for the model to re-attend to visual inputs during inference, these approaches often incur prohibitive training costs and suffer from poor generalizability across different architectures. To address this, we propose FrameRepeat, an automated enhancement framework which features a lightweight repeat scoring module that enables Video-LLMs to autonomously identify which frames should be reinforced. We introduce a novel training strategy, Add-One-In (AOI), that uses MLLM output probabilities to generate supervision signals representing repeat gain. This can be used to train a frame scoring network, which guides the frame repetition behavior. Experimental results across multiple models and datasets demonstrate that FrameRepeat is both effective and generalizable in strengthening important visual cues during the reasoning process.

-> 비디오 추론 시 중요한 프레임을 식별하는 기술로 스포츠 하이라이트 장면 자동 식별에 직접 적용 가능합니다.

### TinyGLASS: Real-Time Self-Supervised In-Sensor Anomaly Detection (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16451v1
- 점수: final 80.0

Anomaly detection plays a key role in industrial quality control, where defects must be identified despite the scarcity of labeled faulty samples. Recent self-supervised approaches, such as GLASS, learn normal visual patterns using only defect-free data and have shown strong performance on industrial benchmarks. However, their computational requirements limit deployment on resource-constrained edge platforms.   This work introduces TinyGLASS, a lightweight adaptation of the GLASS framework designed for real-time in-sensor anomaly detection on the Sony IMX500 intelligent vision sensor. The proposed architecture replaces the original WideResNet-50 backbone with a compact ResNet-18 and introduces deployment-oriented modifications that enable static graph tracing and INT8 quantization using Sony's Model Compression Toolkit.   In addition to evaluating performance on the MVTec-AD benchmark, we investigate robustness to contaminated training data and introduce a custom industrial dataset, named MMS Dataset, for cross-device evaluation. Experimental results show that TinyGLASS achieves 8.7x parameter compression while maintaining competitive detection performance, reaching 94.2% image-level AUROC on MVTec-AD and operating at 20 FPS within the 8 MB memory constraints of the IMX500 platform.   System profiling demonstrates low power consumption (4.0 mJ per inference), real-time end-to-end latency (20 FPS), and high energy efficiency (470 GMAC/J). Furthermore, the model maintains stable performance under moderate levels of training data contamination.

-> 엣지 디바이스에서 실시간 이상 탐지 기술은 프로젝트의 rk3588 기반 엣지 디바이스 개발에 기술적 참고가 됩니다.

### Large Reward Models: Generalizable Online Robot Reward Generation with Vision-Language Models (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16065v1
- 점수: final 80.0

Reinforcement Learning (RL) has shown great potential in refining robotic manipulation policies, yet its efficacy remains strongly bottlenecked by the difficulty of designing generalizable reward functions. In this paper, we propose a framework for online policy refinement by adapting foundation VLMs into online reward generators. We develop a robust, scalable reward model based on a state-of-the-art VLM, trained on a large-scale, multi-source dataset encompassing real-world robot trajectories, human-object interactions, and diverse simulated environments. Unlike prior approaches that evaluate entire trajectories post-hoc, our method leverages the VLM to formulate a multifaceted reward signal comprising process, completion, and temporal contrastive rewards based on current visual observations. Initializing with a base policy trained via Imitation Learning (IL), we employ these VLM rewards to guide the model to correct sub-optimal behaviors in a closed-loop manner. We evaluate our framework on challenging long-horizon manipulation benchmarks requiring sequential execution and precise control. Crucially, our reward model operates in a purely zero-shot manner within these test environments. Experimental results demonstrate that our method significantly improves the success rate of the initial IL policy within just 30 RL iterations, demonstrating remarkable sample efficiency. This empirical evidence highlights that VLM-generated signals can provide reliable feedback to resolve execution errors, effectively eliminating the need for manual reward engineering and facilitating efficient online refinement for robot learning.

-> 비전-언어 모델을 활용한 보상 생성 시스템은 스포츠 자세 및 동작 분석에 적용 가능한 핵심 기술입니다.

### Demystifing Video Reasoning (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16870v1
- 점수: final 80.0

Recent advances in video generation have revealed an unexpected phenomenon: diffusion-based video models exhibit non-trivial reasoning capabilities. Prior work attributes this to a Chain-of-Frames (CoF) mechanism, where reasoning is assumed to unfold sequentially across video frames. In this work, we challenge this assumption and uncover a fundamentally different mechanism. We show that reasoning in video models instead primarily emerges along the diffusion denoising steps. Through qualitative analysis and targeted probing experiments, we find that models explore multiple candidate solutions in early denoising steps and progressively converge to a final answer, a process we term Chain-of-Steps (CoS). Beyond this core mechanism, we identify several emergent reasoning behaviors critical to model performance: (1) working memory, enabling persistent reference; (2) self-correction and enhancement, allowing recovery from incorrect intermediate solutions; and (3) perception before action, where early steps establish semantic grounding and later steps perform structured manipulation. During a diffusion step, we further uncover self-evolved functional specialization within Diffusion Transformers, where early layers encode dense perceptual structure, middle layers execute reasoning, and later layers consolidate latent representations. Motivated by these insights, we present a simple training-free strategy as a proof-of-concept, demonstrating how reasoning can be improved by ensembling latent trajectories from identical models with different random seeds. Overall, our work provides a systematic understanding of how reasoning emerges in video generation models, offering a foundation to guide future research in better exploiting the inherent reasoning dynamics of video models as a new substrate for intelligence.

-> 비디오 모델의 추론 메커니즘 이해는 스포츠 하이라이트 감지 및 전략 분석에 중요한 기술적 기반을 제공합니다.

### Optimal uncertainty bounds for multivariate kernel regression under bounded noise: A Gaussian process-based dual function (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16481v1
- 점수: final 80.0

Non-conservative uncertainty bounds are essential for making reliable predictions about latent functions from noisy data--and thus, a key enabler for safe learning-based control. In this domain, kernel methods such as Gaussian process regression are established techniques, thanks to their inherent uncertainty quantification mechanism. Still, existing bounds either pose strong assumptions on the underlying noise distribution, are conservative, do not scale well in the multi-output case, or are difficult to integrate into downstream tasks. This paper addresses these limitations by presenting a tight, distribution-free bound for multi-output kernel-based estimates. It is obtained through an unconstrained, duality-based formulation, which shares the same structure of classic Gaussian process confidence bounds and can thus be straightforwardly integrated into downstream optimization pipelines. We show that the proposed bound generalizes many existing results and illustrate its application using an example inspired by quadrotor dynamics learning.

-> 가우시안 프로세스 불확정성 정량화 기술은 스포츠 동작 및 전략 분석의 신뢰성을 높이는 데 직접 적용 가능합니다.

### Learning Whole-Body Control for a Salamander Robot (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.16683v1
- 점수: final 80.0

Amphibious legged robots inspired by salamanders are promising in applications in complex amphibious environments. However, despite the significant success of training controllers that achieve diverse locomotion behaviors in conventional quadrupedal robots, most salamander robots relied on central-pattern-generator (CPG)-based and model-based coordination strategies for locomotion control. Learning unified joint-level whole-body control that reliably transfers from simulation to highly articulated physical salamander robots remains relatively underexplored. In addition, few legged robots have tried learning-based controllers in amphibious environments. In this work, we employ Reinforcement Learning to map proprioceptive observations and commanded velocities to joint-level actions, allowing coordinated locomotor behaviors to emerge. To deploy these policies on hardware, we adopt a system-level real-to-sim matching and sim-to-real transfer strategy. The learned controller achieves stable and coordinated walking on both flat and uneven terrains in the real world. Beyond terrestrial locomotion, the framework enables transitions between walking and swimming in simulation, highlighting a phenomenon of interest for understanding locomotion across distinct physical modes.

-> 로봇 동작 학습 기술은 스포츠 동작 분석에 직접 적용 가능

### GeoNVS: Geometry Grounded Video Diffusion for Novel View Synthesis (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.14965v1
- 점수: final 80.0

Novel view synthesis requires strong 3D geometric consistency and the ability to generate visually coherent images across diverse viewpoints. While recent camera-controlled video diffusion models show promising results, they often suffer from geometric distortions and limited camera controllability. To overcome these challenges, we introduce GeoNVS, a geometry-grounded novel-view synthesizer that enhances both geometric fidelity and camera controllability through explicit 3D geometric guidance. Our key innovation is the Gaussian Splat Feature Adapter (GS-Adapter), which lifts input-view diffusion features into 3D Gaussian representations, renders geometry-constrained novel-view features, and adaptively fuses them with diffusion features to correct geometrically inconsistent representations. Unlike prior methods that inject geometry at the input level, GS-Adapter operates in feature space, avoiding view-dependent color noise that degrades structural consistency. Its plug-and-play design enables zero-shot compatibility with diverse feed-forward geometry models without additional training, and can be adapted to other video diffusion backbones. Experiments across 9 scenes and 18 settings demonstrate state-of-the-art performance, achieving 11.3% and 14.9% improvements over SEVA and CameraCtrl, with up to 2x reduction in translation error and 7x in Chamfer Distance.

-> 영상의 새로운 각도 생성 기술로 스포츠 촬영에 간접적 적용 가능하나 스포츠/엣지 특화 아님

### IRIS: Intersection-aware Ray-based Implicit Editable Scenes (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15368v1
- 점수: final 80.0

Neural Radiance Fields achieve high-fidelity scene representation but suffer from costly training and rendering, while 3D Gaussian splatting offers real-time performance with strong empirical results. Recently, solutions that harness the best of both worlds by using Gaussians as proxies to guide neural field evaluations, still suffer from significant computational inefficiencies. They typically rely on stochastic volumetric sampling to aggregate features, which severely limits rendering performance. To address this issue, a novel framework named IRIS (Intersection-aware Ray-based Implicit Editable Scenes) is introduced as a method designed for efficient and interactive scene editing. To overcome the limitations of standard ray marching, an analytical sampling strategy is employed that precisely identifies interaction points between rays and scene primitives, effectively eliminating empty space processing. Furthermore, to address the computational bottleneck of spatial neighbor lookups, a continuous feature aggregation mechanism is introduced that operates directly along the ray. By interpolating latent attributes from sorted intersections, costly 3D searches are bypassed, ensuring geometric consistency, enabling high-fidelity, real-time rendering, and flexible shape editing. Code can be found at https://github.com/gwilczynski95/iris.

-> 스포츠 하이라이트 영상 제작에 효율적인 장면 편집과 실시간 렌더링 기술 적용 가능

### Exposing Cross-Modal Consistency for Fake News Detection in Short-Form Videos (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.14992v1
- 점수: final 80.0

Short-form video platforms are major channels for news but also fertile ground for multimodal misinformation where each modality appears plausible alone yet cross-modal relationships are subtly inconsistent, like mismatched visuals and captions. On two benchmark datasets, FakeSV (Chinese) and FakeTT (English), we observe a clear asymmetry: real videos exhibit high text-visual but moderate text-audio consistency, while fake videos show the opposite pattern. Moreover, a single global consistency score forms an interpretable axis along which fake probability and prediction errors vary smoothly. Motivated by these observations, we present MAGIC3 (Modal-Adversarial Gated Interaction and Consistency-Centric Classifier), a detector that explicitly models and exposes cross-tri-modal consistency signals at multiple granularities. MAGIC3 combines explicit pairwise and global consistency modeling with token- and frame-level consistency signals derived from cross-modal attention, incorporates multi-style LLM rewrites to obtain style-robust text representations, and employs an uncertainty-aware classifier for selective VLM routing. Using pre-extracted features, MAGIC3 consistently outperforms the strongest non-VLM baselines on FakeSV and FakeTT. While matching VLM-level accuracy, the two-stage system achieves 18-27x higher throughput and 93% VRAM savings, offering a strong cost-performance tradeoff.

-> 숏폼 콘텐츠의 진위 여부 판단 기술로 스포츠 영상의 신뢰성 검증에 활용 가능

### Learning Latent Proxies for Controllable Single-Image Relighting (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.15555v1
- 점수: final 80.0

Single-image relighting is highly under-constrained: small illumination changes can produce large, nonlinear variations in shading, shadows, and specularities, while geometry and materials remain unobserved. Existing diffusion-based approaches either rely on intrinsic or G-buffer pipelines that require dense and fragile supervision, or operate purely in latent space without physical grounding, making fine-grained control of direction, intensity, and color unreliable. We observe that a full intrinsic decomposition is unnecessary and redundant for accurate relighting. Instead, sparse but physically meaningful cues, indicating where illumination should change and how materials should respond, are sufficient to guide a diffusion model. Based on this insight, we introduce LightCtrl that integrates physical priors at two levels: a few-shot latent proxy encoder that extracts compact material-geometry cues from limited PBR supervision, and a lighting-aware mask that identifies sensitive illumination regions and steers the denoiser toward shading relevant pixels. To compensate for scarce PBR data, we refine the proxy branch using a DPO-based objective that enforces physical consistency in the predicted cues. We also present ScaLight, a large-scale object-level dataset with systematically varied illumination and complete camera-light metadata, enabling physically consistent and controllable training. Across object and scene level benchmarks, our method achieves photometrically faithful relighting with accurate continuous control, surpassing prior diffusion and intrinsic-based baselines, including gains of up to +2.4 dB PSNR and 35% lower RMSE under controlled lighting shifts.

-> 이미지 보정 기술로 촬영된 스포츠 영상의 품질 향상에 직접 적용 가능

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
