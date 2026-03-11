# CAPP!C_AI 논문 리포트 (2026-03-11)

> 수집 83 | 필터 74 | 폐기 8 | 평가 74 | 출력 60 | 기준 50점

검색 윈도우: 2026-03-10T00:00:00+00:00 ~ 2026-03-11T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 35

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: Streaming Autoregressive Video Generation via Diagonal Distillation

- arXiv: http://arxiv.org/abs/2603.09488v1
- PDF: https://arxiv.org/pdf/2603.09488v1
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 100.0 (llm_adjusted:100 = base:95 + bonus:+5)
- 플래그: 실시간

**개요**
Large pretrained diffusion models have significantly enhanced the quality of generated videos, and yet their use in real-time streaming remains limited. Autoregressive models offer a natural framework for sequential frame synthesis but require heavy computation to achieve high fidelity. Diffusion distillation can compress these models into efficient few-step variants, but existing video distillation approaches largely adapt image-specific methods that neglect temporal dependencies. These techniques often excel in image generation but underperform in video synthesis, exhibiting reduced motion coherence, error accumulation over long sequences, and a latency-quality trade-off. We identify two factors that result in these limitations: insufficient utilization of temporal context during step reduction and implicit prediction of subsequent noise levels in next-chunk prediction (i.e., exposure bias). To address these issues, we propose Diagonal Distillation, which operates orthogonally to existing approaches and better exploits temporal information across both video chunks and denoising steps. Central to our approach is an asymmetric generation strategy: more steps early, fewer steps later. This design allows later chunks to inherit rich appearance information from thoroughly processed early chunks, while using partially denoised chunks as conditional inputs for subsequent synthesis. By aligning the implicit prediction of subsequent noise levels during chunk generation with the actual inference conditions, our approach mitigates error propagation and reduces oversaturation in long-range sequences. We further incorporate implicit optical flow modeling to preserve motion quality under strict step constraints. Our method generates a 5-second video in 2.61 seconds (up to 31 FPS), achieving a 277.3x speedup over the undistilled model.

**선정 근거**
물리적으로 타당한 스포츠 하이라이트 자동 생성 기술로 경기 장면의 인과적 연결을 유지하며 자연스러운 편집 가능

**활용 인사이트**
물리 이벤트 체인 추론 모듈로 주요 장면 분해하고 TCP로 연속성 유지하여 실시간으로 고품질 하이라이트 영상 생성

## 2위: Chain of Event-Centric Causal Thought for Physically Plausible Video Generation

- arXiv: http://arxiv.org/abs/2603.09094v1
- PDF: https://arxiv.org/pdf/2603.09094v1
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 100.0 (llm_adjusted:100 = base:95 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Physically Plausible Video Generation (PPVG) has emerged as a promising avenue for modeling real-world physical phenomena. PPVG requires an understanding of commonsense knowledge, which remains a challenge for video diffusion models. Current approaches leverage commonsense reasoning capability of large language models to embed physical concepts into prompts. However, generation models often render physical phenomena as a single moment defined by prompts, due to the lack of conditioning mechanisms for modeling causal progression. In this paper, we view PPVG as generating a sequence of causally connected and dynamically evolving events. To realize this paradigm, we design two key modules: (1) Physics-driven Event Chain Reasoning. This module decomposes the physical phenomena described in prompts into multiple elementary event units, leveraging chain-of-thought reasoning. To mitigate causal ambiguity, we embed physical formulas as constraints to impose deterministic causal dependencies during reasoning. (2) Transition-aware Cross-modal Prompting (TCP). To maintain continuity between events, this module transforms causal event units into temporally aligned vision-language prompts. It summarizes discrete event descriptions to obtain causally consistent narratives, while progressively synthesizing visual keyframes of individual events by interactive editing. Comprehensive experiments on PhyGenBench and VideoPhy benchmarks demonstrate that our framework achieves superior performance in generating physically plausible videos across diverse physical domains. Our code will be released soon.

**선정 근거**
물리적으로 타당한 영상 생성 기술이 스포츠 하이라이트 자동 편집 및 보정에 직접적으로 적용 가능

## 3위: CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation

- arXiv: http://arxiv.org/abs/2603.09418v1
- PDF: https://arxiv.org/pdf/2603.09418v1
- 코드: https://github.com/53mins/CIGPose
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 98.4 (llm_adjusted:98 = base:95 + bonus:+3)
- 플래그: 코드 공개

**개요**
State-of-the-art whole-body pose estimators often lack robustness, producing anatomically implausible predictions in challenging scenes. We posit this failure stems from spurious correlations learned from visual context, a problem we formalize using a Structural Causal Model (SCM). The SCM identifies visual context as a confounder that creates a non-causal backdoor path, corrupting the model's reasoning. We introduce the Causal Intervention Graph Pose (CIGPose) framework to address this by approximating the true causal effect between visual evidence and pose. The core of CIGPose is a novel Causal Intervention Module: it first identifies confounded keypoint representations via predictive uncertainty and then replaces them with learned, context-invariant canonical embeddings. These deconfounded embeddings are processed by a hierarchical graph neural network that reasons over the human skeleton at both local and global semantic levels to enforce anatomical plausibility. Extensive experiments show CIGPose achieves a new state-of-the-art on COCO-WholeBody. Notably, our CIGPose-x model achieves 67.0\% AP, surpassing prior methods that rely on extra training data. With the additional UBody dataset, CIGPose-x is further boosted to 67.5\% AP, demonstrating superior robustness and data efficiency. The codes and models are publicly available at https://github.com/53mins/CIGPose.

**선정 근거**
전신 자세 추정 기술은 스포츠 동작 분석에 직접적으로 활용 가능한 핵심 기술이다

**활용 인사이트**
67.0% AP 달성으로 선수들의 정확한 동작 분석과 전략적 분석에 활용 가능

## 4위: TrainDeeploy: Hardware-Accelerated Parameter-Efficient Fine-Tuning of Small Transformer Models at the Extreme Edge

- arXiv: http://arxiv.org/abs/2603.09511v1
- PDF: https://arxiv.org/pdf/2603.09511v1
- 발행일: 2026-03-10
- 카테고리: cs.AR, cs.LG
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
On-device tuning of deep neural networks enables long-term adaptation at the edge while preserving data privacy. However, the high computational and memory demands of backpropagation pose significant challenges for ultra-low-power, memory-constrained extreme-edge devices. These challenges are further amplified for attention-based models due to their architectural complexity and computational scale. We present TrainDeeploy, a framework that unifies efficient inference and on-device training on heterogeneous ultra-low-power System-on-Chips (SoCs). TrainDeeploy provides the first complete on-device training pipeline for extreme-edge SoCs supporting both Convolutional Neural Networks (CNNs) and Transformer models, together with multiple training strategies such as selective layer-wise fine-tuning and Low-Rank Adaptation (LoRA). On a RISC-V-based heterogeneous SoC, we demonstrate the first end-to-end on-device fine-tuning of a Compact Convolutional Transformer (CCT), achieving up to 11 trained images per second. We show that LoRA reduces dynamic memory usage by 23%, decreases the number of trainable parameters and gradients by 15x, and reduces memory transfer volume by 1.6x compared to full backpropagation. TrainDeeploy achieves up to 4.6 FLOP/cycle on CCT (0.28M parameters, 71-126M FLOPs) and up to 13.4 FLOP/cycle on Deep-AE (0.27M parameters, 0.8M FLOPs), while expanding the scope of prior frameworks to support both CNN and Transformer models with parameter-efficient tuning on extreme-edge platforms.

**선정 근거**
rk3588 기반 edge device에서 AI 모델 효율적 튜닝을 위한 하드웨어 가속 기술

**활용 인사이트**
최대 11장/초 학습 가능하며, LoRA로 메모리 사용량 23% 감소 및 훈련 파라미터 15x 감소

## 5위: CycleULM: A unified label-free deep learning framework for ultrasound localisation microscopy

- arXiv: http://arxiv.org/abs/2603.09840v1
- PDF: https://arxiv.org/pdf/2603.09840v1
- 발행일: 2026-03-10
- 카테고리: eess.IV, cs.CV
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Super-resolution ultrasound via microbubble (MB) localisation and tracking, also known as ultrasound localisation microscopy (ULM), can resolve microvasculature beyond the acoustic diffraction limit. However, significant challenges remain in localisation performance and data acquisition and processing time. Deep learning methods for ULM have shown promise to address these challenges, however, they remain limited by in vivo label scarcity and the simulation-to-reality domain gap. We present CycleULM, the first unified label-free deep learning framework for ULM. CycleULM learns a physics-emulating translation between the real contrast-enhanced ultrasound (CEUS) data domain and a simplified MB-only domain, leveraging the power of CycleGAN without requiring paired ground truth data. With this translation, CycleULM removes dependence on high-fidelity simulators or labelled data, and makes MB localisation and tracking substantially easier. Deployed as modular plug-and-play components within existing pipelines or as an end-to-end processing framework, CycleULM delivers substantial performance gains across both in silico and in vivo datasets. Specifically, CycleULM improves image contrast (contrast-to-noise ratio) by up to 15.3 dB and sharpens CEUS resolution with a 2.5{\times} reduction in the full width at half maximum of the point spread function. CycleULM also improves MB localisation performance, with up to +40% recall, +46% precision, and a -14.0 μm mean localisation error, yielding more faithful vascular reconstructions. Importantly, CycleULM achieves real-time processing throughput at 18.3 frames per second with order-of-magnitude speed-ups (up to ~14.5{\times}). By combining label-free learning, performance enhancement, and computational efficiency, CycleULM provides a practical pathway toward robust, real-time ULM and accelerates its translation to clinical applications.

**선정 근거**
실시간 처리 성능과 물리 모델링 기술이 스포츠 영상 분석에 직접적으로 적용 가능

## 6위: PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies

- arXiv: http://arxiv.org/abs/2603.09216v1
- PDF: https://arxiv.org/pdf/2603.09216v1
- 발행일: 2026-03-10
- 카테고리: cs.DC
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
On-device deployments of large language models (LLMs) are rapidly proliferating across mobile and edge platforms. LLM inference comprises a compute-intensive prefill phase and a memory bandwidth-intensive decode phase, and the decode phase has been widely recognized as well-suited to processing-in-memory (PIM) in both academia and industry. However, practical PIM-enabled systems face two obstacles between these phases, a memory attribute inconsistency in which prefill favors placing weights in a cacheable region for reuse whereas decode requires weights in a non-cacheable region to reliably trigger PIM, and a weight layout inconsistency between host-friendly and PIM-aware layouts. To address these problems, we introduce \textit{PIM-SHERPA}, a software-only method for efficient on-device LLM inference by resolving PIM memory attribute and layout inconsistencies. PIM-SHERPA provides two approaches, DRAM double buffering (DDB), which keeps a single PIM-aware weights in the non-cacheable region while prefetching the swizzled weights of the next layer into small cacheable buffers, and online weight rearrangement with swizzled memory copy (OWR), which performs the on-demand swizzled memory copy immediately before GEMM. Compared to a baseline PIM emulation system, PIM-SHERPA achieves approximately 47.8 - 49.7\% memory capacity savings while maintaining comparable performance to the theoretical maximum on the Llama 3.2 model. To the best of our knowledge, this is the first work to identify the memory attribute inconsistency and propose effective solutions on product-level PIM-enabled systems.

**선정 근거**
rk3588 엣지 디바이스에서의 효율적인 AI 추론을 위한 메모리 최적화 방법으로 실시간 스포츠 분석에 적용 가능

**활용 인사이트**
PIM 기술을 활용해 메모리 사용량을 47.8-49.7% 절감하면서도 성능을 유지하여 스포츠 영상 처리 속도 향상

## 7위: Decoder-Free Distillation for Quantized Image Restoration

- arXiv: http://arxiv.org/abs/2603.09624v1
- PDF: https://arxiv.org/pdf/2603.09624v1
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Quantization-Aware Training (QAT), combined with Knowledge Distillation (KD), holds immense promise for compressing models for edge deployment. However, joint optimization for precision-sensitive image restoration (IR) to recover visual quality from degraded images remains largely underexplored. Directly adapting QAT-KD to low-level vision reveals three critical bottlenecks: teacher-student capacity mismatch, spatial error amplification during decoder distillation, and an optimization "tug-of-war" between reconstruction and distillation losses caused by quantization noise. To tackle these, we introduce Quantization-aware Distilled Restoration (QDR), a framework for edge-deployed IR. QDR eliminates capacity mismatch via FP32 self-distillation and prevents error amplification through Decoder-Free Distillation (DFD), which corrects quantization errors strictly at the network bottleneck. To stabilize the optimization tug-of-war, we propose a Learnable Magnitude Reweighting (LMR) that dynamically balances competing gradients. Finally, we design an Edge-Friendly Model (EFM) featuring a lightweight Learnable Degradation Gating (LDG) to dynamically modulate spatial degradation localization. Extensive experiments across four IR tasks demonstrate that our Int8 model recovers 96.5% of FP32 performance, achieves 442 frames per second (FPS) on an NVIDIA Jetson Orin, and boosts downstream object detection by 16.3 mAP

**선정 근거**
실시간 처리가 가능한 엣지 기반 이미지 복원 기술로 스포츠 촬영 장비에 직접 적용 가능

**활용 인사이트**
442 FPS의 처리 속도로 스포츠 영상의 품질을 향상시키고 하이라이트 자동 생성에 활용

## 8위: MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging

- arXiv: http://arxiv.org/abs/2603.09116v1
- PDF: https://arxiv.org/pdf/2603.09116v1
- 발행일: 2026-03-10
- 카테고리: eess.IV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
We present MetaSpectra+, a compact multifunctional camera that supports two operating modes: (1) snapshot HDR + hyperspectral or (2) snapshot polarization + hyperspectral imaging. It utilizes a novel metasurface-refractive assembly that splits the incident beam into multiple channels and independently controls each channel's dispersion, exposure, and polarization. Unlike prior multifunctional metasurface imagers restricted to narrow (10-100 nm) bands, MetaSpectra+ operates over nearly the entire visible spectrum (250 nm). Relative to snapshot hyperspectral imagers, it achieves the shortest total track length and the highest reconstruction accuracy on benchmark datasets. The demonstrated prototype reconstructs high-quality hyperspectral datacubes and either an HDR image or two orthogonal polarization channels from a single snapshot.

**선정 근거**
다중 기능 컴팩트 카메라 기술이 스포츠 촬영 및 분석에 적용 가능

**활용 인사이트**
단일 샷으로 고품질 하이퍼스펙트럼 데이터와 HDR 이미지 동시 캡처로 스포츠 장면 분석 정확도 향상

## 9위: Two Teachers Better Than One: Hardware-Physics Co-Guided Distributed Scientific Machine Learning

- arXiv: http://arxiv.org/abs/2603.09032v1
- PDF: https://arxiv.org/pdf/2603.09032v1
- 발행일: 2026-03-10
- 카테고리: cs.LG, cs.AR, cs.CE, cs.DC
- 점수: final 90.0 (llm_adjusted:90 = base:80 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Scientific machine learning (SciML) is increasingly applied to in-field processing, controlling, and monitoring; however, wide-area sensing, real-time demands, and strict energy and reliability constraints make centralized SciML implementation impractical. Most SciML models assume raw data aggregation at a central node, incurring prohibitively high communication latency and energy costs; yet, distributing models developed for general-purpose ML often breaks essential physical principles, resulting in degraded performance. To address these challenges, we introduce EPIC, a hardware- and physics-co-guided distributed SciML framework, using full-waveform inversion (FWI) as a representative task. EPIC performs lightweight local encoding on end devices and physics-aware decoding at a central node. By transmitting compact latent features rather than high-volume raw data and by using cross-attention to capture inter-receiver wavefield coupling, EPIC significantly reduces communication cost while preserving physical fidelity. Evaluated on a distributed testbed with five end devices and one central node, and across 10 datasets from OpenFWI, EPIC reduces latency by 8.9$\times$ and communication energy by 33.8$\times$, while even improving reconstruction fidelity on 8 out of 10 datasets.

**선정 근거**
엣지 디바이스용 분산 컴퓨팅 프레임워크로 물리 인식 처리가 스포츠 분석에 적용 가능

**활용 인사이트**
경량 로컬 인코딩과 물리 인식 디코딩으로 지연 시간 8.9배 감소하며 실시간 스포츠 분석 가능

## 10위: TemporalDoRA: Temporal PEFT for Robust Surgical Video Question Answering

- arXiv: http://arxiv.org/abs/2603.09696v1
- PDF: https://arxiv.org/pdf/2603.09696v1
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 88.8 (llm_adjusted:86 = base:78 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Surgical Video Question Answering (VideoQA) requires accurate temporal grounding while remaining robust to natural variation in how clinicians phrase questions, where linguistic bias can arise. Standard Parameter Efficient Fine Tuning (PEFT) methods adapt pretrained projections without explicitly modeling frame-to-frame interactions within the adaptation pathway, limiting their ability to exploit sparse temporal evidence. We introduce TemporalDoRA, a video-specific PEFT formulation that extends Weight-Decomposed Low-Rank Adaptation by (i) inserting lightweight temporal Multi-Head Attention (MHA) inside the low-rank bottleneck of the vision encoder and (ii) selectively applying weight decomposition only to the trainable low-rank branch rather than the full adapted weight. This design enables temporally-aware updates while preserving a frozen backbone and stable scaling. By mixing information across frames within the adaptation subspace, TemporalDoRA steers updates toward temporally consistent visual cues and improves robustness with minimal parameter overhead. To benchmark this setting, we present REAL-Colon-VQA, a colonoscopy VideoQA dataset with 6,424 clip--question pairs, including paired rephrased Out-of-Template questions to evaluate sensitivity to linguistic variation. TemporalDoRA improves Out-of-Template performance, and ablation studies confirm that temporal mixing inside the low-rank branch is the primary driver of these gains. We also validate on EndoVis18-VQA adapted to short clips and observe consistent improvements on the Out-of-Template split. Code and dataset available at~\href{https://anonymous.4open.science/r/TemporalDoRA-BFC8/}{Anonymous GitHub}.

**선정 근거**
수술 비디오 QA를 위한 시간적 처리 기술은 스포츠 분석으로 적용될 수 있습니다

**활용 인사이트**
경량 시간적 멀티 헤드 어텐션으로 스포츠 영상의 프레임 간 상호작용 분석 정확도 향상

## 11위: M3GCLR: Multi-View Mini-Max Infinite Skeleton-Data Game Contrastive Learning For Skeleton-Based Action Recognition

- arXiv: http://arxiv.org/abs/2603.09367v1
- PDF: https://arxiv.org/pdf/2603.09367v1
- 발행일: 2026-03-10
- 카테고리: cs.CV, cs.AI
- 점수: final 88.0 (llm_adjusted:85 = base:80 + bonus:+5)
- 플래그: 엣지

**개요**
In recent years, contrastive learning has drawn significant attention as an effective approach to reducing reliance on labeled data. However, existing methods for self-supervised skeleton-based action recognition still face three major limitations: insufficient modeling of view discrepancies, lack of effective adversarial mechanisms, and uncontrollable augmentation perturbations. To tackle these issues, we propose the Multi-view Mini-Max infinite skeleton-data Game Contrastive Learning for skeleton-based action Recognition (M3GCLR), a game-theoretic contrastive framework. First, we establish the Infinite Skeleton-data Game (ISG) model and the ISG equilibrium theorem, and further provide a rigorous proof, enabling mini-max optimization based on multi-view mutual information. Then, we generate normal-extreme data pairs through multi-view rotation augmentation and adopt temporally averaged input as a neutral anchor to achieve structural alignment, thereby explicitly characterizing perturbation strength. Next, leveraging the proposed equilibrium theorem, we construct a strongly adversarial mini-max skeleton-data game to encourage the model to mine richer action-discriminative information. Finally, we introduce the dual-loss equilibrium optimizer to optimize the game equilibrium, allowing the learning process to maximize action-relevant information while minimizing encoding redundancy, and we prove the equivalence between the proposed optimizer and the ISG model. Extensive Experiments show that M3GCLR achieves three-stream 82.1%, 85.8% accuracy on NTU RGB+D 60 (X-Sub, X-View) and 72.3%, 75.0% accuracy on NTU RGB+D 120 (X-Sub, X-Set). On PKU-MMD Part I and II, it attains 89.1%, 45.2% in three-stream respectively, all results matching or outperforming state-of-the-art performance. Ablation studies confirm the effectiveness of each component.

**선정 근거**
골격 기반 동작 인식 기술이 스포츠 동작 분석에 직접적으로 적용 가능하여 선수 기술 분석과 자세 교정에 활용 가능

**활용 인사이트**
다중 뷰 증강과 미니맥스 게임 이론을 활용해 스포츠 장면에서 선수들의 움직임을 정확히 분석하고 하이라이트 장면 자동 추출 가능

## 12위: NS-VLA: Towards Neuro-Symbolic Vision-Language-Action Models

- arXiv: http://arxiv.org/abs/2603.09542v1
- PDF: https://arxiv.org/pdf/2603.09542v1
- 발행일: 2026-03-10
- 카테고리: cs.RO
- 점수: final 88.0 (llm_adjusted:85 = base:82 + bonus:+3)
- 플래그: 코드 공개

**개요**
Vision-Language-Action (VLA) models are formulated to ground instructions in visual context and generate action sequences for robotic manipulation. Despite recent progress, VLA models still face challenges in learning related and reusable primitives, reducing reliance on large-scale data and complex architectures, and enabling exploration beyond demonstrations. To address these challenges, we propose a novel Neuro-Symbolic Vision-Language-Action (NS-VLA) framework via online reinforcement learning (RL). It introduces a symbolic encoder to embedding vision and language features and extract structured primitives, utilizes a symbolic solver for data-efficient action sequencing, and leverages online RL to optimize generation via expansive exploration. Experiments on robotic manipulation benchmarks demonstrate that NS-VLA outperforms previous methods in both one-shot training and data-perturbed settings, while simultaneously exhibiting superior zero-shot generalizability, high data efficiency and expanded exploration space. Our code is available.

**선정 근거**
멀티모달 파싱 기술이 스포츠 영상에서 의미 있는 정보를 추출하고 구조화된 지식으로 변환하여 전략 분석에 활용 가능

**활용 인사이트**
통합된 분류법과 3단계 계층적 분석을 통해 경기 영상에서 객체 이벤트를 정확히 식별하고 전략적 인사이트 도출 가능

## 13위: WikiCLIP: An Efficient Contrastive Baseline for Open-domain Visual Entity Recognition

- arXiv: http://arxiv.org/abs/2603.09921v1
- PDF: https://arxiv.org/pdf/2603.09921v1
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 88.0 (llm_adjusted:85 = base:82 + bonus:+3)
- 플래그: 코드 공개

**개요**
Open-domain visual entity recognition (VER) seeks to associate images with entities in encyclopedic knowledge bases such as Wikipedia. Recent generative methods tailored for VER demonstrate strong performance but incur high computational costs, limiting their scalability and practical deployment. In this work, we revisit the contrastive paradigm for VER and introduce WikiCLIP, a simple yet effective framework that establishes a strong and efficient baseline for open-domain VER. WikiCLIP leverages large language model embeddings as knowledge-rich entity representations and enhances them with a Vision-Guided Knowledge Adaptor (VGKA) that aligns textual semantics with visual cues at the patch level. To further encourage fine-grained discrimination, a Hard Negative Synthesis Mechanism generates visually similar but semantically distinct negatives during training. Experimental results on popular open-domain VER benchmarks, such as OVEN, demonstrate that WikiCLIP significantly outperforms strong baselines. Specifically, WikiCLIP achieves a 16% improvement on the challenging OVEN unseen set, while reducing inference latency by nearly 100 times compared with the leading generative model, AutoVER. The project page is available at https://artanic30.github.io/project_pages/WikiCLIP/

**선정 근거**
효율적인 대비 학습 방식으로 스포츠 장면의 시각적 엔티티 인식에 적용 가능

**활용 인사이트**
100배 빠른 추론 지연 시간으로 실시간 스포츠 장면 분석에 효과적

## 14위: Logics-Parsing-Omni Technical Report

- arXiv: http://arxiv.org/abs/2603.09677v1
- PDF: https://arxiv.org/pdf/2603.09677v1
- 코드: https://github.com/alibaba/Logics-Parsing/tree/master/Logics-Parsing-Omni
- 발행일: 2026-03-10
- 카테고리: cs.AI
- 점수: final 86.4 (llm_adjusted:83 = base:80 + bonus:+3)
- 플래그: 코드 공개

**개요**
Addressing the challenges of fragmented task definitions and the heterogeneity of unstructured data in multimodal parsing, this paper proposes the Omni Parsing framework. This framework establishes a Unified Taxonomy covering documents, images, and audio-visual streams, introducing a progressive parsing paradigm that bridges perception and cognition. Specifically, the framework integrates three hierarchical levels: 1) Holistic Detection, which achieves precise spatial-temporal grounding of objects or events to establish a geometric baseline for perception; 2) Fine-grained Recognition, which performs symbolization (e.g., OCR/ASR) and attribute extraction on localized objects to complete structured entity parsing; and 3) Multi-level Interpreting, which constructs a reasoning chain from local semantics to global logic. A pivotal advantage of this framework is its evidence anchoring mechanism, which enforces a strict alignment between high-level semantic descriptions and low-level facts. This enables ``evidence-based'' logical induction, transforming unstructured signals into standardized knowledge that is locatable, enumerable, and traceable. Building on this foundation, we constructed a standardized dataset and released the Logics-Parsing-Omni model, which successfully converts complex audio-visual signals into machine-readable structured knowledge. Experiments demonstrate that fine-grained perception and high-level cognition are synergistic, effectively enhancing model reliability. Furthermore, to quantitatively evaluate these capabilities, we introduce OmniParsingBench. Code, models and the benchmark are released at https://github.com/alibaba/Logics-Parsing/tree/master/Logics-Parsing-Omni.

**선정 근거**
Multimodal parsing techniques could be applicable for analyzing sports footage and extracting meaningful information.

## 15위: A Multi-Prototype-Guided Federated Knowledge Distillation Approach in AI-RAN Enabled Multi-Access Edge Computing System

- arXiv: http://arxiv.org/abs/2603.09727v1
- PDF: https://arxiv.org/pdf/2603.09727v1
- 발행일: 2026-03-10
- 카테고리: cs.LG
- 점수: final 86.4 (llm_adjusted:83 = base:78 + bonus:+5)
- 플래그: 엣지

**개요**
With the development of wireless network, Multi-Access Edge Computing (MEC) and Artificial Intelligence (AI)-native Radio Access Network (RAN) have attracted significant attention. Particularly, the integration of AI-RAN and MEC is envisioned to transform network efficiency and responsiveness. Therefore, it is valuable to investigate AI-RAN enabled MEC system. Federated learning (FL) nowadays is emerging as a promising approach for AI-RAN enabled MEC system, in which edge devices are enabled to train a global model cooperatively without revealing their raw data. However, conventional FL encounters the challenge in processing the non-independent and identically distributed (non-IID) data. Single prototype obtained by averaging the embedding vectors per class can be employed in FL to handle the data heterogeneity issue. Nevertheless, this may result in the loss of useful information owing to the average operation. Therefore, in this paper, a multi-prototype-guided federated knowledge distillation (MP-FedKD) approach is proposed. Particularly, self-knowledge distillation is integrated into FL to deal with the non-IID issue. To cope with the problem of information loss caused by single prototype-based strategy, multi-prototype strategy is adopted, where we present a conditional hierarchical agglomerative clustering (CHAC) approach and a prototype alignment scheme. Additionally, we design a novel loss function (called LEMGP loss) for each local client, where the relationship between global prototypes and local embedding will be focused. Extensive experiments over multiple datasets with various non-IID settings showcase that the proposed MP-FedKD approach outperforms the considered state-of-the-art baselines regarding accuracy, average accuracy and errors (RMSE and MAE).

**선정 근거**
엣지 컴퓨팅 시스템을 위한 연합 학습 접근법이 AI 카메라 장치에 직접 적용 가능

**활용 인사이트**
다양한 비-IID 설정에서 우수한 정확도를 보여주며 다수의 AI 카메라 장치 간 협업 학습 지원

## 16위: Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture

- arXiv: http://arxiv.org/abs/2603.09681v1
- PDF: https://arxiv.org/pdf/2603.09681v1
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
State-of-the-art methods can recover accurate overall 3D human body motion from in-the-wild videos. However, they often fail to capture fine-grained articulations, especially in the feet, which are critical for applications such as gait analysis and animation. This limitation results from training datasets with inaccurate foot annotations and limited foot motion diversity. We address this gap with FootMR, a Foot Motion Refinement method that refines foot motion estimated by an existing human recovery model through lifting 2D foot keypoint sequences to 3D. By avoiding direct image input, FootMR circumvents inaccurate image-3D annotation pairs and can instead leverage large-scale motion capture data. To resolve ambiguities of 2D-to-3D lifting, FootMR incorporates knee and foot motion as context and predicts only residual foot motion. Generalization to extreme foot poses is further improved by representing joints in global rather than parent-relative rotations and applying extensive data augmentation. To support evaluation of foot motion reconstruction, we introduce MOOF, a 2D dataset of complex foot movements. Experiments on MOOF, MOYO, and RICH show that FootMR outperforms state-of-the-art methods, reducing ankle joint angle error on MOYO by up to 30% over the best video-based approach.

**선정 근거**
멀티모달 인터리브 생성 기술은 스포츠 하이라이트 영상 제작에 적용 가능하여 콘텐츠 생성 효율성을 높일 수 있습니다.

**활용 인사이트**
강화학습 기반의 정책 최적화를 통해 스포츠 장면의 연속적인 이미지와 텍스트를 결합한 하이라이트 영상을 자동으로 생성할 수 있습니다.

## 17위: From Ideal to Real: Stable Video Object Removal under Imperfect Conditions

- arXiv: http://arxiv.org/abs/2603.09283v1
- PDF: https://arxiv.org/pdf/2603.09283v1
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Removing objects from videos remains difficult in the presence of real-world imperfections such as shadows, abrupt motion, and defective masks. Existing diffusion-based video inpainting models often struggle to maintain temporal stability and visual consistency under these challenges. We propose Stable Video Object Removal (SVOR), a robust framework that achieves shadow-free, flicker-free, and mask-defect-tolerant removal through three key designs: (1) Mask Union for Stable Erasure (MUSE), a windowed union strategy applied during temporal mask downsampling to preserve all target regions observed within each window, effectively handling abrupt motion and reducing missed removals; (2) Denoising-Aware Segmentation (DA-Seg), a lightweight segmentation head on a decoupled side branch equipped with Denoising-Aware AdaLN and trained with mask degradation to provide an internal diffusion-aware localization prior without affecting content generation; and (3) Curriculum Two-Stage Training: where Stage I performs self-supervised pretraining on unpaired real-background videos with online random masks to learn realistic background and temporal priors, and Stage II refines on synthetic pairs using mask degradation and side-effect-weighted losses, jointly removing objects and their associated shadows/reflections while improving cross-domain robustness. Extensive experiments show that SVOR attains new state-of-the-art results across multiple datasets and degraded-mask benchmarks, advancing video object removal from ideal settings toward real-world applications.

**선정 근거**
비디오 객체 제거 기술은 스포츠 경기 영상 편집에 적용 가능하여 원치 않는 객체를 제거하거나 특정 선수를 분리하는 데 활용될 수 있습니다.

**활용 인사이트**
그림자나 급격한 움직임 등 불완전한 조건에서도 안정적으로 객체를 제거하여 경기 영상을 정리하고 하이라이트 영상 제작 시 특정 선수나 공만 강조할 수 있습니다.

## 18위: When to Lock Attention: Training-Free KV Control in Video Diffusion

- arXiv: http://arxiv.org/abs/2603.09657v1
- PDF: https://arxiv.org/pdf/2603.09657v1
- 발행일: 2026-03-10
- 카테고리: cs.CV, cs.AI, cs.ET, eess.IV
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Maintaining background consistency while enhancing foreground quality remains a core challenge in video editing. Injecting full-image information often leads to background artifacts, whereas rigid background locking severely constrains the model's capacity for foreground generation. To address this issue, we propose KV-Lock, a training-free framework tailored for DiT-based video diffusion models. Our core insight is that the hallucination metric (variance of denoising prediction) directly quantifies generation diversity, which is inherently linked to the classifier-free guidance (CFG) scale. Building upon this, KV-Lock leverages diffusion hallucination detection to dynamically schedule two key components: the fusion ratio between cached background key-values (KVs) and newly generated KVs, and the CFG scale. When hallucination risk is detected, KV-Lock strengthens background KV locking and simultaneously amplifies conditional guidance for foreground generation, thereby mitigating artifacts and improving generation fidelity. As a training-free, plug-and-play module, KV-Lock can be easily integrated into any pre-trained DiT-based models. Extensive experiments validate that our method outperforms existing approaches in improved foreground quality with high background fidelity across various video editing tasks.

**선정 근거**
비디오 확산 모델을 이용한 영상 편집 기술은 스포츠 영상 보정에 활용 가능하여 배경과 전경을 동시에 개선할 수 있습니다.

**활용 인사이트**
선수나 공과 같은 전경 요소를 선명하게 하면서 배경의 일관성을 유지하여 스포츠 하이라이트 영상의 시각적 품질을 향상시킬 수 있습니다.

## 19위: Fine-grained Motion Retrieval via Joint-Angle Motion Images and Token-Patch Late Interaction

- arXiv: http://arxiv.org/abs/2603.09930v1
- PDF: https://arxiv.org/pdf/2603.09930v1
- 발행일: 2026-03-10
- 카테고리: cs.CV, cs.IR
- 점수: final 84.8 (llm_adjusted:81 = base:78 + bonus:+3)
- 플래그: 코드 공개

**개요**
Text-motion retrieval aims to learn a semantically aligned latent space between natural language descriptions and 3D human motion skeleton sequences, enabling bidirectional search across the two modalities. Most existing methods use a dual-encoder framework that compresses motion and text into global embeddings, discarding fine-grained local correspondences, and thus reducing accuracy. Additionally, these global-embedding methods offer limited interpretability of the retrieval results. To overcome these limitations, we propose an interpretable, joint-angle-based motion representation that maps joint-level local features into a structured pseudo-image, compatible with pre-trained Vision Transformers. For text-to-motion retrieval, we employ MaxSim, a token-wise late interaction mechanism, and enhance it with Masked Language Modeling regularization to foster robust, interpretable text-motion alignment. Extensive experiments on HumanML3D and KIT-ML show that our method outperforms state-of-the-art text-motion retrieval approaches while offering interpretable fine-grained correspondences between text and motion. The code is available in the supplementary material.

**선정 근거**
관절 각도 기반 움직임 검색 기술은 스포츠 하이라이트 생성에 적용 가능하여 특정 동작을 기반으로 영상을 검색하고 편집할 수 있습니다.

**활용 인사이트**
텍스트 기반으로 특정 동작을 검색하여 해당 동작이 포함된 영상 장면을 자동으로 추출하고 하이라이트 영상을 생성하는 데 활용할 수 있습니다.

## 20위: Towards Unified Multimodal Interleaved Generation via Group Relative Policy Optimization

- arXiv: http://arxiv.org/abs/2603.09538v1
- PDF: https://arxiv.org/pdf/2603.09538v1
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Unified vision-language models have made significant progress in multimodal understanding and generation, yet they largely fall short in producing multimodal interleaved outputs, which is a crucial capability for tasks like visual storytelling and step-by-step visual reasoning. In this work, we propose a reinforcement learning-based post-training strategy to unlock this capability in existing unified models, without relying on large-scale multimodal interleaved datasets. We begin with a warm-up stage using a hybrid dataset comprising curated interleaved sequences and limited data for multimodal understanding and text-to-image generation, which exposes the model to interleaved generation patterns while preserving its pretrained capabilities. To further refine interleaved generation, we propose a unified policy optimization framework that extends Group Relative Policy Optimization (GRPO) to the multimodal setting. Our approach jointly models text and image generation within a single decoding trajectory and optimizes it with our novel hybrid rewards covering textual relevance, visual-text alignment, and structural fidelity. Additionally, we incorporate process-level rewards to provide step-wise guidance, enhancing training efficiency in complex multimodal tasks. Experiments on MMIE and InterleavedBench demonstrate that our approach significantly enhances the quality and coherence of multimodal interleaved generation.

**선정 근거**
Multimodal generation techniques could be applicable for creating highlight reels from sports footage.

## 21위: A Text-Native Interface for Generative Video Authoring

- arXiv: http://arxiv.org/abs/2603.09072v1
- PDF: https://arxiv.org/pdf/2603.09072v1
- 발행일: 2026-03-10
- 카테고리: cs.HC, cs.AI
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Everyone can write their stories in freeform text format -- it's something we all learn in school. Yet storytelling via video requires one to learn specialized and complicated tools. In this paper, we introduce Doki, a text-native interface for generative video authoring, aligning video creation with the natural process of text writing. In Doki, writing text is the primary interaction: within a single document, users define assets, structure scenes, create shots, refine edits, and add audio. We articulate the design principles of this text-first approach and demonstrate Doki's capabilities through a series of examples. To evaluate its real-world use, we conducted a week-long deployment study with participants of varying expertise in video authoring. This work contributes a fundamental shift in generative video interfaces, demonstrating a powerful and accessible new way to craft visual stories.

**선정 근거**
비디오 제작 인터페이스 기술로 하이라이트 편집과 관련 있으나 스포츠 자동 촬영에 직접적으로 연관되지 않음

## 22위: DCAU-Net: Differential Cross Attention and Channel-Spatial Feature Fusion for Medical Image Segmentation

- arXiv: http://arxiv.org/abs/2603.09530v1
- PDF: https://arxiv.org/pdf/2603.09530v1
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Accurate medical image segmentation requires effective modeling of both long-range dependencies and fine-grained boundary details. While transformers mitigate the issue of insufficient semantic information arising from the limited receptive field inherent in convolutional neural networks, they introduce new challenges: standard self-attention incurs quadratic computational complexity and often assigns non-negligible attention weights to irrelevant regions, diluting focus on discriminative structures and ultimately compromising segmentation accuracy. Existing attention variants, although effective in reducing computational complexity, fail to suppress redundant computation and inadvertently impair global context modeling. Furthermore, conventional fusion strategies in encoder-decoder architectures, typically based on simple concatenation or summation, can not adaptively integrate high-level semantic information with low-level spatial details. To address these limitations, we propose DCAU-Net, a novel yet efficient segmentation framework with two key ideas. First, a new Differential Cross Attention (DCA) is designed to compute the difference between two independent softmax attention maps to adaptively highlight discriminative structures. By replacing pixel-wise key and value tokens with window-level summary tokens, DCA dramatically reduces computational complexity without sacrificing precision. Second, a Channel-Spatial Feature Fusion (CSFF) strategy is introduced to adaptively recalibrate features from skip connections and up-sampling paths through using sequential channel and spatial attention, effectively suppressing redundant information and amplifying salient cues. Experiments on two public benchmarks demonstrate that DCAU-Net achieves competitive performance with enhanced segmentation accuracy and robustness.

**선정 근거**
DCAU-Net의 차이 크로스 어텐션과 채널-공간 특징 융합 기술은 스포츠 장면에서 선수와 객체를 식별하고 중요한 순간을 분석하는 데 직접적으로 적용 가능하며, rk3588 엣지 디바이스에서 효율적으로 실행될 수 있는 계산 효율성 개선 방법을 제공합니다.

**활용 인사이트**
DCAU-Net의 윈도우 레벨 요약 토큰을 활용한 차이 크로스 어텐션은 스포츠 장면에서 판단력 있는 구조를 자동으로 강조하여 하이라이트 장면을 식별하고, 채널-공간 특징 융합은 다양한 수준의 시각 정보를 효과적으로 결합하여 동작 분석 정확도를 향상시킬 수 있습니다.

## 23위: Evolving Prompt Adaptation for Vision-Language Models

- arXiv: http://arxiv.org/abs/2603.09493v1
- PDF: https://arxiv.org/pdf/2603.09493v1
- 발행일: 2026-03-10
- 카테고리: cs.CV, cs.AI
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
The adaptation of large-scale vision-language models (VLMs) to downstream tasks with limited labeled data remains a significant challenge. While parameter-efficient prompt learning methods offer a promising path, they often suffer from catastrophic forgetting of pre-trained knowledge. Toward addressing this limitation, our work is grounded in the insight that governing the evolutionary path of prompts is essential for forgetting-free adaptation. To this end, we propose EvoPrompt, a novel framework designed to explicitly steer the prompt trajectory for stable, knowledge-preserving fine-tuning. Specifically, our approach employs a Modality-Shared Prompt Projector (MPP) to generate hierarchical prompts from a unified embedding space. Critically, an evolutionary training strategy decouples low-rank updates into directional and magnitude components, preserving early-learned semantic directions while only adapting their magnitude, thus enabling prompts to evolve without discarding foundational knowledge. This process is further stabilized by Feature Geometric Regularization (FGR), which enforces feature decorrelation to prevent representation collapse. Extensive experiments demonstrate that EvoPrompt achieves state-of-the-art performance in few-shot learning while robustly preserving the original zero-shot capabilities of pre-trained VLMs.

**선정 근거**
Presents vision-language model adaptation techniques applicable to sports analysis but doesn't address edge computing

## 24위: M2Diff: Multi-Modality Multi-Task Enhanced Diffusion Model for MRI-Guided Low-Dose PET Enhancement

- arXiv: http://arxiv.org/abs/2603.09075v1
- PDF: https://arxiv.org/pdf/2603.09075v1
- 발행일: 2026-03-10
- 카테고리: eess.IV
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
Positron emission tomography (PET) scans expose patients to radiation, which can be mitigated by reducing the dose, albeit at the cost of diminished quality. This makes low-dose (LD) PET recovery an active research area. Previous studies have focused on standard-dose (SD) PET recovery from LD PET scans and/or multi-modal scans, e.g., PET/CT or PET/MRI, using deep learning. While these studies incorporate multi-modal information through conditioning in a single-task model, such approaches may limit the capacity to extract modality-specific features, potentially leading to early feature dilution. Although recent studies have begun incorporating pathology-rich data, challenges remain in effectively leveraging multi-modality inputs for reconstructing diverse features, particularly in heterogeneous patient populations. To address these limitations, we introduce a multi-modality multi-task diffusion model (M2Diff) that processes MRI and LD PET scans separately to learn modality-specific features and fuse them via hierarchical feature fusion to reconstruct SD PET. This design enables effective integration of complementary structural and functional information, leading to improved reconstruction fidelity. We have validated the effectiveness of our model on both healthy and Alzheimer's disease brain datasets. The M2Diff achieves superior qualitative and quantitative performance on both datasets.

**선정 근거**
다중 모달리티 이미지 처리 기술이 스포츠 영상 향상에 활용 가능

## 25위: RiO-DETR: DETR for Real-time Oriented Object Detection

- arXiv: http://arxiv.org/abs/2603.09411v1
- PDF: https://arxiv.org/pdf/2603.09411v1
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 78.4 (llm_adjusted:73 = base:65 + bonus:+8)
- 플래그: 실시간, 코드 공개

**개요**
We present RiO-DETR: DETR for Real-time Oriented Object Detection, the first real-time oriented detection transformer to the best of our knowledge. Adapting DETR to oriented bounding boxes (OBBs) poses three challenges: semantics-dependent orientation, angle periodicity that breaks standard Euclidean refinement, and an enlarged search space that slows convergence. RiO-DETR resolves these issues with task-native designs while preserving real-time efficiency. First, we propose Content-Driven Angle Estimation by decoupling angle from positional queries, together with Rotation-Rectified Orthogonal Attention to capture complementary cues for reliable orientation. Second, Decoupled Periodic Refinement combines bounded coarse-to-fine updates with a Shortest-Path Periodic Loss for stable learning across angular seams. Third, Oriented Dense O2O injects angular diversity into dense supervision to speed up angle convergence at no extra cost. Extensive experiments on DOTA-1.0, DIOR-R, and FAIR-1M-2.0 demonstrate RiO-DETR establishes a new speed--accuracy trade-off for real-time oriented detection. Code will be made publicly available.

**선정 근거**
실시간 방향 객체 검출 기술은 스포츠 장면에서 선수와 객체를 추적하고 분석하는 데 필수적입니다. 이 기술은 자동 촬영 시스템의 핵심으로 작용하여 중요한 순간을 놓치지 않고 캡처할 수 있게 합니다.

**활용 인사이트**
RiO-DETR을 edge 디바이스에 통합하여 실시간으로 선수 위치와 움직임을 추적하고, 이를 바탕으로 자동으로 중요한 순간을 촬영하고 하이라이트를 생성할 수 있습니다. 또한 경기 전략 분석에도 활용할 수 있습니다.

## 26위: PPO-Based Hybrid Optimization for RIS-Assisted Semantic Vehicular Edge Computing

- arXiv: http://arxiv.org/abs/2603.09082v1
- PDF: https://arxiv.org/pdf/2603.09082v1
- 코드: https://github.com/qiongwu86/PPO-Based-Hybrid-Optimization-for-RIS-Assisted-Semantic-Vehicular-Edge-Computing
- 발행일: 2026-03-10
- 카테고리: cs.LG, cs.NI
- 점수: final 78.4 (llm_adjusted:73 = base:60 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
To support latency-sensitive Internet of Vehicles (IoV) applications amidst dynamic environments and intermittent links, this paper proposes a Reconfigurable Intelligent Surface (RIS)-aided semantic-aware Vehicle Edge Computing (VEC) framework. This approach integrates RIS to optimize wireless connectivity and semantic communication to minimize latency by transmitting semantic features. We formulate a comprehensive joint optimization problem by optimizing offloading ratios, the number of semantic symbols, and RIS phase shifts. Considering the problem's high dimensionality and non-convexity, we propose a two-tier hybrid scheme that employs Proximal Policy Optimization (PPO) for discrete decision-making and Linear Programming (LP) for offloading optimization. {The simulation results have validated the proposed framework's superiority over existing methods. Specifically, the proposed PPO-based hybrid optimization scheme reduces the average end-to-end latency by approximately 40% to 50% compared to Genetic Algorithm (GA) and Quantum-behaved Particle Swarm Optimization (QPSO). Moreover, the system demonstrates strong scalability by maintaining low latency even in congested scenarios with up to 30 vehicles.

**선정 근거**
차량 통신에 특화된 엣지 컴퓨팅 최적화 기술로 스포츠 영상 처리에 간접적 적용 가능하나 직접적인 연관성은 낮음

**활용 인사이트**
지연 시간 40-50% 감소 기술을 스포츠 실시간 영상 처리에 적용하면 경기 분석 및 하이라이트 생성 속도 향상 가능

## 27위: Reviving ConvNeXt for Efficient Convolutional Diffusion Models

- arXiv: http://arxiv.org/abs/2603.09408v1
- PDF: https://arxiv.org/pdf/2603.09408v1
- 코드: https://github.com/star-kwon/FCDM
- 발행일: 2026-03-10
- 카테고리: cs.CV, cs.AI, cs.LG
- 점수: final 78.4 (llm_adjusted:73 = base:70 + bonus:+3)
- 플래그: 코드 공개

**개요**
Recent diffusion models increasingly favor Transformer backbones, motivated by the remarkable scalability of fully attentional architectures. Yet the locality bias, parameter efficiency, and hardware friendliness--the attributes that established ConvNets as the efficient vision backbone--have seen limited exploration in modern generative modeling. Here we introduce the fully convolutional diffusion model (FCDM), a model having a backbone similar to ConvNeXt, but designed for conditional diffusion modeling. We find that using only 50% of the FLOPs of DiT-XL/2, FCDM-XL achieves competitive performance with 7$\times$ and 7.5$\times$ fewer training steps at 256$\times$256 and 512$\times$512 resolutions, respectively. Remarkably, FCDM-XL can be trained on a 4-GPU system, highlighting the exceptional training efficiency of our architecture. Our results demonstrate that modern convolutional designs provide a competitive and highly efficient alternative for scaling diffusion models, reviving ConvNeXt as a simple yet powerful building block for efficient generative modeling.

**선정 근거**
효율적인 컨볼루션 확산 모델로 스포츠 영상/이미지 생성에 직접 적용 가능하며 학습 효율성이 높음

**활용 인사이트**
FCDM-XL 모델을 활용해 7배 적은 학습 단계로 고품질 스포츠 콘텐츠 생성 가능하며 4-GPU 시스템으로 효율적 훈련

## 28위: HelixTrack: Event-Based Tracking and RPM Estimation of Propeller-like Objects

- arXiv: http://arxiv.org/abs/2603.09235v1
- PDF: https://arxiv.org/pdf/2603.09235v1
- 발행일: 2026-03-10
- 카테고리: cs.CV
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Safety-critical perception for unmanned aerial vehicles and rotating machinery requires microsecond-latency tracking of fast, periodic motion under egomotion and strong distractors. Frame-based and event-based trackers drift or break on propellers because periodic signatures violate their smooth-motion assumptions. We tackle this gap with HelixTrack, a fully event-driven method that jointly tracks propeller-like objects and estimates their rotations per minute (RPM). Incoming events are back-warped from the image plane into the rotor plane via a homography estimated on the fly. A Kalman Filter maintains instantaneous estimates of phase. Batched iterative updates refine the object pose by coupling phase residuals to geometry. To our knowledge, no public dataset targets joint tracking and RPM estimation of propeller-like objects. We therefore introduce the Timestamped Quadcopter with Egomotion (TQE) dataset with 13 high-resolution event sequences, containing 52 rotating objects in total, captured at distances of 2 m / 4 m, with increasing egomotion and microsecond RPM ground truth. On TQE, HelixTrack processes full-rate events (approx. 11.8x real time) faster than real time and microsecond latency. It consistently outperforms per-event and aggregation-based baselines adapted for RPM estimation.

**선정 근거**
프로펠러 같은 특정 객체에만 적용되는 빠른 움직임 추적 기술로 스포츠와는 간접적 관계

**활용 인사이트**
마이크로초 지연 추적 기술을 스포츠 장면에 적용하면 빠른 동작 분석이 가능하나 모델 수정 필요

## 29위: Cutting the Cord: System Architecture for Low-Cost, GPU-Accelerated Bimanual Mobile Manipulation

- arXiv: http://arxiv.org/abs/2603.09051v1
- PDF: https://arxiv.org/pdf/2603.09051v1
- 발행일: 2026-03-10
- 카테고리: cs.RO
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
We present a bimanual mobile manipulator built on the open-source XLeRobot with integrated onboard compute for less than \$1300. Key contributions include: (1) optimized mechanical design maximizing stiffness-to-weight ratio, (2) a Tri-Bus power topology isolating compute from motor-induced voltage transients, and (3) embedded autonomy using NVIDIA Jetson Orin Nano for untethered operation. The platform enables teleoperation, autonomous SLAM navigation, and vision-based manipulation without external dependencies, providing a low-cost alternative for research and education in robotics and robot learning.

**선정 근거**
rk3588와 유사한 엣지 컴퓨팅 하드웨어이나 로봇 조작에 특화되어 스포츠 촬영과는 직접적 연관성 부족

**활용 인사이트**
저비용 모바일 조작자 아키텍처에서 영향력 있는 부분은 스포츠 촬영 장비의 무선 연결 및 자율 운영 시스템에 참조 가능

## 30위: A Fast Solver for Interpolating Stochastic Differential Equation Diffusion Models for Speech Restoration

- arXiv: http://arxiv.org/abs/2603.09508v1
- PDF: https://arxiv.org/pdf/2603.09508v1
- 발행일: 2026-03-10
- 카테고리: eess.AS
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Diffusion Probabilistic Models (DPMs) are a well-established class of diffusion models for unconditional image generation, while SGMSE+ is a well-established conditional diffusion model for speech enhancement. One of the downsides of diffusion models is that solving the reverse process requires many evaluations of a large Neural Network. Although advanced fast sampling solvers have been developed for DPMs, they are not directly applicable to models such as SGMSE+ due to differences in their diffusion processes. Specifically, DPMs transform between the data distribution and a standard Gaussian distribution, whereas SGMSE+ interpolates between the target distribution and a noisy observation. This work first develops a formalism of interpolating Stochastic Differential Equations (iSDEs) that includes SGMSE+, and second proposes a solver for iSDEs. The proposed solver enables fast sampling with as few as 10 Neural Network evaluations across multiple speech restoration tasks.

**선정 근거**
음성 복원에 초점을 둔 확산 모델로 스포츠 영상 처리에는 직접 적용 불가능

**활용 인사이트**
신경망 평가 횟수를 10회로 줄이는 빠른 샘플링 솔버 기술은 실시간 스포츠 영상 처리에 확장 가능성 있음

## 31위: SPAR-K: Scheduled Periodic Alternating Early Exit for Spoken Language Models

- arXiv: http://arxiv.org/abs/2603.09215v1 | 2026-03-10 | final 76.0

Interleaved spoken language models (SLMs) alternately generate text and speech tokens, but decoding at full transformer depth for every step becomes costly, especially due to long speech sequences. We propose SPAR-K, a modality-aware early exit framework designed to accelerate interleaved SLM inference while preserving perceptual quality.

-> 언어 모델 최적화 기술이 엣지 디바이스의 영상 처리 모델에 간접적으로 적용 가능

## 32위: ENIGMA-360: An Ego-Exo Dataset for Human Behavior Understanding in Industrial Scenarios

- arXiv: http://arxiv.org/abs/2603.09741v1 | 2026-03-10 | final 74.4

Understanding human behavior from complementary egocentric (ego) and exocentric (exo) points of view enables the development of systems that can support workers in industrial environments and enhance their safety. However, progress in this area is hindered by the lack of datasets capturing both views in realistic industrial scenarios.

-> 인간 행동 이해 데이터셋이지만 산업 시나리오에 초점을 맞춰 스포츠 분석과는 간접적 관계

## 33위: Intelligent Spatial Estimation for Fire Hazards in Engineering Sites: An Enhanced YOLOv8-Powered Proximity Analysis Framework

- arXiv: http://arxiv.org/abs/2603.09069v1 | 2026-03-10 | final 74.4

This study proposes an enhanced dual-model YOLOv8 framework for intelligent fire detection and proximity-aware risk assessment, extending conventional vision-based monitoring beyond simple detection to actionable hazard prioritization. The system is trained on a dataset of 9,860 annotated images to segment fire and smoke across complex environments.

-> 공간 분석을 위한 컴퓨터 비전 프레임워크는 스포츠 경기장 모니터링으로 적용될 수 있습니다

## 34위: Embodied Human Simulation for Quantitative Design and Analysis of Interactive Robotics

- arXiv: http://arxiv.org/abs/2603.09218v1 | 2026-03-10 | final 72.0

Physical interactive robotics, ranging from wearable devices to collaborative humanoid robots, require close coordination between mechanical design and control. However, evaluating interactive dynamics is challenging due to complex human biomechanics and motor responses.

-> 인체 운동 시뮬레이션 기술이 동작 분석에 적용 가능하나 로봇 설계에 중점을 둠

## 35위: Training-free Motion Factorization for Compositional Video Generation

- arXiv: http://arxiv.org/abs/2603.09104v1 | 2026-03-10 | final 70.4

Compositional video generation aims to synthesize multiple instances with diverse appearance and motion, which is widely applicable in real-world scenarios. However, current approaches mainly focus on binding semantics, neglecting to understand diverse motion categories specified in prompts.

-> 비디오 생성 기술이 있지만 실제 스포츠 영상 편집보다는 합성 영상 생성에 초점을 맞춤

## 36위: DISPLAY: Directable Human-Object Interaction Video Generation via Sparse Motion Guidance and Multi-Task Auxiliary

- arXiv: http://arxiv.org/abs/2603.09883v1 | 2026-03-10 | final 70.4

Human-centric video generation has advanced rapidly, yet existing methods struggle to produce controllable and physically consistent Human-Object Interaction (HOI) videos. Existing works rely on dense control signals, template videos, or carefully crafted text prompts, which limit flexibility and generalization to novel objects.

-> 인간-객체 상호작용 비디오 생성 기술이 있지만 실제 스포츠 영상 분석보다는 합성에 초점을 맞춤

## 37위: Beyond Scaling: Assessing Strategic Reasoning and Rapid Decision-Making Capability of LLMs in Zero-sum Environments

- arXiv: http://arxiv.org/abs/2603.09337v1 | 2026-03-10 | final 70.4

Large Language Models (LLMs) have achieved strong performance on static reasoning benchmarks, yet their effectiveness as interactive agents operating in adversarial, time-sensitive environments remains poorly understood. Existing evaluations largely treat reasoning as a single-shot capability, overlooking the challenges of opponent-aware decision-making, temporal constraints, and execution under pressure.

-> 전략적 추론 평가 프레임워크가 경기 전략 분석에 간접적으로 참고 가능하나 영상 처리와 직접 연관성 약함

## 38위: POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction with Application to Strong Lens Discovery

- arXiv: http://arxiv.org/abs/2603.09162v1 | 2026-03-10 | final 70.4

Radio interferometry enables high-resolution imaging of astronomical radio sources by synthesizing a large effective aperture from an array of antennas and solving a deconvolution problem to reconstruct the image. Deep learning has emerged as a promising solution to the imaging problem, reducing computational costs and enabling super-resolution.

-> 이미지 처리 기술이 스포츠 영상 처리와 간접적으로 관련 있음

## 39위: 3D UAV Trajectory Estimation and Classification from Internet Videos via Language Model

- arXiv: http://arxiv.org/abs/2603.09070v1 | 2026-03-10 | final 68.8

Reliable 3D trajectory estimation of unmanned aerial vehicles (UAVs) is a fundamental requirement for anti-UAV systems, yet the acquisition of large-scale and accurately annotated trajectory data remains prohibitively expensive. In this work, we present a novel framework that derives UAV 3D trajectories and category information directly from Internet-scale UAV videos, without relying on manual annotations.

-> 비행체 궤적 추론 기술은 스포츠 선수의 움직임 분석에 간접적으로 적용 가능

## 40위: Component-Aware Sketch-to-Image Generation Using Self-Attention Encoding and Coordinate-Preserving Fusion

- arXiv: http://arxiv.org/abs/2603.09484v1 | 2026-03-10 | final 68.0

Translating freehand sketches into photorealistic images remains a fundamental challenge in image synthesis, particularly due to the abstract, sparse, and stylistically diverse nature of sketches. Existing approaches, including GAN-based and diffusion-based models, often struggle to reconstruct fine-grained details, maintain spatial alignment, or adapt across different sketch domains.

-> 스케치에서 이미지 생성 기술은 영상 보정에 간접적으로 활용 가능할 수 있음

## 41위: From Perception to Cognition: How Latency Affects Interaction Fluency and Social Presence in VR Conferencing

- arXiv: http://arxiv.org/abs/2603.09261v1 | 2026-03-10 | final 68.0

Virtual reality (VR) conferencing has the potential to provide geographically dispersed users with an immersive environment, enabling rich social interactions and user experience using avatars. However, remote communication in VR inevitably introduces end-to-end (E2E) latency, which can significantly impact user experience.

-> 실시간 상호작용 지연 시간 분석이 스포츠 실시간 분석과 간접적으로 관련 있음

## 42위: A Short Survey of Averaging Techniques in Stochastic Gradient Methods

- arXiv: http://arxiv.org/abs/2603.09634v1 | 2026-03-10 | final 68.0

Stochastic gradient methods are among the most widely used algorithms for large-scale optimization and machine learning. A key technique for improving the statistical efficiency and stability of these methods is the use of averaging schemes applied to the sequence of iterates generated during optimization.

-> Survey of optimization techniques for machine learning training foundational but not directly related to specific applications

## 43위: The Virtuous Cycle: AI-Powered Vector Search and Vector Search-Augmented AI

- arXiv: http://arxiv.org/abs/2603.09347v1 | 2026-03-10 | final 68.0

Modern AI and vector search are rapidly converging, forming a promising research frontier in intelligent information systems. On one hand, advances in AI have substantially improved the semantic accuracy and efficiency of vector search, including learned indexing structures, adaptive pruning strategies, and automated parameter tuning.

-> AI and vector search intersection could be used in broader system but not directly related to computer vision for sports analysis

## 44위: OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models

- arXiv: http://arxiv.org/abs/2603.09326v1 | 2026-03-10 | final 66.4

Multimodal large language models (MLLMs) have achieved remarkable performance across a wide range of vision language tasks. However, their ability in low-level visual perception, particularly in detecting fine-grained visual discrepancies, remains underexplored and lacks systematic analysis.

-> 미세한 시지각 기술은 상세한 스포츠 분석에 부분적으로 적용될 수 있습니다

## 45위: GeoSolver: Scaling Test-Time Reasoning in Remote Sensing with Fine-Grained Process Supervision

- arXiv: http://arxiv.org/abs/2603.09551v1 | 2026-03-10 | final 66.4

While Vision-Language Models (VLMs) have significantly advanced remote sensing interpretation, enabling them to perform complex, step-by-step reasoning remains highly challenging. Recent efforts to introduce Chain-of-Thought (CoT) reasoning to this domain have shown promise, yet ensuring the visual faithfulness of these intermediate steps remains a critical bottleneck.

-> GeoSolver's reasoning approach could be adapted for sports strategy analysis but is designed for remote sensing.

## 46위: UniField: A Unified Field-Aware MRI Enhancement Framework

- arXiv: http://arxiv.org/abs/2603.09223v1 | 2026-03-10 | final 66.4

Magnetic Resonance Imaging (MRI) field-strength enhancement holds immense value for both clinical diagnostics and advanced research. However, existing methods typically focus on isolated enhancement tasks, such as specific 64mT-to-3T or 3T-to-7T transitions using limited subject cohorts, thereby failing to exploit the shared degradation patterns inherent across different field strengths and severely restricting model generalization.

-> 영상 보정 기술이 스포츠 영상 처리와 간접적으로 관련 있음

## 47위: WS-Net: Weak-Signal Representation Learning and Gated Abundance Reconstruction for Hyperspectral Unmixing via State-Space and Weak Signal Attention Fusion

- arXiv: http://arxiv.org/abs/2603.09037v1 | 2026-03-10 | final 66.0

Weak spectral responses in hyperspectral images are often obscured by dominant endmembers and sensor noise, resulting in inaccurate abundance estimation. This paper introduces WS-Net, a deep unmixing framework specifically designed to address weak-signal collapse through state-space modelling and Weak Signal Attention fusion.

-> Image processing techniques with attention mechanisms could be applicable but specialized for hyperspectral imagery not sports video

## 48위: LAP: A Language-Aware Planning Model For Procedure Planning In Instructional Videos

- arXiv: http://arxiv.org/abs/2603.09743v1 | 2026-03-10 | final 64.0

Procedure planning requires a model to predict a sequence of actions that transform a start visual observation into a goal in instructional videos. While most existing methods rely primarily on visual observations as input, they often struggle with the inherent ambiguity where different actions can appear visually similar.

-> 언어 인식 절차 계획 모델은 스포츠 전략 분석에 간접적으로 적용 가능

## 49위: DFPF-Net: Dynamically Focused Progressive Fusion Network for Remote Sensing Change Detection

- arXiv: http://arxiv.org/abs/2603.09106v1 | 2026-03-10 | final 64.0

Change detection (CD) has extensive applications and is a crucial method for identifying and localizing target changes. In recent years, various CD methods represented by convolutional neural network (CNN) and transformer have achieved significant success in effectively detecting difference areas in bi-temporal remote sensing images.

-> Change detection techniques could potentially be adapted for identifying key moments in sports footage.

## 50위: VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM

- arXiv: http://arxiv.org/abs/2603.09673v1 | 2026-03-10 | final 64.0

Simultaneous Localization and Mapping (SLAM) with 3D Gaussian Splatting (3DGS) enables fast, differentiable rendering and high-fidelity reconstruction across diverse real-world scenes. However, existing 3DGS-SLAM approaches handle measurement reliability implicitly, making pose estimation and global alignment susceptible to drift in low-texture regions, transparent surfaces, or areas with complex reflectance properties.

-> VarSplat's 3D reconstruction and uncertainty modeling could be applied to sports scene analysis.

## 51위: Reading the Mood Behind Words: Integrating Prosody-Derived Emotional Context into Socially Responsive VR Agents

- arXiv: http://arxiv.org/abs/2603.09324v1 | 2026-03-10 | final 64.0

In VR interactions with embodied conversational agents, users' emotional intent is often conveyed more by how something is said than by what is said. However, most VR agent pipelines rely on speech-to-text processing, discarding prosodic cues and often producing emotionally incongruent responses despite correct semantics.

-> Real-time emotion recognition techniques that could be applied to sports analysis

## 52위: Evidential Perfusion Physics-Informed Neural Networks with Residual Uncertainty Quantification

- arXiv: http://arxiv.org/abs/2603.09359v1 | 2026-03-10 | final 64.0

Physics-informed neural networks (PINNs) have shown promise in addressing the ill-posed deconvolution problem in computed tomography perfusion (CTP) imaging for acute ischemic stroke assessment. However, existing PINN-based approaches remain deterministic and do not quantify uncertainty associated with violations of physics constraints, limiting reliability assessment.

-> 신경망 이미지 처리 기술은 관련 있으나 의료 영상에 특화되어 있어 스포츠 분석과 간접적 관련

## 53위: Robust Provably Secure Image Steganography via Latent Iterative Optimization

- arXiv: http://arxiv.org/abs/2603.09348v1 | 2026-03-10 | final 64.0

We propose a robust and provably secure image steganography framework based on latent-space iterative optimization. Within this framework, the receiver treats the transmitted image as a fixed reference and iteratively refines a latent variable to minimize the reconstruction error, thereby improving message extraction accuracy.

-> 이미지 처리 기술이 스포츠 영상 분석에 간접적으로 적용 가능

## 54위: Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging

- arXiv: http://arxiv.org/abs/2603.09137v1 | 2026-03-10 | final 60.0

Osteoporosis is a skeletal disease typically diagnosed using dual-energy X-ray absorptiometry (DXA), which quantifies areal bone mineral density but overlooks bone microarchitecture and surrounding soft tissues. High-resolution peripheral quantitative computed tomography (HR-pQCT) enables three-dimensional microstructural imaging with minimal radiation.

-> Transformer-based segmentation techniques that could be adapted for sports video analysis

## 55위: AutoAgent: Evolving Cognition and Elastic Memory Orchestration for Adaptive Agents

- arXiv: http://arxiv.org/abs/2603.09716v1 | 2026-03-10 | final 60.0

Autonomous agent frameworks still struggle to reconcile long-term experiential learning with real-time, context-sensitive decision-making. In practice, this gap appears as static cognition, rigid workflow dependence, and inefficient context usage, which jointly limit adaptability in open-ended and non-stationary environments.

-> Autonomous agent frameworks that could inform the development of adaptive sports camera systems

## 56위: Unsupervised Domain Adaptation with Target-Only Margin Disparity Discrepancy

- arXiv: http://arxiv.org/abs/2603.09932v1 | 2026-03-10 | final 60.0

In interventional radiology, Cone-Beam Computed Tomography (CBCT) is a helpful imaging modality that provides guidance to practicians during minimally invasive procedures. CBCT differs from traditional Computed Tomography (CT) due to its limited reconstructed field of view, specific artefacts, and the intra-arterial administration of contrast medium.

-> 도메인 적용 기술이 스포츠 분석 모델 적용과 간접적으로 관련 있음

## 57위: TRIP-Bag: A Portable Teleoperation System for Plug-and-Play Robotic Arms and Leaders

- arXiv: http://arxiv.org/abs/2603.09226v1 | 2026-03-10 | final 56.0

Large scale, diverse demonstration data for manipulation tasks remains a major challenge in learning-based robot policies. Existing in-the-wild data collection approaches often rely on vision-based pose estimation of hand-held grippers or gloves, which introduces an embodiment gap between the collection platform and the target robot.

-> Portable teleoperation system has limited relevance to the sports filming edge device project.

## 58위: Reward-Zero: Language Embedding Driven Implicit Reward Mechanisms for Reinforcement Learning

- arXiv: http://arxiv.org/abs/2603.09331v1 | 2026-03-10 | final 54.4

We introduce Reward-Zero, a general-purpose implicit reward mechanism that transforms natural-language task descriptions into dense, semantically grounded progress signals for reinforcement learning (RL). Reward-Zero serves as a simple yet sophisticated universal reward function that leverages language embeddings for efficient RL training.

-> About reinforcement learning with language embeddings, only tangentially related to sports strategy analysis

## 59위: IntroSVG: Learning from Rendering Feedback for Text-to-SVG Generation via an Introspective Generator-Critic Framework

- arXiv: http://arxiv.org/abs/2603.09312v1 | 2026-03-10 | final 52.0

Scalable Vector Graphics (SVG) are central to digital design due to their inherent scalability and editability. Despite significant advancements in content generation enabled by Visual Language Models (VLMs), existing text-to-SVG generation methods are limited by a core challenge: the autoregressive training process does not incorporate visual perception of the final rendered image, which fundamentally constrains generation quality.

-> Focuses on SVG generation using visual feedback, conceptually similar to video processing but different application

## 60위: SurgFed: Language-guided Multi-Task Federated Learning for Surgical Video Understanding

- arXiv: http://arxiv.org/abs/2603.09496v1 | 2026-03-10 | final 50.4

Surgical scene Multi-Task Federated Learning (MTFL) is essential for robot-assisted minimally invasive surgery (RAS) but remains underexplored in surgical video understanding due to two key challenges: (1) Tissue Diversity: Local models struggle to adapt to site-specific tissue features, limiting their effectiveness in heterogeneous clinical environments and leading to poor local predictions. (2) Task Diversity: Server-side aggregation, relying solely on gradient-based clustering, often produces suboptimal or incorrect parameter updates due to inter-site task heterogeneity, resulting in inaccurate localization.

-> 수술 영상 이해 기술은 스포츠 영상 분석에 약간의 관련성 있음

---

## 다시 보기

### Helios: Real Real-Time Long Video Generation Model (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04379v1
- 점수: final 98.4

We introduce Helios, the first 14B video generation model that runs at 19.5 FPS on a single NVIDIA H100 GPU and supports minute-scale generation while matching the quality of a strong baseline. We make breakthroughs along three key dimensions: (1) robustness to long-video drifting without commonly used anti-drifting heuristics such as self-forcing, error-banks, or keyframe sampling; (2) real-time generation without standard acceleration techniques such as KV-cache, sparse/linear attention, or quantization; and (3) training without parallelism or sharding frameworks, enabling image-diffusion-scale batch sizes while fitting up to four 14B models within 80 GB of GPU memory. Specifically, Helios is a 14B autoregressive diffusion model with a unified input representation that natively supports T2V, I2V, and V2V tasks. To mitigate drifting in long-video generation, we characterize typical failure modes and propose simple yet effective training strategies that explicitly simulate drifting during training, while eliminating repetitive motion at its source. For efficiency, we heavily compress the historical and noisy context and reduce the number of sampling steps, yielding computational costs comparable to -- or lower than -- those of 1.3B video generative models. Moreover, we introduce infrastructure-level optimizations that accelerate both inference and training while reducing memory consumption. Extensive experiments demonstrate that Helios consistently outperforms prior methods on both short- and long-video generation. We plan to release the code, base model, and distilled model to support further development by the community.

-> 엣지 컴퓨팅 프레임워크가 AI 카메라 엣지 디바이스 배포에 직접적으로 관련

### Real Eyes Realize Faster: Gaze Stability and Pupil Novelty for Efficient Egocentric Learning (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04098v1
- 점수: final 96.0

Always-on egocentric cameras are increasingly used as demonstrations for embodied robotics, imitation learning, and assistive AR, but the resulting video streams are dominated by redundant and low-quality frames. Under the storage and battery constraints of wearable devices, choosing which frames to keep is as important as how to learn from them. We observe that modern eye-tracking headsets provide a continuous, training-free side channel that decomposes into two complementary axes: gaze fixation captures visual stability (quality), while pupil response captures arousal-linked moments (novelty). We operationalize this insight as a Dual-Criterion Frame Curator that first gates frames by gaze quality and then ranks the survivors by pupil-derived novelty. On the Visual Experience Dataset (VEDB), curated frames at 10% budget match the classification performance of the full stream, and naive signal fusion consistently destroys both contributions. The benefit is task-dependent: pupil ranking improves activity recognition, while gaze-only selection already dominates for scene recognition, confirming that the two signals serve genuinely different roles. Our method requires no model inference and operates at capture time, offering a path toward efficient, always-on egocentric data curation.

-> 안와 추적 데이터를 활용한 효율적인 프레임 선택 방식으로 스포츠 하이라이트 자동 추출에 직접 적용 가능

### RIVER: A Real-Time Interaction Benchmark for Video LLMs (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03985v1
- 점수: final 96.0

The rapid advancement of multimodal large language models has demonstrated impressive capabilities, yet nearly all operate in an offline paradigm, hindering real-time interactivity. Addressing this gap, we introduce the Real-tIme Video intERaction Bench (RIVER Bench), designed for evaluating online video comprehension. RIVER Bench introduces a novel framework comprising Retrospective Memory, Live-Perception, and Proactive Anticipation tasks, closely mimicking interactive dialogues rather than responding to entire videos at once. We conducted detailed annotations using videos from diverse sources and varying lengths, and precisely defined the real-time interactive format. Evaluations across various model categories reveal that while offline models perform well in single question-answering tasks, they struggle with real-time processing. Addressing the limitations of existing models in online video interaction, especially their deficiencies in long-term memory and future perception, we proposed a general improvement method that enables models to interact with users more flexibly in real time. We believe this work will significantly advance the development of real-time interactive video understanding models and inspire future research in this emerging field. Datasets and code are publicly available at https://github.com/OpenGVLab/RIVER.

-> 실시간 비디오 이해 및 상호작용 기술로 스포츠 경기 분석에 적용 가능

### Agentic Peer-to-Peer Networks: From Content Distribution to Capability and Action Sharing (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03753v1
- 점수: final 96.0

The ongoing shift of AI models from centralized cloud APIs to local AI agents on edge devices is enabling \textit{Client-Side Autonomous Agents (CSAAs)} -- persistent personal agents that can plan, access local context, and invoke tools on behalf of users. As these agents begin to collaborate by delegating subtasks directly between clients, they naturally form \emph{Agentic Peer-to-Peer (P2P) Networks}. Unlike classic file-sharing overlays where the exchanged object is static, hash-indexed content (e.g., files in BitTorrent), agentic overlays exchange \emph{capabilities and actions} that are heterogeneous, state-dependent, and potentially unsafe if delegated to untrusted peers. This article outlines the networking foundations needed to make such collaboration practical. We propose a plane-based reference architecture that decouples connectivity/identity, semantic discovery, and execution. Besides, we introduce signed, soft-state capability descriptors to support intent- and constraint-aware discovery. To cope with adversarial settings, we further present a \textit{tiered verification} spectrum: Tier~1 relies on reputation signals, Tier~2 applies lightweight canary challenge-response with fallback selection, and Tier~3 requires evidence packages such as signed tool receipts/traces (and, when applicable, attestation). Using a discrete-event simulator that models registry-based discovery, Sybil-style index poisoning, and capability drift, we show that tiered verification substantially improves end-to-end workflow success while keeping discovery latency near-constant and control-plane overhead modest.

-> 엣지 디바이스 및 AI 에이전트 기술이 프로젝트의 AI 촬영 장치 및 플랫폼에 직접 적용 가능

### SURE: Semi-dense Uncertainty-REfined Feature Matching (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04869v1
- 점수: final 96.0

Establishing reliable image correspondences is essential for many robotic vision problems. However, existing methods often struggle in challenging scenarios with large viewpoint changes or textureless regions, where incorrect cor- respondences may still receive high similarity scores. This is mainly because conventional models rely solely on fea- ture similarity, lacking an explicit mechanism to estimate the reliability of predicted matches, leading to overconfident errors. To address this issue, we propose SURE, a Semi- dense Uncertainty-REfined matching framework that jointly predicts correspondences and their confidence by modeling both aleatoric and epistemic uncertainties. Our approach in- troduces a novel evidential head for trustworthy coordinate regression, along with a lightweight spatial fusion module that enhances local feature precision with minimal overhead. We evaluated our method on multiple standard benchmarks, where it consistently outperforms existing state-of-the-art semi-dense matching models in both accuracy and efficiency. our code will be available on https://github.com/LSC-ALAN/SURE.

-> 시각적 표현 향상 기술은 스포츠 영상 보정 및 하이라이트 편집에 직접적으로 적용 가능하며, 자세 및 동작 분석에도 활용될 수 있습니다.

### Lambdas at the Far Edge: a Tale of Flying Lambdas and Lambdas on Wheels (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04008v1
- 점수: final 94.4

Aggregate Programming (AP) is a paradigm for programming the collective behaviour of sets of distributed devices, possibly situated at the network far edge, by relying on asynchronous proximity-based interactions. The eXchange Calculus (XC), a recently proposed foundational model for AP, is essentially a typed lambda calculus extended with an operator (the exchange operator) providing an implicit communication mechanism between neighbour devices. This paper provides a gentle introduction to XC and to its implementation as a C++ library, called FCPP. The FCPP library and toolchain has been mainly developed at the Department of Computer Science of the University of Turin, where Stefano Berardi spent most of his academic career conducting outstanding research about logical foundation of computer science and transmitting his passion for research to students and young researchers, often exploiting typed lambda calculi. An FCCP program is essentially a typed lambda term, and FCPP has been used to write code that has been deployed on devices at the far edge of the network, including rovers and (soon) Uncrewed Aerial Vehicles (UAVs); hence the title of the paper.

-> 엣지 컴퓨팅 프레임워크가 AI 카메라 엣지 디바이스 배포에 직접적으로 관련

### Architecture and evaluation protocol for transformer-based visual object tracking in UAV applications (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03904v1
- 점수: final 93.6

Object tracking from Unmanned Aerial Vehicles (UAVs) is challenged by platform dynamics, camera motion, and limited onboard resources. Existing visual trackers either lack robustness in complex scenarios or are too computationally demanding for real-time embedded use. We propose an Modular Asynchronous Tracking Architecture (MATA) that combines a transformer-based tracker with an Extended Kalman Filter, integrating ego-motion compensation from sparse optical flow and an object trajectory model. We further introduce a hardware-independent, embedded oriented evaluation protocol and a new metric called Normalized time to Failure (NT2F) to quantify how long a tracker can sustain a tracking sequence without external help. Experiments on UAV benchmarks, including an augmented UAV123 dataset with synthetic occlusions, show consistent improvements in Success and NT2F metrics across multiple tracking processing frequency. A ROS 2 implementation on a Nvidia Jetson AGX Orin confirms that the evaluation protocol more closely matches real-time performance on embedded systems.

-> 객체 추적 기술이 스포츠 경기 자동 촬영에 적용 가능하며, 임베디드 시스템에서 실시간 작동 가능

### Trainable Bitwise Soft Quantization for Input Feature Compression (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.05172v1
- 점수: final 93.6

The growing demand for machine learning applications in the context of the Internet of Things calls for new approaches to optimize the use of limited compute and memory resources. Despite significant progress that has been made w.r.t. reducing model sizes and improving efficiency, many applications still require remote servers to provide the required resources. However, such approaches rely on transmitting data from edge devices to remote servers, which may not always be feasible due to bandwidth, latency, or energy constraints. We propose a task-specific, trainable feature quantization layer that compresses the input features of a neural network. This can significantly reduce the amount of data that needs to be transferred from the device to a remote server. In particular, the layer allows each input feature to be quantized to a user-defined number of bits, enabling a simple on-device compression at the time of data collection. The layer is designed to approximate step functions with sigmoids, enabling trainable quantization thresholds. By concatenating outputs from multiple sigmoids, introduced as bitwise soft quantization, it achieves trainable quantized values when integrated with a neural network. We compare our method to full-precision inference as well as to several quantization baselines. Experiments show that our approach outperforms standard quantization methods, while maintaining accuracy levels close to those of full-precision models. In particular, depending on the dataset, compression factors of $5\times$ to $16\times$ can be achieved compared to $32$-bit input without significant performance loss.

-> RK3588 엣지 장치에서 데이터 전송량을 최대 16배 줄여 실시간 처리와 배터리 효율성을 크게 향상시킬 수 있습니다.

### Yolo-Key-6D: Single Stage Monocular 6D Pose Estimation with Keypoint Enhancements (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03879v1
- 점수: final 93.6

Estimating the 6D pose of objects from a single RGB image is a critical task for robotics and extended reality applications. However, state-of-the-art multi stage methods often suffer from high latency, making them unsuitable for real time use. In this paper, we present Yolo-Key-6D, a novel single stage, end-to-end framework for monocular 6D pose estimation designed for both speed and accuracy. Our approach enhances a YOLO based architecture by integrating an auxiliary head that regresses the 2D projections of an object's 3D bounding box corners. This keypoint detection task significantly improves the network's understanding of 3D geometry. For stable end-to-end training, we directly regress rotation using a continuous 9D representation projected to SO(3) via singular value decomposition. On the LINEMOD and LINEMOD-Occluded benchmarks, YOLO-Key-6D achieves competitive accuracy scores of 96.24% and 69.41%, respectively, with the ADD(-S) 0.1d metric, while proving itself to operate in real time. Our results demonstrate that a carefully designed single stage method can provide a practical and effective balance of performance and efficiency for real world deployment.

-> 6D 포즈 추정 기술이 스포츠 선수 자세 분석에 적용 가능하며 실시간 처리 가능

### EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04090v1
- 점수: final 93.6

Egocentric human motion estimation is essential for AR/VR experiences, yet remains challenging due to limited body coverage from the egocentric viewpoint, frequent occlusions, and scarce labeled data. We present EgoPoseFormer v2, a method that addresses these challenges through two key contributions: (1) a transformer-based model for temporally consistent and spatially grounded body pose estimation, and (2) an auto-labeling system that enables the use of large unlabeled datasets for training. Our model is fully differentiable, introduces identity-conditioned queries, multi-view spatial refinement, causal temporal attention, and supports both keypoints and parametric body representations under a constant compute budget. The auto-labeling system scales learning to tens of millions of unlabeled frames via uncertainty-aware semi-supervised training. The system follows a teacher-student schema to generate pseudo-labels and guide training with uncertainty distillation, enabling the model to generalize to different environments. On the EgoBody3M benchmark, with a 0.8 ms latency on GPU, our model outperforms two state-of-the-art methods by 12.2% and 19.4% in accuracy, and reduces temporal jitter by 22.2% and 51.7%. Furthermore, our auto-labeling system further improves the wrist MPJPE by 13.1%.

-> 인간 동작 추정 기술이 스포츠 동작 및 자세 분석에 직접적으로 적용 가능하며 실시간 처리 가능

### Toward Native ISAC Support in O-RAN Architectures for 6G (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03607v1
- 점수: final 93.6

ISAC is an emerging paradigm in 6G networks that enables environmental sensing using wireless communication infrastructure. Current O-RAN specifications lack the architectural primitives for sensing integration: no service models expose physical-layer observables, no execution frameworks support sub-millisecond sensing tasks, and fronthaul interfaces cannot correlate transmitted waveforms with their reflections.   This article proposes three extensions to O-RAN for monostatic sensing, where transmission and reception are co-located at the base station. First, we specify sensing dApps at the O-DU that process IQ samples to extract delay, Doppler, and angular features. Second, we define E2SM-SENS, a service model enabling xApps to subscribe to sensing telemetry with configurable periodicity. Third, we identify required Open Fronthaul metadata for waveform-echo association. We validate the architecture through a prototype implementation using beamforming and Full-Duplex operation, demonstrating closed-loop control with median end-to-end latency suitable for near-real-time sensing applications. While focused on monostatic configurations, the proposed interfaces extend to bistatic and cooperative sensing scenarios.

-> 저지연 센싱 기술이 실시간 스포츠 분석에 적용 가능

### Adaptive Enhancement and Dual-Pooling Sequential Attention for Lightweight Underwater Object Detection with YOLOv10 (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03807v1
- 점수: final 92.0

Underwater object detection constitutes a pivotal endeavor within the realms of marine surveillance and autonomous underwater systems; however, it presents significant challenges due to pronounced visual impairments arising from phenomena such as light absorption, scattering, and diminished contrast. In response to these formidable challenges, this manuscript introduces a streamlined yet robust framework for underwater object detection, grounded in the YOLOv10 architecture. The proposed method integrates a Multi-Stage Adaptive Enhancement module to improve image quality, a Dual-Pooling Sequential Attention (DPSA) mechanism embedded into the backbone to strengthen multi-scale feature representation, and a Focal Generalized IoU Objectness (FGIoU) loss to jointly improve localization accuracy and objectness prediction under class imbalance. Comprehensive experimental evaluations conducted on the RUOD and DUO benchmark datasets substantiate that the proposed DPSA_FGIoU_YOLOv10n attains exceptional performance, achieving mean Average Precision (mAP) scores of 88.9% and 88.0% at IoU threshold 0.5, respectively. In comparison to the baseline YOLOv10n, this represents enhancements of 6.7% for RUOD and 6.2% for DUO, all while preserving a compact model architecture comprising merely 2.8M parameters. These findings validate that the proposed framework establishes an efficacious equilibrium among accuracy, robustness, and real-time operational efficiency, making it suitable for deployment in resource-constrained underwater settings.

-> 가벼운 모델(2.8M 파라미터)과 이미지 보정 기술이 스포츠 캠용 엣지 디바이스에 직접 적용 가능

### Exploring Challenges in Developing Edge-Cloud-Native Applications Across Multiple Business Domains (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03738v1
- 점수: final 92.0

As the convergence of cloud computing and advanced networking continues to reshape modern software development, edge-cloud-native paradigms have become essential for enabling scalable, resilient, and agile digital services that depend on high-performance, low-latency, and reliable communication. This study investigates the practical challenges of developing, deploying, and maintaining edge-cloud-native applications through in-depth interviews with professionals from diverse domains, including IT, finance, healthcare, education, and industry. Despite significant advancements in cloud technologies, practitioners, particularly those from non-technical backgrounds-continue to encounter substantial complexity stemming from fragmented toolchains, steep learning curves, and operational overhead of managing distributed networking and computing, ensuring consistent performance across hybrid environments, and navigating steep learning curves at the cloud-network boundary. Across sectors, participants consistently prioritized productivity, Quality of Service, and usability over conventional concerns such as cost or migration. These findings highlight the need for operationally simplified, SLA-aware, and developer-friendly platforms that streamline the full application lifecycle. This study contributes a practice-informed perspective to support the alignment of edge-cloud-native systems with the realities and needs of modern enterprises, offering critical insights for the advancement of seamless cloud-network convergence.

-> Edge-클라우드 개발 과제가 AI 카메라 디바이스에 직접 적용 가능

### VisionPangu: A Compact and Fine-Grained Multimodal Assistant with 1.7B Parameters (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04957v1
- 점수: final 90.4

Large Multimodal Models (LMMs) have achieved strong performance in vision-language understanding, yet many existing approaches rely on large-scale architectures and coarse supervision, which limits their ability to generate detailed image captions. In this work, we present VisionPangu, a compact 1.7B-parameter multimodal model designed to improve detailed image captioning through efficient multimodal alignment and high-quality supervision. Our model combines an InternVL-derived vision encoder with the OpenPangu-Embedded language backbone via a lightweight MLP projector and adopts an instruction-tuning pipeline inspired by LLaVA. By incorporating dense human-authored descriptions from the DOCCI dataset, VisionPangu improves semantic coherence and descriptive richness without relying on aggressive model scaling. Experimental results demonstrate that compact multimodal models can achieve competitive performance while producing more structured and detailed captions. The code and model weights will be publicly available at https://www.modelscope.cn/models/asdfgh007/visionpangu.

-> 1.7B 파라미터로 경량화된 멀티모달 모델이 엣지 장치에서 스포츠 장면의 상세한 이해를 가능하게 합니다.

### Person Detection and Tracking from an Overhead Crane LiDAR (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04938v1
- 점수: final 90.4

This paper investigates person detection and tracking in an industrial indoor workspace using a LiDAR mounted on an overhead crane. The overhead viewpoint introduces a strong domain shift from common vehicle-centric LiDAR benchmarks, and limited availability of suitable public training data. Henceforth, we curate a site-specific overhead LiDAR dataset with 3D human bounding-box annotations and adapt selected candidate 3D detectors under a unified training and evaluation protocol. We further integrate lightweight tracking-by-detection using AB3DMOT and SimpleTrack to maintain person identities over time. Detection performance is reported with distance-sliced evaluation to quantify the practical operating envelope of the sensing setup. The best adapted detector configurations achieve average precision (AP) up to 0.84 within a 5.0 m horizontal radius, increasing to 0.97 at 1.0 m, with VoxelNeXt and SECOND emerging as the most reliable backbones across this range. The acquired results contribute in bridging the domain gap between standard driving datasets and overhead sensing for person detection and tracking. We also report latency measurements, highlighting practical real-time feasibility. Finally, we release our dataset and implementations in GitHub to support further research

-> 산업 현장의 LiDAR 기반 사람 감지 및 추적 기술로 스포츠 촬영과 유사한 추적 기술이 적용 가능하나 도메인이 상이함

### Scaling Dense Event-Stream Pretraining from Visual Foundation Models (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03969v1
- 점수: final 89.6

Learning versatile, fine-grained representations from irregular event streams is pivotal yet nontrivial, primarily due to the heavy annotation that hinders scalability in dataset size, semantic richness, and application scope. To mitigate this dilemma, we launch a novel self-supervised pretraining method that distills visual foundation models (VFMs) to push the boundaries of event representation at scale. Specifically, we curate an extensive synchronized image-event collection to amplify cross-modal alignment. Nevertheless, due to inherent mismatches in sparsity and granularity between image-event domains, existing distillation paradigms are prone to semantic collapse in event representations, particularly at high resolutions. To bridge this gap, we propose to extend the alignment objective to semantic structures provided off-the-shelf by VFMs, indicating a broader receptive field and stronger supervision. The key ingredient of our method is a structure-aware distillation loss that grounds higher-quality image-event correspondences for alignment, optimizing dense event representations. Extensive experiments demonstrate that our approach takes a great leap in downstream benchmarks, significantly surpassing traditional methods and existing pretraining techniques. This breakthrough manifests in enhanced generalization, superior data efficiency and elevated transferability.

-> 시각 기반 모델과 이벤트 스트림 처리 기술이 스포츠 영상 분석에 직접적으로 적용 가능하여 하이라이트 장면 추출에 효과적

### DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03744v1
- 점수: final 89.6

Estimating accurate, view-consistent geometry and camera poses from uncalibrated multi-view/video inputs remains challenging - especially at high spatial resolutions and over long sequences. We present DAGE, a dual-stream transformer whose main novelty is to disentangle global coherence from fine detail. A low-resolution stream operates on aggressively downsampled frames with alternating frame/global attention to build a view-consistent representation and estimate cameras efficiently, while a high-resolution stream processes the original images per-frame to preserve sharp boundaries and small structures. A lightweight adapter fuses these streams via cross-attention, injecting global context without disturbing the pretrained single-frame pathway. This design scales resolution and clip length independently, supports inputs up to 2K, and maintains practical inference cost. DAGE delivers sharp depth/pointmaps, strong cross-view consistency, and accurate poses, establishing new state-of-the-art results for video geometry estimation and multi-view reconstruction.

-> 듀얼 스트림 아키텍처로 스포츠 장면의 기하학적 분석 가능

### Optimal Short Video Ordering and Transmission Scheduling for Reducing Video Delivery Cost in Peer-to-Peer CDNs (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03938v1
- 점수: final 88.0

The explosive growth of short video platforms has generated a massive surge in global traffic, imposing heavy financial burdens on content providers. While Peer-to-Peer Content Delivery Networks (PCDNs) offer a cost-effective alternative by leveraging resource-constrained edge nodes, the limited storage and concurrent service capacities of these peers struggle to absorb the intense temporal demand spikes characteristic of short video consumption. In this paper, we propose to minimize transmission costs by exploiting a novel degree of freedom, the inherent flexibility of server-driven playback sequences. We formulate the Optimal Video Ordering and Transmission Scheduling (OVOTS) problem as an Integer Linear Program to jointly optimize personalized video ordering and transmission scheduling. By strategically permuting playlists, our approach proactively smooths temporal traffic peaks, maximizing the offloading of requests to low-cost peer nodes. To solve the OVOTS problem, we provide a rigorous theoretical reduction of the OVOTS problem to an auxiliary Minimum Cost Maximum Flow (MCMF) formulation. Leveraging König's Edge Coloring Theorem, we prove the strict equivalence of these formulations and develop the Minimum-cost Maximum-flow with Edge Coloring (MMEC) algorithm, a globally optimal, polynomial-time solution. Extensive simulations demonstrate that MMEC significantly outperforms baseline strategies, achieving cost reductions of up to 67% compared to random scheduling and 36% compared to a simulated annealing approach. Our results establish playback sequence flexibility as a robust and highly effective paradigm for cost optimization in PCDN architectures.

-> 단편 비디오 전송 최적화 기술이 콘텐츠 공유 플랫폼에 적용 가능

### Point Cloud Feature Coding for Object Detection over an Error-Prone Cloud-Edge Collaborative System (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03890v1
- 점수: final 86.4

Cloud-edge collaboration enhances machine perception by combining the strengths of edge and cloud computing. Edge devices capture raw data (e.g., 3D point clouds) and extract salient features, which are sent to the cloud for deeper analysis and data fusion. However, efficiently and reliably transmitting features between cloud and edge devices remains a challenging problem. We focus on point cloud-based object detection and propose a task-driven point cloud compression and reliable transmission framework based on source and channel coding. To meet the low-latency and low-power requirements of edge devices, we design a lightweight yet effective feature compaction module that compresses the deepest feature among multi-scale representations by removing task-irrelevant regions and applying channel-wise dimensionality reduction to task-relevant areas. Then, a signal-to-noise ratio (SNR)-adaptive channel encoder dynamically encodes the attribute information of the compacted features, while a Low-Density Parity-Check (LDPC) encoder ensures reliable transmission of geometric information. At the cloud side, an SNR-adaptive channel decoder guides the decoding of attribute information, and the LDPC decoder corrects geometry errors. Finally, a feature decompaction module restores the channel-wise dimensionality, and a diffusion-based feature upsampling module reconstructs shallow-layer features, enabling multi-scale feature reconstruction. On the KITTI dataset, our method achieved a 172-fold reduction in feature size with 3D average precision scores of 93.17%, 86.96%, and 77.25% for easy, moderate, and hard objects, respectively, over a 0 dB SNR wireless channel. Our source code will be released on GitHub at: https://github.com/yuanhui0325/T-PCFC.

-> 에지-클라우드 협업 아키텍처에 특징 압축 및 전송 기술이 적용 가능하여 실시간 스포츠 분석 시스템 구축에 유리

### Guiding Diffusion-based Reconstruction with Contrastive Signals for Balanced Visual Representation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04803v1
- 점수: final 86.4

The limited understanding capacity of the visual encoder in Contrastive Language-Image Pre-training (CLIP) has become a key bottleneck for downstream performance. This capacity includes both Discriminative Ability (D-Ability), which reflects class separability, and Detail Perceptual Ability (P-Ability), which focuses on fine-grained visual cues. Recent solutions use diffusion models to enhance representations by conditioning image reconstruction on CLIP visual tokens. We argue that such paradigms may compromise D-Ability and therefore fail to effectively address CLIP's representation limitations. To address this, we integrate contrastive signals into diffusion-based reconstruction to pursue more comprehensive visual representations. We begin with a straightforward design that augments the diffusion process with contrastive learning on input images. However, empirical results show that the naive combination suffers from gradient conflict and yields suboptimal performance. To balance the optimization, we introduce the Diffusion Contrastive Reconstruction (DCR), which unifies the learning objective. The key idea is to inject contrastive signals derived from each reconstructed image, rather than from the original input, into the diffusion process. Our theoretical analysis shows that the DCR loss can jointly optimize D-Ability and P-Ability. Extensive experiments across various benchmarks and multi-modal large language models validate the effectiveness of our method. The code is available at https://github.com/boyuh/DCR.

-> 일반적인 시각적 표현 기술이 스포츠 영상 분석에 적용 가능

### Semantic Bridging Domains: Pseudo-Source as Test-Time Connector (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03844v1
- 점수: final 85.6

Distribution shifts between training and testing data are a critical bottleneck limiting the practical utility of models, especially in real-world test-time scenarios. To adapt models when the source domain is unknown and the target domain is unlabeled, previous works constructed pseudo-source domains via data generation and translation, then aligned the target domain with them. However, significant discrepancies exist between the pseudo-source and the original source domain, leading to potential divergence when correcting the target directly. From this perspective, we propose a Stepwise Semantic Alignment (SSA) method, viewing the pseudo-source as a semantic bridge connecting the source and target, rather than a direct substitute for the source. Specifically, we leverage easily accessible universal semantics to rectify the semantic features of the pseudo-source, and then align the target domain using the corrected pseudo-source semantics. Additionally, we introduce a Hierarchical Feature Aggregation (HFA) module and a Confidence-Aware Complementary Learning (CACL) strategy to enhance the semantic quality of the SSA process in the absence of source and ground truth of target domains. We evaluated our approach on tasks like semantic segmentation and image classification, achieving a 5.2% performance boost on GTA2Cityscapes over the state-of-the-art.

-> 도메인 적응 기술이 엣지 디바이스에서의 스포츠 장면 분석에 적용 가능하여 다양한 환경에서 안정적인 성능 보장

### Scalable Injury-Risk Screening in Baseball Pitching From Broadcast Video (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04864v1
- 점수: final 85.6

Injury prediction in pitching depends on precise biomechanical signals, yet gold-standard measurements come from expensive, stadium-installed multi-camera systems that are unavailable outside professional venues. We present a monocular video pipeline that recovers 18 clinically relevant biomechanics metrics from broadcast footage, positioning pose-derived kinematics as a scalable source for injury-risk modeling. Built on DreamPose3D, our approach introduces a drift-controlled global lifting module that recovers pelvis trajectory via velocity-based parameterization and sliding-window inference, lifting pelvis-rooted poses into global space. To address motion blur, compression artifacts, and extreme pitching poses, we incorporate a kinematics refinement pipeline with bone-length constraints, joint-limited inverse kinematics, smoothing, and symmetry constraints to ensure temporally stable and physically plausible kinematics. On 13 professional pitchers (156 paired pitches), 16/18 metrics achieve sub-degree agreement (MAE $< 1^{\circ}$). Using these metrics for injury prediction, an automated screening model achieves AUC 0.811 for Tommy John surgery and 0.825 for significant arm injuries on 7,348 pitchers. The resulting pose-derived metrics support scalable injury-risk screening, establishing monocular broadcast video as a viable alternative to stadium-scale motion capture for biomechanics.

-> 야구 투구 자세 분석과 부상 예측에 관한 연구로 스포츠 동작 분석 기술이 적용 가능하나, 특정 스포츠(야구)와 부상 예측에 초점이 맞춰져 있어 일반적인 스포츠 촬영 및 하이라이트 편집과는 직접적 연관성은 떨어집니다.

### Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.04977v1
- 점수: final 84.8

Long video understanding is challenging due to dense visual redundancy, long-range temporal dependencies, and the tendency of chain-of-thought and retrieval-based agents to accumulate semantic drift and correlation-driven errors. We argue that long-video reasoning should begin not with reactive retrieval, but with deliberate task formulation: the model must first articulate what must be true in the video for each candidate answer to hold. This thinking-before-finding principle motivates VideoHV-Agent, a framework that reformulates video question answering as a structured hypothesis-verification process. Based on video summaries, a Thinker rewrites answer candidates into testable hypotheses, a Judge derives a discriminative clue specifying what evidence must be checked, a Verifier grounds and tests the clue using localized, fine-grained video content, and an Answer agent integrates validated evidence to produce the final answer. Experiments on three long-video understanding benchmarks show that VideoHV-Agent achieves state-of-the-art accuracy while providing enhanced interpretability, improved logical soundness, and lower computational cost. We make our code publicly available at: https://github.com/Haorane/VideoHV-Agent.

-> 장기 동영상 이해를 위한 가설-검증 다중 에이전트 프레임워크로, 스포츠 영상 분석에 적용 가능하지만 스포츠 특화나 실시간 처리는 다루지 않음.

### Separators in Enhancing Autoregressive Pretraining for Vision Mamba (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03806v1
- 점수: final 84.0

The state space model Mamba has recently emerged as a promising paradigm in computer vision, attracting significant attention due to its efficient processing of long sequence tasks. Mamba's inherent causal mechanism renders it particularly suitable for autoregressive pretraining. However, current autoregressive pretraining methods are constrained to short sequence tasks, failing to fully exploit Mamba's prowess in handling extended sequences. To address this limitation, we introduce an innovative autoregressive pretraining method for Vision Mamba that substantially extends the input sequence length. We introduce new \textbf{S}epara\textbf{T}ors for \textbf{A}uto\textbf{R}egressive pretraining to demarcate and differentiate between different images, known as \textbf{STAR}. Specifically, we insert identical separators before each image to demarcate its inception. This strategy enables us to quadruple the input sequence length of Vision Mamba while preserving the original dimensions of the dataset images. Employing this long sequence pretraining technique, our STAR-B model achieved an impressive accuracy of 83.5\% on ImageNet-1k, which is highly competitive in Vision Mamba. These results underscore the potential of our method in enhancing the performance of vision models through improved leveraging of long-range dependencies.

-> Vision Mamba for autoregressive pretraining is directly applicable to sports video processing and analysis.

### InfinityStory: Unlimited Video Generation with World Consistency and Character-Aware Shot Transitions (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.03646v1
- 점수: final 84.0

Generating long-form storytelling videos with consistent visual narratives remains a significant challenge in video synthesis. We present a novel framework, dataset, and a model that address three critical limitations: background consistency across shots, seamless multi-subject shot-to-shot transitions, and scalability to hour-long narratives. Our approach introduces a background-consistent generation pipeline that maintains visual coherence across scenes while preserving character identity and spatial relationships. We further propose a transition-aware video synthesis module that generates smooth shot transitions for complex scenarios involving multiple subjects entering or exiting frames, going beyond the single-subject limitations of prior work. To support this, we contribute with a synthetic dataset of 10,000 multi-subject transition sequences covering underrepresented dynamic scene compositions. On VBench, InfinityStory achieves the highest Background Consistency (88.94), highest Subject Consistency (82.11), and the best overall average rank (2.80), showing improved stability, smoother transitions, and better temporal coherence.

-> 하이라이트 영상 생성 및 편집 기술로 직접적으로 관련되어 경기 주요 장면 자동 편집에 필수적

### AI+HW 2035: Shaping the Next Decade (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.05225v1
- 점수: final 84.0

Artificial intelligence (AI) and hardware (HW) are advancing at unprecedented rates, yet their trajectories have become inseparably intertwined. The global research community lacks a cohesive, long-term vision to strategically coordinate the development of AI and HW. This fragmentation constrains progress toward holistic, sustainable, and adaptive AI systems capable of learning, reasoning, and operating efficiently across cloud, edge, and physical environments. The future of AI depends not only on scaling intelligence, but on scaling efficiency, achieving exponential gains in intelligence per joule, rather than unbounded compute consumption. Addressing this grand challenge requires rethinking the entire computing stack. This vision paper lays out a 10-year roadmap for AI+HW co-design and co-development, spanning algorithms, architectures, systems, and sustainability. We articulate key insights that redefine scaling around energy efficiency, system-level integration, and cross-layer optimization. We identify key challenges and opportunities, candidly assess potential obstacles and pitfalls, and propose integrated solutions grounded in algorithmic innovation, hardware advances, and software abstraction. Looking ahead, we define what success means in 10 years: achieving a 1000x improvement in efficiency for AI training and inference; enabling energy-aware, self-optimizing systems that seamlessly span cloud, edge, and physical AI; democratizing access to advanced AI infrastructure; and embedding human-centric principles into the design of intelligent systems. Finally, we outline concrete action items for academia, industry, government, and the broader community, calling for coordinated national initiatives, shared infrastructure, workforce development, cross-agency collaboration, and sustained public-private partnerships to ensure that AI+HW co-design becomes a unifying long-term mission.

-> AI+HW 공설계 접근법은 스포츠 촬영 및 분석을 위한 에지 디바이스의 효율성과 성능을 극대화하는 데 핵심적입니다. 특히 에너지 효율성 향상과 시스템 통합은 실시간 스포츠 영상 처리에 필수적입니다.

### HE-VPR: Height Estimation Enabled Aerial Visual Place Recognition Against Scale Variance (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04050v1
- 점수: final 82.4

In this work, we propose HE-VPR, a visual place recognition (VPR) framework that incorporates height estimation. Our system decouples height inference from place recognition, allowing both modules to share a frozen DINOv2 backbone. Two lightweight bypass adapter branches are integrated into our system. The first estimates the height partition of the query image via retrieval from a compact height database, and the second performs VPR within the corresponding height-specific sub-database. The adaptation design reduces training cost and significantly decreases the search space of the database. We also adopt a center-weighted masking strategy to further enhance the robustness against scale differences. Experiments on two self-collected challenging multi-altitude datasets demonstrate that HE-VPR achieves up to 6.1\% Recall@1 improvement over state-of-the-art ViT-based baselines and reduces memory usage by up to 90\%. These results indicate that HE-VPR offers a scalable and efficient solution for height-aware aerial VPR, enabling practical deployment in GNSS-denied environments. All the code and datasets for this work have been released on https://github.com/hmf21/HE-VPR.

-> 경공학 설계와 고도 추정 기술이 스포츠 장면 분석에 간접적으로 적용 가능

### A Baseline Study and Benchmark for Few-Shot Open-Set Action Recognition with Feature Residual Discrimination (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04125v1
- 점수: final 82.4

Few-Shot Action Recognition (FS-AR) has shown promising results but is often limited by a closed-set assumption that fails in real-world open-set scenarios. While Few-Shot Open-Set (FSOS) recognition is well-established for images, its extension to spatio-temporal video data remains underexplored. To address this, we propose an architectural extension based on a Feature-Residual Discriminator (FR-Disc), adapting previous work on skeletal data to the more complex video domain. Extensive experiments on five datasets demonstrate that while common open-set techniques provide only marginal gains, our FR-Disc significantly enhances unknown rejection capabilities without compromising closed-set accuracy, setting a new state-of-the-art for FSOS-AR. The project website, code, and benchmark are available at: https://hsp-iit.github.io/fsosar/.

-> 오픈셋 액션 인식 기술이 스포츠 경기 전략 분석에 적용 가능하며 코드 공개됨

### Semi-Supervised Generative Learning via Latent Space Distribution Matching (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04223v1
- 점수: final 82.4

We introduce Latent Space Distribution Matching (LSDM), a novel framework for semi-supervised generative modeling of conditional distributions. LSDM operates in two stages: (i) learning a low-dimensional latent space from both paired and unpaired data, and (ii) performing joint distribution matching in this space via the 1-Wasserstein distance, using only paired data. This two-step approach minimizes an upper bound on the 1-Wasserstein distance between joint distributions, reducing reliance on scarce paired samples while enabling fast one-step generation. Theoretically, we establish non-asymptotic error bounds and demonstrate a key benefit of unpaired data: enhanced geometric fidelity in generated outputs. Furthermore, by extending the scope of its two core steps, LSDM provides a coherent statistical perspective that connects to a broad class of latent-space approaches. Notably, Latent Diffusion Models (LDMs) can be viewed as a variant of LSDM, in which joint distribution matching is achieved indirectly via score matching. Consequently, our results also provide theoretical insights into the consistency of LDMs. Empirical evaluations on real-world image tasks, including class-conditional generation and image super-resolution, demonstrate the effectiveness of LSDM in leveraging unpaired data to enhance generation quality.

-> 생성적 학습 기술이 스포츠 콘텐츠의 영상/이미지 보정에 적용 가능

### SSR: A Generic Framework for Text-Aided Map Compression for Localization (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04272v1
- 점수: final 80.0

Mapping is crucial in robotics for localization and downstream decision-making. As robots are deployed in ever-broader settings, the maps they rely on continue to increase in size. However, storing these maps indefinitely (cold storage), transferring them across networks, or sending localization queries to cloud-hosted maps imposes prohibitive memory and bandwidth costs. We propose a text-enhanced compression framework that reduces both memory and bandwidth footprints while retaining high-fidelity localization. The key idea is to treat text as an alternative modality: one that can be losslessly compressed with large language models. We propose leveraging lightweight text descriptions combined with very small image feature vectors, which capture "complementary information" as a compact representation for the mapping task. Building on this, our novel technique, Similarity Space Replication (SSR), learns an adaptive image embedding in one shot that captures only the information "complementary" to the text descriptions. We validate our compression framework on multiple downstream localization tasks, including Visual Place Recognition as well as object-centric Monte Carlo localization in both indoor and outdoor settings. SSR achieves 2 times better compression than competing baselines on state-of-the-art datasets, including TokyoVal, Pittsburgh30k, Replica, and KITTI.

-> SSR 프레임워크는 스포츠 영상 압축 및 저장에 직접적으로 적용 가능하며, 텍스트와 시각 정보의 보완적 접근법이 하이라이트 장면 식별에 유용합니다.

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
