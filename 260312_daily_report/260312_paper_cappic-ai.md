# CAPP!C_AI 논문 리포트 (2026-03-12)

> 수집 77 | 필터 72 | 폐기 3 | 평가 72 | 출력 60 | 기준 50점

검색 윈도우: 2026-03-11T00:00:00+00:00 ~ 2026-03-12T00:30:00+00:00 | 임베딩: en_synthetic | run_id: 36

---

## 검색 키워드

autonomous cinematography, sports tracking, camera control, highlight detection, action recognition, keyframe extraction, video stabilization, image enhancement, color correction, pose estimation, biomechanics, tactical analysis, short video, content summarization, video editing, edge computing, embedded vision, real-time processing, content sharing, social platform, advertising system, biomechanics, tactical analysis, embedded vision

---

## 1위: AsyncMDE: Real-Time Monocular Depth Estimation via Asynchronous Spatial Memory

- arXiv: http://arxiv.org/abs/2603.10438v1
- PDF: https://arxiv.org/pdf/2603.10438v1
- 발행일: 2026-03-11
- 카테고리: cs.RO, cs.CV
- 점수: final 100.0 (llm_adjusted:100 = base:95 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Foundation-model-based monocular depth estimation offers a viable alternative to active sensors for robot perception, yet its computational cost often prohibits deployment on edge platforms. Existing methods perform independent per-frame inference, wasting the substantial computational redundancy between adjacent viewpoints in continuous robot operation. This paper presents AsyncMDE, an asynchronous depth perception system consisting of a foundation model and a lightweight model that amortizes the foundation model's computational cost over time. The foundation model produces high-quality spatial features in the background, while the lightweight model runs asynchronously in the foreground, fusing cached memory with current observations through complementary fusion, outputting depth estimates, and autoregressively updating the memory. This enables cross-frame feature reuse with bounded accuracy degradation. At a mere 3.83M parameters, it operates at 237 FPS on an RTX 4090, recovering 77% of the accuracy gap to the foundation model while achieving a 25X parameter reduction. Validated across indoor static, dynamic, and synthetic extreme-motion benchmarks, AsyncMDE degrades gracefully between refreshes and achieves 161FPS on a Jetson AGX Orin with TensorRT, clearly demonstrating its feasibility for real-time edge deployment.

**선정 근거**
에지 디바이스용 실시간 심도 추정 시스템으로 스포츠 장면의 3차원 공간 이해에 필수적이며, rk3588 기반 하드웨어에 최적화되어 있음

**활용 인사이트**
프레임 간 특성 재사용 기술을 통해 선수 위치 추적 및 동작 분석을 실시간으로 수행하고, 경기 장면의 3D 재구성을 통해 하이라이트 장면 자동 생성 가능

## 2위: HyPER-GAN: Hybrid Patch-Based Image-to-Image Translation for Real-Time Photorealism Enhancement

- arXiv: http://arxiv.org/abs/2603.10604v1
- PDF: https://arxiv.org/pdf/2603.10604v1
- 코드: https://github.com/stefanos50/HyPER-GAN
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 96.0 (llm_adjusted:95 = base:82 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Generative models are widely employed to enhance the photorealism of synthetic data for training computer vision algorithms. However, they often introduce visual artifacts that degrade the accuracy of these algorithms and require high computational resources, limiting their applicability in real-time training or evaluation scenarios. In this paper, we propose Hybrid Patch Enhanced Realism Generative Adversarial Network (HyPER-GAN), a lightweight image-to-image translation method based on a U-Net-style generator designed for real-time inference. The model is trained using paired synthetic and photorealism-enhanced images, complemented by a hybrid training strategy that incorporates matched patches from real-world data to improve visual realism and semantic consistency. Experimental results demonstrate that HyPER-GAN outperforms state-of-the-art paired image-to-image translation methods in terms of inference latency, visual realism, and semantic robustness. Moreover, it is illustrated that the proposed hybrid training strategy indeed improves visual quality and semantic consistency compared to training the model solely with paired synthetic and photorealism-enhanced images. Code and pretrained models are publicly available for download at: https://github.com/stefanos50/HyPER-GAN

**선정 근거**
Lightweight real-time image enhancement technology that could be applied to improve the visual quality of sports footage on edge devices.

## 3위: LCAMV: High-Accuracy 3D Reconstruction of Color-Varying Objects Using LCA Correction and Minimum-Variance Fusion in Structured Light

- arXiv: http://arxiv.org/abs/2603.10456v1
- PDF: https://arxiv.org/pdf/2603.10456v1
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 96.0 (llm_adjusted:95 = base:85 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Accurate 3D reconstruction of colored objects with structured light (SL) is hindered by lateral chromatic aberration (LCA) in optical components and uneven noise characteristics across RGB channels. This paper introduces lateral chromatic aberration correction and minimum-variance fusion (LCAMV), a robust 3D reconstruction method that operates with a single projector-camera pair without additional hardware or acquisition constraints. LCAMV analytically models and pixel-wise compensates LCA in both the projector and camera, then adaptively fuses multi-channel phase data using a Poisson-Gaussian noise model and minimum-variance estimation. Unlike existing methods that require extra hardware or multiple exposures, LCAMV enables fast acquisition. Experiments on planar and non-planar colored surfaces show that LCAMV outperforms grayscale conversion and conventional channel-weighting, reducing depth error by up to 43.6\%. These results establish LCAMV as an effective solution for high-precision 3D reconstruction of nonuniformly colored objects.

**선정 근거**
색상 보정 및 최소 분산 융합 기술로 스포츠 장면의 정밀 3D 재구성 및 영상 보정에 적용 가능

**활용 인사이트**
LCAMV를 rk3588 에지 디바이스에 통합하여 실시간으로 스포츠 장면의 3D 재구성 및 색상 왜곡 보정 수행, 최대 43.6%의 깊이 오류 감소로 고품질 영상 생성

## 4위: DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime

- arXiv: http://arxiv.org/abs/2603.10538v1
- PDF: https://arxiv.org/pdf/2603.10538v1
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Scene Graph Generation (SGG) aims to extract a detailed graph structure from an image, a representation that holds significant promise as a robust intermediate step for complex downstream tasks like reasoning for embodied agents. However, practical deployment in real-world applications - especially on resource constrained edge devices - requires speed and resource efficiency, challenges that have received limited attention in existing research. To bridge this gap, we introduce DSFlash, a low-latency model for panoptic scene graph generation designed to overcome these limitations. DSFlash can process a video stream at 56 frames per second on a standard RTX 3090 GPU, without compromising performance against existing state-of-the-art methods. Crucially, unlike prior approaches that often restrict themselves to salient relationships, DSFlash computes comprehensive scene graphs, offering richer contextual information while maintaining its superior latency. Furthermore, DSFlash is light on resources, requiring less than 24 hours to train on a single, nine-year-old GTX 1080 GPU. This accessibility makes DSFlash particularly well-suited for researchers and practitioners operating with limited computational resources, empowering them to adapt and fine-tune SGG models for specialized applications.

**선정 근거**
실시간 영상 처리 및 에지 디바이스에서의 실행 가능성이 스포츠 자동 촬영 장치와 관련 있음

## 5위: Frames2Residual: Spatiotemporal Decoupling for Self-Supervised Video Denoising

- arXiv: http://arxiv.org/abs/2603.10417v1
- PDF: https://arxiv.org/pdf/2603.10417v1
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Self-supervised video denoising methods typically extend image-based frameworks into the temporal dimension, yet they often struggle to integrate inter-frame temporal consistency with intra-frame spatial specificity. Existing Video Blind-Spot Networks (BSNs) require noise independence by masking the center pixel, this constraint prevents the use of spatial evidence for texture recovery, thereby severing spatiotemporal correlations and causing texture loss. To address this, we propose Frames2Residual (F2R), a spatiotemporal decoupling framework that explicitly divides self-supervised training into two distinct stages: blind temporal consistency modeling and non-blind spatial texture recovery. In Stage 1, a blind temporal estimator learns inter-frame consistency using a frame-wise blind strategy, producing a temporally consistent anchor. In Stage 2, a non-blind spatial refiner leverages this anchor to safely reintroduce the center frame and recover intra-frame high-frequency spatial residuals while preserving temporal stability. Extensive experiments demonstrate that our decoupling strategy allows F2R to outperform existing self-supervised methods on both sRGB and raw video benchmarks.

**선정 근거**
비디오 노이즈 제거 기술로 스포츠 경기 촬영 영상의 품질 향상에 적용 가능

## 6위: UHD Image Deblurring via Autoregressive Flow with Ill-conditioned Constraints

- arXiv: http://arxiv.org/abs/2603.10517v1
- PDF: https://arxiv.org/pdf/2603.10517v1
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Ultra-high-definition (UHD) image deblurring poses significant challenges for UHD restoration methods, which must balance fine-grained detail recovery and practical inference efficiency. Although prominent discriminative and generative methods have achieved remarkable results, a trade-off persists between computational cost and the ability to generate fine-grained detail for UHD image deblurring tasks. To further alleviate these issues, we propose a novel autoregressive flow method for UHD image deblurring with an ill-conditioned constraint. Our core idea is to decompose UHD restoration into a progressive, coarse-to-fine process: at each scale, the sharp estimate is formed by upsampling the previous-scale result and adding a current-scale residual, enabling stable, stage-wise refinement from low to high resolution. We further introduce Flow Matching to model residual generation as a conditional vector field and perform few-step ODE sampling with efficient Euler/Heun solvers, enriching details while keeping inference affordable. Since multi-step generation at UHD can be numerically unstable, we propose an ill-conditioning suppression scheme by imposing condition-number regularization on a feature-induced attention matrix, improving convergence and cross-scale consistency. Our method demonstrates promising performance on blurred images at 4K (3840$\times$2160) or higher resolutions.

**선정 근거**
고해상도 이미지 보정 기술이 스포츠 영상 보정에 직접적으로 적용 가능

## 7위: Safety-critical Control Under Partial Observability: Reach-Avoid POMDP meets Belief Space Control

- arXiv: http://arxiv.org/abs/2603.10572v1
- PDF: https://arxiv.org/pdf/2603.10572v1
- 발행일: 2026-03-11
- 카테고리: cs.RO
- 점수: final 93.6 (llm_adjusted:92 = base:82 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Partially Observable Markov Decision Processes (POMDPs) provide a principled framework for robot decision-making under uncertainty. Solving reach-avoid POMDPs, however, requires coordinating three distinct behaviors: goal reaching, safety, and active information gathering to reduce uncertainty. Existing online POMDP solvers attempt to address all three within a single belief tree search, but this unified approach struggles with the conflicting time scales inherent to these objectives. We propose a layered, certificate-based control architecture that operates directly in belief space, decoupling goal reaching, information gathering, and safety into modular components. We introduce Belief Control Lyapunov Functions (BCLFs) that formalize information gathering as a Lyapunov convergence problem in belief space, and show how they can be learned via reinforcement learning. For safety, we develop Belief Control Barrier Functions (BCBFs) that leverage conformal prediction to provide probabilistic safety guarantees over finite horizons. The resulting control synthesis reduces to lightweight quadratic programs solvable in real time, even for non-Gaussian belief representations with dimension $>10^4$. Experiments in simulation and on a space-robotics platform demonstrate real-time performance and improved safety and task success compared to state-of-the-art constrained POMDP solvers.

**선정 근거**
실시간 제어 시스템 및 불확실성 상황에서의 의사결정 기술이 스포츠 장면 분석에 적용 가능

## 8위: From Imitation to Intuition: Intrinsic Reasoning for Open-Instance Video Classification

- arXiv: http://arxiv.org/abs/2603.10300v1
- PDF: https://arxiv.org/pdf/2603.10300v1
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 92.0 (llm_adjusted:90 = base:82 + bonus:+8)
- 플래그: 엣지, 코드 공개

**개요**
Conventional video classification models, acting as effective imitators, excel in scenarios with homogeneous data distributions. However, real-world applications often present an open-instance challenge, where intra-class variations are vast and complex, beyond existing benchmarks. While traditional video encoder models struggle to fit these diverse distributions, vision-language models (VLMs) offer superior generalization but have not fully leveraged their reasoning capabilities (intuition) for such tasks. In this paper, we bridge this gap with an intrinsic reasoning framework that evolves open-instance video classification from imitation to intuition. Our approach, namely DeepIntuit, begins with a cold-start supervised alignment to initialize reasoning capability, followed by refinement using Group Relative Policy Optimization (GRPO) to enhance reasoning coherence through reinforcement learning. Crucially, to translate this reasoning into accurate classification, DeepIntuit then introduces an intuitive calibration stage. In this stage, a classifier is trained on this intrinsic reasoning traces generated by the refined VLM, ensuring stable knowledge transfer without distribution mismatch. Extensive experiments demonstrate that for open-instance video classification, DeepIntuit benefits significantly from transcending simple feature imitation and evolving toward intrinsic reasoning. Our project is available at https://bwgzk-keke.github.io/DeepIntuit/.

**선정 근거**
비디오 분류 프레임워크로 스포츠 장면 분석에 적용 가능

## 9위: PPGuide: Steering Diffusion Policies with Performance Predictive Guidance

- arXiv: http://arxiv.org/abs/2603.10980v1
- PDF: https://arxiv.org/pdf/2603.10980v1
- 발행일: 2026-03-11
- 카테고리: cs.RO
- 점수: final 90.4 (llm_adjusted:88 = base:78 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Diffusion policies have shown to be very efficient at learning complex, multi-modal behaviors for robotic manipulation. However, errors in generated action sequences can compound over time which can potentially lead to failure. Some approaches mitigate this by augmenting datasets with expert demonstrations or learning predictive world models which might be computationally expensive. We introduce Performance Predictive Guidance (PPGuide), a lightweight, classifier-based framework that steers a pre-trained diffusion policy away from failure modes at inference time. PPGuide makes use of a novel self-supervised process: it uses attention-based multiple instance learning to automatically estimate which observation-action chunks from the policy's rollouts are relevant to success or failure. We then train a performance predictor on this self-labeled data. During inference, this predictor provides a real-time gradient to guide the policy toward more robust actions. We validated our proposed PPGuide across a diverse set of tasks from the Robomimic and MimicGen benchmarks, demonstrating consistent improvements in performance.

**선정 근거**
경량 확산 정책 프레임워크로 스포츠 하이라이트 생성 및 동작 분석에 적용 가능

## 10위: Multi-Person Pose Estimation Evaluation Using Optimal Transportation and Improved Pose Matching

- arXiv: http://arxiv.org/abs/2603.10398v1
- PDF: https://arxiv.org/pdf/2603.10398v1
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 89.6 (llm_adjusted:87 = base:82 + bonus:+5)
- 플래그: 엣지

**개요**
In Multi-Person Pose Estimation, many metrics place importance on ranking of pose detection confidence scores. Current metrics tend to disregard false-positive poses with low confidence, focusing primarily on a larger number of high-confidence poses. Consequently, these metrics may yield high scores even when many false-positive poses with low confidence are detected. For fair evaluation taking into account a tradeoff between true-positive and false-positive poses, this paper proposes Optimal Correction Cost for pose (OCpose), which evaluates detected poses against pose annotations as an optimal transportation. For the fair tradeoff between true-positive and false-positive poses, OCpose equally evaluates all the detected poses regardless of their confidence scores. In OCpose, on the other hand, the confidence score of each pose is utilized to improve the reliability of matching scores between the estimated pose and pose annotations. As a result, OCpose provides a different perspective assessment than other confidence ranking-based metrics.

**선정 근거**
Pose estimation evaluation methodology applicable to sports movement analysis

## 11위: Event-based Photometric Stereo via Rotating Illumination and Per-Pixel Learning

- arXiv: http://arxiv.org/abs/2603.10748v1
- PDF: https://arxiv.org/pdf/2603.10748v1
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 88.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Photometric stereo is a technique for estimating surface normals using images captured under varying illumination. However, conventional frame-based photometric stereo methods are limited in real-world applications due to their reliance on controlled lighting, and susceptibility to ambient illumination. To address these limitations, we propose an event-based photometric stereo system that leverages an event camera, which is effective in scenarios with continuously varying scene radiance and high dynamic range conditions. Our setup employs a single light source moving along a predefined circular trajectory, eliminating the need for multiple synchronized light sources and enabling a more compact and scalable design. We further introduce a lightweight per-pixel multi-layer neural network that directly predicts surface normals from event signals generated by intensity changes as the light source rotates, without system calibration. Experimental results on benchmark datasets and real-world data collected with our data acquisition system demonstrate the effectiveness of our method, achieving a 7.12\% reduction in mean angular error compared to existing event-based photometric stereo methods. In addition, our method demonstrates robustness in regions with sparse event activity, strong ambient illumination, and scenes affected by specularities.

**선정 근거**
이벤트 기반 카메라와 경량 학습 방식으로 스포츠 촬영 및 보정에 활용 가능

## 12위: GroundCount: Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations

- arXiv: http://arxiv.org/abs/2603.10978v1
- PDF: https://arxiv.org/pdf/2603.10978v1
- 발행일: 2026-03-11
- 카테고리: cs.CV, cs.AI
- 점수: final 88.0 (llm_adjusted:85 = base:75 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Vision Language Models (VLMs) exhibit persistent hallucinations in counting tasks, with accuracy substantially lower than other visual reasoning tasks (excluding sentiment). This phenomenon persists even in state-of-the-art reasoning-capable VLMs. Conversely, CNN-based object detection models (ODMs) such as YOLO excel at spatial localization and instance counting with minimal computational overhead. We propose GroundCount, a framework that augments VLMs with explicit spatial grounding from ODMs to mitigate counting hallucinations. In the best case, our prompt-based augmentation strategy achieves 81.3% counting accuracy on the best-performing model (Ovis2.5-2B) - a 6.6pp improvement - while reducing inference time by 22% through elimination of hallucination-driven reasoning loops for stronger models. We conduct comprehensive ablation studies demonstrating that positional encoding is a critical component, being beneficial for stronger models but detrimental for weaker ones. Confidence scores, by contrast, introduce noise for most architectures and their removal improves performance in four of five evaluated models. We further evaluate feature-level fusion architectures, finding that explicit symbolic grounding via structured prompts outperforms implicit feature fusion despite sophisticated cross-attention mechanisms. Our approach yields consistent improvements across four of five evaluated VLM architectures (6.2--7.5pp), with one architecture exhibiting degraded performance due to incompatibility between its iterative reflection mechanisms and structured prompts. These results suggest that counting failures stem from fundamental spatial-semantic integration limitations rather than architecture-specific deficiencies, while highlighting the importance of architectural compatibility in augmentation strategies.

**선정 근거**
객체 탐지와 비전-언어 모델 결합으로 스포츠 장면 내 객체 분석에 적용 가능

## 13위: P-GSVC: Layered Progressive 2D Gaussian Splatting for Scalable Image and Video

- arXiv: http://arxiv.org/abs/2603.10551v1
- PDF: https://arxiv.org/pdf/2603.10551v1
- 발행일: 2026-03-11
- 카테고리: cs.CV, cs.MM
- 점수: final 86.4 (llm_adjusted:83 = base:80 + bonus:+3)
- 플래그: 코드 공개

**개요**
Gaussian splatting has emerged as a competitive explicit representation for image and video reconstruction. In this work, we present P-GSVC, the first layered progressive 2D Gaussian splatting framework that provides a unified solution for scalable Gaussian representation in both images and videos. P-GSVC organizes 2D Gaussian splats into a base layer and successive enhancement layers, enabling coarse-to-fine reconstructions. To effectively optimize this layered representation, we propose a joint training strategy that simultaneously updates Gaussians across layers, aligning their optimization trajectories to ensure inter-layer compatibility and a stable progressive reconstruction. P-GSVC supports scalability in terms of both quality and resolution. Our experiments show that the joint training strategy can gain up to 1.9 dB improvement in PSNR for video and 2.6 dB improvement in PSNR for image when compared to methods that perform sequential layer-wise training. Project page: https://longanwang-cs.github.io/PGSVC-webpage/

**선정 근거**
Image and video processing techniques directly applicable to creating photo-like videos and highlights

## 14위: Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity

- arXiv: http://arxiv.org/abs/2603.10990v1
- PDF: https://arxiv.org/pdf/2603.10990v1
- 코드: https://github.com/ZhengyaoFang/CFM
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 86.4 (llm_adjusted:83 = base:80 + bonus:+3)
- 플래그: 코드 공개

**개요**
Recent advances in text-to-image (T2I) generation have greatly improved visual quality, yet producing images that appear visually authentic to real-world photography remains challenging. This is partly due to biases in existing evaluation paradigms: human ratings and preference-trained metrics often favor visually vivid images with exaggerated saturation and contrast, which make generations often too vivid to be real even when prompted for realistic-style images. To address this issue, we present Color Fidelity Dataset (CFD) and Color Fidelity Metric (CFM) for objective evaluation of color fidelity in realistic-style generations. CFD contains over 1.3M real and synthetic images with ordered levels of color realism, while CFM employs a multimodal encoder to learn perceptual color fidelity. In addition, we propose a training-free Color Fidelity Refinement (CFR) that adaptively modulates spatial-temporal guidance scale in generation, thereby enhancing color authenticity. Together, CFD supports CFM for assessment, whose learned attention further guides CFR to refine T2I fidelity, forming a progressive framework for assessing and improving color fidelity in realistic-style T2I generation. The dataset and code are available at https://github.com/ZhengyaoFang/CFM.

**선정 근거**
사실적인 외관을 위한 이미지 향상 기술을 직접적으로 다루어 스포츠 사진 생성에 적용 가능

## 15위: Novel Architecture of RPA In Oral Cancer Lesion Detection

- arXiv: http://arxiv.org/abs/2603.10928v1
- PDF: https://arxiv.org/pdf/2603.10928v1
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 85.6 (llm_adjusted:82 = base:72 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Accurate and early detection of oral cancer lesions is crucial for effective diagnosis and treatment. This study evaluates two RPA implementations, OC-RPAv1 and OC-RPAv2, using a test set of 31 images. OC-RPAv1 processes one image per prediction in an average of 0.29 seconds, while OCRPAv2 employs a Singleton design pattern and batch processing, reducing prediction time to just 0.06 seconds per image. This represents a 60-100x efficiency improvement over standard RPA methods, showcasing that design patterns and batch processing can enhance scalability and reduce costs in oral cancer detection

**선정 근거**
효율적인 배치 처리 기술이 엣지 디바이스에서 스포츠 영상 처리에 적용 가능

## 16위: Muscle Synergy Priors Enhance Biomechanical Fidelity in Predictive Musculoskeletal Locomotion Simulation

- arXiv: http://arxiv.org/abs/2603.10474v1
- PDF: https://arxiv.org/pdf/2603.10474v1
- 발행일: 2026-03-11
- 카테고리: cs.LG, cs.NE, cs.RO
- 점수: final 85.6 (llm_adjusted:82 = base:82 + bonus:+0)

**개요**
Human locomotion emerges from high-dimensional neuromuscular control, making predictive musculoskeletal simulation challenging. We present a physiology-informed reinforcement-learning framework that constrains control using muscle synergies. We extracted a low-dimensional synergy basis from inverse musculoskeletal analyses of a small set of overground walking trials and used it as the action space for a muscle-driven three-dimensional model trained across variable speeds, slopes and uneven terrain. The resulting controller generated stable gait from 0.7-1.8 m/s and on $\pm$ 6$^{\circ}$ grades and reproduced condition-dependent modulation of joint angles, joint moments and ground reaction forces. Compared with an unconstrained controller, synergy-constrained control reduced non-physiological knee kinematics and kept knee moment profiles within the experimental envelope. Across conditions, simulated vertical ground reaction forces correlated strongly with human measurements, and muscle-activation timing largely fell within inter-subject variability. These results show that embedding neurophysiological structure into reinforcement learning can improve biomechanical fidelity and generalization in predictive human locomotion simulation with limited experimental data.

**선정 근거**
근육 시너지 기반 강화 학습으로 스포츠 동작 분석 및 생체역학적 시뮬레이션 구현 가능

**활용 인사이트**
다양한 스포츠 동작을 분석해 개인별 자세 교정 및 전략 분석에 적용 가능

## 17위: A$^2$-Edit: Precise Reference-Guided Image Editing of Arbitrary Objects and Ambiguous Masks

- arXiv: http://arxiv.org/abs/2603.10685v1
- PDF: https://arxiv.org/pdf/2603.10685v1
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 84.0 (llm_adjusted:80 = base:80 + bonus:+0)

**개요**
We propose \textbf{A$^2$-Edit}, a unified inpainting framework for arbitrary object categories, which allows users to replace any target region with a reference object using only a coarse mask. To address the issues of severe homogenization and limited category coverage in existing datasets, we construct a large-scale, multi-category dataset \textbf{UniEdit-500K}, which includes 8 major categories, 209 fine-grained subcategories, and a total of 500,104 image pairs. Such rich category diversity poses new challenges for the model, requiring it to automatically learn semantic relationships and distinctions across categories. To this end, we introduce the \textbf{Mixture of Transformer} module, which performs differentiated modeling of various object categories through dynamic expert selection, and further enhances cross-category semantic transfer and generalization through collaboration among experts. In addition, we propose a \textbf{Mask Annealing Training Strategy} (MATS) that progressively relaxes mask precision during training, reducing the model's reliance on accurate masks and improving robustness across diverse editing tasks. Extensive experiments on benchmarks such as VITON-HD and AnyInsertion demonstrate that A$^2$-Edit consistently outperforms existing approaches across all metrics, providing a new and efficient solution for arbitrary object editing.

**선정 근거**
임의 객체 편 프레임워크로 스포츠 하이라이트 영상 및 이미지 정교한 편집 가능

**활용 인사이트**
거마스크만으로도 객체를 교체해 하이라이트 장면을 자연스럽게 보정하고 향상시킬 수 있음

## 18위: Variance-Aware Adaptive Weighting for Diffusion Model Training

- arXiv: http://arxiv.org/abs/2603.10391v1
- PDF: https://arxiv.org/pdf/2603.10391v1
- 발행일: 2026-03-11
- 카테고리: cs.LG, cs.CV
- 점수: final 82.4 (llm_adjusted:78 = base:78 + bonus:+0)

**개요**
Diffusion models have recently achieved remarkable success in generative modeling, yet their training dynamics across different noise levels remain highly imbalanced, which can lead to inefficient optimization and unstable learning behavior. In this work, we investigate this imbalance from the perspective of loss variance across log-SNR levels and propose a variance-aware adaptive weighting strategy to address it. The proposed approach dynamically adjusts training weights based on the observed variance distribution, encouraging a more balanced optimization process across noise levels. Extensive experiments on CIFAR-10 and CIFAR-100 demonstrate that the proposed method consistently improves generative performance over standard training schemes, achieving lower Fréchet Inception Distance (FID) while also reducing performance variance across random seeds. Additional analysis, including loss-log-SNR visualization, variance heatmaps, and ablation studies, further reveal that the adaptive weighting effectively stabilizes training dynamics. These results highlight the potential of variance-aware training strategies for improving diffusion model optimization.

**선정 근거**
확산 모델 학습 최적화로 스포츠 영상 보정 및 이미지 생성 품질 향상

**활용 인사이트**
다양한 노이즈 수준에서 균형 잡힌 최적화로 더 안정적인 영상 생성 가능

## 19위: Parallel-in-Time Nonlinear Optimal Control via GPU-native Sequential Convex Programming

- arXiv: http://arxiv.org/abs/2603.10711v1
- PDF: https://arxiv.org/pdf/2603.10711v1
- 발행일: 2026-03-11
- 카테고리: cs.RO, eess.SY
- 점수: final 80.0 (llm_adjusted:75 = base:65 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Real-time trajectory optimization for nonlinear constrained autonomous systems is critical and typically performed by CPU-based sequential solvers. Specifically, reliance on global sparse linear algebra or the serial nature of dynamic programming algorithms restricts the utilization of massively parallel computing architectures like GPUs. To bridge this gap, we introduce a fully GPU-native trajectory optimization framework that combines sequential convex programming with a consensus-based alternating direction method of multipliers. By applying a temporal splitting strategy, our algorithm decouples the optimization horizon into independent, per-node subproblems that execute massively in parallel. The entire process runs fully on the GPU, eliminating costly memory transfers and large-scale sparse factorizations. This architecture naturally scales to multi-trajectory optimization. We validate the solver on a quadrotor agile flight task and a Mars powered descent problem using an on-board edge computing platform. Benchmarks reveal a sustained 4x throughput speedup and a 51% reduction in energy consumption over a heavily optimized 12-core CPU baseline. Crucially, the framework saturates the hardware, maintaining over 96% active GPU utilization to achieve planning rates exceeding 100 Hz. Furthermore, we demonstrate the solver's extensibility to robust Model Predictive Control by jointly optimizing dynamically coupled scenarios under stochastic disturbances, enabling scalable and safe autonomy.

**선정 근거**
GPU 기본 최적화 프레임워크로 실시간 스포츠 촬영 및 처리 성능 향상

**활용 인사이트**
rk3588 장치에서 병렬 처리로 지연 시간 감소 및 에너지 효율성 개선 가능

## 20위: Are Video Reasoning Models Ready to Go Outside?

- arXiv: http://arxiv.org/abs/2603.10652v1
- PDF: https://arxiv.org/pdf/2603.10652v1
- 발행일: 2026-03-11
- 카테고리: cs.CV, cs.AI
- 점수: final 80.0 (llm_adjusted:75 = base:75 + bonus:+0)

**개요**
In real-world deployment, vision-language models often encounter disturbances such as weather, occlusion, and camera motion. Under such conditions, their understanding and reasoning degrade substantially, revealing a gap between clean, controlled (i.e., unperturbed) evaluation settings and real-world robustness. To address this limitation, we propose ROVA, a novel training framework that improves robustness by modeling a robustness-aware consistency reward under spatio-temporal corruptions. ROVA introduces a difficulty-aware online training strategy that prioritizes informative samples based on the model's evolving capability. Specifically, it continuously re-estimates sample difficulty via self-reflective evaluation, enabling adaptive training with a robustness-aware consistency reward. We also introduce PVRBench, a new benchmark that injects real-world perturbations into embodied video datasets to assess both accuracy and reasoning quality under realistic disturbances. We evaluate ROVA and baselines on PVRBench, UrbanVideo, and VisBench, where open-source and proprietary models suffer up to 35% and 28% drops in accuracy and reasoning under realistic perturbations. ROVA effectively mitigates performance degradation, boosting relative accuracy by at least 24% and reasoning by over 9% compared with baseline models (QWen2.5/3-VL, InternVL2.5, Embodied-R). These gains transfer to clean standard benchmarks, yielding consistent improvements.

**선정 근거**
실제 환경에서의 강건성 향상으로 다양한 조건에서 스포츠 촬영 안정성 확보

**활용 인사이트**
날씨, 장애물, 카메라 움직임 등 외부 요인에도 강한 스포츠 촬영 시스템 구축 가능

## 21위: An Event-Driven E-Skin System with Dynamic Binary Scanning and real time SNN Classification

- arXiv: http://arxiv.org/abs/2603.10537v1
- PDF: https://arxiv.org/pdf/2603.10537v1
- 발행일: 2026-03-11
- 카테고리: cs.NE
- 점수: final 80.0 (llm_adjusted:75 = base:65 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
This paper presents a novel hardware system for high-speed, event-sparse sampling-based electronic skin (e-skin)that integrates sensing and neuromorphic computing. The system is built around a 16x16 piezoresistive tactile array with front end and introduces a event-based binary scan search strategy to classify the digits. This event-driven strategy achieves a 12.8x reduction in scan counts, a 38.2x data compression rate and a 28.4x equivalent dynamic range, a 99% data sparsity, drastically reducing the data acquisition overhead. The resulting sparse data stream is processed by a multi-layer convolutional spiking neural network (Conv-SNN) implemented on an FPGA, which requires only 65% of the computation and 15.6% of the weight storage relative to a CNN. Despite these significant efficiency gains, the system maintains a high classification accuracy of 92.11% for real-time handwritten digit recognition. Furthermore, a real neuromorphic tactile dataset using Address Event Representation (AER) is constructed. This work demonstrates a fully integrated, event-driven pipeline from analog sensing to neuromorphic classification, offering an efficient solution for robotic perception and human-computer interaction.

**선정 근거**
이벤트 기반 하드웨어 시스템으로 스포츠 장면 처리에 효율적인 접근 방식 제시. 이 논문은 이벤트 기반 이진 스캔 전략을 제안하며, 핵심은 데이터 효율성을 38.2배 향상시키는 것입니다.

**활용 인사이트**
이 시스템을 스포츠 촬영에 적용하면 99% 데이터 희소성을 달성하며, 실시간 분류 정확도 92.11%를 유지하면서 rk3588 엣지 디바이스의 계산량을 65%만 사용할 수 있습니다.

## 22위: BinWalker: Development and Field Evaluation of a Quadruped Manipulator Platform for Sustainable Litter Collection

- arXiv: http://arxiv.org/abs/2603.10529v1
- PDF: https://arxiv.org/pdf/2603.10529v1
- 코드: https://github.com/iit-DLSLab/trash-collection-isaaclab
- 발행일: 2026-03-11
- 카테고리: cs.RO
- 점수: final 78.4 (llm_adjusted:73 = base:60 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
Litter pollution represents a growing environmental problem affecting natural and urban ecosystems worldwide. Waste discarded in public spaces often accumulates in areas that are difficult to access, such as uneven terrains, coastal environments, parks, and roadside vegetation. Over time, these materials degrade and release harmful substances, including toxic chemicals and microplastics, which can contaminate soil and water and pose serious threats to wildlife and human health. Despite increasing awareness of the problem, litter collection is still largely performed manually by human operators, making large-scale cleanup operations labor-intensive, time-consuming, and costly. Robotic solutions have the potential to support and partially automate environmental cleanup tasks. In this work, we present a quadruped robotic system designed for autonomous litter collection in challenging outdoor scenarios. The robot combines the mobility advantages of legged locomotion with a manipulation system consisting of a robotic arm and an onboard litter container. This configuration enables the robot to detect, grasp, and store litter items while navigating through uneven terrains. The proposed system aims to demonstrate the feasibility of integrating perception, locomotion, and manipulation on a legged robotic platform for environmental cleanup tasks. Experimental evaluations conducted in outdoor scenarios highlight the effectiveness of the approach and its potential for assisting large-scale litter removal operations in environments that are difficult to reach with traditional robotic platforms. The code associated with this work can be found at: https://github.com/iit-DLSLab/trash-collection-isaaclab.

**선정 근거**
사물 인식 및 이동 기술이 스포츠 장면 자동 촬영 및 객체 추적에 직접 적용 가능

**활용 인사이트**
사물 탐지 알고리즘을 스포츠 장비로 변형하여 선수 추적과 경기 장면 분석에 활용

## 23위: Learning to Wander: Improving the Global Image Geolocation Ability of LMMs via Actionable Reasoning

- arXiv: http://arxiv.org/abs/2603.10463v1
- PDF: https://arxiv.org/pdf/2603.10463v1
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 76.0 (llm_adjusted:70 = base:70 + bonus:+0)

**개요**
Geolocation, the task of identifying the geographic location of an image, requires abundant world knowledge and complex reasoning abilities. Though advanced large multimodal models (LMMs) have shown superior aforementioned capabilities, their performance on the geolocation task remains unexplored. To this end, we introduce \textbf{WanderBench}, the first open access global geolocation benchmark designed for actionable geolocation reasoning in embodied scenarios. WanderBench contains over 32K panoramas across six continents, organized as navigable graphs that enable physical actions such as rotation and movement, transforming geolocation from static recognition into interactive exploration. Building on this foundation, we propose \textbf{GeoAoT} (Action of Thought), a \underline{Geo}location framework with \underline{A}ction of \underline{T}hough, which couples reasoning with embodied actions. Instead of generating textual reasoning chains, GeoAoT produces actionable plans such as, approaching landmarks or adjusting viewpoints, to actively reduce uncertainty. We further establish an evaluation protocol that jointly measures geolocation accuracy and difficulty-aware geolocation questioning ability. Experiments on 19 large multimodal models show that GeoAoT achieves superior fine-grained localization and stronger generalization in dynamic environments. WanderBench and GeoAoT define a new paradigm for actionable, reasoning driven geolocation in embodied visual understanding.

**선정 근거**
행동 기반 추론 프레임워크로 스포츠 전략 분석 및 장소 식별에 활용 가능. 이 논문은 GeoAoT 프레임워크를 제안하며, 핵심은 추론과 신체적 행동을 결합하는 것입니다.

**활용 인사이트**
WanderBench 벤치마크를 스포츠 전략 분석에 적용하여 경기장 내에서의 위치 식별 정확도를 향상시키고, 실시간 전략 수립을 위한 행동 계획을 생성할 수 있습니다.

## 24위: GLM-OCR Technical Report

- arXiv: http://arxiv.org/abs/2603.10910v1
- PDF: https://arxiv.org/pdf/2603.10910v1
- 발행일: 2026-03-11
- 카테고리: cs.CL
- 점수: final 76.0 (llm_adjusted:70 = base:65 + bonus:+5)
- 플래그: 엣지

**개요**
GLM-OCR is an efficient 0.9B-parameter compact multimodal model designed for real-world document understanding. It combines a 0.4B-parameter CogViT visual encoder with a 0.5B-parameter GLM language decoder, achieving a strong balance between computational efficiency and recognition performance. To address the inefficiency of standard autoregressive decoding in deterministic OCR tasks, GLM-OCR introduces a Multi-Token Prediction (MTP) mechanism that predicts multiple tokens per step, significantly improving decoding throughput while keeping memory overhead low through shared parameters. At the system level, a two-stage pipeline is adopted: PP-DocLayout-V3 first performs layout analysis, followed by parallel region-level recognition. Extensive evaluations on public benchmarks and industrial scenarios show that GLM-OCR achieves competitive or state-of-the-art performance in document parsing, text and formula transcription, table structure recovery, and key information extraction. Its compact architecture and structured generation make it suitable for both resource-constrained edge deployment and large-scale production systems.

**선정 근거**
문서 이해를 위한 효율적인 다중모델 아키텍처로 엣지 디바이스 배포에 적합하지만 스포츠 촬영/분석과 직접적인 관련은 없음

## 25위: Towards Cognitive Defect Analysis in Active Infrared Thermography with Vision-Text Cues

- arXiv: http://arxiv.org/abs/2603.10549v1
- PDF: https://arxiv.org/pdf/2603.10549v1
- 발행일: 2026-03-11
- 카테고리: cs.CV, cs.AI, eess.SP
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Active infrared thermography (AIRT) is currently witnessing a surge of artificial intelligence (AI) methodologies being deployed for automated subsurface defect analysis of high performance carbon fiber-reinforced polymers (CFRP). Deploying AI-based AIRT methodologies for inspecting CFRPs requires the creation of time consuming and expensive datasets of CFRP inspection sequences to train neural networks. To address this challenge, this work introduces a novel language-guided framework for cognitive defect analysis in CFRPs using AIRT and vision-language models (VLMs). Unlike conventional learning-based approaches, the proposed framework does not require developing training datasets for extensive training of defect detectors, instead it relies solely on pretrained multimodal VLM encoders coupled with a lightweight adapter to enable generative zero-shot understanding and localization of subsurface defects. By leveraging pretrained multimodal encoders, the proposed system enables generative zero-shot understanding of thermographic patterns and automatic detection of subsurface defects. Given the domain gap between thermographic data and natural images used to train VLMs, an AIRT-VLM Adapter is proposed to enhance the visibility of defects while aligning the thermographic domain with the learned representations of VLMs. The proposed framework is validated using three representative VLMs; specifically, GroundingDINO, Qwen-VL-Chat, and CogVLM. Validation is performed on 25 CFRP inspection sequences with impacts introduced at different energy levels, reflecting realistic defects encountered in industrial scenarios. Experimental results demonstrate that the AIRT-VLM adapter achieves signal-to-noise ratio (SNR) gains exceeding 10 dB compared with conventional thermographic dimensionality-reduction methods, while enabling zero-shot defect detection with intersection-over-union values reaching 70%.

**선정 근거**
Vision-language models with lightweight adapters could be applicable for analyzing sports activities. 이 논문은 언어-안내 프레임워크를 제안하며, 핵심은 사전 학습된 다중모달 VLM 인코더를 사용하는 것입니다.

**활용 인사이트**
가벼운 어댑터를 사용한 VLM 모델로 스포츠 동작을 제로샷 학습 없이 분석하며, 신경망 학습 없이도 스포츠 자세 및 전략을 이해할 수 있습니다.

## 26위: MoXaRt: Audio-Visual Object-Guided Sound Interaction for XR

- arXiv: http://arxiv.org/abs/2603.10465v1
- PDF: https://arxiv.org/pdf/2603.10465v1
- 발행일: 2026-03-11
- 카테고리: cs.SD, cs.CV, cs.HC
- 점수: final 76.0 (llm_adjusted:70 = base:65 + bonus:+5)
- 플래그: 실시간

**개요**
In Extended Reality (XR), complex acoustic environments often overwhelm users, compromising both scene awareness and social engagement due to entangled sound sources. We introduce MoXaRt, a real-time XR system that uses audio-visual cues to separate these sources and enable fine-grained sound interaction. MoXaRt's core is a cascaded architecture that performs coarse, audio-only separation in parallel with visual detection of sources (e.g., faces, instruments). These visual anchors then guide refinement networks to isolate individual sources, separating complex mixes of up to 5 concurrent sources (e.g., 2 voices + 3 instruments) with ~2 second processing latency. We validate MoXaRt through a technical evaluation on a new dataset of 30 one-minute recordings featuring concurrent speech and music, and a 22-participant user study. Empirical results indicate that our system significantly enhances speech intelligibility, yielding a 36.2% (p < 0.01) increase in listening comprehension within adversarial acoustic environments while substantially reducing cognitive load (p < 0.001), thereby paving the way for more perceptive and socially adept XR experiences.

**선정 근거**
오디오-비주얼 처리 기술이 스포츠 콘텐츠 분석에 간접적으로 적용 가능

**활용 인사이트**
실시간 오디오-비주얼 분리 기술을 경기장 소리와 선수 행동 분석에 적용하여 하이라이트 장면 자동 추출 가능

## 27위: Phase-Interface Instance Segmentation as a Visual Sensor for Laboratory Process Monitoring

- arXiv: http://arxiv.org/abs/2603.10782v1
- PDF: https://arxiv.org/pdf/2603.10782v1
- 발행일: 2026-03-11
- 카테고리: cs.CV
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Reliable visual monitoring of chemical experiments remains challenging in transparent glassware, where weak phase boundaries and optical artifacts degrade conventional segmentation. We formulate laboratory phenomena as the time evolution of phase interfaces and introduce the Chemical Transparent Glasses dataset 2.0 (CTG 2.0), a vessel-aware benchmark with 3,668 images, 23 glassware categories, and five multiphase interface types for phase-interface instance segmentation. Building on YOLO11m-seg, we propose LGA-RCM-YOLO, which combines Local-Global Attention (LGA) for robust semantic representation and a Rectangular Self-Calibration Module (RCM) for boundary refinement of thin, elongated interfaces. On CTG 2.0, the proposed model achieves 84.4% AP@0.5 and 58.43% AP@0.5-0.95, improving over the YOLO11m baseline by 6.42 and 8.75 AP points, respectively, while maintaining near real-time inference (13.67 FPS, RTX 3060). An auxiliary color-attribute head further labels liquid instances as colored or colorless with 98.71% precision and 98.32% recall. Finally, we demonstrate continuous process monitoring in separatory-funnel phase separation and crystallization, showing that phase-interface instance segmentation can serve as a practical visual sensor for laboratory automation.

**선정 근거**
Computer vision segmentation for monitoring, adaptable for sports

**활용 인사이트**
YOLO 기반의 실시간 인스턴스 분할 기술을 선수 개인별 동작 분석 및 경기 상황 모니터링에 적용 가능

## 28위: An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS

- arXiv: http://arxiv.org/abs/2603.10671v1
- PDF: https://arxiv.org/pdf/2603.10671v1
- 발행일: 2026-03-11
- 카테고리: cs.AR, cs.CV, eess.IV
- 점수: final 76.0 (llm_adjusted:70 = base:60 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Recently, progress has been made on the Intra Pattern Copy (IPC) tool for JPEG XS, an image compression standard designed for low-latency and low-complexity coding. IPC performs wavelet-domain intra compensation predictions to reduce spatial redundancy in screen content. A key module of IPC is the displacement vector (DV) search, which aims to solve the optimal prediction reference offset. However, the DV search process is computationally intensive, posing challenges for practical hardware deployment. In this paper, we propose an efficient pipelined FPGA architecture design for the DV search module to promote the practical deployment of IPC. Optimized memory organization, which leverages the IPC computational characteristics and data inherent reuse patterns, is further introduced to enhance the performance. Experimental results show that our proposed architecture achieves a throughput of 38.3 Mpixels/s with a power consumption of 277 mW, demonstrating its feasibility for practical hardware implementation in IPC and other predictive coding tools, and providing a promising foundation for ASIC deployment.

**선정 근거**
이 논문은 JPEG XS 이미지 압축을 위한 FPGA 구현 방법을 제안합니다. 핵심은 저지연 영상 처리를 위한 효율적인 파이프라인 아키텍처 설계입니다. 스포츠 경기 촬영 엣지 디바이스에 직접 적용 가능한 하드웨어 솔루션을 제공합니다.

**활용 인사이트**
277mW 전력 소비로 38.3 Mpixels/s 처리 속도 달성, 실시간 스포츠 영상 압축 및 보정에 활용 가능하며 rk3588 기반 디바이스에 통합하여 저전력 고성능 처리 구현

## 29위: Spatio-Temporal Attention Graph Neural Network: Explaining Causalities With Attention

- arXiv: http://arxiv.org/abs/2603.10676v1
- PDF: https://arxiv.org/pdf/2603.10676v1
- 발행일: 2026-03-11
- 카테고리: cs.LG, cs.CE
- 점수: final 76.0 (llm_adjusted:70 = base:65 + bonus:+5)
- 플래그: 실시간

**개요**
Industrial Control Systems (ICS) underpin critical infrastructure and face growing cyber-physical threats due to the convergence of operational technology and networked environments. While machine learning-based anomaly detection approaches in ICS shows strong theoretical performance, deployment is often limited by poor explainability, high false-positive rates, and sensitivity to evolving system behavior, i.e., baseline drifting. We propose a Spatio-Temporal Attention Graph Neural Network (STA-GNN) for unsupervised and explainable anomaly detection in ICS that models both temporal dynamics and relational structure of the system. Sensors, controllers, and network entities are represented as nodes in a dynamically learned graph, enabling the model to capture inter-dependencies across physical processes and communication patterns. Attention mechanisms provide influential relationships, supporting inspection of correlations and potential causal pathways behind detected events. The approach supports multiple data modalities, including SCADA point measurements, network flow features, and payload features, and thus enables unified cyber-physical analysis. To address operational requirements, we incorporate a conformal prediction strategy to control false alarm rates and monitor performance degradation under drifting of the environment. Our findings highlight the possibilities and limitations of model evaluation and common pitfalls in anomaly detection in ICS. Our findings emphasise the importance of explainable, drift-aware evaluation for reliable deployment of learning-based security monitoring systems.

**선정 근거**
스페이시오-템포럴 어텐션 그래프 신경망으로 스포츠 데이터의 인과 관계 분석에 간접적으로 적용 가능

**활용 인사이트**
시공간 어텐션 메커니즘을 활용하여 선수 간 상호작용과 경기 전략의 인과 관계 분석 가능

## 30위: Semantic Landmark Particle Filter for Robot Localisation in Vineyards

- arXiv: http://arxiv.org/abs/2603.10847v1
- PDF: https://arxiv.org/pdf/2603.10847v1
- 발행일: 2026-03-11
- 카테고리: cs.RO, cs.AI
- 점수: final 72.0 (llm_adjusted:65 = base:55 + bonus:+10)
- 플래그: 엣지, 실시간

**개요**
Reliable localisation in vineyards is hindered by row-level perceptual aliasing: parallel crop rows produce nearly identical LiDAR observations, causing geometry-only and vision-based SLAM systems to converge towards incorrect corridors, particularly during headland transitions. We present a Semantic Landmark Particle Filter (SLPF) that integrates trunk and pole landmark detections with 2D LiDAR within a probabilistic localisation framework. Detected trunks are converted into semantic walls, forming structural row boundaries embedded in the measurement model to improve discrimination between adjacent rows. GNSS is incorporated as a lightweight prior that stabilises localisation when semantic observations are sparse.   Field experiments in a 10-row vineyard demonstrate consistent improvements over geometry-only (AMCL), vision-based (RTAB-Map), and GNSS baselines. Compared to AMCL, SLPF reduces Absolute Pose Error by 22% and 65% across two traversal directions; relative to a NoisyGNSS baseline, APE decreases by 65% and 61%. Row correctness improves from 0.67 to 0.73, while mean cross-track error decreases from 1.40 m to 1.26 m. These results show that embedding row-level structural semantics within the measurement model enables robust localisation in highly repetitive outdoor agricultural environments.

**선정 근거**
Computer vision and localization techniques that could be adapted for sports venue tracking

## 31위: Learning Bimanual Cloth Manipulation with Vision-based Tactile Sensing via Single Robotic Arm

- arXiv: http://arxiv.org/abs/2603.10609v1 | 2026-03-11 | final 72.0

Robotic cloth manipulation remains challenging due to the high-dimensional state space of fabrics, their deformable nature, and frequent occlusions that limit vision-based sensing. Although dual-arm systems can mitigate some of these issues, they increase hardware and control complexity.

-> Vision-based perception techniques that could be applicable to movement analysis in sports

## 32위: UAV-MARL: Multi-Agent Reinforcement Learning for Time-Critical and Dynamic Medical Supply Delivery

- arXiv: http://arxiv.org/abs/2603.10528v1 | 2026-03-11 | final 72.0

Unmanned aerial vehicles (UAVs) are increasingly used to support time-critical medical supply delivery, providing rapid and flexible logistics during emergencies and resource shortages. However, effective deployment of UAV fleets requires coordination mechanisms capable of prioritizing medical requests, allocating limited aerial resources, and adapting delivery schedules under uncertain operational conditions.

-> Multi-agent coordination concepts potentially relevant for camera systems

## 33위: Factorized Neural Implicit DMD for Parametric Dynamics

- arXiv: http://arxiv.org/abs/2603.10995v1 | 2026-03-11 | final 72.0

A data-driven, model-free approach to modeling the temporal evolution of physical systems mitigates the need for explicit knowledge of the governing equations. Even when physical priors such as partial differential equations are available, such systems often reside in high-dimensional state spaces and exhibit nonlinear dynamics, making traditional numerical solvers computationally expensive and ill-suited for real-time analysis and control.

-> 신경망 임플리시트 DMD로 스포츠 동작의 시간적 진화 모델링에 간접적으로 적용 가능

## 34위: FRIEND: Federated Learning for Joint Optimization of multi-RIS Configuration and Eavesdropper Intelligent Detection in B5G Networks

- arXiv: http://arxiv.org/abs/2603.10977v1 | 2026-03-11 | final 72.0

As wireless systems evolve toward Beyond 5G (B5G), the adoption of cell-free (CF) millimeter-wave (mmWave) architectures combined with Reconfigurable Intelligent Surfaces (RIS) is emerging as a key enabler for ultra-reliable, high-capacity, scalable, and secure Industrial Internet of Things (IIoT) communications. However, safeguarding these complex and distributed environments against eavesdropping remains a critical challenge, particularly when conventional security mechanisms struggle to overcome scalability, and latency constraints.

-> 엣지 AI 처리를 사용하지만 무선 보안에 초점을 맞추어 스포츠 영상 분석과 직접적으로 관련이 없음

## 35위: World2Act: Latent Action Post-Training via Skill-Compositional World Models

- arXiv: http://arxiv.org/abs/2603.10422v1 | 2026-03-11 | final 69.6

World Models (WMs) have emerged as a promising approach for post-training Vision-Language-Action (VLA) policies to improve robustness and generalization under environmental changes. However, most WM-based post-training methods rely on pixel-space supervision, making policies sensitive to pixel-level artifacts and hallucination from imperfect WM rollouts.

-> Action understanding framework with potential applications to sports strategy analysis

## 36위: SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning

- arXiv: http://arxiv.org/abs/2603.10446v1 | 2026-03-11 | final 68.0

Generating natural and linguistically accurate sign language avatars remains a formidable challenge. Current Sign Language Production (SLP) frameworks face a stark trade-off: direct text-to-pose models suffer from regression-to-the-mean effects, while dictionary-retrieval methods produce robotic, disjointed transitions.

-> 움직임 합성 기술은 스포츠 동작 분석에 간접적으로 적용 가능

## 37위: Semantic Satellite Communications for Synchronized Audiovisual Reconstruction

- arXiv: http://arxiv.org/abs/2603.10791v1 | 2026-03-11 | final 68.0

Satellite communications face severe bottlenecks in supporting high-fidelity synchronized audiovisual services, as conventional schemes struggle with cross-modal coherence under fluctuating channel conditions, limited bandwidth, and long propagation delays. To address these limitations, this paper proposes an adaptive multimodal semantic transmission system tailored for satellite scenarios, aiming for high-quality synchronized audiovisual reconstruction under bandwidth constraints.

-> Semantic audiovisual processing concepts relevant for video analysis

## 38위: In-Memory ADC-Based Nonlinear Activation Quantization for Efficient In-Memory Computing

- arXiv: http://arxiv.org/abs/2603.10540v1 | 2026-03-11 | final 68.0

In deep networks, operations such as ReLU and hardware-driven clamping often cause activations to accumulate near the edges of the distribution, leading to biased clustering and suboptimal quantization in existing nonlinear (NL) quantization methods. This paper introduces Boundary Suppressed K-Means Quantization (BS-KMQ), a novel NL quantization approach designed to reduce the resolution requirements of analog-to-digital converters (ADCs) in in-memory computing (IMC) systems.

-> 간접적으로 관련되어 있으며, 스포츠 영상 처리를 위한 엣지 디바이스에 적용될 수 있는 AI 최적화 기술을 다룹니다.

## 39위: FutureVLA: Joint Visuomotor Prediction for Vision-Language-Action Model

- arXiv: http://arxiv.org/abs/2603.10712v1 | 2026-03-11 | final 66.4

Predictive foresight is important to intelligent embodied agents. Since the motor execution of a robot is intrinsically constrained by its visual perception of environmental geometry, effectively anticipating the future requires capturing this tightly coupled visuomotor interplay.

-> 시각-운동 예측 모델은 스포츠 전략 분석에 간접적으로 적용 가능

## 40위: Adaptive Manipulation Potential and Haptic Estimation for Tool-Mediated Interaction

- arXiv: http://arxiv.org/abs/2603.10352v1 | 2026-03-11 | final 66.4

Achieving human-level dexterity in contact-rich, tool-mediated manipulation remains a significant challenge due to visual occlusion and the underdetermined nature of haptic sensing. This paper introduces a parameterized Equilibrium Manifold (EM) as a unified representation for tool-mediated interaction, and develops a closed-loop framework that integrates haptic estimation, online planning, and adaptive stiffness control.

-> Tool-mediated manipulation framework indirectly related to sports interaction analysis

## 41위: WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation

- arXiv: http://arxiv.org/abs/2603.10703v1 | 2026-03-11 | final 66.4

Ensuring accessible pedestrian navigation requires reasoning about both semantic and spatial aspects of complex urban scenes, a challenge that existing Large Vision-Language Models (LVLMs) struggle to meet. Although these models can describe visual content, their lack of explicit grounding leads to object hallucinations and unreliable depth reasoning, limiting their usefulness for accessibility guidance.

-> 비전-언어 모델과 분할(segmentation)을 사용하여 보행자 내비게이션을 다루며, 스포츠 분석에 간접적으로 적용 가능할 수 있음

## 42위: Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in Whole-Slide Image Prognosis

- arXiv: http://arxiv.org/abs/2603.10526v1 | 2026-03-11 | final 66.4

Whole-Slide Images (WSIs) are widely used for estimating the prognosis of cancer patients. Current studies generally follow a cancer-specific learning paradigm.

-> 의료 영상 분석을 위한 지식 전달 방법으로, 스포츠 분석에 개념적으로 적용 가능할 수 있음

## 43위: One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination

- arXiv: http://arxiv.org/abs/2603.10360v1 | 2026-03-11 | final 66.4

Current training-free methods tackle MLLM hallucination with separate strategies: either enhancing visual signals or suppressing text inertia. However, these separate methods are insufficient due to critical trade-offs: simply enhancing vision often fails against strong language prior, while suppressing language can introduce extra image-irrelevant noise.

-> Focuses on reducing hallucinations in multimodal language models, which is indirectly related to visual processing but not specifically to sports filming or edge devices.

## 44위: UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis

- arXiv: http://arxiv.org/abs/2603.10852v1 | 2026-03-11 | final 66.4

Breast ultrasound diagnosis typically proceeds from global lesion localization to local sign assessment and then evidence integration to assign a BI-RADS category and determine benignity or malignancy. Many existing methods rely on end-to-end prediction or provide only weakly grounded evidence, which can miss fine-grained lesion cues and limit auditability and clinical review.

-> 계층적 다중 에이전트 접근법이 스포츠 분석에 참고 가능하지만 의료 진단에 특화

## 45위: CodePercept: Code-Grounded Visual STEM Perception for MLLMs

- arXiv: http://arxiv.org/abs/2603.10757v1 | 2026-03-11 | final 66.4

When MLLMs fail at Science, Technology, Engineering, and Mathematics (STEM) visual reasoning, a fundamental question arises: is it due to perceptual deficiencies or reasoning limitations? Through systematic scaling analysis that independently scales perception and reasoning components, we uncover a critical insight: scaling perception consistently outperforms scaling reasoning.

-> 시각적 인식 방법론이 스포츠 동작 분석에 잠재적으로 적용 가능

## 46위: Learning Adaptive Force Control for Contact-Rich Sample Scraping with Heterogeneous Materials

- arXiv: http://arxiv.org/abs/2603.10979v1 | 2026-03-11 | final 66.4

The increasing demand for accelerated scientific discovery, driven by global challenges, highlights the need for advanced AI-driven robotics. Deploying robotic chemists in human-centric labs is key for the next horizon of autonomous discovery, as complex tasks still demand the dexterity of human scientists.

-> 강화학습을 이용한 적응 제어 프레임워크가 스포츠 전략 분석에 간접적으로 참조 가능

## 47위: On the Learning Dynamics of Two-layer Linear Networks with Label Noise SGD

- arXiv: http://arxiv.org/abs/2603.10397v1 | 2026-03-11 | final 66.4

One crucial factor behind the success of deep learning lies in the implicit bias induced by noise inherent in gradient-based training algorithms. Motivated by empirical observations that training with noisy labels improves model generalization, we delve into the underlying mechanisms behind stochastic gradient descent (SGD) with label noise.

-> 간접적으로 관련되어 있으며, 스포츠 분석 시스템에 사용될 수 있는 AI 훈련 방법을 다룹니다.

## 48위: TopGen: Learning Structural Layouts and Cross-Fields for Quadrilateral Mesh Generation

- arXiv: http://arxiv.org/abs/2603.10606v1 | 2026-03-11 | final 64.0

High-quality quadrilateral mesh generation is a fundamental challenge in computer graphics. Traditional optimization-based methods are often constrained by the topological quality of input meshes and suffer from severe efficiency bottlenecks, frequently becoming computationally prohibitive when handling high-resolution models.

-> Computer graphics techniques that could be tangentially related to image enhancement

## 49위: R4-CGQA: Retrieval-based Vision Language Models for Computer Graphics Image Quality Assessment

- arXiv: http://arxiv.org/abs/2603.10578v1 | 2026-03-11 | final 64.0

Immersive Computer Graphics (CGs) rendering has become ubiquitous in modern daily life. However, comprehensively evaluating CG quality remains challenging for two reasons: First, existing CG datasets lack systematic descriptions of rendering quality; and second existing CG quality assessment methods cannot provide reasonable text-based explanations.

-> 컴퓨터 그래픽 이미지 품질 평가를 위한 비전-언어 모델로, 스포츠 영상 분석에 간접적으로 적용 가능할 수 있음

## 50위: LiTo: Surface Light Field Tokenization

- arXiv: http://arxiv.org/abs/2603.11047v1 | 2026-03-11 | final 64.0

We propose a 3D latent representation that jointly models object geometry and view-dependent appearance. Most prior works focus on either reconstructing 3D geometry or predicting view-independent diffuse appearance, and thus struggle to capture realistic view-dependent effects.

-> 3D 객체의 기하학적 구조와 시각적 외관을 모델링하는 방법으로, 스포츠 영상 분석에 개념적으로 적용 가능할 수 있음

## 51위: Bioinspired CNNs for border completion in occluded images

- arXiv: http://arxiv.org/abs/2603.10694v1 | 2026-03-11 | final 64.0

We exploit the mathematical modeling of the border completion problem in the visual cortex to design convolutional neural network (CNN) filters that enhance robustness to image occlusions. We evaluate our CNN architecture, BorderNet, on three occluded datasets (MNIST, Fashion-MNIST, and EMNIST) under two types of occlusions: stripes and grids.

-> Image processing techniques for handling occlusions could be applicable to sports filming where players or objects might be blocked from view.

## 52위: COMIC: Agentic Sketch Comedy Generation

- arXiv: http://arxiv.org/abs/2603.11048v1 | 2026-03-11 | final 64.0

We propose a fully automated AI system that produces short comedic videos similar to sketch shows such as Saturday Night Live. Starting with character references, the system employs a population of agents loosely based on real production studio roles, structured to optimize the quality and diversity of ideas and outputs through iterative competition, evaluation, and improvement.

-> 코미디 영상 생성 기술로 스포츠 하이라이트 편집에 간접적으로 참조 가능

## 53위: Human Presence Detection via Wi-Fi Range-Filtered Doppler Spectrum on Commodity Laptops

- arXiv: http://arxiv.org/abs/2603.10845v1 | 2026-03-11 | final 64.0

Human Presence Detection (HPD) is key to enable intelligent power management and security features in everyday devices. In this paper we propose the first HPD solution that leverages monostatic Wi-Fi sensing and detects user position using only the built-in Wi-Fi hardware of a device, with no need for external devices, access points, or additional sensors.

-> Edge device presence detection with real-time processing

## 54위: RandMark: On Random Watermarking of Visual Foundation Models

- arXiv: http://arxiv.org/abs/2603.10695v1 | 2026-03-11 | final 64.0

Being trained on large and diverse datasets, visual foundation models (VFMs) can be fine-tuned to achieve remarkable performance and efficiency in various downstream computer vision tasks. The high computational cost of data collection and training makes these models valuable assets, which motivates some VFM owners to distribute them alongside a license to protect their intellectual property rights.

-> 시각적 기반 모델 워터마킹이 스포츠 영상 보호용 AI 콘텐츠와 간접적으로 관련

## 55위: End-to-End Chatbot Evaluation with Adaptive Reasoning and Uncertainty Filtering

- arXiv: http://arxiv.org/abs/2603.10570v1 | 2026-03-11 | final 64.0

Large language models (LLMs) combined with retrieval augmented generation have enabled the deployment of domain-specific chatbots, but these systems remain prone to generating unsupported or incorrect answers. Reliable evaluation is therefore critical, yet manual review is costly and existing frameworks often depend on curated test sets and static metrics, limiting scalability.

-> LLM을 이용한 응답 평가 방식이 스포츠 장면 분석에 간접적으로 참조 가능

## 56위: Deep Randomized Distributed Function Computation (DeepRDFC): Neural Distributed Channel Simulation

- arXiv: http://arxiv.org/abs/2603.10750v1 | 2026-03-11 | final 64.0

The randomized distributed function computation (RDFC) framework, which unifies many cutting-edge distributed computation and learning applications, is considered. An autoencoder (AE) architecture is proposed to minimize the total variation distance between the probability distribution simulated by the AE outputs and an unknown target distribution, using only data samples.

-> 간접적으로 관련되어 있으며, 스포츠 분석 시스템에 적용될 수 있는 분산 컴퓨팅을 다룹니다.

## 57위: AI-Generated Rubric Interfaces: K-12 Teachers' Perceptions and Practices

- arXiv: http://arxiv.org/abs/2603.10773v1 | 2026-03-11 | final 64.0

This study investigates K--12 teachers' perceptions and experiences with AI-supported rubric generation during a summer professional development workshop ($n = 25$). Teachers used MagicSchool.ai to generate rubrics and practiced prompting to tailor criteria and performance levels.

-> 간접적으로 관련되어 있으며, 스포츠 분석과 개념적으로 유사한 교육 분야 AI 애플리케이션을 다룹니다.

## 58위: A dataset of medication images with instance segmentation masks for preventing adverse drug events

- arXiv: http://arxiv.org/abs/2603.10825v1 | 2026-03-11 | final 56.0

Medication errors and adverse drug events (ADEs) pose significant risks to patient safety, often arising from difficulties in reliably identifying pharmaceuticals in real-world settings. AI-based pill recognition models offer a promising solution, but the lack of comprehensive datasets hinders their development.

-> 의약품 이미지 분석에 초점을 맞춰 스포츠 프로젝트와는 간접적 관련성만 있음

## 59위: MUNIChus: Multilingual News Image Captioning Benchmark

- arXiv: http://arxiv.org/abs/2603.10613v1 | 2026-03-11 | final 56.0

The goal of news image captioning is to generate captions by integrating news article content with corresponding images, highlighting the relationship between textual context and visual elements. The majority of research on news image captioning focuses on English, primarily because datasets in other languages are scarce.

-> 뉴스 이미지 캡셔닝 벤치마크로 스포츠 분석과 간접적 관련성만 있음

## 60위: Hybrid Self-evolving Structured Memory for GUI Agents

- arXiv: http://arxiv.org/abs/2603.10291v1 | 2026-03-11 | final 54.0

The remarkable progress of vision-language models (VLMs) has enabled GUI agents to interact with computers in a human-like manner. Yet real-world computer-use tasks remain difficult due to long-horizon workflows, diverse interfaces, and frequent intermediate errors.

-> GUI 에이전트 메모리 시스템으로 스포츠 분석에 간접적으로 관련

---

## 다시 보기

### Chain of Event-Centric Causal Thought for Physically Plausible Video Generation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09094v1
- 점수: final 100.0

Physically Plausible Video Generation (PPVG) has emerged as a promising avenue for modeling real-world physical phenomena. PPVG requires an understanding of commonsense knowledge, which remains a challenge for video diffusion models. Current approaches leverage commonsense reasoning capability of large language models to embed physical concepts into prompts. However, generation models often render physical phenomena as a single moment defined by prompts, due to the lack of conditioning mechanisms for modeling causal progression. In this paper, we view PPVG as generating a sequence of causally connected and dynamically evolving events. To realize this paradigm, we design two key modules: (1) Physics-driven Event Chain Reasoning. This module decomposes the physical phenomena described in prompts into multiple elementary event units, leveraging chain-of-thought reasoning. To mitigate causal ambiguity, we embed physical formulas as constraints to impose deterministic causal dependencies during reasoning. (2) Transition-aware Cross-modal Prompting (TCP). To maintain continuity between events, this module transforms causal event units into temporally aligned vision-language prompts. It summarizes discrete event descriptions to obtain causally consistent narratives, while progressively synthesizing visual keyframes of individual events by interactive editing. Comprehensive experiments on PhyGenBench and VideoPhy benchmarks demonstrate that our framework achieves superior performance in generating physically plausible videos across diverse physical domains. Our code will be released soon.

-> 물리적으로 타당한 영상 생성 기술이 스포츠 하이라이트 자동 편집 및 보정에 직접적으로 적용 가능

### Streaming Autoregressive Video Generation via Diagonal Distillation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09488v1
- 점수: final 100.0

Large pretrained diffusion models have significantly enhanced the quality of generated videos, and yet their use in real-time streaming remains limited. Autoregressive models offer a natural framework for sequential frame synthesis but require heavy computation to achieve high fidelity. Diffusion distillation can compress these models into efficient few-step variants, but existing video distillation approaches largely adapt image-specific methods that neglect temporal dependencies. These techniques often excel in image generation but underperform in video synthesis, exhibiting reduced motion coherence, error accumulation over long sequences, and a latency-quality trade-off. We identify two factors that result in these limitations: insufficient utilization of temporal context during step reduction and implicit prediction of subsequent noise levels in next-chunk prediction (i.e., exposure bias). To address these issues, we propose Diagonal Distillation, which operates orthogonally to existing approaches and better exploits temporal information across both video chunks and denoising steps. Central to our approach is an asymmetric generation strategy: more steps early, fewer steps later. This design allows later chunks to inherit rich appearance information from thoroughly processed early chunks, while using partially denoised chunks as conditional inputs for subsequent synthesis. By aligning the implicit prediction of subsequent noise levels during chunk generation with the actual inference conditions, our approach mitigates error propagation and reduces oversaturation in long-range sequences. We further incorporate implicit optical flow modeling to preserve motion quality under strict step constraints. Our method generates a 5-second video in 2.61 seconds (up to 31 FPS), achieving a 277.3x speedup over the undistilled model.

-> 물리적으로 타당한 스포츠 하이라이트 자동 생성 기술로 경기 장면의 인과적 연결을 유지하며 자연스러운 편집 가능

### CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09418v1
- 점수: final 98.4

State-of-the-art whole-body pose estimators often lack robustness, producing anatomically implausible predictions in challenging scenes. We posit this failure stems from spurious correlations learned from visual context, a problem we formalize using a Structural Causal Model (SCM). The SCM identifies visual context as a confounder that creates a non-causal backdoor path, corrupting the model's reasoning. We introduce the Causal Intervention Graph Pose (CIGPose) framework to address this by approximating the true causal effect between visual evidence and pose. The core of CIGPose is a novel Causal Intervention Module: it first identifies confounded keypoint representations via predictive uncertainty and then replaces them with learned, context-invariant canonical embeddings. These deconfounded embeddings are processed by a hierarchical graph neural network that reasons over the human skeleton at both local and global semantic levels to enforce anatomical plausibility. Extensive experiments show CIGPose achieves a new state-of-the-art on COCO-WholeBody. Notably, our CIGPose-x model achieves 67.0\% AP, surpassing prior methods that rely on extra training data. With the additional UBody dataset, CIGPose-x is further boosted to 67.5\% AP, demonstrating superior robustness and data efficiency. The codes and models are publicly available at https://github.com/53mins/CIGPose.

-> 전신 자세 추정 기술은 스포츠 동작 분석에 직접적으로 활용 가능한 핵심 기술이다

### PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09216v1
- 점수: final 96.0

On-device deployments of large language models (LLMs) are rapidly proliferating across mobile and edge platforms. LLM inference comprises a compute-intensive prefill phase and a memory bandwidth-intensive decode phase, and the decode phase has been widely recognized as well-suited to processing-in-memory (PIM) in both academia and industry. However, practical PIM-enabled systems face two obstacles between these phases, a memory attribute inconsistency in which prefill favors placing weights in a cacheable region for reuse whereas decode requires weights in a non-cacheable region to reliably trigger PIM, and a weight layout inconsistency between host-friendly and PIM-aware layouts. To address these problems, we introduce \textit{PIM-SHERPA}, a software-only method for efficient on-device LLM inference by resolving PIM memory attribute and layout inconsistencies. PIM-SHERPA provides two approaches, DRAM double buffering (DDB), which keeps a single PIM-aware weights in the non-cacheable region while prefetching the swizzled weights of the next layer into small cacheable buffers, and online weight rearrangement with swizzled memory copy (OWR), which performs the on-demand swizzled memory copy immediately before GEMM. Compared to a baseline PIM emulation system, PIM-SHERPA achieves approximately 47.8 - 49.7\% memory capacity savings while maintaining comparable performance to the theoretical maximum on the Llama 3.2 model. To the best of our knowledge, this is the first work to identify the memory attribute inconsistency and propose effective solutions on product-level PIM-enabled systems.

-> rk3588 엣지 디바이스에서의 효율적인 AI 추론을 위한 메모리 최적화 방법으로 실시간 스포츠 분석에 적용 가능

### CycleULM: A unified label-free deep learning framework for ultrasound localisation microscopy (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09840v1
- 점수: final 96.0

Super-resolution ultrasound via microbubble (MB) localisation and tracking, also known as ultrasound localisation microscopy (ULM), can resolve microvasculature beyond the acoustic diffraction limit. However, significant challenges remain in localisation performance and data acquisition and processing time. Deep learning methods for ULM have shown promise to address these challenges, however, they remain limited by in vivo label scarcity and the simulation-to-reality domain gap. We present CycleULM, the first unified label-free deep learning framework for ULM. CycleULM learns a physics-emulating translation between the real contrast-enhanced ultrasound (CEUS) data domain and a simplified MB-only domain, leveraging the power of CycleGAN without requiring paired ground truth data. With this translation, CycleULM removes dependence on high-fidelity simulators or labelled data, and makes MB localisation and tracking substantially easier. Deployed as modular plug-and-play components within existing pipelines or as an end-to-end processing framework, CycleULM delivers substantial performance gains across both in silico and in vivo datasets. Specifically, CycleULM improves image contrast (contrast-to-noise ratio) by up to 15.3 dB and sharpens CEUS resolution with a 2.5{\times} reduction in the full width at half maximum of the point spread function. CycleULM also improves MB localisation performance, with up to +40% recall, +46% precision, and a -14.0 μm mean localisation error, yielding more faithful vascular reconstructions. Importantly, CycleULM achieves real-time processing throughput at 18.3 frames per second with order-of-magnitude speed-ups (up to ~14.5{\times}). By combining label-free learning, performance enhancement, and computational efficiency, CycleULM provides a practical pathway toward robust, real-time ULM and accelerates its translation to clinical applications.

-> 실시간 처리 성능과 물리 모델링 기술이 스포츠 영상 분석에 직접적으로 적용 가능

### TrainDeeploy: Hardware-Accelerated Parameter-Efficient Fine-Tuning of Small Transformer Models at the Extreme Edge (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09511v1
- 점수: final 96.0

On-device tuning of deep neural networks enables long-term adaptation at the edge while preserving data privacy. However, the high computational and memory demands of backpropagation pose significant challenges for ultra-low-power, memory-constrained extreme-edge devices. These challenges are further amplified for attention-based models due to their architectural complexity and computational scale. We present TrainDeeploy, a framework that unifies efficient inference and on-device training on heterogeneous ultra-low-power System-on-Chips (SoCs). TrainDeeploy provides the first complete on-device training pipeline for extreme-edge SoCs supporting both Convolutional Neural Networks (CNNs) and Transformer models, together with multiple training strategies such as selective layer-wise fine-tuning and Low-Rank Adaptation (LoRA). On a RISC-V-based heterogeneous SoC, we demonstrate the first end-to-end on-device fine-tuning of a Compact Convolutional Transformer (CCT), achieving up to 11 trained images per second. We show that LoRA reduces dynamic memory usage by 23%, decreases the number of trainable parameters and gradients by 15x, and reduces memory transfer volume by 1.6x compared to full backpropagation. TrainDeeploy achieves up to 4.6 FLOP/cycle on CCT (0.28M parameters, 71-126M FLOPs) and up to 13.4 FLOP/cycle on Deep-AE (0.27M parameters, 0.8M FLOPs), while expanding the scope of prior frameworks to support both CNN and Transformer models with parameter-efficient tuning on extreme-edge platforms.

-> rk3588 기반 edge device에서 AI 모델 효율적 튜닝을 위한 하드웨어 가속 기술

### SURE: Semi-dense Uncertainty-REfined Feature Matching (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04869v1
- 점수: final 96.0

Establishing reliable image correspondences is essential for many robotic vision problems. However, existing methods often struggle in challenging scenarios with large viewpoint changes or textureless regions, where incorrect cor- respondences may still receive high similarity scores. This is mainly because conventional models rely solely on fea- ture similarity, lacking an explicit mechanism to estimate the reliability of predicted matches, leading to overconfident errors. To address this issue, we propose SURE, a Semi- dense Uncertainty-REfined matching framework that jointly predicts correspondences and their confidence by modeling both aleatoric and epistemic uncertainties. Our approach in- troduces a novel evidential head for trustworthy coordinate regression, along with a lightweight spatial fusion module that enhances local feature precision with minimal overhead. We evaluated our method on multiple standard benchmarks, where it consistently outperforms existing state-of-the-art semi-dense matching models in both accuracy and efficiency. our code will be available on https://github.com/LSC-ALAN/SURE.

-> 시각적 표현 향상 기술은 스포츠 영상 보정 및 하이라이트 편집에 직접적으로 적용 가능하며, 자세 및 동작 분석에도 활용될 수 있습니다.

### MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09116v1
- 점수: final 93.6

We present MetaSpectra+, a compact multifunctional camera that supports two operating modes: (1) snapshot HDR + hyperspectral or (2) snapshot polarization + hyperspectral imaging. It utilizes a novel metasurface-refractive assembly that splits the incident beam into multiple channels and independently controls each channel's dispersion, exposure, and polarization. Unlike prior multifunctional metasurface imagers restricted to narrow (10-100 nm) bands, MetaSpectra+ operates over nearly the entire visible spectrum (250 nm). Relative to snapshot hyperspectral imagers, it achieves the shortest total track length and the highest reconstruction accuracy on benchmark datasets. The demonstrated prototype reconstructs high-quality hyperspectral datacubes and either an HDR image or two orthogonal polarization channels from a single snapshot.

-> 다중 기능 컴팩트 카메라 기술이 스포츠 촬영 및 분석에 적용 가능

### Decoder-Free Distillation for Quantized Image Restoration (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09624v1
- 점수: final 93.6

Quantization-Aware Training (QAT), combined with Knowledge Distillation (KD), holds immense promise for compressing models for edge deployment. However, joint optimization for precision-sensitive image restoration (IR) to recover visual quality from degraded images remains largely underexplored. Directly adapting QAT-KD to low-level vision reveals three critical bottlenecks: teacher-student capacity mismatch, spatial error amplification during decoder distillation, and an optimization "tug-of-war" between reconstruction and distillation losses caused by quantization noise. To tackle these, we introduce Quantization-aware Distilled Restoration (QDR), a framework for edge-deployed IR. QDR eliminates capacity mismatch via FP32 self-distillation and prevents error amplification through Decoder-Free Distillation (DFD), which corrects quantization errors strictly at the network bottleneck. To stabilize the optimization tug-of-war, we propose a Learnable Magnitude Reweighting (LMR) that dynamically balances competing gradients. Finally, we design an Edge-Friendly Model (EFM) featuring a lightweight Learnable Degradation Gating (LDG) to dynamically modulate spatial degradation localization. Extensive experiments across four IR tasks demonstrate that our Int8 model recovers 96.5% of FP32 performance, achieves 442 frames per second (FPS) on an NVIDIA Jetson Orin, and boosts downstream object detection by 16.3 mAP

-> 실시간 처리가 가능한 엣지 기반 이미지 복원 기술로 스포츠 촬영 장비에 직접 적용 가능

### Trainable Bitwise Soft Quantization for Input Feature Compression (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.05172v1
- 점수: final 93.6

The growing demand for machine learning applications in the context of the Internet of Things calls for new approaches to optimize the use of limited compute and memory resources. Despite significant progress that has been made w.r.t. reducing model sizes and improving efficiency, many applications still require remote servers to provide the required resources. However, such approaches rely on transmitting data from edge devices to remote servers, which may not always be feasible due to bandwidth, latency, or energy constraints. We propose a task-specific, trainable feature quantization layer that compresses the input features of a neural network. This can significantly reduce the amount of data that needs to be transferred from the device to a remote server. In particular, the layer allows each input feature to be quantized to a user-defined number of bits, enabling a simple on-device compression at the time of data collection. The layer is designed to approximate step functions with sigmoids, enabling trainable quantization thresholds. By concatenating outputs from multiple sigmoids, introduced as bitwise soft quantization, it achieves trainable quantized values when integrated with a neural network. We compare our method to full-precision inference as well as to several quantization baselines. Experiments show that our approach outperforms standard quantization methods, while maintaining accuracy levels close to those of full-precision models. In particular, depending on the dataset, compression factors of $5\times$ to $16\times$ can be achieved compared to $32$-bit input without significant performance loss.

-> RK3588 엣지 장치에서 데이터 전송량을 최대 16배 줄여 실시간 처리와 배터리 효율성을 크게 향상시킬 수 있습니다.

### Person Detection and Tracking from an Overhead Crane LiDAR (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04938v1
- 점수: final 90.4

This paper investigates person detection and tracking in an industrial indoor workspace using a LiDAR mounted on an overhead crane. The overhead viewpoint introduces a strong domain shift from common vehicle-centric LiDAR benchmarks, and limited availability of suitable public training data. Henceforth, we curate a site-specific overhead LiDAR dataset with 3D human bounding-box annotations and adapt selected candidate 3D detectors under a unified training and evaluation protocol. We further integrate lightweight tracking-by-detection using AB3DMOT and SimpleTrack to maintain person identities over time. Detection performance is reported with distance-sliced evaluation to quantify the practical operating envelope of the sensing setup. The best adapted detector configurations achieve average precision (AP) up to 0.84 within a 5.0 m horizontal radius, increasing to 0.97 at 1.0 m, with VoxelNeXt and SECOND emerging as the most reliable backbones across this range. The acquired results contribute in bridging the domain gap between standard driving datasets and overhead sensing for person detection and tracking. We also report latency measurements, highlighting practical real-time feasibility. Finally, we release our dataset and implementations in GitHub to support further research

-> 산업 현장의 LiDAR 기반 사람 감지 및 추적 기술로 스포츠 촬영과 유사한 추적 기술이 적용 가능하나 도메인이 상이함

### VisionPangu: A Compact and Fine-Grained Multimodal Assistant with 1.7B Parameters (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04957v1
- 점수: final 90.4

Large Multimodal Models (LMMs) have achieved strong performance in vision-language understanding, yet many existing approaches rely on large-scale architectures and coarse supervision, which limits their ability to generate detailed image captions. In this work, we present VisionPangu, a compact 1.7B-parameter multimodal model designed to improve detailed image captioning through efficient multimodal alignment and high-quality supervision. Our model combines an InternVL-derived vision encoder with the OpenPangu-Embedded language backbone via a lightweight MLP projector and adopts an instruction-tuning pipeline inspired by LLaVA. By incorporating dense human-authored descriptions from the DOCCI dataset, VisionPangu improves semantic coherence and descriptive richness without relying on aggressive model scaling. Experimental results demonstrate that compact multimodal models can achieve competitive performance while producing more structured and detailed captions. The code and model weights will be publicly available at https://www.modelscope.cn/models/asdfgh007/visionpangu.

-> 1.7B 파라미터로 경량화된 멀티모달 모델이 엣지 장치에서 스포츠 장면의 상세한 이해를 가능하게 합니다.

### Two Teachers Better Than One: Hardware-Physics Co-Guided Distributed Scientific Machine Learning (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09032v1
- 점수: final 90.0

Scientific machine learning (SciML) is increasingly applied to in-field processing, controlling, and monitoring; however, wide-area sensing, real-time demands, and strict energy and reliability constraints make centralized SciML implementation impractical. Most SciML models assume raw data aggregation at a central node, incurring prohibitively high communication latency and energy costs; yet, distributing models developed for general-purpose ML often breaks essential physical principles, resulting in degraded performance. To address these challenges, we introduce EPIC, a hardware- and physics-co-guided distributed SciML framework, using full-waveform inversion (FWI) as a representative task. EPIC performs lightweight local encoding on end devices and physics-aware decoding at a central node. By transmitting compact latent features rather than high-volume raw data and by using cross-attention to capture inter-receiver wavefield coupling, EPIC significantly reduces communication cost while preserving physical fidelity. Evaluated on a distributed testbed with five end devices and one central node, and across 10 datasets from OpenFWI, EPIC reduces latency by 8.9$\times$ and communication energy by 33.8$\times$, while even improving reconstruction fidelity on 8 out of 10 datasets.

-> 엣지 디바이스용 분산 컴퓨팅 프레임워크로 물리 인식 처리가 스포츠 분석에 적용 가능

### TemporalDoRA: Temporal PEFT for Robust Surgical Video Question Answering (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09696v1
- 점수: final 88.8

Surgical Video Question Answering (VideoQA) requires accurate temporal grounding while remaining robust to natural variation in how clinicians phrase questions, where linguistic bias can arise. Standard Parameter Efficient Fine Tuning (PEFT) methods adapt pretrained projections without explicitly modeling frame-to-frame interactions within the adaptation pathway, limiting their ability to exploit sparse temporal evidence. We introduce TemporalDoRA, a video-specific PEFT formulation that extends Weight-Decomposed Low-Rank Adaptation by (i) inserting lightweight temporal Multi-Head Attention (MHA) inside the low-rank bottleneck of the vision encoder and (ii) selectively applying weight decomposition only to the trainable low-rank branch rather than the full adapted weight. This design enables temporally-aware updates while preserving a frozen backbone and stable scaling. By mixing information across frames within the adaptation subspace, TemporalDoRA steers updates toward temporally consistent visual cues and improves robustness with minimal parameter overhead. To benchmark this setting, we present REAL-Colon-VQA, a colonoscopy VideoQA dataset with 6,424 clip--question pairs, including paired rephrased Out-of-Template questions to evaluate sensitivity to linguistic variation. TemporalDoRA improves Out-of-Template performance, and ablation studies confirm that temporal mixing inside the low-rank branch is the primary driver of these gains. We also validate on EndoVis18-VQA adapted to short clips and observe consistent improvements on the Out-of-Template split. Code and dataset available at~\href{https://anonymous.4open.science/r/TemporalDoRA-BFC8/}{Anonymous GitHub}.

-> 수술 비디오 QA를 위한 시간적 처리 기술은 스포츠 분석으로 적용될 수 있습니다

### M3GCLR: Multi-View Mini-Max Infinite Skeleton-Data Game Contrastive Learning For Skeleton-Based Action Recognition (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09367v1
- 점수: final 88.0

In recent years, contrastive learning has drawn significant attention as an effective approach to reducing reliance on labeled data. However, existing methods for self-supervised skeleton-based action recognition still face three major limitations: insufficient modeling of view discrepancies, lack of effective adversarial mechanisms, and uncontrollable augmentation perturbations. To tackle these issues, we propose the Multi-view Mini-Max infinite skeleton-data Game Contrastive Learning for skeleton-based action Recognition (M3GCLR), a game-theoretic contrastive framework. First, we establish the Infinite Skeleton-data Game (ISG) model and the ISG equilibrium theorem, and further provide a rigorous proof, enabling mini-max optimization based on multi-view mutual information. Then, we generate normal-extreme data pairs through multi-view rotation augmentation and adopt temporally averaged input as a neutral anchor to achieve structural alignment, thereby explicitly characterizing perturbation strength. Next, leveraging the proposed equilibrium theorem, we construct a strongly adversarial mini-max skeleton-data game to encourage the model to mine richer action-discriminative information. Finally, we introduce the dual-loss equilibrium optimizer to optimize the game equilibrium, allowing the learning process to maximize action-relevant information while minimizing encoding redundancy, and we prove the equivalence between the proposed optimizer and the ISG model. Extensive Experiments show that M3GCLR achieves three-stream 82.1%, 85.8% accuracy on NTU RGB+D 60 (X-Sub, X-View) and 72.3%, 75.0% accuracy on NTU RGB+D 120 (X-Sub, X-Set). On PKU-MMD Part I and II, it attains 89.1%, 45.2% in three-stream respectively, all results matching or outperforming state-of-the-art performance. Ablation studies confirm the effectiveness of each component.

-> 골격 기반 동작 인식 기술이 스포츠 동작 분석에 직접적으로 적용 가능하여 선수 기술 분석과 자세 교정에 활용 가능

### NS-VLA: Towards Neuro-Symbolic Vision-Language-Action Models (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09542v1
- 점수: final 88.0

Vision-Language-Action (VLA) models are formulated to ground instructions in visual context and generate action sequences for robotic manipulation. Despite recent progress, VLA models still face challenges in learning related and reusable primitives, reducing reliance on large-scale data and complex architectures, and enabling exploration beyond demonstrations. To address these challenges, we propose a novel Neuro-Symbolic Vision-Language-Action (NS-VLA) framework via online reinforcement learning (RL). It introduces a symbolic encoder to embedding vision and language features and extract structured primitives, utilizes a symbolic solver for data-efficient action sequencing, and leverages online RL to optimize generation via expansive exploration. Experiments on robotic manipulation benchmarks demonstrate that NS-VLA outperforms previous methods in both one-shot training and data-perturbed settings, while simultaneously exhibiting superior zero-shot generalizability, high data efficiency and expanded exploration space. Our code is available.

-> 멀티모달 파싱 기술이 스포츠 영상에서 의미 있는 정보를 추출하고 구조화된 지식으로 변환하여 전략 분석에 활용 가능

### WikiCLIP: An Efficient Contrastive Baseline for Open-domain Visual Entity Recognition (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09921v1
- 점수: final 88.0

Open-domain visual entity recognition (VER) seeks to associate images with entities in encyclopedic knowledge bases such as Wikipedia. Recent generative methods tailored for VER demonstrate strong performance but incur high computational costs, limiting their scalability and practical deployment. In this work, we revisit the contrastive paradigm for VER and introduce WikiCLIP, a simple yet effective framework that establishes a strong and efficient baseline for open-domain VER. WikiCLIP leverages large language model embeddings as knowledge-rich entity representations and enhances them with a Vision-Guided Knowledge Adaptor (VGKA) that aligns textual semantics with visual cues at the patch level. To further encourage fine-grained discrimination, a Hard Negative Synthesis Mechanism generates visually similar but semantically distinct negatives during training. Experimental results on popular open-domain VER benchmarks, such as OVEN, demonstrate that WikiCLIP significantly outperforms strong baselines. Specifically, WikiCLIP achieves a 16% improvement on the challenging OVEN unseen set, while reducing inference latency by nearly 100 times compared with the leading generative model, AutoVER. The project page is available at https://artanic30.github.io/project_pages/WikiCLIP/

-> 효율적인 대비 학습 방식으로 스포츠 장면의 시각적 엔티티 인식에 적용 가능

### A Multi-Prototype-Guided Federated Knowledge Distillation Approach in AI-RAN Enabled Multi-Access Edge Computing System (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09727v1
- 점수: final 86.4

With the development of wireless network, Multi-Access Edge Computing (MEC) and Artificial Intelligence (AI)-native Radio Access Network (RAN) have attracted significant attention. Particularly, the integration of AI-RAN and MEC is envisioned to transform network efficiency and responsiveness. Therefore, it is valuable to investigate AI-RAN enabled MEC system. Federated learning (FL) nowadays is emerging as a promising approach for AI-RAN enabled MEC system, in which edge devices are enabled to train a global model cooperatively without revealing their raw data. However, conventional FL encounters the challenge in processing the non-independent and identically distributed (non-IID) data. Single prototype obtained by averaging the embedding vectors per class can be employed in FL to handle the data heterogeneity issue. Nevertheless, this may result in the loss of useful information owing to the average operation. Therefore, in this paper, a multi-prototype-guided federated knowledge distillation (MP-FedKD) approach is proposed. Particularly, self-knowledge distillation is integrated into FL to deal with the non-IID issue. To cope with the problem of information loss caused by single prototype-based strategy, multi-prototype strategy is adopted, where we present a conditional hierarchical agglomerative clustering (CHAC) approach and a prototype alignment scheme. Additionally, we design a novel loss function (called LEMGP loss) for each local client, where the relationship between global prototypes and local embedding will be focused. Extensive experiments over multiple datasets with various non-IID settings showcase that the proposed MP-FedKD approach outperforms the considered state-of-the-art baselines regarding accuracy, average accuracy and errors (RMSE and MAE).

-> 엣지 컴퓨팅 시스템을 위한 연합 학습 접근법이 AI 카메라 장치에 직접 적용 가능

### Logics-Parsing-Omni Technical Report (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09677v1
- 점수: final 86.4

Addressing the challenges of fragmented task definitions and the heterogeneity of unstructured data in multimodal parsing, this paper proposes the Omni Parsing framework. This framework establishes a Unified Taxonomy covering documents, images, and audio-visual streams, introducing a progressive parsing paradigm that bridges perception and cognition. Specifically, the framework integrates three hierarchical levels: 1) Holistic Detection, which achieves precise spatial-temporal grounding of objects or events to establish a geometric baseline for perception; 2) Fine-grained Recognition, which performs symbolization (e.g., OCR/ASR) and attribute extraction on localized objects to complete structured entity parsing; and 3) Multi-level Interpreting, which constructs a reasoning chain from local semantics to global logic. A pivotal advantage of this framework is its evidence anchoring mechanism, which enforces a strict alignment between high-level semantic descriptions and low-level facts. This enables ``evidence-based'' logical induction, transforming unstructured signals into standardized knowledge that is locatable, enumerable, and traceable. Building on this foundation, we constructed a standardized dataset and released the Logics-Parsing-Omni model, which successfully converts complex audio-visual signals into machine-readable structured knowledge. Experiments demonstrate that fine-grained perception and high-level cognition are synergistic, effectively enhancing model reliability. Furthermore, to quantitatively evaluate these capabilities, we introduce OmniParsingBench. Code, models and the benchmark are released at https://github.com/alibaba/Logics-Parsing/tree/master/Logics-Parsing-Omni.

-> Multimodal parsing techniques could be applicable for analyzing sports footage and extracting meaningful information.

### Guiding Diffusion-based Reconstruction with Contrastive Signals for Balanced Visual Representation (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04803v1
- 점수: final 86.4

The limited understanding capacity of the visual encoder in Contrastive Language-Image Pre-training (CLIP) has become a key bottleneck for downstream performance. This capacity includes both Discriminative Ability (D-Ability), which reflects class separability, and Detail Perceptual Ability (P-Ability), which focuses on fine-grained visual cues. Recent solutions use diffusion models to enhance representations by conditioning image reconstruction on CLIP visual tokens. We argue that such paradigms may compromise D-Ability and therefore fail to effectively address CLIP's representation limitations. To address this, we integrate contrastive signals into diffusion-based reconstruction to pursue more comprehensive visual representations. We begin with a straightforward design that augments the diffusion process with contrastive learning on input images. However, empirical results show that the naive combination suffers from gradient conflict and yields suboptimal performance. To balance the optimization, we introduce the Diffusion Contrastive Reconstruction (DCR), which unifies the learning objective. The key idea is to inject contrastive signals derived from each reconstructed image, rather than from the original input, into the diffusion process. Our theoretical analysis shows that the DCR loss can jointly optimize D-Ability and P-Ability. Extensive experiments across various benchmarks and multi-modal large language models validate the effectiveness of our method. The code is available at https://github.com/boyuh/DCR.

-> 일반적인 시각적 표현 기술이 스포츠 영상 분석에 적용 가능

### Improving 3D Foot Motion Reconstruction in Markerless Monocular Human Motion Capture (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09681v1
- 점수: final 85.6

State-of-the-art methods can recover accurate overall 3D human body motion from in-the-wild videos. However, they often fail to capture fine-grained articulations, especially in the feet, which are critical for applications such as gait analysis and animation. This limitation results from training datasets with inaccurate foot annotations and limited foot motion diversity. We address this gap with FootMR, a Foot Motion Refinement method that refines foot motion estimated by an existing human recovery model through lifting 2D foot keypoint sequences to 3D. By avoiding direct image input, FootMR circumvents inaccurate image-3D annotation pairs and can instead leverage large-scale motion capture data. To resolve ambiguities of 2D-to-3D lifting, FootMR incorporates knee and foot motion as context and predicts only residual foot motion. Generalization to extreme foot poses is further improved by representing joints in global rather than parent-relative rotations and applying extensive data augmentation. To support evaluation of foot motion reconstruction, we introduce MOOF, a 2D dataset of complex foot movements. Experiments on MOOF, MOYO, and RICH show that FootMR outperforms state-of-the-art methods, reducing ankle joint angle error on MOYO by up to 30% over the best video-based approach.

-> 멀티모달 인터리브 생성 기술은 스포츠 하이라이트 영상 제작에 적용 가능하여 콘텐츠 생성 효율성을 높일 수 있습니다.

### Scalable Injury-Risk Screening in Baseball Pitching From Broadcast Video (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04864v1
- 점수: final 85.6

Injury prediction in pitching depends on precise biomechanical signals, yet gold-standard measurements come from expensive, stadium-installed multi-camera systems that are unavailable outside professional venues. We present a monocular video pipeline that recovers 18 clinically relevant biomechanics metrics from broadcast footage, positioning pose-derived kinematics as a scalable source for injury-risk modeling. Built on DreamPose3D, our approach introduces a drift-controlled global lifting module that recovers pelvis trajectory via velocity-based parameterization and sliding-window inference, lifting pelvis-rooted poses into global space. To address motion blur, compression artifacts, and extreme pitching poses, we incorporate a kinematics refinement pipeline with bone-length constraints, joint-limited inverse kinematics, smoothing, and symmetry constraints to ensure temporally stable and physically plausible kinematics. On 13 professional pitchers (156 paired pitches), 16/18 metrics achieve sub-degree agreement (MAE $< 1^{\circ}$). Using these metrics for injury prediction, an automated screening model achieves AUC 0.811 for Tommy John surgery and 0.825 for significant arm injuries on 7,348 pitchers. The resulting pose-derived metrics support scalable injury-risk screening, establishing monocular broadcast video as a viable alternative to stadium-scale motion capture for biomechanics.

-> 야구 투구 자세 분석과 부상 예측에 관한 연구로 스포츠 동작 분석 기술이 적용 가능하나, 특정 스포츠(야구)와 부상 예측에 초점이 맞춰져 있어 일반적인 스포츠 촬영 및 하이라이트 편집과는 직접적 연관성은 떨어집니다.

### When to Lock Attention: Training-Free KV Control in Video Diffusion (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09657v1
- 점수: final 85.6

Maintaining background consistency while enhancing foreground quality remains a core challenge in video editing. Injecting full-image information often leads to background artifacts, whereas rigid background locking severely constrains the model's capacity for foreground generation. To address this issue, we propose KV-Lock, a training-free framework tailored for DiT-based video diffusion models. Our core insight is that the hallucination metric (variance of denoising prediction) directly quantifies generation diversity, which is inherently linked to the classifier-free guidance (CFG) scale. Building upon this, KV-Lock leverages diffusion hallucination detection to dynamically schedule two key components: the fusion ratio between cached background key-values (KVs) and newly generated KVs, and the CFG scale. When hallucination risk is detected, KV-Lock strengthens background KV locking and simultaneously amplifies conditional guidance for foreground generation, thereby mitigating artifacts and improving generation fidelity. As a training-free, plug-and-play module, KV-Lock can be easily integrated into any pre-trained DiT-based models. Extensive experiments validate that our method outperforms existing approaches in improved foreground quality with high background fidelity across various video editing tasks.

-> 비디오 확산 모델을 이용한 영상 편집 기술은 스포츠 영상 보정에 활용 가능하여 배경과 전경을 동시에 개선할 수 있습니다.

### From Ideal to Real: Stable Video Object Removal under Imperfect Conditions (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09283v1
- 점수: final 85.6

Removing objects from videos remains difficult in the presence of real-world imperfections such as shadows, abrupt motion, and defective masks. Existing diffusion-based video inpainting models often struggle to maintain temporal stability and visual consistency under these challenges. We propose Stable Video Object Removal (SVOR), a robust framework that achieves shadow-free, flicker-free, and mask-defect-tolerant removal through three key designs: (1) Mask Union for Stable Erasure (MUSE), a windowed union strategy applied during temporal mask downsampling to preserve all target regions observed within each window, effectively handling abrupt motion and reducing missed removals; (2) Denoising-Aware Segmentation (DA-Seg), a lightweight segmentation head on a decoupled side branch equipped with Denoising-Aware AdaLN and trained with mask degradation to provide an internal diffusion-aware localization prior without affecting content generation; and (3) Curriculum Two-Stage Training: where Stage I performs self-supervised pretraining on unpaired real-background videos with online random masks to learn realistic background and temporal priors, and Stage II refines on synthetic pairs using mask degradation and side-effect-weighted losses, jointly removing objects and their associated shadows/reflections while improving cross-domain robustness. Extensive experiments show that SVOR attains new state-of-the-art results across multiple datasets and degraded-mask benchmarks, advancing video object removal from ideal settings toward real-world applications.

-> 비디오 객체 제거 기술은 스포츠 경기 영상 편집에 적용 가능하여 원치 않는 객체를 제거하거나 특정 선수를 분리하는 데 활용될 수 있습니다.

### Fine-grained Motion Retrieval via Joint-Angle Motion Images and Token-Patch Late Interaction (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09930v1
- 점수: final 84.8

Text-motion retrieval aims to learn a semantically aligned latent space between natural language descriptions and 3D human motion skeleton sequences, enabling bidirectional search across the two modalities. Most existing methods use a dual-encoder framework that compresses motion and text into global embeddings, discarding fine-grained local correspondences, and thus reducing accuracy. Additionally, these global-embedding methods offer limited interpretability of the retrieval results. To overcome these limitations, we propose an interpretable, joint-angle-based motion representation that maps joint-level local features into a structured pseudo-image, compatible with pre-trained Vision Transformers. For text-to-motion retrieval, we employ MaxSim, a token-wise late interaction mechanism, and enhance it with Masked Language Modeling regularization to foster robust, interpretable text-motion alignment. Extensive experiments on HumanML3D and KIT-ML show that our method outperforms state-of-the-art text-motion retrieval approaches while offering interpretable fine-grained correspondences between text and motion. The code is available in the supplementary material.

-> 관절 각도 기반 움직임 검색 기술은 스포츠 하이라이트 생성에 적용 가능하여 특정 동작을 기반으로 영상을 검색하고 편집할 수 있습니다.

### Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.04977v1
- 점수: final 84.8

Long video understanding is challenging due to dense visual redundancy, long-range temporal dependencies, and the tendency of chain-of-thought and retrieval-based agents to accumulate semantic drift and correlation-driven errors. We argue that long-video reasoning should begin not with reactive retrieval, but with deliberate task formulation: the model must first articulate what must be true in the video for each candidate answer to hold. This thinking-before-finding principle motivates VideoHV-Agent, a framework that reformulates video question answering as a structured hypothesis-verification process. Based on video summaries, a Thinker rewrites answer candidates into testable hypotheses, a Judge derives a discriminative clue specifying what evidence must be checked, a Verifier grounds and tests the clue using localized, fine-grained video content, and an Answer agent integrates validated evidence to produce the final answer. Experiments on three long-video understanding benchmarks show that VideoHV-Agent achieves state-of-the-art accuracy while providing enhanced interpretability, improved logical soundness, and lower computational cost. We make our code publicly available at: https://github.com/Haorane/VideoHV-Agent.

-> 장기 동영상 이해를 위한 가설-검증 다중 에이전트 프레임워크로, 스포츠 영상 분석에 적용 가능하지만 스포츠 특화나 실시간 처리는 다루지 않음.

### AI+HW 2035: Shaping the Next Decade (2회째 추천)

- arXiv: http://arxiv.org/abs/2603.05225v1
- 점수: final 84.0

Artificial intelligence (AI) and hardware (HW) are advancing at unprecedented rates, yet their trajectories have become inseparably intertwined. The global research community lacks a cohesive, long-term vision to strategically coordinate the development of AI and HW. This fragmentation constrains progress toward holistic, sustainable, and adaptive AI systems capable of learning, reasoning, and operating efficiently across cloud, edge, and physical environments. The future of AI depends not only on scaling intelligence, but on scaling efficiency, achieving exponential gains in intelligence per joule, rather than unbounded compute consumption. Addressing this grand challenge requires rethinking the entire computing stack. This vision paper lays out a 10-year roadmap for AI+HW co-design and co-development, spanning algorithms, architectures, systems, and sustainability. We articulate key insights that redefine scaling around energy efficiency, system-level integration, and cross-layer optimization. We identify key challenges and opportunities, candidly assess potential obstacles and pitfalls, and propose integrated solutions grounded in algorithmic innovation, hardware advances, and software abstraction. Looking ahead, we define what success means in 10 years: achieving a 1000x improvement in efficiency for AI training and inference; enabling energy-aware, self-optimizing systems that seamlessly span cloud, edge, and physical AI; democratizing access to advanced AI infrastructure; and embedding human-centric principles into the design of intelligent systems. Finally, we outline concrete action items for academia, industry, government, and the broader community, calling for coordinated national initiatives, shared infrastructure, workforce development, cross-agency collaboration, and sustained public-private partnerships to ensure that AI+HW co-design becomes a unifying long-term mission.

-> AI+HW 공설계 접근법은 스포츠 촬영 및 분석을 위한 에지 디바이스의 효율성과 성능을 극대화하는 데 핵심적입니다. 특히 에너지 효율성 향상과 시스템 통합은 실시간 스포츠 영상 처리에 필수적입니다.

### M2Diff: Multi-Modality Multi-Task Enhanced Diffusion Model for MRI-Guided Low-Dose PET Enhancement (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09075v1
- 점수: final 80.0

Positron emission tomography (PET) scans expose patients to radiation, which can be mitigated by reducing the dose, albeit at the cost of diminished quality. This makes low-dose (LD) PET recovery an active research area. Previous studies have focused on standard-dose (SD) PET recovery from LD PET scans and/or multi-modal scans, e.g., PET/CT or PET/MRI, using deep learning. While these studies incorporate multi-modal information through conditioning in a single-task model, such approaches may limit the capacity to extract modality-specific features, potentially leading to early feature dilution. Although recent studies have begun incorporating pathology-rich data, challenges remain in effectively leveraging multi-modality inputs for reconstructing diverse features, particularly in heterogeneous patient populations. To address these limitations, we introduce a multi-modality multi-task diffusion model (M2Diff) that processes MRI and LD PET scans separately to learn modality-specific features and fuse them via hierarchical feature fusion to reconstruct SD PET. This design enables effective integration of complementary structural and functional information, leading to improved reconstruction fidelity. We have validated the effectiveness of our model on both healthy and Alzheimer's disease brain datasets. The M2Diff achieves superior qualitative and quantitative performance on both datasets.

-> 다중 모달리티 이미지 처리 기술이 스포츠 영상 향상에 활용 가능

### Evolving Prompt Adaptation for Vision-Language Models (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09493v1
- 점수: final 80.0

The adaptation of large-scale vision-language models (VLMs) to downstream tasks with limited labeled data remains a significant challenge. While parameter-efficient prompt learning methods offer a promising path, they often suffer from catastrophic forgetting of pre-trained knowledge. Toward addressing this limitation, our work is grounded in the insight that governing the evolutionary path of prompts is essential for forgetting-free adaptation. To this end, we propose EvoPrompt, a novel framework designed to explicitly steer the prompt trajectory for stable, knowledge-preserving fine-tuning. Specifically, our approach employs a Modality-Shared Prompt Projector (MPP) to generate hierarchical prompts from a unified embedding space. Critically, an evolutionary training strategy decouples low-rank updates into directional and magnitude components, preserving early-learned semantic directions while only adapting their magnitude, thus enabling prompts to evolve without discarding foundational knowledge. This process is further stabilized by Feature Geometric Regularization (FGR), which enforces feature decorrelation to prevent representation collapse. Extensive experiments demonstrate that EvoPrompt achieves state-of-the-art performance in few-shot learning while robustly preserving the original zero-shot capabilities of pre-trained VLMs.

-> Presents vision-language model adaptation techniques applicable to sports analysis but doesn't address edge computing

### Towards Unified Multimodal Interleaved Generation via Group Relative Policy Optimization (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09538v1
- 점수: final 80.0

Unified vision-language models have made significant progress in multimodal understanding and generation, yet they largely fall short in producing multimodal interleaved outputs, which is a crucial capability for tasks like visual storytelling and step-by-step visual reasoning. In this work, we propose a reinforcement learning-based post-training strategy to unlock this capability in existing unified models, without relying on large-scale multimodal interleaved datasets. We begin with a warm-up stage using a hybrid dataset comprising curated interleaved sequences and limited data for multimodal understanding and text-to-image generation, which exposes the model to interleaved generation patterns while preserving its pretrained capabilities. To further refine interleaved generation, we propose a unified policy optimization framework that extends Group Relative Policy Optimization (GRPO) to the multimodal setting. Our approach jointly models text and image generation within a single decoding trajectory and optimizes it with our novel hybrid rewards covering textual relevance, visual-text alignment, and structural fidelity. Additionally, we incorporate process-level rewards to provide step-wise guidance, enhancing training efficiency in complex multimodal tasks. Experiments on MMIE and InterleavedBench demonstrate that our approach significantly enhances the quality and coherence of multimodal interleaved generation.

-> Multimodal generation techniques could be applicable for creating highlight reels from sports footage.

### DCAU-Net: Differential Cross Attention and Channel-Spatial Feature Fusion for Medical Image Segmentation (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09530v1
- 점수: final 80.0

Accurate medical image segmentation requires effective modeling of both long-range dependencies and fine-grained boundary details. While transformers mitigate the issue of insufficient semantic information arising from the limited receptive field inherent in convolutional neural networks, they introduce new challenges: standard self-attention incurs quadratic computational complexity and often assigns non-negligible attention weights to irrelevant regions, diluting focus on discriminative structures and ultimately compromising segmentation accuracy. Existing attention variants, although effective in reducing computational complexity, fail to suppress redundant computation and inadvertently impair global context modeling. Furthermore, conventional fusion strategies in encoder-decoder architectures, typically based on simple concatenation or summation, can not adaptively integrate high-level semantic information with low-level spatial details. To address these limitations, we propose DCAU-Net, a novel yet efficient segmentation framework with two key ideas. First, a new Differential Cross Attention (DCA) is designed to compute the difference between two independent softmax attention maps to adaptively highlight discriminative structures. By replacing pixel-wise key and value tokens with window-level summary tokens, DCA dramatically reduces computational complexity without sacrificing precision. Second, a Channel-Spatial Feature Fusion (CSFF) strategy is introduced to adaptively recalibrate features from skip connections and up-sampling paths through using sequential channel and spatial attention, effectively suppressing redundant information and amplifying salient cues. Experiments on two public benchmarks demonstrate that DCAU-Net achieves competitive performance with enhanced segmentation accuracy and robustness.

-> DCAU-Net의 차이 크로스 어텐션과 채널-공간 특징 융합 기술은 스포츠 장면에서 선수와 객체를 식별하고 중요한 순간을 분석하는 데 직접적으로 적용 가능하며, rk3588 엣지 디바이스에서 효율적으로 실행될 수 있는 계산 효율성 개선 방법을 제공합니다.

### A Text-Native Interface for Generative Video Authoring (1회째 추천)

- arXiv: http://arxiv.org/abs/2603.09072v1
- 점수: final 80.0

Everyone can write their stories in freeform text format -- it's something we all learn in school. Yet storytelling via video requires one to learn specialized and complicated tools. In this paper, we introduce Doki, a text-native interface for generative video authoring, aligning video creation with the natural process of text writing. In Doki, writing text is the primary interaction: within a single document, users define assets, structure scenes, create shots, refine edits, and add audio. We articulate the design principles of this text-first approach and demonstrate Doki's capabilities through a series of examples. To evaluate its real-world use, we conducted a week-long deployment study with participants of varying expertise in video authoring. This work contributes a fundamental shift in generative video interfaces, demonstrating a powerful and accessible new way to craft visual stories.

-> 비디오 제작 인터페이스 기술로 하이라이트 편집과 관련 있으나 스포츠 자동 촬영에 직접적으로 연관되지 않음

---

이 리포트는 arXiv API를 사용하여 생성되었습니다.
arXiv 논문의 저작권은 각 저자에게 있습니다.
Thank you to arXiv for use of its open access interoperability.
