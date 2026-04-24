# AI-Flagged Content (Cyan Highlights) — DSCSP Paper
> Extracted from Turnitin Report | 40% AI-detected | Submission ID: trn:oid:::1:3487930776

---

## Section I — INTRODUCTION

Steganography, the art of hiding information within innocuous-looking cover media, has historically relied on embedding data into existing images through imperceptible pixel-level modifications [5]. With the advent of powerful generative models, a new paradigm has emerged: generative steganography, where the cover image itself is synthesized to carry the hidden message, eliminating the need for a pre-existing cover and offering inherent plausible deniability.

Diffusion models [8], [16] have recently achieved state-of-the-art image synthesis quality, making them attractive backbones for generative steganography. Existing diffusion-based steganography approaches broadly fall into two categories. Noise-projection methods, exemplified by Diffusion-Stego [18], encode messages directly into the initial noise vector zT before running the reverse diffusion process, achieving a training-free approach but offering limited robustness against post-processing attacks. Model-adaptation methods [13], [21] fine-tune or augment the diffusion model itself, which requires costly retraining and risks degrading the pre-trained model's generation quality.

---

## Section II — RELATED WORK

### B. Deep Learning-Based Steganography

Neural network-based methods jointly learn encoding and decoding. HiDDeN [23] trains an encoder–decoder pair with a discriminator and differentiable JPEG proxy for robustness. SteganoGAN [20] uses GANs to embed messages directly during image generation. These methods achieve higher capacities than classical approaches but are constrained by the quality of their generative backbones.

### C. Diffusion-Based Steganography

Diffusion-Stego [18] projects secret messages into the initial noise vector zT of a pre-trained diffusion model using a training-free orthogonal projection scheme. While elegant, this approach offers limited robustness since the encoded information is distributed globally across the noise space and can be disrupted by spatial transformations.

StegoDiffusion [13] explores invertible diffusion architectures for steganography, leveraging the bijective mapping between image and latent spaces. LDStega [21] operates in the latent space of latent diffusion models, achieving practical robustness through latent-domain message embedding. DERO [22] addresses robustness in diffusion-based watermarking by jointly training a message encoder and decoder with the diffusion model, demonstrating resilience against various image manipulations.

### D. Differentiable Image Processing for Robustness

Training steganographic systems to resist image transformations requires differentiable proxies for non-differentiable operations. The Straight-Through Estimator (STE) [1] enables gradient flow through quantization operations, and has been applied to create differentiable JPEG approximations [15]. We leverage this technique alongside differentiable implementations of blur, noise, cropping, and color transformations to build a comprehensive attack layer for robustness training.

---

## Section III — METHODOLOGY

### A. System Overview

The encoding pipeline operates as follows. Given a payload bit string b ∈ {0, 1}^Nb where Nb = 256:

A key design choice is that the entire DDIM sampling chain runs under torch.no_grad(), and the perturbation δ is added after sampling. This provides a clean gradient path where no gradients need to flow through the diffusion UNet, enabling fast convergence and low memory usage.

### B. Controller Network

The Controller maps a payload b ∈ {0, 1}^256 to a spatial perturbation δ ∈ R^(3×256×256) using a multi-bit spatial tiling scheme.

The 256×256 image is partitioned into an 8×8 grid of 32×32-pixel cells, yielding Nc = 64 cells. Each cell encodes K = 4 bits, giving Nc × K = 256 total bits.

The tanh constrains the perturbation to [−αmax, αmax], ensuring bounded image distortion. The within-cell signal-to-noise ratio for any individual bit is 1/(K−1) = 1/3, providing sufficient separability for the Extractor.

### C. Extractor Network

The Extractor recovers the 256-bit payload from a stego image xs ∈ R^(3×256×256).

The permutation (0, 2, 3, 1) reorders from (B, K, 8, 8) to (B, 8, 8, K), so that flattening yields bit i corresponding to cell ⌊i/K⌋ and within-cell position i mod K—matching the Controller's view(B, 64, 4) grouping.

### D. Frozen Diffusion Backbone

We use a pre-trained DDPM [8] from the HuggingFace diffusers library, specifically google/ddpm-celebahq-256, which generates 256×256 face images.

The UNet is loaded in FP16 precision with all parameters frozen (requires_grad_ = False). Sampling uses the DDIM [16] deterministic sampler with T = 20 inference steps.

To accommodate training on a 6 GB GPU, we adopt an image pool strategy: 128 clean images are pre-generated and cached in GPU memory. The UNet is then moved to CPU, freeing approximately 440 MB of VRAM. The pool is refreshed every 2,000 training steps to maintain diversity.

---

## Section IV — EXPERIMENTS

### B. Training Dynamics

**Phase 2 (steps 3,000–10,000):** Attack probability linearly increases from 0 to 0.7, gradually exposing the system to transformations.

**Phase 3 (steps 10,000–30,000):** Full robustness training. 70% of batches undergo a random attack, solidifying resilience.

We use Adam [11] with β = (0.9, 0.999) and separate parameter groups for the Controller (no weight decay) and Extractor (weight decay 10^−4), both with initial learning rate 5×10^−4. A cosine annealing schedule [12] decays the learning rate to 10^−5 over 30,000 steps. Gradient clipping with max norm 5.0 is applied.

### C. Main Results

**Phase 1 (Clean Convergence, Steps 0–3,000):** The system converges rapidly: BER reaches 0.0000 by step 440, indicating that the Controller–Extractor pair learns the basic encoding–decoding mapping within minutes. PSNR increases from 24.1 dB to 36.8 dB as the Controller learns to minimize the perturbation while maintaining decodability.

**Phase 2 (Robustness Ramp, Steps 3,000–10,000):** As attacks are gradually introduced, a quality–robustness tradeoff emerges. PSNR temporarily dips to approximately 35.0 dB as the Controller adapts its patterns. Clean BER remains at 0.0000 throughout, indicating that the fundamental encoding capacity is not compromised by robustness training.

**Phase 3 (Full Robustness, Steps 10,000–30,000):** With 70% of batches undergoing random attacks, the system stabilizes and PSNR gradually recovers to 36.7 dB. The cosine learning rate decay refines the perturbation patterns, and the orthogonality loss continues decreasing to 11.4.

Table III presents the final evaluation results on 500 test samples generated with the trained model. DSCSP achieves perfect bit recovery with high perceptual quality across all measured metrics.

### D. Baseline Comparison

Table V compares DSCSP with prior steganographic methods using published results. Direct comparison is challenging due to differences in datasets, resolutions, and evaluation protocols; we report the most commonly cited numbers from each paper.

DSCSP achieves the highest PSNR among methods with comparable payload capacity (≥ 256 bits). While HiDDeN offers robustness, its capacity is limited to 30 bits—an order of magnitude lower. Diffusion-Stego matches our capacity in a training-free manner but lacks robustness training.

### E. Steganalysis Detection

The system demonstrates strong robustness against Gaussian noise (perfect recovery across all tested σ values), brightness/contrast adjustments (BER < 0.003 across all combinations), and moderate Gaussian blur (BER < 0.01 up to radius 1.5).

substantially higher BER (0.27–0.36), indicating limitations of the STE approximation for aggressive compression. This gap arises because real JPEG involves chroma subsampling, block-boundary effects, and implementation-specific rounding that the differentiable proxy does not fully capture.

Crop-and-resize remains a fundamental weakness (BER ∼ 0.47–0.50, near random chance), consistent with the spatial tiling architecture's reliance on aligned grid positions between Controller and Extractor.

DSCSP is highly parameter-efficient. The Controller uses only ∼20K trainable parameters—four orders of magnitude smaller than the frozen UNet (114M). The entire trainable system (Controller + Extractor, ∼11.2M parameters) is ∼10× smaller than the frozen backbone. Training completes in approximately 100 minutes on a single laptop GPU with 6 GB VRAM, demonstrating accessibility for resource-constrained settings.

---

## Section V — DISCUSSION

### B. Quality–Robustness Tradeoff

The training dynamics reveal a clear tradeoff: introducing attacks reduces PSNR from a peak of 37.6 dB (step 4,000, 10% attack rate) to 35.0 dB (step 7,000, 40% attack rate), as the Controller must embed stronger, more resilient patterns. During Phase 3, the cosine LR schedule enables gradual refinement, recovering PSNR to 36.7 dB while maintaining robustness—a net loss of only ∼1 dB from the peak clean PSNR.

### C. STE Proxy Limitations

The real JPEG evaluation reveals a significant gap between the differentiable STE proxy and actual JPEG codecs.

### D. Detectability and the Capacity–Undetectability Tradeoff

The perfect steganalysis detection (AUC = 1.0) highlights a fundamental tension in high-capacity steganography. The Controller's spatial tiling creates structured perturbation patterns that, while imperceptible to the human eye (PSNR 36.7 dB, SSIM 0.902), are easily distinguishable by a learned classifier. This is consistent with prior findings that high-capacity methods trade undetectability for payload capacity [2]. Potential mitigations include: (1) adversarial training with a discriminator to regularize the perturbation distribution; (2) adding an explicit steganographic loss that penalizes statistical distinguishability; or (3) reducing the per-cell perturbation amplitude at the cost of lower robustness.

### E. FID as a Distributional Metric

The FID of 13.7 between clean and stego distributions indicates that while individual image pairs show high PSNR and SSIM, the populations exhibit a measurable distributional shift. This is consistent with the steganalysis results: the systematic perturbation patterns shift the statistical properties of the stego distribution relative to the clean distribution, even though per-image distortion is small.

### F. Orthogonality Regularization

The orthogonality loss Lortho decreases from 8,728 at initialization to 11.4 at convergence, indicating that the 4 learned patterns within each cell evolve toward mutual orthogonality. This regularization is essential for multi-bit cells: without it, patterns may collapse into similar directions, reducing the effective per-bit signal strength and increasing BER.

### G. Crop Vulnerability

The spatial tiling architecture's vulnerability to crop attacks (BER ∼ 0.45) represents a fundamental limitation rather than a training deficiency. When the image is cropped, the spatial correspondence between encoded cells and Extractor feature map positions is disrupted.

---

## Section VI — LIMITATIONS AND FUTURE WORK

### B. Real JPEG Robustness

While the STE-based differentiable JPEG proxy enables gradient-based training, our evaluation reveals a substantial gap at low quality factors (Q ≤ 70). More faithful differentiable JPEG proxies—potentially learned from codec input–output pairs—could close this gap and extend robustness to more aggressive compression.

### C. Statistical Undetectability

The perfect steganalysis detection (AUC = 1.0) represents the most significant practical limitation. Incorporating adversarial training with a binary classifier or adapting cover-source mismatch mitigation techniques from classical steganography could reduce detectability, though likely at the expense of capacity or quality.

### D. Error-Correcting Codes

Integrating error-correcting codes (e.g., Reed-Solomon, LDPC) could improve resilience against attacks that cause partial bit errors, at the expense of reduced raw capacity. A differentiable ECC layer could be trained end-to-end with the steganographic pipeline.

### E. Scaling

We demonstrate results at 256 × 256 resolution with 128–512 bits. Scaling to higher resolutions (e.g., 512 × 512) with proportionally larger grids, or adapting DSCSP to latent diffusion models (e.g., Stable Diffusion), would enable larger payloads and more practical image resolutions.

---

## Section VII — ETHICAL CONSIDERATIONS

Steganography is inherently a dual-use technology.

---

## Section VIII — CONCLUSION

The entire system trains in approximately 100 minutes on a 6 GB laptop GPU, demonstrating that high-capacity, robust generative steganography is achievable with modest computational resources. DSCSP establishes sampling-path encoding as a viable and efficient paradigm for diffusion-based steganography, opening directions for future work in adversarial undetectability training, improved JPEG proxies, crop-resilient encoding, and scaling to higher resolutions.

---

*Total flagged: 41 segments | 40% of submission*