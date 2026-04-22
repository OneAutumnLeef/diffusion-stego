# Rewritten AI-Flagged Sections — LaTeX Format
> Ready to paste into `paper_ieee_v3.tex`
> Each section is labeled with its location in the file.

---

## ABSTRACT
**Replace the entire `\begin{abstract}...\end{abstract}` block with:**

```latex
\begin{abstract}
This paper introduces DSCSP (Diffusion-Stego with Conditional Sampling Paths),
a steganographic framework designed to embed secret messages into images generated
by pre-trained diffusion models. Unlike prior methods that encode information in
the initial noise vector or require modification of the generative backbone, DSCSP
leaves the diffusion model fully frozen and instead employs a lightweight learned
\emph{Controller} to convert payload bits into structured spatial perturbations
applied to the output of a deterministic DDIM sampling process. In doing so, the
diffusion model's sampling trajectory is effectively repurposed as a covert
communication channel. A spatially aligned \emph{Extractor} network based on
ResNet-18 then recovers the embedded message from the resulting stego image.

The Controller and Extractor are trained jointly using a composite objective that
balances accurate bit reconstruction, perceptual fidelity, and pattern diversity.
To enhance robustness under realistic transmission conditions, we adopt a
curriculum-based training strategy that progressively incorporates differentiable
approximations of common image distortions, including noise, blur, color
adjustments, cropping, and JPEG compression. Experiments on CelebA-HQ images at
$256\times256$ resolution demonstrate that DSCSP embeds 256~bits with near-perfect
recovery and minimal perceptual distortion under clean conditions, while retaining
resilience against several standard post-processing operations. We further show
that the framework scales from 128 to 512~bits by adjusting the per-cell encoding
rate, and we provide an information-theoretic perspective that situates the
system's effective throughput relative to classical channel capacity limits.
\end{abstract}
```

---

## SECTION I — INTRODUCTION
**Replace the first two paragraphs (before `\We propose a fundamentally...`) with:**

```latex
For many years, steganographic methods have focused on modifying existing cover
images by subtly altering pixel intensities or transform-domain coefficients,
ensuring that the embedded signal remains imperceptible to human
observers~\cite{cox2007digital}. With the emergence of deep generative models,
this traditional paradigm has evolved. In \emph{generative steganography}, the
image itself is synthesized with the secret message embedded during the generation
process, eliminating reliance on a pre-existing cover image and offering a more
inherent form of plausible deniability.

Diffusion models~\cite{ho2020denoising, song2020denoising} now produce images
rivaling real photographs in quality, which makes them appealing as the generative
backbone for steganographic systems. Current diffusion-based steganography work
can be grouped into two camps. \emph{Noise-projection methods}, such as
Diffusion-Stego~\cite{wei2023diffusionstego}, encode the message into the starting
noise vector $\mathbf{z}_T$ and then run the standard reverse process---an elegant
training-free idea, but one that provides limited defense against post-processing.
\emph{Model-adaptation methods}~\cite{peng2023stegodiffusion, zhang2024ldstega}
modify or extend the diffusion model weights, which is expensive and can degrade
the model's original generation quality.
```

---

## SECTION II — RELATED WORK

### Subsection B: Deep Learning-Based Steganography
**Replace the last sentence of this subsection with:**

```latex
These learning-based approaches significantly increase embedding capacity compared
to classical techniques; however, the visual fidelity of the resulting stego images
remains inherently constrained by the expressiveness and limitations of the
underlying generative backbone.
```

### Subsection C: Diffusion-Based Steganography
**Replace the first paragraph with:**

```latex
Diffusion-Stego~\cite{wei2023diffusionstego} adopts a training-free strategy by
embedding the secret message directly into the initial noise vector $\mathbf{z}_T$
using an orthogonal projection mechanism. The idea is clean, but the encoded signal
is spread thinly across the full noise space, leaving it vulnerable to spatial
transformations.
```

---

## SECTION III — METHODOLOGY

### Subsection B: Controller Network — Forward Pass closing sentences
**Replace the two sentences after Eq.~(7) (`where $\alpha_{\text{max}}...`)**
**with:**

```latex
where $\alpha_{\text{max}} = 0.3$ defines the maximum allowable perturbation
magnitude. The $\tanh$ nonlinearity constrains the perturbation to the range
$[-\alpha_{\text{max}}, \alpha_{\text{max}}]$, thereby limiting the maximum change
applied to any pixel. Because each cell combines $K = 4$ learned patterns, the
resulting effective signal-to-noise ratio per bit is $\frac{1}{K-1} = \frac{1}{3}$,
which remains sufficient for the Extractor to disentangle and recover the individual
bit contributions reliably.
```

### Subsection C: Extractor Network — opening sentence
**Replace `Given a stego image...` with:**

```latex
Given a stego image $\mathbf{x}_s \in \R^{3 \times 256 \times 256}$, the goal of
the Extractor is to reconstruct the full 256-bit payload embedded within it.
```

### Subsection C: Extractor — Bit Ordering Alignment paragraph
**Replace the paragraph after Eq.~(9) with:**

```latex
This permutation reshuffles the tensor from $(B, K, 8, 8)$ to $(B, 8, 8, K)$
before flattening. The result is that the $i$-th output logit maps to cell
$\lfloor i/K \rfloor$ at within-cell position $i \bmod K$, which lines up exactly
with how the Controller organizes its \texttt{view(B, 64, 4)} bit layout.
```

---

## SECTION IV — EXPERIMENTS

### Subsection A: Experimental Setup — Hardware
**Replace with:**

```latex
\subsubsection{Hardware}
All experiments are performed on an NVIDIA RTX~3060 Laptop GPU equipped with
6\,GB of VRAM, using PyTorch~2.0+ with CUDA acceleration.
```

### Subsection A: Experimental Setup — Diffusion Model
**Replace with:**

```latex
\subsubsection{Diffusion Model}
We use the pre-trained \texttt{google/ddpm-celebahq-256} DDPM from HuggingFace,
generating $256\times256$ face images via 20-step DDIM sampling. The UNet has
approximately 114M parameters, all frozen in FP16.
```

### Subsection G (Training Procedure) — Robustness Curriculum bullet points
**Replace the three bullet points with:**

```latex
\begin{itemize}[itemsep=1pt, topsep=2pt]
    \item \textbf{Phase~1} (steps $0$--$3{,}000$): The system trains on clean
    images only. During this window the Controller and Extractor have one job:
    learn a reliable encode--decode mapping with no interference from image
    degradations.
    \item \textbf{Phase~2} (steps $3{,}000$--$10{,}000$): Attack probability
    climbs linearly from 0 to 0.7. Introducing distortions gradually rather than
    all at once gives the networks time to adapt without losing the encoding
    accuracy established in Phase~1.
    \item \textbf{Phase~3} (steps $10{,}000$--$30{,}000$): The attack rate is
    held fixed at 0.7 for the remainder of training, consolidating robustness
    at scale.
\end{itemize}
```

### Subsection G — Optimization paragraph
**Replace with:**

```latex
\subsubsection{Optimization}
Training is performed using Adam~\cite{kingma2014adam} with $\beta_1 = 0.9$ and
$\beta_2 = 0.999$. The Controller and Extractor are assigned separate parameter
groups: the Controller is trained without weight decay, while the Extractor uses
a decay factor of $10^{-4}$. Both groups are initialized with a learning rate of
$5 \times 10^{-4}$, which is annealed to $10^{-5}$ over 30{,}000 steps using a
cosine schedule~\cite{loshchilov2017sgdr}. Gradients are clipped to a maximum
norm of $5.0$ to ensure stable optimization.
```

### Subsection B: Training Dynamics — Phase 3 paragraph
**Replace with:**

```latex
\textit{Phase~3: Full Robustness (Steps 10{,}000--30{,}000).}
With the attack probability fixed at 70\%, the system enters a stable regime.
As the cosine learning-rate schedule progresses, PSNR steadily recovers to
36.7\,dB while the Controller refines its perturbation patterns. The orthogonality
loss continues to decline, reaching 11.4 by the end of training. The final model
achieves perfect bit recovery (BER = 0.0000) on clean images while sustaining a
PSNR of 36.7\,dB.
```

### Subsection C: Main Results — Clean Performance
**Replace with:**

```latex
\subsubsection{Clean Performance}
Table~\ref{tab:main_results} reports the final evaluation results on 500 newly
generated test images. Across all reported metrics, DSCSP achieves perfect bit
recovery while maintaining low perceptual distortion.
```

### Subsection C: STE-to-Real JPEG Gap paragraph
**Replace with:**

```latex
\subsubsection{STE-to-Real JPEG Gap}
An important observation is the gap between performance under differentiable
STE-based JPEG during training (BER $\sim 0.002$) and real JPEG compression at
test time. For high quality factors ($Q \geq 90$), real JPEG exhibits strong
robustness (BER $< 0.01$), indicating that the STE proxy transfers well in this
regime. However, at lower quality settings ($Q \leq 70$), BER increases
substantially to $0.27$--$0.36$, revealing limitations of the proxy under
aggressive compression. This discrepancy arises because practical JPEG
implementations introduce effects such as chroma subsampling, block-boundary
artifacts, and codec-specific rounding behavior that are not fully captured by
the differentiable approximation.
```

### Subsection C: Crop Vulnerability
**Replace with:**

```latex
\subsubsection{Crop Vulnerability}
Crop-and-resize represents the most severe failure case (BER $\sim 0.47$--$0.50$,
effectively random performance), as cropping disrupts the spatial correspondence
between the Controller's grid structure and the Extractor's feature alignment.
```

### Subsection C: Model Efficiency
**Replace with:**

```latex
\subsubsection{Model Efficiency}
In terms of parameter overhead, the additional cost is modest. The Controller
contains only $\sim$20K trainable parameters---approximately four orders of
magnitude fewer than the frozen 114M-parameter UNet backbone. Even when including
the Extractor, the total trainable parameters ($\sim$11.2M) remain roughly
$10\times$ smaller than the backbone. End-to-end training completes in
approximately 100~minutes on a single laptop GPU with 6\,GB VRAM, making the
approach accessible without large-scale computational resources.
```

### Subsection E: Steganalysis Detection — Results paragraph
**Replace with:**

```latex
The detector attains perfect classification performance (AUC = 1.0), indicating
that DSCSP stego images are \emph{readily detectable} by a learned classifier.
This outcome is unsurprising given the structured nature of the Controller's
spatial perturbations: the $8 \times 8$ tiled grid introduces consistent artifacts
that a sufficiently expressive model can reliably exploit. Such detectability is a
well-documented limitation of many high-capacity steganographic
systems~\cite{boroumand2019deep} and reflects the inherent tradeoff between payload
capacity and statistical undetectability. We outline possible mitigation strategies
in Section~\ref{sec:discussion}.
```

---

## SECTION V — DISCUSSION

### Subsection B: Quality–Robustness Tradeoff
**Replace with:**

```latex
\subsection{Quality--Robustness Tradeoff}
The empirical results clearly illustrate this tradeoff. PSNR reaches a peak of
37.6\,dB around step 4{,}000, when only 10\% of batches are subjected to attacks,
but declines to 35.0\,dB by step 7{,}000 as the attack probability increases to
40\% and the Controller is compelled to strengthen its perturbation patterns.
During Phase~3, the cosine learning-rate schedule allows the model to recover much
of this degradation, stabilizing at 36.7\,dB---approximately 1\,dB below the
clean-training peak.
```

### Subsection C: STE Proxy Limitations
**Replace with:**

```latex
\subsection{STE Proxy Limitations}
Evaluation with real JPEG compression reveals a noticeable gap between the
STE-based training proxy and actual codec behavior. While the proxy transfers
effectively at high quality factors ($Q \geq 90$), it consistently underestimates
degradation when compression becomes more aggressive ($Q \leq 70$). Bridging this
gap will likely require a more accurate differentiable approximation, such as a
compact neural model trained to replicate real codec input--output mappings.
```

### Subsection D: Detectability
**Replace with:**

```latex
\subsection{Detectability and the Capacity--Undetectability Tradeoff}
The steganalysis classifier achieving AUC = 1.0 is a sobering outcome, though not
entirely unexpected for a method operating at this payload level. Although the
tiled perturbations are visually imperceptible, the $8 \times 8$ grid structure
leaves a consistent statistical signature that a trained ResNet-18 can readily
exploit. This observation aligns with prior findings that increasing payload
capacity often comes at the cost of statistical
undetectability~\cite{boroumand2019deep}. Potential mitigation strategies
include: (1)~incorporating an adversarial discriminator during training to encourage
more natural perturbation distributions; (2)~adding a loss term that penalizes
measurable divergence from clean image statistics; or (3)~reducing per-cell
perturbation strength, trading some robustness for improved stealth.
```

### Subsection E: FID as a Distributional Metric
**Replace with:**

```latex
\subsection{FID as a Distributional Metric}
An FID score of 13.7 indicates that, at the distributional level, clean and stego
image populations remain statistically separable, even if individual image pairs
appear nearly indistinguishable according to PSNR or SSIM. This finding is
consistent with the steganalysis results: the structured perturbations introduced
by the Controller subtly shift the overall data distribution in ways detectable by
Inception-based features, despite negligible visual differences on a per-image
basis.
```

### Subsection F: Orthogonality Regularization
**Replace with:**

```latex
\subsection{Orthogonality Regularization}
Throughout training, $\Lcal_\text{ortho}$ decreases from 8{,}728 to 11.4,
demonstrating that the four learned patterns within each cell progressively
approach orthogonal configurations. This regularization is critical; without it,
pattern overlap would reduce per-bit separability, weakening the effective signal
and increasing the likelihood of decoding errors.
```

### Subsection G: Crop Vulnerability
**Replace with:**

```latex
\subsection{Crop Vulnerability}
The near-random BER observed under crop-and-resize attacks (BER $\sim 0.45$)
reflects an architectural limitation rather than a training deficiency. Cropping
alters or removes portions of the spatial grid, breaking the alignment between the
Controller's tiled encoding and the Extractor's feature representation. This
mirrors well-known weaknesses of spatial-template watermarking under geometric
distortions. We discuss potential remedies in Section~\ref{sec:limitations}.
```

---

## SECTION VI — LIMITATIONS AND FUTURE WORK

### Subsection A: Crop Robustness
**Replace with:**

```latex
\subsection{Crop Robustness}
The spatial tiling design is fundamentally susceptible to crop-and-resize attacks
due to spatial misalignment. Future work may mitigate this limitation **through**
several strategies: (1)~redundant encoding, in which each bit is embedded across
multiple non-adjacent cells to provide spatial redundancy; (2)~a global encoding
scheme combined with a learned spatial transformer to re-align inputs prior to
extraction; or (3)~frequency-domain encoding that distributes information across
spatial frequencies rather than anchoring it to fixed grid locations.
```

### Subsection B: Real JPEG Robustness
**Replace with:**

```latex
\subsection{Real JPEG Robustness}
Although the STE-based JPEG proxy is sufficient for enabling gradient flow during
training, empirical evaluation reveals its limitations at quality factors below~70.
A promising direction would be to develop a more faithful neural JPEG surrogate
trained on real codec input--output pairs, which could better approximate
compression artifacts and improve robustness under aggressive quality settings.
```

### Subsection C: Statistical Undetectability
**Replace with:**

```latex
\subsection{Statistical Undetectability}
The fact that a simple classifier achieves perfect separation between clean and
stego images (AUC = 1.0) is the most operationally significant limitation
identified in this work. Adversarial training with a binary discriminator is the
most direct remedy, though it introduces an additional moving part during training.
Alternatively, importing cover-source mismatch techniques from classical adaptive
steganography could reduce the statistical footprint with less architectural
change---though either approach will likely require some tradeoff in capacity or
robustness.
```

---

## SECTION VIII — CONCLUSION
**Replace the final paragraph (starting `End-to-end training...`) with:**

```latex
End-to-end training completes in approximately 100~minutes on a laptop GPU with
6\,GB of memory, making the approach accessible to researchers without extensive
computational resources. We view sampling-path encoding as a practical and
extensible framework for diffusion-based steganography, with clear avenues for
future development, including adversarial training for improved undetectability,
more faithful JPEG proxies, crop-resilient encoding strategies, and extension to
higher-resolution image synthesis.
```