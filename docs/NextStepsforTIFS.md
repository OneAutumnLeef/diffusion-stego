---

## DSCSP → IEEE TIFS: Complete Publication Agenda

---

### SITUATION ASSESSMENT

Before the agenda, you need to understand what you're actually dealing with. TIFS has a hard bar: the work must be complete, not promising. Right now the paper has three categories of problem:

**Category A — Fatal if unaddressed (desk-reject or major revision rejection risk)**
**Category B — Serious weaknesses a reviewer will demand you fix**
**Category C — Polish and presentation issues**

The agenda is ordered by category, then by dependency.

---

## CATEGORY A — FATAL ISSUES

---

### A1. The detectability result is not publication-ready

This is the biggest problem in the paper. AUC of 0.954 at cycle 11, and 1.000 at cycle 9, means every single stego image is trivially separable from every clean image by a ResNet-18 trained for 12 epochs. The steganalysis literature treats AUC below 0.60 as near-indistinguishable and AUC above 0.90 as fully broken. You are reporting AUC of 0.990–1.000 as a primary result. No TIFS reviewer will accept this as a steganographic system — they will characterise it as a watermarking system with poor robustness, which is a different problem entirely.

**What needs to happen:**

You need to actually close the detectability gap, not just describe it. This requires real training work. Concretely:

**A1a. Run a dedicated adversarial-first training campaign.** The current "Detectability" variant in the four-strategy sweep is the weakest strategy — earliest adversarial schedule, highest attack probability, smallest perturbation budget, four discriminator updates per generator update. It reaches AUC 0.766 on average but never crosses 0.70. You need to push further. Specifically: reduce ϵmax to force the perturbation into the noise floor, increase discriminator updates to 6–8 per generator step, and add a gradient penalty term to the discriminator to prevent mode collapse. The goal is a stable training run that holds AUC below 0.70 for 3000+ steps.

**A1b. Investigate why the SRM discriminator is not creating enough gradient pressure.** The power-spectrum test (p = 1.07×10⁻¹²²) shows the perturbation has strong spectral structure. The discriminator should be suppressing this but isn't. Check whether the SRM filters are actually receiving meaningful gradients during the adversarial phase — the graphify tree shows `discriminator_srmfilterlayer_forward` has degree 2, which is low for a component that should be the core of the detectability mechanism. The SRM layer may be frozen or receiving near-zero gradients due to spectral normalisation clipping.

**A1c. If AUC cannot be pushed below 0.70 within reasonable compute budget, reframe the paper.** This is the honest alternative. The paper would need to pivot to positioning DSCSP as a robust watermarking system rather than a steganographic system, with detectability openly acknowledged as unsolved and the contribution reframed around the trade-off characterisation methodology and the architectural evolution. This is a weaker but defensible TIFS contribution if the experimental results support it.

The current framing — claiming steganography while reporting AUC of 0.990 — is neither fish nor fowl and will be rejected.

---

### A2. The libjpeg gap renders the JPEG robustness claim invalid

Proxy BER 1.17×10⁻⁴ becoming 0.113 under real libjpeg is not a gap — it is a complete failure of JPEG robustness. At BER 0.113, roughly 1 in 9 bits is wrong, which for a 256-bit payload means the decoded message is completely garbled. The paper currently reports this in Section X and then continues to cite the proxy BER as the headline result everywhere else. This is inconsistent and a reviewer will call it out.

**What needs to happen:**

**A2a. Fix the JPEG robustness.** The STE proxy is a known weak approximation of libjpeg. The standard fix is to mix real libjpeg passes into the training channel. `evaluate.py` already has `real_jpeg()` via Pillow. Add a training-time real JPEG augmentation path: with probability p_real (start at 0.1, ramp to 0.3), substitute the DiffJPEG pass with a Pillow JPEG round-trip using a straight-through gradient (detach the Pillow output, use the DiffJPEG output for the backward pass but the Pillow output for the forward loss). This is the standard approach and should close most of the gap within a few thousand steps.

**A2b. Re-evaluate after fixing and report real libjpeg BER as the primary JPEG metric.** The DiffJPEG proxy number can remain as a secondary figure for comparison, but the headline BER must be under real codec conditions. If real libjpeg BER comes down to, say, 0.01–0.05, that is a credible TIFS result for a 256-bit payload at Q75.

**A2c. Until A2a is done, remove JPEG BER from the abstract.** The abstract should not headline a number that Section X reveals to be non-transferable to real conditions. Replace with PSNR and clean BER only, with an explicit note that JPEG robustness is proxy-measured and the real-codec gap is characterised in Section X.

---

### A3. Sample count is too low for the BER claims being made

BER of 1.95×10⁻⁵ (original) or 5.86×10⁻⁵ (current) estimated on 200 samples means the estimate is based on approximately 200 × 256 = 51,200 bit decisions. At BER 5.86×10⁻⁵, you expect roughly 3 bit errors across the entire evaluation set. A 95% confidence interval on 3 observed errors over 51,200 trials is approximately [6×10⁻⁶, 1.7×10⁻⁴] — more than an order of magnitude wide. TIFS reviewers will reject BER claims at the 10⁻⁵ level without a substantially larger sample. This is a statistical validity issue, not a presentation issue.

**What needs to happen:**

**A3a. Increase evaluation to 1000 samples minimum.** This gives 256,000 bit decisions, narrowing the confidence interval to approximately [3×10⁻⁵, 1×10⁻⁴] for the observed BER. C5 from the codebase prompt is already written — execute it. The `--fast` flag preserves development iteration speed.

**A3b. Report bootstrap confidence intervals on all primary BER figures in Table I**, not just in the text. The intervals must appear in the table itself. A reader should be able to see the uncertainty directly.

---

## CATEGORY B — SERIOUS WEAKNESSES

---

### B1. The baseline comparison is not credible in its current form

Table II compares DSCSP against HiDDeN and SteganoGAN using numbers from their original papers at completely different resolutions (128×128 and 360×360 vs 256×256), different payload sizes (30-bit and 4bpp vs 256-bit), and different JPEG codecs. The footnote acknowledges this but the table structure implies comparability. Worse, neither baseline reports adaptive AUC, so the most important column — the one where DSCSP is weakest — has no comparison at all.

**What needs to happen:**

**B1a. Run HiDDeN at CelebA-HQ 256×256 with a 256-bit payload.** HiDDeN is straightforward to train from the original code. A training run to convergence at this resolution and payload size requires modest compute — a 3090 can do it in a day. Report clean BER, real JPEG Q75 BER, PSNR, and adaptive AUC under the same protocol you use for DSCSP.

**B1b. Run SteganoGAN or substitute a more recent baseline.** SteganoGAN at 256×256 with a 256-bit payload is similarly runnable. If the compute is too tight, substitute ReDMark or a recent learned watermarking baseline that has published code and comparable settings. The point is to have at least one matched-conditions number in Table II.

**B1c. Report adaptive AUC for the baselines.** This is the critical column. If HiDDeN and SteganoGAN are also fully detectable (likely, since they were not designed for steganalysis resistance), that is actually a useful finding — it contextualises your AUC numbers. If they are less detectable than DSCSP, that is a problem you need to address.

---

### B2. The crop-resize failure needs either a fix or a stronger justification

Crop Q80 BER of 0.495–0.503 across all eleven cycles means the system has zero crop robustness — it performs at chance. The paper discusses this honestly and proposes three directions, but TIFS expects at least a proof-of-concept implementation of the most promising direction, not just a description.

**What needs to happen:**

**B2a. Implement and test the shift-pool head (C4 from the codebase prompt).** This is architecturally compatible with no retraining of the controller. Add the `shift_pool` head to `GlobalExtractor`, train it for one or two autopilot cycles with aggressive crop augmentation (area ratio 0.60–0.85) in the attack channel, and report crop Q80 BER. Even a partial improvement — say, from 0.50 to 0.35 — demonstrates the direction is viable and gives the paper a concrete result rather than a discussion point.

**B2b. If shift-pool doesn't work, run the translation-robust carrier experiment.** Replace the nearest-neighbour carrier with a spread-spectrum carrier that distributes each bit across multiple spatial positions. This is more involved but is the more robust long-term solution.

**B2c. At minimum, add a paired t-test comparing crop BER to clean BER across all cycles.** The paper mentions paired t-tests justify the crop-vs-clean difference claim but doesn't report the test statistics. Include t-statistic, degrees of freedom, and p-value. This shows you've characterised the failure mode quantitatively even if you haven't fixed it.

---

### B3. The w/o redundancy row in Table VI contradicts the text

Section VII.B states "Table VI reports the corrected result" but the table still shows dashes and "training failed." This is a direct factual contradiction that a reviewer will catch on first reading.

**What needs to happen:**

Run the fixed `ablation_v5.py` (C3 from the codebase prompt is already written) and fill in the actual numbers. This is a no-GPU task — it's a short 8k-step ablation run from a warm start. There is no acceptable excuse for leaving this as "training failed" after the fix has been implemented.

---

### B4. The evaluation protocol inconsistency needs resolution

The paper uses 200–320 samples for some cycles and claims to be moving to 1000. Table I and the cross-dataset tables were evaluated at different sample counts. The bootstrap intervals in the text are for v5 numbers, not v4 numbers. None of this is clearly labelled.

**What needs to happen:**

**B4a. Re-run all primary evaluations (Table I, Table III, Table IV) at a consistent 1000-sample count** after implementing C5. The numbers in the current draft appear to be from a mixed-count evaluation. A consistent evaluation is not optional for TIFS.

**B4b. Report the sample count used for every table** either in the caption or in a "Evaluation Setup" row. This is a TIFS formatting expectation.

---

### B5. The Turnitin AI-detection issue is not fully resolved

The abstract still contains passive hedges ("are offered for orientation," "is left to supplementary / revision"). The conclusion's final sentence is long and passive. The phrase "supplementary / revision" is an unedited internal note.

**What needs to happen:**

**B5a. Rewrite the abstract one more time.** Every passive construction must go. The abstract should read like it was written by a human who did the experiments themselves. Specifically: "are offered for orientation rather than as a matched head-to-head" → "we include these numbers for context, not for direct comparison." "is left to supplementary / revision" → remove entirely or replace with "remains future work."

**B5b. Remove "supplementary / revision" from Section VI.C.** Replace with: "A matched re-run at CelebA-HQ 256×256 with a 256-bit payload is left to future work."

**B5c. Read the full paper aloud.** Every sentence that sounds like a template or a hedge is a Turnitin risk. The goal is prose that sounds like a specific person describing specific experiments they did. The technical sections (IV, V) are already close. The discussion and limitations sections need another pass.

---

### B6. The spectral analysis section is underused

Section VIII has genuinely interesting findings — the KL divergence is low (0.417) but the power-spectrum χ² p-value is 1.07×10⁻¹²² and the SVD Gini is 0.82. These three probes tell a coherent story: the perturbation looks perceptually fine but is spectrally structured and dominated by a few spatial modes. This is exactly the mechanism by which the discriminator fails to suppress detectability. This connection is currently implicit. A TIFS reviewer studying Section VIII will want to see it made explicit and used to motivate specific future work.

**What needs to happen:**

**B6a. Add a paragraph to Section VIII connecting the SVD Gini (0.82) to the detectability failure.** A Gini of 0.82 means the mean perturbation is dominated by 2–3 spatial modes. Those modes are precisely what the ResNet steganalyser is learning to detect. The fix is to reduce the Gini — either through the spectral loss (already present) or by adding an explicit mode-spreading regulariser. This connection gives Section VIII a purpose beyond being a statistical appendix.

**B6b. Plot the top-3 singular vectors of the mean perturbation.** These are the spatial patterns the steganalyser is keying on. Showing them visually would be a genuinely novel contribution and would make Section VIII one of the strongest parts of the paper.

---

## CATEGORY C — POLISH

---

### C1. Table numbering is inconsistent

Tables are numbered I, II, III, IV, V, VI but the cross-dataset table (Table IV in the current draft) appears after Table III in the text. In the original submission it was Table II. The reader has to jump around. Reorder the tables to appear in citation order in the text.

### C2. Figure 4 annotation is stale

Figure 4 shows "cycle 11: AUC drops to 0.6915" but Table I reports cycle 11 AUC as 0.6915 in the original and 0.954 in the current draft. The figure annotation must match the current numbers. This is the kind of inconsistency that tells a reviewer the paper was assembled from multiple versions without a final check.

### C3. The section numbering has a gap

The paper goes Section IX (Discussion) → Section XI (Conclusion) with no Section X listed in the table of contents but Section X appearing in the body as "Limitations." The table of contents in Section II's roadmap says "Section XI" for conclusion but Section IX for discussion. Either the roadmap or the section numbering is wrong. Fix the roadmap in Section II to match the actual section numbers.

### C4. Bootstrap confidence intervals belong in the tables

The paper reports BER bootstrap intervals in the text of Section VIII for v5 numbers, but Table I has no confidence intervals. For TIFS, uncertainty quantification belongs in the primary results table. Add [lower, upper] 95% bootstrap intervals for Clean BER and JPEG Q75 BER in Table I.

### C5. The conclusion is too long and buries the findings

The conclusion currently re-explains the architecture, the training procedure, and the trade-off analysis. TIFS conclusions should be 2–3 paragraphs: what you did, what you found, what it means for the field. Everything else belongs in the body. Cut the conclusion by half and make the three key sentences prominent: the primary BER, the detectability gap, and the crop-resize direction.

---

## EXECUTION SEQUENCE

Here is the order that minimises wasted work, since some experiments feed into others:

**Phase 1 — Fix the experimental gaps (4–6 weeks, requires compute)**

1. A2a — Add real JPEG augmentation to training, re-run adversarial cycles
2. A1a — Run dedicated detectability-first campaign with stronger discriminator
3. B2a — Implement shift-pool head, run 1–2 crop-augmented cycles
4. B1a — Train HiDDeN at matched conditions for baseline comparison
5. B3 — Fix and run the w/o redundancy ablation (no GPU needed, do this first)

**Phase 2 — Re-evaluate everything consistently (1 week)**

6. A3a + B4a — Re-run all primary evaluations at 1000 samples
7. A1b/A1c — Assess whether detectability has improved enough for TIFS framing or whether the paper needs reframing
8. B1c — Evaluate HiDDeN baseline under adaptive steganalysis protocol

**Phase 3 — Rewrite with new numbers (1–2 weeks)**

9. A2c — Update abstract to reflect real JPEG BER as primary metric
10. A1 framing — Rewrite abstract, introduction, and conclusion depending on Phase 2 outcome
11. B6a + B6b — Expand Section VIII with SVD vectors and detectability connection
12. B5a–B5c — Final AI-detection rewrite pass
13. C1–C5 — Consistency and formatting fixes

**Phase 4 — Internal review before submission (1 week)**

14. Read the full paper against the TIFS author checklist
15. Check every number in every table against the evaluation JSON outputs
16. Verify figure annotations match current results
17. Run Turnitin check on the revised draft

---

## HONEST ASSESSMENT OF THE PATH

If Phase 1 experiments succeed — real JPEG BER comes down to below 0.05 and AUC can be pushed below 0.70 on at least one operating point — this paper becomes a legitimate TIFS submission. The architecture is novel, the trade-off characterisation is thorough, the spectral analysis is interesting, and the honest treatment of limitations is actually a strength. That scenario is probably 3–4 months of focused work.

If the detectability gap cannot be closed, the honest path is IH&MMSec with the paper reframed around the trade-off surface methodology and the spectral analysis as the primary contributions. That is still a publishable paper — just not TIFS.

The current draft, as submitted, would receive a rejection from TIFS with a reviewer comment approximately as follows: *"The adaptive AUC of 0.954–1.000 demonstrates that the proposed system is fully detectable by a simple binary classifier, which is inconsistent with the claimed steganographic properties. The JPEG robustness claim relies on a proxy that degrades by three orders of magnitude under real codec conditions. The sample size is insufficient to support BER estimates at the reported precision."* All three of those objections are addressable. The agenda above is how you address them.