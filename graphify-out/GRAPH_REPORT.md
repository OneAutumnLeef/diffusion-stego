# Graph Report - .  (2026-04-20)

## Corpus Check
- 59 files · ~108,244 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 517 nodes · 1290 edges · 23 communities detected
- Extraction: 52% EXTRACTED · 48% INFERRED · 0% AMBIGUOUS · INFERRED: 619 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Clean Image|Clean Image]]
- [[_COMMUNITY_Citation Crop|Citation Crop]]
- [[_COMMUNITY_Loss Dscsp|Loss Dscsp]]
- [[_COMMUNITY_Eval Apply|Eval Apply]]
- [[_COMMUNITY_Generate Compute|Generate Compute]]
- [[_COMMUNITY_Tifs Crop|Tifs Crop]]
- [[_COMMUNITY_Config Main|Config Main]]
- [[_COMMUNITY_Real Tensor|Real Tensor]]
- [[_COMMUNITY_Init Forward|Init Forward]]
- [[_COMMUNITY_Compute Train|Compute Train]]
- [[_COMMUNITY_Append Major|Append Major]]
- [[_COMMUNITY_Compute Bootstrap|Compute Bootstrap]]
- [[_COMMUNITY_Artifact Restore|Artifact Restore]]
- [[_COMMUNITY_Ddim Init|Ddim Init]]
- [[_COMMUNITY_5090 Train|5090 Train]]
- [[_COMMUNITY_Pull Artifacts|Pull Artifacts]]
- [[_COMMUNITY_Init|Init]]
- [[_COMMUNITY_Build Transform|Build Transform]]
- [[_COMMUNITY_Convert Ycbcr|Convert Ycbcr]]
- [[_COMMUNITY_Convert Ycbcr|Convert Ycbcr]]
- [[_COMMUNITY_Init|Init]]
- [[_COMMUNITY_Frozen Unet|Frozen Unet]]
- [[_COMMUNITY_Generate Clean|Generate Clean]]

## God Nodes (most connected - your core abstractions)
1. `DiffusionWrapper` - 95 edges
2. `Controller` - 62 edges
3. `DSCSPLoss` - 56 edges
4. `RedundantCoder` - 56 edges
5. `ContentAdaptiveController` - 52 edges
6. `AttackLayer` - 50 edges
7. `Extractor` - 50 edges
8. `train()` - 37 edges
9. `SRMFilterLayer` - 35 edges
10. `GlobalExtractor` - 28 edges

## Surprising Connections (you probably didn't know these)
- `Crop Robustness Blocker` --semantically_similar_to--> `Crop-and-Resize Vulnerability`  [INFERRED] [semantically similar]
  training_ops/records/20260418_sprint_plan_execution.md → AI_Writing_Report.pdf
- `Detectability Blocker` --semantically_similar_to--> `Perfect Steganalysis Detectability AUC 1.0`  [INFERRED] [semantically similar]
  training_ops/records/20260418_sprint_plan_execution.md → AI_Writing_Report.pdf
- `Sampling-Path Encoding Framing` --semantically_similar_to--> `Sampling-Path Encoding`  [INFERRED] [semantically similar]
  AIcontentv3.md → AI_Writing_Report.pdf
- `Quality-Robustness Tradeoff Concept` --semantically_similar_to--> `Quality-Robustness Discussion Rationale`  [INFERRED] [semantically similar]
  AIcontentv3.md → AI_Writing_Report.pdf
- `STE-to-Real JPEG Gap Concept` --semantically_similar_to--> `STE-to-Real JPEG Robustness Gap`  [INFERRED] [semantically similar]
  AIcontentv3.md → AI_Writing_Report.pdf

## Hyperedges (group relationships)
- **Legacy DSCSP Training-Eval Pipeline** — train_script, train_v2_script, evaluate_script, steganalysis_script, ablation_script [INFERRED 0.84]
- **Publication Gate Evidence Cluster** — concept_publication_targets_bundle, concept_detectability_auc_gate, concept_crop_robustness_blocker, major_log_milestones, ablation_report_results [INFERRED 0.83]
- **Artifact Restore Bootstrap Flow** — agents_instructions, requirements_dependencies, concept_hf_artifact_restore, record_deps_install_20260418_190108 [EXTRACTED 1.00]
- **Sampling-Path Encoding Design Cluster** — Idea_sampling_path_encoding_channel, AIcontentv3_sampling_path_encoding, AI_Writing_Report_sampling_path_encoding, AI_Writing_Report_controller_network, AI_Writing_Report_extractor_resnet18 [INFERRED 0.89]
- **Robustness-Limitations Tradeoff Cluster** — AI_Writing_Report_differentiable_attack_curriculum, AI_Writing_Report_ste_real_jpeg_gap, AI_Writing_Report_crop_vulnerability, AI_Writing_Report_detectability_auc1, AIcontentv3_quality_robustness_tradeoff_concept [INFERRED 0.85]
- **Training Ops Iteration Pipeline** — 20260418_bootstrap_logging_setup_workflow, 20260418_190133_initial_probe_after_deps_probe_run, 20260418_190432_warmup_2000_warmup_run, 20260418_sprint_plan_execution_sprint_run, 20260418_bootstrap_logging_setup_run_logged_sh [INFERRED 0.82]

## Communities

### Community 0 - "Clean Image"
Cohesion: 0.07
Nodes (63): DSCSP v5 Security & Statistical Analysis.  Generates the statistical analysis, KL divergence KL(stego || clean) using diagonal Gaussian fits on     InceptionV, Chi-square test comparing FFT log-magnitude histograms of clean vs stego.     R, SVD of the mean stego delta matrix.     Flat/decaying spectrum = no periodic gr, Compute per-sample BER, PSNR, LPIPS and return bootstrap 95% CI., Save matplotlib figures (skips gracefully if matplotlib is unavailable)., Load controller + extractor from a v5 checkpoint., Generate N (clean, stego) image pairs and compute messages. (+55 more)

### Community 1 - "Citation Crop"
Cohesion: 0.03
Nodes (73): Diffusion Wrapper Tensor Construction Warning, One-Step Probe Validates Pipeline Not Quality, Initial Probe Run After Dependencies, Random-Guess BER Expectation, train_v4.train Invocation, Checkpoint At Step 2000, Curriculum-First Stage Plan, Discriminator Phase Start At Step 1000 (+65 more)

### Community 2 - "Loss Dscsp"
Cohesion: 0.08
Nodes (44): compute_ortho_loss(), generate_image_pool(), get_attack_prob(), main(), DSCSP Ablation Study Framework  Systematically trains and evaluates ablation var, Generate a pool of clean images using the diffusion model., Train a single ablation variant and return results., train_variant() (+36 more)

### Community 3 - "Eval Apply"
Cohesion: 0.07
Nodes (39): brightness_contrast(), _build_dct_matrix(), crop_resize(), DiffJPEG, gaussian_blur(), gaussian_noise(), Differentiable attack layers for robustness training.  All attacks are different, Scale quantization tables by JPEG quality factor. (+31 more)

### Community 4 - "Generate Compute"
Cohesion: 0.08
Nodes (40): generate_pairs(), predict_noise(), Frozen diffusion model wrapper for DSCSP.  Loads a pre-trained HuggingFace DDPM, Get alpha_cumprod for a given timestep value., Get alpha_cumprod for the previous timestep in the DDIM schedule.          Args:, sample_clean(), compute_ssim_single(), eval_attacks() (+32 more)

### Community 5 - "Tifs Crop"
Cohesion: 0.07
Nodes (50): v5 Ablation Report, Ablation Study Script, Converted AI Writing Report, AI-Flagged TIFS Draft Content, DERO [22], Diffusion-Stego [18], HiDDeN [23], LDStega [21] (+42 more)

### Community 6 - "Config Main"
Cohesion: 0.08
Nodes (31): base_config(), _extract_metrics(), _format_val(), main(), phase_variants(), phase_warmstart(), DSCSP v5 Ablation Study.  Trains 6 lightweight variants (8 000 steps each) fro, Launch train_v5.train() in-process with the given config override. (+23 more)

### Community 7 - "Real Tensor"
Cohesion: 0.11
Nodes (26): compute_ssim_batch(), main(), pil_to_tensor(), DSCSP Comprehensive Evaluation Script  Evaluates the trained DSCSP model with:, Compute mean SSIM for a batch. Inputs: (B,3,H,W) in [-1,1]., Convert (3, H, W) tensor in [-1,1] to PIL Image., Convert PIL Image to (1, 3, H, W) tensor in [-1,1]., Apply real JPEG compression via Pillow. (+18 more)

### Community 8 - "Init Forward"
Cohesion: 0.13
Nodes (11): ConvBlock, DownBlock, _group_norm(), _infer_payload_grid(), Content-adaptive steganographic controller for DSCSP v4.  Replaces the tiled-g, Conv + GroupNorm + ReLU block., Downsample by 2x with two conv layers., Upsample by 2x with ConvTranspose + conv. (+3 more)

### Community 9 - "Compute Train"
Cohesion: 0.18
Nodes (14): compute_psnr(), Compute PSNR between two image tensors in [-1, 1] range.      Args:         img1, Aggregate N physical logits to M message logits via mean-pooling.          Diffe, Majority voting for hard bit decisions at evaluation time.          Args:, _build_bootstrap_delta(), compute_spectral_loss(), compute_srm_match_loss(), get_attack_prob() (+6 more)

### Community 10 - "Append Major"
Cohesion: 0.38
Nodes (9): _append_major_log(), _build_cycle_config(), _is_significant(), _load_json(), main(), _meets_required(), _safe_get_step(), _save_json() (+1 more)

### Community 11 - "Compute Bootstrap"
Cohesion: 0.42
Nodes (8): bootstrap_ci(), compute_bootstrap_ci_table(), compute_kl_divergence(), compute_power_spectrum_test(), compute_svd_spectrum(), load_v5_model(), main(), _save_plots()

### Community 12 - "Artifact Restore"
Cohesion: 0.28
Nodes (9): Agent Artifact Restore Instructions, Chat History Archive README, Post-Clone Artifact Restore Workflow, Reproducible Training Logging Workflow, Dependencies Install Run Record, Initial Probe Run Record, Python Dependency Specification, Training Logbook Ledger (+1 more)

### Community 13 - "Ddim Init"
Cohesion: 0.33
Nodes (3): Modified DDIM sampler for DSCSP.  Simplified architecture: the entire DDIM denoi, DDIM sampler that adds a learned perturbation to the final denoised image., StegoDDIM

### Community 14 - "5090 Train"
Cohesion: 1.0
Nodes (1): DSCSP v4 Config Override for RTX 5090 (32GB VRAM).  The RTX 5090 has ample VRA

### Community 15 - "Pull Artifacts"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Init"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Build Transform"
Cohesion: 1.0
Nodes (1): Build the NxN DCT-II transform matrix.

### Community 18 - "Convert Ycbcr"
Cohesion: 1.0
Nodes (1): Convert RGB [0, 255] to YCbCr.

### Community 19 - "Convert Ycbcr"
Cohesion: 1.0
Nodes (1): Convert YCbCr to RGB [0, 255].

### Community 20 - "Init"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "Frozen Unet"
Cohesion: 1.0
Nodes (1): Run the frozen UNet to predict noise. Output is FP32, fully detached.          A

### Community 22 - "Generate Clean"
Cohesion: 1.0
Nodes (1): Generate a clean (unperturbed) image using standard DDIM sampling.         Usefu

## Ambiguous Edges - Review These
- `AI-Flagged TIFS Draft Content` → `Converted AI Writing Report`  [AMBIGUOUS]
  graphify-out/converted/AI_Writing_Report_401b4c80.md · relation: conceptually_related_to
- `Interactive Graph Visualization` → `Graphify Report Summary`  [AMBIGUOUS]
  graphify-out/graph.html · relation: conceptually_related_to
- `AI-Flagged Section Rewrite Plan` → `Turnitin AI Detection Score 40 Percent`  [AMBIGUOUS]
  AIcontentv3.md · relation: conceptually_related_to

## Knowledge Gaps
- **112 isolated node(s):** `DSCSP v5 Ablation Study.  Trains 6 lightweight variants (8 000 steps each) fro`, `Return the base v5 config for all ablation variants.`, `Config for the shared warm-start run (5 000 steps).`, `Launch train_v5.train() in-process with the given config override.`, `Run evaluate_tifs.py on a checkpoint and collect results.` (+107 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `5090 Train`** (2 nodes): `train_v4_5090.py`, `DSCSP v4 Config Override for RTX 5090 (32GB VRAM).  The RTX 5090 has ample VRA`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pull Artifacts`** (1 nodes): `pull_artifacts_from_hf.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Build Transform`** (1 nodes): `Build the NxN DCT-II transform matrix.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Convert Ycbcr`** (1 nodes): `Convert RGB [0, 255] to YCbCr.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Convert Ycbcr`** (1 nodes): `Convert YCbCr to RGB [0, 255].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Frozen Unet`** (1 nodes): `Run the frozen UNet to predict noise. Output is FP32, fully detached.          A`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Generate Clean`** (1 nodes): `Generate a clean (unperturbed) image using standard DDIM sampling.         Usefu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `AI-Flagged TIFS Draft Content` and `Converted AI Writing Report`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Interactive Graph Visualization` and `Graphify Report Summary`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `AI-Flagged Section Rewrite Plan` and `Turnitin AI Detection Score 40 Percent`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `train()` connect `Compute Train` to `Clean Image`, `Loss Dscsp`, `Eval Apply`, `Generate Compute`, `Config Main`, `Real Tensor`, `Append Major`?**
  _High betweenness centrality (0.128) - this node is a cross-community bridge._
- **Why does `DiffusionWrapper` connect `Clean Image` to `Loss Dscsp`, `Eval Apply`, `Generate Compute`, `Real Tensor`, `Compute Train`, `Ddim Init`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `ContentAdaptiveController` connect `Clean Image` to `Init Forward`, `Compute Train`, `Compute Bootstrap`, `Generate Compute`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 90 inferred relationships involving `DiffusionWrapper` (e.g. with `main()` and `DSCSP Ablation Study Framework  Systematically trains and evaluates ablation var`) actually correct?**
  _`DiffusionWrapper` has 90 INFERRED edges - model-reasoned connections that need verification._