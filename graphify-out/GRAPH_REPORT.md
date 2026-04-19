# Graph Report - .  (2026-04-19)

## Corpus Check
- Large corpus: 213 files · ~5,274,963 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 742 nodes · 1484 edges · 36 communities detected
- Extraction: 54% EXTRACTED · 45% INFERRED · 0% AMBIGUOUS · INFERRED: 672 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Core Metrics Pipeline|Core Metrics Pipeline]]
- [[_COMMUNITY_JPEG Eval Pipeline|JPEG Eval Pipeline]]
- [[_COMMUNITY_V4 Mid Training Panels|V4 Mid Training Panels]]
- [[_COMMUNITY_V4 Early Snapshot Panel|V4 Early Snapshot Panel]]
- [[_COMMUNITY_V4 Late Snapshot Panel|V4 Late Snapshot Panel]]
- [[_COMMUNITY_V5 Architecture Objectives|V5 Architecture Objectives]]
- [[_COMMUNITY_V5 Quality Finetune Trends|V5 Quality Finetune Trends]]
- [[_COMMUNITY_V5 Early Snapshot Panel|V5 Early Snapshot Panel]]
- [[_COMMUNITY_V4 Temporal Residual Trends|V4 Temporal Residual Trends]]
- [[_COMMUNITY_V5 Late Snapshot Trends|V5 Late Snapshot Trends]]
- [[_COMMUNITY_V5 Step 22000 Snapshot|V5 Step 22000 Snapshot]]
- [[_COMMUNITY_V5 Mid Snapshot Panel|V5 Mid Snapshot Panel]]
- [[_COMMUNITY_Augmentation Forward Ops|Augmentation Forward Ops]]
- [[_COMMUNITY_Model Block Definitions|Model Block Definitions]]
- [[_COMMUNITY_Ablation Eval Orchestration|Ablation Eval Orchestration]]
- [[_COMMUNITY_Bootstrap Dependency Logs|Bootstrap Dependency Logs]]
- [[_COMMUNITY_Training Major Log Utilities|Training Major Log Utilities]]
- [[_COMMUNITY_Graphify Visualization Internals|Graphify Visualization Internals]]
- [[_COMMUNITY_Workflow Setup Records|Workflow Setup Records]]
- [[_COMMUNITY_Sprint Plan Records|Sprint Plan Records]]
- [[_COMMUNITY_Visualization Utility Module|Visualization Utility Module]]
- [[_COMMUNITY_RunPod Setup Scripts|RunPod Setup Scripts]]
- [[_COMMUNITY_Adversarial Ramp Loss|Adversarial Ramp Loss]]
- [[_COMMUNITY_V4 Step 9000 Panel|V4 Step 9000 Panel]]
- [[_COMMUNITY_V4 Step 9500 Panel|V4 Step 9500 Panel]]
- [[_COMMUNITY_V4 Step 10500 Panel|V4 Step 10500 Panel]]
- [[_COMMUNITY_RTX5090 Config Notes|RTX5090 Config Notes]]
- [[_COMMUNITY_Training Log Templates|Training Log Templates]]
- [[_COMMUNITY_Chat Archive Metadata|Chat Archive Metadata]]
- [[_COMMUNITY_Package Init Node A|Package Init Node A]]
- [[_COMMUNITY_Package Init Node B|Package Init Node B]]
- [[_COMMUNITY_DCT Matrix Builder|DCT Matrix Builder]]
- [[_COMMUNITY_RGB to YCbCr|RGB to YCbCr]]
- [[_COMMUNITY_YCbCr to RGB|YCbCr to RGB]]
- [[_COMMUNITY_Frozen UNet Inference|Frozen UNet Inference]]
- [[_COMMUNITY_Clean DDIM Sampling|Clean DDIM Sampling]]

## God Nodes (most connected - your core abstractions)
1. `DiffusionWrapper` - 59 edges
2. `ContentAdaptiveController` - 52 edges
3. `RedundantCoder` - 39 edges
4. `SRMFilterLayer` - 35 edges
5. `train()` - 33 edges
6. `AttackLayer` - 33 edges
7. `DSCSPLoss` - 33 edges
8. `GlobalExtractor` - 28 edges
9. `DiffJPEG` - 27 edges
10. `Cover-Stego Visual Parity` - 23 edges

## Surprising Connections (you probably didn't know these)
- `Low-Magnitude Residual Pattern` --semantically_similar_to--> `Composite Training Loss`  [INFERRED] [semantically similar]
  outputs_v4/step_000500.png → Idea.md
- `God Nodes Ranking` --conceptually_related_to--> `DSCSP Research Plan`  [INFERRED]
  graphify-out/GRAPH_REPORT.md → Idea.md
- `Differentiable Robustness Augmentations` --semantically_similar_to--> `Crop Robustness Blocker`  [INFERRED] [semantically similar]
  Idea.md → PROJECT_CONTEXT_MANUAL_2026-04-18.md
- `Training Visualization Step 008000` --conceptually_related_to--> `Cycle 11 Evaluation Metrics`  [INFERRED]
  outputs_v4/step_008000.png → PROJECT_CONTEXT_MANUAL_2026-04-18.md
- `Step 019000 residual intensity map` --semantically_similar_to--> `Residual heatmap row at step 019500`  [INFERRED] [semantically similar]
  outputs_v4/step_019000.png → outputs_v5/step_019500.png

## Hyperedges (group relationships)
- **DSCSP Core Design** — idea_sampling_path_encoding, idea_controller_network, idea_extractor_network, idea_ecc_layer, idea_composite_loss [EXTRACTED 1.00]
- **Cycle 11 Outcome Bundle** — project_context_cycle11_metrics, project_context_crop_robustness_blocker, project_context_detectability_pass_cycle11, major_log_cycle11_robust_balance, major_log_crop_q80_ber_0495469, major_log_adaptive_auc_06915 [EXTRACTED 1.00]
- **Outputs V4 Snapshot Sequence 000500 to 008000** — step_000500_training_visualization, step_001000_training_visualization, step_001500_training_visualization, step_002000_training_visualization, step_002500_training_visualization, step_003000_training_visualization, step_003500_training_visualization, step_004000_training_visualization, step_004500_training_visualization, step_005000_training_visualization, step_005500_training_visualization, step_006000_training_visualization, step_006500_training_visualization, step_007000_training_visualization, step_007500_training_visualization, step_008000_training_visualization [EXTRACTED 1.00]
- **Step 008500 embedding visualization triplet** — step_008500_cover_grid, step_008500_stego_grid, step_008500_residual_map [EXTRACTED 1.00]
- **Mid-stage stability pattern in stego visualization** — step_012000_stego_grid, step_014000_stego_grid, step_016000_stego_grid [INFERRED 0.81]
- **Progressive residual suppression across training steps** — step_008500_residual_map, step_012000_residual_map, step_016000_residual_map, step_019000_residual_map [INFERRED 0.84]
- **Early v5 blocky residual phase** — step_000500_training_panel, step_001000_training_panel, step_001500_training_panel, step_002000_training_panel, step_002500_training_panel, step_003000_training_panel, step_003500_training_panel, step_004000_training_panel [INFERRED 0.88]
- **Late v5 striped residual phase** — step_004500_training_panel, step_005000_training_panel, step_005500_training_panel, step_006000_training_panel, step_006500_training_panel, step_007000_training_panel, step_007500_training_panel, step_008000_training_panel, step_008500_training_panel, step_009000_training_panel, step_009500_training_panel, step_010000_training_panel, step_010500_training_panel [INFERRED 0.90]
- **Training panel visualization components** — step_000500_triplet_panel_layout, step_000500_cover_stego_pair, step_000500_blocky_residual_pattern, step_004500_striped_residual_pattern, step_006000_horizontal_banding_pattern [INFERRED 0.86]
- **Early chunk training triplets** — step_011000_cover_row, step_011000_stego_row, step_011000_residual_map, step_011500_cover_row, step_011500_stego_row, step_011500_residual_map, step_012000_cover_row, step_012000_stego_row, step_012000_residual_map, step_012500_cover_row, step_012500_stego_row, step_012500_residual_map, step_013000_cover_row, step_013000_stego_row, step_013000_residual_map, step_013500_cover_row, step_013500_stego_row, step_013500_residual_map, step_014000_cover_row, step_014000_stego_row, step_014000_residual_map [EXTRACTED 1.00]
- **Mid chunk training triplets** — step_014500_cover_row, step_014500_stego_row, step_014500_residual_map, step_015000_cover_row, step_015000_stego_row, step_015000_residual_map, step_015500_cover_row, step_015500_stego_row, step_015500_residual_map, step_016000_cover_row, step_016000_stego_row, step_016000_residual_map, step_016500_cover_row, step_016500_stego_row, step_016500_residual_map, step_017000_cover_row, step_017000_stego_row, step_017000_residual_map, step_017500_cover_row, step_017500_stego_row, step_017500_residual_map [EXTRACTED 1.00]
- **Late chunk training triplets** — step_018000_cover_row, step_018000_stego_row, step_018000_residual_map, step_018500_cover_row, step_018500_stego_row, step_018500_residual_map, step_019000_cover_row, step_019000_stego_row, step_019000_residual_map, step_019500_cover_row, step_019500_stego_row, step_019500_residual_map, step_020000_cover_row, step_020000_stego_row, step_020000_residual_map, step_020500_cover_row, step_020500_stego_row, step_020500_residual_map, step_021000_cover_row, step_021000_stego_row, step_021000_residual_map, step_021500_cover_row, step_021500_stego_row, step_021500_residual_map [EXTRACTED 1.00]
- **V5 Mid-Run Imperceptibility Sequence** — step_022000_training_snapshot, step_022500_training_snapshot, step_023000_training_snapshot, step_023500_training_snapshot, step_024000_training_snapshot, step_024500_training_snapshot, step_025000_training_snapshot, step_025500_training_snapshot, step_026000_training_snapshot, step_026500_training_snapshot, step_027000_training_snapshot, step_027500_training_snapshot, step_028000_training_snapshot, step_028500_training_snapshot, step_029000_training_snapshot, step_029500_training_snapshot, step_030000_training_snapshot, step_030500_training_snapshot, step_031000_training_snapshot, step_031500_training_snapshot, step_032000_training_snapshot, step_032500_training_snapshot, step_022000_cover_stego_visual_parity [INFERRED 0.87]
- **V5 Mid-Run Residual Pattern Consistency** — step_022000_training_snapshot, step_025000_training_snapshot, step_028000_training_snapshot, step_030000_training_snapshot, step_032500_training_snapshot, step_022000_blockwise_residual_signature [INFERRED 0.84]
- **Face Saliency Residual Cases** — step_022500_training_snapshot, step_026000_training_snapshot, step_027000_training_snapshot, step_029500_training_snapshot, step_030500_training_snapshot, step_032000_training_snapshot, step_026000_face_saliency_residuals [EXTRACTED 1.00]
- **Three-Row Cover-Stego-Residual Layout Pattern** — step_033000_cover_grid, step_033500_cover_grid, step_034000_cover_grid, step_034500_cover_grid, step_035000_cover_grid, step_035500_cover_grid, step_036000_cover_grid, step_036500_cover_grid, step_037000_cover_grid, step_037500_cover_grid, step_038000_cover_grid, step_038500_cover_grid, step_039000_cover_grid, step_039500_cover_grid, step_040000_cover_grid, step_040500_cover_grid, step_041000_cover_grid, step_041500_cover_grid, step_042000_cover_grid, step_042500_cover_grid, step_043000_cover_grid, step_043500_cover_grid, step_033000_stego_grid, step_033500_stego_grid, step_034000_stego_grid, step_034500_stego_grid, step_035000_stego_grid, step_035500_stego_grid, step_036000_stego_grid, step_036500_stego_grid, step_037000_stego_grid, step_037500_stego_grid, step_038000_stego_grid, step_038500_stego_grid, step_039000_stego_grid, step_039500_stego_grid, step_040000_stego_grid, step_040500_stego_grid, step_041000_stego_grid, step_041500_stego_grid, step_042000_stego_grid, step_042500_stego_grid, step_043000_stego_grid, step_043500_stego_grid, step_033000_residual_heatmap, step_033500_residual_heatmap, step_034000_residual_heatmap, step_034500_residual_heatmap, step_035000_residual_heatmap, step_035500_residual_heatmap, step_036000_residual_heatmap, step_036500_residual_heatmap, step_037000_residual_heatmap, step_037500_residual_heatmap, step_038000_residual_heatmap, step_038500_residual_heatmap, step_039000_residual_heatmap, step_039500_residual_heatmap, step_040000_residual_heatmap, step_040500_residual_heatmap, step_041000_residual_heatmap, step_041500_residual_heatmap, step_042000_residual_heatmap, step_042500_residual_heatmap, step_043000_residual_heatmap, step_043500_residual_heatmap [EXTRACTED 1.00]
- **Cross-Step Visual Fidelity Consistency** — step_033000_visual_fidelity, step_033500_visual_fidelity, step_034000_visual_fidelity, step_034500_visual_fidelity, step_035000_visual_fidelity, step_035500_visual_fidelity, step_036000_visual_fidelity, step_036500_visual_fidelity, step_037000_visual_fidelity, step_037500_visual_fidelity, step_038000_visual_fidelity, step_038500_visual_fidelity, step_039000_visual_fidelity, step_039500_visual_fidelity, step_040000_visual_fidelity, step_040500_visual_fidelity, step_041000_visual_fidelity, step_041500_visual_fidelity, step_042000_visual_fidelity, step_042500_visual_fidelity, step_043000_visual_fidelity, step_043500_visual_fidelity [INFERRED 0.86]
- **Residual Maps Show Mostly Sparse Perturbations** — step_033000_residual_heatmap, step_033500_residual_heatmap, step_034000_residual_heatmap, step_034500_residual_heatmap, step_035000_residual_heatmap, step_035500_residual_heatmap, step_036000_residual_heatmap, step_036500_residual_heatmap, step_037000_residual_heatmap, step_037500_residual_heatmap, step_038000_residual_heatmap, step_038500_residual_heatmap, step_039000_residual_heatmap, step_039500_residual_heatmap, step_040000_residual_heatmap, step_040500_residual_heatmap, step_041000_residual_heatmap, step_041500_residual_heatmap, step_042000_residual_heatmap, step_042500_residual_heatmap, step_043000_residual_heatmap, step_043500_residual_heatmap [INFERRED 0.80]
- **Late-stage v5 training snapshot sequence** — step_044000_snapshot_v5, step_044500_snapshot_v5, step_045000_snapshot_v5, step_045500_snapshot_v5, step_046000_snapshot_v5, step_046500_snapshot_v5, step_047000_snapshot_v5, step_047500_snapshot_v5, step_048000_snapshot_v5, step_048500_snapshot_v5, step_049000_snapshot_v5, step_049500_snapshot_v5, step_050000_snapshot_v5 [INFERRED 0.90]
- **v5 clean finetune progression** — step_062500_snapshot_v5_clean, step_063000_snapshot_v5_clean, step_063500_snapshot_v5_clean, step_064000_snapshot_v5_clean, step_064500_snapshot_v5_clean, step_065000_snapshot_v5_clean, step_065500_snapshot_v5_clean, step_066000_snapshot_v5_clean [INFERRED 0.92]
- **Shared stego visualization pattern** — step_044000_visual_triplet_layout, step_044000_identity_preservation, step_044000_structured_residual_pattern, step_050000_color_cast_variability [INFERRED 0.85]
- **Early stable quality window** — step_051000_quality_montage, step_051500_quality_montage, step_052000_quality_montage, step_052500_quality_montage, step_053000_quality_montage [INFERRED 0.84]
- **Localized residual spike window** — step_054000_residual_map_grid, step_054500_residual_map_grid, step_057500_residual_map_grid, step_058000_residual_map_grid, step_060000_residual_map_grid, step_061500_residual_map_grid [INFERRED 0.76]
- **Late-phase quality convergence** — step_059000_quality_montage, step_059500_quality_montage, step_060500_quality_montage, step_061000_quality_montage [INFERRED 0.82]
- **Logged Training Operations Pipeline** — readme_training_operations, readme_run_logged_sh, training_log_logbook, 20260418_bootstrap_logging_setup_workflow [EXTRACTED 1.00]
- **Probe to Warmup Progression** — 20260418_190031_initial_probe_run, 20260418_190108_deps_install_run, 20260418_190133_initial_probe_after_deps_run, 20260418_190432_warmup_2000_run [EXTRACTED 1.00]
- **Cover Stego Residual Triplet** — step_062000_cover_row, step_062000_stego_row, step_062000_residual_row [EXTRACTED 1.00]

## Communities

### Community 0 - "Core Metrics Pipeline"
Cohesion: 0.04
Nodes (111): bootstrap_ci(), compute_bootstrap_ci_table(), compute_kl_divergence(), compute_power_spectrum_test(), compute_svd_spectrum(), generate_pairs(), load_v5_model(), main() (+103 more)

### Community 1 - "JPEG Eval Pipeline"
Cohesion: 0.06
Nodes (50): _build_dct_matrix(), DiffJPEG, Apply 8x8 block DCT to a single-channel image., Apply 8x8 block inverse DCT., Differentiable JPEG compression using Straight-Through Estimator (STE).      For, _adaptive_attack(), _adaptive_steganalysis(), _bootstrap_ci() (+42 more)

### Community 2 - "V4 Mid Training Panels"
Cohesion: 0.05
Nodes (53): Step 010000 cover image grid, Step 010000 residual intensity map, Step 010000 stego image grid, Step 013500 cover image grid, Cover image row at step 013500, Step 013500 residual intensity map, Step 013500 stego image grid, Stego image row at step 013500 (+45 more)

### Community 3 - "V4 Early Snapshot Panel"
Cohesion: 0.07
Nodes (46): Cover Image Grid, Residual Difference Heatmap, Stego Image Grid, Visual Fidelity Assessment, Cover Image Grid, Residual Difference Heatmap, Stego Image Grid, Visual Fidelity Assessment (+38 more)

### Community 4 - "V4 Late Snapshot Panel"
Cohesion: 0.07
Nodes (46): Cover Image Grid, Residual Difference Heatmap, Stego Image Grid, Visual Fidelity Assessment, Cover Image Grid, Residual Difference Heatmap, Stego Image Grid, Visual Fidelity Assessment (+38 more)

### Community 5 - "V5 Architecture Objectives"
Cohesion: 0.09
Nodes (44): Adversarial Detectability Objective, Composite Training Loss, Controller Network, Diffusion-Stego Baseline, DSCSP Research Plan, Error-Correcting Coding Layer, Extractor Network, Differentiable Robustness Augmentations (+36 more)

### Community 6 - "V5 Quality Finetune Trends"
Cohesion: 0.06
Nodes (44): Step 051000 quality-training montage with stable identity preservation, Step 051000 residual map grid showing low-to-moderate distributed energy, Step 051500 quality-training montage with stable identity preservation, Step 051500 residual map grid showing low-to-moderate distributed energy, Step 052000 quality-training montage with stable identity preservation, Step 052000 residual map grid showing low-to-moderate distributed energy, Step 052500 quality-training montage with stable identity preservation, Step 052500 residual map grid showing low-to-moderate distributed energy (+36 more)

### Community 7 - "V5 Early Snapshot Panel"
Cohesion: 0.19
Nodes (33): Coarse blocky residual carrier pattern, Near-identity cover to stego pairing in top two rows, outputs_v5/step_000500.png training panel, Three-row panel layout (cover, stego, residual), Shift from v4 low-delta look to v5 high-energy residual carrier, outputs_v5/step_001000.png training panel, outputs_v5/step_001500.png training panel, Occasional severe identity distortion outlier in individual columns (+25 more)

### Community 8 - "V4 Temporal Residual Trends"
Cohesion: 0.1
Nodes (31): Cover and stego rows remain visually near-identical, Possible late-stage residual brightening around step 18500, Residual map sparsity increases across steps 8500-19000, Step 008500 cover image grid, Step 008500 residual intensity map, Step 008500 stego image grid, Step 018500 cover image grid, Cover image row at step 018500 (+23 more)

### Community 9 - "V5 Late Snapshot Trends"
Cohesion: 0.25
Nodes (29): High identity preservation between cover and stego rows, Step 044000 late-stage v5 snapshot, Residual map shows structured high-frequency embedding pattern, Triplet layout: cover, stego, residual heatmap, Step 044500 late-stage v5 snapshot, Step 045000 late-stage v5 snapshot, Step 045500 late-stage v5 snapshot, Step 046000 late-stage v5 snapshot (+21 more)

### Community 10 - "V5 Step 22000 Snapshot"
Cohesion: 0.16
Nodes (26): Blockwise Residual Signature, Cover-Stego Visual Parity, Training Snapshot Step 22000, Training Snapshot Step 22500, Training Snapshot Step 23000, Training Snapshot Step 23500, Training Snapshot Step 24000, Training Snapshot Step 24500 (+18 more)

### Community 11 - "V5 Mid Snapshot Panel"
Cohesion: 0.11
Nodes (25): Step 011000 cover image grid, Cover image row at step 011000, Step 011000 residual intensity map, Step 011000 stego image grid, Stego image row at step 011000, Step 011500 cover image grid, Cover image row at step 011500, Step 011500 residual intensity map (+17 more)

### Community 12 - "Augmentation Forward Ops"
Cohesion: 0.11
Nodes (15): brightness_contrast(), crop_resize(), gaussian_blur(), gaussian_noise(), Differentiable attack layers for robustness training.  All attacks are different, Scale quantization tables by JPEG quality factor., Args:             x: (B, 3, H, W) in [-1, 1]             quality: JPEG quality f, Add zero-mean Gaussian noise. Gradient: identity. (+7 more)

### Community 13 - "Model Block Definitions"
Cohesion: 0.13
Nodes (11): ConvBlock, DownBlock, _group_norm(), _infer_payload_grid(), Content-adaptive steganographic controller for DSCSP v4.  Replaces the tiled-gri, Conv + GroupNorm + ReLU block., Downsample by 2x with two conv layers., Upsample by 2x with ConvTranspose + conv. (+3 more)

### Community 14 - "Ablation Eval Orchestration"
Cohesion: 0.16
Nodes (19): base_config(), _extract_metrics(), _format_val(), main(), phase_variants(), phase_warmstart(), DSCSP v5 Ablation Study.  Trains 6 lightweight variants (8 000 steps each) from, Launch train_v5.train() in-process with the given config override. (+11 more)

### Community 15 - "Bootstrap Dependency Logs"
Cohesion: 0.11
Nodes (19): diffusers ModuleNotFoundError, Initial Probe Run, Diffusion Wrapper Tensor Construction Warning, Random-Guess BER Confirmation, Post-Dependency One-Step Probe, Keep Logs Clean by Warning Cleanup, Chance-Level BER Baseline, Clean-Only Control Plan (+11 more)

### Community 16 - "Training Major Log Utilities"
Cohesion: 0.38
Nodes (9): _append_major_log(), _build_cycle_config(), _is_significant(), _load_json(), main(), _meets_required(), _safe_get_step(), _save_json() (+1 more)

### Community 17 - "Graphify Visualization Internals"
Cohesion: 0.22
Nodes (10): Canvas Hyperedge Overlay Layer, Interactive Graph HTML Viewer, Embedded RAW_NODES and RAW_EDGES Datasets, Search and Node-Info Sidebar, vis-network Rendering Library, Graph Report Corpus Summary, Global Graph Statistics, God Nodes Ranking (+2 more)

### Community 18 - "Workflow Setup Records"
Cohesion: 0.33
Nodes (6): Existing Progress JSON Metrics, Structured Training Ops Workflow Setup, Run Artifacts Layout, run_logged.sh Wrapper, Training Operations Workflow, Bootstrap Logging Setup Entry

### Community 19 - "Sprint Plan Records"
Cohesion: 0.33
Nodes (6): Crop Robustness Publication Blocker, Require Valid Metric Payloads for Passing Gate, Cycle 11 Robust Balance Metrics, Cross-Dataset Evaluator and Gating Fix, Sprint Cycles 9-11 Plan, Sprint Cycles 9-11 Entry

### Community 20 - "Visualization Utility Module"
Cohesion: 0.4
Nodes (3): Visualization utilities for DSCSP., Save side-by-side comparison of clean vs stego images, plus an amplified     dif, save_comparison()

### Community 21 - "RunPod Setup Scripts"
Cohesion: 0.4
Nodes (5): Provision Environment Before Training, Install requirements.txt Dependencies, Dependency Provisioning Run, Dependency Install Unblocks Training, Dependency Install Entry

### Community 22 - "Adversarial Ramp Loss"
Cohesion: 0.5
Nodes (2): Compute adversarial loss ramp factor based on current step., Compute the composite loss with curriculum warmup.          Args:             bi

### Community 23 - "V4 Step 9000 Panel"
Cohesion: 0.67
Nodes (3): Step 009000 cover image grid, Step 009000 residual intensity map, Step 009000 stego image grid

### Community 24 - "V4 Step 9500 Panel"
Cohesion: 0.67
Nodes (3): Step 009500 cover image grid, Step 009500 residual intensity map, Step 009500 stego image grid

### Community 25 - "V4 Step 10500 Panel"
Cohesion: 0.67
Nodes (3): Step 010500 cover image grid, Step 010500 residual intensity map, Step 010500 stego image grid

### Community 26 - "RTX5090 Config Notes"
Cohesion: 1.0
Nodes (1): DSCSP v4 Config Override for RTX 5090 (32GB VRAM).  The RTX 5090 has ample VRAM

### Community 27 - "Training Log Templates"
Cohesion: 1.0
Nodes (2): Training Entry Template, DSCSP Training Logbook

### Community 28 - "Chat Archive Metadata"
Cohesion: 1.0
Nodes (2): Chat History Archive, Current Session Transcript JSONL

### Community 29 - "Package Init Node A"
Cohesion: 1.0
Nodes (0): 

### Community 30 - "Package Init Node B"
Cohesion: 1.0
Nodes (0): 

### Community 31 - "DCT Matrix Builder"
Cohesion: 1.0
Nodes (1): Build the NxN DCT-II transform matrix.

### Community 32 - "RGB to YCbCr"
Cohesion: 1.0
Nodes (1): Convert RGB [0, 255] to YCbCr.

### Community 33 - "YCbCr to RGB"
Cohesion: 1.0
Nodes (1): Convert YCbCr to RGB [0, 255].

### Community 34 - "Frozen UNet Inference"
Cohesion: 1.0
Nodes (1): Run the frozen UNet to predict noise. Output is FP32, fully detached.          A

### Community 35 - "Clean DDIM Sampling"
Cohesion: 1.0
Nodes (1): Generate a clean (unperturbed) image using standard DDIM sampling.         Usefu

## Ambiguous Edges - Review These
- `Step 018500 residual intensity map` → `Possible late-stage residual brightening around step 18500`  [AMBIGUOUS]
  outputs_v4/step_018500.png · relation: references
- `Visual Fidelity Assessment` → `Localized Residual Spike`  [AMBIGUOUS]
  outputs_v5/step_034500.png · relation: conceptually_related_to
- `Visual Fidelity Assessment` → `Localized Residual Spike`  [AMBIGUOUS]
  outputs_v5/step_036000.png · relation: conceptually_related_to
- `Visual Fidelity Assessment` → `Localized Residual Spike`  [AMBIGUOUS]
  outputs_v5/step_040500.png · relation: conceptually_related_to
- `Visual Fidelity Assessment` → `Localized Residual Spike`  [AMBIGUOUS]
  outputs_v5/step_043500.png · relation: conceptually_related_to
- `Step 057500 residual map grid showing localized high-energy facial regions` → `Step 058000 quality-training montage with localized residual hotspot`  [AMBIGUOUS]
  outputs_v5_quality/step_058000.png · relation: shares_data_with

## Knowledge Gaps
- **160 isolated node(s):** `DSCSP v4 Config Override for RTX 5090 (32GB VRAM).  The RTX 5090 has ample VRAM`, `Iterated final training orchestration for DSCSP v4.  This script runs multi-stag`, `DSCSP v5 Ablation Study.  Trains 6 lightweight variants (8 000 steps each) from`, `Return the base v5 config for all ablation variants.`, `Config for the shared warm-start run (5 000 steps).` (+155 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `RTX5090 Config Notes`** (2 nodes): `train_v4_5090.py`, `DSCSP v4 Config Override for RTX 5090 (32GB VRAM).  The RTX 5090 has ample VRAM`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Training Log Templates`** (2 nodes): `Training Entry Template`, `DSCSP Training Logbook`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Chat Archive Metadata`** (2 nodes): `Chat History Archive`, `Current Session Transcript JSONL`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init Node A`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init Node B`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `DCT Matrix Builder`** (1 nodes): `Build the NxN DCT-II transform matrix.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `RGB to YCbCr`** (1 nodes): `Convert RGB [0, 255] to YCbCr.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `YCbCr to RGB`** (1 nodes): `Convert YCbCr to RGB [0, 255].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Frozen UNet Inference`** (1 nodes): `Run the frozen UNet to predict noise. Output is FP32, fully detached.          A`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Clean DDIM Sampling`** (1 nodes): `Generate a clean (unperturbed) image using standard DDIM sampling.         Usefu`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Step 018500 residual intensity map` and `Possible late-stage residual brightening around step 18500`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Visual Fidelity Assessment` and `Localized Residual Spike`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Visual Fidelity Assessment` and `Localized Residual Spike`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Visual Fidelity Assessment` and `Localized Residual Spike`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Visual Fidelity Assessment` and `Localized Residual Spike`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Step 057500 residual map grid showing localized high-energy facial regions` and `Step 058000 quality-training montage with localized residual hotspot`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `train()` connect `Core Metrics Pipeline` to `Training Major Log Utilities`, `JPEG Eval Pipeline`, `Visualization Utility Module`, `Ablation Eval Orchestration`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._