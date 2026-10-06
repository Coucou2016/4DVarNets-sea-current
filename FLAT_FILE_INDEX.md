# FLAT_FILE_INDEX.md

Authoritative map between the **flat** (top-level, no folders) filenames in this repository and their **original** relative paths in the structured tree.

Separator convention: every `/` in the original path becomes `__` (double underscore). Example: `docs/paper/figures/fig_sqg_skill.png` -> `docs__paper__figures__fig_sqg_skill.png`.

See [`FLAT_LAYOUT_NOTICE.md`](FLAT_LAYOUT_NOTICE.md) for the rationale.

## Summary

- Flat files indexed: **333**
- Total payload: **8038356 bytes** (8.04 MB)
- Flat separator: `__`
- Restore the structured, runnable tree with:
  `python flat_unflatten.py --out .`

Counts by group:

| Group | Files | Bytes |
|-------|-------|-------|
| Repository root files | 8 | 598820 |
| Source code (fourdvarnet / data / scripts / tests) | 46 | 279464 |
| Configuration (config/) | 3 | 3446 |
| Documentation (docs/) | 29 | 215941 |
| Paper (docs/paper/) | 41 | 1862091 |
| Report (docs/report/) | 35 | 3032436 |
| Results (results/) | 117 | 1999594 |
| Training history (checkpoints/) | 54 | 46564 |

## Repository root files

| Flat filename | Original path | Bytes |
|---------------|---------------|-------|
| `.gitignore` | `.gitignore` | 640 |
| `LICENSE` | `LICENSE` | 1105 |
| `NOTICE` | `NOTICE` | 1256 |
| `README.md` | `README.md` | 9773 |
| `environment.yml` | `environment.yml` | 825 |
| `pytest.ini` | `pytest.ini` | 59 |
| `report.html` | `report.html` | 585032 |
| `requirements.txt` | `requirements.txt` | 130 |

## Source code (fourdvarnet / data / scripts / tests)

| Flat filename | Original path | Bytes |
|---------------|---------------|-------|
| `data____init__.py` | `data/__init__.py` | 232 |
| `data__dataset.py` | `data/dataset.py` | 13187 |
| `data__natl60.py` | `data/natl60.py` | 20856 |
| `data__natl60__.gitkeep` | `data/natl60/.gitkeep` | 0 |
| `data__real.py` | `data/real.py` | 4655 |
| `data__real__.gitkeep` | `data/real/.gitkeep` | 0 |
| `data__synthetic.py` | `data/synthetic.py` | 4686 |
| `fourdvarnet____init__.py` | `fourdvarnet/__init__.py` | 217 |
| `fourdvarnet__convlstm.py` | `fourdvarnet/convlstm.py` | 2488 |
| `fourdvarnet__geometry.py` | `fourdvarnet/geometry.py` | 9893 |
| `fourdvarnet__losses.py` | `fourdvarnet/losses.py` | 4836 |
| `fourdvarnet__metrics.py` | `fourdvarnet/metrics.py` | 11664 |
| `fourdvarnet__model.py` | `fourdvarnet/model.py` | 5002 |
| `fourdvarnet__observation.py` | `fourdvarnet/observation.py` | 2534 |
| `fourdvarnet__physics.py` | `fourdvarnet/physics.py` | 11504 |
| `fourdvarnet__plotting.py` | `fourdvarnet/plotting.py` | 5859 |
| `fourdvarnet__prior.py` | `fourdvarnet/prior.py` | 1375 |
| `fourdvarnet__repro.py` | `fourdvarnet/repro.py` | 1200 |
| `fourdvarnet__solver.py` | `fourdvarnet/solver.py` | 9855 |
| `scripts__assemble_paper.py` | `scripts/assemble_paper.py` | 10112 |
| `scripts__build_report.py` | `scripts/build_report.py` | 27977 |
| `scripts__download_natl60.py` | `scripts/download_natl60.py` | 6903 |
| `scripts__evaluate.py` | `scripts/evaluate.py` | 11822 |
| `scripts__evaluate_drifters.py` | `scripts/evaluate_drifters.py` | 5432 |
| `scripts__finalize_post_p0.py` | `scripts/finalize_post_p0.py` | 3108 |
| `scripts__generate_synthetic.py` | `scripts/generate_synthetic.py` | 830 |
| `scripts__plot_science.py` | `scripts/plot_science.py` | 10932 |
| `scripts__run_ablation.py` | `scripts/run_ablation.py` | 6346 |
| `scripts__run_post_p0_pipeline.py` | `scripts/run_post_p0_pipeline.py` | 9309 |
| `scripts__run_sensitivity.py` | `scripts/run_sensitivity.py` | 9860 |
| `scripts__run_verify.py` | `scripts/run_verify.py` | 1034 |
| `scripts__smoke_test.py` | `scripts/smoke_test.py` | 3113 |
| `scripts__train.py` | `scripts/train.py` | 10145 |
| `scripts__validate_physics_operators.py` | `scripts/validate_physics_operators.py` | 11799 |
| `scripts__verify_natl60.py` | `scripts/verify_natl60.py` | 3262 |
| `tests__conftest.py` | `tests/conftest.py` | 161 |
| `tests__test_config.py` | `tests/test_config.py` | 818 |
| `tests__test_losses.py` | `tests/test_losses.py` | 2930 |
| `tests__test_metrics.py` | `tests/test_metrics.py` | 2556 |
| `tests__test_model.py` | `tests/test_model.py` | 2692 |
| `tests__test_natl60.py` | `tests/test_natl60.py` | 3311 |
| `tests__test_p0_correctness.py` | `tests/test_p0_correctness.py` | 15402 |
| `tests__test_physics.py` | `tests/test_physics.py` | 4379 |
| `tests__test_real.py` | `tests/test_real.py` | 1341 |
| `tests__test_synthetic.py` | `tests/test_synthetic.py` | 520 |
| `tests__test_training.py` | `tests/test_training.py` | 3327 |

## Configuration (config/)

| Flat filename | Original path | Bytes |
|---------------|---------------|-------|
| `config__default.yaml` | `config/default.yaml` | 1742 |
| `config__paths.yaml` | `config/paths.yaml` | 492 |
| `config__r0_compact_faithful.yaml` | `config/r0_compact_faithful.yaml` | 1212 |

## Documentation (docs/)

| Flat filename | Original path | Bytes |
|---------------|---------------|-------|
| `docs__AUDIT_EVIDENCE.html` | `docs/AUDIT_EVIDENCE.html` | 8410 |
| `docs__AUDIT_EVIDENCE.md` | `docs/AUDIT_EVIDENCE.md` | 5729 |
| `docs__AUDIT_EVIDENCE.pdf` | `docs/AUDIT_EVIDENCE.pdf` | 127170 |
| `docs__PAPER_EXPERIMENTS.md` | `docs/PAPER_EXPERIMENTS.md` | 8033 |
| `docs__REVIEW_RESPONSE_P0.md` | `docs/REVIEW_RESPONSE_P0.md` | 3742 |
| `docs__REVIEW_RESPONSE_ROUND2.md` | `docs/REVIEW_RESPONSE_ROUND2.md` | 2695 |
| `docs__STAGE_EFGH_LEDGER.md` | `docs/STAGE_EFGH_LEDGER.md` | 2006 |
| `docs__chatgpt_collaboration__PASTE_ChatA_GH_LIT_2026-08-16.txt` | `docs/chatgpt_collaboration/PASTE_ChatA_GH_LIT_2026-08-16.txt` | 2879 |
| `docs__chatgpt_collaboration__PASTE_ChatA_M4_GPU_2026-08-16.txt` | `docs/chatgpt_collaboration/PASTE_ChatA_M4_GPU_2026-08-16.txt` | 907 |
| `docs__chatgpt_collaboration__PASTE_ChatB_LIT_FRAMEWORK_2026-08-16.txt` | `docs/chatgpt_collaboration/PASTE_ChatB_LIT_FRAMEWORK_2026-08-16.txt` | 1621 |
| `docs__chatgpt_collaboration__PASTE_ChatB_LIT_FRAMEWORK_2026-08-16_cont.txt` | `docs/chatgpt_collaboration/PASTE_ChatB_LIT_FRAMEWORK_2026-08-16_cont.txt` | 1849 |
| `docs__chatgpt_collaboration__PASTE_ChatB_LIT_FRAMEWORK_2026-08-16_report.txt` | `docs/chatgpt_collaboration/PASTE_ChatB_LIT_FRAMEWORK_2026-08-16_report.txt` | 1870 |
| `docs__chatgpt_collaboration__PASTE_ChatB_REPORT_REVIEW_2026-08-16.txt` | `docs/chatgpt_collaboration/PASTE_ChatB_REPORT_REVIEW_2026-08-16.txt` | 1068 |
| `docs__chatgpt_collaboration__PASTE_R1_ARCH_2026-08-16.txt` | `docs/chatgpt_collaboration/PASTE_R1_ARCH_2026-08-16.txt` | 722 |
| `docs__chatgpt_collaboration__PASTE_R2_METHODS_2026-08-16.txt` | `docs/chatgpt_collaboration/PASTE_R2_METHODS_2026-08-16.txt` | 506 |
| `docs__chatgpt_collaboration__PASTE_R3_RESULTS_2026-08-16.txt` | `docs/chatgpt_collaboration/PASTE_R3_RESULTS_2026-08-16.txt` | 544 |
| `docs__chatgpt_collaboration__PASTE_R4_DISCUSSION_2026-08-16.txt` | `docs/chatgpt_collaboration/PASTE_R4_DISCUSSION_2026-08-16.txt` | 404 |
| `docs__chatgpt_collaboration__PASTE_R5_POLISH_2026-08-16.txt` | `docs/chatgpt_collaboration/PASTE_R5_POLISH_2026-08-16.txt` | 420 |
| `docs__chatgpt_collaboration__SESSION_2026-08-15.md` | `docs/chatgpt_collaboration/SESSION_2026-08-15.md` | 4132 |
| `docs__chatgpt_collaboration__SESSION_2026-08-15_cont.md` | `docs/chatgpt_collaboration/SESSION_2026-08-15_cont.md` | 2495 |
| `docs__chatgpt_collaboration__SESSION_2026-08-15_gpu.md` | `docs/chatgpt_collaboration/SESSION_2026-08-15_gpu.md` | 2284 |
| `docs__chatgpt_collaboration__SESSION_2026-08-16.md` | `docs/chatgpt_collaboration/SESSION_2026-08-16.md` | 5842 |
| `docs__chatgpt_collaboration__SESSION_2026-08-16_cont.md` | `docs/chatgpt_collaboration/SESSION_2026-08-16_cont.md` | 4540 |
| `docs__chatgpt_collaboration__SESSION_2026-08-16_gh.md` | `docs/chatgpt_collaboration/SESSION_2026-08-16_gh.md` | 3644 |
| `docs__chatgpt_collaboration__SESSION_2026-08-16_report.md` | `docs/chatgpt_collaboration/SESSION_2026-08-16_report.md` | 3776 |
| `docs__chatgpt_collaboration__SESSION_5ROUNDS.md` | `docs/chatgpt_collaboration/SESSION_5ROUNDS.md` | 5474 |
| `docs__chatgpt_collaboration__TASK_BRIEF_GPU_NATL60_REVIEW.md` | `docs/chatgpt_collaboration/TASK_BRIEF_GPU_NATL60_REVIEW.md` | 3918 |
| `docs__chatgpt_collaboration__TASK_BRIEF_M4_FIX.md` | `docs/chatgpt_collaboration/TASK_BRIEF_M4_FIX.md` | 7084 |
| `docs__chatgpt_collaboration__rounds__R1_full_log.txt` | `docs/chatgpt_collaboration/rounds/R1_full_log.txt` | 2177 |

## Paper (docs/paper/)

| Flat filename | Original path | Bytes |
|---------------|---------------|-------|
| `docs__paper__01_introduction.md` | `docs/paper/01_introduction.md` | 4524 |
| `docs__paper__02_methods.md` | `docs/paper/02_methods.md` | 5949 |
| `docs__paper__03_experiments.md` | `docs/paper/03_experiments.md` | 4212 |
| `docs__paper__04_discussion.md` | `docs/paper/04_discussion.md` | 2772 |
| `docs__paper__DRAFT_FRAMEWORK.md` | `docs/paper/DRAFT_FRAMEWORK.md` | 3379 |
| `docs__paper__OUTLINE.md` | `docs/paper/OUTLINE.md` | 5336 |
| `docs__paper__PAPER.md` | `docs/paper/PAPER.md` | 23295 |
| `docs__paper__figures__FIGURE_NOTES.md` | `docs/paper/figures/FIGURE_NOTES.md` | 1467 |
| `docs__paper__figures__fig_B2_GPU96_loss.pdf` | `docs/paper/figures/fig_B2_GPU96_loss.pdf` | 31343 |
| `docs__paper__figures__fig_B2_GPU96_loss.png` | `docs/paper/figures/fig_B2_GPU96_loss.png` | 72075 |
| `docs__paper__figures__fig_B2_GPU96_tau_uv.pdf` | `docs/paper/figures/fig_B2_GPU96_tau_uv.pdf` | 33252 |
| `docs__paper__figures__fig_B2_GPU96_tau_uv.png` | `docs/paper/figures/fig_B2_GPU96_tau_uv.png` | 50880 |
| `docs__paper__figures__fig_GPU96_lambda_x_uv.pdf` | `docs/paper/figures/fig_GPU96_lambda_x_uv.pdf` | 37537 |
| `docs__paper__figures__fig_GPU96_lambda_x_uv.png` | `docs/paper/figures/fig_GPU96_lambda_x_uv.png` | 59573 |
| `docs__paper__figures__fig_GPU96_rmse_uv.pdf` | `docs/paper/figures/fig_GPU96_rmse_uv.pdf` | 34251 |
| `docs__paper__figures__fig_GPU96_rmse_uv.png` | `docs/paper/figures/fig_GPU96_rmse_uv.png` | 73060 |
| `docs__paper__figures__fig_GPU96_tau_div.pdf` | `docs/paper/figures/fig_GPU96_tau_div.pdf` | 34566 |
| `docs__paper__figures__fig_GPU96_tau_div.png` | `docs/paper/figures/fig_GPU96_tau_div.png` | 65996 |
| `docs__paper__figures__fig_GPU96_tau_uv.pdf` | `docs/paper/figures/fig_GPU96_tau_uv.pdf` | 33965 |
| `docs__paper__figures__fig_GPU96_tau_uv.png` | `docs/paper/figures/fig_GPU96_tau_uv.png` | 55434 |
| `docs__paper__figures__fig_M3_GPU96_loss.pdf` | `docs/paper/figures/fig_M3_GPU96_loss.pdf` | 31142 |
| `docs__paper__figures__fig_M3_GPU96_loss.png` | `docs/paper/figures/fig_M3_GPU96_loss.png` | 82654 |
| `docs__paper__figures__fig_M4_GPU96_loss.pdf` | `docs/paper/figures/fig_M4_GPU96_loss.pdf` | 31335 |
| `docs__paper__figures__fig_M4_GPU96_loss.png` | `docs/paper/figures/fig_M4_GPU96_loss.png` | 93665 |
| `docs__paper__figures__fig_adv_residual_sanity.pdf` | `docs/paper/figures/fig_adv_residual_sanity.pdf` | 31283 |
| `docs__paper__figures__fig_adv_residual_sanity.png` | `docs/paper/figures/fig_adv_residual_sanity.png` | 60552 |
| `docs__paper__figures__fig_post_p0_rmse_ssh.pdf` | `docs/paper/figures/fig_post_p0_rmse_ssh.pdf` | 35019 |
| `docs__paper__figures__fig_post_p0_rmse_ssh.png` | `docs/paper/figures/fig_post_p0_rmse_ssh.png` | 71008 |
| `docs__paper__figures__fig_post_p0_rmse_uv.pdf` | `docs/paper/figures/fig_post_p0_rmse_uv.pdf` | 35437 |
| `docs__paper__figures__fig_post_p0_rmse_uv.png` | `docs/paper/figures/fig_post_p0_rmse_uv.png` | 77937 |
| `docs__paper__figures__fig_post_p0_tau_uv.pdf` | `docs/paper/figures/fig_post_p0_tau_uv.pdf` | 37319 |
| `docs__paper__figures__fig_post_p0_tau_uv.png` | `docs/paper/figures/fig_post_p0_tau_uv.png` | 61119 |
| `docs__paper__figures__fig_sqg_skill.pdf` | `docs/paper/figures/fig_sqg_skill.pdf` | 28130 |
| `docs__paper__figures__fig_sqg_skill.png` | `docs/paper/figures/fig_sqg_skill.png` | 40378 |
| `docs__paper__figures__fig_synth_ablation_rmse_uv.pdf` | `docs/paper/figures/fig_synth_ablation_rmse_uv.pdf` | 32075 |
| `docs__paper__figures__fig_synth_ablation_rmse_uv.png` | `docs/paper/figures/fig_synth_ablation_rmse_uv.png` | 62085 |
| `docs__paper__figures__fig_synth_ablation_tau_uv.pdf` | `docs/paper/figures/fig_synth_ablation_tau_uv.pdf` | 32999 |
| `docs__paper__figures__fig_synth_ablation_tau_uv.png` | `docs/paper/figures/fig_synth_ablation_tau_uv.png` | 59113 |
| `docs__paper__paper.html` | `docs/paper/paper.html` | 27926 |
| `docs__paper__paper.pdf` | `docs/paper/paper.pdf` | 297964 |
| `docs__paper__pdf_build_note.txt` | `docs/paper/pdf_build_note.txt` | 85 |

## Report (docs/report/)

| Flat filename | Original path | Bytes |
|---------------|---------------|-------|
| `docs__report__PDF_STATUS.txt` | `docs/report/PDF_STATUS.txt` | 87 |
| `docs__report__figures__fig_B2_GPU96_loss.pdf` | `docs/report/figures/fig_B2_GPU96_loss.pdf` | 31343 |
| `docs__report__figures__fig_B2_GPU96_loss.png` | `docs/report/figures/fig_B2_GPU96_loss.png` | 72075 |
| `docs__report__figures__fig_B2_GPU96_tau_uv.pdf` | `docs/report/figures/fig_B2_GPU96_tau_uv.pdf` | 33252 |
| `docs__report__figures__fig_B2_GPU96_tau_uv.png` | `docs/report/figures/fig_B2_GPU96_tau_uv.png` | 50880 |
| `docs__report__figures__fig_GPU96_lambda_x_uv.pdf` | `docs/report/figures/fig_GPU96_lambda_x_uv.pdf` | 37537 |
| `docs__report__figures__fig_GPU96_lambda_x_uv.png` | `docs/report/figures/fig_GPU96_lambda_x_uv.png` | 59573 |
| `docs__report__figures__fig_GPU96_rmse_uv.pdf` | `docs/report/figures/fig_GPU96_rmse_uv.pdf` | 34251 |
| `docs__report__figures__fig_GPU96_rmse_uv.png` | `docs/report/figures/fig_GPU96_rmse_uv.png` | 73060 |
| `docs__report__figures__fig_GPU96_tau_div.pdf` | `docs/report/figures/fig_GPU96_tau_div.pdf` | 34566 |
| `docs__report__figures__fig_GPU96_tau_div.png` | `docs/report/figures/fig_GPU96_tau_div.png` | 65996 |
| `docs__report__figures__fig_GPU96_tau_uv.pdf` | `docs/report/figures/fig_GPU96_tau_uv.pdf` | 33965 |
| `docs__report__figures__fig_GPU96_tau_uv.png` | `docs/report/figures/fig_GPU96_tau_uv.png` | 55434 |
| `docs__report__figures__fig_M3_GPU96_loss.pdf` | `docs/report/figures/fig_M3_GPU96_loss.pdf` | 31142 |
| `docs__report__figures__fig_M3_GPU96_loss.png` | `docs/report/figures/fig_M3_GPU96_loss.png` | 82654 |
| `docs__report__figures__fig_M4_GPU96_loss.pdf` | `docs/report/figures/fig_M4_GPU96_loss.pdf` | 31335 |
| `docs__report__figures__fig_M4_GPU96_loss.png` | `docs/report/figures/fig_M4_GPU96_loss.png` | 93665 |
| `docs__report__figures__fig_adv_residual_sanity.pdf` | `docs/report/figures/fig_adv_residual_sanity.pdf` | 31283 |
| `docs__report__figures__fig_adv_residual_sanity.png` | `docs/report/figures/fig_adv_residual_sanity.png` | 60552 |
| `docs__report__figures__fig_post_p0_rmse_ssh.pdf` | `docs/report/figures/fig_post_p0_rmse_ssh.pdf` | 35019 |
| `docs__report__figures__fig_post_p0_rmse_ssh.png` | `docs/report/figures/fig_post_p0_rmse_ssh.png` | 71008 |
| `docs__report__figures__fig_post_p0_rmse_uv.pdf` | `docs/report/figures/fig_post_p0_rmse_uv.pdf` | 35437 |
| `docs__report__figures__fig_post_p0_rmse_uv.png` | `docs/report/figures/fig_post_p0_rmse_uv.png` | 77937 |
| `docs__report__figures__fig_post_p0_tau_uv.pdf` | `docs/report/figures/fig_post_p0_tau_uv.pdf` | 37319 |
| `docs__report__figures__fig_post_p0_tau_uv.png` | `docs/report/figures/fig_post_p0_tau_uv.png` | 61119 |
| `docs__report__figures__fig_sqg_skill.pdf` | `docs/report/figures/fig_sqg_skill.pdf` | 28130 |
| `docs__report__figures__fig_sqg_skill.png` | `docs/report/figures/fig_sqg_skill.png` | 40378 |
| `docs__report__figures__fig_synth_ablation_rmse_uv.pdf` | `docs/report/figures/fig_synth_ablation_rmse_uv.pdf` | 32075 |
| `docs__report__figures__fig_synth_ablation_rmse_uv.png` | `docs/report/figures/fig_synth_ablation_rmse_uv.png` | 62085 |
| `docs__report__figures__fig_synth_ablation_tau_uv.pdf` | `docs/report/figures/fig_synth_ablation_tau_uv.pdf` | 32999 |
| `docs__report__figures__fig_synth_ablation_tau_uv.png` | `docs/report/figures/fig_synth_ablation_tau_uv.png` | 59113 |
| `docs__report__pdf_build_note.txt` | `docs/report/pdf_build_note.txt` | 87 |
| `docs__report__report.html` | `docs/report/report.html` | 585032 |
| `docs__report__report.md` | `docs/report/report.md` | 5643 |
| `docs__report__report.pdf` | `docs/report/report.pdf` | 956405 |

## Results (results/)

| Flat filename | Original path | Bytes |
|---------------|---------------|-------|
| `results__.gitkeep` | `results/.gitkeep` | 80 |
| `results__SYNTHETIC_ABLATION_NOTES.md` | `results/SYNTHETIC_ABLATION_NOTES.md` | 1174 |
| `results__ablation_8ep_post_m4fix.txt` | `results/ablation_8ep_post_m4fix.txt` | 4538 |
| `results__ablation_summary.json` | `results/ablation_summary.json` | 5167 |
| `results__build_report_gh.txt` | `results/build_report_gh.txt` | 318 |
| `results__deferred_eval_watch.txt` | `results/deferred_eval_watch.txt` | 2778 |
| `results__figures__FIGURE_NOTES.md` | `results/figures/FIGURE_NOTES.md` | 659 |
| `results__figures__fig_B2_GPU96_loss.pdf` | `results/figures/fig_B2_GPU96_loss.pdf` | 31343 |
| `results__figures__fig_B2_GPU96_loss.png` | `results/figures/fig_B2_GPU96_loss.png` | 72075 |
| `results__figures__fig_B2_GPU96_tau_uv.pdf` | `results/figures/fig_B2_GPU96_tau_uv.pdf` | 33252 |
| `results__figures__fig_B2_GPU96_tau_uv.png` | `results/figures/fig_B2_GPU96_tau_uv.png` | 50880 |
| `results__figures__fig_GPU96_lambda_x_uv.pdf` | `results/figures/fig_GPU96_lambda_x_uv.pdf` | 37537 |
| `results__figures__fig_GPU96_lambda_x_uv.png` | `results/figures/fig_GPU96_lambda_x_uv.png` | 59573 |
| `results__figures__fig_GPU96_rmse_uv.pdf` | `results/figures/fig_GPU96_rmse_uv.pdf` | 34251 |
| `results__figures__fig_GPU96_rmse_uv.png` | `results/figures/fig_GPU96_rmse_uv.png` | 73060 |
| `results__figures__fig_GPU96_tau_div.pdf` | `results/figures/fig_GPU96_tau_div.pdf` | 34566 |
| `results__figures__fig_GPU96_tau_div.png` | `results/figures/fig_GPU96_tau_div.png` | 65996 |
| `results__figures__fig_GPU96_tau_uv.pdf` | `results/figures/fig_GPU96_tau_uv.pdf` | 33965 |
| `results__figures__fig_GPU96_tau_uv.png` | `results/figures/fig_GPU96_tau_uv.png` | 55434 |
| `results__figures__fig_M3_GPU96_loss.pdf` | `results/figures/fig_M3_GPU96_loss.pdf` | 31142 |
| `results__figures__fig_M3_GPU96_loss.png` | `results/figures/fig_M3_GPU96_loss.png` | 82654 |
| `results__figures__fig_M4_GPU96_loss.pdf` | `results/figures/fig_M4_GPU96_loss.pdf` | 31335 |
| `results__figures__fig_M4_GPU96_loss.png` | `results/figures/fig_M4_GPU96_loss.png` | 93665 |
| `results__figures__fig_adv_residual_sanity.pdf` | `results/figures/fig_adv_residual_sanity.pdf` | 31283 |
| `results__figures__fig_adv_residual_sanity.png` | `results/figures/fig_adv_residual_sanity.png` | 60552 |
| `results__figures__fig_post_p0_rmse_ssh.pdf` | `results/figures/fig_post_p0_rmse_ssh.pdf` | 35019 |
| `results__figures__fig_post_p0_rmse_ssh.png` | `results/figures/fig_post_p0_rmse_ssh.png` | 71008 |
| `results__figures__fig_post_p0_rmse_uv.pdf` | `results/figures/fig_post_p0_rmse_uv.pdf` | 35437 |
| `results__figures__fig_post_p0_rmse_uv.png` | `results/figures/fig_post_p0_rmse_uv.png` | 77937 |
| `results__figures__fig_post_p0_tau_uv.pdf` | `results/figures/fig_post_p0_tau_uv.pdf` | 37319 |
| `results__figures__fig_post_p0_tau_uv.png` | `results/figures/fig_post_p0_tau_uv.png` | 61119 |
| `results__figures__fig_sqg_skill.pdf` | `results/figures/fig_sqg_skill.pdf` | 28130 |
| `results__figures__fig_sqg_skill.png` | `results/figures/fig_sqg_skill.png` | 40378 |
| `results__figures__fig_synth_ablation_rmse_uv.pdf` | `results/figures/fig_synth_ablation_rmse_uv.pdf` | 32075 |
| `results__figures__fig_synth_ablation_rmse_uv.png` | `results/figures/fig_synth_ablation_rmse_uv.png` | 62085 |
| `results__figures__fig_synth_ablation_tau_uv.pdf` | `results/figures/fig_synth_ablation_tau_uv.pdf` | 32999 |
| `results__figures__fig_synth_ablation_tau_uv.png` | `results/figures/fig_synth_ablation_tau_uv.png` | 59113 |
| `results__gpu_pytest.txt` | `results/gpu_pytest.txt` | 1549 |
| `results__gpu_smoke.txt` | `results/gpu_smoke.txt` | 223 |
| `results__legacy_pre_review2__README.md` | `results/legacy_pre_review2/README.md` | 419 |
| `results__legacy_pre_review2__b2_gpu96_watch.txt` | `results/legacy_pre_review2/b2_gpu96_watch.txt` | 132 |
| `results__legacy_pre_review2__eval_B2_GPU96.txt` | `results/legacy_pre_review2/eval_B2_GPU96.txt` | 1916 |
| `results__legacy_pre_review2__eval_B2_GPU96_err.txt` | `results/legacy_pre_review2/eval_B2_GPU96_err.txt` | 0 |
| `results__legacy_pre_review2__eval_M3_GPU96.txt` | `results/legacy_pre_review2/eval_M3_GPU96.txt` | 960 |
| `results__legacy_pre_review2__eval_M4_GPU96.txt` | `results/legacy_pre_review2/eval_M4_GPU96.txt` | 960 |
| `results__legacy_pre_review2__m4_gpu96_post_nllfix_watch.txt` | `results/legacy_pre_review2/m4_gpu96_post_nllfix_watch.txt` | 2 |
| `results__legacy_pre_review2__m4_gpu96_watch.txt` | `results/legacy_pre_review2/m4_gpu96_watch.txt` | 3972 |
| `results__legacy_pre_review2__metrics_B2_GPU96.json` | `results/legacy_pre_review2/metrics_B2_GPU96.json` | 1761 |
| `results__legacy_pre_review2__metrics_GPU96_expanded_summary.json` | `results/legacy_pre_review2/metrics_GPU96_expanded_summary.json` | 1876 |
| `results__legacy_pre_review2__metrics_M3_GPU96.json` | `results/legacy_pre_review2/metrics_M3_GPU96.json` | 1763 |
| `results__legacy_pre_review2__metrics_M4_GPU96.json` | `results/legacy_pre_review2/metrics_M4_GPU96.json` | 1760 |
| `results__legacy_pre_review2__metrics_M4_GPU96_pre_nllfix.json` | `results/legacy_pre_review2/metrics_M4_GPU96_pre_nllfix.json` | 1766 |
| `results__legacy_pre_review2__train_B2_GPU96.txt` | `results/legacy_pre_review2/train_B2_GPU96.txt` | 836 |
| `results__legacy_pre_review2__train_M3_GPU96.txt` | `results/legacy_pre_review2/train_M3_GPU96.txt` | 1692 |
| `results__legacy_pre_review2__train_M4_GPU96.txt` | `results/legacy_pre_review2/train_M4_GPU96.txt` | 1694 |
| `results__legacy_pre_review2__train_M4_GPU96_post_nllfix.txt` | `results/legacy_pre_review2/train_M4_GPU96_post_nllfix.txt` | 1694 |
| `results__m4fix_eval.txt` | `results/m4fix_eval.txt` | 711 |
| `results__m4fix_train.txt` | `results/m4fix_train.txt` | 415 |
| `results__metrics_B2.json` | `results/metrics_B2.json` | 1430 |
| `results__metrics_M3.json` | `results/metrics_M3.json` | 1765 |
| `results__metrics_M4.json` | `results/metrics_M4.json` | 1779 |
| `results__metrics_M4fix.json` | `results/metrics_M4fix.json` | 1380 |
| `results__metrics_NATL60smoke.json` | `results/metrics_NATL60smoke.json` | 1760 |
| `results__natl60_smoke_eval.txt` | `results/natl60_smoke_eval.txt` | 969 |
| `results__natl60_smoke_train.txt` | `results/natl60_smoke_train.txt` | 176 |
| `results__physics_ops__fig_adv_residual_sanity.pdf` | `results/physics_ops/fig_adv_residual_sanity.pdf` | 31283 |
| `results__physics_ops__fig_adv_residual_sanity.png` | `results/physics_ops/fig_adv_residual_sanity.png` | 60552 |
| `results__physics_ops__fig_sqg_skill.pdf` | `results/physics_ops/fig_sqg_skill.pdf` | 28130 |
| `results__physics_ops__fig_sqg_skill.png` | `results/physics_ops/fig_sqg_skill.png` | 40378 |
| `results__physics_ops__physics_ops_validation.json` | `results/physics_ops/physics_ops_validation.json` | 17574 |
| `results__physics_ops_err.txt` | `results/physics_ops_err.txt` | 0 |
| `results__physics_ops_run.txt` | `results/physics_ops_run.txt` | 498 |
| `results__plot_after_eval.txt` | `results/plot_after_eval.txt` | 2890 |
| `results__plot_after_gh.txt` | `results/plot_after_gh.txt` | 2816 |
| `results__plot_after_m4.txt` | `results/plot_after_m4.txt` | 2816 |
| `results__post_p0__STAGES_EFGH_FINAL.json` | `results/post_p0/STAGES_EFGH_FINAL.json` | 21066 |
| `results__post_p0__ablation_summary.json` | `results/post_p0/ablation_summary.json` | 105431 |
| `results__post_p0__eval_B2_s0_retry.txt` | `results/post_p0/eval_B2_s0_retry.txt` | 1358 |
| `results__post_p0__figure_tau_uv_means.json` | `results/post_p0/figure_tau_uv_means.json` | 1851 |
| `results__post_p0__metrics_B1-s0.json` | `results/post_p0/metrics_B1-s0.json` | 3531 |
| `results__post_p0__metrics_B1-s1.json` | `results/post_p0/metrics_B1-s1.json` | 3542 |
| `results__post_p0__metrics_B1-s2.json` | `results/post_p0/metrics_B1-s2.json` | 3554 |
| `results__post_p0__metrics_B2-s0.json` | `results/post_p0/metrics_B2-s0.json` | 3492 |
| `results__post_p0__metrics_B2-s1.json` | `results/post_p0/metrics_B2-s1.json` | 3547 |
| `results__post_p0__metrics_B2-s2.json` | `results/post_p0/metrics_B2-s2.json` | 3548 |
| `results__post_p0__metrics_M1-s0.json` | `results/post_p0/metrics_M1-s0.json` | 3548 |
| `results__post_p0__metrics_M1-s1.json` | `results/post_p0/metrics_M1-s1.json` | 3542 |
| `results__post_p0__metrics_M1-s2.json` | `results/post_p0/metrics_M1-s2.json` | 3552 |
| `results__post_p0__metrics_M2-s0.json` | `results/post_p0/metrics_M2-s0.json` | 3547 |
| `results__post_p0__metrics_M2-s1.json` | `results/post_p0/metrics_M2-s1.json` | 3551 |
| `results__post_p0__metrics_M2-s2.json` | `results/post_p0/metrics_M2-s2.json` | 3537 |
| `results__post_p0__metrics_M3-s0.json` | `results/post_p0/metrics_M3-s0.json` | 3545 |
| `results__post_p0__metrics_M3-s1.json` | `results/post_p0/metrics_M3-s1.json` | 3530 |
| `results__post_p0__metrics_M3-s2.json` | `results/post_p0/metrics_M3-s2.json` | 3550 |
| `results__post_p0__metrics_M4-s0.json` | `results/post_p0/metrics_M4-s0.json` | 3546 |
| `results__post_p0__metrics_M4-s1.json` | `results/post_p0/metrics_M4-s1.json` | 3542 |
| `results__post_p0__metrics_M4-s2.json` | `results/post_p0/metrics_M4-s2.json` | 3541 |
| `results__post_p0__metrics_R0-s0.json` | `results/post_p0/metrics_R0-s0.json` | 3539 |
| `results__post_p0__metrics_R0-s1.json` | `results/post_p0/metrics_R0-s1.json` | 3546 |
| `results__post_p0__metrics_R0-s2.json` | `results/post_p0/metrics_R0-s2.json` | 3544 |
| `results__post_p0__monitor.txt` | `results/post_p0/monitor.txt` | 748 |
| `results__post_p0__monitor2.txt` | `results/post_p0/monitor2.txt` | 180 |
| `results__post_p0__monitor3.txt` | `results/post_p0/monitor3.txt` | 1371 |
| `results__post_p0__pipeline_err.txt` | `results/post_p0/pipeline_err.txt` | 0 |
| `results__post_p0__pipeline_err2.txt` | `results/post_p0/pipeline_err2.txt` | 0 |
| `results__post_p0__pipeline_run.txt` | `results/post_p0/pipeline_run.txt` | 9784 |
| `results__post_p0__pipeline_run2.txt` | `results/post_p0/pipeline_run2.txt` | 48077 |
| `results__post_p0__restart.txt` | `results/post_p0/restart.txt` | 53 |
| `results__post_p0__sensitivity__sensitivity_summary.json` | `results/post_p0/sensitivity/sensitivity_summary.json` | 8765 |
| `results__post_p0__sensitivity_run.txt` | `results/post_p0/sensitivity_run.txt` | 415 |
| `results__pytest_5rounds.txt` | `results/pytest_5rounds.txt` | 81 |
| `results__pytest_last.txt` | `results/pytest_last.txt` | 1054 |
| `results__pytest_m4fix.txt` | `results/pytest_m4fix.txt` | 81 |
| `results__pytest_partial_last.txt` | `results/pytest_partial_last.txt` | 1079 |
| `results__pytest_post_p0_tooling.txt` | `results/pytest_post_p0_tooling.txt` | 81 |
| `results__smoke_last.txt` | `results/smoke_last.txt` | 223 |
| `results__verify_natl60_last.txt` | `results/verify_natl60_last.txt` | 1329 |

## Training history (checkpoints/)

| Flat filename | Original path | Bytes |
|---------------|---------------|-------|
| `checkpoints__ablation_summary.json` | `checkpoints/ablation_summary.json` | 5851 |
| `checkpoints__history-B1-s0-meta.json` | `checkpoints/history-B1-s0-meta.json` | 97 |
| `checkpoints__history-B1-s0.json` | `checkpoints/history-B1-s0.json` | 1376 |
| `checkpoints__history-B1-s1-meta.json` | `checkpoints/history-B1-s1-meta.json` | 96 |
| `checkpoints__history-B1-s1.json` | `checkpoints/history-B1-s1.json` | 1369 |
| `checkpoints__history-B1-s2-meta.json` | `checkpoints/history-B1-s2-meta.json` | 96 |
| `checkpoints__history-B1-s2.json` | `checkpoints/history-B1-s2.json` | 1375 |
| `checkpoints__history-B1.json` | `checkpoints/history-B1.json` | 96 |
| `checkpoints__history-B2-GPU96.json` | `checkpoints/history-B2-GPU96.json` | 1831 |
| `checkpoints__history-B2-s0-meta.json` | `checkpoints/history-B2-s0-meta.json` | 97 |
| `checkpoints__history-B2-s0.json` | `checkpoints/history-B2-s0.json` | 1374 |
| `checkpoints__history-B2-s1-meta.json` | `checkpoints/history-B2-s1-meta.json` | 96 |
| `checkpoints__history-B2-s1.json` | `checkpoints/history-B2-s1.json` | 1371 |
| `checkpoints__history-B2-s2-meta.json` | `checkpoints/history-B2-s2-meta.json` | 97 |
| `checkpoints__history-B2-s2.json` | `checkpoints/history-B2-s2.json` | 1388 |
| `checkpoints__history-B2.json` | `checkpoints/history-B2.json` | 744 |
| `checkpoints__history-M1-s0-meta.json` | `checkpoints/history-M1-s0-meta.json` | 97 |
| `checkpoints__history-M1-s0.json` | `checkpoints/history-M1-s0.json` | 1385 |
| `checkpoints__history-M1-s1-meta.json` | `checkpoints/history-M1-s1-meta.json` | 96 |
| `checkpoints__history-M1-s1.json` | `checkpoints/history-M1-s1.json` | 1391 |
| `checkpoints__history-M1-s2-meta.json` | `checkpoints/history-M1-s2-meta.json` | 97 |
| `checkpoints__history-M1-s2.json` | `checkpoints/history-M1-s2.json` | 1383 |
| `checkpoints__history-M2-s0-meta.json` | `checkpoints/history-M2-s0-meta.json` | 97 |
| `checkpoints__history-M2-s0.json` | `checkpoints/history-M2-s0.json` | 1378 |
| `checkpoints__history-M2-s1-meta.json` | `checkpoints/history-M2-s1-meta.json` | 95 |
| `checkpoints__history-M2-s1.json` | `checkpoints/history-M2-s1.json` | 1371 |
| `checkpoints__history-M2-s2-meta.json` | `checkpoints/history-M2-s2-meta.json` | 97 |
| `checkpoints__history-M2-s2.json` | `checkpoints/history-M2-s2.json` | 1390 |
| `checkpoints__history-M3-GPU96.json` | `checkpoints/history-M3-GPU96.json` | 1837 |
| `checkpoints__history-M3-s0-meta.json` | `checkpoints/history-M3-s0-meta.json` | 97 |
| `checkpoints__history-M3-s0.json` | `checkpoints/history-M3-s0.json` | 1385 |
| `checkpoints__history-M3-s1-meta.json` | `checkpoints/history-M3-s1-meta.json` | 97 |
| `checkpoints__history-M3-s1.json` | `checkpoints/history-M3-s1.json` | 1391 |
| `checkpoints__history-M3-s2-meta.json` | `checkpoints/history-M3-s2-meta.json` | 97 |
| `checkpoints__history-M3-s2.json` | `checkpoints/history-M3-s2.json` | 1383 |
| `checkpoints__history-M3.json` | `checkpoints/history-M3.json` | 744 |
| `checkpoints__history-M4-GPU96.json` | `checkpoints/history-M4-GPU96.json` | 1846 |
| `checkpoints__history-M4-s0-meta.json` | `checkpoints/history-M4-s0-meta.json` | 97 |
| `checkpoints__history-M4-s0.json` | `checkpoints/history-M4-s0.json` | 1383 |
| `checkpoints__history-M4-s1-meta.json` | `checkpoints/history-M4-s1-meta.json` | 97 |
| `checkpoints__history-M4-s1.json` | `checkpoints/history-M4-s1.json` | 1390 |
| `checkpoints__history-M4-s2-meta.json` | `checkpoints/history-M4-s2-meta.json` | 97 |
| `checkpoints__history-M4-s2.json` | `checkpoints/history-M4-s2.json` | 1380 |
| `checkpoints__history-M4.json` | `checkpoints/history-M4.json` | 741 |
| `checkpoints__history-M4fix.json` | `checkpoints/history-M4fix.json` | 736 |
| `checkpoints__history-NATL60smoke.json` | `checkpoints/history-NATL60smoke.json` | 94 |
| `checkpoints__history-P0smoke.json` | `checkpoints/history-P0smoke.json` | 95 |
| `checkpoints__history-R0-s0-meta.json` | `checkpoints/history-R0-s0-meta.json` | 97 |
| `checkpoints__history-R0-s0.json` | `checkpoints/history-R0-s0.json` | 1378 |
| `checkpoints__history-R0-s1-meta.json` | `checkpoints/history-R0-s1-meta.json` | 96 |
| `checkpoints__history-R0-s1.json` | `checkpoints/history-R0-s1.json` | 1371 |
| `checkpoints__history-R0-s2-meta.json` | `checkpoints/history-R0-s2-meta.json` | 96 |
| `checkpoints__history-R0-s2.json` | `checkpoints/history-R0-s2.json` | 1376 |
| `checkpoints__history.json` | `checkpoints/history.json` | 932 |

## Machine-readable map (JSON)

<!-- FLAT_MAP_JSON_BEGIN -->
```json
{
  "separator": "__",
  "total_files": 333,
  "total_bytes": 8038356,
  "files": [
    {
      "flat": ".gitignore",
      "original": ".gitignore",
      "bytes": 640,
      "group": "root"
    },
    {
      "flat": "LICENSE",
      "original": "LICENSE",
      "bytes": 1105,
      "group": "root"
    },
    {
      "flat": "NOTICE",
      "original": "NOTICE",
      "bytes": 1256,
      "group": "root"
    },
    {
      "flat": "README.md",
      "original": "README.md",
      "bytes": 9773,
      "group": "root"
    },
    {
      "flat": "checkpoints__ablation_summary.json",
      "original": "checkpoints/ablation_summary.json",
      "bytes": 5851,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B1-s0-meta.json",
      "original": "checkpoints/history-B1-s0-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B1-s0.json",
      "original": "checkpoints/history-B1-s0.json",
      "bytes": 1376,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B1-s1-meta.json",
      "original": "checkpoints/history-B1-s1-meta.json",
      "bytes": 96,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B1-s1.json",
      "original": "checkpoints/history-B1-s1.json",
      "bytes": 1369,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B1-s2-meta.json",
      "original": "checkpoints/history-B1-s2-meta.json",
      "bytes": 96,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B1-s2.json",
      "original": "checkpoints/history-B1-s2.json",
      "bytes": 1375,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B1.json",
      "original": "checkpoints/history-B1.json",
      "bytes": 96,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B2-GPU96.json",
      "original": "checkpoints/history-B2-GPU96.json",
      "bytes": 1831,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B2-s0-meta.json",
      "original": "checkpoints/history-B2-s0-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B2-s0.json",
      "original": "checkpoints/history-B2-s0.json",
      "bytes": 1374,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B2-s1-meta.json",
      "original": "checkpoints/history-B2-s1-meta.json",
      "bytes": 96,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B2-s1.json",
      "original": "checkpoints/history-B2-s1.json",
      "bytes": 1371,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B2-s2-meta.json",
      "original": "checkpoints/history-B2-s2-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B2-s2.json",
      "original": "checkpoints/history-B2-s2.json",
      "bytes": 1388,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-B2.json",
      "original": "checkpoints/history-B2.json",
      "bytes": 744,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M1-s0-meta.json",
      "original": "checkpoints/history-M1-s0-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M1-s0.json",
      "original": "checkpoints/history-M1-s0.json",
      "bytes": 1385,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M1-s1-meta.json",
      "original": "checkpoints/history-M1-s1-meta.json",
      "bytes": 96,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M1-s1.json",
      "original": "checkpoints/history-M1-s1.json",
      "bytes": 1391,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M1-s2-meta.json",
      "original": "checkpoints/history-M1-s2-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M1-s2.json",
      "original": "checkpoints/history-M1-s2.json",
      "bytes": 1383,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M2-s0-meta.json",
      "original": "checkpoints/history-M2-s0-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M2-s0.json",
      "original": "checkpoints/history-M2-s0.json",
      "bytes": 1378,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M2-s1-meta.json",
      "original": "checkpoints/history-M2-s1-meta.json",
      "bytes": 95,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M2-s1.json",
      "original": "checkpoints/history-M2-s1.json",
      "bytes": 1371,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M2-s2-meta.json",
      "original": "checkpoints/history-M2-s2-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M2-s2.json",
      "original": "checkpoints/history-M2-s2.json",
      "bytes": 1390,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M3-GPU96.json",
      "original": "checkpoints/history-M3-GPU96.json",
      "bytes": 1837,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M3-s0-meta.json",
      "original": "checkpoints/history-M3-s0-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M3-s0.json",
      "original": "checkpoints/history-M3-s0.json",
      "bytes": 1385,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M3-s1-meta.json",
      "original": "checkpoints/history-M3-s1-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M3-s1.json",
      "original": "checkpoints/history-M3-s1.json",
      "bytes": 1391,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M3-s2-meta.json",
      "original": "checkpoints/history-M3-s2-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M3-s2.json",
      "original": "checkpoints/history-M3-s2.json",
      "bytes": 1383,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M3.json",
      "original": "checkpoints/history-M3.json",
      "bytes": 744,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M4-GPU96.json",
      "original": "checkpoints/history-M4-GPU96.json",
      "bytes": 1846,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M4-s0-meta.json",
      "original": "checkpoints/history-M4-s0-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M4-s0.json",
      "original": "checkpoints/history-M4-s0.json",
      "bytes": 1383,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M4-s1-meta.json",
      "original": "checkpoints/history-M4-s1-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M4-s1.json",
      "original": "checkpoints/history-M4-s1.json",
      "bytes": 1390,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M4-s2-meta.json",
      "original": "checkpoints/history-M4-s2-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M4-s2.json",
      "original": "checkpoints/history-M4-s2.json",
      "bytes": 1380,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M4.json",
      "original": "checkpoints/history-M4.json",
      "bytes": 741,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-M4fix.json",
      "original": "checkpoints/history-M4fix.json",
      "bytes": 736,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-NATL60smoke.json",
      "original": "checkpoints/history-NATL60smoke.json",
      "bytes": 94,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-P0smoke.json",
      "original": "checkpoints/history-P0smoke.json",
      "bytes": 95,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-R0-s0-meta.json",
      "original": "checkpoints/history-R0-s0-meta.json",
      "bytes": 97,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-R0-s0.json",
      "original": "checkpoints/history-R0-s0.json",
      "bytes": 1378,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-R0-s1-meta.json",
      "original": "checkpoints/history-R0-s1-meta.json",
      "bytes": 96,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-R0-s1.json",
      "original": "checkpoints/history-R0-s1.json",
      "bytes": 1371,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-R0-s2-meta.json",
      "original": "checkpoints/history-R0-s2-meta.json",
      "bytes": 96,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history-R0-s2.json",
      "original": "checkpoints/history-R0-s2.json",
      "bytes": 1376,
      "group": "checkpoints"
    },
    {
      "flat": "checkpoints__history.json",
      "original": "checkpoints/history.json",
      "bytes": 932,
      "group": "checkpoints"
    },
    {
      "flat": "config__default.yaml",
      "original": "config/default.yaml",
      "bytes": 1742,
      "group": "config"
    },
    {
      "flat": "config__paths.yaml",
      "original": "config/paths.yaml",
      "bytes": 492,
      "group": "config"
    },
    {
      "flat": "config__r0_compact_faithful.yaml",
      "original": "config/r0_compact_faithful.yaml",
      "bytes": 1212,
      "group": "config"
    },
    {
      "flat": "data____init__.py",
      "original": "data/__init__.py",
      "bytes": 232,
      "group": "code"
    },
    {
      "flat": "data__dataset.py",
      "original": "data/dataset.py",
      "bytes": 13187,
      "group": "code"
    },
    {
      "flat": "data__natl60.py",
      "original": "data/natl60.py",
      "bytes": 20856,
      "group": "code"
    },
    {
      "flat": "data__natl60__.gitkeep",
      "original": "data/natl60/.gitkeep",
      "bytes": 0,
      "group": "code"
    },
    {
      "flat": "data__real.py",
      "original": "data/real.py",
      "bytes": 4655,
      "group": "code"
    },
    {
      "flat": "data__real__.gitkeep",
      "original": "data/real/.gitkeep",
      "bytes": 0,
      "group": "code"
    },
    {
      "flat": "data__synthetic.py",
      "original": "data/synthetic.py",
      "bytes": 4686,
      "group": "code"
    },
    {
      "flat": "docs__AUDIT_EVIDENCE.html",
      "original": "docs/AUDIT_EVIDENCE.html",
      "bytes": 8410,
      "group": "docs"
    },
    {
      "flat": "docs__AUDIT_EVIDENCE.md",
      "original": "docs/AUDIT_EVIDENCE.md",
      "bytes": 5729,
      "group": "docs"
    },
    {
      "flat": "docs__AUDIT_EVIDENCE.pdf",
      "original": "docs/AUDIT_EVIDENCE.pdf",
      "bytes": 127170,
      "group": "docs"
    },
    {
      "flat": "docs__PAPER_EXPERIMENTS.md",
      "original": "docs/PAPER_EXPERIMENTS.md",
      "bytes": 8033,
      "group": "docs"
    },
    {
      "flat": "docs__REVIEW_RESPONSE_P0.md",
      "original": "docs/REVIEW_RESPONSE_P0.md",
      "bytes": 3742,
      "group": "docs"
    },
    {
      "flat": "docs__REVIEW_RESPONSE_ROUND2.md",
      "original": "docs/REVIEW_RESPONSE_ROUND2.md",
      "bytes": 2695,
      "group": "docs"
    },
    {
      "flat": "docs__STAGE_EFGH_LEDGER.md",
      "original": "docs/STAGE_EFGH_LEDGER.md",
      "bytes": 2006,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__PASTE_ChatA_GH_LIT_2026-08-16.txt",
      "original": "docs/chatgpt_collaboration/PASTE_ChatA_GH_LIT_2026-08-16.txt",
      "bytes": 2879,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__PASTE_ChatA_M4_GPU_2026-08-16.txt",
      "original": "docs/chatgpt_collaboration/PASTE_ChatA_M4_GPU_2026-08-16.txt",
      "bytes": 907,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__PASTE_ChatB_LIT_FRAMEWORK_2026-08-16.txt",
      "original": "docs/chatgpt_collaboration/PASTE_ChatB_LIT_FRAMEWORK_2026-08-16.txt",
      "bytes": 1621,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__PASTE_ChatB_LIT_FRAMEWORK_2026-08-16_cont.txt",
      "original": "docs/chatgpt_collaboration/PASTE_ChatB_LIT_FRAMEWORK_2026-08-16_cont.txt",
      "bytes": 1849,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__PASTE_ChatB_LIT_FRAMEWORK_2026-08-16_report.txt",
      "original": "docs/chatgpt_collaboration/PASTE_ChatB_LIT_FRAMEWORK_2026-08-16_report.txt",
      "bytes": 1870,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__PASTE_ChatB_REPORT_REVIEW_2026-08-16.txt",
      "original": "docs/chatgpt_collaboration/PASTE_ChatB_REPORT_REVIEW_2026-08-16.txt",
      "bytes": 1068,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__PASTE_R1_ARCH_2026-08-16.txt",
      "original": "docs/chatgpt_collaboration/PASTE_R1_ARCH_2026-08-16.txt",
      "bytes": 722,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__PASTE_R2_METHODS_2026-08-16.txt",
      "original": "docs/chatgpt_collaboration/PASTE_R2_METHODS_2026-08-16.txt",
      "bytes": 506,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__PASTE_R3_RESULTS_2026-08-16.txt",
      "original": "docs/chatgpt_collaboration/PASTE_R3_RESULTS_2026-08-16.txt",
      "bytes": 544,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__PASTE_R4_DISCUSSION_2026-08-16.txt",
      "original": "docs/chatgpt_collaboration/PASTE_R4_DISCUSSION_2026-08-16.txt",
      "bytes": 404,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__PASTE_R5_POLISH_2026-08-16.txt",
      "original": "docs/chatgpt_collaboration/PASTE_R5_POLISH_2026-08-16.txt",
      "bytes": 420,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__SESSION_2026-08-15.md",
      "original": "docs/chatgpt_collaboration/SESSION_2026-08-15.md",
      "bytes": 4132,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__SESSION_2026-08-15_cont.md",
      "original": "docs/chatgpt_collaboration/SESSION_2026-08-15_cont.md",
      "bytes": 2495,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__SESSION_2026-08-15_gpu.md",
      "original": "docs/chatgpt_collaboration/SESSION_2026-08-15_gpu.md",
      "bytes": 2284,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__SESSION_2026-08-16.md",
      "original": "docs/chatgpt_collaboration/SESSION_2026-08-16.md",
      "bytes": 5842,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__SESSION_2026-08-16_cont.md",
      "original": "docs/chatgpt_collaboration/SESSION_2026-08-16_cont.md",
      "bytes": 4540,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__SESSION_2026-08-16_gh.md",
      "original": "docs/chatgpt_collaboration/SESSION_2026-08-16_gh.md",
      "bytes": 3644,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__SESSION_2026-08-16_report.md",
      "original": "docs/chatgpt_collaboration/SESSION_2026-08-16_report.md",
      "bytes": 3776,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__SESSION_5ROUNDS.md",
      "original": "docs/chatgpt_collaboration/SESSION_5ROUNDS.md",
      "bytes": 5474,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__TASK_BRIEF_GPU_NATL60_REVIEW.md",
      "original": "docs/chatgpt_collaboration/TASK_BRIEF_GPU_NATL60_REVIEW.md",
      "bytes": 3918,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__TASK_BRIEF_M4_FIX.md",
      "original": "docs/chatgpt_collaboration/TASK_BRIEF_M4_FIX.md",
      "bytes": 7084,
      "group": "docs"
    },
    {
      "flat": "docs__chatgpt_collaboration__rounds__R1_full_log.txt",
      "original": "docs/chatgpt_collaboration/rounds/R1_full_log.txt",
      "bytes": 2177,
      "group": "docs"
    },
    {
      "flat": "docs__paper__01_introduction.md",
      "original": "docs/paper/01_introduction.md",
      "bytes": 4524,
      "group": "paper"
    },
    {
      "flat": "docs__paper__02_methods.md",
      "original": "docs/paper/02_methods.md",
      "bytes": 5949,
      "group": "paper"
    },
    {
      "flat": "docs__paper__03_experiments.md",
      "original": "docs/paper/03_experiments.md",
      "bytes": 4212,
      "group": "paper"
    },
    {
      "flat": "docs__paper__04_discussion.md",
      "original": "docs/paper/04_discussion.md",
      "bytes": 2772,
      "group": "paper"
    },
    {
      "flat": "docs__paper__DRAFT_FRAMEWORK.md",
      "original": "docs/paper/DRAFT_FRAMEWORK.md",
      "bytes": 3379,
      "group": "paper"
    },
    {
      "flat": "docs__paper__OUTLINE.md",
      "original": "docs/paper/OUTLINE.md",
      "bytes": 5336,
      "group": "paper"
    },
    {
      "flat": "docs__paper__PAPER.md",
      "original": "docs/paper/PAPER.md",
      "bytes": 23295,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__FIGURE_NOTES.md",
      "original": "docs/paper/figures/FIGURE_NOTES.md",
      "bytes": 1467,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_B2_GPU96_loss.pdf",
      "original": "docs/paper/figures/fig_B2_GPU96_loss.pdf",
      "bytes": 31343,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_B2_GPU96_loss.png",
      "original": "docs/paper/figures/fig_B2_GPU96_loss.png",
      "bytes": 72075,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_B2_GPU96_tau_uv.pdf",
      "original": "docs/paper/figures/fig_B2_GPU96_tau_uv.pdf",
      "bytes": 33252,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_B2_GPU96_tau_uv.png",
      "original": "docs/paper/figures/fig_B2_GPU96_tau_uv.png",
      "bytes": 50880,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_GPU96_lambda_x_uv.pdf",
      "original": "docs/paper/figures/fig_GPU96_lambda_x_uv.pdf",
      "bytes": 37537,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_GPU96_lambda_x_uv.png",
      "original": "docs/paper/figures/fig_GPU96_lambda_x_uv.png",
      "bytes": 59573,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_GPU96_rmse_uv.pdf",
      "original": "docs/paper/figures/fig_GPU96_rmse_uv.pdf",
      "bytes": 34251,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_GPU96_rmse_uv.png",
      "original": "docs/paper/figures/fig_GPU96_rmse_uv.png",
      "bytes": 73060,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_GPU96_tau_div.pdf",
      "original": "docs/paper/figures/fig_GPU96_tau_div.pdf",
      "bytes": 34566,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_GPU96_tau_div.png",
      "original": "docs/paper/figures/fig_GPU96_tau_div.png",
      "bytes": 65996,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_GPU96_tau_uv.pdf",
      "original": "docs/paper/figures/fig_GPU96_tau_uv.pdf",
      "bytes": 33965,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_GPU96_tau_uv.png",
      "original": "docs/paper/figures/fig_GPU96_tau_uv.png",
      "bytes": 55434,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_M3_GPU96_loss.pdf",
      "original": "docs/paper/figures/fig_M3_GPU96_loss.pdf",
      "bytes": 31142,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_M3_GPU96_loss.png",
      "original": "docs/paper/figures/fig_M3_GPU96_loss.png",
      "bytes": 82654,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_M4_GPU96_loss.pdf",
      "original": "docs/paper/figures/fig_M4_GPU96_loss.pdf",
      "bytes": 31335,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_M4_GPU96_loss.png",
      "original": "docs/paper/figures/fig_M4_GPU96_loss.png",
      "bytes": 93665,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_adv_residual_sanity.pdf",
      "original": "docs/paper/figures/fig_adv_residual_sanity.pdf",
      "bytes": 31283,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_adv_residual_sanity.png",
      "original": "docs/paper/figures/fig_adv_residual_sanity.png",
      "bytes": 60552,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_post_p0_rmse_ssh.pdf",
      "original": "docs/paper/figures/fig_post_p0_rmse_ssh.pdf",
      "bytes": 35019,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_post_p0_rmse_ssh.png",
      "original": "docs/paper/figures/fig_post_p0_rmse_ssh.png",
      "bytes": 71008,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_post_p0_rmse_uv.pdf",
      "original": "docs/paper/figures/fig_post_p0_rmse_uv.pdf",
      "bytes": 35437,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_post_p0_rmse_uv.png",
      "original": "docs/paper/figures/fig_post_p0_rmse_uv.png",
      "bytes": 77937,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_post_p0_tau_uv.pdf",
      "original": "docs/paper/figures/fig_post_p0_tau_uv.pdf",
      "bytes": 37319,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_post_p0_tau_uv.png",
      "original": "docs/paper/figures/fig_post_p0_tau_uv.png",
      "bytes": 61119,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_sqg_skill.pdf",
      "original": "docs/paper/figures/fig_sqg_skill.pdf",
      "bytes": 28130,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_sqg_skill.png",
      "original": "docs/paper/figures/fig_sqg_skill.png",
      "bytes": 40378,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_synth_ablation_rmse_uv.pdf",
      "original": "docs/paper/figures/fig_synth_ablation_rmse_uv.pdf",
      "bytes": 32075,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_synth_ablation_rmse_uv.png",
      "original": "docs/paper/figures/fig_synth_ablation_rmse_uv.png",
      "bytes": 62085,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_synth_ablation_tau_uv.pdf",
      "original": "docs/paper/figures/fig_synth_ablation_tau_uv.pdf",
      "bytes": 32999,
      "group": "paper"
    },
    {
      "flat": "docs__paper__figures__fig_synth_ablation_tau_uv.png",
      "original": "docs/paper/figures/fig_synth_ablation_tau_uv.png",
      "bytes": 59113,
      "group": "paper"
    },
    {
      "flat": "docs__paper__paper.html",
      "original": "docs/paper/paper.html",
      "bytes": 27926,
      "group": "paper"
    },
    {
      "flat": "docs__paper__paper.pdf",
      "original": "docs/paper/paper.pdf",
      "bytes": 297964,
      "group": "paper"
    },
    {
      "flat": "docs__paper__pdf_build_note.txt",
      "original": "docs/paper/pdf_build_note.txt",
      "bytes": 85,
      "group": "paper"
    },
    {
      "flat": "docs__report__PDF_STATUS.txt",
      "original": "docs/report/PDF_STATUS.txt",
      "bytes": 87,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_B2_GPU96_loss.pdf",
      "original": "docs/report/figures/fig_B2_GPU96_loss.pdf",
      "bytes": 31343,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_B2_GPU96_loss.png",
      "original": "docs/report/figures/fig_B2_GPU96_loss.png",
      "bytes": 72075,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_B2_GPU96_tau_uv.pdf",
      "original": "docs/report/figures/fig_B2_GPU96_tau_uv.pdf",
      "bytes": 33252,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_B2_GPU96_tau_uv.png",
      "original": "docs/report/figures/fig_B2_GPU96_tau_uv.png",
      "bytes": 50880,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_GPU96_lambda_x_uv.pdf",
      "original": "docs/report/figures/fig_GPU96_lambda_x_uv.pdf",
      "bytes": 37537,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_GPU96_lambda_x_uv.png",
      "original": "docs/report/figures/fig_GPU96_lambda_x_uv.png",
      "bytes": 59573,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_GPU96_rmse_uv.pdf",
      "original": "docs/report/figures/fig_GPU96_rmse_uv.pdf",
      "bytes": 34251,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_GPU96_rmse_uv.png",
      "original": "docs/report/figures/fig_GPU96_rmse_uv.png",
      "bytes": 73060,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_GPU96_tau_div.pdf",
      "original": "docs/report/figures/fig_GPU96_tau_div.pdf",
      "bytes": 34566,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_GPU96_tau_div.png",
      "original": "docs/report/figures/fig_GPU96_tau_div.png",
      "bytes": 65996,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_GPU96_tau_uv.pdf",
      "original": "docs/report/figures/fig_GPU96_tau_uv.pdf",
      "bytes": 33965,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_GPU96_tau_uv.png",
      "original": "docs/report/figures/fig_GPU96_tau_uv.png",
      "bytes": 55434,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_M3_GPU96_loss.pdf",
      "original": "docs/report/figures/fig_M3_GPU96_loss.pdf",
      "bytes": 31142,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_M3_GPU96_loss.png",
      "original": "docs/report/figures/fig_M3_GPU96_loss.png",
      "bytes": 82654,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_M4_GPU96_loss.pdf",
      "original": "docs/report/figures/fig_M4_GPU96_loss.pdf",
      "bytes": 31335,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_M4_GPU96_loss.png",
      "original": "docs/report/figures/fig_M4_GPU96_loss.png",
      "bytes": 93665,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_adv_residual_sanity.pdf",
      "original": "docs/report/figures/fig_adv_residual_sanity.pdf",
      "bytes": 31283,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_adv_residual_sanity.png",
      "original": "docs/report/figures/fig_adv_residual_sanity.png",
      "bytes": 60552,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_post_p0_rmse_ssh.pdf",
      "original": "docs/report/figures/fig_post_p0_rmse_ssh.pdf",
      "bytes": 35019,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_post_p0_rmse_ssh.png",
      "original": "docs/report/figures/fig_post_p0_rmse_ssh.png",
      "bytes": 71008,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_post_p0_rmse_uv.pdf",
      "original": "docs/report/figures/fig_post_p0_rmse_uv.pdf",
      "bytes": 35437,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_post_p0_rmse_uv.png",
      "original": "docs/report/figures/fig_post_p0_rmse_uv.png",
      "bytes": 77937,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_post_p0_tau_uv.pdf",
      "original": "docs/report/figures/fig_post_p0_tau_uv.pdf",
      "bytes": 37319,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_post_p0_tau_uv.png",
      "original": "docs/report/figures/fig_post_p0_tau_uv.png",
      "bytes": 61119,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_sqg_skill.pdf",
      "original": "docs/report/figures/fig_sqg_skill.pdf",
      "bytes": 28130,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_sqg_skill.png",
      "original": "docs/report/figures/fig_sqg_skill.png",
      "bytes": 40378,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_synth_ablation_rmse_uv.pdf",
      "original": "docs/report/figures/fig_synth_ablation_rmse_uv.pdf",
      "bytes": 32075,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_synth_ablation_rmse_uv.png",
      "original": "docs/report/figures/fig_synth_ablation_rmse_uv.png",
      "bytes": 62085,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_synth_ablation_tau_uv.pdf",
      "original": "docs/report/figures/fig_synth_ablation_tau_uv.pdf",
      "bytes": 32999,
      "group": "report"
    },
    {
      "flat": "docs__report__figures__fig_synth_ablation_tau_uv.png",
      "original": "docs/report/figures/fig_synth_ablation_tau_uv.png",
      "bytes": 59113,
      "group": "report"
    },
    {
      "flat": "docs__report__pdf_build_note.txt",
      "original": "docs/report/pdf_build_note.txt",
      "bytes": 87,
      "group": "report"
    },
    {
      "flat": "docs__report__report.html",
      "original": "docs/report/report.html",
      "bytes": 585032,
      "group": "report"
    },
    {
      "flat": "docs__report__report.md",
      "original": "docs/report/report.md",
      "bytes": 5643,
      "group": "report"
    },
    {
      "flat": "docs__report__report.pdf",
      "original": "docs/report/report.pdf",
      "bytes": 956405,
      "group": "report"
    },
    {
      "flat": "environment.yml",
      "original": "environment.yml",
      "bytes": 825,
      "group": "root"
    },
    {
      "flat": "fourdvarnet____init__.py",
      "original": "fourdvarnet/__init__.py",
      "bytes": 217,
      "group": "code"
    },
    {
      "flat": "fourdvarnet__convlstm.py",
      "original": "fourdvarnet/convlstm.py",
      "bytes": 2488,
      "group": "code"
    },
    {
      "flat": "fourdvarnet__geometry.py",
      "original": "fourdvarnet/geometry.py",
      "bytes": 9893,
      "group": "code"
    },
    {
      "flat": "fourdvarnet__losses.py",
      "original": "fourdvarnet/losses.py",
      "bytes": 4836,
      "group": "code"
    },
    {
      "flat": "fourdvarnet__metrics.py",
      "original": "fourdvarnet/metrics.py",
      "bytes": 11664,
      "group": "code"
    },
    {
      "flat": "fourdvarnet__model.py",
      "original": "fourdvarnet/model.py",
      "bytes": 5002,
      "group": "code"
    },
    {
      "flat": "fourdvarnet__observation.py",
      "original": "fourdvarnet/observation.py",
      "bytes": 2534,
      "group": "code"
    },
    {
      "flat": "fourdvarnet__physics.py",
      "original": "fourdvarnet/physics.py",
      "bytes": 11504,
      "group": "code"
    },
    {
      "flat": "fourdvarnet__plotting.py",
      "original": "fourdvarnet/plotting.py",
      "bytes": 5859,
      "group": "code"
    },
    {
      "flat": "fourdvarnet__prior.py",
      "original": "fourdvarnet/prior.py",
      "bytes": 1375,
      "group": "code"
    },
    {
      "flat": "fourdvarnet__repro.py",
      "original": "fourdvarnet/repro.py",
      "bytes": 1200,
      "group": "code"
    },
    {
      "flat": "fourdvarnet__solver.py",
      "original": "fourdvarnet/solver.py",
      "bytes": 9855,
      "group": "code"
    },
    {
      "flat": "pytest.ini",
      "original": "pytest.ini",
      "bytes": 59,
      "group": "root"
    },
    {
      "flat": "report.html",
      "original": "report.html",
      "bytes": 585032,
      "group": "root"
    },
    {
      "flat": "requirements.txt",
      "original": "requirements.txt",
      "bytes": 130,
      "group": "root"
    },
    {
      "flat": "results__.gitkeep",
      "original": "results/.gitkeep",
      "bytes": 80,
      "group": "results"
    },
    {
      "flat": "results__SYNTHETIC_ABLATION_NOTES.md",
      "original": "results/SYNTHETIC_ABLATION_NOTES.md",
      "bytes": 1174,
      "group": "results"
    },
    {
      "flat": "results__ablation_8ep_post_m4fix.txt",
      "original": "results/ablation_8ep_post_m4fix.txt",
      "bytes": 4538,
      "group": "results"
    },
    {
      "flat": "results__ablation_summary.json",
      "original": "results/ablation_summary.json",
      "bytes": 5167,
      "group": "results"
    },
    {
      "flat": "results__build_report_gh.txt",
      "original": "results/build_report_gh.txt",
      "bytes": 318,
      "group": "results"
    },
    {
      "flat": "results__deferred_eval_watch.txt",
      "original": "results/deferred_eval_watch.txt",
      "bytes": 2778,
      "group": "results"
    },
    {
      "flat": "results__figures__FIGURE_NOTES.md",
      "original": "results/figures/FIGURE_NOTES.md",
      "bytes": 659,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_B2_GPU96_loss.pdf",
      "original": "results/figures/fig_B2_GPU96_loss.pdf",
      "bytes": 31343,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_B2_GPU96_loss.png",
      "original": "results/figures/fig_B2_GPU96_loss.png",
      "bytes": 72075,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_B2_GPU96_tau_uv.pdf",
      "original": "results/figures/fig_B2_GPU96_tau_uv.pdf",
      "bytes": 33252,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_B2_GPU96_tau_uv.png",
      "original": "results/figures/fig_B2_GPU96_tau_uv.png",
      "bytes": 50880,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_GPU96_lambda_x_uv.pdf",
      "original": "results/figures/fig_GPU96_lambda_x_uv.pdf",
      "bytes": 37537,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_GPU96_lambda_x_uv.png",
      "original": "results/figures/fig_GPU96_lambda_x_uv.png",
      "bytes": 59573,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_GPU96_rmse_uv.pdf",
      "original": "results/figures/fig_GPU96_rmse_uv.pdf",
      "bytes": 34251,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_GPU96_rmse_uv.png",
      "original": "results/figures/fig_GPU96_rmse_uv.png",
      "bytes": 73060,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_GPU96_tau_div.pdf",
      "original": "results/figures/fig_GPU96_tau_div.pdf",
      "bytes": 34566,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_GPU96_tau_div.png",
      "original": "results/figures/fig_GPU96_tau_div.png",
      "bytes": 65996,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_GPU96_tau_uv.pdf",
      "original": "results/figures/fig_GPU96_tau_uv.pdf",
      "bytes": 33965,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_GPU96_tau_uv.png",
      "original": "results/figures/fig_GPU96_tau_uv.png",
      "bytes": 55434,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_M3_GPU96_loss.pdf",
      "original": "results/figures/fig_M3_GPU96_loss.pdf",
      "bytes": 31142,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_M3_GPU96_loss.png",
      "original": "results/figures/fig_M3_GPU96_loss.png",
      "bytes": 82654,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_M4_GPU96_loss.pdf",
      "original": "results/figures/fig_M4_GPU96_loss.pdf",
      "bytes": 31335,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_M4_GPU96_loss.png",
      "original": "results/figures/fig_M4_GPU96_loss.png",
      "bytes": 93665,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_adv_residual_sanity.pdf",
      "original": "results/figures/fig_adv_residual_sanity.pdf",
      "bytes": 31283,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_adv_residual_sanity.png",
      "original": "results/figures/fig_adv_residual_sanity.png",
      "bytes": 60552,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_post_p0_rmse_ssh.pdf",
      "original": "results/figures/fig_post_p0_rmse_ssh.pdf",
      "bytes": 35019,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_post_p0_rmse_ssh.png",
      "original": "results/figures/fig_post_p0_rmse_ssh.png",
      "bytes": 71008,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_post_p0_rmse_uv.pdf",
      "original": "results/figures/fig_post_p0_rmse_uv.pdf",
      "bytes": 35437,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_post_p0_rmse_uv.png",
      "original": "results/figures/fig_post_p0_rmse_uv.png",
      "bytes": 77937,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_post_p0_tau_uv.pdf",
      "original": "results/figures/fig_post_p0_tau_uv.pdf",
      "bytes": 37319,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_post_p0_tau_uv.png",
      "original": "results/figures/fig_post_p0_tau_uv.png",
      "bytes": 61119,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_sqg_skill.pdf",
      "original": "results/figures/fig_sqg_skill.pdf",
      "bytes": 28130,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_sqg_skill.png",
      "original": "results/figures/fig_sqg_skill.png",
      "bytes": 40378,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_synth_ablation_rmse_uv.pdf",
      "original": "results/figures/fig_synth_ablation_rmse_uv.pdf",
      "bytes": 32075,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_synth_ablation_rmse_uv.png",
      "original": "results/figures/fig_synth_ablation_rmse_uv.png",
      "bytes": 62085,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_synth_ablation_tau_uv.pdf",
      "original": "results/figures/fig_synth_ablation_tau_uv.pdf",
      "bytes": 32999,
      "group": "results"
    },
    {
      "flat": "results__figures__fig_synth_ablation_tau_uv.png",
      "original": "results/figures/fig_synth_ablation_tau_uv.png",
      "bytes": 59113,
      "group": "results"
    },
    {
      "flat": "results__gpu_pytest.txt",
      "original": "results/gpu_pytest.txt",
      "bytes": 1549,
      "group": "results"
    },
    {
      "flat": "results__gpu_smoke.txt",
      "original": "results/gpu_smoke.txt",
      "bytes": 223,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__README.md",
      "original": "results/legacy_pre_review2/README.md",
      "bytes": 419,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__b2_gpu96_watch.txt",
      "original": "results/legacy_pre_review2/b2_gpu96_watch.txt",
      "bytes": 132,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__eval_B2_GPU96.txt",
      "original": "results/legacy_pre_review2/eval_B2_GPU96.txt",
      "bytes": 1916,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__eval_B2_GPU96_err.txt",
      "original": "results/legacy_pre_review2/eval_B2_GPU96_err.txt",
      "bytes": 0,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__eval_M3_GPU96.txt",
      "original": "results/legacy_pre_review2/eval_M3_GPU96.txt",
      "bytes": 960,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__eval_M4_GPU96.txt",
      "original": "results/legacy_pre_review2/eval_M4_GPU96.txt",
      "bytes": 960,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__m4_gpu96_post_nllfix_watch.txt",
      "original": "results/legacy_pre_review2/m4_gpu96_post_nllfix_watch.txt",
      "bytes": 2,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__m4_gpu96_watch.txt",
      "original": "results/legacy_pre_review2/m4_gpu96_watch.txt",
      "bytes": 3972,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__metrics_B2_GPU96.json",
      "original": "results/legacy_pre_review2/metrics_B2_GPU96.json",
      "bytes": 1761,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__metrics_GPU96_expanded_summary.json",
      "original": "results/legacy_pre_review2/metrics_GPU96_expanded_summary.json",
      "bytes": 1876,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__metrics_M3_GPU96.json",
      "original": "results/legacy_pre_review2/metrics_M3_GPU96.json",
      "bytes": 1763,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__metrics_M4_GPU96.json",
      "original": "results/legacy_pre_review2/metrics_M4_GPU96.json",
      "bytes": 1760,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__metrics_M4_GPU96_pre_nllfix.json",
      "original": "results/legacy_pre_review2/metrics_M4_GPU96_pre_nllfix.json",
      "bytes": 1766,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__train_B2_GPU96.txt",
      "original": "results/legacy_pre_review2/train_B2_GPU96.txt",
      "bytes": 836,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__train_M3_GPU96.txt",
      "original": "results/legacy_pre_review2/train_M3_GPU96.txt",
      "bytes": 1692,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__train_M4_GPU96.txt",
      "original": "results/legacy_pre_review2/train_M4_GPU96.txt",
      "bytes": 1694,
      "group": "results"
    },
    {
      "flat": "results__legacy_pre_review2__train_M4_GPU96_post_nllfix.txt",
      "original": "results/legacy_pre_review2/train_M4_GPU96_post_nllfix.txt",
      "bytes": 1694,
      "group": "results"
    },
    {
      "flat": "results__m4fix_eval.txt",
      "original": "results/m4fix_eval.txt",
      "bytes": 711,
      "group": "results"
    },
    {
      "flat": "results__m4fix_train.txt",
      "original": "results/m4fix_train.txt",
      "bytes": 415,
      "group": "results"
    },
    {
      "flat": "results__metrics_B2.json",
      "original": "results/metrics_B2.json",
      "bytes": 1430,
      "group": "results"
    },
    {
      "flat": "results__metrics_M3.json",
      "original": "results/metrics_M3.json",
      "bytes": 1765,
      "group": "results"
    },
    {
      "flat": "results__metrics_M4.json",
      "original": "results/metrics_M4.json",
      "bytes": 1779,
      "group": "results"
    },
    {
      "flat": "results__metrics_M4fix.json",
      "original": "results/metrics_M4fix.json",
      "bytes": 1380,
      "group": "results"
    },
    {
      "flat": "results__metrics_NATL60smoke.json",
      "original": "results/metrics_NATL60smoke.json",
      "bytes": 1760,
      "group": "results"
    },
    {
      "flat": "results__natl60_smoke_eval.txt",
      "original": "results/natl60_smoke_eval.txt",
      "bytes": 969,
      "group": "results"
    },
    {
      "flat": "results__natl60_smoke_train.txt",
      "original": "results/natl60_smoke_train.txt",
      "bytes": 176,
      "group": "results"
    },
    {
      "flat": "results__physics_ops__fig_adv_residual_sanity.pdf",
      "original": "results/physics_ops/fig_adv_residual_sanity.pdf",
      "bytes": 31283,
      "group": "results"
    },
    {
      "flat": "results__physics_ops__fig_adv_residual_sanity.png",
      "original": "results/physics_ops/fig_adv_residual_sanity.png",
      "bytes": 60552,
      "group": "results"
    },
    {
      "flat": "results__physics_ops__fig_sqg_skill.pdf",
      "original": "results/physics_ops/fig_sqg_skill.pdf",
      "bytes": 28130,
      "group": "results"
    },
    {
      "flat": "results__physics_ops__fig_sqg_skill.png",
      "original": "results/physics_ops/fig_sqg_skill.png",
      "bytes": 40378,
      "group": "results"
    },
    {
      "flat": "results__physics_ops__physics_ops_validation.json",
      "original": "results/physics_ops/physics_ops_validation.json",
      "bytes": 17574,
      "group": "results"
    },
    {
      "flat": "results__physics_ops_err.txt",
      "original": "results/physics_ops_err.txt",
      "bytes": 0,
      "group": "results"
    },
    {
      "flat": "results__physics_ops_run.txt",
      "original": "results/physics_ops_run.txt",
      "bytes": 498,
      "group": "results"
    },
    {
      "flat": "results__plot_after_eval.txt",
      "original": "results/plot_after_eval.txt",
      "bytes": 2890,
      "group": "results"
    },
    {
      "flat": "results__plot_after_gh.txt",
      "original": "results/plot_after_gh.txt",
      "bytes": 2816,
      "group": "results"
    },
    {
      "flat": "results__plot_after_m4.txt",
      "original": "results/plot_after_m4.txt",
      "bytes": 2816,
      "group": "results"
    },
    {
      "flat": "results__post_p0__STAGES_EFGH_FINAL.json",
      "original": "results/post_p0/STAGES_EFGH_FINAL.json",
      "bytes": 21066,
      "group": "results"
    },
    {
      "flat": "results__post_p0__ablation_summary.json",
      "original": "results/post_p0/ablation_summary.json",
      "bytes": 105431,
      "group": "results"
    },
    {
      "flat": "results__post_p0__eval_B2_s0_retry.txt",
      "original": "results/post_p0/eval_B2_s0_retry.txt",
      "bytes": 1358,
      "group": "results"
    },
    {
      "flat": "results__post_p0__figure_tau_uv_means.json",
      "original": "results/post_p0/figure_tau_uv_means.json",
      "bytes": 1851,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_B1-s0.json",
      "original": "results/post_p0/metrics_B1-s0.json",
      "bytes": 3531,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_B1-s1.json",
      "original": "results/post_p0/metrics_B1-s1.json",
      "bytes": 3542,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_B1-s2.json",
      "original": "results/post_p0/metrics_B1-s2.json",
      "bytes": 3554,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_B2-s0.json",
      "original": "results/post_p0/metrics_B2-s0.json",
      "bytes": 3492,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_B2-s1.json",
      "original": "results/post_p0/metrics_B2-s1.json",
      "bytes": 3547,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_B2-s2.json",
      "original": "results/post_p0/metrics_B2-s2.json",
      "bytes": 3548,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M1-s0.json",
      "original": "results/post_p0/metrics_M1-s0.json",
      "bytes": 3548,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M1-s1.json",
      "original": "results/post_p0/metrics_M1-s1.json",
      "bytes": 3542,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M1-s2.json",
      "original": "results/post_p0/metrics_M1-s2.json",
      "bytes": 3552,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M2-s0.json",
      "original": "results/post_p0/metrics_M2-s0.json",
      "bytes": 3547,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M2-s1.json",
      "original": "results/post_p0/metrics_M2-s1.json",
      "bytes": 3551,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M2-s2.json",
      "original": "results/post_p0/metrics_M2-s2.json",
      "bytes": 3537,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M3-s0.json",
      "original": "results/post_p0/metrics_M3-s0.json",
      "bytes": 3545,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M3-s1.json",
      "original": "results/post_p0/metrics_M3-s1.json",
      "bytes": 3530,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M3-s2.json",
      "original": "results/post_p0/metrics_M3-s2.json",
      "bytes": 3550,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M4-s0.json",
      "original": "results/post_p0/metrics_M4-s0.json",
      "bytes": 3546,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M4-s1.json",
      "original": "results/post_p0/metrics_M4-s1.json",
      "bytes": 3542,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_M4-s2.json",
      "original": "results/post_p0/metrics_M4-s2.json",
      "bytes": 3541,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_R0-s0.json",
      "original": "results/post_p0/metrics_R0-s0.json",
      "bytes": 3539,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_R0-s1.json",
      "original": "results/post_p0/metrics_R0-s1.json",
      "bytes": 3546,
      "group": "results"
    },
    {
      "flat": "results__post_p0__metrics_R0-s2.json",
      "original": "results/post_p0/metrics_R0-s2.json",
      "bytes": 3544,
      "group": "results"
    },
    {
      "flat": "results__post_p0__monitor.txt",
      "original": "results/post_p0/monitor.txt",
      "bytes": 748,
      "group": "results"
    },
    {
      "flat": "results__post_p0__monitor2.txt",
      "original": "results/post_p0/monitor2.txt",
      "bytes": 180,
      "group": "results"
    },
    {
      "flat": "results__post_p0__monitor3.txt",
      "original": "results/post_p0/monitor3.txt",
      "bytes": 1371,
      "group": "results"
    },
    {
      "flat": "results__post_p0__pipeline_err.txt",
      "original": "results/post_p0/pipeline_err.txt",
      "bytes": 0,
      "group": "results"
    },
    {
      "flat": "results__post_p0__pipeline_err2.txt",
      "original": "results/post_p0/pipeline_err2.txt",
      "bytes": 0,
      "group": "results"
    },
    {
      "flat": "results__post_p0__pipeline_run.txt",
      "original": "results/post_p0/pipeline_run.txt",
      "bytes": 9784,
      "group": "results"
    },
    {
      "flat": "results__post_p0__pipeline_run2.txt",
      "original": "results/post_p0/pipeline_run2.txt",
      "bytes": 48077,
      "group": "results"
    },
    {
      "flat": "results__post_p0__restart.txt",
      "original": "results/post_p0/restart.txt",
      "bytes": 53,
      "group": "results"
    },
    {
      "flat": "results__post_p0__sensitivity__sensitivity_summary.json",
      "original": "results/post_p0/sensitivity/sensitivity_summary.json",
      "bytes": 8765,
      "group": "results"
    },
    {
      "flat": "results__post_p0__sensitivity_run.txt",
      "original": "results/post_p0/sensitivity_run.txt",
      "bytes": 415,
      "group": "results"
    },
    {
      "flat": "results__pytest_5rounds.txt",
      "original": "results/pytest_5rounds.txt",
      "bytes": 81,
      "group": "results"
    },
    {
      "flat": "results__pytest_last.txt",
      "original": "results/pytest_last.txt",
      "bytes": 1054,
      "group": "results"
    },
    {
      "flat": "results__pytest_m4fix.txt",
      "original": "results/pytest_m4fix.txt",
      "bytes": 81,
      "group": "results"
    },
    {
      "flat": "results__pytest_partial_last.txt",
      "original": "results/pytest_partial_last.txt",
      "bytes": 1079,
      "group": "results"
    },
    {
      "flat": "results__pytest_post_p0_tooling.txt",
      "original": "results/pytest_post_p0_tooling.txt",
      "bytes": 81,
      "group": "results"
    },
    {
      "flat": "results__smoke_last.txt",
      "original": "results/smoke_last.txt",
      "bytes": 223,
      "group": "results"
    },
    {
      "flat": "results__verify_natl60_last.txt",
      "original": "results/verify_natl60_last.txt",
      "bytes": 1329,
      "group": "results"
    },
    {
      "flat": "scripts__assemble_paper.py",
      "original": "scripts/assemble_paper.py",
      "bytes": 10112,
      "group": "code"
    },
    {
      "flat": "scripts__build_report.py",
      "original": "scripts/build_report.py",
      "bytes": 27977,
      "group": "code"
    },
    {
      "flat": "scripts__download_natl60.py",
      "original": "scripts/download_natl60.py",
      "bytes": 6903,
      "group": "code"
    },
    {
      "flat": "scripts__evaluate.py",
      "original": "scripts/evaluate.py",
      "bytes": 11822,
      "group": "code"
    },
    {
      "flat": "scripts__evaluate_drifters.py",
      "original": "scripts/evaluate_drifters.py",
      "bytes": 5432,
      "group": "code"
    },
    {
      "flat": "scripts__finalize_post_p0.py",
      "original": "scripts/finalize_post_p0.py",
      "bytes": 3108,
      "group": "code"
    },
    {
      "flat": "scripts__generate_synthetic.py",
      "original": "scripts/generate_synthetic.py",
      "bytes": 830,
      "group": "code"
    },
    {
      "flat": "scripts__plot_science.py",
      "original": "scripts/plot_science.py",
      "bytes": 10932,
      "group": "code"
    },
    {
      "flat": "scripts__run_ablation.py",
      "original": "scripts/run_ablation.py",
      "bytes": 6346,
      "group": "code"
    },
    {
      "flat": "scripts__run_post_p0_pipeline.py",
      "original": "scripts/run_post_p0_pipeline.py",
      "bytes": 9309,
      "group": "code"
    },
    {
      "flat": "scripts__run_sensitivity.py",
      "original": "scripts/run_sensitivity.py",
      "bytes": 9860,
      "group": "code"
    },
    {
      "flat": "scripts__run_verify.py",
      "original": "scripts/run_verify.py",
      "bytes": 1034,
      "group": "code"
    },
    {
      "flat": "scripts__smoke_test.py",
      "original": "scripts/smoke_test.py",
      "bytes": 3113,
      "group": "code"
    },
    {
      "flat": "scripts__train.py",
      "original": "scripts/train.py",
      "bytes": 10145,
      "group": "code"
    },
    {
      "flat": "scripts__validate_physics_operators.py",
      "original": "scripts/validate_physics_operators.py",
      "bytes": 11799,
      "group": "code"
    },
    {
      "flat": "scripts__verify_natl60.py",
      "original": "scripts/verify_natl60.py",
      "bytes": 3262,
      "group": "code"
    },
    {
      "flat": "tests__conftest.py",
      "original": "tests/conftest.py",
      "bytes": 161,
      "group": "code"
    },
    {
      "flat": "tests__test_config.py",
      "original": "tests/test_config.py",
      "bytes": 818,
      "group": "code"
    },
    {
      "flat": "tests__test_losses.py",
      "original": "tests/test_losses.py",
      "bytes": 2930,
      "group": "code"
    },
    {
      "flat": "tests__test_metrics.py",
      "original": "tests/test_metrics.py",
      "bytes": 2556,
      "group": "code"
    },
    {
      "flat": "tests__test_model.py",
      "original": "tests/test_model.py",
      "bytes": 2692,
      "group": "code"
    },
    {
      "flat": "tests__test_natl60.py",
      "original": "tests/test_natl60.py",
      "bytes": 3311,
      "group": "code"
    },
    {
      "flat": "tests__test_p0_correctness.py",
      "original": "tests/test_p0_correctness.py",
      "bytes": 15402,
      "group": "code"
    },
    {
      "flat": "tests__test_physics.py",
      "original": "tests/test_physics.py",
      "bytes": 4379,
      "group": "code"
    },
    {
      "flat": "tests__test_real.py",
      "original": "tests/test_real.py",
      "bytes": 1341,
      "group": "code"
    },
    {
      "flat": "tests__test_synthetic.py",
      "original": "tests/test_synthetic.py",
      "bytes": 520,
      "group": "code"
    },
    {
      "flat": "tests__test_training.py",
      "original": "tests/test_training.py",
      "bytes": 3327,
      "group": "code"
    }
  ]
}
```
<!-- FLAT_MAP_JSON_END -->
