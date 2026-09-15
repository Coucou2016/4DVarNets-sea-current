# 4. Discussion

## 4.1 Soft physics inside a learned solver

A central empirical point is the gap between Stage E and Stages F–G. The standalone eSQG-style map has near-zero τ_uv against truth UV on the same crop, yet enabling a soft SQG residual inside the unrolled cost (M1/M3/M4) improves mean post_p0 τ_uv relative to B2. In other words, the residual acts as a regularizer coupled to multimodal observations and the learned prior, not as a hard current estimate. This distinction matters for interpretation: “physics helps” here means soft constraints in a neural variational loop, not that SQG inversion alone solves SSC.

Advection alone (M2) yields scores close to B2, consistent with Stage E evidence that the heat-budget residual can be dominated by \(\partial_t - \kappa\nabla^2\) on daily frames. Combining SQG and advection (M3) behaves similarly to SQG alone (M1) under this protocol; strain reweighting (M4) yields a further small mean τ_uv gain with reduced seed scatter.

## 4.2 When physics helps or fails

**Helps (this protocol).** Soft SQG-containing losses improve UV skill and SSH RMSE relative to B2 on post_p0 crop96/15ep multi-seed runs, while all learned models beat OI-only geostrophy.

**Fails or is fragile.** (i) Hard SQG as a current map fails on τ_uv (Stage E). (ii) Pre-P0 historical crop96/20ep runs ranked B2 above M3/M4, showing that short protocols and correctness bugs can reverse rankings. (iii) Raw heteroscedastic UV weighting previously starved SSH; scale normalization was required before M4 was interpretable. (iv) Larger compact R0 capacity did not automatically beat B2.

Taken together, physics residuals are implementable and ablatable, but **regime- and hyperparameter-dependent**. Formal generalization to full-domain long training remains **待补充**.

## 4.3 Relation to VarDyn and Fablet et al.

Fablet et al. (2024) demonstrate learned SST–SSH synergy for SSC with 4DVarNets; we do not re-quote their Table scores. Our contribution is the soft residual ablation and operator diagnostics under a transparent protocol. VarDyn (Le Guillou et al., 2025) jointly reconstructs SSH and SST with dynamical constraints; it is a natural dynamical counterpart for tracer mapping, whereas our solver targets SSC with optional soft SQG/advection residuals. Cross-code benchmarking against VarDyn is out of scope here.

## 4.4 Limitations

Hardware (4GB GPU) precludes uncropped ~200×200 × ~200-epoch multi-seed tables. R0 is not byte-faithful to CIA-Oceanix. OSE/drifter evaluation hooks exist but formal scores are **待补充**. Stage H sensitivity (SST coarsening / sparsity) is reported when `results/post_p0/sensitivity/` is populated; otherwise marked incomplete in the audit package.
