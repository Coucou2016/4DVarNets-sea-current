#!/usr/bin/env python3
"""Assemble docs/paper/PAPER.md and docs/paper/paper.html from chapter files."""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"


FRONT = f"""# Physics-constrained 4DVarNet for SST–SSH sea-surface current inversion

**Assembled manuscript (working).** Crop96/20ep ≠ JAMES Table.  
**Public code:** https://github.com/Coucou2016/4DVarNets-sea-current  
**Assembled:** {date.today().isoformat()}

**Axes (nature-writing):** `task=manuscript`, `paper_type=methods`, `journal=generic` (JAMES / GMD-style methods paper; Nature-family *clarity*, not flagship Nature format), `language=en`.

---

## Key Points

- Explicit eSQG-style SQG, SST-advection, and optional strain-aware UV uncertainty residuals can be embedded in a Fablet-like 4DVarNet cost without replacing the ConvLSTM solver.
- On a preliminary NATL60 **crop96 / 20-epoch** OSSE, measured ranking is **B2 > M3 ≳ M4** on τ_uv (0.848 / 0.811 / 0.801); all beat geostrophy (τ_uv ≈ −3.74). Physics extras are not universally additive.
- A raw heteroscedastic NLL scale bug starved M4 SSH (rmse_ssh 0.122); σ-normalized UV loss + retrain recovers rmse_ssh ≈ 0.063 while UV still trails B2. Cropped short runs are **not** paper Table rows.

## Plain Language Summary

Satellites measure sea level and sea-surface temperature more easily than they measure ocean currents directly. Learning models that combine those measurements can estimate currents better than the classical geostrophic approximation. We test whether adding explicit physics “checks” (surface quasi-geostrophy and temperature advection) inside a variational neural solver helps further. In our short, cropped simulation tests, the simpler sea-level + temperature model still wins; the physics checks remain useful to implement and diagnose, but they do not automatically improve scores. We also show how a poorly scaled uncertainty loss can quietly damage sea-level skill, and how to fix that scaling.

## Abstract

Estimating sea-surface currents (SSC) from satellite sea-surface height (SSH) remains limited by altimeter resolution and by the geostrophic approximation. Multimodal 4DVarNet solvers that synergize SSH with sea-surface temperature (SST) already improve SSC relative to geostrophy in NATL60 observing-system simulation experiments (OSSEs). Here we keep a Fablet-style unrolled ConvLSTM 4DVarNet and extend only the variational cost / supervised loss with (i) an effective eSQG-style SQG residual, (ii) an SST advection residual, and (iii) optional strain-heteroscedastic UV uncertainty. Under a controlled ablation (B2: SSH+SST; M3: +SQG+advection; M4: +uncertainty), synthetic short runs can favor physics extras, but on a preliminary NATL60 **crop_size=96, 20-epoch** protocol the measured ranking is **B2 > M3 ≳ M4** on explained variance τ_uv (0.848 / 0.811 / 0.801), with all configurations beating geostrophy (τ_uv ≈ −3.74; rmse_uv ≈ 1.029). Expanded diagnostics from the same evaluations show positive τ_div / τ_vort / τ_strain only for B2. An M4 SSH degradation (rmse_ssh 0.122) is traced to raw NLL scale mismatch and repaired by a σ-normalized UV term (post-fix rmse_ssh 0.063). We therefore frame physics residuals as implementable, ablatable, and **regime-/hyperparameter-dependent**, not universally additive. Cropped short-epoch scores are preliminary and must not be read as JAMES Table rows; uncropped long-epoch NATL60 Table claims remain gated on hardware beyond a 4GB GPU.

---

"""


def _read(name: str) -> str:
    return (PAPER / name).read_text(encoding="utf-8").strip() + "\n\n"


def assemble_md() -> str:
    parts = [
        FRONT,
        _read("01_introduction.md"),
        _read("02_methods.md"),
        _read("03_experiments.md"),
        _read("04_discussion.md"),
        """# 5. Conclusions

1. Explicit eSQG-style SQG, SST-advection, and optional strain-aware UV uncertainty can be embedded in a Fablet-like 4DVarNet cost without changing the ConvLSTM solver.
2. On NATL60 **crop96/20ep**, ranking is **B2 > M3 ≳ M4** on τ_uv; all beat geostrophy. Physics extras are **not universally additive** under this protocol; divergence/vorticity/strain explained variances favor B2 in the same measured JSON.
3. M4 SSH degradation was an NLL scale bug; post-fix retrain recovers rmse_ssh ≈ 0.063 while UV still trails B2.
4. Paper-table claims require uncropped, long-epoch NATL60 (+ optional OSE/drifters). On current 4GB hardware we strengthen measured crop96 diagnostics rather than invent Table scores.

# Open Research

Code, metrics JSON, and figures: https://github.com/Coucou2016/4DVarNets-sea-current (excludes large `*.nc` / `*.pt` weights). NATL60 source data follow the Fablet/Oceanix OSSE distribution cited in Fablet et al. (2024).

# References (key DOIs verified)

- Fablet et al. 2024 JAMES — https://doi.org/10.1029/2023MS003609
- Beauchamp et al. 2023 GMD — https://doi.org/10.5194/gmd-16-2119-2023
- Lapeyre & Klein 2006 JPO — https://doi.org/10.1175/JPO2840.1
- Rio et al. 2016 JTECH — https://doi.org/10.1175/JTECH-D-16-0017.1
- Martin et al. 2023 JAMES — https://doi.org/10.1029/2022MS003589
- González-Haro & Isern-Fontanet related — https://doi.org/10.1029/2019JC015958
- Le Guillou, Chapron & Rio 2025 JAMES (VarDyn) — https://doi.org/10.1029/2024MS004689
- Miracca-Lage et al. 2022 JGR Oceans — https://doi.org/10.1029/2021JC018001
- Yassin & Griffies 2023 JPO — https://doi.org/10.1175/JPO-D-22-0040.1
""",
    ]
    return "".join(parts)


def md_to_html(md: str) -> str:
    """Minimal Markdown→HTML for local viewing (tables + headings + paragraphs)."""
    try:
        import markdown  # type: ignore

        body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    except Exception:
        # Fallback: escape and wrap in <pre>
        import html as html_mod

        body = "<pre>" + html_mod.escape(md) + "</pre>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Physics-constrained 4DVarNet for SST–SSH SSC inversion</title>
<style>
body {{ font-family: "Times New Roman", Times, serif; max-width: 880px; margin: 24px auto; padding: 0 16px; line-height: 1.45; color: #111; }}
h1,h2,h3 {{ line-height: 1.25; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 0.95rem; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; }}
th {{ background: #f4f4f4; }}
code {{ font-family: Consolas, monospace; font-size: 0.9em; }}
.caveat {{ background: #fff8e6; border: 1px solid #e6d39a; padding: 10px 12px; margin: 12px 0; }}
</style>
</head>
<body>
<div class="caveat"><strong>Caveat:</strong> Crop96/20ep numbers are preliminary and are <em>not</em> JAMES paper Table rows.</div>
{body}
</body>
</html>
"""


def main() -> None:
    md = assemble_md()
    out_md = PAPER / "PAPER.md"
    out_html = PAPER / "paper.html"
    out_md.write_text(md, encoding="utf-8")
    out_html.write_text(md_to_html(md), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    main()
