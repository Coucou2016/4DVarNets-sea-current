#!/usr/bin/env python3
"""Assemble docs/paper/PAPER.md and docs/paper/paper.html from chapter files."""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"


FRONT = f"""# Soft differentiable physics in a compact 4DVarNet for SST–SSH sea-surface current inversion

**Manuscript draft (methods / JAMES–GMD style).** Primary scores: post_p0 NATL60 crop96 / 15 epochs / 3 seeds — **not** full-grid JAMES Table rows.  
**Code:** https://github.com/Coucou2016/4DVarNets-sea-current  
**Assembled:** {date.today().isoformat()}

---

## Key Points

- Soft eSQG-style SQG and SST-advection residuals (plus optional strain-aware UV reweighting) can be embedded in a compact 4DVarNet-inspired unrolled cost without claiming a byte-faithful Fablet solver.
- On measured post_p0 crop96/15ep multi-seed NATL60 OSSEs, soft SQG-containing configurations (M1/M3/M4) improve mean τ_uv relative to SSH+SST (B2), while a standalone SQG current map has near-zero τ_uv skill—soft residuals ≠ hard SQG inversion.
- Full-domain ~200-epoch JAMES Table scores remain **待补充**; legacy pre-P0 crop96/20ep ranks are quarantined and are not used as formal claims.

## Plain Language Summary

Satellites observe sea level and sea-surface temperature more readily than ocean currents. Neural variational solvers that combine those observations can estimate currents better than the classical geostrophic approximation. We test whether adding soft physics checks—surface quasi-geostrophy and temperature advection—inside such a solver helps further. In our cropped NATL60 simulation experiments, those soft checks can improve current skill relative to using sea level and temperature alone, even though a stand-alone physics map of currents performs poorly. We treat this as protocol-dependent evidence and do not claim full-domain paper-table scores until longer, uncropped training is available.

## Abstract

Estimating sea-surface currents (SSC) from satellite sea-surface height (SSH) remains limited by altimeter resolution and by the geostrophic approximation. Multimodal 4DVarNet solvers that synergize SSH with sea-surface temperature (SST) improve SSC relative to geostrophy in NATL60 observing-system simulation experiments (OSSEs). Here we use a compact 4DVarNet-inspired ConvLSTM unrolled solver and extend the variational cost with soft, differentiable residuals: an effective eSQG-style SQG term, an SST advection term, and optional strain-aware spatial UV reweighting. In a controlled ablation under a post-P0 protocol (NATL60 crop_size=96, 15 epochs, three seeds), soft SQG-containing models achieve higher mean explained variance τ_uv than SSH+SST alone, while a standalone SQG operator exhibits near-zero τ_uv against truth currents on the same crop. We therefore frame physics residuals as useful soft constraints inside a learned solver, not as hard current maps, and contrast this setting with VarDyn-style dynamical joint SSH–SST mapping. Cropped short-epoch scores must not be read as JAMES Table rows; full-grid long-epoch claims remain gated (**待补充**). All reported numbers are computed in this repository; we do not reproduce external paper tables as our results.

---

"""


def _read(name: str) -> str:
    text = (PAPER / name).read_text(encoding="utf-8").strip() + "\n\n"
    # Strip status banners that belong in drafts, not submission prose.
    lines = []
    for line in text.splitlines(True):
        if line.startswith("**Status:**") or line.startswith("**Nature-skills"):
            continue
        if line.strip() == "---" and not lines:
            continue
        lines.append(line)
    return "".join(lines)


def assemble_md() -> str:
    parts = [
        FRONT,
        _read("01_introduction.md"),
        _read("02_methods.md"),
        _read("03_experiments.md"),
        _read("04_discussion.md"),
        """# 5. Conclusions

1. Soft eSQG-style SQG and SST-advection residuals, with optional strain-aware UV reweighting, can be embedded in a compact 4DVarNet-inspired unrolled cost.
2. Measured post_p0 crop96/15ep multi-seed NATL60 scores show soft SQG-containing configurations improving mean τ_uv relative to B2, while Stage E shows standalone SQG τ_uv ≈ 0—soft physics ≠ hard SQG inversion.
3. Rankings are protocol-dependent; legacy pre-P0 crop96/20ep results are quarantined. Full-grid ~200-epoch JAMES Table scores remain **待补充**.
4. Relative to VarDyn, our contribution is soft SSC-facing residuals inside a learned unrolled solver, not dynamical joint tracer mapping.

# Open Research

Code, metrics JSON, and figures: https://github.com/Coucou2016/4DVarNets-sea-current (excludes large `*.nc` / `*.pt` weights). NATL60 source data follow the OSSE distribution cited in Fablet et al. (2024). Evidence audit: `docs/AUDIT_EVIDENCE.md`.

# References (key DOIs verified)

- Fablet, R., Chapron, B., Le Sommer, J., & Sévellec, F. (2024). Inversion of sea surface currents from satellite-derived SST-SSH synergies with 4DVarNets. *Journal of Advances in Modeling Earth Systems*, 16, e2023MS003609. https://doi.org/10.1029/2023MS003609
- Beauchamp, M., Febvre, Q., Georgenthum, H., & Fablet, R. (2023). 4DVarNet-SSH: end-to-end learning of variational interpolation schemes for nadir and wide-swath satellite altimetry. *Geoscientific Model Development*, 16, 2119–2147. https://doi.org/10.5194/gmd-16-2119-2023
- Le Guillou, F., Chapron, B., & Rio, M.-H. (2025). VarDyn: Dynamical joint-reconstructions of sea surface height and temperature from multi-sensor satellite observations. *Journal of Advances in Modeling Earth Systems*, 17, e2024MS004689. https://doi.org/10.1029/2024MS004689
- Lapeyre, G., & Klein, P. (2006). Dynamics of the upper oceanic layers in terms of surface quasigeostrophy theory. *Journal of Physical Oceanography*, 36, 165–176. https://doi.org/10.1175/JPO2840.1
- Rio, M.-H., Santoleri, R., Bourdalle-Badie, R., et al. (2016). Improving the altimeter-derived surface currents using high-resolution sea surface temperature data: A feasibility study based on model outputs. *Journal of Atmospheric and Oceanic Technology*. https://doi.org/10.1175/JTECH-D-16-0017.1
- Martin, M. J., et al. (2023). Synergistic use of SSH and SST. *Journal of Advances in Modeling Earth Systems*. https://doi.org/10.1029/2022MS003589
- González-Haro, C., & Isern-Fontanet, J. (related). https://doi.org/10.1029/2019JC015958
- Miracca-Lage, M., et al. (2022). *Journal of Geophysical Research: Oceans*. https://doi.org/10.1029/2021JC018001
- Yassin, H., & Griffies, S. M. (2023). *Journal of Physical Oceanography*. https://doi.org/10.1175/JPO-D-22-0040.1
""",
    ]
    return "".join(parts)


def md_to_html(md: str) -> str:
    """Minimal Markdown→HTML for local viewing (tables + headings + paragraphs)."""
    try:
        import markdown  # type: ignore

        body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    except Exception:
        import html as html_mod

        body = "<pre>" + html_mod.escape(md) + "</pre>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Soft physics in compact 4DVarNet for SST–SSH SSC inversion</title>
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
<div class="caveat"><strong>Caveat:</strong> Primary scores are post_p0 NATL60 <em>crop96 / 15 epochs / 3 seeds</em>. They are <em>not</em> JAMES full-domain Table rows. Full-grid long-train claims remain gated.</div>
{body}
</body>
</html>
"""


def try_pdf(html_path: Path, pdf_path: Path) -> str:
    import subprocess

    # Edge / Chrome headless print-to-PDF
    for exe in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ):
        if not Path(exe).is_file():
            continue
        try:
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            uri = html_path.resolve().as_uri()
            subprocess.run(
                [
                    exe,
                    "--headless",
                    "--disable-gpu",
                    f"--print-to-pdf={pdf_path}",
                    uri,
                ],
                check=False,
                capture_output=True,
                timeout=120,
            )
            if pdf_path.is_file() and pdf_path.stat().st_size > 1000:
                return f"PDF via {Path(exe).name}: {pdf_path}"
        except Exception as e:
            continue
    try:
        from weasyprint import HTML  # type: ignore

        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return f"PDF via weasyprint: {pdf_path}"
    except Exception as e:
        return f"PDF unavailable: {e}"


def main() -> None:
    md = assemble_md()
    out_md = PAPER / "PAPER.md"
    out_html = PAPER / "paper.html"
    out_pdf = PAPER / "paper.pdf"
    out_md.write_text(md, encoding="utf-8")
    out_html.write_text(md_to_html(md), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_html}")
    note = try_pdf(out_html, out_pdf)
    print(note)
    (PAPER / "pdf_build_note.txt").write_text(note + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
