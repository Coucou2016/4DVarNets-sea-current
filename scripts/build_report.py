#!/usr/bin/env python3
"""Build self-contained research report (HTML + Markdown + optional PDF).

Reads measured metrics JSON and SciencePlots PNGs under results/, embeds
images as Base64, and writes:
  - docs/report/report.html  (also mirrored to report.html at repo root)
  - docs/report/report.md
  - docs/report/report.pdf    (best-effort; see PDF notes in HTML if fail)

No CDN charts. No external CSS. No relative image paths in HTML.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
OUT = ROOT / "docs" / "report"
PAPER_FIG = ROOT / "docs" / "paper" / "figures"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _m(blob: dict, key: str) -> float | None:
    v = (blob.get("model") or {}).get(key)
    return None if v is None else float(v)


def _g(blob: dict, key: str) -> float | None:
    v = (blob.get("geostrophic") or {}).get(key)
    return None if v is None else float(v)


def _fmt(v: float | None, nd: int = 3) -> str:
    if v is None:
        return "待补充"
    return f"{v:.{nd}f}"


def _b64_png(path: Path) -> str:
    raw = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _fig_block(fig_id: str, title: str, png_name: str, explanation: str, b64_map: dict) -> tuple[str, str]:
    """Return (html, md) for one numbered figure with long explanation."""
    key = png_name
    if key not in b64_map:
        html = f'<figure id="{fig_id}" class="fig"><figcaption><strong>{title}</strong>（图文件缺失：{png_name}）</figcaption><p class="explain">{explanation}</p></figure>'
        md = f"### {title}\n\n*图文件缺失：`{png_name}`*\n\n{explanation}\n"
        return html, md
    src = b64_map[key]
    html = f"""<figure id="{fig_id}" class="fig">
  <img src="{src}" alt="{title}" />
  <figcaption><strong>{title}</strong></figcaption>
  <div class="explain">{explanation}</div>
</figure>"""
    md = f"### {title}\n\n![{title}]({PAPER_FIG.as_posix()}/{png_name})\n\n{explanation}\n"
    # MD keeps relative path for editing; HTML is fully embedded.
    md = f"### {title}\n\n![{title}](../paper/figures/{png_name})\n\n{explanation}\n"
    return html, md


def collect_metrics() -> dict:
    gpu = {}
    for name in ("B2_GPU96", "M3_GPU96", "M4_GPU96"):
        p = RESULTS / f"metrics_{name}.json"
        if p.is_file():
            gpu[name] = _load(p)
    pre = RESULTS / "metrics_M4_GPU96_pre_nllfix.json"
    if pre.is_file():
        gpu["M4_GPU96_pre"] = _load(pre)
    synth = {}
    for name in ("B2", "M3", "M4"):
        p = RESULTS / f"metrics_{name}.json"
        if p.is_file():
            synth[name] = _load(p)
    return {"gpu": gpu, "synth": synth}


def build_b64_map() -> dict[str, str]:
    out = {}
    for p in sorted(FIG.glob("*.png")):
        out[p.name] = _b64_png(p)
    return out


CSS = """
:root {
  --fg: #1a1a1a;
  --muted: #555;
  --line: #d0d5dd;
  --bg: #fafbfc;
  --card: #ffffff;
  --accent: #1f4e79;
  --soft: #eef3f8;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: "Times New Roman", "SimSun", "Noto Serif CJK SC", serif;
  color: var(--fg);
  background: linear-gradient(180deg, #f3f6fa 0%, var(--bg) 220px);
  line-height: 1.75;
  font-size: 16px;
}
.wrap {
  max-width: 920px;
  margin: 0 auto;
  padding: 28px 22px 64px;
}
.cover {
  background: var(--card);
  border: 1px solid var(--line);
  padding: 36px 28px 28px;
  margin-bottom: 28px;
}
.cover h1 {
  margin: 0 0 12px;
  font-size: 1.65rem;
  color: var(--accent);
  line-height: 1.35;
}
.cover .meta { color: var(--muted); font-size: 0.95rem; }
.cover .badge {
  display: inline-block;
  margin-top: 14px;
  padding: 4px 10px;
  background: var(--soft);
  border: 1px solid var(--line);
  font-size: 0.85rem;
  color: var(--accent);
}
nav.toc {
  background: var(--card);
  border: 1px solid var(--line);
  padding: 18px 22px;
  margin-bottom: 28px;
}
nav.toc h2 { margin: 0 0 10px; font-size: 1.15rem; color: var(--accent); }
nav.toc ol { margin: 0; padding-left: 1.3rem; }
nav.toc a { color: var(--accent); text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
section {
  background: var(--card);
  border: 1px solid var(--line);
  padding: 22px 24px 10px;
  margin-bottom: 22px;
}
section h2 {
  margin: 0 0 14px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--accent);
  color: var(--accent);
  font-size: 1.3rem;
}
section h3 { color: #243447; font-size: 1.08rem; margin-top: 1.3em; }
p { margin: 0.7em 0; text-align: justify; }
ul, ol { margin: 0.5em 0 0.9em; }
li { margin: 0.25em 0; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0 8px;
  font-size: 0.92rem;
}
th, td {
  border: 1px solid var(--line);
  padding: 8px 10px;
  text-align: center;
}
th { background: var(--soft); color: var(--accent); }
td.left, th.left { text-align: left; }
.table-note, .caveat {
  color: var(--muted);
  font-size: 0.9rem;
  margin: 4px 0 14px;
}
.caveat {
  background: #fff8e8;
  border-left: 4px solid #c48a1a;
  padding: 10px 12px;
}
.term { font-weight: 600; }
.eq {
  display: block;
  margin: 12px auto;
  padding: 10px 14px;
  background: #f7f9fc;
  border: 1px dashed var(--line);
  text-align: center;
  overflow-x: auto;
  font-family: "Times New Roman", "Cambria Math", serif;
}
figure.fig {
  margin: 18px 0 22px;
  padding: 12px;
  background: #fcfdff;
  border: 1px solid var(--line);
}
figure.fig img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto 8px;
}
figcaption {
  font-size: 0.95rem;
  color: #222;
  margin-bottom: 8px;
}
.explain {
  font-size: 0.92rem;
  color: #333;
  text-align: justify;
  line-height: 1.7;
}
.footer {
  color: var(--muted);
  font-size: 0.85rem;
  text-align: center;
  margin-top: 8px;
}
@media (max-width: 640px) {
  .wrap { padding: 16px 12px 40px; }
  .cover, section, nav.toc { padding-left: 14px; padding-right: 14px; }
  body { font-size: 15px; }
}
"""


def html_table_gpu(metrics: dict) -> str:
    gpu = metrics["gpu"]
    rows = []
    order = [
        ("B2_GPU96", "B2（SSH+SST）"),
        ("M3_GPU96", "M3（+SQG+平流）"),
        ("M4_GPU96", "M4（+应变不确定性，NLL修正后）"),
        ("M4_GPU96_pre", "M4（NLL修正前，存档）"),
    ]
    # geostrophy from B2 blob
    geo_src = gpu.get("B2_GPU96") or next(iter(gpu.values()), None)
    for key, label in order:
        if key not in gpu:
            continue
        b = gpu[key]
        rows.append(
            "<tr>"
            f"<td class='left'>{label}</td>"
            f"<td>{_fmt(_m(b,'tau_uv'))}</td>"
            f"<td>{_fmt(_m(b,'rmse_uv'))}</td>"
            f"<td>{_fmt(_m(b,'rmse_ssh'))}</td>"
            f"<td>{_fmt(_m(b,'tau_div'))}</td>"
            f"<td>{_fmt(_m(b,'lambda_x_ssh_km'),1)}</td>"
            f"<td>{_fmt(_m(b,'lambda_x_uv_km'),1)}</td>"
            "</tr>"
        )
    if geo_src:
        rows.append(
            "<tr>"
            "<td class='left'>地转基线 geo</td>"
            f"<td>{_fmt(_g(geo_src,'tau_uv'))}</td>"
            f"<td>{_fmt(_g(geo_src,'rmse_uv'))}</td>"
            f"<td>{_fmt(_g(geo_src,'rmse_ssh'))}</td>"
            f"<td>{_fmt(_g(geo_src,'tau_div'))}</td>"
            f"<td>{_fmt(_g(geo_src,'lambda_x_ssh_km'),1)}</td>"
            f"<td>{_fmt(_g(geo_src,'lambda_x_uv_km'),1)}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    return f"""<table>
<thead><tr>
<th class="left">配置 ID</th><th>τ_uv ↑</th><th>rmse_uv ↓</th><th>rmse_ssh ↓</th><th>τ_div</th><th>λ_x,ssh (km)</th><th>λ_x,uv (km)</th>
</tr></thead>
<tbody>
{body}
</tbody>
</table>
<p class="table-note">表注：数值来自本地 <code>results/metrics_*_GPU96.json</code>（汇总见 <code>results/metrics_GPU96_expanded_summary.json</code>）。协议为 NATL60 <strong>crop_size=96、20 epochs</strong>，<strong>不等于</strong> Fablet 2024 JAMES 论文 Table 全场长训结果。请勿将本表数字当作论文正式 Table 行引用。B1 SSH-only 本会话未测；全场 200ep 受 4GB GPU 限制暂不可行。</p>"""


def html_table_synth(metrics: dict) -> str:
    synth = metrics["synth"]
    rows = []
    for name, label in (("B2", "B2"), ("M3", "M3"), ("M4", "M4")):
        if name not in synth:
            continue
        b = synth[name]
        rows.append(
            "<tr>"
            f"<td class='left'>{label}</td>"
            f"<td>{_fmt(_m(b,'tau_uv'))}</td>"
            f"<td>{_fmt(_m(b,'rmse_uv'))}</td>"
            f"<td>{_fmt(_m(b,'rmse_ssh'))}</td>"
            f"<td>{_fmt(_g(b,'tau_uv'))}</td>"
            "</tr>"
        )
    body = "\n".join(rows) if rows else "<tr><td colspan='5'>待补充</td></tr>"
    return f"""<table>
<thead><tr>
<th class="left">配置</th><th>模型 τ_uv</th><th>rmse_uv</th><th>rmse_ssh</th><th>地转 τ_uv</th>
</tr></thead>
<tbody>{body}</tbody>
</table>
<p class="table-note">表注：合成 OSSE、约 8 epoch，仅作方向性/接线验证，不可外推为 NATL60 正式结论。</p>"""


def build_documents(metrics: dict, b64: dict) -> tuple[str, str]:
    today = date.today().isoformat()
    # Figure explanations (Chinese, detailed)
    figs = []
    figs.append(
        _fig_block(
            "fig1",
            "图1. 合成 OSSE 消融：表面流场解释方差 τ_uv",
            "fig_synth_ablation_tau_uv.png",
            """<p><strong>读图方式：</strong>横轴为消融配置 B2 / M3 / M4；纵轴为 τ_uv（explained variance of surface currents，表面流速解释方差，越接近 1 表示相对气候学方差的可解释比例越高）。每组给出模型与地转基线对照。</p>
<p><strong>背景与目的：</strong>合成数据用于在无 NATL60 长训前验证物理残差与不确定性项的代码通路与损失量级是否合理。</p>
<p><strong>结论：</strong>在短训合成协议下，M4（含应变相关不确定性）在三配置中 τ_uv 最高，表明该设置下物理项/不确定性项可带来方向性收益；地转基线明显更弱。本图<strong>不能</strong>直接外推到裁剪 NATL60 GPU96 排名。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig2",
            "图2. 合成 OSSE 消融：表面流场均方根误差 RMSE_uv",
            "fig_synth_ablation_rmse_uv.png",
            """<p><strong>读图方式：</strong>纵轴为 RMSE_uv（u、v 分量均方根误差的综合指标，越低越好）。</p>
<p><strong>含义：</strong>与图1互补——τ_uv 强调方差解释，RMSE 强调绝对误差幅值。合成短训中 M4 误差最低，与 τ_uv 排序一致。</p>
<p><strong>注意：</strong>合成场统计结构简化，仅作工程与方向性证据。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig3",
            "图3. NATL60 GPU96（crop96/20ep）τ_uv：B2 / M3 / M4 对比地转",
            "fig_GPU96_tau_uv.png",
            """<p><strong>读图方式：</strong>对比 B2-GPU96、M3-GPU96、M4-GPU96 的模型 τ_uv 与地转 τ_uv。地转 τ_uv 为大幅负值（约 −3.74），表示相对气候学方差，地转重建在该裁剪协议下解释能力很差。</p>
<p><strong>核心结论（实测）：</strong>排序为 <strong>B2 &gt; M3 ≳ M4</strong>，三者均远优于地转。显式加入 eSQG 风格 SQG 残差与 SST 平流残差（M3）以及应变不确定性（M4，NLL 修正后重训）并未在本协议上超过纯 SST–SSH 协同（B2）。</p>
<p><strong>严格边界：</strong>crop96 + 20 epoch ≠ JAMES 论文全场 Table；不得据此宣称“物理项普遍提升技能”。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig4",
            "图4. NATL60 GPU96（crop96/20ep）RMSE_uv",
            "fig_GPU96_rmse_uv.png",
            """<p><strong>读图方式：</strong>RMSE_uv 越低越好。B2 最低（约 0.184），M3（约 0.206）与 M4（约 0.211）接近且均远低于地转（约 1.029）。</p>
<p><strong>与 τ_uv 一致：</strong>支持“学习型 4DVarNet 相对地转有大幅增益，但物理附加项在本短裁剪协议上非单调增益”。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig5",
            "图5. B2-GPU96 训练/验证损失曲线",
            "fig_B2_GPU96_loss.png",
            """<p><strong>读图方式：</strong>横轴 epoch，纵轴为训练损失与验证损失。B2 为 SSH+SST 多模态基线（无 SQG/平流/不确定性）。</p>
<p><strong>含义：</strong>验证损失在约 4.5 量级收敛（最佳约 4.53），表明多模态观测项与先验项尺度匹配正常，可作为 M3/M4 损失尺度的参照。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig6",
            "图6. M3-GPU96 训练/验证损失曲线",
            "fig_M3_GPU96_loss.png",
            """<p><strong>读图方式：</strong>M3 在 B2 基础上打开 SQG 与 SST 平流残差（λ_sqg、λ_adv）。</p>
<p><strong>观察：</strong>最佳验证损失约 5.19，略高于 B2，与最终 τ_uv 略低于 B2 的结果相一致，提示固定权重物理残差可能过约束短训轨迹，或算子在裁剪窗口内存在偏差。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig7",
            "图7. M4-GPU96（NLL 尺度修正后重训）训练/验证损失曲线",
            "fig_M4_GPU96_loss.png",
            """<p><strong>读图方式：</strong>M4 在 M3 上增加应变相关 UV 不确定性监督项。修正前 raw NLL 曾使最佳验证损失膨胀至约 682、SSH RMSE 恶化；本图为 σ 归一化 NLL + MSE 混合后的 20 epoch 重训结果。</p>
<p><strong>结论：</strong>最佳验证损失约 4.83（约第 15 epoch），SSH 误差恢复至约 0.063；流场技能仍略低于 B2。说明损失尺度工程修复有效，但物理/不确定性项仍未在本协议上超越 SST 协同。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig8",
            "图8. NATL60 GPU96（crop96/20ep）τ_div 诊断",
            "fig_GPU96_tau_div.png",
            """<p><strong>读图方式：</strong>纵轴为散度场解释方差 τ_div（explained variance of divergence）。正值表示相对气候学方差有正解释能力；负值表示该诊断上弱于气候学基线。</p>
<p><strong>背景与目的：</strong>主表 τ_uv 之外，补充动力学结构诊断，检验物理残差是否改善散度一致性。</p>
<p><strong>结论：</strong>在本裁剪短训协议上仅 B2 为正（约 0.35），M3/M4 为负。这加强“物理项并非普遍加分”的部分结果叙事，仍<strong>≠</strong>论文正式 Table。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig9",
            "图9. NATL60 GPU96（crop96/20ep）λ_x,uv 诊断",
            "fig_GPU96_lambda_x_uv.png",
            """<p><strong>读图方式：</strong>纵轴为表面流速分辨尺度 λ_x,uv（km，error/signal PSD 比阈值阈值相关定义）。数值来自同一评测 JSON。</p>
<p><strong>结论：</strong>报告实测尺度诊断以充实证据面；因 crop96/20ep，不得把 λ_x 宣称为 JAMES Table 分辨尺度结论。</p>""",
            b64,
        )
    )

    fig_html = "\n".join(h for h, _ in figs)
    fig_md = "\n".join(m for _, m in figs)

    table_gpu = html_table_gpu(metrics)
    table_synth = html_table_synth(metrics)

    # Markdown tables
    def md_gpu() -> str:
        lines = [
            "| 配置 | τ_uv ↑ | rmse_uv ↓ | rmse_ssh ↓ | τ_div | λ_x,uv (km) |",
            "|------|--------|-----------|------------|-------|-------------|",
        ]
        gpu = metrics["gpu"]
        mapping = [
            ("B2_GPU96", "B2"),
            ("M3_GPU96", "M3"),
            ("M4_GPU96", "M4 post-NLL-fix"),
            ("M4_GPU96_pre", "M4 pre-fix"),
        ]
        for k, lab in mapping:
            if k not in gpu:
                continue
            b = gpu[k]
            lines.append(
                f"| {lab} | {_fmt(_m(b,'tau_uv'))} | {_fmt(_m(b,'rmse_uv'))} | {_fmt(_m(b,'rmse_ssh'))} | {_fmt(_m(b,'tau_div'))} | {_fmt(_m(b,'lambda_x_uv_km'),1)} |"
            )
        if "B2_GPU96" in gpu:
            b = gpu["B2_GPU96"]
            lines.append(
                f"| geo | {_fmt(_g(b,'tau_uv'))} | {_fmt(_g(b,'rmse_uv'))} | {_fmt(_g(b,'rmse_ssh'))} | {_fmt(_g(b,'tau_div'))} | {_fmt(_g(b,'lambda_x_uv_km'),1)} |"
            )
        lines.append("")
        lines.append("> 注：NATL60 crop96/20ep，≠ JAMES paper table。全场 200ep / B1 本会话未跑（4GB GPU）。")
        return "\n".join(lines)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>物理约束 4DVarNet 海表流场反演研究报告</title>
<style>
{CSS}
</style>
</head>
<body>
<div class="wrap">

<header class="cover">
  <h1>基于 SST–SSH 协同与物理残差的 4DVarNet 海表流场反演：裁剪 NATL60 OSSE 消融研究报告</h1>
  <p class="meta">项目：4DVarNets-sea-current　｜　报告日期：{today}　｜　实现与验收：Cursor Agent　｜　公开代码：<a href="https://github.com/Coucou2016/4DVarNets-sea-current">github.com/Coucou2016/4DVarNets-sea-current</a></p>
  <p class="meta">基线文献：Fablet et al. (2024), JAMES, doi:10.1029/2023MS003609　｜　顾问通道：ChatGPT（本会话浏览器 MCP 与 Codex 配额均受阻；见 docs/chatgpt_collaboration/SESSION_5ROUNDS.md；文献 DOI 由 WebSearch 核验）</p>
  <span class="badge">证据级别：合成短训（方向性）+ NATL60 GPU96 crop96/20ep（初步）· 非正式论文 Table</span>
</header>

<nav class="toc" id="toc">
  <h2>目录</h2>
  <ol>
    <li><a href="#abstract">摘要</a></li>
    <li><a href="#bg">研究背景与目标</a></li>
    <li><a href="#data">数据与方法</a></li>
    <li><a href="#process">研究过程</a></li>
    <li><a href="#results">结果</a></li>
    <li><a href="#discussion">分析与讨论</a></li>
    <li><a href="#conclusion">结论</a></li>
    <li><a href="#outlook">局限与展望</a></li>
  </ol>
</nav>

<section id="abstract">
  <h2>1. 摘要</h2>
  <p>海表流场（sea surface currents, SSC；海表面水平流速向量场）的卫星反演长期受限于海表高度（sea surface height, SSH；海面动力高度异常）有效分辨率与地转近似（geostrophy；由 SSH 水平梯度经科氏力平衡得到的地转流）对非地转分量的低估。Fablet 等（2024）提出以可训练观测/先验算子嵌入展开变分同化循环的 <span class="term">4DVarNet</span>（四维变分神经网络求解器）框架，利用海表温度（sea surface temperature, SST）与 SSH 的协同提升流场重建。本工作采用<strong>紧凑的 4DVarNet-inspired</strong> ConvLSTM 展开求解器（非字节级忠实复现；求解器自动微分图已按 P0 修正，取消逐步 detach），并在变分代价与监督损失中引入：（i）有效表面准地转（effective surface quasi-geostrophy, eSQG-style）速度残差；（ii）SST 平流–扩散残差；（iii）可选的应变相关 UV <strong>空间重加权</strong>（M4，非不确定性估计头）。</p>
  <p>在受控消融（B2 / M3 / M4）下，合成 OSSE 短训显示 M4 方向性最优；而在 NATL60 <strong>crop96 / 20 epoch</strong> GPU 历史实验（<strong>pre_p0_fix / 非正式论文 Table</strong>）中，排序为 <strong>B2 &gt; M3 ≳ M4</strong>（τ_uv：0.848 / 0.811 / 0.801），三者均显著优于地转（τ_uv ≈ −3.74）。M4 曾因 raw NLL 尺度失控导致 SSH 误差恶化，经 σ 归一化修复并重训后 SSH RMSE 恢复至约 0.063。诚实表述为：物理残差可实现且相对地转仍有增益，但在本短裁剪协议上<strong>并非普遍优于纯 SST–SSH 协同</strong>；正式 Table 结论有待 P0 后全场长训。</p>
</section>

<section id="bg">
  <h2>2. 研究背景与目标</h2>
  <h3>2.1 科学背景</h3>
  <p>卫星高度计直接约束的主要是地转分量；中小尺度与高频运动中非地转过程重要，业务再分析常低估这些贡献。SST 作为高分辨率示踪，可通过表面准地转理论（SQG / eSQG；以表面浮力/温度异常谱关系推断流函数）与热收支/平流反演等途径约束流速（Lapeyre &amp; Klein, 2006；Rio et al., 2016）。深度学习多模态方案（如 Martin et al., 2023；Fablet et al., 2024）进一步表明 SST–SSH 协同可显著提升流场技能。</p>
  <h3>2.2 与基线的关系与缺口</h3>
  <p>Fablet et al. (2024, JAMES) 已证明可学习的多模态观测项对年龄转分量的价值，但变分代价中并未显式写入 eSQG / SST 平流残差，也未系统消融“物理项是否总是加分”。若缺少显式物理项，难以把增益归因于动力学约束；若盲目加入，则在算子失配、权重不当或混合层/非平衡运动主导时可能损害技能。</p>
  <h3>2.3 研究目标</h3>
  <ol>
    <li>在紧凑 4DVarNet-inspired 求解器上实现可开关的 SQG / 平流 / 应变重加权项（求解器图已按 P0 修正）；</li>
    <li>在合成与裁剪 NATL60 OSSE 上完成 B2/M3/M4 公平消融；</li>
    <li>以实测指标给出诚实的部分/负面结果表述，并为全场长训 Table 设定证据门槛；</li>
    <li>产出可复现图表与自包含研究报告（本 HTML）。</li>
  </ol>
  <h3>2.4 术语首次展开</h3>
  <ul>
    <li><strong>OSSE</strong>（observing-system simulation experiment，观测系统模拟实验）：以高分辨率模式真值为“自然运行”，模拟卫星观测采样后做反演评估。</li>
    <li><strong>OSE</strong>（observing-system experiment）：使用真实卫星观测；本报告尚未给出正式 OSE/漂流浮标评分（待补充）。</li>
    <li><strong>τ_uv</strong>：表面流速相对气候学方差的解释方差，越高越好。</li>
    <li><strong>λ_x</strong>：误差/信号功率谱密度比降至 0.5 的分辨尺度（km），越小通常表示更细的有效分辨（需结合谱定义解读）。</li>
    <li><strong>B2 / M3 / M4</strong>：消融编号——B2=SSH+SST；M3=+SQG+平流；M4=+应变 UV 空间重加权（非不确定性估计）。</li>
  </ul>
</section>

<section id="data">
  <h2>3. 数据与方法</h2>
  <h3>3.1 数据</h3>
  <ul>
    <li><strong>合成 OSSE</strong>：<code>data/synthetic_osse.npz</code>，短训（约 8 epoch）方向性验证。</li>
    <li><strong>NATL60</strong>：NEMO 1/60° 北大西洋高分辨率模拟（与 4DVarNet-SSH / Fablet 工作同族 OSSE 设定）。本轮 GPU 实验使用 <strong>crop_size=96</strong>、<strong>20 epochs</strong>（硬件：GTX 950M 4GB）。全场、论文划分、长训（目标约 200 epoch）结果：<strong>待补充</strong>。</li>
  </ul>
  <h3>3.2 状态与求解器</h3>
  <p>状态 \(x=(\\eta,u,v)\)。观测给出稀疏/有缺测 SSH \(y\) 与长度 \(dT\) 的 SST 窗 \(z\)。求解器执行 \(K\) 步展开梯度更新（ConvLSTM），对应 4DVarNet 范式。代码：<code>fourdvarnet/model.py</code>、<code>solver.py</code>、<code>prior.py</code>、<code>observation.py</code>。</p>
  <h3>3.3 变分代价中的物理残差</h3>
  <div class="eq">
  U(x) = λ_obs‖η−y‖²_Ω + λ_sst‖G(η)−H(z)‖² + λ_Φ‖x−Φ(x)‖²<br/>
  + 1_SQG λ_sqg‖(u,v)−A_SQG(z,η)‖² + 1_adv λ_adv‖∂tT + u·∇T − κ∇²T‖²
  </div>
  <p><strong>A_SQG</strong> 为本仓库实现的 <em>effective eSQG-style</em> 算子（非完整三维 SQG 反演）。平流残差对应热收支型约束。开关由 <code>use_sst / use_sqg / use_adv</code> 控制。</p>
  <h3>3.4 M4 应变不确定性与 NLL 尺度</h3>
  <p>令 σ = σ₀(1+α·strain)，默认由真值 UV 计算应变以避免“抬高预测应变压低 NLL”的崩塌。Raw NLL ‖e‖²/σ² 在 σ₀≈0.05 时相对 MSE 可大两个数量级，导致 SSH 项相对变弱。采用 σ 归一化形式并与 MSE 混合（<code>uncert_mse_mix</code>）后重训，SSH 恢复。</p>
  <h3>3.5 评价指标</h3>
  <p>主指标：τ_uv、rmse_uv、rmse_ssh、λ_x（可定义时）。同一划分上始终共报地转基线（<code>fourdvarnet/metrics.py</code>）。</p>
  <h3>3.6 作图规范</h3>
  <p>SciencePlots（science+ieee）+ Times New Roman；中文回退字体 SimSun / Microsoft YaHei（见 <code>fourdvarnet/plotting.py</code>）；导出 DPI=300。</p>
</section>

<section id="process">
  <h2>4. 研究过程</h2>
  <ol>
    <li>复现/实现 Fablet 风格 4DVarNet UV 求解器与多模态观测项；</li>
    <li>加入 eSQG 风格 SQG、SST 平流残差与应变不确定性；建立 B1/B2/M1–M4 消融矩阵；</li>
    <li>合成 OSSE 短训验证通路（M4 方向性最优）；</li>
    <li>NATL60 crop96/20ep：训练并评估 B2、M3、M4；</li>
    <li>诊断 M4 raw NLL 尺度问题 → 代码/配置修复 → M4 全量 20ep 重训；</li>
    <li>SciencePlots 出图；撰写英文论文草稿章节与本中文自包含报告；</li>
    <li>公开仓库：<a href="https://github.com/Coucou2016/4DVarNets-sea-current">https://github.com/Coucou2016/4DVarNets-sea-current</a>（已推送 code+docs+metrics+figures；排除 NATL60 <code>*.nc</code>、checkpoints <code>*.pt</code>、wheels/secrets）。</li>
    <li>ChatGPT 顾问通道：本会话 ≥5 轮尝试见 <code>docs/chatgpt_collaboration/SESSION_5ROUNDS.md</code>（浏览器 tab 创建即丢失；Codex ChatGPT 登录但用量限额至 2026-08-20）。粘贴简报与 GitHub URL 已备；文献 DOI 由 Cursor WebSearch 独立核验后写入论文。</li>
  </ol>
</section>

<section id="results">
  <h2>5. 结果</h2>
  <div class="caveat"><strong>硬性声明：</strong>下列 GPU96 数字仅对应 NATL60 <em>crop96 / 20ep</em> 本地实测，不得写成 JAMES 论文 Table 或“已证明 SQG+平流优于 SST 协同”。</div>

  <h3>5.1 表1. NATL60 GPU96 主结果</h3>
  {table_gpu}

  <h3>5.2 表2. 合成 OSSE（方向性）</h3>
  {table_synth}

  <h3>5.3 图件与详细解读</h3>
  {fig_html}
</section>

<section id="discussion">
  <h2>6. 分析与讨论</h2>
  <p>裁剪短训协议上 B2 领先，说明可学习的 SST–SSH 协同项可能已吸收大量可迁移的示踪结构；固定权重的 SQG/平流残差在窗口截断、算子近似（eSQG-style）或训练预算不足时可能过约束。合成与 NATL60 排名不一致，本身即支持<strong>体制/协议依赖</strong>：物理项不是无条件加分器。</p>
  <p>M4 的 SSH 退化被证明主要是损失尺度工程问题而非“不确定性物理必然损害 SSH”；修复后 SSH 恢复、流场仍略逊于 B2，提示下一步应做 λ 与不确定性超参扫描，而非简单关闭物理项。</p>
  <p>相关文献框架（独立核验 DOI）：Fablet 2024 JAMES (10.1029/2023MS003609)；Beauchamp 2023 GMD 4DVarNet-SSH (10.5194/gmd-16-2119-2023)；Lapeyre &amp; Klein 2006；Rio et al. 2016；González-Haro &amp; Isern-Fontanet 相关；Martin et al. 2023；近期 VarDyn 联合 SSH–SST 动力重建 (10.1029/2024MS004689) 可作为“显式动力约束 vs 学习协同”对照叙述。</p>
</section>

<section id="conclusion">
  <h2>7. 结论</h2>
  <ol>
    <li>物理残差可嵌入 4DVarNet 代价并完成可复现消融；</li>
    <li>crop96/20ep 上 <strong>B2 &gt; M3 ≳ M4</strong>，均大幅优于地转；</li>
    <li>M4 NLL 尺度修复有效恢复 SSH；</li>
    <li>正式论文 Table 与 OSE 评估仍为开放证据门槛。</li>
  </ol>
</section>

<section id="outlook">
  <h2>8. 局限与展望</h2>
  <ul>
    <li>硬件限制导致裁剪训练，可能与物理残差假设的空间上下文冲突；</li>
    <li>缺少多种子置信区间与完整 λ 扫描（待补充）；</li>
    <li>全场 NATL60 长训、OSE/漂流浮标评估未完成（待补充）；</li>
    <li>ChatGPT 浏览器/Codex 顾问自动化本轮受阻（见 SESSION_5ROUNDS.md）；公开 GitHub 已就绪，可由人工在 chatgpt.com 粘贴顾问简报继续迭代。</li>
  </ul>
  <p class="footer">本文件由 <code>scripts/build_report.py</code> 生成：CSS 内联、图片 Base64 嵌入、表格为 HTML。请以 results/metrics_*.json 为数值真源。</p>
</section>

</div>
</body>
</html>
"""

    md = f"""# 基于 SST–SSH 协同与物理残差的 4DVarNet 海表流场反演研究报告

**日期：** {today}  
**证据级别：** 合成短训（方向性）+ NATL60 GPU96 crop96/20ep（初步）· **非正式论文 Table**  
**基线：** Fablet et al. (2024), JAMES, https://doi.org/10.1029/2023MS003609

---

## 目录

1. 摘要
2. 研究背景与目标
3. 数据与方法
4. 研究过程
5. 结果
6. 分析与讨论
7. 结论
8. 局限与展望

---

## 1. 摘要

海表流场（SSC）反演受 SSH 分辨率与地转近似限制。本工作在 Fablet 风格 4DVarNet 求解器上嵌入 eSQG 风格 SQG 残差、SST 平流残差与可选应变 UV 不确定性。NATL60 **crop96/20ep** 实测：**B2 > M3 ≳ M4**（τ_uv 0.848 / 0.811 / 0.801），均优于地转（τ_uv ≈ −3.74）。M4 经 NLL 尺度修正后 SSH RMSE 恢复至约 0.063。物理项可实现且相对地转有增益，但在本协议上并非普遍优于纯 SST–SSH 协同。

---

## 2. 研究背景与目标

详见仓库 `docs/paper/01_introduction.md`。目标：可开关物理残差、公平消融、诚实部分结果、自包含报告。

**术语：** OSSE / OSE / τ_uv / λ_x / B2·M3·M4 — 见 HTML 版首次展开。

---

## 3. 数据与方法

- 合成 OSSE + NATL60 crop96/20ep（全场长训：待补充）
- 代价函数与 σ 归一化 NLL：见 `docs/paper/02_methods.md`
- 作图：SciencePlots + Times New Roman + CJK 回退（SimSun / Microsoft YaHei）

---

## 4. 研究过程

实现 → 合成验证 → GPU96 消融 → M4 NLL 修复重训 → 扩展诊断表/图 → 出图与报告。公开仓库 https://github.com/Coucou2016/4DVarNets-sea-current ；ChatGPT 浏览器 MCP / Codex 配额受阻（见 SESSION_5ROUNDS.md），DOI 由 WebSearch 核验；粘贴简报见 docs/chatgpt_collaboration/。

---

## 5. 结果

### 表1. NATL60 GPU96

{md_gpu()}

### 表2. 合成 OSSE

见 `results/metrics_{{B2,M3,M4}}.json`（方向性）。

### 图件

{fig_md}

---

## 6. 分析与讨论

B2 领先支持“学习协同可能已覆盖部分 SQG/平流可迁移信息”；合成与 NATL60 排名差支持体制依赖。M4 SSH 问题主要为损失尺度。文献 DOI：Fablet 2024；Beauchamp 2023 GMD；Lapeyre & Klein 2006；Rio 2016；Martin 2023；VarDyn 2024MS004689。

---

## 7. 结论

1. 物理残差可嵌入并消融；2. crop96/20ep 上 B2>M3≳M4 且均胜地转；3. M4 NLL 修复有效；4. 正式 Table / OSE 待补充。

---

## 8. 局限与展望

裁剪偏差、多种子与 λ 扫描缺失、全场长训与 OSE/B1 未完成；公开 GitHub 已推送；ChatGPT 自动化顾问通道本会话受阻（人工粘贴简报可继续）。

---

*由 `scripts/build_report.py` 生成。数值真源：`results/metrics_*.json`。*
"""
    return html, md


def try_pdf(html_path: Path, pdf_path: Path) -> str:
    """Best-effort PDF. Returns status note."""
    # 1) weasyprint
    try:
        from weasyprint import HTML  # type: ignore

        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return f"PDF via weasyprint: {pdf_path}"
    except Exception as e1:
        note1 = f"weasyprint failed: {e1}"

    # 2) playwright
    try:
        from playwright.sync_api import sync_playwright  # type: ignore

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.pdf(path=str(pdf_path), format="A4", print_background=True)
            browser.close()
        return f"PDF via playwright: {pdf_path}"
    except Exception as e2:
        note2 = f"playwright failed: {e2}"

    # 3) Microsoft Edge / Chrome headless
    for exe in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ):
        if not Path(exe).is_file():
            continue
        try:
            subprocess.run(
                [
                    exe,
                    "--headless",
                    "--disable-gpu",
                    f"--print-to-pdf={pdf_path}",
                    html_path.resolve().as_uri(),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            if pdf_path.is_file() and pdf_path.stat().st_size > 1000:
                return f"PDF via {Path(exe).name}: {pdf_path}"
        except Exception as e3:
            note1 = note1 + f"; {Path(exe).name}: {e3}"

    return (
        "PDF generation failed on this host. Kept HTML+MD. "
        f"Details: {note1}; {note2}"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics()
    b64 = build_b64_map()
    html, md = build_documents(metrics, b64)

    html_path = OUT / "report.html"
    md_path = OUT / "report.md"
    pdf_path = OUT / "report.pdf"
    html_path.write_text(html, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")

    # Mirror to repo root for convenience
    root_html = ROOT / "report.html"
    shutil.copy2(html_path, root_html)

    pdf_note = try_pdf(html_path, pdf_path)
    note_path = OUT / "PDF_STATUS.txt"
    note_path.write_text(pdf_note + "\n", encoding="utf-8")

    print(f"Wrote {html_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {root_html}")
    print(pdf_note)
    print(f"Embedded PNG count: {len(b64)}")


if __name__ == "__main__":
    main()
