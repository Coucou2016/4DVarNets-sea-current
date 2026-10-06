#!/usr/bin/env python3
"""Build self-contained Chinese research report (HTML + Markdown + optional PDF).

Reads measured metrics JSON and SciencePlots PNGs under results/, embeds
images as Base64, and writes:
  - docs/report/report.html  (also mirrored to report.html at repo root)
  - docs/report/report.md
  - docs/report/report.pdf    (best-effort)

No CDN. No external CSS. No relative image paths in HTML.
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


def _fmt_pm(mean: float | None, std: float | None, nd: int = 3) -> str:
    if mean is None:
        return "待补充"
    if std is None:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f}±{std:.{nd}f}"


def _b64_png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _fig_block(fig_id: str, title: str, png_name: str, explanation: str, b64_map: dict) -> tuple[str, str]:
    if png_name not in b64_map:
        html = (
            f'<figure id="{fig_id}" class="fig"><figcaption><strong>{title}</strong>'
            f'（图文件缺失：{png_name}）</figcaption><div class="explain">{explanation}</div></figure>'
        )
        md = f"### {title}\n\n*图文件缺失：`{png_name}`*\n\n{explanation}\n"
        return html, md
    src = b64_map[png_name]
    html = f"""<figure id="{fig_id}" class="fig">
  <img src="{src}" alt="{title}" />
  <figcaption><strong>{title}</strong></figcaption>
  <div class="explain">{explanation}</div>
</figure>"""
    md = f"### {title}\n\n![{title}](../paper/figures/{png_name})\n\n{explanation}\n"
    return html, md


def collect_metrics() -> dict:
    out: dict = {"gpu": {}, "synth": {}, "post": None, "phys": None, "sens": None, "geo_tau": None}

    legacy = RESULTS / "legacy_pre_review2"
    for name in ("B2_GPU96", "M3_GPU96", "M4_GPU96"):
        p = legacy / f"metrics_{name}.json"
        if not p.is_file():
            p = RESULTS / f"metrics_{name}.json"
        if p.is_file():
            out["gpu"][name] = _load(p)
    pre = legacy / "metrics_M4_GPU96_pre_nllfix.json"
    if pre.is_file():
        out["gpu"]["M4_GPU96_pre"] = _load(pre)

    for name in ("B2", "M3", "M4"):
        p = RESULTS / f"metrics_{name}.json"
        if p.is_file():
            out["synth"][name] = _load(p)

    post = RESULTS / "post_p0" / "ablation_summary.json"
    if post.is_file():
        out["post"] = _load(post)
    phys = RESULTS / "physics_ops" / "physics_ops_validation.json"
    if phys.is_file():
        out["phys"] = _load(phys)
    sens = RESULTS / "post_p0" / "sensitivity" / "sensitivity_summary.json"
    if sens.is_file():
        out["sens"] = _load(sens)

    for seed_name in ("B2-s0", "B1-s0"):
        mp = RESULTS / "post_p0" / f"metrics_{seed_name}.json"
        if mp.is_file():
            out["geo_tau"] = _g(_load(mp), "tau_uv")
            break
    return out


def build_b64_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for folder in (FIG, RESULTS / "physics_ops", OUT / "figures"):
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.png")):
            out.setdefault(p.name, _b64_png(p))
    return out


CSS = """
:root { --fg:#1a1a1a; --muted:#555; --line:#d0d5dd; --bg:#fafbfc; --card:#fff; --accent:#1f4e79; --soft:#eef3f8; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin:0; font-family:"Times New Roman","SimSun","Noto Serif CJK SC",serif; color:var(--fg);
  background:linear-gradient(180deg,#f3f6fa 0%,var(--bg) 220px); line-height:1.75; font-size:16px; }
.wrap { max-width:920px; margin:0 auto; padding:28px 22px 64px; }
.cover { background:var(--card); border:1px solid var(--line); padding:36px 28px 28px; margin-bottom:28px; }
.cover h1 { margin:0 0 12px; font-size:1.65rem; color:var(--accent); line-height:1.35; }
.cover .meta { color:var(--muted); font-size:0.95rem; }
.cover .badge { display:inline-block; margin-top:14px; padding:4px 10px; background:var(--soft);
  border:1px solid var(--line); font-size:0.85rem; color:var(--accent); }
nav.toc { background:var(--card); border:1px solid var(--line); padding:18px 22px; margin-bottom:28px; }
nav.toc h2 { margin:0 0 10px; font-size:1.15rem; color:var(--accent); }
nav.toc ol { margin:0; padding-left:1.3rem; }
nav.toc a { color:var(--accent); text-decoration:none; }
nav.toc a:hover { text-decoration:underline; }
section { background:var(--card); border:1px solid var(--line); padding:22px 24px 10px; margin-bottom:22px; }
section h2 { margin:0 0 14px; padding-bottom:8px; border-bottom:2px solid var(--accent); color:var(--accent); font-size:1.3rem; }
section h3 { color:#243447; font-size:1.08rem; margin-top:1.3em; }
p { margin:0.7em 0; text-align:justify; }
ul,ol { margin:0.5em 0 0.9em; } li { margin:0.25em 0; }
table { width:100%; border-collapse:collapse; margin:12px 0 8px; font-size:0.92rem; }
th,td { border:1px solid var(--line); padding:8px 10px; text-align:center; }
th { background:var(--soft); color:var(--accent); }
td.left,th.left { text-align:left; }
.table-note,.caveat { color:var(--muted); font-size:0.9rem; margin:4px 0 14px; }
.caveat { background:#fff8e8; border-left:4px solid #c48a1a; padding:10px 12px; color:#333; }
.term { font-weight:600; }
.eq { display:block; margin:12px auto; padding:10px 14px; background:#f7f9fc; border:1px dashed var(--line);
  text-align:center; overflow-x:auto; font-family:"Times New Roman","Cambria Math",serif; }
figure.fig { margin:18px 0 22px; padding:12px; background:#fcfdff; border:1px solid var(--line); }
figure.fig img { display:block; max-width:100%; height:auto; margin:0 auto 8px; }
figcaption { font-size:0.95rem; color:#222; margin-bottom:8px; }
.explain { font-size:0.92rem; color:#333; text-align:justify; line-height:1.7; }
.footer { color:var(--muted); font-size:0.85rem; text-align:center; margin-top:8px; }
@media (max-width:640px) { .wrap{padding:16px 12px 40px;} body{font-size:15px;} }
"""


def html_table_post(metrics: dict) -> str:
    post = metrics.get("post")
    if not post:
        return "<p class='caveat'>post_p0 消融汇总缺失（待补充）。</p>"
    summary = post.get("summary") or {}
    geo = metrics.get("geo_tau")
    rows = []
    for name in ("B1", "B2", "M1", "M2", "M3", "M4", "R0"):
        block = summary.get(name) or {}
        mets = block.get("metrics") or {}
        def get(k):
            m = mets.get(k) or {}
            return m.get("mean"), m.get("std")
        tu, ts = get("tau_uv")
        ru, rs = get("rmse_uv")
        su, ss = get("rmse_ssh")
        if tu is None:
            continue
        rows.append(
            "<tr>"
            f"<td class='left'>{name}</td>"
            f"<td>{_fmt_pm(tu, ts)}</td>"
            f"<td>{_fmt_pm(ru, rs)}</td>"
            f"<td>{_fmt_pm(su, ss)}</td>"
            "</tr>"
        )
    if geo is not None:
        rows.append(
            f"<tr><td class='left'>geo（OI-only）</td><td>{_fmt(geo)}</td><td>—</td><td>—</td></tr>"
        )
    body = "\n".join(rows)
    proto = post.get("protocol") or {}
    return f"""<table>
<thead><tr><th class="left">配置</th><th>τ_uv ↑</th><th>rmse_uv ↓</th><th>rmse_ssh ↓</th></tr></thead>
<tbody>{body}</tbody></table>
<p class="table-note">主表：<code>results/post_p0/ablation_summary.json</code>。协议 crop={proto.get('crop_size')}，epochs={proto.get('epochs')}，
seeds={proto.get('seeds')}。<strong>不等于</strong> Fablet 2024 JAMES 全场长训 Table。R0 为更大容量紧凑模型，非字节级官方复现。</p>"""


def html_table_phys(metrics: dict) -> str:
    phys = metrics.get("phys")
    if not phys:
        return "<p class='caveat'>Stage E 物理算子验证缺失（待补充）。</p>"
    sq = (phys.get("sqg") or {}).get("summary") or {}
    adv = (phys.get("advection") or {}).get("summary") or {}
    return f"""<table>
<thead><tr><th class="left">量</th><th>实测</th></tr></thead>
<tbody>
<tr><td class="left">SQG τ_uv 均值</td><td>{_fmt(sq.get('tau_uv_mean'))}</td></tr>
<tr><td class="left">SQG corr_u / corr_v</td><td>{_fmt(sq.get('corr_u_mean'))} / {_fmt(sq.get('corr_v_mean'))}</td></tr>
<tr><td class="left">SQG rmse_uv</td><td>{_fmt(sq.get('rmse_uv_mean'))}</td></tr>
<tr><td class="left">平流 RMS truth / scramble / zero</td><td>{adv.get('rms_truth_mean') and f"{float(adv['rms_truth_mean']):.2e} / {float(adv['rms_scrambled_mean']):.2e} / {float(adv['rms_zero_mean']):.2e}" or '待补充'}</td></tr>
</tbody></table>
<p class="table-note">来源 <code>results/physics_ops/physics_ops_validation.json</code>（crop96，23 个测试日）。SQG 单独作流速图时 τ_uv≈0，故 λ_sqg 宜作软先验。</p>"""


def html_table_legacy(metrics: dict) -> str:
    gpu = metrics.get("gpu") or {}
    if not gpu:
        return "<p>无隔离区 legacy GPU96 文件。</p>"
    rows = []
    for key, label in (
        ("B2_GPU96", "B2"),
        ("M3_GPU96", "M3"),
        ("M4_GPU96", "M4 post-NLL"),
        ("M4_GPU96_pre", "M4 pre-NLL"),
    ):
        if key not in gpu:
            continue
        b = gpu[key]
        rows.append(
            f"<tr><td class='left'>{label}</td><td>{_fmt(_m(b,'tau_uv'))}</td>"
            f"<td>{_fmt(_m(b,'rmse_uv'))}</td><td>{_fmt(_m(b,'rmse_ssh'))}</td></tr>"
        )
    return f"""<table>
<thead><tr><th class="left">配置（隔离）</th><th>τ_uv</th><th>rmse_uv</th><th>rmse_ssh</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="table-note">标签 <code>legacy_pre_review2</code> / <code>pre_p0_fix</code>：仅工程史，不作正式科学排序。</p>"""


def build_documents(metrics: dict, b64: dict) -> tuple[str, str]:
    today = date.today().isoformat()
    figs: list[tuple[str, str]] = []

    figs.append(
        _fig_block(
            "fig_e1",
            "图1. Stage E：eSQG 风格算子相对真值 UV 的技能诊断",
            "fig_sqg_skill.png",
            """<p><strong>读图：</strong>展示 SQG 算子相对 NATL60 真值流速的相关、RMSE 与解释方差等诊断（见同目录 JSON）。</p>
<p><strong>物理含义：</strong>表面准地转（SQG / eSQG）试图用海表温度/浮力异常推断流函数；本仓库实现为<strong>有效</strong> eSQG 风格混合，而非完整三维 SQG 反演。</p>
<p><strong>结论：</strong>相关可较高，但 τ_uv 均值接近 0（甚至略负）。因此把 SQG 当作“单独给出正确流速”会失败；更合理的是在学习型变分代价里作<strong>软约束</strong>。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig_e2",
            "图2. Stage E：SST 平流残差真值 / 扰乱 / 零流对照",
            "fig_adv_residual_sanity.png",
            """<p><strong>读图：</strong>比较真值流速、扰乱流速与零流速下的平流–扩散残差 RMS。</p>
<p><strong>方程角色：</strong>残差 r=∂tT + u·∇T − κ∇²T 衡量温度变化是否与流速平流一致。</p>
<p><strong>结论：</strong>真值残差系统性低于扰乱场（算子对流速敏感）；零流残差有时可不高于真值，说明日尺度上 ∂t−κ∇² 可占主导——平流项单独增益可能有限。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig_p1",
            "图3. post_p0 消融：表面流解释方差 τ_uv（三随机种子均值±标准差）",
            "fig_post_p0_tau_uv.png",
            """<p><strong>读图：</strong>横轴 B1/B2/M1–M4/R0；纵轴 τ_uv（explained variance of surface currents，相对气候学方差的解释比例，越高越好）。误差棒为多种子标准差；地转为 OI-only 基线。</p>
<p><strong>协议：</strong>NATL60 paper-mode，crop96，15 epoch，batch1，seeds {0,1,2}。</p>
<p><strong>结论：</strong>含软 SQG 的 M1/M3/M4 均值高于纯 SST–SSH 的 B2；B2 高于 B1 与地转。这与 Stage E“硬 SQG 无效”并不矛盾——软残差在展开求解器内起正则化作用。</p>
<p><strong>边界：</strong>≠ JAMES 全场 Table；正式 Table 待补充。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig_p2",
            "图4. post_p0 消融：RMSE_uv（均值±标准差）",
            "fig_post_p0_rmse_uv.png",
            """<p><strong>读图：</strong>RMSE_uv 越低越好，与 τ_uv 互补（绝对误差 vs 方差解释）。</p>
<p><strong>结论：</strong>排序与图3大体一致：M1/M3/M4 误差更低。请同时查看多种子离散度，避免单一种子过度解读。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig_p3",
            "图5. post_p0 消融：RMSE_ssh（均值±标准差）",
            "fig_post_p0_rmse_ssh.png",
            """<p><strong>读图：</strong>海表高度重建误差。物理残差主要面向流速，但仍可能通过耦合改变 SSH 技能。</p>
<p><strong>结论：</strong>本协议上 M1/M3/M4 的 SSH RMSE 亦不劣于 B2，未见历史 raw-NLL 尺度失控导致的 SSH 崩塌。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig_l1",
            "图6. 隔离区 legacy GPU96（crop96/20ep，pre_p0_fix）τ_uv",
            "fig_GPU96_tau_uv.png",
            """<p><strong>读图：</strong>历史 B2/M3/M4 与地转对照。该阶段地转诊断曾异常（大幅负 τ_uv），且排序为 B2&gt;M3≳M4。</p>
<p><strong>用途：</strong>仅说明 P0 修正前的工程史与协议依赖性；<strong>不作</strong>当前科学主结论。</p>""",
            b64,
        )
    )
    figs.append(
        _fig_block(
            "fig_s1",
            "图7. 合成 OSSE 短训（8 epoch）τ_uv——方向性通路检查",
            "fig_synth_ablation_tau_uv.png",
            """<p><strong>读图：</strong>合成数据上的快速消融，用于验证损失与物理项代码通路。</p>
<p><strong>边界：</strong>统计结构简化，不得外推为 NATL60 正式排序。</p>""",
            b64,
        )
    )

    fig_html = "\n".join(h for h, _ in figs)
    fig_md = "\n".join(m for _, m in figs)
    table_post = html_table_post(metrics)
    table_phys = html_table_phys(metrics)
    table_legacy = html_table_legacy(metrics)

    sens_note = "待补充"
    if metrics.get("sens"):
        sens_note = "已写入 <code>results/post_p0/sensitivity/sensitivity_summary.json</code>。"

    md_post_lines = [
        "| 配置 | τ_uv ↑ | rmse_uv ↓ | rmse_ssh ↓ |",
        "|------|--------|-----------|------------|",
    ]
    post = metrics.get("post") or {}
    summary = post.get("summary") or {}
    for name in ("B1", "B2", "M1", "M2", "M3", "M4", "R0"):
        mets = (summary.get(name) or {}).get("metrics") or {}
        def pm(k):
            m = mets.get(k) or {}
            return _fmt_pm(m.get("mean"), m.get("std"))
        if not (mets.get("tau_uv") or {}).get("mean"):
            continue
        md_post_lines.append(f"| {name} | {pm('tau_uv')} | {pm('rmse_uv')} | {pm('rmse_ssh')} |")
    if metrics.get("geo_tau") is not None:
        md_post_lines.append(f"| geo | {_fmt(metrics['geo_tau'])} | — | — |")
    md_post = "\n".join(md_post_lines)

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
  <h1>基于 SST–SSH 协同与软物理残差的 4DVarNet 海表流场反演研究报告</h1>
  <p class="meta">项目：4DVarNets-sea-current　｜　日期：{today}　｜　公开代码：
  <a href="https://github.com/Coucou2016/4DVarNets-sea-current">github.com/Coucou2016/4DVarNets-sea-current</a></p>
  <p class="meta">基线文献：Fablet et al. (2024) JAMES doi:10.1029/2023MS003609；Beauchamp et al. (2023) GMD；
  VarDyn / Le Guillou et al. (2025) JAMES doi:10.1029/2024MS004689；Lapeyre &amp; Klein (2006) JPO。</p>
  <span class="badge">主证据：post_p0 crop96/15ep×3seeds + Stage E 物理算子　｜　非正式 JAMES 全场 Table</span>
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
  <p>海表流场（sea surface currents, SSC）卫星反演受海表高度（SSH）有效分辨率与地转近似限制。
  本工作在<strong>紧凑 4DVarNet-inspired</strong>展开求解器中引入可开关的软物理残差：有效 eSQG 风格 SQG、SST 平流，以及可选应变相关 UV 空间重加权（M4，非不确定性头）。</p>
  <p>在本机可完成的 <strong>post_p0</strong> 协议（NATL60 crop96、15 epoch、三随机种子）上，含软 SQG 的配置（M1/M3/M4）平均 τ_uv 高于纯 SST–SSH（B2），且均优于地转；同时 Stage E 显示<strong>单独</strong> SQG 流速图的 τ_uv≈0。
  因而“物理有帮助”应理解为学习型变分环中的软约束，而非硬 SQG 反演。全场约 200 epoch 正式 Table：<strong>待补充</strong>。所有数字均来自本仓库 JSON，未抄录外部论文表。</p>
</section>

<section id="bg">
  <h2>2. 研究背景与目标</h2>
  <h3>2.1 科学背景</h3>
  <p>高度计主要约束地转分量；中小尺度年龄转过程重要。SST 可通过 SQG/eSQG 与热收支/平流途径提供额外约束
  （Lapeyre &amp; Klein, 2006；Rio et al., 2016）。Fablet et al. (2024) 证明可学习多模态 4DVarNet 可提升 SSC；
  VarDyn（Le Guillou et al., 2025）则以动力约束联合重建 SSH 与 SST，作为动力侧对照而非本协议下的 SSC 竞品表。</p>
  <h3>2.2 目标</h3>
  <ol>
    <li>实现可消融的软 SQG / 平流 / 应变重加权；</li>
    <li>在裁剪 NATL60 上给出<strong>本仓库实测</strong>多种子结果；</li>
    <li>用 Stage E 说明硬算子与软残差的差别；</li>
    <li>产出可复现图表与自包含报告（本 HTML）。</li>
  </ol>
  <h3>2.3 术语</h3>
  <ul>
    <li><strong>OSSE</strong>（观测系统模拟实验）：以高分辨率模式真值模拟卫星观测后反演评估。</li>
    <li><strong>τ_uv</strong>：表面流速解释方差，越高越好。</li>
    <li><strong>λ_sqg / λ_adv</strong>：变分代价中 SQG / 平流残差权重。</li>
    <li><strong>B1/B2/M1–M4/R0</strong>：消融编号（见方法）。</li>
    <li><strong>post_p0 / legacy_pre_review2</strong>：协议隔离标签。</li>
  </ul>
</section>

<section id="data">
  <h2>3. 数据与方法</h2>
  <h3>3.1 数据</h3>
  <p>NATL60 Gulf Stream OSSE（paper-mode：obs + oi）。评测用地转基线为 <strong>OI-only</strong> SSH 地转流。合成数据仅用于短训通路检查。</p>
  <h3>3.2 方法要点</h3>
  <p>状态 x=(η,u,v)。展开 K 步对变分代价 U(x) 求梯度，并由 ConvLSTM 参数化更新。软残差写入：</p>
  <div class="eq">U ⊃ λ_sqg ‖(u,v)−A_SQG(z,η)‖² + λ_adv ‖∂tT + u·∇T − κ∇²T‖²</div>
  <p>M4 使用应变相关空间重加权（默认由真值应变构造 σ），<strong>不是</strong>学习型不确定性头。细节见 <code>docs/paper/02_methods.md</code> 与 <code>fourdvarnet/</code>。</p>
</section>

<section id="process">
  <h2>4. 研究过程（可含路径）</h2>
  <ol>
    <li>P0 正确性：全展开自动微分、OI-only 地转基线、尺度与评测聚合修复（见 <code>docs/REVIEW_RESPONSE_P0.md</code>）。</li>
    <li>Stage E：<code>scripts/validate_physics_operators.py</code> → <code>results/physics_ops/</code>。</li>
    <li>Stages F–G：<code>scripts/run_post_p0_pipeline.py</code> 训练/评测 B1–M4/R0 → <code>results/post_p0/metrics_*-s*.json</code>、<code>ablation_summary.json</code>。</li>
    <li>Stage H：<code>scripts/run_sensitivity.py</code> → <code>results/post_p0/sensitivity/</code>（{sens_note}）。</li>
    <li>作图：<code>scripts/plot_science.py</code>（SciencePlots + Times New Roman + CJK 回退）→ <code>results/figures/</code> 并镜像至论文/报告 figures。</li>
    <li>组装：<code>scripts/assemble_paper.py</code>、<code>scripts/build_report.py</code>；证据审计 <code>docs/AUDIT_EVIDENCE.md</code>。</li>
  </ol>
  <div class="caveat">硬件：GTX 950M 4GB。全场 ~200×200、~200 epoch、多种子 JAMES Table 本机不可行（待补充）。权重 <code>*.pt</code> 与 <code>*.nc</code> 不入库。</div>
</section>

<section id="results">
  <h2>5. 结果</h2>
  <h3>5.1 Stage E 物理算子</h3>
  {table_phys}
  <h3>5.2 post_p0 消融主表（本报告主证据）</h3>
  {table_post}
  <h3>5.3 隔离区 legacy（非主结论）</h3>
  {table_legacy}
  <h3>5.4 图件与读图说明</h3>
  {fig_html}
</section>

<section id="discussion">
  <h2>6. 分析与讨论</h2>
  <p><strong>软物理 vs 硬 SQG。</strong> Stage E 表明单独 SQG 不能解释 UV 方差；post_p0 表明把同一类算子作为软残差写入展开求解器后，M1/M3/M4 可高于 B2。二者必须分开表述。</p>
  <p><strong>协议依赖。</strong> 隔离区 pre_p0 历史曾给出 B2&gt;M3≳M4，与当前 post_p0 排序不同，说明短训/正确性修复会改变结论；故全场长训 Table 仍待补充。</p>
  <p><strong>与 VarDyn。</strong> VarDyn 侧重动力约束下的 SSH–SST 联合制图；本工作侧重面向 SSC 的学习型展开求解器 + 可选软残差，不作跨代码表对比。</p>
</section>

<section id="conclusion">
  <h2>7. 结论</h2>
  <ol>
    <li>实现了可消融的软 SQG/平流/应变重加权 4DVarNet-inspired 流程，并给出本仓库实测指标。</li>
    <li>post_p0 上软 SQG 配置平均优于 B2；Stage E 否定“硬 SQG 单独恢复流速”。</li>
    <li>正式全场 JAMES Table、OSE/漂流浮标评分：待补充。</li>
  </ol>
</section>

<section id="outlook">
  <h2>8. 局限与展望</h2>
  <ul>
    <li>4GB 显存限制裁剪与 epoch；不可宣称全场 Table。</li>
    <li>R0 非官方字节级复现；VarDyn 对照未跑。</li>
    <li>下一步：更大显存上的 uncropped 长训、OSE/浮标、敏感性完备化。</li>
  </ul>
</section>

<p class="footer">自包含 HTML（Base64 图，无 CDN）。生成脚本：scripts/build_report.py　｜　{today}</p>
</div>
</body>
</html>
"""

    md = f"""# 基于 SST–SSH 协同与软物理残差的 4DVarNet 海表流场反演研究报告

日期：{today}  
代码：https://github.com/Coucou2016/4DVarNets-sea-current

> 主证据：post_p0 crop96/15ep×3seeds + Stage E。非正式 JAMES 全场 Table。

## 1. 摘要

（见 HTML 版完整叙述。）在 post_p0 协议上，软 SQG 配置（M1/M3/M4）平均 τ_uv 高于 B2；Stage E 显示单独 SQG τ_uv≈0。全场长训 Table：待补充。数字均来自本仓库 JSON。

## 2. 背景与目标

见 HTML。关键文献 DOI：Fablet 2024 `10.1029/2023MS003609`；Beauchamp 2023 `10.5194/gmd-16-2119-2023`；VarDyn 2025 `10.1029/2024MS004689`；Lapeyre & Klein 2006 `10.1175/JPO2840.1`。

## 3. 数据与方法

紧凑 4DVarNet-inspired + 软 SQG/平流/应变重加权。细节：`docs/paper/02_methods.md`。

## 4. 研究过程

见 HTML 路径列表。审计：`docs/AUDIT_EVIDENCE.md`。

## 5. 结果

### post_p0 主表

{md_post}

### 图件

{fig_md}

## 6–8. 讨论 / 结论 / 局限

见 HTML。正式 Table 与 OSE：待补充。
"""
    return html, md


def try_pdf(html_path: Path, pdf_path: Path) -> str:
    try:
        from weasyprint import HTML  # type: ignore

        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return f"PDF via weasyprint: {pdf_path}"
    except Exception as e1:
        note1 = f"weasyprint failed: {e1}"
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
    for exe in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ):
        if not Path(exe).is_file():
            continue
        try:
            subprocess.run(
                [exe, "--headless", "--disable-gpu", f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
                check=False,
                capture_output=True,
                timeout=180,
            )
            if pdf_path.is_file() and pdf_path.stat().st_size > 1000:
                return f"PDF via {Path(exe).name}: {pdf_path}"
        except Exception:
            continue
    return f"PDF unavailable ({note1}; {note2})"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    # Ensure paper figures mirrored for md links
    if FIG.is_dir():
        for p in FIG.glob("*.png"):
            try:
                shutil.copy2(p, OUT / "figures" / p.name)
                PAPER_FIG.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, PAPER_FIG / p.name)
            except OSError as e:
                print(f"warn: could not mirror {p.name}: {e}")

    metrics = collect_metrics()
    b64 = build_b64_map()
    html, md = build_documents(metrics, b64)

    html_path = OUT / "report.html"
    md_path = OUT / "report.md"
    pdf_path = OUT / "report.pdf"
    html_path.write_text(html, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    shutil.copy2(html_path, ROOT / "report.html")

    pdf_note = try_pdf(html_path, pdf_path)
    (OUT / "pdf_build_note.txt").write_text(pdf_note + "\n", encoding="utf-8")
    print(f"Wrote {html_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {ROOT / 'report.html'}")
    print(pdf_note)


if __name__ == "__main__":
    main()
