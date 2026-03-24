import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Figure 0: Bilevel Concept Diagram
# ─────────────────────────────────────────────────────────────────────────────

def make_bilevel_concept_figure():
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Color palette
    outer_face  = '#edf7ed'
    outer_edge  = '#2d6a2d'
    inner_face  = '#dce8f7'
    inner_edge  = '#2c5f8a'
    arrow_col   = '#444444'
    inject_col  = '#8b2d8a'   # purple for the injection arrow label

    # ── Outer rounded rectangle (green) ──────────────────────────────────────
    outer_box = FancyBboxPatch(
        (0.25, 0.25), 9.5, 5.5,
        boxstyle="round,pad=0.15",
        facecolor=outer_face, edgecolor=outer_edge, linewidth=2.0, zorder=1)
    ax.add_patch(outer_box)

    # Outer loop title (top)
    ax.text(5.0, 5.55, 'Outer Loop — optimize search mechanism',
            ha='center', va='center', fontsize=9.5, fontweight='bold',
            color=outer_edge, zorder=5)

    # Outer loop action text (bottom)
    ax.text(5.0, 0.72,
            'analyze trace  \u2192  research mechanism  \u2192  generate code  \u2192  inject',
            ha='center', va='center', fontsize=8.0, color=outer_edge,
            style='italic', zorder=5)

    # ── Inner rounded rectangle (blue) ───────────────────────────────────────
    inner_box = FancyBboxPatch(
        (1.0, 1.45), 8.0, 3.6,
        boxstyle="round,pad=0.12",
        facecolor=inner_face, edgecolor=inner_edge, linewidth=1.6, zorder=2)
    ax.add_patch(inner_box)

    # Inner loop title
    ax.text(5.0, 4.72, 'Inner Loop — optimize task output',
            ha='center', va='center', fontsize=9.0, fontweight='bold',
            color=inner_edge, zorder=5)

    # Inner loop content: propose -> train -> evaluate -> keep/discard
    # Draw four small boxes in a row
    steps = [
        (2.15,  3.30, 'propose'),
        (4.00,  3.30, 'train'),
        (5.85,  3.30, 'evaluate'),
        (7.70,  3.30, 'keep /\ndiscard'),
    ]
    bw, bh = 1.40, 0.62
    for (x, y, lbl) in steps:
        b = FancyBboxPatch(
            (x - bw/2, y - bh/2), bw, bh,
            boxstyle="round,pad=0.05",
            facecolor='white', edgecolor=inner_edge, linewidth=1.2, zorder=4)
        ax.add_patch(b)
        ax.text(x, y, lbl, ha='center', va='center', fontsize=8.0,
                color='#1a1a2e', zorder=5, multialignment='center')

    # Arrows between inner-loop steps
    for i in range(len(steps) - 1):
        x1 = steps[i][0]  + bw/2
        x2 = steps[i+1][0] - bw/2
        y0 = steps[0][1]
        ax.annotate('', xy=(x2, y0), xytext=(x1, y0),
                    arrowprops=dict(arrowstyle='->', color=arrow_col, lw=1.2),
                    zorder=3)

    # Feedback arc: keep/discard -> back to propose (below the row)
    ax.annotate('', xy=(2.15, steps[0][1] - bh/2),
                xytext=(7.70, steps[0][1] - bh/2),
                arrowprops=dict(
                    arrowstyle='->',
                    color='#2c8a3a', lw=1.2,
                    connectionstyle='arc3,rad=0.42'),
                zorder=3)
    ax.text(4.92, 2.28, 'update best config',
            ha='center', va='center', fontsize=7.2,
            color='#2c8a3a', style='italic', zorder=5)

    # ── Injection arrow: outer -> inner (right side) ─────────────────────────
    ax.annotate('', xy=(9.0, 2.65), xytext=(9.0, 0.98),
                arrowprops=dict(
                    arrowstyle='->', color=inject_col, lw=1.5,
                    connectionstyle='arc3,rad=0.0'),
                zorder=6)
    ax.text(9.55, 1.82, 'new\nmechanism\n(Python\ncode)',
            ha='center', va='center', fontsize=7.0,
            color=inject_col, fontweight='bold', zorder=6,
            multialignment='center')

    plt.tight_layout(pad=0.2)
    out = os.path.join(FIGURES_DIR, 'bilevel_concept.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {out}  ({os.path.getsize(out)//1024} KB)')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Architecture Diagram
# ─────────────────────────────────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, text, fontsize=10,
             facecolor='#dce8f7', edgecolor='#2c5f8a',
             linewidth=1.5, text_color='#1a1a2e', bold=False):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.035",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold' if bold else 'normal',
            zorder=4, multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, color='#444444', lw=1.4,
               connectionstyle='arc3,rad=0.0'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=connectionstyle), zorder=2)

def make_architecture_figure():
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # Colors
    L2_edge  = '#2d6a2d';  L2_face  = '#edf7ed'
    L15_edge = '#9b6a00';  L15_face = '#fff8ec'
    L1_edge  = '#2c5f8a';  L1_face  = '#dce8f7'
    arrow_col = '#444444'

    bw, bh = 2.0, 0.72

    # ── Level 1 row — y = 1.25 ───────────────────────────────────────────────
    L1_y = 1.25
    L1_xs = [1.5, 3.7, 5.9]
    labels_L1 = ['Propose\nhyperparameters', 'Train\n(mini-run)', 'Evaluate\nval BPB']
    for x, lbl in zip(L1_xs, labels_L1):
        draw_box(ax, x, L1_y, bw, bh, lbl, fontsize=8.5,
                 facecolor=L1_face, edgecolor=L1_edge)

    # keep/discard diamond
    kx, ky, kr = 8.35, L1_y, 0.46
    diamond = plt.Polygon(
        [(kx, ky+kr), (kx+kr, ky), (kx, ky-kr), (kx-kr, ky)],
        closed=True, facecolor='white', edgecolor=L1_edge, linewidth=1.4, zorder=3)
    ax.add_patch(diamond)
    ax.text(kx, ky, 'Keep?\nDiscard?', ha='center', va='center',
            fontsize=7.5, color='#1a1a2e', zorder=4)

    # L1 arrows left->right
    for x1, x2 in [(L1_xs[0]+bw/2, L1_xs[1]-bw/2),
                   (L1_xs[1]+bw/2, L1_xs[2]-bw/2),
                   (L1_xs[2]+bw/2, kx-kr)]:
        draw_arrow(ax, x1, L1_y, x2, L1_y, color=arrow_col)

    # Feedback arc below
    draw_arrow(ax, kx-kr*0.72, ky-kr*0.72, L1_xs[0], L1_y-bh/2-0.05,
               color='#2c8a3a', lw=1.2, connectionstyle='arc3,rad=-0.38')
    ax.text(4.9, L1_y-1.0, 'keep \u2192 update best config',
            ha='center', fontsize=7.2, color='#2c8a3a', style='italic', zorder=5)

    # Level 1 dashed border + label
    rect1 = plt.Rectangle((0.3, 0.22), 9.4, 1.70,
                           facecolor='none', edgecolor=L1_edge,
                           linewidth=1.0, alpha=0.55, zorder=0, linestyle='--')
    ax.add_patch(rect1)
    ax.text(0.55, 1.07, 'Level 1\nInner Loop', fontsize=7.5, color=L1_edge,
            fontweight='bold', va='center', ha='center', rotation=90, zorder=5)

    # ── Level 1.5 row — y = 3.55 ─────────────────────────────────────────────
    L15_y = 3.55
    L15_xs = [2.2, 5.0, 7.8]
    L15_lbs = ['Analyze\ntrace', 'Freeze / unfreeze\nparameters', 'Inject\nguidance']
    for x, lbl in zip(L15_xs, L15_lbs):
        draw_box(ax, x, L15_y, 2.2, bh, lbl, fontsize=8.5,
                 facecolor=L15_face, edgecolor=L15_edge)

    # L1.5 arrows
    for x1, x2 in [(L15_xs[0]+1.1, L15_xs[1]-1.1),
                   (L15_xs[1]+1.1, L15_xs[2]-1.1)]:
        draw_arrow(ax, x1, L15_y, x2, L15_y, color=arrow_col)

    # Observe trace: down-left from Analyze to L1
    draw_arrow(ax, L15_xs[0], L15_y-bh/2,
               L1_xs[1], L1_y+bh/2+0.05,
               color=L15_edge, lw=1.1, connectionstyle='arc3,rad=0.15')
    ax.text(3.0, 2.55, 'observe\ntrace', ha='center', fontsize=7.0,
            color=L15_edge, style='italic', zorder=5)

    # Inject guidance: down-right from Inject to L1
    draw_arrow(ax, L15_xs[2], L15_y-bh/2,
               L1_xs[2], L1_y+bh/2+0.05,
               color=L15_edge, lw=1.1, connectionstyle='arc3,rad=-0.15')
    ax.text(7.35, 2.55, 'inject\nguidance', ha='center', fontsize=7.0,
            color=L15_edge, style='italic', zorder=5)

    # Outer cycle arc above L1.5
    draw_arrow(ax, L15_xs[2]+1.1, L15_y,
               L15_xs[0]-1.1, L15_y,
               color=L15_edge, lw=1.0, connectionstyle='arc3,rad=-0.45')
    ax.text(5.0, 4.42, 'outer cycle (every ~5 iters)', ha='center',
            fontsize=7.2, color=L15_edge, style='italic', zorder=5)

    # Level 1.5 dashed border + label
    rect15 = plt.Rectangle((0.3, 2.98), 9.4, 1.55,
                            facecolor='none', edgecolor=L15_edge,
                            linewidth=1.0, alpha=0.55, zorder=0, linestyle='--')
    ax.add_patch(rect15)
    ax.text(0.55, 3.75, 'Level 1.5\nOuter\nConfig', fontsize=7.0, color=L15_edge,
            fontweight='bold', va='center', ha='center', rotation=90, zorder=5)

    # ── Level 2 row — y = 5.65 ───────────────────────────────────────────────
    L2_y = 5.65
    L2_xs = [1.5, 3.6, 5.7, 7.8]
    L2_lbs = ['Explore', 'Critique', 'Specify', 'Generate\nCode']
    L2_bw = 1.75
    for x, lbl in zip(L2_xs, L2_lbs):
        draw_box(ax, x, L2_y, L2_bw, bh, lbl, fontsize=8.5,
                 facecolor=L2_face, edgecolor=L2_edge)

    # L2 arrows
    for i in range(3):
        draw_arrow(ax, L2_xs[i]+L2_bw/2, L2_y,
                   L2_xs[i+1]-L2_bw/2, L2_y, color=arrow_col)

    # Generate Code -> down into L1.5
    draw_arrow(ax, L2_xs[3], L2_y-bh/2,
               L15_xs[2], L15_y+bh/2+0.05,
               color=L2_edge, lw=1.6, connectionstyle='arc3,rad=0.0')
    ax.text(8.45, 4.62, 'inject new\noperator', ha='center', fontsize=7.0,
            color=L2_edge, style='italic', zorder=5)

    # Level 2 dashed border + label
    rect2 = plt.Rectangle((0.3, 5.10), 9.4, 1.30,
                           facecolor='none', edgecolor=L2_edge,
                           linewidth=1.0, alpha=0.55, zorder=0, linestyle='--')
    ax.add_patch(rect2)
    ax.text(0.55, 5.75, 'Level 2\nMechanism\nResearch', fontsize=7.0, color=L2_edge,
            fontweight='bold', va='center', ha='center', rotation=90, zorder=5)

    ax.text(5.0, 0.08, 'Three-level bilevel architecture',
            ha='center', fontsize=8.0, color='#555555', style='italic')

    plt.tight_layout(pad=0.3)
    out = os.path.join(FIGURES_DIR, 'architecture.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {out}  ({os.path.getsize(out)//1024} KB)')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Convergence Plot
# ─────────────────────────────────────────────────────────────────────────────

# Raw bpb traces extracted from report.json files.
# None / 0 entries represent crashes / missing values and are handled below.
RAW = {
    'A': {
        1: [1.104926, 1.099455, 1.096541, 1.096392, 1.096958, 1.104669,
            1.098602, 1.098849, 1.099993, 1.098106, 1.104465, 1.103348,
            1.102655, 1.097075, 1.102265, 1.101465, 1.103291, 1.102892,
            1.103016, 1.103372, 1.106494, 1.103706, 1.103419, 1.104103,
            1.103538, 1.104564, 1.105907, 1.104832, 1.100601, 1.103520, 1.103218],
        2: [1.104021, 1.158734, 1.095873, 1.106459, 1.107236, 1.098443,
            1.107215, 1.106447, 1.098239, 1.101973, 1.098088, 1.097392,
            1.097764, 1.097405, 1.098886, 1.097144, 1.099009, 1.097783,
            1.096139, 1.095696, 1.098004, 1.096185, 1.098237, 1.096026,
            1.097682, 1.097680, 1.095995, 1.098510, 1.097955, 1.098225, 1.097886],
        3: [1.105416, 1.158474, 1.097917, 1.095723, 1.102423, 1.099058,
            1.098094, 1.099360, 1.094656, 1.095807, 1.098044, 1.098201,
            1.096376, 1.099423, 1.096066, 1.095851, 1.097276, 1.098413,
            1.096266, 1.096131, 1.098073, 1.095681, 1.097793, 1.097244,
            1.095620, 1.099590, 1.098316, 1.096194, 1.097274, 1.096277, 1.104563],
    },
    'B': {
        1: [1.093949, 1.157096, 1.094391, 1.101268, 1.106670, 1.097744,
            1.103889, 1.097420, 1.104159, 1.093727, 1.104202, 1.159041,
            1.158842, 1.140181, 1.102543, 1.105097, 1.109972, 1.109621,
            1.109262, 1.108900, 1.109920, 1.110854, 1.105812, 1.109945,
            1.102680, 1.102966, None,     None,     1.101607, 1.100623, 1.095669],
        2: [1.102937, 1.156532, 1.095065, 1.095582, 1.096205, 1.098445,
            1.098208, 1.092555, 1.096145, 1.095700, 1.094498, 1.092892,
            1.101091, 1.100224, 1.104662, 1.098635, 1.099258, 1.099400,
            1.101152, 1.100511, 1.102195, 1.108016, 1.108029, 1.107989,
            1.114029, 1.114062, 1.100750, 1.101109, 1.102770, 1.109887, 1.111135],
        3: [1.102621, 1.156510, 1.097509, 1.096661, 1.101438, 1.096620,
            1.095120, 1.096220, 1.095957, 1.095509, 1.094531, 1.096392,
            1.095834, 1.096340, 1.095980, 1.095813, 1.106751, 1.106842,
            1.106960, 1.103762, 1.101755, 1.096033, 1.095846, 1.095725,
            1.095627, 1.095076, 1.095792, 1.095427, 1.093647, 1.095525, 1.093430],
    },
    'C': {
        1: [1.113135, 1.172910, 1.106085, 1.116545, 1.109112, 1.112619,
            1.105621, 1.103146, 1.103284, 1.103992, 1.105388, 1.104619,
            1.174490, 1.110007, 1.119097, 1.120848, 1.111519, 1.112758,
            1.110669, 1.108164, 1.109963, 1.107603, 1.107070, 1.177173,
            1.177118, 1.094075, 1.092607, 1.089541, 1.092200, 1.093315,
            1.089622, 1.094010, 1.047687],
        2: [1.114388, 1.174098, 1.106524, 1.105265, 1.108159, 1.105159,
            1.108278, 1.105693, 1.107761, 1.103468, 1.107899, 1.106577,
            1.159018, 1.108457, 1.108677, 1.109814, 1.109248, 1.109775,
            1.109907, 1.126658, 1.109870, 1.111437, 1.112000, 1.161643,
            1.113512, 1.113569, 1.117205, 1.126880, 1.113253, 1.110994,
            1.111656, 1.111551, 1.113216],
        3: [1.112773, 1.170566, 1.116434, 1.109653, 1.119422, 1.119631,
            1.110661, 1.108950, 1.409488, 1.414232, 1.107743, 1.108012,
            1.172785, 1.111079, 1.113298, 1.114936, 1.115027, 1.117190,
            1.115995, 1.115226, 1.116269, 1.114837, 1.117561, 1.187605,
            1.121426, 1.118650, 1.119581, 1.122227, 1.096162, 1.093497,
            1.054782, None,     1.095010],
    },
    'D': {
        1: [1.094523, 1.156784, 1.097471, 1.100751, 1.106853, 1.107800,
            1.097840, 1.093543, 1.100283, 1.103687, 1.096558, 1.139630,
            1.164546, 1.164602, 1.156534, 1.178502, 1.179424, 1.180908,
            1.163905, 1.164918, 1.134843, 1.096022, 1.095569, 1.097471,
            1.097448, 1.096014, 1.097582, 1.100405, 1.102740, 1.099697,
            1.098669],
        2: [1.103541, 1.157798, 1.101411, 1.102225, 1.100371, 1.102166,
            1.103196, 1.101669, 1.101275, 1.100304, 1.101694, 1.174462,
            1.176914, 1.175952, 1.178483, 1.177756, 1.180971, 1.184570,
            1.192759, 1.176999, 1.177062, 1.101112, 1.102647, 1.103326,
            1.104695, 1.101397, 1.064849, 1.042758, 0,        1.079200,
            1.040898],
        3: [1.102442, 1.156714, 1.096653, 1.099004, 1.096178, 1.102235,
            1.101726, 1.099560, 1.101996, 1.098199, 1.098325, 1.104898,
            1.098201, 1.097932, 1.162270, 1.066580, 1.065990, 1.064057,
            1.063226, 1.065619, 1.067413, 1.098209, 1.106213, 1.106064,
            1.111887, 1.122913, 1.143605, 1.158832, 1.169070, 1.175786,
            1.181361],
    },
}

def running_min(vals, crash_thresh=1.20):
    """Running minimum, treating None/0/crash values as carry-forward."""
    out = []
    cur = float('inf')
    for v in vals:
        if v is None or v == 0 or v > crash_thresh:
            out.append(cur if cur < float('inf') else np.nan)
        else:
            cur = min(cur, v)
            out.append(cur)
    return out


def make_convergence_figure():
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728'}
    group_labels = {
        'A': 'Group A (Level 1)',
        'B': 'Group B (Level 1+1.5)',
        'C': 'Group C (Level 1+1.5+2)',
        'D': 'Group D (Level 1+2)',
    }

    for group in ['A', 'B', 'C', 'D']:
        col = colors[group]
        all_rmins = []
        for rep_vals in RAW[group].values():
            rm = running_min(rep_vals)
            iters = list(range(len(rm)))
            ax.plot(iters, rm, color=col, alpha=0.30, linewidth=1.0, zorder=2)
            all_rmins.append(rm)

        # align lengths for mean
        max_len = max(len(r) for r in all_rmins)
        padded = [r + [r[-1]] * (max_len - len(r)) for r in all_rmins]
        mean_rm = np.nanmean(padded, axis=0)
        ax.plot(range(len(mean_rm)), mean_rm,
                color=col, linewidth=2.6, label=group_labels[group], zorder=3)

    # vertical dashed lines for Level-2 interventions (Group C at iters 10, 20)
    for xpos in [10, 20]:
        ax.axvline(x=xpos, color='#2ca02c', linestyle='--',
                   linewidth=1.2, alpha=0.75, zorder=1)

    ax.set_xlim(0, 32)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Best val BPB (running min)', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.6)

    # annotate intervention lines after limits are finalised
    ylim = ax.get_ylim()
    ytext = ylim[0] + (ylim[1] - ylim[0]) * 0.025
    for xpos in [10, 20]:
        ax.text(xpos + 0.25, ytext, 'L2 intervention',
                fontsize=7.5, color='#2ca02c', va='bottom',
                style='italic', zorder=5)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'convergence.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {out}  ({os.path.getsize(out)//1024} KB)')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Level 2 Research Session Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def make_level2_session_figure():
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    L2_edge = '#2d6a2d'
    L2_face = '#edf7ed'
    arrow_col = '#444444'
    out_color = '#555577'

    # Step boxes: (x_center, label, output_label)
    steps = [
        (1.1,  'Explore',       'hypotheses'),
        (3.1,  'Critique',      'selected\nhypothesis'),
        (5.1,  'Specify',       'spec + interface'),
        (7.1,  'Generate\nCode','Python module'),
        (9.1,  'Validate\n& Inject', 'pass / fail\n→ revert'),
    ]

    bw, bh = 1.55, 0.85
    box_y = 2.45
    out_y  = 1.35

    for (x, lbl, out_lbl) in steps:
        # main box
        box = FancyBboxPatch(
            (x - bw / 2, box_y - bh / 2), bw, bh,
            boxstyle="round,pad=0.04",
            facecolor=L2_face, edgecolor=L2_edge, linewidth=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x, box_y, lbl, ha='center', va='center', fontsize=9.5,
                color='#1a1a2e', fontweight='bold', zorder=4,
                multialignment='center')
        # output label below
        ax.text(x, out_y, out_lbl, ha='center', va='top', fontsize=7.8,
                color=out_color, style='italic', zorder=4,
                multialignment='center')
        # downward tick from box to output label
        ax.annotate('', xy=(x, out_y + 0.12), xytext=(x, box_y - bh / 2),
                    arrowprops=dict(arrowstyle='->', color=out_color,
                                   lw=0.9), zorder=2)

    # horizontal arrows between boxes
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + bw / 2
        x2 = steps[i + 1][0] - bw / 2
        ax.annotate('', xy=(x2, box_y), xytext=(x1, box_y),
                    arrowprops=dict(arrowstyle='->', color=arrow_col,
                                   lw=1.4), zorder=2)

    ax.text(5.0, 3.75, 'Level 2 Research Session (4 LLM calls)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=L2_edge)

    plt.tight_layout(pad=0.3)
    out = os.path.join(FIGURES_DIR, 'level2_session.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {out}  ({os.path.getsize(out)//1024} KB)')


if __name__ == '__main__':
    make_bilevel_concept_figure()
    make_architecture_figure()
    make_convergence_figure()
    make_level2_session_figure()
    print('Done.')
