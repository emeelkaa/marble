import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# ── load ──────────────────────────────────────────────────────────────────────
pred   = np.load('neurobolt/pred.npy').reshape(6, 567, 64)
true   = np.load('neurobolt/true.npy').reshape(6, 567, 64)
labels = pd.read_csv('neurobolt/splits/labels_64_dictionary.csv')

networks = labels['Yeo_networks7'].values

# ── network order & colours ───────────────────────────────────────────────────
NETWORK_ORDER = [
    'VisCent', 'SomMotA', 'DorsAttnB', 'SalVentAttnA',
    'ContA', 'DefaultB', 'No network found',
]

NETWORK_COLORS = {
    'VisCent':          '#8E9AAF',
    'SomMotA':          '#CBC0D3',
    'DorsAttnB':        '#DEE2FF',
    'SalVentAttnA':     '#EFD3D7',
    'ContA':            '#FFD7BA',
    'DefaultB':         '#FEC89A',
    'No network found': '#ECE4DB',
}

NETWORK_LABELS = {
    'VisCent':          'Visual',
    'SomMotA':          'Somatomotor',
    'DorsAttnB':        'Dorsal Attention',
    'SalVentAttnA':     'Salience',
    'ContA':            'Control',
    'DefaultB':         'Default',
    'No network found': 'Unassigned',
}

# ── build reorder index ───────────────────────────────────────────────────────
order = []
for net in NETWORK_ORDER:
    order.append(np.where(networks == net)[0])
order = np.concatenate(order)

networks_sorted = networks[order]

boundaries = []
current = networks_sorted[0]
for i, n in enumerate(networks_sorted):
    if n != current:
        boundaries.append(i)
        current = n

# ── helpers ───────────────────────────────────────────────────────────────────
def fc_corr(x):
    x_c = x - x.mean(axis=0, keepdims=True)
    cov  = x_c.T @ x_c / (x.shape[0] - 1)
    std  = np.sqrt(np.diag(cov) + 1e-8)
    return cov / (std[:, None] * std[None, :] + 1e-8)

from scipy.stats import rankdata

def rescale_fc(pred_mat, true_mat):
    mask = np.triu(np.ones(pred_mat.shape, dtype=bool), k=1)
    p, t = pred_mat[mask], true_mat[mask]
    ranks = rankdata(p, method='ordinal') - 1
    p_scaled = np.sort(t)[ranks]
    out = pred_mat.copy()
    out[mask] = p_scaled
    out.T[mask] = p_scaled
    np.fill_diagonal(out, 1.0)
    return out
'''
def rescale_fc(pred_mat, true_mat):
    mask = np.triu(np.ones(pred_mat.shape, dtype=bool), k=1)
    p, t = pred_mat[mask], true_mat[mask]
    p_scaled = (p - p.mean()) / (p.std() + 1e-8) * t.std() + t.mean()
    out = pred_mat.copy()
    out[mask] = p_scaled
    out.T[mask] = p_scaled
    np.fill_diagonal(out, 1.0)
    return out'''

def threshold_fc(mat, ratio=0.25):
    mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
    thresh = np.percentile(np.abs(mat[mask]), (1 - ratio) * 100)
    out = mat.copy()
    below = np.abs(out) < thresh
    np.fill_diagonal(below, False)
    out[below] = 0
    return out

def reorder(mat):
    return mat[np.ix_(order, order)]

def add_network_boundaries(ax, boundaries, n=64):
    for b in boundaries:
        ax.axhline(b - 0.5, color='white', lw=0.8, alpha=0.9)
        ax.axvline(b - 0.5, color='white', lw=0.8, alpha=0.9)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

def add_network_strips(ax, networks_sorted, boundaries, network_colors):
    n = len(networks_sorted)
    starts = [0] + boundaries
    ends   = boundaries + [n]
    unique_nets = []
    seen = set()
    for net in networks_sorted:
        if net not in seen:
            unique_nets.append(net)
            seen.add(net)
    for start, end, net in zip(starts, ends, unique_nets):
        color = network_colors[net]
        ax.add_patch(mpatches.Rectangle(
            (start - 0.5, -2.5), end - start, 1.5,
            color=color, clip_on=False, transform=ax.transData))
        ax.add_patch(mpatches.Rectangle(
            (-2.5, start - 0.5), 1.5, end - start,
            color=color, clip_on=False, transform=ax.transData))

# ── compute ───────────────────────────────────────────────────────────────────
FOCUS = [2, 5]

corr_true     = {s: reorder(fc_corr(true[s]))                               for s in FOCUS}
corr_pred     = {s: reorder(rescale_fc(fc_corr(pred[s]), fc_corr(true[s]))) for s in FOCUS}
#corr_pred = {s: reorder(fc_corr(pred[s])) for s in FOCUS}
corr_true_thr = {s: threshold_fc(corr_true[s])                              for s in FOCUS}
corr_pred_thr = {s: threshold_fc(corr_pred[s])                              for s in FOCUS}

# shared colour limits
all_vals = np.concatenate([np.triu(corr_true[s], k=1).flatten() for s in FOCUS])
fc_abs   = np.percentile(np.abs(all_vals[all_vals != 0]), 98)

cmap_full = plt.cm.viridis.copy()

cmap_thr = plt.cm.viridis.copy()
cmap_thr.set_under('white')   # exact zeros → white for thresholded matrices

kw     = dict(vmin=-fc_abs, vmax=fc_abs, cmap=cmap_full,
              interpolation='nearest', rasterized=True)
kw_thr = dict(vmin=1e-6,   vmax=fc_abs, cmap=cmap_thr,
              interpolation='nearest', rasterized=True)

# ── plot: 2 rows × 4 cols ─────────────────────────────────────────────────────
# cols: True FC | True FC (top 25%) | Pred FC | Pred FC (top 25%)
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':         9,
    'axes.titleweight': 'bold',
    'axes.titlesize':    9,
    'axes.labelsize':    7,
    'xtick.labelsize':   6.5,
    'ytick.labelsize':   6.5,
})

fig, axes = plt.subplots(2, 4, figsize=(24, 12),
                         gridspec_kw=dict(hspace=0.1, wspace=0.32))

col_specs = [
    ('True FC',                lambda s: corr_true[s],     kw),
    ('Predicted FC',           lambda s: corr_pred[s],     kw),
    ('True (top 25%)',      lambda s: corr_true_thr[s], kw_thr),
    ('Predicted (top 25%)', lambda s: corr_pred_thr[s], kw_thr),
]

roi_ticks = np.arange(9, 64, 10)

for row, s in enumerate(FOCUS):
    last_im = None
    for col, (label, mat_fn, kwargs) in enumerate(col_specs):
        ax = axes[row, col]
        if col >= 2:
            ax.set_facecolor('white')
        mat = mat_fn(s)
        im = ax.imshow(mat, **kwargs)
        last_im = im

        add_network_boundaries(ax, boundaries)
        add_network_strips(ax, networks_sorted, boundaries, NETWORK_COLORS)

        ax.set_xticks(roi_ticks)
        ax.set_xticklabels([str(i + 1) for i in roi_ticks], fontsize=12)
        ax.set_xlabel('ROI', fontsize=12)
        ax.set_yticks([])
        ax.set_title(f'{label}', pad=20, fontsize=18)

    # one colorbar on the right of each row
    cb = fig.colorbar(last_im, ax=axes[row, :], fraction=0.015, pad=0.02)
    cb.ax.tick_params(labelsize=15)
    cb.set_label('Connectivity Strength', fontsize=15)

# ── legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=NETWORK_COLORS[n], label=NETWORK_LABELS[n])
    for n in NETWORK_ORDER
]
fig.legend(handles=legend_patches, loc='lower center', ncol=7,
           fontsize=18, frameon=False, bbox_to_anchor=(0.5, 0.05))

plt.savefig('fc_networks.png', dpi=300, bbox_inches='tight')
plt.show()