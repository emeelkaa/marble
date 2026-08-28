import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

def corr_metric(x, y):
    assert x.shape == y.shape, f'{x.shape} and {y.shape}'
    r = np.corrcoef(x.squeeze(), y.squeeze())[0, 1]
    return r

# ── load ──────────────────────────────────────────────────────────────────────
pred   = np.load('neurobolt/pred.npy').reshape(6, 567, 64)
true   = np.load('neurobolt/true.npy').reshape(6, 567, 64)
pred_unebolt = np.load('neurobolt/pred_une.npy').reshape(6, 567, 64)
true_unebolt = np.load('neurobolt/true_une.npy').reshape(6, 567, 64)
pred_neurobolt = np.load('neurobolt/pred_neurobolt.npy').reshape(6, 567, 64)
true_neurobolt = np.load('neurobolt/true_neurobolt.npy').reshape(6, 567, 64)
pred_biot = np.load('neurobolt/pred_biot.npy').reshape(6, 567, 64)
true_biot = np.load('neurobolt/true_biot.npy').reshape(6, 567, 64)
labels = pd.read_csv('neurobolt/splits/labels_64_dictionary.csv')

networks = labels['Yeo_networks7'].values

# ── network config ────────────────────────────────────────────────────────────
NETWORK_ORDER = ['VisCent', 'SomMotA', 'DorsAttnB', 'SalVentAttnA', 'ContA', 'DefaultB']

NETWORK_LABELS = {
    'VisCent':      'Visual',
    'DorsAttnB':    'Dorsal Attention',
    'DefaultB':     'Default',
    'SalVentAttnA': 'Salience',
    'SomMotA':      'Somatomotor',
    'ContA':        'Control',
}

MODELS = [
    {'name': 'MARBLE',      'pred': pred,           'true': true,           'color': '#FEC89A'},
    {'name': 'UnEBOLT',      'pred': pred_unebolt,    'true': true_unebolt,      'color': '#5B8DB8'},
    {'name': 'NeuroBOLT', 'pred': pred_neurobolt,  'true': true_neurobolt, 'color': '#CBC0D3'},
    {'name': 'BIOT',      'pred': pred_biot,       'true': true_biot,      'color': '#EFD3D7'},
    
]

n_models   = len(MODELS)
n_networks = len(NETWORK_ORDER)
n_subjects = pred.shape[0]

# ── compute scores ────────────────────────────────────────────────────────────
for model in MODELS:
    model['scores'] = {net: [] for net in NETWORK_ORDER}
    for s in range(n_subjects):
        roi_corr = np.array([
            corr_metric(model['pred'][s, :, roi], model['true'][s, :, roi])
            for roi in range(64)
        ])
        for net in NETWORK_ORDER:
            rois = np.where(networks == net)[0]
            if len(rois) == 0:
                continue
            model['scores'][net].append(float(np.mean(roi_corr[rois])))

# ── plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         16,
    'axes.titleweight':  'bold',
    'axes.titlesize':    16,
    'axes.labelsize':    16,
    'xtick.labelsize':   16,
    'ytick.labelsize':   16,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(18, 7))

width       = 0.22
group_gap   = 0.5
net_centers = np.arange(n_networks) * (n_models * width + group_gap)

for m_idx, model in enumerate(MODELS):
    positions = net_centers + (m_idx - n_models / 2 + 0.5) * width
    data      = [model['scores'][net] for net in NETWORK_ORDER]

    bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='#333333', linewidth=1.8),
                    whiskerprops=dict(color='#555555', linewidth=0.9),
                    capprops=dict(color='#555555', linewidth=0.9),
                    boxprops=dict(linewidth=0.8))

    for patch in bp['boxes']:
        patch.set_facecolor(model['color'])
        patch.set_alpha(0.85)

ax.set_xticks(net_centers)
ax.set_xticklabels([NETWORK_LABELS[n] for n in NETWORK_ORDER],
                  fontsize=16)

ax.set_ylabel('Timeseries Correlation ($r$)', fontsize=16)
ax.set_ylim(0.25, 0.65)
ax.yaxis.grid(True, lw=0.5, alpha=0.4)
ax.set_axisbelow(True)

# ── legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=m['color'], edgecolor='#555555',
                   linewidth=0.5, label=m['name'], alpha=0.85)
    for m in MODELS
]
ax.legend(handles=legend_patches, fontsize=16, frameon=False, loc='upper right', handlelength=1.5, handletextpad=0.5)

plt.subplots_adjust(bottom=0.10)
plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
plt.show()