import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ── load ──────────────────────────────────────────────────────────────────────
pred           = np.load('neurobolt/pred.npy').reshape(6, 567, 64)
true           = np.load('neurobolt/true.npy').reshape(6, 567, 64)
pred_unebolt = np.load('neurobolt/pred_une.npy').reshape(6, 567, 64)
true_unebolt = np.load('neurobolt/true_une.npy').reshape(6, 567, 64)
labels         = pd.read_csv('neurobolt/splits/labels_64_dictionary.csv')

networks = labels['Yeo_networks7'].values

# ── config ────────────────────────────────────────────────────────────────────
#NETWORK_ORDER = ['VisCent', 'SomMotA', 'DorsAttnB', 'SalVentAttnA', 'ContA', 'DefaultB']
NETWORK_ORDER = ['VisCent', 'SomMotA', 'SalVentAttnA', 'DefaultB']

NETWORK_COLORS_STRIP = {
    'VisCent':      '#8E9AAF',
    'SomMotA':      '#CBC0D3',
    'DorsAttnB':    '#DEE2FF',
    'SalVentAttnA': '#EFD3D7',
    'ContA':        '#FFD7BA',
    'DefaultB':     '#FEC89A',
}

NETWORK_LABELS = {
    'VisCent':      'Visual',
    'SomMotA':      'Somatomotor',
    'DorsAttnB':    'Dorsal Attention',
    'SalVentAttnA': 'Salience',
    'ContA':        'Control',
    'DefaultB':     'Default Mode',
}

OURS_COLOR      = '#E07B39'  
NEUROBOLT_COLOR = '#5B8DB8' 

# ── helpers ───────────────────────────────────────────────────────────────────
def rescale_timeseries(pred_ts, true_ts):
    t_mean = true_ts.mean(axis=0, keepdims=True)
    t_std  = true_ts.std(axis=0,  keepdims=True) + 1e-8
    p_mean = pred_ts.mean(axis=0, keepdims=True)
    p_std  = pred_ts.std(axis=0,  keepdims=True) + 1e-8
    return (pred_ts - p_mean) / p_std * t_std + t_mean

def best_roi_per_network(pred_ts, true_ts, networks, network_order):
    r_all = np.array([
        np.corrcoef(true_ts[:, roi], pred_ts[:, roi])[0, 1]
        for roi in range(true_ts.shape[1])
    ])
    best = {}
    for net in network_order:
        rois = np.where(networks == net)[0]
        if len(rois) == 0:
            continue
        best_roi = rois[np.argmax(r_all[rois])]
        best[net] = (best_roi, r_all[best_roi])
    return best

# ── settings ──────────────────────────────────────────────────────────────────
FOCUS   = [2, 5]
T_RANGE = (0, 150)

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         15,
    'axes.titleweight':  'bold',
    'axes.titlesize':    15,
    'axes.labelsize':    15,
    'xtick.labelsize':   13,
    'ytick.labelsize':   13,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

t_start, t_end = T_RANGE
time   = np.arange(t_start, t_end)
n_nets = len(NETWORK_ORDER)

for s in FOCUS:
    pred_scaled    = rescale_timeseries(pred[s],           true[s])
    pred_nb_scaled = rescale_timeseries(pred_unebolt[s], true_unebolt[s])
    best           = best_roi_per_network(pred_scaled, true[s], networks, NETWORK_ORDER)
    '''
    fig, axes = plt.subplots(3, 2, figsize=(24, 12),
                             gridspec_kw=dict(hspace=0.3, wspace=0.15))
    '''
    fig, axes = plt.subplots(2, 2, figsize=(24, 10),
                             gridspec_kw=dict(hspace=0.3, wspace=0.15))

    for row, net in enumerate(NETWORK_ORDER):
        #ax = axes[row % 3, row // 3]
        ax = axes[row % 2, row // 2]
        roi, r_ours = best[net]

        y_true = true[s,            t_start:t_end, roi]
        y_ours = pred_scaled[       t_start:t_end, roi]
        y_nb   = pred_nb_scaled[    t_start:t_end, roi]
        r_une   = np.corrcoef(true_unebolt[s, :, roi], pred_unebolt[s, :, roi])[0, 1]

        ax.plot(time, y_true, color='#999999',      lw=1.5, linestyle='--', alpha=0.9, label='True',      zorder=2)
        ax.plot(time, y_ours, color=OURS_COLOR,      lw=1.5, alpha=0.85,               label='MARBLE',      zorder=3)
        ax.plot(time, y_nb,   color=NEUROBOLT_COLOR, lw=1.5, alpha=0.85,               label='UnEBOLT', zorder=3)

        ax.spines['left'].set_color(NETWORK_COLORS_STRIP[net])
        ax.spines['left'].set_linewidth(3)

        roi_name = labels['Difumo_names'].iloc[roi]
        ax.set_ylabel('BOLD', fontsize=15)
        ax.set_title(
            f'{NETWORK_LABELS[net]} '
            f'$\\mathbf{{r_{{mbl}} = {r_ours:.3f}}}$,  $\\mathbf{{r_{{une}} = {r_une:.3f}}}$',
            fontsize=18, loc='left')

        #if row % 3 == 2:
        if row % 2 == 1:
            ax.set_xlabel('Timepoint', fontsize=15)
        else:
            ax.set_xticklabels([])

        if row == 0:
            ax.legend(fontsize=15, frameon=False, loc='upper right',
                      bbox_to_anchor=(1.10, 1.18))

    plt.savefig(f'timeseries_S{s}.png', dpi=300, bbox_inches='tight')
    plt.show()