import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne

# ── load ──────────────────────────────────────────────────────────────────────
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
            'P7', 'P8', 'Fpz', 'Fz', 'Cz', 'Pz', 'POz', 'Oz', 'FT9', 'FT10', 'TP9', 'TP10']

labels   = pd.read_csv('neurobolt/splits/labels_64_dictionary.csv')
networks = labels['Yeo_networks7'].values

attn = np.load('neurobolt/attn.npy')
attn = attn.reshape(6, 567, 64, 832)

n_ch    = len(ch_names)
n_heads = attn.shape[-1] // n_ch
attn_ch = attn[..., :n_heads * n_ch].reshape(6, 567, 64, n_heads, n_ch).mean(axis=3)

attn_mean = attn_ch.mean(axis=(0, 1))  # (64, 26)

# ── MNE montage ───────────────────────────────────────────────────────────────
montage = mne.channels.make_standard_montage('standard_1020')
info    = mne.create_info(ch_names=ch_names, sfreq=1.0, ch_types='eeg')
info.set_montage(montage)

# get 2D channel positions for peak marker
pos_2d = np.array([info['chs'][i]['loc'][:2] for i in range(len(ch_names))])

# ── network config ────────────────────────────────────────────────────────────
NETWORK_ORDER = ['VisCent', 'SomMotA', 'SalVentAttnA', 'DefaultB']

NETWORK_LABELS = {
    'VisCent':      'Visual',
    'SomMotA':      'Somatomotor',
    'DefaultB':     'Default Mode',
    'SalVentAttnA': 'Salience',
}

NETWORK_COLORS = {
    'VisCent':      '#8E9AAF',
    'SomMotA':      '#CBC0D3',
    'DefaultB':     '#FEC89A',
    'SalVentAttnA': '#EFD3D7',
}

# ── compute per-network attention vector ──────────────────────────────────────
net_attn = {}
for net in NETWORK_ORDER:
    roi_idx = np.where(networks == net)[0]
    if len(roi_idx) == 0:
        continue
    v = attn_mean[roi_idx].mean(axis=0)
    net_attn[net] = (v - v.min()) / (v.max() - v.min() + 1e-8)

# ── plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        12,
    'axes.titleweight': 'bold',
    'axes.titlesize':   13,
})

fig = plt.figure(figsize=(17, 5))
gs  = gridspec.GridSpec(1, 5,
                        width_ratios=[1, 1, 1, 1, 0.05],
                        wspace=0.05)

axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
cax  = fig.add_subplot(gs[0, -1])

for ax, net in zip(axes, NETWORK_ORDER):
    data = net_attn[net]

    im, _ = mne.viz.plot_topomap(
        data, info,
        axes=ax,
        show=False,
        contours=4,
        cmap='viridis',
        vlim=(0.0, 1.0),
        sensors=True,
        names=ch_names,
    )

    ax.set_title(NETWORK_LABELS[net], fontsize=13, pad=8,
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor=NETWORK_COLORS[net],
                           edgecolor='none', alpha=0.9))

cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Normalized Attention', fontsize=11)
cbar.ax.tick_params(labelsize=9)

plt.savefig('attn_topomap.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('attn_topomap.png', dpi=300, bbox_inches='tight')
plt.show()