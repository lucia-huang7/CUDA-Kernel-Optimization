"""Render the measured RMSNorm reduction comparison."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

root = Path(__file__).resolve().parents[1]
source = root / 'results/rmsnorm-specialized-confirmation.json'
records = json.loads(source.read_text())['cases']
font_manager.findfont('Times New Roman', fallback_to_default=False)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'svg.fonttype': 'path', 'svg.hashsalt': 'gpu-kernel-lab',
                     'axes.linewidth': 1.5, 'figure.facecolor': 'white'})
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), gridspec_kw={'width_ratios': [1.4, 1]})
x = np.arange(len(records))
methods = [('cuda_warp', 'Previous CUDA', '#9ecfe2'),
           ('cuda_vector128', 'Vector 128', '#5b8851'),
           ('pytorch_compiled', 'PyTorch compiled', '#606060')]
for ax in axes:
    ax.set_facecolor('#efefef')
    ax.grid(axis='y', color='white', linewidth=1.6)
    ax.set_axisbelow(True)
    ax.set_xticks(x, [str(case['rows']) for case in records])
    ax.set_xlabel('Input rows')
for index, (key, label, color) in enumerate(methods):
    values = [case['methods'][key]['graph_us_median'] for case in records]
    axes[0].bar(x + (index - 1) * .24, values, .24, label=label,
                color=color, edgecolor='black', linewidth=1.25)
axes[0].set_ylabel('Latency (μs) ↓')
axes[0].set_ylim(bottom=0)
axes[0].legend(ncol=3, loc='lower left', bbox_to_anchor=(-.015, 1.02), fontsize=9,
               frameon=True, borderpad=.4, handlelength=1.3, columnspacing=1)
reductions = [(1 - case['methods']['cuda_vector128']['graph_us_median'] /
                  case['methods']['cuda_warp']['graph_us_median']) * 100 for case in records]
bars = axes[1].bar(x, reductions, .6, color='#5b8851', edgecolor='black', linewidth=1.25)
axes[1].bar_label(bars, labels=[f'{value:.1f}%' for value in reductions], padding=4,
                  fontsize=10, fontweight='bold')
axes[1].set_ylim(0, max(reductions) * 1.25)
axes[1].set_ylabel('Latency reduction (%) ↑')
fig.subplots_adjust(left=.07, right=.985, bottom=.25, top=.83, wspace=.32)
fig.text(.31, .035, '(a) Width-1536 RMSNorm latency', ha='center', fontfamily='Times New Roman', fontsize=14)
fig.text(.80, .035, '(b) Improvement over previous CUDA', ha='center', fontfamily='Times New Roman', fontsize=14)
out = root / 'assets/figures'
out.mkdir(parents=True, exist_ok=True)
for suffix in ['png', 'svg']:
    filename = out / ('rmsnorm_specialized_v1.' + suffix)
    kwargs = {'metadata': {'Date': None}} if suffix == 'svg' else {}
    fig.savefig(filename, dpi=220, bbox_inches='tight', **kwargs)
    if suffix == 'svg':
        filename.write_text('\n'.join(line.rstrip() for line in filename.read_text().splitlines()) + '\n')
plt.close(fig)
manifest = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'fonts': {'plot': 'DejaVu Sans', 'captions': 'Times New Roman', 'svg': 'outlined'},
            'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('rmsnorm_specialized_v1.*')}}
(out / 'specialized-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Rendered RMSNorm comparison PNG and SVG from saved measurements.')
