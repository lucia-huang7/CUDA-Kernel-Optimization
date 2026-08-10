"""Render independent L4 GEMM confirmation results using the repository figure style."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

root=Path(__file__).resolve().parents[1];source=root/'results/gemm-confirmation.json'
cases=json.loads(source.read_text())['cases']
font_manager.findfont('Times New Roman',fallback_to_default=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'path',
                     'svg.hashsalt':'gpu-kernel-lab-gemm','axes.linewidth':1.5,'figure.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(12.4,4.6),gridspec_kw={'width_ratios':[1.5,1]})
methods=[('naive','CUDA baseline','#9ecfe2'),('tiled','Shared tile','#85a9bb'),
         ('tuned','Tuned CUDA','#5b8851'),('pytorch','PyTorch','#b8b8b8'),('cublas','cuBLAS','#555555')]
for ax,subset in zip(axes,[cases[:2],cases[2:]]):
    x=np.arange(len(subset));ax.set_facecolor('#efefef');ax.grid(axis='y',color='white',linewidth=1.6);ax.set_axisbelow(True)
    for i,(key,label,color) in enumerate(methods):
        vals=[c['methods'][key]['graph_us_median'] for c in subset]
        bars=ax.bar(x+(i-2)*.155,vals,.155,color=color,edgecolor='black',linewidth=1.1,label=label)
        ax.bar_label(bars,labels=[f'{v:.1f}' for v in vals],padding=3,fontsize=8,fontweight='bold')
    top=max(c['methods'][k]['graph_us_median'] for c in subset for k,_,_ in methods)
    ax.set_ylim(0,top*1.22)
    ax.set_xticks(x,[f"{c['m']} × {c['n']} × {c['k']}" for c in subset]);ax.set_xlabel('M × N × K')
    ax.set_ylabel('Latency (μs) ↓')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.51,1),fontsize=10,frameon=True)
fig.subplots_adjust(left=.07,right=.985,bottom=.24,top=.83,wspace=.25)
fig.text(.33,.035,'(a) Small-M and small-K GEMM',ha='center',fontfamily='Times New Roman',fontsize=14)
fig.text(.81,.035,'(b) Square GEMM',ha='center',fontfamily='Times New Roman',fontsize=14)
out=root/'assets/figures';out.mkdir(parents=True,exist_ok=True)
for suffix in ['png','svg']:
    p=out/f'gemm_v1.{suffix}';kwargs={'metadata':{'Date':None}} if suffix=='svg' else {}
    fig.savefig(p,dpi=220,bbox_inches='tight',**kwargs)
    if suffix=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
plt.close(fig)
(out/'gemm-manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'fonts':{'plot':'DejaVu Sans','captions':'Times New Roman','svg':'outlined'},
    'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('gemm_v1.*')}},indent=2)+'\n')
print('Rendered GEMM PNG and SVG from independent confirmation data.')
