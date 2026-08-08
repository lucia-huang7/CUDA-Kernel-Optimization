"""Plot independent L4 LayerNorm confirmation measurements."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

root=Path(__file__).resolve().parents[1]
source=root/'results/layernorm-confirmation.json'
cases=json.loads(source.read_text())['cases']
font_manager.findfont('Times New Roman',fallback_to_default=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'path',
                     'svg.hashsalt':'gpu-kernel-lab-layernorm','axes.linewidth':1.5,'figure.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(12.4,4.5),gridspec_kw={'width_ratios':[1.3,1]})
methods=[('cuda_shared','CUDA baseline','#9ecfe2'),('cuda_vector','Vectorized CUDA','#5b8851'),
         ('pytorch_native','PyTorch native','#b5b5b5'),('pytorch_compiled','PyTorch compiled','#555555')]
for ax,subset in zip(axes,[cases[:4],cases[4:]]):
    x=np.arange(len(subset));ax.set_facecolor('#efefef');ax.grid(axis='y',color='white',linewidth=1.6)
    ax.set_axisbelow(True)
    for i,(key,label,color) in enumerate(methods):
        vals=[c['methods'][key]['graph_us_median'] for c in subset]
        bars=ax.bar(x+(i-1.5)*.19,vals,.19,color=color,edgecolor='black',linewidth=1.2,label=label)
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=3,fontsize=8,fontweight='bold')
    top=max(c['methods'][key]['graph_us_median'] for c in subset for key,_,_ in methods)
    ax.set_ylim(0,top*1.23);ax.set_xticks(x,[str(c['rows']) for c in subset]);ax.set_xlabel('Input rows')
    ax.set_ylabel('Latency (μs) ↓')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.51,1.0),fontsize=10,frameon=True)
fig.subplots_adjust(left=.065,right=.985,bottom=.24,top=.83,wspace=.25)
fig.text(.31,.035,'(a) Small and medium inputs',ha='center',fontfamily='Times New Roman',fontsize=14)
fig.text(.80,.035,'(b) Larger inputs',ha='center',fontfamily='Times New Roman',fontsize=14)
out=root/'assets/figures';out.mkdir(parents=True,exist_ok=True)
for suffix in ['png','svg']:
    p=out/f'layernorm_v1.{suffix}';kwargs={'metadata':{'Date':None}} if suffix=='svg' else {}
    fig.savefig(p,dpi=220,bbox_inches='tight',**kwargs)
    if suffix=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
plt.close(fig)
(out/'layernorm-manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'fonts':{'plot':'DejaVu Sans','captions':'Times New Roman','svg':'outlined'},
    'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('layernorm_v1.*')}},indent=2)+'\n')
print('Rendered LayerNorm PNG and SVG from independent confirmation data.')
