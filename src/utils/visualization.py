"""Visualization utilities"""
import json
import matplotlib.pyplot as plt

def _calculate_pareto_front(times,f1s,results):
    """
    Calculate Pareto front indices filtering by accuracy >= 60%.

    Args:
        times: List of training times
        f1s: List of F1 scores
        results: List of result dictionaries

    Returns:
        List of indices belonging to Pareto front
    """
    p = []
    for i in range(len(times)):
        if results[i]['metrics']['accuracy'] < 0.6:
            continue
        ok = True
        for j in range(len(times)):
            if i!=j and results[j]['metrics']['accuracy'] >= 0.6:
                if times[j]<=times[i] and f1s[j]>=f1s[i] and (times[j]<times[i] or f1s[j]>f1s[i]):
                    ok = False
                    break
        if ok:
            p.append(i)
    return p

def plot_results_comparison(results_file, figsize=(24,14)):
    """
    Create scatter plot and table comparing experiment results.

    Args:
        results_file: Path to JSON file with results
        figsize: Figure size tuple (width, height)
    """
    with open(results_file) as f:
        results = json.load(f)
    times = [r['training_time_seconds'] for r in results]
    f1s = [r['metrics']['f1'] for r in results]
    colors = ['blue' if r['configuration']['employ_quantum'] else 'red' for r in results]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2,1,height_ratios=[1.2,1],hspace=0.5)
    ax1,ax2 = fig.add_subplot(gs[0]),fig.add_subplot(gs[1])
    ax2.axis('off')

    for i,(t,f,c) in enumerate(zip(times,f1s,colors),1):
        ax1.scatter(t,f,c=c,s=150,alpha=0.7,edgecolors='black',linewidth=1.5,zorder=3)
        ax1.annotate(str(i),(t,f),ha='center',va='center',fontsize=9,fontweight='bold',color='white',zorder=4)

    pareto = _calculate_pareto_front(times,f1s,results)
    if len(pareto)>1:
        pts = sorted([(times[i],f1s[i]) for i in pareto])
        ax1.plot([p[0] for p in pts],[p[1] for p in pts],'k--',linewidth=2,alpha=0.5,zorder=2)

    ax1.set_xlabel('Training Time (s)',fontsize=12,fontweight='bold')
    ax1.set_ylabel('F1-score',fontsize=12,fontweight='bold')
    ax1.set_title('Results',fontsize=14,fontweight='bold',pad=20)
    ax1.grid(True,alpha=0.3,zorder=1)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax1.legend(handles=[Patch(facecolor='blue',edgecolor='black',label='Quantum'),Patch(facecolor='red',edgecolor='black',label='Classical'),Line2D([0],[0],color='black',linestyle='--',linewidth=2,label='Pareto Front')],loc='best')

    classical = [(i,r) for i,r in enumerate(results,1) if not r['configuration']['employ_quantum']]
    quantum = [(i,r) for i,r in enumerate(results,1) if r['configuration']['employ_quantum']]
    sel = {i for i,_ in sorted(classical,key=lambda x:x[1]['metrics']['f1'],reverse=True)[:10]}|{i for i,_ in sorted(quantum,key=lambda x:x[1]['metrics']['f1'],reverse=True)[:10]}|{i+1 for i in pareto}
    pareto_ids = {i+1 for i in pareto}

    data = [['ID','Type','Device','Batch Size','Img/Class','Opt','LR','Ep','N Qubits','FMap','Ansatz','Exec Mode','Time','Acc','F1','Stamp']]
    for rid in sorted(sel,key=lambda x:results[x-1]['metrics']['f1'],reverse=True):
        c,m = results[rid-1]['configuration'],results[rid-1]['metrics']
        data.append([str(rid),'Hybrid' if c['employ_quantum'] else 'Classical',c.get('actual_device','N/A'),str(c['batch_size']),str(c.get('images_per_class','N/A')),c['optimizer'],f"{c['learning_rate']:.4f}",str(c['num_epochs']),str(c.get('num_qubits','N/A')),'ZZFMap' if 'ZZFeatureMap' in str(c.get('feature_map','')) else 'N/A','EffSU2' if 'EfficientSU2' in str(c.get('ansatz','')) else 'N/A','Exact' if c.get('execution_mode')=='exact_simulator' else ('Noisy' if c.get('execution_mode')=='noisy_simulator' else 'N/A'),f"{results[rid-1]['training_time_seconds']:.1f}",f"{m['accuracy']*100:.2f}%",f"{m['f1']:.3f}",results[rid-1]['timestamp']])

    t = ax2.table(cellText=data,cellLoc='center',loc='center',colWidths=[.03,.05,.05,.06,.05,.04,.04,.03,.05,.06,.06,.07,.05,.05,.05,.10])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1,2.2)
    for i in range(len(data[0])):
        t[(0,i)].set_facecolor('#4CAF50')
        t[(0,i)].set_text_props(weight='bold',color='white')

    for row_idx in range(1,len(data)):
        rid = int(data[row_idx][0])
        if rid in pareto_ids:
            for col_idx in range(len(data[0])):
                t[(row_idx,col_idx)].set_facecolor('#FFE5B4')

    plt.show()
