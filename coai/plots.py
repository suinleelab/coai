import pdb
import numpy as np
import shap
from matplotlib import pyplot as plt
from sklearn import tree, metrics, base, datasets, utils, exceptions
from collections import Counter

from matplotlib import colors
cdict = {'red':   [[0.0,  1.0, 1.0],
                   [0.5,  0.0, 0.0],
                   [1.0,  0.0, 0.0]],
         'green': [[0.0,  1.0, 1.0],
                   [0.5, 0.0, 0.0],
                   [1.0,  0.0, 0.0]],
         'blue':  [[0.0,  1.0, 1.0],
                   [0.5,  1.0, 1.0],
                   [1.0,  0.0, 0.0]]}
broken_cdict = {'red':   [[0.0,  0.8, 0.8],
                   [0.5,  0.0, 0.0],
                   [1.0,  0.0, 0.0]],
         'green': [[0.0,  0.8, 0.8],
                   [0.5, 0.0, 0.0],
                   [1.0,  0.0, 0.0]],
         'blue':  [[0.0,  1.0, 1.0],
                   [0.5,  1.0, 1.0],
                   [1.0,  0.0, 0.0]]}
newcmp = colors.LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
brokencmp = colors.LinearSegmentedColormap('testCmap', segmentdata=broken_cdict, N=256)
plt.style.use('default')

def get_axis(ax):
    if ax is None: return plt.gca()
    else: return ax
    
def get_twin(ax):
    for other_ax in ax.figure.axes:
        if other_ax is ax:
            continue
        if other_ax.bbox.bounds == ax.bbox.bounds:
            return other_ax
    return None

def auaucc(costs,scores,newfig=True,**kwargs):
    if newfig: plt.figure()
    order = np.argsort(costs)
    plt.step(np.array(costs)[order],np.maximum.accumulate(np.array(scores)[order]),where='post',**kwargs)

def cost_performance_curve(model,ax=None,**kwargs):
    ax = get_axis(ax)
    ax.set_xlabel("Model cost")
    ax.set_ylabel("Model performance")
    ax.step(model.model_costs,model.model_scores,where='post',**kwargs)
    ax.set_xlim(np.min(model.model_costs),np.max(model.model_costs))
    
def first_feature_uses(model):
    return np.array([np.where([i in f for f in model.model_features])[0][0] for i in range(len(model.feature_costs))])


def importance_grid(model,ax=None, ticks=False, fig=None, features=None, maxrow=np.inf,maxcol=np.inf,cmap=newcmp,**kwargs):
    ax = get_axis(ax)
    mat = model.global_attribs.T
#     mat = np.hstack([np.zeros((mat.shape[0],1)),mat])
    starts = first_feature_uses(model)
    order = np.argsort(starts)[::-1]
    # order = np.argsort(np.abs(mat[:,-1])/model.feature_costs)
    
    if features is None and hasattr(model.X,'columns'):
        features = model.X.columns
    elif features is None:
        features = np.arange(mat.shape[1])
    else:
        features = features
    newfeats = features[order]
    
    h,w = min(maxrow,mat.shape[0]),min(maxcol,mat.shape[1])
    widths = model.model_costs[:w]
    heights = np.arange(h+1)[::-1]
    
    X,Y = np.meshgrid(widths,heights)
    im = ax.pcolormesh(X,Y, mat[order][::-1][:h,:w], cmap=cmap, **kwargs)
    
    ax.set_yticks(np.arange(h)+0.5)
    ax.set_ylim(0,h)
    ax.set_yticklabels([k for k in newfeats[-(h):]])
    ax.set_ylabel("Most Important Features")

    if ticks:
        for t in widths:
            # plt.axvline(t,ymin=0,ymax=0.05,color='black')
            plt.axvline(t,color='black',alpha=0.05)
    
    if fig: 
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal',aspect=60)
        cbar.ax.set_xlabel("Feature Importance")
    else:
        ax.set_xlabel("Feature Importance")
        
    plt.tight_layout()

def compare_importances(*shaplist, indiv=True, features=None, k=10, ax=None):
    if ax is None: ax=plt.gca()
    global_attribs = [np.mean(np.abs(s),axis=0) for s in shaplist] if indiv else shaplist
    assert all([len(k)==len(global_attribs[0]) for k in global_attribs])
    global_attribs = np.array(global_attribs).T
    topfeats = []

    def renamer(ind):
        return features[ind] if features is not None else str(ind)

    for i in range(global_attribs.shape[0]):
        for j in range(global_attribs.shape[1]):
            ind = np.argsort(global_attribs[:,j][::-1])[i]
            if ind not in topfeats: topfeats.append(ind)
            if len(topfeats)>=20: break
        if len(topfeats)>=20: break
    for ind in topfeats:
        ax.plot(global_attribs[ind,:],label = renamer(ind), marker='o')
    ax.set_yticklabels([renamer(ind) for ind in topfeats[::-1]])
    ax.legend()

def compare_orders(*orderlist, indiv=True, features=None, ax=None, model_names=None, min_per_model=None, max_per_model=None,higher_first_color='#D55E00',higher_second_color='#0072B2'):
    if model_names is None: model_names=['' for i in orderlist]
    if ax is None: ax=plt.gca()
    # if type(drop_bottom_k) is int: drop_bottom_k = np.array([drop_bottom_k]*len(orderlist))
    # else: drop_bottom_k = np.array(drop_bottom_k)
    if min_per_model is None: min_per_model=10
    if type(min_per_model) is int:
        min_per_model = [min_per_model for o in orderlist]
    min_per_model = np.array(min_per_model)
    if max_per_model is None: max_per_model = min_per_model.copy()
    max_per_model = np.array(max_per_model)
    global_order = orderlist
    assert all([len(o)==len(global_order[0]) for o in global_order])

    # Global order: Columns index models, rows index features. Each row gives index of the $ith$ 
    # most important feature for that model.
    global_order = np.array(global_order).T
    topfeats = []

    def renamer(ind):
        return features[ind] if features is not None else str(ind)

    topfeats = []
    for i,n in enumerate(min_per_model):
        vals = list(global_order[:,i][:n])
        topfeats.extend(vals)

    # for i in range(global_order.shape[0]):
    #     for j in range(global_order.shape[1]):
    #         # Next most important feature
    #         ind = global_order[i,j]
    #         if ind not in topfeats and i<=(global_order.shape[1]-drop_bottom_k[j]): topfeats.append(ind)
    #         if len(topfeats)>=k: break
    #     if len(topfeats)>=k: break


        
    all_vals = []
    for ind in topfeats:
        vals = np.array([global_order.shape[0]-np.where(global_order[:,j]==ind)[0][0] for j in range(global_order.shape[1])]).astype(float)
        vals[vals<(global_order.shape[0]-max_per_model)]=np.nan
        # print(np.where(vals<drop_bottom_k))
        if vals[-1]>vals[0] or np.isnan(vals[0]): color=higher_first_color
        elif vals[0]>vals[-1] or np.isnan(vals[-1]): color=higher_second_color
        else: color = 'gray'
        
        all_vals.append(vals)
        ax.plot(vals,label=renamer(ind), marker='o',color=color)
    ax.set_yticks([v[0] for v in all_vals])
    ax.set_yticklabels([renamer(ind) for ind in topfeats])
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names)
    # ax.set_ylabel("Features")
    # ax.set_ylabel("Models")
    axt = ax.twinx()
    axt.set_yticks([v[-1] for ind,v in zip(topfeats,all_vals) if not np.isnan(v[-1])])
    axt.set_yticklabels([renamer(ind) for ind,v in zip(topfeats,all_vals) if not np.isnan(v[-1])])#[str(s) for s in len(all_vals)-np.arange(len(all_vals))+1])
    axt.set_ylim(*ax.get_ylim())
    # axt.set_ylabel("Feature Importance Rank")

    # ax.legend()
