import os
import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp

def load_data(prefix='DBLP'):
    from .data_loader import data_loader
    dl = data_loader('data/' + prefix)#这里加载了数据加载的信息，节点集，边集，标签训练集、标签测试集
    # print(os.path.abspath('../../data/' + prefix))

    # dl = data_loader('data/'+prefix)
    features = []#这里便得到了所有的节点属性
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]#这里获取节点各类的属性（4057,334）、（14328,4231）、（7723,50）
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)

    adjM = sum(dl.links['data'].values())
    type_mask = np.zeros(dl.nodes['total'], dtype=np.int32)
    
    node_type_num = len(dl.nodes['count'])#这一步获取了节点的不同类型总数，，然后在dl.nodes['count']记录了不同类型节点的各个数量
    
    for i in range(node_type_num):#for i in range（4）：由左闭右开原则，i的结果为0、1、2、3，这里是为了区分不同类型的节点
        type_mask[dl.nodes['shift'][i]: dl.nodes['shift'][i]+dl.nodes['count'][i]] = i
        
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)#这样的矩阵用于存储节点的标签信息
    val_ratio = 0.2#验证率设为0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0] * val_ratio)#1217*0.2=243
    val_idx = train_idx[:split]#这个操作通常用于将数据集划分成训练集和验证集。通过使用 train_idx 中找到的索引，从训练集中选择了一部分用作验证集。
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    if prefix != 'IMDB':
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    # emb = np.load('D:/pycharm_item/AUTOAC/AutoAC-main/data/' + prefix + '/metapath2vec_emb.npy')
    # emb = np.load('/home/yyj/MDNN-AC/AutoAC-main/data/' + prefix + '/metapath2vec_emb.npy')
    return features,\
        adjM, \
        type_mask, \
        labels,\
        train_val_test_idx,\
        dl#,emb

