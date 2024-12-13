import torch.nn as nn
import math
import torch
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from itertools import accumulate

import torch
import dgl
from torch_geometric.nn import knn_graph
from utils.data_loader import *
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from models.model_manager import *
from utils.data_process import *
from utils.tools import *
from scipy.sparse import dia_matrix
from collections import deque
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from retrainer2 import *
from scipy.sparse import dia_matrix, csr_matrix,csc_matrix, coo_matrix
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize
from torch.backends import cudnn
from FixedNet2 import *
import tracemalloc
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import torch as th
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair


from PPNP import *
from GCN import *
from Mean import *


class MDNNModel_3(nn.Module):
    def __init__(self,ranges,args,data_info, idx_info,
                 train_info,adjM,labels,dl,type_mask,train_val_test_idx ,
                 hgs,hgs_1,split_list,heads,select,select_1,
                ):
        super(MDNNModel_3, self).__init__()

        self.args = args

        self.train_idx, self.val_idx, self.test_idx = idx_info
        self._criterion = train_info
        self.labels = labels
        self.type_mask=type_mask
        self.train_val_test_idx=train_val_test_idx
        self.dl=dl
        self.adjM = adjM

        self._logger = args.logger
        self.data_info = data_info
        self.idx_info = idx_info
        self.ranges=ranges
        self.dl = dl
        self._data_info = None
        self._infer_new_features_list =None
        self._writer=None

        self.hgs = hgs
        self.hgs_1 = hgs_1
        # self.hgs_2 = hgs_2


        self.split_list = split_list
        self.heads = heads
        self.select = select
        self.select_1 = select_1
        # self.select_2 = select_2

        self.num_features_list = [args.attn_vec_dim for _ in range(self.args.max_num_views)]
        self.multi_view_interaction_model = M_GCN_t(self.num_features_list, hidden_dim=args.attn_vec_dim).to(
            device)  # self.args.max_features_len
        self.hgnn_model_manager = ModelManager(data_info, idx_info, args)
        # 创建 GNN 模型
        self.hgnn_model = self.hgnn_model_manager.create_model_class().to(device)
        # save_dir = save_dir_name(self.args)  # 调用函数获取路径
        # self._writer = SummaryWriter(f'/home/yyj/MDNN-AC/AutoAC-main/tf-logs/{save_dir}')

        self.hgnn_preprocess = nn.Linear(args.attn_vec_dim, args.hidden_dim, bias=True).to(
            device)  # args.max_features_len
        nn.init.xavier_normal_(self.hgnn_preprocess.weight, gain=1.414).to(device)

        self.fc_list = nn.ModuleList(
            [nn.Linear(in_dim, self.args.attn_vec_dim, bias=True) for in_dim in self.data_info[5]]).to(device)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)


        #基于PPNP补全：
        # 初始化 PPNP 模型
        self.model1 = PPNP1(in_channels=self.args.attn_vec_dim, out_channels=self.args.attn_vec_dim, alpha=0.1).to(device)
        self.model2 = PPNP2(in_channels=self.args.attn_vec_dim, out_channels=self.args.attn_vec_dim, alpha=0.1).to(device)

        # 基于GCN补全：
        # 初始化 GCN 模型
        self.GCN_model1 = GCN1(in_channels=self.args.attn_vec_dim, hidden_channels=self.args.hidden_dim,
                               out_channels=self.args.attn_vec_dim).to(
            device)
        self.GCN_model2 = GCN2(in_channels=self.args.attn_vec_dim, hidden_channels=self.args.hidden_dim,
                               out_channels=self.args.attn_vec_dim).to(
            device)

    def create_retrain_model(self, new_features_list,new_data_info, new_idx_info,
                             new_train_info):  # 用于创建一个用于重新训练的模型。它接受两个参数 alpha 和 node_assign，然后基于这些参数创建一个新的 FixedNet 模型。
        inner_data_info = self.hgnn_model_manager.get_graph_info()
        gnn_model_manager = self.hgnn_model_manager
        model = FixedNet2(new_features_list,new_data_info, new_idx_info, new_train_info, inner_data_info, gnn_model_manager, self.args)

        return model
    def forward(self):
        h = []
        for fc, feature in zip(self.fc_list, self.data_info[0]):
            h.append(fc(feature.to(device)))
        h1=h.copy()
        h2=h.copy()
        # h3=h.copy()
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!基于扩散补全的模块
        #第一个视图
        for i in range(len(self.hgs)):
            preh1 = h1
            #all
            ph1 = []
            s= int(self.select[i])
            h1 = torch.cat(h1[:s+1],0)
            adj = edge_index_to_sparse_mx(torch.stack(self.hgs[i].edges(),0).cpu(),self.hgs[i].num_nodes())
            adj = process_adj(adj)
            h1 = feature_propagation(adj,h1,self.args.T,self.args.alpha).to(device)
            ph1 = torch.split(h1,self.split_list[:s+1],0)
            preh1[s] = ph1[s]
            h1 = preh1
        new_h= torch.cat(h1,0)
        new_h= new_h.squeeze()


        # 第二个视图
        for i in range(len(self.hgs_1)):
            preh2 = h2
            # all
            ph2 = []
            s = int(self.select_1[i])
            h2 = torch.cat(h2[:s + 1], 0)
            adj = edge_index_to_sparse_mx(torch.stack(self.hgs_1[i].edges(), 0).cpu(), self.hgs_1[i].num_nodes())
            adj = process_adj(adj)
            h2= feature_propagation(adj, h2, self.args.T, self.args.alpha).to(device)
            ph2 = torch.split(h2, self.split_list[ :s + 1], 0)
            preh2[s] = ph2[s]
            h2 = preh2
        new_h_2 = torch.cat(h2, 0)
        new_h_2 = new_h_2.squeeze()

        '''''' #多视角加消融，启用
        # ####基于PPNP的方法进行补全：
        # #第一个视图
        # for i in range(len(self.hgs)):
        #     preh1 = h1
        #     #all
        #     ph1 = []
        #     s= int(self.select[i])
        #     h1 = torch.cat(h1[:s+1],0)
        #
        #     out1 = self.model1(h1, torch.stack(self.hgs[i].edges(),0).to(device))
        #     # 使用PPNP输出对缺失值进行补全
        #     x_filled_1 = ppnp_based_aggregation(h1.clone(), out1)
        #
        #     ph1 = torch.split(x_filled_1,self.split_list[:s+1],0)
        #     preh1[s] = ph1[s]
        #     h1 = preh1
        # new_h= torch.cat(h1,0)
        # new_h= new_h.squeeze()
        #
        # # 第二个视图
        # for i in range(len(self.hgs_1)):
        #     preh2 = h2
        #     # all
        #     ph2 = []
        #     s = int(self.select_1[i])
        #     h2 = torch.cat(h2[:s + 1], 0)
        #
        #     out2 = self.model2(h2, torch.stack(self.hgs_1[i].edges(), 0).to(device))
        #     # 使用PPNP输出对缺失值进行补全
        #     x_filled_2 = ppnp_based_aggregation(h2.clone(), out2)
        #
        #     ph2 = torch.split(x_filled_2, self.split_list[ :s + 1], 0)
        #     preh2[s] = ph2[s]
        #     h2 = preh2
        # new_h_2 = torch.cat(h2, 0)
        # new_h_2 = new_h_2.squeeze()

        ####基于GCN的方法进行补全：
        # 第一个视图
        # for i in range(len(self.hgs)):
        #     preh1 = h1
        #     # all
        #     ph1 = []
        #     s = int(self.select[i])
        #     h1 = torch.cat(h1[:s + 1], 0)
        #
        #     out1 = self.GCN_model1(h1, torch.stack(self.hgs[i].edges(), 0).to(device))
        #     # 使用GCN输出对缺失值进行补全
        #     x_filled_1 = gcn_based_aggregation(h1.clone(), out1)
        #
        #     ph1 = torch.split(x_filled_1, self.split_list[:s + 1], 0)
        #     preh1[s] = ph1[s]
        #     h1 = preh1
        # new_h = torch.cat(h1, 0)
        # new_h = new_h.squeeze()
        #
        # # 第二个视图
        # for i in range(len(self.hgs_1)):
        #     preh2 = h2
        #     # all
        #     ph2 = []
        #     s = int(self.select_1[i])
        #     h2 = torch.cat(h2[:s + 1], 0)
        #
        #     out2 = self.GCN_model2(h2, torch.stack(self.hgs_1[i].edges(), 0).to(device))
        #     # 使用GCN输出对缺失值进行补全
        #     x_filled_2 = gcn_based_aggregation(h2.clone(), out2)
        #
        #     ph2 = torch.split(x_filled_2, self.split_list[:s + 1], 0)
        #     preh2[s] = ph2[s]
        #     h2 = preh2
        # new_h_2 = torch.cat(h2, 0)
        # new_h_2 = new_h_2.squeeze()
        #
        # ####基于Mean的方法进行补全：
        # # 第一个视图
        # for i in range(len(self.hgs)):
        #     preh1 = h1
        #     # all
        #     ph1 = []
        #     s = int(self.select[i])
        #     h1 = torch.cat(h1[:s + 1], 0)
        #
        #     x_filled_1 = mean_attribute_aggregation(h1.clone(), self.hgs[i].edges())
        #
        #     ph1 = torch.split(x_filled_1, self.split_list[:s + 1], 0)
        #     preh1[s] = ph1[s]
        #     h1 = preh1
        # new_h = torch.cat(h1, 0)
        # new_h = new_h.squeeze()
        #
        # # 第二个视图
        # for i in range(len(self.hgs_1)):
        #     preh2 = h2
        #     # all
        #     ph2 = []
        #     s = int(self.select_1[i])
        #     h2 = torch.cat(h2[:s + 1], 0)
        #
        #     x_filled_2 = mean_attribute_aggregation(h2.clone(), self.hgs_1[i].edges())
        #
        #     ph2 = torch.split(x_filled_2, self.split_list[:s + 1], 0)
        #     preh2[s] = ph2[s]
        #     h2 = preh2
        # new_h_2 = torch.cat(h2, 0)
        # new_h_2 = new_h_2.squeeze()

        # # 第三个视图
        # for i in range(len(self.hgs_2)):
        #     preh3 = h3
        #     # all
        #     ph3 = []
        #     s = int(self.select_2[i])
        #     h3 = torch.cat(h3[:s + 1], 0)
        #     adj = edge_index_to_sparse_mx(torch.stack(self.hgs_2[i].edges(), 0).cpu(), self.hgs_2[i].num_nodes())
        #     adj = process_adj(adj)
        #     h3 = feature_propagation(adj, h3, self.args.T, self.args.alpha).to(device)
        #     ph3 = torch.split(h3, self.split_list[:s + 1], 0)
        #     preh3[s] = ph3[s]
        #     h3 = preh3
        # new_h_3 = torch.cat(h3, 0)
        # new_h_3 = new_h_3.squeeze()

        ''''''

        views_tensors = []

        views_tensors.append(new_h.to(device))
        views_tensors.append(new_h_2.to(device))
        # views_tensors.append(new_h_3.to(device))


        #复制原始的邻接矩阵 self.args.max_num_views 次
        edge_index_list = [torch.stack(self.data_info[2].edges(),dim=0) for _ in range(self.args.max_num_views)]

        # 构造 view_data
        view_data_list = []
        for i in range(self.args.max_num_views):
            #edge_index = knn_graph(views_tensors[i], k=5, loop=True)
            # 创建一个 Data 对象，并将节点属性矩阵 views_tensors[i] 赋给 x 属性
            view_data = Data(x=views_tensors[i], edge_index=edge_index_list[i])
            view_data_list.append(view_data)
        emb_view_layer, global_emb = self.multi_view_interaction_model(view_data_list)

        '''========基于注意力机制的全局特征融合/基于扩散路径的节点属性聚合框架============================================================================================================================================================================='''
        '''这里推断其实已经结束，那就只剩构建特征以便于计算损失！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！'''


        new_features_list = [global_emb[idx_range.start:idx_range.stop] for idx_range in self.ranges]

        return  emb_view_layer, new_features_list

    def tranin_and_val(self, model,mini_batch_input=None):
        optimizer1 = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        bst_val_loss = [np.inf]  # 使用列表包装，使其成为可变对象
        _earlystop = EarlyStopping_Search(logger=self.args.logger,
                                          patience=self.args.patience_retrain)  # args.patience)#_search)

        for epoch in range(self.args.search_epoch):
            t_start = time.time()
            model.train()
            optimizer1.zero_grad()

            # 前向传播
            multi_view_nodes, new_features_list = model.forward()

            self.input, self.target = convert_np2torch(new_features_list, self.labels, self.args)
            self.combined_features = torch.cat(self.input, dim=0).to(device)
            # 额外增加线性层# 使用额外的线性层对特征进行处理
            self.h = self.hgnn_preprocess(self.combined_features)

            if self.args.use_minibatch is False:
                _, logits = self.hgnn_model_manager.forward_pass(self.hgnn_model,self.h,mini_batch_input)

                if self.args.dataset == 'IMDB':
                   logits = torch.sigmoid(logits).to(device)#softmax sigmoid
                # Calculate train loss
                logits_train = logits[self.train_idx].to(device)
                target_train = self.target[self.train_idx]
            else:
                # node_embedding, logits = [], []
                train_idx_generator = index_generator(batch_size=self.args.batch_size, indices=self.train_idx)

                minibatch_data_info = self.hgnn_model_manager.get_graph_info()
                self.adjlists, self.edge_metapath_indices_list = minibatch_data_info

                for step in range(train_idx_generator.num_iterations()):
                    train_idx_batch = train_idx_generator.next()
                    train_idx_batch.sort()
                    train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                        self.adjlists, self.edge_metapath_indices_list, train_idx_batch, device, self.args.neighbor_samples)

                    _, logits = self.hgnn_model_manager.forward_pass(self.hgnn_model, self.h, (
                    train_g_list, train_indices_list, train_idx_batch_mapped_list, train_idx_batch))
                    if self.args.dataset == 'IMDB':
                        logits = torch.sigmoid(logits).to(device)#softmax sigmoid
                    # Calculate train loss
                    logits_train = logits.to(device)
                    target_train = self.target[train_idx_batch].to(device)

            train_loss = self._criterion(logits_train, target_train)

            #Calculate consistency loss
            #这只有在多个视角的时候才能启用
            loss_consistency1 = loss_each_view(multi_view_nodes[1])

            #多视角的
            loss = train_loss + self.args.beta_1 * loss_consistency1
            #单视角的
            # loss = train_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)  #然后还要补充args.grad_clip！！
            optimizer1.step()
            # model.lr_step(epoch)
            # scheduler.step()
            t_train = time.time()
            #多个视角启用
            self._logger.info(
                'Epoch_batch_{:05d} | lr {:.4f} |Train_Loss {:.4f}|  total_loss_consistency1 {:.4f} | Loss {:.4f} | Time(s) {:.4f}'.format(
                    epoch, optimizer1.state_dict()['param_groups'][0]['lr'], train_loss.item(), loss_consistency1.item(), loss,t_train - t_start))#loss_consistency2
            # #单个视角启用
            # self._logger.info(
            #     'Epoch_batch_{:05d} | lr {:.4f} |Train_Loss {:.4f}| Time(s) {:.4f}'.format(
            #         epoch, optimizer1.state_dict()['param_groups'][0]['lr'], train_loss.item(), t_train - t_start))

            model.eval()
            with torch.no_grad():
                _, infer_new_features_list = model.forward()
                self.infer_input, self.infer_target = convert_np2torch(infer_new_features_list, self.labels, self.args,
                                                                       y_idx=self.val_idx)
                self.infer_combined_features = torch.cat(self.infer_input, dim=0).to(device)
                # 额外增加线性层 # 使用额外的线性层对特征进行处理
                self.infer_h = self.hgnn_preprocess(self.infer_combined_features)

                if self.args.use_minibatch is False:

                    _, infer_logits = self.hgnn_model_manager.forward_pass(self.hgnn_model, self.infer_h, mini_batch_input)

                    if self.args.dataset == 'IMDB':
                        infer_logits = torch.sigmoid(infer_logits).to(device)
                    logits_val = infer_logits[self.val_idx].to(device)

                    val_loss = self._criterion(logits_val, self.infer_target)

                else:
                    logits_val = []
                    val_idx_generator = index_generator(batch_size=self.args.batch_size, indices=self.val_idx,shuffle=False)
                    minibatch_data_info = self.hgnn_model_manager.get_graph_info()
                    self.adjlists, self.edge_metapath_indices_list = minibatch_data_info
                    for iteration in range(val_idx_generator.num_iterations()):
                        val_idx_batch = val_idx_generator.next()
                        val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                            self.adjlists, self.edge_metapath_indices_list, val_idx_batch, device,
                            self.args.neighbor_samples)

                        _, infer_logits = self.hgnn_model_manager.forward_pass(self.hgnn_model, self.infer_h, (
                            val_g_list, val_indices_list, val_idx_batch_mapped_list, val_idx_batch))
                        logits_val.append(infer_logits)
                    infer_logits = torch.cat(logits_val, 0).to(device)
                    if self.args.dataset == 'IMDB':
                        infer_logits = torch.sigmoid(infer_logits).to(device)
                    logits_val = infer_logits.to(device)

                    val_loss = self._criterion(logits_val, self.infer_target)

            t_end = time.time()

            self._logger.info(
                'Epoch {:05d} | lr {:.5f} |Train_Loss {:.4f} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(epoch,optimizer1.state_dict()['param_groups'][0]['lr'],loss,val_loss.item(),t_end - t_train))

            save_dir = save_dir_name(self.args)  # 调用函数获取路径
            self._writer = SummaryWriter(f'/home/yyj/MDNN-AC/AutoAC-main/tf-logs/{save_dir}')
            self._writer.add_scalar(f'{self.args.dataset}_train_loss', loss, global_step=epoch)
            self._writer.add_scalar(f'{self.args.dataset}_val_loss', val_loss, global_step=epoch)
            # # # 这里就是保存模型的代码段！！！！！！！！！！！！！！！！！！这里要改一下
            if is_save(bst_val_loss, loss, val_loss):
                 model.set_new_features_list(infer_new_features_list)
                 model.set_writer(self._writer)
            _earlystop(loss, val_loss.item())
            if _earlystop.early_stop:
                self.args.logger.info('Eearly stopping!')
                break

        # # 冻结第一个模型的权重
        # for param in model.parameters():
        #     param.requires_grad = False
        torch.cuda.empty_cache()
        gc.collect()
    def lr_step(self,epoch):
        self.lr_scheduler.step(epoch)

    # 在属性定义中，我们使用了@property装饰器来创建一个getter方法，
    # 然后使用.setter方法定义一个setter方法，这样我们就可以通过属性访问来设置值。
    def set_new_features_list(self,infer_new_features_list):
        self._infer_new_features_list = infer_new_features_list
    def get_new_features_list(self):
        return self._infer_new_features_list
    def set_data_info(self, new_data_info):
        self._data_info = new_data_info

    def set_writer(self, _writer):
        self._writer = _writer

    def get_data_info(self):
        return self.data_info

    def get_idx_info(self):
        return self.idx_info

    def get_train_info(self):
        return self._criterion

    def get_writer(self):
        return self._writer

    def set_hgnn_model_manager(self, hgnn_model_manager):
        self._hgnn_model_manager = hgnn_model_manager

    def set_hgnn_model(self,hgnn_model):
        self._hgnn_model = hgnn_model

class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MLP_encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout,args):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)
        self.args=args

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.Linear1(x), self.args.slope)
        # x = torch.relu(self.Linear1(x))
        return x

class MLP_classifier(nn.Module):#
    def __init__(self, nfeat, nclass, dropout):
        super(MLP_classifier, self).__init__()
        self.Linear1 = Linear(nfeat, nclass, dropout, bias=True)

    def forward(self, x):
        out = self.Linear1(x)
        return torch.log_softmax(out, dim=1), out

class DDPT(nn.Module):
    def __init__(self, nfeat, nhid, dropout,args, use_bn = False):
        super(DDPT, self).__init__()

        self.encoder = MLP_encoder(nfeat=nfeat,
                                 nhid=nhid,
                                 dropout=dropout,
                                   args=args)

        # self.classifier = MLP_classifier(nfeat=nhid,
        #                                  nclass=nclass,
        #                                  dropout=dropout)

        self.proj_head1 = Linear(nhid, nhid, dropout, bias=True)

        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nfeat)
            self.bn2 = nn.BatchNorm1d(nhid)

    def forward(self, features, eval = False):
        if self.use_bn:
            features = self.bn1(features)
        query_features = self.encoder(features)
        if self.use_bn:
            query_features = self.bn2(query_features)
        return query_features
        # output, emb = self.classifier(query_features)
        # if not eval:
        #     emb = self.proj_head1(query_features)
        # return emb, output

def edge_index_to_sparse_mx(edge_index, num_nodes):
    edge_weight = np.array([1] * len(edge_index[0]))
    adj = csc_matrix((edge_weight, (edge_index[0], edge_index[1])),
                     shape=(num_nodes, num_nodes)).tolil()
    return adj
def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def process_adj(adj):
    adj.setdiag(1)
    '''最终结果是一个新的稀疏矩阵，它只包含 adj.T 中大于 adj 的元素。
    这在某些图操作或图算法中可以用来提取图中权重较大的反向边或进行特定的图过滤操作。'''
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj
def feature_propagation(adj, features, K, alpha):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    adj = adj.to(device)
    features_prop = features.clone()
    for i in range(1, K + 1):
        features_prop = torch.sparse.mm(adj, features_prop)
        features_prop = (1 - alpha) * features_prop + alpha * features
    return features_prop.cpu()
def get_random_batch(n, batch_size):
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    batches = []
    i = 0
    while i + batch_size * 2 < n:
        batches.append(idxs[i:i + batch_size])
        i += batch_size
    batches.append(idxs[i:])
    return batches
def global_knn(x, num_neighbor, batches, knn_metric):
    row = None
    for batch in batches:
        knn_current = kneighbors_graph(x[batch], num_neighbor, metric=knn_metric).tocoo()
        row_current = batch[knn_current.row]
        col_current = batch[knn_current.col]
        if row is None:
            row = row_current
            col = col_current
        else:
            row = np.concatenate((row, row_current))
            col = np.concatenate((col, col_current))
    return row, col
from sklearn.neighbors import kneighbors_graph
def get_knn_graph(x, num_neighbor, batch_size=0, knn_metric='cosine', connected_fast=True):
    if not batch_size:
        adj_knn = kneighbors_graph(x, num_neighbor, metric=knn_metric)
    else:
        if connected_fast:
            print('compute connected fast knn')
            num_neighbor1 = int(num_neighbor / 2)
            batches1 = get_random_batch(x.shape[0], batch_size)
            row1, col1 = global_knn(x, num_neighbor1, batches1, knn_metric)
            num_neighbor2 = num_neighbor - num_neighbor1
            batches2 = get_random_batch(x.shape[0], batch_size)
            row2, col2 = global_knn(x, num_neighbor2, batches2, knn_metric)
            row, col = np.concatenate((row1, row2)), np.concatenate((col1, col2))
        else:
            print('compute fast knn')
            batches = get_random_batch(x.shape[0], batch_size)
            row, col = global_knn(x, num_neighbor, batches, knn_metric)
        adj_knn = coo_matrix((np.ones_like(row), (row, col)), shape=(x.shape[0], x.shape[0]))

    return adj_knn.tolil()