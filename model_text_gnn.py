from config import FLAGS

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import get_scheduler
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros

from sentence_transformers import SentenceTransformer
# import hf text classification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

import numpy as np
import evaluate
from tqdm.auto import tqdm

# import train test split
from sklearn.model_selection import train_test_split

metric = evaluate.load("accuracy")
device = torch.device("cpu")

class TextGNN(nn.Module):
    def __init__(self, pred_type, node_embd_type, num_layers, layer_dim_list, act, bn, num_labels, class_weights, dropout, llm=False, llm_model=None):
        super(TextGNN, self).__init__()
        self.node_embd_type = node_embd_type
        self.layer_dim_list = layer_dim_list
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_labels = num_labels
        if pred_type == 'softmax' or pred_type == 'avg_softmax_ensemble':
            assert layer_dim_list[-1] == num_labels
        elif pred_type == 'mlp':
            num_dims = layer_dim_list[-1]
            if eval(llm):
                num_dims = layer_dim_list[-2]
                if "MiniLM" in llm_model:
                    num_dims += 384
                elif "tas-b" in llm_model:
                    num_dims += 768
            dims = self._calc_mlp_dims(num_dims, num_labels)
            self.mlp = MLP(num_dims, num_labels, num_hidden_lyr=len(dims), hidden_channels=dims, bn=False)
        elif pred_type in ["sm_ensemble_mlp", "full_ensemble_mlp"]:
            num_dims = layer_dim_list[-1] * 2
            if pred_type == "full_ensemble_mlp":
                if "tas-b" in llm_model:
                    num_dims += 768
                num_dims += layer_dim_list[-2]
                assert num_dims == num_labels*2 + 768 + layer_dim_list[-2]
            else:
                assert num_dims == num_labels*2
            assert eval(llm)
            dims = self._calc_mlp_dims(num_dims, num_labels)
            self.mlp = MLP(num_dims, num_labels, num_hidden_lyr=len(dims), hidden_channels=dims, bn=False)
        elif pred_type == "embed_ensemble_mlp":
            num_dims = 768
            num_dims += layer_dim_list[-2]
            assert num_dims == layer_dim_list[-2] + 768
            dims = self._calc_mlp_dims(num_dims, num_labels)
            self.mlp = MLP(num_dims, num_labels, num_hidden_lyr=len(dims), hidden_channels=dims, bn=False)
        elif pred_type == "llm_embed_mlp":
            num_dims = 768
            dims = self._calc_mlp_dims(num_dims, num_labels)
            self.mlp = MLP(num_dims, num_labels, num_hidden_lyr=len(dims), hidden_channels=dims, bn=False)

        self.pred_type = pred_type
        assert len(layer_dim_list) == (num_layers + 1)
        self.act = act
        self.bn = bn
        self.layers = self._create_node_embd_layers()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        llm = eval(llm)
        if llm != False and pred_type not in ["sm_ensemble_mlp", "full_ensemble_mlp", "embed_ensemble_mlp", "llm_embed_mlp", "avg_softmax_ensemble"]:
            print(f'Using LLM: {llm_model}')
            self.llm = SentenceTransformer(llm_model)
        if pred_type in ["sm_ensemble_mlp", "full_ensemble_mlp", "embed_ensemble_mlp", "avg_softmax_ensemble", "llm_embed_mlp"]:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
            self.llm = AutoModelForSequenceClassification.from_pretrained( \
                llm_model, num_labels=num_labels, output_hidden_states=True)
                                                                          

    def forward(self, pyg_graph, dataset, epoch_num):
        acts = [pyg_graph.x]
        for i, layer in enumerate(self.layers):
            ins = acts[-1]
            outs = layer(ins, pyg_graph)
            acts.append(outs)
        
        last_layer = acts[-1]
        
        pred_inds = dataset.node_ids
        if self.pred_type in  ["sm_ensemble_mlp", "full_ensemble_mlp", "avg_softmax_ensemble", "embed_ensemble_mlp", "llm_embed_mlp"]:
            # add code to fine-tune LLM
            if epoch_num == 0:
                print("Fine-tuning LLM on epoch 0")
                self.fine_tune_llm(dataset, pred_inds, self.llm)


        return self._loss(last_layer, dataset, acts[-2])
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    def tokenise(self, docs):
        return self.tokenizer(docs, padding=True,  return_tensors="pt")

    def fine_tune_llm(self, dataset, train_inds, model):
        max_length = 512
        # split train_inds into train and val
        train_inds, val_inds = train_test_split(train_inds, test_size=0.1)

        train_docs = [dataset.docs[i] for i in train_inds]
        val_docs = [dataset.docs[i] for i in val_inds]
        train_labels = dataset.label_inds[train_inds]
        val_labels = dataset.label_inds[val_inds]

        train_encodings = self.tokenizer(train_docs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        val_encodings = self.tokenizer(val_docs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        train_dataset = [{"input_ids": train_encodings["input_ids"][i], "attention_mask": train_encodings["attention_mask"][i], "labels": train_labels[i]} for i in range(len(train_labels))]
        val_dataset = [{"input_ids": val_encodings["input_ids"][i], "attention_mask": val_encodings["attention_mask"][i], "labels": val_labels[i]} for i in range(len(val_labels))]

        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        optimizer = Adam(model.parameters(), lr=5e-5)

        num_epochs = 3
        num_training_steps = num_epochs * len(train_dataloader)
    
        progress_bar = tqdm(range(num_training_steps))

        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        
        model.eval()
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions.detach(), references=batch["labels"].detach())

        results = metric.compute()
        print(results)

    def _loss(self, ins, dataset, prev_layer_gcn=None):
        pred_inds = dataset.node_ids

        prev_layer_gcn = prev_layer_gcn[pred_inds]
        
        if hasattr(self, 'llm') and \
            self.pred_type not in ["sm_ensemble_mlp", "full_ensemble_mlp", "embed_ensemble_mlp", "avg_softmax_ensemble", "llm_embed_mlp"] :
            vals = prev_layer_gcn[pred_inds]
            with torch.no_grad():
                llm_embs = self.llm.encode([dataset.docs[i] for i in pred_inds])
            # llm_embs = torch.tensor(llm_embs, dtype=torch.float, device=FLAGS.device)
            vals = torch.cat((vals, llm_embs), dim=1)
        else:
            vals = ins[pred_inds]

        if self.pred_type == 'softmax':
            y_preds = vals
        elif self.pred_type == 'mlp':
            y_preds = self.mlp(vals)
        elif self.pred_type in  ["sm_ensemble_mlp", "full_ensemble_mlp", "avg_softmax_ensemble"]:
            with torch.no_grad():
                inputs = self.tokenizer([dataset.docs[i] for i in pred_inds], padding=True, truncation=True, max_length=512, return_tensors="pt")
                llm_result = self.llm(**inputs)
                llm_logits = llm_result.logits
                llm_logits = F.softmax(llm_logits, dim=1)
            assert llm_logits.shape[1] == self.num_labels
            # llm_logits = torch.tensor(llm_logits, dtype=torch.float, device=FLAGS.device)
            if self.pred_type == "full_ensemble_mlp":
                with torch.no_grad():
                    llm_embed = llm_result.hidden_states[-1].mean(dim=1)
                inp = torch.cat((vals, prev_layer_gcn, llm_logits, llm_embed), dim=1)
                y_preds = self.mlp(inp)
            elif self.pred_type == "sm_ensemble_mlp":
                y_preds = self.mlp(torch.cat((vals, llm_logits), dim=1))
            elif self.pred_type == "avg_softmax_ensemble":
                y_preds = (llm_logits + vals)/2
        elif self.pred_type in ['embed_ensemble_mlp', 'llm_embed_mlp']:
            inputs = self.tokenizer([dataset.docs[i] for i in pred_inds], padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                llm_result = self.llm(**inputs)
                llm_embed = llm_result.hidden_states[-1].mean(dim=1)
            if 'embed_ensemble_mlp':
                y_preds = self.mlp(torch.cat((llm_embed, prev_layer_gcn), dim=1))
            elif 'llm_embed_mlp':
                y_preds = self.mlp(llm_embed)
        else:
            raise NotImplementedError
        y_true = torch.tensor(dataset.label_inds[pred_inds], dtype=torch.long, device=FLAGS.device)
        loss = self.loss(y_preds, y_true)
        return loss, y_preds.cpu().detach().numpy()

    def _create_node_embd_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            act = self.act if i < self.num_layers - 1 else 'identity'
            # act = 'identity'
            layers.append(NodeEmbedding(
                type=self.node_embd_type,
                in_dim=self.layer_dim_list[i],
                out_dim=self.layer_dim_list[i + 1],
                act=act,
                bn=self.bn,
                dropout=self.dropout if i != 0 else False
            ))
        return layers

    def _calc_mlp_dims(self, mlp_dim, output_dim=1):
        dim = mlp_dim
        dims = []
        while dim > output_dim:
            dim = dim // 2
            dims.append(dim)
        dims = dims[:-1]
        return dims


class NodeEmbedding(nn.Module):
    def __init__(self, type, in_dim, out_dim, act, bn, dropout):
        super(NodeEmbedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.type = type
        if type == 'gcn':
            self.conv = GCNConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        elif type == 'gat':
            self.conv = GATConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        else:
            raise ValueError(
                'Unknown node embedding layer type {}'.format(type))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        if dropout:
            self.dropout = torch.nn.Dropout()

    def forward(self, ins, pyg_graph):
        if self.dropout:
            ins = self.dropout(ins)
        if self.type == 'gcn':
            if FLAGS.use_edge_weights:
                x = self.conv(ins, pyg_graph.edge_index, edge_weight=pyg_graph.edge_attr)
            else:
                x = self.conv(ins, pyg_graph.edge_index)
        else:
            x = self.conv(ins, pyg_graph.edge_index)
        x = self.act(x)
        return x


class MLP(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2,
                 hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels")
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                   for i in range(len(self.layer_channels) - 1)])))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                layer_inputs.append(self.activation(layer(input)))
        # model.store_layer_output(self, layer_inputs[-1])
        if self.bn:
            layer_inputs[-1] = self.bn(layer_inputs[-1])
        return layer_inputs[-1]


def create_act(act, num_parameters=None):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError('Unknown activation function {}'.format(act))


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index = add_self_loops(edge_index, num_nodes)
        loop_weight = torch.full((num_nodes, ),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if x.is_sparse:
            x = torch.sparse.mm(x, self.weight)
        else:
            x = torch.matmul(x, self.weight)

        if not self.cached or self.cached_result is None:
            edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,
                                            self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{j} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions. (default:
            :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated. (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True):
        super(GATConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        if x.is_sparse:
            x = torch.sparse.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        else:
            x = torch.matmul(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes):
        # Compute attention coefficients.
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], num_nodes)

        # Sample attention coefficients stochastically.
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
