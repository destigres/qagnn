import json
import math

import numpy as np

from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
import torch.nn.functional as F
import torch
from json.decoder import JSONDecodeError
import h5py

class QAGNN_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size,
                    dropout=0.1):
        super().__init__()
        assert input_size == output_size
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size//2)

        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, hidden_size//2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, hidden_size//2)
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)

        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype +1 + n_ntype *2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))


        self.k = k
        self.gnn_layers = nn.ModuleList([GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])


        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

    # def get_top_k_indices(self,att,k):
    #     top_k_indices = torch.topk(torch.mean(att[-1],dim=-1)[:len(att[-1])-1000],k)
    #     return
    # def get_bottom_k_indices(self,att,k):
    #     top_k_indices =  torch.topk(torch.mean(att[-1],dim=-1)[:len(att[-1])-1000],k,largest=False)
    #     return


    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra):
        for _ in range(self.k):
            _X,att = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training = self.training)
        return _X,att



    def forward(self, H, A, node_type, node_score, cache_output=False):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        _batch_size, _n_nodes = node_type.size()

        #Embed type
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]


        X = H
        edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node// bs * num_nodes * n_dim
        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]

        _X, att = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra)

        X = _X.view(node_type.size(0), node_type.size(1), -1) #[batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return output,att



class QAGNN(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02,generate=False,num_edges_per_subgraph=.20):
        super().__init__()
        self.init_range = init_range

        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=False, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
        self.svec2nvec = nn.Linear(sent_dim, concept_dim)

        self.concept_dim = concept_dim

        self.generate = generate
        self.num_edges_per_subgraph = num_edges_per_subgraph

        self.activation = GELU()

        self.gnn = QAGNN_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                        input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim, dropout=p_gnn)

        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)

        self.fc = MLP(concept_dim + sent_dim + concept_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)
        self.num_times = 0
        self.total = 0
        self.top_ten_indices = 0
        self.bottom_ten_indices = 0
        self.total_indices = 0
        self.top_mean_edges = 0
        self.bottom_mean_edges = 0
        self.top_percents = []
        self.bottom_percents = []
        self.idx = 0
        self.max_num_edges = 0

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_top_k(self,attn_subgraphs,og_edge_type,og_edge_idx):
        num_subgraphs = len(attn_subgraphs)
        top_k_subgraphs = []
        for i in range(num_subgraphs):
            top_10 = math.ceil(len(torch.mean(attn_subgraphs[i], -1)) * .1)
            indices = torch.topk(torch.mean(attn_subgraphs[i], -1), top_10,largest=True).indices
            top_10_indices = set(og_edge_idx[i].T[indices.cpu().numpy()].cpu().numpy().flatten())
            self.top_ten_indices += len(top_10_indices)
            mean_edges = np.mean([len(torch.where(og_edge_idx[i].T == j)[0].numpy()) for j in top_10_indices])
            if np.isnan(mean_edges):
                self.top_mean_edges += 0
            else:
                self.top_mean_edges += mean_edges
            if indices.is_cuda:
                indices = torch.topk(torch.mean(attn_subgraphs[i], -1), top_10,largest=True).indices.cpu().numpy()
            else:
                indices = torch.topk(torch.mean(attn_subgraphs[i], -1), top_10, largest=True).indices.numpy()
            top_k_subgraphs.append(
                indices
            )
        return top_k_subgraphs


    def break_attn_values_into_subgraphs(self,attn,og_edge_type,og_edge_index):
        num_subgraphs = len(og_edge_index)
        attn_subgraphs = []
        prev_num_edges = 0
        for i in range(num_subgraphs):
            idxs_belonging_to_subgraph = torch.where(sum(attn[0].T[..., :1].flatten() == i for i in set(range(800, 1000))).bool() == True)[0]

            attn_subgraphs.append(attn[-1][prev_num_edges:prev_num_edges+num_edges_in_subgraph])
            prev_num_edges = num_edges_in_subgraph
        return attn_subgraphs

    def get_bottom_k(self,attn_subgraphs,og_edge_type,og_edge_idx):
        num_subgraphs = len(attn_subgraphs)
        top_k_subgraphs = []
        for i in range(num_subgraphs):
            bottom_10 = math.ceil(len(torch.mean(attn_subgraphs[i], -1)) * .1)
            indices = torch.topk(torch.mean(attn_subgraphs[i], -1), bottom_10,largest=False).indices
            bottom_10_indices = set(og_edge_idx[i].T[indices.cpu().numpy()].cpu().numpy().flatten())
            self.bottom_ten_indices += len(bottom_10_indices)
            mean_edges = np.mean([len(torch.where(og_edge_idx[i].T == j)[0].numpy()) for j in bottom_10_indices])
            if np.isnan(mean_edges):
                self.top_mean_edges += 0
            else:
                self.bottom_mean_edges += mean_edges

            if indices.is_cuda:
                indices = torch.topk(torch.mean(attn_subgraphs[i], -1), bottom_10,largest=False).indices.cpu().numpy()
            else:
                indices = torch.topk(torch.mean(attn_subgraphs[i], -1), bottom_10, largest=False).indices.numpy()
            top_k_subgraphs.append(
                indices
            )
        return top_k_subgraphs
    def get_attn_stats(self,attn,og_edge_idx,og_edge_type,node_type_ids,q_id):
        edge_indices,attn_values = attn
        num_subgraphs = len(node_type_ids)
        ## there are self loops between nodes i.e. [0,0] ; [1,1] ... [999,999]
        for i in range(num_subgraphs):
            # edge_idxs_in_graph_i = torch.where(sum(edge_indices.T[..., :1].flatten() == j for j in set(range(i*200, (i+1)*200))).bool() == True)
            offset = i * 200
            num_edge_indices_per_src_node_in_graph_i = [len(torch.where(edge_indices.T[..., :1].flatten() == j)[0].cpu()) for j in range(0+offset,200+offset)]
            num_edge_indices_per_src_node_in_graph_i_sorted = np.argsort(num_edge_indices_per_src_node_in_graph_i)
            max_graph_node_i = num_edge_indices_per_src_node_in_graph_i_sorted[199]
            min_graph_node_i = num_edge_indices_per_src_node_in_graph_i_sorted[0]

            max_src_node_idx = max_graph_node_i + offset
            min_src_node_idx = min_graph_node_i + offset
            edges_with_src_node_max_edge_idx = torch.where(edge_indices.T[..., :1].flatten() == max_src_node_idx)[0].cpu()
            edges_with_src_node_min_edge_idx = torch.where(edge_indices.T[..., :1].flatten() == min_src_node_idx)[0].cpu()

            assert len(edges_with_src_node_max_edge_idx) >= len(edges_with_src_node_min_edge_idx) ## sanity check to ensure that node we've identifed w/ max number of edges has in fact more edges

            attn_over_edges_with_max_src_node = attn_values[edges_with_src_node_max_edge_idx]
            attn_over_edges_with_min_src_node = attn_values[edges_with_src_node_min_edge_idx]
            assert torch.abs(torch.sum(torch.mean(attn_over_edges_with_max_src_node,dim=-1))-1) <= .001 ## checking to see that attn values over edges is ~1
            assert torch.abs(torch.sum(torch.mean(attn_over_edges_with_min_src_node,dim=-1))-1) <= .001## checking to see that attn values over edges is ~1

            avg_attn_on_max_connected_edge_in_graph_i = torch.mean(torch.mean(attn_over_edges_with_max_src_node,dim=-1))
            avg_attn_on_min_connected_edge_in_graph_i = torch.mean(torch.mean(attn_over_edges_with_min_src_node,dim=-1))
            if avg_attn_on_min_connected_edge_in_graph_i >= avg_attn_on_max_connected_edge_in_graph_i:
                self.num_times += 1
            self.total += 1

    # get_edges_for_deletion(self,relevant_nodes_idxs,)
    def get_edges_for_deletion_based_on_node_relevance(self,subgraph_idx,edge_indices,edge_idxs_subgraph_i,node_scores,attn_values,edge_to_type,mode,percent_to_mask,pad_to):
        ## there are self loops between nodes i.e. [0,0] ; [1,1] ... [999,999] - we'll ignore these + any edges out of context node + any edges to context node -- no benefit
        context_node = 200 * (subgraph_idx)
        num_edges_to_mask = int(len(edge_idxs_subgraph_i) * percent_to_mask)
        edge_idxs_mask = []
        descending = mode == 'top'
        relevant_node_idxs_sorted = (torch.argsort(node_scores[subgraph_idx].flatten()[1:],descending=descending) + 1).cpu().numpy().tolist() #we're ignoring context node z.
        assert 0 not in relevant_node_idxs_sorted # assert #we're ignoring context node z.
        if mode == 'top':
            relevant_nodes = relevant_node_idxs_sorted
        else:
            relevant_nodes = relevant_node_idxs_sorted
        for node in relevant_nodes:
            avail_edges_idxs = torch.where((edge_indices.T[...,0] == node) & (edge_indices.T[...,1] != context_node)  & (edge_indices.T[...,1] != node))
            k = max(min(len(attn_values[avail_edges_idxs])-1,abs(len(edge_idxs_mask)-num_edges_to_mask)//2),0)
            top_edges = torch.topk(torch.mean(attn_values[avail_edges_idxs], dim=-1),k=k,largest=descending).indices
            edge_idxs_mask += edge_indices.T[avail_edges_idxs[0][top_edges]].cpu().numpy().tolist()
            edge_idxs_mask += [[edge[1],edge[0]] for edge in edge_indices.T[avail_edges_idxs[0][top_edges]].cpu().numpy().tolist()]
            if len(edge_idxs_mask) >= num_edges_to_mask:
                break
        edges_to_mask, edge_types_of_masked = np.array(edge_idxs_mask)[:num_edges_to_mask],\
             np.array([edge_to_type[tuple(edge)] for edge in edge_idxs_mask])[:num_edges_to_mask]
        to_pad = pad_to - len(edges_to_mask)
        edges_to_mask_pad = np.array(edges_to_mask.tolist() + np.negative(np.ones((to_pad, 2))).tolist())
        edge_types_of_masked_pad = np.array(edge_types_of_masked.tolist() + np.negative(np.ones((to_pad,))).tolist())
        return edges_to_mask_pad,edge_types_of_masked_pad,edges_to_mask,edge_types_of_masked

    def get_edges_to_mask(self,attn,node_scores,og_edge_index,og_edge_type,node_type_ids,q_id,concept_ids):
        # seeing redudant edge....
        question_dict = {}
        with open('/our_dataset/version_1/id_qa_map.pkl', 'rb') as file:
            question_dict = pickle.load(file)
        f = h5py.File('./our_dataset/version_1/csqa_dev.hdf5','r+')

        f.require_dataset("concept_ids",(1221,5,200),maxshape=(1221,5,200),dtype="int64")
        f.require_dataset("node_scores",(1221,5,200,1),dtype="float32")
        f.require_dataset("ground_truth_edge_types",(1221,5,6000),dtype="int64")
        f.require_dataset("ground_truth_adj",(1221,5,6000,2),dtype="int64")
        f.require_dataset("ground_attn",(1221,5,6000,4),dtype="float64")

        f.require_dataset("most_damaging_edges",(1221,5,1200,2),dtype="int64")
        f.require_dataset("most_damaging_edge_types",(1221,5,1200,),dtype="int64")

        f.require_dataset("least_damaging_edges", (1221, 5, 1200,2), dtype="int64")
        f.require_dataset("least_damaging_edge_types", (1221, 5, 1200,), dtype="int64")

        f["concept_ids"][self.idx] = concept_ids.detach().cpu().numpy()
        f["node_scores"][self.idx] = node_scores.detach().cpu().numpy()
        edge_indices, attn_values = attn

        num_subgraphs = len(node_type_ids)
        ## there are self loops between nodes i.e. [0,0] ; [1,1] ... [999,999] - we'll ignore these + any edges out of context node + any edges to context node -- no benefit
        edge_to_type = {}

        for node in range(0, 200):
            edge_to_type[tuple([node, node])] = 99
        edge_to_type[tuple([-1, -1])] = -1
        top,bottom = [],[]
        most_damaging_edges_list,most_damaging_edge_types_list = [],[]
        least_damaging_edges_list,least_damaging_edge_types_list = [],[]
        ground_truth_edges_list,ground_truth_edge_types_list,ground_truth_attn_list = [],[],[]

        for i in range(num_subgraphs):
            beg = i * 200
            end = (i + 1) * 200
            for edge,type in zip(og_edge_index[i].T.cpu().numpy(),og_edge_type[i].cpu().numpy()):
                edge_to_type[tuple(edge)] = type
            #paded  edge
            edges_idxs_in_subgraph_i = torch.where(sum(edge_indices.T[..., :1].flatten() == j for j in set(range(beg, end))).bool() == True)[0]
            edges_in_subgraph_i = edge_indices.T[edges_idxs_in_subgraph_i].cpu().numpy() - beg

            to_pad = 6000 - len(edges_in_subgraph_i)
            edges_in_subgraph_i_padded = np.array(edges_in_subgraph_i.tolist() + np.negative(np.ones((to_pad, 2))).tolist()) #need to do offsetting here

            ground_truth_edges = edges_in_subgraph_i_padded
            ground_truth_edges_types = np.array([edge_to_type[tuple(edge)] for edge in edges_in_subgraph_i_padded])

            ground_truth_edge_types_list.append(ground_truth_edges_types)
            ground_truth_edges_list.append(ground_truth_edges)
            ground_truth_attn_list.append(np.array(attn_values[edges_idxs_in_subgraph_i].cpu().numpy().tolist() + np.negative(np.ones((to_pad,4))).tolist()))

            least_damaging_edges,least_damaging_edge_types,top_edges,_ =  self.get_edges_for_deletion_based_on_node_relevance(i,edge_indices.cpu(),edges_idxs_in_subgraph_i.cpu(),node_scores.cpu(),attn_values.cpu(),edge_to_type,'top',.20,1200)
            least_damaging_edges_list.append(least_damaging_edges)
            least_damaging_edge_types_list.append(least_damaging_edge_types)

            most_damaging_edges, most_damaging_edge_types,bottom_edges,_ = self.get_edges_for_deletion_based_on_node_relevance(i,edge_indices.cpu(),edges_idxs_in_subgraph_i.cpu(),node_scores.cpu(),attn_values.cpu(), edge_to_type,'bottom',.20,1200)

            most_damaging_edges_list.append(most_damaging_edges)
            most_damaging_edge_types_list.append(most_damaging_edge_types)

            top.append(top_edges)
            bottom.append(bottom_edges)

        f['most_damaging_edges'][self.idx] = np.array(most_damaging_edges_list)
        f['most_damaging_edge_types'][self.idx] = np.array(least_damaging_edge_types_list)

        f['least_damaging_edges'][self.idx]= np.array(least_damaging_edges_list)
        f['least_damaging_edge_types'][self.idx] = np.array(least_damaging_edge_types_list)
        f['ground_truth_adj'][self.idx] = np.array(ground_truth_edges_list)
        f['ground_truth_edge_types'][self.idx] = np.array(ground_truth_edge_types_list)
        f['ground_attn'][self.idx] = np.array(ground_truth_attn_list)

        f.close()
        self.idx += 1
        np.save(f"./top_k/{q_id}.npy",np.array(top))
        np.save(f"./bottom_k/{q_id}.npy",np.array(bottom))


    def get_damaging_2_nodes(self,attn_subgraphs,og_egde_idxs,node_type_ids,q_id):
        num_subgraphs = len(attn_subgraphs)
        answer_node_damaging_top = []
        answer_node_damaging_bottom = []
        for i in range(num_subgraphs):
            answer_node = torch.where(node_type_ids[i] == 1)[0]
            if len(answer_node) > 0:
                if len(answer_node) > 1:
                    answer_node = answer_node[0]
                edges_involving_answer_node = torch.where(og_egde_idxs[i].T == answer_node)[0]
                ten_percent = int(len(edges_involving_answer_node) * .1)
                attn_sorted = torch.argsort(torch.mean(attn_subgraphs[i],dim=-1)[torch.where(og_egde_idxs[i].T == answer_node)[0]],descending=True)
                top_ten = edges_involving_answer_node[attn_sorted[:ten_percent]]
                answer_node_damaging_top.append(top_ten.cpu())
                bottom_ten = edges_involving_answer_node[attn_sorted[-ten_percent:]]
                answer_node_damaging_bottom.append(bottom_ten.cpu())
            else:
                answer_node_damaging_top.append([])
                answer_node_damaging_bottom.append([])
        top_k_np = np.array(answer_node_damaging_top, dtype=object)
        bottom_k_np = np.array(answer_node_damaging_bottom, dtype=object)
        np.save(f"./damaging_nodes/bottom_k/{q_id}.npy", bottom_k_np)
        np.save(f"./damaging_nodes/top_k/{q_id}.npy", top_k_np)
    def save_numpy_files(self,top_k,bottom_k,q_ids):
        q_id = q_ids[0]
        top_k_np = np.array(top_k,dtype=object)
        bottom_k_np = np.array(bottom_k,dtype=object)
        np.save(f"./bottom_k/{q_id}.npy",bottom_k_np)
        np.save(f"./top_k/{q_id}.npy",top_k_np)

    def forward(self, sent_vecs, concept_ids, node_type_ids, node_scores, adj_lengths, adj, og_edge_index,og_edge_type,q_ids, emb_data=None, cache_output=False):
        """
        sent_vecs: (batch_size, dim_sent)
        concept_ids: (batch_size, n_node)
        adj: edge_index, edge_type
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns: (batch_size, 1)
        """
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1) #(batch_size, 1, dim_node)
        gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, emb_data) #(batch_size, n_node-1, dim_node)
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1)) #(batch_size, n_node, dim_node)


        #Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2) #[batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
        node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]


        gnn_output,attn = self.gnn(gnn_input, adj, node_type_ids, node_scores)

        # attn_subgraphs = self.break_attn_values_into_subgraphs(attn,og_edge_type,og_edge_index)
        # self.get_attn_stats(attn,og_edge_index,og_edge_type,node_type_ids,q_ids[0])
        # self.get_edges_to_mask(attn,node_scores,og_edge_index,og_edge_type,node_type_ids,q_ids[0],concept_ids)
        # top_k = self.get_top_k(attn_subgraphs,og_edge_type,og_edge_index)
        # bottom_k = self.get_bottom_k(attn_subgraphs,og_edge_type,og_edge_index)
        # self.total_indices += np.sum(adj_lengths.cpu().numpy())
        # self.total += 1

        # self.save_numpy_files(top_k,bottom_k,q_ids)

        Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #1 means masked out

        mask = mask | (node_type_ids == 3) #pool over all KG nodes
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, Z_vecs), 1))
        logits = self.fc(concat)
        return logits, pool_attn


class LM_QAGNN(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.0, encoder_config={},
                 generate=False,
                 num_edges_per_subgraph=.2):
        super().__init__()
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = QAGNN(args, k, n_ntype, n_etype, self.encoder.sent_dim,
                                        n_concept, concept_dim, concept_in_dim, n_attention_head,
                                        fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                                        pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                        init_range=init_range,
                                        generate=generate,
                                        num_edges_per_subgraph=num_edges_per_subgraph)

    def mask_subgraphs(self,q_ids, edge_type, edge_index):
        edges_to_mask_dict  = np.load(f"./top_k/{q_ids[0]}.npy", allow_pickle=True)
        for subgraph in range(len(edge_type)):
            if not len(edges_to_mask_dict[subgraph]) > 0:
                continue
            edges_to_mask = edges_to_mask_dict[subgraph]
            edges_to_mask = list([tuple(x) for x in edges_to_mask])
            edge_idxs_to_mask = np.concatenate([torch.where((edge_index[subgraph].T[..., 0] == x[0]) & (edge_index[subgraph].T[..., 1] == x[1]))[0].cpu().numpy().flatten()
                                                for x in edges_to_mask])
            edge_index_subgraph = edge_index[subgraph].T.to("cpu").cpu().numpy()
            edge_index_subgraph = np.delete(edge_index_subgraph, edge_idxs_to_mask, axis=0)
            edge_index[subgraph] = torch.LongTensor(edge_index_subgraph.T).cuda()
            edge_type_subgraph = edge_type[subgraph].to("cpu").cpu().numpy()
            num_edges_From_context_node = len(np.where(edge_type_subgraph == 0)[0])
            num_edges_to_context_node = len(np.where(edge_type_subgraph == 19)[0])
            edge_type_subgraph = np.delete(edge_type_subgraph, edge_idxs_to_mask, axis=-1)
            # assert len(np.where(edge_type_subgraph == 0)[0]) == num_edges_From_context_node
            # assert len(np.where(edge_type_subgraph == 19)[0]) == num_edges_to_context_node
            edge_type[subgraph] = torch.LongTensor(edge_type_subgraph).cuda()
        for subgraph in range(len(edge_type)):
            edge_type[subgraph] = edge_type[subgraph].cuda()
            edge_index[subgraph] = edge_index[subgraph].cuda()
        return edge_type,edge_index


    def forward(self,q_ids, *inputs, layer_id=-1, cache_output=False, detail=False):
        """
        sent_vecs: (batch_size, num_choice, d_sent)    -> (batch_size * num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        adj_lengths: (batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )
        returns: (batch_size, 1)
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        #Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x,[]) for x in inputs[-2:]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs

        og_edge_index = edge_index
        og_edge_type = edge_type
        # if os.path.isfile(f"bottom_k/{q_ids[0]}.npy"):
        edge_type,edge_index = self.mask_subgraphs(q_ids,edge_type,edge_index)

        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]

        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        logits, attn = self.decoder(sent_vecs.to(node_type_ids.device),
                                    concept_ids,
                                    node_type_ids, node_scores, adj_lengths, adj,
                                    og_edge_index, og_edge_type,
                                    q_ids,
                                    emb_data=None, cache_output=cache_output,
                                    )
        logits = logits.view(bs, nc)
        if not detail:
            return logits, attn
        else:
            return logits, attn, concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc, -1), edge_index_orig, edge_type_orig
            #edge_index_orig: list of (batch_size, num_choice). each entry is torch.tensor(2, E)
            #edge_type_orig: list of (batch_size, num_choice). each entry is torch.tensor(E, )


    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        #edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        #edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        return edge_index, edge_type




class LM_QAGNN_DataLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, use_cache=True):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse

        model_type = MODEL_NAME_TO_CLASS[model_name]
        print ('train_statement_path', train_statement_path)
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)

        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        print ('num_choice', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num, num_choice, args)

        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num, num_choice, args)
        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)

        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path, max_node_num, num_choice, args)
            assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)


        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'train', self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def train_eval(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data)

    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_data=self.test_adj_data)





###############################################################################
############################### GNN architecture ##############################
###############################################################################

from torch.autograd import Variable
def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target



from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import glorot, zeros



class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """
    def __init__(self, args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype; self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        #For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2*emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        #For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))


    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, return_attention_weights=True):
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]

        #Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype +1) #[E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype +1).to(edge_vec.device)
        self_edge_vec[:,self.n_etype] = 1

        head_type = node_type[edge_index[0]] #[E,] #head=src
        tail_type = node_type[edge_index[1]] #[E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype) #[E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype) #[E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1) #[E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype) #[N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1) #[N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0) #[E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0) #[E+N, ?]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1)) #[E+N, emb_dim]

        #Add self loops to edge_index - why???
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1) #[2,1000]
        edge_index = torch.cat([edge_index, loop_index], dim=1)  #[2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings) #[N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index,alpha)
        else:
            return out


    def message(self, edge_index, x_i, x_j, edge_attr): #i: tgt, j:src
        # print ("edge_attr.size()", edge_attr.size()) #[E, emb_dim]
        # print ("x_j.size()", x_j.size()) #[E, emb_dim]
        # print ("x_i.size()", x_i.size()) #[E, emb_dim]
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key   = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]


        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2) #[E, heads]
        src_node_index = edge_index[0] #[E,]
        alpha = softmax(scores, src_node_index) #[E, heads] #group by src side node
        self._alpha = alpha

        #adjust by outgoing degree of src
        E = edge_index.size(1)            #n_edges
        N = int(src_node_index.max()) + 1 #n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index] #[E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1) #[E, heads]

        out = msg * alpha.view(-1, self.head_count, 1) #[E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  #[E, emb_dim]
