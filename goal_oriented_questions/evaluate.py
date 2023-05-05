import random

import torch
from tqdm import tqdm

try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
from torchmetrics.classification import F1Score
from modeling.modeling_qagnn import *
from utils.parser_utils import *
DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}

from collections import defaultdict, OrderedDict
import numpy as np


parser = get_parser("../data")
args, _ = parser.parse_known_args()
parser.add_argument('--mode', default='eval_detail', choices=['train', 'eval_detail'], help='run training or evaluation')
parser.add_argument('--save_dir', default=f'../saved_models/cqsa/', help='model output directory')
parser.add_argument('--save_model', dest='save_model', action='store_true')
parser.add_argument('--load_model_path', default="../saved_models/obqa/obqa_model_hf3.4.0.pt")


# data
parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
parser.add_argument('--train_adj', default=f'../data/{args.dataset}/graph/train.graph.adj.pk')
parser.add_argument('--dev_adj', default=f'../data/{args.dataset}/graph/dev.graph.adj.pk')
parser.add_argument('--test_adj', default=f'../data/{args.dataset}/graph/test.graph.adj.pk')
parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')

# model architecture
parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
parser.add_argument('--gnn_dim', default=100, type=int, help='dimension of the GNN layers')
parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')

parser.add_argument('--max_node_num', default=200, type=int)
parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
parser.add_argument('--subsample', default=1.0, type=float)
parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')

# regularization
parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

# optimization
parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
parser.add_argument('-ebs', '--eval_batch_size', default=1, type=int)
parser.add_argument('--unfreeze_epoch', default=4, type=int)
parser.add_argument('--refreeze_epoch', default=10000, type=int)
parser.add_argument('--fp16', default=False, type=bool_flag, help='use fp16 training. this requires torch>=1.6.0')
parser.add_argument('--drop_partial_batch', default=False, type=bool_flag, help='')
parser.add_argument('--fill_partial_batch', default=False, type=bool_flag, help='')

# generating dataset for Predicting LINK INFO GAIN
parser.add_argument('--mask_percent_links_per',default=.2, type=float, help='percent edges per top/bottom for each subgraph.')
parser.add_argument('--generate',default=False, type=bool_flag,help='generate dataset for link prediction info gain ')

parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
args = parser.parse_args()

if args.simple:
    parser.set_defaults(k=1)
args = parser.parse_args()
args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')

loss_func = nn.CrossEntropyLoss(reduction='mean')

model_path = args.load_model_path

args.ent_emb_paths = ["../data/cpnet/tzw.ent.npy"]

cp_emb = [np.load(path) for path in args.ent_emb_paths]
cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
print('| num_concepts: {} |'.format(concept_num))


model_state_dict, old_args = torch.load(model_path, map_location=torch.device('cpu'))

init_device = torch.device("cpu")

dataset = LM_QAGNN_DataLoader(args, args.train_statements, args.train_adj,
                              args.dev_statements, args.dev_adj,
                              args.test_statements, args.test_adj,
                              batch_size=1, eval_batch_size=args.eval_batch_size,
                              device=(init_device, init_device),
                              model_name=old_args.encoder,
                              max_node_num=old_args.max_node_num, max_seq_length=old_args.max_seq_len,
                              is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                              subsample=args.subsample, use_cache=args.use_cache)

def evaluate_accuracy(eval_set, model, pred_edges):
    n_samples, n_correct = 0, 0
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set):
            logits, _ = model(qids,*input_data)
            loss = loss_func(logits, labels)
            total_loss += loss
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return total_loss / n_samples, n_correct / n_samples



def evaluate_secondary_task(pred_edges,criterion,data,type):
    model = LM_QAGNN(old_args, old_args.encoder, k=old_args.k, n_ntype=4, n_etype=old_args.num_relation, n_concept=concept_num,
                               concept_dim=old_args.gnn_dim,
                               concept_in_dim=concept_dim,
                               n_attention_head=old_args.att_head_num, fc_dim=old_args.fc_dim, n_fc_layer=old_args.fc_layer_num,
                               p_emb=old_args.dropouti, p_gnn=old_args.dropoutg, p_fc=old_args.dropoutf,
                               pretrained_concept_emb=cp_emb, freeze_ent_emb=old_args.freeze_ent_emb,
                               init_range=old_args.init_range,
                               encoder_config={})

    model.load_state_dict(model_state_dict)

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:0")
    model.encoder.to(device0)
    model.decoder.to(device1)

    model.eval()

    statement_dic = {}
    for statement_path in (args.train_statements, args.dev_statements, args.test_statements):
        statement_dic.update(load_statement_dict(statement_path))

    use_contextualized = 'lm' in old_args.ent_emb
    if type == 'val':
        ds = dataset.train()
    else:
        ds = dataset.dev()

    ds.device0 = torch.device("cuda:0")
    ds.device1 = torch.device("cuda:0")

    return evaluate_accuracy(ds, model, pred_edges)


def make_tensors(context_node_embed,node_embeds,edges,kind='imp'):
    if kind == 'imp':
        Y = torch.ones((len(edges)))
    else:
        Y = torch.zeros((len(edges)))
    context_node_copied = context_node_embed.unsqueeze(1).repeat(1, edges.shape[0]).T
    edges_node_emebds = node_embeds[edges.flatten()].reshape(edges.shape[0] , 200*2)
    X = torch.cat((edges_node_emebds, context_node_copied), dim=-1)
    return X,Y

f1 = F1Score(task="binary",threshold=0.5)
def evaluate_primary_task(model,criterion,data,device):
    model.eval()
    losses = []
    f1s = []
    with torch.no_grad():
        for batch in tqdm(data):
            important_edges, non_important_edges = batch['most_damage_edges'], batch['least_damage_edges']
            context_nodes = batch["node_embeds"][:, :, 0, :]
            pad_idxs = torch.zeros(important_edges.shape[0], 5)
            for q in range(important_edges.shape[0]):
                a_edge_outputs = []
                labels = []
                for a in range(important_edges.shape[1]):
                    first_pad_idx = np.where((important_edges.numpy()[q, a] == (-1, -1)).all(axis=-1) == True)[0][0]
                    pad_idxs[q, a] = first_pad_idx
                    # do forward pass here
                    imp_edges = important_edges[q, a, :first_pad_idx, :]
                    no_imp_edges = non_important_edges[q, a, :first_pad_idx, :]

                    if len(imp_edges) == 0:
                        continue

                    inputs_imp, labels_imp = make_tensors(context_nodes[q, a], batch["node_embeds"][q, a], imp_edges,'imp')
                    inputs_non_imp, labels_non_imp = make_tensors(context_nodes[q, a], batch["node_embeds"][q, a], no_imp_edges,'non_imp')

                    inputs_imp = inputs_imp.to(device)
                    labels_imp = labels_imp.to(device)

                    output_imp = model(inputs_imp)
                    loss = criterion(output_imp.flatten(), labels_imp)

                    inputs_non_imp = inputs_non_imp.to(device)
                    labels_non_imp = labels_non_imp.to(device)

                    output_non_imp = model(inputs_non_imp)
                    loss += criterion(output_non_imp.flatten(), labels_non_imp)

                    a_edge_outputs.append(torch.sigmoid(output_imp.flatten().cpu()))
                    a_edge_outputs.append(torch.sigmoid(output_non_imp.flatten().cpu()))
                    labels.append(torch.ones(output_non_imp.shape[0]))
                    labels.append(torch.zeros((output_non_imp.shape[0])))

                    losses.append(loss.item())

                if len(a_edge_outputs) > 0:
                    f = f1(torch.cat(a_edge_outputs),torch.cat(labels).int())
                    f1s.append(f.item())
        return np.mean(losses),np.mean(f1s),important_edges

