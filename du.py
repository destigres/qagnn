# concept decoding utils
import numpy as np
import torch as pt
from transformers import RobertaTokenizer

tk = RobertaTokenizer.from_pretrained("roberta-base")

concepts_path = "data/cpnet/concept.txt"
# array mapping `concept_ids` to concept names
concepts = np.array([line.strip() for line in open(concepts_path, "r").readlines()])

relations_path = "data/cpnet/conceptnet.en.csv"
# array mapping `edge_types` to edge names
# extract each unique edge type from column 0 of relations_path
edge_types = np.array(
    [line.split(",")[0] for line in open(relations_path, "r").readlines()]
)


def decode_attention(concept_ids, edge_ids, edge_attentions, edge_types):
    b, n = concept_ids.shape
    node_pairs = concept_ids.flatten()[edge_ids]
    # drop all placeholder edges where both entities are 1
    mask = (~((node_pairs[0] == 1) & (node_pairs[1] == 1))).repeat(2, 1)
    # mask=(~((node_pairs[:,0]==1) & (node_pairs[:,1]==1))).repeat(2,1)
    node_pairs = node_pairs.masked_select(mask).view(2, -1)

    attn_values, attn_indices = (
        edge_attentions.sum(-1).masked_select(mask).sort(descending=True)
    )

    concepts[node_pairs]
