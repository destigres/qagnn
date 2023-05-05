import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import time
import pickle

class GoalOrientedQuestionDataset(Dataset):
    def __init__(self,edges_file,q_id_file,qa_map_file):
        self.f = h5py.File(edges_file,'r')
        print(self.f.keys())
        with open(qa_map_file, 'rb') as file:
            question_dict = pickle.load(file)
        file.close()
        self.q_ids = np.load(q_id_file,allow_pickle=True)
        if len(self.q_ids.shape) > 1:
            self.q_ids = self.q_ids.flatten()
        self.q_dict = question_dict

        assert len(self.q_ids) == len(self.f["ground_truth_adj"]), "Number of questions between edges dataset file and question_ids file does not match"

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, idx):
        og_sub_graphs = torch.LongTensor(self.f["ground_truth_adj"][idx])
        node_emebds_after_message_pass = torch.FloatTensor(self.f["node_embeds"][idx])
        most_damaging_edges = torch.LongTensor(self.f["most_damaging_edges"][idx])
        least_damaging_edges = torch.LongTensor(self.f["least_damaging_edges"][idx])
        most_damaging_edge_types =  torch.LongTensor(self.f["most_damaging_edge_types"][idx])
        least_damaging_edge_types = torch.LongTensor(self.f["least_damaging_edge_types"][idx])
        og_node_embeds = torch.FloatTensor(self.f["og_node_embeds"][idx])
        node_embeds = torch.FloatTensor(self.f["node_embeds"][idx])
        concept_ids = torch.LongTensor(self.f["concept_ids"][idx])

        q_id = self.q_ids[idx]

        sample = {'og_graphs': og_sub_graphs, 'node_embeds_after_mp': node_emebds_after_message_pass, 'most_damage_edges': most_damaging_edges,
                  'least_damage_edges': least_damaging_edges, 'most_damage_edges': least_damaging_edges, 'most_damage_edge_types': most_damaging_edge_types, 'least_damage_edge_types': least_damaging_edge_types,
                  'q_id': q_id, 'question': self.q_dict[q_id]["question"], 'answer': self.q_dict[q_id]["answerKey"], 'choices': self.q_dict[q_id]["choices"],
                  'og_node_embeds': og_node_embeds,
                  'node_embeds': node_embeds,
                  'concept_ids': concept_ids
                  }
        return sample

if __name__ == '__main__':
    edges_file = '../our_dataset/version_1/csqa_dev.hdf5'
    q_id_file = '../our_dataset/version_1/question_ids.npy'
    qa_map_file = '../our_dataset/version_1/id_qa_map.pkl'
    dev_ds = GoalOrientedQuestionDataset(edges_file,q_id_file,qa_map_file)
    dev_dataloader = DataLoader(dev_ds, batch_size=5, num_workers=8, pin_memory=True,shuffle=True)
    torch.cuda.synchronize()
    start = time.time()
    for batch in dev_dataloader:
        continue
    end = time.time()
    print("name of process", end - start)