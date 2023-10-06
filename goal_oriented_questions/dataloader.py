import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import time
import pickle
import random

class GoalOrientedQuestionDataset(Dataset):
    def __init__(self, edges_file, q_id_file, qa_map_file, emb_file, train_test_split=0.9):
        self.f = h5py.File(edges_file,'r')
        with open(qa_map_file, 'rb') as file:
            question_dict = pickle.load(file)
        file.close()
        self.q_ids = np.load(q_id_file, allow_pickle=True)
        self.q_dict = question_dict
        self.emb = np.load(emb_file, allow_pickle=True)
        self.train_test_split = train_test_split

        self.total_edge_types = max([np.max(self.f['most_damaging_edge_types'][i]) for i in range(self.q_ids.shape[0])]) + 1

        self.train_edges = {}

        assert len(self.q_ids) == len(self.f["ground_truth_adj"]), "Number of questions between edges dataset file and question_ids file does not match"

    def __len__(self):
        """Returns the length of the dataset (or the total number of questions).

        Returns:
            int: The length of the dataset (or the total number of questions).
        """
        return len(self.q_ids)

    def __getitem__(self, idx):
        """Gets the question at a specific index `idx`. 

        Args:
            idx (int): Index of the item to be fetched.

        Returns:
            dict: The question's attributes in dictionary format.
        """
        og_sub_graphs =  np.array(self.f["ground_truth_adj"][idx])
        most_damaging_edges = np.array(self.f["most_damaging_edges"][idx])
        least_damaging_edges = np.array(self.f["least_damaging_edges"][idx])
        most_damaging_edge_types = np.array(self.f["most_damaging_edge_types"][idx])
        least_damaging_edge_types = np.array(self.f["least_damaging_edge_types"][idx])
        pooled_representations = np.array(self.f["pooled_representations"][idx])
        q_id = self.q_ids[idx]

        sample = {'og_graphs': og_sub_graphs, 'pooled_representations': pooled_representations, 'most_damage_edges': most_damaging_edges,
                  'least_damage_edges': least_damaging_edges, 'most_damage_edge_types': most_damaging_edge_types, 'least_damage_edge_types': least_damaging_edge_types,
                  'q_id': q_id, 'question': self.q_dict[q_id]["question"], 'answer': self.q_dict[q_id]["answerKey"], 'choices': self.q_dict[q_id]["choices"]
                  }
        return sample
    
    def link_prediction_iterator(self, train=True, train_samples_to_pick=-1):
        """Generator that iterates over the dataset for link prediction objective. 
        Each edge is either labeled 0 or 1. The input features include:

        1. The pooled representation of the Q/A pair subgraph
        2. The embeddings of both edges
        3. One hot representation of the relation in the edge

        Args:
            train (bool, optional): Whether we want to run in train mode.
            Defaults to True.
            train_samples_to_pick (int, optional): If training is enabled, how
            many train samples on each epoch. Defaults to -1.

        Yields:
            _type_: _description_
        """
        # The edge index starts with 0
        edge_index = 0
        total_questions = len(self.f["ground_truth_adj"])

        split_index = int(self.train_test_split * total_questions)

        if train:
            question_indices = [i for i in range(split_index)]
        else:
            question_indices = [i for i in range(split_index, total_questions)]
        
        # If we haven't tracked the count of valid train edges yet
        if train and self.train_test_split not in self.train_edges:

            count_valid_edges = 0

            for i in question_indices:
                question = self[i]

                for answer_index in range(5):
                    
                    # Iterate over most damaging edges
                    for edge in question['most_damage_edges'][answer_index]:
                        if edge[0] != -1 and edge[0] != 0 and edge[1] != 0:
                            count_valid_edges += 1

                    # Iterate over least damaging edges
                    for edge in question['least_damage_edges'][answer_index]:
                        if edge[0] != -1 and edge[0] != 0 and edge[1] != 0:
                            count_valid_edges += 1
            
            self.train_edges[self.train_test_split] = count_valid_edges
        
        if train and train_samples_to_pick > -1:
            indices = set(random.sample(range(1, count_valid_edges + 1), k=train_samples_to_pick))
        else:
            indices = None

        count_valid_edges = 0

        for i in question_indices:
            question = self[i]

            for answer_index in range(5):
                pooled_representation = question['pooled_representations'][answer_index]
                
                # Iterate over most damaging edges
                for edge, edge_type in zip(question['most_damage_edges'][answer_index], question['most_damage_edge_types'][answer_index]):
                    if edge[0] != -1 and edge[0] != 0 and edge[1] != 0:
                        count_valid_edges += 1
                        if indices == None or count_valid_edges in indices:
                            edge = self.emb[edge - 1]
                            edge = np.reshape(edge, (-1,))
                            one_hot_r = np.zeros((self.total_edge_types,))
                            one_hot_r[edge_type] = 1

                            inputs = np.concatenate([edge, pooled_representation, one_hot_r], axis=0)

                            yield inputs, 1


                # Iterate over least damaging edges
                for edge, edge_type in zip(question['least_damage_edges'][answer_index], question['least_damage_edge_types'][answer_index]):
                    if edge[0] != -1 and edge[0] != 0 and edge[1] != 0:
                        count_valid_edges += 1
                        if indices == None or count_valid_edges in indices:
                            edge = self.emb[edge - 1]
                            edge = np.reshape(edge, (-1,))
                            one_hot_r = np.zeros((self.total_edge_types,))
                            one_hot_r[edge_type] = 1

                            inputs = np.concatenate([edge, pooled_representation, one_hot_r], axis=0)

                            yield inputs, 0
                

if __name__ == '__main__':
    edges_file = '../our_dataset/version_1/csqa_dev.hdf5'
    q_id_file = '../our_dataset/version_1/question_ids.npy'
    qa_map_file = '../our_dataset/version_1/id_qa_map.pkl'
    emb_file = '../our_dataset/version_1/tzw.ent.npy'
    dev_ds = GoalOrientedQuestionDataset(edges_file, q_id_file, qa_map_file, emb_file)
    dev_dataloader = DataLoader(dev_ds, batch_size=5, num_workers=8, pin_memory=True,shuffle=True)
    torch.cuda.synchronize()
    
    total = 0
    for edge, edge_type in dev_ds.link_prediction_iterator(train=True, train_samples_to_pick=1000000):
        total += 1

    print("Total", total)
    
    start = time.time()
    for batch in dev_dataloader:
        continue
    end = time.time()
    print("name of process", end - start)