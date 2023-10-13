import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder

torch.manual_seed(121)

loss_fn = nn.BCELoss()
optimizer = optim.Adam

model = nn.Sequential(
    nn.Linear(3072 + 38, 500),
    nn.ReLU(),
    nn.Linear(500, 200),
    nn.ReLU(),
    nn.Linear(200, 1),
    nn.Sigmoid()
)

# roc_auc_1 = AUC()  # You can define your own AUC function
# pr_auc_1 = AUC()  # You can define your own AUC function
# precision_1 = Precision()  # You can define your own Precision function
# recall_1 = Recall()  # You can define your own Recall function

checkpoint_path = "checkpoints_without_context_nodes_with_edge_types_corrected/cp-{epoch:04d}.ckpt"

latest = None  # You will need to set this to the path of the latest checkpoint if available

if latest is not None:
    initial_epoch = int(latest.split('/')[1][3:7])
else:
    initial_epoch = 0

# Create a callback that saves the model's weights every epoch
cp_callback = torch.save(model.state_dict(), checkpoint_path)

# Define the optimizer
optimizer = optim.Adam(model.parameters())

if latest is not None:
    model.load_state_dict(torch.load(latest))

if __name__ == '__main__':
    import dataloader

    edges_file = '/local2/data/our_dataset/version_1/csqa_dev.hdf5'
    q_id_file = '/local2/data/our_dataset/version_1/question_ids.npy'
    qa_map_file = '/local2/data/our_dataset/version_1/id_qa_map.pkl'
    emb_file = '/local2/data/our_dataset/version_1/tzw.ent.npy'
    dev_ds = dataloader.GoalOrientedQuestionDataset(edges_file, q_id_file, qa_map_file, emb_file)
    dev_dataloader = dataloader.DataLoader(dev_ds, batch_size=5, num_workers=8, pin_memory=True, shuffle=True)

    total = 0

    # Train the model
    gen_valid = dev_ds.batch_itr(batch_size=32, train=False)
    for epoch in range(10):

        gen_train = dev_ds.batch_itr(batch_size=32, train=True, train_samples_to_pick=1000000)
        
        losses = []
        model.train()
        for i, data in enumerate(gen_train):
            inputs, labels = data  # You will need to adjust this based on your data format
            inputs, labels = torch.Tensor(inputs), torch.Tensor(labels)
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            losses.append(loss.item())

            if i % 1000 == 0:
                print("Epoch [" + str(epoch) +  "] Iteration [" + str(i) + "] " + str(np.mean(losses)))
                losses = []
            optimizer.step()

        model.eval()
        for data in gen_valid:
            inputs, labels = data  # You will need to adjust this based on your data format
            inputs, labels = torch.Tensor(inputs), torch.Tensor(labels)
            # Calculate metrics here

    # You may need to adjust this part to calculate the metrics and perform validation
