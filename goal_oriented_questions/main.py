import os
import argparse
from data_utils import setup_datasets
from torch import nn
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from tqdm import tqdm

from torch.optim import Adam
from model import QuestionPredictor

from evaluate import evaluate_primary_task, evaluate_secondary_task

q_id_file_train, q_id_file_test = './data/q_ids_train.npy', './data/q_ids_dev.npy'
edges_train, edges_test = './data/csqa_train.hdf5', './data/csqa_dev.hdf5'
qa_map_file = './data/id_qa_map.pkl'

parser = argparse.ArgumentParser(description='goal-oriented-questions')

# Dataset Locations
parser.add_argument('--q-id-file-train', type=str, default=q_id_file_train, help='q-ids-train')
parser.add_argument('--q-id-file-test', type=str, default=q_id_file_test, help='q-ids-test')
parser.add_argument('--edges-train', type=str, default=edges_train, help='edges-train')
parser.add_argument('--edges-test', type=str, default=edges_test, help='edges-test')
parser.add_argument('--qa-map-file', type=str, default=qa_map_file, help='mapping of q_ids to question content')

# Batch Size Args
parser.add_argument('--ebs', type=int, default=32, help='evaluation batch size ')
parser.add_argument('--tbs', type=int, default=64, help='train batch size ')

# Training Args
parser.add_argument('--epochs', type=int, default=30, help='num epochs')
parser.add_argument('--lr', type=int, default=.0075, help='learning rate')

# Init TensorBoard Writer
writer = SummaryWriter()

criterion = nn.BCEWithLogitsLoss(reduction='mean')


def make_tensors(context_node_embed,node_embeds,most_imp_edges,least_imp_edges):
    Y_imp = torch.ones((len(most_imp_edges)))
    Y_non_imp = torch.zeros((len(most_imp_edges)))
    context_node_copied = context_node_embed.unsqueeze(1).repeat(1, most_imp_edges.shape[0]).T
    Y = torch.cat((Y_imp,Y_non_imp))
    most_imp_edges_node_emebds = node_embeds[most_imp_edges.flatten()].reshape(most_imp_edges.shape[0] , 200*2)
    least_imp_edges_node_emebds = node_embeds[least_imp_edges.flatten()].reshape(least_imp_edges.shape[0] , 200*2)
    most_imp_edges = torch.cat((most_imp_edges_node_emebds, context_node_copied), dim=-1)
    least_imp_edges = torch.cat((least_imp_edges_node_emebds, context_node_copied), dim=-1)
    X = torch.cat((most_imp_edges,least_imp_edges))

    indices = torch.randperm(X.size()[0])
    X = X[indices]
    Y = Y[indices]

    return X,Y

def train_epoch(model, opt, train_data, epoch, device, hyperparams):
    losses = []
    for batch in tqdm(train_data):
        important_edges,non_important_edges = batch['most_damage_edges'], batch['least_damage_edges']
        context_nodes = batch["node_embeds"][:,:,0,:]
        pad_idxs = torch.zeros(important_edges.shape[0],5)
        opt.zero_grad()
        loss = None
        for q in range(important_edges.shape[0]):
            for a in range(important_edges.shape[1]):
                first_pad_idx = np.where((important_edges.numpy()[q,a] == (-1,-1)).all(axis=-1) == True)[0][0]
                pad_idxs[q,a] = first_pad_idx
                # do forward pass here
                imp_edges = important_edges[q, a, :first_pad_idx, :]
                no_imp_edges = non_important_edges[q, a, :first_pad_idx, :]

                if len(imp_edges) == 0:
                    continue

                inputs, labels = make_tensors(context_nodes[q,a],batch["node_embeds"][q,a],imp_edges,no_imp_edges)
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                if not loss:
                    loss = criterion(preds.flatten(),labels)
                else:
                    loss += criterion(preds.flatten(),labels)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    return np.mean(losses)

def train(model: nn.Module, opt: torch.optim, train_data: DataLoader, val_data: DataLoader, device,
          hyperparams):
    best_val_loss = float("inf")
    epoch = 1
    while epoch <= hyperparams.epochs + 1:
        train_loss = train_epoch(model, opt, train_data, epoch, device, hyperparams)

        # evaluate on info gain link prediction
        val_primary_loss, val_primary_acc, important_edges = evaluate_primary_task(model,criterion,val_data,device)

        # evaluate on qa task using QAGNN Model
        # val_secondary_loss, val_secondary_acc = evaluate_secondary_task(important_edges,criterion,val_data,'validation')

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_primary_loss', val_primary_loss, epoch)
        writer.add_scalar('val_primary_acc', val_primary_acc, epoch)
        # writer.add_scalar('val_secondary_loss', val_secondary_loss, epoch)
        # writer.add_scalar('val_secondary_acc', val_secondary_acc, epoch)

        writer.flush()
        print("=" * 89)
        print('End of epoch {} | train loss {:5.5f} | val primary loss {:5.5f} | val primary acc {:5.2f} | val secondary loss {:5.2f} | '
              'val secondary acc {:5.2f} | '.format(epoch, train_loss, val_primary_loss, val_primary_acc, val_primary_acc,
                                                    val_primary_loss))
        print("=" * 89)
        epoch += 1

        if val_primary_loss <= best_val_loss:
            best_val_loss = val_primary_loss
            filename = datetime.now().strftime(f'./weights/%d-%m-%y-%H_%M.pth')
            torch.save(model.state_dict(), filename)


if __name__ == '__main__':
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader = setup_datasets(args)

    input_dim = 600
    model = QuestionPredictor(input_dim)
    model = model.to("cuda:0")
    optimizer = Adam(model.parameters(),lr=args.lr)

    print("=" * 89)
    print("starting training.... ")
    train(model,optimizer,train_dataloader,val_dataloader,torch.device("cuda:0"),args)
    print("ending training.... ")
    print("=" * 89)
    print("starting eval on test ")

    test_primary_loss, test_primary_acc, important_edges  = evaluate_primary_task(model,criterion,test_dataloader)
    # evaluate on qa task using QAGNN Model
    # test_secondary_loss, test_secondary_acc = evaluate_secondary_task(test_pred_edges,criterion,test_dataloader,'test')

    # print('Test primary loss {:5.2f} | Test primary acc {:5.2f} | Test secondary loss {:5.2f} | '
    #       'Test secondary acc {:5.2f} | '.format(test_primary_loss, test_primary_acc, test_secondary_loss,
    #                                             test_secondary_acc))

    writer.add_scalar('test_primary_loss', test_primary_loss)
    writer.add_scalar('test_primary_acc', test_primary_acc)
    # writer.add_scalar('test_secondary_loss', test_secondary_loss)
    # writer.add_scalar('test_secondary_acc', test_secondary_acc)
    print("=" * 89)



