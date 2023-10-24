import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import click
import dataloader
import os
from pathlib import PurePath
from models.feedforward_model import *
import datetime
import argparse

# import qagnn parent folder
import sys

sys.path.append("../")
from qagnn import eval_detail_custom as qagnn_eval_detail_custom


@click.command()
@click.option("--edges_file", default="../data/our_dataset/version_1/csqa_dev.hdf5")
@click.option("--q_id_file", default="../data/our_dataset/version_1/question_ids.npy")
@click.option("--qa_map_file", default="../data/our_dataset/version_1/id_qa_map.pkl")
@click.option("--emb_file", default="../data/our_dataset/version_1/tzw.ent.npy")
@click.option("--model_name")
@click.option("--num_epochs", type=int)
@click.option("--batch_size", type=int)
@click.option("--train_samples_to_pick", default=1000000)
@click.option("--starting_checkpoint_path", default=None)
@click.option("--save_every_n_epochs", default=1)
@click.option("--validate_every_n_epochs", default=1)
@click.option(
    "--top_x_percent", default=0.04, help="percent of each answer subgraph to mask"
)
def main(
    edges_file,
    q_id_file,
    qa_map_file,
    emb_file,
    model_name,
    batch_size,
    num_epochs,
    train_samples_to_pick,
    starting_checkpoint_path,
    save_every_n_epochs,
    validate_every_n_epochs,
    top_x_percent,
):
    ### initial incomplete and complete evaluation
    torch.manual_seed(42)
    np.random.seed(42)

    # evaluate the model on the QAGNN task
    DATASET_NAME = "csqa"
    qagnn_inputs = {
        "mode": "eval_detail",
        "save_dir": "../saved_models/qagnn/",
        "save_model": False,
        "load_model_path": None,
        "num_relation": 38,
        "train_adj": f"../data/{DATASET_NAME}/graph/train.graph.adj.pk",
        "dev_adj": f"../data/{DATASET_NAME}/graph/dev.graph.adj.pk",
        "test_adj": f"../data/{DATASET_NAME}/graph/test.graph.adj.pk",
        "use_cache": True,
        "k": 5,
        "att_head_num": 2,
        "gnn_dim": 100,
        "fc_dim": 200,
        "fc_layer_num": 0,
        "freeze_ent_emb": True,
        "max_node_num": 200,
        "simple": False,
        "subsample": 1.0,
        "init_range": 0.02,
        "dropouti": 0.2,
        "dropoutg": 0.2,
        "dropoutf": 0.2,
        "decoder_lr": 1e-3,
        "mini_batch_size": 1,
        "eval_batch_size": 2,
        "unfreeze_epoch": 4,
        "refreeze_epoch": 10000,
        "fp16": False,
        "drop_partial_batch": False,
        "fill_partial_batch": False,
        "train_statements": f"../data/{DATASET_NAME}/statement/train.statement.jsonl",
        "dev_statements": f"../data/{DATASET_NAME}/statement/dev.statement.jsonl",
        "test_statements": f"../data/{DATASET_NAME}/statement/test.statement.jsonl",
        "load_model_path": "../saved_models/csqa_model_hf3.4.0.pt",
        "ent_emb_paths": ["../data/cpnet/tzw.ent.npy"],
        "cuda": True,
        "inhouse": True,
        "batch_size": 64,
        "inhouse_train_qids": f"../data/{DATASET_NAME}/inhouse_split_qids.txt",
    }
    # get the original (supporting) dataset

    qagnn_parser = argparse.ArgumentParser()
    [
        qagnn_parser.add_argument(f"--{k}", type=type(v), default=v)
        for k, v in qagnn_inputs.items()
    ]
    qagnn_args, _ = qagnn_parser.parse_known_args(qagnn_inputs)

    # qagnn_main(qagnn_args)
    # evaluate on the complete dataset
    qagnn_eval_detail_custom(
        qagnn_args, eval_mode="incomplete", top_x_percent=top_x_percent
    )
    qagnn_eval_detail_custom(
        qagnn_args, eval_mode="complete", top_x_percent=top_x_percent
    )

    ### model training

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam

    if model_name == "feedforward":
        # todo: set params as defauls in the model file
        model = FeedForwardModel()
    else:
        raise NotImplementedError

    # roc_auc_1 = AUC()  # You can define your own AUC function
    # pr_auc_1 = AUC()  # You can define your own AUC function
    # precision_1 = Precision()  # You can define your own Precision function
    # recall_1 = Recall()  # You can define your own Recall function

    # set up checkpointing
    now = datetime.datetime.now()
    checkpoint_dir = PurePath(
        "checkpoints", f"{model_name}", f"{now.strftime('%y-%m-%d_%H:%M:%S')}"
    )
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # starting_checkpoint_path = str(checkpoint_dir / starting_checkpoint_path)

    if starting_checkpoint_path is not None:
        latest = starting_checkpoint_path
        initial_epoch = int(latest.split("/")[1][3:7])
    else:
        initial_epoch = 0

    # Create a callback that saves the model's weights every epoch
    # cp_callback = torch.save(model.state_dict(), starting_checkpoint_path)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters())

    # if latest is not None:
    #     model.load_state_dict(torch.load(latest))
    dev_ds = dataloader.GoalOrientedQuestionDataset(
        edges_file, q_id_file, qa_map_file, emb_file
    )

    # Train the model
    gen_valid = dev_ds.batch_itr(batch_size=batch_size, train=False)
    for epoch in range(num_epochs):
        gen_train = dev_ds.batch_itr(
            batch_size=32, train=True, train_samples_to_pick=train_samples_to_pick
        )

        losses = []
        model.train()
        for i, data in enumerate(gen_train):
            (
                inputs,
                labels,
            ) = data  # You will need to adjust this based on your data format
            inputs, labels = torch.Tensor(inputs), torch.Tensor(labels)
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            losses.append(loss.item())

            if i % 1000 == 0:
                print(f"Epoch [{epoch}] Iteration [{i}] {np.mean(losses)}")
                losses = []
            optimizer.step()

        model.eval()
        for data in gen_valid:
            (
                inputs,
                labels,
            ) = data  # You will need to adjust this based on your data format
            inputs, labels = torch.Tensor(inputs), torch.Tensor(labels)
            # Calculate metrics here

    # You may need to adjust this part to calculate the metrics and perform validation


if __name__ == "__main__":
    main()
