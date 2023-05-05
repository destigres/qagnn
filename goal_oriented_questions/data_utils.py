from Dataset import GoalOrientedQuestionDataset
from torch.utils.data import Dataset, DataLoader,random_split


def setup_datasets(args):
    train_ds = GoalOrientedQuestionDataset(args.edges_train, args.q_id_file_train, args.qa_map_file)
    test_ds = GoalOrientedQuestionDataset(args.edges_test, args.q_id_file_test, args.qa_map_file)

    test_dataloader = DataLoader(test_ds, batch_size=args.ebs, num_workers=8, pin_memory=True, shuffle=False)
    train_size = int(0.8 * len(train_ds))
    test_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(train_ds, [train_size, test_size])
    train_dataloader = DataLoader(train_ds, batch_size=args.tbs, num_workers=8, pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=args.ebs, num_workers=8, pin_memory=True, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

