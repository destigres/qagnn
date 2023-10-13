from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

tf.random.set_seed(121)

loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(500, activation='relu', input_shape=(3072 + 38,)),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

roc_auc_1 = tf.keras.metrics.AUC(name="roc_auc", curve="ROC")

pr_auc_1 = tf.keras.metrics.AUC(name="pr_auc", curve="PR")

precision_1 = tf.keras.metrics.Precision(name="precision")

recall_1 = tf.keras.metrics.Recall(name="recall")

checkpoint_path = "checkpoints_without_context_nodes_with_edge_types_corrected/cp-{epoch:04d}.ckpt"

latest = tf.train.latest_checkpoint("checkpoints_without_context_nodes_with_edge_types_corrected")

if latest != None:
  initial_epoch = int(latest.split('/')[1][3:7])
else:
  initial_epoch = 0

# Create a callback that saves the model's weights every epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch')

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy',
                                                          roc_auc_1,
                                                          pr_auc_1,
                                                          precision_1,
                                                          recall_1])
if latest != None:
  model.load_weights(latest)

if __name__ == '__main__':
    import dataloader

    edges_file = '/local2/data/our_dataset/version_1/csqa_dev.hdf5'
    q_id_file = '/local2/data/our_dataset/version_1/question_ids.npy'
    qa_map_file = '/local2/data/our_dataset/version_1/id_qa_map.pkl'
    emb_file = '/local2/data/our_dataset/version_1/tzw.ent.npy'
    dev_ds = dataloader.GoalOrientedQuestionDataset(edges_file, q_id_file, qa_map_file, emb_file)
    dev_dataloader = dataloader.DataLoader(dev_ds, batch_size=5, num_workers=8, pin_memory=True,shuffle=True)
    
    total = 0

    # Train the model
    gen_train = dev_ds.batch_itr(batch_size=32, train=True, train_samples_to_pick=1000000)
    gen_valid = dev_ds.batch_itr(batch_size=32, train=False)
    history = model.fit(gen_train, epochs=10, batch_size=128, validation_data=gen_valid,
                        callbacks=[cp_callback], initial_epoch=initial_epoch)
    # for edge, edge_type in dev_ds.link_prediction_iterator(train=True, train_samples_to_pick=1000000):
    #     total += 1