import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

attentions = []

for i in range(611):
    arr = np.load(os.path.join('batch_attentions', str(i) + '.npy'))
    print(arr.shape)
    attentions += np.mean(arr, axis=1).tolist()

plt.figure(figsize=(16, 9))
sns.distplot(attentions)
plt.savefig('attention_distribution.jpg')