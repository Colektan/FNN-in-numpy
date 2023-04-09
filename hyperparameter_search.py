import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


with open('idx.pickle', 'rb') as f:
        idx = pickle.load(f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# grid search
init_lr = [0.02, 0.04, 0.06, 0.08]
lam = [1, 0.1, 0.01, 0.001, 0.0001]
size = [600, 400, 200]

color_set = ('#E3E37D', '#968A62')
train_color = color_set[0]
dev_color = color_set[1]

plt.rcParams.update({'font.size' : 5})

idx = 0
score = []
for s in size:
    fig, axes = plt.subplots(5, 8)
    fig.set_figheight(50)
    fig.set_figwidth(10)
    fig.set_tight_layout(1)
    axes = axes.reshape(-1)
    idx = 0
    for lr in init_lr:
            for l in lam:
                    linear_model = nn.models.Model_MLP([train_imgs.shape[-1], s, 10], 'ReLU', lambda_list=[l, l, l])
                    optimizer = nn.optimizer.SGD(init_lr=lr, model=linear_model)
                    scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=30)
                    loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

                    runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn)

                    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=1, log_iters=100, save_dir=r'./best_models')

                    epochs = [i for i in range(len(runner.train_scores))]
                    axes[idx * 2].plot(epochs, runner.train_loss, color=train_color, label="Train loss")
                    axes[idx * 2].plot(epochs, runner.dev_loss, color=dev_color, linestyle="--", label="Dev loss")
                    axes[idx * 2].set_ylabel("loss")
                    axes[idx * 2].set_xlabel("iteration")
                    axes[idx * 2].set_title(f"{lr}, {l}, {s}")
                    axes[idx * 2].legend(loc='upper right')

                    axes[idx * 2 + 1].plot(epochs, runner.train_scores, color=train_color, label="Train scores")
                    axes[idx * 2 + 1].plot(epochs, runner.dev_scores, color=dev_color, linestyle="--", label="Dev scores")
                    axes[idx * 2 + 1].set_ylabel("scores")
                    axes[idx * 2 + 1].set_xlabel("iteration")
                    axes[idx * 2 + 1].set_title(f"{lr}, {l}, {s}")
                    axes[idx * 2 + 1].legend(loc='lower right')
                    score.append(runner.best_score)
                    idx += 1
plt.show()

print(score)


