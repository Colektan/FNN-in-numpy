# plot the score and loss
import matplotlib.pyplot as plt

colors_set = {'Kraftime' : ('#E3E37D', '#968A62')}

def plot(runner, axes, set=colors_set['Kraftime']):
    train_color = set[0]
    dev_color = set[1]
    
    epochs = [i for i in range(len(runner.train_scores))]
    # 绘制训练损失变化曲线
    axes[0].plot(epochs, runner.train_loss, color=train_color, label="Train loss")
    # 绘制评价损失变化曲线
    axes[0].plot(epochs, runner.dev_loss[::2], color=dev_color, linestyle="--", label="Dev loss")
    # 绘制坐标轴和图例
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_title(f"{1e-2} , {1e-2}, {500}")
    axes[0].legend(loc='upper right')
    # 绘制训练准确率变化曲线
    axes[1].plot(epochs, runner.train_scores, color=train_color, label="Train accuracy")
    # 绘制评价准确率变化曲线
    axes[1].plot(epochs, runner.dev_scores[::2], color=dev_color, linestyle="--", label="Dev accuracy")
    # 绘制坐标轴和图例
    axes[1].set_ylabel("score")
    axes[1].set_xlabel("iteration")
    axes[1].set_title(f"{1e-2} , {1e-2}, {500}")
    axes[1].legend(loc='lower right')
    axes[1].set_ylim(0, 1)

if __name__ == '__main__':
    # import sys
    # import pickle
    # with open('temp_runner.pickle', 'rb') as f:
    #     runner = pickle.load(f)
    
    # plt.rcParams.update({'font.size' : 5})
    # _, axes = plt.subplots(5, 8)
    # _.set_figheight(50)
    # _.set_figwidth(10)
    # _.set_tight_layout(1)
    
    # axes = axes.reshape(-1)
    # for i in range(4):
    #     plot(runner, axes[i * 2 : i * 2 + 2])

    # _, axes = plt.subplots(5, 8)
    # _.set_figheight(50)
    # _.set_figwidth(10)
    # _.set_tight_layout(1)
    # axes = axes.reshape(-1)
    # for i in range(4):
    #     plot(runner, axes[i * 2 : i * 2 + 2])
    # plt.show()
    _, axes = plt.subplots(30, 20)
    _.set_tight_layout(1)
    plt.show()