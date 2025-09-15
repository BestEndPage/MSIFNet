import matplotlib.pyplot as plt

def plot_loss_curve(losses, save_path):
    for key in losses.keys():
        plt.plot(range(len(losses[key])), losses[key], label=key)
    plt.xlabel('iteration')
    plt.title(f'loss curve')
    plt.legend()
    plt.savefig(save_path)
    plt.clf()