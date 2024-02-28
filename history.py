import os
import numpy as np


def log_history(loss_history, metric_historys, epochs, opt, prefix=""):
    import matplotlib.pyplot as plt
    save_folder = "./images/history"
    os.makedirs(save_folder, exist_ok=True)

    loss_history = np.array(loss_history)

    epochs = np.array(epochs)
    plt.plot(epochs, loss_history)
    plt.savefig(os.path.join(save_folder, prefix+"loss_history_%s.png" % opt.exp_name))

    plt.clf()
    # epochs = np.unique(epochs // 10) * 10
    if metric_historys is not None:
        for i in range(len(metric_historys)):
            metric_history = np.array(metric_historys[i])
            if i == 1:
                metric_history *= 33
            plt.plot(epochs, metric_history)
        plt.legend(["PSNR", "SSIM"])
        plt.savefig(os.path.join(save_folder, prefix+"metric_history_%s.png" % opt.exp_name))
