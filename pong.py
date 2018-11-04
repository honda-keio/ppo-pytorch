import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from ppo_atari import PPO
from datetime import datetime, timedelta, timezone

if __name__ == "__main__":
    ENV = "PongNoFrameskip-v4"
    max_epochs = 200
    gamma = 0.99
    N = 64
    T = 1250
    batch_size = 64
    v_loss_coef = 0.5
    max_grad_norm = 0.1
    gpu = True
    epsilon = 0.3
    lr_ = 0.00001 
    JST = timezone(timedelta(hours=+9), "JST")
    print(datetime.now(JST))
    for i in range(1,5):
        lr = lr_ * i
        name = "-lr" + str(lr) + "-"
        ppo = PPO(ENV, max_epochs, N, T, batch_size, lr=lr, v_loss_coef=v_loss_coef, max_grad_norm=max_grad_norm, epsilon=epsilon, gpu=gpu)
        try:
            rs = ppo.run(name=name)
        except KeyboardInterrupt:
            pass
        #torch.save(ppo.model.to("cpu").state_dict(), "pong/ppo.pth")
        print(datetime.now(JST))
        plt.plot(range(len(rs)), rs)
        plt.savefig("pong/rs"+name[:-1]+".png")
        plt.close()