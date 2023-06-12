import context

from dataset.colmap_solution import ColmapSolution
from nerf_render.util import uniform_sample, uniform_sample_old
import torch
import matplotlib.pyplot as plt


def test_net_gen():

    test = ColmapSolution("/database/colmap_test", 0)

    test.unit_rescale()

    rays = test.calculate_rays()

    print(rays)


def test_uniform():

    tn = torch.tensor([5]).view(1, 1).expand(1024, 1).float()

    tf = torch.tensor([20]).expand(1024, 1).float()

    t = uniform_sample(tn, tf, 64)

    x = torch.tensor(list(range(64))).view(1, 64, 1).expand(1024, 64, 1).float()

    t = t.reshape(-1).numpy()
    x = x.reshape(-1).numpy()

    print("")

    plt.scatter(x, t)
    plt.show()

def test_uniform2():

    tn = torch.tensor([5]).view(1, 1).expand(1024, 1).float()

    tf = torch.tensor([20]).expand(1024, 1).float()

    t = uniform_sample_old(tn, tf, 64)

    x = torch.tensor(list(range(64))).view(1, 64, 1).expand(1024, 64, 1).float()

    t = t.reshape(-1).numpy()
    x = x.reshape(-1).numpy()

    print("")

    plt.scatter(x, t)
    plt.show()
