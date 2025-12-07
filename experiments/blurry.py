from typing import Optional

import torch
import torchvision
from torch._tensor import Tensor
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torchcfm.models.unet import UNetModel

from dualwcfm.distributions import CouplingDistribution, Distribution
from dualwcfm.distributions.coupling import MarginalDistribution
from dualwcfm.trainers import IterativeTrainer
from dualwcfm.models import ImageDualPotential, ResNetDualPotential, EfficientDualPotential

class ImageSampler:
    """Helper to turn a DataLoader into an infinite sampler."""

    def __init__(self, dataset, batch_size, device):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.iterator = iter(self.loader)
        self.device = device

    def get_batch(self, n):
        try:
            batch, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch, _ = next(self.iterator)

        return batch[:n].to(self.device)


class BlurCoupling(CouplingDistribution):

    def __init__(self, dataset, batch_size=32, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cpu")
        self.sampler = ImageSampler(dataset, batch_size, self.device)
        self.blur_op = T.GaussianBlur(kernel_size=(5, 9), sigma=(2.0, 2.0))

    def coupling_sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.sampler.get_batch(n)
        y = self.blur_op(x)
        return y, x

    def sample(self, n: int) -> Tensor:
        raise NotImplementedError

    def sample_target(self, n: int) -> Tensor:
        return self.sampler.get_batch(n)

    def sample_init(self, n: int) -> Tensor:
        return self.blur_op(self.sampler.get_batch(n))


class IndependentMarginal(Distribution):

    def __init__(self, dataset, is_blurred=False, batch_size=32, device=None):
        self.device = device if device is not None else torch.device("cpu")
        self.sampler = ImageSampler(dataset, batch_size, self.device)
        self.is_blurred = is_blurred
        self.blur_op = T.GaussianBlur(kernel_size=(5, 9), sigma=(2.0, 2.0))

    def sample(self, n: int) -> torch.Tensor:
        data = self.sampler.get_batch(n)
        if self.is_blurred:
            return self.blur_op(data)
        return data


def get_blur_experiment_distributions(batch_size=32):
    tf = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    ds_mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    ds_fashion = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)

    # Pair 1: Coupled (mu, nu) -> Sharp Digits, Blurred Digits
    eval_coupling_dist = BlurCoupling(ds_mnist, batch_size=batch_size)

    # Pair 2: Independent (mu', nu') -> Sharp Clothing, Blurred Clothing + Coupling for evaluation
    coupling_dist = BlurCoupling(ds_fashion, batch_size=batch_size)
    mu_prime = IndependentMarginal(ds_mnist, is_blurred=False, batch_size=batch_size)
    nu_prime = IndependentMarginal(ds_mnist, is_blurred=True, batch_size=batch_size)

    return coupling_dist, eval_coupling_dist, mu_prime, nu_prime



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
coupling_dist, eval_coupling_dist, target_dist, source_dist = get_blur_experiment_distributions(batch_size)

def visualize_experiment():
    batch_size = 8

    y_c, x_c = coupling_dist.coupling_sample(batch_size)

    # x_target = target_dist.sample(batch_size)
    # y_target = source_dist.sample(batch_size)
    y_target, x_target = eval_coupling_dist.coupling_sample(batch_size)

    print(x_c.shape, y_c.shape, x_target.shape, y_target.shape)

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    for i in range(8):
        axes[0, i].imshow(x_c[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5, cmap="gray")
        axes[0, i].axis("off")
    for i in range(8):
        axes[1, i].imshow(y_c[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5, cmap="gray")
        axes[1, i].axis("off")
    for i in range(8):
        axes[2, i].imshow(x_target[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5, cmap="gray")
        axes[2, i].axis("off")
    for i in range(8):
        axes[3, i].imshow(y_target[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5, cmap="gray")
        axes[3, i].axis("off")
    plt.show()

visualize_experiment()




def plot_images(cfm, alpha, coupling_dist, eval_coupling_dist, new_source_dist, new_target_dist, device):
    batch_size = 8

    # new_source_samples = new_source_dist.sample(batch_size).to(device)
    new_source_samples, new_target_samples = eval_coupling_dist.coupling_sample(batch_size)
    new_source_samples = new_source_samples.to(device)
    new_target_samples = new_target_samples.to(device)

    trajectories = cfm.get_trajectory(x0=new_source_samples, n_timesteps=100)

    fig, axes = plt.subplots(3, 8, figsize=(12, 6))
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    for i in range(8):
        axes[0, i].imshow(new_source_samples[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5, cmap="gray")
        axes[0, i].axis("off")
    for i in range(8):
        axes[1, i].imshow(trajectories[-1][i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5, cmap="gray")
        axes[1, i].axis("off")
    for i in range(8):
        axes[2, i].imshow(new_target_samples[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5, cmap="gray")
        axes[2, i].axis("off")

    plt.show()

class UNet(UNetModel):
    def forward(self, x):
        return super().forward(torch.zeros(x.shape[0], device=x.device), x)


f_net = EfficientDualPotential(in_channels=1, base_channels=32).to(device)
g_net = EfficientDualPotential(in_channels=1, base_channels=32).to(device)
flow_model = UNetModel(dim=(1, 32, 32), channel_mult=(1, 2, 3, 4), num_channels=32, num_res_blocks=1).to(device)


iterative_trainer = IterativeTrainer(
    coupling_dist=coupling_dist,
    source_dist=MarginalDistribution(coupling_dist.sample_init),
    target_dist=MarginalDistribution(coupling_dist.sample_target),
    eval_coupling_dist=eval_coupling_dist
)
info = iterative_trainer.train(
    n_iterations=10,
    f_net=f_net,
    g_net=g_net,
    flow_network=flow_model,
    batch_size=64,
    epochs=500,
    learning_rate=1e-4,
    sigma=0.2,
    lambda_reg=0.01,
    return_cfm=False,
    plot_progress=True,
    torch_wrapper=lambda net: net,
    plot_func=plot_images,
    print_every=10,
    eval_every=10,
    device=device,
)