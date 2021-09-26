'''#from argparse import Namespace
###from data.data import load_dataset
from models import get_model
from config import get_arg_parser
###from src.noises.builtins import eval
#from train_utils import train_step, eval, adversarial_eval, save_ckpt, load_ckpt
###import torch.optim as optim
import torch.nn as nn
import numpy as np
import json
from attack_utils import get_attack_fn
import os
import torch.nn as nn
###import train_utils.checkpoints
import torch
import os
import torch.nn as nn
###from models import get_model
import torch
import pandas as pd
###import seaborn as sns
import itertools
###from collections import defaultdict
###from matplotlib import pyplot as plt
from src.noises import *
###from src.datasets import *
from src.smooth import *
from src.models import WideResNet
sns.set_context("notebook", rc={"lines.linewidth": 2})
sns.set_style("whitegrid")
sns.set_palette("husl")
from src.noises import *
# Noises:

eval(Gaussian)
dim = 2
for noise_str in ("Gaussian", "Laplace", "Uniform"):
    noise = eval(noise_str)(lambd=1.0, dim=dim)
    rvs = noise.sample(torch.zeros(10000, dim))
    l2_norms = rvs.norm(p=2, dim=1) / np.sqrt(dim)
    print(f"{noise_str}:\tempirical sigma estimate = {l2_norms.mean():.2f};\ttheoretical sigma = {noise.sigma:.2f}")

for noise_str in ("Gaussian", "Laplace", "Uniform"):
    noise = eval(noise_str)(sigma=1.0, dim=dim)

    rvs = noise.sample(torch.zeros(10000, dim))
    l2_norms = rvs.norm(p=2, dim=1).pow(2) / dim

    print(f"{noise_str}:\tempirical sigma estimate = {l2_norms.mean():.2f};\ttheoretical sigma = {noise.sigma:.2f}")

# Load the model
args = Namespace(adversarial_eval=True, attack_config='./configs/pgd_inf_0.1.json', base_dir='./results/',
                 batch_size=16, dataset='UCR_Wafer', device='cpu', distance_loss=False, eval_freq=5, max_epochs=50,
                 model='resnet', resume=True, test_batch_size=1024, workers=4)
train_loader, test_loader, train_dataset, test_dataset = load_dataset(args)
model = get_model(args)(args, train_loader)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
model, optimizer, best, epoch = train_utils.load_ckpt(args, model, optimizer)

print(train_dataset[1])

sigmas = [0.0, 0.15, 0.25, 0.5, 0.75]
def plot_image(x):
    plt.plot(x.numpy().transpose())


# Probability Lower Bound
axis = np.linspace(0.5, 1.0, 400, endpoint=False)

df = defaultdict(list)

for noise_str in ["Gaussian", "Laplace", "Uniform"]:
    noise = eval(noise_str)(sigma=1.0, dim=dim)
    radii = noise.certify(torch.tensor(axis), adv=1).numpy()
    df["radius"] += radii.tolist()
    df["axis"] += axis.tolist()
    df["noise"] += [noise_str] * len(axis)

df = pd.DataFrame(df)
plt.figure(figsize=(10, 4))
sns.lineplot(x="axis", y="radius", hue="noise", style="noise", data=df)
plt.xlabel("Probability lower bound, $\\hat\\rho$")
plt.ylabel("Certified $\ell_1$ radius")
plt.ylim((0, 3))
'''