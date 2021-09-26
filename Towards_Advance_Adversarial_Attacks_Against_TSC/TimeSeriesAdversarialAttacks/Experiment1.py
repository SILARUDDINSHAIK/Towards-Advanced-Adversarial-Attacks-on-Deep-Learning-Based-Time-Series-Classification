# Import all the libraries
import torch
import pandas as pd
import seaborn as sns
import itertools
from collections import defaultdict
from matplotlib import pyplot as plt
from src.noises import *
from src.datasets import *
from src.smooth import *
from src.models import WideResNet

sns.set_context("notebook", rc={"lines.linewidth": 2})
sns.set_style("whitegrid")
sns.set_palette("husl")
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

# In[4]:


import pandas as pd
import numpy as np
from torchvision.transforms import Compose, ToTensor

Dataset = pd.read_csv('./results/dataset/GunPoint/GunPoint_TRAIN.ts', sep='\t', header=None)
target = Dataset[0]
del Dataset[0]
import torch
import torch.utils.data as data_utils

train = data_utils.TensorDataset(torch.Tensor(np.array(Dataset)), torch.Tensor(np.array(target)))
train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)
x, y = train[1]
import matplotlib.pyplot as plt

torch.Tensor.ndim = property(lambda x: len(x.size()))
N = 42
t = torch.rand(N)
plt.plot(t)
plt.show()
plt.close()
sigmas = [0.0, 0.15, 0.25, 0.5, 0.75]


def plot_image(x):
    plt.plot(x.numpy().transpose())


plt.figure(figsize=(20, 18))
for i, sigma in enumerate(sigmas):
    noise = Gaussian(sigma=sigma, dim=dim)
    sample = (noise.sample(x)).clamp(0, 1)
    plot_image(sample)
    plt.subplot(3, len(sigmas), i + 1)
    plt.axis("on")
    plt.title(f"Gaussian, $\\sigma={sigma:.2f}$")
    plt.savefig("Gaussian, $\\sigma={sigma:.2f}$.jpeg")

for i, sigma in enumerate(sigmas):
    noise = Laplace(sigma=sigma, dim=dim)
    sample = (noise.sample(x)).clamp(0, 1)
    plot_image(sample)
    plt.subplot(3, len(sigmas), i + 1 + len(sigmas))
    plt.axis("on")
    plt.title(f"Laplace, $\\sigma={sigma:.2f}$")

for i, sigma in enumerate(sigmas):
    noise = Uniform(sigma=sigma, dim=dim)
    sample = (noise.sample(x)).clamp(0, 1)
    plot_image(sample)
    plt.subplot(3, len(sigmas), i + 1 + 2 * len(sigmas))
    plt.axis("on")
    plt.title(f"Uniform, $\\sigma={sigma:.2f}$")
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
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = '/Users/silaruddin/Desktop/time-series-adversarials-master/results/UCR_ECG200/MLP/ckpt.best'
state_dict = torch.load(ckpt_path)
model = state_dict['model']
x, y = train[4]
print(x.shape)
plt.figure(figsize=(9, 3))
plt.plot(x)
# plt.axis("on")
x = x.unsqueeze(1).cpu()

# In[40]:


x.shape

# In[29]:


x

# In[20]:


ckpt_path = '/Users/silaruddin/Desktop/time-series-adversarials-master/results/UCR_ECG200/MLP/ckpt.best'

# In[21]:


state_dict = torch.load(ckpt_path)


# In[22]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        torch.nn.Module.dump_patches = True
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
    # model = torch.nn.DataParallel(model)


# cudnn.benchmark = True
model = Model()

# In[23]:


train = data_utils.TensorDataset(torch.Tensor(np.array(Dataset)), torch.Tensor(np.array(target)))
train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)

# In[24]:


state_dict = torch.load(ckpt_path)
# model.load_state_dict({k.replace('fc1.',''):v for k,v in torch.load(ckpt_path).items()})
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[11:]  # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict, strict=False)
model.eval()
# optimizer.load_state_dict(state_dict["optimizer"])
# best = state_dict["train_loss"]
# epoch = state_dict["epoch"]
# model1=state_dict["model"]


# In[30]:


output = model(x)

# In[ ]:


transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# In[ ]:


noise = Uniform(device="cpu", dim=3072, sigma=0.5)

# In[ ]:


preds = smooth_predict_hard(model, x, noise, 256)

# In[ ]:


top_cats = preds.probs.argmax(dim=1)
prob_lb = certify_prob_lb(model, x, top_cats, 0.001, noise, 100)
print(f"Probability lower bound: {prob_lb:.2f}")

# In[ ]:


radius = noise.certify(prob_lb, adv=1)
print(f"Certified Robust L1 Radius: {radius:.2f}")

# In[ ]:


radius = noise.certify(prob_lb, adv=2)
print(f"Certified Robust L2 Radius: {radius:.2f}")

# In[ ]:
