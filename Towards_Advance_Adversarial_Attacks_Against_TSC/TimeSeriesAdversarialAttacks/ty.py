import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

def smooth_predict_hard(model, x, noise, sample_size=64, noise_batch_size=512):
    """
    Make hard predictions for a model smoothed by noise.

    Returns
    -------
    predictions: Categorical, probabilities for each class returned by hard smoothed classifier
    """
    counts = None
    num_samples_left = sample_size

    while num_samples_left > 0:

        shape = torch.Size([x.shape[0], min(num_samples_left, noise_batch_size)]) + x.shape[1:]
        samples = x.unsqueeze(1).expand(shape)
        samples = samples.reshape(torch.Size([-1]) + samples.shape[2:])
        samples = noise.sample(samples.view(len(samples), -1)).view(samples.shape)
        logits = model.forward(samples).view(shape[:2] + torch.Size([-1]))
        top_cats = torch.argmax(logits, dim=2)
        if counts is None:
            counts = torch.zeros(x.shape[0], logits.shape[-1], dtype=torch.float, device=x.device)
        counts += F.one_hot(top_cats, logits.shape[-1]).float().sum(dim=1)
        num_samples_left -= noise_batch_size

    return Categorical(probs=counts)
