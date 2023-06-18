import torch

from functools import partial
from torch.distributions.gamma import Gamma


def anneal_dsm_score_estimation(scorenet, x0, x, labels=None, loss_type='a', hook=None, cond=None, cond_mask=None, gamma=False, L1=False, all_frames=False):
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    version = getattr(net, 'version', 'SMLD').upper()
    net_type = getattr(net, 'type') if isinstance(getattr(net, 'type'), str) else 'v1'

    if all_frames:
        x = torch.cat([x, cond], dim=1)
        cond = None

    # z, perturbed_x
    if version == "SMLD":
        sigmas = net.sigmas
        if labels is None:
            labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)
        used_sigmas = sigmas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        z = torch.randn_like(x)
        perturbed_x = x + used_sigmas * z
    elif version == "DDPM" or version == "DDIM" or version == "FPNDM":
        alphas = net.alphas
        if labels is None:
            labels = torch.randint(0, len(alphas), (x.shape[0],), device=x.device)
        used_alphas = alphas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        if gamma:
            used_k = net.k_cum[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            used_theta = net.theta_t[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            z = Gamma(used_k, 1 / used_theta).sample()
            z = (z - used_k*used_theta) / (1 - used_alphas).sqrt()
        else:
            z = torch.randn_like(x)
        perturbed_x = used_alphas.sqrt() * x + (1 - used_alphas).sqrt() * z
    
    scorenet = partial(scorenet, x0=x0, cond=cond)
    
    # Loss
    if L1:
        def pow_(x):
            return x.abs()
    else:
        def pow_(x):
            return 1 / 2. * x.square()

    # print(labels.shape)
    # assert(0)
    loss = pow_((z - scorenet(perturbed_x, labels, cond_mask=cond_mask)).reshape(len(x), -1)).sum(dim=-1)

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)



def _anneal_dsm_score_estimation(scorenet, x0, x, labels=None, loss_type='a', hook=None, cond=None, cond_mask=None, gamma=False, L1=False, all_frames=False):
    P_mean,P_std,sigma_data=-1.2,1.2,0.5

    rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    n = torch.randn_like(x) * sigma

    # D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
    x = (x+n).to(torch.float32)
    sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
    c_noise = sigma.log() / 4

    F_x = scorenet((c_in * x), c_noise.flatten(), x0=x0, cond=cond, cond_mask=cond_mask)
    D_x = c_skip * x + c_out * F_x.to(torch.float32)


    loss = weight * ((D_x - x) ** 2)
    return loss.sum()