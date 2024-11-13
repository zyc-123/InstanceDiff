import numpy as np
import torch
import math
from tqdm import tqdm

from .BaseSDE import BaseSDE

class driftSDE(BaseSDE):
    def __init__(self,
                 T,
                 drift_net,
                 noise_net,
                 max_sigma,
                 noise_schedule='cosine',
                 drift_schedule='cosine',
                 opt=None,
                 ifTrain=True) -> None:
        """Initialization
        Args:
            T(int): steps number
            drift_net(class): network for predicting the drift at each time step
            noise_net(class): network for predicting the noise at each time step
            noise_schedule(str): noise level schedule type: linear, cosine, or sigmoid
            drift_schedule(str): trajectory type of drift: linear, cosine, or sigmoid
        Returns:
            None
        
        """
        self.T = T
        self.dt = 1.0 / T
        self.max_sigma = max_sigma
        self.drift_net = drift_net
        self.noise_net = noise_net
        self.noise_schedule = self.get_schedule(noise_schedule, sde_opt=opt).cuda()
        self.drift_schedule = self.get_schedule(drift_schedule, sde_opt=opt).cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_module = None
        self.ifTrain = ifTrain

    def set_gpu(self, device):
        self.noise_schedule = self.noise_schedule.to(device)
        self.drift_schedule = self.drift_schedule.to(device)
        self.device = device

    def forward_diffusion(self, x0, xT, device='cuda'):
        """Perform forward drift and diffusion
        Args:
            x0(torch.Tensor): clean images; shape==(B,1,H,W)
            xT(torch.Tensor): fully drifted images; shape==(B,1,H,W)
        Return:
            drifted and noised images: xt = x0 + drift_level*drift + noise_level*noise
        """
        B, C, H, W = x0.shape

        time_idx = torch.randint(1, self.T+1, (B, 1, 1, 1)).long().to(x0.device)

        drift = (xT - x0) * self.drift_schedule[time_idx]
        drifted_x = x0 + drift
        
        std_noise = torch.randn(x0.shape).to(x0.device)
        noise = self.max_sigma * std_noise * torch.sqrt(self.noise_schedule[time_idx])
        drift_noised_x = drifted_x + noise

        return time_idx, drift_noised_x, drift, std_noise, noise

    def get_noise(self, xt, x_cond, t, names, text_encoder, image_context=None):
        text_module = self.noise_net.module.text_module if self.ifTrain else self.noise_net.text_module
        if text_module == "scoremap":
            noise, noise_scoremap = self.noise_net(xt, x_cond, t, names, text_encoder, image_context=image_context)
        else:
            noise = self.noise_net(xt, x_cond, t, names, text_encoder, image_context=image_context)
        return noise

    def get_drift(self, xt, x_cond, t, names, text_encoder, image_context=None):
        text_module = self.drift_net.module.text_module if self.ifTrain else self.drift_net.text_module
        if text_module == "scoremap":
            drift, drift_scoremap = self.drift_net(xt, x_cond, t, names, text_encoder, image_context=image_context)
        else:
            drift = self.drift_net(xt, x_cond, t, names, text_encoder, image_context=image_context)
        return drift
    
    def get_sigma(self, t):
        var = (self.noise_schedule[t]-self.noise_schedule[t-1]) * self.noise_schedule[t-1] / self.noise_schedule[t]
        return self.max_sigma * torch.sqrt(var)

    def reverse_ddpm(self, input, names, text_encoder, reverse_type='scaled', optimize_type="perdict_noise", image_context=None):
        return self.reverse_inputRes(input, names, text_encoder, image_context=image_context)

    def numpy2png(self, img, preffix, t, dir):
        from PIL import Image
        numpy_img = img.repeat((1, 3, 1, 1)).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        numpy_img = numpy_img / 2.0 + 0.5
        numpy_img = (255 * np.clip(numpy_img, 0, 1)).astype(np.uint8)
        png = Image.fromarray(numpy_img)
        png.save(dir + rf"\{preffix}_{t}"+".png")

    def reverse_inputRes(self, input, names, text_encoder, image_context=None):
        x = input + torch.randn(input.shape).to(input.device) * torch.sqrt(self.noise_schedule[self.T]) * self.max_sigma
        for t in tqdm(reversed(range(1, self.T + 1))):
            drift = self.get_drift(x-input, input, t, names, text_encoder, image_context=image_context)
            noise = self.get_noise(x-input, x, t, names, text_encoder, image_context=image_context)
            x = x - drift * (self.drift_schedule[t] - self.drift_schedule[t-1])

            # ODE
            # x = x - noise * (self.noise_schedule[t] - self.noise_schedule[t - 1])

            # SDE
            x = x - noise * torch.sqrt(self.noise_schedule[t]) * self.max_sigma
            x = x + torch.randn(input.shape).to(x.device) * self.max_sigma * torch.sqrt(self.noise_schedule[t-1])
        return x



def create_driftSDE(net, opt):
    """ Creator of drifSDE
    Args:
        net(OrderDict): keys are network name, values are networks objects;
        opt(OrderDict): settings of SDE; 'T', 'drift_schedule', 'noise_schedule'
    Returns:
        An object of driftSDE
    """
    T = opt['T']
    drift_net = net['drift_net']
    noise_net = net['noise_net']
    noise_schedule=opt['drift_schedule']
    drift_schedule=opt['noise_schedule']
    max_sigma = opt['max_sigma']
    ifTrain = True
    if "ifTrain" in opt:
        ifTrain = opt['ifTrain']
    return driftSDE(T,
                    drift_net,
                    noise_net,
                    max_sigma,
                    noise_schedule=noise_schedule,
                    drift_schedule=drift_schedule,
                    opt=opt,
                    ifTrain=ifTrain)


