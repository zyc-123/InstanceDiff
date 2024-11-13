import numpy as np
import torch
import math
from tqdm import tqdm

class BaseSDE():
    def __init__(self) -> None:
        pass

    def linear_beta_schedule(self, steps_num):
        """
        linear schedule
        """
        jump = 1000.0 / (steps_num + 1)
        beta_0 = 0.0001 * jump
        beta_T = 0.02 * jump
        return torch.linspace(beta_0, beta_T, steps_num + 1, dtype=torch.float32)

    def cos_beta_schedule(self, steps_num, eps=0.001):
        t = torch.linspace(0, 1, steps_num+1, dtype=torch.float32)
        f_t = torch.cos(0.5 * math.pi * (t + eps) / (1 + eps)) ** 2
        alpha_bar = f_t / f_t[0]
        betas = np.clip((1 - alpha_bar), 0, 0.999999)
        return betas
    
    def get_alpha_schedule(self, steps_num, eps=0.001):
        t = torch.linspace(0, 1, steps_num+1, dtype=torch.float32)
        f_t = torch.cos(0.5 * math.pi * (t + eps) / (1 + eps)) ** 2
        alpha_bar = f_t / f_t[0]
        return alpha_bar
    
    def get_alpha_schedule(self, steps_num, eps=0.001):
        t = torch.linspace(0, 1, steps_num+1, dtype=torch.float32)
        f_t = torch.cos(0.5 * math.pi * (t + eps) / (1 + eps)) ** 2
        alpha_bar = f_t / f_t[0]
        return alpha_bar

    def sigmoid_schedule(self, steps_num, eps=1e-9):
        thetas = 10 * torch.linspace(0, 1, steps_num+1, dtype=torch.float32) - 5
        betas = (1 - torch.sigmoid(thetas)) * 0.5
        betas = (betas - betas[0]) / (betas[-1] - betas[0])
        betas = torch.clip(betas, eps, 1.0)
        return betas

    def double_cos_beta_schedule(self, steps_num, eps=0.001):
        betas_x = torch.linspace(eps, math.pi, steps_num+1, dtype=torch.float32)
        betas = (1 - torch.cos(betas_x)) * 0.5
        return betas

    def power_schedule(self, steps_num, power=10, eps=0.001):
        t = torch.linspace(0, 1, steps_num+1, dtype=torch.float32)
        f_t = t ** power
        return f_t

    def get_schedule(self, schedule_name, sde_opt=None):
        if schedule_name=='linear':
            return self.linear_beta_schedule(self.T)
        elif schedule_name=='cosine':
            return self.cos_beta_schedule(self.T)
        elif schedule_name=='cosine_alpha':
            return self.get_alpha_schedule(self.T)
        elif schedule_name=='sigmoid':
            return self.sigmoid_schedule(self.T)
        elif schedule_name=='double_cosine':
            return self.double_cos_beta_schedule(self.T)
        elif schedule_name=='power':
            return self.power_schedule(self.T, power=sde_opt['power'])
        else:
            print(f'driftSDE.py::get_sschedule: No implemented schedule: {schedule_name}')

