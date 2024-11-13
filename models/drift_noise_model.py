import torch
import torch.nn as nn
import numpy as np
import clip
import os
from torch.nn.parallel import DataParallel, DistributedDataParallel
from ema_pytorch import EMA
import torch.nn.functional as F
import time
def get_drift_deferential_cosine(t, T):
    """
    use half of cosine curve as the drift trajectory
    """
    dirft_level = (1 - np.cos((t*np.pi)/T))/2
    next_dirft_level = (1 - np.cos(((t+1)*np.pi)/T))/2
    return next_dirft_level - dirft_level

from .modules.MSM_degEmb_Unet import ScoreMapModule
from .modules.modified_clip import CLIPTextContextEncoder
from collections import OrderedDict

from .modules import create_net

from torchvision.transforms.transforms import Resize
from ._modified_BiomedCLIP import HFContextTextEncoder

class CLIPDriftModel():
    def __init__(self,
                 text_encoder_pretrain_path,
                 drift_net_lr,
                 noise_net_lr,
                 weight_decay_drift,
                 beta1,
                 beta2,
                 nepoch,
                 eta_min,
                 dist=False,
                 gpu=True,
                 optimize_type='predict_noise',
                 optimize_target='std',
                 if_train = True,
                 dnet_settings = None,
                 nnet_settings = None,
                 drift_loss='l2',
                 noise_loss='none',
                 if_MultiScoreMap=False,
                 score_map_ch_mult=[1, 1, 2, 4],
                 score_map_ngf=64,
                 use_image_context=False,
                 use_degra_context=False,
                 CLIP_Type="CLIP"
                 ):
        """
        opt: neccesary keys: 'drift_net_lr', 'noise_net_lr', 'weight_decay_drift', 'beta1', 'beta2', 'clip_pretrained'.
        """
        # super(CLIPDriftModel, self).__init__()
        # define CLIP model, freezing parameters
        dnet_settings['use_image_context'] = use_image_context
        dnet_settings['use_degra_context'] = use_degra_context
        nnet_settings['use_image_context'] = use_image_context
        nnet_settings['use_degra_context'] = use_degra_context
        self.dnet_settings = dnet_settings
        self.nnet_settings = nnet_settings
        self.score_map_ch_mult = score_map_ch_mult
        self.use_image_context = use_image_context
        self.use_degra_context = use_degra_context

        self.optimize_target = optimize_target
        self.dist = dist
        token_embed_dim = 512
        if CLIP_Type=="BiomedCLIP":
            self.text_encoder = HFContextTextEncoder()  # max seq len = 512
            self.text_encoder.init_weights(pretrain_path=text_encoder_pretrain_path)
            for param in self.text_encoder.parameters():
                param.requires_grad_(False)
            self.text_encoder.eval()
            token_embed_dim = 768
        else:
            self.text_encoder = CLIPTextContextEncoder(
                context_length=42,
                embed_dim=512,
                transformer_width=512,
                transformer_heads=8,
                transformer_layers=12,
                pretrained=text_encoder_pretrain_path# opt['clip_pretrained']
            )
            self.text_encoder.init_weights()
            for param in self.text_encoder.parameters():
                param.requires_grad_(False)
        self.text_encoder.cuda()

        self.optimize_type = optimize_type # option in ['predict_x0', 'predict_noise']

        self.drift_loss = drift_loss
        self.noise_loss = noise_loss

        print(self.optimize_target)
        print(self.optimize_type)

        self.device = torch.device("cuda" if gpu else "cpu")
        print("DriftModel:", torch.cuda.current_device())

        # text-driven module: attention or score map
        self.drift_prompt = None
        self.noise_prompt = None
        if dnet_settings['text_module']=='scoremap':
            if dnet_settings['if_MultiScoreMap']:
                # score_map_nf = dnet_settings['score_map_ngf']
                # score_map_ch_mult = dnet_settings['score_map_ch_mult']
                self.drift_prompt = nn.ModuleList([ScoreMapModule(visual_dim=score_map_ngf * score_map_ch_mult[i],
                                                                  CLIP_Type=CLIP_Type,
                                                                  token_embed_dim=token_embed_dim).to(self.device) for i in range(len(score_map_ch_mult))]).to(self.device)
            else:
                self.drift_prompt = ScoreMapModule().to(self.device)
            if dist:
                for i in range(len(self.drift_prompt)):
                    self.drift_prompt[i] = DistributedDataParallel(self.drift_prompt[i], device_ids=[torch.cuda.current_device()])
                # self.drift_prompt = DistributedDataParallel(self.drift_prompt, device_ids=[torch.cuda.current_device()])
            else:
                pass
                #self.drift_prompt = DataParallel(self.drift_prompt)
            self.dp_ema = EMA(self.drift_prompt, beta=0.995, update_every=10).to(self.device)
        if nnet_settings['text_module']=='scoremap':
            if nnet_settings['if_MultiScoreMap']:
                # score_map_nf = nnet_settings['score_map_ngf']
                # score_map_ch_mult = nnet_settings['score_map_ch_mult']
                self.noise_prompt = nn.ModuleList([ScoreMapModule(visual_dim=score_map_ngf * score_map_ch_mult[i],
                                                                  CLIP_Type=CLIP_Type,
                                                                  token_embed_dim=token_embed_dim).to(self.device) for i in range(len(score_map_ch_mult))]).to(self.device)
            else:
                self.noise_prompt = ScoreMapModule().to(self.device)
            if dist:
                for i in range(len(self.noise_prompt)):
                    self.noise_prompt[i] = DistributedDataParallel(self.noise_prompt[i], device_ids=[torch.cuda.current_device()])
                # self.noise_prompt = DistributedDataParallel(self.noise_prompt, device_ids=[torch.cuda.current_device()])
            else:
                pass
                #self.noise_prompt = DataParallel(self.noise_prompt)
            self.np_ema = EMA(self.noise_prompt, beta=0.995, update_every=10).to(self.device)

        # define dual network: drift net and noise net
        self.drift_net = create_net(dnet_settings, CLIP_ScoreMapModule=self.drift_prompt).to(self.device)
        self.noise_net = create_net(nnet_settings, CLIP_ScoreMapModule=self.noise_prompt).to(self.device)
        if dist:
            self.drift_net = DistributedDataParallel(self.drift_net, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
            self.noise_net = DistributedDataParallel(self.noise_net, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        else:
            pass
            # self.drift_net = DataParallel(self.drift_net)
            # self.noise_net = DataParallel(self.noise_net)
        self.dn_ema = EMA(self.drift_net, beta=0.995, update_every=10).to(self.device)
        self.nn_ema = EMA(self.noise_net, beta=0.995, update_every=10).to(self.device)

        # define loss function, optimizer, and learning rate scheduler
        if if_train:
            self.l1 = nn.L1Loss(reduction='mean')
            self.l2 = nn.MSELoss(reduction='mean')
            self.drift_optimizer = torch.optim.Adam([{'params': self.drift_net.parameters(), 'lr': drift_net_lr}],
                                                    weight_decay=weight_decay_drift,
                                                    betas=(beta1, beta2))
            self.noise_optimizer = torch.optim.Adam([{'params': self.noise_net.parameters(), 'lr': noise_net_lr}],
                                                    weight_decay=weight_decay_drift,
                                                    betas=(beta1, beta2))

            self.drift_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.drift_optimizer, 
                                                                                T_max=nepoch,
                                                                                eta_min=eta_min)
            self.noise_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.noise_optimizer,
                                                                                T_max=nepoch,
                                                                                eta_min=eta_min)

    def update_lr(self):
        self.drift_lr_scheduler.step()
        self.noise_lr_scheduler.step()

    def get_current_learning_rate(self):
        return self.noise_optimizer.param_groups[0]["lr"]

    def set_sde(self, sde):
        self.sde = sde

    def feed_data(self, data):
        self.input = data['input'].to(self.device)
        self.target = data['target'].to(self.device)
        self.names = data['names']
        if self.use_image_context:
            self.A_emb = data['A_emb'].to(self.device)
        else:
            self.A_emb = None
        time_idx, drift_noised_x, drift, std_noise, noise = self.sde.forward_diffusion(self.target, self.input)
        self.t = time_idx#to(torch.cuda.current_device())
        self.drift_noised_x = drift_noised_x
        self.drift = drift
        self.std_noise = std_noise
        self.noise = noise
    
    def reinit_loss_message(self):
        if self.optimize_type == 'predict_noise' or self.optimize_type == '' or self.optimize_type=="inputRes":
            self.loss_info = {
                'latest': {'l': 0, 'nsml': 0, 'dsml': 0, 'nl': 0, 'dl': 0},
                'avg': {'l': 0, 'dl': 0, 'nl': 0, 'dsml': 0, 'nsml': 0},
                'num': 0}
        elif self.optimize_type == 'predict_x0':
            self.loss_info = {
                'latest': {'l': 0, 'x0sml': 0, 'dsml': 0, 'x0l': 0, 'dl': 0},
                'avg': {'l': 0, 'dl': 0, 'x0l': 0, 'dsml': 0, 'x0sml': 0},
                'num': 0}
        elif self.optimize_type == 'predict_std_noise_scale_drift':
            self.loss_info = {
                'latest': {'l': 0, 'nsml': 0, 'dsml': 0, 'nl': 0, 'dl': 0},
                'avg': {'l': 0, 'dl': 0, 'nl': 0, 'dsml': 0, 'nsml': 0},
                'num': 0}
        elif self.optimize_type == "predict_std_noise_acc_drift":
            self.loss_info = {
                'latest': {'l': 0, 'nsml': 0, 'dsml': 0, 'nl': 0, 'dl': 0},
                'avg': {'l': 0, 'dl': 0, 'nl': 0, 'dsml': 0, 'nsml': 0},
                'num': 0
            }

    def get_loss_message(self):
        num = self.loss_info['num']
        if num<1:
            num=1
        message = ""
        for k in self.loss_info['latest'].keys():
            latest_v = self.loss_info['latest'][k]
            avg_v = self.loss_info['avg'][k] / num
            message += '({}={:4f}/{:4f})'.format(k, latest_v, avg_v)
        return message
    
    def optimize_parameters(self):
        return self.optimize_parameters_inputRes()

    def optimize_score_map(self, score_maps, label, size=[224, 224], mult=[1, 2, 4, 8]):
        loss = []
        for i in range(len(score_maps)):
            score_map = score_maps[i]
            resizer = Resize((size[0] // mult[i], size[1] // mult[i]))
            loss.append(self.l2(score_map, resizer(label)))
        return sum(loss) / 2.0

    def optimize_parameters_inputRes(self):
        """
        description: (x_t-cond, x_t)->noise; (x_t-cond, cond)->drift; use L2 to optimize
        """
        st_time = time.time()
        noise_scoremap = []
        drift_scoremap = []
        if self.dnet_settings["text_module"] == "scoremap":
            pred_drift, drift_scoremap = self.drift_net(self.drift_noised_x - self.input,
                                                        self.input,
                                                        self.t.squeeze(), self.names, self.text_encoder,
                                                        image_context=self.A_emb)
        else:
            pred_drift = self.drift_net(self.drift_noised_x - self.input,
                                                        self.input,
                                                        self.t.squeeze(), self.names, self.text_encoder,
                                                        image_context=self.A_emb)
        if self.nnet_settings["text_module"] == "scoremap":
            pred_noise, noise_scoremap = self.noise_net(self.drift_noised_x - self.input,
                                                        self.drift_noised_x,
                                                        self.t.squeeze(), self.names, self.text_encoder,
                                                        image_context=self.A_emb)
        else:
            pred_noise = self.noise_net(self.drift_noised_x - self.input,
                                                        self.drift_noised_x,
                                                        self.t.squeeze(), self.names, self.text_encoder,
                                                        image_context=self.A_emb)
        # row_size = (self.input.shape[-2], self.input.shape[-1])
        dloss = self.l2(pred_drift, self.input - self.target)
        use_dsm = True
        if "use_dsm" in self.dnet_settings:
            use_dsm = self.dnet_settings["use_dsm"]
        use_nsm = True
        if "use_nsm" in self.nnet_settings:
            use_nsm = self.nnet_settings["use_nsm"]
        if self.dnet_settings["text_module"] == "scoremap" and use_dsm:
            dsm_loss = self.optimize_score_map(drift_scoremap, self.input - self.target) # self.l2(drift_scoremap, self.input - self.target)
        nloss = self.l2(pred_noise, self.std_noise)
        if self.nnet_settings["text_module"] == "scoremap" and use_nsm:
            nsm_loss = self.optimize_score_map(noise_scoremap, self.std_noise)
        # nsm_loss = self.l2(noise_scoremap, self.std_noise)

        loss = dloss + nloss
        if use_dsm:
            loss = loss + dsm_loss
        if use_nsm:
            loss = loss + nsm_loss

        iter_time = time.time() - st_time

        self.noise_optimizer.zero_grad()
        self.drift_optimizer.zero_grad()
        loss.backward()
        self.noise_optimizer.step()
        self.drift_optimizer.step()

        # record loss
        self.loss_info['latest']['l'] = loss.item()
        self.loss_info['latest']['nsml'] = nsm_loss.item() if (len(noise_scoremap)>0 and use_nsm) else 0
        self.loss_info['latest']['dsml'] = dsm_loss.item() if (len(drift_scoremap)>0 and use_dsm) else 0
        self.loss_info['latest']['nl'] = nloss.item()
        self.loss_info['latest']['dl'] = dloss.item()
        self.loss_info['avg']['l'] += loss.item()
        self.loss_info['avg']['dl'] += dloss.item()
        self.loss_info['avg']['nl'] += nloss.item()
        self.loss_info['avg']['dsml'] += dsm_loss.item() if (len(noise_scoremap)>0 and use_nsm) else 0
        self.loss_info['avg']['nsml'] += nsm_loss.item() if (len(drift_scoremap)>0 and use_dsm) else 0
        self.loss_info['num'] += 1

        # return loss.item(), pred_loss.item(), scoremap_loss.item()
        return loss.item(), iter_time

    def optimize_parameters_pred_noise_acc_drift(self):
        st_time = time.time()
        pred_drift, drift_scoremap = self.drift_net(self.drift_noised_x, self.target + self.drift, self.t.squeeze(), self.names,
                                                    self.text_encoder, image_context=self.A_emb)
        pred_noise, noise_scoremap = self.noise_net(self.drift_noised_x, self.input, self.t.squeeze(), self.names,
                                                    self.text_encoder, image_context=self.A_emb)

        dloss = self.l2(pred_drift, self.input - self.target)
        dsm_loss = self.l2(drift_scoremap, self.input - self.drift)
        nloss = self.l2(pred_noise, self.std_noise)
        nsm_loss = self.l2(noise_scoremap, self.std_noise)

        loss = dloss + dsm_loss + nloss + nsm_loss
        iter_time = time.time() - st_time

        self.noise_optimizer.zero_grad()
        self.drift_optimizer.zero_grad()
        loss.backward()
        self.noise_optimizer.step()
        self.drift_optimizer.step()

        # record loss
        self.loss_info['latest']['l'] = loss.item()
        self.loss_info['latest']['nsml'] = nsm_loss.item()
        self.loss_info['latest']['dsml'] = dsm_loss.item()
        self.loss_info['latest']['nl'] = nloss.item()
        self.loss_info['latest']['dl'] = dloss.item()
        self.loss_info['avg']['l'] += loss.item()
        self.loss_info['avg']['dl'] += dloss.item()
        self.loss_info['avg']['nl'] += nloss.item()
        self.loss_info['avg']['dsml'] += dsm_loss.item()
        self.loss_info['avg']['nsml'] += nsm_loss.item()
        self.loss_info['num'] += 1

        return loss.item(), iter_time

    def optimize_parameters_pred_noise_scale_Drift(self):
        st_time = time.time()
        pred_drift, drift_scoremap = self.drift_net(self.drift_noised_x, self.input, self.t.squeeze(), self.names,
                                                    self.text_encoder, image_context=self.A_emb)
        pred_noise, noise_scoremap = self.noise_net(self.drift_noised_x, self.input, self.t.squeeze(), self.names,
                                                    self.text_encoder, image_context=self.A_emb)

        dloss = self.l2(pred_drift, (self.input - self.target) * self.sde.drift_schedule[self.t])
        dsm_loss = self.l2(drift_scoremap, (self.input - self.target) * self.sde.drift_schedule[self.t])
        nloss = self.l2(pred_noise, self.std_noise)
        nsm_loss = self.l2(noise_scoremap, self.std_noise)

        loss = dloss + dsm_loss + nloss + nsm_loss
        iter_time = time.time() - st_time

        self.noise_optimizer.zero_grad()
        self.drift_optimizer.zero_grad()
        loss.backward()
        self.noise_optimizer.step()
        self.drift_optimizer.step()

        # record loss
        self.loss_info['latest']['l'] = loss.item()
        self.loss_info['latest']['nsml'] = nsm_loss.item()
        self.loss_info['latest']['dsml'] = dsm_loss.item()
        self.loss_info['latest']['nl'] = nloss.item()
        self.loss_info['latest']['dl'] = dloss.item()
        self.loss_info['avg']['l'] += loss.item()
        self.loss_info['avg']['dl'] += dloss.item()
        self.loss_info['avg']['nl'] += nloss.item()
        self.loss_info['avg']['dsml'] += dsm_loss.item()
        self.loss_info['avg']['nsml'] += nsm_loss.item()
        self.loss_info['num'] += 1

        return loss.item(), iter_time

    def optimize_parameters_predict_x0(self):
        st_time = time.time()
        pred_drift, drift_scoremap = self.drift_net(self.drift_noised_x, self.input, self.t.squeeze(), self.names, self.text_encoder, image_context=self.A_emb)
        x0, x0_scoremap = self.noise_net(self.drift_noised_x, self.input, self.t.squeeze(), self.names, self.text_encoder, image_context=self.A_emb)

        dloss = self.l2(pred_drift, self.input - self.target)
        dsm_loss = self.l2(drift_scoremap, self.input - self.target)
        x0loss = self.l2(x0, self.target)
        x0sm_loss = self.l2(x0_scoremap, self.target)

        loss = x0loss + x0sm_loss + dloss + dsm_loss
        iter_time = time.time() - st_time

        self.noise_optimizer.zero_grad()
        self.drift_optimizer.zero_grad()
        loss.backward()
        self.noise_optimizer.step()
        self.drift_optimizer.step()

        self.loss_info['latest']['l'] = loss.item()
        self.loss_info['latest']['dl'] = dloss.item()
        self.loss_info['latest']['x0l'] = x0loss.item()
        self.loss_info['latest']['dsml'] = dsm_loss.item()
        self.loss_info['latest']['x0sml'] = x0sm_loss.item()
        self.loss_info['avg']['l'] += loss.item()
        self.loss_info['avg']['dl'] += dloss.item()
        self.loss_info['avg']['x0l'] += x0loss.item()
        self.loss_info['avg']['dsml'] += dsm_loss.item()
        self.loss_info['avg']['x0sml'] += x0sm_loss.item()
        self.loss_info['num'] += 1

        return loss.item(), iter_time

    def optimize_parameters7(self):
        st_time = time.time()
        # pred_drift, drift_scoremap = self.drift_net(self.drift_noised_x, self.input, self.t.squeeze(), self.names, self.text_encoder)
        # mean = self.drift_noised_x - (self.input - (1 - self.sde.drift_schedule[self.t]) * pred_drift)
        # pred_noise, noise_scoremap = self.noise_net(self.drift_noised_x, mean, self.t.squeeze(), self.names, self.text_encoder)
        pred_drift, drift_scoremap = self.drift_net(self.drift_noised_x, self.input, self.t.squeeze(), self.names,
                                                    self.text_encoder, image_context=self.A_emb)
        pred_noise, noise_scoremap = self.noise_net(self.drift_noised_x, self.input, self.t.squeeze(), self.names,
                                                    self.text_encoder, image_context=self.A_emb)

        dloss = self.l2(pred_drift, self.input-self.target)
        dsm_loss = self.optimize_score_map(drift_scoremap, self.input-self.target)
        # dsm_loss = self.l2(drift_scoremap, self.input-self.target)
        nloss = self.l2(pred_noise, self.std_noise)
        nsm_loss = self.optimize_score_map(noise_scoremap, self.std_noise)
        # nsm_loss = self.l2(noise_scoremap, self.std_noise)

        loss = dloss + dsm_loss + nloss + nsm_loss
        iter_time = time.time() - st_time

        self.noise_optimizer.zero_grad()
        self.drift_optimizer.zero_grad()
        loss.backward()
        self.noise_optimizer.step()
        self.drift_optimizer.step()

        # record loss
        self.loss_info['latest']['l'] = loss.item()
        self.loss_info['latest']['nsml'] = nsm_loss.item()
        self.loss_info['latest']['dsml'] = dsm_loss.item()
        self.loss_info['latest']['nl'] = nloss.item()
        self.loss_info['latest']['dl'] = dloss.item()
        self.loss_info['avg']['l'] += loss.item()
        self.loss_info['avg']['dl'] += dloss.item()
        self.loss_info['avg']['nl'] += nloss.item()
        self.loss_info['avg']['dsml'] += dsm_loss.item()
        self.loss_info['avg']['nsml'] += nsm_loss.item()
        self.loss_info['num'] += 1

        # return loss.item(), pred_loss.item(), scoremap_loss.item()
        return loss.item(), iter_time
    
    def optimize_parameters6(self):
        pred_noise, noise_scoremap = self.noise_net(self.drift_noised_x, self.input, self.t.squeeze(), self.names, self.text_encoder)
        
        nloss = self.l2(pred_noise, self.std_noise)
        nsm_loss = self.l2(noise_scoremap, self.std_noise)

        loss = nloss + nsm_loss

        self.noise_optimizer.zero_grad()
        loss.backward()
        self.noise_optimizer.step()

        # record loss 
        self.loss = loss.item()
        self.drift_loss = 0
        self.drift_sm_loss = 0
        self.noise_loss = nloss.item()
        self.noise_sm_loss = nsm_loss.item()
        self.total_loss += loss.item()
        self.total_drift_loss += 0
        self.total_drift_sm_loss += 0
        self.total_noise_loss += nloss.item()
        self.total_noise_sm_loss += nsm_loss.item()
        self.num  += 1

        # return loss.item(), pred_loss.item(), scoremap_loss.item()
        return loss.item()
    
    def optimize_parameters5(self):
        time_drift = self.sde.T*torch.ones_like(self.t.squeeze())
        pred_drift, drift_scoremap = self.drift_net(torch.zeros_like(self.input).to(self.input.device), self.input, time_drift, self.names, self.text_encoder, image_context=self.A_emb)
        expectation_drift_noised_x = self.target + self.sde.drift_schedule[self.t] * (self.input - self.target)
        pred_noise, noise_scoremap = self.noise_net(self.drift_noised_x, expectation_drift_noised_x, self.t.squeeze(), self.names, self.text_encoder, image_context=self.A_emb)
        
        if self.drift_loss=='l2':
            dloss = self.l2(pred_drift, self.input - self.target)
            dsm_loss = self.l2(drift_scoremap, self.input - self.target)
        else:
            dloss = self.l1(pred_drift, self.input - self.target)
            dsm_loss = self.l1(drift_scoremap, self.input - self.target)

        if self.noise_loss=='uni':
            nloss = self.l2(pred_noise + pred_drift, self.std_noise + self.input - self.target)
            nsm_loss = self.l2(noise_scoremap + drift_scoremap, self.std_noise + self.input - self.target)
        else:
            nloss = self.l2(pred_noise, self.std_noise)
            nsm_loss = self.l2(noise_scoremap, self.std_noise)

        pred_loss = dloss + nloss
        scoremap_loss = dsm_loss + nsm_loss
        loss = pred_loss + scoremap_loss

        self.drift_optimizer.zero_grad()
        self.noise_optimizer.zero_grad()
        loss.backward()
        self.drift_optimizer.step()
        self.noise_optimizer.step()

        # record loss 
        self.loss = loss.item()
        self.drift_loss = dloss.item()
        self.drift_sm_loss = dsm_loss.item()
        self.noise_loss = nloss.item()
        self.noise_sm_loss = nsm_loss.item()
        self.total_loss += loss.item()
        self.total_drift_loss += dloss.item()
        self.total_drift_sm_loss += dsm_loss.item()
        self.total_noise_loss += nloss.item()
        self.total_noise_sm_loss += nsm_loss.item()
        self.num  += 1

        # return loss.item(), pred_loss.item(), scoremap_loss.item()
        return loss.item()

    def optimize_parameters4(self):
        pred_drift, drift_scoremap = self.drift_net(torch.zeros_like(self.input).to(self.input.device), self.input, self.sde.T*torch.ones_like(self.t.squeeze()), self.names, self.text_encoder, image_context=self.A_emb)
        # pred_noise, noise_scoremap = self.noise_net(self.drift_noised_x - (self.input-self.target)*self.sde.drift_schedule[self.t], self.input, self.t.squeeze(), self.names, self.text_encoder)
        pred_noise, noise_scoremap = self.noise_net(self.drift_noised_x - (pred_drift.detach())*self.sde.drift_schedule[self.t], self.input, self.t.squeeze(), self.names, self.text_encoder, image_context=self.A_emb)
        
        dloss = self.l1(pred_drift, self.input - self.target)
        dsm_loss = self.l1(drift_scoremap, self.input - self.target)

        nloss = self.l1(pred_noise + pred_drift.detach()/self.sde.max_sigma, self.std_noise + (self.input - self.target)/self.sde.max_sigma)
        nsm_loss = self.l1(noise_scoremap + pred_drift.detach()/self.sde.max_sigma, self.std_noise + (self.input - self.target)/self.sde.max_sigma)

        pred_loss = dloss + nloss
        scoremap_loss = dsm_loss + nsm_loss
        loss = pred_loss + scoremap_loss

        self.drift_optimizer.zero_grad()
        self.noise_optimizer.zero_grad()
        loss.backward()
        self.drift_optimizer.step()
        self.noise_optimizer.step()

        # record loss 
        self.loss = loss.item()
        self.drift_loss = dloss.item()
        self.drift_sm_loss = dsm_loss.item()
        self.noise_loss = nloss.item()
        self.noise_sm_loss = nsm_loss.item()
        self.total_loss += loss.item()
        self.total_drift_loss += dloss.item()
        self.total_drift_sm_loss += dsm_loss.item()
        self.total_noise_loss += nloss.item()
        self.total_noise_sm_loss += nsm_loss.item()
        self.num  += 1

        # return loss.item(), pred_loss.item(), scoremap_loss.item()
        return loss.item()
    
    def optimize_parameters3(self):
        if 'd2nChain' in self.optimize_target:
            pred_drift, drift_scoremap = self.drift_net(torch.zeros_like(self.input).to(self.input.device), self.input, self.sde.T*torch.ones_like(self.t.squeeze()), self.names, self.text_encoder)
            noise_temp = self.drift_noised_x + (1 - self.sde.drift_schedule[self.t]) * pred_drift - self.input
            pred_noise, noise_scoremap = self.noise_net(self.drift_noised_x, noise_temp, self.t.squeeze(), self.names, self.text_encoder)
        else:
            pred_noise, noise_scoremap = self.noise_net(self.drift_noised_x, self.input, self.t.squeeze(), self.names, self.text_encoder)
            pred_drift, drift_scoremap = self.drift_net(self.drift_noised_x, self.input, self.t.squeeze(), self.names, self.text_encoder)
        
        if 'std' in self.optimize_target:
            dloss = self.l1(pred_drift, self.input - self.target)
            dsm_loss = self.l1(drift_scoremap, self.input - self.target)
            if 'uniOptim' in self.optimize_target:
                nloss = self.l1(self.sde.max_sigma*torch.sqrt(self.sde.noise_schedule[self.t]) * pred_noise, self.noise + self.drift - pred_drift)
                nsm_loss = self.l1(self.sde.max_sigma*torch.sqrt(self.sde.noise_schedule[self.t]) * noise_scoremap, self.noise + self.drift - pred_drift)
            else:
                nloss = self.l1(pred_noise, self.std_noise)
                nsm_loss = self.l1(noise_scoremap, self.std_noise)
        elif 'scaled' in self.optimize_target:
            if 'd2nChain' in self.optimize_target:
                dloss = self.l1(pred_drift, self.input - self.target) 
                dsm_loss = self.l1(drift_scoremap, self.input - self.target)  
            else:
                dloss = self.l1(pred_drift, self.drift)
                dsm_loss = self.l1(drift_scoremap, self.drift) 
            if 'uniOptim' in self.optimize_target:
                nloss = self.l1(pred_noise, self.noise + self.drift - pred_drift)
                nsm_loss = self.l1(noise_scoremap, self.noise + self.drift - pred_drift)
            else:
                nloss = self.l1(pred_noise, self.noise)
                nsm_loss = self.l1(noise_scoremap, self.noise)
        else:
            print(f'No implementation of {self.optimize_target}')
        pred_loss = dloss + nloss
        scoremap_loss = dsm_loss + nsm_loss
        loss = pred_loss + scoremap_loss

        self.drift_optimizer.zero_grad()
        self.noise_optimizer.zero_grad()
        loss.backward()
        self.drift_optimizer.step()
        self.noise_optimizer.step()

        # record loss 
        self.loss = loss.item()
        self.drift_loss = dloss.item()
        self.drift_sm_loss = dsm_loss.item()
        self.noise_loss = nloss.item()
        self.noise_sm_loss = nsm_loss.item()
        self.total_loss += loss.item()
        self.total_drift_loss += dloss.item()
        self.total_drift_sm_loss += dsm_loss.item()
        self.total_noise_loss += nloss.item()
        self.total_noise_sm_loss += nsm_loss.item()
        self.num  += 1

        # return loss.item(), pred_loss.item(), scoremap_loss.item()
        return loss.item()

    def set_eval(self):
        self.drift_net.eval()
        self.noise_net.eval()

    def set_gpu(self, device):
        print(f'setting gpu:{device}')
        self.drift_net.to(device)
        self.noise_net.to(device)
        self.drift_prompt.to(device)
        self.noise_prompt.to(device)
        self.text_encoder = self.text_encoder.to(device)
        self.device = device

    def set_train(self):
        self.drift_net.train()
        self.noise_net.train()

    def test(self):
        # print(f'{self.input.device}, {self.target.device}, {self.device}')
        out = self.sde.reverse_ddpm(self.input, self.names, self.text_encoder, reverse_type=self.optimize_target, optimize_type=self.optimize_type, image_context=self.A_emb)
        #out = self.sde.reverse_ode(self.input, self.names, self.text_encoder)
        self.visuals = out.detach().cpu().numpy()
    
    def get_visuals(self):
        return self.visuals
    
    def get_nets(self, use_ema=False):
        if use_ema:
            nets = {
                'noise_net': self.nn_ema,
                'drift_net': self.dn_ema
            }
        else:
            nets = {
                'noise_net': self.noise_net,
                'drift_net': self.drift_net
            }
        return nets
    
    def save_network(self, network, network_label, iter_label, save_dir):
        save_filename = "{}_{}.pth".format(iter_label, network_label)
        # save_path = os.path.join(self.opt["path"]["models"], save_filename)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save(self, iter_label, save_dir):
        if self.dnet_settings['text_module'] == 'scoremap':
            self.save_network(self.drift_prompt, "DP", iter_label, save_dir)
            self.save_network(self.noise_prompt, "NP", iter_label, save_dir)
            self.save_network(self.dp_ema, "DP_ema", 'lastest', save_dir)
            self.save_network(self.np_ema, "NP_ema", 'lastest', save_dir)
        self.save_network(self.drift_net, "DN", iter_label, save_dir)
        self.save_network(self.noise_net, "NN", iter_label, save_dir)
        self.save_network(self.dn_ema, "DN_ema", 'lastest', save_dir)
        self.save_network(self.nn_ema, "NN_ema", 'lastest', save_dir)

    def save_training_state(self, epoch, iter_step, save_dir):
        """Saves training state during training, which will be used for resuming"""
        state = {"epoch": epoch, "iter": iter_step, "schedulers": [self.drift_lr_scheduler, self.noise_lr_scheduler], "optimizers": [self.drift_optimizer, self.noise_optimizer]}
        save_filename = "{}.state".format(iter_step)
        # save_path = os.path.join(self.opt["path"]["training_state"], save_filename)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        self.drift_lr_scheduler, self.noise_lr_scheduler = resume_state['schedulers']
        self.drift_optimizer, self.noise_optimizer = resume_state['optimizers']

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        if self.dist:
            for k, v in load_net.items():
                k_ = k
                if k_.startswith("module."):
                    k_ = k_[7:]
                if "CLIP_ScoreMapModule" in k_ and "CLIP_ScoreMapModule.module" not in k_:
                    k_ = k_.replace("CLIP_ScoreMapModule", "CLIP_ScoreMapModule.module")
                if "online_model" in k_ and "online_model.module" not in k_:
                    k_ = k_.replace("online_model", "online_model.module")
                if "ema_model" in k_ and "ema_model.module" not in k_:
                    k_ = k_.replace("ema_model", "ema_model.module")
                load_net_clean[k_] = v
        else:
            for k, v in load_net.items():
                k_ = k
                if 'module.' in k_:
                    k_ = k_.replace('module.', '')
                load_net_clean[k_] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def load(self, iter_label, save_dir):
        # iter_label = 360000
        if self.dnet_settings['text_module'] == 'scoremap':
            path = os.path.join(save_dir, f"{iter_label}_DP.pth")
            self.load_network(path, self.drift_prompt)
            path = os.path.join(save_dir, f"{iter_label}_NP.pth")
            self.load_network(path, self.noise_prompt)
        path = os.path.join(save_dir, f"{iter_label}_DN.pth")
        self.load_network(path, self.drift_net)
        # iter_label = 540000
        path = os.path.join(save_dir, f"{iter_label}_NN.pth")
        self.load_network(path, self.noise_net)

        iter_label = 'lastest'
        if self.dnet_settings['text_module'] == 'scoremap':
            path = os.path.join(save_dir, f"{iter_label}_DP_ema.pth")
            self.load_network(path, self.dp_ema)
            path = os.path.join(save_dir, f"{iter_label}_NP_ema.pth")
            self.load_network(path, self.np_ema)
        path = os.path.join(save_dir, f"{iter_label}_DN_ema.pth")
        self.load_network(path, self.dn_ema)
        path = os.path.join(save_dir, f"{iter_label}_NN_ema.pth")
        self.load_network(path, self.nn_ema)
        

def create_CLIPDriftModel(train_opt, model_opt, phase='train'):
    text_encoder_pretrain_path  = model_opt['text_encoder_pretrain_path']
    noise_net_lr                = model_opt['noise_net_lr']
    drift_net_lr                = model_opt['drift_net_lr']
    weight_decay_drift          = model_opt['weight_decay_drift']
    beta1                       = model_opt['beta1']
    beta2                       = model_opt['beta2']
    nepoch                      = train_opt['nepoch']
    eta_min                     = model_opt['eta_min']
    if_train = True if phase=='train' else False
    dnet_settings                = model_opt['dnet_settings']
    nnet_settings                = model_opt['nnet_settings']
    optimize_target             = model_opt['optimize_target']
    drift_loss                  = model_opt['drift_loss']
    noise_loss                  = model_opt['noise_loss']
    optimize_type               = model_opt['optimize_type']
    dist                        = train_opt['dist']
    use_image_context           = model_opt['use_image_context']
    use_degra_context           = model_opt['use_degra_context']
    CLIP_Type = "CLIP"
    if "CLIP_Type" in model_opt:
        CLIP_Type = model_opt["CLIP_Type"]

    if 'if_MultiScoreMap' in model_opt:
        if_MultiScoreMap = model_opt['if_MultiScoreMap']
        score_map_ch_mult = model_opt['score_map_ch_mult']
        score_map_ngf = model_opt['score_map_ngf']
    else:
        if_MultiScoreMap = False
        score_map_ch_mult = [1, 1, 2, 4]
        score_map_ngf = 64
    return CLIPDriftModel(  text_encoder_pretrain_path,
                            drift_net_lr=drift_net_lr,
                            noise_net_lr=noise_net_lr,
                            weight_decay_drift=weight_decay_drift,
                            beta1=beta1,
                            beta2=beta2,
                            nepoch=nepoch,
                            eta_min=eta_min,
                            optimize_target=optimize_target,
                            optimize_type=optimize_type,
                            if_train=if_train,
                            dnet_settings = dnet_settings,
                            nnet_settings = nnet_settings,
                            drift_loss=drift_loss,
                            noise_loss=noise_loss,
                            dist=dist,
                            if_MultiScoreMap=if_MultiScoreMap,
                            score_map_ch_mult=score_map_ch_mult,
                            score_map_ngf=score_map_ngf,
                            use_image_context=use_image_context,
                            use_degra_context=use_degra_context,
                            CLIP_Type=CLIP_Type)
    