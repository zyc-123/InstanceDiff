import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
import lpips

import options as option
from models import create_model

from models.SDEs import create_sde

sys.path.insert(0, "../../")
import open_clip
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr
import platform
import yaml

from models.driftSDE import driftSDE

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import random


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.


set_seed(1)

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")

opt_file = parser.parse_args().opt
with open(opt_file, mode="r") as f:
    str_content = f.read()
    opt = yaml.load(str_content, yaml.FullLoader)

# opt = option.dict_to_nonedict(opt)


#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    print(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
train_opt = opt['train']
test_opt = opt['test']

model_opt = opt['models'][test_opt['which_model']]
model = create_model(train_opt, model_opt, phase='test')
# device = model.device
model.load(opt['test']['iter'], opt['test']['pth_dir'])

# clip_model, _preprocess = clip.load("ViT-B/32", device=device)
# if opt['path']['daclip'] is not None:
#     clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=opt['path']['daclip'])
# else:
#     clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# tokenizer = open_clip.get_tokenizer('ViT-B-32')
# clip_model = clip_model.to(device)


# sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
# sde.set_model(model.model)
sde_opt = opt['sdes'][test_opt['which_sde']]
nets = model.get_nets(use_ema=opt['test']['use_ema'])
sde = create_sde(nets, sde_opt)
model.set_sde(sde)

gpu_id = opt['gpu_ids'][0]
model.set_gpu(torch.device(f'cuda:{gpu_id}'))
sde.set_gpu(torch.device(f'cuda:{gpu_id}'))

noise_type = opt['artifact_type']

result_root = os.path.join(test_opt['result_root'], opt['name'])
for artifact_type in opt['artifact_type']:
    result_dir = os.path.join(result_root, artifact_type)
    if not os.path.exists(result_dir):
        print(f"Making directiory {result_dir}")
        os.makedirs(result_dir)
    else:
        print(f"{result_dir} already exists")

with torch.no_grad():
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt["name"]  # path opt['']
        print("\nTesting [{:s}]...".format(test_set_name))
        test_start_time = time.time()
        # dataset_dir = os.path.join(opt["test"]["result_root"], test_set_name)
        # util.mkdir(dataset_dir)

        test_results = OrderedDict()
        for artifact_type in noise_type:  # ['speckle in OCT', 'speckle in ultra sound', 'scatter artifact in CT']:
            test_results[artifact_type] = OrderedDict()
            test_results[artifact_type]['num'] = 0
            for metric in ['RMSE', 'SSIM', 'PSNR']:
                test_results[artifact_type][metric] = []

        test_times = []

        for i, test_data in enumerate(test_loader):
            LQ, GT, names = test_data["LQ"], test_data["GT"], test_data["name"]
            data = {
                'input': LQ.cuda(),
                'target': GT.cuda(),
                'names': names,
                'A_emb': test_data["A_emb"].cuda(),
            }

            if names[0] not in noise_type:
                continue

            # valid Predictor
            model.feed_data(data)

            tic = time.time()
            model.test()
            toc = time.time()
            test_times.append(toc - tic)

            visuals = model.get_visuals().squeeze()

            pred = visuals.reshape((1, 1, visuals.shape[-2], visuals.shape[-1]))
            target = GT.squeeze().unsqueeze(0).unsqueeze(0).detach().cpu().numpy()

            pred = pred / 2 + 0.5
            target = target / 2 + 0.5

            # pred[pred < 0] = 0
            # pred[pred > 1] = 1

            # pred = pred[:,:,64:192,64:192]
            # target = target[:,:,64:192,64:192]

            RMSE = np.sqrt(mse(pred, target))
            PSNR = psnr(pred, target, data_range=1.0)
            SSIM = ssim(pred.squeeze(), target.squeeze(), use_sample_covariance=False, sigma=1.5,
                        gaussian_weights=True,
                        win_size=11, K1=0.01, K2=0.03, data_range=1.0)
            test_results[names[0]]['RMSE'].append(RMSE)
            test_results[names[0]]['SSIM'].append(SSIM)
            test_results[names[0]]['PSNR'].append(PSNR)
            test_results[names[0]]['num'] = test_results[names[0]]['num'] + 1

            to_save = np.concatenate((LQ.squeeze(), visuals.squeeze(), GT.squeeze()), axis=-1)
            img_name = os.path.basename(test_data['GT_path'][0]).split('.')[0]
            save_path = os.path.join(result_root, names[0], f'{i}_{to_save.shape[-1]}x{to_save.shape[-2]}x1.raw')
            to_save.tofile(save_path)

            GT_path = test_data['GT_path'][0]
            info = f'\n Testing {i}, {GT_path}: RMSE={RMSE}, SSIM={SSIM}, PSNR={PSNR}'
            print(info)

        for k1 in test_results.keys():
            message = f'{k1}'
            v1 = test_results[k1]
            for k2 in ['RMSE', 'SSIM', 'PSNR']:
                num = v1['num']
                message = message + f', AVG {k2}: {sum(v1[k2]) / num}'
            print(message)


