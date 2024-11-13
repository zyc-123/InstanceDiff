import argparse
import math
import os
import random
import sys
import copy

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
# from IPython import embed

# import open_clip

import options as option
from models import create_model
from models.SDEs import create_sde
import time

sys.path.insert(0, "../../")
import open_clip
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from tqdm import tqdm
import platform

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import shutil


def store_files(exp_root_dir, args, file_to_be_store=['config']):
    # opt["path"]["experiments_root"]
    dir = os.path.join(exp_root_dir, 'files')
    os.makedirs(dir)
    if 'config' in file_to_be_store:
        target_path = os.path.join(dir, os.path.basename(args.opt))
        shutil.copy(args.opt, target_path)
        file_to_be_store.remove('config')

    for file in file_to_be_store:
        target_path = os.path.join(dir, os.path.basename(file))
        shutil.copy(file, target_path)

def init_dist(backend="nccl", **kwargs):
    backend = 'nccl'
    if platform.system() == "Windows":
        backend = 'gloo'
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
            mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    # rank = torch.distributed.get_rank()
    # rank = int(os.environ["LOCAL_RANK"])
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    print(f'set device {rank % num_gpus}')
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, rank=rank, world_size=2)
    # dist.init_process_group(
    #     backend=backend, rank=rank, world_size=2, **kwargs
    # )  # Initializes the default distributed process group
    # dist.init_process_group(backend=backend, rank=rank, world_size=2)


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    print(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]
    set_seed(seed)

    #### distributed training settings
    print(args.launcher)
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                       and "pretrain_model" not in key
                       and "resume" not in key
                       and "daclip" not in key
                )
            )
            store_files(opt["path"]["experiments_root"], args, file_to_be_store=opt['file_to_be_store'])
            if platform.system() == "Windows":
                os.system("rd log")
            else:
                print("remove ./log")
                os.system("rm -r ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

    #### create train and val dataloader
    dataset_ratio = 1  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_epochs = opt['train']['nepoch']
            total_iters = total_epochs * train_size
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_iters = total_epochs * train_size * dataset_ratio
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:  # main point no distribution
                print(f"Number of train images: {len(train_set)}, iters: {train_size}")
                print(f"Total epochs: {total_epochs}")
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                print(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    train_opt = opt['train']
    model_opt = opt['models'][train_opt['which_model']]
    model = create_model(train_opt, model_opt)  # .cuda()

    #### resume training
    if resume_state:
        print(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"] + 1
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
        model.load(current_step, opt["path"]["models"])
    else:
        current_step = 0
        start_epoch = 0

    # sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    # sde.set_model(model.model)
    # sde = driftSDE(100, model.drift_net, model.noise_net, noise_schedule='cosine', drift_schedule='cosine')
    # model.set_sde(sde)
    nets = model.get_nets()
    sde = create_sde(nets, opt['sdes'][train_opt['which_sde']])
    model.set_sde(sde)

    # scale = opt['degradation']['scale']

    #### training
    print(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    os.makedirs('image', exist_ok=True)

    last_message = "Training Phase (X / X Steps) (loss=X.X)"
    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        train_iterator = tqdm(
            train_loader, desc="", dynamic_ncols=True
        )
        train_loss = 0.0
        model.reinit_loss_message()

        iter_time = [[0.0], [0.0]]
        for ii, train_data in enumerate(train_iterator):
            current_step += 1

            # if current_step > total_iters:
            #     break
            # st_time = time.time()
            LQ, GT, names = train_data["LQ"], train_data["GT"], train_data["name"]

            data = {
                'input': LQ.cuda(),
                'target': GT.cuda(),
                'names': names
            }
            if "A_emb" in train_data:
                data["A_emb"] = (train_data["A_emb"].cuda())

            model.feed_data(data)  # xt, mu, x0
            loss, dur_time = model.optimize_parameters()
            train_loss += loss
            # model.update_learning_rate(
            #     current_step, warmup_iter=opt["train"]["warmup_iter"]
            # )
            # dur_time = time.time() - st_time
            iter_time[rank].append(dur_time)

            # if rank <= 0:
            message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                epoch, current_step, model.get_current_learning_rate()
            )
            message += '(iter time: {}={:4f}, {}={:4f})'.format(0, sum(iter_time[0]) / len(iter_time[0]), 1,
                                                                sum(iter_time[1]) / len(iter_time[1]))
            # message += '(batch size={})'.format(LQ.shape[0])
            message = message + model.get_loss_message()  # f"loss={loss}/{train_loss/(ii+1)}"
            message = f"Training: {ii}/{len(train_loader)}" + message

            train_iterator.set_description(message)

            if current_step % opt["logger"]["print_freq"] == 0:
                print(message)

            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    print("Saving models and training states.")
                    model.save(current_step, opt["path"]["models"])
                    model.save_training_state(epoch, current_step, opt["path"]["training_state"])

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                model.set_eval()

                psnr_accum = 0.0
                rmse_accum = 0.0
                ssim_accum = 0.0
                idx = 0
                with torch.no_grad():
                    val_iterator = tqdm(
                        val_loader, desc="Training Phase (X / X Steps) (loss=X.X)", dynamic_ncols=True
                    )
                    for jj, val_data in enumerate(val_iterator):
                        idx += 1

                        LQ, GT, names = val_data["LQ"], val_data["GT"], val_data["name"]
                        data = {
                            'input': LQ.cuda(),
                            'target': GT.cuda(),
                            'names': names
                        }
                        if "A_emb" in val_data:
                            data["A_emb"] = (val_data["A_emb"].cuda())

                        # valid Predictor
                        model.feed_data(data)
                        model.test()

                        visuals = model.get_visuals().squeeze()

                        pred = visuals.reshape((1, 1, visuals.shape[-2], visuals.shape[-1]))
                        target = GT.squeeze().unsqueeze(0).unsqueeze(0).detach().cpu().numpy()

                        pred = pred / 2.0 + 0.5
                        target = target / 2.0 + 0.5

                        RMSE = np.sqrt(mse(pred, target))
                        PSNR = psnr(pred, target, data_range=1.0)
                        SSIM = ssim(pred.squeeze(), target.squeeze(), use_sample_covariance=False, sigma=1.5,
                                    gaussian_weights=True,
                                    win_size=11, K1=0.01, K2=0.03, data_range=1.0)
                        psnr_accum += PSNR
                        rmse_accum += RMSE
                        ssim_accum += SSIM

                        to_save = np.concatenate((LQ.squeeze(), visuals.squeeze(), GT.squeeze()), axis=-1)
                        to_save.tofile(f'image/{jj}_.raw')

                        val_iterator.set_description(
                            f"Validating: {jj}/{len(val_loader)}, AVG_RMSE={rmse_accum / idx}, AVG_SSIM={ssim_accum / idx}, AVG_PSNR={psnr_accum / idx}")

                        if idx > 9:
                            break

                model.set_train()

                # log
                print("<epoch:{:3d}, iter:{:8,d}".format(
                    epoch, current_step
                ))
                print("# Validation # PSNR: {:.6f} # SSIM: {:.6f} # RMSE: {:.6f}".format(psnr_accum / 100,
                                                                                         ssim_accum / 100,
                                                                                         rmse_accum / 100))

        if epoch % 5 == 0:
            if rank <= 0:
                print(f"Saving models and training states. At Epoch {epoch}")
                model.save(f"epoch_{epoch}", opt["path"]["models"])
                model.save_training_state(epoch, current_step, opt["path"]["training_state"])

    if rank <= 0:
        print("Saving the final model.")
        model.save("latest", opt["path"]["models"])
        print("End of training.")


if __name__ == "__main__":
    main()
