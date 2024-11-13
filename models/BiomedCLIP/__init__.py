from .BiomedCLIP import create_model_from_pretrained, get_tokenizer
# from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import os
import torch
import numpy as np

def get_BiomedCLIP(cfg_file='BiomedCLIP_config.json',
                   checkpoint_path='open_clip_pytorch_model.bin',
                   device='cuda:0'):
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    if checkpoint_path=="open_clip_pytorch_model.bin":
        checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)
    if cfg_file=="BiomedCLIP_config.json":
        cfg_file = os.path.join(os.path.dirname(__file__), cfg_file)
    model, preprocess = create_model_from_pretrained(model_name, cfg_file, checkpoint_path)
    tokenizer = get_tokenizer(model_name, cfg_file)
    # model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    # tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.to(device)
    return model, preprocess, tokenizer