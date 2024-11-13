import json
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, Union
from .transform import PreprocessCfg, merge_preprocess_dict, image_transform_v2, merge_preprocess_kwargs
from pathlib import Path
import os

import torch
from .openai import load_openai_model

from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict, \
    resize_pos_embed, get_cast_dtype, resize_text_pos_embed, set_model_preprocess_cfg
from .coca_model import CoCa
from .tokenizer import HFTokenizer, SimpleTokenizer, DEFAULT_CONTEXT_LENGTH

HF_HUB_PREFIX = 'hf-hub:'


# def _get_hf_config(model_id, cache_dir=None):
#     config_path = download_pretrained_from_hf(model_id, filename='open_clip_config.json', cache_dir=cache_dir)
#     with open(config_path, 'r', encoding='utf-8') as f:
#         config = json.load(f)
#     return config

# def get_model_config(model_name):
#     if model_name in _MODEL_CONFIGS:
#         return deepcopy(_MODEL_CONFIGS[model_name])
#     else:
#         return None

def load_checkpoint(model, checkpoint_path, strict=True):
    if Path(checkpoint_path).suffix in ('.npz', '.npy'):
        from .big_vision import load_big_vision_weights
        load_big_vision_weights(model, checkpoint_path)
        return {}
    if Path(checkpoint_path).suffix in ('.bin'):
        state_dict = torch.load(checkpoint_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.endswith('position_ids'):
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        return model


def _my_get_hf_config(cfg_file):
    with open(cfg_file, 'r') as f:
        config = json.load(f)
        f.close()
    return config


def create_model(
        model_name: str,
        cfg_file: str,
        checkpoint_path: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        **model_kwargs,
):
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())

    # has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    # if has_hf_hub_prefix:
    # model_id = model_name[len(HF_HUB_PREFIX):]
    # TODO
    # checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
    # TODO
    # config = _get_hf_config(model_id, cache_dir)
    config = _my_get_hf_config(cfg_file)
    preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config['preprocess_cfg'])
    model_cfg = config['model_cfg']
    pretrained_hf = False  # override, no need to load original HF text weights
    # else:
    #     model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    #     checkpoint_path = None
    #     model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        # logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
        )
    else:
        # TODO
        model_cfg = model_cfg  # or get_model_config(model_name) # zyc: BiomedCLIP is not in the openai lib
        # if model_name.startswith("E:"):
        #     import json
        #     with open(os.path.join(model_name, "config.json")) as f:
        #         model_cfg = json.load(f)["model_cfg"]
        #         f.close()
        if model_cfg is not None:
            # logging.info(f'Loaded {model_name} model config.')
            pass
        else:
            # logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})  # True
        if pretrained_image:  # False
            if is_timm_model:
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
        cast_dtype = get_cast_dtype(precision)  # None
        is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})  # True
        if is_hf_model:
            # load pretrained weights for HF text model IFF no CLIP weights being loaded
            model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf and not pretrained
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model

        model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)
        if custom_text:
            if "multimodal_cfg" in model_cfg:
                model = CoCa(**model_cfg, cast_dtype=cast_dtype)
            else:
                model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        if precision in ("fp16", "bf16"):  # 'fp32'
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            # manual mixed precision that matches original OpenAI behaviour
            if is_timm_model:
                # FIXME this is a bit janky, create timm based model in low-precision and
                # then cast only LayerNormFp32 instances back to float32 so they don't break.
                # Why? The convert_weights_to_lp fn only works with native models.
                model.to(device=device, dtype=dtype)
                from .transformer import LayerNormFp32

                def _convert_ln(m):
                    if isinstance(m, LayerNormFp32):
                        m.weight.data = m.weight.data.to(torch.float32)
                        m.bias.data = m.bias.data.to(torch.float32)

                model.apply(_convert_ln)
            else:
                model.to(device=device)
                convert_weights_to_lp(model, dtype=dtype)
        elif precision in ("pure_fp16", "pure_bf16"):  # 'fp32'
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            model.to(device=device, dtype=dtype)
        else:
            model.to(device=device)

        pretrained_loaded = False
        # if pretrained: # pretrained==False
        # checkpoint_path = ''
        # pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
        # if pretrained_cfg:
        #     checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
        #     preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
        # elif os.path.exists(pretrained):
        #     checkpoint_path = pretrained

        # if checkpoint_path:
        #     # logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
        #     load_checkpoint(model, checkpoint_path)
        # else:
        #     error_str = (
        #         f'Pretrained weights ({pretrained}) not found for model {model_name}.'
        #         f'Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
        #     # logging.warning(error_str)
        #     raise RuntimeError(error_str)
        # pretrained_loaded = True
        # elif has_hf_hub_prefix: # True
        # logging.info(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
        load_checkpoint(model, checkpoint_path)
        pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, 'image_size', None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg['size'] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    return model


def create_model_from_pretrained(
        model_name: str,
        cfg_file: str,
        checkpoint_path: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        return_transform: bool = True,
        cache_dir: Optional[str] = None,
        **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {}, mean=image_mean, std=image_std, interpolation=image_interpolation, resize_mode=image_resize_mode)

    model = create_model(
        model_name,
        cfg_file,
        checkpoint_path,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        cache_dir=cache_dir,
        require_pretrained=True,
        **model_kwargs,
    )

    if not return_transform:
        return model

    preprocess = image_transform_v2(
        PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return model, preprocess


def get_tokenizer(
        model_name: str = '',
        cfg_file: str = '',
        context_length: Optional[int] = None,
        **kwargs,
):
    if model_name.startswith(HF_HUB_PREFIX):
        model_name = model_name[len(HF_HUB_PREFIX):]
    try:
        config = _my_get_hf_config(cfg_file)['model_cfg']
        # config = _get_hf_config(model_name)['model_cfg']
    except Exception:
        tokenizer = HFTokenizer(
            model_name,
            context_length=context_length or DEFAULT_CONTEXT_LENGTH,
            **kwargs,
        )
        return tokenizer
    # else:
    #     config = get_model_config(model_name)
    #     assert config is not None, f"No valid model config found for {model_name}."

    text_config = config.get('text_cfg', {})
    if 'tokenizer_kwargs' in text_config:
        tokenizer_kwargs = dict(text_config['tokenizer_kwargs'], **kwargs)
    else:
        tokenizer_kwargs = kwargs

    if context_length is None:
        context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)

    if 'hf_tokenizer_name' in text_config:
        # tokenizer = HFTokenizer(
        #     text_config['hf_tokenizer_name'],
        #     context_length=context_length,
        #     **tokenizer_kwargs,
        # )
        tokenizer = HFTokenizer(
            model_name,
            context_length=context_length,
            **tokenizer_kwargs,
        )
    else:
        tokenizer = SimpleTokenizer(
            context_length=context_length,
            **tokenizer_kwargs,
        )

    return tokenizer
