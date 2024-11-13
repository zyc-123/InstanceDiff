import importlib
import sys
import os
def create_net(opt, CLIP_ScoreMapModule=None):
    module_name = opt['module_name']
    class_name = opt['class_name']
    module = importlib.import_module(f'{__package__}.{module_name}')
    create_fn = getattr(module, 'create_'+class_name)

    return create_fn(opt, CLIP_ScoreMapModule=CLIP_ScoreMapModule)
