import importlib

def create_sde(nets, sde_opt):
    sde_class_name = sde_opt['class_name']
    modules = importlib.import_module(f'{__package__}.{sde_class_name}')
    create_fn = getattr(modules, 'create_' + sde_class_name)

    return create_fn(nets, sde_opt)