import logging
import  importlib

def create_model(train_opt, model_opt, phase='train'):
    print(__package__)
    module_name = model_opt['module_name']
    class_name = model_opt['class_name']
    module = importlib.import_module(f'{__package__}.{module_name}')
    create_fn = getattr(module, 'create_' + class_name)

    print(f"Creating model: {module_name}.{class_name}")

    return create_fn(train_opt, model_opt, phase=phase)