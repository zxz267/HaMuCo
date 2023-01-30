from fvcore.common.registry import Registry
MODEL_REGISTRY = Registry('MODEL')

def get_model(model_name):
    return MODEL_REGISTRY.get(model_name)()

