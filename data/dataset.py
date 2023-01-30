from fvcore.common.registry import Registry
DATASET_REGISTRY = Registry('DATASET')


def get_dataset(dataset_name, split):
    return DATASET_REGISTRY.get(dataset_name)(f'./data/{dataset_name}', split)
