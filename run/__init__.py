import os
import inspect
import importlib
from torch.nn import functional as F
from argparse import Namespace


### Helper functions for getting the datasets, backbones, losses, and models
# getting the datasets
def get_all_dataset_names():
    return [dataset.split('.')[0] for dataset in os.listdir('dataset')
            if not dataset.find('__') > -1 and 'py' in dataset]

def get_all_datasets():
    datasets = {}
    for dataset in get_all_dataset_names():
        mod = importlib.import_module('dataset.' + dataset)
        dataset_classes_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'DatasetBase' in str(inspect.getmro(getattr(mod, x))[1:])]
        for d in dataset_classes_name:
            c = getattr(mod, d)
            datasets[c.NAME] = c
    return datasets

def get_dataset(dataset_name, accelerator, args: Namespace):
    """
    Returns the dataset object.
    """
    datasets_set = get_all_datasets()
    assert dataset_name in datasets_set, f'Dataset {dataset_name} not found. Available datasets: {datasets_set.keys()}'
    return datasets_set[dataset_name](accelerator, args)

# getting the models
def get_all_models_names():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

def get_all_models():
    models = {}
    for model in get_all_models_names():
        mod = importlib.import_module('models.' + model)
        class_name = {x.lower():x for x in mod.__dir__()}[model]
        models[model] = getattr(mod, class_name)
    return models

def get_model(args, accelerator):
    """
    Get the network architectures for the backbone.
    """
    models = get_all_models()
    return models[args.model_type](args, accelerator)

# getting the models
def get_all_modelwrappers_names():
    return [modelwrappers.split('.')[0] for modelwrappers in os.listdir('modelwrappers')
            if not modelwrappers.find('__') > -1 and 'py' in modelwrappers]

def get_all_modelwrappers():
    modelwrappers = {}
    for modelwrapper in get_all_modelwrappers_names():
        mod = importlib.import_module('modelwrappers.' + modelwrapper)
        class_name = {x.lower():x for x in mod.__dir__()}[modelwrapper.replace('_', '')]
        modelwrappers[modelwrapper] = getattr(mod, class_name)
    return modelwrappers 

def get_modelwrapper(modelwrapper):
    modelwrappers = get_all_modelwrappers()
    return modelwrappers[modelwrapper]