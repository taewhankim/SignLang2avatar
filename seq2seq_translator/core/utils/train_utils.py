from utils import str2code
import json


def get_scheduler(optimizer, conf):
    config = conf.scheduler
    return str2code(config.name)(optimizer=optimizer, **config[config.name])


def get_optimizer(models, conf):
    params = []
    config = conf.optimizer
    if type(models) != list:
        params = models.parameters()
    else:
        for model in models:
            params += list(model.parameters())
    return str2code(config.name)(params, **config.properties)


class Config:
    def __init__(self, json_path):
        with open(json_path, mode="r") as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, mode="w") as io:
            json.dump(self.__dict__, io, indent=4)

    def update(self, json_path):
        with open(json_path, mode="r") as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__
