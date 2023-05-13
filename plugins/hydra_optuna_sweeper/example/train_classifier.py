# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import hydra
from omegaconf import DictConfig
import optuna 
from hydra.core.hydra_config import HydraConfig
import logging
import sys

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

@hydra.main(version_base=None, config_path="conf", config_name="train_classifier")
def train_classifier(cfg: DictConfig) -> float:
    storage = optuna.storages.JournalStorage( # using predefined journal storage for example
        optuna.storages.JournalFileStorage(HydraConfig.get().sweeper.storage),
    )
    study = optuna.load_study(
        study_name="train_classifier", storage=storage, pruner=optuna.pruners.MedianPruner()
    )
    trial = optuna.Trial(study, HydraConfig.get().job.num) # create a new trial from a frozen trial from storage
    alpha: float = cfg.alpha

    if cfg.get("error", False):
        raise RuntimeError("cfg.error is True")
    
    iris = sklearn.datasets.load_iris()
    classes = list(set(iris.target))
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=0
    )

    clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

    for step in range(1000):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = 1.0 - clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            print ("Pruning...")
            raise optuna.TrialPruned()
        
    return 1.0 - clf.score(valid_x, valid_y)


if __name__ == "__main__":
    train_classifier()
