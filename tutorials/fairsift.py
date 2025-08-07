import os
import sys

import joblib
import numpy as np


# Get the absolute path to the directory where fairsphs.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from fairsphs.py
sys.path.append(os.path.join(base_path, "../"))

from tutorials.algorithms.sift import SIFT
from utils.helpers import get_experiment_params
from tutorials.fairses import FairSES


#Batch Inference Evaluation
class FairSIFT(FairSES):
    def __init__(self, config, model, sensitive_param, population_size=200, threshold=0):
        super().__init__(config, model, sensitive_param, population_size, threshold)
        self.approach_name = "FairSIFT"

        self.setup = SIFT(
            mu=self.population_size,
            bounds=np.array(self.config.input_bounds),
            fitness_func=self.check_discrimination_batch
        )


if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    if classifier_name == 'dnn':
        import tensorflow as tf

        classifier_path = f'models/{config.dataset_name}/dnn_slfc.keras'
        model = tf.keras.models.load_model(classifier_path)
    else:

        classifier_path = f'models/{config.dataset_name}/{classifier_name}.pkl'
        model = joblib.load(classifier_path)

    fairsift = FairSIFT(
        config=config,
        model=model,
        sensitive_param=sensitive_param
    )

    fairsift.run(max_allowed_time=max_allowed_time)
