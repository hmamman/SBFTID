import os
import sys

import numpy as np
# Get the absolute path to the directory where ftat.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Three levels up from ftat.py
sys.path.append(os.path.join(base_path, "../../"))

from tutorials.algorithms.ses import SES


class ESTEO(SES):
    def __init__(self, mu, lambda_, sigma, bounds, fitness_func, TM_size=200, feature_size_rate=2):
        super().__init__(mu, lambda_, sigma, bounds, fitness_func)
        self.features_size = max(1, self.dimension // feature_size_rate)

        self.TM = []
        self.TM_fitness = []
        self.TM_size = TM_size

    def select_features(self):
        self.feature_to_mutate = np.random.choice(self.features_to_select, self.features_size, replace=False)

    def generate_offsprings(self, base_solution):
        offsprings = []

        for _ in range(self.lambda_):
            self.select_features()
            offspring = self.mutation(base_solution)
            offsprings.append(offspring)

        return offsprings

    def teo_selection(self, offsprings, fitness):
        combined_solutions = np.vstack((self.TM, offsprings)) if len(self.TM) > 0 else offsprings
        combined_fitness = np.hstack((self.TM_fitness, fitness)) if len(self.TM_fitness) > 0 else fitness

        best_indices = np.argsort(combined_fitness)[-self.mu:]  # Minimization
        return combined_solutions[best_indices]

    def update_TM(self, offsprings, fitness):
        for offspring, offspring_fitness in zip(offsprings, fitness):
            if offspring_fitness >= 1:
                if len(self.TM) >= self.TM_size:
                    idx = np.random.randint(self.TM_size)
                    if self.TM_fitness[idx] >= offspring_fitness:
                        self.TM[idx] = offspring.copy()
                        self.TM_fitness[idx] = offspring_fitness
                else:
                    self.TM.append(offspring.copy())
                    self.TM_fitness.append(offspring_fitness)

    def run(self):
        offsprings = self.get_offsprings()
        fitness = self.evaluate_fitness(offsprings)
        self.update_TM(offsprings, fitness)

        self.population = self.teo_selection(offsprings, fitness)
