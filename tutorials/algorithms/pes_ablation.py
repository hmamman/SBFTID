import os
import sys

import numpy as np
# Get the absolute path to the directory where ftat.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Three levels up from ftat.py
sys.path.append(os.path.join(base_path, "../../"))

from tutorials.algorithms.ses import SES


class PES(SES):
    def __init__(self, mu, lambda_, sigma, bounds, fitness_func, feature_size_rate=2):
        super().__init__(mu, lambda_, sigma, bounds, fitness_func)

        self.population_fitness = self.evaluate_fitness(self.population)
        self.features_size = max(1, self.dimension // feature_size_rate)

    def select_features(self):
        self.feature_to_mutate = np.random.choice(self.features_to_select, self.features_size, replace=False)

    def generate_offsprings(self, base_solution):
        offsprings = []

        for _ in range(self.lambda_):
            self.select_features()
            offspring = self.mutation(base_solution)
            offsprings.append(offspring)

        return offsprings

    def selection(self, offsprings, fitness):
        combined_population = np.vstack((offsprings, self.population))
        combined_fitness = np.hstack((fitness, self.population_fitness))

        best_indices = np.argsort(combined_fitness)[-self.mu:]  # Minimization
        parents = combined_population[best_indices]
        parents_fitness = combined_fitness[best_indices]
        return parents, parents_fitness

    def evaluate_fitness(self, offsprings):
        fitness = np.array(self.fitness_func(offsprings))
        return fitness

    def run(self):
        offsprings = self.get_offsprings()
        fitness = self.evaluate_fitness(offsprings)

        self.population, self.population_fitness = self.selection(offsprings, fitness)
