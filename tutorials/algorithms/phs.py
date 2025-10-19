import math
import random

import numpy as np


class PHS:
    def __init__(self, mu, bounds, fitness_func, alpha=0.7, TM_size=200, feature_size_rate=10):
        self.mu = mu
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.dimension = len(self.bounds)
        self.alpha = alpha

        self.features_size = max(1, max(3, self.dimension // feature_size_rate))
        self.features_to_select = list(range(self.dimension))
        self.feature_ranges = self.bounds[:, 1] - self.bounds[:, 0]
        self.feature_to_mutate = None

        self.TM = []
        self.TM_fitness = []
        self.TM_size = TM_size

        self.initialize_population()

    def initialize_population(self):
        self.population = []

        # Generate the entire population randomly
        for _ in range(self.mu):
            solution = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            self.population.append(solution)

        self.population = np.array(self.population)

    def select_features(self):
        self.feature_to_mutate = np.random.choice(self.features_to_select, self.features_size, replace=False)

    def memory_based_search(self, base_solution):
        if not self.TM:
            return self.population_based_search(base_solution)

        solution = base_solution.copy()
        # Generate mutations of -1, 1, or 0 for each selected feature
        mutation_step_sizes = np.random.choice([-1, 1, 0], size=len(self.feature_to_mutate))

        template = self.TM[np.random.randint(len(self.TM))]
        offspring = template.copy()

        offspring[self.feature_to_mutate] = solution[self.feature_to_mutate] + mutation_step_sizes

        return np.clip(offspring, self.bounds[:, 0], self.bounds[:, 1])

    def population_based_search(self, base_solution):
        offspring = base_solution.copy()

        # Generate mutations of -1 or 1 for each selected feature
        mutation_step_sizes = np.random.choice([-1, 1], size=len(self.feature_to_mutate))

        offspring[self.feature_to_mutate] += mutation_step_sizes
        return np.clip(offspring, self.bounds[:, 0], self.bounds[:, 1])

    def generate_offsprings(self, base_solution):
        if self.TM and np.random.rand() < self.alpha:
            search_strategy = self.memory_based_search
        else:
            search_strategy = self.population_based_search

        offsprings = []

        self.select_features()
        offspring = search_strategy(base_solution)
        offsprings.append(offspring)
        return offsprings

    def get_offsprings(self):
        all_offsprings = []
        for base_solution in self.population:
            offsprings = self.generate_offsprings(base_solution)
            all_offsprings.extend(offsprings)
        return np.array(all_offsprings)

    def teo_selection(self, offsprings, fitness):
        combined_solutions = np.vstack((self.TM, offsprings)) if len(self.TM) > 0 else offsprings
        combined_fitness = np.hstack((self.TM_fitness, fitness)) if len(self.TM_fitness) > 0 else fitness

        best_indices = np.argsort(combined_fitness)[-self.mu:]
        return combined_solutions[best_indices]

    def update_TM(self, offsprings, fitness):
        for offspring, offspring_fitness in zip(offsprings, fitness):
            if offspring_fitness >= 1:
                if len(self.TM) >= self.TM_size:
                    idx = np.random.randint(self.TM_size)
                    if offspring_fitness >= self.TM_fitness[idx]:
                        self.TM[idx] = offspring.copy()
                        self.TM_fitness[idx] = offspring_fitness
                else:
                    self.TM.append(offspring.copy())
                    self.TM_fitness.append(offspring_fitness)

    def evaluate_fitness(self, offsprings):
        fitness = np.array([self.fitness_func(c) for c in offsprings])
        return fitness

    def run(self):
        offsprings = self.get_offsprings()
        fitness = self.evaluate_fitness(offsprings)
        self.update_TM(offsprings, fitness)

        self.population = self.teo_selection(offsprings, fitness)
