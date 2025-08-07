
from tutorials.algorithms.phs import PHS


class SIFT(PHS):
    def __init__(self, mu, bounds, fitness_func, alpha=0.7, TM_size=200, feature_size_rate=10):
        super().__init__(mu, bounds, fitness_func, alpha, TM_size, feature_size_rate)

    def evaluate_fitness(self, offsprings):
        return self.fitness_func(offsprings)

