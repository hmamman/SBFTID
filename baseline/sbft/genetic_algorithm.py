import numpy as np
import datetime
import math
from typing import Callable, Sequence, Tuple, List, Any, Dict, Optional


class GeneticAlgorithm:
    """
    Search-Based Fairness Testing (SBFT) – genetic algorithm for estimating the *fairness degree*
    of a regression-based ML system (Perera et al., 2022).

    -------------
    Key features
    -------------
    • Solution = full input vector, *including* the sensitive attribute, but genetic operators never
      alter the sensitive gene.
    • Fitness  = |ŷ(x_s=a) – ŷ(x_s=b)| computed on-the-fly; a/b come from a small cached pair that
      is updated whenever a larger difference is found (pc controls cache exploitation).
    • Roulette-wheel parent selection (fitness-proportional).
    • Uniform crossover that copies the sensitive gene unchanged.
    • Two complementary mutation operators:
        – child-1: free uniform mutation in the full variable range;
        – child-2: “bounded” mutation inside the hyper-rectangle defined by its two parents.
    • ri proportion of fresh random test cases injected each generation to avoid stagnation.
    • First-best local search on the current elite solution (±5 % step) every generation.
    """

    def __str__(self) -> str:
        return "SBFT-GA"

    # ------------------------------------------------------------------ #
    #  C O N S T R U C T O R
    # ------------------------------------------------------------------ #
    def __init__(
            self,
            *,
            pop_size: int,
            bounds: Sequence[Tuple[float, float]],
            fitness_func: Callable[[np.ndarray], int],
            sensitive_index: int,
            cross_rate: float = 0.75,
            mut_rate: float = 0.70,
            ri: float = 0.10,
            pc: float = 0.50,
            rng: Optional[np.random.Generator] = None,
    ):
        self.pop_size = pop_size
        self.bounds = np.array(bounds, dtype=int)
        self.D = len(bounds)
        self.s_idx = sensitive_index
        self.s_vals = list(range(self.bounds[sensitive_index][0], self.bounds[sensitive_index][1]))

        # GA & SBFT params
        self.cross_rate = cross_rate
        self.mut_rate = mut_rate
        self.ri = ri
        self.pc = pc
        self.rng = rng or np.random.default_rng()
        self.fitness_func = fitness_func

        # run-time state
        self.pop: np.ndarray = np.empty((pop_size, self.D), dtype=float)
        self.fit: Dict[int, float] = {}
        self.g = 0
        self.start: Optional[datetime.datetime] = None

        # ---- initialise population ----
        self._init_population()
        self._setup()

    # ------------------------------------------------------------------ #
    #  I N I T I A L I S A T I O N
    # ------------------------------------------------------------------ #
    def _init_population(self) -> None:
        self.pop = self.rng.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.pop_size, self.D),
        )
        # clamp sensitive gene to a legal discrete value
        self.pop[:, self.s_idx] = self.rng.choice(self.s_vals, self.pop_size)

        self.fit = self.fitness_func(self.pop)

        # for i in range(self.pop_size):
        #     self.fit[i] = self.fitness_func(self.pop[i])

    # ------------------------------------------------------------------ #
    #  P A R E N T   S E L E C T I O N (roulette wheel)
    # ------------------------------------------------------------------ #

    def _select_parents(self) -> Tuple[np.ndarray, np.ndarray]:
        # 1️⃣  Make sure we are working in float – no silent int arrays.
        # fitness_array = np.array(
        #     [self.fit[i] for i in range(self.pop_size)], dtype=float
        # )
        fitness_array = np.array(
            [self.fit[i] for i in range(self.pop_size)], dtype=float
        )

        # 2️⃣  OPTIONAL: diversity bonus (same for everyone, keeps maths simple)
        #     If you’d rather have *individual* diversity scores, compute a vector
        #     the same size as `fitness_array` instead of this scalar.
        diversity_bonus = float(np.mean(np.std(self.pop, axis=0)))
        fitness_array += diversity_bonus

        # 3️⃣  Sanitise NaN / Inf so they don’t propagate into probabilities.
        fitness_array = np.nan_to_num(fitness_array, nan=0.0, posinf=0.0, neginf=0.0)

        # 4️⃣  Build roulette-wheel probabilities, falling back to uniform if flat.
        total = fitness_array.sum()
        if total == 0.0:
            probs = np.ones(self.pop_size, dtype=float) / self.pop_size
        else:
            probs = fitness_array / total

        # 5️⃣  Sample two distinct parents.
        idxs = self.rng.choice(self.pop_size, 2, p=probs, replace=False)
        return self.pop[idxs[0]].copy(), self.pop[idxs[1]].copy()

    # ------------------------------------------------------------------ #
    #  C R O S S O V E R  &  M U T A T I O N
    # ------------------------------------------------------------------ #
    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() >= self.cross_rate:
            return p1.copy(), p2.copy()

        mask = self.rng.random(self.D) < 0.5
        mask[self.s_idx] = False  # never swap the sensitive gene
        c1, c2 = p1.copy(), p2.copy()
        c1[mask], c2[mask] = p2[mask], p1[mask]
        return c1, c2

    def _mutate_free(self, child: np.ndarray) -> None:
        for j in range(self.D):
            if j == self.s_idx:
                continue
            if self.rng.random() < self.mut_rate:
                child[j] = self.rng.uniform(self.bounds[j, 0], self.bounds[j, 1])

    def _mutate_bounded(self, child: np.ndarray, p_low: np.ndarray, p_high: np.ndarray) -> None:
        for j in range(self.D):
            if j == self.s_idx:
                continue
            if self.rng.random() < self.mut_rate:
                lo, hi = sorted((p_low[j], p_high[j]))
                child[j] = self.rng.uniform(lo, hi)

    # ------------------------------------------------------------------ #
    #  L O C A L   S E A R C H   O N   E L I T E
    # ------------------------------------------------------------------ #
    def _local_search(self, elite: np.ndarray, elite_fit: float) -> Tuple[np.ndarray, float]:
        x = elite.copy()
        best = elite_fit

        idx_order = self.rng.permutation(self.D)
        for j in idx_order:
            if j == self.s_idx:
                continue
            step = 0.05 * (self.bounds[j, 1] - self.bounds[j, 0])
            for delta in (-step, step):
                tmp = x.copy()
                tmp[j] = np.clip(tmp[j] + delta, *self.bounds[j])
                # f = self.fitness_func(tmp)
                f = self.fitness_func([tmp])
                if f > best:
                    x, best = tmp, f
                    break  # first-improvement
        return x, best

    # ------------------------------------------------------------------ #
    #  S I N G L E   G E N E R A T I O N
    # ------------------------------------------------------------------ #
    def _breed(self) -> Tuple[np.ndarray, Dict[int, float]]:
        k_rand = math.ceil(self.ri * self.pop_size)
        offspring: List[np.ndarray] = []

        # main reproductive loop
        while len(offspring) < self.pop_size - k_rand:
            p1, p2 = self._select_parents()
            c1, c2 = self._crossover(p1, p2)

            self._mutate_free(c1)
            self._mutate_bounded(c2, p1, p2)

            offspring.extend([c1, c2])

        #  random insertions
        for _ in range(k_rand):
            child = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            child[self.s_idx] = self.rng.choice(self.s_vals)
            offspring.append(child)

        off = np.stack(offspring[: self.pop_size])
        fitness = self.fitness_func(off)
        off_fit = {i: fit for i, fit in enumerate(fitness)}
        # off_fit = {i: self.fitness_func(ind) for i, ind in enumerate(off)}
        return off, off_fit

    # ------------------------------------------------------------------ #
    #  E V O L U T I O N   L O O P
    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        self.start = datetime.datetime.now()

        # sort initial population by fitness (descending)
        order = np.argsort([-self.fit[i] for i in range(self.pop_size)])
        self.pop = self.pop[order]
        self.fit = {i: self.fit[order[i]] for i in range(self.pop_size)}

        # first elite local search
        elite, elite_f = self._local_search(self.pop[0], self.fit[0])
        self.pop[0], self.fit[0] = elite, elite_f

    # ------------------------------------------------------------------ #
    #  S T O P P I N G   C R I T E R I A
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        offspring, off_fit = self._breed()

        # merge & elitism
        union = np.vstack([self.pop, offspring])
        union_fit_vals = [self.fit[i] for i in range(self.pop_size)] + list(off_fit.values())
        top_idx = np.argsort(-np.array(union_fit_vals))[: self.pop_size]
        self.pop = union[top_idx]

        self.fit = {i: union_fit_vals[top_idx[i]] for i in range(self.pop_size)}

        # local improvement
        elite, elite_f = self._local_search(self.pop[0], self.fit[0])
        self.pop[0], self.fit[0] = elite, elite_f

        self.g += 1
