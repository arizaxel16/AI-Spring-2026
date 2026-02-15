import functools


class GeneticSearch:
    """
    Optimizes a set of candidates explorable through crossovers and mutations.

    Which candidate is better is determined with the `better` function
    """

    def __init__(self, init, crossover, mutate, better, population_size):
        self.init = init
        self.crossover = crossover
        self.mutate = mutate
        self.better = better
        self.population_size = population_size

        # state variables
        self.population = None
        self._best = None
        self.num_solutions = 0

    def reset(self):
        """
        Initializes the population and internal best candidate.
        """
        self.population = self.init(self.population_size)
        self._best = None
        self.num_solutions = len(self.population)
        for ind in self.population:
            if self._best is None or self.better(ind, self._best):
                self._best = ind

    @property
    def best(self):
        return self._best

    @property
    def active(self):
        """
        Returns whether the search should continue.
        """
        return True

    def step(self):
        """
        Executes one GA generation.

        For every pair (i < j) in the population, produce two offspring via
        crossover, then mutate each offspring (unless mutate is None).
        Combine parents + offspring and keep the best population_size individuals.

        Returns:
            bool: True if global best improved in this step, else False.
        """
        pop = self.population
        offspring = []
        for i in range(len(pop)):
            for j in range(i + 1, len(pop)):
                children = self.crossover(pop[i], pop[j])
                if self.mutate is not None:
                    children = [self.mutate(c) for c in children]
                offspring.extend(children)

        self.num_solutions += len(offspring)

        # Check if any offspring improves global best
        improved = False
        for ind in offspring:
            if self._best is None or self.better(ind, self._best):
                self._best = ind
                improved = True

        # Elitist selection: keep best population_size from parents + offspring
        combined = pop + offspring
        combined.sort(
            key=functools.cmp_to_key(
                lambda a, b: -1 if self.better(a, b) else (1 if self.better(b, a) else 0)
            )
        )
        self.population = combined[: self.population_size]

        return improved
