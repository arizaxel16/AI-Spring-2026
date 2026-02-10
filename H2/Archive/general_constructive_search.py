class GeneralConstructiveSearch:
    """
    Implements a general constructive search with BFS/DFS exploration order.
    """

    def __init__(self, expand, goal, better=None, order="dfs"):
        """
        Args:
            expand (Callable): Function that returns successors of a node.
            goal (Callable): Function that returns True if a node is a goal.
            better (Callable, optional): Comparator; True if first arg is strictly better.
            order (str): "dfs" (stack) or "bfs" (queue).
        """
        self.initial = ()
        self.expand_func = expand
        self.goal_func = goal
        self.better_func = better
        self.order_str = order
        self.reset()

    def reset(self):
        self.OPEN = [self.initial]
        self._best = None

    def step(self):
        """
        One logical iteration: pop a node, expand it, check children for goal.
        Returns True iff a new best solution was found.
        """
        if not self.active:
            return False

        # Selection
        if self.order_str == "dfs":
            node = self.OPEN.pop()
        else:
            node = self.OPEN.pop(0)

        # Expansion
        children = self.expand_func(node)

        # Process children: goal nodes update best, non-goal nodes go to OPEN
        found_new_best = False
        non_goal = []
        for child in children:
            if self.goal_func(child):
                if self._best is None:
                    self._best = child
                    found_new_best = True
                elif self.better_func is not None and self.better_func(child, self._best):
                    self._best = child
                    found_new_best = True
            else:
                non_goal.append(child)

        self.OPEN.extend(non_goal)
        return found_new_best

    @property
    def active(self) -> bool:
        if not self.OPEN:
            return False
        if self._best is not None and self.better_func is None:
            return False
        return True

    @property
    def best(self):
        return self._best


def encode_problem(domains, constraints, better=None, order="dfs"):
    """
    Derives expand, goal, better from (D, C, >) for fixed number of variables.
    Nodes are dicts mapping variable -> value.
    """
    variables = sorted(domains.keys())
    n_vars = len(variables)

    def local_expand(node):
        assigned = len(node)
        if assigned >= n_vars:
            return []
        current_var = variables[assigned]
        children = []
        for val in domains[current_var]:
            child = dict(node)
            child[current_var] = val
            if constraints(child):
                children.append(child)
        return children

    def local_goal(node):
        return len(node) == n_vars

    search = GeneralConstructiveSearch(local_expand, local_goal, better, order)
    search.initial = {}
    search.reset()
    return search