class GeneralConstructiveSearch:
    """
    Implements a general constructive search with BFS/DFS exploration order.

    Attributes:
        order (str): Search order strategy ("dfs" or "bfs").
        best (Any): Best known solution found during the search.
        active (bool): True if the search can continue, False otherwise.
    """

    def __init__(self, expand, goal, better=None, order="dfs"):
        """
        Initializes the search instance.

        Args:
            expand (Callable): Function that returns successors of a node.
            goal (Callable): Function that returns True if a node is a goal.
            better (Callable, optional): Comparator that returns True if first arg is better.
            order (str): Exploration strategy ("dfs" or "bfs") to arrange OPEN.

        Notes:
            - The search starts from an empty initial node by default.
        """
        # Store the "Rules of the Game"
        self.expand_func = expand
        self.goal_func = goal
        self.better_func = better
        self.order_str = order

        # Set the starting state
        self.reset()

    def reset(self):
        """
        Resets the search to its initial configuration.
        Useful for re-running the search from scratch.
        """

        # These lines "declare" and "initialize" the variables simultaneously
        self.OPEN = [()]
        self._best = None

    def step(self):
        """
        Executes a single step in the search.

        Returns:
            bool: True iff a new best solution is found; False otherwise.
        """
        # 1. Safety Check: If we aren't active, don't do anything.
        if not self.active:
            return False

        # 2. Selection
        if self.order_str == "dfs":
            node = self.OPEN.pop()
        else:
            node = self.OPEN.pop(0)

        # 3. Goal Test
        if self.goal_func(node):
            is_new_best = False

            # If we have no solution yet, this is the best by default
            if self.best is None:
                is_new_best = True
            # If we have a solution AND a way to compare, check it
            elif self.better_func is not None and self.better_func(node, self.best):
                is_new_best = True

            if is_new_best:
                self._best = node
                return True
            return False

        # 4. Expansion
        children = self.expand_func(node)
        self.OPEN.extend(children)
        return False

    @property
    def active(self) -> bool:
        """
        Indicates whether the search is still ongoing.
        It can stop early after finding the first solution if better is None.

        Returns:
            bool: True if there are nodes left to explore.
        """
        # Rule 1: If there's nothing left to explore, we are definitely not active.
        if not self.OPEN:
            return False

        # Rule 2: If we found a solution AND we aren't looking for a "better" one, stop.
        if self.best is not None and self.better_func is None:
            return False

        # If neither of the above is true, we keep going!
        return True

    @property
    def best(self):
        """
        Returns the current best solution.

        Returns:
            Any: The best node found so far.
        """
        return self._best


def encode_problem(domains, constraints, better, order="dfs"):
    """
    Encodes a fixed-variable problem as a GeneralConstructiveSearch.
    """
    # Get the list of variable names once so we have a consistent order
    var_names = list(domains.keys())
    num_vars = len(var_names)

    def local_goal(node):
        """A goal is reached when all variables have an assignment."""
        return len(node) == num_vars

    def local_expand(node):
        """Generates valid assignments for the next available variable."""
        idx = len(node)

        # If we have assigned all variables, there's nothing more to expand
        if idx >= num_vars:
            return []

        # Identify the next variable and its possible values (the domain)
        current_var = var_names[idx]
        options = domains[current_var]

        # Professional way: Create the list of valid children in one clean step
        # This says: "Give me node + val for every val in options IF it's legal"
        return [node + (val,) for val in options if constraints(node + (val,))]

    # Return the "Engine" configured with these custom rules
    return GeneralConstructiveSearch(local_expand, local_goal, better, order)