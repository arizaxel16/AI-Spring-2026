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
        pass

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
        if self._best is not None and self.better_func is None:
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

    Args:
        domains (dict): Mapping of variable names to domain lists.
        constraints (Callable): Function that returns True if partial assignment is valid.
        better (Callable): Function that compares two full assignments.
        order (str): Exploration strategy ("dfs" or "bfs").

    Returns:
        GeneralConstructiveSearch: Configured search object.
    """
    raise NotImplementedError("You must implement 'encode_problem'")
