from collections import deque


class GeneralConstructiveSearch:
    def __init__(self, w0, succ, goal, better=None, order="dfs"):
        if order not in {"dfs", "bfs"}:
            raise ValueError("order must be 'dfs' or 'bfs'")
        self.w0 = w0
        self.succ = succ
        self.goal = goal
        self.better = better
        self.order = order
        self.reset()

    def reset(self):
        self.OPEN = deque([self.w0]) if self.order == "bfs" else [self.w0]
        self.best_solution = None
        self.solution_count = 0

    @property
    def best(self):
        return self.best_solution

    @property
    def active(self):
        if self.best_solution is not None and self.better is None:
            return False
        return len(self.OPEN) > 0

    def step(self):
        if not self.active:
            return False

        node = self.OPEN.popleft() if self.order == "bfs" else self.OPEN.pop()
        found_solution = False
        if self.goal(node):
            if self.better is None:
                self.best_solution = node
                self.solution_count += 1
                return True
            if self.best_solution is None or self.better(node, self.best_solution):
                self.best_solution = node
            self.solution_count += 1
            found_solution = True

        for successor in self.succ(node):
            if self.goal(successor):
                if self.better is None:
                    self.best_solution = successor
                    self.solution_count += 1
                    return True
                if self.best_solution is None or self.better(successor, self.best_solution):
                    self.best_solution = successor
                self.solution_count += 1
                found_solution = True
            else:
                self.OPEN.append(successor)

        return found_solution


def encode_problem(domains, constraints, better, order="dfs", w0=None):
    if w0 is None:
        w0 = {k: values[0] for k, values in domains.items() if len(values) == 1}

    def expand(n):
        # identify next variable to fix
        next_var = None
        for varname in domains.keys():
            if varname not in n:
                next_var = varname
                break

        successors = []
        if next_var is None:
            return successors
        for v in domains[next_var]:
            new_assignment = n.copy()
            new_assignment[next_var] = v
            if constraints is None or constraints(new_assignment):
                successors.append(new_assignment)
        return successors

    def is_goal(n):
        return len(n) == len(domains)

    return GeneralConstructiveSearch(
        w0=w0, succ=expand, goal=is_goal, better=better, order=order
    )
