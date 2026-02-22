from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
from collections import deque
import heapq


TNode = TypeVar("TNode", bound=Hashable)
TAction = TypeVar("TAction")


@dataclass
class Path(Generic[TNode, TAction]):
    """
    A memory-efficient path representation storing parent pointers.

    head   : the current node/state
    parent : previous Path object (None for root)
    a      : action taken in parent to reach head
    g      : path cost from root to head (only used by UCS; defaults to 0.0)
    """

    head: TNode
    parent: Optional["Path[TNode, TAction]"] = None
    a: Optional[TAction] = None
    g: float = 0.0

    def actions(self) -> List[TAction]:
        """Return the sequence of actions from root to this path head."""
        actions_list = []
        current = self

        while current.parent is not None:
            actions_list.append(current.a)
            current = current.parent
        return actions_list[::-1]



def general_graph_search(
    n0: TNode,
    succ: Callable[[TNode], Iterable[Tuple[TAction, TNode]]],
    is_goal: Callable[[TNode], bool],
    better: Optional[Callable[[Sequence[TAction], Sequence[TAction]], bool]] = None,
) -> Optional[List[TAction]]:
    """
    General Graph Search for path-independent solution quality.

    - Uses CLOSED to ensure we never expand the same node twice.
    - Ensures at most one path to each node is on OPEN at any time.
    - OPEN is FIFO (Remove-First), i.e., graph-search BFS order.
    """
    root_path = Path(n0)
    if is_goal(n0):
        return []

    OPEN = deque([root_path])
    CLOSED = set()
    nodes_on_open = {n0}
    best_actions: Optional[List[TAction]] = None

    while OPEN:
        p = OPEN.popleft()
        if p.head in nodes_on_open:
            nodes_on_open.remove(p.head)

        if p.head in CLOSED:
            continue

        CLOSED.add(p.head)

        if is_goal(p.head):
            current_actions = p.actions()
            if best_actions is None or (better and better(current_actions, best_actions)):
                best_actions = current_actions

            if better is None:
                return best_actions

        for action, next_node in succ(p.head):
            if next_node not in CLOSED and next_node not in nodes_on_open:
                new_path = Path(next_node, p, action)
                OPEN.append(new_path)
                nodes_on_open.add(next_node)

    return best_actions


def uniform_cost_search(
    n0: TNode,
    succ: Callable[[TNode], Iterable[Tuple[TAction, TNode]]],
    is_goal: Callable[[TNode], bool],
    cost: Callable[[TNode, TNode], float],
) -> Optional[List[TAction]]:
    """
    Uniform Cost Search (Dijkstra-style) with parent discarding.

    - OPEN is ordered by g-cost (lowest first).
    - CLOSED contains expanded heads.
    - Parent discarding: if a cheaper path to a node already on OPEN is found,
      update its parent and g.

    Assumes non-negative edge costs (ideally strictly positive as in slides).

    Returns
    -------
    list[action] if a solution exists, otherwise None.
    """
    # 1. Initial Path: Remember that Path objects have a 'g' attribute.
    # For the root, g is 0.0.
    root_path = Path(n0, g=0.0)

    if is_goal(n0):
        return []

    # 2. Data Structures:
    # OPEN must be a list for heapq.
    OPEN = []

    # CLOSED stays a set of expanded nodes.
    CLOSED = set()

    # REPLACING nodes_on_open: Use a dictionary to track the best g-cost
    # found for each node. This is the heart of Parent Discarding.
    best_g = {n0: 0.0}

    # 3. Initial Push: heapq.heappush(OPEN, (priority, item))
    # UCS prioritizes the lowest g-cost.
    heapq.heappush(OPEN, (root_path.g, root_path))

    while OPEN:
        # 4. Pop: Remember heappop returns the tuple (cost, path)
        # You need to unpack both.
        g_cost, p = heapq.heappop(OPEN)

        # 5. The CLOSED Check:
        # Because we might push multiple paths to the same node onto the heap,
        # if we pop a node that is already in CLOSED, skip it (continue).
        if p.head in CLOSED:
            continue

        # 6. Goal Test:
        # In UCS, you MUST check for the goal ONLY when the node is popped.
        # If is_goal(p.head), return the actions immediately.
        if is_goal(p.head):
            return p.actions()

        CLOSED.add(p.head)

        # 7. Expansion:
        # Loop through successors: for action, next_node in succ(p.head):
        for action, next_node in succ(p.head):

            # A. Calculate new_g: current path's g + cost(from, to)
            new_g = g_cost + cost(p.head, next_node)

            # B. The "Parent Discarding" Logic:
            # Check if next_node is in CLOSED. If it is, skip it.
            if next_node in CLOSED:
                continue
            # If next_node is NOT in best_g OR new_g is cheaper than best_g[next_node]:
            if next_node not in best_g or new_g < best_g[next_node]:

                # - Update best_g[next_node] with the new cheaper cost
                best_g[next_node] = new_g
                # - Create new_path with next_node, p, action, and g=new_g
                new_path = Path(next_node, p, action, g=new_g)
                # - Push (new_path.g, new_path) onto the OPEN heap
                heapq.heappush(OPEN, (new_path.g, new_path))

    return None # Return None if the loop finishes without finding a goal