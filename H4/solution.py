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

    action_list = []

    def actions(self) -> List[TAction]:
        """Return the sequence of actions from root to this path head."""
        actions_list = []
        current = self

        while current.parent is not None:
            actions_list.append(current.a)
            current = self.parent
            
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

    Parameters
    ----------
    n0 : initial node (must be hashable)
    succ : successor function returning (action, next_node)
    is_goal : goal predicate on nodes
    better : optional comparator on complete action sequences; returns True iff
             first argument is better than second. If provided, the search does
             NOT stop at first goal; it returns the best goal found.

    Returns
    -------
    list[action] if a solution exists, otherwise None.
    """


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
