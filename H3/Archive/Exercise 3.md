Exercise 3 Answer

1. Why the BCN has no solution:

We have X1, X2, X3 each with domain {0, 1} and all-diff: X1 != X2, X1 != X3, X2 != X3. By the pigeonhole principle, 3 variables cannot all take distinct values from a domain
of only 2 values. Formally: any assignment maps {X1, X2, X3} -> {0, 1}, so by pigeonhole at least two variables must receive the same value, violating some all-diff          
constraint. Therefore no solution exists.

2. What AC-3 concludes — and why:

AC-3 concludes feasible (incorrectly). When it checks arc (X1, X2) with constraint X1 != X2: for every value v in D(X1) = {0, 1}, there exists a compatible value in D(X2) (0
is compatible with 1, and 1 is compatible with 0). No domain gets pruned. The same holds for all 6 arcs. AC-3 terminates with all domains unchanged and reports feasible. This
demonstrates the fundamental limitation of arc consistency: it only checks binary (pairwise) consistency and cannot detect infeasibility that requires reasoning about 3+
variables simultaneously. Detecting this requires a stronger form of consistency (e.g., path consistency or k-consistency for k >= 3).
