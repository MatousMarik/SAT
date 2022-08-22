# Using SAT solver report

## N Queens Puzzle
Find how to place *n* queens to *n x n* chessboard such that no two queens attack each other.

Formula is basicaly list of 3 conditions:
1. There is at least one queen in each row and col.
2. There is at most one queen in each row and col.
3. There is at most one queen on each diagonal.

Solving times for selected 3 solvers, where CDCL is created solver, and the other two are solvers provided by module `pysat`.

| N | CDCL | Glucose4 | Lingeling |
| --- | ---: | ---: | ---: |
| 140 | 42.3953016 | 35.1629159 | 1.2975168 |
| 148 | 43.8990835 | 37.2563885 | 0.6711859 |
| 149 | 58.0692158 | 51.0878447 | 1.7974821 |
| 150 | 60.3326459 | 39.9800227 | 1.5366763 |
| 160 | None | 41.5376908 | 1.7864682 |
| 170 | None | 53.238811 | 5.6790878 |
| 180 | None | 51.1570156 | 6.7237785 |
| 190 | None | 49.5775504 | 24.7912377 |
| 200 | None | 101.9919136 | 7.6854476 |
| 210 | None | None | 5.6858987 |
| 220 | None | None | 2.2958123 |
| 250 | None | 119.81755 | 12.4957197 |

Unfortunatelly since Windows subsystem for Linux were used to run `pysat` module, python process is killed for too great memory usage. Also because of that performance results are probably only approximate.

### Conclusion
Lingeling is the best solver by far, created solver is comparable with Glucose4. Creation of the formula is expensive on time and memory (in the last case there is 250<sup>2</sup> variables and roughly 250<sup>3</sup> clauses).

---

## Backbones

Let the solver find some model and then try opposite literals to find another models. If any opposite literal appear during the search, it is not part of the backbone.
Also ads proven backbone to formula as unit clauses.
- **h1** is a heuristic that sorts literals by 1. VSIDS and 2. number of causes with that literal
- **h2** is a heuristic that sorts literals by lowest number of clauses with opposite literal

| test      |   h1 calls |       h1 time |   h2 calls |   h2 time |
|:----------|---------:|------------:|---------:|------------:|
| uf20-91   |  17      |   0.0073041 |  19.6667 |   0.0122868 |
| uf50-218  |  51      |   0.0904224 |  48.6667 |   0.110082  |
| uf75-325  |  73      |   0.266078  |  52      |   0.436073  |
| uf100-430 |  66.3333 |   0.654802  |  64.6667 |   1.49009   |
| uf125-538 |  93      |   1.24982   |  97.3333 |   3.5125    |
| uf150-645 | 101      |   8.08261   |  81.3333 |  23.3839    |
| uf200-860 | 159.333  | 175.742     | 145      | 151.923     |

The time has to be taken as approximate.

### Conclusion
Heuristic plays key role in backbone algorithm. Adding known backbone to formula allows solver to operate much faster as the 100 calls would normaly take about 100 seconds.

