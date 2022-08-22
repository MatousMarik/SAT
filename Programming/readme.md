# Decision procedures and verification - Programming tasks

Solutions for tasks from http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/prg_tasks.php

## Building SAT solver

### 1. [Tseitin Encoding and DIMACS Format](http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_tseitin.php)
Converts NNF formula from simplified SMT-LIB format to DIMACS.

Run `python3 formula2cnf.py [input [output]] [-e]` to get DIMACS encoding. You can specify `--equivalences` for CNF with equivalences otherwise only left-to-right implications are returned.

### 2. [DPLL Algorithm](http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_dpll.php)
- Write DPLL solver.
### 3. [Watched Literals](http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_watched.php)
- Add watched literals.
### 4. [CDCL Algorithm](http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_cdcl.php)
- Add clause learning.
### 5. [Decision Heuristics](http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_decision.php)
- Add decision heuristics - VSIDS.

Run `python3 dpll[_watched[_cache]].py [input [output]] [-s | -c] [-v]` to get whether formula is satisfiable and optionally its model. By `--sat`, or `--cnf` you can specify input format (for the case file suffix is not ".sat" or ".cnf" or whole input sequence is passed). Use `--verbose` to obtain statistics.

Run 
```
python3 cdcl.py [input [output]] [-s | -c] [-v]
                [--conflict_limit=128]
                [--lbd_limit=3]
                [-a/--assumption [a1 ...]]
                [--heuristic=vsids
                   {default,vsids,random,unsatisfied}
                ]
```
where you can specify initial limits and assupmptions and select heuristic for CDCL solver.

See [report](./report.md).

---
---
## Using SAT solver

### [N Queens Puzzle](http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_n_queens.php)

Run `python3 n_queens.py n` to obtain formula for `n` queens puzzle, where `n` stands for number of queens.

See [report](./using_solver_report.md#n-queens-puzzle)

---
### [Backbones](http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_backbone.php)

Run `python3 backbone.py [input] [--heuristic {vsids,count,neg_count}]` to obtain backbone of given formula. You can specifiy more heuristics, prior has priority and the rest is used for ties.

See [report](./using_solver_report.md#backbones)



