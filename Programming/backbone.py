#!/usr/bin/env python3
"""Backbones task, specification on http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_backbone.php"""
from cdcl_heuristics import CDCL_watched_solver, get_cnf
from cdcl_heuristics import parse_args as cdcl_parse_args
from typing import Optional, Tuple
import sys
from argparse import ArgumentParser, Namespace
from time import perf_counter_ns


def parse_args(args=sys.argv[1:]) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input", nargs="?", type=str, help="Input file.")
    parser.add_argument(
        "--heuristic",
        type=str,
        default=["vsids", "count"],
        nargs="+",
        choices=["vsids", "count", "neg_count"],
        help="Next backbone choice. Can use more - first highest priority.",
    )
    parser.add_argument(
        "-i",
        "--initial_backbone",
        action="store_false",
        help="Don't initial unit prop as backbone.",
    )
    args = parser.parse_args(args)
    return args


def get_backbones(
    cnf: list[list[int]],
    heuristic: list[str] = ["vsids", "count"],
    max_var: Optional[int] = None,
    initial_backbone: bool = True,
) -> Tuple[list[int], int, float]:
    """
    Return backbones, sat calls, time in s.

    ***
    Params:
    - heuristics - next literal to check selection heuristic
    - max_var - optional max_variable
    - initial_backbone - whether to skip checking of literals
        appearing in unit clauses of given cnf
    """
    start = perf_counter_ns()

    solver = CDCL_watched_solver(cnf, max_var)
    max_var = solver.max_var
    sat, model = solver.solve()
    calls = 1

    if not sat:
        # unsat - no backbone
        return [], calls, (perf_counter_ns() - start) / 1_000_000_000

    # first unit prop backbone
    backbone = set(solver.initial_backbone) if initial_backbone else set()
    to_check = list(
        map(
            lambda x: -x,
            filter(lambda x: x not in backbone, model),
        )
    )
    to_check_set = set(to_check)

    # only to have lit_to_clause_indices without learned clauses
    solver.delete_learned_clauses(del_all=True)
    # sort by each of heuristics -> first has highest priority
    for h in reversed(heuristic):
        if h == "vsids":
            vsids = (
                solver.counters[0].tolist()
                + solver.counters[1, -1:0:-1].tolist()
            )
            to_check.sort(key=lambda x: vsids[x])
        elif h == "count":
            to_check.sort(key=lambda x: len(solver.lit_to_clause_indices[x]))
        elif h == "neg_count":
            # sort by lowest appearance of opposite literal
            to_check.sort(
                key=lambda x: len(solver.lit_to_clause_indices[-x]),
                reverse=True,
            )
        else:
            raise ValueError("Unknown heuristic.")

    while to_check:
        lit = to_check.pop()
        if lit not in to_check_set:
            continue
        to_check_set.remove(lit)

        # add unit clause with checked literal
        cnf.append([lit])
        solver = CDCL_watched_solver(cnf, max_var)
        sat, model = solver.solve()
        calls += 1
        if sat:
            cnf.pop()
            to_check_set.difference_update(model)
        else:
            backbone.add(-lit)

    time = (perf_counter_ns() - start) / 1_000_000_000
    return sorted(backbone, key=abs), calls, time


if __name__ == "__main__":
    args = parse_args()
    cnf, max_var = get_cnf(cdcl_parse_args([args.input]))
    bb, calls, time = get_backbones(
        cnf, args.heuristic, max_var, args.initial_backbone
    )
    print("Backbone: {}\nTime: {} s\nSolver calls: {}".format(bb, calls, time))
