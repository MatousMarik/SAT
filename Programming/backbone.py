#!/usr/bin/env python3
"""Backbones task, specification on http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_backbone.php"""
from formula2cnf import get_cnf
from cdcl import CDCL_watched_solver
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
        help="""Next backbone choice. Can use more - first highest priority.
        vsids - decaying sum from initial solve,
        count - number of clauses with literal
        neg_count - lowest number of clauses with opposite literal
        """,
    )
    parser.add_argument(
        "-i",
        "--initial_backbone",
        action="store_false",
        help="Don't use initial unit prop as backbone.",
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

    Get some model then try to add opposite literals
    as unit clauses and solve that new cnf to see whether they are backbone.

    Heuristics are:
    vsids - decaying sum from initial solve,
    count - number of clauses with literal
    neg_count - lowest number of clauses with opposite literal

    ***
    Params:
    - heuristics - next literal to check selection heuristic
        Note: vsids heuristic exploits solver features
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

    # solver.lit_to_clause_indices could be used but then no other solver could be used
    counts = [0] * (max_var * 2 + 1)
    for c in cnf:
        for l in c:
            counts[l] += 1

    # sort by each of heuristics -> first has highest priority
    for h in reversed(heuristic):
        if h == "vsids":
            vsids = (
                solver.counters[0].tolist()
                + solver.counters[1, -1:0:-1].tolist()
            )
            to_check.sort(key=lambda x: vsids[x])
        elif h == "count":
            to_check.sort(key=lambda x: counts[x])
        elif h == "neg_count":
            to_check.sort(
                key=lambda x: counts[-x],
                reverse=True,
            )
        else:
            raise ValueError("Unknown heuristic.")

    while to_check:
        lit = to_check.pop()
        if lit not in to_check_set:
            continue
        to_check_set.remove(lit)

        # add unit clause with checked literal and solve new cnf
        cnf.append([lit])
        solver = CDCL_watched_solver(cnf, max_var)
        sat, model = solver.solve()
        calls += 1

        if sat:
            # remove added unit clause
            cnf.pop()
            # mark all literals that are to be checked but already are in the new model
            # as definitely not part of backbone
            to_check_set.difference_update(model)
        else:
            backbone.add(-lit)

    time = (perf_counter_ns() - start) / 1_000_000_000
    return sorted(backbone, key=abs), calls, time


if __name__ == "__main__":
    args = parse_args()
    cnf, max_var = get_cnf(args.input)
    bb, calls, time = get_backbones(
        cnf, args.heuristic, max_var, args.initial_backbone
    )
    print("Backbone: {}\nTime: {} s\nSolver calls: {}".format(bb, calls, time))
