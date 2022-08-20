#!/usr/bin/env python3
"""Watched DPLL Algorithm. Specification at http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_watched.php"""
from dataclasses import dataclass
from formula2cnf import get_cnf, write_output
from argparse import ArgumentParser, Namespace
from time import perf_counter_ns
from typing import List, Optional
from itertools import chain
import random
import sys


def parse_args(args=sys.argv[1:]) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input", nargs="?", type=str, help="Input file.")
    parser.add_argument("output", nargs="?", type=str, help="Output file.")
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "-s",
        "--sat",
        action="store_const",
        const="SAT",
        dest="format",
        help="Set input format to SMT-LIB (default). Will be automatically overwritten if input file is .sat or .cnf.",
    )
    g.add_argument(
        "-c",
        "--cnf",
        action="store_const",
        const="CNF",
        dest="format",
        help="Set input format to DIMACS. Will be automatically overwritten if input file is .sat or .cnf.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Output CPU time, number of decisions and number of unit propagation steps.",
    )
    args = parser.parse_args(args)
    return args


@dataclass
class WatchedClause:
    """Clause with watchers (list[2 ints])."""

    ci: int  # clause index
    list: list[int]  # list of clause literals
    lbd: int = 0
    watchers: List[int] = None  # two watcher indices into list

    def __post_init__(self):
        """
        If watchers are not provided for initialization
        set them to first indices.
        """
        if self.watchers is None:
            if len(self.list) > 1:
                self.watchers = [0, 1]
            else:
                self.watchers = [0, 0]

    @property
    def w1(self) -> int:
        """Literal watched by first watcher."""
        return self.list[self.watchers[0]]

    @property
    def w2(self) -> int:
        """Literal watched by second watcher."""
        return self.list[self.watchers[1]]

    def watches_more(self) -> bool:
        """Whether watchers watch different literals."""
        self.watchers[0] != self.watchers[1]

    def __hash__(self) -> int:
        return self.ci

    def watch(
        self, assignment: list[bool], literal: int
    ) -> tuple[bool, int, Optional[int]]:
        """
        Return if satisfiable, new watched literal and optionally unit_literal.
        """
        w1, w2 = self.watchers
        # swap so it works only with w1
        if self.list[w2] == -literal:
            w2, w1 = w1, w2

        # search for another watchable
        for w in chain(range(w1 + 1, len(self.list)), range(0, w1 + 1)):
            # it can't be same as w2 and it has to be satisfiable
            if w != w2 and assignment[self.list[w]] != False:
                break

        # not found - only w2 can satisfy
        if w == w1:
            if assignment[self.list[w2]] == False:
                # w2 not satisfied => unsat
                return False, self.list[w1], None
            else:
                # w2 satisfiable => unit literal
                return True, self.list[w1], self.list[w2]

        # update watchers
        self.watchers[:] = w2, w
        return True, self.list[w], None


class DPLL_watched_solver:
    def __init__(self, cnf: list[list[int]], max_var: int) -> None:
        start = perf_counter_ns()
        self.cnf = cnf
        self.max_var = max_var
        # empty watcher list (for each literal -> [clauses])
        self.w_lists: list[list[WatchedClause]] = list(
            [] for _ in range(max_var * 2 + 1)
        )

        # satisfied literals, bool=True if it is first choice and second was not tried
        self.assigned: list[(int, bool)] = []
        # [None] + [values of each literal]
        self.assignment: list[Optional[bool]] = [None] * (max_var * 2 + 1)
        # unassigned variables
        self.unassigned: set[int] = set(range(1, max_var + 1))

        self.initial_unit_literals = None
        self.generate_watched_lists()
        self.counters = [0] * (max_var * 2 + 1)

        self.decisions: int = 0
        self.unit_prop_steps: int = 0
        self.solve_time: int = -1
        self.initialization_time: int = perf_counter_ns() - start

    def generate_watched_lists(self) -> None:
        """Create watched lists - list to clause with watched literal and list of literals from unit clauses."""
        unit_literals = []
        for clause in self.cnf:
            ac = WatchedClause(clause)
            self.w_lists[-clause[0]].append(ac)
            if len(clause) == 1:
                unit_literals.append(clause[0])
            else:
                self.w_lists[-clause[1]].append(ac)
        self.initial_unit_literals = unit_literals

    def rollback(self, literal: Optional[int]) -> None:
        """Unassign all last assigned literals upto 'literal'."""
        if literal is None:
            literal, _ = self.assigned[0]

        while True:
            lit, _ = self.assigned.pop()
            self.unassigned.add(abs(lit))

            # update assignment
            self.assignment[lit], self.assignment[-lit] = None, None

            if lit == literal:
                break

    def unit_prop(self, literal: Optional[int] = None):
        """Set literal satisfied and do unit_propagation."""
        if literal is None:
            u_literals = self.initial_unit_literals
        else:
            self.decisions += 1
            u_literals = [literal]

        first = False

        need_rollback = False
        while u_literals:
            lit = u_literals.pop()
            if abs(lit) not in self.unassigned:
                continue

            self.counters[-lit if first else lit] += 1

            self.unit_prop_steps += 1
            # manage assignment of newly propagated literals
            self.assignment[lit], self.assignment[-lit] = True, False
            # 'pop_all' watched for literal
            watched, self.w_lists[lit] = self.w_lists[lit], []
            while watched:
                watched_c = watched.pop()
                # find if satisfiable, new_watched, unit_literal
                sat, new_wl, new_ul = watched_c.watch(self.assignment, lit)
                self.w_lists[-new_wl].append(watched_c)

                if not sat:
                    # need to not loose not processed watchers
                    self.w_lists[lit].extend(watched)
                    # reset assignment
                    self.assignment[lit], self.assignment[-lit] = None, None
                    if need_rollback:
                        self.rollback(literal)
                    return False
                elif new_ul is not None and abs(new_ul) in self.unassigned:
                    # append new unit_literal
                    u_literals.append(new_ul)

            need_rollback = True
            self.assigned.append((lit, False))
            self.unassigned.remove(abs(lit))
        return True

    def solve(self) -> tuple[bool, list[int]]:
        """Return whether clause is satisfiable and if it is return its satisfied literals."""
        start = perf_counter_ns()

        # assigned only by decision
        assigned = []

        if not self.unit_prop():
            # initial formula is unsatisfiable
            self.solve_time = perf_counter_ns() - start
            return False, None

        while self.unassigned:
            # no heuristic, take random unassigned
            lit = random.sample(self.unassigned, 1)[0]

            # try lit
            if self.counters[lit] or self.counters[-lit]:
                lit = random.choices(
                    [lit, -lit], [self.counters[lit], self.counters[-lit]]
                )[0]
            else:
                lit = random.choices([lit, -lit])[0]

            if self.unit_prop(lit):
                assigned.append((lit, True))
            # try -lit
            elif self.unit_prop(-lit):
                assigned.append((-lit, False))
            else:
                # dead end, need to backtrack
                while assigned:
                    lit, first = assigned.pop()
                    self.rollback(lit)
                    if first and self.unit_prop(-lit):
                        # change was successful
                        assigned.append((-lit, False))
                        break
                # backtracked completely - unsatisfiable
                if not assigned:
                    self.solve_time = perf_counter_ns() - start
                    return False, None
        self.solve_time = perf_counter_ns() - start
        return True, sorted(
            map(lambda x: x[0], self.assigned), key=lambda i: abs(i)
        )

    def get_stats(self) -> tuple[int, int, int, int]:
        """Return initialization time in s, solving time in s, decisions count and unit propagation steps."""
        return (
            self.initialization_time / 1000000000,
            self.solve_time / 1000000000,
            self.decisions,
            self.unit_prop_steps,
        )


def get_string_output(
    sat: bool, model: list[int], stats: tuple[int, int, int, int]
) -> str:
    ret = ["SAT" if sat else "UNSAT"]
    if model:
        ret.append(", ".join(map(str, model)))
    if stats:
        ret.append("Initialization time: {:.3f} s.".format(stats[0]))
        ret.append("Solving time: {:.3f} s.".format(stats[1]))
        ret.append(f"Decisions: {stats[2]}.")
        ret.append(f"Unit propagation steps: {stats[3]}.")
    return "\n".join(ret)


if __name__ == "__main__":
    args = parse_args()
    cnf, max_var = get_cnf(args.input, args.format)
    solver = DPLL_watched_solver(cnf, max_var)
    sat, model = solver.solve()

    result = get_string_output(
        sat, model, solver.get_stats() if args.verbose else None
    )

    write_output(result, args.output)
