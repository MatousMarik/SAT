#!/usr/bin/env python3
"""Watched DPLL Algorithm. Specification at http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_watched.php"""
from dataclasses import dataclass
from formula2cnf import get_cnf, read_input, write_output
from argparse import ArgumentParser, Namespace
from time import perf_counter_ns
from typing import List, Optional
from itertools import chain
import sys, os


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
    parser.add_argument(
        "-m",
        "--minimal_variable",
        action="store_true",
        help="As a decision select unassigned variable with lowest number.",
    )
    args = parser.parse_args(args)
    return args


@dataclass
class WatchedClause:
    """Clause with watchers (list[2 ints])."""

    list: list[int]
    watchers: List = None

    def __post_init__(self):
        if len(self.list) > 1:
            self.watchers = [0, 1]
        else:
            self.watchers = [0, 0]

    def watch(
        self, assignment: list[bool], literal: int
    ) -> tuple[bool, int, Optional[int]]:
        """
        Move watcher and return if satisfiable, new watched literal and optionally unit_literal.

        Note that literal will be negation of one's watcher literal.
        """
        w1, w2 = self.watchers
        # swap so it works only with w1
        if self.list[w2] == -literal:
            w2, w1 = w1, w2

        # the other watcher is False - unsatisfiable
        if assignment[self.list[w2]] == False:
            return False, -self.list[w1], None

        # search for another watchable
        for w in chain(range(w1 + 1, len(self.list)), range(0, w1 + 1)):
            # it can't be same as w2 and it has to be satisfiable
            if w != w2 and assignment[self.list[w]] != False:
                break
        # not found - w2 watches unit_literal
        if w == w1:
            return True, -self.list[w1], self.list[w2]

        # update watchers
        self.watchers[:] = w, w2
        return True, -self.list[w], None


class DPLL_watched_solver:
    def __init__(
        self,
        cnf: list[list[int]],
        max_var: int,
        minimal_decision_variable: bool = False,
    ) -> None:
        start = perf_counter_ns()
        self.cnf = cnf
        self.max_var = max_var
        # empty watcher list (for each literal -> [clauses])
        self.w_lists: list[list[WatchedClause]] = list(
            [] for _ in range(max_var * 2 + 1)
        )

        self.minimal_decision_variable: bool = minimal_decision_variable

        # satisfied literals
        self.assigned: list[int] = []
        # [None] + [values of each literal]
        self.assignment: list[Optional[bool]] = [None] * (max_var * 2 + 1)
        # unassigned variables
        self.unassigned: set[int] = set(range(1, max_var + 1))

        self.initial_unit_literals = None
        self.generate_watched_lists()

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
            literal = self.assigned[0]

        while True:
            lit = self.assigned.pop()
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

        need_rollback = False
        while u_literals:
            lit = u_literals.pop()
            if abs(lit) not in self.unassigned:
                continue

            self.unit_prop_steps += 1
            # manage assignment of newly propagated literals
            self.assignment[lit], self.assignment[-lit] = True, False
            # 'pop_all' watched for literal
            watched, self.w_lists[lit] = self.w_lists[lit], []
            while watched:
                watched_c = watched.pop()
                # find if satisfiable, new_watched, unit_literal
                sat, new_wl, new_ul = watched_c.watch(self.assignment, lit)
                self.w_lists[new_wl].append(watched_c)

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
            self.assigned.append(lit)
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
            if self.minimal_decision_variable:
                lit = min(self.unassigned)
            else:
                # no heuristic, take first unassigned that "pops"
                for lit in self.unassigned:
                    break

            # try lit
            if self.unit_prop(lit):
                assigned.append(lit)
            # try -lit
            elif self.unit_prop(-lit):
                assigned.append(-lit)
            else:
                # dead end, need to backtrack
                while assigned:
                    lit = assigned.pop()
                    self.rollback(lit)
                    # +lit was always tried first, so it can be changed
                    if lit > 0:
                        if self.unit_prop(-lit):
                            # change was successful
                            assigned.append(-lit)
                            break
                # backtracked completely - unsatisfiable
                if not assigned:
                    self.solve_time = perf_counter_ns() - start
                    return False, None
        self.solve_time = perf_counter_ns() - start
        return True, sorted(self.assigned, key=lambda i: abs(i))

    def get_stats(self) -> tuple[float, int, int]:
        """Return time in s, decisions count and unit propagation steps."""
        return (
            (self.initialization_time + self.solve_time) / 1000000000,
            self.decisions,
            self.unit_prop_steps,
        )


def get_string_output(
    sat: bool, model: list[int], stats: tuple[float, int, int]
) -> str:
    ret = ["SAT" if sat else "UNSAT"]
    if model:
        ret.append(", ".join(map(str, model)))
    if stats:
        ret.append("Time: {:.3f} s.".format(stats[0]))
        ret.append(f"Decisions: {stats[1]}.")
        ret.append(f"Unit propagation steps: {stats[2]}.")
    return "\n".join(ret)


if __name__ == "__main__":
    args = parse_args()
    cnf, max_var = get_cnf(args.input, args.format)
    solver = DPLL_watched_solver(cnf, max_var, args.minimal_variable)
    sat, model = solver.solve()

    result = get_string_output(
        sat, model, solver.get_stats() if args.verbose else None
    )

    write_output(result, args.output)
