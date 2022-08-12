#!/usr/bin/env python3
"""Watched CDCL Algorithm. Specification at http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_cdcl.php"""
from dataclasses import dataclass
from formula2cnf import formula2cnf, read_input, write_output
from argparse import ArgumentParser, Namespace
from time import perf_counter_ns
from typing import List, Optional
from itertools import chain
from collections import Counter
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
        "--confilct_limit",
        type=int,
        default=128,
        help="Initial limit on conflicts before reset.",
    )
    parser.add_argument(
        "--lbd_limit",
        type=int,
        default=2,
        help="Initial limit on different decision levels in learned clauses.",
    )
    args = parser.parse_args(args)
    if args.input is not None:
        if args.input.endswith(".sat"):
            args.format = "SAT"
        elif args.input.endswith(".cnf"):
            args.format = "CNF"
    if args.format is None:
        args.format = "SAT"
    return args


def parse_cnf(string: str) -> tuple[list[list[int]], int]:
    lines = string.splitlines()
    lines.reverse()
    line = ""
    while not line.startswith("p cnf "):
        if not lines:
            raise RuntimeError("Invalid formula. No 'p' line.")
        line = lines.pop()
    s_line = line.split(maxsplit=3)
    try:
        max_var = int(s_line[2])
        num_clauses = int(s_line[3])
    except:
        raise RuntimeError(f"Invalid nbvar or nbclauses in '{line}'.")

    cnf = []
    try:
        for line in lines[-num_clauses:]:
            literals = list(map(int, line.split()))
            assert literals[-1] == 0 and all(
                0 < abs(l) <= max_var for l in literals[:-1]
            )
            cnf.append([*literals[:-1]])
    except:
        raise RuntimeError(f"Invalid clause: {line}")
    return cnf, max_var


def get_cnf(args: Namespace) -> tuple[list[list[int]], int]:
    """Return cnf as list of clauses (tuples of ints - literals) and maximal variable."""
    string = read_input(args.input)
    if args.format == "SAT":
        cnf, max_var, _ = formula2cnf(string, False)
    elif args.format == "CNF":
        cnf, max_var = parse_cnf(string)
    else:
        raise RuntimeError("Invalid format.")
    return cnf, max_var


@dataclass
class WatchedClause:
    """Clause with watchers (list[2 ints])."""

    ci: int
    list: list[int]
    lbd: int = 0
    watchers: List = None

    def __post_init__(self):
        if self.watchers is None:
            if len(self.list) > 1:
                self.watchers = [0, 1]
            else:
                self.watchers = [0, 0]

    @property
    def w1(self) -> int:
        return self.list[self.watchers[0]]

    @property
    def w2(self) -> int:
        return self.list[self.watchers[1]]

    def __hash__(self) -> int:
        return self.ci

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


class CDCL_watched_solver:
    LIMIT_MUL = 1.2

    def __init__(
        self,
        cnf: list[list[int]],
        max_var: int,
        conflict_limit: int,
        lbd_limit: int,
    ) -> None:
        start = perf_counter_ns()
        self.cnf = cnf
        self.max_var = max_var
        self.conflict_limit = conflict_limit
        self.lbd_limit = lbd_limit

        # satisfied literals
        self.assigned: list[int] = []
        # [None] + [values of each literal]
        self.assignment: list[Optional[bool]] = [None] * (max_var * 2 + 1)
        # unassigned variables
        self.unassigned: set[int] = set(range(1, max_var + 1))

        self.learned: list[WatchedClause] = []
        self.antecedents: list[WatchedClause] = [None] * (max_var + 1)
        self.decision_levels = [-1] * (max_var + 1)

        # empty watcher list (for each literal -> [clauses])
        self.w_sets: list[set[WatchedClause]] = [
            set() for _ in range(max_var * 2 + 1)
        ]
        self.variables: set[int] = set()
        self.initial_unit_literals: list[int] = []
        self.lit_to_clause_indices: list[set[int]] = [
            set() for _ in range(max_var * 2 + 1)
        ]
        for ci, clause in enumerate(self.cnf):
            # add variables adn lit_to_clauses
            for l in clause:
                self.variables.add(abs(l))
                self.lit_to_clause_indices[l].add(ci)
            ac = WatchedClause(ci, clause)
            # fill watched lists
            self.w_sets[-clause[0]].add(ac)
            if len(clause) == 1:
                # add unit clause literals
                self.unit_literals.append(clause[0])
            else:
                self.w_sets[-clause[1]].add(ac)

        self.nci: int = len(cnf)

        self.decisions: int = 0
        self.unit_prop_steps: int = 0
        self.solve_time: int = -1
        self.restarts: int = 0
        self.initialization_time: int = perf_counter_ns() - start

    def rollback(self, decision_level: int) -> None:
        """Unassign all last assigned literals down to decision level."""
        while (
            self.assigned
            and self.decision_levels[self.assigned[-1]] > decision_level
        ):
            lit = self.assigned.pop()
            self.decision_levels[lit] = -1
            self.unassigned.add(abs(lit))
            self.antecedents[abs(lit)] = None

            # update assignment
            self.assignment[lit], self.assignment[-lit] = None, None

    def unit_prop(
        self, literal: Optional[int] = None, decision_level: int = 0
    ) -> tuple[bool, Optional[WatchedClause]]:
        """
        Set literal satisfied and do unit_propagation.
        Return SAT and conflict clause.
        """
        if literal is None:
            u_literals = self.initial_unit_literals
        else:
            self.decisions += 1
            u_literals = [literal]

        while u_literals:
            lit = u_literals.pop()
            if abs(lit) not in self.unassigned:
                continue

            self.unit_prop_steps += 1
            # manage assignment of newly propagated literals
            self.assignment[lit], self.assignment[-lit] = True, False
            # set decision level
            self.decision_levels[abs(lit)] = decision_level
            # 'pop_all' watched for literal
            watched, self.w_sets[lit] = self.w_sets[lit], set()
            while watched:
                watched_c = watched.pop()
                # find if satisfiable, new_watched, unit_literal
                sat, new_wl, new_ul = watched_c.watch(self.assignment, lit)
                self.w_sets[new_wl].add(watched_c)

                if not sat:
                    # need to not loose not processed watchers
                    self.w_sets[lit].update(watched)
                    # # reset assignment
                    # assignment[lit], assignment[-lit] = None, None
                    # if need_rollback:
                    #     self.rollback(literal)
                    return False, watched_c
                elif new_ul is not None and abs(new_ul) in self.unassigned:
                    # append new unit_literal
                    u_literals.append(new_ul)
                    self.antecedents[abs(new_ul)] = watched_c

            self.assigned.append(lit)
            self.unassigned.remove(abs(lit))
        return True, None

    def decision_literal(self) -> int:
        """Return unassigned literal from most unsatisfied clauses."""
        sat_clauses = set()
        for lit_sat, cis in zip(self.assignment, self.lit_to_clause_indices):
            if lit_sat:
                sat_clauses.update(cis)
        lits = []
        for ci, clause in enumerate(self.cnf):
            if ci not in sat_clauses:
                lits.extend(clause)
        for lit, _ in Counter(lits).most_common():
            if self.assignment[lit] is None:
                return lit

    def process_conflict(
        self, conflict: WatchedClause, decision_level: int
    ) -> tuple[int, int]:
        """Learn assertive clause and returns second highest decision level in it and unit_literal."""
        if decision_level == 0:
            return -1

        # assertive clause literals
        assignment = [*self.assignment]
        acls = set(conflict.list)
        while (
            len(
                [
                    lit
                    for lit in acls
                    if self.decision_levels[abs(lit)] == decision_level
                ]
            )
            > 1
        ):
            while assignment:
                lit = assignment.pop()
                if -lit in acls:
                    acls.update(self.antecedents[abs(lit)].literals)
                    acls.remove(lit)
                    acls.remove(-lit)
                break

        level = 0
        watchers = [None, None]
        decision_levels_in_assertive_clause = [False] * (decision_level + 1)
        lits = list(acls)
        for li, lit in enumerate(lits):
            dl = self.decision_levels[abs(lit)]
            if dl == decision_level:
                unit_literal = lit
                watchers[0] = li
            elif level < dl < decision_level:
                level = dl

            decision_levels_in_assertive_clause[dl] = True

        lbd = sum(decision_levels_in_assertive_clause)
        if len(lits) == 1:
            watchers[1] = watchers[0]
        else:
            found = False
            for lit in reversed(self.assignment):
                if self.decision_levels[abs(lit)] == level:
                    for li, c_lit in enumerate(lits):
                        if abs(lit) == abs(c_lit):
                            watchers[1] = li
                            found = True
                            break
                    if found:
                        break

        new_clause = WatchedClause(self.nci, lits, lbd, watchers)
        self.nci += 1
        self.w_sets[-new_clause.w1].add(new_clause)
        if watchers[0] != watchers[1]:
            self.w_sets[-new_clause.w2].add(new_clause)

        self.learned.append(new_clause)

        return level, unit_literal

    def solve(self) -> tuple[bool, list[int]]:
        """Return whether clause is satisfiable and if it is return its satisfied literals."""
        start = perf_counter_ns()

        # assigned only by decision
        assigned = []

        if not self.unit_prop()[0]:
            # initial formula is unsatisfiable
            self.solve_time = perf_counter_ns() - start
            return False, None

        conflicts = 0
        decision_level = 0
        while self.unassigned:
            # most occupied literal
            lit = self.decision_literal()
            decision_level += 1

            _, conflict = self.unit_prop(lit, decision_level)

            while conflict:
                conflicts += 1

                if conflicts >= self.conflict_limit:
                    self.restarts += 1
                    conflicts = 0
                    decision_level = 0
                    self.conflict_limit *= CDCL_watched_solver.LIMIT_MUL
                    self.lbd_limit *= CDCL_watched_solver.LIMIT_MUL
                    self.rollback(0)
                    # delete learned by lbd
                    new_learned = []
                    for clause in self.learned:
                        if clause.lbd > self.lbd_limit:
                            self.w_sets[clause.w1].remove(clause)
                            if clause.watchers[0] != clause.watchers[1]:
                                self.w_sets[clause.w2].remove(clause)
                        else:
                            new_learned.append(clause)
                    self.learned = new_learned
                    break

            decision_level, new_ulit = self.process_conflicts(
                conflict, decision_level
            )

            if decision_level < 0:
                return False, None

            self.rollback(decision_level)
            _, conflict = self.unit_prop(new_ulit, decision_level)

        self.solve_time = perf_counter_ns() - start
        return True, sorted(self.assigned, key=lambda i: abs(i))

    def get_stats(self) -> tuple[int, int, int, int, int]:
        """Return initialization time in s, solving time in s, decisions count and unit propagation steps."""
        return (
            self.initialization_time / 1000000000,
            self.solve_time / 1000000000,
            self.decisions,
            self.unit_prop_steps,
            self.restarts,
        )


def get_string_output(
    sat: bool, model: list[int], stats: tuple[int, int, int, int, int]
) -> str:
    ret = ["SAT" if sat else "UNSAT"]
    if model:
        ret.append(", ".join(map(str, model)))
    if stats:
        ret.append("Initialization time: {:.3f} s.".format(stats[0]))
        ret.append("Solving time: {:.3f} s.".format(stats[1]))
        ret.append(f"Decisions: {stats[2]}.")
        ret.append(f"Unit propagation steps: {stats[3]}.")
        ret.append(f"Restarts: {stats[4]}.")
    return "\n".join(ret)


if __name__ == "__main__":
    args = parse_args()
    cnf, max_var = get_cnf(args)
    solver = CDCL_watched_solver(
        cnf, max_var, args.conflict_limit, args.lbd_limit
    )
    sat, model = solver.solve()

    result = get_string_output(
        sat, model, solver.get_stats() if args.verbose else None
    )

    write_output(result, args.output)
