#!/usr/bin/env python3
"""Watched CDCL Algorithm with Heuristics. Specification at http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_decision.php"""
from dataclasses import dataclass
from formula2cnf import get_cnf, write_output
from argparse import ArgumentParser, Namespace
from time import perf_counter_ns
from typing import List, Optional, Callable, Tuple
from itertools import chain
from collections import Counter, deque
import sys
import numpy as np


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
        "--conflict_limit",
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
    parser.add_argument(
        "-a",
        "--assumption",
        type=int,
        nargs="+",
        help="List of literals (integers) that are assumed satisfied.",
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        choices=["default", "vsids", "random", "unsatisfied"],
        default="default",
        help="Heuristic for selection of decision literal.",
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


class CDCL_watched_solver:
    LIMIT_MUL = 1.2
    VSIDS_DECAY = 0.75

    def __init__(
        self,
        cnf: list[list[int]],
        max_var: Optional[int] = None,
        conflict_limit: int = 128,
        lbd_limit: int = 2,
        heuristic: str = "default",
        assumption: Optional[list[int]] = None,
    ) -> None:
        start = perf_counter_ns()
        self.cnf = cnf

        if max_var is None:
            max_var = abs(
                max(map(lambda clause: max(clause, key=abs), cnf), key=abs)
            )

        self.max_var = max_var
        self.conflict_limit = conflict_limit
        self.lbd_limit = lbd_limit

        self.assumption: list[int] = None
        if assumption:
            # for popping
            assumption.reverse()
            self.assumption = assumption

        assert self.assumption is None or all(
            -max_var <= a <= max_var for a in self.assumption
        )

        # satisfied literals
        self.assigned: list[int] = []
        # [None] + [values of each literal]
        self.assignment: list[Optional[bool]] = [None] * (max_var * 2 + 1)

        self.learned: list[WatchedClause] = []
        self.antecedents: list[WatchedClause] = [None] * (max_var + 1)
        self.decision_levels = [-1] * (max_var + 1)

        # row for positive and row for negative counters
        # Note: negative literals are also indexing by their abs_value,
        # not from the end by their true=negative value
        self.counters = np.zeros((2, max_var + 1))

        # empty watcher list (for each literal -> [clauses])
        self.w_sets: list[set[WatchedClause]] = [
            set() for _ in range(max_var * 2 + 1)
        ]
        self.variables: set[int] = set()

        self.unit_literals: deque[Tuple[int, WatchedClause]] = deque()
        self.lit_to_clause_indices: list[set[int]] = [
            set() for _ in range(max_var * 2 + 1)
        ]
        self.new_clause_index = 0

        for clause in self.cnf:
            self.add_clause(clause)

        # unassigned variables
        self.unassigned: set[int] = self.variables.copy()

        # backbone found by first unit prop
        self.initial_backbone = None

        # heuristic for literal selection
        self.get_decision_literal: Callable[..., int] = None
        self._set_heuristic(heuristic)

        self.decisions: int = 0
        self.unit_prop_steps: int = 0
        self.solve_time: int = 0
        self.restarts: int = 0
        self.initialization_time: int = perf_counter_ns() - start

    def add_clause(
        self, clause: list[int], lbd: int = 0, watchers: list[int] = None
    ) -> WatchedClause:
        """
        Return new clause, created with provided fields and return its representation as WatchedClause.
        Also add newly created clause into watched list and to lit_to_clause list and if clause is unit,
        add it to unit_literals.
        """
        new_wc = WatchedClause(self.new_clause_index, clause, lbd, watchers)
        # add variables adn lit_to_clauses
        for l in clause:
            self.variables.add(abs(l))
            self.lit_to_clause_indices[l].add(self.new_clause_index)
        # fill watched lists
        self.w_sets[-new_wc.w1].add(new_wc)
        if len(clause) == 1:
            # add unit clause literals
            self.unit_literals.append((clause[0], new_wc))
        else:
            self.w_sets[-new_wc.w2].add(new_wc)

        self.new_clause_index += 1
        return new_wc

    def _set_heuristic(self, h: str) -> Callable[..., int]:
        """Set decision literal selection heuristic."""
        HEURISTICS = {
            "unsatisfied": self.most_unsatisfied_literal,
            "random": self.random_literal,
            "vsids": self.vsids_literal,
            "default": self.vsids_literal,
        }
        if h in HEURISTICS:
            self.get_decision_literal = HEURISTICS[h]
        else:
            self.get_decision_literal = HEURISTICS["default"]

    ### HEURISTICS

    def most_unsatisfied_literal(self) -> int:
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

    def random_literal(self) -> int:
        """Return random unassigned literal."""
        unassigned_variable = np.random.choice(list(self.unassigned))
        if np.random.rand() < 0.5:
            return -unassigned_variable
        return unassigned_variable

    def vsids_literal(self) -> int:
        """Return unassigned literal by vsids heuristics."""
        unassigned_vars = list(self.unassigned)
        max_count_index = self.counters[:, unassigned_vars].argmax()

        if max_count_index >= len(unassigned_vars):
            return -unassigned_vars[max_count_index - len(unassigned_vars)]

        return unassigned_vars[max_count_index]

    ### METHODS

    def set_literal(
        self, lit: int, decision_lvl: int
    ) -> Optional[WatchedClause]:
        """
        Set literal as satisfied in given decision level.

        Return conflict clause or None.
        """
        self.assignment[lit], self.assignment[-lit] = True, False
        self.assigned.append(lit)
        self.unassigned.remove(abs(lit))
        # set decision level
        self.decision_levels[abs(lit)] = decision_lvl

        # 'pop_all' watched for literal
        watched, self.w_sets[lit] = self.w_sets[lit], set()
        while watched:
            watched_c = watched.pop()
            # find if satisfiable, new_watched, unit_literal
            sat, new_wl, new_ul = watched_c.watch(self.assignment, lit)
            self.w_sets[-new_wl].add(watched_c)

            if not sat:
                # need to not loose not processed watchers
                self.w_sets[lit].update(watched)
                return watched_c

            if new_ul is not None:
                # append new unit_literal
                self.unit_literals.append((new_ul, watched_c))
        return None

    def unit_prop(self, decision_level: int = 0) -> Optional[WatchedClause]:
        """
        Return SAT and conflict clause.
        """

        while self.unit_literals:
            lit, unit = self.unit_literals.popleft()
            if abs(lit) not in self.unassigned:
                continue

            self.unit_prop_steps += 1
            self.antecedents[abs(lit)] = unit

            conflict = self.set_literal(lit, decision_level)

            if conflict is not None:
                return conflict

        return None

    def process_conflict(
        self, conflict: WatchedClause, decision_level: int
    ) -> int:
        """Learn assertive clause set unit_literal as decided and returns second highest decision level in it."""
        if decision_level == 0:
            return -1

        # assertive clause literals
        assignment = [*self.assigned]
        # assertive clause literals
        acls = set(conflict.list)

        # find all assertive clause literals
        # update acls with antecedents of literals with given decision_level
        while True:
            same_dl_acls_count = len(
                [
                    lit
                    for lit in acls
                    if self.decision_levels[abs(lit)] == decision_level
                ]
            )
            if not assignment or same_dl_acls_count <= 1:
                break
            while assignment and same_dl_acls_count > 1:
                lit = assignment.pop()
                if -lit in acls:
                    same_dl_acls_count -= 1

                    acls.update(self.antecedents[abs(lit)].list)
                    acls.remove(lit)
                    acls.remove(-lit)

        level = 0
        watchers = [None, None]
        decision_levels_in_assertive_clause = [False] * (decision_level + 1)
        lits = list(acls)

        # vsids decay
        self.counters *= self.VSIDS_DECAY

        # finding level and the unit literal of the assertive clause
        # and counting decision levels for lbd
        for li, lit in enumerate(lits):
            dl = self.decision_levels[abs(lit)]
            if dl == decision_level:
                unit_literal = lit
                watchers[0] = li
            elif level < dl < decision_level:
                level = dl

            decision_levels_in_assertive_clause[dl] = True

            if lit > 0:
                self.counters[0, lit] += 1
            else:
                self.counters[1, -lit] += 1

        lbd = sum(decision_levels_in_assertive_clause)

        # find watchers[1] - last assigned literal
        if len(lits) == 1:
            watchers[1] = watchers[0]
        else:
            found = False
            for lit in reversed(self.assigned):
                if self.decision_levels[abs(lit)] == level:
                    for li, c_lit in enumerate(lits):
                        if abs(lit) == abs(c_lit):
                            watchers[1] = li
                            found = True
                            break
                    if found:
                        break

        new_clause = self.add_clause(lits, lbd, watchers)
        self.learned.append(new_clause)

        self.unit_literals.clear()
        self.unit_literals.appendleft((unit_literal, new_clause))

        return level

    def rollback(self, decision_level: int) -> None:
        """Unassign all last assigned literals down to decision level."""
        if not self.assigned:
            return
        var = abs(self.assigned[-1])
        while self.decision_levels[var] > decision_level:
            self.assigned.pop()
            self.decision_levels[var] = -1
            self.unassigned.add(var)
            self.antecedents[var] = None

            # update assignment
            self.assignment[var], self.assignment[-var] = None, None

            if not self.assigned:
                return
            var = abs(self.assigned[-1])

    def delete_clause(self, clause: WatchedClause) -> None:
        """Remove watched clause from the solver."""
        for l in clause.list:
            self.lit_to_clause_indices[l].remove(clause.ci)
        self.w_sets[-clause.w1].remove(clause)
        if clause.watches_more():
            self.w_sets[-clause.w2].remove(clause)

    def delete_learned_clauses(self, *, del_all: bool = False) -> None:
        """Delete learned clauses with lbd > lbd_limit or all of them if del_all."""
        new_learned = []
        for clause in self.learned:
            if del_all or clause.lbd > self.lbd_limit:
                self.delete_clause(clause)
            else:
                new_learned.append(clause)
        self.learned = new_learned

    def solve(self) -> tuple[bool, list[int]]:
        """Return whether clause is satisfiable and if it is return its satisfied literals."""
        start = perf_counter_ns()

        conflict = self.unit_prop()
        if conflict is not None:
            # initial formula is unsatisfiable
            self.solve_time += perf_counter_ns() - start
            return False, None

        # initial backbone found by initial unit propagation
        self.initial_backbone = set(self.assigned)

        conflicts = 0
        decision_level = 0

        while self.unassigned:
            decision_level += 1
            if self.assumption:
                # literals by assumption
                literal = self.assumption.pop()
            else:
                # literal selection by heuristic
                literal = self.get_decision_literal()

            self.decisions += 1

            self.set_literal(literal, decision_level)

            conflict = self.unit_prop(decision_level)

            while conflict:
                conflicts += 1

                if conflicts >= self.conflict_limit:
                    # restart
                    conflicts = 0
                    decision_level = 0
                    self.restarts += 1
                    self.conflict_limit *= CDCL_watched_solver.LIMIT_MUL
                    self.lbd_limit *= CDCL_watched_solver.LIMIT_MUL
                    self.rollback(0)
                    self.delete_learned_clauses()
                    self.unit_literals.clear()
                    break

                decision_level = self.process_conflict(conflict, decision_level)

                if decision_level < 0:
                    return False, None

                self.rollback(decision_level)
                conflict = self.unit_prop(decision_level)

        self.solve_time += perf_counter_ns() - start
        return True, sorted(self.assigned, key=lambda i: abs(i))

    def get_stats(self) -> tuple[float, float, int, int, int]:
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
    cnf, max_var = get_cnf(args.input, args.format)
    solver = CDCL_watched_solver(
        cnf,
        max_var,
        args.conflict_limit,
        args.lbd_limit,
        args.heuristic,
        args.assumption,
    )
    sat, model = solver.solve()
    s_model = set(model)

    result = get_string_output(
        sat, model, solver.get_stats() if args.verbose else None
    )

    write_output(result, args.output)
