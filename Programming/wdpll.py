#!/usr/bin/env python3
"""Watched DPLL Algorithm. Specification at http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_dpll.php"""
from formula2cnf import formula2cnf, read_input, write_output
from argparse import ArgumentParser, Namespace
from time import perf_counter_ns
from typing import Optional
from dataclasses import dataclass
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
class Literal:
    lit: int
    satisfied: bool = None
    adjacent: set = set()

    def __hash__(self) -> int:
        return self.lit


@dataclass
class ClauseLiteral:
    lit: Literal
    next: "ClauseLiteral"
    clause: "WatchedClause"

    def __post_init__(self):
        self.lit.adjacent.add(self)


@dataclass
class WatchedClause:
    w1: ClauseLiteral = None
    w2: ClauseLiteral = None

    def __post_init__(self):
        first = ClauseLiteral(self.clause[0], None, self)
        prev = first
        for lit in self.clause[1:]:
            prev = ClauseLiteral(lit, prev, self)
        first.next = prev
        self.clause = first
        self.w1 = first
        self.w2 = prev

    def move(self, from_lit: ClauseLiteral) -> tuple[bool, bool]:
        """Move watcher and return is_empty_clause and is_unit_clause."""
        if self.w1 is from_lit:
            while self.w1.lit.satisfied is False:
                self.w1 = self.w1.next


class DPLL_watched_solver:
    def __init__(self, cnf: list[list[int]], max_var: int) -> None:
        start = perf_counter_ns()
        self.cnf = cnf
        self.max_var = max_var
        literals: tuple[Literal] = tuple(
            Literal(l) for l in range(-max_var, max_var + 1)
        )
        self.assigned: list[Literal] = []
        self.unassigned: set[Literal] = set(literals)

        self.watched_clauses = [
            WatchedClause([literals[i] for i in clause]) for clause in cnf
        ]

        self.literals = literals

        self.decisions: int = 0
        self.unit_prop_steps: int = 0
        self.solve_time: int = -1
        self.initialization_time: int = perf_counter_ns() - start

    def rollback(self, literal: Optional[int]) -> None:
        """Unassign all last assigned literals upto 'literal'."""
        counters, a_lists, assigned, unassigned = (
            self.counters,
            self.a_lists,
            self.assigned,
            self.unassigned,
        )
        if literal is None:
            literal = assigned[0]

        while True:
            lit = assigned.pop()
            for ci in a_lists[lit]:
                counters[ci] += 1
            unassigned.add(abs(lit))

            if lit == literal:
                break

    def unit_prop(self, literal: Optional[int] = None):
        """Set literal satisfied and do unit_propagation."""
        counters, a_lists, cnf, assigned, unassigned = (
            self.counters,
            self.a_lists,
            self.cnf,
            self.assigned,
            self.unassigned,
        )
        if literal is None:
            u_literals = [
                cnf[ci][0] for ci, count in enumerate(counters) if count == 1
            ]
        else:
            self.decisions += 1
            u_literals = [literal]
        need_rollback = False
        while u_literals:
            lit = u_literals.pop()
            if abs(lit) not in unassigned:
                continue
            self.unit_prop_steps += 1
            for ci in a_lists[lit]:
                if counters[ci] <= 2:
                    if counters[ci] == 1:
                        for ci2 in a_lists[lit]:
                            if ci == ci2:
                                break
                            counters[ci2] += 1
                        if need_rollback:
                            self.rollback(literal)
                        return False
                    else:
                        for l in cnf[ci]:
                            if abs(l) in unassigned:
                                u_literals.append(l)
                                break
                counters[ci] -= 1
            need_rollback = True
            assigned.append(lit)
            unassigned.remove(abs(lit))
        return True

    def solve(self) -> tuple[bool, list[int]]:
        unassigned, u_prop, rollback = (
            self.unassigned,
            self.unit_prop,
            self.rollback,
        )
        start = perf_counter_ns()
        assigned = []
        if not u_prop():
            self.solve_time = perf_counter_ns() - start
            return False, None

        while unassigned:
            for lit in unassigned:
                break
            if u_prop(lit):
                assigned.append(lit)
            elif u_prop(-lit):
                assigned.append(-lit)
            else:
                while assigned:
                    lit = assigned.pop()
                    rollback(lit)
                    if lit > 0:
                        if u_prop(-lit):
                            assigned.append(-lit)
                            break
                if not assigned:
                    self.solve_time = perf_counter_ns() - start
                    return False, None
        self.solve_time = perf_counter_ns() - start
        return True, sorted(self.assigned, key=lambda i: abs(i))

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
    cnf, max_var = get_cnf(args)
    solver = DPLL_adjacency_solver(cnf, max_var)
    sat, model = solver.solve()

    result = get_string_output(
        sat, model, solver.get_stats() if args.verbose else None
    )

    write_output(result, args.output)
