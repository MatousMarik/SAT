#!/usr/bin/env python3
"""DPLL Algorithm. Specification at http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_dpll.php"""
from dataclasses import dataclass
from formula2cnf import formula2cnf, read_input, write_output
from argparse import ArgumentParser, Namespace
from time import perf_counter_ns
from typing import Optional
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
class AdjacencyClause:
    list: list[int]
    counter: int


class DPLL_adjacency_solver:
    def __init__(self, cnf: list[list[int]], max_var: int) -> None:
        start = perf_counter_ns()
        self.cnf = cnf
        self.max_var = max_var
        self.a_lists: tuple[list[AdjacencyClause]] = tuple(
            [] for _ in range(max_var * 2 + 1)
        )

        self.assigned: list[int] = []
        self.unassigned: set[int] = set(range(1, max_var + 1))

        self.initial_unit_literals = None
        self.generate_adjacency_lists()

        self.decisions: int = 0
        self.unit_prop_steps: int = 0
        self.solve_time: int = -1
        self.initialization_time: int = perf_counter_ns() - start

    def generate_adjacency_lists(self) -> None:
        """Create adjacency lists + counters of possibly satisfied literals in clauses and list of literals from unit clauses."""
        a_lists = self.a_lists
        unit_literals = []
        for clause in self.cnf:
            len_ = len(clause)
            ac = AdjacencyClause(clause, len_)
            if len_ == 1:
                unit_literals.append(clause[0])
            for lit in clause:
                a_lists[-lit].append(ac)
        self.initial_unit_literals = unit_literals

    def rollback(self, literal: Optional[int]) -> None:
        """Unassign all last assigned literals upto 'literal'."""
        a_lists, assigned, unassigned = (
            self.a_lists,
            self.assigned,
            self.unassigned,
        )
        if literal is None:
            literal = assigned[0]

        while True:
            lit = assigned.pop()
            for ac in a_lists[lit]:
                ac.counter += 1
            unassigned.add(abs(lit))

            if lit == literal:
                break

    def unit_prop(self, literal: Optional[int] = None):
        """Set literal satisfied and do unit_propagation."""
        a_lists, assigned, unassigned = (
            self.a_lists,
            self.assigned,
            self.unassigned,
        )
        if literal is None:
            u_literals = self.initial_unit_literals
        else:
            self.decisions += 1
            u_literals = [literal]
        need_rollback = False
        while u_literals:
            lit = u_literals.pop()
            if abs(lit) not in unassigned:
                continue
            self.unit_prop_steps += 1
            for adj_c in a_lists[lit]:
                if adj_c.counter == 1:
                    for adj_c2 in a_lists[lit]:
                        if adj_c is adj_c2:
                            break
                        adj_c2.counter += 1
                    if need_rollback:
                        self.rollback(literal)
                    return False
                elif adj_c.counter == 2:
                    for l in adj_c.list:
                        if abs(l) in unassigned:
                            u_literals.append(l)
                            break
                adj_c.counter -= 1
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
