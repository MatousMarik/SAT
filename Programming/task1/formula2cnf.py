#!/usr/bin/env python3
from argparse import ArgumentParser
from itertools import chain
from functools import partial
import sys, os

from collections import deque
from typing import Iterator, Sequence, List, Dict

sys.path.append(os.path.dirname(__file__))


class EPS:
    FORMULA_START = 0
    L_PAR = FORMULA_START + 1
    AND_VAR_1 = L_PAR + 1
    AND_VAR_2 = AND_VAR_1 + 1
    OR_VAR_1 = AND_VAR_2 + 1
    OR_VAR_2 = OR_VAR_1 + 1
    VARIABLE = OR_VAR_2 + 1
    NOT_VARIABLE = VARIABLE + 1


class ETokens:
    L_PAR = -1
    R_PAR = -2
    AND = -3
    OR = -4
    NOT = -5

    t_map = {"(": L_PAR, ")": R_PAR, "and": AND, "or": OR, "not": NOT}


def translate(string: str):
    v_map = {}
    var = 1

    translated = []
    for word in str.split(string):
        if word[0] == "(":
            translated.append(ETokens.t_map["("])
            word = word[1:]

        add_r = 0
        while word[-1 - add_r] == ")":
            add_r += 1
            if add_r == len(word):
                break
        if add_r:
            word = word[:-add_r]

        if word in ETokens.t_map:
            translated.append(ETokens.t_map[word])
        elif word in v_map:
            translated.append(v_map[word])
        else:
            v_map[word] = var
            translated.append(var)
            var += 1
        if add_r:
            translated.extend([ETokens.t_map[")"]] * add_r)

    return translated, v_map


class EAwaitedType:
    VAR_OR_R = 1
    VAR_OR_R_OR_NOT = 2
    TYPE = 3
    L = 4


def parse2cnf(stream: list[int], num_vars: int, equivalences: bool):
    EAT = EAwaitedType
    max_var = num_vars

    ci = num_vars

    deq = deque()
    rec_q = deque()
    cnf = []
    # awaited position in parenthesis from right side, 0 for most right, -1 for most left
    # -1 initial formula
    type = EAT.VAR_OR_R

    for t in reversed(stream):
        if type == EAT.VAR_OR_R:
            if max_var >= t > 0:
                deq.append(t)
                type = EAT.VAR_OR_R_OR_NOT
            elif t == ETokens.R_PAR:
                rec_q.append(EAT.VAR_OR_R_OR_NOT)
            else:
                raise RuntimeError

        elif type == EAT.VAR_OR_R_OR_NOT:
            if max_var >= t > 0:
                deq.append(t)
                type = EAT.TYPE
            elif t == ETokens.R_PAR:
                rec_q.append(EAT.TYPE)
                type = EAT.VAR_OR_R
            elif t == ETokens.NOT:
                deq.append(-deq.pop())
                type = EAT.L
            else:
                raise RuntimeError

        elif type == EAT.TYPE:
            if t == ETokens.AND:
                # add C <=> (L and R)
                left = deq.pop()
                right = deq.pop()
                cnf.append((-ci, left))
                cnf.append((-ci, right))

                if equivalences:
                    cnf.append((-left, -right, ci))

                deq.append(ci)

                ci += 1
            elif t == ETokens.OR:
                # add C <=> (L or R)
                left = deq.pop()
                right = deq.pop()
                cnf.append((-ci, left, right))

                if equivalences:
                    cnf.append((-left, ci))
                    cnf.append((-right, cnf))

                deq.append(ci)
                ci += 1
            else:
                raise RuntimeError
            type = EAT.L

        elif type == EAT.L:
            if t == ETokens.L_PAR:
                type = rec_q.pop()
            else:
                raise RuntimeError
        else:
            raise RuntimeError

    if len(deq) == 1 and len(rec_q) == 0:
        if type == EAT.VAR_OR_R_OR_NOT:
            return cnf, deq.pop()

        elif type == EAT.VAR_OR_R_OR_NOT and deq.pop() == 1:
            return [(1)], -1

    raise RuntimeError


def to_str(
    cnf: list[Sequence[int]],
    root: int,
    var_map: dict[str, int],
):
    ret = []
    for name, var in sorted(var_map.items(), key=lambda i: i[1]):
        ret.append(f"c variable {var} : {name}")

    if root == -1:
        return ret[0] + "p cnf 1 1\n1 0"

    ret.append(f"c root {root}")
    ret.append(f"c gates {len(var_map) + 1}..{root - 1}")
    ret.append(f"p cnf {len(var_map)} {len(cnf)}")

    for c in cnf:
        ret.append(" ".join(map(str, c)) + " 0")
    return "\n".join(ret)


def formula2cnf(string: str, equivalences: bool) -> str:
    seq, v_map = translate(string)
    cnf, root = parse2cnf(seq, len(v_map), equivalences)
    return to_str(cnf, root, v_map)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", nargs="?", type=str, help="Input file.")
    parser.add_argument("output", nargs="?", type=str, help="Output file.")
    parser.add_argument(
        "-e",
        "--equivalences",
        action="store_true",
        help="CNF with equivalences. Otherwise left-to-right implications only.",
    )
    args = parser.parse_args(
        [r"C:\Users\Maty\Documents\LS_21-22\SAT\Programming\task1\nested_5.sat"]
    )  # TODO: None)

    if args.input is None:
        string = sys.stdin.read()
    else:
        with open(args.input, "r") as f:
            string = f.read()

    result = formula2cnf(string, args.equivalences)

    if args.output is None:
        print(result)
    else:
        with open(args.output, "w") as f:
            f.write(result)
