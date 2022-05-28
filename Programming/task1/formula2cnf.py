#!/usr/bin/env python3
"""Formula to cnf as Tseitin encoding. Specification at http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_tseitin.php"""
from argparse import ArgumentParser
import sys, os

from collections import deque


class ETokens:
    """
    List formula tokens and their int representation
    and provides mapping token_string -> int.
    """

    L_PAR = -1
    R_PAR = -2
    AND = -3
    OR = -4
    NOT = -5

    map = {"(": L_PAR, ")": R_PAR, "and": AND, "or": OR, "not": NOT}
    inv_map = {v: k for k, v in map.items()}

    @classmethod
    def to_str(cls, token: int) -> str:
        """Return name of the token, for variable tokens return "variable"."""
        if token > 0:
            return "variable"
        if token in cls.inv_map:
            return cls.inv_map[token]
        else:
            raise RuntimeError("Invalid token.")


def translate(string: str) -> tuple[list[int], dict[str, int]]:
    """
    Splits sequence by whitespaces and parentheses.
    Return translated sequence and mapping.

    Translated words are ints,
    * positive for variables, where mapping variable_name -> int is given by returned dict
    * negative for tokens from ETokens.
    """
    v_map = {}
    var = 1

    translated = []
    # cycle list of words (parts of string separated by whitespaces)
    for word in str.split(string):
        # trims left parenthesis and adds it as a token
        if word[0] == "(":
            translated.append(ETokens.map["("])
            word = word[1:]

        # trims right parentheses
        add_r = 0
        while word[-1 - add_r] == ")":
            add_r += 1
            if add_r == len(word):
                break
        if add_r:
            word = word[:-add_r]

        # translate strings to ints
        if word in ETokens.map:
            translated.append(ETokens.map[word])
        elif word in v_map:
            translated.append(v_map[word])
        else:
            v_map[word] = var
            translated.append(var)
            var += 1

        # adds right parentheses tokens
        if add_r:
            translated.extend([ETokens.map[")"]] * add_r)

    return translated, v_map


class EAwaitedType:
    """
    Current state of parsed sequence,
    corresponds to awaited token(s).
    """

    VAR_OR_R = 1
    VAR_OR_R_OR_NOT = 2
    AND_OR_OR = 3
    L = 4

    expected_map = {
        1: [1, ETokens.R_PAR],
        2: [1, ETokens.R_PAR, ETokens.NOT],
        3: [ETokens.AND, ETokens.OR],
        4: [ETokens.L_PAR],
    }

    @classmethod
    def expected_names(cls, type: int) -> str:
        """Get names of expected tokens at given state."""
        return ", ".join(map(ETokens.to_str, cls.expected_map[type]))


def parse2cnf(
    stream: list[int], max_var: int, equivalences: bool
) -> tuple[list[tuple[int]], int]:
    """
    Return list of clauses and root gate index.
    ([(1,)], -1 in case of single variable formula)
    Clause is 2 or 3 - tuple of ints, where ints corresponds to variables or gates.

    Parses sequence of tokens from the right side, as if building tree,
    but subtrees - gates are replaced by their indices during the process.
    When whole clause is found it is added to list of all clauses and it is assigned new gate index,
    which is put to stack.

    Params
    ------
    stream: sequence of ints representing ETokens (negative) or variables (positive)
    max_var: maximal int representing variable
    equivalences: specify implications [False] or equivalences [True] between gates and corresponding clauses
    """
    EAT = EAwaitedType
    last_var = max_var

    # index that will be assigned to a new gate, after that it will be increased
    next_gate_index = max_var + 1

    # stack for parsed subtrees
    stack = deque()
    # stack for states when parsing recursively
    rec_s = deque()
    cnf = []
    # awaited position in parentheses from right side,
    # 0 for most right, -1 for most left
    state = EAT.VAR_OR_R

    for token in reversed(stream):
        # awaited variable or right bracket - most right position
        if state == EAT.VAR_OR_R:
            if last_var >= token > 0:
                # token is variable
                stack.append(token)
                state = EAT.VAR_OR_R_OR_NOT
            elif token == ETokens.R_PAR:
                # token is right bracket
                rec_s.append(EAT.VAR_OR_R_OR_NOT)
            else:
                raise RuntimeError(
                    f"Unexpected token; found: {ETokens.to_str(token)}, expected: {EAwaitedType.expected_names(state)}."
                )

        # awaited variable, right bracket or not - second position from right
        elif state == EAT.VAR_OR_R_OR_NOT:
            if last_var >= token > 0:
                # token is variable
                stack.append(token)
                state = EAT.AND_OR_OR
            elif token == ETokens.R_PAR:
                # token is right bracket
                rec_s.append(EAT.AND_OR_OR)
                state = EAT.VAR_OR_R
            elif token == ETokens.NOT:
                # token is not
                # Note: there are only gate or variable tokens on stack,
                #   so there won't be collision between negative variable tokens and ETokens
                stack.append(-stack.pop())
                state = EAT.L
            else:
                raise RuntimeError(
                    f"Unexpected token; found: {ETokens.to_str(token)}, expected: {EAwaitedType.expected_names(state)}."
                )

        # awaited "and" or "or" tokens  - most left position between parentheses
        elif state == EAT.AND_OR_OR:
            if token == ETokens.AND:
                left = stack.pop()
                right = stack.pop()
                if left == right:
                    raise RuntimeError(
                        "Clause would contain same literal twice."
                    )
                if left == -right:
                    raise RuntimeError(
                        "Clause would contain opposite literals."
                    )

                # add C <=> (L and R)
                cnf.append((-next_gate_index, left))
                cnf.append((-next_gate_index, right))

                if equivalences:
                    cnf.append((-left, -right, next_gate_index))

                stack.append(next_gate_index)
                next_gate_index += 1

            elif token == ETokens.OR:
                left = stack.pop()
                right = stack.pop()
                if left == right:
                    raise RuntimeError(
                        "Clause would contain same literal twice."
                    )
                if left == -right:
                    raise RuntimeError(
                        "Clause would contain opposite literals."
                    )

                # add C <=> (L or R)
                cnf.append((-next_gate_index, left, right))

                if equivalences:
                    cnf.append((-left, next_gate_index))
                    cnf.append((-right, cnf))

                stack.append(next_gate_index)
                next_gate_index += 1
            else:
                raise RuntimeError(
                    f"Unexpected token; found: {ETokens.to_str(token)}, expected: {EAwaitedType.expected_names(state)}."
                )
            state = EAT.L

        # awaited left parenthesis
        elif state == EAT.L:
            if token == ETokens.L_PAR:
                state = rec_s.pop()
            else:
                raise RuntimeError(
                    f"Unexpected token; found: {ETokens.to_str(token)}, expected: {EAwaitedType.expected_names(state)}."
                )
        else:
            raise RuntimeError("Unrecognized state.")

    # check correct state of stack
    if len(stack) == 1 and len(rec_s) == 0 and state == EAT.VAR_OR_R_OR_NOT:
        root = stack.pop()
        if root == 1:
            # formula was just a single variable
            return [(1,)], -1
        elif root > last_var:
            return cnf, root
    raise RuntimeError(f"Invalid formula.")


def to_str(
    clauses: list[tuple[int]],
    root: int,
    var_map: dict[str, int],
) -> str:
    """
    Return string representing Tseitin encoding.

    Params
    ------
    clauses: list of tuples of ints representing variables or gates in one clause
    root: root gate (for specification in comment)
    var_map: mapping variable_name -> int (for specifications in comment)
    """
    ret = []
    # adds mapping specification
    for name, var in sorted(var_map.items(), key=lambda i: i[1]):
        ret.append(f"c var {var} : {name}")
    ret.append("c")

    # one variable only
    if root == -1:
        ret.append("p cnf 1 1\n1 0")
        return "\n".join(ret)

    # adds specifications
    ret.append(f"c root : {root}")
    if root - len(var_map) > 1:
        ret.append(f"c gates {len(var_map) + 1}..{root}")
    ret.append("c")
    ret.append(f"p cnf {len(var_map)} {len(clauses)}")

    for c in clauses:
        ret.append(" ".join(map(str, c)) + " 0")
    return "\n".join(ret)


def formula2cnf(formula: str, equivalences: bool) -> str:
    """
    Returns Tseitin encoding of cnf for given formula.
    equivalences specify implications [False] or equivalences [True] between gates and corresponding clauses.
    """
    seq, v_map = translate(formula)
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
    args = parser.parse_args(["test.sat"])

    if args.input is None:
        string = sys.stdin.read()
    else:
        if not os.path.exists(args.input):
            args.input = os.path.join(os.path.dirname(__file__), args.input)
        with open(args.input, "r") as f:
            string = f.read()

    result = formula2cnf(string, args.equivalences)

    if args.output is None:
        print(result)
    else:
        with open(args.output, "w") as f:
            f.write(result)
