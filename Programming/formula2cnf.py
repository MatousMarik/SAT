#!/usr/bin/env python3
"""Formula to cnf as Tseitin encoding. Specification at http://ktiml.mff.cuni.cz/~kucerap/satsmt/practical/task_tseitin.php"""
from argparse import ArgumentParser, Namespace
import sys, os
from typing import Optional, Union
from collections import deque


def parse_cnf(string: str) -> tuple[list[list[int]], int]:
    """
    Parse Dimacs format to cnf as 2D list of ints.
    Also return maximal variable.
    """
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


def get_cnf(
    input: Union[str, None], format: Optional[str] = None
) -> tuple[list[list[int]], int]:
    """Return cnf as list of clauses (tuples of ints - literals) and maximal variable."""
    # overwrite by defaults if needed
    if input is not None:
        if input.endswith(".sat"):
            format = "SAT"
        elif input.endswith(".cnf"):
            format = "CNF"
    if format is None:
        format = "SAT"

    # parse
    string = read_input(input)
    if format == "SAT":
        cnf, max_var, _ = formula2cnf(string, False)
    elif format == "CNF":
        cnf, max_var = parse_cnf(string)
    else:
        raise RuntimeError("Invalid format.")
    return cnf, max_var


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


def _translate(string: str) -> tuple[list[int], dict[str, int]]:
    """
    Splits sequence by whitespaces and parentheses.
    Return translated sequence and mapping.

    Translated words are ints,
    * positive for variables, where mapping variable_name -> int is given by returned dict
    * negative for tokens from ETokens.
    """
    v_map = {}
    var = 1

    # add spaces after brackets
    string = ") ".join(string.split(")"))

    translated = []
    # cycle list of words (parts of string separated by whitespaces)
    for word in str.split(string):
        # trims left parenthesis and adds it as a token
        if word[0] == "(":
            translated.append(ETokens.L_PAR)
            if len(word) == 1:
                continue
            word = word[1:]

        # trims right parentheses
        add_r = False
        if len(word) > 1 and word[-1] == ")":
            add_r = True
            word = word[:-1]

        # translate strings to ints
        if word:
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
            translated.append(ETokens.R_PAR)

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


def _stream_to_cnf(
    stream: list[int], max_var: int, equivalences: bool
) -> tuple[list[list[int]], int]:
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
                raise RuntimeError("Unexpected token.")

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
                raise RuntimeError("Unexpected token.")

        # awaited "and" or "or" tokens  - most left position between parentheses
        elif state == EAT.AND_OR_OR:
            if token == ETokens.AND:
                left = stack.pop()
                right = stack.pop()

                if left == right:
                    # add c <=> (L==R)
                    cnf.append([-next_gate_index, left])
                    if equivalences:
                        cnf.append([-left, next_gate_index])

                else:
                    # add C <=> (L and R)
                    cnf.append([-next_gate_index, left])
                    cnf.append([-next_gate_index, right])

                    if equivalences:
                        if left == -right:
                            raise RuntimeError(
                                "Clause would contain opposite literals."
                            )
                        cnf.append([-left, -right, next_gate_index])

                stack.append(next_gate_index)
                next_gate_index += 1

            elif token == ETokens.OR:
                left = stack.pop()
                right = stack.pop()
                if left == right:
                    # add c <=> (L==R)
                    cnf.append([-next_gate_index, left])
                    if equivalences:
                        cnf.append([-left, next_gate_index])
                else:
                    if left == -right:
                        raise RuntimeError(
                            "Clause would contain opposite literals."
                        )
                    # add C <=> (L or R)
                    cnf.append([-next_gate_index, left, right])

                    if equivalences:
                        cnf.append([-left, next_gate_index])
                        cnf.append([-right, cnf])

                stack.append(next_gate_index)
                next_gate_index += 1
            else:
                raise RuntimeError("Unexpected token.")
            state = EAT.L

        # awaited left parenthesis
        elif state == EAT.L:
            if token == ETokens.L_PAR:
                if not rec_s:
                    raise RuntimeError("Invalid formula.")
                state = rec_s.pop()
            else:
                raise RuntimeError("Unexpected token.")
        else:
            raise RuntimeError("Unrecognized state.")

    # check correct state of stack
    if len(stack) == 1 and len(rec_s) == 0 and state == EAT.VAR_OR_R_OR_NOT:
        root = stack.pop()
        if root == 1:
            # formula was just a single variable
            return [[1]], -1
        elif root > last_var:
            cnf.append([root])
            return cnf, root
    raise RuntimeError("Invalid formula.")


def to_str(
    clauses: list[list[int]],
    root: int,
    var_map: dict[str, int],
) -> str:
    """
    Returns DIMACS encoding of cnf for given formula.

    Params
    ------
    clauses: list of lists of ints representing variables or gates in one clause
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
    ret.append(f"p cnf {root} {len(clauses)}")

    for c in clauses:
        ret.append(" ".join(map(str, c)) + " 0")
    return "\n".join(ret)


def formula2cnf(
    formula: str, equivalences: bool
) -> tuple[list[list[int]], int, dict[str, int]]:
    """
    Return cnf (represented by list of tuples of clause literals), root and (variable_name -> literal) mapping.

    Equivalences specify implications [False] or equivalences [True] between gates and corresponding clauses.
    """
    seq, v_map = _translate(formula)
    cnf, root = _stream_to_cnf(seq, len(v_map), equivalences)
    return cnf, root, v_map


def read_input(input_loc: str) -> str:
    """Read input string from given location."""
    if input_loc is None:
        string = sys.stdin.read()
    else:
        if not os.path.exists(input_loc):
            input_loc = os.path.join(os.path.dirname(__file__), input_loc)
        with open(input_loc, "r") as f:
            string = f.read()
    return string


def write_output(string: str, output_loc: str) -> str:
    if output_loc is None:
        print(string)
    else:
        with open(output_loc, "w") as f:
            f.write(string)


def parse_args(args=sys.argv[1:]) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input", nargs="?", type=str, help="Input file.")
    parser.add_argument("output", nargs="?", type=str, help="Output file.")
    parser.add_argument(
        "-e",
        "--equivalences",
        action="store_true",
        help="CNF with equivalences. Otherwise left-to-right implications only.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    tseitin = read_input(args.input)

    cnf, root, v_map = formula2cnf(tseitin, args.equivalences)

    result = to_str(cnf, root, v_map)

    write_output(result, args.output)
