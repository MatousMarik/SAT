from cdcl_heuristics import *
from dpll_watched_cache import *
import argparse


def gen_queens_cnf(n: int) -> list[list[int]]:
    """
    Returns cnf for n_queens problem

    variable x_{i, j} has index i * n + j + 1
    where i is row and j is column
    """

    def map_to_index(i, j):
        return n * i + j + 1

    cnf = []

    # at least one queen
    for i in range(n):
        cnf.append([map_to_index(i, j) for j in range(n)])

    # at most one queen (row and col)
    for i in range(n):
        for j in range(n):
            for k in range(j):
                cnf.append([-map_to_index(i, k), -map_to_index(i, j)])  # rows
                cnf.append(
                    [-map_to_index(k, i), -map_to_index(j, i)]
                )  # columns

    # at most one queen (diags)
    for i in range(n - 1):
        for j in range(1, n - i):
            for k in range(j):
                cnf.append([-map_to_index(i + j, j), -map_to_index(i + k, k)])
                cnf.append(
                    [
                        -map_to_index(n - i - 1 - j, j),
                        -map_to_index(n - i - 1 - k, k),
                    ]
                )
                if i > 0:
                    cnf.append(
                        [-map_to_index(j, i + j), -map_to_index(k, i + k)]
                    )
                    cnf.append(
                        [
                            -map_to_index(n - j - 1, i + j),
                            -map_to_index(n - k - 1, i + k),
                        ]
                    )

    return cnf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="Number of queens.")
    args = parser.parse_args(["4"])

    cnf = gen_queens_cnf(args.n)

    solver = CDCL_watched_solver(cnf, args.n**2)
    result = solver.solve()
    print(result)
