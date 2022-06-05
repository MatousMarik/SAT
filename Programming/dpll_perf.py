from dpll import *
import pandas

tests = [
    ("uf20-91", "/uf20-0", 20),
    ("uf50-218", "/uf50-0", 20),
    ("uuf50-218", "/uuf50-0", 20),
    ("uf75-325", "/uf75-0", 10),
    ("uuf75-325", "/uuf75-0", 10),
    ("uf100-430", "/uf100-0", 5),
    ("uuf100-430", "/uuf100-0", 5),
    ("uf125-538", "/uf125-0", 3),
    ("uf150-645", "/uf150-0", 3),
]
suffix = ".cnf"

if __name__ == "__main__":
    results = {}
    for test, pref, tests in tests:
        sat = test.startswith("uf")
        solved = 0
        time = 0
        decisions = 0
        ups = 0
        res = []
        for i in range(1, tests + 1):
            solver = DPLL_adjacency_solver(
                *get_cnf(parse_args([test + pref + str(i) + suffix]))
            )
            r_sat, _ = solver.solve()
            it, st, d, ups_ = solver.get_stats()
            res.append(("succ" if r_sat == sat else "fail", it + st, d, ups_))
            time += it + st
            if r_sat == sat:
                solved += 1
            decisions += d
            ups += ups_
        result = pandas.DataFrame(
            res,
            columns=[
                "solution",
                "time in s",
                "decisions",
                "unit propagation steps",
            ],
            index=range(1, tests + 1),
        )
        result.to_csv("dpll_results/" + test + ".csv")
