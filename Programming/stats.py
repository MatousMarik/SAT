#!/usr/bin/env python3
"""Script created for benchmarking for the tasks."""
from dpll import *
from dpll_watched import DPLL_watched_solver as dpll_w
from dpll_watched_cache import DPLL_watched_solver as dpll_wc
from cdcl import *
import numpy as np
import os.path
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import total_ordering
from backbone import get_backbones

tests = [
    # ("uf20-91", "/uf20-0", 20),
    # ("uf50-218", "/uf50-0", 20),
    # ("uuf50-218", "/uuf50-0", 20),
    ("uf75-325", "/uf75-02", 5),
    # ("uuf75-325", "/uuf75-0", 10),
    ("uf100-430", "/uf100-02", 5),
    ("uf125-538", "/uf125-02", 5),
    ("uf150-645", "/uf150-02", 5),
    ("uf200-860", "/uf200-02", 5),
]
suffix = ".cnf"

# solver
solvers = [
    (dpll_wc, "wc_", 7),
]

heuristics = ["unsatisfied", "random", "vsids"]

conflicts = [32, 64, 128, 256]
lbds = [1, 2, 3, 4]


@total_ordering
@dataclass(eq=False)
class Result:
    time: float
    decisions: float
    ups: float
    restarts: float

    conflicts: int
    lbd: int
    heuristics: str = "vsids"

    def __eq__(self, __o: object) -> bool:
        return self.time == __o.time

    def __lt__(self, other) -> bool:
        return self.time < other.time

    def __hash__(self) -> int:
        return int(self.time * self.decisions * self.ups * self.restarts)


def cdcl_heuristics():
    test_count = 15
    variables = [125, 150]
    tests = [("uf125-538", "/uf125-0"), ("uf150-645", "/uf150-0")]
    params = [
        {"conflict_limit": 256, "heuristic": "unsatisfied"},
        {"conflict_limit": 128, "heuristic": "random"},
        {"conflict_limit": 64, "heuristic": "vsids"},
    ]
    columns = ["time", "decisions", "unit propagation steps", "restarts"]

    for test, pref in tests:
        data = [[], [], []]
        for j in range(1, test_count + 1):
            cnf, mv = get_cnf(test + pref + str(j) + suffix)
            for i, kvargs in enumerate(params):
                solver = CDCL_watched_solver(cnf, mv, **kvargs)
                solver.solve()
                it, st, d, ups, resets = solver.get_stats()

                data[i].append([it + st, d, ups, resets])
        for i in range(len(params)):
            df = pd.DataFrame(data[i], columns=columns)
            print(df)
            df.to_csv(
                "dpll_results/" + test + "_" + params[i]["heuristic"] + ".csv"
            )


def cdcl_h_stats():
    variables = [125, 150]
    tests = [("uf125-538", "/uf125-0"), ("uf150-645", "/uf150-0")]
    heuristics = ["unsatisfied", "random", "vsids"]

    columns = ["time", "decisions", "unit propagation steps", "restarts"]
    for test, _ in tests:
        averages = []
        for h in heuristics:
            df = pd.read_csv("dpll_results/" + test + "_" + h + ".csv")
            df = df.mean(axis=0)
            df = df.iloc[1:]
            df = df.to_frame(h).transpose()
            averages.append(df)
        df = pd.concat(averages, axis=0)
        print(df.to_markdown())


def backbone():
    test_count = 3
    tests = [
        ("uf20-91", "/uf20-0"),
        ("uf50-218", "/uf50-0"),
        ("uf75-325", "/uf75-0"),
        ("uf100-430", "/uf100-0"),
        ("uf125-538", "/uf125-0"),
        ("uf150-645", "/uf150-0"),
        ("uf200-860", "/uf200-0"),
    ]

    result = []
    for test, pref in tests:
        tcalls, ttime = 0, 0
        for j in range(1, test_count + 1):
            cnf, mv = get_cnf(test + pref + str(j) + suffix)
            _, calls, time = get_backbones(cnf, ["count"], max_var=mv)
            tcalls += calls
            ttime += time
        tcalls /= test_count
        ttime /= test_count
        result.append((test, tcalls, ttime))

    df = pd.DataFrame(result, columns=["test", "calls", "time"])
    print(df.to_markdown(index=False))
    df.to_csv("dpll_results/backbones_c_f.csv")


def grid():
    results = {}
    for test, pref, t_no in tests:
        print(test)
        statistics = []
        results[test] = statistics
        for c in conflicts:
            for lbd in lbds:
                time = 0
                for j in range(1, t_no + 1):
                    solver = CDCL_watched_solver(
                        *get_cnf(test + pref + str(j) + suffix), c, lbd
                    )
                    sat, _ = solver.solve()
                    assert sat
                    it, st, _, _, _ = solver.get_stats()
                    time += it + st
                time /= t_no
                statistics.append((time, lbd, c))
        statistics.sort()
        print(statistics[:3])
        print()


def comparison():
    for sc, sn, max_i in solvers:
        for i, (test, pref, tests) in enumerate(tests):
            print(test)
            sat = test.startswith("uf")
            solved = 0
            time = 0
            decisions = 0
            ups = 0
            res = []
            for j in range(1, tests + 1):
                solver = sc(*get_cnf(test + pref + str(j) + suffix))
                r_sat, _ = solver.solve()
                it, st, d, ups_ = solver.get_stats()
                res.append(
                    ("succ" if r_sat == sat else "fail", it + st, d, ups_)
                )
                time += it + st
                if r_sat == sat:
                    solved += 1
                decisions += d
                ups += ups_
            result = pd.DataFrame(
                res,
                columns=[
                    "solution",
                    "time in s",
                    "decisions",
                    "unit propagation steps",
                ],
                index=range(1, tests + 1),
            )
            print(result)
            result.to_csv("dpll_results/" + sn + test + ".csv")
            if i == max_i:
                break


def plot_comparison():
    tests = [
        ("uf20-91", "/uf20-0", 20),
        ("uf50-218", "/uf50-0", 50),
        ("uuf50-218", "/uuf50-0", -50),
        ("uf75-325", "/uf75-0", 75),
        ("uuf75-325", "/uuf75-0", -75),
        ("uf100-430", "/uf100-0", 100),
        ("uf125-538", "/uf125-0", 125),
        ("uf150-645", "/uf150-0", 150),
    ]

    solvers = [
        ["dpll", ""],
        ["dpll_wl", "w_"],
        ["dpll_wl_c", "wc_"],
        ["cdcl", "c_"],
    ]

    test_results = []
    test_is = []

    for test, _, t_no in tests:
        frames = []
        for l in solvers:
            solver_name, pref = l
            result = "dpll_results/" + pref + test + ".csv"
            if not os.path.isfile(result):
                df = pd.DataFrame(
                    None,
                    columns=[
                        "solution",
                        "time in s",
                        "decisions",
                        "unit propagation steps",
                    ],
                )
            else:
                df = pd.read_csv(result)
            df = df.mean(axis=0)
            df = df.iloc[1:]
            df = df.to_frame().transpose()
            df.columns = pd.MultiIndex.from_product([[solver_name], df.columns])
            frames.append(df)
        df = pd.concat(frames, axis=1)
        df.insert(0, "examples", t_no)
        test_results.append(df)

    df = pd.concat(test_results)
    # print(df.to_html(index=False))

    df = df[df["examples"] > 0]
    x = df["examples"].tolist()
    y_names = []
    ys = []
    for s, _ in solvers:
        ys.append(df[s]["time in s"].tolist())
        y_names.append(s)
    # ys[0][3] = ys[0][4]
    # ys[2][-1] = ys[0][-1]
    for y, n in zip(ys, y_names):
        plt.plot(x, y, label=n, linestyle="--", marker="o")
    plt.xlabel("examples")
    plt.xticks(x)
    plt.ylabel("time in s")
    plt.ylim(bottom=0, top=40)
    plt.legend()
    plt.savefig("comparison.svg")


if __name__ == "__main__":
    backbone()
