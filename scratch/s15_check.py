from lab import *

if __name__ == "__main__":
    jobs = [(5, None, {"turn": 3, "turnB": 30, "first_turn": 0}, "SRLB"), (8, None, {"turnR": 0.5, "turnL": 4.5, "turnB": 9, "first_turn": 0}, "SRLB"),
            (0, None, {}, "SLRB"), (0, None, {}, "RSLB"), (0, None, {}, "BSRL"), (5, None, {"turn": 3, "turnB": 30}, "SLRB")]
    res = run(jobs, procs=4)
    for j, r in zip(jobs, res): print(j, r)
