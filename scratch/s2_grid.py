from lab import *

if __name__ == "__main__":
    jobs = []
    G = grid(crowd=[0, 0.5, 1, 2, 3], cover=[0, -0.2, -0.4, -0.6], turn=[0, 0.5, 1.5])
    for r in [1, 2, 4, 5, 8]:
        for P in G:
            if r == 4:
                for st in [0, 0.5, 1, 2]:
                    jobs.append((r, None, dict(P, stairs=st), "SRLB"))
            else:
                jobs.append((r, None, P, "SRLB"))
    for r, key in [(3, "rain"), (6, "urgent"), (7, "fragile")]:
        for v in (True, False):
            for P in G:
                jobs.append((r, (key, v), P, "SRLB"))
    print(len(jobs), "jobs")
    res = run(jobs)
    report(jobs, res)
