from lab import *

if __name__ == "__main__":
    jobs = []
    for P in grid(crowd=[0, 0.5, 1, 1.5, 2], stairs=[1, 2, 3, 3.5, 4, 5, 6], cover=[0, -0.3]):
        jobs.append((4, None, P, "SRLB"))
    for P in grid(turn=[3, 5, 10, 100], kB=[1, 1.5, 2, 3]):
        jobs.append((5, None, {"turn": P["turn"], "turnB": P["turn"] * P["kB"]}, "SRLB"))
    for P in grid(turnR=[0, 0.25], turnL=[2.5, 3, 3.5, 4, 5], turnB=[6, 7, 8, 10, 12]):
        jobs.append((8, None, P, "SRLB"))
    for P in grid(turn=[0.25, 0.5, 0.75, 1], turnB=[6, 8, 10, 20], crowd=[2.5, 3, 3.5, 4], cover=[0], stairs=[0]):
        jobs.append((7, ("fragile", True), P, "SRLB"))
    print(len(jobs), "jobs")
    res = run(jobs)
    report(jobs, res, top=5)
