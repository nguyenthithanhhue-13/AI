from lab import *

if __name__ == "__main__":
    jobs = []
    for P in grid(turn=[0.5, 1, 1.5, 2, 3], turnB=[6, 8, 12, 20, 100], crowd=[0]):
        jobs.append((5, None, P, "SRLB"))
    for P in grid(turnR=[0, 0.5, 1], turnL=[1, 1.5, 2, 3], turnB=[2, 3, 4, 5, 6]):
        jobs.append((8, None, P, "SRLB"))
    for P in grid(turn=[0, 0.5, 1, 2], turnB=[0, 2, 4, 8, 20], crowd=[0, 1, 2, 3], cover=[0]):
        jobs.append((7, ("fragile", True), P, "SRLB"))
    for P in grid(crowd=[3, 3.5, 4, 4.5], cover=[-0.3, -0.4, -0.5]):
        jobs.append((6, ("urgent", False), P, "SRLB"))
    for P in grid(crowd=[1], stairs=[0], turn=[0, 0.5, 1], turnB=[0, 1, 2, 4]):
        jobs.append((4, None, P, "SRLB"))
    print(len(jobs), "jobs")
    res = run(jobs)
    report(jobs, res, top=5)
