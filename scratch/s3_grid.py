from lab import *

if __name__ == "__main__":
    jobs = []
    for P in grid(crowd=[3, 4, 5, 6, 10, 50], cover=[0, -0.1, 0.1]):
        jobs.append((1, None, P, "SRLB"))
    for P in grid(cover=[-0.3, -0.5, -0.6, -0.7, -0.8, -0.9], crowd=[0, 0.1]):
        jobs.append((2, None, P, "SRLB"))
    for P in grid(cover=[-0.5, -0.6, -0.7, -0.8], crowd=[0, 0.2, 0.4], normal=[0, 0.2, 0.4]):
        jobs.append((3, ("rain", True), P, "SRLB"))
    for P in grid(crowd=[0.75, 1, 1.25, 1.5], stairs=[-0.5, -0.25, 0, 0.25], cover=[0, -0.2, 0.2]):
        jobs.append((4, None, P, "SRLB"))
    for r in (5, 8):
        for P in grid(turnR=[0, 1, 2, 4], turnL=[0, 1, 2, 4], turnB=[0, 1, 2, 4, 8], first_turn=[0, 1]):
            jobs.append((r, None, P, "SRLB"))
    for P in grid(crowd=[1, 2, 3], cover=[0, -0.3, 0.3], turn=[0, 1, 2], first_turn=[0, 1]):
        jobs.append((7, ("fragile", True), P, "SRLB"))
    for P in grid(crowd=[2, 3, 4, 5], cover=[-0.5, -0.6, -0.7, -0.8]):
        jobs.append((6, ("urgent", False), P, "SRLB"))
    print(len(jobs), "jobs")
    res = run(jobs)
    report(jobs, res, top=5)
