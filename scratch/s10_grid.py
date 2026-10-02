from lab import *

if __name__ == "__main__":
    jobs = []
    for P in grid(crowd=[0, 1, 2], stairs=[1, 2, 3, 4, 6], cover=[0, -0.3, -0.5]):
        jobs.append((4, ("rain", True), P, "SRLB"))
    for P in grid(turn=[2, 2.5, 3, 3.5, 4], turnB=[24, 26, 28, 30, 32, 36, 40]):
        jobs.append((5, None, P, "SRLB"))
    for P in grid(turnR=[0, 0.25, 0.5], turnL=[4, 4.5, 5, 6], turnB=[8, 9, 10, 12, 16]):
        jobs.append((8, None, P, "SRLB"))
    for P in grid(turn=[0.5, 0.75, 1], turnB=[10, 12, 16, 30], crowd=[4, 5, 6, 8], stairs=[0]):
        jobs.append((7, ("fragile", True), P, "SRLB"))
    print(len(jobs), "jobs")
    res = run(jobs)
    report(jobs, res, top=5)
