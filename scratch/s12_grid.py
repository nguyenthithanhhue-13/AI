from lab import *

if __name__ == "__main__":
    jobs = []
    for P in grid(turn=[1.1, 1.2, 1.25, 1.3, 1.4, 1.5], turnB=[12, 13, 14, 15, 16], crowd=[5, 5.25, 5.5, 5.75, 6], cover=[0, -0.1, 0.1]):
        jobs.append((7, ("fragile", True), P, "SRLB"))
    print(len(jobs), "jobs")
    res = run(jobs)
    report(jobs, res, top=12)
