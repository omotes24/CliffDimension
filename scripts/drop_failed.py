"""Delete non-converged solution pickles (ok == False) in a results directory so that run_grid re-solves them.
usage: python scripts/drop_failed.py results/grid [--tag ref]"""
import glob, os, pickle, sys
d = sys.argv[1]
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "*"
n = 0
for f in glob.glob(os.path.join(d, f"sol_{tag}_*.pkl")):
    try:
        r = pickle.load(open(f, "rb"))
    except Exception:
        os.remove(f); n += 1; continue
    if not r.get("ok"):
        os.remove(f); n += 1
print("removed", n)
