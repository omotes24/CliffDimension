"""Experiment 0: freeze the evaluation rules (environment, device, contact model, body, success criterion) and record them
with the code version.  usage: python scripts/exp/exp0_config.py --out results/suite"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from katsumi.exp.common import config_dict
from katsumi.exp.util import save_json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/suite"); a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg = config_dict(root)
    save_json(cfg, os.path.join(a.out, "CONFIG.json"))
    print(open(os.path.join(a.out, "CONFIG.json")).read()[:1500])


if __name__ == "__main__":
    main()
