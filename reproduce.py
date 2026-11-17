"""Regenerate the submission from the data bundle and the trained networks. One command:

    python reproduce.py                           # writes Gallax_AI.csv and prints its SHA-256
    python reproduce.py --expect <sha256>         # and fails unless the file is exactly that one
    python reproduce.py --models DIR --out FILE   # networks from DIR/second_touch_sw2_channels128_sc1_touch_wc, output to FILE
    python reproduce.py --fresh-sync              # redo the broadcast alignment even if it exists

Steps, all deterministic and on the CPU (a GPU's floating point would not give the same bytes):
  1. Align the TV broadcast to the tactical video (possession/broadcast_sync.py), saved to
     data/derived/broadcast_sync.npz; done once, about 20-25 minutes on one CPU core.
  2. Load the 20 networks of the submission: the TCN at 128 channels trained on the second-touch
     labels of the World Cup and of SkillCorner's open games together, short possessions weighted,
     with the touch detector's probabilities as inputs, seeds 0-4 x folds 0-3
     (bench_touch_input.py --labels second_touch --short-weight 2 --hp channels=128 --sc-weight 1
     --seed N --detector wc_res10), from <models>/second_touch_sw2_channels128_sc1_touch_wc, and the
     four touch detectors, trained on the World Cup (touch_detector.py --sources wc --res10), from its
     detector/.
     Nothing is trained here, and nothing reads the ground truth.
  3. Features from the tactical tracking, its missing ball filled from the aligned broadcast; the
     detectors' touch probabilities averaged and added; the networks' probabilities averaged; Viterbi
     at switching cost 7.
  4. Write the CSV through possession/submission.py, which validates it like the official
     evaluation/validate_submission.py, and print its SHA-256.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported anywhere
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "possession"))
import bench  # noqa: E402
import frames  # noqa: E402
import temporal  # noqa: E402

SUBMISSION = "tcn_second_touch_sw2_ch128_sc1_touch_wc_seeds_sync"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=temporal.MODEL_DIR, help="directory holding second_touch_sw2_channels128_sc1_touch_wc/")
    parser.add_argument("--out", default=os.path.join(ROOT, "Gallax_AI.csv"))  # the rules: <team_name>.csv
    parser.add_argument("--expect", help="SHA-256 the output must have")
    parser.add_argument("--fresh-sync", action="store_true", help="redo the broadcast alignment")
    args = parser.parse_args()
    if args.fresh_sync or not os.path.isfile(frames.SYNC_PATH):
        print("1. aligning the broadcast to the tactical video ...", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "possession", "broadcast_sync.py")], check=True)
    else:
        print("1. broadcast alignment found:", os.path.relpath(frames.SYNC_PATH, ROOT))
    print("2-4. networks, decoding and export on the CPU ...", flush=True)
    path = bench.submit(SUBMISSION, model_dir=args.models)
    shutil.copyfile(path, args.out)
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print("wrote {}\nsha256 {}".format(args.out, digest))
    if args.expect and digest != args.expect:
        sys.exit("FAILED: expected sha256 " + args.expect)


if __name__ == "__main__":
    main()
