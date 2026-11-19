import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, required=True)
args = parser.parse_args()
time.sleep(args.seconds)
print("SLEEP_COMPLETED", flush=True)
