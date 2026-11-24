"""Send a MAF to a running server and print one line per sample.

ONCOAI_KEY=<analyst key> python examples/api_client.py my_samples.maf [--url http://127.0.0.1:8080]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("maf")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--min-vaf", type=float, default=0.0)
    args = parser.parse_args()
    key = os.environ.get("ONCOAI_KEY")
    if not key:
        print("set ONCOAI_KEY to an analyst key", file=sys.stderr)
        return 2
    with open(args.maf, "rb") as handle:
        body = handle.read()
    request = urllib.request.Request(
        f"{args.url}/v1/analyse?min_vaf={args.min_vaf}",
        data=body,
        method="POST",
        headers={"x-api-key": key, "content-type": "text/tab-separated-values"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - URL given by the user
        result = json.load(response)
    print(result["notice"])
    print(f"QC: {result['qc']['kept']} of {result['qc']['rows']} rows kept, {result['qc']['samples']} samples")
    for s in result["samples"]:
        p, d = s["profile"], s["prediction"]
        call = "uncertain" if d["abstained"] else d["call"]
        print(f"{p['sample']:<20} TMB {p['tmb_per_mb']:>7.2f}/Mb  {call:<10} (most likely {d['most_likely']}, p={d['confidence']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
