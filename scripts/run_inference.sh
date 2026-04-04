#!/usr/bin/env bash

set -euo pipefail

python inference.py --config configs/default.yaml --image "$1" --text "$2"