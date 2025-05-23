#! /usr/bin/env bash

cd "$(dirname "$0")"
# make sure script is executable
chmod +x evolution.py

# run with:
python evolution.py         \
    ./data_test             \
    0.7                     \
    0.04                    \
    1                       \
    10                      \
    6                       \
    7                       \
    1                       \
    50000                   \
    1.5