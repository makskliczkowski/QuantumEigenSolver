#! /usr/bin/env bash

cd "$(dirname "$0")"
# make sure script is executable
chmod +x evolution.py

# run with:
python evolution.py         \
    ./data_big_times        \
    0.68                    \
    0.06                    \
    4                       \
    20                      \
    13                      \
    15                      \
    1                       \
    100000                  \
    10