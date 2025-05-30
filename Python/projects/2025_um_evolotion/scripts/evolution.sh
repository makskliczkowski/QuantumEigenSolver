#! /usr/bin/env bash

cd "$(dirname "$0")"
# make sure script is executable
chmod +x evolution.py

# run with:
python evolution.py         \
    ./data_big_times        \
    0.68                    \
    0.03                    \
    9                       \
    40                      \
    14                      \
    15                      \
    1                       \
    100000                  \
    5