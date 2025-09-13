#! /usr/bin/env bash

cd "$(dirname "$0")"
# make sure script is executable
echo "Running evolution script in $(pwd)"
chmod +x ./evolution.py

# run with:
# 1. save dir
# 2. alpha_start
# 3. alpha_step
# 4. alpha_num
# 5. number of realizations
# 6. Ns_start
# 7. Ns_stop
# 8. additional parameter
# 9. number of timesteps
# 10. memory per job
# 11. maximal memory
    # --save_dir data_big_times_fixed_f \
python ./evolution.py \
    --save_dir data \
    --alpha_start 1.2 \
    --alpha_step 0.2 \
    --alphas_number 4 \
    --sites_start 9 \
    --sites_end 9 \
    --number_of_realizations 10 \
    --n 1 \
    --time_num 100000 \
    --max_memory 16 \
    --memory_per_worker 4 \
    --model rpm \
    --uniform 1
