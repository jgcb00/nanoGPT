#!/bin/bash

python plot_pg19.py --run-dirs \
    logs/exp14long_Dragon-L-GDN-independent_gn_unique-qk_norm-new_rmsnormweights-adamw_f8657bb7 \
    logs/exp1long_GPT2-L-new_codebase-skyladder-adamw_8f6490ad \
    logs/exp6long_GPT2-L-new_codebase-skyladder-adamw_f13d189e \
    --names \
    exp14-independent_unique-qknorm-new_weights \
    exp1 \
    exp6 \