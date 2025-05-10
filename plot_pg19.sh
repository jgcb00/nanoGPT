#!/bin/bash

python plot_pg19.py --run-dirs \
    logs/exp14long_Dragon-L-GDN-independent_gn_shared-qk_norm-new_rmsnormweights-adamw_b0299b26 \
    logs/exp14long_Dragon-L-GDN-independent_gn_unique-qk_norm-new_rmsnormweights-adamw_f8657bb7 \
    logs/exp14long_Dragon-L-GDN-no_SS-independent_gn_unique-qk_norm-new_rmsnormweights-adamw_8654c809 \
    logs/exp12long_Dragon-L-GDN-independent_gn_unique-qk_norm-new_rmsnormweights-adamw_bb5524f0 \
    logs/exp14long_Dragon-L-GDN-hymba_norm-qk_norm-new_rmsnormweights-adamw_9eeff779 \
    --names \
    exp14-independent_shared-qknorm-new_weights \
    exp14-independent_unique-qknorm-new_weights \
    exp14-no-ss-independent_unique-qknorm-new_weights \
    exp12-independent_unique-qknorm-new_weights \
    exp14-hymba_norm-qknorm-new_weights \