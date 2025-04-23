#!/bin/bash

python plot_pg19.py --run-dirs \
    logs/exp13_Dragon-L-GDN-scalable_softmax-dff-deepseekinit-skyladder_0.6-adamw_ad2556a5 \
    logs/exp13_Dragon-L-scalable_softmax-dff-deepseekinit-adamw_98dc1763 \
    logs/exp13_Dragon-L-scalable_softmax-dff-deepseekinit-repart_middle-adamw_8192bc2a \
    logs/exp13_Dragon-L-scalable_softmax-dff-deepseekinit-skyladder-repart_middle-adamw_0a417b86 \
    logs/exp14_Dragon-L-GDN-rope_to_nope-adamw_dbbb4776 \
    logs/exp14_Dragon-L-GDN-rope_to_nope-skyladder_0.6-adamw_7b191408 \
    logs/exp14_Dragon-L-GDN-rope_to_nope-skyladder-repart_middle-adamw_98777723 \
    logs/exp14_Dragon-L-rope_to_nope-adamw_691e02b2 \
    logs/exp14_Dragon-L-rope_to_nope-skyladder_0.6-adamw_a61ee694 \
    logs/exp14_Dragon-L-rope_to_nope-skyladder-repart_middle-adamw_8357966b \
    --names \
    exp13-GDN-skyladder \
    exp13-Mamba2 \
    exp13-Mamba2-repart_middle \
    exp13-Mamba2-repart_middle-skyladder \
    exp14-GDN \
    exp14-GDN-skyladder \
    exp14-GDN-repart_middle-skyladder \
    exp14-Mamba2 \
    exp14-Mamba2-skyladder \
    exp14-Mamba2-repart_middle-skyladder