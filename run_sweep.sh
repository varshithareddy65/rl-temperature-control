#!/bin/sh
# Smoothness-weight sweep (each run is a full from-scratch training, ~minutes on CPU). Old model is auto-backed-up if paths clash.
for w in 0.3 1.0 3.0; do
  python train.py --tag sw$w --set reward.w_smooth=$w
done
python compare_models.py --models v1=models/ppo_temperature.zip sw0.3=models/ppo_sw0.3.zip sw1.0=models/ppo_sw1.0.zip sw3.0=models/ppo_sw3.0.zip
