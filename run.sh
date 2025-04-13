#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
python mava/systems/eason/anakin/nash_ppo.py

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=1
python mava/systems/eason/anakin/nash_ma_ppo.py

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=2
python mava/systems/ppo/anakin/ff_mappo.py

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=3
python mava/systems/ppo/anakin/ff_ippo.py
