#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

python mava/systems/eason/anakin/nash_ppo.py