#!/bin/bash

# To perform compression on images
python3 input_transfer.py $2 ../transformed/outRef.png --downsampling 4 --quality 75
python3 input_transfer.py $1 ../transformed/inputRef.png --downsampling 4 --quality 75

# To obtain recipes and final reconstruction
python3 run.py $1