#!/bin/bash
set -e

python ../data/_gen_data/generate_two_hop.py \
  --path ../data/_dataset/_org/val_qa_data.json

python ../data/_gen_data/generate_two_hop.py \
  --path ../data/_dataset/_org/train_qa_data.json