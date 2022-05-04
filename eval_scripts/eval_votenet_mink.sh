#!/bin/sh

CHECKPOINT=${1:-/root/code/FastPointTransformer-VoteNet/checkpoints/votenet-mink}

python eval.py checkpoint_dir=$CHECKPOINT model_name=VoteNetMink