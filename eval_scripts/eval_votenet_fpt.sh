#!/bin/sh

CHECKPOINT=${1:-/root/code/FastPointTransformer-VoteNet/checkpoints/votenet-fpt}

python eval.py checkpoint_dir=$CHECKPOINT model_name=VoteNetFPT