#!/bin/bash

set -eux

DEVICE=${DEVICE:-cuda}
MODEL=${MODEL:-DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}
IMAGE_SIZE=${IMAGE_SIZE:-512}

if [[ "$MODEL" == *"224"* ]]; then
    IMAGE_SIZE=224
fi

exec python3 demo.py --weights "checkpoints/$MODEL" --device "$DEVICE" --image_size "$IMAGE_SIZE" --local_network "$@"
