#!/usr/bin/env python3
"""
Download Middlebury 2014 ground-truth disparity for the curated sample pairs.
-----------------------------------------------------------------------------
The Middlebury scenes in ../sample_pairs are the "imperfect" variants
(view1 = im0.png, view2 = im1.png). This fetches, per scene:
    disp0.pfm  (GT disparity for view1)
    disp1.pfm  (GT disparity for view2)
    calib.txt  (focal length, baseline, doffs -> needed to turn disparity into depth)
    im0.png    (only used to verify that view1 really is im0)

Files go to ../data/middlebury_gt/<scene>/ which is git-ignored (each .pfm is ~25MB).
"""

import argparse
import os
import urllib.request

import numpy as np
from PIL import Image

BASE_URL = 'https://vision.middlebury.edu/stereo/data/scenes2014/datasets'

# sample_pairs folder -> Middlebury dataset name
SCENES = {
    '02_motorcycle': 'Motorcycle-imperfect',
    '03_playtable': 'Playtable-imperfect',
    '05_pipes': 'Pipes-imperfect',
}
FILES = ['calib.txt', 'disp0.pfm', 'disp1.pfm', 'im0.png']


def download(url, dst):
    if os.path.exists(dst):
        print(f'  [skip] {dst}')
        return
    print(f'  [get ] {url}')
    tmp = dst + '.part'
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)


def verify_same_image(a, b):
    ia, ib = np.asarray(Image.open(a).convert('RGB')), np.asarray(Image.open(b).convert('RGB'))
    return ia.shape == ib.shape and np.abs(ia.astype(int) - ib.astype(int)).mean() < 1.0


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=os.path.join(script_dir, '..', 'data', 'middlebury_gt'))
    parser.add_argument('--pairs', default=os.path.join(script_dir, '..', 'sample_pairs'))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, '.gitignore'), 'w') as f:
        f.write('*\n')

    for folder, name in SCENES.items():
        print(f'{folder} <- {name}')
        scene_dir = os.path.join(args.out, folder)
        os.makedirs(scene_dir, exist_ok=True)
        for fname in FILES:
            download(f'{BASE_URL}/{name}/{fname}', os.path.join(scene_dir, fname))

        view1 = os.path.join(args.pairs, folder, 'view1.png')
        ok = verify_same_image(view1, os.path.join(scene_dir, 'im0.png'))
        print(f'  view1 == im0.png: {"OK" if ok else "MISMATCH - GT will not line up!"}')
