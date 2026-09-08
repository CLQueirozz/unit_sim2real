#!/bin/bash
mkdir datasets/soybean/train1 -p
mkdir datasets/soybean/train0 -p
mkdir datasets/soybean/test1 -p
mkdir datasets/soybean/test0 -p
for f in datasets/soybean/train/*; do convert -quality 100 -crop 50%x100% +repage $f datasets/soybean/train%d/${f##*/}; done;
for f in datasets/soybean/test/*; do convert -quality 100 -crop 50%x100% +repage $f datasets/soybean/test%d/${f##*/}; done;
mv datasets/soybean/train0 datasets/soybean/trainA
mv datasets/soybean/train1 datasets/soybean/trainB
mv datasets/soybean/test0 datasets/soybean/testA
mv datasets/soybean/test1 datasets/soybean/testB
python train.py --config conf.yaml

