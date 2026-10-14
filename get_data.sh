#!/usr/bin/env sh
# Fetch the NIMS fatigue data as released with Agrawal et al. (2014), IMMI 3:8
# (supplementary file 40192_2013_16_MOESM1_ESM.csv), from a public mirror.
set -e
mkdir -p data
git clone --depth 1 https://github.com/blaiszik/steel-fatigue-machine-learning.git /tmp/nims_src
cp /tmp/nims_src/data/40192_2013_16_MOESM1_ESM.csv data/
echo "data/40192_2013_16_MOESM1_ESM.csv ready"
