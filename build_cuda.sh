#!/usr/bin/bash

echo "Building the astroio library for CUDA..."

[ -d build ] || mkdir build
cd build
cmake .. -DUSE_CUDA=ON \
         -DUSE_OPENMP=OFF \
         -DCMAKE_CUDA_ARCHITECTURES="61-real" \
         -DMAKE_BUILD_TYPE=Release \
         -DCMAKE_C_COMPILER=nvcc \
         -DCMAKE_CXX_COMPILER=nvcc \
         -DCMAKE_CXX_FLAGS="-O3"

make -j 8 VERBOSE=1
sudo make install
