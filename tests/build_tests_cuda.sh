#!/usr/bin/bash

nvcc ./astroio_test.cpp -o astroio_test  -I/usr/local/include -I/usr/local/cuda-11.8/include -L/usr/local/lib -L/usr/local/cuda-11.8/lib64 -lcudart -lm -lpthread -lblink_astroio
