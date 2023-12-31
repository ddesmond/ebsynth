#!/bin/sh
nvcc -gencode arch=compute_50,code=sm_50 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_86,code=sm_86 src/ebsynth.cpp src/ebsynth_cpu.cpp src/ebsynth_cuda.cu -I"include" -DNDEBUG -D__CORRECT_ISO_CPP11_MATH_H_PROTO -O6 -std=c++11 -w -Xcompiler -fopenmp -o bin/ebsynth
