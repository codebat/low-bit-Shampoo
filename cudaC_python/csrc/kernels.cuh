#ifndef kernels_H
#define kernels_H

#include <float.h>
#include <ops.cuh>

template<typename T, int BLOCK_SIZE, int NUM_PER_TH> __global__ void kQuantizeBlockwise8bit(T *A, float *code, const int order, float *absmax, unsigned char *out);
template<typename T, int BLOCK_SIZE, int NUM_PER_TH> __global__ void kQuantizeBlockwise4bit(T *A, float *code, const int order, float *absmax, unsigned char *out);

template<typename T, int TILE_SIZE, int NUM_PER_TH> __global__ void kDequantizeBlockwise8bit(unsigned char *A, float *code, const int order, float *absmax, T *out, const int blocksize);
template<typename T, int TILE_SIZE, int NUM_PER_TH> __global__ void kDequantizeBlockwise4bit(unsigned char *A, float *code, const int order, float *absmax, T *out, const int blocksize);

template<typename T, int BLOCK_SIZE, int NUM_PER_TH> __global__ void kQuantizeBlockwise8bitDiagReal(T *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template<typename T, int BLOCK_SIZE, int NUM_PER_TH> __global__ void kQuantizeBlockwise4bitDiagReal(T *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);

template<typename T, int TILE_SIZE, int NUM_PER_TH> __global__ void kDequantizeBlockwise8bitDiagReal(unsigned char *A, float *code, const int order, float *absmax, float *diag, T *out, const int blocksize);
template<typename T, int TILE_SIZE, int NUM_PER_TH> __global__ void kDequantizeBlockwise4bitDiagReal(unsigned char *A, float *code, const int order, float *absmax, float *diag, T *out, const int blocksize);

#endif
