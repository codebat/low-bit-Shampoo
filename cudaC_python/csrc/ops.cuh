#ifndef ops_H
#define ops_H

#include <stdio.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

template <typename T> void quantizeBlockwise8bit(T *A, float *code, const int order, float *absmax, unsigned char *out, int blocksize);
template <typename T> void quantizeBlockwise4bit(T *A, float *code, const int order, float *absmax, unsigned char *out, int blocksize);

template <typename T> void dequantizeBlockwise8bit(unsigned char *A, float *code, const int order, float *absmax, T *out, int blocksize);
template <typename T> void dequantizeBlockwise4bit(unsigned char *A, float *code, const int order, float *absmax, T *out, int blocksize);

template <typename T> void quantizeBlockwise8bitDiagReal(T *A, float *code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize);
template <typename T> void quantizeBlockwise4bitDiagReal(T *A, float *code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize);

template <typename T> void dequantizeBlockwise8bitDiagReal(unsigned char *A, float *code, const int order, float *absmax, float *diag, T *out, int blocksize);
template <typename T> void dequantizeBlockwise4bitDiagReal(unsigned char *A, float *code, const int order, float *absmax, float *diag, T *out, int blocksize);

#define CUDA_CHECK_RETURN(value) {                              \
  cudaError_t _m_cudaStat = value;                              \
  if (_m_cudaStat != cudaSuccess) {                             \
    fprintf(stderr, "Error %s at line %d in file %s\n",         \
        cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                                                    \
  } }

#endif
