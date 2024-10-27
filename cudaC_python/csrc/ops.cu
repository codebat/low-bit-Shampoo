#include <ops.cuh>
#include <kernels.cuh>

template <typename T> void quantizeBlockwise8bit(T *A, float *code, const int order, float *absmax, unsigned char *out, int blocksize)
{
    int num_blocks_row = order / blocksize;
    num_blocks_row = order % blocksize == 0 ? num_blocks_row : num_blocks_row + 1;
    dim3 grid(order, num_blocks_row);

    switch(blocksize)
    {
        case 2048:
            kQuantizeBlockwise8bit<T, 2048, 4><<<grid, 512>>>(A, code, order, absmax, out);
            break;
        case 1024:
            kQuantizeBlockwise8bit<T, 1024, 4><<<grid, 256>>>(A, code, order, absmax, out);
            break;
        case 512:
            kQuantizeBlockwise8bit<T, 512, 2><<<grid, 256>>>(A, code, order, absmax, out);
            break;
        case 256:
            kQuantizeBlockwise8bit<T, 256, 2><<<grid, 128>>>(A, code, order, absmax, out);
            break;
        case 128:
            kQuantizeBlockwise8bit<T, 128, 2><<<grid, 64>>>(A, code, order, absmax, out);
            break;
        case 64:
            kQuantizeBlockwise8bit<T, 64, 2><<<grid, 32>>>(A, code, order, absmax, out);
            break;
    }

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T> void quantizeBlockwise4bit(T *A, float *code, const int order, float *absmax, unsigned char *out, int blocksize)
{
    int num_blocks_row = order / blocksize;
    num_blocks_row = order % blocksize == 0 ? num_blocks_row : num_blocks_row + 1;
    dim3 grid(order, num_blocks_row);

    switch(blocksize)
    {
        case 2048:
            kQuantizeBlockwise4bit<T, 2048, 4><<<grid, 512>>>(A, code, order, absmax, out);
            break;
        case 1024:
            kQuantizeBlockwise4bit<T, 1024, 4><<<grid, 256>>>(A, code, order, absmax, out);
            break;
        case 512:
            kQuantizeBlockwise4bit<T, 512, 2><<<grid, 256>>>(A, code, order, absmax, out);
            break;
        case 256:
            kQuantizeBlockwise4bit<T, 256, 2><<<grid, 128>>>(A, code, order, absmax, out);
            break;
        case 128:
            kQuantizeBlockwise4bit<T, 128, 2><<<grid, 64>>>(A, code, order, absmax, out);
            break;
        case 64:
            kQuantizeBlockwise4bit<T, 64, 2><<<grid, 32>>>(A, code, order, absmax, out);
            break;
    }

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T> void dequantizeBlockwise8bit(unsigned char *A, float *code, const int order, float *absmax, T *out, int blocksize)
{
    int tile_size = 512;
    int num_blocks_row = order / tile_size;
    num_blocks_row = order % tile_size == 0 ? num_blocks_row : num_blocks_row + 1;
    dim3 grid(order, num_blocks_row);

    kDequantizeBlockwise8bit<T, 512, 8><<<grid, 64>>>(A, code, order, absmax, out, blocksize);

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T> void dequantizeBlockwise4bit(unsigned char *A, float *code, const int order, float *absmax, T *out, int blocksize)
{
    int tile_size = 512;
    int num_blocks_row = order / tile_size;
    num_blocks_row = order % tile_size == 0 ? num_blocks_row : num_blocks_row + 1;
    dim3 grid(order, num_blocks_row);

    kDequantizeBlockwise4bit<T, 512, 8><<<grid, 64>>>(A, code, order, absmax, out, blocksize);

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T> void quantizeBlockwise8bitDiagReal(T *A, float *code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize)
{
    int num_blocks_row = order / blocksize;
    num_blocks_row = order % blocksize == 0 ? num_blocks_row : num_blocks_row + 1;
    dim3 grid(order, num_blocks_row);

    switch(blocksize)
    {
        case 2048:
            kQuantizeBlockwise8bitDiagReal<T, 2048, 4><<<grid, 512>>>(A, code, order, absmax, diag, out);
            break;
        case 1024:
            kQuantizeBlockwise8bitDiagReal<T, 1024, 4><<<grid, 256>>>(A, code, order, absmax, diag, out);
            break;
        case 512:
            kQuantizeBlockwise8bitDiagReal<T, 512, 2><<<grid, 256>>>(A, code, order, absmax, diag, out);
            break;
        case 256:
            kQuantizeBlockwise8bitDiagReal<T, 256, 2><<<grid, 128>>>(A, code, order, absmax, diag, out);
            break;
        case 128:
            kQuantizeBlockwise8bitDiagReal<T, 128, 2><<<grid, 64>>>(A, code, order, absmax, diag, out);
            break;
        case 64:
            kQuantizeBlockwise8bitDiagReal<T, 64, 2><<<grid, 32>>>(A, code, order, absmax, diag, out);
            break;
    }

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T> void quantizeBlockwise4bitDiagReal(T *A, float *code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize)
{
    int num_blocks_row = order / blocksize;
    num_blocks_row = order % blocksize == 0 ? num_blocks_row : num_blocks_row + 1;
    dim3 grid(order, num_blocks_row);

    switch(blocksize)
    {
        case 2048:
            kQuantizeBlockwise4bitDiagReal<T, 2048, 4><<<grid, 512>>>(A, code, order, absmax, diag, out);
            break;
        case 1024:
            kQuantizeBlockwise4bitDiagReal<T, 1024, 4><<<grid, 256>>>(A, code, order, absmax, diag, out);
            break;
        case 512:
            kQuantizeBlockwise4bitDiagReal<T, 512, 2><<<grid, 256>>>(A, code, order, absmax, diag, out);
            break;
        case 256:
            kQuantizeBlockwise4bitDiagReal<T, 256, 2><<<grid, 128>>>(A, code, order, absmax, diag, out);
            break;
        case 128:
            kQuantizeBlockwise4bitDiagReal<T, 128, 2><<<grid, 64>>>(A, code, order, absmax, diag, out);
            break;
        case 64:
            kQuantizeBlockwise4bitDiagReal<T, 64, 2><<<grid, 32>>>(A, code, order, absmax, diag, out);
            break;
    }

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T> void dequantizeBlockwise8bitDiagReal(unsigned char *A, float *code, const int order, float *absmax, float *diag, T *out, int blocksize)
{
    int tile_size = 512;
    int num_blocks_row = order / tile_size;
    num_blocks_row = order % tile_size == 0 ? num_blocks_row : num_blocks_row + 1;
    dim3 grid(order, num_blocks_row);

    kDequantizeBlockwise8bitDiagReal<T, 512, 8><<<grid, 64>>>(A, code, order, absmax, diag, out, blocksize);

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

template <typename T> void dequantizeBlockwise4bitDiagReal(unsigned char *A, float *code, const int order, float *absmax, float *diag, T *out, int blocksize)
{
    int tile_size = 512;
    int num_blocks_row = order / tile_size;
    num_blocks_row = order % tile_size == 0 ? num_blocks_row : num_blocks_row + 1;
    dim3 grid(order, num_blocks_row);

    kDequantizeBlockwise4bitDiagReal<T, 512, 8><<<grid, 64>>>(A, code, order, absmax, diag, out, blocksize);

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template void quantizeBlockwise8bit<__nv_bfloat16>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out, int blocksize);
template void quantizeBlockwise8bit<float>(float *A, float *code, const int order, float *absmax, unsigned char *out, int blocksize);

template void quantizeBlockwise4bit<__nv_bfloat16>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out, int blocksize);
template void quantizeBlockwise4bit<float>(float *A, float *code, const int order, float *absmax, unsigned char *out, int blocksize);

template void dequantizeBlockwise8bit<__nv_bfloat16>(unsigned char *A, float *code, const int order, float *absmax, __nv_bfloat16 *out, int blocksize);
template void dequantizeBlockwise8bit<float>(unsigned char *A, float *code, const int order, float *absmax, float *out, int blocksize);

template void dequantizeBlockwise4bit<__nv_bfloat16>(unsigned char *A, float *code, const int order, float *absmax, __nv_bfloat16 *out, int blocksize);
template void dequantizeBlockwise4bit<float>(unsigned char *A, float *code, const int order, float *absmax, float *out, int blocksize);

template void quantizeBlockwise8bitDiagReal<__nv_bfloat16>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize);
template void quantizeBlockwise8bitDiagReal<float>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize);

template void quantizeBlockwise4bitDiagReal<__nv_bfloat16>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize);
template void quantizeBlockwise4bitDiagReal<float>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize);

template void dequantizeBlockwise8bitDiagReal<__nv_bfloat16>(unsigned char *A, float *code, const int order, float *absmax, float *diag, __nv_bfloat16 *out, int blocksize);
template void dequantizeBlockwise8bitDiagReal<float>(unsigned char *A, float *code, const int order, float *absmax, float *diag, float *out, int blocksize);

template void dequantizeBlockwise4bitDiagReal<__nv_bfloat16>(unsigned char *A, float *code, const int order, float *absmax, float *diag, __nv_bfloat16 *out, int blocksize);
template void dequantizeBlockwise4bitDiagReal<float>(unsigned char *A, float *code, const int order, float *absmax, float *diag, float *out, int blocksize);
