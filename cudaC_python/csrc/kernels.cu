#include <kernels.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

template <int CODE_LEN>
__device__ unsigned char dQuantize(float* smem_code, float x)
{
    int pivot = CODE_LEN / 2 - 1;
    int upper_pivot = CODE_LEN - 1;
    int lower_pivot = 0;

    float lower = -1.0f;
    float upper = 1.0f;

    float val = smem_code[pivot];
    for(int i = CODE_LEN / 4; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
        }
        val = smem_code[pivot];
    }

    if(upper_pivot == CODE_LEN - 1)
        upper = smem_code[upper_pivot];
    if(lower_pivot == 0)
        lower = smem_code[lower_pivot];

    if(x > val)
    {
        float midpoint = (upper+val)*0.5f;
        if(x > midpoint)
            return upper_pivot;
        else
            return pivot;
    }
    else
    {
        float midpoint = (lower+val)*0.5f;
        if(x < midpoint)
            return lower_pivot;
        else
            return pivot;
    }
}

template<typename T, int BLOCK_SIZE, int NUM_PER_TH>
__global__ void kQuantizeBlockwise8bit(T *A, float *code, const int order, float *absmax, unsigned char *out)
{
    const int base_idx = blockIdx.x * order;
    const int base_idy = blockIdx.y * BLOCK_SIZE;
    const int base_id = base_idx + base_idy;
    int valid_items = order - base_idy > BLOCK_SIZE ? BLOCK_SIZE : order - base_idy;
    float local_abs_max = -FLT_MAX;

    T vals[NUM_PER_TH];
    unsigned char qvals[NUM_PER_TH];

    typedef cub::BlockLoad<T, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockStore<unsigned char, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
    typedef cub::BlockReduce<float, BLOCK_SIZE/NUM_PER_TH> BlockReduce;

    __shared__ typename LoadT::TempStorage loadt;
    __shared__ typename StoreChar::TempStorage storec;
    __shared__ typename BlockReduce::TempStorage reduce;
    __shared__ float smem_code[256];
    __shared__ float smem_absmax_value[1];

    for(int i = threadIdx.x; i < 256; i+=blockDim.x)
        smem_code[i] = code[i];
   
    __syncthreads();
    LoadT(loadt).Load(&(A[base_id]), vals, valid_items, (T)0.0f);

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
        local_abs_max = fmaxf(local_abs_max, fabsf((float)vals[j]));

    local_abs_max = BlockReduce(reduce).Reduce(local_abs_max, cub::Max());

    if(threadIdx.x == 0)
        smem_absmax_value[0] = local_abs_max;

    __syncthreads();
    if(threadIdx.x == 0)
        absmax[blockIdx.x * gridDim.y + blockIdx.y] = local_abs_max;
    else
        local_abs_max = smem_absmax_value[0];
    __syncwarp();

    if(local_abs_max > 0)
        local_abs_max = 1.0f / local_abs_max;

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
    {
        qvals[j] = dQuantize<256>(smem_code, ((float)vals[j])*local_abs_max);
    }

    __syncthreads();
    StoreChar(storec).Store(&(out[base_id]), qvals, valid_items);
}

template<typename T, int BLOCK_SIZE, int NUM_PER_TH>
__global__ void kQuantizeBlockwise4bit(T *A, float *code, const int order, float *absmax, unsigned char *out)
{
    const int base_idx = blockIdx.x * order;
    const int base_idy = blockIdx.y * BLOCK_SIZE;
    const int base_id = base_idx + base_idy;
    int valid_items = order - base_idy > BLOCK_SIZE ? BLOCK_SIZE : order - base_idy;
    float local_abs_max = -FLT_MAX;

    T vals[NUM_PER_TH];
    unsigned char qvals[NUM_PER_TH/2];

    typedef cub::BlockLoad<T, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockStore<unsigned char, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH/2, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
    typedef cub::BlockReduce<float, BLOCK_SIZE/NUM_PER_TH> BlockReduce;

    __shared__ typename LoadT::TempStorage loadt;
    __shared__ typename StoreChar::TempStorage storec;
    __shared__ typename BlockReduce::TempStorage reduce;
    __shared__ float smem_code[16];
    __shared__ float smem_absmax_value[1];

    for(int i = threadIdx.x; i < 16; i+=blockDim.x)
        smem_code[i] = code[i];
   
    __syncthreads();
    LoadT(loadt).Load(&(A[base_id]), vals, valid_items, (T)0.0f);

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
        local_abs_max = fmaxf(local_abs_max, fabsf((float)vals[j]));

    local_abs_max = BlockReduce(reduce).Reduce(local_abs_max, cub::Max());

    if(threadIdx.x == 0)
        smem_absmax_value[0] = local_abs_max;

    __syncthreads();
    if(threadIdx.x == 0)
        absmax[blockIdx.x * gridDim.y + blockIdx.y] = local_abs_max;
    else
        local_abs_max = smem_absmax_value[0];
    __syncwarp();

    if(local_abs_max > 0)
        local_abs_max = 1.0f / local_abs_max;

    #pragma unroll (NUM_PER_TH / 2)
    for(int j = 0; j < (NUM_PER_TH / 2); j++)
    {
        qvals[j] = dQuantize<16>(smem_code, ((float)vals[2 * j]) * local_abs_max) + 
                  (dQuantize<16>(smem_code, ((float)vals[2 * j + 1]) * local_abs_max) << 4);
    }

    __syncthreads();
    StoreChar(storec).Store(&(out[base_id / 2]), qvals, (valid_items + 1) / 2);
}

template<typename T, int TILE_SIZE, int NUM_PER_TH>
__global__ void kDequantizeBlockwise8bit(unsigned char *A, float *code, const int order, float *absmax, T *out, const int blocksize)
{
    const int gridDim_y = (order + blocksize - 1) / blocksize;
    const int base_idx = blockIdx.x * order;
    const int base_idy = blockIdx.y * TILE_SIZE;
    const int base_id = base_idx + base_idy;
    int absmax_idy = (base_idy + threadIdx.x * NUM_PER_TH) / blocksize;
    if(absmax_idy >=gridDim_y)
        absmax_idy = gridDim_y - 1;

    int valid_items = order - base_idy > TILE_SIZE ? TILE_SIZE : order - base_idy;
    float local_abs_max = -FLT_MAX;

    T vals[NUM_PER_TH];
    unsigned char qvals[NUM_PER_TH];

    typedef cub::BlockLoad<unsigned char, TILE_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
    typedef cub::BlockStore<T, TILE_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ typename LoadChar::TempStorage loadchar;
    __shared__ typename StoreT::TempStorage storet;

    local_abs_max = __ldg(&absmax[blockIdx.x * gridDim_y + absmax_idy]);
   
    __syncthreads();
    LoadChar(loadchar).Load(&(A[base_id]), qvals, valid_items, 127);

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
        vals[j] = __ldg(&code[qvals[j]])*local_abs_max;

    __syncthreads();
    StoreT(storet).Store(&(out[base_id]), vals, valid_items);
}

template<typename T, int TILE_SIZE, int NUM_PER_TH>
__global__ void kDequantizeBlockwise4bit(unsigned char *A, float *code, const int order, float *absmax, T *out, const int blocksize)
{
    const int gridDim_y = (order + blocksize - 1) / blocksize;
    const int base_idx = blockIdx.x * order;
    const int base_idy = blockIdx.y * TILE_SIZE;
    const int base_id = base_idx + base_idy;
    int absmax_idy = (base_idy + threadIdx.x * NUM_PER_TH) / blocksize;
    if(absmax_idy >=gridDim_y)
        absmax_idy = gridDim_y - 1;

    int valid_items = order - base_idy > TILE_SIZE ? TILE_SIZE : order - base_idy;
    float local_abs_max = -FLT_MAX;

    T vals[NUM_PER_TH];
    unsigned char qvals[NUM_PER_TH/2];

    typedef cub::BlockLoad<unsigned char, TILE_SIZE/NUM_PER_TH, NUM_PER_TH/2, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
    typedef cub::BlockStore<T, TILE_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ typename LoadChar::TempStorage loadchar;
    __shared__ typename StoreT::TempStorage storet;

    local_abs_max = __ldg(&absmax[blockIdx.x * gridDim_y + absmax_idy]);

    __syncthreads();
    LoadChar(loadchar).Load(&(A[base_id / 2]), qvals, (valid_items + 1) / 2, 7);

    #pragma unroll (NUM_PER_TH / 2)
    for(int j = 0; j < (NUM_PER_TH / 2); j++)
    {
        vals[2 * j] = __ldg(&code[qvals[j] & 15]) * local_abs_max;
        vals[2 * j + 1] = __ldg(&code[qvals[j] >> 4]) * local_abs_max;
    }

    __syncthreads();
    StoreT(storet).Store(&(out[base_id]), vals, valid_items);
}

template<typename T, int BLOCK_SIZE, int NUM_PER_TH>
__global__ void kQuantizeBlockwise8bitDiagReal(T *A, float *code, const int order, float *absmax, float *diag, unsigned char *out)
{
    const int base_idx = blockIdx.x * order;
    const int base_idy = blockIdx.y * BLOCK_SIZE;
    const int base_id = base_idx + base_idy;
    int valid_items = order - base_idy > BLOCK_SIZE ? BLOCK_SIZE : order - base_idy;
    float local_abs_max = -FLT_MAX;

    T vals[NUM_PER_TH];
    unsigned char qvals[NUM_PER_TH];

    typedef cub::BlockLoad<T, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockStore<unsigned char, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
    typedef cub::BlockReduce<float, BLOCK_SIZE/NUM_PER_TH> BlockReduce;

    __shared__ typename LoadT::TempStorage loadt;
    __shared__ typename StoreChar::TempStorage storec;
    __shared__ typename BlockReduce::TempStorage reduce;
    __shared__ float smem_code[256];
    __shared__ float smem_absmax_value[1];

    for(int i = threadIdx.x; i < 256; i+=blockDim.x)
        smem_code[i] = code[i];
   
    __syncthreads();
    LoadT(loadt).Load(&(A[base_id]), vals, valid_items, (T)0.0f);

    if((BLOCK_SIZE/NUM_PER_TH) * blockIdx.y + threadIdx.x == blockIdx.x / NUM_PER_TH)
    {
        diag[blockIdx.x] = (float)vals[blockIdx.x % NUM_PER_TH];
        vals[blockIdx.x % NUM_PER_TH] = (T)0.0f;
    }

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
        local_abs_max = fmaxf(local_abs_max, fabsf((float)vals[j]));

    local_abs_max = BlockReduce(reduce).Reduce(local_abs_max, cub::Max());

    if(threadIdx.x == 0)
        smem_absmax_value[0] = local_abs_max;

    __syncthreads();
    if(threadIdx.x == 0)
        absmax[blockIdx.x * gridDim.y + blockIdx.y] = local_abs_max;
    else
        local_abs_max = smem_absmax_value[0];
    __syncwarp();

    if(local_abs_max > 0)
        local_abs_max = 1.0f / local_abs_max;

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
    {
        qvals[j] = dQuantize<256>(smem_code, ((float)vals[j])*local_abs_max);
    }

    __syncthreads();
    StoreChar(storec).Store(&(out[base_id]), qvals, valid_items);
}

template<typename T, int BLOCK_SIZE, int NUM_PER_TH>
__global__ void kQuantizeBlockwise4bitDiagReal(T *A, float *code, const int order, float *absmax, float *diag, unsigned char *out)
{
    const int base_idx = blockIdx.x * order;
    const int base_idy = blockIdx.y * BLOCK_SIZE;
    const int base_id = base_idx + base_idy;
    int valid_items = order - base_idy > BLOCK_SIZE ? BLOCK_SIZE : order - base_idy;
    float local_abs_max = -FLT_MAX;

    T vals[NUM_PER_TH];
    unsigned char qvals[NUM_PER_TH/2];

    typedef cub::BlockLoad<T, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    typedef cub::BlockStore<unsigned char, BLOCK_SIZE/NUM_PER_TH, NUM_PER_TH/2, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;
    typedef cub::BlockReduce<float, BLOCK_SIZE/NUM_PER_TH> BlockReduce;

    __shared__ typename LoadT::TempStorage loadt;
    __shared__ typename StoreChar::TempStorage storec;
    __shared__ typename BlockReduce::TempStorage reduce;
    __shared__ float smem_code[16];
    __shared__ float smem_absmax_value[1];

    for(int i = threadIdx.x; i < 16; i+=blockDim.x)
        smem_code[i] = code[i];
   
    __syncthreads();
    LoadT(loadt).Load(&(A[base_id]), vals, valid_items, (T)0.0f);

    if((BLOCK_SIZE/NUM_PER_TH) * blockIdx.y + threadIdx.x == blockIdx.x / NUM_PER_TH)
    {
        diag[blockIdx.x] = (float)vals[blockIdx.x % NUM_PER_TH];
        vals[blockIdx.x % NUM_PER_TH] = (T)0.0f;
    }

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
        local_abs_max = fmaxf(local_abs_max, fabsf((float)vals[j]));

    local_abs_max = BlockReduce(reduce).Reduce(local_abs_max, cub::Max());

    if(threadIdx.x == 0)
        smem_absmax_value[0] = local_abs_max;

    __syncthreads();
    if(threadIdx.x == 0)
        absmax[blockIdx.x * gridDim.y + blockIdx.y] = local_abs_max;
    else
        local_abs_max = smem_absmax_value[0];
    __syncwarp();

    if(local_abs_max > 0)
        local_abs_max = 1.0f / local_abs_max;

    #pragma unroll (NUM_PER_TH / 2)
    for(int j = 0; j < (NUM_PER_TH / 2); j++)
    {
        qvals[j] = dQuantize<16>(smem_code, ((float)vals[2 * j]) * local_abs_max) + 
                  (dQuantize<16>(smem_code, ((float)vals[2 * j + 1]) * local_abs_max) << 4);
    }

    __syncthreads();
    StoreChar(storec).Store(&(out[base_id / 2]), qvals, (valid_items + 1) / 2);
}

template<typename T, int TILE_SIZE, int NUM_PER_TH>
__global__ void kDequantizeBlockwise8bitDiagReal(unsigned char *A, float *code, const int order, float *absmax, float *diag, T *out, const int blocksize)
{
    const int gridDim_y = (order + blocksize - 1) / blocksize;
    const int base_idx = blockIdx.x * order;
    const int base_idy = blockIdx.y * TILE_SIZE;
    const int base_id = base_idx + base_idy;
    int absmax_idy = (base_idy + threadIdx.x * NUM_PER_TH) / blocksize;
    if(absmax_idy >=gridDim_y)
        absmax_idy = gridDim_y - 1;

    int valid_items = order - base_idy > TILE_SIZE ? TILE_SIZE : order - base_idy;
    float local_abs_max = -FLT_MAX;

    T vals[NUM_PER_TH];
    unsigned char qvals[NUM_PER_TH];

    typedef cub::BlockLoad<unsigned char, TILE_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
    typedef cub::BlockStore<T, TILE_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ typename LoadChar::TempStorage loadchar;
    __shared__ typename StoreT::TempStorage storet;

    local_abs_max = __ldg(&absmax[blockIdx.x * gridDim_y + absmax_idy]);
   
    __syncthreads();
    LoadChar(loadchar).Load(&(A[base_id]), qvals, valid_items, 127);

    #pragma unroll NUM_PER_TH
    for(int j = 0; j < NUM_PER_TH; j++)
        vals[j] = __ldg(&code[qvals[j]])*local_abs_max;

    if((TILE_SIZE/NUM_PER_TH) * blockIdx.y + threadIdx.x == blockIdx.x / NUM_PER_TH)
    {
        vals[blockIdx.x % NUM_PER_TH] = (T)diag[blockIdx.x];
    }

    __syncthreads();
    StoreT(storet).Store(&(out[base_id]), vals, valid_items);
}

template<typename T, int TILE_SIZE, int NUM_PER_TH>
__global__ void kDequantizeBlockwise4bitDiagReal(unsigned char *A, float *code, const int order, float *absmax, float *diag, T *out, const int blocksize)
{
    const int gridDim_y = (order + blocksize - 1) / blocksize;
    const int base_idx = blockIdx.x * order;
    const int base_idy = blockIdx.y * TILE_SIZE;
    const int base_id = base_idx + base_idy;
    int absmax_idy = (base_idy + threadIdx.x * NUM_PER_TH) / blocksize;
    if(absmax_idy >=gridDim_y)
        absmax_idy = gridDim_y - 1;

    int valid_items = order - base_idy > TILE_SIZE ? TILE_SIZE : order - base_idy;
    float local_abs_max = -FLT_MAX;

    T vals[NUM_PER_TH];
    unsigned char qvals[NUM_PER_TH/2];

    typedef cub::BlockLoad<unsigned char, TILE_SIZE/NUM_PER_TH, NUM_PER_TH/2, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
    typedef cub::BlockStore<T, TILE_SIZE/NUM_PER_TH, NUM_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

    __shared__ typename LoadChar::TempStorage loadchar;
    __shared__ typename StoreT::TempStorage storet;

    local_abs_max = __ldg(&absmax[blockIdx.x * gridDim_y + absmax_idy]);

    __syncthreads();
    LoadChar(loadchar).Load(&(A[base_id / 2]), qvals, (valid_items + 1) / 2, 7);

    #pragma unroll (NUM_PER_TH / 2)
    for(int j = 0; j < (NUM_PER_TH / 2); j++)
    {
        vals[2 * j] = __ldg(&code[qvals[j] & 15]) * local_abs_max;
        vals[2 * j + 1] = __ldg(&code[qvals[j] >> 4]) * local_abs_max;
    }

    if((TILE_SIZE/NUM_PER_TH) * blockIdx.y + threadIdx.x == blockIdx.x / NUM_PER_TH)
    {
        vals[blockIdx.x % NUM_PER_TH] = (T)diag[blockIdx.x];
    }

    __syncthreads();
    StoreT(storet).Store(&(out[base_id]), vals, valid_items);
}

//==============================================================
//                   TEMPLATE DEFINITIONS
//==============================================================

template __device__ unsigned char dQuantize<256>(float* smem_code, float x);
template __device__ unsigned char dQuantize<16>(float* smem_code, float x);

template __global__ void kQuantizeBlockwise8bit<__nv_bfloat16, 2048, 4>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise8bit<__nv_bfloat16, 1024, 4>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise8bit<__nv_bfloat16, 512, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise8bit<__nv_bfloat16, 256, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise8bit<__nv_bfloat16, 128, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise8bit<__nv_bfloat16, 64, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise8bit<float, 2048, 4>(float *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise8bit<float, 1024, 4>(float *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise8bit<float, 512, 2>(float *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise8bit<float, 256, 2>(float *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise8bit<float, 128, 2>(float *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise8bit<float, 64, 2>(float *A, float *code, const int order, float *absmax, unsigned char *out);

template __global__ void kQuantizeBlockwise4bit<__nv_bfloat16, 2048, 4>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise4bit<__nv_bfloat16, 1024, 4>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise4bit<__nv_bfloat16, 512, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise4bit<__nv_bfloat16, 256, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise4bit<__nv_bfloat16, 128, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise4bit<__nv_bfloat16, 64, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise4bit<float, 2048, 4>(float *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise4bit<float, 1024, 4>(float *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise4bit<float, 512, 2>(float *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise4bit<float, 256, 2>(float *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise4bit<float, 128, 2>(float *A, float *code, const int order, float *absmax, unsigned char *out);
template __global__ void kQuantizeBlockwise4bit<float, 64, 2>(float *A, float *code, const int order, float *absmax, unsigned char *out);

template __global__ void kDequantizeBlockwise8bit<__nv_bfloat16, 512, 8>(unsigned char *A, float *code, const int order, float *absmax, __nv_bfloat16 *out, const int blocksize);
template __global__ void kDequantizeBlockwise8bit<float, 512, 8>(unsigned char *A, float *code, const int order, float *absmax, float *out, const int blocksize);

template __global__ void kDequantizeBlockwise4bit<__nv_bfloat16, 512, 8>(unsigned char *A, float *code, const int order, float *absmax, __nv_bfloat16 *out, const int blocksize);
template __global__ void kDequantizeBlockwise4bit<float, 512, 8>(unsigned char *A, float *code, const int order, float *absmax, float *out, const int blocksize);

template __global__ void kQuantizeBlockwise8bitDiagReal<__nv_bfloat16, 2048, 4>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise8bitDiagReal<__nv_bfloat16, 1024, 4>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise8bitDiagReal<__nv_bfloat16, 512, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise8bitDiagReal<__nv_bfloat16, 256, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise8bitDiagReal<__nv_bfloat16, 128, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise8bitDiagReal<__nv_bfloat16, 64, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise8bitDiagReal<float, 2048, 4>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise8bitDiagReal<float, 1024, 4>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise8bitDiagReal<float, 512, 2>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise8bitDiagReal<float, 256, 2>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise8bitDiagReal<float, 128, 2>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise8bitDiagReal<float, 64, 2>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);

template __global__ void kQuantizeBlockwise4bitDiagReal<__nv_bfloat16, 2048, 4>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise4bitDiagReal<__nv_bfloat16, 1024, 4>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise4bitDiagReal<__nv_bfloat16, 512, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise4bitDiagReal<__nv_bfloat16, 256, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise4bitDiagReal<__nv_bfloat16, 128, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise4bitDiagReal<__nv_bfloat16, 64, 2>(__nv_bfloat16 *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise4bitDiagReal<float, 2048, 4>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise4bitDiagReal<float, 1024, 4>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise4bitDiagReal<float, 512, 2>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise4bitDiagReal<float, 256, 2>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise4bitDiagReal<float, 128, 2>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);
template __global__ void kQuantizeBlockwise4bitDiagReal<float, 64, 2>(float *A, float *code, const int order, float *absmax, float *diag, unsigned char *out);

template __global__ void kDequantizeBlockwise8bitDiagReal<__nv_bfloat16, 512, 8>(unsigned char *A, float *code, const int order, float *absmax, float *diag, __nv_bfloat16 *out, const int blocksize);
template __global__ void kDequantizeBlockwise8bitDiagReal<float, 512, 8>(unsigned char *A, float *code, const int order, float *absmax, float *diag, float *out, const int blocksize);

template __global__ void kDequantizeBlockwise4bitDiagReal<__nv_bfloat16, 512, 8>(unsigned char *A, float *code, const int order, float *absmax, float *diag, __nv_bfloat16 *out, const int blocksize);
template __global__ void kDequantizeBlockwise4bitDiagReal<float, 512, 8>(unsigned char *A, float *code, const int order, float *absmax, float *diag, float *out, const int blocksize);
