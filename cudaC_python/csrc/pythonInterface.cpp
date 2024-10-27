#include <ops.cuh>

void quantizeBlockwise8bit_bf16(__nv_bfloat16 *A, float * code, const int order, float *absmax, unsigned char *out, int blocksize){ quantizeBlockwise8bit<__nv_bfloat16>(A, code, order, absmax, out, blocksize); }
void quantizeBlockwise8bit_fp32(float *A, float * code, const int order, float *absmax, unsigned char *out, int blocksize){ quantizeBlockwise8bit<float>(A, code, order, absmax, out, blocksize); }

void quantizeBlockwise4bit_bf16(__nv_bfloat16 *A, float * code, const int order, float *absmax, unsigned char *out, int blocksize){ quantizeBlockwise4bit<__nv_bfloat16>(A, code, order, absmax, out, blocksize); }
void quantizeBlockwise4bit_fp32(float *A, float * code, const int order, float *absmax, unsigned char *out, int blocksize){ quantizeBlockwise4bit<float>(A, code, order, absmax, out, blocksize); }

void dequantizeBlockwise8bit_bf16(unsigned char *A, float * code, const int order, float *absmax, __nv_bfloat16 *out, int blocksize){ dequantizeBlockwise8bit<__nv_bfloat16>(A, code, order, absmax, out, blocksize); }
void dequantizeBlockwise8bit_fp32(unsigned char *A, float * code, const int order, float *absmax, float *out, int blocksize){ dequantizeBlockwise8bit<float>(A, code, order, absmax, out, blocksize); }

void dequantizeBlockwise4bit_bf16(unsigned char *A, float * code, const int order, float *absmax, __nv_bfloat16 *out, int blocksize){ dequantizeBlockwise4bit<__nv_bfloat16>(A, code, order, absmax, out, blocksize); }
void dequantizeBlockwise4bit_fp32(unsigned char *A, float * code, const int order, float *absmax, float *out, int blocksize){ dequantizeBlockwise4bit<float>(A, code, order, absmax, out, blocksize); }

void quantizeBlockwise8bit_diagreal_bf16(__nv_bfloat16 *A, float * code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize){ quantizeBlockwise8bitDiagReal<__nv_bfloat16>(A, code, order, absmax, diag, out, blocksize); }
void quantizeBlockwise8bit_diagreal_fp32(float *A, float * code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize){ quantizeBlockwise8bitDiagReal<float>(A, code, order, absmax, diag, out, blocksize); }

void quantizeBlockwise4bit_diagreal_bf16(__nv_bfloat16 *A, float * code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize){ quantizeBlockwise4bitDiagReal<__nv_bfloat16>(A, code, order, absmax, diag, out, blocksize); }
void quantizeBlockwise4bit_diagreal_fp32(float *A, float * code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize){ quantizeBlockwise4bitDiagReal<float>(A, code, order, absmax, diag, out, blocksize); }

void dequantizeBlockwise8bit_diagreal_bf16(unsigned char *A, float * code, const int order, float *absmax, float *diag, __nv_bfloat16 *out, int blocksize){ dequantizeBlockwise8bitDiagReal<__nv_bfloat16>(A, code, order, absmax, diag, out, blocksize); }
void dequantizeBlockwise8bit_diagreal_fp32(unsigned char *A, float * code, const int order, float *absmax, float *diag, float *out, int blocksize){ dequantizeBlockwise8bitDiagReal<float>(A, code, order, absmax, diag, out, blocksize); }

void dequantizeBlockwise4bit_diagreal_bf16(unsigned char *A, float * code, const int order, float *absmax, float *diag, __nv_bfloat16 *out, int blocksize){ dequantizeBlockwise4bitDiagReal<__nv_bfloat16>(A, code, order, absmax, diag, out, blocksize); }
void dequantizeBlockwise4bit_diagreal_fp32(unsigned char *A, float * code, const int order, float *absmax, float *diag, float *out, int blocksize){ dequantizeBlockwise4bitDiagReal<float>(A, code, order, absmax, diag, out, blocksize); }

extern "C"
{
    void cquantize_blockwise_8bit_bf16(__nv_bfloat16 *A, float * code, const int order, float *absmax, unsigned char *out, int blocksize){ quantizeBlockwise8bit_bf16(A, code, order, absmax, out, blocksize); }
    void cquantize_blockwise_8bit_fp32(float *A, float * code, const int order, float *absmax, unsigned char *out, int blocksize){ quantizeBlockwise8bit_fp32(A, code, order, absmax, out, blocksize); }

    void cquantize_blockwise_4bit_bf16(__nv_bfloat16 *A, float * code, const int order, float *absmax, unsigned char *out, int blocksize){ quantizeBlockwise4bit_bf16(A, code, order, absmax, out, blocksize); }
    void cquantize_blockwise_4bit_fp32(float *A, float * code, const int order, float *absmax, unsigned char *out, int blocksize){ quantizeBlockwise4bit_fp32(A, code, order, absmax, out, blocksize); }

    void cdequantize_blockwise_8bit_bf16(unsigned char *A, float * code, const int order, float *absmax, __nv_bfloat16 *out, int blocksize){ dequantizeBlockwise8bit_bf16(A, code, order, absmax, out, blocksize); }
    void cdequantize_blockwise_8bit_fp32(unsigned char *A, float * code, const int order, float *absmax, float *out, int blocksize){ dequantizeBlockwise8bit_fp32(A, code, order, absmax, out, blocksize); }

    void cdequantize_blockwise_4bit_bf16(unsigned char *A, float * code, const int order, float *absmax, __nv_bfloat16 *out, int blocksize){ dequantizeBlockwise4bit_bf16(A, code, order, absmax, out, blocksize); }
    void cdequantize_blockwise_4bit_fp32(unsigned char *A, float * code, const int order, float *absmax, float *out, int blocksize){ dequantizeBlockwise4bit_fp32(A, code, order, absmax, out, blocksize); }

    void cquantize_blockwise_diagreal_8bit_bf16(__nv_bfloat16 *A, float * code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize){ quantizeBlockwise8bit_diagreal_bf16(A, code, order, absmax, diag, out, blocksize); }
    void cquantize_blockwise_diagreal_8bit_fp32(float *A, float * code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize){ quantizeBlockwise8bit_diagreal_fp32(A, code, order, absmax, diag, out, blocksize); }

    void cquantize_blockwise_diagreal_4bit_bf16(__nv_bfloat16 *A, float * code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize){ quantizeBlockwise4bit_diagreal_bf16(A, code, order, absmax, diag, out, blocksize); }
    void cquantize_blockwise_diagreal_4bit_fp32(float *A, float * code, const int order, float *absmax, float *diag, unsigned char *out, int blocksize){ quantizeBlockwise4bit_diagreal_fp32(A, code, order, absmax, diag, out, blocksize); }

    void cdequantize_blockwise_diagreal_8bit_bf16(unsigned char *A, float * code, const int order, float *absmax, float *diag, __nv_bfloat16 *out, int blocksize){ dequantizeBlockwise8bit_diagreal_bf16(A, code, order, absmax, diag, out, blocksize); }
    void cdequantize_blockwise_diagreal_8bit_fp32(unsigned char *A, float * code, const int order, float *absmax, float *diag, float *out, int blocksize){ dequantizeBlockwise8bit_diagreal_fp32(A, code, order, absmax, diag, out, blocksize); }

    void cdequantize_blockwise_diagreal_4bit_bf16(unsigned char *A, float * code, const int order, float *absmax, float *diag, __nv_bfloat16 *out, int blocksize){ dequantizeBlockwise4bit_diagreal_bf16(A, code, order, absmax, diag, out, blocksize); }
    void cdequantize_blockwise_diagreal_4bit_fp32(unsigned char *A, float * code, const int order, float *absmax, float *diag, float *out, int blocksize){ dequantizeBlockwise4bit_diagreal_fp32(A, code, order, absmax, diag, out, blocksize); }
}
