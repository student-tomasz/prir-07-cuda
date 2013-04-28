#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "args.h"
#include "exitmf.h"
#include "vec.h"

__global__ void cube(float *d_vc, float *d_v, int v_len);
void cuda_cube(float *d_vc, float *d_v, int v_len);
void seq_cube(float *h_vc, float *h_v, int v_len);
void cuda_vecalloc(float **d_v, int v_len);
void cuda_vecnsend(float *d_v, float *h_v, int v_len);
void cuda_vecnrecv(float *h_v, float *d_v, int v_len);
void cuda_vecfree(float *d_v);


int main(int argc, char *argv[])
{
    float *v, *vc;
    int v_len, vc_len;
    char *v_fpath, *vc_fpath;
    float *h_v, *h_vc, *d_v, *d_vc;
    clock_t start, finish;
    int b_cnt, i;

    args(argc, argv, &v_fpath, &vc_fpath, &b_cnt);
    vecread(v_fpath, &v, &v_len);
    vecread(vc_fpath, &vc, &vc_len);

    /* CUDA */
    vecalloc(&h_v, v_len);
    vecalloc(&h_vc, v_len);
    vecncpy(h_v, v, v_len);
    cuda_vecalloc(&d_v, v_len);
    cuda_vecalloc(&d_vc, v_len);
    start = clock();
    cuda_vecnsend(d_v, h_v, v_len);
    for (i = 0; i < b_cnt; i++) {
        cuda_cube(d_vc, d_v, v_len);
    }
    cuda_vecnrecv(h_vc, d_vc, v_len);
    finish = clock();
    if (veccmp(h_vc, vc, v_len) != 0) exitmf("Those CUDA results suck! Abort.\n");
    printf("CUDA OK in %fs\n", ((float)finish-start)/CLOCKS_PER_SEC);
    cuda_vecfree(d_vc);
    cuda_vecfree(d_v);
    free(h_vc);
    free(h_v);

    /* Sequential */
    vecalloc(&h_v, v_len);
    vecalloc(&h_vc, v_len);
    vecncpy(h_v, v, v_len);
    start = clock();
    for (i = 0; i < b_cnt; i++) {
        seq_cube(h_vc, h_v, v_len);
    }
    finish = clock();
    if (veccmp(h_vc, vc, v_len) != 0) exitmf("Those sequential results suck! Abort.\n");
    printf("SEQ OK in %fs\n", ((float)finish-start)/CLOCKS_PER_SEC);
    free(h_vc);
    free(h_v);

    free(vc);
    free(v);
    return EXIT_SUCCESS;
}

__global__ void cube(float *d_vc, float *d_v, int v_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < v_len) {
        d_vc[i] = d_v[i] * d_v[i] * d_v[i];
    }
}

void cuda_cube(float *d_vc, float *d_v, int v_len)
{
    int threads_per_block = 512;
    int blocks_per_grid = v_len / threads_per_block;
    cube<<<blocks_per_grid, threads_per_block>>>(d_vc, d_v, v_len);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) exitmf("Failed at cube\n");
}

void seq_cube(float *h_vc, float *h_v, int v_len)
{
    int i;
    for (i = 0; i < v_len; i++) {
        h_vc[i] = h_v[i] * h_v[i] * h_v[i];
    }
}

void cuda_vecalloc(float **d_v, int v_len)
{
    cudaError_t rslt = cudaMalloc(d_v, sizeof(**d_v) * v_len);
    if (rslt != cudaSuccess) exitmf("Failed at cudaMalloc\n");
}

void cuda_vecnsend(float *d_v, float *h_v, int v_len)
{
    cudaError_t rslt = cudaMemcpy(d_v, h_v, sizeof(*d_v) * v_len, cudaMemcpyHostToDevice);
    if (rslt != cudaSuccess) exitmf("Failed at cudaMemcpyHostToDevice\n");
}

void cuda_vecnrecv(float *h_v, float *d_v, int v_len)
{
    cudaError_t rslt = cudaMemcpy(h_v, d_v, sizeof(*h_v) * v_len, cudaMemcpyDeviceToHost);
    if (rslt != cudaSuccess) exitmf("Failed at cudaMemcpyDeviceToHost\n");
}

void cuda_vecfree(float *d_v)
{
    cudaError_t rslt = cudaFree(d_v);
    if (rslt != cudaSuccess) exitmf("Failed at cudaFree\n");

    // rslt = cudaDeviceReset();
    // if (rslt != cudaSuccess) exitmf("Failed at cudaDeviceReset\n");
}
