#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "args.h"
#include "exitmf.h"
#include "vec.h"

__global__ void cube(float *v, int v_len);
void cuda_cube(float *v, int v_len);
void seq_cube(float *v, int v_len);
void cuda_init(float **d_v, float *h_v, int v_len);
void cuda_send(float *d_v, float *h_v, int v_len);
void cuda_recv(float *h_v, float *d_v, int v_len);
void cuda_fin(float *h_v, float *d_v, int v_len);


int main(int argc, char *argv[])
{
    float *v, *vc;
    int v_len, vc_len;
    char *v_fpath, *vc_fpath;
    float *h_v, *d_v;
    clock_t start, finish;

    args(argc, argv, &v_fpath, &vc_fpath);
    vecread(v_fpath, &v, &v_len);
    vecread(vc_fpath, &vc, &vc_len);

    /* CUDA */
    vecncpy(&h_v, v, v_len);
    cuda_init(&d_v, h_v, v_len);
    start = clock();
    cuda_send(d_v, h_v, v_len);
    cuda_cube(d_v, v_len);
    cuda_recv(h_v, d_v, v_len);
    finish = clock();
    if (veccmp(h_v, vc, v_len) != 0) exitmf("Those CUDA results suck! Abort.\n");
    printf("CUDA OK in %fs\n", ((float)finish-start)/CLOCKS_PER_SEC);
    cuda_fin(h_v, d_v, v_len);

    /* Sequential */
    vecncpy(&h_v, v, v_len);
    start = clock();
    seq_cube(h_v, v_len);
    finish = clock();
    if (veccmp(h_v, vc, v_len) != 0) exitmf("Those sequential results suck! Abort.\n");
    printf("SEQ OK in %fs\n", ((float)finish-start)/CLOCKS_PER_SEC);
    free(h_v);

    free(v);
    free(vc);
    return EXIT_SUCCESS;
}

__global__ void cube(float *v, int v_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < v_len) {
        v[i] = v[i] * v[i] * v[i];
    }
}

void cuda_cube(float *d_v, int v_len)
{
    int threads_per_block = 256;
    int blocks_per_grid = v_len / threads_per_block;
    cube<<<blocks_per_grid, threads_per_block>>>(d_v, v_len);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) exitmf("Failed at cube\n");
}

void seq_cube(float *v, int v_len)
{
    int i;
    for (i = 0; i < v_len; i++) {
        v[i] = v[i] * v[i] * v[i];
    }
}

void cuda_init(float **d_v, float *h_v, int v_len)
{
    cudaError_t rslt = cudaMalloc(d_v, sizeof(**d_v) * v_len);
    if (rslt != cudaSuccess) exitmf("Failed at cudaMalloc\n");
}

void cuda_send(float *d_v, float *h_v, int v_len)
{
    cudaError_t rslt = cudaMemcpy(d_v, h_v, sizeof(*d_v) * v_len, cudaMemcpyHostToDevice);
    if (rslt != cudaSuccess) exitmf("Failed at cudaMemcpyHostToDevice\n");
}

void cuda_recv(float *h_v, float *d_v, int v_len)
{
    cudaError_t rslt = cudaMemcpy(h_v, d_v, sizeof(*h_v) * v_len, cudaMemcpyDeviceToHost);
    if (rslt != cudaSuccess) exitmf("Failed at cudaMemcpyDeviceToHost\n");
}

void cuda_fin(float *h_v, float *d_v, int v_len)
{
    free(h_v);

    cudaError_t rslt = cudaFree(d_v);
    if (rslt != cudaSuccess) exitmf("Failed at cudaFree\n");

    // rslt = cudaDeviceReset();
    // if (rslt != cudaSuccess) exitmf("Failed at cudaDeviceReset\n");
}
