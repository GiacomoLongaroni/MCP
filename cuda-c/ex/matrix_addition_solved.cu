#include <stdio.h>

#define ROWS 4000  // Number of rows in the matrices
#define COLS 6000  // Number of columns in the matrices

__global__ void matrixAdd(float* A, float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Size in bytes for the ROWS x COLS matrix
    int size = ROWS * COLS * sizeof(float);  

    // Host memory allocation
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < ROWS * COLS; i++) {
        h_A[i] = 1.0 + (float)rand()/RAND_MAX;
        h_B[i] = 2.0 + (float)rand()/RAND_MAX;
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);  // 16x16 threads per block
    dim3 numBlocks((COLS + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, ROWS, COLS);

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print part of the result matrix C for verification
    printf("Result matrix C[0] (cpu): %f\n", h_A[0]+h_B[0]);
    printf("Result matrix C[0] (gpu): %f\n\n", h_C[0]);

    printf("Result matrix C[-1] (cpu): %f\n", h_A[ROWS*COLS-1]+h_B[ROWS*COLS-1]);
    printf("Result matrix C[-1] (gpu): %f\n", h_C[ROWS*COLS-1]);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
