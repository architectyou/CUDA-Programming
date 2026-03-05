/*naive implementation of MM (matrix multiplication)*/
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

// gpu kernel이지 일반 kernel이 아니라는것을 설명하는 것
__global__ void mmnaive(int*A, int*B, int*C, int N){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N){
        int sum = 0;
        for (int k = 0; k < N; k++){
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void cpu_mm_naive(const int*A, const int*B, int*C, int N){
    for (int i = 0; i<N ; i++) {
        for (int j = 0; j < N ; j++) {
            int sum = 0;
            for (int k = 0; k < N ; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(){
    const int N=512;

    //allocat memory
    int * h_A = new int[N*N];
    int * h_B = new int[N*N];
    int * h_C = new int[N*N];
    int * h_C_gpu = new int[N*N];

    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A,N*N*sizeof(int));
    cudaMalloc((void**)&d_B,N*N*sizeof(int));
    cudaMalloc((void**)&d_C,N*N*sizeof(int));

    //initialize values
    for(int i=0;i<N*N;i++){
        h_A[i]=std::rand() % 10;
        h_B[i]=std::rand() % 10;
        h_C[i]=0;
        h_C_gpu[i]=0;
    }

    cpu_mm_naive(h_A, h_B, h_C, N);

    //copy the host vectors to the device 
    cudaMemcpy(d_A,h_A,N*N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,N*N*sizeof(int),cudaMemcpyHostToDevice);

    //block size 할당하기
    int block_size=16;
    dim3 threadsperblock(block_size, block_size);
    dim3 numBlocks((N+block_size-1)/block_size, (N+block_size-1)/block_size);

    mmnaive<<<numBlocks,threadsperblock>>>(d_A,d_B,d_C,N);

    cudaMemcpy(h_C_gpu,d_C,N*N*sizeof(int),cudaMemcpyDeviceToHost);

    long temp = 0;
    for (int i=0; i < N*N; i++) {
        temp +=  std::abs(h_C_gpu[i] - h_C[i]);
    }
    if (temp==0) {
        std::cout<<"the results are correct.";
    } else {
        std::cout<<"wrong results";
    }

    // 출력 결과 확인하기 & 메모리 해제
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_gpu;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}