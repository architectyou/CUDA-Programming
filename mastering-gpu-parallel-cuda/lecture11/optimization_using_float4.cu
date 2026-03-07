/*naive implementation of MM (matrix multiplication)*/
#include <iostream>
#include <cuda_runtime.h>

#define N 512 // Assume N is a multiple of TILE SIZE
#define TILE_SIZE 16 // Could also use 32 for some GPUs

// gpu kernel이지 일반 kernel이 아니라는것을 설명하는 것
// tiled MM (assuming N % TILE_SIZE ==0)
__global__ void matrixMulTiled(const float*A, const float*B, float*C){
    // shared memory에 변수 또는 배열을 정의하는 것.
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y*TILE_SIZE + threadIdx.y;
    int col = blockIdx.x*TILE_SIZE + threadIdx.x;

    // we need to move the data from global to shared memory
    float sum=0.0;
    int numtiles= N/TILE_SIZE;

    for(int t = 0;t<numtiles;t++){
        sA[threadIdx.y][threadIdx.x] = A[row*N+(t*TILE_SIZE+threadIdx.x)];
        sB[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE + threadIdx.y)*N + col];
        __syncthreads();

        for (int k=0; k<TILE_SIZE; k++){
            sum += sA[threadIdx.y][k]*sB[k][threadIdx.x];
            __syncthreads();
        }
    }
    C[row*N+col]=sum;
}

void cpu_mm_naive(float*A, float*B, float*C){
    for (int i = 0; i<N ; i++) {
        for (int j = 0; j < N ; j++) {
            float sum = 0;
            for (int k = 0; k < N ; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(){
    //allocat memory
    float * h_A = new float[N*N];
    float * h_B = new float[N*N];
    float * h_C = new float[N*N];
    float * h_C_gpu = new float[N*N];

    // initialize input matrices
    for (int i = 0; i < N*N; i++){
        h_A[i] = static_cast<float>(i%10); //arbitrary
        h_B[i] = static_cast<float>((i*2) % 10);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,N*N*sizeof(float));
    cudaMalloc(&d_B,N*N*sizeof(float));
    cudaMalloc(&d_C,N*N*sizeof(float));

    //copy the host vectors to the device 
    cudaMemcpy(d_A,h_A,N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,N*N*sizeof(float),cudaMemcpyHostToDevice);
    
    //block size 할당하기
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);
    cpu_mm_naive(h_A, h_B, h_C);

    matrixMulTiled<<<blocks,threads>>>(d_A,d_B,d_C);

    cudaMemcpy(h_C_gpu,d_C,N*N*sizeof(float),cudaMemcpyDeviceToHost);

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