/*naive implementation of MM (matrix multiplication)*/
#include <iostream>
#include <cuda_runtime.h>

#define N 512 // Assume N is a multiple of TILE SIZE
#define TILE_SIZE 16 // Could also use 32 for some GPUs

__global__ void mmTiledVec4(const float*A, const float*B, float*C, int n){
    // shared memory에 변수 또는 배열을 정의.
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // int row = blockIdx.y*TILE_SIZE + threadIdx.y;
    // int col = blockIdx.x*TILE_SIZE + threadIdx.x; 
    // //thread에서 요소들을 읽어오는 방식 때문에 y가 아니라 x를 변경해야 함.
    // thread가 계산할 위치 정의
    int Row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int baseCol = blockIdx.x * TILE_SIZE + (threadIdx.x * 4);
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // 벡터나 변수를 정의하는 대신 배열 정의
    int numtiles = n / TILE_SIZE;

    for(int t = 0;t<numtiles;t++){
        // sA[threadIdx.y][threadIdx.x] = A[row*N+(t*TILE_SIZE+threadIdx.x)]; -> 하나의 데이터에 대해서만 global memory-> shared memory로 load.
        // global -> shared load (srcA : 행렬 A에서 읽어올 주소)
        int globalAcol = t*TILE_SIZE + (threadIdx.x * 4);
        const float* srcA = A + (Row*n + globalAcol);
        //sA[threadIdx.y][threadIdx.x*4] = srcA;
        // float 4개를 한번에 load
        *reinterpret_cast<float4*>(&sA[threadIdx.y][threadIdx.x*4]) = *reinterpret_cast<const float4*>(srcA);


        // sB[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE + threadIdx.y)*N + col];
        int globalBrow = t*TILE_SIZE + threadIdx.y;
        const float* srcB = B + (globalBrow*n + baseCol);
        
        float* dstB = &sB[threadIdx.y][threadIdx.x*4];
        float4 vecB = *reinterpret_cast<const float4*>(srcB);
        *reinterpret_cast<float4*>(dstB) = vecB;

        __syncthreads();


        // compute partial sum
        for (int k=0; k<TILE_SIZE; k++){
            float aVal = sA[threadIdx.y][k];
            float4 bVal4 = *reinterpret_cast<float4*>(&sB[k][threadIdx.x*4]);
            sum[0] += aVal * bVal4.x;
            sum[1] += aVal * bVal4.y;
            sum[2] += aVal * bVal4.z;
            sum[3] += aVal * bVal4.w;
        }

        __syncthreads();
    }
    // write final partial sum
    if(Row < n){
        float * outC = C + (Row * n + baseCol);
        outC[0] = sum[0];
        outC[1] = sum[1];
        outC[2] = sum[2];
        outC[3] = sum[3];
    }
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
    srand(0);
    for(int i=0; i<N*N; i++){
        h_A[i] = float(rand()%10);
        h_B[i] = float(rand()%10);
        h_C[i] = 0.0f;
        h_C_gpu[i] = 0.0f;
    }

    cpu_mm_naive(h_A, h_B, h_C);

    float *d_A, *d_B, *d_C;
    size_t nBytes = N*N*sizeof(float);
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);
    cudaMalloc((void**)&d_C, nBytes);

    //copy the host vectors to the device 
    cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);
    
    //block size 할당하기
    dim3 block(4, 16);
    dim3 grid(N / TILE_SIZE, N / TILE_SIZE);
    mmTiledVec4<<<grid,block>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu,d_C,nBytes,cudaMemcpyDeviceToHost);

    long long diff = 0;
    for(int i=0; i<N*N; i++){
        diff += (long long)fabs(h_C_gpu[i] - h_C[i]);
    }
    if(diff==0) {
        std::cout << "Results match!\n";
    } else {
        std::cout << "Mismatch, diff=" << diff << "\n";
    }

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_gpu;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}