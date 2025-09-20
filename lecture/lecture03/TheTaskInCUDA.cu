// [헤더] CUDA 런타임과 표준 라이브러리, 예제 데이터 헤더 포함
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
# include <stdio.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include "TheEmployeesSalary.h"
# include <stdlib.h>


// [선언] 호스트 보조 함수: H2D 복사 → 커널 실행 → D2H 복사까지 수행
cudaError_t thehelperfunction(const double* tarrary, double* tnewSalaries, int tsize, int TPB, int BPG);
 

// [디바이스 커널] 각 원소에 15% 인상 + 5000 적용
// global kernel function -> which can only access host function
__global__ void TaskDoer(const double* array, double* newSalaries, int size)
{
    // 병렬화하려면 우선 thread index를 가져와야 함.
    // int ID = blockIdx.x * blockDim.x +  threadIdx.x;
    // newSalaries[ID] = array[ID] + (array[ID] * 15 / 100) + 5000;

    // thread index가 SIZE를 초기화하면? ->
    int ID = blockIdx.x * blockDim.x +  threadIdx.x;
    if (ID < size){
        newSalaries[ID] = array[ID] * 1.15 + 5000.0;
    }

}

int main()
{
    // [입력 크기] 급여 배열의 원소 개수 계산
    int size = sizeof(TheArrayOfSalaries) / sizeof(TheArrayOfSalaries[0]);

    // device 는 host pointer, host variable에 직접 access 불가능
    // variable 할당
    // (주의) 디바이스 포인터는 helper에서 할당/해제합니다. 여기서는 호스트 출력 버퍼만 준비합니다.
    double* newSalaries = (double*)malloc(size * sizeof(double));
    if (newSalaries == NULL) {
        fprintf(stderr, "malloc failed!\n");
        return 1;
    }

    // 사용할 thread 수 지정
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // [실행] H2D → 커널 → D2H 전체 흐름을 helper가 수행
    cudaError_t cudaStatus = thehelperfunction(TheArrayOfSalaries, newSalaries, size, threadsPerBlock, blocksPerGrid);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "The helper function failed!\n");
        free(newSalaries);
        return 1;
    }

    // [검증 출력] 결과 일부/전체 출력
    for (int i = 0; i < size; i ++)
    {
        printf("%f\n", newSalaries[i]);
    }

    free(newSalaries);
    return 0;
}

// [호스트 보조 함수 정의] 디바이스 메모리 관리, 커널 실행, 에러 확인 및 복사
cudaError_t thehelperfunction(const double* tarrary, double* tnewSalaries, int tsize, int TPB, int BPG)
{
    double * d_Array = 0;
    double * d_NewSalaries = 0;

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Set device function failed! Is a GPU Present in your machine?");
        goto AnError;
    }

    cudaStatus = cudaMalloc((void**)& d_Array, tsize*sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto AnError;
    }

    cudaStatus = cudaMalloc((void**)& d_NewSalaries, tsize*sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto AnError;
    }

    cudaStatus = cudaMemcpy(d_Array, tarrary, tsize*sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto AnError;
    }

    // [커널 실행]
    TaskDoer<<<BPG, TPB>>>(d_Array, d_NewSalaries, tsize);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError failed!\n");
        goto AnError;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
        goto AnError;
    }

    // [결과 복사]
    cudaStatus = cudaMemcpy(tnewSalaries, d_NewSalaries, tsize*sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed!");
        goto AnError;
    }

AnError:
    cudaFree(d_Array);
    cudaFree(d_NewSalaries);
    return cudaStatus;
}