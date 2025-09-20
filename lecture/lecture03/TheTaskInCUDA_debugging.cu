#include <driver_types.h>
# include <stdio.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include "TheEmployeesSalary.h"
# include <stdlib.h>
 
cudaError_t thehelperfunction(){
    
}

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
    int size = sizeof(TheArrayOfSalaries) / sizeof(TheArrayOfSalaries[0]);
    // device 는 host pointer, host variable에 직접 access 불가능
    // variable 할당
    double* d_Array;
    double* d_NewSalaries;
    double* newSalaries;

    // memory 할당
    cudaMalloc((void**)& d_Array, size * sizeof(double)); // 장치에 8byte memory 할당 + 배열이므로 *100
    cudaMalloc((void**)& d_NewSalaries, size * sizeof(double));

    cudaMemcpy(d_Array, TheArrayOfSalaries, size * sizeof(double), cudaMemcpyHostToDevice);
    // 사용할 thread 수 지정
    int threadsPerBlock=256;
    int blocksPerGrid=(size + threadsPerBlock - 1) / threadsPerBlock;
    
    TaskDoer<<<blocksPerGrid, threadsPerBlock>>>(d_Array, d_NewSalaries, size);
    cudaDeviceSynchronize();
    newSalaries = (double*)malloc(size * sizeof(double));
    cudaMemcpy(newSalaries, d_NewSalaries, size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i ++)
    {
        printf("%f\n", newSalaries[i]);
    }
    free(newSalaries);
    cudaFree(d_Array);
    cudaFree(d_NewSalaries);
    return 0;
}