#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thread>
#include <atomic>


__device__ int lock = 0;
// [CUDA 커널] 락을 사용하여 공유 배열에 안전하게 접근하는 커널
__global__ void lockKernel(int* sharedArray, int size)
{
    // 현재 스레드의 ID 가져오기
    int tid = threadIdx.x;
    
    // 스레드 ID가 배열 크기보다 작은 경우에만 실행
    if (tid < size) {
        printf("T%03d: trying to acquire lock\n", tid);
        
        // 락 획득 시도: atomicCAS로 lock이 0이면 1로 변경
        // 0이 아니면 다른 스레드가 락을 보유 중이므로 대기
        while (atomicCAS(&lock, 0, 1) != 0) {
            // 메모리 펜스로 블록 내 스레드 간 메모리 일관성 보장
            __threadfence_block();
        }
        
        // 락 획득 성공
        printf("T%03d: acquired lock\n", tid);
        
        // 임계 영역: 공유 배열에 스레드 ID 값 쓰기
        printf("T%03d: write %d\n", tid, tid);
        sharedArray[tid] = tid;
        
        // 락 해제: atomicExch로 lock을 0으로 설정
        printf("T%03d: releasing lock\n", tid);
        atomicExch(&lock, 0);
    }
}

int main()
{
    // host-side turn lock: 0이면 GPU0 차례, 1이면 GPU1 차례
    static std::atomic<int> turn{0};

    auto worker = [&](int deviceId, int size, int runs) {
        // 각 GPU 전용 디바이스/호스트 버퍼 준비
        cudaSetDevice(deviceId);
        int* d_sharedArray = nullptr;
        cudaMalloc((void**)&d_sharedArray, size * sizeof(int));
        int* h_array = (int*)malloc(size * sizeof(int));

        for (int r = 0; r < runs; ++r) {
            // 자신의 차례까지 대기
            while (turn.load(std::memory_order_acquire) != deviceId) {
                std::this_thread::yield();
            }

            // 해당 GPU에서 디바이스 락 초기화
            int zero = 0;
            cudaMemcpyToSymbol(lock, &zero, sizeof(int));

            // 커널 실행
            lockKernel<<<1, size>>>(d_sharedArray, size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "[GPU%d][run %d] launch error: %s\n", deviceId, r, cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();

            // 결과 복사 및 일부 출력
            cudaMemcpy(h_array, d_sharedArray, size * sizeof(int), cudaMemcpyDeviceToHost);
            printf("[GPU%d][run %d] first 8 results: ", deviceId, r);
            for (int i = 0; i < 8 && i < size; ++i) {
                printf("%d ", h_array[i]);
            }
            printf("\n");

            // 다음 GPU 차례로 토글
            turn.store(1 - deviceId, std::memory_order_release);
        }

        free(h_array);
        cudaFree(d_sharedArray);
    };

    // 실행 파라미터
    int size = 1024;
    int runsPerGPU = 2; // GPU0 → GPU1 → GPU0 → GPU1 순으로 2회씩 실행

    // 두 GPU용 스레드 실행
    std::thread t0(worker, 0, size, runsPerGPU);
    std::thread t1(worker, 1, size, runsPerGPU);

    t0.join();
    t1.join();

    return 0;
}