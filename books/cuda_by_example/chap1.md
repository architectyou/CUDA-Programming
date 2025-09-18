# Why CUDA? Why Now?

## Parallel Computing
- for 30 years, the important increasing performance was increase the speed at the processor's clock operated
- most desktop processors have clock speed between 1GHz and 4GHz (1000 times faster than original personal computer)
- Increasing CPU clock speed is always been a reliable source for improved performance
- To increase the performance, super computer manufacturers have extracted massive leaps in performance by steadily increasing the number of processors
- super computers -> have ten or hundreds of thousands of process cores working in tandem
- 그래서 점점 cpu 제조 회사들이 multi-core processor unit의 형태로 출시하기 시작했음

## History of GPU Computing and CUDA
### History of GPU Computing
- 1980-1990년대 microsoft windows -> new type of processor에 대한 포문을 열었음.
- 이른 1990년대, 사용자들이 PC를 위한 2D display accelerator를 구매하기 시작
    - To assist in the display and usability of graphical operating systems
- 1980년대, Silicon Graphics 라는 곳에서 3D Graphics가 사용되기 시작
    - Providing toools to create stunning cinematic effects
- 1992년도, Silicon Graphics가 OpenGL 라이브러리 발표
    - `OpenGL` : to be used as standardized, platform-independent method for writing 3D graphics applications
- 1990년대 중반부터 3D graphics에 대한 수요가 급격히 증가하기 시작했음. 이 때 두 가지의 상당히 중요한 개발 개념이 등장
    1. Released of immersive first-person games
        - the popularity of the nascent first-person shooter genre would significantly accelerate the adoption of 3D graphics in consumer computing
    2. Companies such as NVIDIA, ATI Technologies and 3dfx interactive began releasing graphics accelerators that were affordable enough to attract widespread attention
- Transform and lighting -> 3D Graphic rendering process
    - were already integral parts of the OpenGL graphics pipeline,
    - NVIDIA GeForce 256 marked the beginning of a natural progression where increasingly more of the graphics pipeline would be implemented directly on the graphics processor
- from a parallel computing standpoints -> NIVIDA release of the GeForce3 series in 2001 represents arguably the most important breakthrough in GPU technology
    - This standards required that compliant hardware contain both programmable vertex and programmable pixel shading stages
    - 개발자들이 GPU를 이용해 어느 정도 "Computation"을 제어할 수 있게 되었음

- APIs such as OpenGL and DirectX were still the only way to interact with a GPU

## Usage of CUDA C