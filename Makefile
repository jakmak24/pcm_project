all: grandrule.c
	gcc grandrule.c -o prog
	
omp: grandrule.c
	gcc grandrule.c -fopenmp -o prog_omp
    
gpu: grandrule.cu
	nvcc grandrule.cu -o prog_gpu