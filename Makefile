all: grandrule.c
	gcc grandrule.c -o prog
	
omp: grandrule.c
	cc grandrule.c -fopenmp -o prog_omp