#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define STAR 10000

char *RULES_FILE = (char *)"rule_2M.csv";
const int RULES_COUNT = 2000000;
char *TR_FILE = (char *)"transactions_tiny.csv";
const int TR_COUNT = 20000;

const int RULE_SIZE = 11;
const int TR_SIZE = RULE_SIZE - 1;
int rules_f[RULES_COUNT*RULE_SIZE];
int data_f[TR_COUNT*TR_SIZE];


int** alloc_two_d(int rows, int cols) {
    int **array = (int **)calloc(rows, sizeof(int*));
    for (int row = 0; row < rows; row++) {
        array[row] = (int *)calloc(cols, sizeof(int));
    }
    return array;
}

int** load_csv(char *csv_file, int rows, int cols){
    int **data = alloc_two_d(rows, cols);
    FILE* file = fopen(csv_file, "r");
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            if(!fscanf(file, "%d;", &data[row][col])) {
                fscanf(file, "%*c;");
                data[row][col] = STAR;
            }
        }
    }
    fclose(file);
    return data;
}

int cmpfunc (const void * a, const void * b) {
	const int **r1 = (const int**)a;
	const int **r2 = (const int**)b;

	int i = 0;
	int cmp = 0;
	
	while(i<10){
		cmp = (*r1)[i]-(*r2)[i];
		if ( cmp != 0){
			return cmp;
		}else{
			i++;
		}
	}
	return 0;
}

__global__
void gpu_kernel(int* data, int tr_count, int tr_size, int* rules, int rules_count,int rule_size, int* result, int result_size){
	
	int tr = blockIdx.x*blockDim.x + threadIdx.x;
    if(tr >= tr_count)return;
    
    int start_col = 0;
    for (int row = 0; row < rules_count; row++) {
        
        int ok = 1;
        
        while(rules[row*rule_size + start_col] == STAR){
            start_col++;
        }
        
        for (int col = start_col; ok && col < tr_size; col++) {
            
            if (data[tr*tr_size + col] != rules[row*rule_size+col] && rules[row*rule_size+col] != STAR) {
                ok = 0;
            }
        }
        if (ok) {
            result[tr*result_size+ rules[row*rule_size+rule_size-1]]+=1;
        }
    }
}

void gpu_search(int* data_f, int tr_count,int tr_size, int* rules_f, int rules_count,int rule_size){
	
    int* data_g;
    int* rules_g;
    int* result_g;
    
    int result_f [tr_count*100];
    for(int i=0;i<tr_count;i++){
        for(int j=0;j<100;j++){
            result_f[i*100+j]=0;
        }
    }
    cudaError_t err;
    
    err = cudaMalloc((void **)&data_g, tr_count*tr_size*sizeof(int));
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaMalloc((void **)&rules_g, rules_count*rule_size*sizeof(int));
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaMalloc((void **)&result_g, tr_count*100*sizeof(int));
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    
    err = cudaMemcpy(data_g, data_f, tr_count*tr_size*sizeof(int), cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaMemcpy(rules_g, rules_f, rules_count*rule_size*sizeof(int), cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaMemcpy(result_g, result_f, tr_count*100*sizeof(int), cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    
    gpu_kernel<<<5,256>>>(data_g, tr_count, tr_size, rules_g, rules_count, rule_size, result_g, 100);
    
    err = cudaMemcpy(result_f,result_g , tr_count*100*sizeof(int), cudaMemcpyDeviceToHost);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    
    // for(int i=0;i<2000;i++){
        // for(int j=0;j<100;j++){
            // printf("%d,",result_f[i*100+j]);
        // }
        // printf("\n");
    // }
    
    err = cudaFree(data_g);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaFree(rules_g);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaFree(result_g);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
     
}




int main(){

    struct timeval start, end;
	gettimeofday(&start, NULL);

	printf("Loading rules\n");
    int **rules = load_csv(RULES_FILE, RULES_COUNT, RULE_SIZE);
	printf("Loading transactions\n");
    int **data = load_csv(TR_FILE, TR_COUNT, TR_SIZE);
	
	printf("Sorting rules\n");
	qsort(rules, RULES_COUNT, sizeof(rules[0]), cmpfunc);
	
    
    for(int i=0;i<RULES_COUNT;i++){
        for(int j=0;j<RULE_SIZE;j++){
            rules_f[i*RULE_SIZE + j]=rules[i][j];
        }
    }
    for(int i=0;i<TR_COUNT;i++){
        for(int j=0;j<TR_SIZE;j++){
            data_f[i*TR_SIZE + j]=data[i][j];
        }
    }
    
    printf("GPU: start\n");
	gettimeofday(&start, NULL);
	gpu_search(data_f,TR_COUNT,TR_SIZE,rules_f,RULES_COUNT,RULE_SIZE);
	gettimeofday(&end, NULL);
	printf("GPU: %f\n",(end.tv_sec  - start.tv_sec)+ (end.tv_usec - start.tv_usec) / 1.e6);
	
    return 0;
}

  
  
