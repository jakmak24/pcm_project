#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define STAR -1

void load_csv(int*data, char *csv_file, int rows, int cols, int cols_t){
    FILE* file = fopen(csv_file, "r");
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            if(!fscanf(file, "%d;", &data[row*cols_t + col])) {
                fscanf(file, "%*c;");
                data[row*cols_t + col] = STAR;
            }
        }
    }
    fclose(file);
}

int cmpfunc (const void * a, const void * b) {
	const int *r1 = (const int*)a;
	const int *r2 = (const int*)b;
    int mask_index = 11;
    int mask_diff = (*(r1+mask_index)) - (*(r2+mask_index));
    if (mask_diff) {
        return mask_diff;
    }
    return (*(r1+mask_index + 1)) - (*(r2+mask_index+1));
}


int cmphsh (const void * a, const void * b) {
    int r1 = *(int*) a;
    int *r2 = (int*) b;
    return r1 - (*(r2+12));
    
}

void searchCPU(int *rules, int rules_count ,int rule_size,int rule_t_size, int *data,int tr_count,int tr_size,int* mask_indexes, int MAX_MASK){
    
    for (int tr = 0; tr < tr_count; tr++) {
        for (int mask = 0; mask < MAX_MASK ; mask++) {
            int tmp_mask = mask;
            int hash = 0;
            for (int i = 0; i <tr_size; i++) {
                if (tmp_mask % 2 == 0) {
                    hash += data[tr*tr_size + i];
                }
                tmp_mask /= 2;
            }
            int index_start = mask_indexes[mask];
            int index_end = mask_indexes[mask + 1];
            
            if (index_start != index_end) {

                int *res = (int *)bsearch(&hash, rules+(index_start)*rule_t_size , index_end - index_start, sizeof(int)*rule_t_size, cmphsh);
                if (res) {
                    while (res > rules && (*(res +rule_size+1)) == (*(res - rule_t_size +rule_size+1))) {
                        res = res - rule_t_size;
                    }
                    while (res < rules + rules_count*rule_t_size && (*(res + rule_size + 1)) == hash) {
                        int ok = 1;
                        for (int i = 0; ok && i < tr_size; i++) {
                            ok = ((*(res+i)) == STAR || ((*res+i)) == data[tr*tr_size + i]);
                        }
                        if (ok) {
                           printf("%d: %d\n", tr, (*(res + rule_size - 1)));
                        }
                        res+=rule_t_size;
                        printf("%d: %d, previous %d\n", hash, (*(res + rule_size + 1)), (*(res - rule_t_size+ rule_size + 1)));
                    }
                }
            }
        }
    }
}

__device__
int cmphsh_gpu (const int * a, const int * b) {
    int r1 = *(int*) a;
    int *r2 = (int*) b;
    return r1 - (*(r2+12));
}

__device__
int *bsearch_gpu(const int *key, const int *base, size_t num, size_t size,int (*cmp)(const int *key, const int *elt)){
	const int *pivot;
	int result;

	while (num > 0) {
		pivot = base + (num >> 1) * size;
		result = cmp(key, pivot);

		if (result == 0)
			return (int *)pivot;

		if (result > 0) {
			base = pivot + size;
			num--;
		}
		num >>= 1;
	}
	return NULL;
}

__global__
void search_kernel(int *rules, int rules_count ,int rule_size,int rule_t_size, int *data,int tr_count,int tr_size,int* mask_indexes, int MAX_MASK, int*result,int result_size){
    
    int tr = blockIdx.x*blockDim.x + threadIdx.x;
    if(tr >= tr_count)return;
    
    for (int mask = 0; mask < MAX_MASK ; mask++) {
        int tmp_mask = mask;
        int hash = 0;
        for (int i = 0; i <tr_size; i++) {
            if (tmp_mask % 2 == 0) {
                hash += data[tr*tr_size + i];
            }
            tmp_mask /= 2;
        }
        int index_start = mask_indexes[mask];
        int index_end = mask_indexes[mask + 1];
        
        if (index_start != index_end) {

            int *res = (int *)bsearch_gpu(&hash, rules+(index_start)*rule_t_size, index_end - index_start, sizeof(int)*rule_t_size, cmphsh_gpu);
            if (res) {
                while (res > rules && (*(res +rule_size+1)) == (*(res - rule_t_size +rule_size+1))) {
                    res = res - rule_t_size;
                }
                while (res < rules + rules_count*rule_t_size && (*(res + rule_size + 1)) == hash) {
                    int ok = 1;
                    for (int i = 0; ok && i < tr_size; i++) {
                        ok = ((*(res+i)) == STAR || ((*res+i)) == data[tr*tr_size + i]);
                    }
                    if (ok) {
                       result[tr*result_size + (*(res + rule_size - 1)) ]+=1;
                    }
                    res+=rule_t_size;
                }
            }
        }
    }
    
}

void searchGPU(int *rules, int rules_count ,int rule_size,int rule_t_size, int *data,int tr_count,int tr_size,int* mask_indexes, int MAX_MASK){
    
    cudaError_t err;
    int* data_g;
    int* rules_g;
    int* mask_indexes_g;
    int* result_g ;
    
    int* result = (int*)calloc(tr_count*100,sizeof(int));
    
    err = cudaMalloc((void **)&data_g, tr_count*tr_size*sizeof(int));
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaMalloc((void **)&rules_g, rules_count*rule_t_size*sizeof(int));
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaMalloc((void **)&mask_indexes_g, (MAX_MASK+1)*sizeof(int));
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaMalloc((void **)&result_g, tr_count*100*sizeof(int));
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    
    err = cudaMemcpy(data_g, data, tr_count*tr_size*sizeof(int), cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaMemcpy(rules_g, rules, rules_count*rule_t_size*sizeof(int), cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaMemcpy(mask_indexes_g, mask_indexes, tr_count*100*sizeof(int), cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaMemcpy(result_g, result, tr_count*100*sizeof(int), cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

    int BLOCK_SIZE =256;
    int BLOCK_DIM = 4;
    
    search_kernel<<<BLOCK_DIM,BLOCK_SIZE>>>(rules_g,rules_count ,rule_size,rule_t_size,data_g,tr_count,tr_size,mask_indexes_g,MAX_MASK,result_g,100);
   
    err = cudaMemcpy(result, result_g , tr_count*100*sizeof(int), cudaMemcpyDeviceToHost);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
   
    for(int i=0;i<BLOCK_DIM*BLOCK_SIZE;i++){
        for(int j=0;j<100;j++){
            printf("%d",result[i*100+j]);
        }
        printf("\n");
    }
    
    err = cudaFree(data_g);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaFree(rules_g);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaFree(mask_indexes_g);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    err = cudaFree(result_g);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    
   
}


int main(){
    char *rules_file = (char *)"rule_2M.csv";
    int rules_count = 2000000;
    int rule_size = 11;
    int rule_t_size = rule_size+2;
    char *transactions_file = (char *)"transactions_tiny.csv";
    int tr_count = 2000;
    int tr_size = rule_size - 1;
    
    const int MAX_MASK = (1<<tr_size);

	struct timeval start, end;

	printf("Loading rules\n");
    int *rules = (int*)calloc(rules_count*(rule_t_size),sizeof(int));
    load_csv(rules,rules_file, rules_count, rule_size, rule_t_size);
	printf("Loading transactions\n");
    int *data = (int*)calloc(tr_count*(tr_size),sizeof(int));
    load_csv(data,transactions_file, tr_count, tr_size, tr_size);
	
    for (int i = 0; i < rules_count; i++) {
        int mask = 0;
        int hash = 0;
        for (int col = 0; col <tr_size ; col++) {
            if (rules[i*rule_t_size + col] == STAR) {
                mask |= 1 << (col);
            } else {
                hash += rules[i*rule_t_size + col];
            }
        }
        rules[i*rule_t_size + rule_size] = mask;
        rules[i*rule_t_size + rule_size + 1] = hash;
    }
    


	printf("Sorting rules\n");
	qsort(rules, rules_count, sizeof(int)*rule_t_size, cmpfunc);
    


    int *mask_indexes = (int*)calloc(MAX_MASK + 1,sizeof(int));
    
    int cur_mask = rules[0 + rule_size];
    for (int i = 1; i < rules_count; i++) {
        if (cur_mask != rules[i*rule_t_size + rule_size]) {
            for (int j = cur_mask + 1; j <= rules[i*rule_t_size + rule_size]; j++) {
                mask_indexes[j] = i;
            }
            cur_mask = rules[i*rule_t_size + rule_size];
        }
        
        if( i==(rules_count-1)){
            for(int j= cur_mask+1;j<MAX_MASK+1;j++){
                mask_indexes[j]=rules_count;
            }
        }
    }
    
	// printf("Sorted: start\n");
	// gettimeofday(&start, NULL);
    // searchCPU(rules,rules_count ,rule_size,rule_t_size,data,tr_count,tr_size,mask_indexes,MAX_MASK);
	// gettimeofday(&end, NULL);
	// printf("Sorted: %f\n",(end.tv_sec  - start.tv_sec)+ (end.tv_usec - start.tv_usec) / 1.e6);
    
    printf("Sorted: start\n");
	gettimeofday(&start, NULL);
    searchGPU(rules,rules_count ,rule_size,rule_t_size,data,tr_count,tr_size,mask_indexes,MAX_MASK);
	gettimeofday(&end, NULL);
	printf("Sorted: %f\n",(end.tv_sec  - start.tv_sec)+ (end.tv_usec - start.tv_usec) / 1.e6);
	
    return 0;
}
