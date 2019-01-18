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

__host__ __device__ 
int bsearch(int key, int *rules,int rule_t_size,int start,int end){

    int pivot,result;

	while (end > start) {
		pivot =  start + ((end-start)>>1);
		result = key - rules[pivot*rule_t_size + 12];

		if (result == 0)
			return pivot;

		if (result > 0) {
			start = pivot + 1;
		}else{
            end = pivot;
        }
	}
	return -1;
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

                int found = bsearch(hash, rules, rule_t_size, index_start,index_end);
                if (found != -1) {
                    
                    while (found > 0 && (rules[found*rule_t_size + rule_size+1] == rules[(found-1)*rule_t_size + rule_size+1])) {
                        found--;
                    }
                    while (found < rules_count && rules[found*rule_t_size +rule_size+1] == hash) {
                        int ok = 1;
                        for (int i = 0; ok && i < tr_size; i++) {
                            ok = (rules[found*rule_t_size +i] == STAR || (rules[found*rule_t_size +i] == data[tr*tr_size + i]));
                        }
                        if (ok) {
                           //printf("%d,%d\n",tr,rules[found*rule_t_size +rule_size -1]);
                        }
                        found++;
                    }
                }
            }
        }
    }
}


__global__
void search_kernel(int curr_batch_size,int *rules, int rules_count ,int rule_size,int rule_t_size, int *data,int tr_count,int tr_size,int* mask_indexes, int MAX_MASK, int*result,int result_size){
    
    int tr = blockIdx.x*blockDim.x + threadIdx.x;
    if(tr >= curr_batch_size)return;
    
    for (int i=tr*result_size;i<(tr+1)*result_size;i++){
        result[i]=0;
    }
    
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

            int found = bsearch(hash, rules, rule_t_size, index_start,index_end);
            if (found != -1) {
                
                while (found > 0 && (rules[found*rule_t_size + rule_size+1] == rules[(found-1)*rule_t_size + rule_size+1])) {
                    found--;
                }
                while (found < rules_count && rules[found*rule_t_size +rule_size+1] == hash) {
                    int ok = 1;
                    for (int i = 0; ok && i < tr_size; i++) {
                        ok = (rules[found*rule_t_size +i] == STAR || (rules[found*rule_t_size +i] == data[tr*tr_size + i]));
                    }
                    if (ok) {
                       result[tr*result_size + rules[found*rule_t_size +rule_size -1] ]+=1;
                    }
                    found++;
                }
            }
        }
    }
    
}

void cudaAssert(int line,cudaError_t err){
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,line);}
}

void searchGPU(int *rules, int rules_count ,int rule_size,int rule_t_size, int *data,int tr_count,int tr_size,int* mask_indexes, int MAX_MASK){
    
    int* data_g;
    int* rules_g;
    int* mask_indexes_g;
    int* result_g;
    
    int BLOCK_SIZE =256;
    int BLOCK_DIM = 16;
    int batch_size = BLOCK_SIZE*BLOCK_DIM;
    int curr_batch_size = batch_size;
    int batch_count = (tr_count+batch_size-1)/batch_size; //divide and round up;
    
    cudaAssert(__LINE__,cudaMalloc((void **)&rules_g, rules_count*rule_t_size*sizeof(int)));
    cudaAssert(__LINE__,cudaMalloc((void **)&mask_indexes_g, (MAX_MASK+1)*sizeof(int)));
    cudaAssert(__LINE__,cudaMemcpy(rules_g, rules, rules_count*rule_t_size*sizeof(int), cudaMemcpyHostToDevice));
    cudaAssert(__LINE__,cudaMemcpy(mask_indexes_g, mask_indexes, (MAX_MASK+1)*sizeof(int), cudaMemcpyHostToDevice));
    
    int* result = (int*)calloc(batch_size*100,sizeof(int));
    cudaAssert(__LINE__,cudaMalloc((void **)&result_g, batch_size*100*sizeof(int)));
    cudaAssert(__LINE__,cudaMalloc((void **)&data_g, batch_size*tr_size*sizeof(int)));
    
    for (int batch_nr=0;batch_nr<batch_count;batch_nr++){
        
        if(batch_nr==batch_count-1){
            curr_batch_size = tr_count - batch_nr*batch_size;
        }
        cudaAssert(__LINE__,cudaMemcpy(data_g, data+batch_nr*batch_size*tr_size , curr_batch_size*tr_size*sizeof(int), cudaMemcpyHostToDevice));

        search_kernel<<<BLOCK_DIM,BLOCK_SIZE>>>(curr_batch_size,rules_g,rules_count ,rule_size,rule_t_size,data_g,tr_count,tr_size,mask_indexes_g,MAX_MASK,result_g,100);
        cudaAssert(__LINE__,cudaThreadSynchronize());
        
        cudaAssert(__LINE__,cudaMemcpy(result, result_g , curr_batch_size*100*sizeof(int), cudaMemcpyDeviceToHost));
        
        // for(int j=0;j<curr_batch_size;j++){
            // printf("%d:",batch_nr*batch_size+j);
            // for(int k=0;k<100;k++){
                // printf("%d",result[j*100+k]);
            // }
            // printf("\n");
        // }
        
    }
    
    free(result);
    cudaAssert(__LINE__,cudaFree(data_g));
    cudaAssert(__LINE__,cudaFree(rules_g));
    cudaAssert(__LINE__,cudaFree(mask_indexes_g));
    cudaAssert(__LINE__,cudaFree(result_g));

    
}


int main(){
    char *rules_file = (char *)"rule_2M.csv";
    int rules_count = 2000000;
    int rule_size = 11;
    int rule_t_size = rule_size+2;
    char *transactions_file = (char *)"transactions_0.csv";
    int tr_count = 1000000;
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
