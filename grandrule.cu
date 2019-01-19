#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define STAR -1
struct timeval start, end;

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

void save_result_csv(char *csv_file, char* result, int*data, int rows, int cols_result, int cols_data){
    FILE* file = fopen(csv_file, "w");
    for (int row = 0; row < rows; row++) {
        for (int result_col =0 ; result_col<cols_result;result_col++){
            for(int i=0;i<result[row*cols_result + result_col];i++){
                for(int j=0;j<cols_data;j++){
                    fprintf(file,"%d;",data[row*cols_data+j]);
                }
                fprintf(file,"%d\n",result_col);
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

__host__ __device__
void process_transaction(int tr,int *rules, int rules_count ,int rule_size,int rule_t_size, int *data,int tr_count,int tr_size,int* mask_indexes, int MAX_MASK, char* result,int result_size){
    
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
                       result[tr*result_size + rules[found*rule_t_size + rule_size -1]]+=1;
                    }
                    found++;
                }
            }
        }
    }
    
    
}

void process_on_CPU(int *rules, int rules_count ,int rule_size,int rule_t_size, int *data,int tr_count,int tr_size,int* mask_indexes, int MAX_MASK, char* result, int result_size){
    
    char transactions_file[20] = "transactions_0.csv";
    char out_file[20] = "out_0.csv";

    for(int i =0;i<2;i++){
        transactions_file[13] = '0'+i;
        out_file[4] = '0'+i;
        printf("Loading transactions_%d\n",i);
        load_csv(data,transactions_file, tr_count, tr_size, tr_size);
        
        printf("CPU: start_%d\n",i);
        gettimeofday(&start, NULL);
        
        #pragma omp parallel for
        for (int tr = 0; tr < tr_count; tr++) {
          process_transaction(tr,rules,rules_count,rule_size,rule_t_size,data,tr_count,tr_size,mask_indexes,MAX_MASK,result,result_size);
        }
        
        gettimeofday(&end, NULL);
        printf("CPU: end_%d : %f s\n",i,(end.tv_sec  - start.tv_sec)+ (end.tv_usec - start.tv_usec) / 1.e6);
        
        //save_result_csv(out_file,result,data,tr_count,result_size,tr_size);
    }
}


__global__
void process_batch_kernel(int curr_batch_size,int *rules, int rules_count ,int rule_size,int rule_t_size, int *data,int tr_count,int tr_size,int* mask_indexes, int MAX_MASK, char*result,int result_size){
    
    int tr = blockIdx.x*blockDim.x + threadIdx.x;
    if(tr >= curr_batch_size)return;
    
    process_transaction(tr,rules,rules_count,rule_size,rule_t_size,data,tr_count,tr_size,mask_indexes,MAX_MASK,result,result_size);
    
}

void cudaAssert(int line,cudaError_t err){
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,line);}
}

void process_on_GPU(int *rules, int rules_count ,int rule_size,int rule_t_size, int *data,int tr_count,int tr_size,int* mask_indexes, int MAX_MASK,char *result,int result_size){
    
    int* data_g;
    int* rules_g;
    int* mask_indexes_g;
    char* result_g;
    
    int BLOCK_SIZE =256;
    int BLOCK_DIM = 16;
    int batch_size = BLOCK_SIZE*BLOCK_DIM;
    int batch_count = (tr_count+batch_size-1)/batch_size; //divide and round up;
    
    cudaAssert(__LINE__,cudaMalloc((void **)&rules_g, rules_count*rule_t_size*sizeof(int)));
    cudaAssert(__LINE__,cudaMalloc((void **)&mask_indexes_g, (MAX_MASK+1)*sizeof(int)));
    cudaAssert(__LINE__,cudaMemcpy(rules_g, rules, rules_count*rule_t_size*sizeof(int), cudaMemcpyHostToDevice));
    cudaAssert(__LINE__,cudaMemcpy(mask_indexes_g, mask_indexes, (MAX_MASK+1)*sizeof(int), cudaMemcpyHostToDevice));
    
    cudaAssert(__LINE__,cudaMalloc((void **)&result_g, batch_size*result_size*sizeof(char)));
    cudaAssert(__LINE__,cudaMalloc((void **)&data_g, batch_size*tr_size*sizeof(int)));
    
    char transactions_file[20] = "transactions_0.csv";
    char out_file[20] = "out_0.csv";

    for(int i =0;i<2;i++){
        transactions_file[13] = '0'+i;
        out_file[4] = '0'+i;
        printf("Loading transactions_%d\n",i);
        load_csv(data,transactions_file, tr_count, tr_size, tr_size);
    
        printf("GPU: start_%d\n",i);
        gettimeofday(&start, NULL);
        
        int curr_batch_size = batch_size;
        for (int batch_nr=0;batch_nr<batch_count;batch_nr++){
            
            if(batch_nr==batch_count-1){
                curr_batch_size = tr_count - batch_nr*batch_size;
            }
            cudaAssert(__LINE__,cudaMemcpy(data_g, data+batch_nr*batch_size*tr_size , curr_batch_size*tr_size*sizeof(int), cudaMemcpyHostToDevice));

            process_batch_kernel<<<BLOCK_DIM,BLOCK_SIZE>>>(curr_batch_size,rules_g,rules_count ,rule_size,rule_t_size,data_g,tr_count,tr_size,mask_indexes_g,MAX_MASK,result_g,result_size);
            cudaAssert(__LINE__,cudaThreadSynchronize());
            
            cudaAssert(__LINE__,cudaMemcpy(result+batch_nr*batch_size*result_size, result_g , curr_batch_size*result_size*sizeof(char), cudaMemcpyDeviceToHost));
            
        }

        gettimeofday(&end, NULL);
        printf("GPU: end_%d : %f s\n",i,(end.tv_sec  - start.tv_sec)+ (end.tv_usec - start.tv_usec) / 1.e6);
        //save_result_csv(out_file,result,data,tr_count,result_size,tr_size);
    }
    
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
    int tr_count = 1000000;
    int tr_size = rule_size - 1;
    
    const int MAX_MASK = (1<<tr_size);

	printf("Loading rules\n");
    int *rules = (int*)calloc(rules_count*(rule_t_size),sizeof(int));
    load_csv(rules,rules_file, rules_count, rule_size, rule_t_size);

    	
    int *data = (int*)calloc(tr_count*(tr_size),sizeof(int));
    
    int result_size = 100;
    char *result = (char*)calloc(tr_count*(result_size),sizeof(char));
    
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
    
    process_on_CPU(rules,rules_count ,rule_size,rule_t_size,data,tr_count,tr_size,mask_indexes,MAX_MASK,result,result_size);
    process_on_GPU(rules,rules_count,rule_size,rule_t_size,data,tr_count,tr_size,mask_indexes,MAX_MASK,result,result_size);
	
    return 0;
}
