#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define STAR -1

int** alloc_two_d(int rows, int cols) {
    int **array = calloc(rows, sizeof(int*));
    for (int row = 0; row < rows; row++) {
        array[row] = calloc(cols, sizeof(int));
    }
    return array;
}

int** load_csv(char *csv_file, int rows, int cols, int add_cols){
    int **data = alloc_two_d(rows, cols + add_cols);
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
    int mask_index = 11;
    int mask_diff = (*r1)[mask_index] - (*r2)[mask_index];
    if (mask_diff) {
        return mask_diff;
    }
    return (*r1)[mask_index + 1] - (*r2)[mask_index + 1];
}

int cmphsh (const void * a, const void * b) {
    int r1 = *(int*) a;
    int **r2 = (int**) b;
    return r1 - (*r2)[12];
}

int main(){
    char *rules_file = "rule_2M.csv";
    int rules_count = 2000000;
    int rule_size = 11;
    char *transactions_file = "transactions_tiny.csv";
    int tr_count = 5000;
    int tr_size = rule_size - 1;
    
    const int MAX_MASK = (1<<tr_size);

	struct timeval start, end;
	gettimeofday(&start, NULL);

	printf("Loading rules\n");
    int **rules = load_csv(rules_file, rules_count, rule_size, 2);
	printf("Loading transactions\n");
    int **data = load_csv(transactions_file, tr_count, tr_size, 0);
	
    for (int i = 0; i < rules_count; i++) {
        int mask = 0;
        int hash = 0;
        for (int col = tr_size - 1; col >= 0; col--) {
            if (rules[i][col] == STAR) {
                mask |= 1 << (tr_size - col - 1);
            } else {
                hash += rules[i][col];
            }
        }
        rules[i][rule_size] = mask;
        rules[i][rule_size + 1] = hash;
    }

	printf("Sorting rules\n");
	qsort(rules, rules_count, sizeof(rules[0]), cmpfunc);

    int *mask_indexes = calloc(MAX_MASK + 1,sizeof(int));
    
    int cur_mask = rules[0][rule_size];
    for (int i = 1; i < rules_count; i++) {
        if (cur_mask != rules[i][rule_size]) {
            for (int j = cur_mask + 1; j <= rules[i][rule_size]; j++) {
                mask_indexes[j] = i;
            }
            cur_mask = rules[i][rule_size];
        }
        
        if( i==(rules_count-1)){
            for(int j= cur_mask+1;j<MAX_MASK+1;j++){
                mask_indexes[j]=rules_count;
            }
        }
    }

	printf("Sorted: start\n");
	gettimeofday(&start, NULL);

#pragma omp parallel for
    for (int tr = 0; tr < tr_count; tr++) {
        for (int mask = 0; mask < MAX_MASK ; mask++) {
            int tmp_mask = mask;
            int hash = 0;
            for (int i = tr_size - 1; i >= 0; i--) {
                if (tmp_mask % 2 == 0) {
                    hash += data[tr][i];
                }
                tmp_mask /= 2;
            }
            int index_start = mask_indexes[mask];
            int index_end = mask_indexes[mask + 1];
            if (index_start != index_end) {

                int **res = bsearch(&hash, &rules[index_start], index_end - index_start, sizeof(rules[0]), cmphsh);
                
                if (res) {
                    while (res > rules && (*res)[rule_size + 1] == (*(res - 1))[rule_size + 1]) {
                        res = res - 1;
                    }
                    while (res < rules + rules_count && (*res)[rule_size + 1] == hash) {
                        int ok = 1;
                        for (int i = 0; ok && i < tr_size; i++) {
                            ok == (*res)[i] == STAR || (*res)[i] == data[tr][i];
                        }
                        if (ok) {
                            //printf("%d: %d\n", tr, (*res)[rule_size - 1]);
                        }
                        res++;

                    //printf("%d: %d, previous %d\n", hash, (*res)[rule_size + 1], (*(res - 1))[rule_size + 1]);
                    }
                }
            }
        }
    }
	
	gettimeofday(&end, NULL);
	printf("Sorted: %f\n",(end.tv_sec  - start.tv_sec)+ (end.tv_usec - start.tv_usec) / 1.e6);
	
    return 0;
}
