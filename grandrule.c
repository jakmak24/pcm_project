#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STAR 10000


int** alloc_two_d(int rows, int cols) {
    int **array = calloc(rows, sizeof(int*));
    for (int row = 0; row < rows; row++) {
        array[row] = calloc(cols, sizeof(int));
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

void dummy_search(int** data, int tr_count, int** rules, int rules_count){
	
	#pragma omp parallel for
	for (int tr = 0; tr < tr_count; tr++) {
		for (int row = 0; row < rules_count; row++) {
			int ok = 1;
			for (int col = 0 ; ok && col < 10; col++) {
				
				if (data[tr][col] != rules[row][col] && rules[row][col] != STAR) {
					ok = 0;
				}
				
			}
			if (ok) {
                //printf("%d,%d\n", tr,rules[row][10]);
            }
		}
	}
	
}



void sorted_search(int** data, int tr_count, int** rules, int rules_count){
	
	#pragma omp parallel for
	for (int tr = 0; tr < tr_count; tr++) {
		//printf("%d\n", tr);

		int start_col = 0;
        for (int row = 0; row < rules_count; row++) {
			
			int ok = 1;
			
			while(rules[row][start_col] == STAR){
				start_col++;
			}
			
            for (int col = start_col; ok && col < 10; col++) {
				
                if (data[tr][col] != rules[row][col] && rules[row][col] != STAR) {
                    ok = 0;
                }
            }
            if (ok) {
                //printf("%d,%d\n", tr,rules[row][10]);
            }
			
        }
    }
	
}



int main(){
    char *rules_file = "rule_2M.csv";
    int rules_count = 2000000;
    int tr_count = 20000;
    int rule_size = 11;
    int tr_size = rule_size - 1;
	
	struct timeval start, end;
	gettimeofday(&start, NULL);

	printf("Loading rules\n");
    int **rules = load_csv(rules_file, rules_count, rule_size);
	printf("Loading transactions\n");
    int **data = load_csv("transactions_tiny.csv", tr_count, tr_size);
	
	// gettimeofday(&start, NULL);
	// dummy_search(data,tr_count,rules,rules_count);
	// gettimeofday(&end, NULL);
	// printf("Dummy: %f\n",(end.tv_sec  - start.tv_sec)+ (end.tv_usec - start.tv_usec) / 1.e6);
	
	printf("Sorting rules\n");
	qsort(rules, rules_count, sizeof(rules[0]), cmpfunc);
	
	printf("Sorted: start\n");
	gettimeofday(&start, NULL);
	
	sorted_search(data,tr_count,rules,rules_count);
	
	gettimeofday(&end, NULL);
	printf("Sorted: %f\n",(end.tv_sec  - start.tv_sec)+ (end.tv_usec - start.tv_usec) / 1.e6);
	
    return 0;
}

  
  
