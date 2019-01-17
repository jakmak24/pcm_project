#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define STAR 10000


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

int main(){
    char *rules_file = "tiny.csv";
    int rules_count = 15;
    int tr_count = 15;
    int rule_size = 4;
    int tr_size = rule_size - 1;
	
	struct timeval start, end;
	gettimeofday(&start, NULL);

	printf("Loading rules\n");
    int **rules = load_csv(rules_file, rules_count, rule_size, 2);
	printf("Loading transactions\n");
    int **data = load_csv("tiny_tr.csv", tr_count, tr_size, 0);
	
    for (int i = 0; i < rules_count; i++) {
        int mask = 0;
        int hash = 0;
        for (int col = tr_size - 1; col >= 0; col--) {
            if (rules[i][col] == STAR) {
                mask |= 1 << col;
            } else {
                hash += rules[i][col];
            }
        }
        rules[i][rule_size] = mask;
        rules[i][rule_size + 1] = hash;
    }

    for (int i = 0; i < rules_count; i++) {
        for (int j = 0; j < rule_size + 2; j++) {
            printf("%5.1d ", rules[i][j]);
        }
        printf("\n");
    }

	printf("Sorting rules\n");
	qsort(rules, rules_count, sizeof(rules[0]), cmpfunc);
	
	printf("Sorted: start\n");
	gettimeofday(&start, NULL);
	
    //search
	
	gettimeofday(&end, NULL);
	printf("Sorted: %f\n",(end.tv_sec  - start.tv_sec)+ (end.tv_usec - start.tv_usec) / 1.e6);
	
    return 0;
}

  
  
