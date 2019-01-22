#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#define STAR 100001

int** alloc_two_d(int rows, int cols) {
    int **array = calloc(rows, sizeof(int*));
    for (int row = 0; row < rows; row++) {
        array[row] = calloc(cols, sizeof(int));
    }
    return array;
}

void free_two_d(int*** array, int rows)
{
    int **a = *array;
    for (int row = 0; row < rows; row++) {
        free(a[row]);
    }
    free(a);
    *array = NULL;
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
    const int *r1 = *(const int**)a;
    const int *r2 = *(const int**)b;
    //printf("lol %d %d\n", r1[0], r2[0]);

    int i = 0;
    int cmp = 0;

    while(i<10){
        cmp = r1[i]-r2[i];
        if ( cmp != 0){
            return cmp;
        }else{
            i++;
        }
    }
    return 0;
}

int main(){
    char *rules_file = "rule_2M.csv";
    int rules_count = 2000000;
    int tr_count = 100;
    int rule_size = 11;
    int tr_size = rule_size - 1;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int **rules_tmp = load_csv(rules_file, rules_count, rule_size);
    int **data = load_csv("transactions_0.csv", tr_count, tr_size);

    gettimeofday(&start, NULL);

    qsort(rules_tmp, rules_count, sizeof(rules_tmp[0]), cmpfunc);

    int **rules = alloc_two_d(rules_count, rule_size);

    for (int i = 0; i < rules_count; i++) {
        memcpy(rules[i], rules_tmp[i], rule_size * sizeof(int));
    }

    free_two_d(&rules_tmp, rules_count);

    gettimeofday(&end, NULL);
    printf("Sorted, time for preprocessing: %f\n",(end.tv_sec  - start.tv_sec)+ (end.tv_usec - start.tv_usec) / 1.e6);

    gettimeofday(&start, NULL);

#pragma omp parallel for
    for (int tr = 0; tr < tr_count; tr++) {
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
            }
        }
    }

    gettimeofday(&end, NULL);
    printf("Sorted, time for %d transactions: %f\n", tr_count, (end.tv_sec  - start.tv_sec)+ (end.tv_usec - start.tv_usec) / 1.e6);

    return 0;
}
