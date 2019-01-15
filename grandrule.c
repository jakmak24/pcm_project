#include <stdio.h>
#include <stdlib.h>

#define STAR -1

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


int main(){
    char *rules_file = "rule_tiny.csv";
    int rules_count = 20000;
    int tr_count = 20000;
    int rule_size = 11;
    int tr_size = rule_size - 1;

    int **rules = load_csv(rules_file, rules_count, rule_size);
    int **data = load_csv("transactions_tiny.csv", tr_count, tr_size);

    for (int tr = 0; tr < tr_count; tr++) {
        for (int row = 0; row < rules_count; row++) {
            int ok = 1;
            for (int col = 0; ok && col < 10; col++) {
                if (data[tr][col] != rules[row][col] && rules[row][col] != -1) {
                    ok = 0;
                }
            }
            if (ok) {
                printf("%d, ", rules[row][rule_size - 1]);
            }
        }
        printf("\n");
    }
    return 0;
}

  
  
