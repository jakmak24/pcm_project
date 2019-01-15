#include <stdio.h>
#include <stdlib.h>

#define STAR -1

char *rules_file = "rule_tiny.csv";
int rules_count = 20000;
int rule_size = 11;

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
    int **rules = load_csv(rules_file, rules_count, rule_size);

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < rule_size; j++) {
            printf("%5.d ", rules[i][j]);
        }
        printf("\n");
    }
    return 0;
}

  
  
