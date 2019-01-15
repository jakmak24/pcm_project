#include <stdio.h>


#define STAR = -1

char [] RULES_FILE = "rule_tiny.csv";
int rule_size = 20000

struct rule {
    short c[10];
    short class;
};


void load_rules(,int rule_size,char[] rules_filename){
    
    FILE* my_file = fopen(rules_filename);
    char buf[bufSize];
    
    while (fgets(buf, sizeof(buf), fp) != NULL){
        buf[strlen(buf) - 1] = '\0'; // eat the newline fgets() stores
        printf("%s\n", buf);
    }
    
  fclose(fp);
}


int main(){
    
    struct rule_list[rule_size];
    load rules(RULES_FILE)
    
    //process rules
    
    //load some transations
    //process transactions
    //save transactions
    
    
    return 0
}

  
  