#include <iostream>

using namespace std;

#define array_number 64

int matrix[array_number][array_number];

int **double_array(size_t n) {
    int **result = new int*[8];
    for (int i = 0; i < n; ++i) {
            result[i] = matrix[i];
            for (int j = 0; j < n; ++j){
                result[i][j] = j;
            }
    }

    return result;
}


char a[1];

int main() {
    // cout<<"A magic print! If you comment this, the program will break."<<endl;
    int **result = double_array(array_number);
    // cout<<&a<<endl;
    setvbuf(stdout, a, _IOFBF, 0);
    cout<<&a<<endl;
    for (int i = 0; i < array_number; ++i) {
        cout<<"print address of result[i] "<<&result[i][0]<<endl;
    }
    free(result);
}