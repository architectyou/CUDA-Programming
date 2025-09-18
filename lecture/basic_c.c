# include <stdio.h>
# define size 5

void arrayAdd(const int*a, const int*b, int*c, int Size) {
    for (int i=0; i<Size; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int a[size] = {1, 2, 3, 4, 5};
    int b[size] = {10, 20, 30, 40, 50};
    int c[size]; // 이렇게 아무것도 없이 선언해도 되고, 첫번째 Index에 0을 넣어 선언해도 됨. 

    arrayAdd(a, b, c, size);
    printf("Array addition Result : \n");
    printf("{1, 2, 3, 4, 5} + {10, 20, 30, 40, 50} = {%d, %d, %d, %d, %d}\n", c[0], c[1], c[2], c[3], c[4]);
    return 0;
}

