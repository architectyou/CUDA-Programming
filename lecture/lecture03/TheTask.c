#include <stdio.h>
#include "TheEmployeesSalary.h" // localheader file을 사용하기 위해서는 "" 로 사용해야 함.

void TaskDoer(const double TheSalaries[], double TheSumArray[], int SIZE)
{
    for (int i = 0; i <SIZE; i++)
    {
        TheSumArray[i] = TheSalaries[i] + (TheSalaries[i] * 15/100) + 5000; 
    }
}

int main()
{
    int size = sizeof(TheArrayOfSalaries) / sizeof(TheArrayOfSalaries[0]); // sizeof function returns the size of the array 
    double TheNewSalaries[size];
    TaskDoer(TheArrayOfSalaries, TheNewSalaries, size);
    for (int i = 0; i < size; i++){
        printf("%f\n", TheNewSalaries[i]);
    }
    return 0;
}