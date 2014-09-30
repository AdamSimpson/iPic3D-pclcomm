#include <iostream>
#include "assert.h"
#include "Alloc.h"

void test_1D()
{
    size_t length = 100;

    // Allocate arrays
    char   *a1 = newArray1<char>(length); 
    float  *a2 = newArray1<float>(length);
    double *a3 = newArray1<double>(length);

    // Fill with data on host
    for(int i=0; i<length; i++) {
        a1[i] = 'h';
        a2[i] = 0x80000000;
        a3[i] = 0x1.921fb5p+1;
    }    

    // Fill with data on device
    #pragma acc parallel loop present(a1,a2,a3)
    for(int i=0; i<length; i++) {
        a1[i] = 'd';
        a2[i] = 0X436A508C;
        a3[i] = 0x1.f58000p+10;
    }

    // Assert host data
    for(int i=0; i<length; i++) {
        assert(a1[i] == 'h');
        assert(a2[i] == 0x80000000);
        assert(a3[i] == 0x1.921fb5p+1);
    }

    // Copy GPU data to host and assert
    #pragma acc update host(a1[0:length], a2[0:length], a3[0:length])
    for(int i=0; i<length; i++) {
        assert(a1[i] == 'd');
        assert(a2[i] == 0X436A508C);
        assert(a3[i] == 0x1.f58000p+10);
    }
    
    std::cout<<"Passed test_1D"<<std::endl;
}

int main(int argc, char *argv[])
{
    test_1D();

    return 0;
}
