#include <iostream>
#include "assert.h"
#include "Alloc.h"
#include "openacc.h"
#include "stdio.h"

void test_1D()
{
    size_t dim = 100;

    // Allocate arrays
    char   *a1 = newArray1<char>(dim); 
    float  *a2 = newArray1<float>(dim);
    double *a3 = newArray1<double>(dim);

    // Fill with data on host
    for(int i=0; i<dim; i++) {
        a1[i] = 'h';
        a2[i] = 0x80000000;
        a3[i] = 0x1.921fb5p+1;
    }    

    // Fill with data on device
    #pragma acc parallel loop present(a1,a2,a3)
    for(int i=0; i<dim; i++) {
        a1[i] = 'd';
        a2[i] = 0X436A508C;
        a3[i] = 0x1.f58000p+10;
    }

    // Assert host data
    for(int i=0; i<dim; i++) {
        assert(a1[i] == 'h');
        assert(a2[i] == 0x80000000);
        assert(a3[i] == 0x1.921fb5p+1);
    }

    // Copy GPU data to host and assert
    #pragma acc update host(a1[0:dim], a2[0:dim], a3[0:dim])
    for(int i=0; i<dim; i++) {
        assert(a1[i] == 'd');
        assert(a2[i] == 0X436A508C);
        assert(a3[i] == 0x1.f58000p+10);
    }
    
    // Need to free here

    std::cout<<"Passed test_1D"<<std::endl;
}

void test_2D()
{
    size_t dim1 = 100;
    size_t dim2 = 100;

    // Allocate arrays
    char   **a1 = newArray2<char>(dim1, dim2);
    float  **a2 = newArray2<float>(dim1, dim2);
    double **a3 = newArray2<double>(dim1, dim2);

    // Fill with data on host
    for(int i=0; i<dim1; i++) {
       for(int j=0; j<dim2; j++) {
            a1[i][j] = 'h';
            a2[i][j] = 0x80000000;
            a3[i][j] = 0x1.921fb5p+1;
        }
    }

    // Fill with data on device
    #pragma acc parallel present(a1,a2,a3)
    {
        #pragma acc loop
        for(int i=0; i<dim1; i++) {
            #pragma acc loop
            for(int j=0; j<dim2; j++) {
                a1[i][j] = 'd';
                a2[i][j] = 0X436A508C;
                a3[i][j] = 0x1.f58000p+10;
            }
        }
    }

    // Assert host data
    for(int i=0; i<dim1; i++) {
        for(int j=0; j<dim2; j++) {
            assert(a1[i][j] == 'h');
            assert(a2[i][j] == 0x80000000);
            assert(a3[i][j] == 0x1.921fb5p+1);
        }
    }

    // Copy GPU data to host and assert
    for(int i=0; i<dim1; i++) {
        // Note that update host(a1[i][0:dim2]) DOESN'T work
        #pragma acc update host(a1[i:1][0:dim2], a2[i:1][0:dim2], a3[i:1][0:dim2])
        for(int j=0; j<dim2; j++) {
            assert(a1[i][j] == 'd');
            assert(a2[i][j] == 0X436A508C);
            assert(a3[i][j] == 0x1.f58000p+10);
        }
    }

    // Need to free here
    std::cout<<"Passed test_2D"<<std::endl;
}

int main(int argc, char *argv[])
{
    test_1D();
    test_2D();

    return 0;
}
