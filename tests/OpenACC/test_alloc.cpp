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
    // If contiguous we can do memcpy, otherwise we must acc update host on the inner most dimension
    // which is horrible for performance
    acc_memcpy_from_device(a1[0], acc_deviceptr(a1[0]), sizeof(char)*dim1*dim2);
    acc_memcpy_from_device(a2[0], acc_deviceptr(a2[0]), sizeof(float)*dim1*dim2);
    acc_memcpy_from_device(a3[0], acc_deviceptr(a3[0]), sizeof(double)*dim1*dim2);

    for(int i=0; i<dim1; i++) {
        // Note that update host(a1[i][0:dim2]) DOESN'T work
//        #pragma acc update host(a1[i:1][0:dim2], a2[i:1][0:dim2], a3[i:1][0:dim2])
        for(int j=0; j<dim2; j++) {
            assert(a1[i][j] == 'd');
            assert(a2[i][j] == 0X436A508C);
            assert(a3[i][j] == 0x1.f58000p+10);
        }
    }

    // Need to free here
    std::cout<<"Passed test_2D"<<std::endl;
}

void test_3D()
{
    size_t dim1 = 100;
    size_t dim2 = 100;
    size_t dim3 = 100;

    // Allocate arrays
    char   ***a1 = newArray3<char>(dim1, dim2, dim3);
    float  ***a2 = newArray3<float>(dim1, dim2, dim3);
    double ***a3 = newArray3<double>(dim1, dim2, dim3);

    // Fill with data on host
    for(int i=0; i<dim1; i++) {
       for(int j=0; j<dim2; j++) {
           for(int k=0; k<dim3; k++) {
                a1[i][j][k] = 'h';
                a2[i][j][k] = 0x80000000;
                a3[i][j][k] = 0x1.921fb5p+1;
            }
        }
    }

    // Fill with data on device
    #pragma acc parallel present(a1,a2,a3)
    {
        #pragma acc loop
        for(int i=0; i<dim1; i++) {
            #pragma acc loop
            for(int j=0; j<dim2; j++) {
                #pragma acc loop
                for(int k=0; k<dim3; k++) {
                    a1[i][j][k] = 'd';
                    a2[i][j][k] = 0X436A508C;
                    a3[i][j][k] = 0x1.f58000p+10;
                }
            }
        }
    } // acc parallel present

    // Assert host data
    for(int i=0; i<dim1; i++) {
        for(int j=0; j<dim2; j++) {
            for(int k=0; k<dim3; k++) {
                assert(a1[i][j][k] == 'h');
                assert(a2[i][j][k] == 0x80000000);
                assert(a3[i][j][k] == 0x1.921fb5p+1);
            }
        }
    }

    // Copy GPU data to host and assert
    // If contiguous we can do memcpy, otherwise we must acc update host on the inner most dimension
    // which is horrible for performance
    acc_memcpy_from_device(a1[0][0], acc_deviceptr(a1[0][0]), sizeof(char)*dim1*dim2*dim3);
    acc_memcpy_from_device(a2[0][0], acc_deviceptr(a2[0][0]), sizeof(float)*dim1*dim2*dim3);
    acc_memcpy_from_device(a3[0][0], acc_deviceptr(a3[0][0]), sizeof(double)*dim1*dim2*dim3);
    for(int i=0; i<dim1; i++) {
        for(int j=0; j<dim2; j++) {
            for(int k=0; k<dim3; k++) {
                // Note that update host(a1[i][0:dim2]) DOESN'T work
//                #pragma acc update host(a1[i:1][j:1][0:dim2], a2[i:1][j:1][0:dim2], a3[i:1][j:1][0:dim2])
                assert(a1[i][j][k] == 'd');
                assert(a2[i][j][k] == 0X436A508C);
                assert(a3[i][j][k] == 0x1.f58000p+10);
            }
        }
    }

    // Need to free here
    std::cout<<"Passed test_3D"<<std::endl;
}

void test_4D()
{
    size_t dim1 = 100;
    size_t dim2 = 100;
    size_t dim3 = 100;
    size_t dim4 = 100;

    // Allocate arrays
    char   ****a1 = newArray4<char>(dim1, dim2, dim3, dim4);
    float  ****a2 = newArray4<float>(dim1, dim2, dim3, dim4);
    double ****a3 = newArray4<double>(dim1, dim2, dim3, dim4);

    // Fill with data on host
    for(int i=0; i<dim1; i++) {
       for(int j=0; j<dim2; j++) {
           for(int k=0; k<dim3; k++) {
               for(int t=0; t<dim4; t++) {
                    a1[i][j][k][t] = 'h';
                    a2[i][j][k][t] = 0x80000000;
                    a3[i][j][k][t] = 0x1.921fb5p+1;
                }
            }
        }
    }

    // Fill with data on device
    #pragma acc parallel present(a1,a2,a3)
    {
        #pragma acc loop
        for(int i=0; i<dim1; i++) {
            #pragma acc loop
            for(int j=0; j<dim2; j++) {
                #pragma acc loop
                for(int k=0; k<dim3; k++) {
                    #pragma acc loop
                    for(int t=0; t<dim4; t++) {
                        a1[i][j][k][t] = 'd';
                        a2[i][j][k][t] = 0X436A508C;
                        a3[i][j][k][t] = 0x1.f58000p+10;
                    }
                }
            }
        }
    } // acc parallel present

    // Assert host data
    for(int i=0; i<dim1; i++) {
        for(int j=0; j<dim2; j++) {
            for(int k=0; k<dim3; k++) {
                for(int t=0; t<dim4; t++) {
                    assert(a1[i][j][k][t] == 'h');
                    assert(a2[i][j][k][t] == 0x80000000);
                    assert(a3[i][j][k][t] == 0x1.921fb5p+1);
                }
            }
        }
    }

    // Copy GPU data to host and assert
    // If contiguous we can do memcpy, otherwise we must acc update host on the inner most dimension
    // which is horrible for performance
    acc_memcpy_from_device(a1[0][0][0], acc_deviceptr(a1[0][0][0]), sizeof(char)*dim1*dim2*dim3*dim4);
    acc_memcpy_from_device(a2[0][0][0], acc_deviceptr(a2[0][0][0]), sizeof(float)*dim1*dim2*dim3*dim4);
    acc_memcpy_from_device(a3[0][0][0], acc_deviceptr(a3[0][0][0]), sizeof(double)*dim1*dim2*dim3*dim4);
    for(int i=0; i<dim1; i++) {
        for(int j=0; j<dim2; j++) {
            for(int k=0; k<dim3; k++) {
                for(int t=0; t<dim4; t++) {
                    // Note that update host(a1[i][0:dim2]) DOESN'T work
//                    #pragma acc update host(a1[i:1][j:1][k:1][0:dim2], a2[i:1][j:1][k:1][0:dim2], a3[i:1][j:1][k:1][0:dim2])
                    assert(a1[i][j][k][t] == 'd');
                    assert(a2[i][j][k][t] == 0X436A508C);
                    assert(a3[i][j][k][t] == 0x1.f58000p+10);
                }
            }
        }
    }

    // Need to free here
    std::cout<<"Passed test_4D"<<std::endl;
}

int main(int argc, char *argv[])
{
    test_1D();
    test_2D();
    test_3D();
    test_4D();

    return 0;
}
