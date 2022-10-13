# Measuring_bulkflows
Codes to implement various methods to calculate the Bulk Flow from a peculiar velocity survey. Currently in the processing of being debugged for issues. Probably wouldn't rely on it for accurate calculations at this point..

More info to come.


Should be able to compile the code like this: 

`g++-11 -I/usr/local/include -L/usr/local/lib -lgsl -lgslcblas -fopenmp -lm -llapack -lblas measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.cpp -o measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.exe`

although you might not need these flags: `-I/usr/local/include -L/usr/local/lib`

For testing things/printing resulting matrices (technically a vector of vectors) to the terminal use: 

`print_2D_matrix(matrix_object, "desired_file_name");` 

For saving it to a file use: 

`write_matrix_2_file(matrix_object, "desired_file_name");`
