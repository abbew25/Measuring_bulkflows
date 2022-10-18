# Measuring_bulkflows
Codes to implement various methods to calculate the Bulk Flow from a peculiar velocity survey. This code calculates the MLE Bulk Flow (Kaiser, 1988) and the Minimum Variance Estimator by Feldman et al. (2010) Bulk Flow.

Some bug fixes have been made that caused the code to generate biased results for irregular survey geometries. This has been updated now. 

I should note here, some parts of this code have been largely inspired by code shared with me by Cullan Howlett to calculate the velocity covariance matrix.

This code should obtain similar results to the publicly available code of Morag Scrimgeour that computes the same MVE Bulk Flow.

Should be able to compile the code like this: 

`g++-11 -I/usr/local/include -L/usr/local/lib -lgsl -lgslcblas -fopenmp -lm -llapack -lblas measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.cpp -o measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.exe`

although you might not need these flags: `-I/usr/local/include -L/usr/local/lib`

For testing things/printing resulting matrices (technically a vector of vectors) to the terminal use: 

`print_2D_matrix(matrix_object, "desired_file_name");` 

For saving it to a file use: 

`write_matrix_2_file(matrix_object, "desired_file_name");`


The code reads in 2 input files, a power spectrum (powerspectrum.csv, which is a file with 1 column and 2 rows, a list of k values and a list of P(k) values). The second file is an example mock survey, which has the following measurements for galaxies: right ascension, declination, observed spectroscopic redshift, observed radial peculiar velocity (the code actually doesn't read this value in),  observed radial peculiar velocity uncertainty/error (the code actually doesn't read this value in also), observed log-distance ratio, observed log-distance ratio uncertainty/error. 


