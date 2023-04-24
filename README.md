# Measuring_bulkflows
Codes to implement various methods to calculate the Bulk Flow from a peculiar velocity survey. The code `measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.cpp` calculates the MLE Bulk Flow (Kaiser, 1988) and the Minimum Variance Estimator by Watkins, Feldman and Hudson (2009) Bulk Flow. It can also include modelling for the higher order moments of the velocity field by changing the number of mode functions included in the analysis with the variable `num_mode_funcs` (see Hudson et al, 2010, for more details about higher order modes of the peculiar velocity field). 

Some bug fixes have been made that caused the code to generate biased results for irregular survey geometries. This has been updated now. 

Parts of this code have been largely inspired by code shared with me by Cullan Howlett to calculate the velocity covariance matrix.

This code should obtain similar results to the publicly available code of Morag Scrimgeour that computes the same MVE Bulk Flow.

The code should be able to be compiled like this: 

`g++-11 -I/usr/local/include -L/usr/local/lib -lgsl -lgslcblas -fopenmp -lm -llapack -lblas measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.cpp -o measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.exe`.

For print one of the 2D matrices (a vector of vectors object) to the terminal use: 

`print_2D_matrix(matrix_object, "desired_file_name");` 

For saving this kind of object to a file use: 

`write_matrix_2_file(matrix_object, "desired_file_name");`

The code reads in 2 input files, a power spectrum (powerspectrum.csv, which is a file with 1 column and 2 rows, a list of k values and a list of P(k) values). The second file is an example mock survey, which has the following measurements for galaxies: right ascension, declination + 90 degrees, observed spectroscopic redshift, observed radial peculiar velocity,  observed radial peculiar velocity uncertainty/error, observed log-distance ratio, observed log-distance ratio uncertainty/error. 

The code uses the estimator of Watkins et al (2015) to convert observed log-distance ratios (and their uncertainties) to a peculiar velocity observation and uncertainty, although it can be altered to read in directly the observed peculiar velocity observation and uncertainty instead in the function `read_in_mock_file_data()`.

The code outputs bulk flows in a choice of arbitrary coordinates, defined by 
$\hat{x} = \cos(\mathrm{RA})\sin(\mathrm{Dec})$,
$\hat{y} = \sin(\mathrm{RA})\sin(\mathrm{Dec})$, 
$\hat{z} = \cos(\mathrm{Dec})$. 
