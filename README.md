# Measuring_bulkflows

## Overview 
Codes to implement various methods to calculate the Bulk Flow from a peculiar velocity survey. The code `measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.cpp` calculates the MLE Bulk Flow (Kaiser, 1988) and the Minimum Variance Estimator by Watkins, Feldman and Hudson (2009) Bulk Flow. It can also include modelling for the higher order moments of the velocity field by changing the number of mode functions included in the analysis with the variable `num_mode_funcs` (see Hudson et al, 2010, for more details about higher order modes of the peculiar velocity field). Parts of this code have been largely inspired by code shared with me by Cullan Howlett to calculate the velocity covariance matrix. This code should obtain similar results to the publicly available code of Morag Scrimgeour that computes the same MVE Bulk Flow.

The additional code `BF_MVEpeery_MLEnusser.cpp` calculates the MVE described in Peery et al (2018) and the MLE method described in Nusser (2014). 

This repository will also soon have a PDF file showing various plots and results from applying the bulk flow estimators to mock data, in support of my upcoming paper to be submitted for publication.

## Updates 
Some bug fixes have been made that caused the code to generate biased results for irregular survey geometries. This has been updated now. 

## Required installations

You will need a gcc/gnu compiler, and packages blas, lapack, gsl, fopenmp, eigen. 

## Compilation 
The code `measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.cpp` should be able to be compiled like this: 

`g++ -I/usr/local/include -L/usr/local/lib -lgsl -lgslcblas -fopenmp -lm -llapack -lblas measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.cpp -o measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.exe`.

The code `BF_MVEpeery_MLEnusser.cpp` is compiled using:

`g++ -lgsl -lgslcblas -lblas -llapack -lm -fopenmp -O3 -I/opt/homebrew/include/eigen3/ -I/usr/local/include -L/usr/local/lib BF_MVEpeery_MLEnusser_code.cpp -o BF_MVEpeery_MLEnusser_code.exe -Wall -Wextra`.

The extra flags for the eigen package are because I have included a new function in this code that does block matrix inversion for the velocity covariance matrix that can compute the inverse of a very large matrix (up to 50,000 x 50,000 in size), it is slow but the lapack function throws an error when the matrix is this large. You will need to use multiple threads with fopenmp to get the matrix inversion done for a matrix this large. 

## Other functions 
For print one of the 2D matrices (a vector of vectors object) to the terminal use: 

`print_2D_matrix(matrix_object, "desired_file_name");` 

For saving this kind of object to a file use: 

`write_matrix_2_file(matrix_object, "desired_file_name");`

## Input files
The codes reads in 2 input files, a power spectrum (powerspectrum.csv, which is a file with 1 column and 2 rows, a list of k values and a list of P(k) values). The second file is an example mock survey (example_surveymock.dat), which has the following measurements for galaxies: right ascension, declination, observed spectroscopic redshift, observed log-distance ratio, observed log-distance ratio uncertainty/error, n(r) (mean number density of objects (per Mpc * 1e6) in the survey at the point the galaxy lies at). The last 3 columns have the real vx,vy and vz velocities.

The codes uses the estimator of Watkins et al (2015) to convert observed log-distance ratios (and their uncertainties) to a peculiar velocity observation and uncertainty, although it can be altered to read in directly the observed peculiar velocity observation and uncertainty instead (if desired by the user) the function `read_in_mock_file_data()`.


## Output 
The codes outputs bulk flow modes and the uncertainty for each mode with the results in a choice of arbitrary coordinates, defined by 
$\hat{x} = \cos(\mathrm{RA})\sin(\mathrm{Dec}+90)$,
$\hat{y} = \sin(\mathrm{RA})\sin(\mathrm{Dec}+90)$, 
$\hat{z} = \cos(\mathrm{Dec}+90)$. These should match the coordinate directions of the real velocities given in the last columns of the input example file (for comparing the output of the estimators to the expectation using the real velocities of the objects in the simulation).

