// compile with eigen, lapack, fopenmp, gsl, use flag -O3 for significant speed up....

/*
g++ -lgsl -lgslcblas -lblas -llapack -lm -fopenmp -O3 -I/opt/homebrew/include/eigen3/ 
-I/usr/local/include -L/usr/local/lib BF_MVEpeery_MLEnusser_code.cpp 
-o BF_MVEpeery_MLEnusser_code.exe -Wall -Wextra
*/

// ------------------------------------------------------------------------------------------------------------

// script for MVE analysis 

// load libraries 
#include <iostream> 
#include <fstream>
#include <sstream>
#include <string>
#include <vector> 
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <chrono>
#include <random> 
#include <ctime>
#include <omp.h> 
#include <Eigen/Core>
#include <Eigen/Dense> 

using namespace std;
using namespace std::chrono;
using namespace Eigen; 

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
string mock_data_file_name = "example_surveymock.dat";
string powerspectrum_file_name = "powerspectrum.csv";


void read_in_mock_file_data(); // function decleration to read in mock data - function definition can be found below main()
void read_in_density_power_spectrum(); // function to read in the matter power spectrum (for a range of different ks)
double Hubble_parameter(double scalefactor); // get hubble parameter at given scalefactor, in km/s/Mpc
vector<double> energy_densities(double scalefactor); // get the energy densities at the time corresponding to scalefactor 
double deceleration_parameter(double scalefactor); // get the deceleration parameter at the time corresponding to scalefactor 
double dq_dt(double scalefactor); // get of deceleration parameter w.r.t. time 
double a_double_dot(double scalefactor);
double jerk_parameter(double scalefactor); // get the jerk parameter at the present day at the time corresponding to scalefactor 
double compute_z_mod(double z_observation); // equation to compute for PV computations 
double func_to_integrate_for_galaxy_distance(double zed); // function to be integrated, used by the function defined below 
double distance_galaxy(double zed); // get the distance to a galaxy computed using given cosmological model at zed
void compute_obs_PVs_from_obs_logdist_ratios(); // get the peculiar velocities (and also their errors)
double integrand_pk_j0_kA(double k, void * params_int_from_above); // function to get integrand of P(k)j_0(kA)
double integrand_pk_j2_kA(double k, void * params_int_from_above); // function to get integrand of P(k)j_2(kA)
double integrand_pk(double k, void * params_int_from_above); 
void get_real_galaxy_distances(); // function to get distances to each galaxy in real space 

vector< vector<double> > compute_R_ij();
vector< vector<double> > compute_matrix_inverse( vector< vector<double> > input_matrix);
void precompute_stuff();

void print_2D_matrix(vector< vector<double> > matrix_2_print, string name);
double mode_func_dot(double modefunc_index, double ra_angle, double dec_angle, double r_dist); // function to get doc
// product between object position vectors and mode functions 
void write_matrix_2_file(vector< vector<double> > matrix_2_write, string filepathname);
vector< vector<double> > calculate_covariance_velocity_moments(vector< vector<double> > matrix_weights_pn, 
vector< vector<double> > matrix_G_mn); 

// creating a structure for the parameters of integration for the integration function to get P(k)j_0(kA) and func for P(k)j_2(kA)
struct params_int2
{
  double A;
};

// calling external functions in LAPACK
extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
}

// function which calls LAPACK to get inverse of 2D array object 
void inverse(double* A, int N)
{
    int *IPIV = new int[N];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;

    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

    delete[] IPIV;
    delete[] WORK;
}

double B_val(vector< vector<double> > G_mn_matrix_inv);
vector<double> Di_vector(double Bval, vector< vector<double> > G_mn_inv, vector< vector<double> > Q_in);
vector<double> Beta_i_vector(vector<double> Di, vector< vector<double> > lambda_ij, vector<double> Li);
vector<double> Li_vector(double B, vector< vector<double> > G_mn_matrix_inv);
vector< vector<double> > M_ij_matrix(vector< vector<double> > G_mn_inv, vector<double> Li_vec);
vector< vector<double> > lambda_ij_matrix(vector< vector<double> > inv_M_ik, vector< vector<double> > G_mn_inv,
vector< vector<double> > Q_km, vector<double> Dk);
vector< vector<double> > compute_Q_qi(); 
vector< vector<double> > compute_MLE_weights_nusser_weight();
vector< vector<double> > weights_in_peery2018(vector< vector<double> > G_mn_inv, vector< vector<double> > Q_im,
vector< vector<double> > lambdaij, vector<double> Betai);

vector< vector<double> > block_matrix_inverse(vector< vector<double> > input_matrix);
MatrixXd get_vec_slice(vector< vector<double> > input_matrix, int start_0, int end_0, int start_1, int end_1);


//----------------------------------------------------------------------------------//

double H0 = 67.32; // km/s/MPC (Hubble constant)
double Omega_matter_0 = 0.30; // dimensionless matter density 
double Omega_Lambda_0 = 1 - Omega_matter_0; // cosmological constant 
double eos_matter = 0.0; // equation of state for matter
double eos_lambda = -1.0; // Eq of state for dark energy
double light_speed = 299792.458; // speed of light in km/s
double sigma_star = 300.0; // ( km/s ) - this is the the 1D velocity dispersion 
int num_mode_funcs = 3; // use 3 for just BF, 9 for BF + shear, 19 for BF + shear + octupoles
int num_ideal_objects_read_in = 10000; // objects to read in from a data file with an 'ideal' survey selection 
double distance_max_ideal_survey = 500.0; // mpc 
double distance_min_ideal_survey = 0.0; // mpc 
double max_ideal_RA = 360.0;



// vectors where we will save data from in file (mock data)
vector<double> RA,Dec,z_obs,logdist,logdist_err, nbar; 

// get data from file with power spectrum computed with CLASS
vector<double> k_vector, Pk_vector;

// spline for integration for power spectrum 
gsl_spline *P_mm_k_spline;
gsl_interp_accel * P_mm_k_acc;

// spline for A with integral result of int j_0(kA) P(k) dk and for A with result of int j_2(kA) P(k) dk
gsl_spline *P_mm_k_j0_kA_spline, *P_mm_k_j2_kA_spline;
gsl_interp_accel * P_mm_k_j0_kA_acc, *P_mm_k_j2_kA_acc;

double int_over_pk; 

// vectors we will compute ourselves
vector<double> object_pvs, object_pv_errs, realspace_galaxy_distances; // a vector with the peculiar velocities, 
//a vector with the peculiar velocity uncertainties


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


int main (int argc, char **argv) { 


    auto start = high_resolution_clock::now(); // starting timer for script execution

    /*-----------------------------------------------------------------------------------------------*/

    // call our function to read in the mock data 
    read_in_mock_file_data();
    // call our function to read in the matter density power spectrum
    read_in_density_power_spectrum();
    // setting up splines 
    precompute_stuff();
    // convert log distance ratios to peculiar velocities and uncertainties on PVs
    compute_obs_PVs_from_obs_logdist_ratios();
    // get distances to each galaxy in real space 
    get_real_galaxy_distances();

    cout << "read in and main set up done" << endl; 
    /*-----------------------------------------------------------------------------------------------*/
    
    
    // run Peery MVE 
    vector< vector<double> > Q_im = compute_Q_qi();
    cout << "Qim done" << endl; 

    // compute the matrix G_ij of the galaxies (G_ij (aka R_ij) = peculiar velocity data covariance matrix)
    vector< vector<double> > G_ij = compute_R_ij(); // getting velocity covariance part 
    //vector< vector<double> > G_ij_inverse = compute_matrix_inverse(G_ij);
    vector< vector<double> > G_ij_inverse = block_matrix_inverse(G_ij); // use block matrix inverse for bigger matrices 
    cout << "G_ij, and inverse done" << endl; 

    double B = B_val(G_ij_inverse);

    cout << "B: " << B << endl;

    vector<double> Li = Li_vector(B, G_ij_inverse);

    vector< vector<double> > Mij = M_ij_matrix(G_ij_inverse, Li);
    vector< vector<double> > Mij_inverse = compute_matrix_inverse(Mij);

    //print_2D_matrix(Mij_inverse, "Mij inverse");

    vector<double> Di = Di_vector(B, G_ij_inverse, Q_im);

    vector< vector<double> > lambdaij = lambda_ij_matrix(Mij_inverse, G_ij_inverse, Q_im, Di);

    //print_2D_matrix(lambdaij, "lambdaij matrix");

    vector<double> Betai = Beta_i_vector(Di, lambdaij, Li);


    vector< vector<double> > weights_mve_peery = weights_in_peery2018(G_ij_inverse, Q_im, lambdaij, Betai);

    cout << "Peery MVE done" << endl;  
    
    /*-----------------------------------------------------------------------------------------------*/
    
    // check constraints are being met 

    // sum _n w_i,n g_n,j = delta_ij
    
    vector< vector<double> > constraint_one(num_mode_funcs, vector<double>(num_mode_funcs));

    for (int i = 0; i < num_mode_funcs; i++){
      for (int j = 0; j < num_mode_funcs; j++){
        constraint_one[i][j] = 0;
        for (int n = 0; n < z_obs.size(); n++){
          constraint_one[i][j] += weights_mve_peery[i][n]*mode_func_dot(j, RA[n],Dec[n],realspace_galaxy_distances[n]);
        }
      }
    }

    print_2D_matrix(constraint_one, "printing constraint equation result one");

    // sum_n w_in c z_n = 0

    vector<double> constraint_two(num_mode_funcs);

    for (int i = 0; i < num_mode_funcs; i++){
      constraint_two[i] = 0;
      for (int n = 0; n < z_obs.size(); n++){
        constraint_two[i] += weights_mve_peery[i][n]*light_speed*z_obs[n];
      }
    }
    
    cout << "printing constraint equation result two:" << endl;

    for (int i = 0; i < num_mode_funcs; i++){
        cout << constraint_two[i] << " ";
    }
    cout << endl; 

    /*-----------------------------------------------------------------------------------------------*/

    // run Nusser MLE 

    vector< vector<double> > weights_mle_nusser = compute_MLE_weights_nusser_weight();

    cout << "Nusser MLE done" << endl; 

    /*-----------------------------------------------------------------------------------------------*/


    // calculate BFs

    double BF_x_mle = 0; 
    double BF_y_mle = 0; 
    double BF_z_mle = 0;
    double BF_x_mve = 0; 
    double BF_y_mve = 0; 
    double BF_z_mve = 0;

    for (int m = 0; m < z_obs.size(); m++){

        BF_x_mle += weights_mle_nusser[0][m]*object_pvs[m];
        BF_y_mle += weights_mle_nusser[1][m]*object_pvs[m];
        BF_z_mle += weights_mle_nusser[2][m]*object_pvs[m];
        BF_x_mve += weights_mve_peery[0][m]*object_pvs[m];
        BF_y_mve += weights_mve_peery[1][m]*object_pvs[m];
        BF_z_mve += weights_mve_peery[2][m]*object_pvs[m];

    }


    // calculate covariance matrix of BFs for each method 
    vector< vector<double> > cov_matrix_mle = calculate_covariance_velocity_moments(weights_mle_nusser, G_ij);
    vector< vector<double> > cov_matrix_mve = calculate_covariance_velocity_moments(weights_mve_peery, G_ij);

    /*-----------------------------------------------------------------------------------------------*/

    
    // write the results to a file 
    ofstream results_file;
    results_file.open(("MVE_result_peery_MLE_result_nusser.txt" ));

    results_file << BF_x_mve << " " << BF_y_mve << " " << BF_z_mve << " " << endl;
    results_file << sqrt(cov_matrix_mve[0][0]) << " " << sqrt(cov_matrix_mve[1][1]) << " " << sqrt(cov_matrix_mve[2][2]) << endl; 
    
    results_file <<  " ---------------------------------------------------------------" << endl; 

    results_file << BF_x_mle << " " << BF_y_mle << " " << BF_z_mle << endl; 
    results_file << " " << sqrt(cov_matrix_mle[0][0]) << " " << sqrt(cov_matrix_mle[1][1]) << " " << sqrt(cov_matrix_mle[2][2]) << endl; 

    results_file.close();
    

    /*-----------------------------------------------------------------------------------------------*/
    

    // finalizing 
    gsl_spline_free(P_mm_k_spline);
    gsl_interp_accel_free(P_mm_k_acc);

    gsl_spline_free(P_mm_k_j0_kA_spline);
    gsl_interp_accel_free(P_mm_k_j0_kA_acc);

    gsl_spline_free(P_mm_k_j2_kA_spline);
    gsl_interp_accel_free(P_mm_k_j2_kA_acc);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Elapsed seconds while running program: " << duration.count()/1e6 << endl;


    return 0;

}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


void read_in_mock_file_data() {


  string line; // we will loop through the lines in the file 
  int numoflines = 0; // counter for lines in file 
  ifstream mockdata; //instantiating object for mockdata file to read in 
  mockdata.open(mock_data_file_name);
  if (!(mockdata.is_open())) {throw "Failed to open input file: mock data.";}
  
  // we want to loop through the lines in the file so we can save the data to vectors 
  while ( getline(mockdata,line) )
  {
    
    stringstream ss(line);
    vector<string> splitline;

    while( ss.good() )
    {
    string substr;
    getline( ss, substr, ' ' );
    splitline.push_back( substr );
    }
    
    // put data in globally defined vectors ... 

    RA.push_back(stod(splitline[0])); // right ascension (measured)
    Dec.push_back(stod(splitline[1])-90.0); // declination (measured)
    z_obs.push_back(stod(splitline[2])); // observed redshift in sim
    logdist.push_back((stod(splitline[3]))); // observed log distance ratio
    logdist_err.push_back(stod(splitline[4])); // uncertainty on log distance ratio (as determined from observations)
    nbar.push_back(stod(splitline[5])*1e-6); 

    numoflines += 1;

  } // end of while loop through mock data file 

  mockdata.close();

} // end of function to read in mock data 


/* ----------------------------------------------------------- */ 


void read_in_density_power_spectrum(){

  string line; // we will loop through the lines in the file 
  int numoflines = 0; // counter for lines in file 

  ifstream powerspectrum_data; //instantiating object for power spectrum file to read in 
  powerspectrum_data.open(powerspectrum_file_name);
  if (!(powerspectrum_data.is_open())) {throw "Failed to open input file: power spectrum.";}

  while ( getline(powerspectrum_data,line) ){
    
    if (numoflines == 0) {

      stringstream ss(line);

      while( ss.good() )
      {
        string substr;
        getline( ss, substr, ' ' );
        k_vector.push_back( stod(substr)*h );
      }


    } else if (numoflines == 1) {

      stringstream ss(line);

      while( ss.good() )
      {
        string substr;
        getline( ss, substr, ' ' );
        Pk_vector.push_back( stod(substr)/(h*h*h) );
      }

    }

    numoflines += 1; 
  }


}

/* ----------------------------------------------------------- */ 

double Hubble_parameter(double scalefactor) {

  // hubble parameter at given scalefactor  
  double H_a = H0*pow( Omega_matter_0*pow(scalefactor, -3.0*(1.0 + eos_matter)) + Omega_Lambda_0*pow(scalefactor, -3.0*(1.0 + eos_lambda) ), 0.5 );

  return H_a; 

}

/* ----------------------------------------------------------- */ 

vector<double> energy_densities(double scalefactor){

  double omega_matter_a = Omega_matter_0*(pow(scalefactor, -3.0*(1.0 + eos_matter)));

  double omega_lambda_a = Omega_Lambda_0*(pow(scalefactor, -3.0*(1.0 + eos_lambda)));

  vector<double> densities_vec;

  densities_vec.push_back(omega_matter_a);
  densities_vec.push_back(omega_lambda_a);

  return densities_vec;

}

/* ----------------------------------------------------------- */ 

double deceleration_parameter(double scalefactor){

  vector<double> vec_densities = energy_densities(scalefactor);
  double Om_a = vec_densities[0];
  double Ol_a = vec_densities[1];

  double q_a = 0.5*(pow(H0,2)/pow(Hubble_parameter(scalefactor),2))*( Om_a*(1.0 + 3.0*eos_matter) + Ol_a*(1.0 + 3.0*eos_lambda) );

  return q_a;

}

/* ----------------------------------------------------------- */ 

double a_double_dot(double scalefactor){

  double res = -1.0*scalefactor*pow(Hubble_parameter(scalefactor),2)*deceleration_parameter(scalefactor);
  return res;

}

/* ----------------------------------------------------------- */ 

double dq_dt(double scalefactor){

  // dq_dt = dq_da * da_dt using chain rule 

  double dadt = scalefactor*Hubble_parameter(scalefactor);
  double del_a = 1e-3;
  double dqda_approx = (deceleration_parameter(scalefactor) - deceleration_parameter(scalefactor - del_a))/(del_a);

  return dadt*dqda_approx;

}

/* ----------------------------------------------------------- */ 

double jerk_parameter(double scalefactor){

  double q_a = deceleration_parameter(scalefactor);

  double dq_a_dt = dq_dt(scalefactor); 

  double Ha = Hubble_parameter(scalefactor);

  double j_a = q_a + 2.0*(pow(q_a,2)) - dq_a_dt/Ha;

  return j_a;

}

/* ----------------------------------------------------------- */ 

double compute_z_mod(double z_observation){

  double q0 = deceleration_parameter(1.0);
  double j0 = jerk_parameter(1.0);

  //cout << q0 << " " << j0 << endl; 

  double z_mod = z_observation*( 1.0 + 0.5*(1.0 - q0)*z_observation - (1.0/6.0)*(j0 - q0 - 3.0*(pow(q0, 2)) + 1.0 )*(pow(z_observation, 2)) );
  //cout << z_mod << endl;
  return z_mod;

}

/* ----------------------------------------------------------- */ 

void compute_obs_PVs_from_obs_logdist_ratios(){

  // using the approximation of Watkins and Feldman (2015) for getting PV and PV uncertainty
  int loop_length = z_obs.size();

  for (int i = 0; i < loop_length; i++) {

    double z_mod_obs = compute_z_mod(z_obs[i]);

    double v_peculiar_obs = ((light_speed*z_mod_obs)/(1.0 + z_mod_obs))*logdist[i]*log(10);
    object_pvs.push_back(v_peculiar_obs);

    double v_peculiar_obs_err = ((light_speed*z_mod_obs)/(1.0 + z_mod_obs))*logdist_err[i]*log(10);
    object_pv_errs.push_back(v_peculiar_obs_err);

  }

}

/* ----------------------------------------------------------- */ 

double func_to_integrate_for_galaxy_distance(double zed, void * p){

  double a = 1.0/(1.0 + zed);

  double result = (light_speed/H0)/sqrt(  Omega_matter_0*pow(a, -3.0*(eos_matter + 1.0) ) +  Omega_Lambda_0*pow(a, -3.0*(eos_lambda + 1.0) ) ); 
  return result;

}

/* ----------------------------------------------------------- */ 

double distance_galaxy(double zed){ //This calculates the distance given the redshift (red) for the galaxy

  double result, error;
  gsl_function F;
  gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
  F.function = &func_to_integrate_for_galaxy_distance;
  gsl_integration_qags(&F, 0.0, zed, 1e-9, 1e-9, 1000, w, &result, &error);
  gsl_integration_workspace_free(w);
  return result;

}

/* ----------------------------------------------------------- */ 

double integrand_pk_j0_kA(double k, void * params_int_from_above){

  params_int2 these_params = *(params_int2*)params_int_from_above;
  
  double pk_val = gsl_spline_eval(P_mm_k_spline, k, P_mm_k_acc);
  double res = pk_val*gsl_sf_bessel_jl(0, k*these_params.A);

  return res;
}

/* ----------------------------------------------------------- */ 

double integrand_pk_j2_kA(double k, void * params_int_from_above){

  params_int2 these_params = *(params_int2*)params_int_from_above;
  
  double pk_val = gsl_spline_eval(P_mm_k_spline, k, P_mm_k_acc);
  double res = pk_val*gsl_sf_bessel_jl(2, k*these_params.A);

  return res;
}

/* ----------------------------------------------------------- */ 

double integrand_pk(double k, void * params_int_from_above){

  //params_int2 these_params = *(params_int2*)params_int_from_above;
  
  double pk_val = gsl_spline_eval(P_mm_k_spline, k, P_mm_k_acc);

  return pk_val;
}

/* ----------------------------------------------------------- */ 

void get_real_galaxy_distances(){

  // get the actual distance to the galaxies (r) - in real space and save the result
    for (int zval = 0; zval < z_obs.size(); zval++){

        //cout << z_obs[zval] << endl; 
        double dist_redshift_space = distance_galaxy(z_obs[zval]); 
        //realspace_galaxy_distances.push_back( dist_redshift_space/pow(10.0, logdist[zval])  );
        realspace_galaxy_distances.push_back(dist_redshift_space);
    }

}

/* ----------------------------------------------------------- */ 

void precompute_stuff(){

    // make a spline for looking up the value of the power spectrum at a given k
    double k_array[k_vector.size()], Pk_array[k_vector.size()];

    copy(k_vector.begin(), k_vector.end(), k_array); // need to copy vectors to arrays to use for the spline
    copy(Pk_vector.begin(), Pk_vector.end(), Pk_array);

    P_mm_k_acc = gsl_interp_accel_alloc(); // steps to create the gsl spline object for power spectrum integration 
    P_mm_k_spline = gsl_spline_alloc(gsl_interp_cspline, k_vector.size());
    gsl_spline_init(P_mm_k_spline, k_array, Pk_array, k_vector.size());

    // to speed up our construction of the matrices G_ij and Q_qi in the next steps we need to precompute some integrals 
    // need to get the result of integral P(k)dk, P(k)j_0(kA)dk, P(k)j_2(kA)dk
    double A_min = 0; // A = sqrt( r_1^2 + r_2^2 - 2r_1 r_2 cos(a)) where a is the angle between galaxies 1 and 2, r_i is the distance to ith galaxy
    double A_max = 1800.0; // Mpcs (corresponds to roughly an object at z = 0.5 which should be more than far enough for our survey)
    int bins = 15000;
    double kmin = k_vector[0];
    double kmax = k_vector[(k_vector.size() - 1.0)];
    double A_vector[bins], integral_j0_pk[bins], integral_j2_pk[bins];
    // compute the integrals for a range of A values, store results in vectors, then create a spline object to look at later 
    
    #pragma omp parallel for 
    for (int A_index = 0; A_index < bins; A_index++){

      double A = A_index*(A_max - A_min)/bins;
      A_vector[A_index] = A;
      //cout << A << endl; 
      // take these and set the values in the struct params_int2 for the integral, do the integration 
      params_int2 custom_params_int0, custom_params_int2; // our custom struct
      double result0, error0, result2, error2; // integration result and error 

      gsl_function F0, F2;
      F0.function = &integrand_pk_j0_kA;
      F2.function = &integrand_pk_j2_kA;

      F0.params = &custom_params_int0;
      F2.params = &custom_params_int2;
      gsl_integration_workspace * w0 = gsl_integration_workspace_alloc(1000);
      gsl_integration_workspace * w2 = gsl_integration_workspace_alloc(1000);  
      
      custom_params_int0.A = A;
      custom_params_int2.A = A;

      gsl_integration_qags(&F0, kmin, kmax, 5e-9, 5e-9, 1000, w0, &result0, &error0);
      gsl_integration_workspace_free(w0);
      gsl_integration_qags(&F2, kmin, kmax, 5e-9, 5e-9, 1000, w2, &result2, &error2);
      gsl_integration_workspace_free(w2);

      integral_j0_pk[A_index] = result0;
      integral_j2_pk[A_index] = result2;

      //cout << result2 << endl; 


    }
    // end of pragma omp parallel for 
     
     // create the spline objects 
    P_mm_k_j0_kA_acc = gsl_interp_accel_alloc(); // steps to create the gsl spline object for power spectrum integral (with j_0(kA))
    P_mm_k_j2_kA_acc = gsl_interp_accel_alloc(); // like above line ^ 
    P_mm_k_j0_kA_spline = gsl_spline_alloc(gsl_interp_cspline, bins);
    P_mm_k_j2_kA_spline = gsl_spline_alloc(gsl_interp_cspline, bins);
    gsl_spline_init(P_mm_k_j0_kA_spline, A_vector, integral_j0_pk, bins);
    gsl_spline_init(P_mm_k_j2_kA_spline, A_vector, integral_j2_pk, bins);


    params_int2 custom_params_int4; // our custom struct
    double result4, error4; // integration result and error 

    gsl_function F4;
    F4.function = &integrand_pk;

    F4.params = &custom_params_int4;
    gsl_integration_workspace * w4 = gsl_integration_workspace_alloc(1000);

    gsl_integration_qags(&F4, kmin, kmax, 5e-9, 5e-9, 1000, w4, &result4, &error4);
    gsl_integration_workspace_free(w4);
  
    int_over_pk = result4; 

}

/* ----------------------------------------------------------- */ 

vector< vector<double> > compute_R_ij(){

    vector< vector<double> > R_ij_matrix(z_obs.size(), vector<double>(z_obs.size()));
    double integral_prefactor = pow(Omega_matter_0, 1.1)*(H0*H0)/(2.0*(M_PI*M_PI));

    // R_ij_eps
    #pragma omp parallel for 
      for (int i = 0; i < z_obs.size(); i ++){

          for (int j = 0; j < z_obs.size(); j ++){

            R_ij_matrix[i][j] = 0.0; 
            
          
            if (i == j){
                double p1 = ( pow(object_pv_errs[i],2.0)  + pow(sigma_star, 2.0) );
                double p2 = integral_prefactor*(1.0/3.0)*int_over_pk; 
                R_ij_matrix[i][i] = (p1 + p2);
                
            }
            
          }
      }
    // end of nested pragma omp 
    
    // we will use threading to split up the matrix elements between threads to speed up the task of building this matrix
    #pragma omp parallel for
    for (int i = 0; i < z_obs.size(); i++){

        double integral_prefactor = pow(Omega_matter_0, 1.1)*(H0*H0)/(2.0*(M_PI*M_PI));

        for(int j = i; j < z_obs.size(); j++){ // since the matrix is symmetric we only need to compute half of the elements really 

            if (i == j){
              continue; 
            } else { 

              // get the distances to the ith and jth galaxies 
              double idist = realspace_galaxy_distances[i];
              double jdist = realspace_galaxy_distances[j];

              //get the angle between the ith and jth galaxies in the sky (radians)
              double val = mode_func_dot(0, RA[i], Dec[i], idist)*mode_func_dot(0, RA[j], Dec[j], jdist) + 
              mode_func_dot(1, RA[i], Dec[i], idist)*mode_func_dot(1, RA[j], Dec[j], jdist) 
              + mode_func_dot(2, RA[i], Dec[i], idist)*mode_func_dot(2, RA[j], Dec[j], jdist);
              
              if (val >= 1.0 ) { val = 1.0; }
              if (val <= -1.0) { val = -1.0;}

              double alpha = acos( val  );

              //if (i == j){alpha = 0.0;}

              //if (RA[i] == RA[j] && Dec[i] == Dec[j]) { alpha = 0; }

              double A = pow(  (pow(idist,2) + pow(jdist,2) - 2.0*idist*jdist*cos(alpha)) , 0.5);

              //if (i == j) { A = 0.0; }

              if ( round((pow(idist,2) + pow(jdist,2) - 2.0*idist*jdist*cos(alpha))*1000.0)/1000.0 == 0.0  ) { A = 0.0; }

              if (idist == jdist && alpha == 0.0){ A = 0.0; }

              if ((isnan(A) | isnan(alpha) )) {
                // cout << "idist: " << idist << endl; 
                // cout << "jdist: " << jdist << endl; 
                // cout << "A: " << A << endl;
                // cout << "alpha: " << alpha << endl;
                // cout << val << endl;
                cout << "A or alpha are nan (in compute_R_ij())." << endl;
                throw "A or alpha are nan (in compute_R_ij()).";
              }

              if (abs(A) < 1e-6) { A = 0.0; }

              if (A < 0 || A > 2000.0) { 
                throw "A is out of range.";
              }

              if (A == 0){ 
                R_ij_matrix[i][j] = integral_prefactor*(1.0/3.0)*int_over_pk; 
                R_ij_matrix[j][i] = integral_prefactor*(1.0/3.0)*int_over_pk; 
                
              } else {
                  
                double term1 = integral_prefactor*(1.0/3.0)*cos(alpha)*(gsl_spline_eval(P_mm_k_j0_kA_spline, A, P_mm_k_j0_kA_acc));
                double term2 = - integral_prefactor*(1.0/3.0)*cos(alpha)*( 2.0*gsl_spline_eval(P_mm_k_j2_kA_spline, A, P_mm_k_j2_kA_acc) );
                double term3 = integral_prefactor*(1.0/pow(A,2))*gsl_spline_eval(P_mm_k_j2_kA_spline, A, P_mm_k_j2_kA_acc)*idist*jdist*pow(sin(alpha), 2);
            
                R_ij_matrix[i][j] = (term1 + term2 + term3); 
                R_ij_matrix[j][i] = (term1 + term2 + term3); 

              } 

              //if (i != j) { R_ij_matrix[j][i] = R_ij_matrix[i][j]; }

            } // end else i == j statement 

        } // end j loop
    } // end i loop 
    // end of nested pragma omp parallel for 
    

    return R_ij_matrix;
}

/* ----------------------------------------------------------- */ 

vector< vector<double> > compute_matrix_inverse( vector< vector<double> > input_matrix){

    // put the matrix into a 1D vector and get inverse from LAPACK

    vector<double> input_copy_for_inverse(pow(input_matrix.size(),2));
    
    #pragma omp parallel for 
    for (int i = 0; i < input_matrix.size(); i++){
      for (int j = 0; j < input_matrix.size(); j++){

        input_copy_for_inverse[j + i*(input_matrix.size())] = input_matrix[i][j];

      } // loop i end 
    } // loop j end 
    // end of nested pragma omp parallel for 
    

    inverse(input_copy_for_inverse.data(), input_matrix.size());

    vector< vector<double> > return_inverse(input_matrix.size(), vector<double>(input_matrix.size()));


    #pragma omp parallel for 
    for (int i = 0; i < input_matrix.size(); i++){
      for (int j = 0; j < input_matrix.size(); j++){

        return_inverse[i][j] = input_copy_for_inverse[j + i*(input_matrix.size())];

      } // loop i end 
    } // loop j end 
    // end of nested pragma omp parallel for 

    return return_inverse;
}

/* ----------------------------------------------------------- */

void print_2D_matrix(vector< vector<double> > matrix_2_print, string name){

    // printing to the terminal
    cout << "-----------------------------------------------------------" << endl;
    cout << name << ": " << endl;
    
    int dim_1 = matrix_2_print.size();
    int dim_2 = matrix_2_print[0].size();

    for (int i = 0; i < dim_1; i++){
        for (int j = 0; j < dim_2; j++){
            

            cout << matrix_2_print[i][j] << " ";

        }

        cout << endl;
    }

    cout << "-----------------------------------------------------------" << endl;

}

/* ----------------------------------------------------------- */

void write_matrix_2_file(vector< vector<double> > matrix_2_write, string filepathname){

    ofstream matrixfile;
    matrixfile.open((filepathname));
    
    int dim_1 = matrix_2_write.size();
    int dim_2 = matrix_2_write[0].size();

    for (int i = 0; i < dim_1; i++){
        for (int j = 0; j < dim_2; j++){
            

            matrixfile << matrix_2_write[i][j] << " ";

        }

        matrixfile << endl;
    }

    matrixfile.close();
}

/* ----------------------------------------------------------- */ 


vector< vector<double> > calculate_covariance_velocity_moments(vector< vector<double> > matrix_weights_pn, vector< vector<double> > matrix_G_mn){
    
    vector< vector<double> > covariance_matrix_moments_ab(num_mode_funcs, vector<double>(num_mode_funcs));
    
    #pragma omp parallel for 
    for (int a = 0; a < num_mode_funcs; a++){
        for (int b = a; b < num_mode_funcs; b++){

          covariance_matrix_moments_ab[a][b] = 0.0; 
         
            for (int m = 0; m < z_obs.size(); m++){
                for (int n = 0; n < z_obs.size(); n++){
                    
                  covariance_matrix_moments_ab[a][b] += matrix_weights_pn[a][m]*matrix_weights_pn[b][n]*matrix_G_mn[m][n];
                    
                } // end of n loop
            } // end of m loop

        if (a != b) { covariance_matrix_moments_ab[b][a] = covariance_matrix_moments_ab[a][b]; }
            
        } // end of b loop
    } // end of a loop
    // end of nested pragma omp parallel for 
    
    return covariance_matrix_moments_ab;
}


/* ----------------------------------------------------------- */ 

double mode_func_dot(double modefunc_index, double ra_angle, double dec_angle, double r_dist){

  double res = 0;

  double xhat = cos(ra_angle*M_PI/180.0)*sin(dec_angle*M_PI/180.0);
  double yhat = sin(ra_angle*M_PI/180.0)*sin(dec_angle*M_PI/180.0);
  double zhat = cos(dec_angle*M_PI/180.0);

  // BF moments ------------------------------------------------------------------------

  if (modefunc_index == 0){ // BF x

    res = xhat;

  } else if (modefunc_index == 1){ // BF y

    res = yhat; 

  } else if (modefunc_index == 2){ // BF z

    res = zhat; 
  // shear moments ----------------------------------------------------------------------
  
  } else if (modefunc_index == 3){ // shear xx

    res = r_dist*xhat*xhat;

  } else if (modefunc_index == 4){ // shear yy

    res = r_dist*yhat*yhat;

  } else if (modefunc_index == 5){ // shear zz

    res = r_dist*zhat*zhat;

  } else if (modefunc_index == 6){ // shear xy

    res = 2.0*r_dist*xhat*yhat;

  } else if (modefunc_index == 7){ // shear yz

    res = 2.0*r_dist*zhat*yhat;

  } else if (modefunc_index == 8){ // shear xz

    res = 2.0*r_dist*xhat*zhat;

  // octupole moments -------------------------------------------------------------------- 

  } else if (modefunc_index == 9){ // octupole xxx

    res = pow(r_dist,2)*xhat*xhat*xhat - xhat*( M_PI*pow(r_dist,4)/3.0 ); 

  } else if (modefunc_index == 10){ // octupole yyy 

    res = pow(r_dist,2)*yhat*yhat*yhat - yhat*( M_PI*pow(r_dist,4)/3.0 );

  } else if (modefunc_index == 11){ // octupole zzz 

    res = pow(r_dist,2)*zhat*zhat*zhat - zhat*( M_PI*pow(r_dist,4)/3.0 );

  } else if (modefunc_index == 12){ // octupole xxy

    res = 3.0*pow(r_dist,2)*xhat*xhat*yhat - yhat*( M_PI*pow(r_dist,4)/3.0 );

  } else if (modefunc_index == 13){ // octupole yyz 

    res = 3.0*pow(r_dist,2)*yhat*yhat*zhat - zhat*( M_PI*pow(r_dist,4)/3.0 );

  } else if (modefunc_index == 14){ // octupole zzx 

    res = 3.0*pow(r_dist,2)*xhat*zhat*zhat - xhat*( M_PI*pow(r_dist,4)/3.0 );

  } else if (modefunc_index == 15){ // octupole yyx

    res = 3.0*pow(r_dist,2)*xhat*yhat*yhat - xhat*( M_PI*pow(r_dist,4)/3.0 );

  } else if (modefunc_index == 16){ // octupole zzy

    res = 3.0*pow(r_dist,2)*yhat*zhat*zhat - yhat*( M_PI*pow(r_dist,4)/3.0 );

  } else if (modefunc_index == 17){ // octupole xxz 

    res = 3.0*pow(r_dist,2)*xhat*xhat*zhat - zhat*( M_PI*pow(r_dist,4)/3.0 );

  } else if (modefunc_index == 18){ // octupole xyz 

    res = 6.0*pow(r_dist,2)*xhat*yhat*zhat;

  }

  return res; 
}

/* ----------------------------------------------------------- */

double B_val(vector< vector<double> > G_mn_matrix_inv){

    double B = 0; 

    for (int m = 0; m < z_obs.size(); m++){
        for (int n = 0; n < z_obs.size(); n++){

            B += G_mn_matrix_inv[m][n]*light_speed*light_speed*z_obs[n]*z_obs[m];

        }
    }

    return B;
}

/* ----------------------------------------------------------- */

vector<double> Li_vector(double Bv, vector< vector<double> > G_mn_matrix_inv){

    vector<double> Li(num_mode_funcs);

    for (int i = 0; i < num_mode_funcs; i++){

        Li[i] = 0; 

        for (int m = 0; m < z_obs.size(); m++){
            for (int n = 0; n < z_obs.size(); n++){

                Li[i] += G_mn_matrix_inv[m][n]*mode_func_dot(i,RA[n],Dec[n], realspace_galaxy_distances[n])*light_speed*z_obs[m]/Bv; 

            }
        }
    }

    return Li;

}

/* ----------------------------------------------------------- */

vector< vector<double> > M_ij_matrix(vector< vector<double> > G_mn_inv, vector<double> Li_vec){

    vector< vector<double> > M_ij_mat(num_mode_funcs, vector<double>(num_mode_funcs));

    for (int i = 0; i < num_mode_funcs; i++){
        for (int j = 0; j < num_mode_funcs; j++){

            M_ij_mat[i][j] = 0.0; 

            for (int m = 0; m < z_obs.size(); m++){
                for (int n = 0; n < z_obs.size(); n++){

                    double mode_ni = mode_func_dot(i,RA[n],Dec[n],realspace_galaxy_distances[n]);
                    double mode_mj = mode_func_dot(j,RA[m],Dec[m],realspace_galaxy_distances[m]);

                    M_ij_mat[i][j] += 0.5*G_mn_inv[n][m]*(mode_ni - (Li_vec[i]*light_speed*z_obs[n]))*mode_mj;

                }
            }


        }
    }

    return M_ij_mat; 
    
}

/* ----------------------------------------------------------- */

vector<double> Di_vector(double Bval, vector< vector<double> > G_mn_inv, vector< vector<double> > Q_in_mat){

    vector<double> Di_vec(num_mode_funcs);

    for (int i = 0; i < num_mode_funcs; i++){

        Di_vec[i] = 0;

        for (int m = 0; m < z_obs.size(); m++){
            for (int n = 0; n < z_obs.size(); n++){
                
                Di_vec[i] += G_mn_inv[m][n]*Q_in_mat[i][n]*light_speed*z_obs[m]/Bval;

            }
        }

    }

    return Di_vec;

}

/* ----------------------------------------------------------- */

vector<double> Beta_i_vector(vector<double> Di_mat, vector< vector<double> > lambda_ij_mat, vector<double> Li_mat){

    vector<double> Beta_i_vec(num_mode_funcs);

    for (int i = 0; i < num_mode_funcs; i++){

        Beta_i_vec[i] = 0.0; 

        double sumj = 0;

        for (int j = 0; j < num_mode_funcs; j++){

            sumj += lambda_ij_mat[i][j]*Li_mat[j];

        }
        Beta_i_vec[i] = 2.0*Di_mat[i] - sumj;
    }

    return Beta_i_vec;
}

/* ----------------------------------------------------------- */

vector< vector<double> > lambda_ij_matrix(vector< vector<double> > inv_M_ik, vector< vector<double> > G_mn_inv,
vector< vector<double> > Q_km, vector<double> Dk){

    vector< vector<double> > lambda_ij(num_mode_funcs, vector<double>(num_mode_funcs));

    for (int i = 0; i < num_mode_funcs; i++){
        for (int j = 0; j < num_mode_funcs; j++){

            lambda_ij[i][j] = 0.0; 

            for (int k = 0; k < num_mode_funcs; k++){

                double mn_sum = 0;

                for (int m = 0; m < z_obs.size(); m++){
                    for (int n = 0; n < z_obs.size(); n++){

                        double mod_nk = mode_func_dot(k, RA[n],Dec[n],realspace_galaxy_distances[n]); 
                        mn_sum += G_mn_inv[n][m]*(Q_km[i][m] - (Dk[i]*light_speed*z_obs[m]))*mod_nk;

                
                    }
                }
            
                double delta_ik = 0; 
                if (i == k){
                  delta_ik = 1.0; 
                }

                lambda_ij[i][j] += inv_M_ik[j][k]*(mn_sum - delta_ik);


            }


        }
    }

    return lambda_ij;
}

/* ----------------------------------------------------------- */

vector< vector<double> > weights_in_peery2018(vector< vector<double> > G_mn_inv, vector< vector<double> > Q_im,
vector< vector<double> > lambdaij_mat, vector<double> Betai){

    vector< vector<double> > w_in(num_mode_funcs, vector<double>(z_obs.size()));

    for (int i = 0; i < num_mode_funcs; i++){
        for (int n = 0; n < z_obs.size(); n++){

            w_in[i][n] = 0; 

            for (int m = 0; m < z_obs.size(); m++){

                double sum_j = 0;

                for (int j = 0; j < num_mode_funcs; j++){
                    sum_j += 0.5*lambdaij_mat[i][j]*mode_func_dot(j, RA[m],Dec[m],realspace_galaxy_distances[m]);
                }

                w_in[i][n] += G_mn_inv[n][m]*(Q_im[i][m] - sum_j - (0.5*Betai[i]*light_speed*z_obs[m]));

            }

        }
    }


    return w_in; 
}

/* ----------------------------------------------------------- */

vector< vector<double> > compute_MLE_weights_nusser_weight(){

  vector< vector<double> > A_pj_matrix(num_mode_funcs, vector<double>(num_mode_funcs));

  // calculate elements of A_pj ----------
  for (int p = 0; p < num_mode_funcs; p++){
    for (int j = 0; j < num_mode_funcs; j++){

      A_pj_matrix[p][j] = 0.0; 

      for (int n = 0; n < z_obs.size(); n++){

        double weighting = 1.0/( nbar[n]*pow(realspace_galaxy_distances[n],2) ); 
        
        double ghat_p_n = mode_func_dot(p, RA[n], Dec[n], realspace_galaxy_distances[n]);
        double ghat_j_n = mode_func_dot(j, RA[n], Dec[n], realspace_galaxy_distances[n]);

        A_pj_matrix[p][j] += ghat_p_n*ghat_j_n*weighting; 
      
      } // loop over galaxies 

    } // loop over p
  } // loop over j

  cout << " A pj matrix done" << endl;

  // invert A_pj -------------------------

  vector< vector<double> > A_pj_matrix_inv = compute_matrix_inverse(A_pj_matrix);

  vector< vector<double> > weights_pn(num_mode_funcs, vector<double>(z_obs.size()));

  for(int p = 0; p < num_mode_funcs; p++){
    for (int n = 0; n < z_obs.size(); n++){

      weights_pn[p][n] = 0.0; 

      for (int j = 0; j < num_mode_funcs; j++){

        double weighting = 1.0/( nbar[n]*pow(realspace_galaxy_distances[n],2) ); 
        
        double ghat_j_n = mode_func_dot(j, RA[n], Dec[n], realspace_galaxy_distances[n]);

        weights_pn[p][n] += (A_pj_matrix_inv[p][j]*ghat_j_n*weighting); 

      } // loop over j

    } // loop over galaxies 
  } // loop over p

    return weights_pn;

}

/* ----------------------------------------------------------- */ 

vector< vector<double> > compute_Q_qi(){

    // initialise vectors of data we will need from the ideal data - only going to use these in this function
    // so they dont need to be defined globally
    vector<double> RA_ideal_survey, Dec_ideal_survey, r_ideal_survey;

    // now creating some 'data' from an ideal survey that is full sky and has a gaussian number density of objects 
    // calculating some parameters to get the number of objects as a function of r appropriately given the number of objects 
    // we want and also the gaussian width R_I we set, and the maximum and minimum distances to objects we want to consider 
    
    //---------------------------------------------------------------------------------------//
    
    // okay now generate galaxies with values for distance with distribution given by nice gaussian and other sky coordinates uniform (or whatever is appropriate for sky area)
    int counter = 0;
    random_device rd;
    default_random_engine eng(10);
    uniform_real_distribution<double> dist_r(distance_min_ideal_survey, distance_max_ideal_survey);
    uniform_real_distribution<double> dist_dec(0.0, 1.0); 
    uniform_real_distribution<double> dist_ra(0.0, 360.0);
    uniform_real_distribution<double> dist_rand(0.0, 1.0);


    while (counter < num_ideal_objects_read_in){

        RA_ideal_survey.push_back( dist_ra(eng) );
        Dec_ideal_survey.push_back( acos(2.0*dist_dec(eng)-1.0)*180.0/M_PI );
        r_ideal_survey.push_back(dist_r(eng));
        counter += 1;
        
    } //end while loop 
    
    //---------------------------------------------------------------------------------------//
    
    // now actually constructing the matrix cov(s_i, s'_j) - cov matrix of ideal data velocities with real data from surveys
    vector< vector<double> > cov_si_sjdash(realspace_galaxy_distances.size(), vector<double>(r_ideal_survey.size())); 

    double kmin = k_vector[0];
    double kmax = k_vector[(k_vector.size() - 1.0)];
    double integral_prefactor = pow(Omega_matter_0, 1.1)*(H0*H0)/(2.0*(M_PI*M_PI));

    // now we can construct the covariance between the real data and the ideal data 
    #pragma omp parallel for 
    for (int i = 0; i < realspace_galaxy_distances.size(); i++){
        for(int jdash = 0; jdash < r_ideal_survey.size(); jdash++){

            // get the distances to the ith and jth galaxies 
            double idist = realspace_galaxy_distances[i];
            double jddist = r_ideal_survey[jdash];

            // get the angle between the ith and jth galaxies in the sky (radians)
            double val = mode_func_dot(0, RA[i], Dec[i], idist)*mode_func_dot(0, RA_ideal_survey[jdash], Dec_ideal_survey[jdash], jddist) + 
            mode_func_dot(1, RA[i], Dec[i], idist)*mode_func_dot(1, RA_ideal_survey[jdash], Dec_ideal_survey[jdash], jddist) 
            + mode_func_dot(2, RA[i], Dec[i], idist)*mode_func_dot(2, RA_ideal_survey[jdash], Dec_ideal_survey[jdash], jddist);
            
            if (val >= 1.0 ) { val = 1.0; }
            if (val <= -1.0) { val = -1.0; }
            
            double alpha = acos( val );

            if (RA[i] == RA_ideal_survey[jdash] && Dec[i] == Dec_ideal_survey[jdash]){ alpha = 0.0; }

            double A = pow(  (pow(idist,2) + pow(jddist,2) - 2.0*idist*jddist*cos(alpha)) , 0.5);

            if (idist == jddist && Dec[i] == Dec_ideal_survey[jdash] && RA[i] == RA_ideal_survey[jdash]) { A = 0.0; }

            if ( round((pow(idist,2) + pow(jddist,2) - 2.0*idist*jddist*cos(alpha))*1000.0)/1000.0 == 0.0  ) { A = 0.0; }

            if ((isnan(A) | isnan(alpha) )) {
              cout << "A: " << A << endl;
              cout << "alpha: " << alpha << endl;
              throw "A or alpha are nan (in compute_Q_qi()).";
              }

            if (abs(A) < 1e-6) { A = 0.0; }

            if (A < 0.0 || A > 2000.0) { 
              throw "A is out of range.";
            }

            if (A == 0){ 
                cov_si_sjdash[i][jdash] = integral_prefactor*(1.0/3.0)*int_over_pk;
                
            
            } else {
                double term1 = integral_prefactor*(1.0/3.0)*cos(alpha)*(gsl_spline_eval(P_mm_k_j0_kA_spline, A, P_mm_k_j0_kA_acc)); 
                double term2 = - integral_prefactor*(1.0/3.0)*cos(alpha)*(2.0*gsl_spline_eval(P_mm_k_j2_kA_spline, A, P_mm_k_j2_kA_acc) ); 
                double term3 = integral_prefactor*(1.0/pow(A,2))*gsl_spline_eval(P_mm_k_j2_kA_spline, A, P_mm_k_j2_kA_acc)*idist*jddist*pow(sin(alpha), 2);

                cov_si_sjdash[i][jdash] = term1 + term2 + term3; 
            
            } 

        } // end loop through ideal survey data 
    } // end loop through real data 
    // end of nested pragma omp parallel for 

    // ok so have the covariance matrix. Now get the matrix Q_qi
    
    vector< vector<double> > Q_qi(num_mode_funcs, vector<double>(realspace_galaxy_distances.size())); 

    //#pragma omp parallel for 
    for (int i = 0; i < realspace_galaxy_distances.size(); i++){ // loop through i objects in real data
        for(int q = 0; q < num_mode_funcs; q++){ // loop through q orthogonal modes 

              Q_qi[q][i] = 0.0; 

            for (int jdash = 0; jdash < r_ideal_survey.size(); jdash++){ // loop through ideal survey gals 

              //Q_qi[q][i] += 3.0*mode_func_dot(q, RA_ideal_survey[jdash], Dec_ideal_survey[jdash], r_ideal_survey[jdash])*cov_si_sjdash[i][jdash]/(r_ideal_survey.size());
              Q_qi[q][i] += mode_func_dot(q, RA_ideal_survey[jdash], Dec_ideal_survey[jdash], r_ideal_survey[jdash])*cov_si_sjdash[i][jdash]/(r_ideal_survey.size());


            } // end loop through ideal survey gals

        } // end q loop
    } // end i loop
    // end of nested pragma omp parallel for 
   
    return Q_qi;

}

/* ----------------------------------------------------------- */ 

vector< vector<double> > block_matrix_inverse(vector< vector<double> > input_matrix){

  cout << "------------------------------------------" << endl;
  cout << "Computing inverse of very large matrix with Blocks" << endl;
  cout << "------------------------------------------" << endl;

  int size_input_total = input_matrix.size();

  int size_A = size_input_total/2;

  if (size_input_total%2 != 0){
    size_A = (size_input_total+1)/2;
  }

  int size_D = size_input_total - size_A;


  // size A x A 
  MatrixXd A_matrix = get_vec_slice(input_matrix, 0, size_A-1, 0, size_A-1);
  MatrixXd A_mat_inv = A_matrix.inverse();
  A_matrix.resize(0,0);
  // size D x D 
  MatrixXd D_matrix = get_vec_slice(input_matrix, size_A, size_input_total-1, size_A, size_input_total-1);

  // size D x A 
  MatrixXd B_matrix = get_vec_slice(input_matrix, 0, size_A-1, // index 1 
  size_A, size_input_total-1); // index 0 
  // size A x D 
  MatrixXd C_matrix = get_vec_slice(input_matrix, size_A, size_input_total-1, // index 1 
  0, size_A-1); // index 0 

  cout << "matrix slices defined" << endl;

  // size D x D (equal to D block of total inverse)
  MatrixXd P_mat = (D_matrix - (C_matrix*A_mat_inv)*B_matrix).inverse();
  D_matrix.resize(0,0);

  cout << "P_matrix / D block of total inverse calculated" << endl;

  // inverse total storage as vector of vectors 
  vector< vector<double> > total_inverse_mat(size_input_total, vector<double>(size_input_total));

  // size A x A 
  MatrixXd BlockA_totalinv = A_mat_inv + A_mat_inv*((B_matrix*P_mat)*(C_matrix*A_mat_inv));
  #pragma omp parallel for 
  for (int i = 0; i < size_A; i++){
    for (int j = 0; j < size_A; j++){
        total_inverse_mat[i][j] = BlockA_totalinv(i,j);
    }
  }
  // end of nested pragma omp parallel for 

  BlockA_totalinv.resize(0,0);
  cout << "inverse Block A stored" << endl;

  #pragma omp parallel for 
  for (int i = 0; i < size_D; i++){
    for (int j = 0; j < size_D; j++){
        total_inverse_mat[(i+size_A)][(j+size_A)] = P_mat(i,j);
    }
  }
  // end of nested pragma omp parallel for 

  cout << "inverse Block D stored" << endl;

  // size D x A 
  MatrixXd BlockB_totalinv = -1*A_mat_inv*(B_matrix*P_mat);
  B_matrix.resize(0,0);
  #pragma omp parallel for 
  for (int i = size_A; i < (size_D+size_A); i++){
    for (int j = 0; j < size_A; j++){
        total_inverse_mat[i][j] = BlockB_totalinv(j,i-size_A);
    }
  }
  // end of nested pragma omp parallel for 

  BlockB_totalinv.resize(0,0);
  cout << "inverse Block B stored" << endl;


  // size A x D 
  MatrixXd BlockC_totalinv = -1*(P_mat*C_matrix)*A_mat_inv;
  P_mat.resize(0,0);
  C_matrix.resize(0,0);
  A_mat_inv.resize(0,0);
  #pragma omp parallel for 
  for (int i = 0; i < size_A; i++){
    for (int j = size_A; j < (size_D+size_A); j++){
        total_inverse_mat[i][j] = BlockC_totalinv(j-size_A,i);
    }
  }
  // end of nested pragma omp parallel for 

  cout << "inverse Block C stored" << endl;
  BlockC_totalinv.resize(0,0);
  cout << "Finished inverting matrices using Block method." << endl; 
  cout << "------------------------------------------" << endl;

  return total_inverse_mat;

}

/* ----------------------------------------------------------- */ 


MatrixXd get_vec_slice(vector< vector<double> > input_matrix,
int start_0, int end_0, int start_1, int end_1){

    MatrixXd returnslicemat(end_0+1-start_0, end_1+1-start_1);

    #pragma omp parallel for 
    for (int i = start_0; i < end_0+1; i++){
        for (int j = start_1; j < end_1+1; j++){
            returnslicemat(i-start_0,j-start_1) = input_matrix[i][j];
        }
    }
    // end of nested pragma omp parallel for 

    return returnslicemat;

}

/* ----------------------------------------------------------- */ 


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
