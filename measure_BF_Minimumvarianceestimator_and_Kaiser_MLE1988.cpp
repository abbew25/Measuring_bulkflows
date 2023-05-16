
// computes the BF using the MVE of Watkins 2009 first, and then using MLE of Kaiser 1988 second


// To compile: 
// g++ -lgsl -lgslcblas -fopenmp -lm -llapack -lblas measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.cpp -o measure_BF_Minimumvarianceestimator_and_Kaiser_MLE1988.exe
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

using namespace std;
using namespace std::chrono;


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////


// declare global variables and functions - full function definitions can be found after main()
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

// creating a structure for the parameters of integration for the integration function to get P(k)j_0(kA) and func for P(k)j_2(kA)
struct params_int2
{
  double A;
};

double integrand_pk_j0_kA(double k, void * params_int_from_above); // function to get integrand of P(k)j_0(kA)
double integrand_pk_j2_kA(double k, void * params_int_from_above); // function to get integrand of P(k)j_2(kA)
void get_real_galaxy_distances(); // function to get distances to each galaxy in real space 
vector< vector<double> > compute_R_ij();
vector< vector<double> > compute_matrix_inverse( vector< vector<double> > input_matrix);

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

void precompute_stuff();

void print_2D_matrix(vector< vector<double> > matrix_2_print, string name);
vector< vector<double> > compute_M_pq(vector< vector<double> > G_ij_inverse_matrix);
double number_objects_ideal_distribution(double r, double A);

vector< vector<double> > compute_Q_qi(); // get the covariance of the real data velocities with an simulated ideal survey 

vector< vector<double> > compute_lagrange_multiplier(vector< vector<double> > inv_M_ql, 
vector< vector<double> > G_mn_inv, vector< vector<double> > Q_pm);

// finally get velocity weights 
vector< vector<double> > compute_weights_for_velocity_components(vector< vector<double> > G_mn_inv, 
vector< vector<double> > Q_lm, vector< vector<double> > lagrange_multiplier_matrix); 

// function to check condition we used to calculate langrange multiplier is correct
vector< vector<double> > check_results_condition_met(vector< vector<double> > matrix_weights_pn); 

// function to calculate < u_i, u_j > where u_i is the ith moment of the velocity field
vector< vector<double> > calculate_covariance_velocity_moments(vector< vector<double> > matrix_weights_pn, vector< vector<double> > matrix_G_mn); 

double number_objects_ideal_distribution(double r, double A); // distribution of galaxies in an ideal survey 
// cosmological parameters

vector< vector<double> > compute_MLE_weights(); // compute the weights for the objects with MLE method instead of MVE

double mode_func_dot(double modefunc_index,
double ra_angle, double dec_angle, double r_dist); // function to get doc product between object position vectors and mode functions 

double integrand_pk_j2_klimAgoesto0(double k, void * params_int_from_above); // function to get integral over k of 
// P(k)j)2(kA)/A^2 in the limit A goes to zero, which becomes int of k of P(k)k^2/15

double integrand_pk(double k, void * params_int_from_above); // integrate over the power spectrum 

void write_matrix_2_file(vector< vector<double> > matrix_2_write, string filepathname);

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

double H0 = 67.32; // km/s/MPC (Hubble constant).
double Omega_matter_0 = 0.30; // dimensionless matter density 
double Omega_Lambda_0 = 1 - Omega_matter_0; // cosmological constant 
double eos_matter = 0.0; // equation of state for matter
double eos_lambda = -1.0; // Eq of state for dark energy
double light_speed = 299792.458; // speed of light in km/s
double sigma_star = 300.0; // ( km/s ) - this is the the 1D velocity dispersion 
int num_mode_funcs = 3; // use 3 for just BF, 9 for BF + shear, 19 for BF + shear + octupoles
int num_ideal_objects_read_in = 10000; // objects to read in from a data file with an 'ideal' survey selection 
double R_I = 70.0; // Mpc, ideal survey gaussian width / standard deviation 
double distance_max_ideal_survey = 500.0; // mpc 
double distance_min_ideal_survey = 0.0; // mpc 
double C = 2.0*R_I*R_I; // just a constant to use later 

// vectors where we will save data from in file (mock data)
vector<double> RA,Dec,z_obs,logdist,logdist_err; 

// get data from file with power spectrum computed with CLASS
vector<double> k_vector, Pk_vector;

// spline for integration for power spectrum 
gsl_spline *P_mm_k_spline;
gsl_interp_accel * P_mm_k_acc;

// spline for A with integral result of int j_0(kA) P(k) dk and for A with result of int j_2(kA) P(k) dk
gsl_spline *P_mm_k_j0_kA_spline, *P_mm_k_j2_kA_spline;
gsl_interp_accel * P_mm_k_j0_kA_acc, *P_mm_k_j2_kA_acc;

double int_over_k_Pk_ksquared_over_15, int_over_pk; 

double Pkdk_integral = 0; // result of integral P(k) dk from kmin to kmax 

// vectors we will compute ourselves
vector<double> object_pvs, object_pv_errs, realspace_galaxy_distances; // a vector with the peculiar velocities, 
//a vector with the peculiar velocity uncertainties


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char **argv) {

    auto start = high_resolution_clock::now(); // starting timer for script execution

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

    // compute the matrix G_ij of the galaxies (G_ij (aka R_ij) = peculiar velocity data covariance matrix)
    vector< vector<double> > G_ij = compute_R_ij(); // getting velocity covariance part 

    vector< vector<double> > G_ij_inverse = compute_matrix_inverse(G_ij);

    cout << "compute G_ij done" << endl;

    // now construct the matrix M_pq (only works right now for 3 modes for BF only)
    vector< vector<double> > M_pq = compute_M_pq(G_ij_inverse);

    cout << "compute M_pq done" << endl;

    vector< vector<double> > inv_M_pq = compute_matrix_inverse(M_pq); // computing the inverse of this matrix
  
    // now construct the matrix Q_qi (size = num_mode_funcs*num_galaxies in real survey (size mock_data))
    vector< vector<double> > Q_qi_mat = compute_Q_qi();

    cout << "compute Q_qi done" << endl; 

    // ok, now construct the lagrange multiplier 
    vector< vector<double> > lambda_pq = compute_lagrange_multiplier(inv_M_pq, G_ij_inverse, Q_qi_mat);

    cout << "compute lambda done" << endl;

    // now we can finally calculate the weights 
    vector< vector<double> > weights_pn = compute_weights_for_velocity_components(G_ij_inverse, Q_qi_mat, lambda_pq);

    cout << "compute MVE weights done" << endl;

    double BF_x = 0; // treating q = 0 as the x direction
    double BF_y = 0; // treating q = 1 as the y direction
    double BF_z = 0; // treating q = 2 as the z direction

    for (int i = 0; i < z_obs.size(); i++){
        BF_x += weights_pn[0][i]*object_pvs[i];
        BF_y += weights_pn[1][i]*object_pvs[i];
        BF_z += weights_pn[2][i]*object_pvs[i];
    }

    // now calculate the error bars for the BF components
    vector< vector<double> > cov_vel_moments_ab = calculate_covariance_velocity_moments(weights_pn, G_ij);
    
    // NEED TO COMMENT THIS BACK IN to write results to a file ------------------------------
    
    
    // write the results to a file 
    ofstream results_file;
    results_file.open(("BF_MVE_result.txt" ));
    results_file << BF_x << " " << BF_y << " " << BF_z << endl;
    results_file << " " << sqrt(cov_vel_moments_ab[0][0]) << " " << sqrt(cov_vel_moments_ab[1][1]) << " " << sqrt(cov_vel_moments_ab[2][2]) << endl; 
    results_file.close();

    // write the covariance matrix to a file 
    ofstream cov_file;
    cov_file.open(("BF_MVE_cov.txt" ));
    for (int i = 0; i < num_mode_funcs; i++){
      for (int j = 0; j < num_mode_funcs; j++){
        cov_file << cov_vel_moments_ab[i][j] <<  " ";
      }
      cov_file << endl;
    }
    cov_file.close();

    // calculate the MLE weights 

    vector< vector<double> > MLE_weights_pn = compute_MLE_weights();

    // calculate the MLE covariance matrix

    vector< vector<double> > cov_matrix_MLE = calculate_covariance_velocity_moments(MLE_weights_pn, G_ij);

    // calculate MLE bulk flow 

    double BF_x_mle = 0; // treating q = 0 as the x direction
    double BF_y_mle = 0; // treating q = 1 as the y direction
    double BF_z_mle = 0; // treating q = 2 as the z direction

    for (int i = 0; i < z_obs.size(); i++){
        BF_x_mle += MLE_weights_pn[0][i]*object_pvs[i];
        BF_y_mle += MLE_weights_pn[1][i]*object_pvs[i];
        BF_z_mle += MLE_weights_pn[2][i]*object_pvs[i];
    }

    // write the results to a file 
    ofstream results_file2;
    results_file2.open(("BF_MLE_result.txt" ));
    results_file2 << " " << BF_x_mle << " " << BF_y_mle << " " << BF_z_mle << endl; 
    results_file2 << " " << sqrt(cov_matrix_MLE[0][0]) << " " << sqrt(cov_matrix_MLE[1][1]) << " " << sqrt(cov_matrix_MLE[2][2]) << endl; 
    results_file2.close();

    // write the MLE covariance matrix to a file 
    ofstream cov_file2;
    cov_file2.open(("BF_MLE_cov.txt" ));
    for (int i = 0; i < num_mode_funcs; i++){
      for (int j = 0; j < num_mode_funcs; j++){

        cov_file2 << cov_matrix_MLE[i][j] <<  " ";

      }
      cov_file2 << endl;
    }
    cov_file2.close();
    

    // -----------------------------------------------------------

    // -----------------------------------------------------------

    // finally do:
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
  while ( getline(mockdata,line))
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
        Dec.push_back(stod(splitline[1])); // declination (measured)
        z_obs.push_back(stod(splitline[2])); // observed redshift in sim
        logdist.push_back(stod(splitline[3])); // observed log distance ratio
        logdist_err.push_back(stod(splitline[4])); // uncertainty on log distance ratio (as determined from observations)

    
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
        k_vector.push_back( stod(substr) );
      }


    } else if (numoflines == 1) {

      stringstream ss(line);

      while( ss.good() )
      {
        string substr;
        getline( ss, substr, ' ' );
        Pk_vector.push_back( stod(substr) );
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

  double z_mod = z_observation*( 1.0 + 0.5*(1.0 - q0)*z_observation - (1.0/6.0)*(j0 - q0 - 3.0*(pow(q0, 2)) + 1.0 )*(pow(z_observation, 2)) );
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
  gsl_integration_qags(&F, 0.0, zed, 1e-8, 1e-8, 1000, w, &result, &error);
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

double integrand_pk_j2_klimAgoesto0(double k, void * params_int_from_above){

  double pk_val = gsl_spline_eval(P_mm_k_spline, k, P_mm_k_acc);
  double res = pk_val*(k*k)/15.0;

  return res;
}

/* ----------------------------------------------------------- */ 

double integrand_pk(double k, void * params_int_from_above){

  double pk_val = gsl_spline_eval(P_mm_k_spline, k, P_mm_k_acc);

  return pk_val;
}

/* ----------------------------------------------------------- */ 


void get_real_galaxy_distances(){

  // get the actual distance to the galaxies (r) - in real space and save the result
    for (int zval = 0; zval < z_obs.size(); zval++){

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
    double A_max = 2000.0; // Mpcs (corresponds to roughly an object at z = 0.5 which should be more than far enough for our survey)
    int bins = 10000;
    double kmin = k_vector[0];
    double kmax = k_vector[(k_vector.size() - 1.0)];
    double A_vector[bins], integral_j0_pk[bins], integral_j2_pk[bins];
    // compute the integrals for a range of A values, store results in vectors, then create a spline object to look at later 
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

      gsl_integration_qags(&F0, kmin, kmax, 1e-9, 1e-9, 1000, w0, &result0, &error0);
      //cout << "j0" << endl;
      gsl_integration_workspace_free(w0);
      gsl_integration_qags(&F2, kmin, kmax, 1e-9, 1e-9, 1000, w2, &result2, &error2);
      //cout << "j2" << endl; 
      gsl_integration_workspace_free(w2);

      integral_j0_pk[A_index] = result0;
      integral_j2_pk[A_index] = result2;


    }
    // create the spline objects 
    P_mm_k_j0_kA_acc = gsl_interp_accel_alloc(); // steps to create the gsl spline object for power spectrum integral (with j_0(kA))
    P_mm_k_j2_kA_acc = gsl_interp_accel_alloc(); // like above line ^ 
    P_mm_k_j0_kA_spline = gsl_spline_alloc(gsl_interp_cspline, bins);
    P_mm_k_j2_kA_spline = gsl_spline_alloc(gsl_interp_cspline, bins);
    gsl_spline_init(P_mm_k_j0_kA_spline, A_vector, integral_j0_pk, bins);
    gsl_spline_init(P_mm_k_j2_kA_spline, A_vector, integral_j2_pk, bins);

    // do the integral for P(k) J_2(kA)/A^2 dk for lim A -> 0
    // take these and set the values in the struct params_int2 for the integral, do the integration 
    params_int2 custom_params_int3; // our custom struct
    double result3, error3; // integration result and error 

    gsl_function F3;
    F3.function = &integrand_pk_j2_klimAgoesto0;

    F3.params = &custom_params_int3;
    gsl_integration_workspace * w3 = gsl_integration_workspace_alloc(1000);

    gsl_integration_qags(&F3, kmin, kmax, 1e-8, 1e-8, 1000, w3, &result3, &error3);
    gsl_integration_workspace_free(w3);
  
    int_over_k_Pk_ksquared_over_15 = result3;


    params_int2 custom_params_int4; // our custom struct
    double result4, error4; // integration result and error 

    gsl_function F4;
    F4.function = &integrand_pk;

    F4.params = &custom_params_int4;
    gsl_integration_workspace * w4 = gsl_integration_workspace_alloc(1000);

    gsl_integration_qags(&F4, kmin, kmax, 1e-8, 1e-8, 1000, w4, &result4, &error4);
    gsl_integration_workspace_free(w4);
  
    int_over_pk = result4;

}

/* ----------------------------------------------------------- */ 

vector< vector<double> > compute_R_ij(){

    vector< vector<double> > R_ij_matrix(z_obs.size(), vector<double>(z_obs.size()));

    // R_ij_eps
    for (int i = 0; i < z_obs.size(); i ++){

        for (int j = 0; j < z_obs.size(); j ++){
        
          if (i == j){
              R_ij_matrix[i][i] = ( pow(object_pv_errs[i],2.0)  + pow(sigma_star, 2.0) );
          } else {
              R_ij_matrix[i][j] = 0.0; 
          }

        }
    }

    // we will use threading to split up the matrix elements between threads to speed up the task of building this matrix
    for (int i = 0; i < z_obs.size(); i++){
        double kmin = k_vector[0];
        double kmax = k_vector[(k_vector.size() - 1.0)];
        double integral_prefactor = pow(Omega_matter_0, 1.1)*(H0*H0)/(2.0*(M_PI*M_PI));

        for(int j = i; j < z_obs.size(); j++){ // since the matrix is symmetric we only need to compute half of the elements really 

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

            if (i == j){alpha = 0.0;}

            if (RA[i] == RA[j] && Dec[i] == Dec[j]) { alpha = 0; }

            double A = pow(  (pow(idist,2) + pow(jdist,2) - 2.0*idist*jdist*cos(alpha)) , 0.5);

            if (i == j) { A = 0.0; }

            if ( round((pow(idist,2) + pow(jdist,2) - 2.0*idist*jdist*cos(alpha))*1000.0)/1000.0 == 0.0  ) { A = 0.0; }

            if (idist == jdist && alpha == 0.0){ A = 0.0; }

            if ((isnan(A) | isnan(alpha) )) {
              cout << "idist: " << idist << endl; 
              cout << "jdist: " << jdist << endl; 
              cout << "A: " << A << endl;
              cout << "alpha: " << alpha << endl;
              cout << val << endl;
              cout << "A or alpha are nan (in compute_R_ij())." << endl;
              throw "A or alpha are nan (in compute_R_ij()).";
              }

          if (abs(A) < 1e-6) { A = 0.0; }

            if (A == 0){ 
                R_ij_matrix[i][j] += integral_prefactor*(1.0/3.0)*int_over_pk; 
                
            } else {

                
                double term1 = integral_prefactor*(1.0/3.0)*cos(alpha)*(gsl_spline_eval(P_mm_k_j0_kA_spline, A, P_mm_k_j0_kA_acc));
                double term2 = - integral_prefactor*(1.0/3.0)*cos(alpha)*( 2.0*gsl_spline_eval(P_mm_k_j2_kA_spline, A, P_mm_k_j2_kA_acc) );
                double term3 = integral_prefactor*(1.0/pow(A,2))*gsl_spline_eval(P_mm_k_j2_kA_spline, A, P_mm_k_j2_kA_acc)*idist*jdist*pow(sin(alpha), 2);
            
                R_ij_matrix[i][j] += term1 + term2 + term3; 

            } 

            if (i != j) { R_ij_matrix[j][i] = R_ij_matrix[i][j]; }

        } // end j loop
    } // end i loop 

    return R_ij_matrix;

}


/* ----------------------------------------------------------- */ 


vector< vector<double> > compute_matrix_inverse( vector< vector<double> > input_matrix){

    // put the matrix into a 1D vector and get inverse from LAPACK

    vector<double> input_copy_for_inverse(pow(input_matrix.size(),2));
    
    for (int i = 0; i < input_matrix.size(); i++){
      for (int j = 0; j < input_matrix.size(); j++){

        input_copy_for_inverse[j + i*(input_matrix.size())] = input_matrix[i][j];

      } // loop i end 
    } // loop j end 
    
    inverse(input_copy_for_inverse.data(), input_matrix.size());

    vector< vector<double> > return_inverse(input_matrix.size(), vector<double>(input_matrix.size()));

    for (int i = 0; i < input_matrix.size(); i++){
      for (int j = 0; j < input_matrix.size(); j++){

        return_inverse[i][j] = input_copy_for_inverse[j + i*(input_matrix.size())];

      } // loop i end 
    } // loop j end 


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

vector< vector<double> > compute_M_pq(vector< vector<double> > G_ij_inverse_matrix){

    vector< vector<double> > M_pq_matrix( num_mode_funcs, vector<double>(num_mode_funcs) );

    for (int p = 0; p < num_mode_funcs; p++){ // looping through pth mode function
        for (int q = 0; q < num_mode_funcs; q++){ //looping through qth mode function

        M_pq_matrix[p][q] = 0.0;

        // work out dot products for mode functions
          for(int m = 0; m < z_obs.size(); m++){ // looping through mth galaxy in the inverse covariance matrix
            for(int n = 0; n < z_obs.size(); n++){ // looping through nth galaxy in the inverse covariance matrix

              M_pq_matrix[p][q] += 0.5*G_ij_inverse_matrix[m][n]*mode_func_dot(p,RA[m],Dec[m],realspace_galaxy_distances[m])*mode_func_dot(q,RA[n],Dec[n],realspace_galaxy_distances[n]);
              
            } // nth galaxy end 
          } // mth galaxy end 

        } // qth loop end
      } // pth loop end

    return M_pq_matrix;
}

/* ----------------------------------------------------------- */ 

// probability density function of (r) for gaussian radial distribution of objects 
double number_objects_ideal_distribution(double r, double A){

    double nr = A*pow(r,2.0)*exp( -(pow(r,2))/(2.0*pow(R_I,2)));
    return nr;

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
    
    double norm = 0.0; // normalisation of integral over density function 

    double integral_res = (  M_PI*C*(  pow(M_PI, 0.5)*pow(C,0.5) ) ); 

    // to normalize the probability density function to integrate up to 1 
    norm = num_ideal_objects_read_in/integral_res;

    // okay now generate galaxies with values for distance with distribution given by nice gaussian and other sky coordinates uniform (or whatever is appropriate for sky area)
    int counter = 0;
    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<double> dist_r(distance_min_ideal_survey, distance_max_ideal_survey);
    uniform_real_distribution<double> dist_dec(0.0, 1.0); 
    uniform_real_distribution<double> dist_ra(0.0, 360.0);
    uniform_real_distribution<double> dist_rand(0.0, 1.0);

    while (counter < num_ideal_objects_read_in){

        // randomly select a value for r
        double sample_r_value = dist_r(eng);

        // calculate the probability of drawing this sample with gaussian prob func
        double P_r = number_objects_ideal_distribution(sample_r_value, norm)/num_ideal_objects_read_in;
        
        // draw a random number from a uniform distribution
        double random_num = dist_rand(eng);
        
        // compare the probability of drawing the sample vs the number draw from the uniform dist
        if (random_num < P_r){
            counter += 1;
            
            RA_ideal_survey.push_back( dist_ra(eng) );
            Dec_ideal_survey.push_back( acos(2.0*dist_dec(eng)-1.0)*180.0/M_PI );
            r_ideal_survey.push_back(sample_r_value);

            
        } 
    } //end while loop 
    
    //---------------------------------------------------------------------------------------//

    // now actually constructing the matrix cov(s_i, s'_j) - cov matrix of ideal data velocities with real data from surveys
    vector< vector<double> > cov_si_sjdash(realspace_galaxy_distances.size(), vector<double>(r_ideal_survey.size())); 

    double kmin = k_vector[0];
    double kmax = k_vector[(k_vector.size() - 1.0)];
    double integral_prefactor = pow(Omega_matter_0, 1.1)*(H0*H0)/(2.0*(M_PI*M_PI));

    // now we can construct the covariance between the real data and the ideal data 
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

    // ok so have the covariance matrix. Now get the matrix Q_qi
    
    vector< vector<double> > Q_qi(num_mode_funcs, vector<double>(realspace_galaxy_distances.size())); 
    for (int i = 0; i < realspace_galaxy_distances.size(); i++){ // loop through i objects in real data
        for(int q = 0; q < num_mode_funcs; q++){ // loop through q orthogonal modes 

              Q_qi[q][i] = 0.0; 

            for (int jdash = 0; jdash < r_ideal_survey.size(); jdash++){ // loop through ideal survey gals 

              Q_qi[q][i] += 3.0*mode_func_dot(q, RA_ideal_survey[jdash], Dec_ideal_survey[jdash], r_ideal_survey[jdash])*cov_si_sjdash[i][jdash]/(r_ideal_survey.size());

            } // end loop through ideal survey gals

        } // end q loop
    } // end i loop
    
    return Q_qi;

}

/* ----------------------------------------------------------- */ 

vector< vector<double> > compute_lagrange_multiplier(vector< vector<double> > inv_M_ql, 
vector< vector<double> > G_mn_inv, vector< vector<double> > Q_pm){

  vector< vector<double> > lambda_pq_matrix(num_mode_funcs, vector<double>(num_mode_funcs));

  for (int p = 0; p < num_mode_funcs; p++){
    for (int q = 0; q < num_mode_funcs; q++){

      lambda_pq_matrix[p][q] = 0.0; 

      for (int l = 0; l < num_mode_funcs; l++){

        double delta_pl = 0;
        if (l == p) { delta_pl = 1; }

        double sum_mn = 0;

        for(int m = 0; m < z_obs.size(); m++){ //looping through galaxies
          for (int n = 0; n < z_obs.size(); n++){ //looping through galaxies

              double g_ln_mode = mode_func_dot(l, RA[n], Dec[n], realspace_galaxy_distances[n]); 

            sum_mn += G_mn_inv[m][n]*Q_pm[p][m]*( g_ln_mode );

          } // end of n loop 
        } // end of m loop

        lambda_pq_matrix[p][q] += inv_M_ql[q][l]*( sum_mn - delta_pl  );

      } // end of l loop 
    } // end of q loop
  } // end of p loop

  return lambda_pq_matrix;

}

/* ----------------------------------------------------------- */ 

vector< vector<double> > compute_weights_for_velocity_components(vector< vector<double> > G_mn_inv, 
vector< vector<double> > Q_pm, vector< vector<double> > lagrange_multiplier_matrix){

    vector< vector<double> > weights_pn(num_mode_funcs, vector<double>(z_obs.size()));

    for(int p = 0; p < num_mode_funcs; p++){
      for(int n = 0; n < z_obs.size(); n++){

        weights_pn[p][n] = 0.0; 

        for(int m = 0; m < z_obs.size(); m++){

          double sum_q = 0;

          for(int q = 0; q < num_mode_funcs; q++){

            sum_q += lagrange_multiplier_matrix[p][q]*mode_func_dot(q, RA[m], Dec[m], realspace_galaxy_distances[m]);

          } // loop over q 

          weights_pn[p][n] += G_mn_inv[m][n]*( Q_pm[p][m] - 0.5*(sum_q) );

        } // loop over m
      } // loop over n
    } // loop over p

    return weights_pn;

}

/* ----------------------------------------------------------- */ 


vector< vector<double> > check_results_condition_met(vector< vector<double> > matrix_weights_pn ){
    
    vector< vector<double> > matrix(num_mode_funcs, vector<double>(num_mode_funcs));
    
    for (int p = 0; p < num_mode_funcs; p++){
        for (int q = 0; q < num_mode_funcs; q++){

          matrix[p][q] = 0.0; 
            
            for (int n = 0; n < z_obs.size(); n++){
                    
              matrix[p][q] += matrix_weights_pn[p][n]*mode_func_dot(q, RA[n], Dec[n], realspace_galaxy_distances[n]);

            } // loop through n
            
        } // loop through q
    } // loop through p
    
    return matrix;
    
}

/* ----------------------------------------------------------- */

vector< vector<double> > calculate_covariance_velocity_moments(vector< vector<double> > matrix_weights_pn, vector< vector<double> > matrix_G_mn){
    
    vector< vector<double> > covariance_matrix_moments_ab(num_mode_funcs, vector<double>(num_mode_funcs));
    
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
    
    
    return covariance_matrix_moments_ab;
}

/* ----------------------------------------------------------- */

double mode_func_dot(double modefunc_index, double ra_angle, double dec_angle, double r_dist){

  double res = 0;

  double xhat = cos(ra_angle*M_PI/180.0)*sin((dec_angle+90.0)*M_PI/180.0);
  double yhat = sin(ra_angle*M_PI/180.0)*sin((dec_angle+90.0)*M_PI/180.0);
  double zhat = cos((dec_angle+ 90.0)*M_PI/180.0);

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

vector< vector<double> > compute_MLE_weights(){

  vector< vector<double> > A_pj_matrix(num_mode_funcs, vector<double>(num_mode_funcs));

  // calculate elements of A_pj ----------
  for (int p = 0; p < num_mode_funcs; p++){
    for (int j = 0; j < num_mode_funcs; j++){

      A_pj_matrix[p][j] = 0.0;

      for (int n = 0; n < z_obs.size(); n++){

      A_pj_matrix[p][j] += mode_func_dot(p, RA[n], Dec[n], realspace_galaxy_distances[n])*mode_func_dot(j, RA[n], Dec[n], realspace_galaxy_distances[n])/( pow(object_pv_errs[n],2) + pow( sigma_star ,2) );

      } // loop over galaxies 

    } // loop over p
  } // loop over j

  // invert A_pj -------------------------

  vector< vector<double> > A_pj_matrix_inv = compute_matrix_inverse(A_pj_matrix);

  vector< vector<double> > weights_pn(num_mode_funcs, vector<double>(z_obs.size()));

  for(int p = 0; p < num_mode_funcs; p++){
    for (int n = 0; n < z_obs.size(); n++){

      weights_pn[p][n] = 0.0; 

      for (int j = 0; j < num_mode_funcs; j++){

      weights_pn[p][n] += A_pj_matrix_inv[p][j]*mode_func_dot(j, RA[n], Dec[n], realspace_galaxy_distances[n])/( pow(object_pv_errs[n],2) + pow( sigma_star ,2) );

      } // loop over j

    } // loop over galaxies 
  } // loop over p

    return weights_pn;

}

/* ----------------------------------------------------------- */

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
