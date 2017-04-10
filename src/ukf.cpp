#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

	n_x_ = 5;
	n_aug_ = n_x_ + 2; // 7
	n_sig_ = 2 * n_aug_ + 1; // 15
	lambda_ = 3 - n_x_;

	is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

	// initial sigma point matrix
	Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);

	// initial time
	time_us_ = 0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

	// initial weights vector
	weights_ = VectorXd(11);

	// noise covariance matrix
	Q_ = MatrixXd(2, 2);
	Q_ << std_a_*std_a_, 0,
				0, std_yawdd_*std_yawdd_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

	long long timestamp = meas_package.timestamp_;
	VectorXd z = meas_package.raw_measurements_;

	double px, py;

	if (!is_initialized_) {

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			double rho, phi;
			rho = z(0);
			phi = z(1);
			px = rho * cos(phi);
			py = rho * sin(phi);
		} else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
			px = z(0);
			py = z(1);
		} else {
			cout << "Warning: Incorrect sensor type" << endl;
			return;
		}

		// initial state
		x_ << px, py, 0, 0, 0;

		// initial covariance matrix
		P_.fill(0.0);

		// define weights to calculate predicted state mean
		weights_.fill(0.5/(lambda_ + n_aug_));
		weights_(0) = lambda_/(lambda_+n_aug_);

		// state is now initialized
		is_initialized_ = true;
		return;
	}

	// timestep in us
	double delta_t = (timestamp - time_us_) / 1E6;
	time_us_ = timestamp;

	Prediction(delta_t);

  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */


}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
	double nu_a {0}; //TODO: Value
	double nu_psi_dd {0}; //TODO: Value

	// augmented state x
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.head(n_x_) = x_;
	x_aug(5) = nu_a;
	x_aug(6) = nu_psi_dd;

	// augmented P
	MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug.bottomRightCorner(2, 2) = Q_;

	// square root of P_aug
	MatrixXd A = P_aug.llt().matrixL();

	// augmented Sigma points
	MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
	double k = sqrt(lambda_ + n_aug_);
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug_ ; i++) {
		VectorXd sig_col = k*A.col(i);
		Xsig_aug.col(i+1) = x_aug + sig_col;
		Xsig_aug.col(i+1+n_aug_) = x_aug - sig_col;
	}

	// predict sigma points and state
	VectorXd x_pred = VectorXd::Zero(n_x_);
	for (int i = 0; i < n_sig_; i++) {
		VectorXd sigma_point = Process(Xsig_aug.col(i), delta_t);
		Xsig_pred_.col(i) = sigma_point;
		x_pred += weights_(i)*sigma_point;
	}

	// predict covariance matrix
	VectorXd x_diff = VectorXd(n_x_);
	MatrixXd P_pred = MatrixXd::Zero(n_x_, n_x_);
	for (int i = 0; n_sig_; i++) {
		x_diff = Xsig_pred_.col(i) - x_pred;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
		P_pred += weights_(i)*x_diff*x_diff.transpose();
	}

	// update date
	x_ = x_pred;

	// update covariance matrix
	P_ = P_pred;
}

/**
 * @param x_k_aug augmented state
 * @param delta_t
 * @return predicted state
 */
VectorXd UKF::Process(const VectorXd &x_k_aug, double delta_t) {
	double px = x_k_aug(0);
	double py = x_k_aug(1);
	double v = x_k_aug(2);
	double psi = x_k_aug(3);
	double psi_d = x_k_aug(4);
	double nu_a = x_k_aug(5);
	double nu_psi_dd = x_k_aug(6);
	double delta_px, delta_py;

	VectorXd x_noise = VectorXd(n_x_); // predicted state
	VectorXd x_k1 = VectorXd(n_x_); // predicted state
	VectorXd xd_dt = VectorXd(n_x_); // integral x_dot for delta_t

	if (abs(psi_d) < 0.001) {
		delta_px = v*cos(psi)*delta_t;
		delta_py = v*sin(psi)*delta_t;
	}
	else {
		delta_px = v/psi_d*(sin(psi+psi_d*delta_t)-sin(psi));
		delta_py = v/psi_d*(cos(psi)-cos(psi+psi_d*delta_t));
	}

	xd_dt << delta_px, delta_py, 0, psi_d*delta_t, 0;

	// noise component, assuming straight line driving
	x_noise<< 0.5*delta_t*delta_t*cos(psi)*nu_a,
						0.5*delta_t*delta_t*sin(psi)*nu_a,
						delta_t*nu_a,
						0.5*delta_t*delta_t*nu_psi_dd,
						delta_t*nu_psi_dd;

	VectorXd x_k {px, py, v, psi, psi_d};
	return x_k + xd_dt + x_noise;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
