#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

	long nb_est = estimations.size();
	assert(nb_est != 0);
	assert(nb_est == ground_truth.size());

	long nb_dim = estimations[0].size();
	assert(nb_dim != 0);

	VectorXd error = VectorXd::Zero(nb_dim);
	VectorXd residual = VectorXd::Zero(nb_dim);

	for (int i = 0; i < nb_est; i++) {
		residual = estimations[i] - ground_truth[i];
		residual = residual.array() * residual.array();
		error += residual;
	}

	VectorXd rmse = sqrt(error/nb_est);

	return rmse;
}
