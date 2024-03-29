﻿#include "WatermarkFunctions.h"
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <vector>
#include <memory>
#include <Eigen/Dense>

using std::cout;

//constructor to initialize all the necessary data
WatermarkFunctions::WatermarkFunctions(const Eigen::ArrayXXf& image, const std::string w_file_path, const int p, const float psnr, const int num_threads) 
	:image(image), p(p), pad(pad), rows(image.rows()), cols(image.cols()), padded_rows(rows + 2 * pad), padded_cols(cols + 2 * pad), elems(rows* cols),
	w(load_W(w_file_path, image.rows(), image.cols())), p_squared(static_cast<int>(std::pow(p, 2))), p_squared_minus_one_div_2((p_squared - 1) / 2), psnr(psnr), num_threads(num_threads)  {
}

//helper method to load the random noise matrix W from the file specified.
Eigen::ArrayXXf WatermarkFunctions::load_W(const std::string w_file, const Eigen::Index rows, const Eigen::Index cols) {
	std::ifstream w_stream(w_file.c_str(), std::ios::binary);
	if (!w_stream.is_open()) {
		std::string error_str("Error opening '" + w_file + "' file for Random noise W array");
		throw std::exception(error_str.c_str());
	}
	w_stream.seekg(0, std::ios::end);
	const auto total_bytes = w_stream.tellg();
	w_stream.seekg(0, std::ios::beg);
	if (rows * cols * sizeof(float) != total_bytes) {
		std::string error_str("Error: W file total elements != image dimensions! W file total elements: " + std::to_string(total_bytes / (sizeof(float))) + std::string(", Image width: ") + std::to_string(cols) + std::string(", Image height: ") + std::to_string(rows));
		throw std::exception(error_str.c_str());
	}
	std::unique_ptr<float> w_ptr(new float[rows * cols]);
	w_stream.read(reinterpret_cast<char*>(&w_ptr.get()[0]), total_bytes);
	return Eigen::Map<Eigen::ArrayXXf>(w_ptr.get(), cols, rows).transpose().eval();
}

//generate p x p neighbors
Eigen::VectorXf WatermarkFunctions::create_neighbors(const Eigen::ArrayXXf& padded_image, const int i, const int j, const int p, const int p_squared)
{
	const int neighbor_size = (p - 1) / 2;
	//x_: will contain all the neighbors minus the current pixel value
	Eigen::VectorXf x_(p_squared - 1);
	const int i0 = i - neighbor_size;
	const int j0 = j - neighbor_size;
	const int i1 = i + neighbor_size;
	const int j1 = j + neighbor_size;
	const Eigen::ArrayXXf x_temp = padded_image.block(i0, j0, i1 - i0 + 1, j1 - j0 + 1);
	//ignore the central pixel value
	std::memcpy(x_.data(), x_temp.data(), sizeof(float) * p_squared_minus_one_div_2);
	std::memcpy(x_.data() + 4, x_temp.data() + 5, sizeof(float) * p_squared_minus_one_div_2);
	return x_;
}
void WatermarkFunctions::compute_NVF_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded, Eigen::ArrayXXf& m_nvf)
{
	m_nvf = Eigen::ArrayXXf::Constant(rows, cols, 0.0f);
	const int neighbor_size = (p - 1) / 2;
#pragma omp parallel for
	for (int i = pad; i < rows + pad; i++) {
		Eigen::ArrayXXf neighb = Eigen::ArrayXXf::Constant(p, p, 0.0f);
		float variance;
		for (int j = pad; j < cols + pad; j++) {
			variance = 0.0f;
			const float* neighb_d = neighb.data();
			const int i0 = i - neighbor_size;
			const int j0 = j - neighbor_size;
			const int i1 = i + neighbor_size;
			const int j1 = j + neighbor_size;
			neighb = padded.block(i0, j0, i1 - i0 + 1, j1 - j0 + 1);
			const float mean = neighb.mean();
			for (int ii = 0; ii < p; ii++)
				for (int jj = 0; jj < p; jj++)
					variance += powf(neighb_d[jj * p + ii] - mean, 2);
			variance = variance / (p_squared - 1);
			m_nvf(i - pad, j - pad) = 1.0f - (1.0f / (1.0f + variance));
		}
	}
}

Eigen::ArrayXXf WatermarkFunctions::make_and_add_watermark(bool is_custom_mask) {
	Eigen::ArrayXXf padded = Eigen::ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = image;
	Eigen::ArrayXXf m, u;
	if (is_custom_mask) {
		compute_NVF_mask(image, padded, m);
	}
	else {
		Eigen::ArrayXXf error_sequence;
		Eigen::MatrixXf coefficients;
		compute_prediction_error_mask(image, padded, m, error_sequence, coefficients, true);
	}
	u = m * w;
	float divisor = std::sqrt(u.square().sum() / (rows * cols));
	float a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divisor;
	return image + (a * u).eval();
}

//create NVF mask and return the watermarked image
Eigen::ArrayXXf WatermarkFunctions::make_and_add_watermark_NVF()
{
	return make_and_add_watermark(true);
}

//create ME mask and return the watermarked image
Eigen::ArrayXXf WatermarkFunctions::make_and_add_watermark_prediction_error()
{
	return make_and_add_watermark(false);
}

//compute Prediction error mask
void WatermarkFunctions::compute_prediction_error_mask(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded_image, Eigen::ArrayXXf& m_e, Eigen::ArrayXXf& error_sequence, Eigen::MatrixXf& coefficients, const bool mask_needed)
{
	m_e = Eigen::ArrayXXf::Constant(rows, cols, 0.0f);
	Eigen::MatrixXf Rx = Eigen::ArrayXXf::Constant(p_squared - 1, p_squared - 1, 0.0f);
	Eigen::MatrixXf rx = Eigen::ArrayXXf::Constant(p_squared - 1, 1, 0.0f);
	std::vector<Eigen::MatrixXf> Rx_all(num_threads);
	std::vector<Eigen::MatrixXf> rx_all(num_threads);
	for (int i = 0; i < num_threads; i++) {
		Rx_all[i] = Rx;
		rx_all[i] = rx;
	}
#pragma omp parallel for
	for (int i = pad; i < rows + pad; i++) {
		Eigen::VectorXf x_;
		for (int j = pad; j < cols + pad; j++) {
			//calculate p^-1 neighbors
			x_ = create_neighbors(padded_image, i, j, p, p_squared);
			//calculate Rx and rx
			Rx_all[omp_get_thread_num()].noalias() += (x_ * x_.transpose().eval());
			rx_all[omp_get_thread_num()].noalias() += (x_ * image(i - pad, j - pad));
		}
	}
	//reduction sums of Rx,rx of each thread
	for (int i = 0; i < num_threads; i++) {
		Rx.noalias() += Rx_all[i];
		rx.noalias() += rx_all[i];
	}
	coefficients = Rx.fullPivLu().solve(rx);
	coefficients.transposeInPlace();

	//calculate ex(i,j)
	compute_error_sequence(image, padded_image, coefficients, error_sequence);
	if (mask_needed) {
		Eigen::ArrayXXf error_sequence_abs = error_sequence.abs().eval();
		m_e = error_sequence_abs / error_sequence_abs.maxCoeff();
	}
}

//computes the prediction error sequence 
void WatermarkFunctions::compute_error_sequence(const Eigen::ArrayXXf& image, const Eigen::ArrayXXf& padded, Eigen::MatrixXf& coefficients, Eigen::ArrayXXf& error_sequence)
{
	error_sequence = Eigen::ArrayXXf::Constant(rows, cols, 0.0f);
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		Eigen::VectorXf x_;
		for (int j = 0; j < cols; j++) {
			x_ = create_neighbors(padded, i + pad, j + pad, p, p_squared);
			error_sequence(i, j) = image(i, j) - (coefficients * x_)(0);
		}
	}
}
//main mask detector for Me and NVF masks
float WatermarkFunctions::mask_detector(const Eigen::ArrayXXf& watermarked_image, const bool is_custom_mask)
{
	Eigen::MatrixXf a_z;
	Eigen::ArrayXXf m, e_z, padded = Eigen::ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = watermarked_image;
	if (is_custom_mask) {
		compute_prediction_error_mask(watermarked_image, padded, m, e_z, a_z, false); //no need for Me mask
		compute_NVF_mask(watermarked_image, padded, m);
	}
	else {
		compute_prediction_error_mask(watermarked_image, padded, m, e_z, a_z, true);
	}

	Eigen::ArrayXXf u = m * w;
	Eigen::ArrayXXf e_u;
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = u;
	compute_error_sequence(u, padded, a_z, e_u);
	float dot_ez_eu, d_ez, d_eu;
#pragma omp parallel sections
	{
#pragma omp section
		dot_ez_eu = (e_z * e_u).eval().sum();
#pragma omp section
		d_ez = std::sqrt(e_z.square().eval().sum());
#pragma omp section
		d_eu = std::sqrt(e_u.square().eval().sum());
	}
	return dot_ez_eu / (d_ez * d_eu);
}
