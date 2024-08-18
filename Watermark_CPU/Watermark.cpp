#include "pch.h"
#include "eigen_rgb_array.hpp"
#include "Watermark.hpp"
#include <cmath>
#include <fstream>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

#define ME_MASK_CALCULATION_REQUIRED_NO false
#define ME_MASK_CALCULATION_REQUIRED_YES true

using namespace Eigen;
using std::string;

//constructor to initialize all the necessary data
Watermark::Watermark(const EigenArrayRGB& image_rgb, const ArrayXXf& image, const string &w_file_path, const int p, const float psnr)
	:image_rgb(image_rgb), image(image), w(load_W(w_file_path, image.rows(), image.cols())), p(p), p_squared(static_cast<int>(std::pow(p, 2))), p_squared_minus_one_div_2((p_squared - 1) / 2), 
	pad(p / 2), num_threads(omp_get_max_threads()), rows(image.rows()), cols(image.cols()), padded_rows(rows + 2 * pad), padded_cols(cols + 2 * pad), psnr(psnr) {
}

//helper method to load the random noise matrix W from the file specified.
ArrayXXf Watermark::load_W(const string &w_file, const Index rows, const Index cols) {
	std::ifstream w_stream(w_file.c_str(), std::ios::binary);
	if (!w_stream.is_open())
		throw std::runtime_error(string("Error opening '" + w_file + "' file for Random noise W array\n"));
	w_stream.seekg(0, std::ios::end);
	const auto total_bytes = w_stream.tellg();
	w_stream.seekg(0, std::ios::beg);
	if (rows * cols * sizeof(float) != total_bytes)
		throw std::runtime_error(string("Error: W file total elements != image dimensions! W file total elements: " + std::to_string(total_bytes / (sizeof(float))) + ", Image width: " + std::to_string(cols) + ", Image height: " + std::to_string(rows) + "\n"));
	std::unique_ptr<float> w_ptr(new float[rows * cols]);
	w_stream.read(reinterpret_cast<char*>(&w_ptr.get()[0]), total_bytes);
	return Map<ArrayXXf>(w_ptr.get(), cols, rows).transpose().eval();
}

//generate p x p neighbors
void Watermark::create_neighbors(const ArrayXXf& array, VectorXf& x_, const int i, const int j)
{
	const int neighbor_size = (p - 1) / 2;
	const int start_row = i - neighbor_size;
	const int start_col = j - neighbor_size;
	const int end_row = i + neighbor_size;
	const int end_col = j + neighbor_size;
	const auto x_temp = array.block(start_row, start_col, end_row - start_row + 1, end_col - start_col + 1).reshaped();
	//ignore the central pixel value
	x_(seq(0, p_squared_minus_one_div_2 - 1)) = x_temp(seq(0, p_squared_minus_one_div_2 - 1));
	x_(seq(p_squared_minus_one_div_2, p_squared - 2)) = x_temp(seq(p_squared_minus_one_div_2 + 1, p_squared - 1));
}
void Watermark::compute_NVF_mask(const ArrayXXf& image, const ArrayXXf& padded, ArrayXXf& m_nvf)
{
	m_nvf = ArrayXXf(rows, cols);
	const int neighbor_size = (p - 1) / 2;
#pragma omp parallel for
	for (int i = pad; i < rows + pad; i++) {
		for (int j = pad; j < cols + pad; j++) {
			const int start_row = i - neighbor_size;
			const int end_row = i + neighbor_size;
			const int start_col = j - neighbor_size;
			const int end_col = j + neighbor_size;
			const auto neighb = padded.block(start_row, start_col, end_row - start_row + 1, end_col - start_col + 1);
			const float variance = (neighb - neighb.mean()).matrix().squaredNorm() / (p_squared - 1);
			m_nvf(i - pad, j - pad) = variance / (1.0f + variance);
		}
	}
}

EigenArrayRGB Watermark::make_and_add_watermark(MASK_TYPE mask_type) {
	ArrayXXf padded = ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = image;
	ArrayXXf m, u;
	if (mask_type == MASK_TYPE::NVF)
		compute_NVF_mask(image, padded, m);
	else {
		ArrayXXf error_sequence;
		VectorXf coefficients;
		compute_prediction_error_mask(padded, m, error_sequence, coefficients, ME_MASK_CALCULATION_REQUIRED_YES);
	}
	u = m * w;
	float divisor = std::sqrt(u.square().sum() / (rows * cols));
	float a = (255.0f / std::sqrt(std::pow(10.0f, psnr / 10.0f))) / divisor;
	const ArrayXXf u_strength = u * a;
	
	EigenArrayRGB watermarked_image;
#pragma omp parallel for
	for (int channel = 0; channel < 3; channel++)
		watermarked_image[channel] = (image_rgb[channel] + u_strength).cwiseMax(0).cwiseMin(255);
	return watermarked_image;
}

//compute Prediction error mask
void Watermark::compute_prediction_error_mask(const ArrayXXf& padded_image, ArrayXXf& m_e, ArrayXXf& error_sequence, VectorXf& coefficients, const bool mask_needed)
{
	MatrixXf Rx = ArrayXXf::Constant(p_squared - 1, p_squared - 1, 0.0f);
	MatrixXf rx = ArrayXXf::Constant(p_squared - 1, 1, 0.0f);
	std::vector<MatrixXf> Rx_all(num_threads);
	std::vector<MatrixXf> rx_all(num_threads);
	for (int i = 0; i < num_threads; i++) {
		Rx_all[i] = Rx;
		rx_all[i] = rx;
	}
#pragma omp parallel for
	for (int i = pad; i < rows + pad; i++) {
		VectorXf x_(p_squared - 1);
		MatrixXf Rx_pixel;
		VectorXf rx_pixel;
		for (int j = pad; j < cols + pad; j++) {
			//calculate p^2 - 1 neighbors
			create_neighbors(padded_image, x_, i, j);
			//calculate Rx and rx
			Rx_pixel.noalias() = x_ * x_.transpose();
			rx_pixel.noalias() = x_ * padded_image(i, j);
			Rx_all[omp_get_thread_num()].noalias() += Rx_pixel;
			rx_all[omp_get_thread_num()].noalias() += rx_pixel;
		}
	}
	//reduction sums of Rx,rx of each thread
	for (int i = 0; i < num_threads; i++) {
		Rx += Rx_all[i];
		rx += rx_all[i];
	}
	coefficients = Rx.fullPivLu().solve(rx);
	//calculate ex(i,j)
	compute_error_sequence(padded_image, coefficients, error_sequence);
	if (mask_needed) {
		ArrayXXf error_sequence_abs = error_sequence.abs().eval();
		m_e = error_sequence_abs / error_sequence_abs.maxCoeff();
	}
}

//computes the prediction error sequence 
void Watermark::compute_error_sequence(const ArrayXXf& padded, const VectorXf& coefficients, ArrayXXf& error_sequence)
{
	error_sequence = ArrayXXf(rows, cols);
#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		VectorXf x_(p_squared - 1);
		const int padded_i = i + pad;
		for (int j = 0; j < cols; j++) {
			const int padded_j = j + pad;
			create_neighbors(padded, x_, padded_i, padded_j);
			error_sequence(i, j) = padded(padded_i, padded_j) - x_.dot(coefficients);
		}
	}
}
//main mask detector for Me and NVF masks
float Watermark::mask_detector(const ArrayXXf& watermarked_image, MASK_TYPE mask_type)
{
	VectorXf a_z;
	ArrayXXf m, e_z, padded = ArrayXXf::Constant(padded_rows, padded_cols, 0.0f);
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = watermarked_image;
	if (mask_type == MASK_TYPE::NVF) {
		compute_prediction_error_mask(padded, m, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_NO);
		compute_NVF_mask(watermarked_image, padded, m);
	}
	else {
		compute_prediction_error_mask(padded, m, e_z, a_z, ME_MASK_CALCULATION_REQUIRED_YES);
	}

	ArrayXXf e_u;
	padded.block(pad, pad, (padded_rows - pad) - pad, (padded_cols - pad) - pad) = (m * w);
	compute_error_sequence(padded, a_z, e_u);
	float dot_ez_eu, d_ez, d_eu;
	
#pragma omp parallel sections
	{
#pragma omp section
		dot_ez_eu = e_z.cwiseProduct(e_u).sum();
#pragma omp section
		d_ez = std::sqrt(e_z.matrix().squaredNorm());
#pragma omp section
		d_eu = std::sqrt(e_u.matrix().squaredNorm());
	}
	return dot_ez_eu / (d_ez * d_eu);
}
