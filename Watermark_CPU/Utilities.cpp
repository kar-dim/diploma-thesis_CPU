#include "pch.h"
#include "cimg_init.hpp"
#include "eigen_rgb_array.hpp"
#include "Utilities.hpp"
#include <chrono>
#include <string>

using namespace cimg_library;
using namespace Eigen;

using std::string;

string add_suffix_before_extension(const string& file, const string& suffix) {
	auto dot = file.find_last_of('.');
	return dot == string::npos ? file + suffix : file.substr(0, dot) + suffix + file.substr(dot);
}

CImg<float> eigen_rgb_array_to_cimg(const EigenArrayRGB& array_rgb) {
	const auto rows = array_rgb[0].rows();
	const auto cols = array_rgb[0].cols();
	CImg<float> cimg_image(static_cast<unsigned int>(cols), static_cast<unsigned int>(rows), 1, 3);
	//a parallel pixel by pixel copy for loop is faster instead of three parallel (channel) bulk memory copies
	//because cimg and eigen use different memory layouts, and transposing is required which would make the copy much slower
#pragma omp parallel for
	for (int y = 0; y < rows; ++y)
		for (int x = 0; x < cols; ++x)
			for (int channel = 0; channel < 3; channel++)
				cimg_image(x, y, 0, channel) = array_rgb[channel](y, x);
	return cimg_image;
}

EigenArrayRGB cimg_to_eigen_rgb_array(CImg<float>& rgb_image) {
	const int rows = rgb_image.height();
	const int cols = rgb_image.width();
	//a parallel pixel by pixel copy for loop is faster instead of three parallel (channel) bulk memory copies
	//because cimg and eigen use different memory layouts, and transposing is required which would make the copy much slower
	EigenArrayRGB rgb_array = { ArrayXXf(rows,cols), ArrayXXf(rows,cols), ArrayXXf(rows, cols) };
#pragma omp parallel for
	for (int y = 0; y < rgb_image.height(); y++)
		for (int x = 0; x < rgb_image.width(); x++)
			for (int channel = 0; channel < 3; channel++)
				rgb_array[channel](y,x) = rgb_image(x, y, 0, channel);
	return rgb_array;
}

ArrayXXf eigen_rgb_array_to_grayscale_array(const EigenArrayRGB& array_rgb, const float r_weight, const float g_weight, const float b_weight) {
	return (array_rgb[0] * r_weight) + (array_rgb[1] * g_weight) + (array_rgb[2] * b_weight);
}

//χρονομέτρηση
namespace timer {
	void start() {
		start_timex = std::chrono::high_resolution_clock::now();
	}
	void end() {
		cur_timex = std::chrono::high_resolution_clock::now();
	}
	double secs_passed() {
		return (double)std::chrono::duration_cast<std::chrono::microseconds>(cur_timex - start_timex).count() / 1000000;
	}
}