#pragma once
#include "pch.h"
#include "cimg_init.hpp"
#include "eigen_rgb_array.hpp"
#include <chrono>
#include <string>

std::string add_suffix_before_extension(const std::string& file, const std::string& suffix);
cimg_library::CImg<float> eigen_rgb_array_to_cimg(const EigenArrayRGB& image_rgb);
EigenArrayRGB cimg_to_eigen_rgb_array(cimg_library::CImg<float>& rgb_image);
Eigen::ArrayXXf eigen_rgb_array_to_grayscale_array(const EigenArrayRGB& image_rgb, const float r_weight, const float g_weight, const float b_weight);

namespace timer {
	static std::chrono::time_point<std::chrono::steady_clock> start_timex, cur_timex;
	void start();
	void end();
	double secs_passed();
}