#include "cimg_init.hpp"
#include "eigen_rgb_array.hpp"
#include "INIReader.h"
#include "Utilities.hpp"
#include "Watermark.hpp"
#include "Watermark_CPU.hpp"
#include <cstdlib>
#include <Eigen/Dense>
#include <exception>
#include <format>
#include <iostream>
#include <omp.h>
#include <string>
#include <thread>

#define R_WEIGHT 0.299f
#define G_WEIGHT 0.587f
#define B_WEIGHT 0.114f

using namespace cimg_library;
using namespace Eigen;
using std::cout;
using std::string;

/*!
 *  \brief  This is a project implementation of my Thesis with title:
 *			EFFICIENT IMPLEMENTATION OF WATERMARKING ALGORITHMS AND
 *			WATERMARK DETECTION IN IMAGE AND VIDEO USING GPU
 *  \author Dimitris Karatzas
 */
int main(int argc, char** argv)
{
	const INIReader inir("settings.ini");
	if (inir.ParseError() < 0) {
		cout << "Could not load configuration file, exiting..";
		exit_program(EXIT_FAILURE);
	}
	const string image_path = inir.Get("paths", "image", "NO_IMAGE");
	const bool show_fps = inir.GetBoolean("options", "execution_time_in_fps", false);
	const int p = inir.GetInteger("parameters", "p", 5);
	const float psnr = static_cast<float>(inir.GetReal("parameters", "psnr", 30.0f));
	const string w_file = inir.Get("paths", "w_path", "w.txt");
	int num_threads = inir.GetInteger("parameters", "threads", 0);
	if (num_threads <= 0 || num_threads > 256) {
		auto threads_supported = std::thread::hardware_concurrency();
		num_threads = threads_supported == 0 ? 2 : threads_supported;
	}
	int loops = inir.GetInteger("parameters", "loops_for_test", 5); 
	loops = loops <= 0 || loops > 64 ? 5 : loops;

	//openmp initialization
	omp_set_num_threads(num_threads);
#pragma omp parallel for
	for (int i = 0; i < 24; i++) {}

	const CImg<float> rgb_image_cimg(image_path.c_str());
	const int rows = rgb_image_cimg.height();
	const int cols = rgb_image_cimg.width();

	if (cols <= 16 || rows <= 16 || rows >= 16384 || cols >= 16384) {
		cout << "Image dimensions too low or too high\n";
		exit_program(EXIT_FAILURE);
	}
	if (p <= 0 || p % 2 != 1 || p > 9) {
		cout << "p parameter must be a positive odd number less than 9\n";
		exit_program(EXIT_FAILURE);
	}
	if (psnr <= 0) {
		cout << "PSNR must be a positive number\n";
		exit_program(EXIT_FAILURE);
	}

	cout << "Using " << omp_get_num_threads() << " parallel threads.\n";
	cout << "Each test will be executed " << loops << " times. Average time will be shown below\n";
	cout << "Image size is: " << rows << " rows and " << cols << " columns\n\n";

	//copy from cimg to Eigen
	timer::start();
	const EigenArrayRGB array_rgb = cimg_to_eigen_rgb_array(rgb_image_cimg);
	const ArrayXXf array_grayscale = eigen_rgb_array_to_grayscale_array(array_rgb, R_WEIGHT, G_WEIGHT, B_WEIGHT);
	timer::end();
	cout << "Time to load image from disk and initialize CImg and Eigen memory objects: " << timer::secs_passed() << " seconds\n\n";
	
	//tests begin
	try {
		//initialize main class responsible for watermarking and detection
		Watermark watermark_obj(array_rgb, array_grayscale, w_file, p, psnr);

		double secs = 0;
		//NVF mask calculation
		EigenArrayRGB watermark_NVF, watermark_ME;
		for (int i = 0; i < loops; i++) {
			timer::start();
			watermark_NVF = watermark_obj.make_and_add_watermark(MASK_TYPE::NVF);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << std::format("Calculation of NVF mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, execution_time(show_fps, secs / loops));
		
		secs = 0;
		//Prediction error mask calculation
		for (int i = 0; i < loops; i++) {
			timer::start();
			watermark_ME = watermark_obj.make_and_add_watermark(MASK_TYPE::ME);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << std::format("Calculation of ME mask with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, execution_time(show_fps, secs / loops));

		const ArrayXXf watermarked_NVF_gray = eigen_rgb_array_to_grayscale_array(watermark_NVF, R_WEIGHT, G_WEIGHT, B_WEIGHT);
		const ArrayXXf watermarked_ME_gray = eigen_rgb_array_to_grayscale_array(watermark_ME, R_WEIGHT, G_WEIGHT, B_WEIGHT);

		float correlation_nvf, correlation_me;
		secs = 0;
		//NVF mask detection
		for (int i = 0; i < loops; i++) {
			timer::start();
			correlation_nvf = watermark_obj.mask_detector(watermarked_NVF_gray, MASK_TYPE::NVF);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << std::format("Calculation of the watermark correlation (NVF) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, execution_time(show_fps, secs / loops));

		secs = 0;
		//Prediction error mask detection
		for (int i = 0; i < loops; i++) {
			timer::start();
			correlation_me = watermark_obj.mask_detector(watermarked_ME_gray, MASK_TYPE::ME);
			timer::end();
			secs += timer::secs_passed();
		}
		cout << std::format("Calculation of the watermark correlation (ME) of an image with {} rows and {} columns and parameters:\np = {}  PSNR(dB) = {}\n{}\n\n", rows, cols, p, psnr, execution_time(show_fps, secs / loops));

		cout << std::format("Correlation [NVF]: {:.16f}\n", correlation_nvf);
		cout << std::format("Correlation [ME]: {:.16f}\n", correlation_me);

		//save watermarked images to disk
		if (inir.GetBoolean("options", "save_watermarked_files_to_disk", false)) {
			cout << "\nSaving watermarked files to disk...\n";
#pragma omp parallel sections 
			{
#pragma omp section
				save_watermarked_image(image_path, "_W_NVF", watermark_NVF);
#pragma omp section
				save_watermarked_image(image_path, "_W_ME", watermark_ME);
			}
			cout << "Successully saved to disk\n";
		}
	}
	catch (const std::exception& e) {
		cout << e.what() << "\n";
		exit_program(EXIT_FAILURE);
	}
	exit_program(EXIT_SUCCESS);
}

//calculate execution time in seconds, or show FPS value
string execution_time(const bool show_fps, const double seconds) {
	return show_fps ? std::format("FPS: {:.2f} FPS", 1.0 / seconds) : std::format("{:.6f} seconds", seconds);
}

//save the provided Eigen RGB array containing a watermarked image to disk
void save_watermarked_image(const string& image_path, const string& suffix, const EigenArrayRGB& watermark) {
	std::string watermarked_file = add_suffix_before_extension(image_path, suffix);
	eigen_rgb_array_to_cimg(watermark).save_png(watermarked_file.c_str());
}

//exits the program with the provided exit code
void exit_program(const int exit_code) {
	std::system("pause");
	std::exit(exit_code);
}