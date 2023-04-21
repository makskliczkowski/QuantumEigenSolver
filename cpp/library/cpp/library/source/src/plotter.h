#pragma once

#ifdef PLOT
	#ifdef _DEBUG
		#undef _DEBUG
		#include <python.h>
		#define _DEBUG
	#else
		#include <python.h>
	#endif

	// plotting
	#define WITHOUT_NUMPY
	#define WITH_OPENCV
	#include "matplotlib-cpp/matplotlibcpp.h"
	namespace plt = matplotlibcpp;

	template<typename _T>
	void plot(v_1d<_T> x, v_1d<_T> v, std::string xlabel = "", std::string ylabel = "", std::string name = "") {
		plt::plot(x, v);
		plt::xlabel(xlabel);
		plt::ylabel(ylabel);
		plt::title(name);
	};
	template<typename _T>
	void inline plot(arma::Col<_T> x, arma::Col<_T> v, std::string xlabel = "", std::string ylabel = "", std::string name = "") {
		plt::plot(arma::conv_to<v_1d<double>>::from(x), arma::conv_to<v_1d<double>>::from(v));
		plt::xlabel(xlabel);
		plt::ylabel(ylabel);
		plt::title(name);
	};
	
	#define PLOTV(x, v, xl, yl, title) plot(x, v, xl, yl, title)

	template<typename _type>
	void inline scatter(v_1d<_type> x, v_1d<_type> v, std::string xlabel = "", std::string ylabel = "", std::string name = "") {
		plt::scatter_colored(x, v);
		plt::xlabel(xlabel);
		plt::ylabel(ylabel);
		plt::title(name);
	};
	#define SCATTER(x, v, xl, yl, title) scatter(x, v, xl, yl, title)

	// -------------------------- SAVE THE FIGURE 
	void inline save_fig(std::string name, bool show = false) {
		plt::save(name);
		if (show) plt::show();
		plt::close();
	}

	#define SAVEFIG(name, show) save_fig(name, show)
	#define SHOWFIG plt::show();
	#define CLOSEFIG plt::close();
#else 
	#define PLOT_V1D(x, v, xl, yl, n)
	#define SCATTER_V1D(x, v, xl, yl, n)
	#define SAVEFIG(name, show)
	#define SHOWFIG
	#define CLOSEFIG
#endif