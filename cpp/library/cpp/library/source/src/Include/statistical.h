#pragma once

#ifndef ALG_H
#include "../lin_alg.h"
#endif // !ALG_H

// ######################################################## binning ########################################################

/*
* @brief bin the data to calculate the correlation time approximation
* @param seriesData data to be binned
* @param bins the vector to save the average into
* @param binSize the size of a given single bin
*/
template<typename _T>
inline void binning(const _T& seriesData, _T& bins, uint binSize) {
	if (binSize * bins.size() > seriesData.size()) throw "Cannot create bins of insufficient elements";
	for (int i = 0; i < bins.size(); i++)
		bins[i] = arma::mean(seriesData.subvec(binSize * i, binSize * (i + 1) - 1));
}


// ######################################################## statistical meassures ########################################################

/*
* @brief Approximate the correlation error
* @param bins the bin average of the data
* @returns approximation of a statistical correlation error
*/
template<typename _T>
inline _T correlationError(const arma::Col<_T>& bins) {
	return arma::real(sqrt(arma::var(bins) / bins.size()));
}

///*
//* @brief for an std::vector calculates the standard deviation
//*/
//inline double stddev(const v_1d<double>& v)
//{
//	auto mean = std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
//	v_1d<double> diff(v.size());
//	std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
//	auto sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
//	return std::sqrt(sq_sum / double(v.size()));
//}