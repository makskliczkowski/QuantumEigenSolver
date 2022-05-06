#pragma once
#ifndef BINARY_H
#include "../../src/binary.h"
#endif // ! BINARY_H


#ifndef OPERATORS_H
#define OPERATORS_H

/* 
* @brief multiplication of sigma_xi | state >
* @param L lattice dimensionality (base vector length)
* @param sites the sites to meassure correlation at
*/
static std::pair<u64, double> sigma_x(u64 base_vec, int L, const v_1d<int>& sites) {
	for (auto const& site : sites)
		base_vec = flip(base_vec, L - 1 - site);
	return std::make_pair(base_vec, 1.0);
};

/*
* @brief multiplication of sigma_yi | state >
* @param L lattice dimensionality (base vector length)
* @param sites the sites to meassure correlation at
*/
static std::pair<u64, cpx> sigma_y(u64 base_vec, int L, const v_1d<int>& sites) {
	auto tmp = base_vec;
	cpx val = 1.0;
	for (auto const& site : sites) {
		val *= checkBit(tmp, L - 1 - site) ? imn : -imn;
		tmp = flip(tmp, L - 1 - site);
	}
	return std::make_pair(tmp, val);
};

/*
* @brief multiplication of sigma_zi | state >
* @param L lattice dimensionality (base vector length)
* @param sites the sites to meassure correlation at
*/
static std::pair<u64, double> sigma_z(u64 base_vec, int L, const v_1d<int>& sites) {
	double val = 1.0;
	for (auto const& site : sites)
		val *= checkBit(base_vec, L - 1 - site) ? 1.0 : -1.0;
	return std::make_pair(base_vec, val);
};


static std::pair<u64, cpx> spin_flip(u64 base_vec, int L, v_1d<int> sites) {
	if (sites.size() > 2) throw "Not implemented such exotic operators, choose 1 or 2 sites\n";
	auto tmp = base_vec;
	cpx val = 0.0;
	auto it = sites.begin() + 1;
	auto it2 = sites.begin();
	if (!(checkBit(base_vec, L - 1 - *it))) {
		tmp = flip(tmp, L - 1 - *it);
		val = 2.0;
		if (sites.size() > 1) {
			if (checkBit(base_vec, L - 1 - *it2)) {
				tmp = flip(tmp, L - 1 - *it2);
				val *= 2.0;
			}
			else val = 0.0;
		}
	}
	else val = 0.0;
	return std::make_pair(tmp, val);
};

#endif // !OPERATORS_H
