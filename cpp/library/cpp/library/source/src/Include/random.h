#pragma once
#ifndef RANDOM_H
#define RANDOM_H

#include "../xoshiro_pp.h"
#include <random>
#include <ctime>
#include <numeric>

// -------------------------------------------------------- RANDOM NUMBER CLASS --------------------------------------------------------

/*
* @brief Random number generator class based on Xorshiro256
* @link 
*/
class randomGen {
private:
	XoshiroCpp::Xoshiro256PlusPlus engine;
public:
	explicit randomGen(std::uint64_t seed = std::random_device{}()) 
	{
		this->newSeed(seed);
	}

	/*
	* @brief move integer to get optimal seed
	*/
	uint64_t seedInit(uint64_t n) const
	{
		uint64_t z = (n += 0x9e3779b97f4a7c15);
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		return z ^ (z >> 31);
	}

	/*
	* @brief initialize seed
	*/
	void newSeed(std::uint64_t seed) {
		this->engine = XoshiroCpp::Xoshiro256PlusPlus(this->seedInit(seed));
	}

	// --------------------- WRAPPERS ON RANDOM FUNCTIONS ---------------------

	template <typename T>
	T xavier(T in, T out, double xav = 6.0) {
		return std::uniform_real_distribution<T>(-1., +1.)(engine) * std::sqrt(xav / (in + out));
	}

	template <typename T>
	T kaiming(T in) {
		return std::uniform_real_distribution<T>(-1., +1.)(engine) * std::sqrt(6.0 / in);
	}

	/*
	* @brief random real uniform distribution : _min <= x < _max
	* @param _min smallest value
	* @param _max highest value
	*/
	template <typename T>
	T random(T _min = 0, T _max = 1) {
		return std::uniform_real_distribution<T>(_min, _max)(engine);
	}

	template <typename T>
	T randomInt(T _min, T _max) {
		return static_cast<long long>(_min + static_cast<T>((_max - _min) * this->random()));
	}

	/*
	* @brief random normal distribution
	*/
	template <typename T>
	T randomNormal(T _mean = 0, T _std = 1) {
		return std::normal_distribution(_mean, _std)(engine);
	}

	template <typename T>
	bool bernoulli(T p) {
		return std::bernoulli_distribution(p)(engine);
	}

	// --------------------- OTHER ---------------------
	
	/*
	* @brief creates random vector with a given strength
	*/
	arma::vec createRanVec(int _size, double _strength) {
		arma::vec o = arma::ones(_size);
		for (auto i = 0; i < _size; i++)
			o(i) = (this->random<double>() * 2.0 - 1) * _strength;
		return o;
	}
};

#endif // !RANDOM_H
