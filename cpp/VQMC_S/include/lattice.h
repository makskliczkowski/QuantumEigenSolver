#pragma once
#ifndef BINARY_H
#include "../src/binary.h"
#endif // !BINARY_H


#ifndef LATTICE_H
#define LATTICE_H
using namespace std;
// -------------------------------------------------------- GENERAL LATTICE --------------------------------------------------------

/*
* Pure virtual lattice class, it will allow to distinguish between different geometries in the models
*/
class Lattice {
protected:
	// ----------------------- LATTICE PARAMETERS
	unsigned int dim = 1;											// the dimensionality of the lattice 1,2,3
	unsigned int Ns = 1;											// number of lattice sites
	string type;													// type of the lattice
	int _BC = 0;													// boundary conditions 0 = PBC, 1 = OBC
	v_2d<int> nearest_neighbors;									// vector of the nearest neighbors
	v_2d<int> next_nearest_neighbors;								// vector of the next nearest neighbors
	v_2d<int> coordinates;											// vector of real coordiates allowing to get the distance between lattice points
	v_3d<int> spatialNorm;											// norm for averaging over all spatial sites

public:
	virtual ~Lattice() = default;
	// ----------------------- VIRTUAL GETTERS
	virtual int get_Lx() const = 0;
	virtual int get_Ly() const = 0;
	virtual int get_Lz() const = 0;
	virtual std::tuple<int, int, int> getSiteDifference(uint i, uint j) const = 0;

	// ----------------------- GETTERS
	virtual int get_norm(int x, int y, int z) const = 0;
	auto get_Ns()											const RETURNS(this->Ns);												// returns the number of sites
	auto get_Dim()											const RETURNS(this->dim);												// returns dimension of the lattice
	auto get_nn(int lat_site, int nei_num)					const RETURNS(this->nearest_neighbors[lat_site][nei_num]);				// returns given nearest nei at given lat site
	auto get_nnn(int lat_site, int nei_num)					const RETURNS(this->next_nearest_neighbors[lat_site][nei_num]);			// returns given next nearest nei at given lat site
	auto get_nn_number(int lat_site)						const RETURNS(this->nearest_neighbors[lat_site].size());				// returns the number of nn
	auto get_nnn_number(int lat_site)						const RETURNS(this->next_nearest_neighbors[lat_site].size());			// returns the number of nnn
	auto get_coordinates(int lat_site, int axis)			const RETURNS(this->coordinates[lat_site][axis]);						// returns the given coordinate
	auto get_type()											const RETURNS(this->type);												// returns the type of the lattice as a string
	auto get_nei(int lat_site, int corr_len)				const;

// ----------------------- CALCULATORS
	void calculate_nn();
	virtual void calculate_nn_pbc() = 0;
	virtual void calculate_nn_obc() = 0;
	virtual void calculate_nnn_pbc() = 0;
	virtual void calculate_coordinates() = 0;
};

inline void Lattice::calculate_nn() {
	switch (this->_BC)
	{
	case 0:
		this->calculate_nn_pbc();
		//this->calculate_nnn_pbc();
		break;
	case 1:
		this->calculate_nn_obc();
		break;
	default:
		this->calculate_nn_pbc();
		break;
	}
}

inline auto Lattice::get_nei(int lat_site, int corr_len) const
{
	switch (this->_BC) {
	case 0:
		return myModuloEuclidean((lat_site + corr_len), this->Ns);
		break;
	case 1:
		return (lat_site + corr_len) > this->Ns ? - 1 : (lat_site + corr_len);
		break;
	default:
		return myModuloEuclidean((lat_site + corr_len), this->Ns);
	}
}

#endif // !LATTICE_H