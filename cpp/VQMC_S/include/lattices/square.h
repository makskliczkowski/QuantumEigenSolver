#pragma once
#ifndef LATTICE_H
#include "lattice.h"
#endif // !LATTICE_H

// -------------------------------------------------------- SQUARE LATTICE --------------------------------------------------------

class SquareLattice : public Lattice {
private:
	int Lx;																												// spatial x-length
	int Ly;																												// spatial y-length
	int Lz;																												// spatial z-length
public:
	// CONSTRUCTORS
	~SquareLattice() = default;	
	SquareLattice() = default;
	SquareLattice(int Lx, int Ly = 1, int Lz = 1, int dim = 1, int _BC = 0);											// general constructor

	// GETTERS
	int get_Lx() const override { return this->Lx; };
	int get_Ly() const override { return this->Ly; };
	int get_Lz() const override { return this->Lz; };
	int get_norm(int x, int y, int z) const override { return this->spatialNorm[x + this->Lx - 1][y + this->Ly - 1][z + this->Lz - 1]; };
	std::tuple<int, int, int> getSiteDifference(uint i, uint j) const override;

	// CALCULATORS
	void calculate_nn_pbc() override;
	void calculate_nn_obc() override;
	void calculate_nnn_pbc() override;
	void calculate_coordinates() override;
};

/*
* @brief Constructor for the square lattice
*/
SquareLattice::SquareLattice(int Lx, int Ly, int Lz, int dim, int _BC) 
	: Lx(Lx), Ly(Ly), Lz(Lz)
{
	this->dim = dim;
	this->_BC = _BC;
	this->type = "square";
	// fix sites depending on _BC
	switch (this->dim)
	{
	case 1:
		this->Ly = 1; this->Lz = 1;
		break;
	case 2:
		this->Lz = 1;
		break;
	default:
		break;
	}
	this->Ns = this->Lx * this->Ly * this->Lz;

	this->calculate_nn();
	this->calculate_coordinates();
	// spatial norm
	this->spatialNorm = v_3d<int>(2 * Lx - 1, v_2d<int>(2 * Ly - 1, v_1d<int>(2 * Lz - 1, 0)));
	// for (int i = 0; i < this->Ns; i++) {
	// 	for (int j = 0; j < this->Ns; j++) {
	// 		const auto [x, y, z] = this->getSiteDifference(i, j);
	// 		spatialNorm[x][y][z]++;
	// 	}
	// }
}
/*
* @brief Returns the real space difference between lattice site cooridinates given in ascending order.
* From left to right. Then second row left to right etc.
* @param i First coordinate
* @param j Second coordinate
* @return Three-dimensional tuple (vector of vec[i]-vec[j])
*/
std::tuple<int, int, int> SquareLattice::getSiteDifference(uint i, uint j) const
{
	const int z = this->get_coordinates(i, 2) - this->get_coordinates(j, 2);
	const int y = this->get_coordinates(i, 1) - this->get_coordinates(j, 1);
	const int x = this->get_coordinates(i, 0) - this->get_coordinates(j, 0);
	return std::tuple<int, int, int>(x + Lx - 1, y + Ly - 1, z + Lz - 1);
}

/*
* @brief Calculate the nearest neighbors with PBC
*/
void SquareLattice::calculate_nn_pbc()
{
	switch (this->dim)
	{
	case 1:
		//* One dimension 
		this->nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->nearest_neighbors[i][0] = myModuloEuclidean(i + 1, Lx);											// right
			this->nearest_neighbors[i][1] = myModuloEuclidean(i - 1, Lx);											// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(4, 0));
		for (int i = 0; i < Ns; i++) {
			this->nearest_neighbors[i][0] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i + 1, Lx);		// right
			this->nearest_neighbors[i][1] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i - 1, Lx);		// left
			this->nearest_neighbors[i][2] = myModuloEuclidean(i + Lx, Ns);											// bottom
			this->nearest_neighbors[i][3] = myModuloEuclidean(i - Lx, Ns);											// top
		}
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}

/*
* @brief Calculate the nearest neighbors with OBC
*/
void SquareLattice::calculate_nn_obc()
{
	switch (this->dim)
	{
	case 1:
		//* One dimension 
		this->nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->nearest_neighbors[i][0] = (i + 1) >= Lx ? -1 : i + 1;										// right
			this->nearest_neighbors[i][1] = (i - 1) == 0 ? -1 : i - 1;										// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		//this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(4, 0));
		//for (int i = 0; i < Ns; i++) {
		//	this->nearest_neighbors[i][0] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i + 1, Lx);		// right
		//	this->nearest_neighbors[i][1] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i - 1, Lx);		// left
		//	this->nearest_neighbors[i][2] = myModuloEuclidean(i + Lx, Ns);											// bottom
		//	this->nearest_neighbors[i][3] = myModuloEuclidean(i - Lx, Ns);											// top
		//}
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}

/*
* @brief Calculate the next nearest neighbors with PBC
*/
void SquareLattice::calculate_nnn_pbc()
{
	switch (this->dim)
	{
	case 1:
		/* One dimension */
		this->nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->nearest_neighbors[i][0] = myModuloEuclidean(i + 2, Lx);											// right
			this->nearest_neighbors[i][1] = myModuloEuclidean(i - 2, Lx);											// left
		}
		break;
	case 2:
		/* Two dimensions */
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}
/*
* @brief Returns real space coordinates from a lattice site number
*/
void SquareLattice::calculate_coordinates()
{
	const int LxLy = Lx * Ly;
	this->coordinates = v_2d<int>(this->Ns, v_1d<int>(3, 0));
	for (int i = 0; i < Ns; i++) {
		this->coordinates[i][0] = i % Lx;												// x axis coordinate
		this->coordinates[i][1] = (static_cast<int>(1.0 * i / Lx)) % Ly;				// y axis coordinate
		this->coordinates[i][2] = (static_cast<int>(1.0 * i / (LxLy))) % Lz;			// z axis coordinate
		//std::cout << "(" << this->coordinates[i][0] << "," << this->coordinates[i][1] << "," << this->coordinates[i][2] << ")\n";
	}
}