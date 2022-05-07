#include "../include/lattices/hexagonal.h"

/*
* Constructor for the hexagonal lattice
*/
HexagonalLattice::HexagonalLattice(int Lx, int Ly, int Lz, int dim, int _BC)
	: Lx(Lx), Ly(Ly), Lz(Lz)
{
	this->dim = dim;
	this->_BC = _BC;
	this->type = "hexagonal";
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
	this->Ns = 2 * this->Lx * this->Ly * this->Lz;

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
std::tuple<int, int, int> HexagonalLattice::getSiteDifference(uint i, uint j) const
{
	return std::tuple<int, int, int>(1, 1, 1);
}

/*
* Calculate the nearest neighbors with PBC
*/
void HexagonalLattice::calculate_nn_pbc()
{
	switch (this->dim)
	{
	case 1:
		// One dimension 
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(1, 0));
		for (int i = 0; i < Lx; i++) {
			// z bond only
			this->nearest_neighbors[2 * i][0] = 2 * i + 1;
			this->nearest_neighbors[2 * i + 1][0] = 2 * i;
		}
		break;
	case 2:
		// Two dimensions 
		// numeration begins from the bottom as 0 to the second as 1 with lattice vectors move
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(3, 0));
		for (int i = 0; i < Lx; i++) {
			for (int j = 0; j < Ly; j++) {
				auto current_elem_a = 2 * i + 2 * Lx * j;
				auto current_elem_b = 2 * i + 2 * Lx * j + 1;

				auto up = myModuloEuclidean(j + 1, Ly);
				auto down = myModuloEuclidean(j - 1, Ly);
				auto right = myModuloEuclidean(i + 1, Lx);
				auto left = myModuloEuclidean(i - 1, Lx);

				// y and x bonding depends on current y level as the hopping between sites changes
				auto x_change_i_a = (2 * i + 2 * down * Lx + 1); // false;
				auto y_change_i_a = (2 * left + 2 * down * Lx + 1); // true;

				auto x_change_i_b = (2 * left + 2 * up * Lx); // true;
				auto y_change_i_b = (2 * i + 2 * up * Lx); // false;
				if (myModuloEuclidean(j, 2) == 1) {
					x_change_i_a = (2 * right + 2 * down * Lx + 1); // true
					x_change_i_b = (2 * i + 2 * up * Lx); // false

					y_change_i_a = (2 * i + 2 * down * Lx + 1); // false;
					y_change_i_b = (2 * right + 2 * up * Lx); // true;
				}

				// x bonding
				this->nearest_neighbors[current_elem_a][2] = x_change_i_a;
				this->nearest_neighbors[current_elem_b][2] = x_change_i_b;
				// y bonding
				this->nearest_neighbors[current_elem_a][1] = y_change_i_a;
				this->nearest_neighbors[current_elem_b][1] = y_change_i_b;
				// z bonding
				this->nearest_neighbors[current_elem_a][0] = current_elem_a + 1;
				this->nearest_neighbors[current_elem_b][0] = current_elem_b - 1;
			}
		}
		//		stout << this->nearest_neighbors << EL;
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}

/*
* @brief Calculate the nearest neighbors with OBC - WORKING HELLA FINE 2D
*/
void HexagonalLattice::calculate_nn_obc()
{
	switch (this->dim)
	{
	case 1:
		// One dimension 
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(1, 0));
		for (int i = 0; i < Lx; i++) {
			// z bond only
			this->nearest_neighbors[2 * i][0] = 2 * i + 1;
			this->nearest_neighbors[2 * i + 1][0] = 2 * i;
		}
		break;
	case 2:
		// Two dimensions 
		// numeration begins from the bottom as 0 to the second as 1 with lattice vectors move
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(3, -1));
		for (int i = 0; i < Lx; i++) {
			for (int j = 0; j < Ly; j++) {
				auto current_elem_a = 2 * i + 2 * Lx * j;
				auto current_elem_b = 2 * i + 2 * Lx * j + 1;

				auto up = j + 1;
				auto down = j - 1;
				auto right = i + 1;
				auto left = i - 1;

				// y and x bonding depends on current y level as the hopping between sites changes
				auto x_change_i_a = -1;
				auto x_change_i_b = -1;
				auto y_change_i_a = -1;
				auto y_change_i_b = -1;

				if (down >= 0) {
					x_change_i_a = (2 * i + 2 * down * Lx + 1); // false;
					if (left >= 0)
						y_change_i_a = (2 * left + 2 * down * Lx + 1); // true;
				}
				if (up < Ly) {
					if (left >= 0)
						x_change_i_b = (2 * left + 2 * up * Lx); // true;
					y_change_i_b = (2 * i + 2 * up * Lx); // false;
				}
				if (myModuloEuclidean(j, 2) == 1) {
					if (down >= 0) {
						if (right < Lx)
							x_change_i_a = (2 * right + 2 * down * Lx + 1); // true
						y_change_i_a = (2 * i + 2 * down * Lx + 1); // false;
					}
					if (up < Ly) {
						x_change_i_b = (2 * i + 2 * up * Lx); // false
						if (right < Lx)
							y_change_i_b = (2 * right + 2 * up * Lx); // true;
					}
				}

				// x bonding
				this->nearest_neighbors[current_elem_a][2] = x_change_i_a;
				this->nearest_neighbors[current_elem_b][2] = x_change_i_b;
				// y bonding
				this->nearest_neighbors[current_elem_a][1] = y_change_i_a;
				this->nearest_neighbors[current_elem_b][1] = y_change_i_b;
				// z bonding
				this->nearest_neighbors[current_elem_a][0] = current_elem_a + 1;
				this->nearest_neighbors[current_elem_b][0] = current_elem_b - 1;
			}
		}
		// stout << this->nearest_neighbors << EL;
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
void HexagonalLattice::calculate_nnn_pbc()
{
	switch (this->dim)
	{
	case 1:
		/* One dimension */
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
void HexagonalLattice::calculate_coordinates()
{

}
