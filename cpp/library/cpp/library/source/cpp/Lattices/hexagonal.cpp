#include "../../src/Lattices/hexagonal.h"

/*
* @brief Constructor for the hexagonal lattice
*/
HexagonalLattice::HexagonalLattice(int Lx, int Ly, int Lz, int dim, int _BC)
	: Lx(Lx), Ly(Ly), Lz(Lz)
{
	this->dim = dim;
	this->_BC = static_cast<BoundaryConditions>(_BC);
	this->type = getSTR_BoundaryConditions(this->_BC);
	// fix sites depending on _BC
	switch (this->dim)
	{
	case 1:
		this->Ly = 1; this->Lz = 1;
		this->nnForward = { 0 };
		this->nnnForward = { 0 };
		break;
	case 2:
		this->Lz = 1;
		this->nnForward = { 0, 1, 2 };
		this->nnnForward = { 0, 1, 2 };
		break;
	default:
		break;
	}

	// we take 2 * Ly because of the fact that we have two elements in one elementary cell always
	this->Ns = 2 * this->Lx * this->Ly * this->Lz;

	// neighbors
	this->calculate_nn();
	this->calculate_nnn();

	// coordinates
	this->calculate_coordinates();
	this->calculate_spatial_norm();


	this->a1 = arma::vec({ sqrt(3) * this->a / 2.0, 3 * this->a / 2.0, 0 });
	this->a2 = arma::vec({ -sqrt(3) * this->a / 2.0, 3 * this->a / 2.0, 0 });
	this->a3 = arma::vec({ 0, 0, this->c });

	this->kVec = arma::mat(this->Lx * this->Ly * this->Lz, 3, arma::fill::zeros);

	//! make vectors
	this->calculate_kVec();
}

// ------------------------------------------------------------- Getters -------------------------------------------------------------

/*
* @brief returns the nn for a given x direction at a given lattice site
*/
int HexagonalLattice::get_nn(int site, Lattice::direction d) const
{
	switch (d) {
	case X:
		return this->nn[site][0];
		break;
	case Y:
		return this->dim >= 2 ? this->nn[site][1] : this->nn[site][0];
		break;
	case Z:
		return this->dim == 3 ? this->nn[site][2] : this->nn[site][0];
		break;
	default:
		return this->nn[site][0];
		break;
	}
}

/*
* @brief returns the real space vector for a given multipliers of reciprocal vectors
*/
arma::vec HexagonalLattice::getRealVec(int x, int y, int z) const
{
	// elementary cell Y value (two atoms in each elementary cell)
	auto yFloor = std::floor(double(y) / 2.0);
	// how much should we move in y direction with a1 + a2 (each y / 4 gives additional movement in a1 + a2)
	auto yMove = std::floor(double(yFloor) / 2.0);

	// go in y direction
	arma::vec tmp = (yMove * (this->a1 + this->a2)) + (z * this->a3);
	tmp += (double)modEUC<int>(Y, 2) * this->a1;

	// go in x is direction, working for negative
	return tmp + x * (this->a1 - this->a2);
}

// ------------------------------------------------------------- nearest neighbors -------------------------------------------------------------

/*
* @brief Calculate the nearest neighbors with PBC
*/
void HexagonalLattice::calculate_nn_pbc()
{
	switch (this->dim)
	{
	case 1:
		// One dimension - just a chain of 2*Lx elems
		this->nn = v_2d<int>(this->Ns, v_1d<int>(2, 0));
		for (uint i = 0; i < Ns; i++) {
			// z bond only
			this->nn[i][0] = (i + 1) % this->Ns;										// this is the neighbor top
			this->nn[i][1] = modEUC<int>(i-1, this->Ns);							// this is the neighbor bottom
		}
		break;
	case 2:
		// Two dimensions 
		// numeration begins from the bottom as 0 to the second as 1 with lattice vectors move
		this->nn = v_2d<int>(Ns, v_1d<int>(3, 0));
		// over Lx
		for (int i = 0; i < Lx; i++) {
			// over big Y
			for (int j = 0; j < Ly; j++) {
				// current elements a corresponding to first site, b corresponding to the second one
				auto current_elem_a = 2 * i + 2 * Lx * j;								// lower
				auto current_elem_b = 2 * i + 2 * Lx * j + 1;							// upper

				// check the elementary cells
				auto up = modEUC<int>(j + 1, this->Ly);
				auto down = modEUC<int>(j - 1, this->Ly);
				auto right = modEUC<int>(i + 1, this->Lx);
				auto left = modEUC<int>(i - 1, this->Lx);

				// y and x bonding depends on current y level as the hopping between sites changes 
				
				auto y_bond_a = -1;
				auto y_bond_b = -1;
				auto x_bond_a = -1;
				auto x_bond_b = -1;
				if (modEUC(j, 2) == 0) {
					// right 
					// neighbor x does not change for a but y changes -> y_bond
					y_bond_a = (2 * i + 2 * down * this->Lx + 1);							// site b is the neighbor for a

					// left 
					// neighbor y does change for a and x changes -> x_bond
					x_bond_a = (2 * left + 2 * down * this->Lx + 1);							// site b is the neighbor for a

					// left 
					// neighbor x does change for a and y changes -> x_bond
					y_bond_b = (2 * left + 2 * up * this->Lx);
					// right 
					// neighbor x does not change for a but y changes -> y_bond
					x_bond_b = (2 * i + 2 * up * this->Lx);
				}
				else
				{
					// right a - x changes for a, y changes for a
					y_bond_a = (2 * right + 2 * down * this->Lx + 1);
					// left b - x does not change y changes
					y_bond_b = (2 * i + 2 * up * this->Lx);
					// left a - x does not change, y changes;
					x_bond_a = (2 * i + 2 * down * this->Lx + 1);
					// right b - x changes, y changes;
					x_bond_b = (2 * right + 2 * up * this->Lx);
				}

				// x bonding
				this->nn[current_elem_a][2] = x_bond_a;
				this->nn[current_elem_b][2] = x_bond_b;
				// y bonding
				this->nn[current_elem_a][1] = y_bond_a;
				this->nn[current_elem_b][1] = y_bond_b;
				// z bonding
				this->nn[current_elem_a][0] = current_elem_a + 1;
				this->nn[current_elem_b][0] = current_elem_b - 1;
			}
		}
		stout << this->nn << EL;
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
		// One dimension - just a chain of 2*Lx elems
		this->nn = v_2d<int>(this->Ns, v_1d<int>(2, 0));
		for (uint i = 0; i < this->Ns; i++) {
			// z bond only
			this->nn[i][0] = (i + 1) >= this->Ns ? i + 1 : -1;						// this is the neighbor top
			this->nn[i][1] = modEUC<int>(i - 1, this->Ns);						// this is the neighbor bottom
		}
		break;
	case 2:
		// Two dimensions 
		// numeration begins from the bottom as 0 to the second as 1 with lattice vectors move
		this->nn = std::vector<std::vector<int>>(Ns, std::vector<int>(3, -1));
		for (int i = 0; i < Lx; i++) {
			for (int j = 0; j < Ly; j++) {
				auto current_elem_a = 2 * i + 2 * Lx * j;
				auto current_elem_b = 2 * i + 2 * Lx * j + 1;

				auto up = j + 1;
				auto down = j - 1;
				auto right = i + 1;
				auto left = i - 1;

				auto y_bond_a = -1;
				auto y_bond_b = -1;
				auto x_bond_a = -1;
				auto x_bond_b = -1;
				if (modEUC(j, 2) == 0) {
					// right 
					// neighbor x does not change for a but y changes -> y_bond
					y_bond_a = down >= 0 ? (2 * i + 2 * down * Lx + 1) : -1;							// site b is the neighbor for a

					// left 
					// neighbor y does change for a and x changes -> x_bond
					x_bond_a = left >= 0 && down >= 0 ? (2 * left + 2 * down * Lx + 1) : -1;			// site b is the neighbor for a

					// left 
					// neighbor x does change for a and y changes -> x_bond
					y_bond_b = left >= 0 && up < Ly ? (2 * left + 2 * up * Lx) : -1;

					// right 
					// neighbor x does not change for a but y changes -> y_bond
					x_bond_b = up < Ly ? (2 * i + 2 * up * Lx) : -1;
				}
				else 
				{
					// right a - x changes for a, y changes for a
					y_bond_a = right < Lx && down >= 0 ? (2 * right + 2 * down * Lx + 1) : -1;
					// left b - x does not change y changes
					y_bond_b = up < Ly ? (2 * i + 2 * up * Lx) : -1;
					// left a - x does not change, y changes;
					x_bond_a = down >= 0 ? (2 * i + 2 * down * Lx + 1) : -1;
					// right b - x changes, y changes;
					x_bond_b = right < Lx && up < Ly ? (2 * right + 2 * up * Lx) : -1;
				}

				// x bonding
				this->nn[current_elem_a][2] = x_bond_a;
				this->nn[current_elem_b][2] = x_bond_b;
				// y bonding
				this->nn[current_elem_a][1] = y_bond_a;
				this->nn[current_elem_b][1] = y_bond_b;
				// z bonding
				this->nn[current_elem_a][0] = current_elem_a + 1;
				this->nn[current_elem_b][0] = current_elem_b - 1;
			}
		}
		// stout << this->nn << EL;
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}

/*
* @brief Calculate the nearest neighbors with MBC - WORKING HELLA FINE 2D
*/
void HexagonalLattice::calculate_nn_mbc()
{
	switch (this->dim)
	{
	case 1:
		// One dimension - just a chain of 2*Lx elems
		this->nn = v_2d<int>(this->Ns, v_1d<int>(2, 0));
		for (uint i = 0; i < Ns; i++) {
			// z bond only
			this->nn[i][0] = (i + 1) % Ns;							// this is the neighbor top
			this->nn[i][1] = modEUC<int>(i - 1, Ns);			// this is the neighbor bottom
		}
		break;
	case 2:
		// Two dimensions 
		// numeration begins from the bottom as 0 to the second as 1 with lattice vectors move
		this->nn = v_2d<int>(Ns, v_1d<int>(3, 0));
		// over Lx
		for (int i = 0; i < Lx; i++) {
			// over big Y
			for (int j = 0; j < Ly; j++) {
				// current elements a corresponding to first site, b corresponding to the second one
				auto current_elem_a = 2 * i + 2 * Lx * j;								// lower
				auto current_elem_b = 2 * i + 2 * Lx * j + 1;							// upper

				// check the elementary cells
				auto up = j + 1;
				auto down = j - 1;
				auto right = modEUC<int>(i + 1, Lx);
				auto left = modEUC<int>(i - 1, Lx);

				// y and x bonding depends on current y level as the hopping between sites changes 

				auto y_bond_a = -1;
				auto y_bond_b = -1;
				auto x_bond_a = -1;
				auto x_bond_b = -1;
				if (modEUC(j, 2) == 0) {
					// right 
					// neighbor x does not change for a but y changes -> y_bond
					y_bond_a = down >= 0 ? (2 * i + 2 * down * Lx + 1) : -1;							// site b is the neighbor for a

					// left 
					// neighbor y does change for a and x changes -> x_bond
					x_bond_a = left >= 0 && down >= 0 ? (2 * left + 2 * down * Lx + 1) : -1;			// site b is the neighbor for a

					// left 
					// neighbor x does change for a and y changes -> x_bond
					y_bond_b = left >= 0 && up < Ly ? (2 * left + 2 * up * Lx) : -1;

					// right 
					// neighbor x does not change for a but y changes -> y_bond
					x_bond_b = up < Ly ? (2 * i + 2 * up * Lx) : -1;
				}
				else 
				{
					// right a - x changes for a, y changes for a
					y_bond_a = right < Lx && down >= 0 ? (2 * right + 2 * down * Lx + 1) : -1;
					// left b - x does not change y changes
					y_bond_b = up < Ly ? (2 * i + 2 * up * Lx) : -1;
					// left a - x does not change, y changes;
					x_bond_a = down >= 0 ? (2 * i + 2 * down * Lx + 1) : -1;
					// right b - x changes, y changes;
					x_bond_b = right < Lx && up < Ly ? (2 * right + 2 * up * Lx) : -1;
				}

				// x bonding
				this->nn[current_elem_a][2] = x_bond_a;
				this->nn[current_elem_b][2] = x_bond_b;
				// y bonding
				this->nn[current_elem_a][1] = y_bond_a;
				this->nn[current_elem_b][1] = y_bond_b;
				// z bonding
				this->nn[current_elem_a][0] = current_elem_a + 1;
				this->nn[current_elem_b][0] = current_elem_b - 1;
			}
		}
		stout << this->nn << EL;
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}

/*
* @brief Calculate the nearest neighbors with SBC - WORKING HELLA FINE 2D
*/
void HexagonalLattice::calculate_nn_sbc()
{
	this->nn = v_2d<int>(this->Ns);
	switch (this->dim)
	{
	case 1:
		// One dimension - just a chain of 2*Lx elems
		for (uint i = 0; i < Ns; i++) {
			this->nn[i] = v_1d<int>(2, 0);
			// z bond only
			this->nn[i][0] = (i + 1) % Ns;												// this is the neighbor top
			this->nn[i][1] = modEUC<int>(i - 1, Ns);									// this is the neighbor bottom
		}
		break;
	case 2:
		// Two dimensions 
		// numeration begins from the bottom as 0 to the second as 1 with lattice vectors move
		this->nn = v_2d<int>(Ns, v_1d<int>(3, 0));
		// over Lx
		for (int i = 0; i < Lx; i++) {
			// over big Y
			for (int j = 0; j < Ly; j++) {
				// current elements a corresponding to first site, b corresponding to the second one
				auto current_elem_a = 2 * i + 2 * Lx * j;								// lower
				auto current_elem_b = 2 * i + 2 * Lx * j + 1;							// upper

				// check the elementary cells
				auto up = (int)(j + 1, Ly);
				auto down = modEUC<int>(j - 1, Ly);
				auto right = i + 1;
				auto left = i - 1;

				// y and x bonding depends on current y level as the hopping between sites changes 

				auto y_bond_a = -1;
				auto y_bond_b = -1;
				auto x_bond_a = -1;
				auto x_bond_b = -1;
				if (modEUC(j, 2) == 0) {
					// right 
					// neighbor x does not change for a but y changes -> y_bond
					y_bond_a = down >= 0 ? (2 * i + 2 * down * Lx + 1) : -1;							// site b is the neighbor for a

					// left 
					// neighbor y does change for a and x changes -> x_bond
					x_bond_a = left >= 0 && down >= 0 ? (2 * left + 2 * down * Lx + 1) : -1;			// site b is the neighbor for a

					// left 
					// neighbor x does change for a and y changes -> x_bond
					y_bond_b = left >= 0 && up < Ly ? (2 * left + 2 * up * Lx) : -1;

					// right 
					// neighbor x does not change for a but y changes -> y_bond
					x_bond_b = up < Ly ? (2 * i + 2 * up * Lx) : -1;
				}
				else
				{
					// right a - x changes for a, y changes for a
					y_bond_a = right < Lx&& down >= 0 ? (2 * right + 2 * down * Lx + 1) : -1;
					// left b - x does not change y changes
					y_bond_b = up < Ly ? (2 * i + 2 * up * Lx) : -1;
					// left a - x does not change, y changes;
					x_bond_a = down >= 0 ? (2 * i + 2 * down * Lx + 1) : -1;
					// right b - x changes, y changes;
					x_bond_b = right < Lx&& up < Ly ? (2 * right + 2 * up * Lx) : -1;
				}

				// x bonding
				this->nn[current_elem_a][2] = x_bond_a;
				this->nn[current_elem_b][2] = x_bond_b;
				// y bonding
				this->nn[current_elem_a][1] = y_bond_a;
				this->nn[current_elem_b][1] = y_bond_b;
				// z bonding
				this->nn[current_elem_a][0] = current_elem_a + 1;
				this->nn[current_elem_b][0] = current_elem_b - 1;
			}
		}
		stout << this->nn << EL;
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}
// ------------------------------------------------------------- next nearest neighbors -------------------------------------------------------------

/*
* @brief Calculate the next nearest neighbors with PBC
*/
void HexagonalLattice::calculate_nnn_pbc()
{
	this->nnn = v_2d<int>(this->Ns);
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
* @brief Calculate the next nearest neighbors with PBC
*/
void HexagonalLattice::calculate_nnn_obc()
{
	this->nnn = v_2d<int>(this->Ns);
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

// ------------------------------------------------------------- coordinates -------------------------------------------------------------

/*
* @brief Returns real space coordinates from a lattice site number
*/
void HexagonalLattice::calculate_coordinates()
{
	const int LxLy = Lx * Ly;
	this->coord = v_2d<int>(this->Ns, v_1d<int>(3, 0));
	// we must categorize elements by pairs
	for (uint i = 0; i < Ns; i++) {
		this->coord[i][0] = (int(1.0 * i / 2.0)) % Lx;						// x axis coordinate
		this->coord[i][1] = (int(1.0 * i / (2.0 * Lx))) % Ly;				// y axis coordinate
		this->coord[i][2] = (int(1.0 * i / (LxLy))) % Lz;					// z axis coordinate			

		// we calculate the big Y that is enumerated normally accordingly and then calculate the small y which is twice bigger or twice bigger + 1
		if (i % 2 == 0)
			this->coord[i][1] = this->coord[i][1] * 2;
		else
			this->coord[i][1] = this->coord[i][1] * 2 + 1;

		//stout << VEQ(i) << "->(" << this->coordinates[i][0] << "," << this->coordinates[i][1] << "," << this->coordinates[i][2] << ")\n";
	}


}

/*
* @brief calculates the matrix of all k vectors
*/
void HexagonalLattice::calculate_kVec()
{
	const auto two_pi_over_Lx = TWOPI / Lx / a;
	const auto two_pi_over_Ly = TWOPI / Ly / a;
	const auto two_pi_over_Lz = TWOPI / Lz / c;

	const arma::vec b1 = { 1. / sqrt(3), 1. / 3., 0 };
	const arma::vec b2 = { -1. / sqrt(3), 1. / 3., 0 };
	const arma::vec b3 = { 0, 0, 1 };


	for (int qx = 0; qx < Lx; qx++) {
		double kx = -PI + two_pi_over_Lx * qx;
		for (int qy = 0; qy < Ly; qy++) {
			double ky = -PI + two_pi_over_Ly * qy;
			for (int qz = 0; qz < Lz; qz++) {
				double kz = -PI + two_pi_over_Lz * qz;
				uint iter = qz * (Lx * Ly) + qy * Lx + qx;
				this->kVec.row(iter) = (kx * b1 + ky * b2 + kz * b3).st();
			}
		}
	}

}
