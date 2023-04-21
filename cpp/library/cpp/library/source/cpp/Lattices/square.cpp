#include "../../src/Lattices/square.h"

// ############################################################################################################################################

/*
* @brief Constructor for the square lattice
*/
SquareLattice::SquareLattice(int Lx, int Ly, int Lz, int dim, int _BC)
	: Lx(Lx), Ly(Ly), Lz(Lz)
{
	this->dim		=		dim;
	this->_BC		=		static_cast<BoundaryConditions>(_BC);
	this->type_		=		LatticeTypes::SQ;
	this->type		=		SSTR(getSTR_LatticeTypes(this->type_));

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
		this->nnForward = { 0,1 };
		this->nnnForward = { 0,1 };
		break;
	case 3:
		this->nnForward = { 0,1,2 };
		this->nnnForward = { 0,1,2 };
		break;
	default:
		break;
	}
	this->Ns = this->Lx * this->Ly * this->Lz;

	// neighbors
	this->calculate_nn();
	this->calculate_nnn();

	// coordinates
	this->calculate_coordinates();
	this->calculate_spatial_norm();

	this->a1 = { this->a, 0, 0 };
	this->a2 = { 0, this->b, 0 };
	this->a3 = { 0, 0, this->c };

	// calculate k_space vectors
	this->kVec = arma::mat(this->Ns, 3, arma::fill::zeros);
	this->calculate_kVec();
	LOGINFO("Created " + this->type + " lattice", LOG_TYPES::INFO, 1);
	LOGINFO(this->get_info(), LOG_TYPES::TRACE, 2);
}

// ------------------------------------------------------------- Getters -------------------------------------------------------------

/*
* @brief returns the nn for a given x direction at a given lattice site
*/
int SquareLattice::get_nn(int site, Lattice::direction d) const
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

// ------------------------------------------------------------- nearest neighbors -------------------------------------------------------------

/*
* @brief Calculate the nearest neighbors with PBC
*/
void SquareLattice::calculate_nn_pbc()
{
	this->nn = v_2d<int>(this->Ns);
	switch (this->dim)
	{
	case 1:
		// One dimension 
		for (auto i = 0; i < Lx; i++) {
			this->nn[i] = v_1d<int>(2, 0);
			this->nn[i][0] = modEUC<int>(i + 1, Lx);								// right
			this->nn[i][1] = modEUC<int>(i - 1, Lx);								// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		for (uint i = 0; i < this->Ns; i++) {
			this->nn[i] = v_1d<int>(4, 0);
			this->nn[i][0] = int(1.0 * i / Lx) * Lx + modEUC<int>(i + 1, Lx);		// right
			this->nn[i][1] = modEUC<int>(i + Lx, Ns);								// top
			this->nn[i][2] = int(1.0 * i / Lx) * Lx + modEUC<int>(i - 1, Lx);		// left
			this->nn[i][3] = modEUC<int>(i - Lx, Ns);								// bottom
		}
		break;
	case 3:
		// Three dimensions
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		for (uint i = 0; i < Ns; i++) {
			this->nn[i] = v_1d<int>(6, 0);
			int x = i % Lx;
			int y = int(1.0 * i / Lx) % Ly;
			int z = int(1.0 * i / Lx / Ly) % Lz;
			this->nn[i][0] = z * Lx * Ly + y * Lx + modEUC<int>(i + 1, Lx);			// right - x
			this->nn[i][1] = z * Lx * Ly + modEUC<int>(i + Lx, Lx * Ly);			// right - y
			this->nn[i][2] = modEUC<int>(i + Lx * Ly, Ns);							// right - z

			this->nn[i][3] = z * Lx * Ly + y * Lx + modEUC<int>(i - 1, Lx);			// left - x
			this->nn[i][4] = z * Lx * Ly + modEUC<int>(i - Lx, Lx * Ly);			// left - y
			this->nn[i][5] = modEUC<int>(i - Lx * Ly, Ns);							// left - z
		}
		break;
	default:
		break;
	}
}

// ############################################################################################################################################

/*
* @brief Calculate the nearest neighbors with OBC
*/
void SquareLattice::calculate_nn_obc()
{
	this->nn = v_2d<int>(this->Ns);
	switch (this->dim)
	{
	case 1:
		//* One dimension 
		for (auto i = 0; i < Lx; i++) {
			this->nn[i] = v_1d<int>(2, 0);
			this->nn[i][0] = (i + 1) >= Lx ? -1 : i + 1;								// right
			this->nn[i][1] = (i - 1) < 0 ? -1 : i - 1;									// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		for (int i = 0; i < (int)this->Ns; i++) {
			this->nn[i] = v_1d<int>(4, 0);
			int x = i % Lx;
			int y = int(1.0 * i / Lx) % Ly;
			this->nn[i][0] = (i + 1) < (y + 1) * Lx ? i + 1 : -1;						// right
			this->nn[i][1] = i + Lx < (int)this->Ns ? i + Lx : -1;						// top
			this->nn[i][2] = (i - 1) >= y * Lx ? i - 1 : -1;							// left
			this->nn[i][3] = i - Lx >= 0 ? i - Lx : -1;									// bottom
		}
		break;
	case 3:
		// Three dimensions
		// numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour
		for (int i = 0; i < (int)Ns; i++) {
			this->nn[i] = v_1d<int>(6, 0);
			int x = i % Lx;
			int y = int(1.0 * i / Lx) % Ly;
			int z = int(1.0 * i / Lx / Ly) % Lz;
			this->nn[i][0] = z * Lx * Ly + y * Lx + (i + 1 < (z * Lx * Ly + (y + 1) * Lx) ? i + 1 : -1);				// right - x
			this->nn[i][1] = z * Lx * Ly + (i + Lx < ((z + 1)* Lx* Ly) ? i + Lx : -1);									// right - y
			this->nn[i][2] = i + Lx * Ly < (int)this->Ns ? i + Lx * Ly : -1;											// right - z

			this->nn[i][3] = z * Lx * Ly + y * Lx + (i - 1 >= (z * Lx * Ly + y * Lx) ? i - 1 : -1);						// left - x
			this->nn[i][4] = z * Lx * Ly + (i - Lx >= (z * Lx * Ly) ? i - Lx : -1);										// left - y
			this->nn[i][5] = i - Lx * Ly >= 0 ? i - Lx * Ly : -1;														// left - z
		}
	default:
		break;
	}
}

// ############################################################################################################################################

/*
* !TODO
* @brief Calculate the nearest neighbors with MBC [PBC->x;OBC->y] TODEFINE
*/
void SquareLattice::calculate_nn_mbc()
{
	this->nn = v_2d<int>(this->Ns);
	switch (this->dim)
	{
	case 1:
		//* One dimension 
		for (int i = 0; i < this->Lx; i++) {
			this->nn[i] = v_1d<int>(2, 0);
			this->nn[i][0] = (i + 1) >= Lx ? -1 : i + 1;							// right
			this->nn[i][1] = (i - 1) == 0 ? -1 : i - 1;								// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		for (int i = 0; i < (int)this->Ns; i++) {
			this->nn[i] = v_1d<int>(4, 0);
			int x = i % Lx;
			int y = int(1.0 * i / Lx) % Ly;
			this->nn[i][0] = (i + 1) < (y + 1) * Lx ? y * Lx + x + 1 : -1;			// right
			this->nn[i][1] = i + Lx < (int)this->Ns ? i + Lx : -1;					// top
			this->nn[i][2] = (i - 1) >= y * Lx ? y * Lx + x - 1 : -1;				// left
			this->nn[i][3] = i - Lx >= 0 ? i - Lx : -1;								// bottom
		}
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}

// ############################################################################################################################################

/*
* @brief Calculate the nearest neighbors with SBC [OBC->x;PBC->y,] TODEFINE
*/
void SquareLattice::calculate_nn_sbc()
{
	this->nn = v_2d<int>(this->Ns);
	switch (this->dim)
	{
	case 1:
		//* One dimension 
		for (int i = 0; i < Lx; i++) {
			this->nn[i] = v_1d<int>(2, 0);
			this->nn[i][0] = (i + 1) >= Lx ? -1 : i + 1;							// right
			this->nn[i][1] = (i - 1) == 0 ? -1 : i - 1;								// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		for (uint i = 0; i < Ns; i++) {
			this->nn[i] = v_1d<int>(2, 0);
			int x = i % Lx;
			int y = static_cast<int>(1.0 * i / Lx) % Ly;
			//this->nn[i][0] = (i + 1) < (y + 1) * Lx ? y * Lx + x + 1 : -1;		// right
			//this->nn[i][1] = i + Lx < Ns ? i + Lx : -1;							// top
			//this->nn[i][2] = (i - 1) >= y * Lx ? y * Lx + x - 1 : -1;				// left
			//this->nn[i][3] = i - Lx >= 0 ? i - Lx : -1;							// bottom
		}
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
void SquareLattice::calculate_nnn_pbc()
{
	this->nnn = v_2d<int>(this->Ns);
	switch (this->dim)
	{
	case 1:
		/* One dimension */
		for (int i = 0; i < this->Lx; i++) {
			this->nnn[i] = v_1d<int>(2, 0);
			this->nnn[i][0] = modEUC<int>(i + 2, Lx);									// right
			this->nnn[i][1] = modEUC<int>(i - 2, Lx);									// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		for (uint i = 0; i < Ns; i++) {
			this->nnn[i] = v_1d<int>(4, 0);
			this->nnn[i][0] = int(1.0 * i / Lx) * Lx + modEUC<int>(i + 2, Lx);		// right
			this->nnn[i][1] = modEUC<int>(i + 2 * Lx, Ns);							// top
			this->nnn[i][2] = int(1.0 * i / Lx) * Lx + modEUC<int>(i - 2, Lx);		// left
			this->nnn[i][3] = modEUC<int>(i - 2 * Lx, Ns);							// bottom
		}
		break;
	case 3:
		// Three dimensions
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		for (uint i = 0; i < this->Ns; i++) {
			this->nnn[0] = v_1d<int>(6, 0);
			int x = i % Lx;
			int y = static_cast<int>(1.0 * i / Lx) % Ly;
			int z = static_cast<int>(1.0 * i / Lx / Ly) % Lz;
			this->nnn[i][0] = z * Lx * Ly + y * Lx + modEUC<int>(i + 2, Lx);		// right - x
			this->nnn[i][1] = z * Lx * Ly + modEUC<int>(i + 2 * Lx, Lx * Ly);		// right - y
			this->nnn[i][2] = modEUC<int>(i + 2 * Lx * Ly, Ns);						// right - z

			this->nnn[i][3] = z * Lx * Ly + y * Lx + modEUC<int>(i - 2, Lx);		// left - x
			this->nnn[i][4] = z * Lx * Ly + modEUC<int>(i - 2 * Lx, Lx * Ly);		// left - y
			this->nnn[i][5] = modEUC<int>(i - 2 * Lx * Ly, Ns);						// left - z
		}
		break;
	default:
		break;
	}
}

/*
* @brief Calculate the next nearest neighbors with OBC
*/
void SquareLattice::calculate_nnn_obc()
{
	this->nnn = v_2d<int>(this->Ns);
	switch (this->dim)
	{
	case 1:
		/* One dimension */
		for (int i = 0; i < Lx; i++) {
			this->nnn[i] = v_1d<int>(2, 0);
			this->nnn[i][0] = (i + 2) >= Lx ? -1 : i + 2;										// right
			this->nnn[i][1] = (i - 2) < 0 ? -1 : i - 2;										// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		break;
	case 3:
		// Three dimensions
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		break;
	default:
		break;
	}
}

// ------------------------------------------------------------- coordinates -------------------------------------------------------------

/*
* @brief Returns real space coordinates from a lattice site number
*/
void SquareLattice::calculate_coordinates()
{
	const int LxLy = Lx * Ly;
	this->coord = v_2d<int>(this->Ns, v_1d<int>(3, 0));
	for (int i = 0; i < (int)this->Ns; i++) {
		this->coord[i][0] = i % Lx;												// x axis coordinate
		this->coord[i][1] = (static_cast<int>(1.0 * i / Lx)) % Ly;				// y axis coordinate
		this->coord[i][2] = (static_cast<int>(1.0 * i / LxLy)) % Lz;			// z axis coordinate			
	}
}

// ############################################################################################################################################

/*
* @brief calculates all the kVec for a square lattice
*/
void SquareLattice::calculate_kVec()
{
	const auto two_pi_over_Lx = TWOPI / a / Lx;
	const auto two_pi_over_Ly = TWOPI / b / Ly;
	const auto two_pi_over_Lz = TWOPI / c / Lz;

	for (int qx = 0; qx < Lx; qx++) {
		double kx = -PI + two_pi_over_Lx * qx;
		for (int qy = 0; qy < Ly; qy++) {
			double ky = -PI + two_pi_over_Ly * qy;
			for (int qz = 0; qz < Lz; qz++) {
				double kz = -PI + two_pi_over_Lz * qz;
				uint iter = qz * (Lx * Ly) + qy * Lx + qx;
				this->kVec.row(iter) = { kx, ky, kz };
			}
		}
	}
}
