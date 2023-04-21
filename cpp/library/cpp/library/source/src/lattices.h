#pragma once 

/*******************************
* Contains the possible methods
* for general lattice class.
*******************************/

#ifndef COMMON_H
	#include "common.h"
#endif

#ifndef LATTICE_H
#define LATTICE_H

// ########################################################	GENERAL LATTICE ########################################################

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
enum LatticeTypes { SQ, HEX };							//%
														//%
BEGIN_ENUM(LatticeTypes)								//%
{														//%
	DECL_ENUM_ELEMENT(SQ),								//%
	DECL_ENUM_ELEMENT(HEX)								//%
}														//%
END_ENUM(LatticeTypes);									//%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
enum BoundaryConditions {	PBC = 0, OBC = 1,			//%
							MBC = 2, SBC = 3 };			//%
BEGIN_ENUM(BoundaryConditions)							//%
{														//%
	DECL_ENUM_ELEMENT(PBC),								//%
	DECL_ENUM_ELEMENT(OBC),								//%
	DECL_ENUM_ELEMENT(MBC),								//%
	DECL_ENUM_ELEMENT(SBC)								//%
}														//%
END_ENUM(BoundaryConditions);							//%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Pure virtual lattice class, it will allow to distinguish between different geometries in the models
*/
class Lattice {
protected:
	// ----------------------- LATTICE PARAMETERS
	BoundaryConditions _BC	= BoundaryConditions::PBC;		// boundary conditions 0 = PBC, 1 = OBC, 2 = MBC [PBC->x;OBC->y, OBC->z], 3 = SBC [OBC->x, PBC->y, OBC->z],
	LatticeTypes type_		= LatticeTypes::SQ;				// enum type of the lattice
	std::string type		= "";							// type of the lattice
	
	unsigned int dim		= 1;							// the dimensionality of the lattice 1,2,3
	unsigned int Ns			= 1;							// number of lattice sites
	
	// --- nn --- 
	v_2d<int> nn;											// vector of the nearest neighbors
	v_1d<uint> nnForward;									// number of nearest neighbors forward
	
	// --- nnn --- 
	v_2d<int> nnn;											// vector of the next nearest neighbors
	v_1d<uint> nnnForward;									// number of nearest neighbors forward
	
	// --- coords ---
	v_2d<int> coord;										// vector of real coordiates allowing to get the distance between lattice points
	v_3d<int> spatialNorm;									// norm for averaging over all spatial sites

	// reciprocal vectors
	arma::vec a1, a2, a3;									// base vectors of the lattice
	arma::mat kVec;											// allowed values of k - to be used in the lattice
public:
	enum direction {
		X, Y, Z
	};
	virtual ~Lattice() = default;

	// ----------------------- VIRTUAL GETTERS -----------------------
	virtual int get_Lx()									const = 0;
	virtual int get_Ly()									const = 0;
	virtual int get_Lz()									const = 0;
	t_3d<int> getSiteDifference(t_3d<int> i, uint j)		const;
	t_3d<int> getSiteDifference(uint i, uint j)				const;

	// ----------------------- GETTERS -----------------------
	virtual arma::vec getRealVec(int x, int y, int z)		const = 0;
	virtual int getNorm(int x, int y, int z)				const = 0;

	// ----------------------- FORWARD
	virtual uint get_nn_ForwardNum(int site, int num)		const = 0;
	virtual uint get_nnn_ForwardNum(int site, int num)		const = 0;
	virtual v_1d<uint> get_nn_ForwardNum(int site, v_1d<uint> p)	const = 0;														// with placeholder returns vector of nn
	virtual v_1d<uint> get_nnn_ForwardNum(int site, v_1d<uint> p)	const = 0;														// with placeholder returns vector of nnn
	virtual int get_nn(int site, direction d)				const = 0;																// retruns nn in a given direction x 
	
	// ----------------------- GETTERS NEI -----------------------
	auto get_nn_ForwardNum(int site)						const -> uint	{ return (uint)this->nnForward.size(); };				// with no placeholder returns number of nn
	auto get_nnn_ForwardNum(int site)						const -> uint	{ return (uint)this->nnnForward.size(); };				// with no placeholder returns number of nnn
	auto get_nn(int site, int nei_num)						const -> int	{ return this->nn[site][nei_num]; };					// returns given nearest nei at given lat site
	auto get_nnn(int site, int nei_num)						const -> int	{ return this->nnn[site][nei_num]; };					// returns given next nearest nei at given lat site
	auto get_nn(int site)									const -> uint	{ return (uint)this->nn[site].size(); };				// returns the number of nn
	auto get_nnn(int site)									const -> uint   { return (uint)this->nnn[site].size(); };				// returns the number of nnn
	auto get_nei(int lat_site, int corr_len)				const -> int;

	// ----------------------- GETTERS OTHER -----------------------
	BoundaryConditions get_BC()								const			{ return this->_BC; };									// returns the boundary conditions
	LatticeTypes get_Type()									const			{ return this->type_; };								// returns the type of the lattice as a string
	std::string get_type()									const			{ return this->type; };									// returns the type of the lattice as a string
	arma::mat get_kVec()									const			{ return this->kVec; };									// returns all k vectors in the RBZ
	arma::subview_row<double> get_kVec(uint row)							{ return this->kVec.row(row); };						// returns the given k vector row
	v_3d<int> get_spatial_norm()							const			{ return this->spatialNorm; };							// returns the spatial norm
	auto get_spatial_norm(int x, int y, int z)				const -> int    { return this->spatialNorm[x][y][z]; };					// returns the spatial norm at x,y,z
	auto get_coordinates(int site, direction axis)			const -> int	{ return this->coord[site][axis]; };					// returns the given coordinate
	auto get_Ns()											const -> uint	{ return this->Ns; };									// returns the number of sites
	auto get_Dim()											const -> uint	{ return this->dim; };									// returns dimension of the lattice
	auto get_info()											const -> std::string 
	{
		std::string _inf;
		strSeparatedP(_inf, ',', 3,
			this->type, 
			getSTR_BoundaryConditions(this->_BC), 
			VEQV(d, this->dim), VEQ(Ns),
			VEQV(Lx, this->get_Lx()), VEQV(Ly, this->get_Ly()), VEQV(Lz, this->get_Lz()));
		return _inf;
	};


	// ----------------------- CALCULATORS -----------------------
	void calculate_nn();
	void calculate_nnn();
	void calculate_spatial_norm();
	
	// ------ nn ------
	virtual void calculate_nn_pbc() = 0;
	virtual void calculate_nn_obc() = 0;
	virtual void calculate_nn_mbc() = 0;
	virtual void calculate_nn_sbc() = 0;
	
	// ------ nnn ------ 
	virtual void calculate_nnn_pbc() = 0;
	virtual void calculate_nnn_obc() = 0;
	
	// ------ coords ------ 
	virtual void calculate_coordinates() = 0;

	// ----------------------- SYMMETRY -----------------------
	virtual t_3d<int> getNumElems() = 0;																							// returns the number of elements if the symmetry is possible
	virtual t_3d<int> getSymPosInv(int x, int y, int z) = 0;																		// from symmetrised form return coordinates
	virtual t_3d<int> getSymPos(int x, int y, int z) = 0;																			// from given coordinates return their symmetrised form
	virtual bool symmetry_checker(int xx, int yy, int zz) = 0;

private:
	virtual void calculate_kVec() = 0;
};

/*
* @brief calculates the nearest neighbors
*/
inline void Lattice::calculate_nn() {
	switch (this->_BC)
	{
	case BoundaryConditions::PBC:
		this->calculate_nn_pbc();
		break;
	case BoundaryConditions::OBC:
		this->calculate_nn_obc();
		break;
	case BoundaryConditions::MBC:
		this->calculate_nn_mbc();
		break;
	case BoundaryConditions::SBC:
		this->calculate_nn_sbc();
		break;
	default:
		this->calculate_nn_pbc();
		break;
	}
	LOGINFO("Created NN. Using: " + SSTR(getSTR_BoundaryConditions(this->_BC)), LOG_TYPES::INFO, 2);
}

/*
* @brief calculates the next nearest neighbors
*/
inline void Lattice::calculate_nnn()
{
	switch (this->_BC)
	{
	case 0:
		this->calculate_nnn_pbc();
		break;
	case BoundaryConditions::OBC:
		this->calculate_nnn_obc();
		break;
	default:
		this->calculate_nnn_pbc();
		break;
	}
	LOGINFO("Created NNN. Using: " + SSTR(getSTR_BoundaryConditions(this->_BC)), LOG_TYPES::INFO, 2);
}

/*
* @brief calculates the spatial repetition of difference between the lattice sites considering _BC and enumeration
*/
inline void Lattice::calculate_spatial_norm()
{
	// spatial norm
	auto [x_n, y_n, z_n] = this->getNumElems();
	this->spatialNorm = SPACE_VEC(x_n, y_n, z_n, int);

	for (uint i = 0; i < this->Ns; i++) {
		for (uint j = 0; j < this->Ns; j++) {
			const auto [xx, yy, zz] = this->getSiteDifference(i, j);
			auto [a, b, c] = this->getSymPos(xx, yy, zz);
			spatialNorm[a][b][c]++;
		}
	}
}

/*
* @brief gets the neighbor from a given lat_site lattice site at corr_len length
*/
inline int Lattice::get_nei(int lat_site, int corr_len) const
{
	switch (this->_BC) 
	{
	case BoundaryConditions::PBC:
		return modEUC<int>(lat_site + corr_len, this->Ns);
		break;
	case BoundaryConditions::OBC:
		return uint(lat_site + corr_len) > this->Ns ? -1 : (lat_site + corr_len);
		break;
	default:
		return modEUC<int>(lat_site + corr_len, this->Ns);
	}
}

/*
* @brief Returns the real space difference between lattice site cooridinates given in ascending order.
* From left to right. Then second row left to right etc.
* @param i First coordinate
* @param j Second coordinate
* @return Three-dimensional tuple (vector of vec[i]-vec[j])
*/
inline t_3d<int> Lattice::getSiteDifference(t_3d<int> i, uint j) const
{
	const auto& [x1, y1, z1] = i;
	const int z = z1 - this->get_coordinates(j, Z);
	const int y = y1 - this->get_coordinates(j, Y);
	const int x = x1 - this->get_coordinates(j, X);
	// returns the site difference
	return std::tuple<int, int, int>(x, y, z);
}

/*
* @brief Returns the real space difference between lattice site cooridinates given in ascending order.
* From left to right. Then second row left to right etc.
* @param i First coordinate
* @param j Second coordinate
* @return Three-dimensional tuple (vector of vec[i]-vec[j])
*/
inline t_3d<int> Lattice::getSiteDifference(uint i, uint j) const
{
	const int z = this->get_coordinates(i, Z) - this->get_coordinates(j, Z);
	const int y = this->get_coordinates(i, Y) - this->get_coordinates(j, Y);
	const int x = this->get_coordinates(i, X) - this->get_coordinates(j, X);
	// returns the site difference
	return std::tuple<int, int, int>(x, y, z);
}

#endif // !LATTICE_H