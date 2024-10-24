#pragma once
/***********************************
* Defines the Hilbert space with the
* usage of Local and Global symmetry.
* It is a base for further Hamiltonian
* developments.
***********************************/

#ifndef HILBERT_H
#define HILBERT_H

#ifndef HILBERTSYM_H
	#include "../hilbert_sym.h"
#endif
#ifndef GLOBALSYM_H
	#include "global_symmetries.h"
#endif

#include <mutex>
#include <shared_mutex>

// ##########################################################################################################################################

namespace Hilbert 
{
	constexpr double SYM_NORM_THRESHOLD = 1e-6;

#ifndef HILBERT_EMPTY_CHECK
#define HILBERT_EMPTY_CHECK(NH, TODO)	if (Nh == 0) { \
											LOGINFO("Skipping creation due to signal or empty Hilbert space.", LOG_TYPES::FINISH, 2); TODO ;}																													
#endif

	template <typename _T>
	class HilbertSpace {
	private:
		void mappingKernel(u64 start, u64 stop, v_1d<u64>& mapThreaded, v_1d<_T>& normThreaded, int t);
		void mappingKernelRepr();

	protected:
		using MutexType										= std::shared_timed_mutex;
		using ReadLock										= std::shared_lock<MutexType>;
		using WriteLock										= std::unique_lock<MutexType>;
		
		using GSymV											= v_1d<GlobalSyms::GlobalSym>;
		using SymOpV										= v_1d<Operators::Operator<_T>>;
		using SymGV											= v_1d<std::pair<Operators::SymGenerators, int>>;
		using RPair											= std::pair<u64, _T>;
		using RPairV										= v_1d<std::pair<u64, _T>>;

		mutable MutexType Mutex;
		uint threadNum										= 1;				// get number of threads
		uint Ns												= 1;				// number of lattice sites
		uint Nhl											= 2;				// number of local possibilities
		uint Nint											= 1;				// number of fermionic modes such that total V=L^D*N_int
		u64 Nh												= 1;				// number of states in the Hilbert space	
		u64 NhFull											= 1;				// full Hilbert space
		std::shared_ptr<Lattice> lat;

		// --------------------------- SYMMERTIES ----------------------------
		GSymV symGroupGlobal_;													// stores the global symmetry group representatives
		SymOpV symGroup_;														// stores the local symmetry group representatives
		SymGV symGroupSec_;														// stores the local symmetry group and their sectors for convenience

		// ------------------------ NORM AND MAPPING  ------------------------
		v_1d<_T> normalization_								= {};				// stores the representative normalization
		v_1d<u64> mapping_									= {};				// stores the symmetry representative mapping
		v_1d<u64> fullMap_									= {};				// stores the map of the representatives to a Hilbert space without the global syms
		RPairV reprMap_										= {};				// stores the map from the Hilbert space to the corresponding representative index and value of return (optional)

	public:
		// -------------------------- CONSTRUCTORS ---------------------------
		~HilbertSpace()
		{
			LOGINFO("Hilbert space destructor called.", LOG_TYPES::INFO, 3);
			this->fullMap_.clear();
			this->mapping_.clear();
			this->normalization_.clear();
		}
		
		HilbertSpace() = default;

		/*
		* @brief Create the Hilbert space class allowing for creation of specific symmetry 
		* sectors or a general Hilbert space for the system.
		* @param _lat general lattice to be used for creation of the system
		* @param _gen vector of symmetry generators with their eigenvalues
		* @param _glob global symmetries 
		* @param _Nhl number of local degrees of freedom on a give lattice site (2 for spins, 4 for 1/2-spin fermions
		* @param _Nint number of fermionic modes for a given lattice site
		*/
		HilbertSpace(std::shared_ptr<Lattice> _lat,
			SymGV _gen											= {},
			GSymV _glob											= {},
			uint _Nhl											= 2, 
			uint _Nint											= 1,
			bool _genereateReprezentativesMap					= false)
			: Nhl(_Nhl), Nint(_Nint), lat(_lat)
		{
			this->Ns				=				this->lat->get_Ns();
			this->NhFull			=				(u64)std::pow(this->Nhl, this->Ns * this->Nint);

			// set symmetry elements
			this->symGroupGlobal_	=				_glob;
			this->normalization_	=				v_1d<_T>();
			this->symGroup_			=				v_1d<Operators::Operator<_T>>();
			this->mapping_			=				v_1d<u64>();
			this->reprMap_			=				v_1d<std::pair<u64, _T>>();

			auto _t					=				NOW;
			// initialize
			this->generateSymGroup(_gen);
			this->generateMapping();
			if (_genereateReprezentativesMap)
				this->mappingKernelRepr();

			if (this->Nh == this->NhFull)
				LOGINFO("Produced the full Hilbert space - no symmetries are used", LOG_TYPES::WARNING, 2);

			if (this->Nh <= 0)
				LOGINFO("No states in the Hilbert space", LOG_TYPES::WARNING, 2);
			LOGINFO(_t, "Hilbert Space Creator: " + this->getSymInfo(), 3);
		};

		/*
		* @brief Assign constructor
		*/
		HilbertSpace(const HilbertSpace<_T>& _H)
			:	Ns(_H.Ns), 
				Nhl(_H.Nhl), 
				Nint(_H.Nint), 
				Nh(_H.Nh), 
				NhFull(_H.NhFull), 
				lat(_H.lat),
				symGroupGlobal_(_H.symGroupGlobal_), 
				symGroup_(_H.symGroup_), 
				symGroupSec_(_H.symGroupSec_),
				normalization_(_H.normalization_), 
				mapping_(_H.mapping_), 
				fullMap_(_H.fullMap_),
				reprMap_(_H.reprMap_)
		{
				WriteLock lhs_lk(this->Mutex, std::defer_lock);
				ReadLock  rhs_lk(_H.Mutex	, std::defer_lock);
				std::lock(lhs_lk, rhs_lk);
		};

		/*
		* @brief Move constructor
		*/
		HilbertSpace(HilbertSpace<_T>&& _H)
			:	Ns(std::move(_H.Ns)), 
				Nhl(std::move(_H.Nhl)),
				Nint(std::move(_H.Nint)),
				Nh(std::move(_H.Nh)),
				NhFull(std::move(_H.NhFull)),
				lat(std::move(_H.lat)),
				symGroupGlobal_(std::move(_H.symGroupGlobal_)),
				symGroup_(std::move(_H.symGroup_)),
				symGroupSec_(std::move(_H.symGroupSec_)),
				normalization_(std::move(_H.normalization_)),
				mapping_(std::move(_H.mapping_)),
				fullMap_(std::move(_H.fullMap_)),
				reprMap_(std::move(_H.reprMap_))
		{
			WriteLock lhs_lk(this->Mutex, std::defer_lock);
			ReadLock  rhs_lk(_H.Mutex, std::defer_lock);
			std::lock(lhs_lk, rhs_lk);
		};
	
		// -------------------------- ASSIGN OPERATOR -------------------------
		HilbertSpace<_T>& operator=(const HilbertSpace<_T>& _H)
		{
			if (this != &_H) 
			{
				WriteLock lhs_lk(this->Mutex, std::defer_lock);
				ReadLock  rhs_lk(_H.Mutex	, std::defer_lock);
				std::lock(lhs_lk, rhs_lk);

				this->Ns				= _H.Ns;
				this->Nh				= _H.Nh;
				this->Nhl				= _H.Nhl;
				this->Nint				= _H.Nint;
				this->NhFull			= _H.NhFull;
				this->symGroupGlobal_	= _H.symGroupGlobal_;
				this->symGroupSec_		= _H.symGroupSec_;
				this->normalization_	= _H.normalization_;
				this->symGroup_			= _H.symGroup_;
				this->mapping_			= _H.mapping_;
				this->fullMap_			= _H.fullMap_;
				this->lat				= _H.lat;
			}
			return *this;
		};

		// ------------------------- INNER GENERATORS -------------------------
		void generateSymGroup(const v_1d<std::pair<Operators::SymGenerators, int>>& g);	// generates symmetry groups taking the comutation into account
		void generateMapping();															// generates mapping from reduced hilbert space to original
		void generateFullMap();															// generates full map if a global symmetry is present

		std::pair<u64, _T> findRep(u64 baseIdx)			const;							// returns the representative index and symmetry return eigval
		std::pair<u64, _T> findRep(u64 baseIdx, _T nB)	const;							// returns the representative and symmetry eigval taking the second symmetry sector beta

		// ------------------------ FULL HILBERT SPACE ------------------------

		arma::Col<_T> castToFull(const arma::Col<_T>& _s);

		// ----------------------------- GETTERS ------------------------------
		BoundaryConditions getBC()						const					{ return this->lat->get_BC();												};
		std::shared_ptr<Lattice> getLattice()			const					{ return this->lat;															};
		auto getLatticeSize()							const -> uint			{ return this->Ns;															};
		auto getLocalHilbertSize()						const -> uint			{ return this->Nhl;															};
		auto getFullHilbertSize()						const -> u64			{ return this->NhFull;														};
		auto getHilbertSize()							const -> u64			{ return this->Nh;															};
		auto getNum()									const -> uint			{ return this->Nint;														};
		auto getNorm()									const -> v_1d<_T>		{ return this->normalization_;												};
		auto getNorm(u64 k)								const -> _T				{ return this->normalization_.size() != 0 ? this->normalization_[k] : 1.0;	};
		auto getMapping()								const -> v_1d<u64>		{ return this->mapping_;													};
		auto getMapping(u64 k)							const -> u64			{ return this->mapping_.empty() ? k : this->mapping_[k];					};
		auto getRepr()									const -> RPairV			{ return this->reprMap_;													};
		auto getRepr(u64 k)								const -> RPair			{ if(!this->reprMap_.empty()) return this->reprMap_[k];						};
		auto getSymGroup()								const -> SymOpV			{ return this->symGroup_;													};
		auto getSymGroupGlob()							const -> GSymV			{ return this->symGroupGlobal_;												};

		v_1d<u64>		getFullMap()					const;					// returns the full map (taking global symmetries into account)
		arma::SpMat<_T> getSymRot()						const;					// returns the symmetry rotation matrix
		arma::SpMat<_T> getSymRot(const v_1d<u64>& fMap)const;					// returns the symmetry rotation matrix
		_T				getSymNorm(u64 baseIdx)			const;					// returns the symmetry normalization (how many states map into that one)
		std::string		getSymInfo()					const;

		// ----------------------------- CHECKERS -----------------------------
		bool			checkSym()						const					{ return !(this->Nh == this->NhFull);										};
		bool			checkLSym()						const					{ return this->symGroup_.size() != 0;										};
		bool			checkGSym()						const					{ return this->symGroupGlobal_.size() != 0;									};
		bool			checkU1()						const					{ for (const GlobalSyms::GlobalSym& g : this->symGroupGlobal_) if (g.getName() == GlobalSyms::GlobalSymGenerators::U1) return true; return false; };
		int				checkU1Val()					const					{ for (const GlobalSyms::GlobalSym& g : this->symGroupGlobal_) if (g.getName() == GlobalSyms::GlobalSymGenerators::U1) return (int)g.getVal(); return -INT_MAX; };
	};

	// ##########################################################################################################################################

	/*
	* @brief Based on the input symmetry generators, allows to create a local point symmetry group mix.
	* @param g enum types generators with corresponding sector value
	*/
	template<typename _T>
	inline void HilbertSpace<_T>::generateSymGroup(const v_1d<std::pair<Operators::SymGenerators, int>>& g)
	{
		// no symmetries! - there are no global and local symmetries to be used, therefore return
		if (g.empty() && !this->checkGSym())
			return;

		// copy the generators to modify them
		v_1d<std::pair<Operators::SymGenerators, int>> genIn = g;

		// find translation symmetry and check its sector for the parity correspondence
		Operators::Operator<_T> T	=			Operators::Operator<_T>(this->lat);
		bool containsT_				=			false;
		bool containsTCpx_			=			false;


		// ******** G L O B A L S ********
		// check if contains U(1)
		bool containsU1_			=			this->checkU1();

		// go through the local generators
		for (auto i = 0; i < genIn.size(); i++) {
			const auto [gen, sec] = genIn[i];
			
			// if it is a translation - proceed
			if (static_cast<Operators::SymGenerators>(gen) == Operators::SymGenerators::T)
			{
				// check if PBC, otherwise remove translation
				containsT_ = (this->lat->get_BC() == (uint)BoundaryConditions::PBC);

				// create the translation operator
				//if(containsT_)
				this->symGroupSec_.push_back(genIn[i]);
				T = Operators::symChoice<_T>(genIn[i], this->lat);

				// erease the translation as we will include it later on
				genIn.erase(genIn.begin() + i);

				//  say something
				if (!containsT_) {
					LOGINFOG("Not using translation due to the boundary conditions - BC", LOG_TYPES::WARNING, 1);
					continue;
				}
				
				// otherwise check sectors
				if ((sec != 0 && !(sec == this->Ns / 2 && this->Ns % 2 == 0)))
					containsTCpx_ = true;
				break;
			}
		}
		if (containsT_) {
			LOGINFOG("Using translation!", LOG_TYPES::INFO, 1);
			if(containsTCpx_)
				LOGINFOG("In complex sector!", LOG_TYPES::INFO, 2);
		}

		// remove reflection symmetry if we are in the complex sector - it doesn't commute then!
		for (auto i = 0; i < genIn.size(); i++)
		{
			const auto [gen, sec] = genIn[i];
			if (static_cast<Operators::SymGenerators>(gen) == Operators::SymGenerators::R && containsTCpx_) {
				genIn.erase(genIn.begin() + i);
				break;
			}
		}
		
		// check the existance of parity when U(1) is used
		if (containsU1_) 
		{
			uint removed = 0;
			LOGINFOG("Using U(1)!", LOG_TYPES::INFO, 1);
			for (auto i = 0; i < genIn.size(); i++)
			{
				const auto [gen, sec] = genIn[i];
				// remove if not in the half-filling or uneven system sizes
				if ((	static_cast<Operators::SymGenerators>(gen) == Operators::SymGenerators::PX	||
						static_cast<Operators::SymGenerators>(gen) == Operators::SymGenerators::PY) &&
						((this->checkU1Val() != this->Ns / 2) || (this->Ns % 2 != 0)))
				{
					genIn.erase(genIn.begin() + (i - removed));
					removed++;
					LOGINFOG("Removing parity in X and/or Y direction", LOG_TYPES::INFO, 1);
				}
			}
		}
		
		// save all for convenience
		for(auto& g : genIn)
			this->symGroupSec_.push_back(g);

		// --------------------------------------------------------------------------------
		LOGINFO(2);
		LOGINFO(LOG_TYPES::INFO, "", 40, '%', 0);
		LOGINFOG("Using local: ", LOG_TYPES::INFO, 0);
		for (auto& g : genIn) 
		{
			const auto [gen, sec] = g;
			LOGINFOG(SSTR(Operators::getSTR_SymGenerators(gen)) + ":" + VEQ(sec), LOG_TYPES::INFO, 1);
		}
		if (containsT_)
			LOGINFOG(SSTR(Operators::getSTR_SymGenerators(Operators::SymGenerators::T)) + ":" + VEQ(T.getVal()), LOG_TYPES::INFO, 1);

		LOGINFOG("Using global: ", LOG_TYPES::INFO, 0);
		for (auto& g : this->symGroupGlobal_) 
		{
			LOGINFOG(SSTR(GlobalSyms::getSTR_GlobalSymGenerators(g.getName())) + ":" + VEQ(g.getVal()), LOG_TYPES::INFO, 1);
		}
		LOGINFO(LOG_TYPES::INFO, "", 40, '%', 0);
		LOGINFO(2);

		// add neutral element
		// this->symGroup_.push_back(Operators::Operator<_T>(this->lat));

		// go through all of the combinations
		const auto SIZE_GEN = genIn.size();
		for (auto i = 0; i <= SIZE_GEN; i++)
		{
			std::string bitmask(i, 1);              	// i leading 1's
			bitmask.resize(SIZE_GEN, 0);   				// N - i trailing 0's

			// go through all bitmask permutations corresponding to using those specific operator combinations
			do
			{
				// start with a neutral operator
				Operators::Operator<_T> OP_(this->lat);
				for (int i = 0; i < SIZE_GEN; ++i)		// [0..N-1] integers
				{
					// current permutation contains the usage of this operator make it's combination
					if (bitmask[i])
					{
						// create operator based on the type
						auto _OP = Operators::symChoice<_T>(genIn[i], this->lat);
						OP_ = OP_ % _OP;
					}
				}
				this->symGroup_.push_back(OP_);
			} while (std::prev_permutation(bitmask.begin(), bitmask.end())); // loop over all combinations with bitmask
		}

		// handle the translation - the translation operator multiplies all combination of symmetry group operators
		if (containsT_) 
		{
			v_1d<Operators::Operator<_T>> symGroupT_ = this->symGroup_;
			auto Tp = T;
			for (uint i = 1; i < this->lat->get_Ns(); i++) {
				// add combinations with other symmetries
				for (auto Go : symGroupT_)
					this->symGroup_.push_back(Tp % Go);
				// move next translation
				Tp = Tp % T;
			}
		}
	}

	// ##########################################################################################################################################

	/*
	* @brief Creates the information string about the Hilbert space and symmetries
	* @returns string containing the information about all the symmetries
	*/
	template<typename _T>
	inline std::string Hilbert::HilbertSpace<_T>::getSymInfo() const
	{
		std::string tmp = ",";
		if(this->checkLSym())
			// start with local symmetries
			for (auto& g : this->symGroupSec_) {
				auto& [gen, val] = g;
				tmp += SSTR(Operators::getSTR_SymGenerators(static_cast<Operators::SymGenerators>(gen)));
				tmp += "=";
				tmp += STR(val);
				tmp += ",";
			}
		if(this->checkGSym())
			// start wit global symmetries
			for (const GlobalSyms::GlobalSym& g : this->symGroupGlobal_) {
				auto name = g.getName();
				auto val = g.getVal();
				tmp += SSTR(GlobalSyms::getSTR_GlobalSymGenerators(static_cast<GlobalSyms::GlobalSymGenerators>(name)));
				tmp += "=";
				tmp += STRP(val, 2);
				tmp += ",";
			}

		// remove last ","
		if(!tmp.empty())
			tmp.pop_back();
		
		// return the name string
		return tmp;
	}

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ############################################################ REPRESENTATIVES #############################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################

	/*
	* @brief Find representatives of other state by using various combinations of the symmetries. We aim to look for the smallest 
	* representative possible generated from the usage of symmetry generators combinations. We find also the value of returning to that state.
	* @param @base_idx current base vector index to act on it with symmetry generators
	* @returns pair containing the representative index and symmetry eigenvalue connected with returning to it
	*/
	template<typename _T>
	inline std::pair<u64, _T> Hilbert::HilbertSpace<_T>::findRep(u64 baseIdx) const
	{
		// if the map exists, return the value already saved
		if (!this->reprMap_.empty())
			return this->reprMap_[baseIdx];

		// start with a biggest value possible
		u64 SEC = INT64_MAX;
		// setup starting symmetry eigenvalue - nothing needs to be done
		_T val	= 1.0;
		// go through all symmetry generators to find the smallest baseIdx - representative
		for (const auto& G : this->symGroup_) 
		{
			// act!
			auto [state, retVal] = G(baseIdx);

			// check if state is smaller
			if (state < SEC) {
				SEC = state;
				val = retVal;
			}
		}
		return std::make_pair(SEC, val);
	}

	// ##########################################################################################################################################

	/*
	* @brief Finds the representative for a given baseIdx in sector_alfa. The usage of this procedure is handy when acting with matrix
	* on a representative |\\bar n>. Then, the matrix in general transforms |\\bar n>->|m> and we look for its representative |\\bar m>, to which the
	* distance needs to be calculated.
	* It means that we have a base idx in sector alfa on the right side 
	* and we look for the representative after acting on it. Then, we take the normalisation in sector beta. 
	* @warning (the same is used for creating the Hamiltonian with beta = alfa)
	* @returns representative binary number and eigenvalue from symmetries to return to that state from baseIdx
	*/
	template<typename _T>
	inline std::pair<u64, _T> Hilbert::HilbertSpace<_T>::findRep(u64 baseIdx, _T nB) const
	{
		// check if mapping exists, if not - return the same state and 1.0
		if (this->mapping_.empty())
			return std::make_pair(baseIdx, 1.0);

		// if the map exists, return the value already saved
		if (!this->reprMap_.empty())
		{
			auto [idx, symEig] = this->reprMap_[baseIdx];
			return std::make_pair(idx, this->normalization_[idx] / nB * algebra::conjugate(symEig));
		}

		// find representative already in the mapping (can be that the matrix element already changes the state to the representative)
		u64 idx = binarySearch(this->mapping_, 0, static_cast<ull>(this->Nh) - 1, baseIdx);

		// if is in range (so has been found in the mapping)
		if (idx < this->mapping_.size()) return std::make_pair(idx, this->normalization_[idx] / nB);

		// need to find the representative by acting
		auto [min, symEig] = this->findRep(baseIdx);
		idx = binarySearch(this->mapping_, 0, static_cast<ull>(this->Nh) - 1, min);

		// if is in range
		if (idx < this->mapping_.size()) return std::make_pair(idx, this->normalization_[idx] / nB * algebra::conjugate(symEig));

		// haven't found the representative - different block sector
		return std::make_pair(u64(0), 0.0);
	}

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ############################################################# N O R M A L S ##############################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################

	/*
	* @brief From applying symmetry operators the function finds the normalisation for a given state. The
	* normalization constitutes of finding the sum of generator eigenvalues for a given symmetry that return to the same state.
	* (Connected to the period of a given state)
	* @param base_idx current base vector index to act on with symmetries
	* @returns The value of the normalization.
	* @link Based on Prof. Laeuchli lecture
	*/
	template<typename _T>
	inline _T Hilbert::HilbertSpace<_T>::getSymNorm(u64 baseIdx) const
	{
		_T norm = 0.0;
		for (auto& G : this->symGroup_) {
			// if we return to the same state by acting with symmetry group operators
			const auto [newIdx, val] = G(baseIdx);
			if (newIdx == baseIdx)
				norm += val;
		}
		return std::sqrt(norm);
	}

	// ##########################################################################################################################################

	/*
	* @brief Creates a symmetry rotation matrix that reproduces the full Hiblert state (without the local symmetries included)
	* @param fMap full mapping between current Hilber space and the full Hilbert space (usefull when having global symmetries), otherwise empty
	* @returns the rotation matrix that transforms to the full Hilbert space (without the local symmetries included)
	*/
	template<typename _T>
	inline arma::SpMat<_T> HilbertSpace<_T>::getSymRot(const v_1d<u64>& fMap) const
	{
		// check the maximal dimension of the Hilbert space (if we have global symmetries)
		const u64 maxDim	= fMap.empty() ? this->NhFull : fMap.size();

		// find index helping function
		auto find_index		= [&](u64 idx) { return (!fMap.empty()) ? binarySearch(fMap, 0, maxDim - 1, idx) : idx; };

		// generate sparse mapping
		arma::SpMat<_T> U(maxDim, this->Nh);

		// iterate states
		auto symSize = this->symGroup_.size();

		// if no symmetries return empty
		if (symSize == 0)
			return U;

		for (u64 k = 0; k < this->Nh; k++)

			for (auto& G : this->symGroup_) 
			{
				// find new corresponding state from the generators
				auto [idx, val] = G(this->mapping_[k]);
				// apply the mapping
				auto idxM = find_index(idx);
				// use only if exists in sector
				if (idxM < maxDim)
					U(idxM, k) += algebra::conjugate(val / (this->normalization_[k] * std::sqrt(double(symSize))));
			}
		return U;
	}

	/*
	* @brief Creates a symmetry rotation matrix that reproduces the full Hiblert state. Generates the full map if necessary (global symmetries check).
	*/
	template<typename _T>
	inline arma::SpMat<_T> HilbertSpace<_T>::getSymRot() const
	{
		// check the maximal dimension of the Hilbert space (if we have global symmetries)
		v_1d<u64> fMap		= (this->symGroupGlobal_.empty()) ? v_1d<u64>() : this->getFullMap();
		const u64 maxDim	= fMap.empty() ? this->NhFull : fMap.size();

		// find index helping function
		std::function<u64(u64)> find_index;
		if(fMap.empty()) 
			find_index = [&](u64 idx) { return idx; };
		else
			find_index = [&](u64 idx) { return binarySearch(fMap, 0, maxDim - 1, idx); };

		// generate sparse mapping
		arma::SpMat<_T> U(maxDim, this->Nh);

		// iterate states
		double symSize = (double)this->symGroup_.size();

		// if no symmetries return empty
		if (symSize == 0)
			return U;

		// out of the given state, generates the full state in the Hilbert space
		for (u64 k = 0; k < this->Nh; k++)
		{
			for (auto& G : this->symGroup_) 
			{
				// find new corresponding state from the generators acting on the representative
				auto [idx, val] =	G(this->mapping_[k]);
				// apply the mapping
				idx				=	find_index(idx);
				// use only if exists in sector
				if (idx < maxDim)
					U(idx, k)	+=	algebra::conjugate(val / (this->normalization_[k] * std::sqrt(symSize)));
			}
		}
		return U;
	}

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ############################################################# M A P P I N G ##############################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################

	/*
	* @brief For a given range of states in a full Hilbert space, get a transformation to equivalence class representatives (ECR).
	* This means that we make it by acting on each basis state with all symmetry generators.
	*/
	template<typename _T>
	inline void HilbertSpace<_T>::mappingKernel(u64 start, u64 stop, v_1d<u64>& mapThreaded, v_1d<_T>& normThreaded, int t)
	{
		// go through each state
		for (u64 j = start; j < stop; j++) 
		{
			// check all global conservation
			bool globalChecker	= true;
			// go through all global symmetries to check if the basis state exists in this sector
			for (auto& Glob : this->symGroupGlobal_)
				globalChecker	= globalChecker && Glob(j);

			// if the global symmetry is not conserved, the state for sure is not in the basis
			if (!globalChecker)
				continue;

			// check the representative
			const auto [SEC, _] = this->findRep(j);
			
			// if this state is the smallest already, it for sure is the representative, otherwise we have already used it
			if (SEC == j) 
			{
				// normalisation condition -- check if state is in the basis (if zero we cannot use it)
				_T N = getSymNorm(j);
				if (std::abs(N) > SYM_NORM_THRESHOLD) 
				{
					mapThreaded.push_back(j);
					normThreaded.push_back(N);
				}
			}
		}
	}

	/*
	* @brief For all states in full Hilbert space, get a transformation to equivalence class representatives (ECR) and save it for later.
	*/
	template<typename _T>
	inline void HilbertSpace<_T>::mappingKernelRepr()
	{
		LOGINFO("Creating the map of representatives", LOG_TYPES::INFO, 1);
		// go through each state
#pragma omp parallel for num_threads(this->threadNum)
		for (u64 j = 0; j < this->NhFull; j++) 
		{
			u64 idx						= INT64_MAX;
			// check all global conservation
			bool globalChecker			= true;
			// go through all global symmetries to check if the basis state exists in this sector
			for (auto& Glob : this->symGroupGlobal_)
				globalChecker			= globalChecker && Glob(j);

			// if the global symmetry is not conserved, the state for sure is not in the basis
			if (!globalChecker)
			{
				this->reprMap_.push_back(std::make_pair(INT64_MAX, 0.0));
				continue;
			}
			// already in the map
			idx							= binarySearch(this->mapping_, 0, static_cast<ull>(this->Nh) - 1, j);
			if (idx < this->mapping_.size())
			{
				this->reprMap_.push_back(std::make_pair(idx, 1.0));
				continue;
			}

			// otherwise check the representative
			const auto [SEC, symEig]	= this->findRep(j);
			idx							= binarySearch<u64>(this->mapping_, 0, static_cast<ull>(this->Nh) - 1, SEC);

			// if is in range
			if (idx < this->mapping_.size()) 
				this->reprMap_.push_back(std::make_pair(idx, algebra::conjugate(symEig)));
			else
				this->reprMap_.push_back(std::make_pair(INT64_MAX, 0.0));
		}
		LOGINFO("Finished the map of representatives", LOG_TYPES::INFO, 2);
	}

	/*
	* @brief Splits the mapping genertation onto threads, where each finds basis states in the reduced Hilbert space within a given range.
	* The mapping is retrieved by concatenating the resulting maps from each thread.
	*/
	template<typename _T>
	inline void HilbertSpace<_T>::generateMapping()
	{
		// if no symmetries we don't need mapping
		if (this->symGroupGlobal_.empty() && this->symGroup_.empty())
		{ 
			this->Nh	= this->NhFull;
			return;
		}

		u64 start		= 0;
		u64 powNs		= NhFull;
		u64 stop		= powNs;
#ifndef DEBUG
		int numThreads = this->threadNum;
#else
		int numThreads = 1;
#endif // !DEBUG

		// if we can thread it then we do!
		if (numThreads != 1) {
			//Threaded
			v_2d<u64> mapThreaded(numThreads);
			v_2d<_T> normThreaded(numThreads);
			v_1d<std::thread> threads;

			// -------- reserve threads --------
			threads.reserve(numThreads);
			for (auto t = 0; t < numThreads; t++) 
			{
				start	= (u64)(powNs / (double)numThreads * t);
				stop	= ((t + 1) == numThreads ? powNs : u64(powNs / (double)numThreads * (double)(t + 1)));

				mapThreaded[t]	= v_1d<u64>();
				normThreaded[t] = v_1d<_T>();
				threads.emplace_back(&HilbertSpace<_T>::mappingKernel, this, start, stop, std::ref(mapThreaded[t]), std::ref(normThreaded[t]), t);
			}

			// -------- join the threads together --------
			for (auto& thread : threads)
				thread.join();

			for (auto& t : mapThreaded)
				this->mapping_.insert(this->mapping_.end(), std::make_move_iterator(t.begin()), std::make_move_iterator(t.end()));

			for (auto& t : normThreaded)
				this->normalization_.insert(this->normalization_.end(), std::make_move_iterator(t.begin()), std::make_move_iterator(t.end()));
		}
		else
			this->mappingKernel(start, stop, this->mapping_, this->normalization_, 0);

		// set the Hilbert space size
		this->Nh = this->mapping_.size();
	}

	// ##########################################################################################################################################
	// ##########################################################################################################################################
	// ############################################################ F U L L  M A P ##############################################################
	// ##########################################################################################################################################
	// ##########################################################################################################################################

	/*
	* @brief If the global symmetry exists, creates a transformation map to the base without it. Otherwise returns nothing. This means that 
	* for each representative, we have a full Hilbert space state in the map.
	* Hiblert space.
	* @returns Full map to the basis without the global symmetries. 
	*/
	template<typename _T>
	inline v_1d<u64> HilbertSpace<_T>::getFullMap() const 
	{
		// if the fullMap alread
		if (!this->fullMap_.empty()) return this->fullMap_;

		v_1d<u64> fullMap = {};
		if (!this->symGroupGlobal_.empty())
		{
			for (u64 j = 0; j < this->NhFull; j++) 
			{
				// check globals
				bool globalChecker = true;
				for (auto& Glob : this->symGroupGlobal_)
					globalChecker = globalChecker && Glob(j);

				// if is in, go on with that state
				if (globalChecker)
					fullMap.push_back(j);
			}
		}
		return fullMap;
	}

	/*
	* @brief Generates full mapping to the full Hilbert space using global symmetries.
	*/
	template<typename _T>
	inline void Hilbert::HilbertSpace<_T>::generateFullMap()
	{
		this->fullMap_ = {};
		if (!this->symGroupGlobal_.empty())
		{
			WriteLock lock(this->Mutex);
			LOGINFOG("Creating full global symmetry map!", LOG_TYPES::INFO, 1);
			for (u64 j = 0; j < this->NhFull; j++) 
			{
				// check globales
				bool globalChecker = true;
				for (auto& Glob : this->symGroupGlobal_)
					globalChecker = globalChecker && Glob(j);
				// if is in, goomb
				if (globalChecker)
					this->fullMap_.push_back(j);
			}
		}
	}

	// ##########################################################################################################################################

	/*
	* @brief Calculates the cast to the full Hilbert space when global symmetry is present. The state will have mostly zeros outside the globals.
	* @param _s state to be transformed
	* @returns if global symmetry is present, we transform the state; otherwise we return the same state. 
	*/
	template<typename _T>
	inline arma::Col<_T> Hilbert::HilbertSpace<_T>::castToFull(const arma::Col<_T>& _s)
	{
		if (!this->checkGSym())
			return _s;
		else 
		{
			// make a placeholder to the full Hilbert space
			arma::Col<_T> fS(this->NhFull, arma::fill::zeros);

			// if not exist, generate a full map
			if (this->fullMap_.empty()) 
				this->generateFullMap();

			// create new vector state
			for (int i = 0; i < this->fullMap_.size(); i++)
				fS(this->fullMap_[i]) = _s(i);
			return fS;
		}
	}

	// ##########################################################################################################################################

};


#endif // !SYMMETRIES_H