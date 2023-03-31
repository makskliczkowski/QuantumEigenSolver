#pragma once

/*******************************
* Definitions for the operators 
* and Hiblert space that handles
* the symmetries etc.
*******************************/

#include "../source/src/common.h"
#include "../source/src/lattices.h"
#include "../source/src/binary.h"

/*
* @brief Implemented symmetry types
*/
enum SymGenerators { T, R, PX, PY, PZ };

// ##########################################################################################################################################

namespace Operators {

	/*
	* @brief A class describing the local operator acting on specific states
	*/
	template<typename _T>
	class Operator {
		typedef std::function<std::pair<u64, _T>(u64)> repType;					// type returned for representing, what it does with state and value it returns
		std::shared_ptr<Lattice> lat_;											// lattice type to be used later on
		_T eigVal_ = 1.0;														// eigenvalue for symmetry generator (if there is inner value)
		repType fun_;															// function allowing to use symmetry

	public:

		Operator(std::shared_ptr<Lattice>& _lat, _T _eigVal, repType&& _fun)
			: lat_(_lat), eigVal_(_eigVal), fun_(_fun) {
			this->init();
		};
		//Operator(std::shared_ptr<Lattice>& _lat, _T _eigVal, repType _fun)
		//	: lat_(_lat), eigVal_(_eigVal), fun_(_fun)							{ this->init(); };
		Operator(const Operator<_T>& o)
			: eigVal_(o.eigVal_), fun_(o.fun_), lat_(o.lat_) {};
		Operator(Operator<_T>&& o)
			: eigVal_(std::move(o.eigVal_)), fun_(std::move(o.fun_)), lat_(std::move(o.lat_)) {};

		// ---------- static ----------
		static auto E(u64 s)							-> std::pair<u64, _T>	{ std::make_pair(s, _T(1.0)); };

		// ---------- virtual functions to override ----------
		virtual void init() {};

		// ---------- override operators -----------
		virtual auto operator()(u64 state) const		-> std::pair<u64, _T>	{ const auto [s, val] = this->fun_(state); return std::make_pair(s, eigVal_ * val); };
		virtual auto operator()(u64 state)				-> std::pair<u64, _T>	{ const auto [s, val] = this->fun_(state); return std::make_pair(s, eigVal_ * val); };

		// ---------- SETTERS -----------
		auto setFun(const repType& _fun) -> void { this->fun_ = _fun; };
		auto setFun(repType&& _fun) -> void { this->fun_ = std::move(_fun); };

		// ---------- OPERATORS JOIN ----------
		template<typename _T1, typename _T2>
		friend Operator<_T2> operator*(const Operator<_T1>& A, const Operator<_T2>& B) {
			Operator<_T2> op(A);
			auto _fun = [A, B](u64 s) {
				auto [s1, v1] = A(s);
				auto [s2, v2] = B(s1);
				return std::make_pair(s2, v1 * v2);
			};
			op.setFun(std::move(_fun));
			return op;
		};

		// ---------- OPERATORS CAST ----------
		template <class _TOut>
		operator Operator<_TOut>() {
			auto _fun = [&](u64 s) {
				const auto [s1, v1] = this->fun_(s);
				return std::make_pair(s1, _TOut(v1));
			};
			Operator<_TOut> op(this->lat_,
				static_cast<_TOut>(this->eigVal_),
				_fun);
			return op;
		}
	};

	/*
	* @brief generates translation symmetry
	* @param kx K sector in X direction
	* @param ky K sector in Y direction
	* @param kz K sector in Z direction
	* @param dim dimension of translation
	* @param base base of the Hilbert space
	*/
	template <typename _T>
	Operator<_T> makeTranslation(std::shared_ptr<Lattice> lat, int kx, int ky = 0, int kz = 0, int8_t dim = 1, uint base = 2) {
		auto Kx = TWOPI * kx / double(lat->get_Lx());
		auto Ky = TWOPI * ky / double(lat->get_Ly());
		auto Kz = TWOPI * kz / double(lat->get_Lz());

		std::function<std::pair<u64, _T>(u64)> cyclicShift;						// cyclic shift function with boundary preservation
		_T val = 1.0;															// exponent used as generator
		// create lambda function
		switch (dim) {
		case 1:
			cyclicShift = std::function(
				[lat, Kx, Ky, Kz](u64 state)
				{
					return std::make_pair(rotateLeft(state, lat->get_Ns()), cpx(1.0, 0.0));
				}
			);
			val = std::exp(I * cpx(Kx));
			break;
		case 2:
			cyclicShift = std::function(
				[lat, Kx, Ky, Kz](u64 state)
				{
					auto tmpState = state;
					for (auto i = 0; i < lat->get_Lx(); i++)
						tmpState = rotateLeft(state, lat->get_Ns());
					return std::make_pair(tmpState, cpx(1.0, 0.0));
				}
			);
			val = std::exp(I * cpx(Ky));
			break;
		case 3:
			cyclicShift = std::function(
				[lat, Kx, Ky, Kz](u64 state)
				{
					auto tmpState = state;
					for (auto i = 0; i < lat->get_Lx() * lat->get_Ly(); i++)
						tmpState = rotateLeft(state, lat->get_Ns());
					return std::make_pair(tmpState, cpx(1.0, 0.0));
				}
			);
			val = std::exp(I * cpx(Kz));
			break;
		};
		return Operator<_T>(lat, val, std::move(cyclicShift));
	}

	template <>
	Operator<double> makeTranslation(std::shared_ptr<Lattice> lat, int kx, int ky, int kz, int8_t dim, uint base) {
		auto Kx = TWOPI * kx / double(lat->get_Lx());
		auto Ky = TWOPI * ky / double(lat->get_Ly());
		auto Kz = TWOPI * kz / double(lat->get_Lz());

		std::function<std::pair<u64, double>(u64)> cyclicShift;						// cyclic shift function with boundary preservation
		double val = 1.0;															// exponent used as generator
		// create lambda function
		switch (dim) {
		case 1:
			cyclicShift = std::function(
				[lat, Kx, Ky, Kz](u64 state)
				{
					return std::make_pair(rotateLeft(state, lat->get_Ns()), 1.0);
				}
			);
			val = std::real(std::exp(I * cpx(Kx)));
			break;
		case 2:
			cyclicShift = std::function(
				[lat, Kx, Ky, Kz](u64 state)
				{
					auto tmpState = state;
					for (auto i = 0; i < lat->get_Lx(); i++)
						tmpState = rotateLeft(state, lat->get_Ns());
					return std::make_pair(tmpState, 1.0);
				}
			);
			val = std::real(std::exp(I * cpx(Ky)));
			break;
		case 3:
			cyclicShift = std::function(
				[lat, Kx, Ky, Kz](u64 state)
				{
					auto tmpState = state;
					for (auto i = 0; i < lat->get_Lx() * lat->get_Ly(); i++)
						tmpState = rotateLeft(state, lat->get_Ns());
					return std::make_pair(tmpState, 1.0);
				}
			);
			val = std::real(std::exp(I * cpx(Kz)));
			break;
		};
		return Operator<double>(lat, val, std::move(cyclicShift));
	}

	/*
	* @brief Reflection symmetry
	*/
	template <typename _T>
	Operator<_T> makeReflection(std::shared_ptr<Lattice> lat, int sec, int base = 2) {
		auto reflect = [lat, base](u64 state) 
		{ 
			return std::make_pair(revBits(state, lat->get_Ns(), base),
									_T(1.0)); 
		};
		return Operator<_T>(lat, _T(sec), std::function<std::pair<u64, _T>(u64)>(reflect));
	}

	/*
	* @brief Parity with \sigma^x
	*/
	template <typename _T>
	Operator<_T> makeFlipX(std::shared_ptr<Lattice> lat, int sec) {
		auto flipX = [lat](u64 state) 
		{
			return std::make_pair(flipAll(state, lat->get_Ns()),
									_T(1.0)); 
		};
		return Operator<_T>(lat, _T(sec), std::function<std::pair<u64, _T>(u64)>(flipX));
	}

	/*
	* @brief Parity with \sigma^y
	*/
	template <typename _T>
	Operator<_T> makeFlipY(std::shared_ptr<Lattice> lat, int sec) {
		auto flipY = [lat](u64 state)
		{
			int spinUps = lat->get_Ns() - __builtin_popcountll(state);
			return std::make_pair(flipAll(state, lat->get_Ns()),
									_T((spinUps % 2 == 0 ? 1.0 : -1.0) * std::pow(I, lat->get_Ns())));
		};
		return Operator<_T>(lat, _T(sec), std::function<std::pair<u64, _T>(u64)>(flipY));
	}

	/*
	* @brief Parity with \sigma^z
	*/
	template <typename _T>
	Operator<_T> makeFlipZ(std::shared_ptr<Lattice> lat, int sec) {
		auto flipZ = [lat](u64 state)
		{
			int spinUps = lat->get_Ns() - __builtin_popcountll(state);
			return std::make_pair(state, 
									_T(spinUps % 2 == 0 ? 1.0 : -1.0));
		};
		return Operator<_T>(lat, _T(sec), std::function<std::pair<u64, _T>(u64)>(flipZ));
	}
};

// ##########################################################################################################################################

namespace GlobalSyms {
	
	class GlobalSym {
		typedef std::function<bool(u64, double)> repType;						// type returned for checking 
	protected:
		repType check_;															// function that check global sym
	public:
		double val_;															// value connected to global symmetry
		std::shared_ptr<Lattice> lat_;											// lattice type to be used later on

		// constructors
		GlobalSym(std::shared_ptr<Lattice> _lat) : val_(0), lat_(_lat) {};

		// ---------- SETTERS -----------
		auto setFun(const repType& _fun)				-> void					{ this->check_ = _fun; };
		auto setFun(repType&& _fun)						-> void					{ this->check_ = std::move(_fun); };

		// ---------- CHECKER OVERLOAD ------------

		virtual auto operator()(u64 state) const		-> bool					{ return this->check_(state, val_); };
		virtual auto operator()(u64 state)				-> bool					{ return this->check_(state, val_); };

		// ---------- check the symmetry existence ----------
		auto check(u64 state, bool outCond) const		-> bool { return this->check_(state, val_) && outCond; };
	};

	/*
	* @brief describes the global check of U(1) symmetry
	*/
	auto U1Sym(u64 _state, double _val) -> bool { return __builtin_popcountll(_state) == _val; };
};

// ##########################################################################################################################################

namespace Hilbert {
	constexpr int SYM_NORM_THRESHOLD					=						1e-4;

	template <typename _T>
	class HilbertSpace {
	private:
		void mappingKernel(u64 start, u64 stop, v_1d<u64>& mapThreaded, v_1d<_T> normThreaded, int t);

	protected:
		uint threadNum									= 1;					// get number of threads
		uint Ns											= 1;					// number of lattice sites
		uint Nhl										= 2;					// number of local possibilities
		uint Nint										= 1;					// number of fermionic modes such that total V=L^D*N_int
		u64 Nh											= 1;					// number of states in the Hilbert space	
		u64 NhFull										= 1;					// full Hilbert space
		std::shared_ptr<Lattice> lat;

		// ------------------------ symmetries ------------------------
		v_1d<GlobalSyms::GlobalSym> symGroupGlobal_;							// stores the global symmetry group representatives
		v_1d<Operators::Operator<_T>> symGroup_;								// stores the local symmetry group representatives

		// ------------------------ symmetry normalization and mapping to a reduced Hilbert space ------------------------
		v_1d<_T> normalization_;												// stores the representative normalization
		v_1d<u64> mapping_;														// stores the symmetry representative mapping

	public:

		// ------------------------ constructors etc -------------------------

		/*
		* @brief
		* @param _lat
		* @param _gen
		* @param _glob
		* @param _Nhl
		* @param _Nint
		*/
		HilbertSpace(std::shared_ptr<Lattice> _lat,
			v_1d<SymGenerators> _gen = {},
			v_1d<GlobalSyms::GlobalSym> _glob = {},
			uint _Nhl = 2, uint _Nint = 1)
			: lat(_lat), Nhl(_Nhl), Nint(_Nint)
		{
			this->Ns = this->lat->get_Ns();
			this->NhFull = std::pow(this->Nhl, this->Ns * this->Nint);

			// set symmetry elements
			this->symGroupGlobal_ = v_1d<GlobalSyms::GlobalSym>();
			this->normalization_ = v_1d<_T>();
			this->symGroup_ = v_1d<Operators::Operator<_T>>();
			this->mapping_ = v_1d<u64>();

			// initialize
			this->generateSymGroup(_gen);
			this->generateMapping();

			if (this->Nh == this->NhFull)
				stoutd("\t\t\t->Produced the full Hilbert space\n");

			// no states in Hilbert space
			if (this->Nh <= 0)
				stoutd("\t\t\t->No states in the Hilbert space\n");
		}

		// ------------------------ inner generators -------------------------
		void generateSymGroup(v_1d<SymGenerators> gen);							// generates symmetry groups taking the comutation into account
		void generateMapping();													// generates mapping from reduced hilbert space to original

		std::pair<u64, _T> findRep(u64 baseIdx)			const;					// returns the representative index and symmetry return eigval
		std::pair<u64, _T> findRep(u64 baseIdx, _T nB)	const;					// returns the representative and symmetry eigval taking the second symmetry sector beta

		// ------------------------ getters ------------------------
		u64 getLatticeSize()							const					{ return this->Ns; };
		u64 getHilbertSize()							const					{ return this->Nh; };
		u64 getFullHilbertSize()						const					{ return this->NhFull; };
		u64 getLocalHilbertSize()						const					{ return this->Nhl; };
		std::shared_ptr<Lattice> getLattice()			const					{ return this->lat; };
		auto getNorm()									const -> v_1d<_T>		{ return this->normalization_; };
		auto getSymGroup()								const -> v_1d<Operators::Operator<_T>>	{ return this->symGroup_; };
		auto getSymGroupGlob()							const -> v_1d<GlobalSyms::GlobalSym>	{ return this->symGroupGlobal_; };
		
		v_1d<u64> getFullMap()							const;					// returns the full map taking global symmetries into account
		arma::SpMat<_T> getSymRot()						const;					// returns the symmetry rotation matrix
		arma::SpMat<_T> getSymRot(const v_1d<u64>& fMap)const;					// returns the symmetry rotation matrix
		_T getSymNorm(u64 baseIdx)						const;					// returns the symmetry normalization
		
	};

	// ##########################################################################################################################################

	/*
	* @brief Find representatives of other EC generated by reflection, spin-flip and (reflection x spin-flip) symmetry
	* @param @base_idx current base vector index to act on with symmetries
	*/
	template<typename _T>
	inline std::pair<u64, _T> Hilbert::HilbertSpace<_T>::findRep(u64 baseIdx) const
	{
		u64 SEC = INT64_MAX;
		_T val = 1.0;
		for (const Operators::Operator<_T>& G : this->symGroup_) {
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
	* @brief finds the representative for a given baseIdx in sector_alfa normalisation potentailly from other symmetry sector beta
	* @warning (the same is used creating the Hamiltonian with beta = alfa)
	* @returns representative binary number and eigenvalue from symmetries to return to that state from baseIdx
	*/
	template<typename _T>
	inline std::pair<u64, _T> Hilbert::HilbertSpace<_T>::findRep(u64 baseIdx, _T nB) const
	{
		if (this->mapping_.empty())
			return std::make_pair(baseIdx, 1.0);

		// found representative already in the mapping
		u64 idx = binarySearch(this->mapping_, 0, static_cast<ull>(this->Ns) - 1, baseIdx);

		// if is in range
		if (idx < static_cast<u64>(this->Ns * this->Nint)) return std::make_pair(idx, this->normalization_[idx] / nB);

		// need to find the representative by acting
		auto [min, symEig] = this->findRep(baseIdx);
		idx = binarySearch(this->mapping_, 0, static_cast<ull>(this->Ns) - 1, min);

		// if is in range
		if (idx < static_cast<u64>(this->Ns * this->Nint)) return std::make_pair(idx, this->normalization_[idx] / nB * std::conj(symEig));

		// haven't found the representative - different block sector
		return std::make_pair(u64(0), 0.0);
	}

	template <>
	inline std::pair<u64, double> Hilbert::HilbertSpace<double>::findRep(u64 baseIdx, double nB) const
	{
		if (this->mapping_.empty())
			return std::make_pair(baseIdx, 1.0);

		// found representative already in the mapping
		u64 idx = binarySearch(this->mapping_, 0, static_cast<ull>(this->Ns) - 1, baseIdx);

		// if is in range
		if (idx < this->Ns) return std::make_pair(idx, this->normalization_[idx] / nB);

		// need to find the representative by acting
		auto [min, symEig] = this->findRep(baseIdx);
		idx = binarySearch(this->mapping_, 0, static_cast<ull>(this->Ns) - 1, min);

		// if is in range
		if (idx < this->Ns) return std::make_pair(idx, this->normalization_[idx] / nB * symEig);

		// haven't found the representative - different block sector
		return std::make_pair(u64(0), 0.0);
	}

	// ##########################################################################################################################################

	/*
	* @brief From applying symmetry operators the function finds the normalisation for a given state
	* @param base_idx current base vector index to act on with symmetries
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

	template<>
	inline double Hilbert::HilbertSpace<double>::getSymNorm(u64 baseIdx) const
	{
		double norm = 0.0;
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
	* @brief Creates a symmetry rotation matrix that reproduces the full Hiblert state
	* @param fMap full mapping between current Hilber space and the full Hilbert space (usefull when having global symmetries), otherwise empty
	*/
	template<typename _T>
	inline arma::SpMat<_T> HilbertSpace<_T>::getSymRot(const v_1d<u64>& fMap) const
	{
		// check the maximal dimension of the Hilbert space (if we have global symmetries)
		const u64 maxDim = fMap.empty() ? ULLPOW(this->Ns) : fMap.size();
		
		// find index helping function
		auto find_index = [&](u64 idx) { return (!fMap.empty()) ? binarySearch(fMap, 0, maxDim - 1, idx) : idx; };
		
		// generate sparse mapping
		arma::SpMat<_T> U(maxDim, this->Nh);
		
		// iterate states
		auto symSize = this->symGroup_.size();

		// if no symmetries return empty
		if (symSize == 0)
			return U;

		for (u64 k = 0; k < this->Nh; k++)
			for (auto& G : this->symGroup_) {
				// find new corresponding state from the generators
				auto [idx, val] = G(this->mapping_[k]);
				// apply the mapping
				auto idxM = find_index(idx);
				// use only if exists in sector
				if (idxM < maxDim)
					U(idxM, k) += std::conj(val / (this->normalization_[k] * std::sqrt(double(symSize))));
			}
		return U;
	}

	template<>
	inline arma::SpMat<double> HilbertSpace<double>::getSymRot(const v_1d<u64>& fMap) const
	{
		// check the maximal dimension of the Hilbert space (if we have global symmetries)
		const u64 maxDim = fMap.empty() ? ULLPOW(this->Ns) : fMap.size();
		
		// find index helping function
		auto find_index = [&](u64 idx) { return (!fMap.empty()) ? binarySearch(fMap, 0, maxDim - 1, idx) : idx; };
		
		// generate sparse mapping
		arma::SpMat<double> U(maxDim, this->Nh);

		// iterate states
		auto symSize = this->symGroup_.size();

		// if no symmetries return empty
		if (symSize == 0)
			return U;

		for (u64 k = 0; k < this->Nh; k++)
			for (auto& G : this->symGroup_) {
				// find new corresponding state from the generators
				auto [idx, val] = G(this->mapping_[k]);
				// apply the mapping
				idx = find_index(idx);
				// use only if exists in sector
				if (idx < maxDim)
					U(idx, k) += (val / (this->normalization_[k] * std::sqrt(double(symSize))));
			}
		return U;
	}

	/*
	* @brief Creates a symmetry rotation matrix that reproduces the full Hiblert state
	*/
	template<typename _T>
	inline arma::SpMat<_T> HilbertSpace<_T>::getSymRot() const
	{
		// check the maximal dimension of the Hilbert space (if we have global symmetries)
		v_1d<u64> fMap = (this->symGroupGlobal_.empty()) ? v_1d<u64>() : this->getFullMap();
		const u64 maxDim = fMap.empty() ? ULLPOW(this->Ns) : fMap.size();

		// find index helping function
		auto find_index = [&](u64 idx) { return (!fMap.empty()) ? binarySearch(fMap, 0, maxDim - 1, idx) : idx; };

		// generate sparse mapping
		arma::SpMat<_T> U(maxDim, this->Nh);

		// iterate states
		auto symSize = this->symGroup_.size();

		// if no symmetries return empty
		if (symSize == 0)
			return U;

		for (u64 k = 0; k < this->Nh; k++)
			for (auto& G : this->symGroup_) {
				// find new corresponding state from the generators
				auto [idx, val] = G(this->mapping_[k]);
				// apply the mapping
				idx = find_index(idx);
				// use only if exists in sector
				if (idx < maxDim)
					U(idx, k) += std::conj(val / (this->normalization_[k] * std::sqrt(double(symSize))));
			}
		return U;
	}

	template<>
	inline arma::SpMat<double> HilbertSpace<double>::getSymRot() const
	{
		// check the maximal dimension of the Hilbert space (if we have global symmetries)
		v_1d<u64> fMap = (this->symGroupGlobal_.empty()) ? v_1d<u64>() : this->getFullMap();
		const u64 maxDim = fMap.empty() ? this->NhFull : fMap.size();

		// find index helping function
		auto find_index = [&](u64 idx) { return (!fMap.empty()) ? binarySearch(fMap, 0, maxDim - 1, idx) : idx; };

		// generate sparse mapping
		arma::SpMat<double> U(maxDim, this->Nh);

		// iterate states
		auto symSize = this->symGroup_.size();
		
		// if no symmetries return empty
		if (symSize == 0)
			return U;

		for (u64 k = 0; k < this->Nh; k++)
			for (auto& G : this->symGroup_) {
				// find new corresponding state from the generators
				auto [idx, val] = G(this->mapping_[k]);
				// apply the mapping
				idx = find_index(idx);
				// use only if exists in sector
				if (idx < maxDim)
					U(idx, k) += (val / (this->normalization_[k] * std::sqrt(double(symSize))));
			}
		return U;
	}

	// ##########################################################################################################################################

	template<typename _T>
	inline void HilbertSpace<_T>::mappingKernel(u64 start, u64 stop, v_1d<u64>& mapThreaded, v_1d<_T> normThreaded, int t)
	{
		for (u64 j = start; j < stop; j++) {

			// check all global conservation
			bool globalChecker = true;
			for (auto& Glob : this->symGroupGlobal_)
				globalChecker = globalChecker & Glob(j);

			// if is not conserved
			if (!globalChecker)
				continue;

			// check the representative
			const auto [SEC, _] = this->findRep(j);
			if (SEC == j) {
				// normalisation condition -- check if state in basis
				_T N = getSymNorm(j);
				if (std::abs(N) > SYM_NORM_THRESHOLD) {
					mapThreaded.push_back(j);
					normThreaded.push_back(N);
				}
			}
		}
	}

	/*
	* @brief  Splits the mapping onto threads, where each finds basis states in the reduced Hilbert space within a given range.
	* The mapping is retrieved by concatenating the resulting maps from each thread
	*/
	template<typename _T>
	inline void HilbertSpace<_T>::generateMapping()
	{
		// if no symmetries we don't need mapping
		if (this->symGroupGlobal_.empty() && this->symGroup_.empty())
			return;

		u64 start					=					0;
		u64 powNs					=					static_cast<u64>(std::pow(this->Nhl, this->Ns * this->Nint));
		u64 stop					=					powNs;
#ifndef DEBUG
		int numThreads				=					this->threadNum;
#else
		int numThreads				=					1;
#endif // !DEBUG

		// if we can thread it then we do!
		if (numThreads != 1) {
			//Threaded
			v_2d<u64> mapThreaded(numThreads);
			v_2d<_T> normThreaded(numThreads);
			v_1d<std::thread> threads;

			// -------- reserve threads --------
			threads.reserve(numThreads);
			for (auto t = 0; t < numThreads; t++) {
				start				=					(u64)(powNs / (double)numThreads * t);
				stop				=					((t + 1) == numThreads ? powNs : u64(powNs / (double)numThreads * (double)(t + 1)));

				mapThreaded[t]		=					v_1d<u64>();
				normThreaded[t]		=					v_1d<_T>();
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
	
	/*
	* @brief returns the ful mapping if the global symmetry exists, otherwise returns nothing
	*/
	template<typename _T>
	inline v_1d<u64> HilbertSpace<_T>::getFullMap() const {
		v_1d<u64> fullMap = {};
		if(!this->symGroupGlobal_.empty())
			for (u64 j = 0; j < this->Nh; j++) {
				// check globales
				bool globalChecker = true;
				for (auto& Glob : this->symGroupGlobal_)
					globalChecker = globalChecker & Glob(j);
				// if is in, goomb
				if (globalChecker)
					fullMap.push_back(j);
			}
		return fullMap;
	}

};