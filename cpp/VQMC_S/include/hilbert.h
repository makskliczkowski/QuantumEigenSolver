#pragma once
#include "../source/src/common.h"
#include "../source/src/lattices.h"

/*
* @brief A class describing the local operator acting on specific states
*/
template<typename _T>
class Operator {
	typedef std::function<std::pair<u64, _T>(u64)> repType;									// type returned for representing 

public:
	_T eigVal_;																				// eigenvalue for symmetry generator
	repType fun_;																			// function allowing to use symmetry
	std::shared_ptr<Lattice> lat_;															// lattice type to be used later on

	Operator(std::shared_ptr<Lattice>& _lat, _T _eigVal, repType&& _fun) 
		: lat_(_lat), eigVal_(_eigVal), fun_(_fun){ this->init(); };
	Operator(const Operator<_T>& o) 
		: eigVal_(o.eigVal_), fun_(o.fun_), lat_(o.lat_) {};
	Operator(Operator<_T>&& o) 
		: eigVal_(std::move(o.eigVal_)), fun_(std::move(o.fun_)), lat_(std::move(o.lat_)) {};
	
	// ---------- virtual functions to override ----------
	virtual void init() {};

	// ---------- override operators -----------
	virtual std::pair<u64, _T> operator()(u64 state) const	{ return this->fun_(state); };
	virtual std::pair<u64, _T> operator()(u64 state)		{ return this->fun_(state); };
	
	// ---------- SETTERS -----------
	void setFun(const repType& _fun)					{ this->fun_ = _fun; };
	void setFun(repType&& _fun)							{ this->fun_ = std::move(_fun); };

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

//template<typename _T>
class GlobalSym {
public:
	double val_;
	std::shared_ptr<Lattice> lat_;															// lattice type to be used later on
	
	// constructors
	GlobalSym(std::shared_ptr<Lattice> _lat) : val_(0), lat_(_lat) {};

	// check the symmetry existence
	virtual bool check(u64 state) const = 0;
	bool check(u64 state, bool outerCondition) { return this->check(state) && outerCondition; };
};

class U1Sym : public GlobalSym{
public:
	U1Sym(double _val, std::shared_ptr<Lattice> _lat) 
		: GlobalSym(_lat) 
	{
		this->val_ = _val;
	}

	bool check(u64 state)	    const override { return __builtin_popcountll(state) == val_; };
};

template <typename _T>
class HilbertSpace {
	u64 Nh;																					// number of states in the Hilbert space	

	// symmetries
	v_1d<GlobalSym*> symGroupGlobal_;
	v_1d<Operator<_T>> symGroup_;

	// symmetry normalization and mapping to a reduced hilbert space
	v_1d<_T> normalization_;																// stores the representative normalization
	v_1d<u64> mapping_;																		// stores the symmetry representative

public:
	_T getSymNorm(u64 baseIdx) const;

};