#pragma once
#include "../source/src/common.h"
#include "../source/src/lattices.h"

/*
* @brief A class describing the local operator acting on specific states
*/
template<typename _T>
class Operator {
	typedef std::function<std::pair<u64, _T>(u64)> repType;									// type returned for representing 

protected:
	_T eigVal_;																				// eigenvalue for symmetry generator
	repType fun_;																			// function allowing to use symmetry
	std::shared_ptr<Lattice> lat_;															// lattice type to be used later on

public:
	Operator(std::shared_ptr<Lattice>& _lat, _T _eigVal, repType&& _fun) 
		: lat_(_lat), eigVal_(_eigVal), fun_(_fun){ this->init(); };
	Operator(const Operator<_T>& o) 
		: eigVal_(o.eigVal_), fun_(o.fun_), lat_(o.lat_) {};
	Operator(Operator<_T>&& o) 
		: eigVal_(std::move(o.eigVal_)), fun_(std::move(o.fun_)), lat_(std::move(o.lat_)) {};
	
	// ---------- virtual functions to override ----------
	virtual void init() {};

	// ---------- override operators -----------
	virtual repType operator()(u64 state)				RETURNS(this->fun_(state));
	

	// ---------- SETTERS -----------
	void setFun(const repType& _fun)					{ this->fun_ = _fun; };
	void setFun(repType&& _fun)							{ this->fun_ = std::move(_fun); };

	// ---------- join operators together ----------
	template<typename _T1, typename _T2>
	friend Operator<_T2> operator*(const Operator<_T1>& A, const Operator<_T2>& B) {
		Operator<_T2> op(A);
		auto _fun = [A, B](u64 s) {
			const auto [s1, v1] = A(s);
			const auto [s2, v2] = B(s1);
			return std::make_pair(s2, v1 * v2);
		};
		op.setFun(std::move(_fun));
		return op;
	};
};

