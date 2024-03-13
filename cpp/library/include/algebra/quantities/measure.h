#pragma once

#ifndef OPERATORS_H
#	include "../operators.h"
#endif

#include "../../quantities/statistics.h"

/*
* @brief Class that stores the measurements is able to save them.
*/
template <typename _T = std::complex<double>>
class Measurement
{
private:

	using MeasureGlobal = std::vector<_T>;
	using MeasureLocal	= std::vector<arma::Col<_T>>;
	using MeasureCorr	= std::vector<arma::Mat<_T>>;

	// store the operator averages
	std::vector<_T> valG_;
	std::vector<arma::Col<_T>> valL_;
	std::vector<arma::Mat<_T>> valC_;

	// store the many body operator matrices
	v_1d<arma::SpMat<_T>> MG_;
	v_2d<arma::SpMat<_T>> ML_;
	v_3d<arma::SpMat<_T>> MC_;

	// types for the operators
	//using OPG			= Operators::OpVec_glb_t;						
	using OPG			= v_1d<Operators::Operator<_T>>;
	//using OPL			= Operators::OpVec_loc_t;						
	using OPL			= v_1d<Operators::Operator<_T, uint>>;
	//using OPC			= Operators::OpVec_cor_t;						
	using OPC			= v_1d<Operators::Operator<_T, uint, uint>>;
	using MeasureTuple  = std::tuple<std::vector<_T>, std::vector<arma::Col<_T>>, std::vector<arma::Mat<_T>>>;

protected:
	std::string dir_	= "";
	uint threads_		= 1;
	uint Ns_			= 1;

	// store the lattice if necessary
	std::shared_ptr<Lattice> lat_;
	// global operators
	OPG opG_;
	// local operators
	OPL opL_;
	// correlation operators
	OPC opC_;
public:
	~Measurement();
	Measurement(size_t _Ns,	const strVec& _operators);
	Measurement(std::shared_ptr<Lattice> _lat,	const strVec& _operators);
	Measurement(size_t _Ns, const std::string& _dir, const OPG& _opG,
													 const OPL& _opL = {},
													 const OPC& _opC = {},
													 uint _threadNum = 1);
	Measurement(size_t _Ns, const std::string& _dir, const OPG& _opG,
													 const strVec& _opGN,
													 const OPL& _opL		= {},
													 const strVec& _opLN	= {},
													 const OPC& _opC		= {},
													 const strVec& _opCN	= {},
													 uint _threadNum		= 1);
	Measurement(std::shared_ptr<Lattice> _lat,	const std::string& _dir,
												const OPG& _opG,
												const OPL& _opL = {},
												const OPC& _opC = {},
												uint _threadNum = 1);

	// measure the observables
	template<typename _C>
	std::tuple<std::vector<_T>, std::vector<arma::Col<_T>>, std::vector<arma::Mat<_T>>> measureS(const _C& _state);

	template<typename _C>
	void measure(const _C& _state);

	template<typename _C>
	std::vector<_T> measureG(const _C& _state, int _cut = -1);

	// ---------------------------------------------------------------

	// measure the observables (offdiagonal)

	template<typename _C>
	std::tuple<std::vector<_T>, std::vector<arma::Col<_T>>, std::vector<arma::Mat<_T>>> measureS(const _C& _stateL, const _C& _stateR);

	template<typename _C>
	void measure(const _C& _stateL, const _C& _stateR);

	//template<typename _C>
	//void measure(const _C& _stateL, const _C& _stateR, MeasureGlobal& _mg, MeasureLocal& _ms = {}, MeasureCorr& _mc = {});

	template<typename _C>
	std::vector<_T> measureG(const _C& _stateL, const _C& _stateR, int _cut = -1);

	// save the measurements
	void save(const std::string _additional = "", const strVec & _ext = {".h5"});

public:
	auto clear() -> void;
	auto initializeMatrices(u64 _dim) -> void;

	// ########### CHECKERS ############
	
	template <typename _C>
	auto checkG_mat(const _C& _ev)	const noexcept -> bool;



	// ############ GETTERS ############

	auto getNs()					const noexcept -> size_t					{ return Ns_;			};
	auto getDir()					const noexcept -> std::string				{ return dir_;			};
	auto getThreads()				const noexcept -> uint						{ return threads_;		};
	auto getOpG()					const noexcept -> OPG						{ return opG_;			};
	auto getOpG_mat()				const noexcept -> v_1d<arma::SpMat<_T>>&	{ return MG_;			};
	auto getOpG_mat(uint i)			const noexcept -> v_1d<arma::SpMat<_T>>&	{ return MG_[i];		};
	auto getOpGN(uint i)			const noexcept -> std::string				{ return opG_[i].getNameS(); };
	auto getOpL()					const noexcept -> OPL						{ return opL_;			};
	auto getOpLN(uint i)			const noexcept -> std::string				{ return opL_[i].getNameS(); };
	auto getOpC()					const noexcept -> OPC						{ return opC_;			};
	auto getOpCN(uint i)			const noexcept -> std::string				{ return opC_[i].getNameS(); };
	auto getLat()					const noexcept -> std::shared_ptr<Lattice>	{ return lat_;			};
	auto getValG()					const noexcept -> const v_1d<_T>&			{ return valG_;			};
	auto getValL()					const noexcept -> const v_1d<arma::Col<_T>>&{ return valL_;			};
	auto getValC()					const noexcept -> const v_1d<arma::Mat<_T>>&{ return valC_;			};

	// ############ SETTERS ############

	auto setNs(size_t _Ns)					noexcept -> void					{ Ns_ = _Ns;			};
	auto setDir(const std::string& _dir)	noexcept -> void					{ dir_ = _dir;			};
	auto setThreads(uint _threads)			noexcept -> void					{ threads_ = _threads;	};
	auto setOpG(const OPG& _opG)			noexcept -> void					{ opG_ = _opG;			};
	auto setOpL(const OPL& _opL)			noexcept -> void					{ opL_ = _opL;			};
	auto setOpC(const OPC& _opC)			noexcept -> void					{ opC_ = _opC;			};
};

// ############################################################################################################################################################

template <typename _T>
Measurement<_T>::~Measurement() 
{ 
	LOGINFO("Destroying the measurement of the NQS averages.", LOG_TYPES::TRACE, 3);
	this->clear();
}

template<typename _T>
inline Measurement<_T>::Measurement(size_t _Ns, const strVec & _operators)
	: threads_(1), Ns_(_Ns)
{

}

template<typename _T>
inline Measurement<_T>::Measurement(std::shared_ptr<Lattice> _lat, const strVec & _operators)
{
	this->lat_ = _lat;
}

template<typename _T>
inline Measurement<_T>::Measurement(size_t _Ns, const std::string & _dir, 
									const OPG & _opG, const OPL & _opL, const OPC & _opC, uint _threadNum)
	: dir_(_dir), threads_(_threadNum), Ns_(_Ns), opG_(_opG), opL_(_opL), opC_(_opC)
{
	CONSTRUCTOR_CALL;
	// create directory
	makeDir(_dir);
}

template<typename _T>
inline Measurement<_T>::Measurement(size_t _Ns, const std::string & _dir,
									const OPG & _opG, const strVec & _opGN,
									const OPL & _opL, const strVec & _opLN,
									const OPC & _opC, const strVec & _opCN,
									uint _threadNum)
	: Measurement<_T>(_Ns, _dir, _opG, _opL, _opC, _threadNum)
{
	if (_opGN.size() == _opG.size())
	{
		for(auto i = 0; i < _opGN.size(); ++i)
			this->opG_[i].setNameS(_opGN[i]);
	}
	if (_opLN.size() == _opL.size())
	{
		for (auto i = 0; i < _opLN.size(); ++i)
			this->opL_[i].setNameS(_opLN[i]);
	}
	if (_opCN.size() == _opC.size())
	{
		for (auto i = 0; i < _opCN.size(); ++i)
			this->opC_[i].setNameS(_opCN[i]);
	}
}

template<typename _T>
inline Measurement<_T>::Measurement(std::shared_ptr<Lattice> _lat, const std::string & _dir, 
									const OPG & _opG, const OPL & _opL, const OPC & _opC, uint _threadNum)
	: dir_(_dir), threads_(_threadNum), lat_(_lat), opG_(_opG), opL_(_opL), opC_(_opC)
{
	CONSTRUCTOR_CALL;
	// create directory
	makeDir(_dir);
}

// ############################################################################################################################################################

// ############################################################ M E A S U R E ###############################################################################

// ############################################################################################################################################################

template<typename _T>
inline auto Measurement<_T>::clear() -> void
{
	// clear the averages
	valG_.clear();
	valL_.clear();
	valC_.clear();

	// clear the matrices
	MG_.clear();
	ML_.clear();
	MC_.clear();
}

// ############################################################################################################################################################

template<typename _T>
inline void Measurement<_T>::initializeMatrices(u64 _dim)
{
	this->clear();

	BEGIN_CATCH_HANDLER
	{
		// measure global
		for (const auto& _op : this->opG_)
		{
			arma::SpMat<_T> _Min = _op.template generateMat<typename arma::SpMat>(_dim);
			// push the operator
			this->MG_.push_back(_Min);
		}

	}
	END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);
	LOGINFO("Initialized the measurement matrices.", LOG_TYPES::TRACE, 3);

	//BEGIN_CATCH_HANDLER
	//{
	//	// measure local
	//	for (auto& _op : this->opL_)
	//	{
	//		_op->resetMB();
	//		// go through the local operators
	//		for (auto i = 0; i < this->Ns_; ++i)
	//		{
	//			_op->setManyBodyMat(_H, i);
	//			_op->applyManyBody(_state, i, 0);
	//		}
	//	}
	//}
	//END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);

	//BEGIN_CATCH_HANDLER
	//{
	//	// measure correlation
	//	for (auto& _op : this->opC_)
	//	{
	//		_op->resetMB();
	//		for (auto i = 0; i < this->Ns_; ++i)
	//		{
	//			for (auto j = 0; j < this->Ns_; ++j)
	//			{
	//				_op->setManyBodyMat(_H, i, j);
	//				_op->applyManyBody(_state, i, j);
	//			}
	//		}
	//	}
	//}
	//END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);
}

// ############################################################################################################################################################

template<typename _T>
template<typename _C>
inline std::vector<_T> Measurement<_T>::measureG(const _C & _state, int _cut)
{
	BEGIN_CATCH_HANDLER
	{
		// save into column
		v_1d<_T> _valG	= v_1d<_T>(_cut >= 0 ? _cut : this->opG_.size());

		// measure global
		for (size_t i = 0; i < this->MG_.size(); ++i)
		{
			if ((int)i == _cut)
				break;
			_valG[i] = Operators::applyOverlap(_state, this->MG_[i]);
		}

		return _valG;
	}
	END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);
	return {};
}

template<typename _T>
template<typename _C>
inline std::tuple<std::vector<_T>, std::vector<arma::Col<_T>>, std::vector<arma::Mat<_T>>> Measurement<_T>::measureS(const _C & _state)
{
	v_1d<_T> _valG			  = this->measureG(_state);

	v_1d<arma::Col<_T>> _valL = v_1d<arma::Col<_T>>(this->Ns_, arma::Col<_T>(this->opL_.size(), arma::fill::zeros));
	BEGIN_CATCH_HANDLER
	{
	// measure local
	for (size_t i = 0; i < this->ML_.size(); ++i)
		for (auto j = 0; j < this->Ns_; ++j)
			_valL[i](j) = Operators::applyOverlap(_state, this->ML_[i][j]);
	}
	END_CATCH_HANDLER("Problem in the measurement of local operators.", ;);

	v_1d<arma::Mat<_T>> _valC = v_1d<arma::Mat<_T>>(this->opC_.size(), arma::Mat<_T>(this->Ns_, this->Ns_, arma::fill::zeros));
	BEGIN_CATCH_HANDLER
	{
	// measure local
	for (size_t i = 0; i < this->MC_.size(); ++i)
		for (auto j = 0; j < this->Ns_; ++j)
			for (auto k = 0; k < this->Ns_; ++k)
				_valC[i](j, k) = Operators::applyOverlap(_state, this->MC_[i][j][k]);
	}
	END_CATCH_HANDLER("Problem in the measurement of correlation operators.", ;);

	return std::make_tuple(_valG, _valL, _valC);
}

template<typename _T>
template<typename _C>
inline void Measurement<_T>::measure(const _C & _state)
{
	const auto& [_valG, _valL, _valC] = this->measureS(_state);
	this->valG_ = _valG;
	this->valL_ = _valL;
	this->valC_ = _valC;
}

// ############################################################################################################################################################

template<typename _T>
template<typename _C>
inline std::vector<_T> Measurement<_T>::measureG(const _C & _stateL, const _C & _stateR, int _cut)
{
	BEGIN_CATCH_HANDLER
	{
		// save into column
		auto _valG = v_1d<_T>(_cut >= 0 ? _cut : this->opG_.size());

		// measure global
		for (size_t i = 0; i < this->MG_.size(); ++i)
		{
			if ((int)i == _cut)
				break;
			_valG[i] = Operators::applyOverlap(_stateL, _stateR, this->MG_[i]);
		}
		return _valG;
	}
	END_CATCH_HANDLER("Problem in the measurement of global operators.", ;);
	return {};
}

template<typename _T>
template<typename _C>
inline std::tuple<std::vector<_T>, std::vector<arma::Col<_T>>, std::vector<arma::Mat<_T>>> Measurement<_T>::measureS(const _C & _stateL, const _C & _stateR)
{
	auto _valG = this->measureG(_stateL, _stateR);
	auto _valL = v_1d<arma::Col<_T>>(this->Ns_, arma::Col<_T>(this->opL_.size(), arma::fill::zeros));
	BEGIN_CATCH_HANDLER
	{
		// measure local

		for (size_t i = 0; i < this->ML_.size(); ++i)
			for (auto j = 0; j < this->Ns_; ++j)
				_valL[i](j) = Operators::applyOverlap(_stateL, _stateR, this->ML_[i][j]);
	}
	END_CATCH_HANDLER("Problem in the measurement of local operators.", ;);

	auto _valC = v_1d<arma::Mat<_T>>(this->opC_.size(), arma::Mat<_T>(this->Ns_, this->Ns_, arma::fill::zeros));
	BEGIN_CATCH_HANDLER
	{
	// measure correlation
	for (size_t i = 0; i < this->MC_.size(); ++i)
		for (auto j = 0; j < this->Ns_; ++j)
			for (auto k = 0; k < this->Ns_; ++k)
				_valC[i](j, k) = Operators::applyOverlap(_stateL, _stateR, this->MC_[i][j][k]);
	}
	END_CATCH_HANDLER("Problem in the measurement of correlation operators.", ;);

	return std::make_tuple(_valG, _valL, _valC);
}

template<typename _T>
template<typename _C>
inline void Measurement<_T>::measure(const _C & _stateL, const _C & _stateR)
{
	const auto& [_valG, _valL, _valC] = this->measureS(_stateL, _stateR);
	this->valG_ = _valG;
	this->valL_ = _valL;
	this->valC_ = _valC;
}

// #############

/*
* @brief Check the norm of the operators.
* @param _ev The eigenvector of the system.
*/
template<typename _T>
template<typename _C>
inline bool Measurement<_T>::checkG_mat(const _C& _ev) const noexcept 
{
	LOGINFO("", LOG_TYPES::TRACE, 20, 3);
	for (int i = 0; i < this->MG_.size(); ++i)
	{
		auto _name				= this->opG_[i].getNameS();
		// check the norm
		auto _op_transformed	= SystemProperties::transform_operator(arma::Mat<_T>(this->MG_[i]), arma::Mat<_T>(_ev));
		auto _op_norm			= SystemProperties::hilber_schmidt_norm(_op_transformed);
		auto _op_norm_nt		= SystemProperties::hilber_schmidt_norm(arma::Mat<_T>(this->MG_[i]));
		LOGINFO("[" + _name + "]" + VEQ(_op_norm), LOG_TYPES::TRACE, 1);
		LOGINFO("[" + _name + "]" + VEQ(_op_norm_nt), LOG_TYPES::TRACE, 2);
	}
	LOGINFO("", LOG_TYPES::TRACE, 20, 3);
	return true;
}

// ############################################################################################################################################################

template<typename _T>
inline void Measurement<_T>::save(const std::string _additional, const strVec & _ext)
{
}