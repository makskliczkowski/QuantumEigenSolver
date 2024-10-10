namespace Operators
{
	namespace OperatorTypes
	{
		// known operators
		enum OperatorsAvailable { 
			E =	0, 
			Sx, 
			Sy, 
			Sz,
			SzR,			// random superposition of Sz
			// quadratic
			ni, 			// site occupation
			nq,				// site modulation 
			nn,				// site correlation 
			nk,				// quasi-momentum occupation
			nr, 			// site occupation (Random)
			E2,				// identity squared - not used
		};

		BEGIN_ENUM(OperatorsAvailable)
		{
			DECL_ENUM_ELEMENT(E),
			DECL_ENUM_ELEMENT(Sx),
			DECL_ENUM_ELEMENT(Sy),
			DECL_ENUM_ELEMENT(Sz),
			DECL_ENUM_ELEMENT(SzR),
			DECL_ENUM_ELEMENT(ni),
			DECL_ENUM_ELEMENT(nq),
			DECL_ENUM_ELEMENT(nn),
			DECL_ENUM_ELEMENT(nk),
			DECL_ENUM_ELEMENT(nr),
			DECL_ENUM_ELEMENT(E2)
		}
		END_ENUM(OperatorsAvailable)

		/*
		* @brief Checks if the operator needs integer indices
		* @param _op the operator
		* @returns true if the operator needs integer indices	
		*/
		inline bool needsIntegerIdx(OperatorsAvailable _op)
		{
			switch (_op)
			{
			// case OperatorsAvailable::nq:
			case OperatorsAvailable::nk:
			case OperatorsAvailable::nr:
				return false;
			default:
				return true;
			}
		}
		// --------------------------------------------------------------------------------------------

		/*
		* @brief Check if the operator uses Hilbert space dimension rather than the integer indices
		* @param _op the operator
		* @returns true if the operator uses Hilbert space dimension
		*/
		inline bool needsHilbertSpaceDim(OperatorsAvailable _op)
		{
			switch (_op)
			{
			case OperatorsAvailable::E:
			case OperatorsAvailable::E2:
				return false;
			// quadratic operators do use Nh!
			case OperatorsAvailable::ni:
			case OperatorsAvailable::nq:
			case OperatorsAvailable::nn:
			case OperatorsAvailable::nk:
			case OperatorsAvailable::nr:
				return true;
			default:
				return false;
			}
			return false;
		}
	}

	// ##########################################################################################################################################


	class OperatorNameParser
	{
	private:
		static inline int precision_ = 3;
		size_t L_;
		size_t Nh_; 
		std::string Lstr_;
		std::string Nhstr_;

		void initMap()
		{
			for(int fooInt = static_cast<int>(OperatorTypes::OperatorsAvailable::E); fooInt != static_cast<int>(OperatorTypes::OperatorsAvailable::E2); fooInt++ )
			{
				if(fooInt == static_cast<int>(OperatorTypes::OperatorsAvailable::E2) || fooInt == static_cast<int>(OperatorTypes::OperatorsAvailable::E)) 
					continue;
				
				// setup the name 
				std::string fooStr 		= OperatorTypes::getSTR_OperatorsAvailable(static_cast<OperatorTypes::OperatorsAvailable>(fooInt));
				operator_map_[fooStr] 	= static_cast<OperatorTypes::OperatorsAvailable>(fooInt);
			}
		}

	public:
		// create a map of operators
		std::map<std::string, Operators::OperatorTypes::OperatorsAvailable> operator_map_;

		// --------------------------------------------------------------------------------------------

		OperatorNameParser(size_t L) : L_(L), Nh_(L), Lstr_(std::to_string(L)), Nhstr_(Lstr_) 
		{
			this->initMap();
		};

		OperatorNameParser(size_t L, size_t Nh) : L_(L), Nh_(Nh), Lstr_(std::to_string(L)), Nhstr_(std::to_string(Nh)) 
		{
			this->initMap();
		};

		// --------------------------------------------------------------------------------------------

		
		// parse input 
		strVec parse(const strVec& _inputs);

	private:
		// parse single input
		strVec parse(const std::string& _input);

		// parse default (without the site separator)
		strVec parseDefault(const std::string& _input);

		// parse with the sites after "/"
		std::string parseSingleOperator(const std::string& _input);

		// parse with correlation after "-"
		strVec parseCorrelationOperator(const std::string& _input);

		// parse multiple operators
		strVec parseMultipleOperators(const std::string& _input);

		// parse range of sites
		strVec parseRangeOperators(const std::string& _input);

		// --------------------------------------------------------------------------------------------
		
		// resolve the operator name from the input sites
		std::pair<std::string, std::string> resolveOperatorSeparator(const std::string& _input);

		// resolve the site and return a long double (for the indices parsing)
		long double resolveSite(const std::string& _site, bool _usesHilbert = false);

		// std::string resolveSite(double _site);

		std::vector<long double> resolveSites(const strVec& _sites, bool _usesHilbert);

		strVec resolveSitesMultiple(const std::string& _sites, bool _needsIntIdx = true, bool _usesHilbert = false);

		// resolve the correlation recursively
		void resolveCorrelation(const std::vector<strVec>& _list, strVec& _currentCombination, size_t _depth, strVec& _out);

		// --------------------------------------------------------------------------------------------
	public:

		/*
		* @brief Creates a global operator from the input string - this allows for its further usage in the calculations.
		* (creating matrices, acting on states, etc.)
		* @param _input the input string
		* @param _operator the operator to create
		* @returns true if the operator was created successfully
		*/
		template <typename _T>
		bool createGlobalOperator(const std::string& _input, std::shared_ptr<Operator<_T>>& _operator,
				bool _usesRealAllowed 		= true,
				bool _useHilbertAllowed 	= false,
				randomGen* _rgen 			= nullptr)
		{
			// resolve the operator and the sites based on the input
			auto [op, sites] 		= this->resolveOperatorSeparator(_input);
			
			// check if the operator is known
			if (!this->operator_map_.contains(op))
				return false;

			// check if the operator uses the Hilbert space or the lattice size
			bool _usesHilbert 		= OperatorTypes::needsHilbertSpaceDim(this->operator_map_[op]);
			
			// get the dimension - either the Hilbert space or the lattice size (depending on the character of the operator)
			size_t _dimension 		= _usesHilbert ? this->Nh_ : this->L_;

			// check if the sites contain the correlation or random operator
			v_1d<long double> _sites 	= { 0 };
			bool _containsRandom 		= false;
			if (_containsRandom = sites.find(OPERATOR_SEP_RANDOM) != std::string::npos; !_containsRandom)
				_sites = this->resolveSites(splitStr(sites, OPERATOR_SEP_CORR), _usesHilbert);

			// filter the operators
			if (!_useHilbertAllowed && (_usesHilbert || _containsRandom))
				return false;
			else if (!_usesRealAllowed && !_usesHilbert)
				return false;

			// create the operator
			switch (operator_map_[op])
			{
			// !!!!! SPIN OPERATORS !!!!!
			case OperatorTypes::OperatorsAvailable::Sx: 
				_operator = std::make_shared<Operator<_T>>(Operators::SpinOperators::sig_x(_dimension, Vectors::convert<uint>(_sites)));
				break;
			case OperatorTypes::OperatorsAvailable::Sy:
				// return Operators::SpinOperators::sig_y(this->L_, _sites);
				break;
			case OperatorTypes::OperatorsAvailable::Sz:
				_operator = std::make_shared<Operator<_T>>(Operators::SpinOperators::sig_z(_dimension, Vectors::convert<uint>(_sites)));
				break;
			case OperatorTypes::OperatorsAvailable::SzR:
				_operator = std::make_shared<Operator<_T>>(Operators::SpinOperators::RandomSuperposition::sig_z(_dimension));
				break;
			// !!!!! QUADRATIC OPERATORS !!!!!
			case OperatorTypes::OperatorsAvailable::ni:
				_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::site_occupation(_dimension, _sites[0]));	
				break;
			case OperatorTypes::OperatorsAvailable::nq:
				_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::site_nq(_dimension, _sites[0]));
				break;
			case OperatorTypes::OperatorsAvailable::nn:
				if(_sites.size() == 1)
					_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::nn_correlation(_dimension, _sites[0], _sites[0]));
				else if (_sites.size() > 1)
					_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::nn_correlation(_dimension, _sites[0], _sites[1]));
				break;
			case OperatorTypes::OperatorsAvailable::nk:
				if (_sites[0] == 0)
					_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::quasimomentum_occupation(_dimension));
				// else 
					// return Operators::QuadraticOperators::quasimomentum_occupation(this->L_, _sites[0]);
				break;
			// !!!!!! RANDOM OPERATOR !!!!!!
			case OperatorTypes::OperatorsAvailable::nr:
			{
				if (_rgen)
				{
					// create the random operator
					v_1d<double> _rcoefs = _rgen->random<double>(-1.0, 1.0, _dimension);
					_operator = std::make_shared<Operator<_T>>(Operators::QuadraticOperators::site_occupation_r(_dimension, _rcoefs));
				}
				else 
					return false;
				break;	
			}
			default:
				return false;
			};

			return true;
		}

		/*
		* @brief Creates a global operator from the input string - this allows for its further usage in the calculations.
		* (creating matrices, acting on states, etc.)
		* allows for filtering the operators on being quadratic or many-body operators
		* @param _input the input string
		* @param _operator the operator to create
		* @param _uses real if the operator uses real space indices (can be also momentum)
		* @param _usesHilbert if the operator uses the Hilbert space dimension (quadraic operators)
		* @returns a pair of the operator and the names of the operators
		*/
		template <typename _T>
		std::pair<std::vector<std::shared_ptr<Operator<_T>>>, strVec> createGlobalOperators(const strVec& _inputs,
																							bool _usesReal 		= true,
																							bool _usesHilbert 	= false,
																							randomGen* _rgen 	= nullptr)
		{
			std::vector<std::shared_ptr<Operator<_T>>> ops;

			// parse the input strings
			strVec _outStr = this->parse(_inputs);

			// create the operators
			LOGINFO("Using operators: ", LOG_TYPES::INFO, 4);
			strVec _msgs = {};
			for (int i = 0; i < _outStr.size(); i++)
				_msgs.push_back(STR(i) + ")" + _outStr[i]);
			LOGINFO(_msgs, LOG_TYPES::INFO, 4);
			
			// try to parse the operators
			strVec _outOperators = {};
			for (auto& op : _outStr)
			{
				std::shared_ptr<Operator<_T>> _opin;

				// check if the operator is valid
				if (this->createGlobalOperator<_T>(op, _opin, _usesReal, _usesHilbert, _rgen))
				{
					LOGINFO("Correctly parsed operator: " + op, LOG_TYPES::INFO, 4);
					ops.push_back(_opin);
					_outOperators.push_back(op);
				}
			}

			return std::make_pair(ops, _outOperators);
		}

		// --------------------------------------------------------------------------------------------
	};
};