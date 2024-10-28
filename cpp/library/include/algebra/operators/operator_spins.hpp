namespace Operators
{
    /*
	* @brief The spin operator namespace. Contains the most common spin operators.
	*/
	namespace SpinOperators
	{	
		template <typename _T = double>
		std::pair<u64, _T> sig_x(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_x(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		Operators::Operator<_T> sig_x(size_t _Ns, size_t _part);
		template <typename _T = double>
		Operators::Operator<_T> sig_x(size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		Operators::Operator<_T> sig_x(size_t _Ns);
		template <typename _T = double>
		Operators::Operator<_T, uint> sig_x_l(size_t _Ns);

		template <typename _T = double>
		std::pair<u64, _T> sig_z(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		std::pair<_OP_V_T, _T> sig_z(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		Operators::Operator<_T> sig_z(size_t _Ns, size_t _part);
		template <typename _T = double>
		Operators::Operator<_T> sig_z(size_t _Ns, const v_1d<uint>& sites);
		template <typename _T = double>
		Operators::Operator<_T> sig_z(size_t _Ns);
		template <typename _T = double>
		Operators::Operator<_T, uint> sig_z_l(size_t _Ns);
	};

	namespace SpinOperators
	{
		// ##########################################################################################################################################
		namespace RandomSuperposition {
			static inline std::vector<double> superpositions = { 0.3888, 0.1021, 0.3092, -0.3006, -0.9239, 0.7622, 0.4685, 0.8464, 0.4395, -0.1038, 0.3524, -0.7478, 0.0176, -0.9207, -0.7081, 0.0704 };
			Operators::Operator<double> sig_z(size_t _Ns);
			Operators::Operator<double> sig_z_vanish(size_t _Ns);
			// Operators::Operator<double> sig_z_vanish_r(size_t _Ns);
		};
	};
};