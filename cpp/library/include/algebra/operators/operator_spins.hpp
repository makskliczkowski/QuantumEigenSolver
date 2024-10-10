namespace Operators
{
    /*
	* @brief The spin operator namespace. Contains the most common spin operators.
	*/
	namespace SpinOperators
	{
		std::pair<u64, double> sig_x(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		std::pair<_OP_V_T, double> sig_x(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_x(size_t _Ns, size_t _part);
		Operators::Operator<double> sig_x(size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_x(size_t _Ns);

		std::pair<u64, double> sig_z(u64 base_vec, size_t _Ns, const v_1d<uint>& sites);
		std::pair<_OP_V_T, double> sig_z(_OP_V_T_CR base_vec, size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_z(size_t _Ns, size_t _part);
		Operators::Operator<double> sig_z(size_t _Ns, const v_1d<uint>& sites);
		Operators::Operator<double> sig_z(size_t _Ns);

		// ##########################################################################################################################################
		namespace RandomSuperposition {
			static inline std::vector<double> superpositions = { 0.3888, 0.1021, 0.3092, -0.3006, -0.9239, 0.7622, 0.4685, 0.8464, 0.4395, -0.1038, 0.3524, -0.7478, 0.0176, -0.9207, -0.7081, 0.0704 };
			Operators::Operator<double> sig_z(size_t _Ns);
		};
	}
};