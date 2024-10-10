namespace Operators
{
	/*
	* @brief For Quadratic Operators, we will treat the operators as acting on the integer index as it was not the configuration!
	*/
	namespace QuadraticOperators
	{
		// -------- n_i Operators --------

		Operators::Operator<double> site_occupation(size_t _Ns, const size_t _site);
		Operators::Operator<double> site_occupation_r(size_t _Ns, const v_1d<double>& _coeffs);
		Operators::Operator<double> site_occupation_r(size_t _Ns, const v_1d<size_t>& _sites, const v_1d<double>& _coeffs);

		// -------- n_q Operators --------
		
		Operators::Operator<double> site_nq(size_t _Ns, const size_t _momentum);

		// ------ n_i n_j Operators ------

		Operators::Operator<double> nn_correlation(size_t _Ns, const size_t _site_plus, const size_t _site_minus);

		// --- quasimomentum Operators ---

		Operators::Operator<std::complex<double>> quasimomentum_occupation(size_t _Ns, const size_t _momentum);
		Operators::Operator<double> quasimomentum_occupation(size_t _Ns);

		// ----- kinectic Operators ------

		Operators::Operator<double> kinetic_energy(size_t _Nx, size_t _Ny, size_t _Nz);
	}
};