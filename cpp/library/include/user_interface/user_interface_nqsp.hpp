/**
* @struct NqsP
* @brief Parameters for neural network quantum states (NQS).
* 
* @var v_1d<u64> layersDim Dimension of each layer in the neural network.
* @var type Type of the neural network quantum state. Default is NQSTYPES::RBM_T.
* @var uint nVisible Number of visible units. Default is 1.
* @var double nqs_nh Number of hidden units. Default is 1.
* @var uint nLayers Number of layers. Default is 2.
* @var uint nFlips Number of flips. Default is 1.
* 
* @section Training Parameters
* @var double nqs_tr_pinv Pseudoinverse for the NQS. Default is 1e-5.
* @var double nqs_tr_pc Percentage of samples for display. Default is 5.0.
* @var uint nqs_tr_bs Block size for training. Default is 8.
* @var uint nqs_tr_th Thermalize when training. Default is 50.
* @var uint nqs_tr_mc Inner blocks for training (MC steps). Default is 500.
* @var uint nqs_tr_epo Samples (outer loop for training). Default is 1000.
* @var bool nqs_tr_rst Reset state before each step? Default is false.
* @var double nqs_tr_reg Regularization for the NQS SR method. Default is 1e-7.
* @var int nqs_tr_regs Regularization scheduler. Default is 0.
* @var int nqs_tr_regp Regularization scheduler patience. Default is 10.
* @var double nqs_tr_regd Regularization decay. Default is 0.96.
* @var int nqs_tr_prec Preconditioner for the NQS SR method. Default is 0.
* @var int nqs_tr_sol Solver for the NQS SR method. Default is 1.
* @var double nqs_tr_tol Solver tolerance. Default is 1e-7.
* @var int nqs_tr_iter Max iterations for the NQS SR method. Default is 5000.
* 
* @section Time Evolution Parameters
* @var bool nqs_te Time evolution. Default is false.
* @var uint nqs_te_mc Samples (outer loop for time evolution). Default is 1.
* @var uint nqs_te_th Thermalize when time evolution. Default is 0.
* @var uint nqs_te_bn Inner blocks for time evolution. Default is 100.
* @var uint nqs_te_bs Block size for time evolution. Default is 4.
* @var bool nqs_te_rst Reset state before each step? Default is false.
* @var double nqs_te_dt Initial time step for time evolution. Default is 0.01.
* @var double nqs_te_tf Final time for time evolution. Default is 1.0.
* @var uint nqs_te_tlog Use logarithmic time steps? Default is 0.
* @var bool nqs_te_rk Use Runge-Kutta method for time evolution? Default is false.
* 
* @section Excited States Parameters
* @var uint nqs_ex_mc Samples (outer loop for collecting - excited states). Default is 1.
* @var uint nqs_ex_th Thermalize when collecting - excited states. Default is 0.
* @var uint nqs_ex_bn Inner blocks for collecting - excited states. Default is 100.
* @var uint nqs_ex_bs Block size for collecting - excited states. Default is 4.
* @var double nqs_ex_beta Beta for the excited states.
* 
* @section Collecting Parameters
* @var uint nqs_col_mc Samples (outer loop for collecting). Default is 1.
* @var uint nqs_col_th Thermalize when collecting. Default is 0.
* @var uint nqs_col_bn Inner blocks for collecting. Default is 100.
* @var uint nqs_col_bs Block size for collecting. Default is 4.
* @var bool nqs_col_rst Reset state before each step? Default is false.
* 
* @section Learning Rate Parameters
* @var int nqs_sch Learning rate scheduler. Default is 0.
* @var double nqs_lr Initial learning rate. Default is 1e-3.
* @var double nqs_lrd Learning rate decay. Default is 0.96.
* @var int nqs_lr_pat Learning rate decay pattern. Default is 10.
* 
* @section Early Stopping Parameters
* @var int nqs_es_pat Patience for early stopping. Default is 5.
* @var double nqs_es_del Delta for early stopping. Default is 1e-3.
* 
* @section Miscellaneous Parameters
* @var bool nqs_ed Use exact diagonalization for the NQS. Default is false.
* @var std::string _loadNQS Static string for loading NQS. Default is an empty string.
* @var std::string loadNQS_ String for loading NQS. Default is an empty string.
* 
* @brief Sets default values for the parameters.
*/
#ifndef USER_INTERFACE_PARAMS_NQSP_HPP
#define USER_INTERFACE_PARAMS_NQSP_HPP
#include "./user_interface_symp.hpp"

namespace UI_PARAMS 
{

	// ----------------------------------------------------------------

	// !TODO 
	// Neural network quantum states params
	struct NqsP 
	{
		v_1d<u64> layersDim;
		UI_PARAM_CREATE_DEFAULT(type, NQS_NS::NQSTYPES, NQS_NS::NQSTYPES::RBM_T);
		UI_PARAM_CREATE_DEFAULT(nVisible, uint, 1);
		UI_PARAM_CREATE_DEFAULTD(nqs_nh, double, 1);
		UI_PARAM_CREATE_DEFAULT(nLayers, uint, 2);
		UI_PARAM_CREATE_DEFAULT(nFlips, uint, 1);
		
		UI_PARAM_CREATE_DEFAULTD(nqs_tr_pinv, double, 1e-5);// pseudoinverse for the NQS
		UI_PARAM_CREATE_DEFAULTD(nqs_tr_pc, double, 5.0);	// percentage of the samples to be used for display
		// training
		UI_PARAM_CREATE_DEFAULT(nqs_tr_bs, uint, 8);		// block size for training
		UI_PARAM_CREATE_DEFAULT(nqs_tr_th, uint, 50);		// thermalize when training
		UI_PARAM_CREATE_DEFAULT(nqs_tr_mc, uint, 500);		// number of inner blocks for training - this is rather crucial - is Monte Carlo steps
		UI_PARAM_CREATE_DEFAULT(nqs_tr_epo, uint, 1000);	// number of samples - outer loop for training
		UI_PARAM_CREATE_DEFAULT(nqs_tr_rst, bool, 0);		// should I reset the state before each step?
		UI_PARAM_CREATE_DEFAULT(nqs_tr_pt, int, 0);			// should I use the parallel tempering - this is the number of replicas - 0 - no, > 0 - yes
		// timeevolution
		UI_PARAM_CREATE_DEFAULT(nqs_te, bool, 0);			// time evolution - do or do not? - 0 - no, 1 - yes
		UI_PARAM_CREATE_DEFAULT(nqs_te_mc, uint, 1);		// number of samples - outer loop for time evolution
		UI_PARAM_CREATE_DEFAULT(nqs_te_th, uint, 0);		// thermalize when time evolution
		UI_PARAM_CREATE_DEFAULT(nqs_te_bn, uint, 100);		// number of inner blocks for time evolution
		UI_PARAM_CREATE_DEFAULT(nqs_te_bs, uint, 4);		// block size for time evolution
		UI_PARAM_CREATE_DEFAULT(nqs_te_rst, bool, 0);		// should I reset the state before each step?
		UI_PARAM_CREATE_DEFAULTD(nqs_te_dt, double, 0.01);	// time step for the time evolution - initial time step
		UI_PARAM_CREATE_DEFAULTD(nqs_te_tf, double, 1.0);	// final time for the time evolution - final time
		UI_PARAM_CREATE_DEFAULT(nqs_te_tlog, uint, 0);		// use the logarithmic time steps? - 0 - no, > 0 - yes (use this number as the number of steps)
		UI_PARAM_CREATE_DEFAULT(nqs_te_rk, bool, 0);		// use the Runge-Kutta method for the time evolution - 0 - Euler, 1 - Runge-Kutta (2nd order)
		// regularization
		UI_PARAM_CREATE_DEFAULTD(nqs_tr_reg, double, 1e-7); // regularization for the NQS SR method
		UI_PARAM_CREATE_DEFAULT(nqs_tr_regs, int, 0);		// regularization for the NQS SR method - scheduler
		UI_PARAM_CREATE_DEFAULT(nqs_tr_regp, int, 10);		// regularization for the NQS SR method - scheduler patience
		UI_PARAM_CREATE_DEFAULTD(nqs_tr_regd, double, 0.96);// regularization for the NQS SR method - decay
		// preconditioner
		UI_PARAM_CREATE_DEFAULT(nqs_tr_prec, int, 0);		// preconditioner for the NQS SR method - 0 - identity, 1 - Jacobi, 2 - Incomplete Cholesky, 3 - SSOR
		// solver type
		UI_PARAM_CREATE_DEFAULT(nqs_tr_sol, int, 1);		// solver for the NQS SR method
		UI_PARAM_CREATE_DEFAULTD(nqs_tr_tol, double, 1e-7); // solver for the NQS SR method - tolerance
		UI_PARAM_CREATE_DEFAULT(nqs_tr_iter, int, 5000);	// solver for the NQS SR method - maximum number of iterations
		// for collecting - excited states
		UI_PARAM_CREATE_DEFAULT(nqs_ex_mc, uint, 1);		// number of samples - outer loop for collecting - excited states
		UI_PARAM_CREATE_DEFAULT(nqs_ex_th, uint, 0);		// thermalize when collecting - excited states
		UI_PARAM_CREATE_DEFAULT(nqs_ex_bn, uint, 100);		// number of inner blocks for collecting - excited states
		UI_PARAM_CREATE_DEFAULT(nqs_ex_bs, uint, 4);		// block size for collecting - excited states
		UI_PARAM_CREATE_DEFAULTV(nqs_ex_beta, double);		// beta for the excited states - if not set, then only the ground state is calculated
		// for collecting
		UI_PARAM_CREATE_DEFAULT(nqs_col_mc, uint, 1);		// number of samples - outer loop for collecting
		UI_PARAM_CREATE_DEFAULT(nqs_col_th, uint, 0);		// thermalize when collecting
		UI_PARAM_CREATE_DEFAULT(nqs_col_bn, uint, 100);		// number of inner blocks for collecting
		UI_PARAM_CREATE_DEFAULT(nqs_col_bs, uint, 4);		// block size for collecting
		UI_PARAM_CREATE_DEFAULT(nqs_col_rst, bool, 0);		// should I reset the state before each step?
		// learning rate
		UI_PARAM_CREATE_DEFAULT(nqs_sch, int, 0);			// learning rate scheduler - 0 - constant, 1 - exponential decay (default), 2 - step decay, 3 - cosine decay, 4 - adaptive
		UI_PARAM_CREATE_DEFAULTD(nqs_lr, double, 1e-3);		// learning rate (initial)
		UI_PARAM_CREATE_DEFAULTD(nqs_lrd, double, 0.96);	// learning rate decay
		UI_PARAM_CREATE_DEFAULT(nqs_lr_pat, int, 10);		// learning rate decay pattern
		// early stopping
		UI_PARAM_CREATE_DEFAULT(nqs_es_pat, int, 5);		// use the early stopping
		UI_PARAM_CREATE_DEFAULTD(nqs_es_del, double, 1e-3);	// patience for the early stopping
		UI_PARAM_CREATE_DEFAULT(nqs_ed, bool, false);		// use the exact diagonalization for the NQS
		// weight load directory
		inline static const std::string _loadNQS	        = ""; 
		std::string loadNQS_								= "";

		void setDefault() 
		{
			UI_PARAM_SET_DEFAULT(nqs_nh);
			UI_PARAM_SET_DEFAULT(nVisible);
			UI_PARAM_SET_DEFAULT(nLayers);
			UI_PARAM_SET_DEFAULT(nFlips);
			// training
			UI_PARAM_SET_DEFAULT(nqs_tr_epo);
			UI_PARAM_SET_DEFAULT(nqs_tr_mc);
			UI_PARAM_SET_DEFAULT(nqs_tr_bs);
			UI_PARAM_SET_DEFAULT(nqs_tr_th);
			UI_PARAM_SET_DEFAULT(nqs_tr_rst);			
			UI_PARAM_SET_DEFAULT(nqs_lr);
			UI_PARAM_SET_DEFAULT(loadNQS);
			// collection
			UI_PARAM_SET_DEFAULT(nqs_col_mc);
			UI_PARAM_SET_DEFAULT(nqs_col_th);
			UI_PARAM_SET_DEFAULT(nqs_col_bn);
			UI_PARAM_SET_DEFAULT(nqs_col_bs);
			UI_PARAM_SET_DEFAULT(nqs_col_rst);
			// time evolution
			UI_PARAM_SET_DEFAULT(nqs_te);
			UI_PARAM_SET_DEFAULT(nqs_te_mc);
			UI_PARAM_SET_DEFAULT(nqs_te_th);
			UI_PARAM_SET_DEFAULT(nqs_te_bn);
			UI_PARAM_SET_DEFAULT(nqs_te_bs);
			UI_PARAM_SET_DEFAULT(nqs_te_rst);
			UI_PARAM_SET_DEFAULT(nqs_te_dt);
			UI_PARAM_SET_DEFAULT(nqs_te_tf);
			UI_PARAM_SET_DEFAULT(nqs_te_tlog);
			UI_PARAM_SET_DEFAULT(nqs_te_rk);
			// regularization
			UI_PARAM_SET_DEFAULT(nqs_tr_reg);
			UI_PARAM_SET_DEFAULT(nqs_tr_regs);
			UI_PARAM_SET_DEFAULT(nqs_tr_regd);
			UI_PARAM_SET_DEFAULT(nqs_tr_regp);
			// preconditioner
			UI_PARAM_SET_DEFAULT(nqs_tr_prec);
			// solver type
			UI_PARAM_SET_DEFAULT(nqs_tr_sol);
			UI_PARAM_SET_DEFAULT(nqs_tr_tol);
			UI_PARAM_SET_DEFAULT(nqs_tr_iter);
			// early stopping
			UI_PARAM_SET_DEFAULT(nqs_es_pat);
			UI_PARAM_SET_DEFAULT(nqs_es_del);
			// excited states
			UI_PARAM_SET_DEFAULT(nqs_ex_mc);
			UI_PARAM_SET_DEFAULT(nqs_ex_th);
			UI_PARAM_SET_DEFAULT(nqs_ex_bn);
			UI_PARAM_SET_DEFAULT(nqs_ex_bs);
		}
	};
};

#endif