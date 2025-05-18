'''
NQS training class.
'''


import os
import json
import time
import numpy as np
from contextlib import contextmanager
from tqdm import trange
from typing import Union

from general_python.common.plot import Plotter, colorsCycle, linestylesCycle
from general_python.common.flog import Logger
from general_python.common.timer import timeit

#! NQS
from NQS.nqs import NQS
from NQS.tdvp import TDVP
from general_python.algebra.ode import IVP
from general_python.ml.schedulers import Parameters, EarlyStopping

class NQSTrainer:
    def __init__( self,
                nqs             : NQS,
                ode_solver      : IVP,
                tdvp            : TDVP,
                n_batch         : int,
                lr_scheduler    : Parameters,
                reg_scheduler   : Parameters,
                early_stopper   : EarlyStopping,
                logger          : Logger,
                **kwargs
            ):
        #! set the objects
        self.nqs                = nqs
        self.ode_solver         = ode_solver
        self.tdvp               = tdvp
        self.lr_scheduler       = lr_scheduler
        self.reg_scheduler      = reg_scheduler
        self.early_stopper      = early_stopper
        self.logger             = logger

        #! should I jit compile the ansatz - it is compiled but we can move it till the very end
        self.ansatz             = nqs.ansatz
        self.loc_energy         = nqs.local_energy
        self.flat_grad          = nqs.flat_grad
        
        #! JIT-compiled single step (y, t, *, configs, configs_ansatze, probabilities, int_step)
        self._single_step       = nqs.wrap_single_step_jax(batch_size = n_batch)

        # storage for history
        self.history            = []
        self.history_std        = []
        self.lr_history         = []
        self.reg_history        = []
        self.timings            = {"sample": [], "step": [], "update": [], "gradient": [], "prepare": [], "solve": []}

    # ------------------------------------------------------

    @contextmanager
    def _time(self, phase: str, fn, *args, **kwargs):
        """
        Context manager to time a function call and store elapsed time.

        Yields:
            result of fn(*args, **kwargs)
        """
        result, elapsed = timeit(fn, *args, **kwargs)
        self.timings[phase].append(elapsed)
        yield result

    # ------------------------------------------------------

    def _update_lr(self, epoch: int, last_loss: float, update_lr: bool = True):
        """
        Update the learning rate based on the epoch and last loss.
        Parameters
        ----------
        epoch : int
            Current epoch number.
        last_loss : float
            Last recorded loss.
        """
        if update_lr and self.lr_scheduler is not None:
            lr = self.lr_scheduler(epoch, last_loss)
            self.ode_solver.set_dt(float(lr))
            return lr
        return self.ode_solver.dt
    
    def _update_reg(self, epoch: int, last_loss: float, update_reg: bool = True):
        """
        Update the regularization based on the epoch and last loss.
        Parameters
        ----------
        epoch : int
            Current epoch number.
        last_loss : float
            Last recorded loss.
        update_reg : bool
            Whether to update the regularization.
        """
        if update_reg and self.reg_scheduler is not None:
            reg = self.reg_scheduler(epoch, last_loss)
            self.tdvp.set_diag_shift(reg)
            return reg
        return self.tdvp.sr_diag_shift
    
    def _reset_history(self, reset: bool = False):
        """
        Reset the history of the training.
        Parameters
        ----------
        reset : bool
            Whether to reset the history.
        """
        if reset:
            self.history        = []
            self.history_std    = []
            self.timings        = {"sample": [], "step": [], "update": []}
        else:
            self.history        = list(self.history)
            self.history_std    = list(self.history_std)
            self.timings        = {k: list(v) for k, v in self.timings.items()}
    
    def train(self,
            n_epochs            : int,
            reset               : bool = False,
            use_lr_scheduler    : bool = True,
            use_reg_scheduler   : bool = True,
            **kwargs):
        
        #! reset the history
        self._reset_history(reset=reset)
        
        #! reset the early stopping
        self.early_stopper.reset()

        #! get the current parameters
        last_params = self.nqs.get_params()
        
        #! create the progress bar
        pbar = trange(n_epochs, desc="Training", leave=True)
        
        #! go through the epochs
        for epoch in pbar:
            
            #! schedulers
            lr          = self._update_lr(epoch, np.real(self.history[-1]) if self.history else None, use_lr_scheduler)
            reg         = self._update_reg(epoch, np.real(self.history[-1]) if self.history else None, use_reg_scheduler)

            #! sampling
            with self._time("sample", self.nqs.sample, reset=reset) as sample_out:
                (_, _), (cfgs, cfgs_psi), probs = sample_out

            #! energy + gradient
            params = self.nqs.get_params(unravel=True) # gives a vector instead of a dict
            with self._time("step", self.ode_solver.step, 
                                    f               = self.tdvp,
                                    y               = params,
                                    t               = 0.0,
                                    est_fn          = self._single_step,
                                    configs         = cfgs,
                                    configs_ansatze = cfgs_psi,
                                    probabilities   = probs) as step_out:
                dparams, _, (info, meta) = step_out

            #! update
            with self._time("update", self.nqs.set_params, dparams,
                                    shapes  = meta[0],
                                    sizes   = meta[1],
                                    iscpx   = meta[2]) as update_out:
                pass

            #! record
            mean_E      = info.mean_energy
            std_E       = info.std_energy
            std_E_real  = np.real(std_E)
            mean_E_real = np.real(mean_E)
            self.history.append(mean_E_real)
            self.history_std.append(std_E_real)

            #! add other times
            self.timings["gradient"].append(info.times['gradient'])
            self.timings["prepare"].append(info.times['prepare'])
            self.timings["solve"].append(info.times['solve'])
            
            #! progress bar
            times       = {p: self.timings[p][-1] for p in ("sample","step","update","gradient","prepare","solve")}
            total       = sum(times.values())
            pbar.set_postfix({
                "E/N"           :   f"{mean_E / self.nqs.size:.4e} ± {std_E_real / self.nqs.size:.4e}",
                "lr"            :   f"{lr:.1e}",
                "sig"           :   f"{reg:.1e}",
                **{f"t_{k}"     :   f"{v:.2e}s" for k,v in times.items()},
                "t_total"       :   f"{total:.2e}s",
                }, refresh=True)

            #! check for NaN
            if np.isnan(mean_E) or np.isnan(std_E):
                self.logger.warning(f"NaN at epoch {epoch}, stopping.")
                break
            
            #! update the last parameters
            last_params = self.nqs.get_params()
            
            if self.early_stopper(mean_E_real):
                self.logger.info(f"Early stopping at epoch {epoch}.")
                break
        
        #! set the last parameters in the end
        self.nqs.set_params(last_params)

        #! convert to arrays
        self.history        = np.array(self.history)
        self.history_std    = np.array(self.history_std)
        self.lr_history     = np.array(self.lr_scheduler.history)
        self.reg_history    = np.array(self.reg_scheduler.history)
        
        for k in self.timings:
            self.timings[k] = np.array(self.timings[k])

        return self.history, self.history_std, self.timings

    # ------------------------------------------------------
    
    def _last_size(self, n: Union[float, int] = 0.1, default_mult: float = 0.1):
        """
        Get the last size of the history.
        Parameters
        ----------
        n : float or int
            If float, it is a fraction of the history length.
            If int, it is the number of last elements to consider.
        """
        if n <= 1:
            n = int(n * len(self.history))
        
        if n > len(self.history):
            n = int(default_mult * len(self.history))
        return n

    def _save_json(self, params: dict, filename: str):
        """
        Save the parameters to a json file.
        Parameters
        ----------
        params : dict
            Parameters to save.
        filename : str
            Filename to save the parameters to.
        """
        with open(filename, "w") as f:
            json.dump(params, f, indent=4)
        self.logger.info(f"Saved parameters to {filename}", color="green", lvl=1)

    def report_gs(self, 
                eigv    : float               =   None, 
                last_n  : Union[float, int]   =   15,
                savedir : str                 =   None,
                plot_kw : dict                =   {},
                ):
        """
        Print time breakdown and plot energy, std, epoch times.
        """
        self.timings    = {k: np.array(v) for k, v in self.timings.items()}
        # time breakdown
        samp            = self.timings["sample"].sum()
        step            = self.timings["step"].sum()
        upd             = self.timings["update"].sum()
        prepare         = self.timings["prepare"].sum()
        solve           = self.timings["solve"].sum()
        gradient        = self.timings["gradient"].sum()
        tot             = samp + step + upd
        self.logger.info("Time breakdown (s):", color="blue", lvl=0)
        self.logger.info(f"- sampling: {samp:.2e} ({samp / tot * 100:.1f}%)", lvl=1)
        self.logger.info(f"- compute : {step:.2e} ({step / tot * 100:.1f}%)", lvl=1)
        self.logger.info(f"- update  : {upd:.2e} ({upd / tot * 100:.1f}%)", lvl=1)
        self.logger.info(f"- prepare : {prepare:.2e} ({prepare / tot * 100:.1f}%)", lvl=1)
        self.logger.info(f"- solve   : {solve:.2e} ({solve / tot * 100:.1f}%)", lvl=1)
        self.logger.info(f"- gradient: {gradient:.2e} ({gradient / tot * 100:.1f}%)", lvl=1)
        self.logger.info(f"- total   : {tot:.2e}", color="green")

        # prepare data
        energies        = self.history[~np.isnan(self.history)][:-2] / self.nqs.size
        energies_std    = self.history_std[~np.isnan(self.history_std)][:-2] / self.nqs.size
        sample_times    = self.timings["sample"] / self.nqs.size
        step_times      = self.timings["step"]   / self.nqs.size
        update_times    = self.timings["update"] / self.nqs.size
        grad_times      = self.timings["gradient"] / self.nqs.size
        prepare_times   = self.timings["prepare"] / self.nqs.size
        solve_times     = self.timings["solve"] / self.nqs.size
        epoch_times     = np.sum([self.timings[k] for k in self.timings.keys()], axis=0) / self.nqs.size
        last_mean       = np.nanmean(energies[-self._last_size(last_n):])
        last_std        = np.nanstd(energies[-self._last_size(last_n):])
        last_time       = np.nanmean(epoch_times[-self._last_size(last_n):])

        # plotting
        if plot_kw is not None:
            fig, ax     = Plotter.get_subplots(nrows    = 2,
                                            ncols       = 1,
                                            figsize     = (4, 5),
                                            dpi         = 150,
                                            sharex      = True)
            max_all     = np.nanmax(np.real(self.history))
            min_all     = np.nanmin(np.real(self.history))
            inset_up    = last_mean > (max_all - min_all) / 3
            
            if inset_up:
                axin    = ax[0].inset_axes(plot_kw.get("inset_axes", [0.6, 0.3, 0.38, 0.3]), zorder=10)
            else:
                axin    = ax[0].inset_axes(plot_kw.get("inset_axes", [0.2, 0.02, 0.38, 0.3]), zorder=10)
            ax          = [ax[0], axin, ax[1]]
            
            #! energies
            x           = np.arange(len(energies))
            Plotter.plot(ax[0], x=x, y=energies, marker="o", markersize=0.5, lw=1)
            Plotter.hline(ax[0], val=last_mean, lw=0.5, ls=':', label=f"Mean {last_mean:.3e}")
            # Plot upper and lower std limits at each epoch
            
            if eigv is not None:
                Plotter.hline(ax[0], val=eigv / self.nqs.size, color='r', linestyle="--", lw=0.5, label=f"GS {eigv/self.nqs.size:.3e}")

            #! fill between
            ax[0].fill_between(
                x,
                energies - energies_std,
                energies + energies_std,
                color   = "gray",
                alpha   = 0.2,
                label   = r"$\pm \sigma_E$",
            )
            ymin        = np.min(energies - energies_std)
            ymax        = np.max(energies + energies_std)
            ymargin     = 0.05 * (ymax - ymin)
            xlim        = (1, x[-1])
            ylim        = (ymin - ymargin, ymax + ymargin)
            ylim        = plot_kw.get("ylim_0", ylim)
            Plotter.set_legend(ax[0], loc=plot_kw.get("legend_loc_0", "upper right"),
                            fontsize=plot_kw.get("fontsize", 8), frameon=True, framealpha=0.8)
            Plotter.set_ax_params(ax[0], 
                            ylabel      =   r"$E/N_s$",
                            yscale      =   plot_kw.get("yscale_0", 'symlog'),
                            xscale      =   plot_kw.get("xscale_0", 'log'),
                            ylim        =   ylim,
                            xlim        =   xlim,
                            fontsize    =   plot_kw.get("fontsize", 8))
            Plotter.set_tickparams(ax[0], maj_tick_l=2, min_tick_l=1)
            Plotter.set_annotate_letter(ax[0], iter=0,
                                        fontsize=plot_kw.get("fontsize", 8),
                                        x = plot_kw.get("annotate_x", 0.1),
                                        y = plot_kw.get("annotate_y", 0.1))
            ax[0].set_title(f"GS train {self.nqs._hamiltonian}", fontsize=8)
            
            #! std
            xlim        = (1, x[-1])
            ylim        = plot_kw.get("ylim_1", None)
            Plotter.plot(ax[1], x=x, y=energies_std, marker="o", markersize=0.5, lw=1)
            Plotter.set_ax_params(ax[1], ylabel=r"$\sigma_E/N_s$",
                            xlabel="$i$", yscale='log',
                            xlim=xlim, ylim=ylim,
                            xscale='log')
            Plotter.set_tickparams(ax[1], maj_tick_l=2, min_tick_l=1)
            if not inset_up:
                Plotter.set_label_cords(ax[1], which='x', inX=0.5, inY=1.1)
                Plotter.set_ax_params(ax[1], which='x', tickPos='top', scale='log')
                ax[1].set_xticklabels([])

            #! epoch times
            xlim        = (1, x[-1] + 1)
            ylim        = plot_kw.get("ylim_2", None)
            Plotter.plot(ax[2], x=np.arange(len(epoch_times)), y=epoch_times,
                        markersize=0.5, lw=1, color=next(colorsCycle), label="epoch")
            Plotter.plot(ax[2], x=np.arange(len(sample_times)), y=sample_times, ls=next(linestylesCycle),
                        markersize=0.5, lw=1, alpha=0.5, label="sample", color=next(colorsCycle))
            Plotter.plot(ax[2], x=np.arange(len(step_times)), y=step_times, ls=next(linestylesCycle),
                        markersize=0.5, lw=1, alpha=0.5, label="step", color=next(colorsCycle))
            Plotter.plot(ax[2], x=np.arange(len(update_times)), y=update_times, ls=next(linestylesCycle),
                        markersize=0.5, lw=1, alpha=0.5, label="update", color=next(colorsCycle))
            Plotter.plot(ax[2], x=np.arange(len(grad_times)), y=grad_times, ls=next(linestylesCycle),
                        markersize=0.5, lw=1, alpha=0.5, label="gradient", color=next(colorsCycle))
            Plotter.plot(ax[2], x=np.arange(len(prepare_times)), y=prepare_times, ls=next(linestylesCycle),
                        markersize=0.5, lw=1, alpha=0.5, label="prepare", color=next(colorsCycle))
            Plotter.plot(ax[2], x=np.arange(len(solve_times)), y=solve_times, ls=next(linestylesCycle),
                        markersize=0.5, lw=1, alpha=0.5, label="solve", color=next(colorsCycle))
            Plotter.set_ax_params(ax[2], xlabel="Epoch", ylabel=r"$t_{\rm epoch}[s/N_s]$",
                            xscale=plot_kw.get("xscale_2", 'log'),
                            yscale=plot_kw.get("yscale_2", 'log'),
                            xlim=xlim, ylim=ylim)
            Plotter.set_tickparams(ax[2], maj_tick_l=2, min_tick_l=1)
            Plotter.set_legend(ax[2], loc=plot_kw.get("legend_loc_2", "upper right"),
                            fontsize=plot_kw.get("fontsize", 8), frameon=True, framealpha=0.8)
            Plotter.set_annotate_letter(ax[2], iter=1, 
                                        fontsize=plot_kw.get("fontsize", 8),
                                        x = plot_kw.get("annotate_x", 0.1),
                                        y = plot_kw.get("annotate_y", 0.1))
            fig.tight_layout()

            if savedir is not None:
                #! create the directory
                os.makedirs(savedir, exist_ok=True)
                
                # save figure
                fig_path = os.path.join(savedir, "gs_train.png")
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Saved figure to '{fig_path}' and JSON to '{savedir}'.")

                # prepare JSON data
                info_path   = os.path.join(savedir, "gs_train_info.json")
                self._save_json({
                                    "gs_energy":   last_mean,
                                    "gs_std":      last_std,
                                    "epoch_time":  last_time,
                                    "n_points":    len(energies),
                                    "history":     list(self.history),
                                    "lr_history":  list(self.lr_history),
                                    "reg_history": list(self.reg_history),
                                }, info_path)
                try:
                    params_path = os.path.join(savedir, "gs_train_params.json")
                    params_nqs  = self.nqs.get_params(unravel=True)
                    # transform jax to numpy
                    params_nqs  = list(np.array(params_nqs))
                    self._save_json({'params': params_nqs}, params_path)
                except Exception as e:
                    self.logger.warning(f"Failed to save parameters: {e}")
                    self.logger.warning("Parameters are not saved.")
                    self.logger.warning("You can save them manually using `nqs.get_params()`.", color="red", lvl=1)
            return fig, ax
        return None, None
    
    # ------------------------------------------------------
    
    def __repr__(self):
        return f"NQSTrainer(nqs={self.nqs}, ode_solver={self.ode_solver}, tdvp={self.tdvp}"
    
    def __str__(self):
        return f"NQSTrainer(nqs={self.nqs}, ode_solver={self.ode_solver}, tdvp={self.tdvp}"
    
    def __call__(self, *args, **kwargs):
        """
        Call the train method with the given arguments.
        """
        return self.train(*args, **kwargs)
    
# ------------------------------------------------------
#! EOF