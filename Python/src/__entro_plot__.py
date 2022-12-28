from .__helper__ import *
from .__models__ import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='w'
import matplotlib.colors as mcolors
import struct
from scipy import stats

FLOAT_SIZE = 8

kPSep = os.sep

colors_ls = (list(mcolors.TABLEAU_COLORS)[:120])
colors_ls_cyc = itertools.cycle(list(mcolors.TABLEAU_COLORS)[:120])
markers_ls = ['o','s','v', '+', 'o', '*']
markers = itertools.cycle(markers_ls)

mpl.rcParams.update(mpl.rcParamsDefault)
#mpl.rcParams['figure.dpi'] = 400
#mpl.rcParams['figure.figsize'] = 400
mpl.rcParams['lines.markersize']='4'
from matplotlib.ticker import ScalarFormatter, NullFormatter
# -------------------------------------------- PLOT HELPERS ------------------------------------

'''
Inserting inset with a specific format
'''
def add_subplot_axes(ax,rect):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.15
    y_labelsize *= rect[3]**0.15
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

# -------------------------------------------- PAGE --------------------------------------------
def page_result(d_a, d_b):
    if d_a <= d_b:
        return psi(d_a * d_b + 1) - psi(d_b + 1) - (d_a - 1)/(2*d_b)
    else:
        return psi(d_a * d_b + 1) - psi(d_a + 1) - (d_b - 1)/(2*d_a)

def page_result_var(d_a, d_b):
    return (((d_a + d_b)/(d_a*d_b + 1.0)) * polygamma(1, d_b + 1)) - polygamma(1, d_a*d_b + 1) - ((d_a-1)*(d_a + 2.0 * d_b - 1.0))/(4.0 * d_b * d_b * (d_a * d_b  + 1.0))

def page_thermodynamic(f, L):
    return f * L * np.log(2.) - np.power(2., -np.abs(1.-2.*f)*L - 1.0)

def page_thermodynamic_var(f, L):
    return (1/2 - 1/4 * (1.0 if f == 1/2 else 0.0)) * np.power(2., -(1+np.abs(1-2*f))*L)

def their_result(fraction_h, V):
    frac = fraction_h if fraction_h <= 1 else (V+2)*fraction_h/np.power(2.0, V)
    val = (V-1.0)*np.log(2)/2.0
    val += 2.0 * (np.exp(-np.power(erfinv(frac), 2.0)) - 1.0) / (frac * np.pi)
    return  val + (-2.0 + 2.0 * frac + np.exp(-np.power(erfinv(frac), 2.0)))*erfinv(frac)/(2.0*frac*np.sqrt(np.pi)) 
# -------------------------------------------- OTHER --------------------------------------------

# standard maximum value
page_difference = lambda x, l: np.array(page_thermodynamic(0.5, l) - x).flatten()
# page results value not in thermodynamic limit
digamma_difference = lambda x, l: np.array(page_result(np.power(2, l//2), np.power(2, l//2)) - x).flatten()
# page results value not in thermodynamic limit
digamma_difference_v = lambda x, l: np.array(x-page_result_var(np.power(2, l//2), np.power(2, l//2))).flatten()

their_limit_l = 200

########################################## PLOT ALL TOGETHER - PARAMETER SWEEP ##########################################

'''
Plot the heatmap of the pivoted values of the dataframe where column is the z'th value and labels can be set
'''
def plot_heatmap_values(pivot : pd.DataFrame, column : str, xlabel : str, ylabel : str, title : str):
    
    x = np.array(pivot.index)
    y = np.array([float(i[-1]) for i in pivot.columns])
    
    x_center = 0.5 * (x[:-1] + x[1:])
    y_center = (y[:-1] + y[1:])/2

    X, Y = np.meshgrid(x_center, y_center)

    z_mx = np.array(pivot[column])
    total_min = find_nearest_idx(pivot, column, 0.0)
    total_min = pivot[column].loc[total_min]
    
    # max and min for the cmap
    z_min, z_max = np.min(z_mx), np.max(z_mx)

    # plot info
    fig, ax = plt.subplots(figsize = (8, 8))
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(f'$S_r = N\log (2) /2 - 1/2 + (1/2 - \log(2))/2$\n$S-S_r$, type={column}, \n{title}')
    ax.tick_params(labelrotation=0)

    # norm for the cmap
    divnorm=mcolors.TwoSlopeNorm(vmin=z_min, vcenter=(z_max + z_min) / 2.0, vmax=z_max)

    c = ax.pcolormesh(x, y, z_mx.T, cmap='RdBu', norm=divnorm, shading='gouraud', label = column)#, vmin=z_min, vmax=abs(z_min))
    # contour needs the centers
    cset = ax.contour(X, Y, z_mx[1:,:-1].T, cmap='gray')
    ax.clabel(cset, inline=True)

    # limits
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_xlim(np.min(x), np.max(x))

    fig.colorbar(c, ax=ax)
    plt.legend()
    #plt.savefig(f'type={column}, \n{tit}', facecolor='white')
    plt.show()

'''
Plot entropy mean values in two parameter sweep
- pivot : pivot table to be taken from
- sample num : number of samples of index
- col : which entropy is it
- title : sets the title for the plot
- xlabel : sets the xlabel
- label : sets the col_a label acctually
'''
def plot_two_params_entro(pivot : pd.DataFrame, sample_num : int,
                          col : str, label : str, title : str, xlabel : str,
                          N : int):
    # iterate through the indices
    for val in pivot.sample(sample_num).index:
        pivot[col].loc[val].plot(label = f'{label}$={val}$')

    plt.axhline(-constant_page_correction(N) + page_val(N), label = '$S_{Page} - S_{rand}$', linestyle = "--", color='black')
    plt.axhline(0.0, linestyle='--', color = 'gray', label = '$S_{rand}$')
    plt.title(col + f',L={N},' + title)
    plt.ylabel('$<S> - S_{rand}$')
    plt.xlabel(xlabel)
    plt.legend()

 
    
########################################## PLOT FIT #################################################
'''

'''
def plot_fit(values, x, ax, label, color, fit_fun):
    # extract the power law behavior
    #log_val = np.log(values)
    #log_1_ov_x = np.log(1/x)
    #funct=(lambda x, a : np.power(1.0/x, a))
    #popt1, pcov1 = curve_fit(funct, 1.0/x, values)
    #print(popt1)

    
    #print(x, 1.0/x, values)
    
    param_bounds= ([0, -np.inf, 0],[np.inf,np.inf, min(values)]) if (fit_fun == fit_one_over_v2) else (([-np.inf, 0],[np.inf, min(values)]) if fit_fun == fit_one_over_v else ([-np.inf, -np.inf],[np.inf, np.inf]))
    popt, pcov = curve_fit(fit_fun, 1.0/x, values, bounds=param_bounds)
    
    xran = np.arange(10.0, 1e5, 2)
    #fit_val = fit_one_over_v(xran, popt[0], popt[1], popt[2])
    #fit_val = fit_one_over_v(xran, popt[0], popt[1])
    fit_val = fit_fun(xran, *popt)
    #print(xran[0:5], 1.0/xran[0:5],fit_val[0:5])
    #ax.plot(1.0/xran,funct(xran, *popt1), '--', color = color, alpha = 0.5)
    ax.plot(1.0/xran, fit_val, '--', color = color, alpha = 0.5)
    ax.scatter(1.0/np.max(xran), fit_fun(np.max(xran), *popt), s = 25, c = color)
    
    
    #ins = ax.inset_axes([0.1,0.1,0.2,0.2])
    #xs = np.array([1e14] + [i for i in range(int(1e4), int(1e5), 2)])
    #ins.plot(1.0/xs, fit_fun(xs, *popt), c=color)
    
    text = f'{label}:{popt}'
    return text, popt


'''
Fits the change of the entanglement entropy in terms of fraction
'''
def fit_fractions(values, x, ax, label, color):
    function = lambda x, a, b, c: c * np.exp(a*(np.power(x,b)))
    param_bounds= ([0, 0, 0],[np.inf,np.inf,np.inf])
    popt, pcov = curve_fit(function, x, values, bounds=param_bounds)
    xran = np.arange(np.min(x), np.max(x), 1e-4)
    fit_val = function(xran, *popt)
    lbl = r'$%se^{%sx^%s}$'%(f'{popt[2]:.2f}', f'{popt[0]:.2f}', f'{popt[1]:.2f}')
    ax.plot(xran, fit_val, '--', color = color, alpha = 0.5, label = label + f', fit: {lbl}')
    text = f'{label}:{popt}'
    return text, popt

########################################## PLOT DIFFERENCE ##########################################

'''
Plots the difference between average entropy and a Page value
- L : list of all system sizes
- fractions : list of fractions to be plotted
- xscale, yscale : scales
- fit_fun : fit_function for size scaling
- y_lim : limit of y_axis 
- av_name : what sector shall be plotted on [0][1]
- read_log : if we shall read existing log or create new one
- su2 : sz symmetry conservation
- verbose : talk?
- fit_frac_num : number of constant fractions to be used 
- fit_frac_step : step of the constant fraction on [1][0]
'''
def plot_difference_log(L : list, fractions : list, directory : str,
                    params : dict,
                    xscale = 'linear', yscale = 'linear',
                    fit_fun = fit_one_over_v, ylim = [1e-2, 1],
                    av_name = 'imag',
                    read_log = True, su2 = False, verbose = False,
                    fit_frac_num = 100, fit_frac_step = 0.01,
                    ):
    
    directory_l = lambda l: directory + f"{kPSep}resultsXYZ{l}{kPSep}"
    fontsize = 13.5
    # ----------------------------- SET THE FIG AND AX --------------------------------
    fig, ax = plt.subplots(2, 2, figsize = (18,18))
    color_real, color_other = tuple(random.sample(colors_ls, 2))
    marker_real, marker_other = tuple(random.sample(markers_ls, 2))
    marker_s = 32
    frac = fractions[0]
    # create the dataframe for the averages
    frac_df = pd.DataFrame(index=fractions + ['max'], columns = L).fillna(0)

    inverse_L = 1.0 / np.array(L)
    # ----------------------------------------- PLOT PROPERTIES ------------------------------------------
    
    xlim = [0, 1.0/(np.min(L)-1)]
    xticks = [0.0] + list(1.0/np.array(np.arange(np.min(L)-2, np.max(L)+2, 2)))
    xlabels = [0] + [f'1/{i}' for i in np.arange(np.min(L)-2, np.max(L)+2, 2)]
    ################# [0][0] ##################
    ax[0][0].set_yscale(yscale)
    ax[0][0].set_xscale(xscale)
    ax[0][0].set_xticks(xticks, xlabels, rotation = -45)
    if xscale == 'linear':
        ax[0][0].set_xlim(xlim)
    ax[0][0].set_xlabel("$1/V$", fontsize = fontsize)
    ax[0][0].set_ylabel("$S_{max} - <S_A>_{sec}$", fontsize = fontsize)
    ax[0][0].set_title("$[V\ln{2}/2 - 1/2] - <S_A>$ for different sectors. " + f"ν = {fractions[0]}.", fontsize = fontsize)
    
    ################# [0][1] ##################
    ax[0][1].set_yscale(yscale)
    ax[0][1].set_xscale(xscale)  
    ax[0][1].set_xlabel("$1/V$")
    #ax[0][1].set_yticks(yticks, ylabels)
    ax[0][1].set_xticks(xticks, xlabels, rotation = -45)
    if xscale == 'linear':
        ax[0][1].set_xlim(xlim)
    if yscale == 'linear':
        ax[0][1].set_ylim(ylim)
    ax[0][1].set_ylabel("$S_{page} - <S_A>_{av}^{ν}$", fontsize = fontsize)
    ax[0][1].set_title("$S_{page} - <S_A>_{av}^{ν}$, different ν" + f", {av_name} sectors", fontsize = fontsize)
    
    ################# [1][0] ##################
    ax[1][0].set_yscale(yscale)
    ax[1][0].set_xlim(0.0, 0.5)
    ax[1][0].set_xlabel("Constant fraction")
    ax[1][0].set_ylabel("$S_{page} - <S_A>_{v}$", fontsize = fontsize)
    ax[1][0].set_title("Change of entanglement entropy difference from Page in fraction size", fontsize = fontsize)
    
    ################# [1][1] ##################
    ax[1][1].set_yscale(yscale)
    ax[1][1].set_xscale(xscale)
    #ax[1][1].set_yticks(yticks, ylabels)
    ax[1][1].set_xticks(xticks, xlabels, rotation = -45)
    if xscale == 'linear':
        ax[1][1].set_xlim(xlim)
    if yscale == 'linear':
        ax[1][1].set_ylim(ylim)
    ax[1][1].set_xlabel("$1/V$", fontsize = fontsize)
    ax[1][1].set_ylabel("$S_{page} - \overline{<S_A>}^{ν}$", fontsize = 16)
    ax[1][1].set_title("$S_{page} - \overline{<S_A>}^{ν}$ averaged over sectors - " + f"ν = {fractions[0]}", fontsize = fontsize)

    
    # set the fractions columns
    frac_cols = [f'S_f={frac:.3f}' for frac in fractions]
    all_cols = frac_cols + ['S_max']
    # save all average entropies into one df
    all_df = pd.DataFrame(columns=all_cols)
    all_df['Ns']=np.array(L)
    all_df.set_index('Ns', inplace=True)
    # set page values
    page_vals = np.array([page_thermodynamic(0.5, l) for l in L])
    
    # --------------------------------------------- GET THE VALUES ------------------------------------------------------
    real_plot, img_plot = None, None

    av_real = []
    av_imag = []
    av_all = []
    maxima = []
    # iterate all lattice sizes
    for l in L:
        # read the log file
        df = get_log_file(directory=directory_l(l), read_log=read_log, su2=su2)
        df = parse_dataframe(df, params)
        # set the gap ratio
        set_gap_ratios_df_log(df=df, directory=directory_l(l))
        # check the fractions that are not yet calculated and calculate them
        frac_left = [float(frac.split('=')[-1]) for frac in frac_cols if frac not in df.columns]
        if len(frac_left) != 0:
            set_entropies_df_log(df=df, directory=directory_l(l), fractions=frac_left, set_max=True, verbose=verbose)
            
        # sum the hilbert space
        sum_hilbert_space = float(df['Nh'].sum())
        print(f'L={l}:N={int(sum_hilbert_space)}')
        # norm in the sum later
        divider = [int(sum_hilbert_space * fr) if fr < 1.0 else int(len(df) * fr) for fr in fractions]
        
        # check all seperately and save the averages
        for index, row in df.iterrows():
            marker = marker_real if row['sec'] == 'real' else marker_other
            color = color_real if row['sec'] == 'real' else color_other
            if row['sec'] == 'real':
                real_plot = ax[0][0].scatter(1/l, -row[f'S_f={frac:.3f}'] + page_thermodynamic(0.5, l),
                        marker = marker, color = color, s=marker_s)
            else:
                img_plot = ax[0][0].scatter(1/l, -row[f'S_f={frac:.3f}'] + page_thermodynamic(0.5, l),
                        marker = marker, color = color, s=marker_s)
            # multiply entropy by Nh
            Nh = float(row['Nh'])
            nums = [int(Nh * fr) if fr < 1.0 else fr for fr in fractions]
            for i, col in enumerate(frac_cols):
                df.loc[index, col] = row[col] * nums[i]
        
        # save the sum and divide by the total hilbert for an average
        all_df.loc[l] = df[frac_cols].sum()
        
        # add maximum
        all_df.loc[l, 'S_max'] = df['S_max'].max()
        for i, col in enumerate(frac_cols):
            all_df.loc[l, col] = all_df.loc[l, col] / divider[i]
        
        df['i'] = np.ones(len(df))    
        # groupby and save averages
        df_sec = df.groupby('sec').sum()
        for index, row in df_sec.iterrows():
            Nh = float(row['Nh'])
            length = int(row['i'])
            for i, col in enumerate(frac_cols):
                num = int(Nh * fractions[i]) if fractions[i] < 1.0 else int(length * fractions[i])
                df_sec.loc[index, col] = row[col] / num
            df_sec.loc[index, 'S_max'] /= length
        # save all fractions         
        frac_df[l] = np.array(df_sec.loc[av_name, all_cols])
        # save the averages
        av_real.append(float(df_sec.loc['real',f'S_f={frac:.3f}']))
        av_imag.append(float(df_sec.loc['imag',f'S_f={frac:.3f}']))
        maxima.append(float(df[f'S_max'].max()))
        av_all.append(float(df_sec[f'S_f={frac:.3f}'].mean()))      
        
    # ------------------------------------- PAGE -------------------------------------------
    for col in all_cols:
        all_df[col] = page_vals - all_df[col]
    all_df['page'] = page_vals
    
    # ------------------------------------- FITS -------------------------------------------
    fits = {}
    marker_s = 8
    # plot averages
    L = np.array(L)
    if len(av_real) != 0:
        color = next(colors_ls_cyc)
        values = np.array([page_vals[i] - av_real[i] for i in range(len(L))]).flatten()
        #t, popt = plot_fit(values[np.isfinite(values)], inverse_L[np.isfinite(values)], ax[1][1], '', color, fit_fun)
        ax[1][1].plot(inverse_L, values, marker = next(markers), color = color, label = f"real sectors", markersize=marker_s)
        #fits['real']=popt
    
    if len(av_imag) != 0:
        color = next(colors_ls_cyc)
        values = np.array([page_vals[i] - av_imag[i] for i in range(len(L))]).flatten()
        
        #, fit:{t}t, popt = plot_fit(values[np.isfinite(values)], inverse_L[np.isfinite(values)], ax[1][1], '', color, fit_fun)
        ax[1][1].plot(inverse_L, values, marker = next(markers), color = color, label = f"imaginary sectors", markersize=marker_s)
        #fits['img']=popt
        
    if len(av_all) != 0:    
        color = next(colors_ls_cyc)
        values = np.array([page_vals[i] - av_all[i] for i in range(len(L))]).flatten()
        #t, popt = plot_fit(values[np.isfinite(values)], inverse_L[np.isfinite(values)], ax[1][1], '', color, fit_fun), fit:{t}
        ax[1][1].plot(inverse_L, values, marker = next(markers), color = color, label = f"all sectors", markersize=marker_s)
        #fits['all']=popt
    
    if len(maxima) != 0:
        color = next(colors_ls_cyc)
        values = np.array([page_vals[i] - maxima[i] for i in range(len(L))]).flatten()
        ax[1][1].plot(inverse_L, values, marker = next(markers), color = color, label = f"outliers", markersize=marker_s)

    # ------------------------------------- ALL FRACTIONS -----------------------------------
    print(frac_df)
    l = 100
    fits_av = {}
    for i, row in frac_df.iterrows():
        color=next(colors_ls_cyc)
        if type(i) == float:
            limit = page_thermodynamic(1/2, l) - their_result(float(i), l)
            ax[0][1].plot(inverse_L, page_vals-np.array(row), marker = next(markers), color = color, label=f"ν={i}")
            t, popt = plot_fit(np.array(row), inverse_L, ax[0][1], '', color, fit_fun)
            txt = f"limit : {limit:.4f},v={i}"
            ax[0][1].axhline(y=limit, color = color, linestyle='--', alpha = 0.3, label = txt)
            fits_av[i] = popt
        else:
            ax[0][1].plot(inverse_L, page_vals-np.array(row), marker = next(markers), markersize = marker_s, color = color, label=f"ν={i}")
    
    
    # ------------------------------------- FRACTIONS FIT -----------------------------------
    
    fractions_fit = [0.5 - fit_frac_step * i for i in range(fit_frac_num) if (0.5 - fit_frac_step * i > 0)]
    fractions_fit_lb = [f'S_f={fr:.3f}' for fr in fractions_fit]
    ax_sub = ax[1][0].inset_axes([0.25,0.65,0.3,0.3])
    ax_sub.set_title('Vanishing fraction')
    
    r = []
    for l in L:
        color = next(colors_ls_cyc)
        marker = next(markers)
        # read the log file
        df = get_log_file(directory=directory_l(l), read_log=read_log, su2=su2)
        df = parse_dataframe(df, params)
        # constant fractions
        minim = df['Nh'].min() // 3
        step = minim / 25
        const_fractions_fit = [20] + [float(minim - step * i) for i in range(round(minim / step))]
        const_fractions_lb = [f'S_f={fr:.3f}' for fr in const_fractions_fit]
        frac_together = np.array(fractions_fit + const_fractions_fit)
        # set the entropies
        set_entropies_df_log(df=df, directory=directory_l(l),
                             fractions= frac_together, set_max=True, verbose=verbose)
        
        # take the hilbert sum
        sum_hilbert_space = df['Nh'].sum()
        
        # constant fractions plot
        for index, row in df.iterrows():
            Nh = float(row['Nh'])
            df.loc[index, fractions_fit_lb] = row[fractions_fit_lb] * Nh / sum_hilbert_space
        const_frac_df = page_thermodynamic(1/2.0, l) - df[fractions_fit_lb].sum()
        text, popt = fit_fractions(np.array(const_frac_df), np.array(fractions_fit), ax[1][0], f'L={l}', color)
    
        r.append(ax[1][0].scatter(np.array(fractions_fit), np.array(const_frac_df), color=color, marker=marker))
        
        # set vanishing fractions
        vanish_frac_df = page_thermodynamic(1/2.0, l) - df[const_fractions_lb].mean(axis=0) 
        ax_sub.scatter(np.array(const_fractions_fit)/sum_hilbert_space, np.array(vanish_frac_df),color=color, marker=marker)
            
    ax_sub.set_xlabel('$N_{fr}/N_{H}$')    
        
    
    # ------------------------------------- PLOT PROPERTIES -----------------------------------
    # plot limit according to Nuclear Physics B 966(2021)
    limit = page_thermodynamic(0.5, their_limit_l) - their_result(fractions[0], their_limit_l)
    print(limit)
    text = f"limit : {limit:.4f},v={fractions[0]}"
    lim = ax[0][0].axhline(y=limit, color='blue', linestyle='--', alpha = 0.8)
    ax[0][0].legend([real_plot, img_plot, lim], ['real sectors', 'imaginary sectors', text], fontsize = fontsize, loc = 'upper left')
    ax[0][1].legend(fontsize = fontsize, loc = 'best')
    ax[1][0].legend(fontsize=fontsize, loc = 'lower right')   
    
    ax[1][1].axhline(y=limit, color='blue', linestyle='--', alpha = 0.8, label = text)
    ax[1][1].legend(fontsize = fontsize, loc='best')
    
    plt.savefig(directory + f'xyz_size_scaling_{"su2" if su2 else ""}.png', facecolor='white')
    plt.savefig(directory + f'xyz_size_scaling_{"su2" if su2 else ""}.pdf', facecolor='white')
    return all_df, all_cols
    
################################################# EIGENSTATES #####################################################

'''
Plot the histogram of the eigenstates in the middle of the spectrum averaged over some interval
'''
def plot_eig_hist(L, frac : int, directory : str, params : dict, bins = 100, dens = False):
    function = lambda x, mu, sig: np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    directory_l = lambda l: directory + f"{kPSep}resultsXYZ{l}{kPSep}"
    x_data = np.zeros(bins)
    g = np.zeros(bins)
    
    # check which col will be used for plotting gaussianity
    col_gauss = None
    length = 1
    for key in params:
        if len(params[key]) > 1:
            col_gauss = key
            length = len(params[key])
            break
    
    # create axes
    textsize = 24
    markersize = 55
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize = (32, 10)) 

    ax_sub = ax.inset_axes([0.65,0.1,0.3,0.35])
    ax_sub.set_title(f'Vanishing fraction v={frac}')
    ax_sub.set_xlabel('L')
    ax_sub.set_ylabel('$S_{page}-<S>$')
    
    # save the parameters
    colors = [next(colors_ls_cyc) for i in range(len(L))]
    marker = [next(markers) for i in range(len(L))]
    
    color_param = [next(colors_ls_cyc) for i in range(length)]

    ls = []
    # create handles for legend
    hand_fidel = []
    hand_gauss = []
    hand_entro = []
    for i, l in enumerate(L):
        log = get_log_file(directory=directory_l(l), read_log=False, su2=False)
        if len(log) == 0: continue
        log = parse_dataframe(log, params)
        set_gap_ratios_df_log(log, directory_l(l))
        set_entropies_df_log(log, directory_l(l), [frac], False)
        
        # iterate rows that are left
        j = 0
        for idx, row in log.iterrows():     
            short = row['model_short']
            print(short)
            states = get_eigenstates(directory_l(l), short, 'states')
            
            if len(states) == 0:
                break
            if states.dtype == [('real', '<f8'), ('imag', '<f8')]: states = states.view('complex')
            states = np.square(np.abs(states))
            states_num = states.shape[-1]
            # find modulus fidelity
            fidel_in = 1.0/modulus_fidelity(states)                        
            states = states[:, states_num//2-frac//2 : states_num//2+frac//2].flatten()  * np.sqrt(row['Nh'])
            #if SYM:
            #    states=np.append(states, np.zeros(row['Nh']*states_num))
            
            # find the gaussianity
            gauss_in = gaussianity(states)       
            
            # find the total histogram
            hist, x_data = np.histogram(states, bins, density = dens)
            # which column to take for entropy
            col = f"S_f={frac:.3f}"      
            S = page_thermodynamic(1/2, l) - row[col]
            print(S)
            # plot the histogram
            lbl = f',{col_gauss}={row[col_gauss]}' if col_gauss is not None else ''
            ax.plot(x_data[0:-1], hist, label = f'$L={l}{lbl},<r>={row["gapratios"]:.4f},S_p-S={S:.4f}$', linestyle = '--', linewidth=3.5)
        
        
            ################## PLOTS #######################
            a, b, c = None, None, None
            if col_gauss is not None:
                val = row[col_gauss]
                a=ax2.scatter(val, fidel_in, marker = marker[i], color = colors[i], s=markersize + 15)
                             #, label = f'L={ls[i]},$1/M$')
                b=ax3.scatter(val, gauss_in, marker = marker[i], color = colors[i], s=markersize + 15)
                             #, label = f'L={ls[i]},$\Gamma$')
                c=ax_sub.scatter(1./l, S, color=color_param[j], s = markersize)#, label = f'{col_gauss}={params[col_gauss]}')
            else:
                print("L")
                a=ax2.scatter(1./l, fidel_in, color = color_param[j], label = '$1/M$', s = markersize + 15)
                b=ax3.scatter(1./l, gauss_in, color = color_param[j], label = '$\Gamma$', s = markersize + 15)
                c=ax_sub.scatter(1./l, S, color = color_param[j], s = markersize) 
            hand_entro.append(c)
            if j == 0 : 
                hand_fidel.append(a)
                hand_gauss.append(b)
            j+=1
    
    
    # plot gaussian    
    g = function(x_data, 0.0, 1.0) * 0.82
    ax.plot(x_data, g, label = 'Normal', color = 'black', linewidth=5.5, alpha = 0.3)
    
    # plot limit of the entropies on subaxis
    limit = page_thermodynamic(1/2, 100) - their_result(100, 100)
    ax_sub.axhline(limit, label = f'v={frac},Huang:{limit:.5f}', color = next(colors_ls_cyc))

    # plot 2 if is not None 
    ax2.axhline(np.pi/2.0, label = '$\pi / 2$')
    #ax2.set_yticks([np.pi/2.0], ['$\pi/2$'])
    ax2.set_title('Modulus fidelity inverse', fontsize=textsize)
    ax2.set_xlabel('$1/h_z$', fontsize=textsize)
    ax2.set_ylabel('$1/<M>$', fontsize=textsize)
    if col_gauss is not None:
        ax2.set_xlim(0.1, np.max(params[col_gauss])+0.1)
        ax2.set_xlabel(str(col_gauss), fontsize=textsize)
        ax2.legend(hand_fidel, [f'L={l}' for l in L], fontsize=textsize)    
    
    # plot 2 if is not None 
    ax3.axhline(np.pi/2.0)
    #ax3.set_yticks([np.pi/2.0], ['$\pi/2$'])
    ax3.set_xlabel('1/$h_z$', fontsize=textsize)
    ax3.set_title('Gaussianity', fontsize=textsize)
    ax3.set_ylabel('$<\Gamma >$', fontsize=textsize)
    if col_gauss is not None:
        ax3.set_xlim(0.1, np.max(params[col_gauss])+0.1)
        ax3.set_xlabel(str(col_gauss), fontsize=textsize)
        ax3.legend(hand_gauss, [f'L={l}' for l in L], fontsize=textsize)      
    
    
    # plot style  
    ax.set_ylim(0)
    ax.set_xlim(0, max(x_data))
    ax.set_xlabel('$D^{1/2}c$', fontsize=textsize)
    ax.set_ylabel('$P(z)$', fontsize=textsize)
    ax.set_title(f'Normalized histogram for {int(frac)} eigenstates\n{print_dict(params)}', fontsize=textsize)
    ax.legend()
    if col_gauss is not None:
        ax_sub.legend(hand_entro[-len(params[col_gauss]):], [f'{col_gauss}={i}' for i in params[col_gauss]])
    plt.savefig(directory + f'gauss_xyz_{print_dict(params)}.png', facecolor='white')
    plt.savefig(directory + f'gauss_xyz_{print_dict(params)}.pdf', facecolor='white') 
        
################################################# ENTROPIES TOGETHER #####################################################
'''

'''
def plot_df_together(L, dfs, energies_ls, directory, su2 : bool, xlim = [-10,15]):
    fontsize = 12.5
    fig, ax = plt.subplots(1, len(L), figsize = (14,10))
    for i in range(len(L)):
        ax[i].set_xlabel('Energy', fontsize = fontsize)
        ax[i].set_ylabel('S(E)', fontsize = fontsize)
    
    for i, l in enumerate(L):
        entropies = dfs[i].loc[l//2]
        energies = energies_ls[i]
        #entropies = np.array(entropies,:].T).flatten()

        color = next(colors_ls_cyc)
            
        ax[i].scatter(np.array(energies), np.array(list(entropies)),
                label = str(l), alpha = 0.5,
                c = color, marker=next(markers))

        val = page_result(np.power(2.0, l//2), np.power(2.0, l//2))
        ax[i].axhline(y = val, color = color, label = f'Page/L--L={l}')

    #ax.set_ylim([0, (page_result(np.power(2.0, max(L)//2), np.power(2.0, max(L)//2))+0.2)/max(L)])
    for i in range(len(L)):
        ax[i].legend(fontsize = fontsize)
        ax[i].set_ylim([0,7.5])
        ax[i].set_xlim(xlim)
    plt.savefig(directory + f"_rainbows,type={model_name},sym={SYM},L={L},{'su2' if su2 else ''}.png", facecolor='white')
    plt.savefig(directory + f"_rainbows,type={model_name},sym={SYM},L={L},{'su2' if su2 else ''}.pdf", facecolor='white')
    
'''

'''        
def plot_rescale_df(L, dfs, energies_ls, directory, su2 : bool):
    fontsize = 12.5
    fig, ax = plt.subplots(figsize = (14,10))
    ax.set_xlabel('Energy', fontsize = fontsize)
    ax.set_ylabel('S(E)', fontsize = fontsize)
    ax.set_title("$[S(E/l) - max(S(E/l))]/l)$")
    kolorki = []
    maxima = []
    for i, l in enumerate(L):
        res = page_result(np.power(2,l//2), np.power(2,l//2))
        entropies = np.array(dfs[i].loc[l//2])
        maxima.append(np.max(entropies))
        entropies = entropies/l
        entropies -= np.max(entropies)
        energies = energies_ls[i].flatten()/l
        
        color = next(colors_ls_cyc)
        kolorki.append(color)
        ax.scatter(energies, entropies,
                    label = str(l), alpha = 0.5,
                    c = color, marker=next(markers))
    print(maxima)
    for i, l in enumerate(L):
        res = page_result(np.power(2,l//2), np.power(2,l//2))
        print(l, (res)/l)
        ax.axhline(res/l - maxima[i]/l, color = kolorki[i], label = f'$[Digamma-max(S_L(E/L))/L--L={l}$')
    ax.legend(fontsize = fontsize)
    plt.savefig(directory + f"_rescale,type={model_name},sym={SYM},L={L},{'su2' if su2 else ''}.png", facecolor='white')
    plt.savefig(directory + f"_rescale,type={model_name},sym={SYM},L={L},{'su2' if su2 else ''}.pdf", facecolor='white')
    
    
    
'''

'''    
def maxima_df(L, dfs, ens, directory, su2 : bool, bin_num = 200):
    fig, ax = plt.subplots(3, figsize = (12,24))
    fontsize=12.5
    kolorki = []
    
    mean_idxs = []
    mean_vals = []
    
    max_idxs = []
    max_vals = []
    
    dos_idxs = []
    dos_vals = []
    
    ax[0].set_xlabel('E/V', fontsize = fontsize)
    ax[0].set_title('DOS')
    ax[1].set_xlabel('E/V', fontsize = fontsize)
    ax[2].set_xlabel('V', fontsize = fontsize)
    
    ax_sub = ax[1].inset_axes([0.1,0.1,0.3,0.3])
    
    for i, l in enumerate(L):
        entropies = dfs[i]
        energies = ens[i].flatten()
        
        color = next(colors_ls_cyc)
        kolorki.append(color)
        # ---------------------------------------------------------------------------------------------------------------
        df_roll = moving_average(entropies, roll_number)
        #print(df_roll.iloc[-1])
        roll_max = find_maximum(df_roll).iloc[-1]
        df_roll = df_roll.iloc[-1].to_numpy().flatten()

        # ---------------------------------------------------------------------------------------------------------------

        #print(entropies.columns, model.N)
        energies = energies
        idx_mean = find_nearest_idx_np(energies, np.mean(energies))
        val_mean = energies[idx_mean]
        
        print(val_mean, idx_mean)
        
        roll_val = energies[roll_max]
        
        # take entropies
        entropies = np.array(entropies.loc[l//2]).flatten()


        # ---------------------------------------------------------------------------------------------------------------
        dos, bins = np.histogram(energies, bins=bin_num)
        dos = np.array(dos, dtype=np.float32)
        bins = np.array(bins[1:])
        ax[0].scatter(bins, dos, color = color, label = str(l))
        parameters, covariance = curve_fit(gauss, bins, dos)
        gausik = gauss(bins, *parameters)
        
        
        # ---------------------------------------------------------------------------------------------------------------
        mean_idxs.append(idx_mean)
        mean_vals.append(val_mean)
        
        max_idxs.append(roll_max)
        max_vals.append(roll_val)
        
        dos_n = find_nearest_idx_np(energies, parameters[2])
        dos_v = energies[dos_n]
        dos_idxs.append(dos_n)
        dos_vals.append(dos_v)
        
    
        if l == max(L):
            idx_sub = find_nearest_idx_np(energies, np.max(energies))
            sub_num = int(0.08 * len(energies))
            ax_sub.scatter(energies[idx_mean - sub_num : idx_mean+sub_num], entropies[idx_mean - sub_num:idx_mean + sub_num], label = str(l), s = 1, color = 'yellow')
            ax_sub.plot(energies[idx_mean - sub_num : idx_mean+sub_num], df_roll[idx_mean - sub_num:idx_mean + sub_num], color = 'black')
            #ax_sub.scatter(energies, entropies/l, label = str(l))
            ax_sub.axvline(val_mean, label = f'mean={val_mean:.3e}', color = 'red')
            ax_sub.axvline(roll_val, label = f'max={roll_val:.3e}', color = 'blue')
            ax_sub.axvline(dos_v, label = f'dos={dos_v:.3e}', color = 'green')
            ax_sub.set_xticks([])
            #ax_sub.set_xticklabels([f'{i:.2e}' for i in [val_mean, roll_val, dos_v]], fontsize = 6, rotation = 45)
            ax_sub.set_yticks([])
            
        # ---------------------------------------------------------------------------------------------------------------
        ax[0].plot(bins, gausik, color=color)
        idx_max = np.argmax(gausik)
        ax[0].axvline(bins[idx_max], color = color, alpha = 0.5)

        
        ax[1].scatter(np.array(energies), np.array(entropies),
                   label = str(l),
                   c = color, marker=next(markers), alpha = 0.5)
        ax[1].plot(energies, df_roll, color ='red')

    ax[2].plot(L, dos_vals, label = 'dos', marker = next(markers))
    ax[2].plot(L, max_vals, label = 'roll_max', marker = next(markers))
    ax[2].plot(L, mean_vals, label = 'mean', marker = next(markers))
    
    ax[2].legend(fontsize = fontsize)
    ax[1].legend(fontsize = fontsize)
    ax[0].legend(fontsize = fontsize)
    ax_sub.legend(fontsize = 6)
    plt.savefig(directory + f'_xyz_maxima_{"su2" if su2 else ""}.png', facecolor = 'white')
    plt.savefig(directory + f'_xyz_maxima_{"su2" if su2 else ""}.pdf', facecolor = 'white')
    return [(mean_idxs[i], mean_vals[i])for i in range(len(L))],  [(max_idxs[i], max_vals[i])for i in range(len(L))],  [(dos_idxs[i], dos_vals[i])for i in range(len(L))]
