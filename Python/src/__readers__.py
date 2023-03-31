from .__entro_plot__ import *
from .__models__ import *
import h5py
import tqdm
import pandas as pd
from numba import njit


model_name = 'xyz'
roll_number = 4
SYM = True

# define the value taken for the middle of the spectrum
#IDX_VAL = 'roll'
IDX_VAL = 'mean'
# IDX_VAL = 'max'
# IDX_VAL = 'dos'

####################################################### GET LOG FILE #########################################################

'''
Parses the model according to a given string
- model string : contains all the information about the model
'''
def parse_model(model_string : str):
    #print(model_string)
    model_split = model_string.split(',')
    #print(model_split)
    model_name = model_split[0]
    columns = [str(i.split('=')[0]) for i in model_split[1:]]
    values = [float(i.split('=')[-1]) for i in model_split[1:]]
    if SYM:
        # find index of k
        columns.append('sec')
        k_index = columns.index('k')
        Ns_index = columns.index('Ns')
        k = int(values[k_index])
        Ns = int(values[Ns_index])
        if k==0 or k==Ns//2:
            values.append('real')
        else:
            values.append('imag')
        
    return columns, values

'''
Reads the entropies_log.dat file and creates a Dataframe out of it
- directory : directory of the log file
- head : if there is some predefined header
'''
def get_log_file(directory : str, head = None, read_log = True, su = False):
    #df = pd.read_csv(directory + 'entropies_log.dat', sep = '\t')
    #print(df)
    files = list(os.listdir(directory))[1:]
    files = [file for file in files if file.startswith('_') and file.endswith('.h5')]
    files = [file for file in files if 'sym' in file] if SYM else [file for file in files if 'sym' not in file]
    files.sort()
    df = pd.DataFrame()
    if read_log:
        df = pd.read_csv(directory + 'entropies_log.dat', sep = '\t', index_col=False,
                header = None, names = ['model', 'S_max', 'S_f=0.250', 'S_f=0.100', 'S_f=0.125', 'S_f=0.500', 'S_f=50.000', 'S_f=200.000', 'S_f=500.000', 'Nh']).sort_values('model').dropna(axis=1)
        df['model_short'] = df['model'].apply(lambda x : str(x).replace('0000', '00').replace('000', '0'))
    else:
        df['model'] = files
        df['model'] = df['model'].apply(lambda x : x[:-3])
        df['model_short'] = df['model']

    # check su
    if su:
        df = df[df['model_short'].str.contains('su')]
    else:
        df = df[~df['model_short'].str.contains('su')]
        
    # check head
    if head is not None:
        df = df.head(head)
    models = df['model'].to_list()
    value = []
    cols = None
    for i in models:
        columns, values = parse_model(i)
        if len(columns) != 0:
            cols = columns
            value.append(values)
    df[cols] = value

    if not read_log:
        df['Nh'] = df['spectrum_num'].astype(int)
    if SYM:
        df = df.sort_values(['k','p','x'])
    df.reset_index(inplace=True)
    df.drop(['index'], axis = 1, inplace=True)
    return df

'''
Takes the log DataFrame and calculates a gap_ratio for each of the models in it
- df : DataFrame of the log
- directory : directory where all the h5 files are stored
- use_mls : if we shall divide the gapratio by the `mean level spacing`
'''
def set_gap_ratios_df_log(df : pd.DataFrame, directory : str, use_mls = False):
    df.loc[:,'gapratios'] = np.zeros(len(df))
    # takes the Hilbert space sizes
    hilbert_sizes = df.loc[:,'Nh'].to_list()
    # takes the names of the models
    model_shorts = df.loc[:,'model_short'].to_list()
    #print(len(model_shorts), len(hilbert_sizes))
    gapratios = []

    for i, short in enumerate(tqdm.tqdm(model_shorts)):
        # read the energy to calculate the gap ratio
        #print(short, hilbert_sizes[i], df[df['model_short'] == short]['Nh'])
        filename = short + '.h5'
        if not 'spectrum' in short:
            filename = short + ',spectrum_num=' + str(int(hilbert_sizes[i])) + '.h5'
            
        # read the energy from a given filename (h5 Database)
        en = read_h5_file(directory, filename, 'energy')
        #ent = read_h5_file(directory, filename, 'entropy')
                
        if len(en) == 0: 
            gapratios.append(0)
            continue
        # calculate the gapratio
        gap_rat = gap_ratio(np.array(en), 0.5, use_mean_lvl_spacing=use_mls)
        #df.loc[short, 'gapratios'] = gap_rat
        gapratios.append(gap_rat)
    df['gapratios'] = gapratios

'''
Takes the log DataFrame and calculates the entropy fractions for each of the models in it
- df : DataFrame of the log
- directory : directory where all the h5 files are stored
- use_mls : if we shall divide the gapratio by the `mean level spacing`
'''
def set_entropies_df_log(df : pd.DataFrame, directory : str, fractions = [200, 0.1], set_max = False, set_min = False, idx_e = -1, verbose=True):
    col = [f'S_f={i:.3f}' for i in fractions] + (['S_max'] if set_max else []) + ['S_min'] if set_min else []
    df[col] = np.zeros((len(df), len(fractions)+ (1 if set_max else 0) + (1 if set_min else 0)))

    # takes the Hilbert space sizes
    hilbert_sizes = df.loc[:,'Nh'].to_list()
    # takes the names of the models
    model_shorts = df.loc[:,'model_short'].to_list()
    
    entropies = []
   
    for i, short in enumerate(tqdm.tqdm(model_shorts)):
        if verbose: print("\t\t"+short)
        # read the energy to calculate the gap ratio
        filename = short + '.h5'
        if not 'spectrum' in short:
            filename = short + ',spectrum_num=' + str(int(hilbert_sizes[i])) + '.h5'
            
        # read all entropies
        ent, av_idx, Nh, _, dictionary = read_entropies(directory, filename, 1.0, verbose=verbose)
        # print(ent)
        if len(ent) == 0:
            entropies.append(np.array([0 for i in fractions] + [0, 0]))
            continue
        
        # iterate fractions
        means = []
        for frac in fractions:
            entropy = get_entropies(ent, av_idx, frac)
            if not entropy.empty:
                means.append(mean_entropy(entropy, idx_e))
                if set_max:
                    means.append(np.max(entropy.iloc[idx_e]))
                if set_min:
                    means.append(np.min(entropy.iloc[idx_e]))
            else:
                means = means + [0, 0, 0]
        #df.loc[short, [f'S_f={i:.3f}' for i in fractions]] = np.array(means)   
        entropies.append(np.array(means))
    df.loc[:,col] = np.array(entropies)     

'''
Takes the log DataFrame and calculates the information entropy fractions for each of the models in it
'''
def set_info_entro_df_log(df : pd.DataFrame, directory : str, verbose=True):
    df.loc[:,'S_info'] = np.zeros(len(df))
    # takes the Hilbert space sizes
    hilbert_sizes = df.loc[:,'Nh'].to_list()
    # takes the names of the models
    model_shorts = df.loc[:,'model_short'].to_list()
    # takes the sectors
    sectors = df.loc[:,'sec'].to_list()
    #print(len(model_shorts), len(hilbert_sizes))
    entropies = []

    for i, short in enumerate(tqdm.tqdm(model_shorts)):
        # read the states
        filename = short + '.h5'
        if not 'spectrum' in short:
            filename = short + ',spectrum_num=' + str(int(hilbert_sizes[i])) + '.h5'
            
        # read the energy from a given filename (h5 Database)
        states = read_h5_file(directory, filename, 'states')
        if sectors[i] == 'imag':
            states = states.view('complex')
                
        if len(states) == 0: 
            entropies.append(0)
            continue
        
        # calculate the gapratio
        sinfo = info_entropy(states, short) / np.log(0.48 * float(hilbert_sizes[i]))
        #df.loc[short, 'gapratios'] = gap_rat
        entropies.append(sinfo)
    df['S_info'] = entropies

'''
Takes the log dataframe and concatenates the eigenstates into one flat array
'''
def read_eigenstates_df_log(df : pd.DataFrame, directory : str, sec : str, rescale_type = 'var', verbose=True):
    # takes the Hilbert space sizes
    hilbert_sizes = df.loc[:,'Nh'].to_list()
    # takes the names of the models
    model_shorts = df.loc[:,'model_short'].to_list()
    # save the states (to be flattened later)
    states_saved = np.array([])
    # Jiaozi lambda function
    bound = 0.02
    Jiaozi = lambda dE: 1.0/bound if np.abs(dE) <= bound / 2.0 else 0.0
    for i, short in enumerate(tqdm.tqdm(model_shorts)):
        # read the states
        filename = short + '.h5'
        if not 'spectrum' in short:
            filename = short + ',spectrum_num=' + str(int(hilbert_sizes[i])) + '.h5'
            
        # read the energy from a given filename (h5 Database)
        states = read_h5_file(directory, filename, 'states')
        if (sec == 'imag' or type(states[0]) == np.void):
            states = states.view('complex')
            
        # rescale if Jiaozi
        if rescale_type == 'Jiaozi':
            energies = np.array(read_h5_file(directory, filename, 'energy')).flatten()
            av_en = np.mean(energies)
            idx_mean = find_nearest_idx_np(energies, av_en)
            energies = energies[idx_mean - 50 : idx_mean + 50]
            # take 50 states less because we don't have left and right bounds
            states = states.T
            start = 25
            state_num = 50
            states_prime = np.zeros_like(states[start:start + state_num])
            for alpha, _ in enumerate(states_prime):
                # get current energy
                E_alpha = energies[start + alpha]
                av = np.zeros_like(states_prime[0])
                norm = 0.
                # iterate other states
                for alpha_prime, E in enumerate(energies):
                    state_prime = np.array(states[alpha_prime]).copy()
                    # get energy difference
                    dE = E - E_alpha
                    tmp = Jiaozi(dE)
                    # print(dE, "yes" if np.abs(dE) <= bound / 2.0 else "no!")
                    if tmp != 0.0:
                        av += tmp * np.square(np.abs(state_prime))
                        norm += tmp
                # print(norm*bound)
                av = np.sqrt(av / norm)
                states_prime[alpha] = np.divide(states[start + alpha], av) 
            states = states_prime
            
        # continue on that states
        states = states.flatten()
        if len(states) == 0:
            continue 
            # rescale by variance  

        if rescale_type == 'var':
            states = (states * np.complex64(np.sqrt(2.0 * hilbert_sizes[i]))) if (sec == 'imag' or type(states[0]) == np.void) else (states * np.sqrt(hilbert_sizes[i]))
            # var = np.var(states)
            # states = states * np.complex128(var) if (sec == 'imag' or type(states[0]) == np.void) else (states * var)

        # append
        states_saved = np.concatenate([states_saved, states])
        
        # count missing momenta twice
        if sec == 'imag':
            states_saved = np.concatenate([states_saved, states])

    return states_saved

'''
Takes the log dataframe and concatenates the gap ratios into one flat array
'''
def read_gap_ratios_df_log(df : pd.DataFrame, directory : str, verbose=True):
    # takes the Hilbert space sizes
    hilbert_sizes = df.loc[:,'Nh'].to_list()
    # takes the names of the models
    model_shorts = df.loc[:,'model_short'].to_list()
    # save the states (to be flattened later)
    ratios_saved = np.array([])
    
    for i, short in enumerate(tqdm.tqdm(model_shorts)):
        # read the states
        filename = short + '.h5'
        if not 'spectrum' in short:
            filename = short + ',spectrum_num=' + str(int(hilbert_sizes[i])) + '.h5'
            
        # read the energy from a given filename (h5 Database)
        en = read_h5_file(directory, filename, 'energy')
                
        if len(en) == 0: 
            continue
        # calculate the gapratio
        gap_rat = gap_ratio(np.array(en), 0.5, use_mean_lvl_spacing=False, return_mean=False)
        
        ratios_saved = np.concatenate([ratios_saved, gap_rat])

    return ratios_saved.flatten()
       
'''
Group the dataframe according to two parameters of the model
'''
def log_group_two_params(df : pd.DataFrame, col_a : str, col_b : str, columns, rescale = True):
    # copy df not to destroy it
    tmp = df.copy()
    
    # save ho;bert spaces
    tmp['Nh2'] = tmp['Nh']
    tmp[tmp['sec'] == 'imag']['Nh2'] *= 2
    
    # perform averaging over sectors
    if rescale:
        for col in columns:
            tmp[col] = tmp[col] * tmp['Nh2']
            tmp[tmp['sec'] == 'imag'][['Nh2',col]] *= 2.

        # groupby to average
        tmp = tmp.groupby([col_b,col_a])[columns + ['Nh', 'Nh2']].sum().reset_index(col_b)
        for col in columns:
            tmp[col] = tmp[col] / tmp['Nh2']
    else:
        return tmp.groupby([col_b,col_a])[columns + ['Nh']].mean().reset_index(col_b)    
    return tmp
   
####################################################### READ THE DATABASE SAVED FOR THE MODEL #######################################################

'''
Use 'energy' for energy reading and 'entropy' for entropy reading
- directory : the directory to be read from
- file : file to be read from
- data_col : either `entropy` or 'energy' or 'states'
'''
def read_h5_file(directory, file, data_col : str):  
    if not os.path.exists(directory + file):
        print(directory + file, "doesn't exists")
        return pd.DataFrame()
    if os.stat(directory + file).st_size == 0:
        print(f"Removing : {directory + file}")
        os.remove(directory + file)  
        return pd.DataFrame()
    try:
        with h5py.File(directory + file, "r") as f: 
            
            a_group_key = list(f.keys())
            if data_col not in a_group_key:
                raise Exception("Not in columns")
            
            
            array = f[data_col][()]
            
            array=np.array(array).T
            if data_col == 'energy':
                return pd.DataFrame(array, columns = [data_col])
            elif data_col == 'entropy':
                # read the number of lattice sites 
                size_x = array.shape[0]
                size_y = len(array.flatten())//size_x
                return pd.DataFrame(array, index=np.arange(1, size_x+1), columns=np.arange(1, size_y+1))
            else:
                return np.array(array)
            
    except Exception as e:
        print(e)
        print(f"couldn't open {directory + file}")
        return pd.DataFrame()
    return pd.DataFrame()
        
####################################################### READ ENTROPIES #######################################################

'''
Reads the entropies from a file with a given fraction of elements in the middle. Depends on index
'''
def read_entropies(directory : str, file : str, fraction : float, verbose = False):
    df = read_h5_file(directory, file, 'entropy')
    en = read_h5_file(directory, file, 'energy').to_numpy().flatten()
    # print(df)
    N = len(en)
    # if there are no entropies -> return empty
    if len(df) == 0:
        return pd.DataFrame(), 1, 1, N, {}

    columns = df.columns
    cols_to_take = columns
    
    # -------------- IDX MAX ----------------
    # find moving average
    df_roll = moving_average(df, roll_number)
    idx_roll = find_maximum(df_roll).iloc[-1]

    # -------------- IDX MEAN ---------------
    idx_mean = find_nearest_idx_np(en, np.mean(en))
    
    # -------------- IDX DOS ---------------
    dos, bins = np.histogram(en, bins=100)
    parameters, covariance = curve_fit(gauss, np.array(bins[1:]), np.array(dos, dtype=np.float32))
    idx_dos = find_nearest_idx_np(en, parameters[2])
    
    # -------------- IDX OUTLIER ---------------
    idx_outlier = find_maximum(df).iloc[-1] - 1
    val_outlier = en[idx_outlier]
    
    ################ PRINT INFO ################
    if verbose:
        # indices
        print(f"\t\t\tidx_roll={idx_roll},\tidx_mean={idx_mean},\tidx_dos={idx_dos},\tidx_outlier={idx_outlier}\t\t->we take {IDX_VAL}")
        # energies
        print("\t\tEnergies:")
        for index in [idx_roll, idx_mean, idx_dos, idx_outlier]:
            print(f"\t\t\tE[{index}]={float(en[index])}")
        print("\t\tEntropies:")
        for index in [idx_roll, idx_mean, idx_dos, idx_outlier]:
            print(f"\t\t\tS[{index}]={float(df[index].iloc[-1])}")
            
    av_idx = idx_roll if (IDX_VAL == 'roll') else \
        (idx_mean if (IDX_VAL == 'mean') else \
            (idx_outlier if (IDX_VAL == 'max') else idx_dos))
        
    # create the dictionary of indices
    idx_dic = {
        'idx':(av_idx, en[av_idx]),
        'roll':(idx_roll, en[idx_roll]),
        'mean':(idx_mean, en[idx_mean]),
        'dos':(idx_dos, en[idx_dos]),
        'max':(idx_outlier, en[idx_outlier])
        }
    
    if fraction == 1.0:
        # if it is already the file that we wanted
        if verbose:
            print(f"\t\t\ttaking the whole dataframe")
        return df, av_idx, N, N, idx_dic
    else:
        bad, av_idx, lower, upper = get_values_num(fraction, cols_to_take, df.shape[0], av_idx)
        # ----------------------- use the maximal value to get that ------------------------
        
        if bad: 
            print(f'\t\t->get_values_num() returned {bad, av_idx, lower, upper}')
            return pd.DataFrame(), av_idx, len(cols_to_take), N, idx_dic       
        
        cols_to_take = cols_to_take[lower : upper]
        return df.loc[:,cols_to_take], av_idx, len(cols_to_take), N, idx_dic

'''
From full entropies DataFrame take the one corresponding to our fraction
'''
def get_entropies(df : pd.DataFrame, av_idx : int, fraction : float):
    # if there are no entropies -> return empty
    if len(df) == 0:
        return pd.DataFrame()

    columns = df.columns
    cols_to_take = columns
    
    if fraction == 1.0:
        # if it is already the file that we wanted
        print(f"\t\t\ttaking the whole dataframe")
        return df
    else:
        bad, av_idx, lower, upper = get_values_num(fraction, cols_to_take, df.shape[0], av_idx)
        # ----------------------- use the maximal value to get that ------------------------
        
        if bad: 
            print(f'\t\t->get_values_num() returned {bad, av_idx, lower, upper}')
            return pd.DataFrame()    
        
        cols_to_take = cols_to_take[lower : upper]
        return df.loc[:,cols_to_take]
    
####################################################### READ BINARY ENTROPIES 

def read_binary(directory, L) -> pd.DataFrame:
    if not os.path.exists(directory):
        print("directory does not exist")
        return pd.DataFrame()
    
    size_x = L // 2
    files = os.listdir(directory)
    df = None
    for file in files:
        if file.startswith('entropies') and file.endswith('bin'):
            with open(directory + file, mode='rb') as file: # b is important -> binary
                idx = 0
                values = []
                index = np.arange(1, L // 2 + 1)
                while file:
                    binary = file.read(FLOAT_SIZE)
                    if not binary:
                        break    
                    val = struct.unpack("d", binary)[0]
                    values.append(val)
                    idx += 1
                spectrum_num = len(values)//size_x
                columns = [i for i in range(1, spectrum_num+1)]
                
                values = np.array(values)
                values = values.reshape(spectrum_num, size_x)
                df = pd.DataFrame(index, columns = [0])
                values = np.matrix.transpose(np.array(values))
                df = pd.concat([df, pd.DataFrame(values, columns=columns)], axis=1)
                df.rename({df.columns[0]:'Ls'}, inplace=True, axis=1)
                df.set_index('Ls',inplace=True)
    return df

####################################################### READ HDF5 ENTROPIES 

def read_entro_h5(directory, L) -> pd.DataFrame:
    if not os.path.exists(directory):
        print("directory does not exist")
        return pd.DataFrame()
    
    size_x = L // 2
    files = os.listdir(directory)
    df = pd.DataFrame()
    for file in files:
        if file.startswith('entropies') and file.endswith('h5'):
            with h5py.File(directory + file, "r") as f:
                # Print all root level object names (aka keys) 
                # these can be group or dataset names 
                #print("Keys: %s" % f.keys())
                # get first object name/key; may or may NOT be a group
                #a_group_key = list(f.keys())[0]
                
                # get the object type for a_group_key: usually group or dataset
                #print(type(f[a_group_key])) 

                # If a_group_key is a group name, 
                # this gets the object names in the group and returns as a list
                #data = list(f[a_group_key])

                # If a_group_key is a dataset name, 
                # this gets the dataset values and returns as a list
                #data = list(f[a_group_key])
                # preferred methods to get dataset values:
                #ds_obj = f[a_group_key]      # returns as a h5py dataset object
                #ds_arr = f[a_group_key][()]  # returns as a numpy array
                #print(f['entropy'][()])
                array = np.array(f['entropy'][()]).T
                size_y = len(array.flatten())//size_x

                df = pd.DataFrame(array, index=np.arange(1, size_x+1), columns=np.arange(1, size_y+1))
    
    if len(df) == 0:
        df = read_binary(directory, L)
        
    return df

####################################################### ENERGIES #######################################################

####################################################### READ ENERGIES 

def read_energy(directory) -> pd.DataFrame:
    df = pd.DataFrame()

    if not os.path.exists(directory):
        print(directory, "doesn't exists")
        return pd.DataFrame()
    
    tmp = "s"
    files = os.listdir(directory)
    for file in files:
        if file.startswith('energies') and not file.endswith('bin'):
            df = pd.read_csv(directory + file, names = ["energy"])
            save = False
            if str(df['energy'].iloc[0]).startswith('this'):
                values = [float(str(i).split('=')[-1]) for i in list(df['energy'])]
                df['energy'] = values
                save = True
            #print(df)
            if len(df) != 0:
                if save:
                    df.to_csv(directory + file, sep = "\t", header=False)
                break            

    if len(df) == 0:
        return pd.DataFrame()
    
    return df

def read_energy_ising(directory) -> pd.DataFrame:
    df = pd.DataFrame()

    if not os.path.exists(directory):
        print(directory, "doesn't exists")
        return pd.DataFrame()
    
    tmp = "s"
    files = os.listdir(directory)
    for file in files:
        if file.startswith('energies') and not file.endswith('bin'):
            df = pd.read_csv(directory + file, names = ["energy"], sep='\t')
            save = False
            if str(df['energy'].iloc[0]).startswith('this'):
                values = [float(str(i).split('=')[-1]) for i in list(df['energy'])]
                df['energy'] = values
                save = True
            #print(df)
            if len(df) != 0:
                if save:
                    df.to_csv(directory + file, sep = "\t", header=False)
                break            

    if len(df) == 0:
        return pd.DataFrame()
    
    return df

####################################################### READ ENERGIES BINARY #######################################################

def read_energy_bin(directory) -> pd.DataFrame:
    df = pd.DataFrame()

    if not os.path.exists(directory):
        print(directory, "doesn't exists")
        return pd.DataFrame()
    
    tmp = "s"
    files = os.listdir(directory)
    for file in files:
        if file.startswith('energies') and file.endswith('bin'):
            with open(directory + file, mode='rb') as file: # b is important -> binary
                idx = 0
                values = []
                while file:
                    binary = file.read(FLOAT_SIZE)
                    if not binary:
                        break    
                    val = struct.unpack("d", binary)[0]
                    values.append(val)
                    idx += 1
                spectrum_num = len(values)
                
                values = np.array(values)
                df = pd.DataFrame(values, index = np.arange(spectrum_num), columns = ['energy'])

    if len(df) == 0:
        if model_name=='xyz':
            return read_energy(directory)
        else:
            return read_energy_ising(directory)
    
    return df

####################################################### READ ENERGIES HDF5 #######################################################

def read_energy_h5(directory, file) ->pd.DataFrame:
    if not os.path.exists(directory):
        print("directory does not exist")
        return pd.DataFrame()
    
    #files = os.listdir(directory)
    df = pd.DataFrame()
    try:
        with h5py.File(directory + file, "r") as f:
            # Print all root level object names (aka keys) 
            # these can be group or dataset names 
            #print("Keys: %s" % f.keys())
            # get first object name/key; may or may NOT be a group
            #a_group_key = list(f.keys())[0]
            
            # get the object type for a_group_key: usually group or dataset
            #print(type(f[a_group_key])) 

            # If a_group_key is a group name, 
            # this gets the object names in the group and returns as a list
            #data = list(f[a_group_key])

            # If a_group_key is a dataset name, 
            # this gets the dataset values and returns as a list
            #data = list(f[a_group_key])
            # preferred methods to get dataset values:
            #ds_obj = f[a_group_key]      # returns as a h5py dataset object
            #ds_arr = f[a_group_key][()]  # returns as a numpy array
            #print(f['entropy'][()])
            array = np.array(f['energy'][()]).T
            df = pd.DataFrame(array, columns = ['energy'])
    except:
        print(f"couldn't open {directory + file}")
        return pd.DataFrame()
    if len(df) == 0:
        df = read_energy_bin(directory)
    return df


####################################################### EIGENSTATES #######################################################

'''
From a given .h5 file extracts the eigenstates
'''
def get_eigenstates(directory : str, model_short : str, which = 'states_r'):
    eigs = read_h5_file(directory, model_short + '.h5', which)
    
    return eigs