from .__entro_plot__ import *
import h5py
model_name = 'xyz'
roll_number = 91
sym = True

####################################################### GET LOG FILE #########################################################

def parse_model(model_string : str):
    #print(model_string)
    model_split = model_string.split(',')
    #print(model_split)
    model_name = model_split[0]
    columns = [str(i.split('=')[0]) for i in model_split[1:]]
    values = [float(i.split('=')[-1]) for i in model_split[1:]]
    
    return columns, values

def get_log_file(directory, head = None):
    df = pd.read_csv(directory + 'entropies_log.dat', sep = '\t',
                       header = None, names = ['model', 'max_ent', '200_ent', 'tmp']).drop(columns = ['tmp'], axis = 1).dropna()
    if head is not None:
        df = df.head(head)
    
    models = df['model'].to_list()
    value = []
    columns = None
    for i in models:
        columns, values = parse_model(i)
        value.append(values)
    df[columns] = value
        #df.iloc[i][columns] = values
    df['model_short'] = df['model'].apply(lambda x : str(x).replace('0000', '00').replace('000', '0'))
    return df
    

####################################################### READ ENTROPIES #######################################################

def read_entropies(directory, L, fraction):
    df = read_entro_h5(directory, L)
    en = read_energy_h5(directory).to_numpy().flatten()
    N = len(en)
    if len(df) == 0:
        return pd.DataFrame(), 1, 1, N, {}

    columns = df.columns
    cols_to_take = columns
    
    # find moving average
    df_roll = moving_average(df, roll_number)
    # -------------- IDX MAX ----------------
    idx_max = find_maximum(df_roll).iloc[-1]
    #print( , df_roll[idx_max], np.max(df_roll))
    # -------------- IDX MEAN ---------------
    idx_mean = find_nearest_idx_np(en, np.mean(en))
    # -------------- IDX DOS ---------------
    dos, bins = np.histogram(np.array(en).flatten(), bins=100)
    parameters, covariance = curve_fit(gauss, np.array(bins[1:]), np.array(dos, dtype=np.float32))
    idx_dos = find_nearest_idx_np(en, parameters[2])
    
    idx_brute_max = find_maximum(df).iloc[-1] - 1
    #print('L=',L, df.max(axis=1))
    val_brute_max = en[idx_brute_max]
    
    print(f"\t\t\tidx_roll={idx_max},idx_mean={idx_mean},idx_dos={idx_dos},idx_max={idx_brute_max},we take {INDEX_VAL}")
    print(f"\t\t\tE[idx_roll]={float(en[idx_max])},E[idx_mean]={float(en[idx_mean])},E[idx_dos]={float(en[idx_dos])},E[idx_max]={float(val_brute_max)}")
    av_idx = idx_max if INDEX_VAL == 'roll' else (idx_mean if INDEX_VAL == 'mean' else (idx_brute_max if INDEX_VAL == 'max' else idx_dos))
    print(f"\t\t\tEntropy[idx]={df[av_idx].iloc[-1]},Entropy_max={float(np.max(df.iloc[-1]))},entropy_roll_max={np.max(df_roll.iloc[-1])},entropy_roll_max[idx]={df_roll[av_idx].iloc[-1]}")
    idx_dic = {'roll':(idx_max, en[idx_max]), 'mean':(idx_mean, en[idx_mean]), 'dos':(idx_dos, en[idx_dos]), 'idx':(av_idx, en[av_idx]), 'max':(idx_brute_max, en[idx_brute_max])}
    if fraction == 1.0:
        # if it is already the file that we wanted
        print(f"\t\t\ttaking the whole dataframe")

        return df, av_idx, N, N, idx_dic
    else:
        bad, av_idx, lower, upper = get_values_num(fraction, cols_to_take, L, av_idx)
        # ----------------------- use the maximal value to get that ------------------------
        
        if bad: 
            print(f'\t\t->get_values_num() returned {bad, av_idx, lower, upper}')
            return pd.DataFrame(), av_idx, len(cols_to_take), N, idx_dic       
        
        cols_to_take = cols_to_take[lower : upper]
        return df.loc[:,cols_to_take], av_idx, len(cols_to_take), N, idx_dic

####################################################### READ BINARY ENTROPIES #######################################################

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

####################################################### READ HDF5 ENTROPIES #######################################################
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

####################################################### READ ENERGIES #######################################################


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


def read_energy_h5(directory) ->pd.DataFrame:
    if not os.path.exists(directory):
        print("directory does not exist")
        return pd.DataFrame()
    
    files = os.listdir(directory)
    df = pd.DataFrame()
    for file in files:
        if file.startswith('energies') and file.endswith('h5'):
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


