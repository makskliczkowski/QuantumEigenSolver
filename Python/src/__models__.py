from .__readers__ import *
import numpy as np


##################################################################### ISING #####################################################################

class ising_transverse:
    def __init__(self, Ns, J=1.0, J0=0.0, g=0.9, g0=0.0, h=0.81, w=0.0, sym = False, k = 0, p = 1, x = 1, bc = 0):
        # if constant fraction means that we need to take care of how many do we take from given sector
        self.quiet = True
        # to set the indices
        self.index = 1.0
        
        self.mean_energy = 0.0
        self.mean_energy_sec = 0.0
        self.e_idx_mean = 0.0
        self.average_idx_indices_col = {}
        self.N = 1
        self.Ns = 1
        self.N_full = 1
        
        self.all_symmetries = []
        self.real_symmetries = []
        self.imag_symmetries = []
        self.all_symmetries_str = []
        self.real_symmetries_str = []
        self.imag_symmetries_str = []
        self.bc = bc
        self.set_Ns(Ns)
        
        self.J = J
        self.J0 = J0
        self.g = g
        self.g0 = g0
        self.h = h
        self.w = w

        self.k = k
        self.p = p
        self.x = x
        self.sym = sym
    
    '''
    Setting the number of lattice sites
    '''
    def set_Ns(self, Ns):
        self.Ns = Ns
        self.N = 1
        self.N_full = np.power(2, self.Ns)
        if self.bc == 0:
            self.all_symmetries = [(i, 1) for i in range(0, self.Ns)] + [(0, -1)] + ([(Ns//2, -1)] if self.Ns % 2 == 0 else [])
            self.real_symmetries = [(0, 1), (0, -1)] + ([(Ns//2, -1), (Ns//2, 1)] if self.Ns % 2 == 0 else [])
            self.all_symmetries_str = [str(i) for i in self.all_symmetries]
            self.real_symmetries_str = [str(i) for i in self.real_symmetries]
        else:
            self.all_symmetries = [(0, 1), (0, -1)]
            self.real_symmetries = [(0, 1), (0, -1)]
            self.all_symmetries_str = [str(i) for i in self.all_symmetries]
            self.real_symmetries_str = [str(i) for i in self.real_symmetries]
        self.imag_symmetries = list(set(self.all_symmetries).difference(self.real_symmetries))
        self.imag_symmetries_str = [str(i) for i in self.imag_symmetries] 
    
# --------------------------------------------- GETTERS ---------------------------------
    '''
    Gets the information about the model
    '''        
    def get_info(self):
        if not self.sym:
            return f"Ns={self.Ns},J={self.J:.2f},J0={self.J0:.2f},g={self.g:.2f},g0={self.g0:.2f},h={self.h:.2f},w={self.w:.2f},bc={self.bc}"
        else:
            return f"Ns={self.Ns},J={self.J:.2f},g={self.g:.2f},h={self.h:.2f},k={self.k:.2f},p={self.p},x={self.x},bc={self.bc}"
    
    '''
    Gets the information about the model without symmetries
    '''            
    def get_info_wo_sym(self):
        if not self.sym:
            return self.get_info()
        else:
            return f"Ns={self.Ns},J={self.J:.2f},g={self.g:.2f},h={self.h:.2f},bc={self.bc}"
    
    '''
    Gets the directory with the model
    '''                
    def get_dir(self, dir = "resultsSym"):
        directory = dir + str(self.Ns) + kPSep
        if self.sym:
            return directory + f"_ising_sym," + self.get_info() + kPSep
        else:
            return directory + f"_ising," + self.get_info() + kPSep    
        
    '''
    Gets the symmetries
    '''
    def get_symmetries(self, sectors = 'all'):
        if sectors == 'all':
            return self.all_symmetries
        elif sectors == 'real':
            return self.real_symmetries
        elif sectors == 'imaginary':
            return self.imag_symmetries
        else:
            return [(self.k, self.p)]
    # ------------------------------------------------------------------------------------------- CONCATENATES -------------------------------------------------------------------------------------------
    
    '''
    Concatenates all entropies from different sectors that we want together
    '''
    def concat_entropies(self, fraction, dir = "{kPSep}resultsSym", sector = 'all'):
        df = pd.DataFrame(columns = ['sector'] + [i for i in range(1, self.Ns//2 + 1)]).fillna(0)        
        syms = self.get_symmetries(sector)
        N = 0
        k = self.k
        p = self.p
        for i in syms:
            self.k = i[0]
            self.p = i[1]
            df_tmp, idx, val_num, N, idx_val = read_entropies(self.get_dir(dir), self.Ns, fraction)
            # append to check
            df_tmp2 = pd.DataFrame()
            for index, row in df_tmp.iterrows():
                df_tmp2[index] = np.array(row)
                
            df_tmp2['sector'] = [(self.k,self.p) for i in range(len(df_tmp2))]
            # append to the dataframe
            
            df = df.append(df_tmp2, ignore_index = True)
            
        self.k = k
        self.p = p
        
        print(self.get_info() + "\n\t->" + f"N={len(df)}")
        return df    
    
    # ------------------------------- getting different symmetry sectors entropies together --------------------------------
    
    '''
    Gets all different symmetry sectors mean from all entropies file
    '''
    def symmetry_sectors_all(self, fraction, dir = "{kPSep}resultsSym", var = False):
    
        df = pd.DataFrame(index=[i for i in range(1, self.Ns//2+1)])
        
        k = self.k
        p = self.p
        col_name = 'av_S' if not var else "var"
        N_all = 0
        for i in self.all_symmetries:
            self.k = i[0]
            self.p = i[1]

            print(f"\t-->For {self.Ns},sym={i},frac={fraction},bc={self.bc} reading the whole entropies file binary")
            df_tmp, idx, val_num, N, idx_dic = read_entropies(self.get_dir(dir), self.Ns, fraction)
            
            if not df_tmp.empty:
                print(f'\t--->Ns={self.Ns},k={self.k},p={self.p},{idx_dic}')
                if var:
                    df_tmp[col_name] = df_tmp.var(axis=1)
                else:
                    df_tmp[col_name] = df_tmp.mean(axis=1)           
                df[str(i) + f':{N}:{val_num}'] = np.array(df_tmp.loc[:,col_name])
            else:
                print(f"\t\t\t->For {self.Ns} missing {i},frac={fraction},bc={self.bc}")
            N_all += N
        
        self.k = k
        self.p = p    
        print(f"\t->{('Did' if (self.N_full==N_all) else 'Didnt')} capture whole Hilbert space - for Ns = {self.Ns}, bc = {self.bc}, frac = {fraction} -- {N_all}/{self.N_full}")
        return df    
    
    '''
    Averages over specific symmetry sectors in one dataframe
    '''
    def average_symmetry_sectors_df(self, df : pd.DataFrame):
        av_real = 'av_Sr'
        N_r = 0
        N_r_total = 0
        av_other = 'av_So'
        N_o = 0
        N_o_total = 0
        av_together = 'av_S'
        N_t = 0
        N_total = 0
        
        df_tmp = pd.DataFrame(index=[i for i in range(1, self.Ns//2+1)], columns=[av_real, av_other, av_together]).fillna(0)
        for col in df.columns:
            N_tmp = int(col.split(':')[-2])
            symmetry = col.split(':')[0]
            N_tmp_total = int(col.split(':')[-1])
            if symmetry in self.real_symmetries_str:
                #print('real:')
                #print('\t', sym)
                N_r_total += N_tmp_total
                N_r += N_tmp
                
                #df_tmp.loc[:,av_real] += N_tmp_total * np.array(df.loc[:,col])
                df_tmp.loc[:,av_real] += N_tmp * np.array(df.loc[:,col])
                #df_tmp.loc[:,av_real] += np.log2(N_tmp) * np.array(df.loc[:,col])
            else:
                #print('imag:')
                #print('\t', sym)
                N_o_total += N_tmp_total
                N_o += N_tmp
                
                #df_tmp.loc[:,av_other] += N_tmp_total * np.array(df.loc[:,col])
                df_tmp.loc[:,av_other] += N_tmp * np.array(df.loc[:,col])
                #df_tmp.loc[:,av_other] += np.log2(N_tmp) * np.array(df.loc[:,col])
            
            N_total += N_tmp_total
            N_t += N_tmp
            
            #df_tmp.loc[:,av_together] += N_tmp_total * np.array(df.loc[:,col])
            df_tmp.loc[:,av_together] += N_tmp * np.array(df.loc[:,col])
            #df_tmp.loc[:,av_together] += np.log2(N_tmp) * np.array(df.loc[:,col])
            #print(df_tmp)
        print(f"\t\t->Total number of states = {N_total} out of {N_t} frac = {N_total/N_t :.4f}\n")
        
        #df_tmp[av_real] /= N_r_total
        #df_tmp[av_other] /= N_o_total
        #df_tmp[av_together] /= N_total
        
        df_tmp[av_real] /= N_r
        df_tmp[av_other] /= N_o
        df_tmp[av_together] /= N_t
        
        #df_tmp[av_real] = df_tmp[av_real] / np.log2(N_r)
        #df_tmp[av_other] = df_tmp[av_other] / np.log2(N_o)
        #df_tmp[av_together] = df_tmp[av_together] / np.log2(N_t)
        
        return df_tmp
    
##################################################################### XYZ #####################################################################

class xyz:
    def __init__(self, Ns, Ja = 1.0, Jb = 1.0, hx = 0.2, hz = 0.8, da = 0.9, db = 0.9, ea=0.5, eb = 0.5, sym = False, k = 0, p = 1, x = 1, bc = 0):
        # if constant fraction means that we need to take care of how many do we take from given sector
        self.quiet = True
        # to set the indices
        self.index = 1.0
        
        self.mean_energy = 0.0
        self.mean_energy_sec = 0.0
        self.e_idx_mean=0.0
        self.average_idx_indices_col = {}
        self.N = 1
        self.Ns = 1
        self.N_full = 1
        self.all_symmetries = []
        self.real_symmetries = []
        self.imag_symmetries = []
        self.all_symmetries_str = []
        self.real_symmetries_str = []
        self.imag_symmetries_str = []
        self.bc = bc
        self.set_Ns(Ns)
        
        self.Ja = Ja
        self.Jb = Jb
        self.hx = hx
        self.hz = hz
        self.da = da
        self.db = db
        self.ea = ea
        self.eb = eb

        self.k = k
        self.p = p
        self.x = x
        self.sym = sym
        self.pb = 1
    
    '''
    Setting the number of lattice sites
    '''        
    def set_Ns(self, Ns):
        self.Ns = Ns
        self.N = 1
        self.N_full = np.power(2, self.Ns)
        if self.bc == 0:
            self.all_symmetries = [(i, 1) for i in range(0, self.Ns)] + [(0, -1)] + ([(Ns//2, -1)] if self.Ns % 2 == 0 else [])
            self.real_symmetries = [(0, 1), (0, -1)] + ([(Ns//2, -1), (Ns//2, 1)] if self.Ns % 2 == 0 else [])
            self.all_symmetries_str = [str(i) for i in self.all_symmetries]
            self.real_symmetries_str = [str(i) for i in self.real_symmetries]
        else:
            self.all_symmetries = [(0, 1), (0, -1)]
            self.real_symmetries = [(0, 1), (0, -1)]
            self.all_symmetries_str = [str(i) for i in self.all_symmetries]
            self.real_symmetries_str = [str(i) for i in self.real_symmetries]
        self.imag_symmetries = list(set(self.all_symmetries).difference(self.real_symmetries))
        self.imag_symmetries_str = [str(i) for i in self.imag_symmetries] 
    
# --------------------------------------------- GETTERS ---------------------------------
    '''
    Gets the information about the model
    '''        
    def get_info(self):
        if not self.sym:
            return f"Ns={self.Ns},Ja={self.Ja:.2f},Jb={self.Jb:.2f},hx={self.hx:.2f},hz={self.hz:.2f},da={self.da:.2f},db={self.db:.2f},ea={self.ea:.2f},eb={self.eb:.2f},pb={self.pb},bc={self.bc}"
        else:
            return f"Ns={self.Ns},Ja={self.Ja:.2f},Jb={self.Jb:.2f},hx={self.hx:.2f},hz={self.hz:.2f},da={self.da:.2f},db={self.db:.2f},ea={self.ea:.2f},eb={self.eb:.2f},k={self.k:.2f},p={self.p},x={self.x},bc={self.bc}"
        
    '''
    Gets the information about the model without symmetries
    '''            
    def get_info_wo_sym(self):
        if not self.sym:
            return self.get_info()
        else:
            return f"Ns={self.Ns},Ja={self.Ja:.2f},Jb={self.Jb:.2f},hx={self.hx:.2f},hz={self.hz:.2f},da={self.da:.2f},db={self.db:.2f},ea={self.ea:.2f},eb={self.eb:.2f},bc={self.bc}"
     
    '''
    Gets the directory with the model
    '''                
    def get_dir(self, dir = "resultsSym"):
        directory = dir + str(self.Ns) + kPSep
        if self.sym:
            return directory + f"_xyz_sym," + self.get_info() + kPSep
        else:
            return directory + f"_xyz," + self.get_info() + kPSep
        
    '''
    Gets the symmetries
    '''
    def get_symmetries(self, sectors = 'all'):
        if sectors == 'all':
            return self.all_symmetries
        elif sectors == 'real':
            return self.real_symmetries
        elif sectors == 'imaginary':
            return self.imag_symmetries
        else:
            return [(self.k, self.p)]
    # ------------------------------------------------------------------------------------------- CONCATENATES -------------------------------------------------------------------------------------------
    
    '''
    Concatenates all entropies from different sectors that we want together
    '''
    def concat_entropies(self, fraction, dir = "{kPSep}resultsSym", sector = 'all'):
        df = pd.DataFrame(columns = ['sector'] + [i for i in range(1, self.Ns//2 + 1)]).fillna(0)        
        syms = self.get_symmetries(sector)
        N = 0
        k = self.k
        p = self.p
        for i in syms:
            self.k = i[0]
            self.p = i[1]
            df_tmp, idx, val_num, N, idx_val = read_entropies(self.get_dir(dir), self.Ns, fraction)
            # append to check
            df_tmp2 = pd.DataFrame()
            for index, row in df_tmp.iterrows():
                df_tmp2[index] = np.array(row)
                
            df_tmp2['sector'] = [(self.k,self.p) for i in range(len(df_tmp2))]
            # append to the dataframe
            
            df = df.append(df_tmp2, ignore_index = True)
            
        self.k = k
        self.p = p
        
        print(self.get_info() + "\n\t->" + f"N={len(df)}")
        return df    
    
    # ------------------------------- getting different symmetry sectors entropies together --------------------------------
    
    '''
    Gets all different symmetry sectors mean from all entropies file
    '''
    def symmetry_sectors_all(self, fraction, dir = "{kPSep}resultsSym", var = False):
    
        df = pd.DataFrame(index=[i for i in range(1, self.Ns//2+1)])
        
        k = self.k
        p = self.p
        col_name = 'av_S' if not var else "var"
        N_all = 0
        for i in self.all_symmetries:
            self.k = i[0]
            self.p = i[1]

            print(f"\t-->For {self.Ns},sym={i},frac={fraction},bc={self.bc} reading the whole entropies file binary")
            df_tmp, idx, val_num, N, idx_dic = read_entropies(self.get_dir(dir), self.Ns, fraction)
            
            if not df_tmp.empty:
                print(f'\t--->Ns={self.Ns},k={self.k},p={self.p},{idx_dic}')
                if var:
                    df_tmp[col_name] = df_tmp.var(axis=1)
                else:
                    df_tmp[col_name] = df_tmp.mean(axis=1)           
                df[str(i) + f':{N}:{val_num}'] = np.array(df_tmp.loc[:,col_name])
            else:
                print(f"\t\t\t->For {self.Ns} missing {i},frac={fraction},bc={self.bc}")
            N_all += N
        
        self.k = k
        self.p = p    
        print(f"\t->{('Did' if (self.N_full==N_all) else 'Didnt')} capture whole Hilbert space - for Ns = {self.Ns}, bc = {self.bc}, frac = {fraction} -- {N_all}/{self.N_full}")
        return df    
    
    '''
    Averages over specific symmetry sectors in one dataframe
    '''
    def average_symmetry_sectors_df(self, df : pd.DataFrame):
        av_real = 'av_Sr'
        N_r = 0
        N_r_total = 0
        av_other = 'av_So'
        N_o = 0
        N_o_total = 0
        av_together = 'av_S'
        N_t = 0
        N_total = 0
        
        df_tmp = pd.DataFrame(index=[i for i in range(1, self.Ns//2+1)], columns=[av_real, av_other, av_together]).fillna(0)
        for col in df.columns:
            N_tmp = int(col.split(':')[-2])
            symmetry = col.split(':')[0]
            N_tmp_total = int(col.split(':')[-1])
            if symmetry in self.real_symmetries_str:
                #print('real:')
                #print('\t', sym)
                N_r_total += N_tmp_total
                N_r += N_tmp
                
                #df_tmp.loc[:,av_real] += N_tmp_total * np.array(df.loc[:,col])
                df_tmp.loc[:,av_real] += N_tmp * np.array(df.loc[:,col])
                #df_tmp.loc[:,av_real] += np.log2(N_tmp) * np.array(df.loc[:,col])
            else:
                #print('imag:')
                #print('\t', sym)
                N_o_total += N_tmp_total
                N_o += N_tmp
                
                #df_tmp.loc[:,av_other] += N_tmp_total * np.array(df.loc[:,col])
                df_tmp.loc[:,av_other] += N_tmp * np.array(df.loc[:,col])
                #df_tmp.loc[:,av_other] += np.log2(N_tmp) * np.array(df.loc[:,col])
            
            N_total += N_tmp_total
            N_t += N_tmp
            
            #df_tmp.loc[:,av_together] += N_tmp_total * np.array(df.loc[:,col])
            df_tmp.loc[:,av_together] += N_tmp * np.array(df.loc[:,col])
            #df_tmp.loc[:,av_together] += np.log2(N_tmp) * np.array(df.loc[:,col])
            #print(df_tmp)
        print(f"\t\t->Total number of states = {N_total} out of {N_t} frac = {N_total/N_t :.4f}\n")
        
        #df_tmp[av_real] /= N_r_total
        #df_tmp[av_other] /= N_o_total
        #df_tmp[av_together] /= N_total
        
        df_tmp[av_real] /= N_r
        df_tmp[av_other] /= N_o
        df_tmp[av_together] /= N_t
        
        #df_tmp[av_real] = df_tmp[av_real] / np.log2(N_r)
        #df_tmp[av_other] = df_tmp[av_other] / np.log2(N_o)
        #df_tmp[av_together] = df_tmp[av_together] / np.log2(N_t)
        
        return df_tmp

    
####################################################### CREATE A MODEL #######################################################

def create_model(model_name, L, k = 0, p = 1, x = 1, bc = 0, sym = True):
    if model_name=="xyz":
        return xyz(L, k=k, p=p, x=x, bc=bc, sym=sym)
    else:
        return ising_transverse(L, k=k, p=p, x=x, bc=bc, sym=sym)
    
####################################################### PARSE A MODEL #######################################################


    
    #if model_name.starts_with("__xyz"):
    #    return xyz(L, k=k, p=p, x=x, bc=bc, sym=sym)
    #else:
    #    return ising_transverse(L, k=k, p=p, x=x, bc=bc, sym=sym)