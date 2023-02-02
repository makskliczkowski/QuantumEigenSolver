

import pandas as pd
import subprocess

df = pd.read_csv('params_sweep.csv', index_col=0)
df['p'] = (df['p'] + 1)/2
df['x'] = (df['x'] + 1)/2
with open("commands.txt",'w') as f:
    for i, row in df.iterrows():
        string = " ".join([ str(row['Ns']),
                            str(row['hz']),
                            str(row['hx']),
                            str(row['Jb']),
                            str(row['da']),
                            str(row['bc']),
                                str(row['k']),
                            str(row['p']),
                            str(row['x'])])
        f.write(string + '\n')
    
    #print("skrypt_run_symmetries_single_param.sh")
    #subprocess.check_call(['skrypt_run_symmetries_single_param.sh', str(row['Ns']),
    #                       str(row['hz']),
    #                       str(row['hx']),
    #                       str(row['Jb']),
    #                       str(row['da']),
    #                       str(row['bc']),
    #                       str(row['k']),
    #                       str(row['p']),
    #                       str(row['x'])])
#