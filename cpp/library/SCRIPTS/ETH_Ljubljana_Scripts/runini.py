import subprocess
import sys
import os

# load the ini file to get what scripts to run
iniFile =   sys.argv[1]
if not iniFile.endswith(".ini"):
    raise Exception(f"{iniFile} is not '.ini' file")

resultError = lambda result: '0' not in result

###################################################################

# iterate me
iter    =   0
# save the output to know what scripts were run
outp    =   open(f"output_{iniFile.replace('.ini', '')}.dat", "a+")
# open the file to read
with open(iniFile, "r+") as f:
    # read the lines to run
    lines = f.readlines()
    for l in lines:
        # split the commands, get rid of endline
        tmp     = l.split("\n")[0]
        # split the commands, additional information to the script
        tmp     = tmp.split('"')
        # the script to run
        left    = tmp[0].split(" ")
        # additional information to the script
        mid     = tmp[1]
        # argumnents to SLURM
        right   = tmp[-1].split(" ")[1:]
        # concatenate the commands together
        tmp     = left[:-1] + [mid] + right
        # check what we running
        print(tmp)
        # run the script within a subprocess
        result  = subprocess.run(tmp, stdout=subprocess.PIPE)
        # see what the result is
        result  = result.stdout.decode('utf-8')
        
        if resultError(result):
            print(result)
            break
        # write the lines already running
        outp.writelines(str(tmp) + "\n")
        iter    += 1
    # go back to the beginning
    f.seek(0)
    # clear file
    f.truncate()
    # save only the lines that have not been run
    lines = lines[iter:]
    f.writelines(lines)
# close the output
outp.close()

###################################################################

# if the file is empty, remove it
# if os.path.getsize(iniFile) == 0:
    # print("\t", "REMOVING: ", f"iniFile")
    # os.remove(iniFile)

