import os
import  subprocess
import sys

# read system size
Ns      = int(sys.argv[1])

iter    = 0
outp    =   open(f"output{Ns}.dat", "a+")
with open(f"input{Ns}_inter.ini", "r+") as f:
    # readlines from the file
    lines = f.readlines()
    for l in lines:
        # split and get the line
        tmp     = l.split("\n")[0]
        # get the parameters
        tmp     = tmp.split('"')
        left    = tmp[0].split(" ")
        mid     = tmp[1]
        right   = tmp[-1].split(" ")[1:]
        # join them back
        tmp     = left[:-1] + [mid] + right
        print(tmp)
        # run what is written there
        result  = subprocess.run(tmp, stdout=subprocess.PIPE, check=True)
        result  = result.stdout.decode('utf-8')
        # if couldn't run, break!
        if '0' not in result:
            print(result)
            break
        # write output for logging
        outp.writelines(str(tmp) + "\n")
        iter    += 1
    # truncate
    f.seek(0)
    f.truncate()
    lines = lines[iter:]
    f.writelines(lines)
outp.close()
if os.path.getsize(f"input{Ns}.ini") == 0:
    print("REMOVING: ", f"input{Ns}.ini")
    os.remove(f"input{Ns}.ini")

