import os
import subprocess
import sys

iter    =   0
outp    =   open(f"output_Manifold.dat", "a+")
with open(f"input_Manifold.ini", "r+") as f:
    lines = f.readlines()
    for l in lines:
        tmp     = l.split("\n")[0]
        tmp     = tmp.split('"')
        left    = tmp[0].split(" ")
        mid     = tmp[1]
        right   = tmp[-1].split(" ")[1:]
        tmp     = left[:-1] + [mid] + right
        print(tmp)
        result  = subprocess.run(tmp, stdout=subprocess.PIPE)
        result  = result.stdout.decode('utf-8')
        if '0' not in result:
            print(result)
            break
        outp.writelines(str(tmp) + "\n")
        iter+=1
    f.seek(0)
    f.truncate()
    lines = lines[iter:]
    f.writelines(lines)
outp.close()
if os.path.getsize(f"input_Manifold.ini") == 0:
    print("REMOVING: ", f"input_Manifold.ini")
    os.remove(f"input_Manifold.ini")

