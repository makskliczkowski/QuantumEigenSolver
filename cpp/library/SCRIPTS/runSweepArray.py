import os
import  subprocess
import sys
import time

print(sys.argv)
Ns      = int(sys.argv[1])
JobN    = int(sys.argv[3])
ArrJobN = int(sys.argv[2])

iter    =   0
outp    =   open(f"output{Ns}.dat", "a+")


wLine   =   lambda FUN, L, eta1, dlt1, SYMS, a: f"bash ../qsolver.o -fun ${FUN} -mod 1 -bc 0 -l 0 -d 1 -Lx ${L} -Ly 1 -Lz 1 -J1 1.0 -J2 0 -hx 0 -hz 0 -eta1 ${eta1} -eta2 0 -dlt1 ${dlt1} -dlt2 0 -S 1 ${SYMS} -th 24 -dir SUSY/ >& ./LOG/log_${a}.txt"

with open(f"input{Ns}.ini", "r+") as f:
    lines   = f.readlines()
    l       = []
    if len(lines) > JobN:
        l   = lines[JobN]
    else:
        exit()
        
    tmp     = l.split("\n")[0]
    tmp     = tmp.split('"')
    left    = tmp[0].split(" ")
    mid     = tmp[1]
    right   = tmp[-1].split(" ")[1:]
    tmp     = left[:-1] + [mid] + right
    result  = subprocess.run(wLine(tmp[-1], tmp[2], tmp[4], tmp[3], tmp[5], tmp[5] + str(JobN)), stdout=subprocess.PIPE)
    # result = subprocess.run(tmp, stdout=subprocess.PIPE)
   # result = result.stdout.decode('utf-8')
    #if '0' not in result:
    #    print(result)
    #    break
    outp.writelines(str(tmp) + "\n")
    
    # terminate
    if JobN == ArrJobN - 1:          
        time.sleep(1)
        f.seek(0)
        f.truncate()
        lines = lines[ArrJobN:]
        f.writelines(lines)        
outp.close()

if os.path.getsize(f"input{Ns}.ini") == 0:
    print("REMOVING: ", f"input{Ns}.ini")
    os.remove(f"input{Ns}.ini")

