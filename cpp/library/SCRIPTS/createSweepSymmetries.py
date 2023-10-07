import subprocess
import os
import sys
VALNONE=-1000
SCRIPT="./skrypt_run_symmetries_sweep.sh"

eta1 = 0.0
dlt1 = float(sys.argv[2])
Ns = int(sys.argv[1])
SYM = Ns >= 22

TIM = ""
MEM = ""
CPU = 1
FUN = 22
if Ns <= 20:
    TIM = "99:59:59"
    MEM = "64gb"
    CPU = 16
elif Ns <= 21:
    TIM = "10:59:59"
    MEM = "64gb"
    CPU = 48
elif Ns <= 23:
    TIM = "49:59:59"
    MEM = "64gb"
    CPU = 48
elif Ns < 24:
    TIM = "99:59:59"
    MEM = "192gb"
    CPU = 48
else:
    print("WHAT")
    TIM = "199:59:59"
    MEM = "370gb"
    CPU = 48


useU1 = False if eta1 != 0. else True
useSz = True
useSy = False #if Ns % 2 != 0 else True

pys = [-1, 1] if useSy else [None]
pzs = [-1, 1] if useSz else [None]
# u1s = [i for i in range(0, Ns//2 + 1)] if useU1 else [None]
# u1s = [i for i in range(Ns//2 + 1, Ns + 1)] if useU1 else [None]
# u1s = [i for i in range(0, Ns + 1)] if useU1 else [None]
u1s = ([i for i in range(0, Ns//2)] + [i for i in range(Ns//2 + 1, Ns + 1)]) if useU1 else [None]
# u1s = [10, 11, 12, 13, 14]
# if Ns//2 not in u1s and Ns >= 24:
#     TIM = "99:59:59"
#     MEM = "192gb"
#     CPU = 48

if Ns >= 25:
    u1s = [Ns // 4, Ns // 2]
ks = [i for i in range(Ns//2+1)]
if Ns >= 26:
    u1s = [Ns//4, Ns//2]
    ks = [0, Ns//2]
if Ns >= 28:
    u1s = [Ns//4]
    

print("START")
f = open(f"input{Ns}.ini", "w")
if(SYM):
    FUN = 21
    for U1 in u1s:
        SYMU1 = f"-U1 {U1} " if U1 is not None else ""
        for k in ks:
            # if(U1 == Ns // 2) and k != 0:
            #     continue
            SYMK = f"-k {k} " 
            rs  = [-1, 1] if (k == 0 or (k == Ns // 2 and Ns % 2 == 0)) else [None]
            pxs = [-1, 1] if (Ns % 2 == 0 and U1 == Ns // 2) else [None]
            for px in pxs:
                SYMPX = f"-px {px} " if px is not None else ""
                for py in pys:
                    SYMPY = f"-py {py} " if py is not None else ""
                    for pz in pzs:
                        SYMPZ = f"-pz {pz} " if pz is not None else ""
                        for r in rs:
                            SYMR = f"-x {r} " if r is not None else ""
                            SYMS = (SYMU1 + SYMK + SYMPX + SYMPY + SYMPZ + SYMR)[:-1]
                            print("\n-------------\n", "->DOING: ", SYMS, "-------------")
                            #os.system(f'sh {SCRIPT} {Ns} {dlt1} {eta1} "{SYMS}"')
                            f.writelines([f'sh {SCRIPT} {Ns} {dlt1} {eta1} "{SYMS}" {TIM} {MEM} {CPU} {FUN}\n'])
                            # print(subprocess.run([SCRIPT, str(Ns), str(dlt1), str(eta1), SYMS], shell=True))
else:
    FUN = 22
    f.writelines([f'sh {SCRIPT} {Ns} {dlt1} {eta1} "" {TIM} {MEM} {CPU} {FUN}\n'])
f.close()
