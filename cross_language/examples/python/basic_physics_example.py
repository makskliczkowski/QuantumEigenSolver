import numpy as np

from QES.general_python.common.binary import int2base
from QES.general_python.lattices.square import SquareLattice
from QES.general_python.physics.eigenlevels import gap_ratio, entropy_vonNeuman
from QES.general_python.algebra.eigen.factory import choose_eigensolver


def main():
    print("--- Python Physics Example ---")

    bits = int2base(13, 6, spin=True, spin_value=1.0)
    print("int2base(13,6):", bits.tolist())

    lat = SquareLattice(lx=2, ly=3, dim=2)
    print("square lattice Ns:", int(lat.Ns))

    vals = np.array([-2.0, -1.25, -0.2, 0.7, 1.5, 2.4, 3.1], dtype=float)
    gr = gap_ratio(vals, fraction=0.8, use_mean_lvl_spacing=True)
    print("gap ratio mean/std:", float(gr["mean"]), float(gr["std"]))

    psi = np.array([0.2, -0.3, 0.1, 0.25, -0.4, 0.15, 0.5, -0.2, 0.05, 0.1, -0.15, 0.2, -0.1, 0.3, -0.25, 0.35], dtype=float)
    psi = psi / np.linalg.norm(psi)
    s = entropy_vonNeuman(psi, 4, 2, TYP="SCHMIDT")
    print("entropy:", float(s))

    A = np.array([
        [2.0, 0.5, 0.0, 0.0],
        [0.5, 1.0, 0.25, 0.0],
        [0.0, 0.25, 1.5, 0.75],
        [0.0, 0.0, 0.75, 3.0],
    ], dtype=float)
    res = choose_eigensolver(method="exact", A=A, hermitian=True)
    print("lowest eigenvalue:", float(np.sort(np.real(res.eigenvalues))[0]))


if __name__ == "__main__":
    main()
