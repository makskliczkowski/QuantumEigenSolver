'''
file    : QES/Algebra/Model/Noninteracting/Conserving/free_fermions.py
author  : Maksymilian Kliczkowski
date    : 2025-04-29
version : 1.0
Title   : Free Fermions Model 

'''

import numpy as np
from typing import List, Optional, Union

# import the Hamiltonian and HilbertSpace classes
from Algebra.hamil import Hamiltonian, HilbertSpace

class FreeFermions(Hamiltonian):


    def __init__(self, ns: int, hilbert_space: HilbertSpace):
        super().__init__(ns, hilbert_space)