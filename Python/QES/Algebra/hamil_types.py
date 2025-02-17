from enum import Enum, unique

# --- ED limits (using hexadecimal where applicable)
UI_LIMITS_MAXFULLED             = 0x40000
UI_LIMITS_MAXPRINT              = 0x8
UI_LIMITS_SI_STATENUM           = 100
UI_LIMITS_MIDDLE_SPEC_STATENUM  = 200

# ############################ EXISTING MODELS ############################
@unique
class Hamiltonians(Enum):
    NONE            = 0
    ISING_M         = 1
    XYZ_M           = 2
    HEI_KIT_M       = 3
    QSM_M           = 4
    RP_M            = 5
    ULTRAMETRIC_M   = 6
    # quadratic (or noninteracting) models:
    FREE_FERMIONS_M = 100
    AUBRY_ANDRE_M   = 101
    SYK2_M          = 102
    ANDERSON_M      = 103
    POWER_LAW_RANDOM_BANDED_M = 104

MY_MODELS_LIST = [
    Hamiltonians.ISING_M,
    Hamiltonians.XYZ_M,
    Hamiltonians.HEI_KIT_M,
    Hamiltonians.QSM_M,
    Hamiltonians.RP_M,
    Hamiltonians.ULTRAMETRIC_M,
    Hamiltonians.FREE_FERMIONS_M,
    Hamiltonians.AUBRY_ANDRE_M,
    Hamiltonians.SYK2_M,
    Hamiltonians.ANDERSON_M,
    Hamiltonians.POWER_LAW_RANDOM_BANDED_M,
    Hamiltonians.NONE,
]

# Maximum interacting model value defined as the value of ULTRAMETRIC_M.
MY_MODELS_MAX_INTERACTING = Hamiltonians.ULTRAMETRIC_M.value

################################################################################

def check_noninteracting(model: Hamiltonians) -> bool:
    """
    Check if a model is noninteracting.
    
    In this implementation, a model is considered noninteracting if its value is
    greater than or equal to FREE_FERMIONS_M or if it is equal to RP_M.
    """
    return (model.value >= Hamiltonians.FREE_FERMIONS_M.value) or (model == Hamiltonians.RP_M)

def check_dense(model: Hamiltonians) -> bool:
    """
    Check if a model is dense.
    
    Dense models are defined as those corresponding to:
        - POWER_LAW_RANDOM_BANDED_M,
        - ULTRAMETRIC_M, or
        - RP_M.
    """
    return model in (Hamiltonians.POWER_LAW_RANDOM_BANDED_M, Hamiltonians.ULTRAMETRIC_M, Hamiltonians.RP_M)

################################################################################