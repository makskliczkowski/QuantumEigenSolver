########################################################################
#! RESOLVE PATH
########################################################################

import sys
import os

from pathlib import Path
cwd         = Path.cwd()
mod_path    = Path(__file__).resolve()
qes_path    = Path(__file__).parent.parent.parent
lib_path    = qes_path / 'QES'
print("Current working directory:", cwd)
print("Module path:", mod_path)
print("QES path:", qes_path)
print("Library path:", lib_path, "\n\tWith folders:", os.listdir(lib_path))
sys.path.insert(0, str(lib_path))