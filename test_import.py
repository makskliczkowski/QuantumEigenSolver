#!/usr/bin/env python3

import sys
import os

# Add the Python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyqusolver', 'Python'))

try:
    import QES
    print("✓ QES import successful")
    print(f"QES version: {QES.__version__}")
    
    # Test individual modules
    modules_to_test = [
        'QES.Algebra',
        'QES.Solver', 
        'QES.NQS',
        'QES.general_python',
        'QES.general_python.physics',
        'QES.general_python.ml'
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module} import successful")
        except Exception as e:
            print(f"✗ {module} import failed: {e}")
            
except Exception as e:
    print(f"✗ QES import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
