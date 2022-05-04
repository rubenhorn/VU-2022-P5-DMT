# Import this at the very top of your file to accelerate your scikit-learn code
print('Optimizing sklearn functions...')
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
