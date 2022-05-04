# Import this at the very top of your file to accelerate your scikit-learn code
print('Auto-optimizing sklearn functions...')
from sklearnex import patch_sklearn, config_context
patch_sklearn()

def on_gpu():
    print('Offloading to GPU...')
    return config_context(target_offload="gpu:0", allow_fallback_to_host=False)
