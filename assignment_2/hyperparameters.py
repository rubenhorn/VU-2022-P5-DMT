# TODO set to False for actual training
use_small_dataset = True
random_state = 42
# Number of folds or None
cv = 3
# Number of samples using random parameters (non exhaustive search)
n_iter = 10

rf_params = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'random_state': [random_state],
}

param_grid = [{}]
# Run inference on all cores
param_grid[0]['classifier__n_jobs'] = [-1]
# key_prefix_rf = 'classifier__estimator__'
# for key, value in rf_params.items():
#     param_grid[0][key_prefix_rf + key] = value
