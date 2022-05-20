random_state = 42
# Number of folds or None
cv = 3
# Number of samples using random parameters (non exhaustive search)
n_iter = 10
tf_epochs = 50
learning_rate = 0.03
momentum = 0.9

rf_params = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'random_state': [random_state],
}

sgd_params = {
    'loss': ['log', 'hinge'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'random_state': [random_state],
    'max_iter': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
}

rbf_params = {
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'random_state': [random_state],
}

param_grid_pipeline = [{}]

key_prefix_sgd = 'classifier__estimator__'
for key, value in sgd_params.items():
    param_grid_pipeline[0][key_prefix_sgd + key] = value

key_prefix_rbf = 'rbf__'
for key, value in rbf_params.items():
    # NOTE: If you remove the RBFSampler, you have to comment out the following line
    param_grid_pipeline[0][key_prefix_rbf + key] = value
    pass

# Run inference on all cores
param_grid_pipeline[0]['classifier__n_jobs'] = [-1]

kmeans_params = {
    # 'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'n_clusters': [8,9,10,11,12,13,14,15,16,17,18],
    'random_state': [random_state],
}

param_grid_clustering = [{}]

key_prefix_kmeans = 'clustering__'
for key, value in kmeans_params.items():
    param_grid_clustering[0][key_prefix_kmeans + key] = value
