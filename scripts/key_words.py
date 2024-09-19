from enum import Enum

class Keywords(Enum):
    LINEAR_REAGRESSION_KEYWORDS = [
        'fit_intercept',
        'copy_X',
        'n_jobs',
        'positive'
    ]
    RANDOM_FOREST_KEYWORDS = [
        'n_estimators',
        'criterion',
        'max_depth',
        'min_samples_split',
        'min_samples_leaf',
        'min_weight_fraction_leaf',
        'max_features',
        'max_leaf_nodes',
        'min_impurity_decrease',
        'bootstrap',
        'oob_score',
        'n_jobs',
        'random_state',
        'verbose',
        'warm_start',
        'ccp_alpha',
        'max_samples',
        'monotonic_cst',
        'estimator_',
        'estimators_',
        'n_features_in_',
        'feature_names_in_',
        'n_outputs_',
        'oob_score_',
        'oob_prediction_',
        'estimators_samples_'
    ]