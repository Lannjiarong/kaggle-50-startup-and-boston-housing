import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
TEST_SIZE = 0.2
RANDOM_STATE = 42
ALGORITHM_NAMES = ('Lasso', 'Forward SFS', 'RFE', 'ElasticNet', 'Exhaustive')


def load_datasets():
    startups = pd.read_csv(ROOT / '50_Startups.csv')
    startups_x = pd.get_dummies(
        startups.drop(columns='Profit'),
        columns=['State'],
        prefix=['State'],
        drop_first=True,
        dtype=float,
    )

    boston = pd.read_csv(ROOT / 'BostonHousing.csv')
    boston.columns = [column.upper() for column in boston.columns]

    return {
        '50startups': (startups_x.astype(float), startups['Profit'].astype(float), 'Profit'),
        'boston': (boston.drop(columns='MEDV').astype(float), boston['MEDV'].astype(float), 'MEDV'),
    }


def rank_features(x_train, y_train):
    feature_names = list(x_train.columns)
    x_scaled = StandardScaler().fit_transform(x_train)
    y_scaled = StandardScaler().fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()

    lasso = Lasso(alpha=0.05, max_iter=10000).fit(x_scaled, y_scaled)
    lasso_order = np.argsort(-np.abs(lasso.coef_), kind='stable')

    elastic_net = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000).fit(x_scaled, y_scaled)
    elastic_order = np.argsort(-np.abs(elastic_net.coef_), kind='stable')

    forward_order = []
    remaining = list(range(len(feature_names)))
    for _ in feature_names:
        best_feature = None
        best_score = -np.inf
        for feature_index in remaining:
            candidates = forward_order + [feature_index]
            model = LinearRegression().fit(x_train.iloc[:, candidates], y_train)
            score = model.score(x_train.iloc[:, candidates], y_train)
            if score > best_score:
                best_feature = feature_index
                best_score = score
        forward_order.append(best_feature)
        remaining.remove(best_feature)

    rfe_rankings = []
    for count in range(1, len(feature_names) + 1):
        selector = RFE(LinearRegression(), n_features_to_select=count)
        selector.fit(x_train, y_train)
        rfe_rankings.append(np.flatnonzero(selector.support_).tolist())

    exhaustive_rankings = best_subset_rankings(x_train, y_train)

    return {
        'Lasso': [lasso_order[:count].tolist() for count in range(1, len(feature_names) + 1)],
        'Forward SFS': [forward_order[:count] for count in range(1, len(feature_names) + 1)],
        'RFE': rfe_rankings,
        'ElasticNet': [elastic_order[:count].tolist() for count in range(1, len(feature_names) + 1)],
        'Exhaustive': exhaustive_rankings,
    }


def best_subset_rankings(x_train, y_train):
    values = x_train.to_numpy(dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales == 0] = 1
    standardized = (values - means) / scales
    centered_target = np.asarray(y_train, dtype=float) - np.mean(y_train)
    gram = standardized.T @ standardized
    cross = standardized.T @ centered_target
    target_sum_squares = centered_target @ centered_target

    rankings = []
    for count in range(1, values.shape[1] + 1):
        best_subset = None
        best_rss = np.inf
        for subset in itertools.combinations(range(values.shape[1]), count):
            subset_cross = cross[list(subset)]
            beta = np.linalg.lstsq(gram[np.ix_(subset, subset)], subset_cross, rcond=None)[0]
            rss = max(0.0, target_sum_squares - beta @ subset_cross)
            if rss < best_rss:
                best_rss = rss
                best_subset = list(subset)
        rankings.append(best_subset)
    return rankings


def calculate_model(selected_features, x_train, y_train, x_test, y_test, test_indices):
    feature_names = list(x_train.columns)
    selected_names = [feature_names[index] for index in selected_features]
    x_train_selected = x_train[selected_names].to_numpy(dtype=float)
    x_test_selected = x_test[selected_names].to_numpy(dtype=float)
    train_design = np.column_stack((np.ones(len(y_train)), x_train_selected))
    test_design = np.column_stack((np.ones(len(y_test)), x_test_selected))
    y_train_values = np.asarray(y_train, dtype=float)
    y_test_values = np.asarray(y_test, dtype=float)

    beta = np.linalg.lstsq(train_design, y_train_values, rcond=None)[0]
    train_residuals = y_train_values - train_design @ beta
    train_rss = float(train_residuals @ train_residuals)
    train_tss = float(np.sum((y_train_values - np.mean(y_train_values)) ** 2))
    degrees_residual = len(y_train) - len(selected_names) - 1
    residual_variance = train_rss / degrees_residual if degrees_residual > 0 else 0.0
    covariance = np.linalg.pinv(train_design.T @ train_design) * residual_variance
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_statistics = np.divide(beta, standard_errors, out=np.zeros_like(beta), where=standard_errors > 0)
    t_pvalues = (
        2 * stats.t.sf(np.abs(t_statistics), degrees_residual)
        if degrees_residual > 0 else np.ones_like(t_statistics)
    )
    t_critical = stats.t.ppf(0.975, degrees_residual) if degrees_residual > 0 else 1.96

    y_predicted = test_design @ beta
    residuals = y_test_values - y_predicted
    test_rss = float(residuals @ residuals)
    test_tss = float(np.sum((y_test_values - np.mean(y_test_values)) ** 2))
    r2 = 1.0 - test_rss / test_tss if test_tss > 0 else 0.0
    test_degrees_residual = len(y_test) - len(selected_names) - 1
    adjusted_r2 = (
        1.0 - (1.0 - r2) * (len(y_test) - 1) / test_degrees_residual
        if test_degrees_residual > 0 else r2
    )

    model_degrees = len(selected_names)
    f_stat = (
        ((train_tss - train_rss) / model_degrees) / residual_variance
        if model_degrees > 0 and residual_variance > 0 else 0.0
    )
    f_pvalue = float(stats.f.sf(f_stat, model_degrees, degrees_residual)) if degrees_residual > 0 else 1.0

    coefficients = [{
        'feature': 'Intercept',
        'coef': float(beta[0]),
        'stderr': float(standard_errors[0]),
        'tstat': float(t_statistics[0]),
        'pvalue': float(t_pvalues[0]),
        'ci_lower': float(beta[0] - t_critical * standard_errors[0]),
        'ci_upper': float(beta[0] + t_critical * standard_errors[0]),
        'vif': None,
    }]

    for index, feature in enumerate(selected_names, start=1):
        vif = calculate_vif(x_train[selected_names], feature)
        coefficients.append({
            'feature': feature,
            'coef': float(beta[index]),
            'stderr': float(standard_errors[index]),
            'tstat': float(t_statistics[index]),
            'pvalue': float(t_pvalues[index]),
            'ci_lower': float(beta[index] - t_critical * standard_errors[index]),
            'ci_upper': float(beta[index] + t_critical * standard_errors[index]),
            'vif': vif,
        })

    outlier_indices = np.argsort(-np.abs(residuals), kind='stable')[:10]
    outliers = []
    for rank, position in enumerate(outlier_indices, start=1):
        actual = float(y_test_values[position])
        residual = float(residuals[position])
        outliers.append({
            'rank': rank,
            'index': int(test_indices[position]),
            'actual': round(actual, 2),
            'predicted': round(float(y_predicted[position]), 2),
            'residual': round(residual, 2),
            'abs_error': round(abs(residual), 2),
            'pct_error': round(abs(residual / actual) * 100, 2) if actual != 0 else 0.0,
        })

    return {
        'k': len(selected_names),
        'features': selected_names,
        'rmse': float(np.sqrt(test_rss / len(y_test_values))),
        'mae': float(np.mean(np.abs(residuals))),
        'r2': float(r2),
        'adj_r2': float(adjusted_r2),
        'f_stat': float(f_stat),
        'f_pvalue': f_pvalue,
        'coefficients': coefficients,
        'outliers': outliers,
    }


def calculate_vif(x_train, feature):
    if x_train.shape[1] == 1:
        return 1.0
    others = x_train.drop(columns=feature)
    model = LinearRegression().fit(others, x_train[feature])
    r2 = min(float(model.score(others, x_train[feature])), 1.0)
    return round(min(1.0 / max(1.0 - r2, 1e-12), 100.0), 2)


def train_dataset(x, y):
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    x_train = x.iloc[train_indices].reset_index(drop=True)
    x_test = x.iloc[test_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)
    rankings = rank_features(x_train, y_train)

    models = {}
    for algorithm in ALGORITHM_NAMES:
        models[algorithm] = [
            calculate_model(features, x_train, y_train, x_test, y_test, test_indices)
            for features in rankings[algorithm]
        ]

    return models, {
        'rows': len(y),
        'train_rows': len(train_indices),
        'test_rows': len(test_indices),
        'features': x.shape[1],
        'feature_defaults': {name: float(value) for name, value in x_train.mean().items()},
    }


def main():
    datasets = load_datasets()
    model_data = {}
    training_info = {
        'split': '80/20 train/test',
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'metric_source': 'held-out test set',
        'datasets': {},
    }

    for name, (x, y, target) in datasets.items():
        model_data[name], dataset_info = train_dataset(x, y)
        training_info['datasets'][name] = {'target': target, **dataset_info}
        best = min(model_data[name]['Lasso'], key=lambda item: item['rmse'])
        print(
            f'{name}: trained {dataset_info["train_rows"]} rows, tested '
            f'{dataset_info["test_rows"]} rows; Lasso best test RMSE '
            f'k={best["k"]}: {best["rmse"]:.4f}'
        )

    startup_json = json.dumps(model_data['50startups'], indent=2, ensure_ascii=False, allow_nan=False)
    boston_json = json.dumps(model_data['boston'], indent=2, ensure_ascii=False, allow_nan=False)
    combined_json = json.dumps(model_data, indent=2, ensure_ascii=False, allow_nan=False)
    training_json = json.dumps(training_info, indent=2, ensure_ascii=False, allow_nan=False)

    (ROOT / 'startup_model_data.json').write_text(startup_json + '\n', encoding='utf-8')
    (ROOT / 'boston_model_data.json').write_text(boston_json + '\n', encoding='utf-8')
    (ROOT / 'model_data.js').write_text(
        'window.PROBLEMS_DATA = ' + combined_json + ';\n'
        'window.MODEL_DATA = window.PROBLEMS_DATA["50startups"];\n'
        'window.TRAINING_INFO = ' + training_json + ';\n',
        encoding='utf-8',
    )
    print('Generated startup_model_data.json, boston_model_data.json, and model_data.js.')


if __name__ == '__main__':
    main()
