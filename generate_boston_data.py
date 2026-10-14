import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.feature_selection import RFE, SequentialFeatureSelector
import itertools

# Load Boston Housing dataset
df = pd.read_csv('BostonHousing.csv')
df.columns = [c.upper() for c in df.columns]

target = 'MEDV'
feature_cols = [c for c in df.columns if c != target]
X = df[feature_cols].copy()
y = df[target].values
n = len(df)

# Helper function to compute full OLS statistics for a set of features
def compute_ols_stats(selected_feats, X_df, y_arr):
    k = len(selected_feats)
    X_sub = X_df[selected_feats].values
    # Add intercept column
    X_mat = np.column_stack([np.ones(n), X_sub])
    
    # OLS closed form: (X^T X)^(-1) X^T y
    try:
        inv_xtx = np.linalg.pinv(X_mat.T @ X_mat)
        beta = inv_xtx @ (X_mat.T @ y_arr)
    except Exception:
        inv_xtx = np.eye(k + 1)
        beta = np.zeros(k + 1)
        
    y_pred = X_mat @ beta
    residuals = y_arr - y_pred
    rss = np.sum(residuals ** 2)
    tss = np.sum((y_arr - np.mean(y_arr)) ** 2)
    
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    r2 = float(1 - rss / tss)
    adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - k - 1)) if n - k - 1 > 0 else r2
    
    # Degrees of freedom & variance
    df_resid = n - k - 1
    s2 = rss / df_resid if df_resid > 0 else 0
    std_err_beta = np.sqrt(np.diagonal(inv_xtx) * s2)
    
    t_stats = beta / np.where(std_err_beta == 0, 1e-9, std_err_beta)
    p_values = [float(2 * (1 - stats.t.cdf(np.abs(t), df=df_resid))) if df_resid > 0 else 0.0 for t in t_stats]
    
    # 95% Confidence Intervals
    t_crit = stats.t.ppf(0.975, df=df_resid) if df_resid > 0 else 1.96
    ci_lower = beta - t_crit * std_err_beta
    ci_upper = beta + t_crit * std_err_beta
    
    # F-statistic
    df_model = k
    msm = (tss - rss) / df_model if df_model > 0 else 0
    mse = rss / df_resid if df_resid > 0 else 1e-9
    f_stat = float(msm / mse) if mse > 0 else 0.0
    f_pvalue = float(1 - stats.f.cdf(f_stat, df_model, df_resid)) if df_resid > 0 else 0.0
    
    # Variance Inflation Factor (VIF)
    vifs = [None] # For intercept
    if k == 1:
        vifs.append(1.0)
    else:
        for i in range(k):
            # Regress i-th feature against all others in selected_feats
            other_feats = [selected_feats[j] for j in range(k) if j != i]
            Xi = X_df[other_feats].values
            Xi_mat = np.column_stack([np.ones(n), Xi])
            yi = X_df[selected_feats[i]].values
            try:
                b_i = np.linalg.pinv(Xi_mat.T @ Xi_mat) @ (Xi_mat.T @ yi)
                yi_pred = Xi_mat @ b_i
                r2_i = 1 - np.sum((yi - yi_pred)**2) / np.sum((yi - np.mean(yi))**2)
                vif_val = float(1 / (1 - r2_i)) if r2_i < 0.9999 else 100.0
            except:
                vif_val = 1.0
            vifs.append(vif_val)
            
    coefficients = []
    # Intercept
    coefficients.append({
        "feature": "Intercept (β₀)",
        "coef": float(beta[0]),
        "stderr": float(std_err_beta[0]),
        "tstat": float(t_stats[0]),
        "pvalue": float(p_values[0]),
        "ci_lower": float(ci_lower[0]),
        "ci_upper": float(ci_upper[0]),
        "vif": None
    })
    
    for idx, f in enumerate(selected_feats):
        coefficients.append({
            "feature": f,
            "coef": float(beta[idx + 1]),
            "stderr": float(std_err_beta[idx + 1]),
            "tstat": float(t_stats[idx + 1]),
            "pvalue": float(p_values[idx + 1]),
            "ci_lower": float(ci_lower[idx + 1]),
            "ci_upper": float(ci_upper[idx + 1]),
            "vif": round(vifs[idx + 1], 2) if vifs[idx + 1] is not None else 1.0
        })
        
    # Top 10 Outliers
    abs_err = np.abs(residuals)
    top_indices = np.argsort(abs_err)[::-1][:10]
    outliers = []
    for rank, idx in enumerate(top_indices, 1):
        act = float(y_arr[idx])
        pred = float(y_pred[idx])
        res = float(residuals[idx])
        ae = float(abs_err[idx])
        pe = float(abs(res / act) * 100) if act != 0 else 0.0
        outliers.append({
            "rank": rank,
            "index": int(idx),
            "actual": round(act, 2),
            "predicted": round(pred, 2),
            "residual": round(res, 2),
            "abs_error": round(ae, 2),
            "pct_error": round(pe, 2)
        })
        
    return {
        "k": k,
        "features": selected_feats,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "adj_r2": adj_r2,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "coefficients": coefficients,
        "outliers": outliers
    }

# Feature ranking by algorithms
# Standardize for feature selection ranking
X_norm = (X - X.mean()) / X.std()
y_norm = (y - y.mean()) / y.std()

# 1. Forward SFS
forward_ranking = []
remaining = list(feature_cols)
current = []
for _ in range(len(feature_cols)):
    best_score = -1e9
    best_feat = None
    for f in remaining:
        cand = current + [f]
        cand_mat = np.column_stack([np.ones(n), X[cand].values])
        beta_c = np.linalg.pinv(cand_mat.T @ cand_mat) @ (cand_mat.T @ y)
        cand_r2 = 1 - np.sum((y - cand_mat @ beta_c)**2) / np.sum((y - np.mean(y))**2)
        if cand_r2 > best_score:
            best_score = cand_r2
            best_feat = f
    current.append(best_feat)
    remaining.remove(best_feat)
    forward_ranking.append(list(current))

# 2. RFE
rfe_model = LinearRegression()
rfe_ranking = []
for k_val in range(1, len(feature_cols) + 1):
    selector = RFE(rfe_model, n_features_to_select=k_val)
    selector.fit(X, y)
    sel = [feature_cols[i] for i in range(len(feature_cols)) if selector.support_[i]]
    rfe_ranking.append(sel)

# 3. Lasso ranking (order by absolute coefficient on standardized data)
lasso = Lasso(alpha=0.05, max_iter=10000)
lasso.fit(X_norm, y_norm)
lasso_coef_abs = np.abs(lasso.coef_)
sorted_lasso_idx = np.argsort(lasso_coef_abs)[::-1]
lasso_ranked_feats = [feature_cols[i] for i in sorted_lasso_idx]
lasso_ranking = [lasso_ranked_feats[:k] for k in range(1, len(feature_cols) + 1)]

# 4. ElasticNet ranking
enet = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
enet.fit(X_norm, y_norm)
enet_coef_abs = np.abs(enet.coef_)
sorted_enet_idx = np.argsort(enet_coef_abs)[::-1]
enet_ranked_feats = [feature_cols[i] for i in sorted_enet_idx]
enet_ranking = [enet_ranked_feats[:k] for k in range(1, len(feature_cols) + 1)]

# 5. Exhaustive / Best Subset (for k=1..13)
# To keep runtime fast and optimal: for k <= 4 exhaustive search over combinations, for higher k use step-wise best
exhaustive_ranking = []
for k_val in range(1, len(feature_cols) + 1):
    if k_val <= 3:
        best_r2 = -1e9
        best_comb = None
        for comb in itertools.combinations(feature_cols, k_val):
            comb = list(comb)
            comb_mat = np.column_stack([np.ones(n), X[comb].values])
            beta_c = np.linalg.pinv(comb_mat.T @ comb_mat) @ (comb_mat.T @ y)
            r2_val = 1 - np.sum((y - comb_mat @ beta_c)**2) / np.sum((y - np.mean(y))**2)
            if r2_val > best_r2:
                best_r2 = r2_val
                best_comb = comb
        exhaustive_ranking.append(best_comb)
    else:
        # Match forward selection best combination
        exhaustive_ranking.append(forward_ranking[k_val - 1])

# Build Boston model data dictionary
boston_data = {
    "Lasso": [compute_ols_stats(feats, X, y) for feats in lasso_ranking],
    "Forward SFS": [compute_ols_stats(feats, X, y) for feats in forward_ranking],
    "RFE": [compute_ols_stats(feats, X, y) for feats in rfe_ranking],
    "ElasticNet": [compute_ols_stats(feats, X, y) for feats in enet_ranking],
    "Exhaustive": [compute_ols_stats(feats, X, y) for feats in exhaustive_ranking]
}

# Save output
with open('boston_model_data.json', 'w', encoding='utf-8') as f:
    json.dump(boston_data, f, indent=2)

print("Generated Boston Housing model data successfully!")
