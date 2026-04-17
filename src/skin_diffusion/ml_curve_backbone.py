import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


def parse_float_csv(csv_text, fallback_values):
    # Parse comma-separated floats and fall back to defaults when input is empty.
    parsed_values = []
    token_list = str(csv_text).split(",")
    token_index = 0
    while token_index < len(token_list):
        token = token_list[token_index].strip()
        if token != "":
            parsed_values.append(float(token))
        token_index += 1

    if len(parsed_values) == 0:
        fallback_out = []
        fallback_index = 0
        while fallback_index < len(fallback_values):
            fallback_out.append(float(fallback_values[fallback_index]))
            fallback_index += 1
        return fallback_out
    return parsed_values


def choose_curve_log_transform(values):
    # Use log-space only when the target spread is wide enough to justify it.
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 2:
        return {"kind": "identity"}

    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size < 2:
        return {"kind": "identity"}

    min_value = max(float(np.min(positive_values)), 1e-30)
    max_value = float(np.max(positive_values))
    spread_ratio = max_value / min_value
    if spread_ratio < 20.0:
        return {"kind": "identity"}

    eps_value = max(min_value * 1e-3, 1e-30)
    return {"kind": "log10", "eps": eps_value}


def apply_curve_transform(values, transform):
    # Transform targets before PCA fitting.
    if transform["kind"] == "identity":
        return values
    eps_value = float(transform["eps"])
    clipped = np.maximum(values, eps_value)
    return np.log10(clipped)


def invert_curve_transform(values, transform):
    # Map transformed values back onto the original flux scale.
    if transform["kind"] == "identity":
        return values
    clipped = np.clip(values, -300.0, 300.0)
    return np.power(10.0, clipped)


def fit_curve_backbone(
    x_train_raw,
    j_train_true,
    x_val_raw,
    j_val_true,
    curve_components=20,
    alpha_grid=None,
    use_xgboost=False,
):
    # Fit PCA + regressor curve backbone.  When use_xgboost is True the candidate pool
    # includes XGBoost multi-output regressors alongside Ridge for a stronger corrective baseline.
    if alpha_grid is None:
        alpha_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_val_scaled = scaler.transform(x_val_raw)

    transform = choose_curve_log_transform(j_train_true)
    j_train_transformed = apply_curve_transform(j_train_true, transform)
    max_components = min(
        int(curve_components),
        int(j_train_transformed.shape[0]),
        int(j_train_transformed.shape[1]),
    )
    if max_components < 1:
        raise ValueError("Backbone requires at least one PCA component")

    pca = PCA(n_components=max_components)
    z_train = pca.fit_transform(j_train_transformed)

    # Build candidate list: always include Ridge, optionally add XGBoost 
    candidates = []
    alpha_index = 0
    while alpha_index < len(alpha_grid):
        candidates.append(
            {"family": "ridge", "alpha": float(alpha_grid[alpha_index])}
        )
        alpha_index += 1

    if use_xgboost:
        from xgboost import XGBRegressor
        for n_est, depth, lr in [(300, 3, 0.05), (400, 4, 0.05), (500, 5, 0.03)]:
            candidates.append({"family": "xgboost", "n_estimators": n_est, "max_depth": depth, "learning_rate": lr})

    best_label = None
    best_model = None
    best_val_rmse = float("inf")
    best_family = "ridge"
    best_hyperparams = {}

    for cand in candidates:
        family = cand["family"]
        if family == "ridge":
            model = Ridge(alpha=cand["alpha"])
            label = "ridge_alpha=" + str(cand["alpha"])
            hparams = {"alpha": cand["alpha"]}
        else:
            base_xgb = XGBRegressor(
                n_estimators=int(cand["n_estimators"]),
                max_depth=int(cand["max_depth"]),
                learning_rate=float(cand["learning_rate"]),
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=0,
                n_jobs=1,
            )
            model = MultiOutputRegressor(base_xgb)
            label = "xgb_n" + str(cand["n_estimators"]) + "_d" + str(cand["max_depth"])
            hparams = {
                "n_estimators": int(cand["n_estimators"]),
                "max_depth": int(cand["max_depth"]),
                "learning_rate": float(cand["learning_rate"]),
            }

        model.fit(x_train_scaled, z_train)
        z_val_pred = model.predict(x_val_scaled)
        j_val_pred_transformed = pca.inverse_transform(z_val_pred)
        j_val_pred = invert_curve_transform(j_val_pred_transformed, transform)
        j_val_pred = np.maximum(j_val_pred, 0.0)
        val_rmse = float(np.sqrt(mean_squared_error(j_val_true, j_val_pred)))
        if val_rmse < best_val_rmse:
            best_label = label
            best_model = model
            best_val_rmse = val_rmse
            best_family = family
            best_hyperparams = hparams

    if best_model is None:
        raise ValueError("Backbone fit failed")

    print("backbone selected:", best_label, "val_rmse=", best_val_rmse)

    return {
        "scaler": scaler,
        "pca": pca,
        "transform": transform,
        "curve_transform": transform,
        "ridge": best_model,
        "ridge_alpha": float(best_hyperparams.get("alpha", 0.0)),
        "backbone_family": str(best_family),
        "backbone_hyperparams": best_hyperparams,
        "val_rmse": float(best_val_rmse),
    }


def predict_curve_backbone(backbone_model, x_raw):
    # Predict base J(t) and return the scaled tabular features used by downstream stages.
    x_scaled = backbone_model["scaler"].transform(x_raw).astype(np.float32)
    z_pred = backbone_model["ridge"].predict(x_scaled)
    j_pred_transformed = backbone_model["pca"].inverse_transform(z_pred)

    transform = backbone_model.get("transform")
    if transform is None:
        transform = backbone_model.get("curve_transform")
    if transform is None:
        transform = {"kind": "identity"}

    j_pred = invert_curve_transform(j_pred_transformed, transform)
    j_pred = np.maximum(j_pred, 0.0).astype(np.float32)
    return j_pred, x_scaled
