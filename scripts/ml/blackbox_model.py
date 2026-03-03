import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def choose_transform(values):
    # Use log10 when positive values span a wide range.
    # Zero values are allowed; they are clipped by apply_transform.
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return {"kind": "identity"}

    positive = finite[finite > 0.0]
    if positive.size < 2:
        return {"kind": "identity"}

    min_value = max(float(np.min(positive)), 1e-30)
    max_value = float(np.max(positive))
    spread_ratio = max_value / min_value

    if spread_ratio < 20.0:
        return {"kind": "identity"}

    eps = max(min_value * 1e-3, 1e-30)
    transform = {}
    transform["kind"] = "log10"
    transform["eps"] = eps
    return transform


def choose_scalar_log_transform(values):
    # Scalars are trained in log-space to handle wide dynamic ranges.
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        transform = {}
        transform["kind"] = "identity"
        return transform

    positive = finite[finite > 0.0]
    if positive.size == 0:
        # Fallback for unexpected non-positive data.
        transform = {}
        transform["kind"] = "identity"
        return transform

    min_value = float(np.min(positive))
    eps = max(min_value * 1e-3, 1e-30)
    transform = {}
    transform["kind"] = "log10"
    transform["eps"] = eps
    return transform


def apply_transform(values, transform):
    # Apply target transform before fitting.
    if transform["kind"] == "identity":
        return values
    eps = transform["eps"]
    clipped = np.maximum(values, eps)
    return np.log10(clipped)


def invert_transform(values, transform):
    # Convert predictions back to original units.
    if transform["kind"] == "identity":
        return values
    # Clamp log-space predictions before exponentiation to avoid inf values.
    # 10^300 is already far beyond any expected scale in this project.
    clipped = np.clip(values, -300.0, 300.0)
    return np.power(10.0, clipped)


def build_scalar_candidates(use_xgboost, alpha_grid):
    # Build candidate models for scalar targets.
    candidates = []

    ridge_params = []
    for alpha in alpha_grid:
        entry = {}
        entry["alpha"] = alpha
        ridge_params.append(entry)
    candidates.append({"family": "ridge", "params": ridge_params})

    if use_xgboost:
        xgb_params = []
        xgb_params.append({"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05})
        xgb_params.append({"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05})
        candidates.append({"family": "xgboost", "params": xgb_params})
    return candidates


def build_curve_candidates(use_xgboost, alpha_grid):
    # Build candidate models for PCA curve coordinates.
    candidates = []

    ridge_params = []
    for alpha in alpha_grid:
        entry = {}
        entry["alpha"] = alpha
        ridge_params.append(entry)
    candidates.append({"family": "ridge", "params": ridge_params})

    if use_xgboost:
        xgb_params = []
        xgb_params.append({"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05})
        xgb_params.append({"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05})
        candidates.append({"family": "xgboost_multioutput", "params": xgb_params})
    return candidates


def make_scalar_estimator(family, params):
    # Construct a scalar estimator from family + params.
    if family == "ridge":
        return Ridge(alpha=float(params["alpha"]))
    if family == "xgboost":
        return XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=0,
            n_jobs=1,
        )
    raise ValueError("Unknown scalar model family: " + str(family))


def make_curve_estimator(family, params):
    # Construct a curve estimator in PCA space.
    if family == "ridge":
        return Ridge(alpha=float(params["alpha"]))
    if family == "xgboost_multioutput":
        base = XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=0,
            n_jobs=1,
        )
        return MultiOutputRegressor(base)
    raise ValueError("Unknown curve model family: " + str(family))


def fit_scalar_models(X_train, y_train, X_val, y_val, target_names, use_xgboost, alpha_grid, min_rows_for_xgboost):
    # Fit one model per scalar target and pick the best on validation RMSE.
    # We train each scalar target separately because scales/availability differ.
    models = []
    for target_idx in range(y_train.shape[1]):
        target_name = target_names[target_idx]
        ytr = y_train[:, target_idx]
        yva = y_val[:, target_idx]
        mask_train = np.isfinite(ytr)
        mask_val = np.isfinite(yva)
        train_rows = int(np.sum(mask_train))
        val_rows = int(np.sum(mask_val))

        # Avoid complex models when a target has limited supervision.
        use_xgboost_for_target = bool(use_xgboost) and (train_rows >= min_rows_for_xgboost)
        candidates = build_scalar_candidates(use_xgboost_for_target, alpha_grid)

        # If a target has no valid training values, keep a placeholder model.
        if train_rows == 0:
            model_info = {}
            model_info["target_name"] = target_name
            model_info["model_family"] = None
            model_info["transform"] = {"kind": "identity"}
            model_info["hyperparams"] = {}
            model_info["estimator"] = None
            model_info["train_rows"] = train_rows
            model_info["val_rows"] = val_rows
            models.append(model_info)
        else:
            # Keep scalar targets in log-space for stable fitting.
            transform = choose_scalar_log_transform(ytr[mask_train])
            ytr_transformed = apply_transform(ytr[mask_train], transform)

            best_rmse = float("inf")
            best_family = None
            best_params = {}
            best_estimator = None

            # Try each candidate setting and keep the lowest validation RMSE.
            for candidate in candidates:
                family = candidate["family"]
                for params in candidate["params"]:
                    estimator = make_scalar_estimator(family, params)
                    estimator.fit(X_train[mask_train], ytr_transformed)

                    if val_rows == 0:
                        # No validation rows: rank by train RMSE as fallback.
                        pred_transformed = estimator.predict(X_train[mask_train])
                        pred = invert_transform(pred_transformed, transform)
                        rmse = float(np.sqrt(mean_squared_error(ytr[mask_train], pred)))
                    else:
                        pred_transformed = estimator.predict(X_val[mask_val])
                        pred = invert_transform(pred_transformed, transform)
                        rmse = float(np.sqrt(mean_squared_error(yva[mask_val], pred)))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_family = family
                        best_params = dict(params)
                        best_estimator = estimator

            model_info = {}
            model_info["target_name"] = target_name
            model_info["model_family"] = best_family
            model_info["transform"] = transform
            model_info["hyperparams"] = best_params
            model_info["estimator"] = best_estimator
            model_info["train_rows"] = train_rows
            model_info["val_rows"] = val_rows
            models.append(model_info)

    return models


def predict_scalar_models(models, X):
    # Predict all scalar targets for one feature matrix.
    target_count = len(models)
    y_pred = np.full((X.shape[0], target_count), np.nan, dtype=float)

    for i in range(target_count):
        model_info = models[i]
        estimator = model_info["estimator"]
        transform = model_info["transform"]
        if estimator is not None:
            pred_transformed = estimator.predict(X)
            y_pred[:, i] = invert_transform(pred_transformed, transform)
    return y_pred


def fit_curve_model(X_train, J_train, X_val, J_val, curve_components, use_xgboost, alpha_grid):
    # Train curve model: transform J, compress with PCA, regress PCA scores.
    # This reduces curve dimensionality so the regressor has a manageable target.
    transform = choose_transform(J_train)
    J_train_transformed = apply_transform(J_train, transform)

    max_components = min(curve_components, J_train_transformed.shape[0], J_train_transformed.shape[1])
    pca = PCA(n_components=max_components)
    Z_train = pca.fit_transform(J_train_transformed)

    candidates = build_curve_candidates(use_xgboost, alpha_grid)
    best_rmse = float("inf")
    best_family = None
    best_params = {}
    best_estimator = None

    for candidate in candidates:
        family = candidate["family"]
        for params in candidate["params"]:
            estimator = make_curve_estimator(family, params)
            estimator.fit(X_train, Z_train)

            Z_val_pred = estimator.predict(X_val)
            J_val_pred_transformed = pca.inverse_transform(Z_val_pred)
            J_val_pred = invert_transform(J_val_pred_transformed, transform)
            rmse = float(np.sqrt(mean_squared_error(J_val, J_val_pred)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_family = family
                best_params = dict(params)
                best_estimator = estimator

    curve_config = {}
    curve_config["model_family"] = best_family
    for key, value in best_params.items():
        curve_config[key] = value
    return pca, transform, best_estimator, curve_config


def predict_curve_model(pca, curve_transform, estimator, X):
    # Predict J(t) by decoding predicted PCA coordinates.
    Z_pred = estimator.predict(X)
    J_pred_transformed = pca.inverse_transform(Z_pred)
    J_pred = invert_transform(J_pred_transformed, curve_transform)
    return np.maximum(J_pred, 0.0)
