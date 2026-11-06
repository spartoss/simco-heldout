"""Model zoo for the supervised DOS benchmark.

Every model is a Pipeline fitted on the outer-train women only:
  median imputation -> drop constant columns -> StandardScaler -> [TopK screen] -> model
with its hyperparameters chosen by GridSearchCV on a GroupKFold(5) over the *training* women
(no visit of a test woman ever reaches tuning). Grids are fixed here, before any result was seen.
``TopK`` keeps the k features with the largest univariate F statistic, recomputed inside every fit.

Families (the notebook groups them the same way):
  null       mean
  linear     ridge, lasso, enet, pls, pcr, svr_lin
  kernel/NN-free nonlinear   svr_rbf, krr_rbf, knn, gp
  bagging    rf, et, bag_tree, bag_ridge
  boosting   xgb_p43 (paper's XGBoost config, untuned), xgb, hgb, gbr, ada
  neural     mlp (sklearn); torch DNN ensembles live in run_dnn.py, DeepSets on cells in run_deepsets.py
  ensemble   stack (super learner: base models' inner-OOF predictions -> non-negative linear blend)
"""
import json

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.feature_selection import VarianceThreshold, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

P43 = "/scratch/users/bmerwan/simco_work/onset_test/param_grid_files/params_43.json"


class TopK(BaseEstimator, TransformerMixin):
    def __init__(self, k=200):
        self.k = k

    def fit(self, X, y):
        f, _ = f_regression(X, y)
        f = np.nan_to_num(f)
        k = X.shape[1] if self.k in (None, "all") else min(int(self.k), X.shape[1])
        self.idx_ = np.sort(np.argsort(-f)[:k])
        return self

    def transform(self, X):
        return X[:, self.idx_]


class XGB(RegressorMixin, BaseEstimator):
    """sklearn-native wrapper: xgboost 1.7.6 predates sklearn 1.6's estimator tags, so a bare
    XGBRegressor cannot sit in a Pipeline / GridSearchCV here."""

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0, colsample_bytree=1.0,
                 gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, min_child_weight=1.0, random_state=0):
        self.n_estimators, self.max_depth, self.learning_rate = n_estimators, max_depth, learning_rate
        self.subsample, self.colsample_bytree, self.gamma = subsample, colsample_bytree, gamma
        self.reg_alpha, self.reg_lambda, self.min_child_weight = reg_alpha, reg_lambda, min_child_weight
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBRegressor
        self.m_ = XGBRegressor(**self.get_params(), verbosity=0, n_jobs=1).fit(X, y)
        self.feature_importances_ = self.m_.feature_importances_
        return self

    def predict(self, X):
        return self.m_.predict(X)


class PLS1(PLSRegression):
    """PLSRegression returning 1-D predictions (sklearn's is (n, 1))."""

    def predict(self, X, copy=True):
        return np.ravel(super().predict(X, copy=copy))


def ys(est):
    """Standardise y (DOS sd ~31 d) for scale-sensitive learners."""
    return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())


def pipe(model, topk=False):
    steps = [("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(1e-10)),
             ("sc", StandardScaler())]
    if topk:
        steps.append(("sel", TopK()))
    return Pipeline(steps + [("m", model)])


def logs(a, b, n):
    return list(np.logspace(a, b, n))


def zoo(seed=0):
    """name -> (family, estimator, grid). Grids are in Pipeline parameter syntax."""
    p43 = json.load(open(P43))
    Z = {
        "mean": ("null", pipe(DummyRegressor()), {}),
        # --- linear
        "ridge": ("linear", pipe(Ridge()), {"m__alpha": logs(0, 5, 11)}),
        "lasso": ("linear", pipe(Lasso(max_iter=20000)), {"m__alpha": logs(-0.5, 1.5, 9)}),
        "enet": ("linear", pipe(ElasticNet(max_iter=20000)),
                 {"m__alpha": logs(-0.5, 2, 8), "m__l1_ratio": [0.1, 0.5, 0.9]}),
        "pls": ("linear", pipe(PLS1(scale=False)), {"m__n_components": [1, 2, 3, 4, 6, 8, 12]}),
        "pcr": ("linear", pipe(Pipeline([("pca", PCA(random_state=seed)), ("r", Ridge())])),
                {"m__pca__n_components": [5, 10, 20, 40], "m__r__alpha": [1, 10, 100, 1000]}),
        "svr_lin": ("linear", pipe(ys(SVR(kernel="linear")), topk=True),
                    {"sel__k": [100, 500, "all"], "m__regressor__C": logs(-4, -1, 4)}),
        # --- kernel / instance-based
        "svr_rbf": ("kernel", pipe(ys(SVR(kernel="rbf")), topk=True),
                    {"sel__k": [50, 200, 1000, "all"], "m__regressor__C": [0.3, 1, 3, 10],
                     "m__regressor__epsilon": [0.1, 0.5]}),
        "krr_rbf": ("kernel", pipe(ys(KernelRidge(kernel="rbf")), topk=True),
                    {"sel__k": [50, 200, 1000, "all"], "m__regressor__alpha": [0.03, 0.3, 3]}),
        "knn": ("kernel", pipe(KNeighborsRegressor(weights="distance"), topk=True),
                {"sel__k": [20, 100, 500], "m__n_neighbors": [5, 10, 20]}),
        "gp": ("kernel", pipe(GaussianProcessRegressor(
            ConstantKernel(1.0) * RBF(10.0, (1e-1, 1e4)) + WhiteKernel(1.0, (1e-3, 10)),
            normalize_y=True, n_restarts_optimizer=2, random_state=seed), topk=True),
            {"sel__k": [20, 100, 500]}),
        # --- bagging
        "rf": ("bagging", pipe(RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=1)),
               {"m__max_features": ["sqrt", 0.1, 0.33], "m__min_samples_leaf": [1, 3, 6]}),
        "et": ("bagging", pipe(ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=1)),
               {"m__max_features": ["sqrt", 0.1, 0.33], "m__min_samples_leaf": [1, 3, 6]}),
        "bag_tree": ("bagging", pipe(BaggingRegressor(DecisionTreeRegressor(), n_estimators=300,
                                                      random_state=seed)),
                     {"m__max_samples": [0.5, 1.0], "m__max_features": [0.2, 1.0],
                      "m__estimator__max_depth": [4, None]}),
        "bag_ridge": ("bagging", pipe(BaggingRegressor(Ridge(), n_estimators=200, random_state=seed)),
                      {"m__max_features": [0.05, 0.2], "m__estimator__alpha": [10, 100, 1000]}),
        # --- boosting
        "xgb_p43": ("boosting", pipe(XGB(**p43, random_state=seed)), {}),
        "xgb": ("boosting", pipe(XGB(n_estimators=400, learning_rate=0.03, subsample=0.7, random_state=seed)),
                {"m__max_depth": [2, 4, 6], "m__colsample_bytree": [0.05, 0.3],
                 "m__min_child_weight": [1, 5], "m__reg_lambda": [1, 10]}),
        "hgb": ("boosting", pipe(HistGradientBoostingRegressor(max_iter=300, early_stopping=False,
                                                               random_state=seed)),
                {"m__learning_rate": [0.03, 0.1], "m__max_depth": [2, 4], "m__max_features": [0.1, 0.5],
                 "m__l2_regularization": [0, 10]}),
        "gbr": ("boosting", pipe(GradientBoostingRegressor(n_estimators=500, learning_rate=0.02,
                                                           subsample=0.7, random_state=seed)),
                {"m__max_depth": [2, 3], "m__max_features": ["sqrt", 0.1], "m__loss": ["squared_error", "huber"]}),
        "ada": ("boosting", pipe(AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=300,
                                                   random_state=seed)),
                {"m__learning_rate": [0.03, 0.3], "m__loss": ["linear", "square"]}),
        # --- neural (sklearn)
        "mlp": ("neural", pipe(ys(MLPRegressor(max_iter=2000, early_stopping=False, random_state=seed,
                                               learning_rate_init=1e-3)), topk=True),
                {"sel__k": [100, 1000], "m__regressor__hidden_layer_sizes": [(256, 64), (512, 256, 64)],
                 "m__regressor__alpha": [1, 10, 100]}),
    }
    return Z


STACK_BASE = ["ridge", "enet", "pls", "svr_rbf", "knn", "rf", "et", "xgb", "hgb"]
# mid-grid settings for the super learner's base models (fixed, so the stack is one level of nesting)
STACK_FIX = {
    "ridge": {"m__alpha": 1000}, "enet": {"m__alpha": 3.0, "m__l1_ratio": 0.5}, "pls": {"m__n_components": 3},
    "svr_rbf": {"sel__k": 200, "m__regressor__C": 1}, "knn": {"sel__k": 100, "m__n_neighbors": 10},
    "rf": {"m__max_features": 0.1, "m__min_samples_leaf": 3}, "et": {"m__max_features": 0.1, "m__min_samples_leaf": 3},
    "xgb": {"m__max_depth": 2, "m__colsample_bytree": 0.3, "m__min_child_weight": 5, "m__reg_lambda": 10},
    "hgb": {"m__learning_rate": 0.03, "m__max_depth": 2, "m__max_features": 0.1, "m__l2_regularization": 10},
}


def stack(inner_cv, seed=0):
    """Super learner. inner_cv = list of GroupKFold (train, val) index pairs on the outer-train rows."""
    Z = zoo(seed)
    base = [(n, Z[n][1].set_params(**STACK_FIX[n])) for n in STACK_BASE]
    return StackingRegressor(base, final_estimator=LinearRegression(positive=True), cv=inner_cv,
                             n_jobs=1, passthrough=False)
