# logit_boost.py
# This module provides estimators for discrete choice models, including multinomial logit,
# nested logit, latent class, random coefficients models, and BLP along with evaluation metrics.

import numpy as np
from scipy.optimize import minimize, basinhopping
from typing import Optional
from scipy.special import logsumexp
import warnings
from statsmodels.sandbox.regression.gmm import IV2SLS
import numba as nb


class Metrics:
    """This class provides various metrics for model evaluation."""

    @staticmethod
    def rho2(model, X: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Compute McFadden's pseudo R^2."""
        if model.__class__.__name__ not in ["LogitModel"]:
            warnings.warn(
                "Invalid model type. Only LogitModel and NestedLogitModel are supported.",
                UserWarning,
            )
            return None
        ll_null = model._negative_log_likelihood(np.zeros(model.nparams), X, y)
        ll_full = model.fun
        return 1 - (ll_full / ll_null) if ll_null != 0 else 0

    @staticmethod
    def AIC(model) -> float:
        """Compute Akaike Information Criterion (AIC)."""
        k = model.nparams
        return 2 * k + 2 * model.fun

    @staticmethod
    def BIC(model) -> float:
        """Compute Bayesian Information Criterion (BIC)."""
        k = model.nparams
        n = model.observations
        return np.log(n) * k + 2 * model.fun

    @staticmethod
    def HQC(model) -> float:
        """Compute Hannan-Quinn Criterion (HQC)."""
        k = model.nparams
        n = model.observations
        return 2 * k * np.log(np.log(n)) + 2 * model.fun

    @staticmethod
    def CAIC(model) -> float:
        """Compute Corrected Akaike Information Criterion (CAIC)."""
        k = model.nparams
        n = model.observations
        return 2 * k + 2 * model.fun + 2 * k * (k + 1) / (n - k - 1)


class LogitModel:
    """
    Multinomial logit model estimator.

    Attributes:
        coef_ : Optional[np.ndarray]
            Estimated coefficients (K,).
        fun : Optional[float]
            Minimized negative log-likelihood value.
        observations : Optional[int]
            Number of observations (N).
        nalternatives : Optional[int]
            Number of alternatives (p).
        nfeatures : Optional[int]
            Number of attributes (K).
        nparams : Optional[int]
            Number of parameters.
    """

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.fun: Optional[float] = None
        self.observations: Optional[int] = None
        self.nalternatives: Optional[int] = None
        self.nfeatures: Optional[int] = None
        self.nparams: Optional[int] = None

    def _negative_log_likelihood(
        self, params: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Compute the negative log-likelihood for optimization.

        Parameters:
            params : ndarray
                Current parameter values (K,).
            X : ndarray
                Design matrix (N, p, K).
            y : ndarray
                Choice indicators (N,).

        Returns:
            float
                Negative log-likelihood value.
        """
        utilities = np.einsum("npk,k->np", X, params)
        selected_utilities = utilities[np.arange(self.observations), y]
        log_denoms = logsumexp(utilities, axis=1)
        log_likelihood = np.sum(selected_utilities - log_denoms)
        return -log_likelihood

    def _jac(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian (gradient) of the negative log-likelihood.

        Parameters:
            params : ndarray
                Current parameter values (K,).
            X : ndarray
                Design matrix (N, p, K).
            y : ndarray
                Choice indicators (N,).

        Returns:
            ndarray
                Gradient vector (K,).
        """
        utilities = np.einsum("npk,k->np", X, params)
        utilities -= np.max(utilities, axis=1, keepdims=True)
        exp_utilities = np.exp(utilities)
        sum_exp = np.sum(exp_utilities, axis=1, keepdims=True)
        probabilities = exp_utilities / sum_exp
        observed = np.sum(X[np.arange(self.observations), y, :], axis=0)
        predicted = np.einsum("np,npk->k", probabilities, X)
        return predicted - observed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogitModel":
        """
        Fit the multinomial logit model.

        Parameters:
            X : ndarray
                Design matrix of shape (N, p, K).
            y : ndarray
                Choice indicators of shape (N,).

        Returns:
            self : LogitModel
                The fitted model instance.

        Raises:
            ValueError: If input shapes or values are invalid.
            RuntimeError: If optimization fails.
        """
        N, p, K = X.shape
        self.observations = N
        self.nalternatives = p
        self.nfeatures = K
        self.nparams = K

        if X.ndim != 3:
            raise ValueError("X must be a 3D array (N x p x K)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if np.any(y < 0) or np.any(y >= p):
            raise ValueError(f"y values must be between 0 and {p-1}")

        initial_params = np.zeros(K)
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            args=(X, y),
            jac=self._jac,
            method="L-BFGS-B",
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        self.coef_ = result.x
        self.fun = result.fun
        print(f"Optimized parameters:\n{self.coef_}")
        print(f"Maximized LL: {-self.fun}")
        return self


class NestedLogitModel:
    """
    Nested logit model estimator.

    Attributes:
        coef_ : Optional[ndarray]
            Estimated coefficients.
        fun : Optional[float]
            Minimized negative log-likelihood value.
        observations : Optional[int]
            Number of observations (N).
        nalternatives : Optional[int]
            Number of alternatives (p).
        nfeatures : Optional[int]
            Number of attributes (K-1, excluding group ID).
        ngroups : Optional[int]
            Number of groups.
        nparams : Optional[int]
            Number of parameters.
        multi_groups : Optional[ndarray]
            Indices of groups with multiple alternatives.
    """

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.fun: Optional[float] = None
        self.observations: Optional[int] = None
        self.nalternatives: Optional[int] = None
        self.nfeatures: Optional[int] = None
        self.ngroups: Optional[int] = None
        self.nparams: Optional[int] = None
        self.multi_groups: Optional[np.ndarray] = None

    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        X_features: np.ndarray,
        group_ids: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Compute the negative log-likelihood for optimization.

        Parameters:
            params : ndarray
                Current parameter values.
            X_features : ndarray
                Feature matrix (N, p, K-1).
            group_ids : ndarray
                Group identifiers (N, p).
            y : ndarray
                Choice indicators (N,).

        Returns:
            float
                Negative log-likelihood value.
        """
        betas = params[: self.nfeatures]
        rhos = np.ones(self.ngroups)
        rhos[self.multi_groups] = params[self.nfeatures :]
        utilities = np.einsum("npk,k->np", X_features, betas)
        N = self.observations
        scaled_utilities = utilities / rhos[group_ids]
        inclusive_value = np.full((N, self.ngroups), -np.inf)
        for g in range(self.ngroups):
            mask = group_ids == g
            masked_scaled = np.where(mask, scaled_utilities, -np.inf)
            inclusive_value[:, g] = logsumexp(masked_scaled, axis=1)
        log_nest_terms = rhos * inclusive_value
        log_denom = logsumexp(log_nest_terms, axis=1)
        selected_groups = group_ids[np.arange(N), y]
        rho_selected = rhos[selected_groups]
        inc_selected = inclusive_value[np.arange(N), selected_groups]
        selected_utilities = utilities[np.arange(N), y]
        log_probs = (
            selected_utilities / rho_selected
            + (rho_selected - 1) * inc_selected
            - log_denom
        )
        ll = np.sum(log_probs)
        return -ll

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ngroups: int,
        optimizer: str = "L-BFGS-B",
    ) -> "NestedLogitModel":
        """
        Fit the nested logit model.

        Parameters:
            X : ndarray
                Design matrix of shape (N, p, K).
            y : ndarray
                Choice indicators of shape (N,).
            ngroups : int
                Number of unique groups.
            optimizer : str, optional
                Optimization algorithm (default: 'L-BFGS-B').

        Returns:
            self : NestedLogitModel
                The fitted model instance.

        Raises:
            ValueError: If input shapes or values are invalid.
            RuntimeError: If optimization fails.
        """
        N, p, K = X.shape
        self.observations = N
        self.nalternatives = p
        self.nfeatures = K - 1
        self.ngroups = ngroups

        if X.ndim != 3:
            raise ValueError("X must be a 3D array (N x p x K)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if np.any(y < 0) or np.any(y >= p):
            raise ValueError(f"y values must be between 0 and {p-1}")
        if ngroups <= 0:
            raise ValueError("ngroups must be positive")

        group_ids = X[:, :, -1].astype(int)
        if np.any(group_ids < 0) or np.any(group_ids >= ngroups):
            raise ValueError(f"Group IDs must be between 0 and {ngroups-1}")

        X_features = X[:, :, : self.nfeatures]

        # Detect multi-alternative groups
        first_obs_groups = group_ids[0, :]
        group_sizes = np.array([np.sum(first_obs_groups == g) for g in range(ngroups)])
        self.multi_groups = np.where(group_sizes > 1)[0]
        num_multi = len(self.multi_groups)
        self.nparams = self.nfeatures + num_multi

        # Initial parameters and bounds
        initial_params = np.concatenate(
            (np.zeros(self.nfeatures), np.full(num_multi, 0.5))
        )
        bounds = [(None, None)] * self.nfeatures + [(1e-3, 1 - 1e-3)] * num_multi

        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            args=(X_features, group_ids, y),
            method=optimizer,
            bounds=bounds,
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        self.coef_ = result.x
        self.fun = result.fun

        # Reconstruct rhos
        rhos = np.ones(ngroups)
        rhos[self.multi_groups] = result.x[self.nfeatures :]

        print(f"Optimized parameters for features:\n{result.x[:self.nfeatures]}")
        print(
            f"Optimized parameters for within-group correlations (rho fixed to 1 for single-alternative groups):\n{rhos}"
        )
        print(f"Maximized LL: {-self.fun}")
        return self


class LatentClassModel:
    """
    Latent class model estimator (discrete heterogeneity distribution).

    Attributes:
        coef_ : Optional[ndarray]
            Estimated coefficients [beta_class1, ..., beta_classC, delta1, ..., delta{C-1}].
        fun : Optional[float]
            Minimized negative log-likelihood value.
        indiv_likelihood : Optional[ndarray]
            Class-conditional likelihoods (num_individuals, nclasses).
        observations : Optional[int]
            Number of observations (N).
        nalternatives : Optional[int]
            Number of alternatives (p).
        nfeatures : Optional[int]
            Number of attributes (K).
        nclasses : Optional[int]
            Number of latent classes.
        nparams : Optional[int]
            Number of parameters.
        indiv_data : Optional[list]
            Precomputed per-individual data.
    """

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.fun: Optional[float] = None
        self.indiv_likelihood: Optional[np.ndarray] = None
        self.observations: Optional[int] = None
        self.nalternatives: Optional[int] = None
        self.nfeatures: Optional[int] = None
        self.nclasses: Optional[int] = None
        self.nparams: Optional[int] = None
        self.indiv_data: Optional[list[tuple[np.ndarray, np.ndarray]]] = None

    def _negative_log_likelihood(
        self, params: np.ndarray, indiv_id: np.ndarray
    ) -> float:
        """
        Compute the negative log-likelihood for optimization.

        Parameters:
            params : ndarray
                Flattened parameters.
            indiv_id : ndarray
                Individual IDs (N,).

        Returns:
            float
                Negative log-likelihood.
        """
        betas = params[: self.nfeatures * self.nclasses].reshape(
            self.nclasses, self.nfeatures
        )
        deltas = np.append(params[self.nfeatures * self.nclasses :], 0.0)
        log_weights = deltas - logsumexp(deltas)
        negll = 0.0
        indiv_liks = []
        for X_i, y_i in self.indiv_data:
            utilities = np.einsum("npk,ck->npc", X_i, betas)
            selected_u = utilities[np.arange(X_i.shape[0]), y_i, :]
            log_denom = logsumexp(utilities, axis=1)
            log_lik_obs = selected_u - log_denom
            log_lik_class = np.sum(log_lik_obs, axis=0)
            log_marginal = logsumexp(log_lik_class + log_weights)
            negll -= log_marginal
            indiv_liks.append(np.exp(log_lik_class))
        self.indiv_likelihood = np.array(indiv_liks)
        return negll

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        nclasses: int,
        indiv_id: np.ndarray,
        niter: int = 10,
        stepsize: float = 0.5,
    ) -> "LatentClassModel":
        """
        Fit the latent class model.

        Parameters:
            X : ndarray
                Design matrix (N, p, K).
            y : ndarray
                Choices (N,).
            nclasses : int
                Number of latent classes.
            indiv_id : ndarray
                Individual IDs (N,).
            niter : int, optional
                Number of basin-hopping iterations (default: 10).
            stepsize : float, optional
                Step size for perturbations (default: 0.5).

        Returns:
            self : LatentClassModel
                The fitted model instance.

        Raises:
            ValueError: If input shapes or values are invalid.
            RuntimeError: If optimization fails.
        """
        N, p, K = X.shape
        self.observations = N
        self.nalternatives = p
        self.nfeatures = K
        self.nclasses = nclasses
        self.nparams = K * nclasses + nclasses - 1

        if X.ndim != 3:
            raise ValueError("X must be a 3D array (N x p x K)")
        if y.ndim != 1 or indiv_id.ndim != 1:
            raise ValueError("y and indiv_id must be 1D arrays")
        if X.shape[0] != y.shape[0] or X.shape[0] != indiv_id.shape[0]:
            raise ValueError("X, y, and indiv_id must have the same number of rows")
        if np.any(y < 0) or np.any(y >= p):
            raise ValueError(f"y values must be between 0 and {p-1}")
        if len(np.unique(indiv_id)) < 1:
            raise ValueError("indiv_id must have at least one unique individual")

        # Precompute per-individual data
        unique_indiv = np.unique(indiv_id)
        self.indiv_data = []
        for i in unique_indiv:
            mask = indiv_id == i
            self.indiv_data.append((X[mask], y[mask]))

        initial_params = np.ones(self.nparams)

        def callback(x: np.ndarray, f: float, accept: bool) -> None:
            print(f"Current NLL: {f}")

        result = basinhopping(
            self._negative_log_likelihood,
            initial_params,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "args": (indiv_id,),
            },
            niter=niter,
            stepsize=stepsize,
            callback=callback,
        )

        if not result.lowest_optimization_result.success:
            raise RuntimeError(
                f"Optimization failed: {result.lowest_optimization_result.message}"
            )

        self.coef_ = result.x
        self.fun = result.fun

        deltas = np.append(self.coef_[self.nfeatures * self.nclasses :], 0.0)
        weights = np.exp(deltas) / np.sum(np.exp(deltas))
        print(
            f"Optimized parameters for features:\n{self.coef_[:self.nfeatures * self.nclasses]}"
        )
        print(f"Optimized parameters for class weights:\n{np.round(weights, 2)}")
        print(f"Maximized LL: {-self.fun}")
        return self

    def segment_assign(self) -> np.ndarray:
        """
        Assign a segment to each individual based on their likelihoods.

        Returns:
            ndarray
                Segment assignments.
        """
        if self.indiv_likelihood is None:
            raise RuntimeError("Model is not fitted yet.")
        deltas = np.append(self.coef_[self.nfeatures * self.nclasses :], 0.0)
        weights = np.exp(deltas) / np.sum(np.exp(deltas))
        posterior = self.indiv_likelihood * weights
        posterior /= np.sum(posterior, axis=1, keepdims=True)
        return np.argmax(posterior, axis=1)


class RandomCoefficientsModel:
    """
    Random Coefficients Model for choice data.

    Attributes:
        coef_ : Optional[np.ndarray]
            Estimated coefficients.
        fun : Optional[float]
            Minimized negative log-likelihood value.
        observations : Optional[int]
            Number of observations (N).
        nalternatives : Optional[int]
            Number of alternatives (p).
        nfeatures : Optional[int]
            Number of attributes (K).
        draws : Optional[int]
            Number of draws.
        nparams : Optional[int]
            Number of parameters.
        BIC : float
            Bayesian Information Criterion.
        means : Optional[np.ndarray]
            Means of coefficients.
        std : Optional[np.ndarray]
            Standard deviations of coefficients.
        sigma : Optional[np.ndarray]
            Covariance matrix.
        homo_covariates : Optional[np.ndarray]
            Homogeneous covariates indicators.
        indiv_data : Optional[list]
            Precomputed per-individual data.
    """

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.fun: Optional[float] = None
        self.observations: Optional[int] = None
        self.nalternatives: Optional[int] = None
        self.nfeatures: Optional[int] = None
        self.draws: Optional[int] = None
        self.nparams: Optional[int] = None
        self.BIC: float = np.inf
        self.means: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.homo_covariates: Optional[np.ndarray] = None
        self.indiv_data: Optional[list[tuple[np.ndarray, np.ndarray]]] = None
        self.nfactors = None

    def _decompose_params(
        self,
        params: np.ndarray,
        draw_matrix: np.ndarray,
        method: Optional[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        K = self.nfeatures
        if method != "factor-analytic":
            means = params[:K]
            cov = np.append(params[K:], np.zeros(K * (K + 1) // 2 - len(params[K:])))
            gamma = np.zeros((K, K), dtype=cov.dtype)
            idx = np.triu_indices(K)
            gamma[idx] = cov
            param_matrix = means + draw_matrix @ gamma
            sigma = gamma.T @ gamma
        else:
            F = draw_matrix.shape[1]
            means = params[:K]
            cov = np.append(params[K:], np.zeros(F * K - len(params[K:]))).reshape(K, F)
            param_matrix = means + draw_matrix @ cov.T
            sigma = cov @ cov.T
        std = np.sqrt(np.diag(sigma))
        return param_matrix, means, std, sigma

    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        draw_matrix: np.ndarray,
        method: Optional[str],
    ) -> float:
        param_matrix, _, _, _ = self._decompose_params(params, draw_matrix, method)
        negll = 0.0
        for X_indiv, y_one_hot in self.indiv_data:
            utilities = np.einsum("dk,npk->dnp", param_matrix, X_indiv)
            max_u = np.max(utilities, axis=2, keepdims=True)
            exp_u = np.exp(utilities - max_u)
            sum_exp = np.sum(exp_u, axis=2, keepdims=True)
            probs = exp_u / sum_exp
            lik_per_obs = np.sum(probs * y_one_hot[None, :, :], axis=2)
            log_lik_per_obs = np.log(lik_per_obs + 1e-300)
            log_lik_per_draw = np.sum(log_lik_per_obs, axis=1)
            log_avg_lik = logsumexp(log_lik_per_draw) - np.log(self.draws)
            negll -= log_avg_lik
        return negll

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indiv_id: np.ndarray,
        method: Optional[str] = None,
        draws: int = 10,
        homo_covariates: Optional[np.ndarray] = None,
        niteration: int = 5,
        optimizer: str = "L-BFGS-B",  # "Powell"
        step_size: float = 0.5,
        nfactors: Optional[int] = None,
    ) -> "RandomCoefficientsModel":
        """
        Fit the Random Coefficients Model.

        Parameters:
            X : ndarray
                Design matrix (N x p x K).
            y : ndarray
                Choices (N,).
            indiv_id : ndarray
                Individual identifiers (N,).
            method : str, optional
                Model method: None (default) or 'factor-analytic'.
            draws : int, optional
                Number of draws (default: 10).
            homo_covariates : ndarray, optional
                Homogeneous covariates indicators.
            niteration : int, optional
                Number of iterations (default: 5).
            optimizer : str, optional
                Optimization algorithm (default: 'Powell').
            step_size : float, optional
                Step size (default: 0.5).

        Returns:
            RandomCoefficientsModel
                Fitted model instance.
        """
        self.draws = draws
        N, p, K = X.shape
        self.observations = N
        self.nalternatives = p
        self.nfeatures = K
        if homo_covariates is None:
            homo_covariates = np.zeros(K, dtype=int)
        self.homo_covariates = homo_covariates
        if nfactors is not None:
            self.nfactors = nfactors
            if method != "factor-analytic":
                raise ValueError(
                    "nfactors can only be used with factor-analytic method"
                )
            else:
                print("Using factor-analytic method with", nfactors, "factors.")

        if X.ndim != 3:
            raise ValueError("X must be a 3D array (N x p x K)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if np.any(y < 0) or np.any(y >= p):
            raise ValueError(f"y values must be between 0 and {p-1}")
        if method not in [None, "factor-analytic"]:
            raise ValueError(f"Unknown method: {method}")
        if method == "factor-analytic" and K < 2:
            raise ValueError("Factor-analytic requires at least 2 attributes")
        if draws % 2 != 0:
            draws += 1
            warnings.warn(f"Draws should be even: using {draws}", UserWarning)
        if np.sum(homo_covariates) > 0 and np.min(
            np.where(homo_covariates == 1)[0]
        ) < np.max(np.where(homo_covariates == 0)[0]):
            raise ValueError("Homogeneous covariates must be at the end")
        nhomo = np.sum(homo_covariates)
        nhetero = K - nhomo

        # Precompute per-individual data
        unique_indiv = np.unique(indiv_id)
        self.indiv_data = []
        for i in unique_indiv:
            mask = indiv_id == i
            X_i = X[mask]
            y_i = y[mask]
            num_obs = len(y_i)
            y_one_hot = np.zeros((num_obs, p))
            y_one_hot[np.arange(num_obs), y_i] = 1
            self.indiv_data.append((X_i, y_one_hot))

        def callback(x: np.ndarray, f: float, accept: bool) -> None:
            print(f"Current NLL: {f}")

        if method != "factor-analytic":
            self.nparams = int(nhetero * (nhetero + 1) / 2 + K)
            initial_params = np.ones(self.nparams)
            draw_matrix = np.hstack(
                (
                    np.random.normal(size=(draws // 2, nhetero)),
                    np.zeros((draws // 2, nhomo)),
                )
            )
            draw_matrix = np.vstack((draw_matrix, -draw_matrix))

            result = basinhopping(
                self._negative_log_likelihood,
                initial_params,
                minimizer_kwargs={
                    "method": optimizer,
                    "args": (draw_matrix, method),
                },
                niter=niteration,
                stepsize=step_size,
                callback=callback,
            )
            self.coef_ = result.x
            self.fun = result.fun
            _, self.means, self.std, self.sigma = self._decompose_params(
                result.x, draw_matrix, method
            )
        else:
            if self.nfactors is not None:
                F = self.nfactors
                cur_nparam = int(nhetero * F - (F - 1) + K)
                initial_params = np.ones(cur_nparam)
                draw_matrix = np.random.normal(size=(draws // 2, F))
                draw_matrix = np.vstack((draw_matrix, -draw_matrix))

                result = basinhopping(
                    self._negative_log_likelihood,
                    initial_params,
                    minimizer_kwargs={
                        "method": optimizer,
                        "args": (draw_matrix, method),
                    },
                    niter=niteration,
                    stepsize=step_size,
                    callback=callback,
                )
                cur_BIC = np.log(N) * cur_nparam + 2 * result.fun
                print("BIC: ", cur_BIC)
            else:
                for F in range(1, K + 1):
                    cur_nparam = int(nhetero * F - (F - 1) + K)
                    initial_params = np.ones(cur_nparam)
                    draw_matrix = np.random.normal(size=(draws // 2, F))
                    draw_matrix = np.vstack((draw_matrix, -draw_matrix))

                    result = basinhopping(
                        self._negative_log_likelihood,
                        initial_params,
                        minimizer_kwargs={
                            "method": optimizer,
                            "args": (draw_matrix, method),
                        },
                        niter=niteration,
                        stepsize=step_size,
                        callback=callback,
                    )
                    cur_BIC = np.log(N) * cur_nparam + 2 * result.fun
                    print("current BIC: ", cur_BIC)
                    if cur_BIC > self.BIC:
                        break
            self.nparams = cur_nparam
            self.coef_ = result.x
            self.fun = result.fun
            self.BIC = cur_BIC
            self.nfactor = F
            _, self.means, self.std, self.sigma = self._decompose_params(
                result.x, draw_matrix, method
            )

        if not result.lowest_optimization_result.success:
            raise RuntimeError(
                f"Optimization failed: {result.lowest_optimization_result.message}"
            )

        print(f"Maximized LL: {-self.fun}")
        if method == "factor-analytic":
            print(f"The optimal number of factors is: {self.nfactor}")
        print(f"The means of the coefficients are: {self.means}")
        print(f"The standard deviations of the coefficients are: {self.std}")
        print(f"The covariance matrix of the coefficients is: {self.sigma}")
        return self


@nb.jit(nopython=True)
def _contraction_mapping(
    gamma,
    nu,
    X,
    shares,
    log_shares,
    log_outside,
    D,
    t,
    p,
    nfeatures,
    tol=1e-6,
    max_iter=10000,
):
    sigma = np.zeros((nfeatures, nfeatures))
    i, j = np.triu_indices(nfeatures)
    for k in range(gamma.shape[0]):
        sigma[i[k], j[k]] = gamma[k]

    draws = nu @ sigma
    delta = log_shares - log_outside

    for iter_count in range(max_iter):
        # Compute utilities: delta + draws @ X.T (D, t*p)
        utils = delta + draws @ X.T

        # Reshape to (D, t, p) for per-market max subtraction
        ## Subtract 200 per market per draw to avoid exp overflow
        utils_reshaped = utils.reshape(D, t, p)
        utils_reshaped -= 200

        exp_util_reshaped = np.exp(utils_reshaped)  # Now safe from inf

        # Market sums: sum(exp) + exp(-max) for outside (but since normalized, adjust)
        # Remember to subtract 200 for outside utility as well
        market_sums = np.sum(exp_util_reshaped, axis=2) + np.exp(-200)  # (D, t)

        denom = market_sums[:, :, np.newaxis]  # (D, t, 1)
        approx_shares_reshaped = exp_util_reshaped / denom  # (D, t, p)
        reshaped = approx_shares_reshaped.reshape(D, t * p)
        approx_shares = np.sum(reshaped, axis=0) / D  # (t*p,)

        delta += log_shares - np.log(approx_shares)

        if np.max(np.abs(approx_shares - shares)) < tol:
            return delta

    raise ValueError("Contraction mapping did not converge")


class BLP:
    def __init__(self, X, Z, shares, outside, D, t, p, nu=None):
        """
        initialize BLP model
        ----------
        X : ndarray
            characteristics matrix
        Z : ndarray
            instruments matrix
        shares : ndarray
            market shares vector
        outside : ndarray
            outside option shares (scalar or per-market array of length t)
        D : int
            number of draws
        t : int
            number of markets
        p : int
            number of products in each market
        nu : ndarray, optional
            sampling error term (D x nfeatures)
        """
        self.X = X.astype(np.float64)
        self.Z = Z.astype(np.float64)
        self.shares = shares.astype(np.float64)
        self.outside = np.asarray(outside).astype(np.float64)
        self.nfeatures = X.shape[1]
        self.D = D
        self.t = t
        self.p = p
        if nu is None:
            nu = np.random.normal(size=(D // 2, self.nfeatures))
            nu = np.vstack([nu, -nu])
        self.nu = nu.astype(np.float64)

        # Precompute projection matrix and first stage for TSLS
        self.Proj = self.Z @ np.linalg.inv(self.Z.T @ self.Z) @ self.Z.T
        self.FirstStage = self.Proj @ self.X
        self.tslspre = np.linalg.inv(self.FirstStage.T @ self.FirstStage)

        # Precompute logs, handle broadcasting for outside
        self.log_shares = np.log(self.shares)
        self.log_outside = np.log(self.outside)
        if self.log_outside.ndim == 0:  # scalar
            self.log_outside = np.full_like(self.log_shares, self.log_outside)
        elif len(self.log_outside) == self.t:
            self.log_outside = np.repeat(self.log_outside, self.p)

    def TSLS(self, delta):
        params = self.tslspre @ (self.FirstStage.T @ delta)
        res_2sls = delta - (self.X @ params)
        # model_2sls = IV2SLS(delta, self.X, self.Z)
        # result = model_2sls.fit()
        # res_2sls = delta - result.predict()
        loss = res_2sls.T @ self.Proj @ res_2sls
        return loss, params

    def contraction_mapping(self, gamma):
        return _contraction_mapping(
            gamma.astype(np.float64),
            self.nu,
            self.X,
            self.shares,
            self.log_shares,
            self.log_outside,
            self.D,
            self.t,
            self.p,
            self.nfeatures,
        )

    def outer(self, gamma):
        delta = self.contraction_mapping(gamma)
        loss, coef = self.TSLS(delta)
        return loss, coef

    def fit(self, method="L-BFGS-B", disp=True):
        gamma_init = np.ones(self.nfeatures * (self.nfeatures + 1) // 2) / 10
        res = minimize(
            lambda gamma: self.outer(gamma)[0],
            gamma_init,
            method=method,
            options={"disp": disp, "maxiter": 1000},
        )
        self.gamma_hat = res.x
        self.loss, self.beta_hat = self.outer(self.gamma_hat)
        return self.gamma_hat, self.beta_hat

    def summary(self):
        print("Estimated gamma:", self.gamma_hat)
        print("Estimated beta:", self.beta_hat)
        print("Final loss:", self.loss)
