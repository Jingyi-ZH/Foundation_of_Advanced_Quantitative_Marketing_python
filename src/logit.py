# logit.py
# This module provides estimators for discrete choice models, including multinomial logit,
# nested logit, and latent class models, along with evaluation metrics.

import numpy as np
from scipy.optimize import minimize
import warnings
from scipy.optimize import basinhopping
from statsmodels.sandbox.regression.gmm import IV2SLS


class Metrics:
    """This class provides various metrics for model evaluation."""

    @staticmethod
    def rho2(model, X, y):
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
    def AIC(model):
        """Compute Akaike Information Criterion (AIC)."""
        k = model.nparams
        return 2 * k + 2 * model.fun

    @staticmethod
    def BIC(model):
        """Compute Bayesian Information Criterion (BIC)."""
        k = model.nparams
        n = model.observations
        return np.log(n) * k + 2 * model.fun

    @staticmethod
    def HQC(model):
        """Compute Hannan-Quinn Criterion (HQC)."""
        k = model.nparams
        n = model.observations
        return 2 * k * np.log(np.log(n)) + 2 * model.fun

    @staticmethod
    def CAIC(model):
        """Compute Corrected Akaike Information Criterion (CAIC)."""
        k = model.nparams
        n = model.observations
        return 2 * k + 2 * model.fun + 2 * k * (k + 1) / (n - k - 1)


class LogitModel:
    """
    Multinomial logit model estimator.

    Attributes:
        coef_ : ndarray
            Estimated coefficients (K,).
        fun : float
            Minimized negative log-likelihood value.
        observations : int
            Number of observations (N).
        nalternatives : int
            Number of alternatives (p).
        nfeatures : int
            Number of attributes (K).
        nparams : int
            Number of parameters.
    """

    def __init__(self):
        self.coef_ = None
        self.fun = None
        self.observations = None
        self.nalternatives = None
        self.nfeatures = None
        self.nparams = None

    def _negative_log_likelihood(self, params, X, y):
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
        # One-hot encode y for likelihood calculation
        y_one_hot = np.zeros((self.observations, self.nalternatives))
        y_one_hot[np.arange(self.observations), y] = 1

        # Compute utilities (N, p)
        utilities = np.einsum("npk,k->np", X, params)  # Efficient for 3D arrays

        # Stabilized softmax to compute probabilities
        utilities -= np.max(utilities, axis=1, keepdims=True)
        exp_utilities = np.exp(utilities)
        sum_exp = np.sum(exp_utilities, axis=1, keepdims=True)
        probabilities = exp_utilities / sum_exp

        # Log-likelihood (with epsilon to avoid log(0))
        likelihoods = np.sum(probabilities * y_one_hot, axis=1)
        log_likelihood = np.sum(np.log(likelihoods + 1e-10))

        return -log_likelihood

    def fit(self, X, y):
        """
        Fit the multinomial logit model.

        Parameters:
            X : ndarray
                Design matrix of shape (N, p, K), where:
                - N is the number of observations,
                - p is the number of alternatives,
                - K is the number of attributes.
            y : ndarray
                Choice indicators of shape (N,), with integer values from 0 to p-1.

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

        # Input validation
        if X.ndim != 3:
            raise ValueError(
                "X must be a 3D array (N observations x p alternatives x K attributes)"
            )
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations (rows)")
        if np.any(y < 0) or np.any(y >= p):
            raise ValueError(f"y values must be integers between 0 and {p-1}")

        # Initial parameter guess (small values to avoid overflow)
        initial_params = np.zeros(K)

        # Optimize negative log-likelihood
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            args=(X, y),
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
        coef_ : ndarray
            Estimated coefficients (K-1 + ngroups,).
        fun : float
            Minimized negative log-likelihood value.
        observations : int
            Number of observations (N).
        nalternatives : int
            Number of alternatives (p).
        nfeatures : int
            Number of attributes (K-1, excluding group ID).
        ngroups : int
            Number of groups.
        nparams : int
            Number of parameters.
    """

    def __init__(self):
        self.coef_ = None
        self.fun = None
        self.observations = None
        self.nalternatives = None
        self.nfeatures = None
        self.ngroups = None
        self.nparams = None

    def _negative_log_likelihood(self, params, X, y):
        """
        Compute the negative log-likelihood for optimization.

        Parameters:
            params : ndarray
                Current parameter values.
            X : ndarray
                Design matrix (N, p, K).
            y : ndarray
                Choice indicators (N,).

        Returns:
            float
                Negative log-likelihood value.
        """
        # One-hot encode y for likelihood calculation
        y_one_hot = np.zeros((self.observations, self.nalternatives))
        y_one_hot[np.arange(self.observations), y] = 1

        lambda_ = params[-self.ngroups :]
        rhos = 1 / (1 + np.exp(-lambda_))  # Ensure 0 < rho < 1
        feature_params = params[: self.nfeatures]
        group_ids = X[:, :, -1].astype(int)  # (N, p)
        X_features = X[:, :, : self.nfeatures]
        utilities = np.einsum("npk,k->np", X_features, feature_params)
        exp_utilities = np.exp(utilities / rhos[group_ids])  # (N, p)

        # Compute nest utilities
        nest_utility = []
        for i in range(self.ngroups):
            group_mask = group_ids == i
            sum_exp = np.sum(exp_utilities * group_mask, axis=1, keepdims=True)
            nest_utility.append(sum_exp)
        nest_utility = np.column_stack(nest_utility)  # (N, ngroups)

        nest_utility_rho = nest_utility**rhos  # (N, ngroups)
        prob_nest = nest_utility_rho / np.sum(
            nest_utility_rho, axis=1, keepdims=True
        )  # (N, ngroups)
        prob_nest_expanded = prob_nest[
            np.arange(self.observations)[:, None], group_ids
        ]  # (N, p)

        nest_u_expanded = nest_utility[
            np.arange(self.observations)[:, None], group_ids
        ]  # (N, p)
        prob_item = exp_utilities / nest_u_expanded  # (N, p)

        probabilities = prob_nest_expanded * prob_item  # (N, p)

        # Negative log-likelihood (with regularization to avoid log(0))
        log_probabilities = np.log(probabilities + 1e-10)
        nll = -np.sum(y_one_hot * log_probabilities) + 1e-4 * np.dot(rhos, rhos)

        return nll

    def fit(self, X, y, ngroups, nsingle_class=None):
        """
        Fit the nested logit model.

        Parameters:
            X : ndarray
                Design matrix of shape (N, p, K), where:
                - N is the number of observations,
                - p is the number of alternatives,
                - K is the number of attributes (last column is group ID).
            y : ndarray
                Choice indicators of shape (N,), with integer values from 0 to p-1.
            ngroups : int
                Number of unique groups.
            nsingle_class : int, optional
                Number of single-class groups (for parameter adjustment).

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
        if nsingle_class is not None:
            self.nparams = self.nfeatures + ngroups - nsingle_class
        else:
            self.nparams = self.nfeatures + ngroups

        # Input validation
        if X.ndim != 3:
            raise ValueError(
                "X must be a 3D array (N observations x p alternatives x K attributes)"
            )
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations (rows)")
        if np.any(y < 0) or np.any(y >= p):
            raise ValueError(f"y values must be integers between 0 and {p-1}")
        if np.any(X[:, :, -1] < 0) or np.any(X[:, :, -1] >= ngroups):
            raise ValueError(f"Group IDs must be integers between 0 and {ngroups-1}")
        if ngroups <= 0:
            raise ValueError("ngroups must be a positive integer")

        # Initial parameter guess
        initial_params = np.zeros(self.nfeatures + ngroups)

        # Optimize negative log-likelihood
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            args=(X, y),
            method="L-BFGS-B",
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        self.coef_ = result.x
        self.fun = result.fun

        lambda_ = self.coef_[-self.ngroups :]
        rhos = np.round(1 / (1 + np.exp(-lambda_)), 2)

        print(f"Optimized parameters for features:\n{self.coef_[:self.nfeatures]}")
        print(
            f"Optimized parameters for within-group correlations (rho = 0.5 will be assigned to groups with only one product):\n{rhos}"
        )
        print(f"Maximized LL: {-self.fun}")

        return self


class LatentClassModel:
    """
    Latent class model estimator (discrete heterogeneity distribution).

    This model assumes a finite number of latent classes, each with its own parameters
    for a multinomial logit choice model. Class probabilities are estimated using
    softmax with one class fixed for identification.

    Attributes:
        coef_ : ndarray
            Estimated coefficients [beta_class1, ..., beta_classC, delta1, ..., delta{C-1}].
        fun : float
            Minimized negative log-likelihood value.
        indiv_likelihood : ndarray
            Class-conditional likelihoods (num_individuals, nclasses).
        observations : int
            Number of observations (N).
        nalternatives : int
            Number of alternatives (p).
        nfeatures : int
            Number of attributes (K).
        nclasses : int
            Number of latent classes.
        nparams : int
            Number of parameters.
    """

    def __init__(self):
        self.coef_ = None
        self.fun = None
        self.indiv_likelihood = None
        self.observations = None
        self.nalternatives = None
        self.nfeatures = None
        self.nclasses = None
        self.nparams = None

    def _class_likelihood(self, params, X, y):
        """
        Compute the likelihood for a single class and individual's observations.

        Parameters:
            params : ndarray
                Class-specific parameters (K,).
            X : ndarray
                Design matrix for individual (num_obs, p, K).
            y : ndarray
                Choices for individual (num_obs,).

        Returns:
            float
                Likelihood value.
        """
        num_obs = X.shape[0]
        y_one_hot = np.zeros((num_obs, self.nalternatives))
        y_one_hot[np.arange(num_obs), y] = 1

        utilities = np.einsum("npk,k->np", X, params)
        utilities -= np.max(utilities, axis=1, keepdims=True)
        exp_utilities = np.exp(utilities)
        sum_exp = np.sum(exp_utilities, axis=1, keepdims=True)
        probabilities = exp_utilities / sum_exp

        likelihoods_per_obs = np.sum(probabilities * y_one_hot, axis=1)
        indiv_likelihood = np.prod(likelihoods_per_obs)

        return indiv_likelihood

    def _negative_log_likelihood(self, full_params, X, y, indiv_id):
        """
        Compute the negative log-likelihood for optimization.

        Parameters:
            full_params : ndarray
                Flattened parameters.
            X : ndarray
                Design matrix (N, p, K).
            y : ndarray
                Choices (N,).
            indiv_id : ndarray
                Individual IDs (N,).

        Returns:
            tuple
                (negative log-likelihood, class-conditional likelihoods).
        """
        full_params = np.append(full_params, 0)
        deltas = full_params[-self.nclasses :]
        weights = np.exp(deltas) / np.sum(np.exp(deltas))

        indiv_likelihoods = []
        negll = 0.0

        for i in np.unique(indiv_id):
            mask = indiv_id == i
            X_indiv = X[mask]
            y_indiv = y[mask]

            marginal_likelihood = 0.0
            class_likelihoods = []

            for c in range(self.nclasses):
                beta_start = c * self.nfeatures
                class_params = full_params[beta_start : beta_start + self.nfeatures]
                class_lik = self._class_likelihood(class_params, X_indiv, y_indiv)
                class_likelihoods.append(class_lik)
                marginal_likelihood += class_lik * weights[c]

            indiv_likelihoods.append(class_likelihoods)
            negll += -np.log(marginal_likelihood)

        indiv_likelihoods = np.array(indiv_likelihoods)
        # np.sum(np.log(np.sum(lls * rhos,axis=1)))
        return (negll, indiv_likelihoods)

    def fit(self, X, y, nclasses, indiv_id):
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

        # Input validation
        if X.ndim != 3:
            raise ValueError(
                "X must be a 3D array (N observations x p alternatives x K attributes)"
            )
        if y.ndim != 1 or indiv_id.ndim != 1:
            raise ValueError("y and indiv_id must be 1D arrays")
        if X.shape[0] != y.shape[0] or X.shape[0] != indiv_id.shape[0]:
            raise ValueError(
                "X, y, and indiv_id must have the same number of observations (rows)"
            )
        if np.any(y < 0) or np.any(y >= p):
            raise ValueError(f"y values must be integers between 0 and {p-1}")
        if len(np.unique(indiv_id)) < 1:
            raise ValueError("indiv_id must have at least one unique individual")

        # Initial parameter guess
        initial_params = np.ones(self.nparams)

        # Callback to monitor
        def callback(x, f, accept):
            print(f"Current NLL: {f}")

        result = basinhopping(
            lambda params, X, y, indiv_id: self._negative_log_likelihood(
                params, X, y, indiv_id
            )[0],
            initial_params,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "args": (X, y, indiv_id),
                # "bounds": bounds  # If applicable
            },
            niter=10,  # Number of basin-hopping iterations; increase for better search
            stepsize=0.5,  # Initial step size for perturbations
            callback=callback,
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        self.coef_ = result.x
        self.fun = result.fun
        _, self.indiv_likelihood = self._negative_log_likelihood(
            result.x, X, y, indiv_id
        )

        deltas = np.append(self.coef_[-self.nclasses + 1 :], 0)
        weights = np.exp(deltas) / np.sum(np.exp(deltas))
        print(
            f"Optimized parameters for features:\n{self.coef_[:self.nfeatures * self.nclasses]}"
        )
        print(f"Optimized parameters for class weights (pie):\n{np.round(weights, 2)}")
        print(f"Maximized LL: {-self.fun}")

        return self

    def segment_assign(self):
        """
        Assign a segment to each individual based on their likelihoods.
        """
        if self.indiv_likelihood is None:
            raise RuntimeError("Model is not fitted yet.")
        full_params = np.append(self.coef_, 0)
        deltas = full_params[-self.nclasses :]
        weights = np.exp(deltas) / np.sum(np.exp(deltas))
        posterior = self.indiv_likelihood * weights
        posterior /= np.sum(posterior, axis=1, keepdims=True)

        # Assign each individual to the segment with the highest posterior
        return np.argmax(posterior, axis=1)


import numpy as np
from scipy.optimize import basinhopping
from typing import Optional, Union


class RandomCoefficientsModel:
    """
    Random Coefficients Model for choice data.
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
        self.nfactors: Optional[int] = None

    def _class_likelihood(
        self, params: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Compute the likelihood for a single draw of parameters and individual's observations.

        Parameters:
            params : ndarray
                A specific draw of parameters (K,).
            X : ndarray
                Design matrix for individual (num_obs, p, K).
            y : ndarray
                Choices for individual (num_obs,).

        Returns:
            float
                Likelihood value.
        """
        num_obs = X.shape[0]
        y_one_hot = np.zeros((num_obs, self.nalternatives))
        y_one_hot[np.arange(num_obs), y] = 1

        utilities = np.einsum("npk,k->np", X, params)
        utilities -= np.max(utilities, axis=1, keepdims=True)
        exp_utilities = np.exp(utilities)
        sum_exp = np.sum(exp_utilities, axis=1, keepdims=True)
        probabilities = exp_utilities / sum_exp

        likelihoods_per_obs = np.sum(probabilities * y_one_hot, axis=1)
        indiv_likelihood = np.prod(likelihoods_per_obs)

        return indiv_likelihood

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
            sigma = (cov @ draw_matrix.T) @ (draw_matrix @ cov.T)
        # calculate standard variation from sigma
        std = np.sqrt(np.diag(sigma))

        return param_matrix, means, std, sigma

    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        draw_matrix: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        indiv_id: np.ndarray,
        method: Optional[str],
    ) -> float:
        param_matrix, _, _, _ = self._decompose_params(params, draw_matrix, method)
        negll = 0.0
        for i in np.unique(indiv_id):
            mask = indiv_id == i
            X_indiv = X[mask]
            y_indiv = y[mask]
            indiv_likelihood = np.array(
                [self._class_likelihood(row, X_indiv, y_indiv) for row in param_matrix]
            )
            negll -= np.log(np.mean(indiv_likelihood))
        return negll

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indiv_id: np.ndarray,
        method: Optional[str] = None,
        draws: Optional[int] = 10,
        homo_covariates: Optional[np.ndarray] = None,
        niteration: Optional[int] = 5,
        optimizer: Optional[str] = "L-BFGS-B",
        step_size: Optional[float] = 0.5,
        nfactors: Optional[int] = None,
    ) -> "RandomCoefficientsModel":
        """
        Fit the Random Coefficients Model.

        Parameters:
            X : ndarray
                Design matrix (N observations x p alternatives x K attributes).
            y : ndarray
                Choices (N observations,).
            indiv_id : ndarray
                Individual identifiers (N observations,).
            method : str, optional
                Model method: None (default) or 'factor-analytic'.
            draws : int, optional
                Number of draws for simulation (default: 10).
            niteration : int, optional
                Number of iterations for optimization (default: 5).
            optimizer : str, optional
                Optimization algorithm (default: 'L-BFGS-B').
            step_size : float, optional
                Step size for optimization (default: 0.5).

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
            homo_covariates = np.array([0] * K)
        self.homo_covariates = homo_covariates
        if nfactors is not None:
            self.nfactors = nfactors
            if method != "factor-analytic":
                raise ValueError(
                    "nfactors can only be used with factor-analytic method"
                )
            else:
                print("Using factor-analytic method with", nfactors, "factors.")

        # Input validation
        if X.ndim != 3:
            raise ValueError(
                "X must be a 3D array (N observations x p alternatives x K attributes)"
            )
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations (rows)")
        if np.any(y < 0) or np.any(y >= p):
            raise ValueError(f"y values must be integers between 0 and {p-1}")
        if method not in [None, "factor-analytic"]:
            raise ValueError(f"Unknown method: {method}")
        if method == "factor-analytic" and K < 2:
            raise ValueError("Factor-analytic method requires at least 2 attributes.")
        if draws % 2 != 0:
            draws = draws + 1
            warnings.warn(
                f"The number of draws should be even: taking {draws} draws", UserWarning
            )
        if (
            np.sum(homo_covariates) > 0
            and np.where(homo_covariates == 1)[0][0]
            < np.where(homo_covariates == 0)[0][-1]
        ):
            raise ValueError(
                "You must put homogeneous covariates to the end of the data."
            )
        nhomo = len(np.where(homo_covariates == 1)[0])
        nhetero = K - nhomo
        self.nhomo = nhomo

        def callback(x: np.ndarray, f: float, accept: bool) -> None:
            print(f"Current NLL: {f}")

        if method != "factor-analytic":
            self.nparams = int(nhetero * (nhetero + 1) / 2 + K)
            initial_params = np.ones(self.nparams)
            draw_matrix = np.hstack(
                (
                    np.random.normal(size=(self.draws // 2, nhetero)),
                    np.zeros((self.draws // 2, nhomo)),
                )
            )
            draw_matrix = np.vstack((draw_matrix, -draw_matrix))

            result = basinhopping(
                self._negative_log_likelihood,
                initial_params,
                minimizer_kwargs={
                    "method": optimizer,
                    "args": (draw_matrix, X, y, indiv_id, method),
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
                        "args": (draw_matrix, X, y, indiv_id, method),
                    },
                    niter=niteration,
                    stepsize=step_size,
                    callback=callback,
                )
                cur_BIC = np.log(N) * cur_nparam + 2 * result.fun
                print("BIC: ", cur_BIC)
            else:
                for F in range(1, self.nfeatures + 1):
                    cur_nparam = int(nhetero * F - (F - 1) + K)
                    initial_params = np.ones(cur_nparam)
                    draw_matrix = np.random.normal(size=(self.draws // 2, F))
                    draw_matrix = np.vstack((draw_matrix, -draw_matrix))

                    result = basinhopping(
                        self._negative_log_likelihood,
                        initial_params,
                        minimizer_kwargs={
                            "method": optimizer,
                            "args": (draw_matrix, X, y, indiv_id, method),
                        },
                        niter=niteration,
                        stepsize=step_size,
                        callback=callback,
                    )
                    cur_BIC = np.log(self.observations) * cur_nparam + 2 * result.fun
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

        # Check optimization success
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


class BLP:
    def __init__(self, X, Z, shares, outside, nfeatures, D, t, p, nu=None):
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
            outside option shares
        nfeatures : int
            number of features
        D : int
            number of draws
        t : int
            number of markets
        p : int
            number of products in each market
        nu : ndarray, optional
            sampling error term (D x nfeatures)
        """
        self.X = X
        self.Z = Z
        self.shares = shares
        self.outside = outside
        self.nfeatures = nfeatures
        self.D = D
        self.t = t
        self.p = p
        if nu is None:
            nu = np.random.normal(size=(D // 2, nfeatures))
            nu = np.vstack([nu, -nu])
        self.nu = nu

    def TSLS(self, delta):
        model_2sls = IV2SLS(delta, self.X, self.Z)
        result = model_2sls.fit()
        res_2sls = delta - result.predict()
        loss = (
            res_2sls.T @ self.Z @ np.linalg.inv(self.Z.T @ self.Z) @ self.Z.T @ res_2sls
        )
        return loss, result.params

    def contraction_mapping(self, gamma):
        sigma = np.zeros((self.nfeatures, self.nfeatures))
        idx = np.triu_indices(self.nfeatures)
        sigma[idx] = gamma

        draws = self.nu @ sigma
        init_delta = np.log(self.shares) - np.log(self.outside)
        delta = init_delta.copy()

        for _ in range(10000):
            exp_util = np.exp(delta.T + draws @ self.X.T)
            denom = np.tile(
                (np.sum(exp_util.reshape(self.D, self.t, self.p), axis=2) + 1).reshape(
                    self.D, self.t, 1
                ),
                (1, 1, self.p),
            ).reshape(self.D, self.t * self.p)

            approx_shares = (exp_util / denom).mean(axis=0)
            if np.max(np.abs(approx_shares - self.shares)) < 1e-6:
                return delta
            delta = delta + np.log(self.shares) - np.log(approx_shares)

        raise ValueError("Contraction mapping did not converge")

    def outer(self, gamma):
        delta = self.contraction_mapping(gamma)
        loss, coef = self.TSLS(delta)
        return loss, coef

    def fit(self, method="Powell", disp=True):
        gamma_init = np.ones(self.nfeatures * (self.nfeatures + 1) // 2) / 10
        res = minimize(
            lambda gamma: self.outer(gamma)[0],
            gamma_init,
            method=method,
            options={"disp": disp},
        )
        self.gamma_hat = res.x
        self.loss, self.beta_hat = self.outer(self.gamma_hat)
        return self.gamma_hat, self.beta_hat

    def summary(self):
        print("Estimated gamma:", self.gamma_hat)
        print("Estimated beta:", self.beta_hat)
        print("Final loss:", self.loss)
