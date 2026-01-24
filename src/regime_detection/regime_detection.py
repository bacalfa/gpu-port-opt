"""
Market Regime Detection Module
==============================

GPU-accelerated market regime detection using multiple methodologies:
- Hidden Markov Models (HMM)
- Bayesian Changepoint Detection
- Clustering-based regime identification
- Volatility regime detection

Features:
- Multiple regime detection algorithms
- GPU acceleration with cuML when available
- Probabilistic regime assignments
- Regime transition analysis
- Visualization and interpretation tools

Author: Bruno Abreu Calfa
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# GPU acceleration with automatic fallback
try:
    import cudf
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA

    GPU_AVAILABLE = True
    print("✓ GPU acceleration enabled (cuDF/cuML)")
except ImportError:
    GPU_AVAILABLE = False
    cudf = pd
    print("⚠ GPU not available, using CPU (pandas/sklearn)")

# Standard ML libraries (always available)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# HMM library
try:
    from hmmlearn import hmm

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠ hmmlearn not available. Install with: pip install hmmlearn")

# Bayesian libraries
try:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pt = None
    print("⚠ PyMC not available. Install with: pip install pymc")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class GPURegimeAccelerator:
    """Utility class for GPU/CPU operations in regime detection"""

    @staticmethod
    def to_gpu(df: pd.DataFrame):
        """Convert pandas DataFrame to cuDF if GPU available"""
        if GPU_AVAILABLE:
            try:
                return cudf.from_pandas(df)
            except Exception as e:
                logger.warning(f"GPU conversion failed: {e}. Using CPU.")
                return df
        return df

    @staticmethod
    def to_cpu(df):
        """Convert cuDF DataFrame to pandas"""
        if GPU_AVAILABLE and isinstance(df, cudf.DataFrame):
            return df.to_pandas()
        return df

    @staticmethod
    def get_kmeans(n_clusters: int, random_state: int = 42):
        """Get KMeans (GPU or CPU)"""
        if GPU_AVAILABLE:
            return cuKMeans(n_clusters=n_clusters, random_state=random_state)
        else:
            return KMeans(n_clusters=n_clusters, random_state=random_state)

    @staticmethod
    def get_pca(n_components: int):
        """Get PCA (GPU or CPU)"""
        if GPU_AVAILABLE:
            return cuPCA(n_components=n_components)
        else:
            return PCA(n_components=n_components)


class VolatilityRegimeDetector:
    """
    Simple but effective volatility-based regime detection

    Identifies:
    - Low volatility regime
    - Normal volatility regime
    - High volatility regime
    - Crisis regime (extreme volatility)
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize volatility regime detector

        Args:
            use_gpu: Whether to use GPU if available
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu = GPURegimeAccelerator()

        logger.info(f"VolatilityRegimeDetector initialized (GPU: {self.use_gpu})")

    def detect_regimes(
        self, returns: pd.DataFrame, window: int = 21, quantiles: list[float] = [0.25, 0.50, 0.75]
    ) -> pd.DataFrame:
        """
        Detect volatility regimes using rolling volatility

        Args:
            returns: Return data (Date × Asset)
            window: Rolling window for volatility calculation
            quantiles: Quantile thresholds for regime boundaries

        Returns:
            DataFrame with regime labels (0=Low, 1=Normal, 2=High, 3=Crisis)
        """
        logger.info(f"Detecting volatility regimes (window={window})")

        if self.use_gpu:
            returns = self.gpu.to_gpu(returns)

        # Calculate rolling volatility (annualized)
        volatility = returns.rolling(window=window).std() * np.sqrt(252)

        # Use market-wide volatility (average across assets)
        if self.use_gpu:
            avg_volatility = volatility.mean(axis=1).to_pandas()
        else:
            avg_volatility = volatility.mean(axis=1)

        # Define regime thresholds based on quantiles
        thresholds = avg_volatility.quantile(quantiles).values

        # Assign regimes
        regimes = pd.Series(index=avg_volatility.index, dtype=int)
        regimes[:] = 1  # Default: normal volatility

        regimes[avg_volatility <= thresholds[0]] = 0  # Low volatility
        regimes[(avg_volatility > thresholds[1]) & (avg_volatility <= thresholds[2])] = (
            2  # High volatility
        )
        regimes[avg_volatility > thresholds[2]] = 3  # Crisis

        # Create output DataFrame
        result = pd.DataFrame(
            {
                "regime": regimes,
                "volatility": avg_volatility,
                "regime_name": regimes.map({0: "Low Vol", 1: "Normal", 2: "High Vol", 3: "Crisis"}),
            }
        )

        # Add regime probabilities (crisp assignments = 100% probability)
        for i in range(4):
            result[f"prob_regime_{i}"] = (regimes == i).astype(float)

        logger.info(f"✓ Detected regimes - Distribution: {regimes.value_counts().to_dict()}")

        return result


class ClusteringRegimeDetector:
    """
    Clustering-based regime detection using K-Means

    Groups market states based on:
    - Returns
    - Volatility
    - Correlation structure
    - Macro indicators (if provided)
    """

    def __init__(self, n_regimes: int = 4, use_gpu: bool = True):
        """
        Initialize clustering regime detector

        Args:
            n_regimes: Number of regimes to detect
            use_gpu: Whether to use GPU if available
        """
        self.n_regimes = n_regimes
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu = GPURegimeAccelerator()
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None

        logger.info(
            f"ClusteringRegimeDetector initialized (n_regimes={n_regimes}, GPU={self.use_gpu})"
        )

    def prepare_features(
        self, returns: pd.DataFrame, volatility_window: int = 21, correlation_window: int = 60
    ) -> pd.DataFrame:
        """
        Prepare features for clustering

        Args:
            returns: Return data
            volatility_window: Window for volatility calculation
            correlation_window: Window for correlation calculation

        Returns:
            DataFrame with regime features
        """
        logger.info("Preparing features for clustering...")

        # 1. Rolling volatility (market-wide)
        vol = returns.rolling(window=volatility_window).std() * np.sqrt(252)
        avg_vol = vol.mean(axis=1)

        # 2. Rolling correlation (average pairwise correlation)
        rolling_corr = []
        for i in range(len(returns)):
            if i < correlation_window:
                rolling_corr.append(np.nan)
            else:
                window_returns = returns.iloc[i - correlation_window : i]
                corr_matrix = window_returns.corr()
                # Average correlation (excluding diagonal)
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                avg_corr = corr_matrix.values[mask].mean()
                rolling_corr.append(avg_corr)

        avg_corr = pd.Series(rolling_corr, index=returns.index)

        # 3. Market return (equal-weighted)
        market_return = returns.mean(axis=1)

        # 4. Return dispersion (cross-sectional std)
        return_dispersion = returns.std(axis=1)

        # Combine features
        features = pd.DataFrame(
            {
                "market_return": market_return,
                "volatility": avg_vol,
                "correlation": avg_corr,
                "dispersion": return_dispersion,
            }
        )

        # Remove NaNs
        features = features.dropna()

        logger.info(f"✓ Features prepared: {features.shape}")

        return features

    def detect_regimes(
        self, returns: pd.DataFrame, use_pca: bool = True, n_components: int = 3
    ) -> pd.DataFrame:
        """
        Detect regimes using K-Means clustering

        Args:
            returns: Return data
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components

        Returns:
            DataFrame with regime assignments and probabilities
        """
        logger.info("Detecting regimes using K-Means clustering...")

        # Prepare features
        features = self.prepare_features(returns)

        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        features_scaled_df = pd.DataFrame(
            features_scaled, index=features.index, columns=features.columns
        )

        # Optional PCA
        if use_pca:
            logger.info(f"Applying PCA (n_components={n_components})")
            self.pca = self.gpu.get_pca(n_components=n_components)

            if self.use_gpu:
                features_gpu = self.gpu.to_gpu(features_scaled_df)
                features_pca = self.pca.fit_transform(features_gpu)
                features_pca = self.gpu.to_cpu(features_pca)
            else:
                features_pca = self.pca.fit_transform(features_scaled_df)

            # Explained variance
            if hasattr(self.pca, "explained_variance_ratio_"):
                var_exp = self.pca.explained_variance_ratio_
                if self.use_gpu:
                    var_exp = var_exp.to_numpy() if hasattr(var_exp, "to_numpy") else var_exp
                logger.info(f"  PCA explained variance: {var_exp}")

            X = features_pca
        else:
            X = features_scaled_df.values

        # K-Means clustering
        logger.info(f"Running K-Means (n_clusters={self.n_regimes})")
        self.kmeans = self.gpu.get_kmeans(n_clusters=self.n_regimes)

        if self.use_gpu:
            X_gpu = cudf.DataFrame(X) if not isinstance(X, cudf.DataFrame) else X
            regimes = self.kmeans.fit_predict(X_gpu)
            regimes = regimes.to_numpy() if hasattr(regimes, "to_numpy") else regimes
        else:
            regimes = self.kmeans.fit_predict(X)

        # Calculate distances to cluster centers for probabilities
        if self.use_gpu:
            distances = self.kmeans.transform(X_gpu)
            distances = distances.to_numpy() if hasattr(distances, "to_numpy") else distances
        else:
            distances = self.kmeans.transform(X)

        # Convert distances to probabilities (softmax)
        # Probability proportional to inverse distance
        inv_distances = 1.0 / (distances + 1e-10)
        probabilities = inv_distances / inv_distances.sum(axis=1, keepdims=True)

        # Create result DataFrame
        result = pd.DataFrame(index=features.index)
        result["regime"] = regimes

        # Add probabilities for each regime
        for i in range(self.n_regimes):
            result[f"prob_regime_{i}"] = probabilities[:, i]

        # Add regime names (to be assigned based on characteristics)
        regime_chars = self._characterize_regimes(features, regimes)
        result["regime_name"] = result["regime"].map(regime_chars)

        # Add original features for reference
        result = result.join(features)

        # Distribution of regimes
        regime_dist = pd.Series(regimes).value_counts().to_dict()
        logger.info(f"✓ K-Means completed - Regime distribution: {regime_dist}")

        return result

    def _characterize_regimes(self, features: pd.DataFrame, regimes: np.ndarray) -> dict[int, str]:
        """
        Characterize regimes based on their feature means

        Args:
            features: Feature DataFrame
            regimes: Regime assignments

        Returns:
            Dictionary mapping regime ID to name
        """
        regime_means = {}
        for i in range(self.n_regimes):
            mask = regimes == i
            regime_means[i] = features[mask].mean()

        # Create regime names based on characteristics
        regime_names = {}

        for i in range(self.n_regimes):
            means = regime_means[i]

            # Characterize by volatility and returns
            if means["volatility"] < features["volatility"].median():
                vol_desc = "Low Vol"
            else:
                vol_desc = "High Vol"

            return_desc = "Bull" if means["market_return"] > 0 else "Bear"

            regime_names[i] = f"{return_desc} {vol_desc}"

        return regime_names


class HMMRegimeDetector:
    """
    Hidden Markov Model (HMM) based regime detection

    Uses Gaussian HMM to identify latent market states based on
    observed returns and volatility patterns.
    """

    def __init__(self, n_regimes: int = 4, n_iter: int = 1000):
        """
        Initialize HMM regime detector

        Args:
            n_regimes: Number of hidden states
            n_iter: Maximum iterations for EM algorithm
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn not available. Install with: pip install hmmlearn")

        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.model = None
        self.scaler = StandardScaler()

        logger.info(f"HMMRegimeDetector initialized (n_regimes={n_regimes})")

    def prepare_features(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM

        Args:
            returns: Return data

        Returns:
            Feature array
        """
        # Use market return and rolling volatility
        market_return = returns.mean(axis=1)
        volatility = returns.std(axis=1)

        features = pd.DataFrame({"market_return": market_return, "volatility": volatility}).dropna()

        return features

    def detect_regimes(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regimes using Gaussian HMM

        Args:
            returns: Return data

        Returns:
            DataFrame with regime assignments and probabilities
        """
        logger.info("Detecting regimes using Gaussian HMM...")

        # Prepare features
        features = self.prepare_features(returns)

        # Standardize
        X = self.scaler.fit_transform(features)

        # Train Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=42,
            tol=1e-6,
        )

        logger.info("Fitting HMM...")
        self.model.fit(X)

        # Predict regimes (Viterbi algorithm)
        regimes = self.model.predict(X)

        # Get regime probabilities (forward-backward algorithm)
        probabilities = self.model.predict_proba(X)

        # Create result DataFrame
        result = pd.DataFrame(index=features.index)
        result["regime"] = regimes

        # Add probabilities
        for i in range(self.n_regimes):
            result[f"prob_regime_{i}"] = probabilities[:, i]

        # Characterize regimes
        regime_chars = self._characterize_regimes_hmm(features, regimes)
        result["regime_name"] = result["regime"].map(regime_chars)

        # Add features
        result = result.join(features)

        # Log results
        regime_dist = pd.Series(regimes).value_counts().to_dict()
        logger.info(f"✓ HMM completed - Regime distribution: {regime_dist}")
        logger.info(f"  Log-likelihood: {self.model.score(X):.2f}")

        return result

    def _characterize_regimes_hmm(
        self, features: pd.DataFrame, regimes: np.ndarray
    ) -> dict[int, str]:
        """Characterize HMM regimes"""
        regime_means = {}
        for i in range(self.n_regimes):
            mask = regimes == i
            regime_means[i] = features[mask].mean()

        regime_names = {}
        for i in range(self.n_regimes):
            means = regime_means[i]

            if means["market_return"] > 0 and means["volatility"] < features["volatility"].median():
                regime_names[i] = "Bull Low Vol"
            elif (
                means["market_return"] > 0
                and means["volatility"] >= features["volatility"].median()
            ):
                regime_names[i] = "Bull High Vol"
            elif (
                means["market_return"] <= 0
                and means["volatility"] < features["volatility"].median()
            ):
                regime_names[i] = "Bear Low Vol"
            else:
                regime_names[i] = "Bear High Vol"

        return regime_names


class BayesianRegimeDetector:
    """
    Bayesian changepoint detection for regime identification

    Uses PyMC to detect structural breaks and regime transitions
    in market data using Bayesian inference.
    """

    def __init__(self, n_regimes: int = 4):
        """
        Initialize Bayesian regime detector

        Args:
            n_regimes: Expected number of regimes
        """
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC not available. Install with: pip install pymc")

        self.n_regimes = n_regimes
        self.trace = None

        logger.info(f"BayesianRegimeDetector initialized (n_regimes={n_regimes})")

    @staticmethod
    def _assign_regimes_variable(
        changepoints_sorted, n_changepoints, max_changepoints: int, T: int, n_regimes: int
    ):
        """
        Assign each time point to a regime based on variable number of changepoint locations.
        Only uses the first n_changepoints changepoints, then maps segments to regimes.

        Args:
            changepoints_sorted: Sorted changepoint locations (PyTensor variable, may have more than needed)
            n_changepoints: Actual number of changepoints to use (PyTensor variable)
            max_changepoints: Maximum number of changepoints (concrete Python int)
            T: Total number of time points
            n_regimes: Number of regimes to assign to

        Returns:
            Array of regime indices for each time point with shape (T,)
        """
        # Create time indices [0, 1, 2, ..., T-1]
        time_indices = pt.arange(T, dtype="int32")
        time_indices = pt.flatten(time_indices)

        # Initialize all to segment 0 (regime 0)
        segment_idx = pt.zeros(T, dtype="int32")

        # For each potential changepoint, only use it if its index < n_changepoints
        # This way we only use the first n_changepoints changepoints
        # Use max_changepoints (concrete int) for the Python range loop
        for i in range(max_changepoints):
            # Check if this changepoint should be used (i < n_changepoints)
            use_cp = pt.lt(pt.constant(i, dtype="int32"), n_changepoints)
            cp = changepoints_sorted[i]

            # If this changepoint is used, increment segment for all subsequent time points
            segment_idx = pt.switch(
                pt.and_(use_cp, time_indices >= cp),
                pt.constant(i + 1, dtype="int32"),
                segment_idx,
            )

        # Map segments to regimes using modulo (round-robin)
        # This ensures we use all n_regimes even if we have more segments
        regime_idx = segment_idx % n_regimes

        return pt.flatten(regime_idx)

    def _detect_changepoints_simple(
        self,
        returns_clean: pd.Series,
        returns_array: np.ndarray,
        T: int,
        n_samples: int,
        n_chains: int,
    ) -> pd.DataFrame:
        """
        Simpler changepoint model with fixed number of changepoints.
        Better convergence but less flexible than variable model.

        Args:
            returns_clean: Clean return series
            returns_array: Array of returns
            T: Number of time points
            n_samples: Number of MCMC samples
            n_chains: Number of chains

        Returns:
            DataFrame with regime probabilities
        """
        n_changepoints = self.n_regimes - 1  # Fixed number
        returns_std = float(np.std(returns_array))
        returns_mean = float(np.mean(returns_array))
        min_spacing = max(5, T // 50)

        with pm.Model() as model:
            # Regime parameters
            regime_means = pm.Normal(
                "regime_means", mu=returns_mean, sigma=returns_std * 0.5, shape=self.n_regimes
            )
            regime_sds = pm.HalfNormal("regime_sds", sigma=returns_std * 0.3, shape=self.n_regimes)

            if n_changepoints > 0:
                # Continuous changepoint locations
                changepoints_continuous = pm.Uniform(
                    "changepoints_continuous",
                    lower=float(T * 0.1),
                    upper=float(T * 0.9),
                    shape=n_changepoints,
                )

                # Sort and add spacing constraint
                changepoints_sorted_cont = pm.Deterministic(
                    "changepoints_sorted", pt.sort(changepoints_continuous)
                )
                changepoints_sorted = pt.cast(pt.round(changepoints_sorted_cont), "int32")

                # Spacing constraint
                if n_changepoints > 1:
                    spacing_diffs = changepoints_sorted_cont[1:] - changepoints_sorted_cont[:-1]
                    spacing_penalty = pt.sum(
                        pt.switch(
                            spacing_diffs < min_spacing,
                            -100.0 * (min_spacing - spacing_diffs) ** 2,
                            0.0,
                        )
                    )
                    pm.Potential("spacing_constraint", spacing_penalty)

                # Assign regimes
                regime_idx = self._assign_regimes_variable(
                    changepoints_sorted,
                    pt.constant(n_changepoints, dtype="int32"),
                    n_changepoints,
                    T,
                    self.n_regimes,
                )
            else:
                regime_idx = pt.zeros(T, dtype="int32")

            # Likelihood
            obs_means = regime_means[pt.flatten(regime_idx)]
            obs_sds = regime_sds[pt.flatten(regime_idx)]
            pm.Normal(
                "likelihood",
                mu=pt.flatten(obs_means),
                sigma=pt.flatten(obs_sds),
                observed=returns_array,
            )

            # Sample
            self.trace = pm.sample(
                draws=n_samples,
                chains=n_chains,
                tune=2000,
                return_inferencedata=True,
                progressbar=True,
                target_accept=0.90,
                init="jitter+adapt_diag",
                random_seed=42,
            )

        # Extract results (similar to main method)
        if n_changepoints > 0:
            changepoint_samples = self.trace.posterior["changepoints_sorted"].values
            changepoint_means = changepoint_samples.mean(axis=(0, 1)).astype(int)
            changepoint_means = np.unique(changepoint_means)
            T_clean = len(returns_clean)
            changepoint_means = changepoint_means[
                (changepoint_means >= int(T_clean * 0.1)) & (changepoint_means < int(T_clean * 0.9))
            ]

            regimes = np.zeros(len(returns_clean), dtype=int)
            for i, cp in enumerate(sorted(changepoint_means)):
                regimes[cp:] = (i + 1) % self.n_regimes
        else:
            regimes = np.zeros(len(returns_clean), dtype=int)

        result = pd.DataFrame(index=returns_clean.index)
        result["regime"] = regimes
        for i in range(self.n_regimes):
            result[f"prob_regime_{i}"] = (regimes == i).astype(float)
        regime_mapping = {i: f"Regime {i + 1}" for i in range(self.n_regimes)}
        result["regime_name"] = result["regime"].map(regime_mapping)

        return result

    def detect_changepoints(
        self,
        returns: pd.Series | pd.DataFrame,
        n_samples: int = 2000,
        n_chains: int = 4,
        use_simple_model: bool = False,
    ) -> pd.DataFrame:
        """
        Detect changepoints using Bayesian inference

        Args:
            returns: Return series (single asset) or DataFrame (multiple assets).
                     If DataFrame, uses market return (mean across assets).
            n_samples: Number of MCMC samples
            n_chains: Number of MCMC chains
            use_simple_model: If True, use a simpler fixed-number changepoint model
                             (better convergence, but fixed number of changepoints)

        Returns:
            DataFrame with regime probabilities
        """
        logger.info("Detecting changepoints using Bayesian inference...")
        logger.warning(
            "Bayesian changepoint detection is computationally intensive. This may take several minutes."
        )

        # Handle both Series and DataFrame input
        if isinstance(returns, pd.DataFrame):
            # For multi-asset data, use market return (mean across assets)
            logger.info(
                f"  Input is DataFrame with {len(returns.columns)} assets. Using market return."
            )
            returns_series = returns.mean(axis=1)
        else:
            returns_series = returns

        returns_clean = returns_series.dropna()
        T = len(returns_clean)
        returns_array = returns_clean.values

        # Ensure returns_array is 1D with shape (T,)
        # Convert to numpy array and ensure it's 1D
        returns_array = np.asarray(returns_array)

        if returns_array.ndim > 1:
            # If 2D, this shouldn't happen after taking mean, but handle it
            if returns_array.shape[1] > 1:
                logger.warning(
                    f"  returns_array has shape {returns_array.shape}. "
                    "Taking mean across second dimension."
                )
                returns_array = returns_array.mean(axis=1)
            else:
                returns_array = returns_array.flatten()

        # Final flatten to ensure 1D
        returns_array = returns_array.flatten()

        # Ensure length matches T (should be exact match)
        if len(returns_array) != T:
            logger.warning(
                f"  returns_array length {len(returns_array)} != T {T}. "
                "Truncating or this indicates a data alignment issue."
            )
            if len(returns_array) > T:
                returns_array = returns_array[:T]
            else:
                # This shouldn't happen, but pad if needed
                returns_array = np.pad(
                    returns_array, (0, T - len(returns_array)), mode="constant", constant_values=0.0
                )

        # Final shape check
        assert returns_array.shape == (T,), (
            f"Expected returns_array shape ({T},), got {returns_array.shape}. "
            "This indicates a shape mismatch that will cause PyMC errors."
        )

        # Option for simpler model if convergence is problematic
        if use_simple_model:
            logger.info("  Using simpler fixed-changepoint model for better convergence")
            return self._detect_changepoints_simple(
                returns_clean, returns_array, T, n_samples, n_chains
            )

        with pm.Model() as model:
            # Variable number of changepoints model with improved convergence
            # Maximum number of changepoints (reasonable upper bound)
            max_changepoints = min(T - 1, max(15, self.n_regimes * 6))

            # Prior on number of changepoints (Poisson with mean around n_regimes-1)
            # Use a more informative prior to help convergence
            lambda_cp = pm.Gamma(
                "lambda_cp",
                alpha=3.0,  # More informative
                beta=1.0 / max(1, self.n_regimes - 1),
            )
            n_changepoints_raw = pm.Poisson("n_changepoints_raw", mu=lambda_cp)

            # Truncate to reasonable bounds [0, max_changepoints]
            n_changepoints = pm.Deterministic(
                "n_changepoints", pt.clip(n_changepoints_raw, 0, max_changepoints)
            )

            # Regime means and std devs (fixed number of regimes)
            # Use more informative priors based on data characteristics
            returns_std = float(np.std(returns_array))
            returns_mean = float(np.mean(returns_array))

            regime_means = pm.Normal(
                "regime_means", mu=returns_mean, sigma=returns_std * 0.5, shape=self.n_regimes
            )
            regime_sds = pm.HalfNormal("regime_sds", sigma=returns_std * 0.3, shape=self.n_regimes)

            # Use CONTINUOUS changepoint locations (easier for NUTS to sample)
            # Then round to integers for indexing
            # This avoids the discrete parameter space that causes divergences
            changepoints_continuous = pm.Uniform(
                "changepoints_continuous",
                # lower=float(T * 0.1),
                # upper=float(T * 0.9),
                lower=float(T * 0.01),
                upper=float(T * 0.99),
                shape=max_changepoints,
            )

            # Sort the continuous changepoints
            changepoints_sorted_continuous = pm.Deterministic(
                "changepoints_sorted_continuous", pt.sort(changepoints_continuous)
            )

            # Round to integers for indexing (but keep continuous for sampling)
            changepoints_sorted = pm.Deterministic(
                "changepoints_sorted",
                pt.cast(pt.round(changepoints_sorted_continuous), "int32"),
            )

            # Add constraint: minimum spacing between changepoints
            # This prevents degenerate solutions and improves convergence
            min_spacing = max(5, T // 200)  # At least 5 time points or 0.5% of data
            if max_changepoints > 1:
                # Penalize changepoints that are too close together
                # Use a soft constraint with exponential penalty
                spacing_diffs = (
                    changepoints_sorted_continuous[1:] - changepoints_sorted_continuous[:-1]
                )
                # Penalty increases exponentially as spacing decreases below min_spacing
                spacing_penalty = pt.sum(
                    pt.switch(
                        spacing_diffs < min_spacing,
                        -100.0 * (min_spacing - spacing_diffs) ** 2,  # Quadratic penalty
                        0.0,
                    )
                )
                pm.Potential("spacing_constraint", spacing_penalty)

            # Assign regime to each observation based on changepoint locations
            # The number of actual changepoints determines how many segments we have
            # We map segments to regimes in round-robin fashion
            regime_idx = pm.Deterministic(
                "regime_idx",
                self._assign_regimes_variable(
                    changepoints_sorted, n_changepoints, max_changepoints, T, self.n_regimes
                ),
            )

            # Get the mean and std for each observation based on its regime
            # regime_idx should have shape (T,), regime_means has shape (n_regimes,)
            # Advanced indexing: regime_means[regime_idx] should give shape (T,)

            # Ensure regime_idx is explicitly 1D with shape (T,)
            regime_idx_flat = pt.flatten(regime_idx)

            # Index regime parameters - this should give shape (T,)
            obs_means = regime_means[regime_idx_flat]
            obs_sds = regime_sds[regime_idx_flat]

            # Explicitly flatten to ensure shape is (T,) not (1, T) or (T, 1)
            # The error (1, 1905) suggests a transpose issue, so flatten ensures (T,)
            obs_means = pt.flatten(obs_means)
            obs_sds = pt.flatten(obs_sds)

            # Likelihood: observations follow Normal distribution with regime-specific parameters
            likelihood = pm.Normal(
                "likelihood",
                mu=obs_means,
                sigma=obs_sds,
                observed=returns_array,
            )

            logger.info("  Sampling from posterior (this may take a while)...")
            logger.info(f"  Using {n_chains} chains, {n_samples} samples each")
            logger.info(f"  Max changepoints: {max_changepoints}, Min spacing: {min_spacing}")

            # Sample with improved settings for convergence
            # Use higher target_accept for better exploration, but may be slower
            # Consider using init="adapt_diag" for better initialization
            try:
                self.trace = pm.sample(
                    draws=n_samples,
                    chains=n_chains,
                    tune=2000,  # Longer tuning for better adaptation
                    return_inferencedata=True,
                    progressbar=True,
                    target_accept=0.9,  # Slightly lower than 0.95 for better mixing
                    init="jitter+adapt_diag",  # Better initialization
                    random_seed=42,  # For reproducibility
                )
            except Exception as e:
                logger.warning(f"  Sampling failed with error: {e}")
                logger.info("  Trying with simpler initialization...")
                # Fallback: try with simpler settings
                self.trace = pm.sample(
                    draws=n_samples,
                    chains=n_chains,
                    tune=2000,
                    return_inferencedata=True,
                    progressbar=True,
                    target_accept=0.85,
                    init="adapt_diag",
                    random_seed=42,
                )

        logger.info("✓ Bayesian inference completed")

        # Extract changepoint estimates (now variable number)
        n_changepoints_samples = self.trace.posterior["n_changepoints"].values
        n_changepoints_mean = int(np.round(n_changepoints_samples.mean()))

        logger.info(
            f"  Estimated number of changepoints: {n_changepoints_mean:.1f} "
            f"(posterior mean, range: {int(n_changepoints_samples.min())}-{int(n_changepoints_samples.max())})"
        )

        # Extract changepoint locations
        changepoint_samples_all = self.trace.posterior["changepoints_sorted"].values
        # Shape: (chains, draws, max_changepoints)

        # Get the most likely changepoint locations
        # We'll use the posterior mean of the first n_changepoints_mean changepoints
        if n_changepoints_mean > 0:
            # Extract only the first n_changepoints_mean changepoints from each sample
            changepoint_means = (
                changepoint_samples_all[:, :, :n_changepoints_mean].mean(axis=(0, 1)).astype(int)
            )

            # Remove duplicates and sort
            changepoint_means = np.unique(changepoint_means)
            # Filter to valid range (same as model bounds)
            T_clean = len(returns_clean)
            changepoint_means = changepoint_means[
                (changepoint_means >= int(T_clean * 0.1)) & (changepoint_means < int(T_clean * 0.9))
            ]

            logger.info(f"  Detected changepoints at indices: {changepoint_means}")

            # Create regimes based on changepoints
            # Map segments to regimes in round-robin fashion
            regimes = np.zeros(len(returns_clean), dtype=int)
            for i, cp in enumerate(sorted(changepoint_means)):
                regimes[cp:] = (i + 1) % self.n_regimes
        else:
            # No changepoints: all observations in regime 0
            regimes = np.zeros(len(returns_clean), dtype=int)
            logger.info("  No changepoints detected (single regime model)")

        # Create result DataFrame
        result = pd.DataFrame(index=returns_clean.index)
        result["regime"] = regimes

        # Simple probability assignment (can be improved)
        for i in range(self.n_regimes):
            result[f"prob_regime_{i}"] = (regimes == i).astype(float)

        # Add regime names - convert numpy array to Series for map operation
        regime_mapping = {i: f"Regime {i + 1}" for i in range(self.n_regimes)}
        result["regime_name"] = result["regime"].map(regime_mapping)

        regime_dist = result["regime"].value_counts().to_dict()
        logger.info(f"✓ Regimes assigned - Distribution: {regime_dist}")

        return result


class RegimeAnalyzer:
    """
    Analyze and visualize regime detection results
    """

    @staticmethod
    def analyze_regime_characteristics(
        regimes: pd.DataFrame, returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze characteristics of each regime

        Args:
            regimes: Regime DataFrame (from any detector)
            returns: Original return data

        Returns:
            DataFrame with regime statistics
        """
        logger.info("Analyzing regime characteristics...")

        # Align data
        common_idx = regimes.index.intersection(returns.index)
        regimes_aligned = regimes.loc[common_idx]
        returns_aligned = returns.loc[common_idx]

        # Market return
        market_return = returns_aligned.mean(axis=1)

        # Calculate statistics per regime
        regime_stats = []

        unique_regimes = regimes_aligned["regime"].unique()

        for regime_id in sorted(unique_regimes):
            mask = regimes_aligned["regime"] == regime_id

            # Get regime name
            regime_name = (
                regimes_aligned.loc[mask, "regime_name"].iloc[0]
                if "regime_name" in regimes_aligned.columns
                else f"Regime {regime_id}"
            )

            # Calculate statistics
            regime_returns = market_return[mask]

            stats = {
                "regime_id": regime_id,
                "regime_name": regime_name,
                "n_days": mask.sum(),
                "pct_days": (mask.sum() / len(mask)) * 100,
                "mean_return": regime_returns.mean(),
                "volatility": regime_returns.std() * np.sqrt(252),
                "sharpe_ratio": (regime_returns.mean() * 252)
                / (regime_returns.std() * np.sqrt(252)),
                "skewness": regime_returns.skew(),
                "kurtosis": regime_returns.kurtosis(),
                "max_return": regime_returns.max(),
                "min_return": regime_returns.min(),
            }

            regime_stats.append(stats)

        result = pd.DataFrame(regime_stats)

        logger.info("✓ Regime analysis completed")

        return result

    @staticmethod
    def calculate_regime_transitions(regimes: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime transition matrix

        Args:
            regimes: Regime DataFrame

        Returns:
            Transition probability matrix
        """
        logger.info("Calculating regime transition matrix...")

        regime_series = regimes["regime"]
        n_regimes = regime_series.nunique()

        # Initialize transition matrix
        transitions = np.zeros((n_regimes, n_regimes))

        # Count transitions
        for i in range(len(regime_series) - 1):
            from_regime = regime_series.iloc[i]
            to_regime = regime_series.iloc[i + 1]
            transitions[from_regime, to_regime] += 1

        # Convert to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_probs = np.divide(
            transitions, row_sums, where=row_sums != 0, out=np.zeros_like(transitions)
        )

        # Create DataFrame
        regime_names = []
        for i in range(n_regimes):
            mask = regime_series == i
            if mask.any() and "regime_name" in regimes.columns:
                name = regimes.loc[mask, "regime_name"].iloc[0]
            else:
                name = f"Regime {i}"
            regime_names.append(name)

        result = pd.DataFrame(transition_probs, index=regime_names, columns=regime_names)

        logger.info("✓ Transition matrix calculated")

        return result


def main():
    """Main demonstration function"""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    # Load data
    logger.info("Loading data...")

    try:
        prices = pd.read_csv("data/raw/asset_prices.csv", index_col=0, parse_dates=True)
        returns = prices.pct_change().dropna()
        logger.info(f"✓ Loaded data: {returns.shape}")
    except FileNotFoundError:
        logger.error("Data not found. Run download_data.py first.")
        return

    # Test each detector
    print("\n" + "=" * 70)
    print("REGIME DETECTION DEMONSTRATION")
    print("=" * 70)

    # 1. Volatility-based
    print("\n1. Volatility Regime Detection")
    print("-" * 70)
    vol_detector = VolatilityRegimeDetector(use_gpu=True)
    vol_regimes = vol_detector.detect_regimes(returns)
    print(vol_regimes[["regime", "regime_name", "volatility"]].tail())

    # 2. Clustering-based
    print("\n2. Clustering Regime Detection")
    print("-" * 70)
    cluster_detector = ClusteringRegimeDetector(n_regimes=4, use_gpu=True)
    cluster_regimes = cluster_detector.detect_regimes(returns)
    print(cluster_regimes[["regime", "regime_name"]].tail())

    # 3. HMM-based (if available)
    if HMM_AVAILABLE:
        print("\n3. HMM Regime Detection")
        print("-" * 70)
        hmm_detector = HMMRegimeDetector(n_regimes=4)
        hmm_regimes = hmm_detector.detect_regimes(returns)
        print(hmm_regimes[["regime", "regime_name"]].tail())

    # 4. Bayesian-based (if available)
    if PYMC_AVAILABLE:
        print("\n4. Bayesian Regime Detection")
        print("-" * 70)
        bayesian_detector = BayesianRegimeDetector(n_regimes=4)
        bayesian_regimes = bayesian_detector.detect_changepoints(returns)
        print(bayesian_regimes[["regime", "regime_name"]].tail())

    # Analyze regimes
    print("\n" + "=" * 70)
    print("REGIME ANALYSIS")
    print("=" * 70)

    analyzer = RegimeAnalyzer()

    # Characteristics
    print("\nRegime Characteristics:")
    characteristics = analyzer.analyze_regime_characteristics(cluster_regimes, returns)
    print(characteristics)

    # Transitions
    print("\nRegime Transition Matrix:")
    transitions = analyzer.calculate_regime_transitions(cluster_regimes)
    print(transitions)

    # Save results
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_regimes.to_parquet(output_dir / "regimes_clustering.parquet")
    characteristics.to_csv(output_dir / "regime_characteristics.csv")
    transitions.to_csv(output_dir / "regime_transitions.csv")

    print("\n✓ Results saved to data/processed/")

    return cluster_regimes


if __name__ == "__main__":
    regimes = main()
