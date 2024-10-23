from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dask import compute, delayed
from numpydantic import NDArray
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._stats.base import Stat


@dataclass
class BootStrapResult:
    """
    Data class to store the results of a bootstrap operation.

    Attributes
    ----------
        idx: Index of the bootstrap iteration.
        xx (NDArray): Array representing the x-values.
        result (NDArray): Array containing the smoothed results.
    """

    idx: int
    xx: NDArray
    result: NDArray


def _single_bootstrap(idx: int, data: pd.DataFrame, xx, frac, delta) -> BootStrapResult:
    result = sm.nonparametric.lowess(
        endog=data["y"],
        exog=data["x"],
        xvals=xx,
        frac=frac,
        delta=delta,
    )

    # Select the second column if result is two-dimensional, else reformat and select
    result_array: NDArray = result[:, 1] if result.ndim > 1 else np.column_stack((xx, result))[:, 1]

    return BootStrapResult(idx=idx, xx=xx, result=result_array)


@dataclass
class LowessDask(Stat):
    """Perform locally-weighted regression (LOWESS) to smooth data.

    This statistical method allows fitting a smooth curve to your data
    using a local regression. It can be useful to visualize the trend of the data.


    Parameters
    ----------
    frac
        The fraction of data used when estimating each y-value.
    gridsize
        The number of points in the grid to which the LOWESS is applied.
        Higher values result in a smoother curve.
    delta
        Distance within which to use linear-interpolation instead of weighted regression. Default is 0.0.
    num_bootstrap
        The number of bootstrap samples to use for confidence intervals. Default is None.
    alpha
        Confidence level for the intervals. Default is 0.95.
    clip_min
        Minimum value to clip the smoothed curve to. Default is None.
    clip_max
        Maximum value to clip the smoothed curve to. Default is None.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with the smoothed curve's 'x', 'y', 'ymin', and 'ymax' coordinates.

    """

    frac: float = 0.2
    gridsize: int = 100
    delta: float = 0.0
    num_bootstrap: int | None = None
    alpha: float = 0.95
    clip_max: float | None = None
    clip_min: float | None = None

    def __post_init__(self):
        # Type checking for the arguments
        if not isinstance(self.frac, float) or not (0 < self.frac <= 1):
            raise ValueError("frac must be a float between 0 and 1.")
        if not isinstance(self.gridsize, int) or self.gridsize <= 0:
            raise ValueError("gridsize must be a positive integer.")
        if self.num_bootstrap is not None and (not isinstance(self.num_bootstrap, int) or self.num_bootstrap <= 0):
            raise ValueError("num_bootstrap must be a positive integer or None.")
        if not isinstance(self.alpha, float) or not (0 < self.alpha < 1):
            raise ValueError("alpha must be a float between 0 and 1.")

    def _fit_predict(self, data: pd.DataFrame):
        x = data["x"]
        xx = np.linspace(x.min(), x.max(), self.gridsize)
        result = sm.nonparametric.lowess(endog=data["y"], exog=x, frac=self.frac, delta=self.delta, xvals=xx)
        if result.ndim == 1:  # Handle single-dimensional return values
            yy = result
        else:
            yy = result[:, 1]  # Select the predicted y-values
        if self.clip_max or self.clip_min:
            np.clip(yy, a_min=self.clip_min, a_max=self.clip_max, out=yy)
            np.clip(xx, a_min=self.clip_min, a_max=self.clip_max, out=xx)
        return pd.DataFrame({"x": xx, "y": yy})

    def _bootstrap_resampling(self, data):
        xx = np.linspace(data["x"].min(), data["x"].max(), self.gridsize)
        bootstrap_estimates = np.empty((self.num_bootstrap, len(xx)))

        tasks = [delayed(_single_bootstrap)(idx, data, xx, self.frac, self.delta) for idx in range(self.num_bootstrap)]

        bootstrap_results: list[BootStrapResult] = compute(*tasks)
        for br in bootstrap_results:
            bootstrap_estimates[br.idx, :] = br.result[:]
        if self.clip_max or self.clip_min:
            np.clip(bootstrap_estimates, a_min=self.clip_min, a_max=self.clip_max, out=bootstrap_estimates)
        lower_bound = np.percentile(bootstrap_estimates, (1 - self.alpha) / 2 * 100, axis=0)
        upper_bound = np.percentile(bootstrap_estimates, (1 + self.alpha) / 2 * 100, axis=0)
        return pd.DataFrame({"ymin": lower_bound, "ymax": upper_bound})

    def __call__(self, data: DataFrame, groupby: GroupBy, orient: str, scales) -> DataFrame:  # noqa: D102
        if orient == "x":
            xvar = data.columns[0]
            yvar = data.columns[1]
        else:
            xvar = data.columns[1]
            yvar = data.columns[0]

        renamed_data = data.rename(columns={xvar: "x", yvar: "y"})
        renamed_data = renamed_data.dropna(subset=["x", "y"])

        grouping_vars = [str(v) for v in data if v in groupby.order]

        if not grouping_vars:
            # If no grouping variables, directly fit and predict
            smoothed = self._fit_predict(renamed_data)
        else:
            # Apply the fit_predict method for each group separately
            smoothed = groupby.apply(renamed_data, self._fit_predict)

        if self.num_bootstrap:
            if not grouping_vars:
                bootstrap_estimates = self._bootstrap_resampling(renamed_data)
            else:
                bootstrap_estimates = groupby.apply(renamed_data, self._bootstrap_resampling)
        return smoothed.join(bootstrap_estimates[["ymin", "ymax"]]) if self.num_bootstrap else smoothed
