import webbrowser
from collections.abc import Iterable

import dask
from dask.delayed import Delayed
from distributed import Client, progress


class DaskLocalCluster:
    """A Dask Local Cluster helper class."""

    def __init__(self, n_workers: int, threads_per_worker: int, **client_kwargs):
        """Initialize the DaskLocalCluster class.

        Parameters
        ----------
        n_workers
            The number of workers to use.
        threads_per_worker
            The number of threads per worker.
        **client_kwargs
            Additional keyword arguments to pass to the Dask Client constructor.
        """
        self.client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, **client_kwargs)

    def __call__(self, open_dashboard: bool = True) -> Client:
        """Open the Dask dashboard.

        Parameters
        ----------
        open_dashboard
            Whether to open the dashboard. Default is `True`.

        Returns
        -------
        The Dask Client object.
        """
        if open_dashboard:
            webbrowser.open(self.client.dashboard_link)
        return self.client


class DaskSetupDelayed:
    """A helper class for setting up delayed objects in a Dask cluster."""

    def __init__(self, delayed_objects: Iterable[Delayed]):
        """Initialize the DaskSetupDelayed class.

        Parameters
        ----------
        delayed_objects
            The delayed objects to set up.
        """
        self._delayed_objects = delayed_objects

    def compute(self):
        """Compute the delayed function.

        Returns
        -------
            The results of the delayed computations.
        """
        x = dask.persist(*self._delayed_objects)
        progress(x)
        results = dask.compute(*x)
        match results:
            case list():
                return results
            case tuple():
                return list(results)
