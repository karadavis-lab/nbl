import webbrowser
from collections.abc import Iterable

import dask
from dask.delayed import Delayed
from distributed import Client, progress


class DaskLocalCluster:
    """_summary_."""

    def __init__(self, n_workers: int, threads_per_worker: int, **client_kwargs):
        """_summary_.

        Parameters
        ----------
        n_workers : int
            _description_
        threads_per_worker : int
            _description_
        """
        self.client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, **client_kwargs)

    def __call__(self, open_dashboard: bool = True) -> Client:
        """_summary_.

        Parameters
        ----------
        open_dashboard : bool, optional
            _description_, by default True

        Returns
        -------
        Client
            _description_
        """
        if open_dashboard:
            webbrowser.open(self.client.dashboard_link)
        return self.client


class DaskSetupDelayed:
    """_summary_."""

    def __init__(self, delayed_objects: Iterable[Delayed]):
        """_summary_.

        Parameters
        ----------
        delayed_objects : Iterable[Delayed]
            _description_
        """
        self._delayed_objects = delayed_objects

    def compute(self):
        """Compute the delayed function.

        Returns
        -------
        list[Any]
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
