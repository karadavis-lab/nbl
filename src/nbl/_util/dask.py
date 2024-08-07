import webbrowser
from collections.abc import Iterable

from dask.delayed import Delayed
from distributed import Client, as_completed, get_client


class DaskLocalCluster:
    """_summary_."""

    def __init__(self, n_workers: int, threads_per_worker: int):
        """_summary_.

        Parameters
        ----------
        n_workers : int
            _description_
        threads_per_worker : int
            _description_
        """
        self.client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker)

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

    def compute(self, progress_bar: bool | str, tqdm_kwargs: dict):
        """Compute the delayed function.

        Parameters
        ----------
        return_futures : bool
            If True, pass the futuers to `as_completed` to use.

        Returns
        -------
        as_completed
            A class which yields futures as they are finished.
        """
        client = get_client()

        match progress_bar:
            case "rich":
                from tqdm.rich import tqdm
            case "classic":
                from tqdm.auto import tqdm
            case _:
                raise ValueError(f"progress_bar must be one of 'rich' or 'classic', not {progress_bar}")

        futures = client.compute(collections=self._delayed_objects, allow_other_workers=True)

        _results = [r for _f, r in tqdm(as_completed(futures, with_results=True), total=len(self._delayed_objects))]

        return _results
