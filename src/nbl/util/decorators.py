import functools
import inspect
import os
import warnings
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar

import anndata as ad
import spatialdata as sd
from upath import UPath

P = ParamSpec("P")
T = TypeVar("T")


def get_annotation(bound: inspect.BoundArguments, parameter_name: str) -> Any | None:
    """Gets the annotation of a parameter from a bound arguments object.

    Parameters
    ----------
    bound
        The bound arguments object.
    parameter_name
        The name of the parameter to get the annotation for.

    Returns
    -------
    The annotation of the parameter.
    """
    param = bound.signature.parameters[parameter_name]
    return param.annotation


def deprecation_alias(version: str, **aliases: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorate a function to warn user of use of arguments set for deprecation.

    Parameters
    ----------
    aliases
        Deprecation argument aliases to be mapped to the new arguments.

    Returns
    -------
    A decorator that can be used to mark an argument for deprecation and substituting it with the new argument.

    Raises
    ------
    TypeError
        If the provided aliases are not of string type.

    Example
    -------
    Assuming we have an argument 'table' set for deprecation and we want to warn the user and substitute with 'tables':

    ```python
    @deprecation_alias(table="tables")
    def my_function(tables: AnnData | dict[str, AnnData]):
        pass
    ```
    """

    def deprecation_decorator(f: Callable[P, T]) -> Callable[Concatenate[str, P], T]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Callable[Concatenate[str, P], T]:
            class_name = f.__qualname__
            _rename_kwargs(f.__name__, kwargs, aliases, class_name, dep_version=version)
            return f(*args, **kwargs)

        return wrapper

    return deprecation_decorator


def _rename_kwargs(
    func_name: str, kwargs: dict[str, Any], aliases: dict[str, str], class_name: None | str, dep_version: str
) -> None:
    """Rename function arguments set for deprecation and gives warning in case of usage of these arguments."""
    for alias, new in aliases.items():
        if alias in kwargs:
            class_name = class_name + "." if class_name else ""
            if new in kwargs:
                raise TypeError(
                    f"{class_name}{func_name} received both {alias} and {new} as arguments!"
                    f" {alias} is being deprecated in Ark version {dep_version}, only use {new} instead."
                )
            warnings.warn(
                message=(
                    f"`{alias}` is being deprecated as an argument to `{class_name}{func_name}` in Ark "
                    f"version {dep_version}, switch to `{new}` instead."
                ),
                category=DeprecationWarning,
                stacklevel=3,
            )
            kwargs[new] = kwargs.pop(alias)


def path_alias(*args_to_convert: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that converts path-like arguments to `upath.UPath` objects.

    Parameters
    ----------
    *args_to_convert
        Variable-length argument list of strings representing the names of arguments to convert.

    Returns
    -------
    A decorator function that can be applied to another function.
    """

    def path_decorator(f: Callable[P, T]) -> Callable[Concatenate[str, P], T]:
        sig = inspect.signature(f)

        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Callable[Concatenate[str, P], T]:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            _to_upath(bound, *args_to_convert)
            return f(*bound.args, **bound.kwargs)

        return wrapper

    return path_decorator


def _to_upath(bound: inspect.BoundArguments, *args_to_convert: P.args) -> None:
    for arg in args_to_convert:
        if arg in bound.arguments:
            annotation = get_annotation(bound, arg)
            if annotation == (str | os.PathLike):
                bound.arguments.update({arg: UPath(bound.arguments[arg])})


def check_inplace(*args_to_check: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that checks if specified arguments are modified in-place.

    Parameters
    ----------
    *args_to_check
        Variable-length argument list of strings representing the names of arguments to check, and copy in-place
        if necessary.

    Returns
    -------
    A decorator function that can be applied to another function.
    """

    def inplace_decorator(f: Callable[P, T]) -> Callable[Concatenate[str, P], T]:
        sig = inspect.signature(f)

        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Callable[Concatenate[str, P], T]:
            bound = sig.bind(*args, **kwargs)
            func_name = f.__qualname__
            bound.apply_defaults()
            _copy_inplace(bound, *args_to_check, func_name=func_name)
            return f(*bound.args, **bound.kwargs)

        return wrapper

    return inplace_decorator


def _copy_inplace(bound: inspect.BoundArguments, *args_to_check: P.args, func_name: str) -> None:
    if "inplace" in bound.arguments:
        inplace = bound.arguments.get("inplace", False)
        if not inplace:
            for arg in args_to_check:
                if arg in bound.arguments:
                    match bound.signature[arg]:
                        case sd.SpatialData():
                            bound.arguments[arg] = bound.arguments[arg].deepcopy()
                        case ad.AnnData():
                            bound.arguments[arg] = bound.arguments[arg].copy()
                        case _:
                            continue
    else:
        raise ValueError(f"Missing 'inplace' argument in function {func_name} signature.")


def catch_warnings(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that catches warnings and suppresses them.

    Parameters
    ----------
    func
        The function to decorate.

    Returns
    -------
    The decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Callable[Concatenate[str, P], T]:
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper
