import cv2
import numpy as np
import xarray as xr
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import uniform_filter
from skimage.morphology import disk

from typing import Union, Tuple, Any
import dask.array as darr
from scipy.ndimage import gaussian_filter


def remove_background(varr: xr.DataArray, method: str, wnd: int) -> xr.DataArray:
    """
    Remove background from a video.
    
    Parameters
    ----------
    varr : xr.DataArray
        The input movie data
    method : str
        Either 'uniform' or 'tophat'
    wnd : int
        Window size in pixels
    """
    selem = disk(wnd)
    res = xr.apply_ufunc(
        remove_background_perframe,
        varr,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[varr.dtype],
        kwargs=dict(method=method, wnd=wnd, selem=selem),
    )
    return res.rename(varr.name + "_subtracted")


def remove_background_perframe(
    fm: np.ndarray, 
    method: str, 
    wnd: int, 
    selem: np.ndarray
) -> np.ndarray:
    """
    Remove background from a single frame.
    
    Parameters
    ----------
    fm : np.ndarray
        Input frame
    method : str
        Either 'uniform' or 'tophat'
    wnd : int
        Window size for uniform filter
    selem : np.ndarray
        Structuring element for tophat
    """
    if method == "uniform":
        return fm - uniform_filter(fm, wnd)
    elif method == "tophat":
        return cv2.morphologyEx(
            fm,
            cv2.MORPH_TOPHAT,
            selem
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def stripe_correction(varr, reduce_dim="height", on="mean"):
    if on == "mean":
        temp = varr.mean(dim="frame")
    elif on == "max":
        temp = varr.max(dim="frame")
    elif on == "perframe":
        temp = varr
    else:
        raise NotImplementedError("on {} not understood".format(on))
    mean1d = temp.mean(dim=reduce_dim)
    varr_sc = varr - mean1d
    return varr_sc.rename(varr.name + "_Stripe_Corrected")


def denoise(varr: xr.DataArray, method: str, **kwargs) -> xr.DataArray:
    """
    Denoise the movie frame by frame.

    Parameters
    ----------
    varr : xr.DataArray
        The input movie data, should have dimensions "height", "width" and "frame"
    method : str
        The method to use to denoise each frame:
        - "gaussian": applies cv2.GaussianBlur
        - "anisotropic": applies medpy.filter.smoothing.anisotropic_diffusion
        - "median": applies cv2.medianBlur
        - "bilateral": applies cv2.bilateralFilter

    Returns
    -------
    xr.DataArray
        The denoised movie with "_denoised" appended to its name
    """
    def _wrap_median(frame, **kwargs):
        return cv2.medianBlur(frame, **kwargs)

    def _wrap_gaussian(frame, **kwargs):
        return cv2.GaussianBlur(frame, **kwargs)

    def _wrap_bilateral(frame, **kwargs):
        return cv2.bilateralFilter(frame, **kwargs)

    if method == "gaussian":
        func = _wrap_gaussian
    elif method == "anisotropic":
        func = anisotropic_diffusion
    elif method == "median":
        func = _wrap_median
    elif method == "bilateral":
        func = _wrap_bilateral
    else:
        raise NotImplementedError(f"denoise method {method} not understood")

    res = xr.apply_ufunc(
        func,
        varr,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[varr.dtype],
        kwargs=kwargs,
    )
    
    return res.rename(varr.name + "_denoised")