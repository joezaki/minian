import os
import functools as fct
import itertools as itt
import numpy as np
import xarray as xr
import dask
from scipy import linalg
import scipy.sparse as scisps
import ffmpeg
from typing import Callable, List, Optional, Tuple, Union
from uuid import uuid4

from vispy import scene, use
from vispy.scene import visuals
from vispy.visuals.filters import IsolineFilter
from vispy.scene.cameras import Magnify1DCamera
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
use('pyqt5')

from .cnmf import compute_AtC, smooth_sig
from .utilities import custom_arr_optimize


def write_video(
    arr: xr.DataArray,
    vname: Optional[str] = None,
    vpath: Optional[str] = ".",
    norm=True,
    options={"crf": "18", "preset": "ultrafast"},
) -> str:
    """
    Write a video from a movie array using `python-ffmpeg`.

    Parameters
    ----------
    arr : xr.DataArray
        Input movie array. Should have dimensions: ("frame", "height", "width")
        and should only be chunked along the "frame" dimension.
    vname : str, optional
        The name of output video. If `None` then a random one will be generated
        using :func:`uuid4.uuid`. By default `None`.
    vpath : str, optional
        The path to the folder containing the video. By default `"."`.
    norm : bool, optional
        Whether to normalize the values of the input array such that they span
        the full pixel depth range (0, 255). By default `True`.
    options : dict, optional
        Optional output arguments passed to `ffmpeg`. By default `{"crf": "18",
        "preset": "ultrafast"}`.

    Returns
    -------
    fname : str
        The absolute path to the video file.

    See Also
    --------
    ffmpeg.output
    """
    if not vname:
        vname = "{}.mp4".format(uuid4())
    fname = os.path.join(vpath, vname)
    if norm:
        arr_opt = fct.partial(
            custom_arr_optimize, rename_dict={"rechunk": "merge_restricted"}
        )
        with dask.config.set(array_optimize=arr_opt):
            arr = arr.astype(np.float32)
            arr_max = arr.max().compute().values
            arr_min = arr.min().compute().values
        den = arr_max - arr_min
        arr -= arr_min
        arr /= den
        arr *= 255
    arr = arr.clip(0, 255).astype(np.uint8)
    w, h = arr.sizes["width"], arr.sizes["height"]
    process = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="gray", s="{}x{}".format(w, h))
        .filter("pad", int(np.ceil(w / 2) * 2), int(np.ceil(h / 2) * 2))
        .output(fname, pix_fmt="yuv420p", vcodec="libx264", r=30, **options)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for blk in arr.data.blocks:
        process.stdin.write(np.array(blk).tobytes())
    process.stdin.close()
    process.wait()
    return fname



def generate_videos(
    varr: xr.DataArray,
    Y: xr.DataArray,
    A: Optional[xr.DataArray] = None,
    C: Optional[xr.DataArray] = None,
    AC: Optional[xr.DataArray] = None,
    nfm_norm: int = None,
    gain=1.5,
    vpath=".",
    vname="minian.mp4",
    options={"crf": "18", "preset": "ultrafast"},
) -> str:
    """
    Generate a video visualizaing the result of minian pipeline.

    The resulting video contains four parts: Top left is a original reference
    movie supplied as `varr`; Top right is the input to CNMF algorithm supplied
    as `Y`; Bottom right is a movie `AC` representing cellular activities as
    computed by :func:`minian.cnmf.compute_AtC`; Bottom left is a residule movie
    computed as the difference between `Y` and `AC`. Since the CNMF algorithm
    contains various arbitrary scaling process, a normalizing scalar is computed
    with least square using a subset of frames from `Y` and `AC` such that their
    numerical values matches.

    Parameters
    ----------
    varr : xr.DataArray
        Input reference movie data. Should have dimensions ("frame", "height",
        "width"), and should only be chunked along "frame" dimension.
    Y : xr.DataArray
        Movie data representing input to CNMF algorithm. Should have dimensions
        ("frame", "height", "width"), and should only be chunked along "frame"
        dimension.
    A : xr.DataArray, optional
        Spatial footprints of cells. Only used if `AC` is `None`. By default
        `None`.
    C : xr.DataArray, optional
        Temporal activities of cells. Only used if `AC` is `None`. By default
        `None`.
    AC : xr.DataArray, optional
        Spatial-temporal activities of cells. Should have dimensions ("frame",
        "height", "width"), and should only be chunked along "frame" dimension.
        If `None` then both `A` and `C` should be supplied and
        :func:`minian.cnmf.compute_AtC` will be used to compute this variable.
        By default `None`.
    nfm_norm : int, optional
        Number of frames to randomly draw from `Y` and `AC` to compute the
        normalizing factor with least square. By default `None`.
    gain : float, optional
        A gain factor multiplied to `Y`. Useful to make the results visually
        brighter. By default `1.5`.
    vpath : str, optional
        Desired folder containing the resulting video. By default `"."`.
    vname : str, optional
        Desired name of the video. By default `"minian.mp4"`.
    options : dict, optional
        Output options for `ffmpeg`, passed directly to :func:`write_video`. By
        default `{"crf": "18", "preset": "ultrafast"}`.

    Returns
    -------
    fname : str
        Absolute path of the resulting video.
    """
    if AC is None:
        print("generating traces")
        AC = compute_AtC(A, C)
    print("normalizing")
    gain = 255 / Y.max().compute().values * gain
    Y = Y * gain
    if nfm_norm is not None:
        norm_idx = np.sort(
            np.random.choice(np.arange(Y.sizes["frame"]), size=nfm_norm, replace=False)
        )
        Y_sub = Y.isel(frame=norm_idx).values.reshape(-1)
        AC_sub = scisps.csc_matrix(AC.isel(frame=norm_idx).values.reshape((-1, 1)))
        lsqr = scisps.linalg.lsqr(AC_sub, Y_sub)
        norm_factor = lsqr[0].item()
        del Y_sub, AC_sub
    else:
        norm_factor = gain
    AC = AC * norm_factor
    res = Y - AC
    print("writing videos")
    vid = xr.concat(
        [
            xr.concat([varr, Y], "width", coords="minimal"),
            xr.concat([res, AC], "width", coords="minimal"),
        ],
        "height",
        coords="minimal",
    )
    return write_video(vid, vname, vpath, norm=False, options=options)