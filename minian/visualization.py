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
from vispy.color import colormap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
use('pyqt5')

from .cnmf import compute_AtC, smooth_sig
from .utilities import custom_arr_optimize


def visualize_raw_video(
        varr,
        title
):
    cur_frame = varr.sel(frame = 0)

    qt_app = QApplication.instance()
    win = QWidget()
    win.setWindowTitle(title)
    layout = QVBoxLayout()
    win.setLayout(layout)
    
    # VisPy canvas and view
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
    layout.addWidget(canvas.native)
    grid = canvas.central_widget.add_grid(spacing=1)
    
    view1 = scene.ViewBox(pos=(0,0), parent=canvas.scene)
    view2 = scene.ViewBox(pos=(0,1), parent=canvas.scene)
    view3 = scene.ViewBox(pos=(1,0), parent=canvas.scene)

    grid.add_widget(view1, 0, 0, col_span=2, row_span=2)
    grid.add_widget(view2, 0, 2, col_span=1, row_span=2)
    grid.add_widget(view3, 2, 0, col_span=3)


    frame_im = scene.Image(cur_frame, parent=view1.scene)
    histogram = visuals.Histogram(
        data=cur_frame.stack(stacked_dims=['height','width']),
        bins=50,
        orientation='v',
        color='darkblue',
        parent=view2.scene)
    frames = varr.frame
    min_vals = scene.Line(
        pos=np.column_stack((frames, varr.min(['height','width']))),
                            color='#ff7f0e', width=2, parent=view3.scene)
    max_vals = scene.Line(
        pos=np.column_stack((frames, varr.max(['height','width']))),
                            color='#1f77b4', width=2, parent=view3.scene)
    mean_vals = scene.Line(
        pos=np.column_stack((frames, varr.mean(['height','width']))),
                            color='black', width=2, parent=view3.scene)
    cur_vline = scene.Line(
        pos=np.column_stack(((0,0),(-10,300))),
                             color='red', width=2, parent=view3.scene)

    view1.camera = 'panzoom'
    view2.camera = 'panzoom'
    view3.camera = 'panzoom'
    view1.camera.set_range()
    view2.camera.set_range(y=(0,255))
    view3.camera.set_range(x=(frames[0],frames[-1]))

    # Slider
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, varr.sizes['frame'] - 1)
    slider.setValue(0)
    layout.addWidget(slider)
    
    # Update line on slider change
    def update_plot(index):
        cur_frame = varr.sel(frame=index)
        frame_im.set_data(cur_frame)

        nonlocal histogram
        histogram.parent = None
        histogram = visuals.Histogram(
            data=cur_frame.stack(stacked_dims=['height','width']),
            bins=50,
            orientation='v',
            color='darkblue',
            parent=view2.scene)

        cur_vline.set_data(np.column_stack(((index,index),(-10,300))))

        win.setWindowTitle(f'frame = {index}')
        canvas.update()

    slider.valueChanged.connect(update_plot)

    win.show()
    qt_app.exec_()


def visualize_before_after(
        before,
        after,
        title,
        im_scaling=1
):

    # if objects are just images, add a frame dim
    if 'frame' not in before.dims:
        before = before.expand_dims({'frame': 1})
    if 'frame' not in after.dims:
        after = after.expand_dims({'frame': 1})

    qt_app = QApplication.instance()
    win = QWidget()
    win.setWindowTitle(title)
    layout = QVBoxLayout()
    win.setLayout(layout)
    
    width  = before.sizes['width']
    height = before.sizes['height']
    width  *= im_scaling
    height *= im_scaling
    
    # VisPy canvas and view
    canvas = scene.SceneCanvas(keys='interactive', size=(width*2,height), bgcolor='white')
    layout.addWidget(canvas.native)
    grid = canvas.central_widget.add_grid(spacing=1)
    
    view1 = scene.ViewBox(pos=(0,0), size=(width,height), parent=canvas.scene)
    view2 = scene.ViewBox(pos=(0,1), size=(width,height), parent=canvas.scene)

    grid.add_widget(view1, 0, 0)
    grid.add_widget(view2, 0, 1)

    # add before and after images
    before_im = scene.Image(before.isel(frame=0), parent=view1.scene)
    after_im = scene.Image(after.isel(frame=0), parent=view1.scene)
    view2.add(after_im)

    view1.camera = scene.PanZoomCamera(rect=((0, 0), (width, height)))
    view2.camera = scene.PanZoomCamera(rect=((0, 0), (width, height)))
    view1.camera.aspect = 1
    view2.camera.aspect = 1
    view1.camera.link(view2.camera)

    # Slider
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, before.sizes['frame'] - 1)
    slider.setValue(0)
    layout.addWidget(slider)
    
    # Update line on slider change
    def update_plot(index):
        before_im.set_data(before.isel(frame=index))
        after_im.set_data(after.isel(frame=index))
        win.setWindowTitle(f'{title}: frame = {index}')
        canvas.update()

    slider.valueChanged.connect(update_plot)

    win.show()
    qt_app.exec_()


def visualize_preprocess(
        frame,
        func,
        title,
        im_scaling=1,
        **kwargs
):
    # create list of processed images and subtitles
    pkey = kwargs.keys()
    pval = kwargs.values()
    image_ls = [func(frame, **dict(zip(pkey, params))) for params in itt.product(*pval)]
    title_ls = [str(dict(zip(pkey, params))) for params in itt.product(*pval)]

    qt_app = QApplication.instance()
    win = QWidget()
    win.setWindowTitle(title)
    layout = QVBoxLayout()
    win.setLayout(layout)
    
    width  = frame.sizes['width']
    height = frame.sizes['height']
    width  *= im_scaling
    height *= im_scaling
    
    # VisPy canvas and view
    canvas = scene.SceneCanvas(keys='interactive', size=(width*2,height*2), bgcolor='white')
    layout.addWidget(canvas.native)
    grid = canvas.central_widget.add_grid(spacing=1)
    
    view1 = scene.ViewBox(pos=(0,0), size=(width,height), parent=canvas.scene)
    view2 = scene.ViewBox(pos=(0,1), size=(width,height), parent=canvas.scene)
    view3 = scene.ViewBox(pos=(1,0), size=(width,height), parent=canvas.scene)
    view4 = scene.ViewBox(pos=(1,1), size=(width,height), parent=canvas.scene)
    
    grid.add_widget(view1, 0, 0)
    grid.add_widget(view2, 0, 1)
    grid.add_widget(view3, 1, 0)
    grid.add_widget(view4, 1, 1)

    # add original image (subplot 1)
    orig_image = scene.Image(
        image_ls[0],
        parent=view1.scene
        )

    # add original contour (subplot 2)
    orig_contour = scene.Image(
        image_ls[0],
        interpolation='cubic',
        parent=view2.scene
        )
    iso = IsolineFilter(level=5, width=2, color='white')
    orig_contour.attach(iso)

    # add processed image (subplot 3)
    processed_image = scene.Image(
        image_ls[0],
        parent=view3.scene
        )
    
    # add processed image contour (subplot 4)
    processed_contour = scene.Image(
        image_ls[0],
        interpolation='cubic',
        parent=view4.scene
        )
    iso = IsolineFilter(level=5, width=2, color='white')
    processed_contour.attach(iso)
    
    # share axes
    view1.camera = scene.PanZoomCamera(rect=((0, 0), (width, height)))
    view2.camera = scene.PanZoomCamera(rect=((0, 0), (width, height)))
    view3.camera = scene.PanZoomCamera(rect=((0, 0), (width, height)))
    view4.camera = scene.PanZoomCamera(rect=((0, 0), (width, height)))
    view1.camera.link(view2.camera)
    view1.camera.link(view3.camera)
    view1.camera.link(view4.camera)
    
    # Slider
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, len(image_ls) - 1)
    slider.setValue(0)
    layout.addWidget(slider)
    
    # Update line on slider change
    def update_plot(index):
        processed_image.set_data(image_ls[index])
        processed_contour.set_data(image_ls[index])
        win.setWindowTitle(f'{title}: {title_ls[index]}')
        canvas.update()

    slider.valueChanged.connect(update_plot)
    
    win.show()
    qt_app.exec_()


def visualize_motion(
        motion,
        magnify=False
):
    
    qt_app = QApplication.instance()
    win = QWidget()
    layout = QVBoxLayout(win)

    canvas = scene.SceneCanvas(keys='interactive', size=(1000, 400), show=True, bgcolor='white')
    canvas._send_hover_events = True
    layout.addWidget(canvas.native)

    # Add a ViewBox with pan/zoom
    view = canvas.central_widget.add_view()
    if magnify:
        view.camera = Magnify1DCamera(mag=4, size_factor=0.6, radius_ratio=0.6)
    else:
        view.camera = 'panzoom'

    # Add X and Y axes
    x_axis = scene.AxisWidget(orientation='bottom', axis_label='frame',
                            axis_color='black', text_color='black', tick_color='black')
    y_axis = scene.AxisWidget(orientation='left', axis_label='motion',
                            axis_color='black', text_color='black', tick_color='black')
    x_axis.stretch = (1, 0.1)
    y_axis.stretch = (0.1, 1)
    # Link axes to the view
    grid = canvas.central_widget.add_grid()
    grid.add_widget(y_axis, row=0, col=0)
    grid.add_widget(x_axis, row=1, col=1)
    grid.add_widget(view,   row=0, col=1)
    x_axis.link_view(view)
    y_axis.link_view(view)

    frames = motion.frame
    width_line = scene.Line(pos=np.column_stack((frames, motion.sel(shift_dim="width"))),
                            color='#ff7f0e', width=1, parent=view.scene)
    height_line = scene.Line(pos=np.column_stack((frames, motion.sel(shift_dim="height"))),
                             color='#1f77b4', width=1, parent=view.scene)
    view.camera.set_range()
    
    win.setLayout(layout)
    win.show()
    qt_app.exec_()


def visualize_seeds(
        max_proj,
        seeds,
        mask=None
):
    
    if mask is None:
        mask = np.repeat(True, seeds.shape[0])
    else:
        mask = seeds[mask]
    good_seeds = seeds[mask].copy()
    bad_seeds  = seeds[np.invert(mask)].copy()

    qt_app = QApplication.instance()
    win = QWidget()
    layout = QVBoxLayout(win)

    canvas = scene.SceneCanvas(keys='interactive', show=True)
    canvas._send_hover_events = True
    layout.addWidget(canvas.native)

    # Add a ViewBox with pan/zoom
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'

    # add seeds
    good_seeds_scatter = visuals.Markers()
    good_seeds_scatter.set_data(pos=good_seeds[['width','height']].values,
                                edge_width=0, face_color=('white'), size=3, symbol='o')
    view.add(good_seeds_scatter)
    if bad_seeds.shape[0] > 0:
        bad_seeds_scatter = visuals.Markers()
        bad_seeds_scatter.set_data(pos=bad_seeds[['width','height']].values,
                                edge_width=0, face_color=('red'), size=3, symbol='o')
        view.add(bad_seeds_scatter)

    # add max proj
    max_proj_im = scene.Image(max_proj, parent=view.scene)

    view.camera.set_range()
    view.camera.aspect = 1
    
    win.setLayout(layout)
    win.show()
    qt_app.exec_()


def visualize_pnr_refine(
        Y_hw_chk,
        example_seeds,
        noise_freq_list,
        cols=3,
        magnify=False,
        link_views=False
):
    # compute signals for all pnr levels
    example_trace = Y_hw_chk.sel(
        height=example_seeds["height"].to_xarray(),
        width=example_seeds["width"].to_xarray(),
    ).rename(**{"index": "seed"})
    arrays_dict = {}
    for freq in noise_freq_list:
        trace_smth_low = smooth_sig(example_trace, freq).compute()
        trace_smth_high = smooth_sig(example_trace, freq, btype="high").compute()
        arrays_dict[freq] = {'low':trace_smth_low,
                             'high':trace_smth_high}

    # begin plotting
    qt_app = QApplication.instance()
    win = QWidget()
    layout = QVBoxLayout(win)

    canvas = scene.SceneCanvas(keys='interactive', size=(1000, 400), show=True, bgcolor='white')
    canvas._send_hover_events = True
    layout.addWidget(canvas.native)

    grid = canvas.central_widget.add_grid()
    plot_coords = list(itt.product(range(int(np.ceil(len(example_seeds)/cols))),range(cols)))
    view_ls = []
    frames = Y_hw_chk.frame
    low_lines_ls = []
    high_lines_ls = []
    for i in np.arange(len(example_seeds)):
        view_ls.append(scene.ViewBox(pos=plot_coords[i], border_color='black'))
        if magnify:
            view_ls[i].camera = Magnify1DCamera(mag=4, size_factor=0.6, radius_ratio=0.6)
        else:
            view_ls[i].camera = 'panzoom'
        grid.add_widget(view_ls[i], plot_coords[i][0], plot_coords[i][1])
        low_line = scene.Line(
            pos=np.column_stack((frames, arrays_dict[noise_freq_list[0]]['low'][i])),
            color='#ff7f0e',
            width=1,
            parent=view_ls[i].scene
            )
        low_lines_ls.append(low_line)
        high_line = scene.Line(
            pos=np.column_stack((frames, arrays_dict[noise_freq_list[0]]['high'][i])),
            color='#1f77b4',
            width=1,
            parent=view_ls[i].scene
            )
        high_lines_ls.append(high_line)

    # link views
    if link_views:
        for i in np.arange(1,len(view_ls)):
            view_ls[0].camera.link(view_ls[i].camera)
    
    # Slider
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, len(noise_freq_list) - 1)
    slider.setValue(0)
    layout.addWidget(slider)
    
    # Update line on slider change
    def update_plot(index):
        for i in np.arange(len(example_seeds)):
            low_data = arrays_dict[noise_freq_list[index]]['low'][i]
            high_data = arrays_dict[noise_freq_list[index]]['high'][i]
            low_lines_ls[i].set_data(pos=np.column_stack((frames, low_data)))
            high_lines_ls[i].set_data(pos=np.column_stack((frames, high_data)))
            view_ls[i].camera.set_range(y=(min(low_data.min(),high_data.min()),
                                           max(low_data.max(),high_data.max())))
        win.setWindowTitle(f'noise frequency: {noise_freq_list[index]}')
        canvas.update()

    update_plot(0)
    slider.valueChanged.connect(update_plot)
    
    win.show()
    qt_app.exec_()


def visualize_initialization(
        A,
        C,
        b,
        f
):
    qt_app = QApplication.instance()
    win = QWidget()
    win.setWindowTitle('Initialization Visualization')
    layout = QVBoxLayout()
    win.setLayout(layout)
    
    # VisPy canvas and view
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
    layout.addWidget(canvas.native)
    grid = canvas.central_widget.add_grid(spacing=1)

    # add data to subplots
    view_ls = []
    col_spans = [1, 2, 1, 2]
    plot_coords = list(itt.product(range(2),range(2)))
    data_to_plot = {
        'A': A.max("unit_id").compute().astype(np.float32),
        'C': C.compute().astype(np.float32),
        'b': b.compute().astype(np.float32),
        'f': f.compute().astype(np.float32)
    }
    for i, (var, data) in enumerate(data_to_plot.items()):
        view = scene.ViewBox(pos=plot_coords[i], parent=canvas.scene)
        view_ls.append(view)
        view_ls[i].camera = 'panzoom'
        grid.add_widget(view_ls[i], plot_coords[i][0], plot_coords[i][1], col_span=col_spans[i])
        if var == 'f':
            plot = scene.Line(pos=np.column_stack((f.frame, f)),
                            color="#07117b", width=1, parent=view.scene)
        else:
            plot = scene.Image(data, parent=view_ls[i])
        view_ls[i].add(plot)
        view_ls[i].camera.set_range()
    
    # link the spatial subplots together
    view_ls[0].camera.link(view_ls[2].camera)
    
    win.setLayout(layout)
    win.show()
    qt_app.exec_()


def normalize(a: np.ndarray) -> np.ndarray:
    """
    Normalize an input array to range (0, 1) using :func:`numpy.interp`.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    a_norm : np.ndarray
        Normalized array.
    """
    return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, +1))


def visualize_spatial_params(
        units,
        A_dict,
        C_dict,
        norm=True
):
    sprs_ls = list(A_dict.keys())

    if norm:
        for sprs in sprs_ls:
            C_dict[sprs] = xr.apply_ufunc(
                normalize,
                C_dict[sprs].chunk(dict(frame=-1)),
                input_core_dims=[["frame"]],
                output_core_dims=[["frame"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[C_dict[sprs].dtype],
            )
            C_dict[sprs] = C_dict[sprs].compute()

    qt_app = QApplication.instance()
    win = QWidget()
    win.setWindowTitle('Initialization Visualization')
    layout = QVBoxLayout()
    win.setLayout(layout)
    
    # VisPy canvas and view
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
    layout.addWidget(canvas.native)
    grid = canvas.central_widget.add_grid()

    view1 = scene.ViewBox(pos=(0,0), parent=canvas.scene, border_color='black')
    view2 = scene.ViewBox(pos=(0,1), parent=canvas.scene, border_color='black')
    view3 = scene.ViewBox(pos=(1,0), parent=canvas.scene, border_color='black')
    grid.add_widget(view1, 0, 0)
    grid.add_widget(view2, 0, 1, row_span=2)
    grid.add_widget(view3, 1, 0)

    width, height = A_dict[sprs_ls[0]].sizes['width'], A_dict[sprs_ls[0]].sizes['height']
    view1.camera = scene.PanZoomCamera(rect=((0, 0), (width, height)))
    view2.camera = 'panzoom'
    view3.camera = scene.PanZoomCamera(rect=((0, 0), (width, height)))
    view1.camera.link(view3.camera)
    
    # plot footprints
    A_binary = scene.Image((A_dict[sprs_ls[0]] > 0).sum("unit_id").astype(np.float32), parent=view1.scene)

    A_cont = scene.Image(A_dict[sprs_ls[0]].sum("unit_id").astype(np.float32), parent=view3.scene)

    C_ls = []
    for i, _ in enumerate(units):
        line = scene.Line(pos=np.column_stack((C_dict[sprs_ls[0]].frame,
                                               C_dict[sprs_ls[0]][i,:]+i)),
                                               color='#07117b', width=1, parent=view2.scene)
        C_ls.append(line)
    view2.camera.set_range()

    # Slider
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, len(sprs_ls) - 1)
    slider.setValue(0)
    layout.addWidget(slider)
    
    # Update line on slider change
    def update_plot(index):
        A_binary.set_data((A_dict[sprs_ls[index]] > 0).sum("unit_id"))
        A_cont.set_data(A_dict[sprs_ls[index]].sum("unit_id"))
        for i, unit in enumerate(units):
            C_ls[i].parent = None
            if unit in C_dict[sprs_ls[index]].unit_id:
                C_ls[i] = scene.Line(pos=np.column_stack((C_dict[sprs_ls[index]].frame,
                                                    C_dict[sprs_ls[index]].sel(unit_id=unit)+i)),
                                                    color='#07117b', width=1, parent=view2.scene)
        win.setWindowTitle(f'sparse penalty: {sprs_ls[index]}')
        canvas.update()

    update_plot(0)
    slider.valueChanged.connect(update_plot)
    
    win.setLayout(layout)
    win.show()
    qt_app.exec_()


def visualize_spatial_update(
        A,
        A_new
):
    
    qt_app = QApplication.instance()
    win = QWidget()
    win.setWindowTitle('Initialization Visualization')
    layout = QVBoxLayout()
    win.setLayout(layout)
    
    # VisPy canvas and view
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
    layout.addWidget(canvas.native)
    grid = canvas.central_widget.add_grid()

    plot_coords = list(itt.product(range(2),range(2)))
    data_to_plot = {
        'A'        : A.max("unit_id").compute().astype(np.float32),
        'A_bin'    : (A.fillna(0) > 0).sum("unit_id").compute().astype(np.uint8),
        'A_new'    : A_new.max("unit_id").compute().astype(np.float32),
        'A_new_bin': (A_new > 0).sum("unit_id").compute().astype(np.uint8)
    }
    view_ls = []
    for i, (var, data) in enumerate(data_to_plot.items()):
        view = scene.ViewBox(pos=(plot_coords[i]), parent=canvas.scene, border_color='black')
        view_ls.append(view)
        grid.add_widget(view_ls[i], plot_coords[i][0], plot_coords[i][1])
        width, height = data.sizes['width'], data.sizes['height']
        view_ls[i].camera = scene.PanZoomCamera(rect=((0, 0), (width, height)))

        plot = scene.Image(data, parent=view_ls[i].scene)

    for i in np.arange(1,4):
        view_ls[0].camera.link(view_ls[i].camera)
    
    win.show()
    qt_app.exec_()


def visualize_spatial_bg(
        b,
        f,
        b_new,
        f_new
):
    
    qt_app = QApplication.instance()
    win = QWidget()
    win.setWindowTitle('Initialization Visualization')
    layout = QVBoxLayout()
    win.setLayout(layout)
    
    # VisPy canvas and view
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
    layout.addWidget(canvas.native)
    grid = canvas.central_widget.add_grid()

    plot_coords = list(itt.product(range(2),range(2)))
    data_to_plot = {
        'b'     : b.compute().astype(np.float32),
        'f'     : f.compute().astype(np.float16),
        'b_new' : b_new.compute().astype(np.float32),
        'f_new' : f_new.compute().astype(np.float16)
    }
    view_ls = []
    for i, (var, data) in enumerate(data_to_plot.items()):
        view = scene.ViewBox(pos=(plot_coords[i]), parent=canvas.scene, border_color='black')
        view_ls.append(view)
        grid.add_widget(view_ls[i], plot_coords[i][0], plot_coords[i][1])

        if var[0] == 'f':
            plot = scene.Line(pos=np.column_stack((data.frame, data)),
                              color='#07117b', width=1, parent=view_ls[i].scene)
            view_ls[i].camera = 'panzoom'
            view_ls[i].camera.set_range()
        else:
            width, height = data.sizes['width'], data.sizes['height']
            view_ls[i].camera = scene.PanZoomCamera(rect=((0, 0), (width, height)))
            plot = scene.Image(data, parent=view_ls[i].scene)
    
    # temporarily link spatial and temporal plots by index
    view_ls[0].camera.link(view_ls[2].camera)
    view_ls[1].camera.link(view_ls[3].camera)

    win.show()
    qt_app.exec_()


def construct_G(g: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Construct a convolving matrix from AR coefficients.

    Parameters
    ----------
    g : np.ndarray
        Input AR coefficients.
    T : np.ndarray
        Number of time samples of the AR process.

    Returns
    -------
    G : np.ndarray
        A `T` x `T` matrix that can be used to multiply with a timeseries to
        convolve the AR process.

    See Also
    --------
    minian.cnmf.update_temporal :
        for more background on the role of AR process in the pipeline
    """
    cur_c, cur_r = np.zeros(T), np.zeros(T)
    cur_c[0] = 1
    cur_r[0] = 1
    cur_c[1 : len(g) + 1] = -g
    return linalg.toeplitz(cur_c, cur_r)


def convolve_G(s: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Convolve an AR process to input timeseries.

    Despite the name, only AR coefficients are needed as input. The convolving
    matrix will be computed using :func:`construct_G`.

    Parameters
    ----------
    s : np.ndarray
        The input timeseries, presumably representing spike signals.
    g : np.ndarray
        The AR coefficients.

    Returns
    -------
    c : np.ndarray
        Convolved timeseries, presumably representing calcium dynamics.

    See Also
    --------
    minian.cnmf.update_temporal :
        for more background on the role of AR process in the pipeline
    """
    G = construct_G(g, len(s))
    try:
        c = np.linalg.inv(G).dot(s)
    except np.linalg.LinAlgError:
        c = s.copy()
    return c


def construct_pulse_response(
    g: np.ndarray, length=500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a model pulse response corresponding to certain AR coefficients.

    Parameters
    ----------
    g : np.ndarray
        The AR coefficients.
    length : int, optional
        Number of timepoints in output. By default `500`.

    Returns
    -------
    s : np.ndarray
        Model spike with shape `(length,)`, zero everywhere except the first
        timepoint.
    c : np.ndarray
        Model convolved calcium response, with same shape as `s`.

    See Also
    --------
    minian.cnmf.update_temporal :
        for more background on the role of AR process in the pipeline
    """
    s = np.zeros(length)
    s[np.arange(0, length, 500)] = 1
    c = convolve_G(s, g)
    return s, c


def visualize_temporal_params(
    units,
    params,
    YA_dict, # raw signal in grey
    C_dict, # fitted signal in orange
    S_dict, # fitted spike in steelblue
    g_dict, # 
    sig_dict,
    A_dict,
    norm=True,
    magnify=True
):

    cur_params = {param:params[param][0] for param in params}
    cur_cell = [units[0]]
    frames = YA_dict[tuple(cur_params.values())].frame

    activities_dict = {
        'YA' : YA_dict,
        'C'  : C_dict,
        'S'  : S_dict,
        # 'sig': sig_dict
    }

    if norm:
        for var in activities_dict.keys():
            for param in activities_dict[var]:
                activities_dict[var][param] = xr.apply_ufunc(
                    normalize,
                    activities_dict[var][param].chunk(dict(frame=-1)).compute(),
                    input_core_dims=[["frame"]],
                    output_core_dims=[["frame"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[activities_dict[var][param].dtype],
                    )

    # compute pulse simulation
    s_pul_dict = {}
    c_pul_dict = {}
    for param_ls in S_dict.keys():
        f_crd = YA_dict[param_ls].coords["frame"]
        pul_crd = f_crd.values[:500]

        s_pul, c_pul = xr.apply_ufunc(
            construct_pulse_response,
            g_dict[param_ls].compute(),
            input_core_dims=[["lag"]],
            output_core_dims=[["t"], ["t"]],
            vectorize=True,
            kwargs=dict(length=len(pul_crd)),
            output_sizes=dict(t=len(pul_crd)),
        )
        s_pul, c_pul = (s_pul.assign_coords(t=pul_crd), c_pul.assign_coords(t=pul_crd))
        if norm:
            c_pul = xr.apply_ufunc(
                normalize,
                c_pul.chunk(dict(t=-1)),
                input_core_dims=[["t"]],
                output_core_dims=[["t"]],
                dask="parallelized",
                output_dtypes=[c_pul.dtype],
            ).compute()
        s_pul_dict[param_ls] = s_pul
        c_pul_dict[param_ls] = c_pul
    
    # begin plotting
    qt_app = QApplication.instance()
    win = QWidget()
    layout = QVBoxLayout(win)

    canvas = scene.SceneCanvas(keys='interactive', size=(1000, 400), show=True, bgcolor='white')
    canvas._send_hover_events = True
    layout.addWidget(canvas.native)

    grid = canvas.central_widget.add_grid()
    view1 = scene.ViewBox(pos=(0,0), parent=canvas.scene, border_color='black')
    view2 = scene.ViewBox(pos=(1,0), parent=canvas.scene, border_color='black')
    view3 = scene.ViewBox(pos=(1,1), parent=canvas.scene, border_color='black')
    grid.add_widget(view1, 0, 0, col_span=2)
    grid.add_widget(view2, 1, 0)
    grid.add_widget(view3, 1, 1)

    if magnify:
        view1.camera = Magnify1DCamera(mag=4, size_factor=0.6, radius_ratio=0.6)
    else:
        view1.camera = 'panzoom'
    view2.camera = 'panzoom'
    view3.camera = 'panzoom'

    # initialize data in all subplots
    s_plot = scene.Line(pos=np.column_stack((frames, S_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))),
                                             color="#3c8a1b", width=1, parent=view1.scene)
    c_plot = scene.Line(pos=np.column_stack((frames, C_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))),
                                             color="#ff8b32", width=1, parent=view1.scene)
    ya_plot = scene.Line(pos=np.column_stack((frames, YA_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))),
                                              color="#a3a3a3", width=1, parent=view1.scene)
    
    s_pul_plot = scene.Line(pos=np.column_stack((pul_crd, s_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))),
                            color='red', width=2, parent=view2.scene)
    c_pul_plot = scene.Line(pos=np.column_stack((pul_crd, c_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))),
                            color='steelblue', width=2, parent=view2.scene)
    view2.camera.set_range()
    
    a_plot = scene.Image(A_dict[tuple(cur_params.values())][0,:,:], parent=view3.scene)
    view3.camera.set_range()
    
    # Slider configs
    cell_slider = QSlider(Qt.Horizontal)
    cell_slider.setRange(0, YA_dict[tuple(cur_params.values())].shape[0] - 1)
    cell_slider.setValue(0)
    layout.addWidget(cell_slider)

    param_slider_ls = []
    for param in params:
        param_slider = QSlider(Qt.Horizontal)
        param_slider.setRange(0, len(params[param]) - 1)
        param_slider.setValue(0)
        layout.addWidget(param_slider)
        param_slider_ls.append(param_slider)

    view1.camera.set_range()

    # Update which cell is plotted
    def update_cell(index):
        cur_cell[0] = units[index]
        
        s_plot.set_data(np.column_stack((frames, S_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
        c_plot.set_data(np.column_stack((frames, C_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
        ya_plot.set_data(np.column_stack((frames, YA_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))

        s_pul_plot.set_data(np.column_stack((pul_crd, s_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
        c_pul_plot.set_data(np.column_stack((pul_crd, c_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))

        a_plot.set_data(A_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))

        win.setWindowTitle(f'{cur_params};  cell: {units[index]}')
        canvas.update()

    update_cell(0)
    cell_slider.valueChanged.connect(update_cell)

    # Update parameters one by one
    def update_subplots():
        ya_plot.set_data(np.column_stack((frames, YA_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
        c_plot.set_data(np.column_stack((frames, C_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
        s_plot.set_data(np.column_stack((frames, S_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))

        s_pul_plot.set_data(np.column_stack((pul_crd, s_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
        c_pul_plot.set_data(np.column_stack((pul_crd, c_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
        win.setWindowTitle(f'{cur_params};  cell: {cur_cell}')

    def update_p(index):
        cur_params['p'] = params['p'][index]
        update_subplots()

    def update_sprs(index):
        cur_params['sparse_penal'] = params['sparse_penal'][index]
        update_subplots()

    def update_add(index):
        cur_params['add_lag'] = params['add_lag'][index]
        update_subplots()
    
    def update_noise(index):
        cur_params['noise_freq'] = params['noise_freq'][index]
        update_subplots()

    update_funcs = [update_p, update_sprs, update_add, update_noise]
    for i, param in enumerate(params):
        param_slider_ls[i].valueChanged.connect(update_funcs[i])

    win.setLayout(layout)
    win.show()
    qt_app.exec_()


def visualize_temporal_components(
        C=None,
        S=None,
        C_new=None,
        S_new=None,
        title='Temporal Update'
):
    
    qt_app = QApplication.instance()
    win = QWidget()
    win.setWindowTitle(title)
    layout = QVBoxLayout()
    win.setLayout(layout)
    
    # VisPy canvas and view
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
    layout.addWidget(canvas.native)
    grid = canvas.central_widget.add_grid()

    plot_coords = list(itt.product(range(2),range(2)))
    data_to_plot = {
        'C'     : C,
        'S'     : S,
        'C_new' : C_new,
        'S_new' : S_new,
    }
    view_ls = []
    for i, (var, data) in enumerate(data_to_plot.items()):
        view = scene.ViewBox(pos=(plot_coords[i]), parent=canvas.scene, border_color='black')
        view_ls.append(view)
        grid.add_widget(view_ls[i], plot_coords[i][0], plot_coords[i][1])
        if data is None:
            view_ls[i].camera = 'panzoom'
        else:
            width, height = data.sizes['frame'], data.sizes['unit_id']
            view_ls[i].camera = scene.PanZoomCamera(rect=((0, 0), (width, height)))
            plot = scene.Image(data, parent=view_ls[i].scene)

    for i in np.arange(1,4):
        view_ls[0].camera.link(view_ls[i].camera)
    
    win.show()
    qt_app.exec_()


def jackson_pollock_plot(
        A_array,
        max_proj,
        title,
        method='maxidx',
        threshold=0,
        cm=colormap.get_colormap('Spectral_r'),
        alpha=0.7
):
    rand_color = np.random.choice(np.arange(1,A_array.shape[0]+1), A_array.shape[0], replace=False)

    if method == 'forloop':
        maxA = A_array.max('unit_id').values
        for i in range(A_array.shape[0]):
            maxA[(A_array[i,:,:]>threshold)] = rand_color[i]

    elif method == 'matmul':
        A_array = (A_array.values > threshold)
        maxA = (A_array.T * rand_color).T.max(axis=0).astype(np.float32)

    elif method == 'maxidx':
        rand_color -= 1
        A_array = A_array.values[rand_color,:,:]
        maxA = np.argmax(A_array, axis=0).astype(np.float32)
    else:
        raise Exception('Invalid method chosen.')

    # convert maxA colors to remove space where there are no cells
    maxA[maxA <= threshold] = np.nan
    maxA = normalize(maxA)
    maxA = np.array([cm[maxA[i,:]] for i in np.arange(maxA.shape[0])])
    maxA[:,:,-1] = alpha

    # begin plotting
    qt_app = QApplication.instance()
    win = QWidget()
    win.setWindowTitle(title)
    layout = QVBoxLayout()
    win.setLayout(layout)

    canvas = scene.SceneCanvas(keys='interactive')
    layout.addWidget(canvas.native)
    grid = canvas.central_widget.add_grid(spacing=1)

    max_proj_view = scene.ViewBox(pos=(0,0), border_color='white', parent=canvas.scene)
    a_view        = scene.ViewBox(pos=(0,1), border_color='white', parent=canvas.scene)
    grid.add_widget(max_proj_view, 0, 0)
    grid.add_widget(a_view, 0, 1)

    max_proj_im = scene.Image(max_proj.astype(np.float32), cmap='gray', parent=max_proj_view.scene)
    a_im        = scene.Image(maxA.astype(np.float32), cmap='Spectral_r', parent=a_view.scene)

    max_proj_view.camera = 'panzoom'
    a_view.camera        = 'panzoom'
    max_proj_view.camera.aspect = 1
    a_view.camera.aspect = 1
    max_proj_view.camera.link(a_view.camera)
    max_proj_view.camera.set_range()
    a_view.camera.set_range()

    win.show()
    qt_app.exec_()


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