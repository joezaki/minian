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
from vispy.visuals.transforms import STTransform
from vispy.scene.cameras import Magnify1DCamera
from vispy.color import colormap
from vispy import gloo
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
use('pyqt5')

from .cnmf import compute_AtC, smooth_sig
from .utilities import custom_arr_optimize

class Vis:
    '''
    A general class to create a VisPy canvas and populate
    it with subplots, to be used with the minian pipeline.
    '''

    def __init__(
            self,
            title,
            width=1000,
            height=500,
            bgcolor='white',
            grid_margin=0,
            grid_padding=20,
            grid_spacing=20
    ):
        self.title = title
        self.qt_app = QApplication.instance()
        self.win = QWidget()
        self.win.setWindowTitle(title)
        self.layout = QVBoxLayout()
        self.win.setLayout(self.layout)

        self.width, self.height = (width, height)
        
        # VisPy canvas and view
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(self.width,self.height),
            bgcolor=bgcolor
            )
        self.canvas._send_hover_events = True
        self.layout.addWidget(self.canvas.native)
        self.grid = self.canvas.central_widget.add_grid(
            margin=grid_margin,
            padding=grid_padding,
            spacing=grid_spacing
            )

        # dictionary of all views and axes to be added
        self.view_coords = {}
        self.view_dict = {}

        self.max_gl_size = gloo.gl.glGetParameter(gloo.gl.GL_MAX_TEXTURE_SIZE)



    def show(self):
        '''
        Render canvas with any additional plots in a pop-up window.
        '''
        
        self.win.show()
        self.qt_app.exec_()
    


    def add_axes_view(
            self,
            name,
            x_label=None,
            y_label=None,
            col_span=1,
            row_span=1,
            magnify=False,
            add_xaxis=True,
            add_yaxis=True
            ):
        '''
        Add x- and y-axes to a given subplot, given the view
        of that subplot.
        '''

        row, col = self.view_coords[name]
        nested_grid = self.grid.add_grid(row, col, row_span=row_span, col_span=col_span)
        nested_grid.spacing = 0

        # Add X and Y axes
        axis_kwargs = dict(
            text_color='black',
            tick_color='black',
            tick_width=1,
            axis_width=2,
            major_tick_length=5,
            minor_tick_length=3
        )

        if add_xaxis:
            xaxis = scene.AxisWidget(orientation='bottom', axis_label=x_label, **axis_kwargs)
            xaxis.height_max = 40
            nested_grid.add_widget(xaxis, row=1, col=1)
        
        if add_yaxis:
            yaxis = scene.AxisWidget(orientation='left', axis_label=y_label, **axis_kwargs)
            yaxis.width_max = 40
            nested_grid.add_widget(yaxis, row=0, col=0)

        # Link axes to the view
        view = nested_grid.add_view(row=0, col=1)
        view.border_color = 'black'
        if magnify:
            view.camera = Magnify1DCamera(mag=1, size_factor=2*col_span, radius_ratio=1)
        else:
            view.camera = scene.cameras.PanZoomCamera()

        if add_xaxis:
            xaxis.link_view(view)
        if add_yaxis:
            yaxis.link_view(view)

        return view
    

    def draw_multiscale_image(
            self,
            data,
            view
    ):
        '''
        If a matrix is larger than the GL_MAX_TEXTURE_SIZE limit, use this
        to plot the same image as multiple smaller images tiled together.
        '''

        frames = data.sizes['frame']
        unit_ids = data.sizes['unit_id']
        clim = (data.min().values, data.max().values)

        # compute tile indices
        frame_tiles = np.ceil(frames / self.max_gl_size)
        unit_id_tiles = np.ceil(unit_ids / self.max_gl_size)
        frame_indices = np.array_split(data['frame'], frame_tiles)
        unit_id_indices = np.array_split(data['unit_id'], unit_id_tiles)

        # plot all tiles
        image_ls = []
        for frame_tile in frame_indices:
            for unit_id_tile in unit_id_indices:
                image = scene.Image(data.sel(
                    frame=frame_tile,
                    unit_id=unit_id_tile),
                    clim=clim,
                    parent=view.scene
                    )
                image.transform = STTransform(translate=(frame_tile[0],unit_id_tile[0]))
                image_ls.append(image)
        

    
    def visualize_raw_video(
            self, 
            varr
            ):
        '''
        Visualize original raw video along with histogram of pixel values for
        each frame, as well as the mean, min, and max value across the video.
        '''
        cur_frame = varr.sel(frame=0)

        self.view_coords = {'image':(0,0), 'hist':(0,2), 'line':(1,0)}
        self.axis_labels = {'image':{'x':'width', 'y':'height'},
                            'hist': {'x':'frequency', 'y':'fluorescence'},
                            'line': {'x':'frame', 'y':'fluorescence'}}
        col_spans = {'image':2, 'hist':1, 'line':2}
        self.view_dict = {name:self.add_axes_view(
            name=name,
            x_label=self.axis_labels[name]['x'],
            y_label=self.axis_labels[name]['y'],
            col_span=col_spans[name]) for name in self.view_coords.keys()}
        
        frame_im = scene.Image(cur_frame, parent=self.view_dict['image'].scene)
        histogram = visuals.Histogram(
            data=cur_frame.stack(stacked_dims=['height','width']),
            bins=50,
            orientation='v',
            color='darkblue',
            parent=self.view_dict['hist'].scene
        )
        
        frames = varr.frame
        min_vals = scene.Line(
            pos=np.column_stack((frames, varr.min(['height', 'width']))),
            color='#ff7f0e', width=2, parent=self.view_dict['line'].scene
        )
        max_vals = scene.Line(
            pos=np.column_stack((frames, varr.max(['height', 'width']))),
            color='#1f77b4', width=2, parent=self.view_dict['line'].scene
        )
        mean_vals = scene.Line(
            pos=np.column_stack((frames, varr.mean(['height', 'width']))),
            color='black', width=2, parent=self.view_dict['line'].scene
        )
        cur_vline = scene.Line(
            pos=np.column_stack(((0,0), (-20, 320))),
            color='red', width=2, parent=self.view_dict['line'].scene
        )
        
        self.view_dict['image'].camera.rect = (0, 0, cur_frame.sizes['width'], cur_frame.sizes['height'])
        self.view_dict['image'].camera.aspect = 1
        self.view_dict['hist'].camera.set_range(y=(0,255))
        self.view_dict['line'].camera.set_range(x=(frames[0],frames[-1]))

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, varr.sizes['frame'] - 1)
        slider.setValue(0)
        self.layout.addWidget(slider)
        
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
                parent=self.view_dict['hist'].scene)
            self.view_dict['hist'].camera.set_range(y=(0,255))

            cur_vline.set_data(np.column_stack(((index,index),(-20,320))))

            self.win.setWindowTitle(f'frame = {index}')
            self.canvas.update()

        slider.valueChanged.connect(update_plot)



    def visualize_before_after(
            self,
            before,
            after,
            scale_image=False
    ):
        '''
        Visualize two side-by-side images or videos before (left) and
        after (right) of a transformation.
        '''        

        # if objects are just images, add a frame dim
        if 'frame' not in before.dims:
            before = before.expand_dims({'frame': 1})
        if 'frame' not in after.dims:
            after = after.expand_dims({'frame': 1})

        # create views
        self.view_coords = {'before':(0,0), 'after':(0,1)}
        before_view = self.add_axes_view(name='before', x_label='width', y_label='height')
        after_view = self.add_axes_view(name='after', x_label='width', y_label='height')

        # add images to views
        before_im = scene.Image(before.sel(frame=0), parent=self.canvas.scene)
        before_view.add(before_im)
        after_im = scene.Image(after.sel(frame=0), parent=before_view)
        after_view.add(after_im)

        before_view.camera.link(after_view.camera)
        before_view.camera.rect = (0, 0, before.sizes['width'], before.sizes['height'])
        after_view.camera.rect = (0, 0, after.sizes['width'], after.sizes['height'])
        if not scale_image:
            before_view.camera.aspect=1
            after_view.camera.aspect=1

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, before.sizes['frame'] - 1)
        slider.setValue(0)
        self.layout.addWidget(slider)
        
        # Update line on slider change
        def update_plot(index):
            before_im.set_data(before.isel(frame=index))
            after_im.set_data(after.isel(frame=index))
            self.win.setWindowTitle(f'{self.title}: frame = {index}')
            self.canvas.update()

        slider.valueChanged.connect(update_plot)
    


    def visualize_preprocess(
            self,
            frame,
            func,
            scale_image=False,
            **kwargs
    ):
        
        width  = frame.sizes['width']
        height = frame.sizes['height']
        
        # create list of processed images and subtitles
        pkey = kwargs.keys()
        pval = kwargs.values()
        image_ls = [func(frame, **dict(zip(pkey, params))) for params in itt.product(*pval)]
        title_ls = [str(dict(zip(pkey, params))) for params in itt.product(*pval)]

        self.view_coords = {
            'orig'             : (0,0),
            'orig_contour'     : (0,1),
            'processed'        : (1,0),
            'processed_contour': (1,1)
        }
        orig_view      = self.add_axes_view('orig', x_label='width', y_label='height')
        orig_cont_view = self.add_axes_view('orig_contour', x_label='width', y_label='height')
        proc_view      = self.add_axes_view('processed', x_label='width', y_label='height')
        proc_cont_view = self.add_axes_view('processed_contour', x_label='width', y_label='height')

        # add original image (subplot 1)
        orig_image = scene.Image(
            image_ls[0],
            parent=orig_view.scene
            )

        # add original contour (subplot 2)
        orig_contour = scene.Image(
            image_ls[0],
            interpolation='cubic',
            parent=orig_cont_view.scene
            )
        iso = IsolineFilter(level=5, width=2, color='white')
        orig_contour.attach(iso)

        # add processed image (subplot 3)
        processed_image = scene.Image(
            image_ls[0],
            parent=proc_view.scene
            )
        
        # add processed image contour (subplot 4)
        processed_contour = scene.Image(
            image_ls[0],
            interpolation='cubic',
            parent=proc_cont_view.scene
            )
        iso = IsolineFilter(level=5, width=2, color='white')
        processed_contour.attach(iso)
        
        # share axes
        orig_view.camera.set_range()
        orig_view.camera.link(orig_cont_view.camera)
        orig_view.camera.link(proc_view.camera)
        orig_view.camera.link(proc_cont_view.camera)

        if not scale_image:
            orig_view.camera.aspect = 1
            orig_cont_view.camera.aspect = 1
            proc_view.camera.aspect = 1
            proc_cont_view.camera.aspect = 1
        orig_view.camera.rect      = (0, 0, frame.sizes['width'], frame.sizes['height'])
        orig_cont_view.camera.rect = (0, 0, frame.sizes['width'], frame.sizes['height'])
        proc_view.camera.rect      = (0, 0, frame.sizes['width'], frame.sizes['height'])
        proc_cont_view.camera.rect = (0, 0, frame.sizes['width'], frame.sizes['height'])

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, len(image_ls) - 1)
        slider.setValue(0)
        self.layout.addWidget(slider)
        
        # Update line on slider change
        def update_plot(index):
            processed_image.set_data(image_ls[index])
            processed_contour.set_data(image_ls[index])
            self.win.setWindowTitle(f'{self.title}: {title_ls[index]}')
            self.canvas.update()

        slider.valueChanged.connect(update_plot)



    def visualize_motion(
            self,
            motion,
            magnify=False
    ):

        # Add a ViewBox with pan/zoom
        self.view_coords = {'motion': (0,0)}
        view = self.add_axes_view('motion', x_label='frame', y_label='motion', magnify=magnify)

        frames = motion.frame
        width_line = scene.Line(pos=np.column_stack((frames, motion.sel(shift_dim="width"))),
                                color='#ff7f0e', width=1, parent=view.scene)
        height_line = scene.Line(pos=np.column_stack((frames, motion.sel(shift_dim="height"))),
                                color='#1f77b4', width=1, parent=view.scene)
        view.camera.set_range()



    def visualize_seeds(
            self,
            max_proj,
            seeds,
            mask=None,
            marker_scaling=1.5
    ):
        
        if mask is None:
            mask = np.repeat(True, seeds.shape[0])
        else:
            mask = seeds[mask]
        good_seeds = seeds[mask].copy()
        bad_seeds  = seeds[np.invert(mask)].copy()

        self.view_coords = {'seeds':(0,0)}
        view = self.add_axes_view('seeds', x_label='width', y_label='height')

        # add seeds
        if good_seeds.shape[0] > 0:
            good_seeds_scatter = visuals.Markers()
            good_seeds_scatter.set_data(pos=good_seeds[['width','height']].values,
                                        edge_width=0, face_color=('white'),
                                        size=good_seeds['seeds']*marker_scaling, symbol='o')
            good_seeds_scatter.antialias = 0
            view.add(good_seeds_scatter)
        if bad_seeds.shape[0] > 0:
            bad_seeds_scatter = visuals.Markers()
            bad_seeds_scatter.set_data(pos=bad_seeds[['width','height']].values,
                                       edge_width=0, face_color=('red'),
                                       size=bad_seeds['seeds']*marker_scaling, symbol='o')
            bad_seeds_scatter.antialias = 0
            view.add(bad_seeds_scatter)

        # add max proj
        max_proj_im = scene.Image(max_proj, parent=view.scene)

        view.camera.rect = (0, 0, max_proj.sizes['width'], max_proj.sizes['height'])
        view.camera.aspect = 1
    


    def visualize_pnr_refine(
            self,
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
        
        # start plotting
        seeds = example_seeds.index
        view_dict = {}
        plot_coords = list(itt.product(range(int(np.ceil(len(seeds)/cols))),range(cols)))
        self.view_coords = {seed:coord for seed, coord in zip(seeds, plot_coords)}
        frames = Y_hw_chk.frame

        low_lines_dict = {}
        high_lines_dict = {}
        for i, seed in enumerate(seeds):
            view_dict[seed] = self.add_axes_view(name=seed, magnify=magnify)
            low_lines_dict[seed] = scene.Line(
                pos=np.column_stack((frames, arrays_dict[noise_freq_list[0]]['low'][i])),
                color='#ff7f0e',
                width=1,
                parent=view_dict[seed].scene
                )
            high_lines_dict[seed] = scene.Line(
                pos=np.column_stack((frames, arrays_dict[noise_freq_list[0]]['high'][i])),
                color='#1f77b4',
                width=1,
                parent=view_dict[seed].scene
                )

        # link views
        if link_views:
            for seed in seeds:
                view_dict[seed].camera.link(view_dict[seeds[0]].camera)

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, len(noise_freq_list) - 1)
        slider.setValue(0)
        self.layout.addWidget(slider)
        
        # Update line on slider change
        def update_plot(index):
            for i, seed in enumerate(seeds):
                low_data = arrays_dict[noise_freq_list[index]]['low'][i]
                high_data = arrays_dict[noise_freq_list[index]]['high'][i]
                low_lines_dict[seed].set_data(pos=np.column_stack((frames, low_data)))
                high_lines_dict[seed].set_data(pos=np.column_stack((frames, high_data)))
                view_dict[seed].camera.set_range(y=(min(low_data.min(),high_data.min()),
                                            max(low_data.max(),high_data.max())))
            self.win.setWindowTitle(f'noise frequency: {noise_freq_list[index]}')
            self.canvas.update()

        update_plot(0)
        slider.valueChanged.connect(update_plot)
    


    def visualize_initialization(
            self,
            A,
            C,
            b,
            f,
            multiscale=True,
    ):

        data_to_plot = {
            'A': A.max("unit_id").compute().astype(np.float32),
            'C': C.compute().astype(np.float32),
            'b': b.compute().astype(np.float32),
            'f': f.compute().astype(np.float32)
        }

        # add data to subplots
        col_spans = {'A':1, 'C':2, 'b':1, 'f':2}
        plot_coords = list(itt.product(range(2),range(2)))
        self.view_coords = {name:coord for name, coord in zip(data_to_plot.keys(), plot_coords)}
        x_labels = {'A': 'width', 'C': 'frame', 'b': 'width', 'f': 'frame'}
        y_labels = {'A': 'height', 'C': 'unit_id', 'b': 'height', 'f': 'f'}

        view_dict = {name:self.add_axes_view(name, col_span=col_spans[name],
                                             x_label=x_labels[name],
                                             y_label=y_labels[name]) for name in data_to_plot.keys()}

        a_plot = scene.Image(data_to_plot['A'], parent=view_dict['A'].scene)
        b_plot = scene.Image(data_to_plot['b'], parent=view_dict['b'].scene)
        if multiscale:
            self.draw_multiscale_image(data=data_to_plot['C'], view=view_dict['C'])
        else:
            scene.Image(data_to_plot['C'], parent=view_dict['C'].scene)
        scene.Line(
            pos=np.column_stack((f.frame, f)),
            color="#07117b", width=1, parent=view_dict['f'].scene
        )

        view_dict['A'].camera.rect = (0, 0, A.sizes['width'], A.sizes['height'])
        view_dict['b'].camera.rect = (0, 0, b.sizes['width'], b.sizes['height'])
        view_dict['C'].camera.rect = (0, 0, C.sizes['frame'], C.sizes['unit_id'])
        view_dict['f'].camera.set_range()
        
        # link the spatial subplots together
        view_dict['A'].camera.aspect = 1
        view_dict['b'].camera.aspect = 1
        view_dict['A'].camera.link(view_dict['b'].camera)
    


    def visualize_spatial_params(
            self,
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

        self.view_coords = {'a_binary':(0,0),
                            'a_cont'  :(1,0),
                            'temporal':(0,1)}
        a_bin_view = self.add_axes_view(
            name='a_binary',
            x_label='width',
            y_label='height',
            row_span=1)
        a_cont_view = self.add_axes_view(
            name='a_cont',
            x_label='width',
            y_label='height',
            row_span=1)
        temp_view = self.add_axes_view(
            name='temporal',
            x_label='frame',
            row_span=2)
        
        a_bin_view.camera.link(a_cont_view.camera)
        a_bin_view.camera.aspect = 1
        a_cont_view.camera.aspect = 1

        # plot footprints
        A_binary = scene.Image((A_dict[sprs_ls[0]] > 0).sum("unit_id").astype(np.float32), parent=a_bin_view.scene)

        A_cont = scene.Image(A_dict[sprs_ls[0]].sum("unit_id").astype(np.float32), parent=a_cont_view.scene)
        a_bin_view.camera.rect = (0, 0, A_dict[sprs_ls[0]].sizes['width'], A_dict[sprs_ls[0]].sizes['height'])
        a_cont_view.camera.rect = (0, 0, A_dict[sprs_ls[0]].sizes['width'], A_dict[sprs_ls[0]].sizes['height'])

        # plot temporal components
        C_ls = []
        for i, unit in enumerate(units):
            line = scene.Line(pos=np.column_stack((C_dict[sprs_ls[0]].frame,
                                                C_dict[sprs_ls[0]][i,:]+i)),
                                                color='#07117b', width=1, parent=temp_view.scene)
            text = scene.visuals.Text(str(unit), color='black', parent=temp_view.scene)
            text.font_size= 10
            text.pos = -10, i+0.5
            C_ls.append(line)
        temp_view.camera.set_range()

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, len(sprs_ls) - 1)
        slider.setValue(0)
        self.layout.addWidget(slider)
        
        # Update line on slider change
        def update_plot(index):
            A_binary.set_data((A_dict[sprs_ls[index]] > 0).sum("unit_id"))
            A_cont.set_data(A_dict[sprs_ls[index]].sum("unit_id"))
            for i, unit in enumerate(units):
                C_ls[i].parent = None
                if unit in C_dict[sprs_ls[index]].unit_id:
                    C_ls[i] = scene.Line(pos=np.column_stack((C_dict[sprs_ls[index]].frame,
                                                        C_dict[sprs_ls[index]].sel(unit_id=unit)+i)),
                                                        color='#07117b', width=1, parent=temp_view.scene)
            self.win.setWindowTitle(f'sparse penalty: {sprs_ls[index]}')
            self.canvas.update()

        update_plot(0)
        slider.valueChanged.connect(update_plot)



    def visualize_spatial_update(
            self,
            A,
            A_new
    ):

        plot_coords = list(itt.product(range(2),range(2)))
        data_to_plot = {
            'A'        : A.max("unit_id").compute().astype(np.float32),
            'A_bin'    : (A.fillna(0) > 0).sum("unit_id").compute().astype(np.uint8),
            'A_new'    : A_new.max("unit_id").compute().astype(np.float32),
            'A_new_bin': (A_new > 0).sum("unit_id").compute().astype(np.uint8)
        }
        self.view_coords = {name:coord for name, coord in zip(data_to_plot.keys(), plot_coords)}
        view_ls = []
        for i, (name, data) in enumerate(data_to_plot.items()):
            view = self.add_axes_view(name, x_label='width', y_label='height')
            view_ls.append(view)

            plot = scene.Image(data, parent=view_ls[i].scene)
            view.camera.rect = (0, 0, data.sizes['width'], data.sizes['height'])
            view.camera.aspect = 1

        for i in np.arange(1,4):
            view_ls[0].camera.link(view_ls[i].camera)



    def visualize_spatial_bg(
            self,
            b,
            f,
            b_new,
            f_new
    ):

        plot_coords = list(itt.product(range(2),range(2)))
        data_to_plot = {
            'b'     : b.compute().astype(np.float32),
            'f'     : f.compute().astype(np.float16),
            'b_new' : b_new.compute().astype(np.float32),
            'f_new' : f_new.compute().astype(np.float16)
        }
        self.view_coords = {name:coord for name, coord in zip(data_to_plot.keys(), plot_coords)}
        view_ls = []
        col_spans = {'b':1, 'f':2, 'b_new':1, 'f_new':2}
        x_labels = {'b':'width', 'f':'frame', 'b_new':'width', 'f_new':'frame'}
        y_labels = {'b':'height','f':'f',     'b_new':'height','f_new':'f'}
        for i, (name, data) in enumerate(data_to_plot.items()):
            view = self.add_axes_view(name,
                                      col_span=col_spans[name],
                                      x_label=x_labels[name],
                                      y_label=y_labels[name])
            view_ls.append(view)

            if name[0] == 'f':
                plot = scene.Line(pos=np.column_stack((data.frame, data)),
                                color='#07117b', width=1, parent=view_ls[i].scene)
                view_ls[i].camera.set_range()
            else:
                plot = scene.Image(data, parent=view_ls[i].scene)
                view_ls[i].camera.aspect = 1
                view_ls[i].camera.rect = (0, 0, data.sizes['width'], data.sizes['height'])
        
        # temporarily link spatial and temporal plots by index
        view_ls[0].camera.link(view_ls[2].camera)
        view_ls[1].camera.link(view_ls[3].camera)



    def visualize_temporal_params(
            self,
            units,
            params,
            YA_dict,
            C_dict,
            S_dict,
            g_dict,
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
            'S'  : S_dict
        }

        if norm:
            for var in activities_dict.keys():
                if activities_dict[var] is not None:
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
        if g_dict is not None:
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
        self.view_coords = {'temporal' :(0,0),
                            'pulse'    :(1,0),
                            'footprint':(1,1)}
        col_spans = {'temporal'  :2,
                     'pulse'     :1,
                     'footprint' :1}
        temp_view  = self.add_axes_view(
            name='temporal',
            x_label='frame',
            y_label='Intensity (A.U.)',
            col_span=col_spans['temporal'],
            magnify=magnify
            )
        if g_dict is not None:
            pulse_view = self.add_axes_view(
                name='pulse',
                x_label='t',
                y_label='Response (A.U.)',
                col_span=col_spans['pulse']
                )
        footprint_view = self.add_axes_view(
            name='footprint',
            x_label='width',
            y_label='height',
            col_span=col_spans['footprint']
            )

        # initialize data in all subplots
        if activities_dict['S'] is not None:
            s_plot = scene.Line(pos=np.column_stack((frames, S_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))),
                                                    color="#3c8a1b", width=1, parent=temp_view.scene)
        c_plot = scene.Line(pos=np.column_stack((frames, C_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))),
                                                color="#ff8b32", width=1, parent=temp_view.scene)
        ya_plot = scene.Line(pos=np.column_stack((frames, YA_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))),
                                                color="#a3a3a3", width=1, parent=temp_view.scene)
        
        if g_dict is not None:
            s_pul_plot = scene.Line(pos=np.column_stack((pul_crd, s_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))),
                                    color='red', width=2, parent=pulse_view.scene)
            c_pul_plot = scene.Line(pos=np.column_stack((pul_crd, c_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))),
                                    color='steelblue', width=2, parent=pulse_view.scene)
            pulse_view.camera.set_range()
        
        cur_footprint = A_dict[tuple(cur_params.values())][0,:,:]
        a_plot = scene.Image(cur_footprint, parent=footprint_view.scene)
        footprint_view.camera.rect = (0, 0, cur_footprint.sizes['width'], cur_footprint.sizes['height'])

        # Slider configs
        cell_slider = QSlider(Qt.Horizontal)
        cell_slider.setRange(0, len(units) - 1)
        cell_slider.setValue(0)
        self.layout.addWidget(cell_slider)

        param_slider_ls = []
        for param in params:
            param_slider = QSlider(Qt.Horizontal)
            param_slider.setRange(0, len(params[param]) - 1)
            param_slider.setValue(0)
            self.layout.addWidget(param_slider)
            param_slider_ls.append(param_slider)

        temp_view.camera.set_range()

        # Update which cell is plotted
        def update_cell(index):
            cur_cell[0] = units[index]
            if activities_dict['S'] is not None:
                s_plot.set_data(np.column_stack((frames, S_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
            c_plot.set_data(np.column_stack((frames, C_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
            ya_plot.set_data(np.column_stack((frames, YA_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))

            if g_dict is not None:
                s_pul_plot.set_data(np.column_stack((pul_crd, s_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
                c_pul_plot.set_data(np.column_stack((pul_crd, c_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))

            a_plot.set_data(A_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))

            self.win.setWindowTitle(f'{cur_params};  cell: {cur_cell[0]}')
            self.canvas.update()

        update_cell(0)
        cell_slider.valueChanged.connect(update_cell)

        # Update parameters one by one
        def update_subplots():
            ya_plot.set_data(np.column_stack((frames, YA_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
            c_plot.set_data(np.column_stack((frames, C_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
            s_plot.set_data(np.column_stack((frames, S_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))

            if g_dict is not None:
                s_pul_plot.set_data(np.column_stack((pul_crd, s_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
                c_pul_plot.set_data(np.column_stack((pul_crd, c_pul_dict[tuple(cur_params.values())].sel(unit_id=cur_cell[0]))))
            self.win.setWindowTitle(f'{cur_params};  cell: {cur_cell[0]}')

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



    def visualize_temporal_components(
            self,
            components_dict,
            cols=2,
            multiscale=True
    ):
        
        plot_coords = list(itt.product(range(int(np.ceil(len(components_dict)/cols))),range(cols)))
        self.view_coords = {name:coord for name, coord in zip(components_dict.keys(), plot_coords)}

        unit_id_max = np.max([data['unit_id'].max() for data in components_dict.values() if data is not None])
        frame_max   = np.max([data['frame'].max()   for data in components_dict.values() if data is not None])

        view_ls = []
        for i, (name, data) in enumerate(components_dict.items()):
            view = self.add_axes_view(name, x_label='frame', y_label='unit_id')
            view_ls.append(view)
            if data is not None:
                if multiscale:
                    self.draw_multiscale_image(data=data, view=view_ls[i])
                else:
                    scene.Image(data, parent=view_ls[i].scene)
                view.camera.rect = (0, 0, frame_max, unit_id_max)

        for i in np.arange(1,len(components_dict)):
            view_ls[0].camera.link(view_ls[i].camera)



    def jackson_pollock_plot(
            self,
            max_proj,
            A,
            method='maxidx',
            threshold=0,
            cm=colormap.get_colormap('Spectral_r'),
            alpha=0.7
    ):
        
        rand_color = np.random.choice(np.arange(1,A.shape[0]+1), A.shape[0], replace=False)

        if method == 'forloop':
            maxA = A.max('unit_id').values
            for i in range(A.shape[0]):
                maxA[(A[i,:,:]>threshold)] = rand_color[i]

        elif method == 'matmul':
            A = (A.values > threshold)
            maxA = (A.T * rand_color).T.max(axis=0).astype(np.float32)

        elif method == 'maxidx':
            rand_color -= 1
            A = A.values[rand_color,:,:]
            maxA = np.argmax(A, axis=0).astype(np.float32)
        else:
            raise Exception('Invalid method chosen.')

        # convert maxA colors to remove space where there are no cells
        maxA[maxA <= threshold] = np.nan
        maxA = normalize(maxA)
        maxA = np.array([cm[maxA[i,:]] for i in np.arange(maxA.shape[0])])
        maxA[:,:,-1] = alpha

        # start plotting
        self.view_coords = {'max_proj':(0,0), 'A':(0,1)}
        max_proj_view = self.add_axes_view('max_proj', x_label='width', y_label='height')
        a_view        = self.add_axes_view('A', x_label='width', y_label='height')

        max_proj_im = scene.Image(max_proj.astype(np.float32), cmap='gray', parent=max_proj_view.scene)
        a_im        = scene.Image(maxA.astype(np.float32), cmap='Spectral_r', parent=a_view.scene)

        max_proj_view.camera.aspect = 1
        a_view.camera.aspect = 1
        max_proj_view.camera.link(a_view.camera)
        width, height = (max_proj.sizes['width'], max_proj.sizes['height'])
        max_proj_view.camera.rect = (0, 0, width, height)
        a_view.camera.rect = (0, 0, width, height)


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