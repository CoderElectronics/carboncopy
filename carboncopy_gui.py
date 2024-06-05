"""
Carbon Copy by Ari Stehney

Scan in carbon pieces to a DXF file.
"""

from PIL import Image
from nicegui import ui, events
import io, base64, math, re, sys, os

from skimage.morphology import convex_hull_image
from skimage.util import invert
from skimage import feature
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.filters import threshold_otsu
import skimage

from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import ezdxf
from ezdxf.addons.drawing.properties import LayoutProperties
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

import timeit
from threading import Timer

# Fix recursion overflow errors
sys.setrecursionlimit(32768)

# State variables
topx, topy, botx, boty = 0, 0, 0, 0
p1x, p1y, p2x, p2y = 0, 0, 0, 0
btn_area_cntr = True
btn_corner_cntr = True
btn_r_mode = False
in_known_dist = 0

loaded_b64, loaded_rgba, loaded_b64_type = None, None, None
im_cropped, im_bin, im_final = None, None, None
doc_buf = None
im_thresh_s = 128

im_poly_alpha = 0.03
im_poly_beta = 0.25
im_poly_gamma = 0.035
im_poly_scalar = 1
im_poly_gauss_sigma = 3

def regressive_polyimg_to_dxf_multi_spline(img, msp, poly_scalar=1, s_foreground_thresh=160, alpha=0.05, beta=15,
                                           gamma=0.035, gaussian_sigma=3, preview=False, pass_bool=None, dppi=1):
    im_gray = np.array(img.convert('L'))
    im_gray_thresh = im_gray.copy()
    im_gray_thresh[im_gray_thresh < s_foreground_thresh] = 0
    im_bool = im_gray < s_foreground_thresh

    if pass_bool is not None:
        im_bool = pass_bool

    im_bin = im_bool.astype(np.uint8)

    labels, num = skimage.morphology.label(im_bin, background=None, return_num=True, connectivity=2)

    splinelst = []

    for l_num in range(1, num + 1):
        tr_arr = (labels == l_num) * 1.
        tr_bool = np.invert(ndimage.binary_fill_holes(labels == l_num))

        convimg = convex_hull_image(tr_arr)
        cent_pt = ndimage.measurements.center_of_mass(tr_arr)
        cent_conv_pt = np.array(ndimage.measurements.center_of_mass(convimg))

        edges = feature.canny(convimg, sigma=gaussian_sigma)

        indices = np.where(edges == [1])
        coordinates = [list(a) for a in zip(indices[0], indices[1])]
        coordinates.sort(key=lambda p: math.atan2(p[1] - cent_pt[1], p[0] - cent_pt[0]))

        init = (coordinates - cent_conv_pt) * poly_scalar + cent_conv_pt

        snake = active_contour(
            gaussian(im_gray_thresh, sigma=gaussian_sigma, preserve_range=False),
            init,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        tr_inside = tr_arr.copy()
        tr_inside[tr_bool] = invert(tr_inside[tr_bool])
        tr_inside = invert(tr_inside)

        if preview:
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))

            ax[1].imshow(tr_inside, cmap=plt.cm.gray)

            ax[0].imshow(im_bin, cmap=plt.cm.gray)
            ax[0].plot(init[:, 1], init[:, 0], '--r', lw=3)
            ax[0].plot(snake[:, 1], snake[:, 0], '-b', lw=3)
            ax[0].scatter([cent_pt[1]], [cent_pt[0]])
            ax[0].set_xticks([]), ax[0].set_yticks([])
            ax[0].axis([0, im_gray_thresh.shape[1], im_gray_thresh.shape[0], 0])

            plt.show()

        fit_points = [(ptt[1]*dppi, ptt[0]*dppi, 0) for ptt in snake] + [(snake[0][1]*dppi, snake[0][0]*dppi, 0)]
        splinelst.append(msp.add_spline(fit_points))

        regressive_polyimg_to_dxf_multi_spline(img, msp,
                                               poly_scalar=poly_scalar,
                                               s_foreground_thresh=s_foreground_thresh,
                                               alpha=alpha,
                                               beta=beta,
                                               gamma=gamma,
                                               gaussian_sigma=gaussian_sigma,
                                               preview=preview,
                                               pass_bool=tr_inside.astype(bool),
                                               dppi=dppi)

    return splinelst

def procDxf():
    global im_cropped, im_bin, im_thresh_s, in_known_dist, p1x, p1y, p2x, p2y, topx, topy, botx, boty, im_final, doc_buf

    # im_thresh_s : threshold
    # in_known_dist: float mm
    # p1x, p1y, p2x, p2y : known dim values
    # topx, topy, botx, boty : crop values

    #print("p1: ({}, {}), p2: ({}, {})".format(p1x, p1y, p2x, p2y))

    planar_dist = float(in_known_dist)
    pixel_dist = math.sqrt((p2x - p1x)**2 + (p2y - p1y)**2)
    dpp = planar_dist / pixel_dist

    #print("planar: {}, pixel: {}, dpp: {}".format(planar_dist, pixel_dist, dpp))

    #print("alpha: {}, beta: {}, gamma: {}, gaussian_sigma: {}, poly_scalar: {}".format(im_poly_alpha, im_poly_beta, im_poly_gamma, im_poly_gauss_sigma, im_poly_scalar))

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    splist = regressive_polyimg_to_dxf_multi_spline(im_cropped, msp,
                                                    poly_scalar=im_poly_scalar,
                                                    s_foreground_thresh=im_thresh_s,
                                                    alpha=im_poly_alpha, # 0.03
                                                    beta=im_poly_beta, # 0.25
                                                    gamma=im_poly_gamma, # 0.035
                                                    gaussian_sigma=im_poly_gauss_sigma,
                                                    preview=False,
                                                    dppi=dpp)

    ctx = RenderContext(doc)
    msp_properties = LayoutProperties.from_layout(msp)
    msp_properties.set_colors("#eaeaea")

    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1))
    out = MatplotlibBackend(ax)

    # override the layout properties and render the modelspace
    Frontend(ctx, out).draw_layout(
        msp,
        finalize=True,
        layout_properties=msp_properties,
    )

    # cache image for GUI
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    im_final = Image.open(img_buf)

    # cache DXF for download
    doc.saveas("cache.dxf")

    with open("cache.dxf", "rb") as fh:
        doc_buf = fh.read()

    os.remove("cache.dxf")
    
# Create Widgets

@ui.refreshable
def panel_tweak_binarization():
    global im_thresh_s
    @ui.refreshable
    def binarImage():
        global im_gray, im_bool, im_bin, im_thresh_s, im_cropped
        im_gray = np.array(im_cropped.convert('L'))
        im_bool = im_gray > im_thresh_s
        im_bin = (im_gray > im_thresh_s) * 255

        img_updated = Image.fromarray(np.uint8(im_bin))

        ui.image(img_updated).props('fit=scale-down')

    ui.label('Binarization').classes('text-h4')
    ui.label('Tweak the threshold below until only the object is shown and completely solid without gaps. Increasing the threshold may cause the volume to be larger than the original.')

    binarImage()

    with ui.row():
        def update_binar_img(thr):
            global im_thresh_s
            im_thresh_s = thr
            binarImage.refresh()

        ui.label("Threshold for binarization")
        ui.slider(min=0, max=255, value=im_thresh_s, on_change=lambda e: update_binar_img(e.value)).props('label-always')

    ui.run()

@ui.refreshable
def panel_tweak_polyreg():
    global im_poly_alpha, im_poly_beta, im_poly_gamma, im_poly_scalar, im_poly_gauss_sigma

    def update_poly_params(param, thr):
        globals()[param] = thr

    ui.label('Polygon Regression').classes('text-h4')
    ui.label('Modify the parameters to tune the shape fitting algorithm, higher values may result in perimeter overfitting or strange issues.')

    with ui.column():
        with ui.row():
            ui.number(label='Alpha', value=im_poly_alpha, format='%.4f',
                      on_change=lambda e: update_poly_params('im_poly_alpha', e.value))

            ui.number(label='Beta', value=im_poly_beta, format='%.4f',
                      on_change=lambda e: update_poly_params('im_poly_beta', e.value))

            ui.number(label='Gamma', value=im_poly_gamma, format='%.4f',
                      on_change=lambda e: update_poly_params('im_poly_gamma', e.value))

            ui.number(label='Gaussian sigma', value=im_poly_gauss_sigma, format='%i',
                      on_change=lambda e: update_poly_params('im_poly_gauss_sigma', e.value))

            ui.number(label='Polygon scalar', value=im_poly_scalar, format='%.2f',
                      on_change=lambda e: update_poly_params('im_poly_scalar', e.value))

# Main UI windows
@ui.refreshable
def export_panel():
    global im_final, doc_buf

    if im_final is not None:
        with ui.interactive_image(im_final) as ii:
            ui.label('35.5s').classes('absolute bottom-0 left-0 m-2 text-white p-4 bg-black backdrop-opacity-10 rounded-md')

            ui.button("Download .DXF", on_click=lambda: ui.download(doc_buf, 'output.dxf'), icon='download') \
                .props('flat fab color=blue') \
                .classes('absolute bottom-0 right-0 m-2')
    else:
        with ui.column():
            ui.markdown("#### No image processed<br>")
            ui.markdown("You must load an image and process first before exporting.")

@ui.refreshable
def uploader_panel():
    with ui.card().classes('no-shadow border-[1px] w-full'):
        ui.label('Upload an image file and press upload to continue to the next step.')
        def handle_upload(e: events.UploadEventArguments):
            global loaded_rgba, loaded_b64, loaded_b64_type

            loaded_raw = e.content.read()
            loaded_b64 = base64.b64encode(loaded_raw)
            loaded_rgba = Image.open(io.BytesIO(loaded_raw))
            loaded_b64_type = e.type

            def mouse_handler(e: events.MouseEventArguments):
                global p1x, p1y, p2x, p2y, topx, topy, botx, boty, btn_r_mode, btn_corner_cntr, btn_area_cntr

                if e.type == 'mousedown':
                    if btn_r_mode:
                        if btn_corner_cntr:
                            p1x, p1y = e.image_x, e.image_y
                            ii.content += f'<circle cx="{p1x}" cy="{p1y}" r="10" fill="none" stroke="orange" stroke-width="4" />'
                        else:
                            p2x, p2y = e.image_x, e.image_y
                            ii.content += f'<circle cx="{p2x}" cy="{p2y}" r="10" fill="none" stroke="blue" stroke-width="4" />'

                        btn_corner_cntr = not btn_corner_cntr
                    else:
                        if btn_area_cntr:
                            topx, topy = e.image_x, e.image_y
                            ii.content = f'<circle cx="{topx}" cy="{topy}" r="5" fill="none" stroke="red" stroke-width="2" />'
                        else:
                            botx, boty = e.image_x, e.image_y
                            ii.content += f'<circle cx="{botx}" cy="{boty}" r="5" fill="none" stroke="green" stroke-width="2" />'
                            ii.content += f'<rect x="{topx}" y="{topy}" width="{botx-topx}" height="{boty-topy}" style="fill:yellow;stroke:red;stroke-width:3;opacity:0.2" rx="0" ry="0" fill="none" stroke="red" stroke-width="2" />'

                        btn_area_cntr = not btn_area_cntr

            with ui.interactive_image(f'data:{loaded_b64_type};base64,{loaded_b64.decode()}', on_mouse=mouse_handler, events=['mousedown', 'mouseup'], cross=True) as ii:
                def set_r_mode(rm):
                    global btn_r_mode
                    btn_r_mode = rm

                ui.button("Active Area", on_click=lambda: set_r_mode(False), icon='crop_free') \
                    .props('flat fab color=blue') \
                    .classes('absolute bottom-0 left-0 m-2') # Cropping mode

                ui.button("Ref Points", on_click=lambda: set_r_mode(True), icon='rounded_corner') \
                    .props('flat fab color=blue') \
                    .classes('absolute bottom-0 right-0 m-2') # Corner point mode

            with ui.row().classes('w-full'):
                def proc_image_step():
                    global im_cropped, im_thresh_s, topx, topy, botx, boty

                    im_cropped = loaded_rgba.crop((topx, topy, botx, boty))
                    im_thresh_s = threshold_otsu(np.array(im_cropped.convert('L')))
                    tweak_algorithm_panel.refresh()

                    ui.notify('Loaded image successfully, going to next tab.', type='success')
                    Timer(1, lambda: panels.set_value('Tweak Algorithm')).start()

                ui.space()

                def set_known_dist(indist):
                    global in_known_dist
                    in_known_dist = indist

                ui.number(label='Ref. Point Distance', value=0, format='%.2f',
                          on_change=lambda e: set_known_dist(e.value))

                ui.button("Process image", on_click=proc_image_step).classes('mt-5')

        ui.upload(on_upload=handle_upload).props('accept=.png,.jpg').classes('max-w-full w-full')

@ui.refreshable
def tweak_algorithm_panel():
    global loaded_b64, loaded_b64_type

    if im_cropped is not None:
        with ui.tabs() as tabs:
            binar = ui.tab('Binarization', icon='opacity')
            polyreg = ui.tab('Polygon Regression', icon='select_all')

        with ui.tab_panels(tabs, value=binar).classes('w-full'):
            with ui.tab_panel(binar):
                panel_tweak_binarization()

            with ui.tab_panel(polyreg):
                panel_tweak_polyreg()

        with ui.row().classes('w-full'):
            def proc_image_step():
                # stuff here
                procDxf()

                export_panel.refresh()

                ui.notify('Processed image successfully, going to next tab.', type='success')
                Timer(1, lambda: panels.set_value('Export')).start()

            ui.space()
            ui.button("Process and continue", on_click=proc_image_step).classes('mt-5')

    else:
        with ui.column():
            ui.markdown("#### No image loaded<br>")
            ui.markdown("You must load an image first before changing settings.")

# UI Elements
dark_mode_status = 0
def dark_mode_toggle():
    global dark_mode_status
    if dark_mode_status == 1:
        ui.dark_mode().disable()
        dark_mode_status = 0
    else:
        ui.dark_mode().enable()
        dark_mode_status = 1

@ui.refreshable
def main_panel():
    global panels

    with ui.header().classes(replace='row items-center w-full') as header:
        with ui.tabs().classes('w-full') as tabs:
            ui.tab('Upload File')
            ui.tab('Tweak Algorithm')
            ui.tab('Export')
            ui.space()

            ui.button('Theme', icon='brightness_6', on_click=dark_mode_toggle).classes('mr-2')
            ui.button('Stop Server', icon='logout', on_click=lambda: exit()).classes('mr-2').props('color="red"')

    with ui.tab_panels(tabs, value='Upload File').classes('w-full') as panels:
        with ui.tab_panel('Upload File'):
            uploader_panel()

        with ui.tab_panel('Tweak Algorithm'):
            tweak_algorithm_panel()

        with ui.tab_panel('Export'):
            export_panel()

"""
Main app runtime loop
"""

if __name__ in {"__main__", "__mp_main__"}:
    main_panel()

    ui.dark_mode().enable()
    ui.run() #uvicorn_reload_excludes="")