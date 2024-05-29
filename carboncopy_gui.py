"""
Carbon Copy by Ari Stehney

Scan in carbon pieces to a DXF file.
"""

from concurrent.futures import process
from PIL import Image
from scipy.interpolate import interp1d
import numpy as np
import math, re, ezdxf, sys
from scipy import ndimage as ndi
from skimage import feature
from nicegui import ui, events
import io, base64

from threading import Timer
from fastapi.responses import StreamingResponse

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
im_cropped = None
im_gray, im_bool, im_bin = None, None, None
im_thresh_s = 128

def procDxf():
    global im_gray, im_bool, im_bin, im_thresh_s, in_known_dist, p1x, p1y, p2x, p2y, topx, topy, botx, boty

    # im_thresh_s : threshold
    # in_known_dist: float mm
    # p1x, p1y, p2x, p2y : known dim values
    # topx, topy, botx, boty : crop values

    print("p1: ({}, {}), p2: ({}, {})".format(p1x, p1y, p2x, p2y))

    planar_dist = float(in_known_dist)
    pixel_dist = math.sqrt((p2x - p1x)**2 + (p2y - p1y)**2)
    print("Planar: {}, Pixel: {}".format(planar_dist, pixel_dist))
    dpp = planar_dist / pixel_dist

    input_gray = np.array(im_cropped.convert('L'))
    input_bool_inv = im_gray < im_thresh_s

    edges = feature.canny(input_bool_inv, sigma=3)
    w_orig, h_orig = im_cropped.size

    xU_scale = interp1d([0, len(edges[0])], [0, w_orig])
    yU_scale = interp1d([0, len(edges)],[0, h_orig])

    addPt = []
    edgeMx = []
    def procEdge(i, j):
        edgematrix = [
            [i+1, j+1],
            [i+1, j],
            [i, j+1],
            [i-1, j-1],
            [i-1, j],
            [i, j-1],
            [i-1, j+1],
            [i+1, j-1],
        ]

        for k in edgematrix:
            if (edges[k[0], k[1]] == 1) and (not (k in addPt)):
                addPt.append([k[0], k[1]])
                edgeMx.append([i, j, k[0], k[1]])
                try:
                    procEdge(k[0], k[1])
                except RecursionError:
                    messagebox.showwarning("Warning","Please increase recursion depth!")
                    return

    for j in range(0, len(edges[0])-1):
        for i in range(0, len(edges)-1):
            if edges[i, j]==1 and (not ([i, j] in addPt)):
                procEdge(i, j)

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    for item in edgeMx:
        msp.add_line((xU_scale(item[1])*dpp, yU_scale(item[0])*dpp), (xU_scale(item[3])*dpp, yU_scale(item[2])*dpp), dxfattribs={"layer": "MainLayer"})

    doc.saveas("test.dxf")
    
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
    ui.label('Tweak the threshold below until only the object is shown, and it is completely solid without gaps.')

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
def panel_tweak_lcom():
    ui.label('LCOM Reduction').classes('text-h4')
    ui.label('Local center of mass size')

@ui.refreshable
def panel_tweak_tangvec():
    ui.label('Tangency/Vectorization').classes('text-h4')
    ui.label('Spline conversion parameters')

# Main UI windows
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
                    global im_cropped, topx, topy, botx, boty

                    im_cropped = loaded_rgba.crop((topx, topy, botx, boty))
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
        with ui.splitter(value=15).classes('w-full h-[500px]') as splitter:
            with splitter.before:
                with ui.tabs().props('vertical').classes('w-full') as tabs:
                    binar = ui.tab('Binarization', icon='opacity')
                    lcom = ui.tab('LCOM Reduction', icon='blur_on')
                    tvec = ui.tab('Tangency/Vectorization', icon='polyline')
            with splitter.after:
                with ui.tab_panels(tabs, value=binar) \
                        .props('vertical').classes('w-full h-full'):
                    with ui.tab_panel(binar):
                        panel_tweak_binarization()

                    with ui.tab_panel(lcom):
                        panel_tweak_lcom()

                    with ui.tab_panel(tvec):
                        panel_tweak_tangvec()

        with ui.row().classes('w-full'):
            def proc_image_step():
                # stuff here
                procDxf()

                ui.notify('Processed image successfully, going to next tab.', type='success')
                Timer(1, lambda: panels.set_value('Export')).start()

            ui.space()
            ui.button("Process and continue", on_click=proc_image_step).classes('mt-5')

    else:
        with ui.column():
            ui.markdown("#### No image loaded<br>")
            ui.markdown("You must load an image first before changing settings.")
@ui.refreshable
def export_panel():
    pass

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