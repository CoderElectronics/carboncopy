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

rect_id = None
p1_id = None
p2_id = None
enabled = False

# Helper FNs
def previewProc(pathm):
    global width, height
    global new_width
    global input_img
    input_img = Image.open(pathm)
    width, height = input_img.size
    new_width = int((width / height) * 512)
    prv_img = input_img.resize((new_width, 512))

    return prv_img

# Setup window
"""window = tk.Tk()
window.title("Carbon Copy Tool")
window.geometry('%sx%s' % (500, 512))
window.minsize(500, 512)
window.maxsize(500, 512)

img = ImageTk.PhotoImage(previewProc(path))
canvas = tk.Canvas(window, width=new_width, height=512,
                   borderwidth=0, highlightthickness=0)
canvas.grid(column=0, row=0)
canvas.img = img  # Keep reference in case this code is put into a function.
img_con = canvas.create_image(0, 0, image=img, anchor=tk.NW)

# Create selection rectangle (invisible since corner points are equal).
rect_id = canvas.create_rectangle(topx, topy, topx, topy,
                                  dash=(2,2), fill='', outline='red')

p1_id = canvas.create_oval(p1x-10, p1y-10, p1x+10, p1y+10,
                                  dash=(2,2), outline="green")
p2_id = canvas.create_oval(p2x-10, p2y-10, p2x+10, p2y+10,
                                  dash=(2,2), outline="blue")"""

# Buttons here
def procDims():
    clks = [p1x == 0, p1y == 0, p2x == 0, p2y == 0]
    if botx == topx and boty == topy:
        messagebox.showwarning("Warning","Please drag and select an area of the image first!")
    elif True in clks:
        messagebox.showwarning("Warning","Please select 2 points as known dimensions!")
    elif re.match(r'^-?\d+(?:\.\d+)$', knownlocpos.get("1.0",END)) is None:
        messagebox.showwarning("Warning","Enter a valid dimension!")
    else:
        xScl = interp1d([0, new_width], [0, width])
        yScl = interp1d([0, 512],[0, height])
        proc_img_crop = input_img.crop((int(xScl(topx)), int(yScl(topy)), int(xScl(botx)), int(yScl(boty))))

        adjustmentPanel(proc_img_crop)

def adjustmentPanel(proc_img_crop):
    global adj_new_width, adj_prv_img

    panel = tk.Toplevel(window)
    panel.title("Adjust Image Processing Settings")
    adj_width, adj_height = proc_img_crop.size
    adj_new_width = int((adj_width / adj_height) * 512)
    adj_prv_img = proc_img_crop.resize((adj_new_width, 512))

    panel.geometry("{}x{}".format(adj_new_width+200, 512))
    panel.minsize(adj_new_width+200, 512)
    panel.maxsize(adj_new_width+200, 512)

    global adj_canvas, adj_img_con, thresh_s

    adj_img = ImageTk.PhotoImage(adj_prv_img)
    adj_canvas = tk.Canvas(panel, width=adj_new_width, height=512,
                    borderwidth=0, highlightthickness=0)
    adj_canvas.grid(row=0, column=0)
    adj_canvas.img = adj_img  # Keep reference in case this code is put into a function.
    adj_img_con = adj_canvas.create_image(0, 0, image=adj_img, anchor=tk.NW)

    adj_frame = Frame(panel)
    adj_frame.grid(row=0, column=1)

    Label(adj_frame, text="Bianarize Threshold").grid(row=0, column=0, padx=25)
    thresh_s = Scale(adj_frame, from_=0, to=255, orient=HORIZONTAL)
    thresh_s.grid(row=1, column=0, pady=25, padx=(25, 25))
    thresh_s.bind("<ButtonRelease-1>", processArgs)
    thresh_s.set(70)

    processArgs(1)

    previewbtn = Button(adj_frame, text="Continue", command=procDxf)
    previewbtn.grid(row=2, column=0, sticky="sew", pady=25, padx=25)

"""def processArgs(event):
    global im_gray, im_bool, im_bin
    im_gray = np.array(adj_prv_img.convert('L'))
    im_bool = im_gray > thresh_s.get()
    im_bin = (im_gray > thresh_s.get()) * 255

    img_updated = ImageTk.PhotoImage(Image.fromarray(np.uint8(im_bin)))
    adj_canvas.itemconfig(adj_img_con, image=img_updated)
    adj_canvas.config(width=adj_new_width, height=512)
    adj_canvas.img = img_updated"""

def procDxf():
    # s_thresh.get() : threshold
    # p1x, p1y, p2x, p2y : known dim values
    # topx, topy, botx, boty : crop values

    xScl = interp1d([0, new_width], [0, width])
    yScl = interp1d([0, 512],[0, height])

    planar_dist = float(knownlocpos.get("1.0",END))
    pixel_dist = math.sqrt((xScl(p2x) - xScl(p1x))**2 + (yScl(p2y) - yScl(p1y))**2)
    dpp = planar_dist / pixel_dist

    input_crop = input_img.crop((int(xScl(topx)), int(yScl(topy)), int(xScl(botx)), int(yScl(boty))))
    input_gray = np.array(input_crop.convert('L'))
    input_bool_inv = im_gray < thresh_s.get()

    edges = feature.canny(input_bool_inv, sigma=3)
    w_orig, h_orig = input_crop.size
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

    f = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
    if f is None:
        return
    fn = f.name
    f.close()

    doc.saveas(fn)
    
# Create Widgets
"""home_frame = Frame(window)
home_frame.grid(row=0, column=1)

Label(home_frame, text="Right click on two of crosshairs on the page, and measure the distance (float value)", wraplength=105).grid(row=0, column=1)

Label(home_frame, text="\nDistance in mm:").grid(row=1, column=1)

knownlocpos = tk.Text(home_frame, height=1, width=12)
knownlocpos.grid(row=2, column=1, pady=25, padx=15)

Label(home_frame, text="Drag to select the area to be profiled then click process to continue.", wraplength=105).grid(row=3, column=1)

endbtn = Button(home_frame, text="Process Image", command=procDims, state=DISABLED)
endbtn.grid(row=4, column=1, sticky="sew", pady=25, padx=25)

menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", command=openImg)
menubar.add_cascade(label="File", menu=filemenu)
window.config(menu=menubar)

# Main Loop
canvas.bind("<Button-2>", set_pin_loc)
canvas.bind('<Button-1>', get_mouse_posn)
canvas.bind('<B1-Motion>', update_sel_rect)
window.mainloop()"""

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
                global topx, topy, botx, boty, btn_r_mode, btn_corner_cntr, btn_area_cntr

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