"""
Carbon Copy by Ari Stehney

Scan in carbon pieces to a DXF file.
"""

from concurrent.futures import process
import tkinter as tk
from tkinter import DISABLED, HORIZONTAL, N, S, E, W, END, filedialog as fd, messagebox, filedialog
from tkinter.font import NORMAL
from PIL import Image, ImageTk
from scipy.interpolate import interp1d
import numpy as np
from tkinter.ttk import *
import math, re, ezdxf, sys
from scipy import ndimage as ndi
from skimage import feature
sys.setrecursionlimit(32768)

# State variables
topx, topy, botx, boty = 0, 0, 0, 0
p1x, p1y, p2x, p2y = 0, 0, 0, 0
btn3_cntr = False
rect_id = None
p1_id = None
p2_id = None
path = "assets/dummyimg.jpg"
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

# Callbacks
def get_mouse_posn(event):
    global topy, topx
    topx, topy = event.x, event.y

def update_sel_rect(event):
    global rect_id
    global topy, topx, botx, boty
    botx, boty = event.x, event.y
    canvas.coords(rect_id, topx, topy, botx, boty)  # Update selection rect.

def set_pin_loc(event):
    global btn3_cntr, p1x, p1y, p2x, p2y
    if btn3_cntr:
        p1x, p1y = event.x, event.y
        canvas.coords(p1_id, p1x-10, p1y-10, p1x+10, p1y+10)
    else:
        p2x, p2y = event.x, event.y
        canvas.coords(p2_id, p2x-10, p2y-10, p2x+10, p2y+10)
    btn3_cntr = not btn3_cntr

# Setup window
window = tk.Tk()
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
                                  dash=(2,2), outline="blue")

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

def processArgs(event):
    global im_gray, im_bool, im_bin
    im_gray = np.array(adj_prv_img.convert('L'))
    im_bool = im_gray > thresh_s.get()
    im_bin = (im_gray > thresh_s.get()) * 255

    img_updated = ImageTk.PhotoImage(Image.fromarray(np.uint8(im_bin)))
    adj_canvas.itemconfig(adj_img_con, image=img_updated)
    adj_canvas.config(width=adj_new_width, height=512)
    adj_canvas.img = img_updated

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
home_frame = Frame(window)
home_frame.grid(row=0, column=1)

Label(home_frame, text="Right click on two of crosshairs on the page, and measure the distance (float value)", wraplength=105).grid(row=0, column=1)

Label(home_frame, text="\nDistance in mm:").grid(row=1, column=1)

knownlocpos = tk.Text(home_frame, height=1, width=12)
knownlocpos.grid(row=2, column=1, pady=25, padx=15)

Label(home_frame, text="Drag to select the area to be profiled then click process to continue.", wraplength=105).grid(row=3, column=1)

endbtn = Button(home_frame, text="Process Image", command=procDims, state=DISABLED)
endbtn.grid(row=4, column=1, sticky="sew", pady=25, padx=25)

# Menubar logic here
def openImg():
    path = fd.askopenfilename()
    img_updated = ImageTk.PhotoImage(previewProc(path))
    canvas.itemconfig(img_con, image=img_updated)
    canvas.config(width=new_width, height=512)
    canvas.img = img_updated
    endbtn.config(state=NORMAL)

menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", command=openImg)
menubar.add_cascade(label="File", menu=filemenu)
window.config(menu=menubar)

# Main Loop
canvas.bind("<Button-2>", set_pin_loc)
canvas.bind('<Button-1>', get_mouse_posn)
canvas.bind('<B1-Motion>', update_sel_rect)
window.mainloop()