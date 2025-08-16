import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import matplotlib.pyplot as plt
from cv2 import dnn_superres
import numpy as np

# Globals
img = None
mask = None
filename = None

# ───────────────────────────────
# Browse Image
# ───────────────────────────────
def browse_image():
    global img, filename
    filename = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
    if filename:
        img = cv2.imread(filename)
        statuslabel.config(text="Image selected")
        plt.title("Selected Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# ───────────────────────────────
# Browse Mask
# ───────────────────────────────
def browse_mask():
    global mask
    maskfile = filedialog.askopenfilename(filetypes=[("Mask", "*.png")])
    if maskfile:
        mask = cv2.imread(maskfile, 0)
        statuslabel.config(text="Mask selected")
        plt.title("Selected Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.show()

# ───────────────────────────────
# Inpaint (Fixed)
# ───────────────────────────────
def inpaint_image():
    global img, mask
    if img is None or mask is None:
        messagebox.showwarning("Error", "Select both image and mask first")
        return

    # Resize mask if dimensions don't match
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Ensure mask is single-channel grayscale
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Ensure mask is uint8
    mask = mask.astype(np.uint8)

    try:
        dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite("output_inpaint.png", dst)
        statuslabel.config(text="Inpainting complete → output_inpaint.png")
        plt.title("Inpainted Output")
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    except cv2.error as e:
        messagebox.showerror("OpenCV Error", f"Inpainting failed:\n{e}")

# ───────────────────────────────
# Super Resolution
# ───────────────────────────────
def super_resolve():
    global img
    if img is None:
        messagebox.showwarning("Error", "Select image first")
        return
    try:
        sr = dnn_superres.DnnSuperResImpl_create()
        model_path = "EDSR_x2.pb"
        if not os.path.exists(model_path):
            messagebox.showwarning("Model Missing", "Download EDSR_x2.pb from OpenCV model zoo.")
            return
        sr.readModel(model_path)
        sr.setModel("edsr", 2)
        result = sr.upsample(img)
        cv2.imwrite("output_superres.png", result)
        statuslabel.config(text="Super resolution done → output_superres.png")
        plt.title("Super Resolution Output")
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    except Exception as e:
        messagebox.showwarning("Error", f"Super-resolution failed: {e}")

# ───────────────────────────────
# Deblur (Sharpening filter)
# ───────────────────────────────
def deblur_image():
    global img
    if img is None:
        messagebox.showwarning("Error", "Select image first")
        return
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    result = cv2.filter2D(img, -1, kernel)
    cv2.imwrite("output_deblur.png", result)
    statuslabel.config(text="Deblurring complete → output_deblur.png")
    plt.title("Deblurred Output")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# ───────────────────────────────
# GUI Setup
# ───────────────────────────────
parent = tk.Tk()
parent.title("Vision Suite - Inpainting, Super-Res, Deblurring")
frame = tk.Frame(parent, padx=10, pady=10)
frame.pack()

tk.Label(frame, text="Vision Suite", fg="red", font=("Times New Roman", 18)).pack()
ttk.Separator(frame, orient='horizontal').pack(fill='x')

tk.Button(frame, text="Select Image", command=browse_image).pack()
tk.Button(frame, text="Select Mask", command=browse_mask).pack()
tk.Button(frame, text="Inpaint", command=inpaint_image).pack()
tk.Button(frame, text="Super Resolution", command=super_resolve).pack()
tk.Button(frame, text="Deblur", command=deblur_image).pack()

ttk.Separator(frame, orient='horizontal').pack(fill='x')
statuslabel = tk.Label(frame, text="", fg="blue")
statuslabel.pack()

parent.mainloop()