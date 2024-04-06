import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os

def iou(y_true, y_pred):
    smooth = 1e-5
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred))
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def iou_loss(y_true,y_pred):
    return -iou(y_true,y_pred)

# Defining the custom metric function
def dice_coefficient(y_true, y_pred):
    smooth = 1e-15
    intersection = K.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

segmentation_model = load_model('C:\\Users\\sairam\\Documents\\Nikhil Project\\My models\\USS_UNET_MODEL.h5', custom_objects={'iou_loss': iou_loss, 'dice_coefficient': dice_coefficient,'iou':iou})

classification_model = tf.keras.models.load_model('C:\\Users\\sairam\\Documents\\Nikhil Project\\My models\\_BIMODEL_model (1).h5')

def load_image(file_path):
    image = Image.open(file_path)
    image = Image.open(file_path)
    image = image.resize((256, 256))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    return image

def update_image():
    file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
    if file_path:
        global image
        image = load_image(file_path)

        image_pil = Image.fromarray((image[0] * 255).astype(np.uint8))
        image_label.image = ImageTk.PhotoImage(image_pil)
        image_label.config(image=image_label.image)

        segment_button.pack()

def start_segmentation():
    global mask
    mask = segmentation_model.predict(image)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    mask_image = Image.fromarray((mask[0] * 255).astype(np.uint8))

    mask_label.image = ImageTk.PhotoImage(mask_image)
    mask_label.config(image=mask_label.image)

    classify_button.pack()

def start_classification():
    prediction = classification_model.predict(mask)
    result_label.config(text=f"Output: {'Strictured' if prediction[0][0] > 0.45 else 'Non-Strictured'}, Value: {prediction[0][0]}")

root = tk.Tk()
root.title("URETHRAL STRICTURE CLASSIFIER")


logo_image = Image.open("logo.jpg")
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(root, image=logo_photo)
logo_label.image = logo_photo  
logo_label.pack()
logo_label.pack(side=RIGHT)

image_label = tk.Label(root)
image_label.pack()
result_label = tk.Label(root, text="")
result_label.pack()

mask_label = tk.Label(root)  # Define mask_label here
mask_label.pack()

update_image_button = tk.Button(root, text="LOAD IMAGE", command=update_image,bg="sky blue", fg="white")
update_image_button.pack()
update_image_button.pack(side=LEFT)

segment_button = tk.Button(root, text="SEGMENT", command=start_segmentation,bg="red", fg="white")
segment_button.pack()
segment_button.pack(side=LEFT)

classify_button = tk.Button(root, text="CLASSIFY", command=start_classification,bg="green", fg="white")
classify_button.pack()
classify_button.pack(side=LEFT)



root.mainloop()
