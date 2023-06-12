import math

import cv2
import numpy as np
# import pandas as pd
from keras.models import load_model
from PIL import Image, ImageTk
from keras.preprocessing import image
from keras.utils import img_to_array
import tkinter as tk
from tkinter import filedialog


model = load_model('detection.h5')
test_datagen = image.ImageDataGenerator(rescale=1.0 / 255.0)

# ////////////////////////////////////////////////////////////////////////////////////////////////////

Pred_result=''

def open_image():
    global Pred_result
    filepath = filedialog.askopenfilename(title="Select an image file",
                                          filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.gif"), ("All files", "*.*")))
    if filepath:
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
        image = Image.fromarray(image)
        
        im  = image.copy()
        image = image.resize((400, 300))  # Adjust the size of the image as needed
        photo = ImageTk.PhotoImage(image)

        im = im.resize((250, 250))
        im1 = img_to_array(im)
        
        img = test_datagen.standardize(np.copy(im1))
        img_array = np.expand_dims(np.asarray(img), axis=0)
        prediction = model.predict(img_array)
        if prediction[0] > 0.5:
            label = "Plastic " + str(prediction[0])
            Pred_result = label

        # Scale the window size based on the image dimensions
        window_width = image.width + 100
        window_height = image.height + 200
        root.geometry(f"{window_width}x{window_height}")
        
        # Update the image label with the new photo
        image_label.configure(image=photo)
        image_label.image = photo  # Keep a reference to the image
        


def dt2():
    text = str(Pred_result)
    text_label.config(text=text , font=("Arial", 24, "bold"))



# Create the main window
root = tk.Tk()
root.title("Prediction Waste Plastic")

# Create a frame for the image
image_frame = tk.Frame(root)
image_frame.pack()

# Create a button to open the image
button = tk.Button(image_frame, text="Open Image", command=open_image)
button.pack()

# Create a button to display the entered text
text_button = tk.Button(root, text="Display Predection", command=dt2)
text_button.pack()

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Create a label for the displayed text
text_label = tk.Label(root, text="Pred", font=("Helvetica", 24, "bold"))
text_label.pack()

# Set the initial size of the window
window_width = 400
window_height = 300
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")


# Start the Tkinter event loop
root.mainloop()
