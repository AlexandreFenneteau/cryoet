import glob
import yaml  
import os  
import cv2  
import matplotlib.pyplot as plt  
import numpy as np
from matplotlib.widgets import Slider  
import matplotlib

COLORS = [matplotlib.colors.ListedColormap(['black', 'tab:red']), #apo-ferritine, ok
          matplotlib.colors.ListedColormap(['black', 'tab:orange']), #beta-gal, ok
          matplotlib.colors.ListedColormap(['black', 'tab:green']), #ribosome, ok
          matplotlib.colors.ListedColormap(['black', 'tab:brown']), #thyroglob ok => confusion avec virus????
          matplotlib.colors.ListedColormap(['black', 'tab:pink']), #pas detectee ?????
          ]
  
# Function to parse the YAML file  
def parse_yaml(yaml_file):  
    with open(yaml_file, 'r') as f:  
        data = yaml.safe_load(f)  
    return data  
  
# Function to plot image with bounding boxes  
def plot_image_with_boxes(image_path, msk_path, ax):  
    # Read image      
    image = cv2.imread(image_path)  

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    ax.clear()  
    ax.imshow(image)  

    for i in range(1, 6):
        msk_label = np.zeros_like(msk)
        msk_label[msk == i] = 1
        ax.imshow(msk_label, alpha=msk_label * 0.6, cmap=COLORS[i-1])
      
    # Display image name  
    ax.set_title(f"Image: {os.path.basename(image_path)}")  
    plt.draw()  
  
# Main function to visualize images with a slider  
def visualize_dataset(img_paths, msk_paths):  
    # Parse YAML and load data  
      
    # Create figure and axis for display  
    fig, ax = plt.subplots()  
    plt.subplots_adjust(bottom=0.25)  
      
    # Initialize the first image  
    idx = 0  
    plot_image_with_boxes(img_paths[idx], msk_paths[idx], ax)  
      
    # Create slider  
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')  
    slider = Slider(ax_slider, 'Image', 0, len(img_paths) - 1, valinit=0, valstep=1)  
      
    # Update function for slider  
    def update(val):  
        idx = int(slider.val)  
        plot_image_with_boxes(img_paths[idx], msk_paths[idx], ax)  

     # Key press event handler  
    def on_key(event):  
        if event.key == 'right':  # Next image  
            if slider.val < len(img_paths) - 1:  
                slider.val += 1  
                update(slider.val)  
        elif event.key == 'left':  # Previous image  
            if slider.val > 0:  
                slider.val -= 1  
                update(slider.val)  
      
    slider.on_changed(update)  
    fig.canvas.mpl_connect('key_press_event', on_key)
      
    plt.show()  
  
from tqdm import tqdm
# Example usage  
###################################################################
img_data_glob = sorted(glob.glob(r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\yolo\coronal\images\train\TS_5_4_*.png"))
msk_data_glob = sorted(glob.glob(r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\yolo\coronal\msks\train\TS_5_4_*.png"))
visualize_dataset(img_data_glob, msk_data_glob)  
###################################################################
# msk_data_glob = sorted(glob.glob(r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\yolo\sagittal\msks\train\*.png"))
# labels = np.array([])
# for msk_path in tqdm(msk_data_glob):
#     msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
#     current_labels = np.unique(msk)
#     labels = np.unique(np.concat([labels, current_labels]))
# 
#     if 6 in labels:
#         print(msk_path)
# print(labels)
