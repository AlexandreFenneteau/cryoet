import glob
import yaml  
import os  
import cv2  
import matplotlib.pyplot as plt  
from matplotlib.widgets import Slider  

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple",
          "tab:red", "tab:pink", "tab:olive", "tab:cyan"]
  
# Function to parse the YAML file  
def parse_yaml(yaml_file):  
    with open(yaml_file, 'r') as f:  
        data = yaml.safe_load(f)  
    return data  
  
# Function to read image and annotation files  
def load_data(data, yaml_path, split):  
    path = os.path.abspath(os.path.join(os.path.dirname(yaml_path), os.pardir, os.pardir, os.pardir, data['path']))
    classes = data['names']  
    train_images_path = os.path.abspath(os.path.join(path, data[split]))
    train_labels_path = train_images_path.replace('images', 'labels')  
      
    # Gather image and label files  
    images = sorted(glob.glob(os.path.join(train_images_path, '*.png')))
    labels = []
    for image in images:
        label_path = os.path.join(train_labels_path, os.path.basename(image.replace('.png', '.txt')))
        labels.append(label_path if os.path.exists(label_path) else None)  
      
    return images, labels, classes  
  
# Function to plot image with bounding boxes  
def plot_image_with_boxes(image_path, label_path, class_names, ax):  
    # Read image  
    image = cv2.imread(image_path)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    ax.clear()  
    ax.imshow(image)  
      
    # Draw bounding boxes if label file exists  
    if not(label_path is None):  
        with open(label_path, 'r') as f:  
            lines = f.readlines()  
        for line in lines:  
            data = line.strip().split()  
            class_id, x_center, y_center, width, height = map(float, data)  
            class_id = int(class_id)  
              
            # Convert normalized coordinates to absolute pixel values  
            h, w, _ = image.shape  
            x_center *= w  
            y_center *= h  
            width *= w  
            height *= h  

            #width+= 4
            #height += 4
              
            # Calculate bounding box coordinates  
            x1 = int(x_center - width / 2)  
            y1 = int(y_center - height / 2)  
            #x2 = int(x_center + width / 2)  
            #y2 = int(y_center + height / 2)  
              
            # Draw rectangle and label  
            ax.add_patch(plt.Rectangle((x1, y1), width, height, edgecolor=COLORS[class_id], facecolor='none', linewidth=2))  
            #ax.text(x1, y1 - 5, class_names[class_id], color='red', fontsize=10, backgroundcolor='white')  
      
    # Display image name  
    ax.set_title(f"Image: {os.path.basename(image_path)}")  
    plt.draw()  
  
# Main function to visualize images with a slider  
def visualize_dataset(yaml_file, split: str = "train"):  
    # Parse YAML and load data  
    data = parse_yaml(yaml_file)  
    images, labels, class_names = load_data(data, yaml_file, split)  
      
    # Create figure and axis for display  
    fig, ax = plt.subplots()  
    plt.subplots_adjust(bottom=0.25)  
      
    # Initialize the first image  
    idx = 0  
    plot_image_with_boxes(images[idx], labels[idx], class_names, ax)  
      
    # Create slider  
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')  
    slider = Slider(ax_slider, 'Image', 0, len(images) - 1, valinit=0, valstep=1)  
      
    # Update function for slider  
    def update(val):  
        idx = int(slider.val)  
        plot_image_with_boxes(images[idx], labels[idx], class_names, ax)  

     # Key press event handler  
    def on_key(event):  
        if event.key == 'right':  # Next image  
            if slider.val < len(images) - 1:  
                slider.val += 1  
                update(slider.val)  
        elif event.key == 'left':  # Previous image  
            if slider.val > 0:  
                slider.val -= 1  
                update(slider.val)  
      
    slider.on_changed(update)  
    fig.canvas.mpl_connect('key_press_event', on_key)
      
    plt.show()  
  
# Example usage  
#yaml_file = r'C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\yolo\coronal_data.yaml'  # Replace with the path to your YAML file  
yaml_file = r'C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\yolo\axial_data.yaml'  # Replace with the path to your YAML file  
#yaml_file = r'C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\yolo\sagittal_data.yaml'  # Replace with the path to your YAML file  
visualize_dataset(yaml_file, split="val")  