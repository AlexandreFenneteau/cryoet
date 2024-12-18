import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib.widgets import Slider  
  
class ImageSlicer:  
    def __init__(self, volume, mask):  
        """  
        Initialize the ImageSlicer with a 3D volume.  
  
        :param volume: A 3D numpy array representing the image volume.  
        """  
        self.volume = volume  
        self.mask = mask
        self.z, self.x, self.y = volume.shape  
  
        # Initial slice indices  
        self.axial_index = self.z // 2  
        self.coronal_index = self.y // 2  
        self.sagittal_index = self.x // 2  
  
        # Set up the figure and the axis  
        self.fig, self.ax = plt.subplots(1, 1)  
        plt.subplots_adjust(left=0.25, bottom=0.25)  
  
        # Display the initial slices  
        self.axial_im = self.ax.imshow(self.volume[self.axial_index, :, :], cmap='gray')  
        self.axial_msk = self.ax.imshow(self.mask[self.axial_index, :, :], cmap='jet', alpha=0.2)
  
        self.ax.set_title('Axial [x, y]')  
  
        # Create sliders for each dimension  
        axcolor = 'lightgoldenrodyellow'  
        ax_axial = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)  
  
        self.s_axial = Slider(ax_axial, 'Axial [z slice]', 0, self.z - 1, valinit=self.axial_index, valstep=1)  
  
        # Register the update functions with each slider  
        self.s_axial.on_changed(self.update_axial)  
  
        plt.show()  
  
    def update_axial(self, val):  
        """Update the axial slice."""  
        self.axial_index = int(self.s_axial.val)  
        self.axial_im.set_data(self.volume[self.axial_index, :, :])  
        self.axial_msk.set_data(self.mask[self.axial_index, :, :])  
        self.fig.canvas.draw_idle()  
  
# Example usage  
if __name__ == "__main__":  
    # Create a sample 3D numpy array (e.g., a volumetric image)  
    # For demonstration, let's create a 3D array with random values  
    import torch
    test_path = r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\train\TS_99_9_res-0_img.pt" 
    msk_path = r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\train\TS_99_9_res-0_msk-all.pt" 
    volume_data = torch.load(test_path)
    msk_data = torch.load(msk_path)
  
    # Initialize and run the image slicer  
    slicer = ImageSlicer(volume_data, msk_data)  