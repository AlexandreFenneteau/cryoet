import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib.widgets import Slider, RadioButtons  
  
class ImageSlicer:  
    def __init__(self, volume, mask, slicing_mode="axial"):  
        """  
        Initialize the ImageSlicer with a 3D volume and mask.  
  
        :param volume: A 3D numpy array representing the image volume.  
        :param mask: A 3D numpy array representing the mask.  
        :param slicing_mode: A string specifying the initial slicing mode ("axial", "sagittal", or "coronal").  
        """  
        self.volume = volume  
        self.mask = mask  
        self.z, self.y, self.x = volume.shape  
        self.slicing_mode = slicing_mode  
  
        # Initial slice indices  
        self.axial_index = self.z // 2  
        self.coronal_index = self.y // 2  
        self.sagittal_index = self.x // 2  

        self.particle_cmps = ["spring", "summer", "autumn", "winter", "cool", "Wistia"]
  
        # Set up the figure and the axis  
        self.fig, self.ax = plt.subplots(1, 1)  
        plt.subplots_adjust(left=0.25, bottom=0.25)  
  
        # Display the initial slice  
        self.im = None  
        self.msk = None  
        self.update_display()  
  
        # Create sliders  
        axcolor = 'lightgoldenrodyellow'  
        self.slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)  
        self.s_slider = Slider(self.slider_ax, 'Slice', 0, self.get_max_index() - 1, valinit=self.get_current_index(), valstep=1)  
        self.s_slider.on_changed(self.update_slice)  
  
        # Create radio buttons for slicing mode selection  
        self.radio_ax = plt.axes([0.025, 0.4, 0.15, 0.15], facecolor=axcolor)  
        self.radio = RadioButtons(self.radio_ax, ('axial', 'sagittal', 'coronal'))  
        self.radio.on_clicked(self.change_slicing_mode)  
  
        plt.show()  
  
    def get_current_index(self):  
        """Get the current slice index based on the slicing mode."""  
        if self.slicing_mode == "axial":  
            return self.axial_index  
        elif self.slicing_mode == "sagittal":  
            return self.sagittal_index  
        elif self.slicing_mode == "coronal":  
            return self.coronal_index  
  
    def get_max_index(self):  
        """Get the maximum index for the current slicing mode."""  
        if self.slicing_mode == "axial":  
            return self.z  
        elif self.slicing_mode == "sagittal":  
            return self.x  
        elif self.slicing_mode == "coronal":  
            return self.y  
  
    def update_display(self):  
        """Update the displayed slice based on the slicing mode."""  
        if self.im is not None:  
            self.im.remove()  
        if self.msk is not None:  
            self.msk.remove()  
  
        if self.slicing_mode == "axial":  
            self.im = self.ax.imshow(self.volume[self.axial_index, :, :], cmap='gray')  
            if self.mask is not None:
                img = torch.zeros(self.mask.shape[2:], dtype=int)
                for label in range(self.mask.shape[0]):
                    img[self.mask[label, self.axial_index]] = label + 1
                self.msk = self.ax.imshow(img, cmap="jet", alpha=0.4)  
            self.ax.set_title('Axial [x, y]')  
        elif self.slicing_mode == "sagittal":  
            self.ax.set_title('Sagittal [z, y]')  
            self.im = self.ax.imshow(self.volume[:, :, self.sagittal_index], cmap='gray')  
            if not(self.mask is None):
                img = torch.zeros((self.mask.shape[1], self.mask.shape[2]), dtype=int)
                for label in range(self.mask.shape[0]):
                    img[self.mask[label, :, :, self.sagittal_index]] = label + 1
                self.msk = self.ax.imshow(img, cmap="jet", alpha=0.4)  
        elif self.slicing_mode == "coronal":  
            self.ax.set_title('Coronal [z, x]')  
            self.im = self.ax.imshow(self.volume[:, self.coronal_index, :], cmap='gray')  
            if not(self.mask is None):
                img = torch.zeros((self.mask.shape[1], self.mask.shape[3]), dtype=int)
                for label in range(self.mask.shape[0]):
                    img[self.mask[label, :, self.coronal_index, :]] = label + 1
                self.msk = self.ax.imshow(img, cmap="jet", alpha=0.4)  
  
        self.fig.canvas.draw_idle()  
  
    def update_slice(self, val):  
        """Update the slice index and the displayed slice."""  
        new_index = int(self.s_slider.val)  
        if self.slicing_mode == "axial":  
            self.axial_index = new_index  
        elif self.slicing_mode == "sagittal":  
            self.sagittal_index = new_index  
        elif self.slicing_mode == "coronal":  
            self.coronal_index = new_index  
  
        self.update_display()  
  
    def change_slicing_mode(self, label):  
        """Change the slicing mode and update the display."""  
        self.slicing_mode = label  
        self.s_slider.valmax = self.get_max_index() - 1  
        self.s_slider.set_val(self.get_current_index())  
        self.update_display()  
  
  
# Example usage  
if __name__ == "__main__":  
    # Create a sample 3D numpy array (e.g., a volumetric image)  
    # For demonstration, let's create a 3D array with random values  
    import torch
    test_path = r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\pytorch\train\TS_5_4_res-0_img.pt" 
    msk_path = r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\pytorch\train\TS_5_4_res-0_msk.pt" 
    #msk_path = None
    volume_data = torch.load(test_path)
    if msk_path:
        msk_data = torch.load(msk_path)
    else:
        msk_data = None
  
    # Initialize and run the image slicer  
    slicer = ImageSlicer(volume_data, msk_data, slicing_mode="axial")  