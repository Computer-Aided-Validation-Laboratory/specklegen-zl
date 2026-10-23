import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from pathlib import Path
from PIL import Image


class FourierAnalysis:
    """
    A class that analyses an image using fast fourer transforms.

    Attributes
    ----------
        image (np.ndarray): the image being analysed.
        width (int): width of image.
        height (int): height of image
        
    """

    def __init__(self, image: np.ndarray):

        self.image= image 

        self.width= None
        self.height= None 

        self.freq_x = None
        self.freq_y = None 

        self.power_x = None
        self.power_y = None

        self.x_size = None
        self.y_size = None 

    def FFT(self):
        """
        Performs a 2D FFT on the speckle pattern to find average speckle size and spacing.

            Args
            ----------
                None
            
            Returns
            ----------
                freq_x, freq_y (np.ndarray): sample frequency per pixels for rows (x) and columns (y).
                power_x, power_y (np.ndarray): power spectrum of rows (x) and columns (y). 
                x_size (float): average size of speckles in x direction from fft analysis.*
                y_size (float): average size of speckles in y direction from fft analysis.*
                
                    *Would be better to combine these and have a radial size or area as speckles are not squares.
                
                
        """
        self.width = np.shape(self.image)[0]
        self.height = np.shape(self.image)[1]

        freq_x_i = fftshift(fftfreq(self.width))
    
        freq_y_i = fftshift(fftfreq(self.height))
    
        mask_x = freq_x_i > 0 
        mask_y = freq_y_i > 0
    
        self.freq_x = freq_x_i[mask_x]
        self.freq_y = freq_y_i[mask_y]
    
        Fx = fftshift(fft(self.image,axis = 0))
        Fy = fftshift(fft(self.image,axis = 1))

        self.power_x = np.sum(np.abs(Fx)**2, axis = 1)[mask_x]
    
        #Fy = fftshift(fft(self.grid, axis = 1))
        self.power_y = np.sum(np.abs(Fy)**2, axis = 0)[mask_y]
    
        xi = np.argmin(self.power_x)
        yi = np.argmin(self.power_y)
    
        peak_freq_x = self.freq_x[xi]
        self.x_size = 1/ peak_freq_x 
    
        peak_freq_y = self.freq_y[yi]
        self.y_size = 1/ peak_freq_y
    
        return self.freq_x, self.freq_y, self.power_x, self.power_y, self.x_size, self.y_size

    def plot_FFT(self, save_path: Path | None = None):
        """
        Plots and saves the Fourier transform graphs.

            Args
            ----------
                save_path (Path): if left empty then image will not save, other wise will save as specified.

            Returns
            ----------
                None (plots nad saves fft graphs).

        """

        
        fig, ax = plt.subplots(1,2, figsize = (10,6))
        
        ax[0].plot(self.freq_x, self.power_x)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("Frequency [pixels $^{-1}$]")
        ax[0].set_ylabel("Rows Power Spectrum")
        ax[0].axvline(1 / self.x_size, color = "r", linestyle = "--", label = f" Average pixel size: {np.round(self.x_size,2)} pixels")
        ax[0].legend()
    
        ax[1].plot(self.freq_y, self.power_y)
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("Frequency [pixels $^{-1}$]")
        ax[1].set_ylabel("Columns Power Spectrum")
        ax[1].axvline(1/ self.y_size, color="r", linestyle="--", label = f"Avergae pixel size:  {np.round(self.y_size,2)} pixels")
        ax[1].legend()

        plt.show()

        if save_path is not None:

            save_path.parent.mkdir(parents=True, exist_ok=True)

            print(save_path.resolve())

            
            fig.savefig(save_path, dpi = 300, bbox_inches="tight")


  
      
    