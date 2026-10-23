import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fft2, fftshift, fftfreq
from scipy.ndimage import convolve
from scipy.signal import find_peaks, peak_widths


class SpeckleImage:
    def __init__(self, width: int, height: int, speckle_radius: int, bw_ratio: float):
        self.width = width
        self.height = height
        self.speckle_radius = speckle_radius
        self.bw_ratio = bw_ratio

        self.grid = np.full((self.height, self.width), 0)

    """
    def smooth_edges(self, scale = 2):

        self.width *= scale
        self.height *= scale 

        self.speckle_radius *= scale

        self.grid = np.full((self.height, self.width), 0)
    """
    def calculate_speckle_position(self, scale = 1):
        """
        Calculates the position of the centre of the speckles from the speckle radius and bw ratio.

        Outputs the position of the speckle centres as x and y coordinates.

        """

        speckle_area = np.pi * (self.speckle_radius**2)

        n_speckles = (self.bw_ratio * self.width * self.height) / speckle_area

        self.x_step = self.width / (np.sqrt(self.width / self.height * n_speckles))
        self.y_step = self.height / (np.sqrt(self.height / self.width * n_speckles))

        # circle centres:
        x_centres = np.arange(self.x_step, self.width, self.x_step)
        y_centres = np.arange(self.y_step, self.height, self.y_step)

        # make centres into coordinates
        self.speckle_pos = np.array(np.meshgrid(x_centres, y_centres)).T.reshape(-1, 2)

        return self.speckle_pos, self.x_step, self.y_step


    def randomise_speckles(self):
        """
        Randomises the positions of the speckles.
        """
        self.speckle_pos[:, 0] += np.random.uniform(-self.x_step / 2, self.x_step / 2, size=self.speckle_pos.shape[0])
        self.speckle_pos[:, 1] += np.random.uniform(-self.y_step / 2, self.y_step / 2, size=self.speckle_pos.shape[0])

        return self.speckle_pos

    def circle(self, centre_x: float, centre_y: float, x: float, y: float, radius: int):

        return (x - centre_x) ** 2 + (y - centre_y) ** 2 <= self.speckle_radius**2

    def generate_speckle_pattern(self):
        """
        Generates the speckle pattern by finding the pixels that should be coloured black - these satisfy the circle equation.

        """
        centre_pixels = np.floor(self.speckle_pos).astype(int)
        self.grid[centre_pixels[:, 0], centre_pixels[:, 1]] = 1

        # want step size to be an integer
        rx = np.floor(self.x_step).astype(int)
        ry = np.floor(self.y_step).astype(int)

        # Precompute the circle mask once, this is if there was a circle at coordinates 0,0
        #later want to displace the mask essentially around the random centre coordinates
        dy, dx = np.ogrid[-ry:ry, -rx:rx]

        circle_mask = self.circle(0, 0, dx, dy, self.speckle_radius)
        
        for xc, yc in centre_pixels:

            x0 = max(0, xc-rx)
            x1 = min(self.width, xc+rx)

            y0 = max(0, yc - ry)
            y1 = min(self.height, yc+ry)

            region = self.grid[x0:x1, y0:y1] 

            #need to crop mask 
            mx0 = x0 - (xc-rx)
            mx1 = mx0 + (x1 - x0)

            my0 = y0 - (yc - ry)
            my1 = my0 + (y1 - y0) 


            circle_cropped = circle_mask[mx0:mx1, my0:my1]

            region[circle_cropped] = 1


        print(f"Total plotted pixels: {np.sum(self.grid)}")

        print(f"Plotted bw ratio: {np.sum(self.grid) / (self.width * self.height)}")

        return self.grid

    def smooth_edges(self):

        kernel = np.ones((3, 3)) / 9

        smoothed = convolve(self.grid.astype(float),kernel,mode="constant")

        self.grid = smoothed

        return self.grid
        
    def FFT(self):
        """
        Performs a 2D FFT on the speckle pattern to find average speckle size and spacing.

        """
        freq_x = fftshift(fftfreq(self.width))

        freq_y = fftshift(fftfreq(self.height))

        mask_x = freq_x > 0

        mask_y = freq_y > 0

        freq_x_f = freq_x[mask_x]
        freq_y_f = freq_y[mask_y]

        Fx = fftshift(fft(self.grid,axis = 0))
        Fy = fftshift(fft(self.grid,axis = 1))
        #Fx = fftshift(fft(self.grid, axis = 0))
        power_x = np.sum(np.abs(Fx)**2, axis = 1)[mask_x]

        #Fy = fftshift(fft(self.grid, axis = 1))
        power_y = np.sum(np.abs(Fy)**2, axis = 0)[mask_y]

        xi = np.argmin(power_x)
        yi = np.argmin(power_y)

        peak_freq_x = freq_x_f[xi]
        x_size = 1/ peak_freq_x 

        peak_freq_y = freq_y_f[yi]
        y_size = 1/ peak_freq_y

        return freq_x_f, freq_y_f, power_x, power_y, x_size, y_size


    def plot_FFT(self, save : bool = False, filename: str | None = None):

        fig, ax = plt.subplots(1,2, figsize = (10,6))

        freq_x_f, freq_y_f, power_x, power_y, x_size, y_size = self.FFT()

        ax[0].plot(freq_x_f, power_x)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("Frequency [pixels $^{-1}$}]")
        ax[0].set_ylabel("Rows Power Spectrum")
        ax[0].axvline(1 / x_size, color = "r", linestyle = "--", label = f" Average pixel size: {np.round(x_size,2)} pixels")
        ax[0].legend()

        ax[1].plot(freq_y_f, power_y)
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("Frequency 1/pixels")
        ax[1].set_ylabel("Columns Power Spectrum")
        ax[1].axvline(1/ y_size, color="r", linestyle="--", label = f"Avergae pixel size:  {np.round(y_size,2)} pixels")
        ax[1].legend()

        if save:
            if filename is None:
                filename = "FFT_plot.png"
        
            plt.savefig(filename, bbox_inches="tight")


        plt.show()

        plt.close()

    def show_image(self):

        plt.imshow(self.grid, cmap="binary")
        # turn off x and y axes labels
        # plt.axis('off')

        plt.show()

    def save_image(self, filename: str | None = None):

        if filename is None:
            filename = "speckle_pattern.png"

        plt.imsave(filename, self.grid, cmap="binary")
