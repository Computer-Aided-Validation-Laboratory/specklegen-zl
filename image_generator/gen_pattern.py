import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fft2, fftshift
from scipy.ndimage import zoom
from scipy.signal import find_peaks, peak_widths


class SpeckleImage:
    def __init__(self, width: int, height: int, speckle_radius: int, bw_ratio: float):
        self.width = width
        self.height = height
        self.speckle_radius = speckle_radius
        self.bw_ratio = bw_ratio

        self.grid = np.full((self.height, self.width), 0)

    def calculate_speckle_position(self):
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
        self.speckle_pos[:, 0] += np.random.uniform(
            -self.x_step / 2, self.x_step / 2, size=self.speckle_pos.shape[0]
        )
        self.speckle_pos[:, 1] += np.random.uniform(
            -self.y_step / 2, self.y_step / 2, size=self.speckle_pos.shape[0]
        )

        return self.speckle_pos

    def circle(self, centre_x: float, centre_y: float, x: float, y: float):

        return (x - centre_x) ** 2 + (y - centre_y) ** 2 <= self.speckle_radius**2

    def generate_speckle_pattern(self):
        """
        Generates the speckle pattern by finding the pixels that should be coloured black - these satisfy the circle equation.

        """
        
        """
        for x_centre,y_centre in centre_pixels:
            for x in range(x_centre - int(self.x_step), x_centre + int(self.x_step)):
                for y in range(y_centre - int(self.y_step), y_centre + int(self.y_step)):
                            if 0 <= x < self.width and 0 <= y < self.height and self.circle(x_centre, y_centre, x, y):
                                self.grid[y, x] = 1
        """
        centre_pixels = np.floor(self.speckle_pos).astype(int)
        self.grid[centre_pixels[:, 0], centre_pixels[:, 1]] = 1
        


        # want step size to be an integer
        rx = np.floor(self.x_step).astype(int)
        ry = np.floor(self.y_step).astype(int)

        # Precompute the circle mask once, this is if there was a circle at coordinates 0,0
        #later want to displace the mask essentially around the random centre coordinates
        dy, dx = np.ogrid[-ry:ry, -rx:rx]

        circle_mask = self.circle(0, 0, dx, dy)
        
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
        """
        Smooths the edges of the speckles using interpolation.

        """
        self.grid = zoom(self.grid, 2, order=2)

        return self.grid

    def FFT(self, plot=True):
        """
        Performs a 2D FFT on the speckle pattern to find average speckle size and spacing.

        """
        row_fft = fftshift(fft(self.grid, axis=0))
        column_fft = fftshift(fft(self.grid, axis=1))

        # find power spectrum

        power_spectrum_rows = np.abs(row_fft) ** 2
        power_spectrum_columns = np.abs(column_fft) ** 2

        peaks_x, _ = find_peaks(np.sum(power_spectrum_rows, axis=0))
        peaks_y, _ = find_peaks(np.sum(power_spectrum_columns, axis=1))

        peak_freq_x = peaks_x[np.argmax(np.sum(power_spectrum_rows, axis=0)[peaks_x])]
        peak_freq_y = peaks_y[
            np.argmax(np.sum(power_spectrum_columns, axis=1)[peaks_y])
        ]

        avg_speckle_size_x = 1 / peak_freq_x
        avg_speckle_size_y = 1 / peak_freq_y

        # plot x and y projections of the fft to find the frequency with the highest magnitude in each direction
        if plot == True:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].plot(np.sum(power_spectrum_rows, axis=0))
            ax[0].set_xscale("log")
            ax[0].set_yscale("log")
            ax[0].set_xlabel("Frequency (1/pixels)")
            ax[0].set_title("X projection of POWER SPECTRUM")
            ax[0].axvline(
                peak_freq_x, color="r", linestyle="--", label="Peak Frequency"
            )

            ax[1].plot(np.sum(power_spectrum_columns, axis=1))
            ax[1].set_xscale("log")
            ax[1].set_yscale("log")
            ax[1].set_xlabel("Frequency (1/pixels)")
            ax[1].set_title("Y projection of POWER SPECTRUM")
            ax[1].axvline(
                peak_freq_y, color="r", linestyle="--", label="Peak Frequency"
            )

            plt.show()
        # the power spectrum records the intensity of the speckles in the frequency domain - this is related to the size and spacing of the speckles in the spatial domain

        # fwhm of peak shown in figures plotted above is related to the average speckle size in the spatial domain - the larger the fwhm, the smaller the speckles in the spatial domain

        print(avg_speckle_size_x)
        print(avg_speckle_size_y)

        return avg_speckle_size_x, avg_speckle_size_y

    def show_image(self):

        plt.imshow(self.grid, cmap="binary")
        # turn off x and y axes labels
        # plt.axis('off')

        plt.show()

    def save_image(self, filename: str):
        plt.imsave(filename, self.grid, cmap="binary")
