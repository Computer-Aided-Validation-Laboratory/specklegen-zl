import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.ndimage import convolve
from pathlib import Path


class SpeckleImage:
    """
    A class that creates a random speckle pattern.

    Attributes: 
        width, height (int): image dimensions (pixels)
        speckle_radius (int): radius of circles that are the speckles in the speckle pattern (pixels)
        bw_ratio (float): proportions of pixels that are 'speckles' 
    
    """



    def __init__(self, width: int, height: int, speckle_radius: int, bw_balance: float):
        self.width: int = width
        self.height: int = height
        self.speckle_radius: int = speckle_radius
        self.bw_balance: float = bw_balance

        self.grid = np.full((self.height, self.width), 1)

        self.speckle_pos = None
        self.x_step = None
        self.y_step = None


    def calculate_speckle_position(self):
        """
        Calculates the position of the centre of the speckles from the speckle radius and bw ratio.

        Args 
        ---------
            None

        Returns
        ---------
            speckle_pos (np.ndarray): an array of the uniformly spaced speckle centre coordinates.
            x_step (float): the spacing between speckle centres in the x-axis.
            y_step (float): the spacing between speckle centres in the y-axis.


        """

        speckle_area = np.pi * (self.speckle_radius**2)

        n_speckles = (self.bw_balance * self.width * self.height) / speckle_area

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
        Randomises the speckles by randomly displacing the centres. 

            Args 
            ---------
                None
        
            Returns
            ---------
                speckle_pos (np.ndarray): randomised coordinates of the speckle centres. 
        
        
        """
        
        self.speckle_pos[:, 0] += np.random.uniform(-self.x_step / 2, self.x_step / 2, size=self.speckle_pos.shape[0])
        self.speckle_pos[:, 1] += np.random.uniform(-self.y_step / 2, self.y_step / 2, size=self.speckle_pos.shape[0])

        return self.speckle_pos

    def circle(self, centre_x: float, centre_y: float, x: float, y: float):
        """
        Determines if a pixel is within the bounds of a circle

            Args
            ----------
                centre_x (float): x coordinate of the centre.
                centre_y (float): y coordinate of the centre.
                x (float): x coordinate that is being checked.
                y (float): y coordiante that is being checked.
            
            Returns
            ----------
                bool: do the pixel coordinates satisfy the circle equation.
        
        """

        return (x - centre_x) ** 2 + (y - centre_y) ** 2 <= self.speckle_radius**2

    def generate_speckle_pattern(self, inverted: bool = False):
        """
        Generates the speckle pattern by finding which pixels make up the speckles.

            Args
            ----------
                inverted (bool): if set to True then the speckle pattern will be white speckles on a black background
            
            Returns
            ----------
                grid (np.ndarray): an array of zeros and ones which forms the speckle image
                                    0 is coloured black and 1 is coloured white. 

        """
        centre_pixels = np.floor(self.speckle_pos).astype(int)
        self.grid[centre_pixels[:, 0], centre_pixels[:, 1]] = 0

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

            region[circle_cropped] = 0


        print(f"Total plotted pixels: {np.sum(self.grid)}")

        print(f"Plotted bw balance: {1 - np.sum(self.grid) / (self.width * self.height)}")

        if inverted == True:
            self.grid = 1-self.grid

        return self.grid

    def smooth_edges(self):
        """
        Smooths the edges of the speckles in the image.

            Args
            ----------
                None

            Returns
            ----------
                grid (np.ndarray): array now consists of floats between 0 and 1 which represent greyscale.
        """

        kernel = np.ones((3, 3)) / 9

        smoothed = convolve(self.grid.astype(float),kernel,mode="constant")

        self.grid = smoothed

        return self.grid


    def show_image(self):
        """
        Displays the speckle image.

            Args
            ----------
                None
            
            Returns
            ----------
                None (just shows image)
        """

        np.clip(self.grid,0,1)

        plt.imshow(self.grid, cmap="gray")


        plt.show()

    def save_image(self, bits: int = 8, save_path : Path | None = None):
        """
        Saves the speckle image in a lossless format.

            Args
            ---------
                bits (int): number of bits that encode the saved image (8-bit, 16-bit).
                save_path (str): if left empty then will not save, otherwise image will save as specified

            Returns
            ---------
                None (just saves image)
        """

        max_val = (1 << bits) - 1

        if bits == 8:
            dtype = np.uint8
        else: 
            dtype = np.uint16

        out = np.rint(self.grid * max_val).astype(dtype)


        if save_path is not None:
            save_path.parent.mkdir(parents = True, exist_ok = True)
            print(save_path.resolve())



