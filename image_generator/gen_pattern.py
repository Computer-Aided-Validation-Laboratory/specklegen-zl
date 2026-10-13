from ast import Pass

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


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

        self.x_step = self.width / (np.sqrt(self.width/self.height * n_speckles))
        self.y_step = self.height / (np.sqrt(self.height/self.width * n_speckles))

        # circle centres:
        x_centres = np.arange(self.x_step / 2, self.width, self.x_step)
        y_centres = np.arange(self.y_step / 2, self.height, self.y_step)


        # make centres into coordinates
        self.speckle_pos = np.array(np.meshgrid(x_centres, y_centres)).T.reshape(-1, 2)

        return self.speckle_pos, self.x_step, self.y_step

    def randomise_speckles(self):
        """
        Randomises the positions of the speckles.

        """
        self.speckle_pos[:,0] += np.random.uniform(-self.x_step, self.x_step, size=self.speckle_pos.shape[0])
        self.speckle_pos[:,1] += np.random.uniform(-self.y_step, self.y_step, size=self.speckle_pos.shape[0])

        return self.speckle_pos

    def circle(self, centre_x: float, centre_y: float, x: float, y: float):

        return (x - centre_x) ** 2 + (y - centre_y) ** 2 <= self.speckle_radius**2

    def generate_speckle_pattern(self):
        """
        Generates the speckle pattern by finding the pixels that should be coloured black - these satisfy the circle equation.

        """

        centre_pixels = np.floor(self.speckle_pos).astype(int)

        #want to pick out a pixel and check if it satisfies the circle equation for any of the centre coordinates

        #want to speed this up
        #instead check pixels that are within a step size of the sentre

        for x_centre,y_centre in centre_pixels:

            for x in range(x_centre - int(self.x_step), x_centre + int(self.x_step)):
                for y in range(y_centre - int(self.y_step), y_centre + int(self.y_step)):
                            if 0 <= x < self.width and 0 <= y < self.height and self.circle(x_centre, y_centre, x, y):
                                self.grid[y, x] = 1

        print(f"Total plotted pixels: {np.sum(self.grid)}")

        print(f"Plotted bw ratio: {np.sum(self.grid) / (self.width * self.height)}")

        
        return self.grid
    
    def smooth_edges(self):
        """
        Smooths the edges of the speckles using interpolation.

        """
        self.grid = zoom(self.grid, 2, order=2)

        return self.grid
        
    def show_image(self):

        plt.imshow(self.grid, cmap="binary")
        #turn off x and y axes labels
        plt.axis('off')

        plt.show()

    def save_image(self, filename: str):
        plt.imsave(filename, self.grid, cmap="binary")

