from ast import Pass
from tokenize import Ignore

import matplotlib.pyplot as plt
import numpy as np


class speckle_image:
    def __init__(self, width, height, speckle_radius, bw_ratio):
        self.width = width
        self.height = height
        self.speckle_radius = speckle_radius
        self.bw_ratio = bw_ratio

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

    def circle(self, centre_x, centre_y, x, y):

        return (x - centre_x) ** 2 + (y - centre_y) ** 2 <= self.speckle_radius**2

    def speckle_pattern(self):

        # first lets just plot a dot at the centre of each speckle

        self.grid = np.full((self.height, self.width), 0)

        centre_pixels = np.floor(self.speckle_pos).astype(int)

        #want to pick out a pixel and check if it satisfies the circle equationfor any of the centre coordinates

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
    
    def image(self):

        plt.imshow(self.grid, cmap="binary")
        plt.show()