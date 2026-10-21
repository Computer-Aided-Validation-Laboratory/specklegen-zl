"""Configure and generate a speckle image."""

from gen_pattern import SpeckleImage

#First we want to define the parameters used in generating the speckle pattern:

image_width = 500

image_height = 500

speckle_radius = 5 #pixels

bw_balance = 0.5

params = SpeckleImage(image_width, image_height, speckle_radius, bw_balance)


params.calculate_speckle_position()

params.randomise_speckles()

params.generate_speckle_pattern()

#params.smooth_edges()

#params.FFT()


params.show_image()


#params.save_image("speckle_pattern.png")
