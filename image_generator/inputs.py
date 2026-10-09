"""Configure and generate a speckle image."""


from gen_pattern import speckle_image

image_width = 1000

image_height = 1000

speckle_radius = 5 #pixels

bw_balance = 0.4



params = speckle_image(image_width, image_height, speckle_radius, bw_balance)


params.calculate_speckle_position()

params.speckle_pattern()

params.image()




