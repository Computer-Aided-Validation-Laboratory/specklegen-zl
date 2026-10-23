import sys
import os
#from pathlib import Path


print("Working directory:", os.getcwd())
print("Python executable:", sys.executable)
print("Import paths:", sys.path)

#sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_generator import SpeckleImage

#Define image parameters 
image_width = 500

image_height = 500

speckle_radius = 5 #pixels

bw_balance = 0.5

#generate pattern with user-defined parameters 
pattern = SpeckleImage(image_width, image_height, speckle_radius, bw_balance)

#calculate positions of speckles
pattern.calculate_speckle_position()

#randomise the speckles
pattern.randomise_speckles()

#make speckle pattern
pattern.generate_speckle_pattern()

#smooth the edges of the speckles 
pattern.smooth_edges()

pattern.show_image()

#params.save_image("speckle_pattern.png")