from generator import SpeckleImage
from generator import FourierAnalysis
from pathlib import Path



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
pattern.generate_speckle_pattern(inverted= False)

#smooth the edges of the speckles 
pattern.smooth_edges()

#plotting speckle pattern
pattern.show_image()

#save the speckle pattern to desired folder
pattern.save_image(save_path= Path("example_images/speckle_pattern.png"))

#FFT analysis

fft = FourierAnalysis(pattern.grid)

#perform fast fourier transform to find average speckle size
fft.FFT()

#plot and save fft graphs. 
fft.plot_FFT( save_path=Path("example_images/FFT_analysis.png") )