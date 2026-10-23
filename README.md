# specklegen-zl

# Overview
This is a Python Package that generates a simple random speckle pattern using NumPy based on user inputs such as image dimensions, speckle radius and black/white balance. The user can also analyse the image using fast fourier transforms, which output the average speckle size in the x and y directions. 

# Features 
- Generate speckle patterns using SpeckleImage.
- Visualise the speckle patterns.
- Save the generated patterns in a lossless format.
- Analyse the speckle pattern using FourierAnalysis.
- Plot and save the power spectrum of the speckle pattern to ind average speckle sizes.

# Installation
## Requirements
- Python 3.12 or newer 

Clone the repository: 
```
git clone <repository-url>
cd specklegen-zl
```
Install the package:
```
python -m pip install .
```
For development, install in editable mode instead:

```
python -m pip install -e .
```

## Structure
The package has two source code modules - one for generating the speckle pattern and one for the Fourier analysis.
- ``speckle_pattern.py`` contains the class SpeckleImage. The user can input image dimensions, black/white balance and speckle radius. The code first generates speckle positions uniformly distributed in a grid. These positions are randomly displaced. The speckles are smoothed so they do not appear 'blocky' and instead there is a gradient between the white background and black speckles (or vice versa if the user chooses to invert the image for white speckles on a black background). The user can the plot and save the image.
- ``FFT_analysis.py`` contains the class FourierAnalysis, which performs a fast fourier transform (fft) on the speckle pattern to find the average speckle size in the horizontal and vertical directions.
```
.gitignore
LICENSE
README.md
pyproject.toml
example_images
src/
|
|----generator/
|      |---- __init__.py
|      |---- speckle_pattern.py
|      |---- FFT_analysis.py
|
|----examples
       |----example_speckle_pattern.py
```

