# maskflow
Calculate flow field of an image pair and mask people in the flow field.

## Dependencies
### for Mask RCNN:
numpy   
scipy   
Pillow  
cython  
matplotlib  
scikit-image    
tensorflow>=1.3.0   
keras>=2.0.8  
opencv-python 
h5py  
imgaug  
IPython[all]  
coco PythonAPI (run `make` and `python setup.py build_ext install` in /coco/PythonAPI/ folder)
### for liteflownet:
pytorch   
cupy

## Usage
```# from root directory
import maskflow
first_image = ... # an image array
second_image = ... # another image array
num_people, color_images = maskflow.maskPeople(first_image, second_image)
```
`first_image` and `second_image` form the image pair for optical flow calculation. shape: (height, width, 3)   
`num_people` is the number of people detected in the first image.   
`color_images` contains images with masked flow for each person detected. shape: (num_peope, height, width, 3)
