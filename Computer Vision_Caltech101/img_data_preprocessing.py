
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

#Image_Path = "C:\\Soumick\\Prostate Classification task\\train1"
Image_Path = "C:\\Courses\\Selected Topics in Image Understanding\\101_ObjectCategories"
os.chdir(Image_Path)
list_fams = os.listdir(os.getcwd()) # vector of strings with family names
list_fams = sorted(list_fams, key=str.lower) #to ensure that the file names is sorted

img_files_list = [] 
labels_img_list = [] 
label_name = []
for directory, sub_directory, image_list in os.walk(Image_Path):
    #print(directory)
    #print(image_list)
    label_img_list = directory.split('\\')
    for image_name in image_list:
        if ".jpg" in image_name.lower():  # check whether the file's DICOM
            img_files_list.append(os.path.join(directory,image_name))
            #labels_img_list.append(label_name)
            label_name.append(label_img_list[-1])
    
# Get ref file
sample_ds = cv2.imread(img_files_list[0], 0)
plt.imshow(sample_ds, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
check_pixel_dim = (int(len(sample_ds[0])), int(len(sample_ds[1])), len(img_files_list))

'''# Load spacing values (in mm)
check_pixel_spacing = (float(sample_ds.PixelSpacing[0]), float(sample_ds.PixelSpacing[1]), float(sample_ds.SliceThickness))'''

# The array is sized based on 'check_pixel_dim'
image_array = []


# loop through all the DICOM files
for image in img_files_list:
    # read the file
    ds = cv2.imread(image, 0)
    
    img = cv2.resize(ds, (128, 128))
    
    #image_array[:, :] = img
    image_array.append(img)
'''im_array = np.asarray(image_array, dtype=np.float32)  '''  



