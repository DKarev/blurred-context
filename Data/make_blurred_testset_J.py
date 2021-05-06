from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import pandas as pd
import os
import itertools

object_data = pd.read_csv('/home/dimitar/experiments_I_and_J/expIJ/test_expJ_Color_oriimg.txt', header=None)
binary_data = pd.read_csv('/home/dimitar/experiments_I_and_J/expIJ/test_expJ_Color_binimg.txt', header=None)
oriimage_path = '/home/mengmi/Projects/Proj_context2/Matlab/Stimulus/keyframe_expH'
binimage_path = '/home/mengmi/Projects/Proj_context2/Matlab/Stimulus/keyframe_expA'



blur_list = [0, 1, 2, 4, 8, 16, 32, 64]

results_ori = 'Datasets/MSCOCO/testColor_blurimg_J'
for blur in blur_list:
    for i in range(1, 56):
        stri = str(i)
        if i<=9:
            stri = '0' + str(i)
            
        new_folder = os.path.join(results_ori + '_' + str(blur), 'cate' + stri)
        if os.path.exists(new_folder) is False:
            os.makedirs(new_folder)

prev_s = ''
data = zip(object_data.iterrows(), binary_data.iterrows())
for (_, s), (_, s1) in data:

    if prev_s != s[0].split('/')[0]:
        prev_s = s[0].split('/')[0]
        print("Processing: " + str(prev_s))

    # Read the image and its binary mask
    image = Image.open(os.path.join(oriimage_path, s[0]))
    bin_mask = np.array(Image.open(os.path.join(binimage_path, s1[0])))

    # Find the corners of the bounding box
    A = np.argwhere(bin_mask >= 200)
    top, left = A[0]
    bottom, right = A[-1]
    if bottom < A[-2][0] or right < A[-2][0]:
        bottom, right = A[-2]

    # Create a mask around the target object
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([ (left,bottom), (right,top) ], fill=255)
        
    for blur in blur_list:
        # Blur the image but preserve the target object
        blurred = image.filter(ImageFilter.GaussianBlur(blur))
        blurred.paste(image, mask=mask)

        # Save the image
        results = results_ori + ('_' + str(blur))
        new_file = os.path.join(results_ori + '_' + str(blur), s[0])
        blurred.save(new_file)

    
