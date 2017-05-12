import numpy as np
import scipy
from scipy import io
from PIL import Image
import os
import re

def my_savemat(img, filename, var):
    (w, h) = img.size
    d = len(img.getbands())
    npa = np.array(img.getdata()).reshape(h, w, d)
    scipy.io.savemat(filename, {var:npa}) 

in_data_dir = "./train_data/291/"
out_data_dir = in_data_dir + "train/"

if not os.path.exists(out_data_dir):
    os.mkdir(out_data_dir)

file_list = [f for f in os.listdir(in_data_dir) if re.search(r'.*\.(jpg|bmp)$', f)]
file_list = map(lambda x:in_data_dir+x, file_list)

# for naming output files
count = 0

for f in file_list:

    print ("Processing ", f, "...\n")

    ext = os.path.splitext(f)[1]
    ext = ""

    img_in = Image.open(f)
    (width, height) = img_in.size

    # Make it a multiple of 12 (for /2, /3, /4 scaling)
    img_raw = img_in.crop((0, 0, width-(width%12), height-(height%12)))

    (width, height) = img_raw.size
    patch_size = 41
    stride = 41
    x_size = (width-patch_size)//stride+1
    y_size = (height-patch_size)//stride+1

    img_2 = img_raw.resize((width//2, height//2), Image.BICUBIC)
    img_3 = img_raw.resize((width//3, height//3), Image.BICUBIC)
    img_4 = img_raw.resize((width//4, height//4), Image.BICUBIC)

    for x in range(0, x_size, stride):
        for y in range(0, y_size, stride):

            # Original orientation
            patch_name = str(count)

            patch = img_raw.crop((x, y, x+patch_size-1, y+patch_size-1))
            my_savemat(patch, out_data_dir + patch_name + ext, "patch")
            patch = img_2.crop((x, y, x+patch_size-1, y+patch_size-1))
            my_savemat(patch, out_data_dir + patch_name + "_2" + ext, "patch")
            patch = img_3.crop((x, y, x+patch_size-1, y+patch_size-1))
            my_savemat(patch, out_data_dir + patch_name + "_3" + ext, "patch")
            patch = img_4.crop((x, y, x+patch_size-1, y+patch_size-1))
            my_savemat(patch, out_data_dir + patch_name + "_4" + ext, "patch")

            count = count+1;

            # Rotate 90
            patch_name = str(count)

            patch = img_raw.crop((x, y, x+patch_size-1, y+patch_size-1)).rotate(90)
            my_savemat(patch, out_data_dir + patch_name + ext, "patch")
            patch = img_2.crop((x, y, x+patch_size-1, y+patch_size-1)).rotate(90)
            my_savemat(patch, out_data_dir + patch_name + "_2" + ext, "patch")
            patch = img_3.crop((x, y, x+patch_size-1, y+patch_size-1)).rotate(90)
            my_savemat(patch, out_data_dir + patch_name + "_3" + ext, "patch")
            patch = img_4.crop((x, y, x+patch_size-1, y+patch_size-1)).rotate(90)
            my_savemat(patch, out_data_dir + patch_name + "_4" + ext, "patch")

            count = count+1

            # LR flip
            patch_name = str(count)

            patch = img_raw.crop((x, y, x+patch_size-1, y+patch_size-1)).transpose(Image.FLIP_LEFT_RIGHT)
            my_savemat(patch, out_data_dir + patch_name + ext, "patch")
            patch = img_2.crop((x, y, x+patch_size-1, y+patch_size-1)).transpose(Image.FLIP_LEFT_RIGHT)
            my_savemat(patch, out_data_dir + patch_name + "_2" + ext, "patch")
            patch = img_3.crop((x, y, x+patch_size-1, y+patch_size-1)).transpose(Image.FLIP_LEFT_RIGHT)
            my_savemat(patch, out_data_dir + patch_name + "_3" + ext, "patch")
            patch = img_4.crop((x, y, x+patch_size-1, y+patch_size-1)).transpose(Image.FLIP_LEFT_RIGHT)
            my_savemat(patch, out_data_dir + patch_name + "_4" + ext, "patch")

            count = count+1

            # rotate 90, then LR flip
            patch_name = str(count)

            patch = img_raw.crop((x, y, x+patch_size-1, y+patch_size-1)).rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
            my_savemat(patch, out_data_dir + patch_name + ext, "patch")
            patch = img_2.crop((x, y, x+patch_size-1, y+patch_size-1)).rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
            my_savemat(patch, out_data_dir + patch_name + "_2" + ext, "patch")
            patch = img_3.crop((x, y, x+patch_size-1, y+patch_size-1)).rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
            my_savemat(patch, out_data_dir + patch_name + "_3" + ext, "patch")
            patch = img_4.crop((x, y, x+patch_size-1, y+patch_size-1)).rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
            my_savemat(patch, out_data_dir + patch_name + "_4" + ext, "patch")

            count = count+1
