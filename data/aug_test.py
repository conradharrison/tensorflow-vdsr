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

in_data_dir = "./test_data/Set14/"
out_data_dir = in_data_dir + "test/"

if not os.path.exists(out_data_dir):
    os.mkdir(out_data_dir)

file_list = [f for f in os.listdir(in_data_dir) if re.search(r'.*\.bmp$', f)]
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

    patch_name = str(count)

    my_savemat(img_raw, out_data_dir + patch_name + ext, "img_raw")
    my_savemat(img_raw.resize((width//2, height//2), Image.BICUBIC), out_data_dir + patch_name + "_2" + ext, "img_2")
    my_savemat(img_raw.resize((width//3, height//3), Image.BICUBIC), out_data_dir + patch_name + "_3" + ext, "img_3")
    my_savemat(img_raw.resize((width//4, height//4), Image.BICUBIC), out_data_dir + patch_name + "_4" + ext, "img_4")

    count = count+1

