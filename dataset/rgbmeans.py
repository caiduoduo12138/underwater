#compute RGBmean
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

R_channel = 0
G_channel = 0
B_channel = 0
R_channel_square = 0
G_channel_square = 0
B_channel_square = 0
pixels_num = 0

imgs = []
root_path ='/home/cai/data/experiment/images/' # change your image path here!
f = os.listdir(root_path)
len_ = len(f)
for i,filename in enumerate(f):
    img_path = root_path + '{}'.format(filename)
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    pixels_num += h * w

    R_temp = img[:, :, 0]
    R_channel += np.sum(R_temp)
    R_channel_square += np.sum(np.power(R_temp, 2.0))
    G_temp = img[:, :, 1]
    G_channel += np.sum(G_temp)
    G_channel_square += np.sum(np.power(G_temp, 2.0))
    B_temp = img[:, :, 2]
    B_channel = B_channel + np.sum(B_temp)
    B_channel_square += np.sum(np.power(B_temp, 2.0))
    print('{0}/{1}'.format(i,len_))

R_mean = R_channel / pixels_num
G_mean = G_channel / pixels_num
B_mean = B_channel / pixels_num

R_mean_norm = R_mean/255.0
G_mean_norm = G_mean/255.0
B_mean_norm = B_mean/255.0

"""   
S^2
= sum((x-x')^2 )/N = sum(x^2+x'^2-2xx')/N
= {sum(x^2) + sum(x'^2) - 2x'*sum(x) }/N
= {sum(x^2) + N*(x'^2) - 2x'*(N*x') }/N
= {sum(x^2) - N*(x'^2) }/N
= sum(x^2)/N - x'^2
"""

R_std = np.sqrt(R_channel_square / pixels_num - R_mean * R_mean)
G_std = np.sqrt(G_channel_square / pixels_num - G_mean * G_mean)
B_std = np.sqrt(B_channel_square / pixels_num - B_mean * B_mean)

R_std_norm = R_std/255.0 
G_std_norm = G_std/255.0 
B_std_norm = B_std/255.0 

print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_std is %f, G_std is %f, B_std is %f" % (R_std, G_std, B_std))
print('\n')
print("R_mean_norm is %f, G_mean_norm is %f, B_mean_norm is %f" % (R_mean_norm, G_mean_norm, B_mean_norm))
print("R_std_norm is %f, G_std_norm is %f, B_std_norm is %f" % (R_std_norm, G_std_norm, B_std_norm))
