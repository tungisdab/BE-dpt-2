import numpy as np


np.set_printoptions(threshold=np.inf)
check_feature = np.load(r"E:\codeKHANH\video_recognition\API2\feature_new\cats\v2_cat.mp4.npy")
print(check_feature.shape)
print(check_feature)