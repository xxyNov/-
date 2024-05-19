import cv2
import numpy as np


#  image2 是目标图像，image1 是原始图像

H = np.load('Matrix_result/homography_matrix_3.npy')

image1 = cv2.imread('test_images/3.2.jpg')
image2 = cv2.imread('test_images/3.1.jpg')

points_src = np.float32([[0, 0], [0, image2.shape[0]], [image2.shape[1], image2.shape[0]], [image2.shape[1], 0]]).reshape(-1, 1, 2)
points_dst = np.float32([[0, 0], [0, image1.shape[0]], [image1.shape[1], image1.shape[0]], [image1.shape[1], 0]]).reshape(-1, 1, 2)
H_inv, _ = cv2.findHomography(points_dst, points_src, cv2.RANSAC, 5.0)

restored_image = cv2.warpPerspective(image2, H_inv, (image1.shape[1], image1.shape[0]))


# 显示原始图像、目标图像和逆变换后的图像
cv2.imshow('Original Image', image1)
cv2.imshow('Target Image', image2)
cv2.imshow('Restored Image', restored_image)

# 等待按键后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()