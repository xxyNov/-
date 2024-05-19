import os

import cv2
import numpy as np
import time

from setuptools.sandbox import save_path


def read_images(left_image_path, right_image_path):
    # 以灰度模式读取图像
    left_image = cv2.imread(left_image_path, 0)
    right_image = cv2.imread(right_image_path, 0)
    return left_image, right_image

def ncc(left_block, right_block):
    # 计算两个块之间的归一化互相关（NCC）
    product = np.mean((left_block - left_block.mean()) * (right_block - right_block.mean()))
    stds = left_block.std() * right_block.std()

    if stds == 0:
        return 0
    else:
        return product / stds

def ssd(left_block, right_block):
    # 计算两个块之间的平方差之和（SSD）
    return np.sum(np.square(np.subtract(left_block, right_block)))

def sad(left_block, right_block):
    # 计算两个块之间的绝对差之和（SAD）
    return np.sum(np.abs(np.subtract(left_block, right_block)))

def select_similarity_function(method):
    # 根据方法名称选择相似性度量函数
    if method == 'ncc':
        return ncc
    elif method == 'ssd':
        return ssd
    elif method == 'sad':
        return sad
    else:
        raise ValueError("未知方法")

def compute_disparity_map(left_image, right_image, block_size, disparity_range, method='ncc'):
    # 初始化视差图
    height, width = left_image.shape
    disparity_map = np.zeros((height, width), np.uint8)
    half_block_size = block_size // 2
    similarity_function = select_similarity_function(method)

    # 遍历图像中的每个像素
    for row in range(half_block_size, height - half_block_size):
        for col in range(half_block_size, width - half_block_size):
            best_disparity = 0
            best_similarity = float('inf') if method in ['ssd', 'sad'] else float('-inf')

            # 定义一个基于当前像素的比较块
            left_block = left_image[row - half_block_size:row + half_block_size + 1,
                         col - half_block_size:col + half_block_size + 1]

            # 遍历不同的视差
            for d in range(disparity_range):
                if col - d < half_block_size:
                    continue

                # 定义用于比较的第二个块
                right_block = right_image[row - half_block_size:row + half_block_size + 1,
                              col - d - half_block_size:col - d + half_block_size + 1]

                # 计算相似性度量
                similarity = similarity_function(left_block, right_block)

                # 如有必要，更新最佳相似性和视差
                if method in ['ssd', 'sad']:
                    # 对于SSD和SAD，我们对最小值感兴趣
                    if similarity < best_similarity:
                        best_similarity = similarity
                        best_disparity = d
                else:
                    # 对于NCC，我们对最大值感兴趣
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_disparity = d

            # 将最佳视差赋给视差图
            disparity_map[row, col] = best_disparity * (256. / disparity_range)

    return disparity_map

def main():
    # 定义输入图像的路径
    left_image_path = 'im2.png'
    right_image_path = 'im6.png'

    # 加载图像
    left_image, right_image = read_images(left_image_path, right_image_path)

    # 记录开始时间
    start_time = time.time()

    # 定义块大小和视差范围
    block_size = 21
    disparity_range = 64

    # 指定相似性度量方法 ('ncc', 'ssd', 或 'sad')
    method = 'ssd'  # 更改此字符串以在方法间切换

    # 使用选定的方法计算视差图
    disparity_map = compute_disparity_map(left_image, right_image, block_size, disparity_range, method=method)

    # 指定保存视差图的文件夹路径
    save_path = 'D:\Github\CV-5\cvLab\CV-5_4\\results'  # 确保替换为实际的文件夹路径

    # 确保save_path是字符串类型
    if not isinstance(save_path, str):
        raise TypeError("save_path must be a string")

    # 如果文件夹不存在，则创建它
    if not os.path.exists(save_path):
        os.makedirs(save_path)

        # 组合文件夹路径和文件名
    save_image_path = os.path.join(save_path, 'ssd_size_21.png')

    # 保存图像到文件
    cv2.imwrite(save_image_path, disparity_map)

    # 为了显示，调整视差图的大小
    scale_factor = 3.0
    resized_image = cv2.resize(disparity_map, (0, 0), fx=scale_factor, fy=scale_factor)

    # 显示结果
    cv2.imshow('disparity_map_resized', resized_image)
    print('时间消耗:', time.time() - start_time)

    # 等待按键后关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()