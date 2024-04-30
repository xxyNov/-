python image_stitching.py --images images/datasets_name --output output_name.png --crop 1

其中`images/datasets_name`为待拼接图像所在文件夹，`output_name.png`为处理拼接保存后的图像；这里使用了相对路径，因为在项目根目录下运行了终端,不确定在根目录最好使用完整的绝对路径。` --crop 1`为是否裁剪黑色边框，缺省则不裁剪。