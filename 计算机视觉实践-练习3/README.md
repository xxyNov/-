训练后的权重文件存储在test文件夹下，将需要处理的PNG图片存储在test/LR/目录下，然后在控制台运行

python test.py --model_name [model file name]命令，例如python test.py --model_name 4pp_eusr_pirm.pb，输出结果将存储在test/SR/目录下

set5数据集的输出结果在results文件夹下的outputs目录下。

计算PSNR、SSIM指标值的代码为项目中的PSNR_SSIM.py文件。