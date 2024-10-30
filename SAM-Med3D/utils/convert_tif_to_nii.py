import os
import numpy as np
import nibabel as nib
import tifffile as tiff


def convert_tif_to_nii(input_folder):
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            # 获取完整路径
            filepath = os.path.join(input_folder, filename)

            # 读取TIF图像
            tiff_image = tiff.imread(filepath)

            # 将TIF图像转换为NIfTI格式
            nii_image = nib.Nifti1Image(tiff_image, affine=np.eye(4))

            # 生成新的文件名（覆盖原始文件）
            nii_filepath = os.path.splitext(filepath)[0] + ".nii.gz"
            # print(nii_filepath)

            # # 保存为NIfTI格式
            nib.save(nii_image, nii_filepath)
            #
            # # 删除原始TIF文件
            os.remove(filepath)
            #
            # print(f"Converted {filename} to {nii_filepath} and deleted the original file.")


# 设置要处理的文件夹路径
# input_folder = '/data/kfchen/nnUNet/nnUNet_raw/Dataset181_deflu_gamma_nii/imagesTr'
# input_folder = '/data/kfchen/nnUNet/nnUNet_raw/Dataset181_deflu_gamma_nii/imagesTs'
input_folder = '/data/kfchen/nnUNet/nnUNet_raw/Dataset181_deflu_gamma_nii/labelsTr'

# 调用转换函数
convert_tif_to_nii(input_folder)
