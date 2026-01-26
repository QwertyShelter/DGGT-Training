# DGGT-Training 复现

添加文件如下 `main.py`, `main.ipynb`, `my_train.py`, `pts3d_batch.py`, `utils/dataset_tool.py`; 其中 `train.py` 为官方提供训练代码
- `main.py`: 单卡训练代码，现在已经跑通
- `main.ipynb`: 单次推理以及 3D 点云可视化脚本，现已跑通
- `my_train.py`: 多卡训练代码初始版本，现在已经迁移到 `main_count.py` 中
- `main_count.py`: 目前使用的多卡训练代码，包含时间和内存监测，现已跑通
- `pts3d_batch.py`: 批量推理以及 3D 点云可视化，现已跑通

同时，对 `datasets/dataset.py`, `dggt/utils/pose_enc.py`, `dggt/models/vggt.py` 有所修改，主要是删除了 vggt 原本的 `SkyGaussian Head` 