import os

validate_root = "../../dataset/waymo-scene-flow/valid/"
outputfile = "waymo_val_list.txt"

with open(outputfile, 'w', encoding='utf-8') as f:
    # 遍历文件夹
    for root, dirs, files in os.walk(validate_root):
        for file in files:
            filename = file.split(".")[0]
            f.write(filename + '\n')
    