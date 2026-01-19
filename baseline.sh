python render.py --image_dir davis/bear --scene_names 3 5 7 --input_views 1 --sequence_length 40 --start_idx 0 --mode 2 \
    --ckpt_path checkpoints/model_latest_waymo.pt --output_path output -images

python datasets/preprocess_waymo.py \
    --data_root ../dataset/waymo-dataset/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split validation \
    --scene_list_file data/waymo_val_list.txt \
    --scene_ids 0 1 2 \
    --num_workers 8 \
    --process_keys images lidar calib pose dynamic_masks ground \
    --json_folder_to_save data/annotations/waymo

python datasets/tools/extract_masks.py \
    --data_root data/waymo/processed/validation \
    --segformer_path=../SegFormer \
    --checkpoint=checkpoints/segformer.b5.1024x1024.city.160k.pth \
    --split_file data/waymo_example_scenes.txt \
    --process_dynamic_mask

python inference.py \
    --image_dir data/waymo/processed/validation \
    --scene_names 0 1 2 \
    --input_views 1 \
    --intervals 2 \
    --sequence_length 4 \
    --start_idx 0 \
    --mode 2 \
    --ckpt_path checkpoints/model_latest_waymo.pt \
    --output_path output/inference \
    -images \
    -depth \
    -metrics


torchrun --nproc_per_node=1 --master_port=0000 my_train.py \
  --image_dir data/waymo/processed/validation \
  --scene_names 0 1 2 \
  --log_dir logs/test


python my_train.py \
  --image_dir data/waymo/processed/validation \
  --scene_names 0 1 2 \
  --log_dir logs/test


python train.py \
  --image_dir data/waymo/processed/validation \
  --scene_names 0 1 2 \
  --log_dir logs/test