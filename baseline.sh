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
  --log_dir logs/test \
  --local_rank 4


python train.py \
  --image_dir data/waymo/processed/validation \
  --scene_names 0 1 2 \
  --log_dir logs/test


CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=12345 my_train.py \
  --image_dir data/waymo/processed/validation --batch_size 1 \
  --max_epoch 10

CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=12345 my_train.py \
  --image_dir data/waymo/processed/validation --batch_size 2 \
  --max_epoch 10 \
 # --ckpt_path logs/test/ckpt/model_final.pt


CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=12345 train.py \
  --image_dir data/waymo/processed/validation --batch_size 1 \
  --max_epoch 10 --log_dir logs/test

python datasets/preprocess_waymo.py \
    --data_root ../../dataset/waymo-scene-flow/ \
    --target_dir ../../dataset/waymo_processed/ \
    --dataset waymo \
    --split validation \
    --scene_list_file data/waymo_val_list.txt \
    --start_idx 69 \
    --num_scenes 133 \
    --num_workers 8 \
    --process_keys images lidar calib pose dynamic_masks ground \
    --json_folder_to_save ../../dataset/annotations/waymo


python main.py \
  --exp_name waymo_test_full \
  --image_dir ../../dataset/waymo_processed/validation --batch_size 1 \
  --max_epoch 50 --local_rank 4 \
  --save_image 5 --save_ckpt 10