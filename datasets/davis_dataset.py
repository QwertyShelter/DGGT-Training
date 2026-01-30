
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets.dataset import load_and_preprocess_images

class DavisDataset(Dataset):
    def __init__(self, data_path, interval=1, seq_len=4, partial=False):
        
        self.data_list = []
        self.seq_len = seq_len
        self.interval = interval
        
        if partial:
            test_list = [  
                            "boxing-fisheye",
                            "car-shadow",
                            "car-roundabout",
                            "car-turn",
                            "cat-girl",
                            "color-run",
                            "cows",
                            "drift-turn",
                            "india",
                            "lucia",
                            "scooter-gray",
                            "varanus-cage",
                            "walking"
                        ]
            
        self.image_path = os.path.join(data_path, 'JPEGImages', '480p')
        self.dymask_path = os.path.join(data_path, 'Annotations', '480p')
        self.skymask_path = os.path.join(data_path, 'sky_masks')

        for item in os.listdir(self.image_path):
            if not partial or item in test_list:
                item_path = os.path.join(self.image_path, item)
                if os.path.isdir(item_path):
                    self.data_list.append(item)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        data_name = self.data_list[idx]

        image_path_list = []
        skymask_path_list = []
        dymask_path_list = []

        for file in os.listdir(os.path.join(self.image_path, data_name)):
            filename = file.split('.')[0]
            image_path_list.append(os.path.join(self.image_path, data_name, filename + '.jpg'))
            skymask_path_list.append(os.path.join(self.skymask_path, data_name, filename + '.png'))
            dymask_path_list.append(os.path.join(self.dymask_path, data_name, filename + '.png'))

        idx = np.linspace(0, len(image_path_list) - 1, num=self.seq_len, dtype=int)
        
        image_seq = [image_path_list[i] for i in idx]
        images = load_and_preprocess_images(image_seq)          # [T, C, H, W]

        skymask_seq = [skymask_path_list[i] for i in idx]
        skymasks = load_and_preprocess_images(skymask_seq)      # [S, C, H, W]

        dymask_seq = [dymask_path_list[i] for i in idx]
        dymasks = load_and_preprocess_images(dymask_seq)        # [S, C, H, W]

        
        sequence_length = images.shape[0]
        indices = [i * self.interval for i in range(sequence_length)]
        intervals = [self.interval for _ in range(sequence_length - 1)]
        
        timestamps = np.array(indices)
        timestamps = timestamps / timestamps[-1] * (sequence_length / 4)

        input_dict = {
            "name": data_name,
            "images": images,
            "masks": skymasks,
            "dynamic_mask": dymasks,
            "timestamps": timestamps,
            "intervals": intervals
        }

        return input_dict
    

class NaiveDavisDataset(Dataset):
    def __init__(self, data_path, interval=1, seq_len=4, partial=False):

        self.data_list = []
        self.seq_len = seq_len
        self.interval = interval

        if partial:
            test_list = [  
                            "boxing-fisheye",
                            "car-shadow",
                            "car-roundabout",
                            "car-turn",
                            "cat-girl",
                            "color-run",
                            "cows",
                            "drift-turn",
                            "india",
                            "lucia",
                            "scooter-gray",
                            "varanus-cage",
                            "walking"
                        ]

        for item in os.listdir(data_path):
            if not partial or item in test_list:
                item_path = os.path.join(data_path, item)
                if os.path.isdir(item_path):
                    self.data_list.append(item_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        data_path = self.data_list[idx]

        image_path_list = []

        for root, dirs, files in os.walk(data_path):
            for file in sorted(files):
                file = os.path.join(root, file)
                image_path_list.append(file)

        images = load_and_preprocess_images(image_path_list)    # [T, C, H, W]
        sequence_length = images.shape[0]
        images = images[::(sequence_length // self.seq_len)]  # downsample to fixed length
        sequence_length = images.shape[0]

        bg_mask = np.ones((1, images.shape[0], images.shape[2], images.shape[3]), dtype=bool)   # [B, T, H, W]
        bg_mask = torch.from_numpy(bg_mask)

        name = data_path.split('/')[-1]

        sequence_length = images.shape[0]
        indices = [i * self.interval for i in range(sequence_length)]
        intervals = [self.interval for _ in range(sequence_length - 1)]

        timestamps = np.array(indices)
        timestamps = timestamps / timestamps[-1] * (sequence_length / 4)

        '''
        start_idx = 0
        indices = [start_idx + i for i in range(T)]
        
        timestamps = np.array(indices) - start_idx
        timestamps = timestamps / timestamps[-1] * (T / 4)
        '''

        input_dict = {
            "name": name,
            "images": images,
            "masks": bg_mask,
            "timestamps": timestamps,
            "intervals": intervals
        }

        return input_dict

    
'''
def extract_frames_from_video(video_path, max_frames=100):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)
    
    frame_count = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    return frames


def preprocess(root_dir, save_root, max_frames=100):
    for video in os.listdir(root_dir):
        video_name = video.split('.')[0]
        video_path = os.path.join(root_dir, video)
        frames = extract_frames_from_video(video_path, max_frames)

        save_dir = os.path.join(save_root, video_name)
        os.makedirs(save_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame_path = os.path.join(save_dir, f'frame_{i:03d}.png')
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    root_dir = '/data/wangpeifeng/dataset/davis_videos'
    save_root = 'davis'
    preprocess(root_dir, save_root, max_frames=100)
'''