# https://github.com/edenton/svg/blob/master/data/kth.py
import numpy as np
import os
import pickle
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .h5 import HDF5Dataset


class KTHDataset(Dataset):

    def __init__(self, data_dir, frames_per_sample=5, train=True, random_time=True, random_horizontal_flip=True,
                 total_videos=-1, with_target=True, start_at=0):

        self.data_dir = data_dir                    # '/path/to/Datasets/KTH64_h5' (with shard_0001.hdf5 and persons.pkl in it)
        self.train = train
        self.frames_per_sample = frames_per_sample
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)
        self.with_target = with_target
        self.start_at = start_at

        # Read h5 files as dataset
        self.videos_ds = HDF5Dataset(self.data_dir)

        # Persons
        with open(os.path.join(data_dir, 'persons.pkl'), 'rb') as f:
            self.persons = pickle.load(f)

        # Train
        self.train_persons = list(range(1, 21))
        self.train_idx = sum([self.persons[p] for p in self.train_persons], [])
        # Test
        self.test_persons = list(range(21, 26))
        self.test_idx = sum([self.persons[p] for p in self.test_persons], [])

        print(f"Dataset length: {self.__len__()}")

        p='train' if self.train else 'test'
        path1 = os.path.join(self.data_dir,p+'1.npz')
        path2 = os.path.join(self.data_dir,p+'2.npz')
        if os.path.exists(path1) and os.path.exists(path2):
            self.videos = np.load(path1,allow_pickle=True)['arr_0']
            self.targets = np.load(path2,allow_pickle=True)['arr_0']
        else:
            self.videos=[]
            self.targets=[]
            for index in range(self.__len__()):
                video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
                shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
                idx = self.train_idx[int(idx_in_shard)] if self.train else self.test_idx[int(idx_in_shard)]

                with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
                    video_len = f['len'][str(idx)][()]
                    imgs = []
                    for i in range(video_len):
                        imgs.append(f[str(idx)][str(i)][()])
                    self.targets.append(int(f['target'][str(idx)][()]))
                    
                self.videos.append(imgs)             
                print(index,end='\r')

            print()
            print(len(self.videos))
            print(type(self.videos[0][0]))

            arr=np.array(self.videos)
            arr2=np.array(self.targets)
            np.savez(path1,arr)
            np.savez(path2,arr2)

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len

    def __len__(self):
        return self.total_videos if self.total_videos > 0 else len(self.train_idx) if self.train else len(self.test_idx)

    def max_index(self):
        return len(self.train_idx) if self.train else len(self.test_idx)

    def __getitem__(self, index, time_idx=0):

        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        idx = self.train_idx[int(idx_in_shard)] if self.train else self.test_idx[int(idx_in_shard)]

        prefinals = []
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx)][()] - self.start_at
            if self.random_time and video_len > self.frames_per_sample:
                time_idx = np.random.choice(video_len - self.frames_per_sample)
            time_idx += self.start_at
            for i in range(time_idx, min(time_idx + self.frames_per_sample, video_len)):
                img = f[str(idx)][str(i)][()]
                arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img))
                prefinals.append(arr)
            target = int(f['target'][str(idx)][()])

        # video_len = len(self.videos[index]) - self.start_at
        # if self.random_time and video_len > self.frames_per_sample:
        #     time_idx = np.random.choice(video_len - self.frames_per_sample)
        # time_idx += self.start_at
        # for i in range(time_idx, min(time_idx + self.frames_per_sample, video_len)):
        #     img = self.videos[index][i]
        #     arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img))
        #     prefinals.append(arr)
        # target = self.targets[index]

        if self.with_target:
            return torch.stack(prefinals), torch.tensor(target)
        else:
            return torch.stack(prefinals)
