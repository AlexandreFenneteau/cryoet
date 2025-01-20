import os
from abc import ABC
from typing import List

import torch
import torchio as tio
from torchio import Queue, Subject, ScalarImage, LabelMap, SubjectsDataset, SubjectsLoader, UniformSampler
import lightning as L

class TrainDataset(ABC):
    def __init__(self,
                 pytorch_train_folder: str,
                 train_exps: List[str] = ["TS_5_4", "TS_6_6", "TS_69_2", "TS_86_3", "TS_99_9"],
                 val_exps: List[str] = ["TS_6_4", "TS_73_6"],
                 particle_types: List[str] = ['apo-ferritin', 'beta-galactosidase',
                                              'ribosome', 'thyroglobulin', 'virus-like-particle'],
                 batch_size = 32,
                 res: str = '0',
                 patch_size: List[int] = [5, 256, 256],
                 patch_cache_size: int = 128,
                 n_patch_per_subject: int = 180):

        #self.input_folder = os.path.join(CONF.DATA_DIR, "preproc", "pytorch", "train")
        self.input_folder = pytorch_train_folder
        self.train_exps = train_exps
        self.val_exps = val_exps
        self.particle_types = particle_types
        self.batch_size = batch_size
        self.res = res
        self.patch_size = patch_size
        self.patch_cache_size = patch_cache_size
        self.n_patch_per_subject = n_patch_per_subject

        self.img_pattern = os.path.join(self.input_folder, "{exp_name}_res-{res}_img.pt")
        self.msk_pattern = os.path.join(self.input_folder, "{exp_name}_res-{res}_msk.pt")

    def _get_experiment(self, exp_name: str) -> Subject:
        subject_pars = {}
        subject_pars["tomogram"] = ScalarImage(tensor=torch.load(self.img_pattern.format(exp_name=exp_name,
                                                                                         res=self.res)).unsqueeze(0))
        subject_pars[f"mask"] = LabelMap(tensor=torch.load(self.msk_pattern.format(exp_name=exp_name,
                                                                                   res=self.res)))
        return Subject(**subject_pars)

    def _get_preproc_list(self):
        intensity_scaler = tio.transforms.RescaleIntensity(
            out_min_max=(-1, 1), percentiles=(0.5, 99.5), p=1.)
        return [intensity_scaler]

    def _get_augmentation_list(self):
        return [
            tio.transforms.RandomFlip(axes = (0, 1, 2), flip_probability=0.2, p=0.5),
            tio.transforms.RandomAffine(scales=0.1, degrees=10, translation=0., default_pad_value = -1., p=0.5),
            tio.transforms.RandomBlur(1.5, p=0.5),
            tio.transforms.RandomNoise(mean=0., std=0.2, p=0.5)
        ]

    def _get_dataset(self, split: str) -> SubjectsDataset:
        exp_list = []
        exp_names = self.train_exps if split == "train" else self.val_exps
        for exp_name in exp_names:
            exp_list.append(self._get_experiment(exp_name))
        transforms = tio.Compose(self._get_preproc_list() + self._get_augmentation_list())
        return SubjectsDataset(exp_list, transform=transforms)


    def get_loader(self,
                   split: str) -> SubjectsLoader:
        #voir https://torchio.readthedocs.io/patches/patch_training.html#torchio.data.Queue
        assert split in ["train", "val"], "Only train or val"
        
        dataset = self._get_dataset(split)
        sampler = UniformSampler(self.patch_size)
        queue = Queue(subjects_dataset=dataset,
                      max_length=self.patch_cache_size,
                      samples_per_volume=self.n_patch_per_subject,
                      sampler=sampler,
                      shuffle_patches=True,
                      shuffle_subjects=True,
                      #shuffle_patches=False,
                      #shuffle_subjects=False,
                      num_workers=0)
        loader = SubjectsLoader(queue,
                                batch_size=self.batch_size,
                                num_workers=0)

        return loader


class CZIIDataModule(L.LightningDataModule):
    def __init__(self,
                 pytorch_train_folder: str,
                 train_exps: List[str] = ["TS_5_4", "TS_6_6", "TS_69_2", "TS_86_3", "TS_99_9"],
                 val_exps: List[str] = ["TS_6_4", "TS_73_6"],
                 particle_types: List[str] = ['apo-ferritin', 'beta-galactosidase',
                                              'ribosome', 'thyroglobulin', 'virus-like-particle'],
                 batch_size=32, res='0', patch_size=(32, 32, 32), patch_cache_size=32, n_patch_per_subject=16):
        super().__init__()
        self.pytorch_train_folder = pytorch_train_folder
        self.train_exps = train_exps
        self.val_exps = val_exps
        self.particle_types = particle_types
        self.batch_size = batch_size
        self.res=res
        self.patch_size=patch_size
        self.patch_cache_size=patch_cache_size
        self.n_patch_per_subject=n_patch_per_subject

    def prepare_data(self):
        # download only
        self.train_dataset = TrainDataset(self.pytorch_train_folder,
                                          self.train_exps, self.val_exps,
                                          self.particle_types, self.batch_size,
                                          self.res, self.patch_size, self.patch_cache_size,
                                          self.n_patch_per_subject)

    def setup(self, stage):
        # transform
        self.__train_loader = self.train_dataset.get_loader("train")
        self.__val_loader = self.train_dataset.get_loader("val")

    def train_dataloader(self):
        return self.__train_loader

    def val_dataloader(self):
        return self.__val_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    train_ds = TrainDataset(r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\pytorch\train",
                            batch_size=32, res='0', patch_size=(32, 32, 32),
                            patch_cache_size=32, n_patch_per_subject=16)
    loader = train_ds.get_loader("val")
    i = 0
    for i, batch in enumerate(loader):
        print(i)
        print(batch["tomogram"]['data'].shape)
        print(batch["mask"]['data'].shape)
        #fig, ax = plt.subplots(1)
        #img = load["tomogram"]["data"][0, 0, 0].numpy()
        #ax.imshow(img, cmap="gray", vmin=-1., vmax=1.)
        #msk = torch.zeros(load["tomogram"]["data"].shape[-2:], dtype=torch.int)
        #msk_tensor = load["mask"]["data"][0, :, 0]
        #for label in range(5):
        #    msk[msk_tensor[label] == 1] = label

        #msk = msk.numpy()
        ##ax.imshow(msk.numpy(), cmap="jet")
        #ax.imshow(msk, alpha=(msk != 0).astype(float) * 0.5, cmap="jet")
        #fig.set_size_inches((10,10))
        #fig_path = os.path.join(r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\src\test_img", f"test_img_slice_{i}.jpg")
        #plt.savefig(fig_path)
        #print(f"{fig_path}... Written!")
        #plt.close()
        #i+=1
