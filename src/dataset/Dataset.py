import os
from abc import ABC
from typing import List

import torch
from torchio import Queue, Subject, ScalarImage, LabelMap, SubjectsDataset, SubjectsLoader, UniformSampler

from config import CONF

class TrainDataset(ABC):
    def __init__(self,
                 train_exps: List[str] = ["TS_5_4", "TS_6_6", "TS_69_2", "TS_86_3", "TS_99_9"],
                 val_exps: List[str] = ["TS_6_4", "TS_73_6"],
                 particle_types: List[str] = ['apo-ferritin', 'beta-amylase', 'beta-galactosidase',
                                              'ribosome', 'thyroglobulin', 'virus-like-particle'],
                 batch_size = 32,
                 res: List[str] = ['0', '1', '2'],
                 patch_size: List[int] = [5, 630, 630],
                 patch_cache_size: int = 128,
                 n_patch_per_subject: int = 180):

        self.input_folder = os.path.join(CONF.DATA_DIR, "preproc", "pytorch", "train")
        self.train_exps = train_exps
        self.val_exps = val_exps
        self.particle_types = particle_types
        self.batch_size = batch_size
        self.res = res
        self.patch_size = patch_size
        self.patch_cache_size = patch_cache_size
        self.n_patch_per_subject = n_patch_per_subject

        self.img_pattern = os.path.join(self.input_folder, "{exp_name}_res-{res}_img.pt")
        self.msk_pattern = os.path.join(self.input_folder, "{exp_name}_res-{res}_msk-{particle_type}.pt")

    def _get_experiment(self, exp_name: str) -> Subject:
        subject_pars = {}
        for res in self.res:
            subject_pars[f"tomogram_{res}"] = ScalarImage(tensor=torch.load(self.img_pattern.format(exp_name=exp_name,
                                                                                             res=res)).unsqueeze(0))
            for particle_type in self.particle_types:
                subject_pars[f"mask_{res}_{particle_type}"] = LabelMap(tensor=torch.load(self.img_pattern.format(exp_name=exp_name,
                                                                                                          res=res,
                                                                                                          particle_type=particle_type)).unsqueeze(0))
                                    
        return Subject(**subject_pars)

    def _get_dataset(self, split: str) -> SubjectsDataset:
        exp_list = []
        exp_names = self.train_exps if split == "train" else self.val_exps
        for exp_name in exp_names:
            exp_list.append(self._get_experiment(exp_name))
        return SubjectsDataset(exp_list)


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
                      shuffle_subjects=True)
        loader = SubjectsLoader(queue,
                                batch_size=self.batch_size,
                                num_workers=0)

        return loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_ds = TrainDataset(batch_size=2, res=['0'], patch_cache_size=10, n_patch_per_subject=10)
    loader = train_ds.get_loader("val")
    for i, load in enumerate(loader):
        fig, ax = plt.subplots(2, 5)
        for batch in range(2):
            for z_slice in range(5):
                ax[batch, z_slice].imshow(load["tomogram_0"]["data"][batch, 0, z_slice], cmap="gray")
                ax[batch, z_slice].imshow(load["mask_0_ribosome"]["data"][batch, 0, z_slice], alpha=0.2, cmap="jet")
        plt.savefig(f"toto_{i}.jpg")
