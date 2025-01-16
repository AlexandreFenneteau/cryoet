import os
import json
from abc import ABC
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import pandas as pd

from copick_management import CopickManagement
from torch_sphere import TorchSphere

class PreprocData(ABC):

    def __init__(self, copick_management: CopickManagement, out_folder: str, overwrite: bool = False, write_msks: bool = False):
        super().__init__()

        self.particle_int_label = {
            'apo-ferritin': 1,
            'beta-galactosidase': 2,
            'ribosome': 3,
            'thyroglobulin': 4,
            'virus-like-particle': 5
        }

        self.mask_particle_channels = ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']
        #                                      0                  1                 2             3                    4

        self.copick = copick_management
        self.out_folder = out_folder
        self.preproc_folder = os.path.join(
            self.out_folder, self.copick.split)
        self.overwrite = overwrite
        self.write_msks = write_msks

        os.makedirs(self.preproc_folder, exist_ok=True)

    def preproc_zarr_img(self, experiment_name):
        experiment_resolutions = {}
        label_path = os.path.join(
            self.preproc_folder, f"{experiment_name}_label_coords.csv")
        if self.copick.split == "train":
            # assemble labels and put them into preproc folder
            if not(os.path.exists(label_path)) or self.overwrite:
                label_dicts = self.copick.get_run_labels(experiment_name)
                df = pd.DataFrame(label_dicts)
                df.to_csv(label_path, index=False, header=True)
                print(f"{label_path} ... written!")
            else:
                df = pd.read_csv(label_path)
        else:
            df = None
        # convert images in pytorch format and put it into preproc folder
        for zarray, res_dict in self.copick.get_zarray(experiment_name):
            res_name = list(res_dict.keys())[0]
            experiment_resolutions.update(res_dict)
            
            tensor_save_path = os.path.join(self.preproc_folder, f"{experiment_name}_res-{res_name}_img.pt")
            msk_save_path = os.path.join(self.preproc_folder, f"{experiment_name}_res-{res_name}_msk.pt") if self.copick.split == 'train' else None
            
            if self.copick.split == "train" and self.write_msks:
                if not os.path.exists(msk_save_path) or self.overwrite:
                    msk_tensor = self._generate_label_mask(zarray.shape, res_dict[res_name], label_dicts)
                    torch.save(msk_tensor, msk_save_path)
                    print(f"{msk_save_path} ... written!")
            if not os.path.exists(tensor_save_path) or self.overwrite:
                zarr_tensor = torch.tensor(np.array(zarray, dtype=np.float32))
                torch.save(zarr_tensor, tensor_save_path)
                print(f"{tensor_save_path} ... written!")
        exp_res_path = os.path.join(self.preproc_folder, f"{experiment_name}_res-detail.json")
        if not os.path.exists(exp_res_path) or self.overwrite:
            with open(exp_res_path, "w") as f:
                json.dump(experiment_resolutions, f, indent="\t")
            print(f"{exp_res_path} ... written!")
        return tensor_save_path, msk_save_path, df, experiment_resolutions

    def preproc_zarr_imgs(self):
        for experiment_name in self.copick.get_experiment_names():
            self.preproc_zarr_img(experiment_name)

    def _generate_label_mask(self,
                             tensor_shape: Tuple[int, int, int],
                             res_dict: Dict[str, float],
                             label_dicts: List[Dict[str, Union[str, int, float]]]) -> torch.Tensor:
        msk_tensor_dict = {}
        # msk_tensor = torch.zeros(tensor_shape, dtype=int)
        x_size_a, y_size_a, z_size_a = res_dict["x"], res_dict["y"], res_dict["z"]
        for particle_dict in label_dicts:
            particle_type = particle_dict["particle_type"]
            if particle_type in self.mask_particle_channels:
                # print(f"76: {particle_type = }")
                radius_a = particle_dict["radius"]
                x_pos_a, y_pos_a, z_pos_a = particle_dict["x"], particle_dict["y"], particle_dict["z"]
                # particle_coords are in coordinates [x, y, z]
                # with x: width, y: height and z: depth
                # however images are in matrix indexing  indexed by:
                # z: slice, i: row, j: columns
                # slice and depth are representing the same thing in this context
                # however, the row (i) represents the height (y)
                # and, the column (j) represents the width (x).
                # So, image volume is indexed by:
                # z: (slice/depth), i/y (row/height), j/x (column/width)
                # 0 ------> j (x)
                # |
                # |
                # i (y)
                # then mask is indexed by [z, y, x]
    
                particle_center_vox_coord = (
                    round(z_pos_a / z_size_a),
                    round(y_pos_a / y_size_a),
                    round(x_pos_a / x_size_a),
                )
                # here we approximate the radius conversion as the average of voxel size in 3D
                # this approximation only holds because the voxel_size in each direction is really close
                radius_vox = 3 * radius_a / (x_size_a + y_size_a + z_size_a)
    
                if particle_type not in msk_tensor_dict:
                    msk_tensor_dict[particle_type] = torch.zeros(
                        tensor_shape, dtype=int)
    
                TorchSphere.add_sphere(msk_tensor_dict[particle_type],
                                       radius_vox,
                                       particle_center_vox_coord)
        tensor_concat = []
        for particle_name in self.mask_particle_channels:
            # print(f"110: {particle_name = }")
            msk_tensor = msk_tensor_dict[particle_name]
            msk_tensor[msk_tensor > 0] = 1
            tensor_concat.append(msk_tensor.unsqueeze(0).bool())

        return torch.cat(tensor_concat)