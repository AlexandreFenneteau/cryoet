import os
import glob
import json
from abc import ABC
from typing import Dict, List, Union

import torch
import pandas as pd

from config import CONF

from copick_mgmt import get_copick_root, get_experiment_names, get_run_labels, get_zarr_tensors

class PreprocData(ABC):

	def __init__(self, split: str = "train", overwrite: bool = False):
		assert split in ["train", "test"], "Should only be test of train split"

		self.split: str = split
		self.copick_root = get_copick_root(split)
		self.preproc_folder = os.path.join(CONF.DATA_DIR, "preproc", self.split)
		self.overwrite = overwrite

		os.makedirs(self.preproc_folder, exist_ok=True)
		self._preproc_zarr_imgs()


	def _preproc_zarr_imgs(self):
		experiment_resolutions = {}
		for experiment_name in get_experiment_names(self.copick_root):
			label_path = os.path.join(self.preproc_folder, f"{experiment_name}_label_coords.csv")
			if (len(glob.glob(os.path.join(self.preproc_folder, f"{experiment_name}_res-*_img.pt"))) < 3) or self.overwrite:
				if self.split == "train":
					#assemble labels and put them into preproc folder
					label_dicts = get_run_labels(self.copick_root, experiment_name)
					df = pd.DataFrame(label_dicts)
					df.to_csv(label_path, index=False, header=True)
				#convert images in pytorch format and put it into preproc folder
				for zarr_tensor, res_dict in get_zarr_tensors(self.copick_root, experiment_name):
					experiment_resolutions.update(res_dict)
					res_name = list(res_dict.keys())[0]
					tensor_save_path = os.path.join(self.preproc_folder, f"{experiment_name}_res-{res_name}_img.pt")
					torch.save(zarr_tensor, tensor_save_path)
					print(f"{tensor_save_path} ... written!")
				exp_res_path = os.path.join(self.preproc_folder, f"{experiment_name}_res-detail.json")
				with open(exp_res_path, "w") as f:
					json.dump(experiment_resolutions, f, indent="\t")
				print(f"{exp_res_path} ... written!")


	def _generate_label_mask(img_tensor: torch.Tensor, res_dict: Dict[str, float], label_dicts: List[Dict[str, Union[str, int, float]]]):
		msk_tensor = torch.zeros(img_tensor.shape, dtype=int)
		x_size_a, y_size_a, z_size_a = res_dict["x"], res_dict["y"], res_dict["z"]
		for particle_dict in label_dicts:
			label_int = label_dicts["label"]
			radius_a = label_dicts["radius"]
			x_pos_a, y_pos_a, z_pos_a = label_dicts["x"], label_dicts["y"], label_dicts["z"]
			particle_center_vox_coord = (
				round(x_pos_a / x_size_a),
				round(y_pos_a / y_size_a),
				round(z_pos_a / z_size_a),
			)
			# here we approximate the radius conversion as the average of voxel size in 3D
			# this approximation only holds because the voxel_size in each direction is really close
			radius_vox = round(3 * radius_a / (x_size_a + y_size_a + z_size_a))


			

			


if __name__ == "__main__":
	train_preproc = PreprocData()
	#test_preproc = PreprocData("test")