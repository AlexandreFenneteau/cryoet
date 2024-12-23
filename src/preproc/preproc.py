import csv
import os
import glob
import json
from abc import ABC
from typing import Dict, List, Tuple, Union

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import pandas as pd

import torchio as tio
import yaml

from config import CONF

from copick_mgmt import get_copick_root, get_experiment_names, get_run_labels, get_zarr_tensors
from geometry import TorchSphere

class PreprocData(ABC):

	def __init__(self, split: str = "train", overwrite: bool = False):
		assert split in ["train", "test"], "Should only be test of train split"
		super().__init__()

		self.particle_int_label = {
			'apo-ferritin': 1,
			'beta-amylase': 2,
			'beta-galactosidase': 3,
			'ribosome': 4,
			'thyroglobulin': 5,
			'virus-like-particle': 6
		}

		self.split: str = split
		self.copick_root = get_copick_root(split)
		self.preproc_folder = os.path.join(CONF.DATA_DIR, "preproc", "pytorch", self.split)
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

					if self.split == "train":
						msk_tensor_dict = self._generate_label_mask(zarr_tensor.shape, res_dict[res_name], label_dicts)
						all_msk_tensor = torch.zeros(zarr_tensor.shape, dtype=int)
						for particle_type in msk_tensor_dict:
							msk_tensor = msk_tensor_dict[particle_type]
							all_msk_tensor[msk_tensor > 0] = msk_tensor[msk_tensor > 0] * self.particle_int_label[particle_type]
							msk_save_path = os.path.join(self.preproc_folder, f"{experiment_name}_res-{res_name}_msk-{particle_type}.pt")
							torch.save(msk_tensor, msk_save_path)
							print(f"{msk_save_path} ... written!")
						all_msk_save_path = os.path.join(self.preproc_folder, f"{experiment_name}_res-{res_name}_msk-all.pt")
						torch.save(all_msk_tensor, all_msk_save_path)
						print(f"{all_msk_save_path} ... written!")
						
					tensor_save_path = os.path.join(self.preproc_folder, f"{experiment_name}_res-{res_name}_img.pt")
					torch.save(zarr_tensor, tensor_save_path)
					print(f"{tensor_save_path} ... written!")
				exp_res_path = os.path.join(self.preproc_folder, f"{experiment_name}_res-detail.json")
				with open(exp_res_path, "w") as f:
					json.dump(experiment_resolutions, f, indent="\t")
				print(f"{exp_res_path} ... written!")


	@staticmethod
	def _generate_label_mask(tensor_shape: Tuple[int, int, int],
							 res_dict: Dict[str, float],
							 label_dicts: List[Dict[str, Union[str, int, float]]]) -> Dict[str, torch.Tensor]:
		msk_tensor_dict = {}
		#msk_tensor = torch.zeros(tensor_shape, dtype=int)
		x_size_a, y_size_a, z_size_a = res_dict["x"], res_dict["y"], res_dict["z"]
		for particle_dict in label_dicts:
			particle_type = particle_dict["particle_type"]
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
			# then mask is indexed by [z, y, x] instead of [z, x, y]

			particle_center_vox_coord = (
				round(z_pos_a / z_size_a),
				round(y_pos_a / y_size_a),
				round(x_pos_a / x_size_a),
			)
			# here we approximate the radius conversion as the average of voxel size in 3D
			# this approximation only holds because the voxel_size in each direction is really close
			radius_vox = 3 * radius_a / (x_size_a + y_size_a + z_size_a)

			if particle_type not in msk_tensor_dict:
				msk_tensor_dict[particle_type] = torch.zeros(tensor_shape, dtype=int)

			TorchSphere.add_sphere(msk_tensor_dict[particle_type],
						  		   radius_vox,
								   particle_center_vox_coord)

		for msk_tensor in msk_tensor_dict.values():
			msk_tensor[msk_tensor > 0] = 1

		return msk_tensor_dict


class ExportYOLOFormat(ABC):
	"""Convert the pytorch 3D data as coco images.
		see https://docs.ultralytics.com/datasets/detect/ """

	def __init__(self, slicing: str,
				 split_dict: Dict[str, List[str]] = {"train": ["TS_5_4", "TS_6_4", "TS_69_2", "TS_86_3", "TS_99_9"],
										  			 "val": ["TS_6_6", "TS_73_6"],
													 "test": ["TS_5_4", "TS_6_4", "TS_69_2"]}):
		assert slicing in ['x', 'y', 'z'], f"{slicing = } should be only 'x', 'y' or 'z'"
		super().__init__()
		self.preproc_pytorch_folder = os.path.join(CONF.DATA_DIR, "preproc", "pytorch")
		self.preproc_yolo_folder = os.path.join(CONF.DATA_DIR, "preproc", "yolo")
		self.slicing = slicing
		self.split_dict = split_dict
		self.img_axes_maps = {'z': 0, 'x': 1, 'y': 2}

		self.slicing_name = "sagittal" if self.slicing == 'x' else "coronal" if self.slicing == 'y' else "axial"
		self.slicing_idx = self.img_axes_maps[self.slicing]

		self.width_axe = 'x' if self.slicing == 'z' else 'z'
		self.height_axe = 'x' if self.slicing == 'y' else 'y'

		self.width_ax_idx = self.img_axes_maps[self.width_axe]
		self.height_ax_idx = self.img_axes_maps[self.height_axe]

		self.intensity_scaler = tio.transforms.RescaleIntensity(out_min_max=(0, 2**16), percentiles=(0.5, 99.5))
		self.out_img_dim = (640, 640) #width, height of images

		#for split in [train, test]
		for torch_split in ["train", "test"]:
			if torch_split == "train":
				yolo_splits = ["train", "val"]
			else:
				yolo_splits = ["test"]
			
			for yolo_split in yolo_splits:
				yolo_img_path = os.path.join(self.preproc_yolo_folder, self.slicing_name, "images", yolo_split)
				os.makedirs(yolo_img_path, exist_ok=True)
				#1: for each exp (only 0 res) in [train, val, test split]:
				for exp_name in self.split_dict[yolo_split]:
					pt_path = os.path.join(self.preproc_pytorch_folder, torch_split, f"{exp_name}_res-0_img.pt")
					#load_pytorch 3d array res 0
					tensor_data = torch.load(pt_path)
					# rescale intensity in (0, 2**16) to be used as 16bit depth img
					#intensity rescale to 0 -> 2^{16}  #torchio rescaleintensity? #scipy?
					tensor_data = self.intensity_scaler(tio.ScalarImage(tensor=tensor_data.unsqueeze(0))).data[0]

					img_shape = tensor_data.shape

					n_slices = tensor_data.shape[self.slicing_idx]
					#for i in each slice
					for i in range(n_slices):
						# we only keep each of three slice in a channel max(0, i-1), i, min(i+1, 639)
						# for borders, we duplicate the middle in the missing direction
						img_path = os.path.join(yolo_img_path, f"{exp_name}_{i:03}.png")
						if not(os.path.exists(img_path)):
							i_slice_min = max(0, i-1)
							i_slice_max = min(i+1, n_slices-1)
							if self.slicing_idx == 0:
								three_channel_slice = torch.cat([tensor_data[i_slice_min, :, :].unsqueeze(-1),
																tensor_data[i, :, :].unsqueeze(-1),
																tensor_data[i_slice_max, :, :].unsqueeze(-1)], dim=-1)
							elif self.slicing_idx == 1:
								three_channel_slice = torch.cat([tensor_data[:, i_slice_min, :].unsqueeze(-1),
																tensor_data[:, i, :].unsqueeze(-1),
																tensor_data[:, i_slice_max, :].unsqueeze(-1)], dim=-1)
							else:
								three_channel_slice = torch.cat([tensor_data[:, :, i_slice_min, :].unsqueeze(-1),
																tensor_data[:, :, i, :].unsqueeze(-1),
																tensor_data[:, :, i_slice_max, :].unsqueeze(-1)], dim=-1)
							three_channel_slice = three_channel_slice.numpy()
							#plt.imshow(three_channel_slice/(2**16), cmap='gray')
							#plt.show()
							#cv2.imshow("Image", three_channel_slice.astype(np.uint16))
							#cv2.waitKey(0)
							#cv2.destroyAllWindows()
							#resizing of image to 640, 640
							three_channel_slice = cv2.resize(three_channel_slice, self.out_img_dim, interpolation=cv2.INTER_CUBIC)
							#cv2.imshow("Image resize", three_channel_slice.astype(np.uint16))
							#cv2.waitKey(0)
							#cv2.destroyAllWindows()
							#save image at self.preproc_yolo_folder/slicing/images/split/exp_name_{i:trois decimales} in png 16 bit depth
							cv2.imwrite(img_path, three_channel_slice.astype(np.uint16))
			
					if yolo_split != "test":

						label_csv_path = os.path.join(self.preproc_pytorch_folder, torch_split, f"{exp_name}_label_coords.csv")
						res_detail_path = os.path.join(self.preproc_pytorch_folder, torch_split, f"{exp_name}_res-detail.json")

						yolo_label_path = os.path.join(self.preproc_yolo_folder, self.slicing_name, "labels", yolo_split)
						os.makedirs(yolo_label_path, exist_ok=True)
						#load res_detail
						with open(res_detail_path, 'r') as f:
							res = json.load(f)['0']

						#load label csv
						with open(label_csv_path, 'r') as csv_file:
							reader = csv.DictReader(csv_file)
							# for each particle
							for particle_dict in reader:
								# keys: experiment, particle_type, label, radius, x, y, z
								# world_coordinates (wc) = csv x, y, z
								#w_x, w_y, w_z = particle_dict['x'], particle_dict['y'], particle_dict['z']
								w_c = {'x': float(particle_dict['x']),
			   						   'y': float(particle_dict['y']),
									   'z': float(particle_dict['z'])}
								# voxel_coordinates (vc) = wc / res_detail
								#v_x, v_y, v_z = w_x / res_x, w_y / res_y, w_z / res_z
								v_c = {key: val / res[key] for key, val in w_c.items()}
								# get the coordinates relative to image
								#r_x = v_x / img_shape[self.img_axes_maps['x']]
								#r_y = v_y / img_shape[self.img_axes_maps['y']]
								#y_z = v_z / img_shape[self.img_axes_maps['z']]
								r_c = {key: val / img_shape[self.img_axes_maps[key]] for key, val in v_c.items()}

								particle_radius_vox = (float(particle_dict["radius"]) * 3.) / (res['x'] + res['y'] + res['z'])
								# w, h = label['radius'] / widht and height of image
								w = particle_radius_vox / img_shape[self.width_ax_idx]
								h = particle_radius_vox / img_shape[self.height_ax_idx]
								
								# class = label['label'] - 1
								particle_cls = int(particle_dict["label"]) - 1 # in dict labels are from 1 to 6, yolo from 0 to 5
								# for i in range(max(0, wc - radius), min(wc + radius, 639)):
								center_slice_idx = int(v_c[self.slicing])
								first_slice = max(0, center_slice_idx - int(particle_radius_vox))
								last_slice = min(center_slice_idx + int(particle_radius_vox), img_shape[self.slicing_idx])
								for i in range(first_slice, last_slice):
									# fichier = self.preproc_yolo_folder/slicing/labels/split/exp_name_{i:trois decimales} in txt
									label_txt_path = os.path.join(yolo_label_path, f"{exp_name}_{i:03}.txt")
									#	write_line(class, yolo_center_x, yolo_center_y, w, h)
									with open(label_txt_path, 'a') as append_f:
										append_f.write(f"{particle_cls} {r_c[self.width_axe]} {r_c[self.height_axe]} {w} {h}\n")
				break
			break
		

		#2 write data_{slicing}.yaml at self.preproc_yolo_folder with
		yolo_yml_path = os.path.join(self.preproc_yolo_folder, f"{self.slicing_name}_data.yaml")
		yolo_yml_dict = {
			"path": ".",
			"train": os.path.join(self.slicing_name, "images", "train"),
			"val": os.path.join(self.slicing_name, "images", "val"),
			"test": os.path.join(self.slicing_name, "images", "val"),
			"names": {
				0: "apo-ferritin",
				1: "beta-amylase",
				2: "beta-galactosidase",
				3: "ribosome",
				4: "thyroglobulin",
				5: "virus-like-particle"
			}
		}
		with open(yolo_yml_path, "w") as f:
			yaml.dump(yolo_yml_dict, f)
		# path = .
		# train: {slicing}/images/train
		# val: {slicing}/images/val
		# test: {slicing}/images/test

		# names:
		# 	0: apo-ferritin
		# 	1: beta-amylase
		# 	2: beta-galactosidase
		# 	3: ribosome
		# 	4: thyroglobulin
		# 	5: virus-like-particle






if __name__ == "__main__":
	train_preproc = PreprocData("train", overwrite=True)
	#test_preproc = PreprocData("test")
	toto = ExportYOLOFormat('z')