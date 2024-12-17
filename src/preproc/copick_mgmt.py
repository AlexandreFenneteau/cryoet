import os
import json
from typing import List, Tuple, Dict, Generator, Union
import numpy as np

import zarr
import copick
from copick.impl.filesystem import CopickRootFSSpec
import zarr.hierarchy

import torch

from config import CONF


def get_copick_root(split: str = "train") -> CopickRootFSSpec :
	assert split in ["train", "test"], "Split can only be 'train' or 'test'."
	copick_config_path = os.path.join(CONF.BRUT_DATA_PATH, f"{split}_copic_config.json")
	if not os.path.exists(copick_config_path):
		config_dict = {
			"name": "czii_cryoet_mlchallenge_2024",
			"description": f"2024 CZII CryoET ML Challenge {split}ing data.",
			"version": "1.0.0",

			"pickable_objects": [
				{
					"name": "apo-ferritin",
					"is_particle": True,
					"pdb_id": "4V1W",
					"label": 1,
					"color": [  0, 117, 220, 128],
					"radius": 60,
					"map_threshold": 0.0418
				},
				{
					"name": "beta-amylase",
					"is_particle": True,
					"pdb_id": "1FA2",
					"label": 2,
					"color": [153,  63,   0, 128],
					"radius": 65,
					"map_threshold": 0.035
				},
				{
					"name": "beta-galactosidase",
					"is_particle": True,
					"pdb_id": "6X1Q",
					"label": 3,
					"color": [ 76,   0,  92, 128],
					"radius": 90,
					"map_threshold": 0.0578
				},
				{
					"name": "ribosome",
					"is_particle": True,
					"pdb_id": "6EK0",
					"label": 4,
					"color": [  0,  92,  49, 128],
					"radius": 150,
					"map_threshold": 0.0374
				},
				{
					"name": "thyroglobulin",
					"is_particle": True,
					"pdb_id": "6SCJ",
					"label": 5,
					"color": [ 43, 206,  72, 128],
					"radius": 130,
					"map_threshold": 0.0278
				},
				{
					"name": "virus-like-particle",
					"is_particle": True,
					"pdb_id": "6N4V",			
					"label": 6,
					"color": [255, 204, 153, 128],
					"radius": 135,
					"map_threshold": 0.201
				}
			],


			"static_root": os.path.join(CONF.BRUT_DATA_PATH, split, "static"),
		}

		if split == "train":
			config_dict["overlay_root"] = os.path.join(CONF.BRUT_DATA_PATH, split, "overlay")
			config_dict["overlay_fs_args"] = {"auto_mkdir": True}
		else:
			config_dict["overlay_root"] = os.path.join(CONF.BRUT_DATA_PATH, split, "overlay")
			config_dict["overlay_fs_args"] = {"auto_mkdir": False}

		with open(copick_config_path, "w") as f:
			json.dump(config_dict, f, indent='\t')

	return copick.from_file(copick_config_path)


def get_experiment_names(root: CopickRootFSSpec) -> List[str]:
	return [run.name for run in root.runs]


def get_zarr_img(root: CopickRootFSSpec, run_name: str) -> Tuple[zarr.hierarchy.Group, Dict[str, Dict[str, float]]]:
	tomogram = root.get_run(run_name).get_voxel_spacing(CONF.COPICK_VOX_SIZE).get_tomograms(CONF.TRAIN_TOMO_TYPE)[0]
	zarr_img = zarr.open(tomogram.zarr())
	zarr_attrs = zarr_img.attrs.asdict()

	zarr_ds_attrs = zarr_attrs['multiscales'][0]['datasets']
	resolutions_dict = {item['path']: {'z': item['coordinateTransformations'][0]['scale'][0],
									   'y': item['coordinateTransformations'][0]['scale'][1],
									   'x': item['coordinateTransformations'][0]['scale'][2]} for item in zarr_ds_attrs}

	return zarr_img, resolutions_dict


def get_zarr_tensors(root: CopickRootFSSpec, exp_name: str) -> Generator[Tuple[torch.Tensor, Dict[str, Dict[str, float]]], None, None]:
	zarr_img, res_dict = get_zarr_img(root, exp_name)
	for res, zarray in zarr_img.arrays():
		yield torch.tensor(np.array(zarray, dtype=np.float32)), {res: res_dict[res]}


def get_pickable_objects_properties(root: CopickRootFSSpec) -> Dict[str, Dict[str, Union[str, Tuple[int], int, float]]]:
	"""Return pickable object properties.
	
	Should look like:
	{'apo-ferritin': {'color': (0, 117, 220, 128),
					'label': 1,
					'map_threshold': 0.0418,
					'pdb_id': '4V1W',
					'radius': 60.0},
	'beta-amylase': {'color': (153, 63, 0, 128),
					'label': 2,
					'map_threshold': 0.035,
					'pdb_id': '1FA2',
					'radius': 65.0},
	'beta-galactosidase': {'color': (76, 0, 92, 128),
							'label': 3,
							'map_threshold': 0.0578,
							'pdb_id': '6X1Q',
							'radius': 90.0},
	'ribosome': {'color': (0, 92, 49, 128),
				'label': 4,
				'map_threshold': 0.0374,
				'pdb_id': '6EK0',
				'radius': 150.0},
	'thyroglobulin': {'color': (43, 206, 72, 128),
					'label': 5,
					'map_threshold': 0.0278,
					'pdb_id': '6SCJ',
					'radius': 130.0},
	'virus-like-particle': {'color': (255, 204, 153, 128),
							'label': 6,
							'map_threshold': 0.201,
							'pdb_id': '6N4V',
							'radius': 135.0}}
	"""
	properties_dict = {}
	for raw_property in root.pickable_objects:
		properties_dict[raw_property.name] = {'label': raw_property.label,
									  		  'color': raw_property.color,
									  		  'radius': raw_property.radius,
											  'pdb_id': raw_property.pdb_id,
											  'map_threshold': raw_property.map_threshold}

	return properties_dict


def assert_eye_matrix(list_matrix: List[List[float]]):
	"""assert that the matrix is [[1., 0., 0., 0.],
								  [0., 1., 0., 0.]
								  [0., 0., 1., 0.]
								  [0., 0., 0., 1.]
								 ]
	"""
	assert list_matrix[0][0] == 1.
	assert sum(list_matrix[0]) == 1.

	assert list_matrix[1][1] == 1.
	assert sum(list_matrix[1]) == 1.

	assert list_matrix[2][2] == 1.
	assert sum(list_matrix[2]) == 1.

	assert list_matrix[3][3] == 1.
	assert sum(list_matrix[3]) == 1.


def get_run_labels(root: CopickRootFSSpec, experiment_name: str) -> List[Dict[str, Union[str, int, float]]]:
	run = root.get_run(experiment_name)
	pickable_properties = get_pickable_objects_properties(root)
	ov_path = run.overlay_path

	list_data = []
	for particle_name in pickable_properties:
		json_path = os.path.join(ov_path, "Picks", f"{particle_name}.json")
		with open(json_path, 'r') as f:
			json_data = json.load(f)
		for point in json_data["points"]:
			assert_eye_matrix(point["transformation_"])
			line_data = {
				"experiment": experiment_name,
				"particle_type": particle_name, 
				"label": pickable_properties[particle_name]["label"],
				"radius": pickable_properties[particle_name]["radius"], 
				"x": point["location"]['x'], 
				"y": point["location"]['y'], 
				"z": point["location"]['z']
			}
			list_data.append(line_data)
	return list_data


if __name__ == "__main__":
	train_copick_root = get_copick_root()
	# print(f"{copick_root.config = }")
	# print(f"{copick_root.runs = }")
	run_names = get_experiment_names(train_copick_root)
	#t, res_dict = next(iter(get_zarr_arrays(copick_root, run_names[0])))
	get_run_labels(train_copick_root, run_names[0])