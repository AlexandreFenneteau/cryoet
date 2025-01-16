from abc import ABC
from math import ceil
from typing import Dict, Tuple, Union
import numpy as np

import torch

class TorchSphere(ABC):
	sphere_mask_dict: Dict[Union[float, int], torch.Tensor] = {}
	#indexed by radius in integer/float
	# {20: torch.tensor([0, 0, 1, 1, 1, 0,
	#					[0, 1, 1, 1, 1, 0],
	#                    ....])}

	@classmethod
	def add_sphere(cls, mask_tens: torch.Tensor, radius: Union[float, int],
				   sphere_center: Tuple[int, int, int]) -> None:

		if radius not in cls.sphere_mask_dict:
			cls._generate_sphere_mask(radius)

		ceil_radius = ceil(radius)

		x_min = sphere_center[0] - ceil_radius
		x_max = sphere_center[0] + ceil_radius
		x_min_offset = -x_min if -x_min > 0 else 0
		x_max_offset = x_max - mask_tens.shape[0] if x_max - mask_tens.shape[0] > 0 else 0

		y_min = sphere_center[1] - ceil_radius
		y_max = sphere_center[1] + ceil_radius
		y_min_offset = -y_min if -y_min > 0 else 0
		y_max_offset = y_max - mask_tens.shape[1] if y_max - mask_tens.shape[1] > 0 else 0

		z_min = sphere_center[2] - ceil_radius
		z_max = sphere_center[2] + ceil_radius
		z_min_offset = -z_min if -z_min > 0 else 0
		z_max_offset = z_max - mask_tens.shape[2] if z_max - mask_tens.shape[2] > 0 else 0

		
		sphere_patch_shape = cls.sphere_mask_dict[radius].shape

		mask_tens[x_min + x_min_offset: x_max - x_max_offset,
			      y_min + y_min_offset: y_max - y_max_offset,
			      z_min + z_min_offset: z_max - z_max_offset] += cls.sphere_mask_dict[radius][x_min_offset: sphere_patch_shape[0]-x_max_offset,
																							  y_min_offset: sphere_patch_shape[1]-y_max_offset,
																							  z_min_offset: sphere_patch_shape[2]-z_max_offset]
		
	@classmethod
	def _generate_sphere_mask(cls, radius: Union[int, float]) -> None:
		"""Generate an n-dimensional spherical mask.
		From https://stackoverflow.com/questions/46626267/how-to-generate-a-sphere-in-3d-numpy-array"""

		shape = (ceil(radius) * 2,) * 3
		position = position = (ceil(radius),) * 3
		assert len(position) == len(shape)
		n = len(shape)
		position = np.array(position).reshape((-1,) + (1,) * n)
		arr = np.linalg.norm(np.indices(shape) - position, axis=0)
		cls.sphere_mask_dict[radius] = torch.tensor((arr < radius).astype(int))