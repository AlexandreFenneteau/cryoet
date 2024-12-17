from abc import ABC
from math import ceil
from typing import Dict, Tuple

import torch

class TorchSphere(ABC):
	sphere_dist_dict: Dict[torch.Tensor] = {}
	#indexed by radius in integer
	# {20: torch.tensor([0, 0, 1, 1, 1, 0,
	#					[0, 1, 1, 1, 1, 0],
	#                    ....])}


	@classmethod
	def add_sphere(cls, tens: torch.Tensor, radius: float, label: int,
				   sphere_center: Tuple[int, int, int]):
		pass
		
		#if radius_vox not in cls.sphere_mask_dict:
		#	pass


	
	#@classmethod
	#def _generate_sphere_mask(radius_vox: int):
	#	"""Generate a tensor of size """

	@classmethod
	def _generate_sphere_mask(cls, radius):
		"""Generate an n-dimensional spherical mask.
		From https://stackoverflow.com/questions/46626267/how-to-generate-a-sphere-in-3d-numpy-array"""

		shape = (ceil(radius) * 2,) * 3
		position = position = (ceil(radius),) * 3
		assert len(position) == len(shape)
		n = len(shape)
		position = np.array(position).reshape((-1,) + (1,) * n)
		arr = np.linalg.norm(np.indices(shape) - position, axis=0)
		cls.sphere_dist_dict[radius] = torch.tensor((arr <= radius).astype(int))

if __name__ == "__main__":
	import numpy as np




	
	sp_idx = sphere_idx((3, 3, 3), 1, (1, 1, 1))
	print(sp_idx)
