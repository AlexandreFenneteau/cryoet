import csv
import os
import json
from abc import ABC
from typing import List

import cv2
import numpy as np
import torch

import torchio as tio
import yaml

class ExportYOLOFormat(ABC):
    """Convert the pytorch 3D data as coco images.
            see https://docs.ultralytics.com/datasets/detect/ """

    def __init__(self, preproc_pytorch_dir: str, out_dir: str, slicing: str,
                 overwrite: bool = False, slicing_radius_ratio: float = 0.8,
                 write_msk: bool = False, out_img_dim = (640, 640)):
        assert slicing in [
            'x', 'y', 'z'], f"{slicing = } should be only 'x', 'y' or 'z'"
        super().__init__()
        self.preproc_pytorch_folder = preproc_pytorch_dir
        self.slicing = slicing
        self.slicing_radius_ratio = slicing_radius_ratio
        self.img_axes_maps = {'z': 0, 'y': 1, 'x': 2}
        self.write_msk = write_msk 

        self.overwrite = overwrite

        self.slicing_name = "sagittal" if self.slicing == 'x' else "coronal" if self.slicing == 'y' else "axial"
        self.slicing_idx = self.img_axes_maps[self.slicing]

        self.out_dir = os.path.join(out_dir, self.slicing_name)

        self.width_axe = 'y' if self.slicing == 'x' else 'x'
        self.height_axe = 'y' if self.slicing == 'z' else 'z'

        self.width_ax_idx = self.img_axes_maps[self.width_axe]
        self.height_ax_idx = self.img_axes_maps[self.height_axe]

        self.intensity_scaler = tio.transforms.RescaleIntensity(
            out_min_max=(0, 2**16), percentiles=(0.5, 99.5))
        self.out_img_dim = out_img_dim  # width, height of images

    
    def write_exp_imgs(self, exp_name: str, split: str) -> List[str]:
        pt_path = os.path.join(self.preproc_pytorch_folder, f"{exp_name}_res-0_img.pt")
        # load_pytorch 3d array res 0
        tensor_data = torch.load(pt_path)
        # rescale intensity in (0, 2**16) to be used as 16bit depth img
        # intensity rescale to 0 -> 2^{16}  #torchio rescaleintensity? #scipy?
        tensor_data = self.intensity_scaler(
            tio.ScalarImage(tensor=tensor_data.unsqueeze(0))).data[0]

        n_slices = tensor_data.shape[self.slicing_idx]
        img_output_dir = os.path.join(self.out_dir, "images", split)
        os.makedirs(img_output_dir, exist_ok=True)
        # for i in each slice
        output_paths = []
        for i in range(n_slices):
            # we only keep each of three slice in a channel max(0, i-1), i, min(i+1, 639)
            # for borders, we duplicate the middle in the missing direction
            img_path = os.path.join(
                img_output_dir, f"{exp_name}_{i:04}.png")
            output_paths.append(img_path)
            if not (os.path.exists(img_path)) or self.overwrite:
                i_slice_min = max(0, i-1)
                i_slice_max = min(i+1, n_slices-1)
                if self.slicing_idx == 0:
                    three_channel_slice = torch.cat([tensor_data[i_slice_min, :, :].unsqueeze(-1),
                                                        tensor_data[i, :,
                                                                    :].unsqueeze(-1),
                                                        tensor_data[i_slice_max, :, :].unsqueeze(-1)], dim=-1)
                elif self.slicing_idx == 1:
                    three_channel_slice = torch.cat([tensor_data[:, i_slice_min, :].unsqueeze(-1),
                                                        tensor_data[:, i,
                                                                    :].unsqueeze(-1),
                                                        tensor_data[:, i_slice_max, :].unsqueeze(-1)], dim=-1)
                else:
                    three_channel_slice = torch.cat([tensor_data[:, :, i_slice_min].unsqueeze(-1),
                                                        tensor_data[:, :,
                                                                    i].unsqueeze(-1),
                                                        tensor_data[:, :, i_slice_max].unsqueeze(-1)], dim=-1)
                three_channel_slice = three_channel_slice.numpy()
                if self.out_img_dim is not None:
                    three_channel_slice = cv2.resize(
                        three_channel_slice, self.out_img_dim, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(
                    img_path, three_channel_slice.astype(np.uint16))
                print(f"{img_path} ... written!")
        return output_paths

    def write_exp_imgs_msks(self, exp_name: str, split: str) -> List[str]:
        pt_path = os.path.join(self.preproc_pytorch_folder, f"{exp_name}_res-0_msk.pt")
        # load_pytorch 3d array res 0
        tensor_data = torch.load(pt_path)

        n_slices = tensor_data.shape[self.slicing_idx + 1] # tensor_data shape [channel, z, y, x]
        img_output_dir = os.path.join(self.out_dir, "msks", split)
        os.makedirs(img_output_dir, exist_ok=True)
        # for i in each slice
        output_paths = []
        for i in range(n_slices):
            # we only keep each of three slice in a channel max(0, i-1), i, min(i+1, 639)
            # for borders, we duplicate the middle in the missing direction
            msk_path = os.path.join(
                img_output_dir, f"{exp_name}_{i:04}.png")
            output_paths.append(msk_path)
            if not (os.path.exists(msk_path)) or self.overwrite:
                if self.slicing_idx == 0:
                    msk_slice = torch.zeros(tensor_data.shape[2:], dtype=int)
                    for msk_id in range(tensor_data.shape[0]):
                        torch_slice = tensor_data[msk_id, i, :, :]
                        msk_slice[torch_slice == 1] = msk_id + 1
                elif self.slicing_idx == 1:
                    msk_slice = torch.zeros((tensor_data.shape[1], tensor_data.shape[3]), dtype=int)
                    for msk_id in range(tensor_data.shape[0]):
                        torch_slice = tensor_data[msk_id, :, i, :]
                        msk_slice[torch_slice == 1] = msk_id + 1
                else:
                    msk_slice = torch.zeros((tensor_data.shape[1], tensor_data.shape[2]), dtype=int)
                    for msk_id in range(tensor_data.shape[0]):
                        torch_slice = tensor_data[msk_id, :, :, i]
                        msk_slice[torch_slice == 1] = msk_id + 1
                msk_slice = msk_slice.unsqueeze(-1).numpy().astype(np.uint8)
                if self.out_img_dim is not None:
                    msk_slice = cv2.resize(msk_slice, self.out_img_dim, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(
                    msk_path, msk_slice.astype(np.uint8))
                print(f"{msk_path} ... written!")
        return output_paths

    def write_exp_bboxs(self, exp_name: str, split: str,
                        resolution: str ='0', original_img_shape=(184, 630, 630)):
        label_csv_path = os.path.join(self.preproc_pytorch_folder, f"{exp_name}_label_coords.csv")
        res_detail_path = os.path.join(self.preproc_pytorch_folder, f"{exp_name}_res-detail.json")

        yolo_label_path = os.path.join(
            self.out_dir, "labels", split)
        os.makedirs(yolo_label_path, exist_ok=True)
        # load res_detail
        with open(res_detail_path, 'r') as f:
            res = json.load(f)[resolution]

        # load label csv
        with open(label_csv_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            # for each particle
            for particle_dict in reader:
                # keys: experiment, particle_type, label, radius, x, y, z
                # world_coordinates (wc) = csv x, y, z
                # w_x, w_y, w_z = particle_dict['x'], particle_dict['y'], particle_dict['z']
                w_c = {'x': float(particle_dict['x']),
                        'y': float(particle_dict['y']),
                        'z': float(particle_dict['z'])}
                # voxel_coordinates (vc) = wc / res_detail
                # v_x, v_y, v_z = w_x / res_x, w_y / res_y, w_z / res_z
                v_c = {key: val / res[key]
                        for key, val in w_c.items()}
                # get the coordinates relative to image
                # r_x = v_x / img_shape[self.img_axes_maps['x']]
                # r_y = v_y / img_shape[self.img_axes_maps['y']]
                # y_z = v_z / img_shape[self.img_axes_maps['z']]
                r_c = {
                    key: val / original_img_shape[self.img_axes_maps[key]] for key, val in v_c.items()}

                particle_radius_vox = (
                    float(particle_dict["radius"]) * 3.) / (res['x'] + res['y'] + res['z'])
                # w, h = label['radius'] / widht and height of image
                w = particle_radius_vox / original_img_shape[self.width_ax_idx]
                h = particle_radius_vox / original_img_shape[self.height_ax_idx]

                # class = label['label'] - 1
                # in dict labels are from 1 to 6, yolo from 0 to 5
                particle_cls = int(particle_dict["label"]) - 1

                # for each slice in the (diameter_particle x slicing_radius_ratio)
                # create the bounding box
                center_slice_idx = int(v_c[self.slicing])
                first_slice = max(
                    0, center_slice_idx - int(particle_radius_vox * self.slicing_radius_ratio))
                last_slice = min(center_slice_idx + int(
                    particle_radius_vox * self.slicing_radius_ratio), original_img_shape[self.slicing_idx])
                for i in range(first_slice, last_slice):
                    label_txt_path = os.path.join(yolo_label_path, f"{exp_name}_{i:04}.txt")
                    # write_line(class, yolo_center_x, yolo_center_y, w, h)
                    with open(label_txt_path, 'a') as append_f:
                        append_f.write(
                            f"{particle_cls} {r_c[self.width_axe]} {r_c[self.height_axe]} {w} {h}\n")

    def write_training_imgs(self, split: str, experiment_names: List[str]):
        assert split in ["train", "val"], f"{split = } should be 'train' or 'val'"
        for exp_name in experiment_names:
            self.write_exp_imgs(exp_name, split)
            self.write_exp_bboxs(exp_name, split)
            if self.write_msk:
                self.write_exp_imgs_msks(exp_name, split)

    def write_testing_img(self, experiment_name) -> List[str]:
        output_paths = self.write_exp_imgs(experiment_name, 'test')
        return output_paths
    
    def write_yolo_yml_file(self, yolo_yml_path: str):
        # 2 write data_{slicing}.yaml at self.preproc_yolo_folder with
        yolo_yml_dict = {
            "path": os.path.abspath(self.out_dir),
            "train": '/'.join(["images", "train"]),
            "val": '/'.join(["images", "val"]),
            "names": {
                    0: "apo-ferritin",
                    1: "beta-galactosidase",
                    2: "ribosome",
                    3: "thyroglobulin",
                    4: "virus-like-particle"
            }
        }
        with open(yolo_yml_path, "w") as f:
            yaml.dump(yolo_yml_dict, f)


if __name__ == "__main__":
    from preproc_data import PreprocData
    from copick_management import CopickManagement
    from yolo_preproc import ExportYOLOFormat
    
    # Preproc 
    copick_mgt = CopickManagement(out_data_dir=r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc",
                                  in_data_dir=r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\czii-cryo-et-object-identification",
                                  split="train")
    
    preproc = PreprocData(copick_mgt,
                          out_folder=r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\pytorch",
                          overwrite=False,
                          write_msks=True)
    
    out = preproc.preproc_zarr_imgs()
    
    train_yolo_export = ExportYOLOFormat(
        preproc_pytorch_dir=r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\pytorch\train",
        out_dir=r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\yolo",
        slicing='z',
        overwrite=False,
        slicing_radius_ratio=0.8,
        write_msk=True,
        out_img_dim=None)

    train_yolo_export.write_training_imgs('train', ["TS_5_4", "TS_6_4", "TS_69_2", "TS_86_3", "TS_99_9", "TS_6_6", "TS_73_6"]) # all
    
    train_yolo_export = ExportYOLOFormat(
        preproc_pytorch_dir=r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\pytorch\train",
        out_dir=r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\yolo",
        slicing='y',
        overwrite=False,
        slicing_radius_ratio=0.8,
        write_msk=True,
        out_img_dim=None)

    train_yolo_export.write_training_imgs('train', ["TS_5_4", "TS_6_4", "TS_69_2", "TS_86_3", "TS_99_9", "TS_6_6", "TS_73_6"]) # all
    
    train_yolo_export = ExportYOLOFormat(
        preproc_pytorch_dir=r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\pytorch\train",
        out_dir=r"C:\Users\AlexandreFenneteau\Travail\perso\cryoet\data\preproc\yolo",
        slicing='x',
        overwrite=False,
        slicing_radius_ratio=0.8,
        write_msk=True,
        out_img_dim=None)

    train_yolo_export.write_training_imgs('train', ["TS_5_4", "TS_6_4", "TS_69_2", "TS_86_3", "TS_99_9", "TS_6_6", "TS_73_6"]) # all
    #train_yolo_export.write_training_imgs('train', ["TS_5_4"])
    #train_yolo_export.write_training_imgs('val', ["TS_6_6", "TS_73_6"])
    #train_yolo_export.write_training_imgs('val', ["TS_6_6"])
