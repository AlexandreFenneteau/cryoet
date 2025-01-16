import torch
from dataset.Dataset import CZIIDataModule, TrainDataset
from model.mpunext import LightningDenseMpunext
import lightning as L

n_slices = 5
n_classes = 5

model = LightningDenseMpunext(n_inputs=n_slices, n_outputs=n_classes, dropout_p=0.3, fm_size=[6, 7, 8, 9], lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_dir = "."

callbacks = [L.pytorch.callbacks.StochasticWeightAveraging(swa_lrs=1e-4, device=None),
             L.pytorch.callbacks.ModelCheckpoint(
                 monitor="val_loss",
                 save_top_k=-1,
                 every_n_epochs=1,
                 mode='min')]


#train_datamodule = CZIIDataModule(batch_size=32, res='0', patch_size=(5, 32, 32), patch_cache_size=32, n_patch_per_subject=16)

train_ds = TrainDataset(batch_size=128, res='0', patch_size=(5, 256+128, 256+128), patch_cache_size=256, n_patch_per_subject=128)
val_loader = train_ds.get_loader("val")
train_loader = train_ds.get_loader("train")

trainer = L.Trainer(callbacks=callbacks, max_epochs=10, default_root_dir=log_dir, log_every_n_steps=5, check_val_every_n_epoch=1)

trainer.fit(model, train_dataloaders=train_loader,
             val_dataloaders=val_loader)

