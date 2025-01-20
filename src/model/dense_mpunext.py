"""Mix between the MPUnet published in JMI and Unet++ declared with Pytorch"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


def conv2d(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     padding=1)



class DenseMpunext(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, dropout_p: float, fm_size = [6, 7, 8, 9]):
        super().__init__()
        self._activ = F.silu
        # Encoder
        #   First level
        #self.conv000 = conv2d(n_inputs, 6)
        self.conv000 = conv2d(n_inputs, fm_size[0])
        self.conv001 = conv2d(fm_size[0], fm_size[0])
        self.drop000 = nn.Dropout2d(dropout_p)
        self.inorm000 = nn.InstanceNorm2d(fm_size[0])
        self.inorm001 = nn.InstanceNorm2d(fm_size[0])
        #   Second level
        self.conv100 = conv2d(fm_size[0], fm_size[1])
        self.conv101 = conv2d(fm_size[1], fm_size[1])
        self.drop100 = nn.Dropout2d(dropout_p)
        self.inorm100 = nn.InstanceNorm2d(fm_size[1])
        self.inorm101 = nn.InstanceNorm2d(fm_size[1])
        self.conv1to0 = conv2d(fm_size[1], fm_size[0])
        #   Third level
        self.conv200 = conv2d(fm_size[1], fm_size[2])
        self.conv201 = conv2d(fm_size[2], fm_size[2])
        self.drop200 = nn.Dropout2d(dropout_p)
        self.inorm200 = nn.InstanceNorm2d(fm_size[2])
        self.inorm201 = nn.InstanceNorm2d(fm_size[2])
        self.conv2to1 = conv2d(fm_size[2], fm_size[1])

        # Bottleneck
        self.conv300 = conv2d(fm_size[2], fm_size[3])
        self.conv301 = conv2d(fm_size[3], fm_size[3])
        self.drop300 = nn.Dropout2d(dropout_p)
        self.inorm300 = nn.InstanceNorm2d(fm_size[3])
        self.inorm301 = nn.InstanceNorm2d(fm_size[3])
        self.conv3to2 = conv2d(fm_size[3], fm_size[2])

        # Middle of Unet++
        #   First level
        #       Second column
        self.conv002 = conv2d(fm_size[0], fm_size[0])
        self.conv003 = conv2d(fm_size[0], fm_size[0])
        self.conv_seg0 = nn.Conv2d(in_channels=fm_size[0],
                                   out_channels=n_outputs,
                                   kernel_size=1,
                                   padding=0)
        self.drop002 = nn.Dropout2d(dropout_p)
        self.inorm002 = nn.InstanceNorm2d(fm_size[0])
        self.inorm003 = nn.InstanceNorm2d(fm_size[0])
        #       Third column
        self.conv004 = conv2d(fm_size[0], fm_size[0])
        self.conv005 = conv2d(fm_size[0], fm_size[0])
        self.conv_seg1 = nn.Conv2d(in_channels=fm_size[0],
                                   out_channels=n_outputs,
                                   kernel_size=1,
                                   padding=0)
        self.drop004 = nn.Dropout2d(dropout_p)
        self.inorm004 = nn.InstanceNorm2d(fm_size[0])
        self.inorm005 = nn.InstanceNorm2d(fm_size[0])
        #   Second level
        #       Second column
        self.conv102 = conv2d(fm_size[1], fm_size[1])
        self.conv103 = conv2d(fm_size[1], fm_size[1])
        self.drop102 = nn.Dropout2d(dropout_p)
        self.inorm102 = nn.InstanceNorm2d(fm_size[1])
        self.inorm103 = nn.InstanceNorm2d(fm_size[1])

        # Decoder
        #   Third level
        self.conv202 = conv2d(fm_size[2], fm_size[2])
        self.conv203 = conv2d(fm_size[2], fm_size[2])
        self.drop202 = nn.Dropout2d(dropout_p)
        self.inorm202 = nn.InstanceNorm2d(fm_size[2])
        self.inorm203 = nn.InstanceNorm2d(fm_size[2])
        #   Second level
        self.conv104 = conv2d(fm_size[1], fm_size[1])
        self.conv105 = conv2d(fm_size[1], fm_size[1])
        self.drop104 = nn.Dropout2d(dropout_p)
        self.inorm104 = nn.InstanceNorm2d(fm_size[1])
        self.inorm105 = nn.InstanceNorm2d(fm_size[1])
        #   First level
        self.conv006 = conv2d(fm_size[0], fm_size[0])
        self.conv007 = conv2d(fm_size[0], n_outputs)
        self.drop006 = nn.Dropout2d(dropout_p)
        self.inorm006 = nn.InstanceNorm2d(fm_size[0])

    # def _activ(self, input_layer):
    #     # Activation function
    #     return F.leaky_relu(input_layer,
    #                         negative_slope=self.lr_slope)

    def forward(self, image):
        # Encoder
        #   First level
        fms = self._activ(self.inorm000(self.conv000(image)))
        fms = fms_00 = fms + self._activ(self.inorm001(self.conv001(self.drop000(fms))))
        #   Second level
        fms = self._activ(self.inorm100(self.conv100(F.max_pool2d(fms, (2, 2)))))
        fms = fms_10 = fms + self._activ(self.inorm101(self.conv101(self.drop100(fms))))
        #   Third level
        fms = self._activ(self.inorm200(self.conv200(F.max_pool2d(fms, (2, 2)))))
        fms = fms_20 = fms + self._activ(self.inorm201(self.conv201(self.drop200(fms))))

        # Bottleneck
        fms = self._activ(self.inorm300(self.conv300(F.max_pool2d(fms, (2, 2)))))
        fms = fms + self._activ(self.inorm301(self.conv301(self.drop300(fms))))

        # Middle of Unet++
        #   Second level
        #       Second column
        fms_11 = self._activ(self.inorm102(self.conv102(fms_10 + F.interpolate(self.conv2to1(fms_20),
                                                                               scale_factor=2,
                                                                               mode='nearest')
                                                        )
                                          )
                            )

        fms_11 = fms_11 + self._activ(self.inorm103(self.conv103(self.drop102(fms_11))))

        #   First level
        #       Second column
        fms_01 = self._activ(self.inorm002(self.conv002(fms_00 + F.interpolate(self.conv1to0(fms_10),
                                                                               scale_factor=2,
                                                                               mode='nearest')
                                                                             )))
        fms_01 = fms_01 + self._activ(self.inorm003(self.conv003(self.drop002(fms_01))))
        seg_01 = self.conv_seg0(fms_01)
        #       Third column
        fms_02 = self._activ(self.inorm004(self.conv004(fms_00 + fms_01 + F.interpolate(self.conv1to0(fms_11),
                                                                                        scale_factor=2,
                                                                                        mode='nearest')
                                                                             )))
        fms_02 = fms_02 + self._activ(self.inorm005(self.conv005(self.drop004(fms_02))))
        seg_02 = self.conv_seg1(fms_02)

        # Decoder
        #   Third level
        fms = self._activ(self.inorm202(self.conv202(fms_20 + F.interpolate(self.conv3to2(fms),
                                                                             scale_factor=2,
                                                                             mode='nearest')
                                                                            )))
        fms = fms + self._activ(self.inorm203(self.conv203(self.drop202(fms))))

        #   Second level
        fms = self._activ(self.inorm104(self.conv104(fms_10 + fms_11 + F.interpolate(self.conv2to1(fms),
                                                                                     scale_factor=2,
                                                                                     mode='nearest')
                                                                                    )))
        fms = fms + self._activ(self.inorm105(self.conv105(self.drop104(fms))))
        #   First level
        fms = self._activ(self.inorm006(self.conv006(fms_00 + fms_01 + fms_02 +
                                                     F.interpolate(self.conv1to0(fms),
                                                                   scale_factor=2,
                                                                   mode='nearest')
                                                                   )))
        fms = torch.sigmoid(self.conv007(self.drop006(fms)) + seg_01 + seg_02)
        return fms


def torch_weighted_batch_dice_loss_brut(target: torch.Tensor, output: torch.Tensor,
                                        epsilon: float = 1e-5) -> torch.Tensor:
    n_msk_vox = torch.sum(target)
    eval_weight = [1., 2., 1., 2., 1.] #There are five particles of interest, with three "easy" particles (ribosome, virus-like particles, and apo-ferritin) assigned a weight of 1 and two "hard" particles (thyroglobulin and β-galactosidase) assigned a weight of 2.
    weighted_target = target.clone().detach().to(float)
    for label in range(target.shape[1]): #weight by number of voxel in batch for each class
        weighted_target[:, label] *= eval_weight[label] * n_msk_vox / max(1., torch.sum(target[:, label])) 
    return -dice_score(weighted_target, output, epsilon)


def dice_score(target: torch.Tensor, output: torch.Tensor, epsilon: float = 1e-5):
    return (2.0 * (output * target).sum() + epsilon) / (
            (output + target).sum() + epsilon)

class LightningDenseMpunext(L.LightningModule):
    def __init__(self, n_inputs: int, n_outputs: int, dropout_p: float, fm_size = [6, 7, 8, 9], lr=1e-3):
        super().__init__()
        self.dense_mpunext = DenseMpunext(n_inputs, n_outputs, dropout_p, fm_size)
        self.lr = lr
        self.save_hyperparameters()

    def _pred_metric_loss_step(self, batch, batch_idx, split_str):
        img, msk = batch['tomogram']['data'], batch['mask']['data']
        #img_shape [batch, channel = 1, z, y, x]
        #msk_shape [batch, labels, z, y, x] on veut prédire uniquement la slice du milieu mais tous les labels
        assert msk.shape[2] % 2 == 1, f"Msk shape must not be pair, {msk.shape = }"
        actual_msk = msk[:, :, msk.shape[2]//2] # a verifier
        pred_msk = self.dense_mpunext(img.squeeze(1)) # remove "channel" dimension of img
        assert actual_msk.shape == pred_msk.shape, f"Shape of masks must be equal {actual_msk.shape = }, {pred_msk.shape = }"
        loss = torch_weighted_batch_dice_loss_brut(actual_msk, pred_msk)
        self.log(f"{split_str}_loss", loss)
        for label in range(actual_msk.shape[1]):
            self.log(f"{split_str}_dice_label-{label}", dice_score(actual_msk[:, label], pred_msk[:, label]))
        return loss

    def training_step(self, batch, batch_idx):
        return self._pred_metric_loss_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._pred_metric_loss_step(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.dense_mpunext(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.dense_mpunext.parameters(), lr=self.lr)
        return optimizer



if __name__ == '__main__':
    model = DenseMpunext(5, 5, 0.3, fm_size=[6, 7, 8, 9])
    test = torch.rand((1, 5, 640, 640))
    res = model(test)
    pass
