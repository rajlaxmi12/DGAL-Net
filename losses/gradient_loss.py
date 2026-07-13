"""
==========================================================
DGAL-Net v2
Gradient Loss
==========================================================

Gradient Loss encourages preservation of edges and fine
structures by minimizing the difference between image
gradients of the prediction and the ground truth.

Loss:

L_grad = ||Gx(pred)-Gx(gt)||1 + ||Gy(pred)-Gy(gt)||1

==========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """
    Edge-aware Gradient Loss
    """

    def __init__(self):
        super().__init__()

        # Sobel X Kernel
        sobel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]]
        )

        # Sobel Y Kernel
        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]]
        )

        # Shape -> (1,1,3,3)
        self.register_buffer(
            "sobel_x",
            sobel_x.view(1, 1, 3, 3)
        )

        self.register_buffer(
            "sobel_y",
            sobel_y.view(1, 1, 3, 3)
        )

    def compute_gradient(self, image):
        """
        Compute Sobel gradients.

        Args
        ----
        image : (B,C,H,W)

        Returns
        -------
        grad_x
        grad_y
        """

        B, C, H, W = image.shape

        grad_x = []
        grad_y = []

        for c in range(C):

            channel = image[:, c:c+1]

            gx = F.conv2d(
                channel,
                self.sobel_x,
                padding=1,
            )

            gy = F.conv2d(
                channel,
                self.sobel_y,
                padding=1,
            )

            grad_x.append(gx)
            grad_y.append(gy)

        grad_x = torch.cat(grad_x, dim=1)
        grad_y = torch.cat(grad_y, dim=1)

        return grad_x, grad_y

    def forward(
        self,
        prediction,
        target,
    ):
        """
        Parameters
        ----------
        prediction : (B,C,H,W)

        target : (B,C,H,W)

        Returns
        -------
        Scalar Gradient Loss
        """

        pred_gx, pred_gy = self.compute_gradient(prediction)

        target_gx, target_gy = self.compute_gradient(target)

        loss_x = F.l1_loss(
            pred_gx,
            target_gx,
        )

        loss_y = F.l1_loss(
            pred_gy,
            target_gy,
        )

        loss = loss_x + loss_y

        return loss
