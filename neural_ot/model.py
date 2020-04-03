import torch
import torch.nn as nn


class NeuralOT(nn.Module):
    def __init__(self, source_dual_net, target_dual_net, source_to_target_net, regularization_parameter=1e-2,
                 regularization_mode='entropy'):
        super().__init__()
        self.u = source_dual_net
        self.v = target_dual_net
        self.f = source_to_target_net
        self.eps = regularization_parameter
        if regularization_mode not in ['entropy', 'l2']:
            raise ValueError("``regularization_mode`` must be ``'entropy'`` or ``'l2'``.")
        self.mode = regularization_mode

    @staticmethod
    def l2_distances(x, y):
        """
        Parameters
        ----------
        x : torch.tensor
            Batch of images from source domain of shape ``(N_s, C, H, W)``.
        y : torch.tensor
            Batch of images from target domain of shape ``(N_t, C, H, W)``.

        Returns
        -------
        distances : torch.tensor
            Tensor of shape ``(N_s, N_t)`` with pairwise l2 distances between source and target images.
        """
        return torch.sum((x[:, None] - y) ** 2, dim=(-3, -2, -1))

    def plan_criterion(self, x, y):
        """
        Parameters
        ----------
        x : torch.tensor
            Batch of images from source domain of shape ``(N_s, C, H, W)``.
        y : torch.tensor
            Batch of images from target domain of shape ``(N_t, C, H, W)``.

        Returns
        -------
        loss : torch.tensor
            Loss for training dual neural networks.
        """
        self.u.train()
        self.v.train()
        u = self.u(x)
        v = self.v(y)
        c = self.l2_distances(x, y)

        if self.mode == 'entropy':
            regularization_term = -self.eps * torch.exp((u[:, None] + v - c) / self.eps)
        else:
            regularization_term = -torch.relu(u[:, None] + v - c) ** 2 / (4 * self.eps)

        return -torch.mean(u[:, None] + v + regularization_term)

    def mapping_criterion(self, x, y):
        """
        Parameters
        ----------
        x : torch.tensor
            Batch of images from source domain of shape ``(N, C, H, W)``.
        y : torch.tensor
            Batch of images from target domain of shape ``(N, C, H, W)``.

        Returns
        -------
        loss : torch.tensor
            Loss for training mapping neural network.
        """
        self.u.eval()
        self.v.eval()
        self.f.train()
        u = self.u(x).detach()
        v = self.v(y).detach()
        c = self.l2_distances(x, y)
        mapped = self.f(x)  # shape ``(N, C, H, W)``

        d = self.l2_distances(mapped, y)
        if self.mode == 'entropy':
            h = torch.exp((u[:, None] + v - c) / self.eps)
        else:
            h = torch.relu(u[:, None] + v - c) / (2 * self.eps)

        return torch.mean(d * h)

    def map(self, x):
        self.f.eval()
        return self.f(x)
