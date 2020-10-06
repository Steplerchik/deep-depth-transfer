import torch
from .unsupervised_depth_model import UnsupervisedDepthModel


class ScaledUnsupervisedDepthModel(UnsupervisedDepthModel):
    def __init__(self, pose_net, depth_net, criterion, optimizer_parameters, result_visualizer=None,
                 *args, **kwargs):
        super().__init__(pose_net, depth_net, criterion, optimizer_parameters, result_visualizer, *args, **kwargs)
        self._log_min_depth = torch.nn.Parameter(torch.tensor(0))
        self._log_scale = torch.nn.Parameter(torch.tensor(0))

    def depth(self, x):
        out = self._depth_net(x, is_return_depth=False)
        out = torch.exp(self._log_min_depth) + torch.exp(self._log_scale) * torch.sigmoid(out)
        return out

    def loss(self, batch):
        result = super().loss(batch)
        result["log_scale"] = self._log_scale
        result["log_min_depth"] = self._log_min_depth
        return result
