import torch
import torch.nn as nn


class DeepNNDendroMatrix(nn.Module):
    def __init__(self, device, num_features, layer_sizes, path_mat, p=1, init_deltas=False, init_root=True):
        """
            param p: type of norm to take for dendronet loss
        """
        super(DeepNNDendroMatrix, self).__init__()
        self.device = device
        self.num_features = num_features
        first_layer_size = [self.num_features]
        self.layer_sizes = first_layer_size + layer_sizes
        self.p = p
        self.path_mat = path_mat
        self.num_edges = self.path_mat.shape[0]
        root_layers = []
        delta_layers = []

        # creating the weights present at the root node for each layer
        for i in range(len(layer_sizes) - 1):
            root_layers.append(nn.Parameter(torch.zeros(self.layer_sizes[i], self.layer_sizes[i + 1]),
                                                 device=device, dtype=torch.double, requires_grad=True))
        # initializing the root weights if needed
        if init_root:
            for i in range(len(root_layers)):
                torch.nn.init.normal_(root_layers[i], mean=0.0, std=0.01)

        self.root_layers = nn.ParameterList(root_layers)

        #  creating the delta matrices for each layer
        for i in range(len(layer_sizes)):
            delta_layers.append(nn.Parameter(torch.zeros(self.layer_sizes[i+1], self.layer_sizes[i], self.num_edges),
                                                  device=device, dtype=torch.double, requires_grad=True))
        if init_deltas:
            for i in range(len(delta_layers)):
                torch.nn.init.normal_(delta_layers[i], mean=0.0, std=0.01)

        self.delta_layers = nn.ParameterList(delta_layers)

    def delta_loss(self):
        total = 0
        for delta_mat in self.delta_layers:
            total += torch.norm(delta_mat, p=self.p)
        return total

    def forward(self, x, node_idx):
        in_out = x
        for root_layer, delta_layer in zip(self.root_layers, self.delta_layers):
            effective_weights = torch.add(root_layer, torch.matmul(delta_layer, self.path_mat[:, node_idx]).permute(1, 0, 2))
            in_out = torch.nn.functional.relu(torch.sum((in_out * effective_weights.permute(-1, 0, 1)), dim=1).T)
