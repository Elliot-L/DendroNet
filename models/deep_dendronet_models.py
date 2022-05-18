import torch
import torch.nn as nn


class DeepNNDendroMatrix(nn.Module):
    def __init__(self, device, num_features, layer_sizes, path_mat, p=1, init_deltas=False, init_root=True, delta_layers=None, root_layers=None):
        """
            param p: type of norm to take for dendronet loss
            layer_sizes: number of nodes for hidden layers and output layer
            (the size of the input layer is the same as the number of features)
        """
        super(DeepNNDendroMatrix, self).__init__()
        self.device = device
        self.num_features = num_features
        first_layer_size = [self.num_features]
        self.layer_sizes = first_layer_size + layer_sizes
        self.p = p
        self.path_mat = torch.tensor(path_mat, device=device, dtype=torch.double)
        self.num_edges = self.path_mat.shape[0]

        if root_layers == None:
            root_layers = []
            # creating the weights present at the root node for each layer
            for i in range(len(self.layer_sizes) - 1):
                new_root = nn.Parameter(torch.zeros(size=(self.layer_sizes[i], self.layer_sizes[i + 1]),
                                                device=device, dtype=torch.double, requires_grad=True))
                root_layers.append(new_root)
            # initializing the root weights if needed
            if init_root:
                for i in range(len(root_layers)):
                    torch.nn.init.normal_(root_layers[i], mean=0.0, std=0.01)

            self.root_layers = nn.ParameterList(root_layers)
        else:
            root_layers = [nn.Parameter(layer) for layer in root_layers]
            self.root_layers = nn.ParameterList(root_layers)

        if delta_layers == None:
            delta_layers = []
            #  creating the delta matrices for each layer
            for i in range(len(self.layer_sizes) - 1):
                new_delta = nn.Parameter(torch.zeros(size=(self.layer_sizes[i+1], self.layer_sizes[i], self.num_edges),
                                                      device=device, dtype=torch.double, requires_grad=True))
                delta_layers.append(new_delta)
            if init_deltas:
                for i in range(len(delta_layers)):
                    torch.nn.init.normal_(delta_layers[i], mean=0.0, std=0.01)

            self.delta_layers = nn.ParameterList(delta_layers)
        else:
            delta_layers = [nn.Parameter(layer) for layer in delta_layers]
            self.delta_layers = nn.ParameterList(delta_layers)

    def delta_loss(self, idx):
        total = 0
        for delta_mat in self.delta_layers:
            total += torch.norm(delta_mat, p=self.p)
        return total

        """
        if idx is not None:
            edges = torch.max(self.path_mat[:, idx], dim=1)
            mat_slice = self.delta_mat.T[torch.nonzero(edges.values == 1.0).reshape(-1)]
            return torch.norm(mat_slice, p=self.p)
        return torch.norm(self.delta_mat, p=self.p)
        """

    def root_loss(self):
        total = 0
        for root_mat in self.root_layers:
            total += torch.norm(root_mat, p=self.p)
        return total

    def forward(self, x, node_idx):
        in_out = x
        # print(in_out.shape)
        for root_layer, delta_layer in zip(self.root_layers, self.delta_layers):
            effective_weights = torch.add(root_layer, torch.matmul(delta_layer, self.path_mat[:, node_idx]).permute(2, 1, 0))
            # print("effective")
            # print(effective_weights.shape)
            in_out = torch.nn.functional.relu(torch.sum((in_out * effective_weights.permute(-1, 0, 1)), dim=-1).T)
            # print("in out")
            # print(in_out.shape)
        return torch.squeeze(in_out)
