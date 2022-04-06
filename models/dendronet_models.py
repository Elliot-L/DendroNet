import torch
import torch.nn as nn


class DendroMatrixLinReg(nn.Module):
    def __init__(self, device, root_weights, path_mat, delta_mat, p=1, init_deltas=False, init_root=True):
        """
        param p: type of norm to take for dendronet loss
        """
        super(DendroMatrixLinReg, self).__init__()
        self.device = device
        self.path_mat = torch.tensor(path_mat, device=device, dtype=torch.double)
        self.p = p
        self.root_weights = nn.Parameter(torch.tensor(root_weights, device=device, dtype=torch.double,
                                                      requires_grad=True))
        if init_root:
            torch.nn.init.normal_(self.root_weights, mean=0.0, std=0.01)
        self.delta_mat = nn.Parameter(torch.tensor(delta_mat, device=device, dtype=torch.double, requires_grad=True))
        if init_deltas:
            torch.nn.init.normal_(self.delta_mat, mean=0.0, std=0.01)

    def delta_loss(self, idx):
        if idx is not None:
            edges = torch.max(self.path_mat[:, idx], dim=1)
            mat_slice = self.delta_mat.T[torch.nonzero(edges.values == 1.0).reshape(-1)]
            return torch.norm(mat_slice, p=self.p)
        return torch.norm(self.delta_mat, p=self.p)

    # node_idx identifies the paths relevant to all samples in x, in the same order
    def forward(self, x, node_idx):
        self.path_mat = self.path_mat.to(x.device)
        effective_weights = torch.add(self.root_weights, torch.matmul(self.delta_mat, self.path_mat[:, node_idx]).T)
        # this works for linreg with bias-in only
        #  print("Inside: " + str(x.size()))
        return torch.sum((x * effective_weights), dim=1)

class DendroMatrixLogReg(DendroMatrixLinReg):
    def __init__(self, device, root_weights, path_mat, delta_mat, p=1, init_deltas=False):
        super(DendroMatrixLogReg, self).__init__(device, root_weights, path_mat, delta_mat, p, init_deltas)

    # node_idx identifies the paths relevant to all samples in x, in the same order
    def forward(self, x, node_idx):
        effective_weights = torch.add(self.root_weights, torch.matmul(self.delta_mat, self.path_mat[:, node_idx]).T)
        # this works for logreg with bias-in only, note that this is a one-output version requiring BCELoss
        return torch.sigmoid(torch.sum((x * effective_weights), dim=1))

    # todo: incorporate into forward to eliminate repeat code? Currently used for feature importance analysis
    def get_effective_weights(self, node_idx):
        return torch.add(self.root_weights, torch.matmul(self.delta_mat, self.path_mat[:, node_idx]).T)


"""
EXPERIMENTAL - NEEDS TO BE TESTED
For now, this is being implemented as a 1-hidden layer neural net, with variable layer sizes
todo: can this be generalized to take in a varying number of layers as well?
"""
class DendroMatrixNN(nn.Module):
    def __init__(self, device, num_features, layer_sizes, path_mat, p=1, init_deltas=False):
        """
        param p: type of norm to take for dendronet loss
        """
        super(DendroMatrixNN, self).__init__()
        assert len(layer_sizes) == 2, 'unsupported number of layer sizes'
        self.device = device
        self.num_features = num_features
        self.layer_sizes = layer_sizes
        self.p = p
        self.init_deltas = init_deltas
        self.path_mat = torch.tensor(path_mat, device=device, dtype=torch.double)
        self.num_edges = self.path_mat.shape[0]
        # creating the weights present at the root node for each layer
        self.root_lin1 = nn.Parameter(torch.zeros(size=(num_features, layer_sizes[0]), device=device,
                                                  dtype=torch.double, requires_grad=True))
        self.root_lin2 = nn.Parameter(torch.zeros(size=(layer_sizes[0], layer_sizes[1]), device=device,
                                                  dtype=torch.double, requires_grad=True))
        # initializing the root layers
        torch.nn.init.normal_(self.root_lin1, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.root_lin2, mean=0.0, std=0.01)

        # creating the delta matrices for each layer, of size (outputs, inputs, num_edges)
        self.delta_mat1 = nn.Parameter(torch.zeros(size=(layer_sizes[0], num_features, self.num_edges),
                                                   device=device, dtype=torch.double, requires_grad=True))
        self.delta_mat2 = nn.Parameter(torch.zeros(size=(layer_sizes[1], layer_sizes[0], self.num_edges),
                                                   device=device, dtype=torch.double, requires_grad=True))
        if init_deltas:
            torch.nn.init.normal_(self.delta_mat1, mean=0.0, std=0.01)
            torch.nn.init.normal_(self.delta_mat2, mean=0.0, std=0.01)

    def delta_loss(self):
        return torch.norm(self.delta_mat1, p=self.p) + torch.norm(self.delta_mat2, p=self.p)

    def forward(self, x, node_idx):
        # operations for the first layer, with relu activation
        effective_weights_lin1 = torch.add(self.root_lin1, torch.matmul(self.delta_mat1, self.path_mat[:, node_idx]).T)
        lin1_out = torch.nn.functional.relu(torch.sum((x * effective_weights_lin1.permute(-1, 0, 1)), dim=-1).T)

        # operations for the output layer, raw scores for use with CrossEntropyLoss, assumes bias is in
        # todo: confirm 1 output with MSELoss is equivalent for regression here, should be fine
        effective_weights_lin2 = torch.add(self.root_lin2, torch.matmul(self.delta_mat2, self.path_mat[:, node_idx]).T)
        return torch.sum((lin1_out * effective_weights_lin2.permute(-1, 0, 1)), dim=-1).T