import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def get_last_layer(self):
        return self.linear

    def forward(self, x):
        # try:
        outputs = torch.sigmoid(self.linear(x))
        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()
        return outputs


class LinearMapping(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearMapping, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)

    def get_last_layer(self):
        return self.linear

    def forward(self, x):
        # try:
        outputs = self.linear(x)
        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()
        return outputs
