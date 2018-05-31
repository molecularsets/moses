import torch

dtype = torch.FloatTensor

x = torch.randn(1, 10).type(dtype)
y = torch.randn(10, 1).type(dtype)

print((x @ y).squeeze().shape)

if __name__ == '__main__':
    pass
