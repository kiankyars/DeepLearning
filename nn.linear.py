import torch

linear = torch.nn.Linear(3, 3)
inputs = torch.rand(3, 3)

a= linear(inputs)
# tensor([[-0.4848,  0.4174, -0.1633],
#     [-0.4428,  0.5404, -0.1470],
#     [-0.7272,  0.0770,  0.0321]], grad_fn=<AddmmBackward0)
b = (inputs @ linear.weight.T).add(linear.bias)
# tensor([[-0.4848,  0.4174, -0.1633],
#     [-0.4428,  0.5404, -0.1470],
#     [-0.7272,  0.0770,  0.0321]], grad_fn=<AddBackward0)
print(a,b)
i = torch.rand(1,3)
layer = torch.nn.Linear(3, 2, False)
print(layer.weight)
print(layer(i))
x = torch.randn(1, requires_grad=True) + torch.randn(1)
print(x)
y = torch.randn(2, requires_grad=True).sum()
print(y)
