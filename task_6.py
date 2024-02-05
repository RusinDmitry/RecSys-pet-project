import torch
def function02(tensor_input: torch.Tensor) -> torch.Tensor:
    shape = tensor_input.shape # две оси (матрица)

    tensor = torch.rand(shape, requires_grad=True)

    tensor.to(torch.float32)
    return tensor