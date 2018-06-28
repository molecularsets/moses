import torch


def assert_check(t, dims=None, dtype=None, device=None):
    assert isinstance(t, torch.Tensor), "Tensor initial check"

    if dims is not None:
        assert t.dim() == len(dims)
        for i, dim in enumerate(dims):
            if dim is not None and dim > 0:
                assert t.size(i) == dim, "Tensor dim check"

    if dtype is not None:
        assert t.dtype == dtype, "Tensor type check"

    if device is not None:
        assert t.device == device, "Tensor device check"


def get_device(config):
    return torch.device(
        f'cuda:{config.device_code}'
        if config.device_code >= 0 and torch.cuda.is_available()
        else 'cpu'
    )
