from typing import Optional, List
from torch import Tensor


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, dtype):
        cast_tensor = self.tensors.to(device)
        cast_tensor = cast_tensor.to(dtype)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
            cast_mask = cast_mask.to(dtype)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)