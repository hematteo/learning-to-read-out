"""DTensor-safe encoder initialization for llamascopium's ``Crosscoder``.

llamascopium 2.0.0b34 initializes ``W_E`` by rearranging the full ``W_D``. Under
head-parallel training, where both are DTensors sharded on ``n_heads``, that
rearrange deadlocked in the 4-GPU Pythia-6.9B and OLMo-2-7B runs. Only the two
trailing dims are permuted, so the transpose is done on each local shard and
wrapped back with ``W_E``'s placements. Single-device behavior is unchanged.

Importing this module applies the patch; ``wu_adapter`` imports it.
"""

import einops
import torch
from llamascopium.models.crosscoder import Crosscoder
from llamascopium.utils.timer import timer
from torch.distributed.tensor import DTensor


@timer.time("init_encoder_with_decoder_transpose")
@torch.no_grad()
def init_encoder_with_decoder_transpose(self: Crosscoder, factor: float = 1.0) -> None:
    sharded = isinstance(self.W_D, DTensor)
    W_D = self.W_D.to_local() if sharded else self.W_D
    transposed = einops.rearrange(W_D, "n_heads d_sae d_model -> n_heads d_model d_sae").clone().contiguous() * factor
    if sharded:
        transposed = DTensor.from_local(transposed, device_mesh=self.W_E.device_mesh, placements=self.W_E.placements)
    self.W_E.copy_(transposed)


Crosscoder.init_encoder_with_decoder_transpose = init_encoder_with_decoder_transpose
