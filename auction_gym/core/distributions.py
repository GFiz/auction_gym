from ray.rllib.models.torch.torch_distributions import TorchSquashedGaussian
from ray.rllib.utils.typing import TensorType

class TorchUnitIntervalGaussian(TorchSquashedGaussian):
    @classmethod
    def from_logits(cls, logits: TensorType, **kwargs):
        # Map tanh-squashed output to [0, 1]
        return super().from_logits(logits, low=0.0, high=1.0, **kwargs)