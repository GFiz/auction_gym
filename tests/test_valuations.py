import pytest
import torch
from auction_gym.core.valuation import (
    DeterministicValuation,
    UniformValuation,
    NormalValuation,
    GammaValuation
)

def test_deterministic_valuation():
    """Test that DeterministicValuation returns a fixed value."""
    val_model = DeterministicValuation(0.75)
    assert val_model() == 0.75
    # This class no longer uses a distribution, so we check the error
    with pytest.raises(NotImplementedError):
        val_model.get_distribution()

def test_uniform_valuation():
    """Test that UniformValuation returns a value within the specified range."""
    val_model = UniformValuation(low=0.2, high=0.8)
    sample = val_model()
    assert 0.2 <= sample <= 0.8
    # Check that the underlying distribution is Uniform
    dist = val_model.get_distribution()
    assert isinstance(dist, torch.distributions.Uniform)
    assert dist.low == 0.2
    assert dist.high == 0.8

def test_normal_valuation():
    """Test that NormalValuation samples from a normal distribution."""
    # We can't test the exact sample, but we can check the distribution
    mean, std = 0.5, 0.1
    val_model = NormalValuation(mean=mean, std=std)
    dist = val_model.get_distribution()
    assert isinstance(dist, torch.distributions.Normal)
    assert dist.loc == mean
    assert dist.scale == std
    # Sample a bunch of times and check if the mean is close
    samples = [val_model() for _ in range(1000)]
    assert abs(sum(samples) / len(samples) - mean) < 0.1

def test_gamma_valuation():
    """Test that GammaValuation samples from a gamma distribution."""
    shape, scale = 2.0, 1.0
    val_model = GammaValuation(shape=shape, scale=scale)
    dist = val_model.get_distribution()
    assert isinstance(dist, torch.distributions.Gamma)
    assert dist.concentration == shape  # concentration is shape
    assert dist.rate == 1 / scale     # rate is 1/scale in torch
    # Sample and check that it's positive
    sample = val_model()
    assert sample >= 0

def test_valuation_input_validation():
    """Test that invalid inputs to valuation models raise errors."""
    with pytest.raises(ValueError):
        DeterministicValuation(-1.0)
    with pytest.raises(ValueError):
        UniformValuation(-0.5, 1.0)
    with pytest.raises(ValueError):
        UniformValuation(0.8, 0.2) # high < low
    with pytest.raises(ValueError):
        NormalValuation(0.5, -0.1)
    with pytest.raises(ValueError):
        GammaValuation(-1.0, 1.0) 