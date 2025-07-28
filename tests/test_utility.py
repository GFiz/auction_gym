import pytest
import numpy as np

from auction_gym.core.utility import LinearUtility, BaseUtility


class RiskAverseUtility(BaseUtility):
    """A custom utility model for testing."""
    def __call__(self, won: bool, valuation: float, payment: float) -> float:
        if won:
            return np.log(1 + valuation - payment)
        return 0.0


class QuadraticUtility(BaseUtility):
    """A quadratic utility model for testing."""
    def __call__(self, won: bool, valuation: float, payment: float) -> float:
        if won:
            return (valuation - payment) ** 2
        return 0.0


class ConstantUtility(BaseUtility):
    """A constant utility model for testing."""
    def __call__(self, won: bool, valuation: float, payment: float) -> float:
        if won:
            return 1.0
        return 0.0


def test_linear_utility():
    """Test LinearUtility function directly."""
    utility = LinearUtility()
    
    # Test winning case
    reward = utility(won=True, valuation=10.0, payment=6.0)
    assert reward == 4.0
    
    # Test losing case
    reward = utility(won=False, valuation=10.0, payment=6.0)
    assert reward == 0.0
    
    # Test edge case: valuation equals payment
    reward = utility(won=True, valuation=5.0, payment=5.0)
    assert reward == 0.0


def test_risk_averse_utility():
    """Test RiskAverseUtility function directly."""
    utility = RiskAverseUtility()
    
    # Test winning case
    reward = utility(won=True, valuation=10.0, payment=6.0)
    expected = np.log(1 + 10.0 - 6.0)  # log(5)
    assert np.isclose(reward, expected)
    
    # Test losing case
    reward = utility(won=False, valuation=10.0, payment=6.0)
    assert reward == 0.0
    
    # Test edge case: valuation equals payment
    reward = utility(won=True, valuation=5.0, payment=5.0)
    expected = np.log(1 + 5.0 - 5.0)  # log(1) = 0
    assert np.isclose(reward, expected)


def test_quadratic_utility():
    """Test QuadraticUtility function directly."""
    utility = QuadraticUtility()
    
    # Test winning case
    reward = utility(won=True, valuation=10.0, payment=6.0)
    expected = (10.0 - 6.0) ** 2  # 4^2 = 16
    assert reward == expected
    
    # Test losing case
    reward = utility(won=False, valuation=10.0, payment=6.0)
    assert reward == 0.0
    
    # Test edge case: valuation equals payment
    reward = utility(won=True, valuation=5.0, payment=5.0)
    assert reward == 0.0


def test_constant_utility():
    """Test ConstantUtility function directly."""
    utility = ConstantUtility()
    
    # Test winning case
    reward = utility(won=True, valuation=10.0, payment=6.0)
    assert reward == 1.0
    
    # Test losing case
    reward = utility(won=False, valuation=10.0, payment=6.0)
    assert reward == 0.0
    
    # Test edge case: valuation equals payment
    reward = utility(won=True, valuation=5.0, payment=5.0)
    assert reward == 1.0


def test_utility_edge_cases():
    """Test utility functions with edge cases."""
    linear_utility = LinearUtility()
    risk_averse_utility = RiskAverseUtility()
    
    # Test with zero valuation
    assert linear_utility(won=True, valuation=0.0, payment=0.0) == 0.0
    assert risk_averse_utility(won=True, valuation=0.0, payment=0.0) == 0.0
    
    # Test with very small values
    assert linear_utility(won=True, valuation=0.001, payment=0.0001) == 0.0009
    assert np.isclose(risk_averse_utility(won=True, valuation=0.001, payment=0.0001), 
                     np.log(1 + 0.0009))
    
    # Test with large values
    assert linear_utility(won=True, valuation=1000.0, payment=500.0) == 500.0
    assert np.isclose(risk_averse_utility(won=True, valuation=1000.0, payment=500.0), 
                     np.log(501))


@pytest.mark.parametrize("utility_class,expected_winner_reward,expected_loser_reward", [
    (LinearUtility, 0.2, 0.0),  # valuation - payment = 0.8 - 0.6 = 0.2
    (RiskAverseUtility, np.log(1.2), 0.0),  # log(1 + 0.8 - 0.6) = log(1.2)
    (QuadraticUtility, 0.04, 0.0),  # (0.8 - 0.6)^2 = 0.04
    (ConstantUtility, 1.0, 0.0),  # constant 1.0 for winner
])
def test_utility_consistency(utility_class, expected_winner_reward, expected_loser_reward):
    """Test that different utility functions behave consistently."""
    utility = utility_class()
    
    # Test winning case with same valuation/payment
    winner_reward = utility(won=True, valuation=0.8, payment=0.6)
    assert np.isclose(winner_reward, expected_winner_reward)
    
    # Test losing case
    loser_reward = utility(won=False, valuation=0.8, payment=0.6)
    assert np.isclose(loser_reward, expected_loser_reward)


def test_utility_negative_payment():
    """Test utility functions with negative payment (edge case)."""
    linear_utility = LinearUtility()
    risk_averse_utility = RiskAverseUtility()
    
    # Test with negative payment (shouldn't happen in practice but good to test)
    assert linear_utility(won=True, valuation=5.0, payment=-2.0) == 7.0
    assert np.isclose(risk_averse_utility(won=True, valuation=5.0, payment=-2.0), 
                     np.log(8))  # log(1 + 5 - (-2)) = log(8) 