import pytest
from auction_gym.core.auction import FirstPriceAuction, SecondPriceAuction

# Test cases: bids, expected_winner, expected_price
AUCTION_TEST_CASES = [
    ([0.5, 0.8, 0.3], 1, 0.5),   # Clear winner
    ([0.9, 0.9, 0.1], 0, 0.9),   # Tie for first place, winner is first one
    ([0.5], 0, 0.0),             # Single bidder
    ([], None, 0.0),             # No bidders
]

@pytest.mark.parametrize("bids, expected_winner, _", AUCTION_TEST_CASES)
def test_first_price_auction_winner(bids, expected_winner, _):
    """Test that the first-price auction allocates to the highest bidder."""
    auction = FirstPriceAuction()
    winner, _ = auction.run_auction(bids)
    assert winner == expected_winner

@pytest.mark.parametrize("bids, winner_idx, expected_price", [
    ([0.5, 0.8, 0.3], 1, 0.8),   # Winner pays their own bid
    ([0.9, 0.9, 0.1], 0, 0.9),   # Tie, winner pays their bid
    ([0.5], 0, 0.5),             # Single bidder pays their bid
])
def test_first_price_auction_payment(bids, winner_idx, expected_price):
    """Test that the winner of a first-price auction pays their own bid."""
    auction = FirstPriceAuction()
    payment = auction.payment(bids, winner_idx)
    assert payment == expected_price

@pytest.mark.parametrize("bids, expected_winner, _", AUCTION_TEST_CASES)
def test_second_price_auction_winner(bids, expected_winner, _):
    """Test that the second-price auction allocates to the highest bidder."""
    auction = SecondPriceAuction()
    winner, _ = auction.run_auction(bids)
    assert winner == expected_winner

@pytest.mark.parametrize("bids, winner_idx, expected_price", [
    ([0.5, 0.8, 0.3], 1, 0.5),   # Winner pays the second-highest bid
    ([0.9, 0.9, 0.1], 0, 0.9),   # Tie, winner pays the other highest bid
    ([0.5], 0, 0.0),             # Single bidder pays 0
])
def test_second_price_auction_payment(bids, winner_idx, expected_price):
    """Test that the winner of a second-price auction pays the second-highest bid."""
    auction = SecondPriceAuction()
    payment = auction.payment(bids, winner_idx)
    assert payment == expected_price 