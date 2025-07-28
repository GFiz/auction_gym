import pytest
import numpy as np
from unittest.mock import Mock, patch

from auction_gym.core.mechanism import (
    MechanismMeta, 
    AuctionMechanism, 
    SecondPriceAuction, 
    FirstPriceAuction
)


class TestMechanismMeta:
    """Test the MechanismMeta metaclass functionality."""
    
    def test_mechanism_registration(self):
        """Test that mechanisms are automatically registered with their mechanism_type."""
        # Clear registry for clean test
        MechanismMeta._registry.clear()
        
        # Create a test mechanism class
        class TestMechanism(AuctionMechanism):
            mechanism_type = 'test_mechanism'
            
            def allocate(self, bids):
                return [1, 0]
            
            def payment(self, bids, allocations):
                return [10.0, 0.0]
        
        # Check that the mechanism was registered
        assert 'test_mechanism' in MechanismMeta._registry
        assert MechanismMeta._registry['test_mechanism'] == TestMechanism
    
    def test_abstract_class_not_registered(self):
        """Test that abstract classes are not registered."""
        # Clear registry for clean test
        MechanismMeta._registry.clear()
        
        # Abstract class should not be registered
        class AbstractMechanism(AuctionMechanism):
            mechanism_type = 'abstract_mechanism'
            # No implementation of abstract methods
        
        assert 'abstract_mechanism' not in MechanismMeta._registry
    
    def test_get_available_mechanism_types(self):
        """Test getting list of available mechanism types."""
        # Clear registry and add some test mechanisms
        MechanismMeta._registry.clear()
        
        class Mechanism1(AuctionMechanism):
            mechanism_type = 'mechanism1'
            
            def allocate(self, bids):
                return [1, 0]
            
            def payment(self, bids, allocations):
                return [10.0, 0.0]
        
        class Mechanism2(AuctionMechanism):
            mechanism_type = 'mechanism2'
            
            def allocate(self, bids):
                return [0, 1]
            
            def payment(self, bids, allocations):
                return [0.0, 10.0]
        
        available_types = MechanismMeta.get_available_mechanism_types()
        assert 'mechanism1' in available_types
        assert 'mechanism2' in available_types
        assert len(available_types) == 2
    
    def test_get_mechanism_class_success(self):
        """Test successfully getting a mechanism class by type."""
        # Clear registry and add a test mechanism
        MechanismMeta._registry.clear()
        
        class TestMechanism(AuctionMechanism):
            mechanism_type = 'test_mechanism'
            
            def allocate(self, bids):
                return [1, 0]
            
            def payment(self, bids, allocations):
                return [10.0, 0.0]
        
        retrieved_class = MechanismMeta.get_mechanism_class('test_mechanism')
        assert retrieved_class == TestMechanism
    
    def test_get_mechanism_class_not_found(self):
        """Test getting a mechanism class that doesn't exist."""
        # Clear registry
        MechanismMeta._registry.clear()
        
        with pytest.raises(ValueError) as exc_info:
            MechanismMeta.get_mechanism_class('nonexistent_mechanism')
        
        assert "Mechanism type 'nonexistent_mechanism' not registered" in str(exc_info.value)
        assert "Available types: []" in str(exc_info.value)


class TestAuctionMechanism:
    """Test the abstract AuctionMechanism class."""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that AuctionMechanism cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AuctionMechanism()
    
    def test_class_methods_delegate_to_metaclass(self):
        """Test that class methods properly delegate to the metaclass."""
        # Clear registry and add a test mechanism
        MechanismMeta._registry.clear()
        
        class TestMechanism(AuctionMechanism):
            mechanism_type = 'test_mechanism'
            
            def allocate(self, bids):
                return [1, 0]
            
            def payment(self, bids, allocations):
                return [10.0, 0.0]
        
        # Test get_available_mechanism_types
        available_types = AuctionMechanism.get_available_mechanism_types()
        assert 'test_mechanism' in available_types
        
        # Test get_mechanism_class
        retrieved_class = AuctionMechanism.get_mechanism_class('test_mechanism')
        assert retrieved_class == TestMechanism


class TestSecondPriceAuction:
    """Test the SecondPriceAuction class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mechanism = SecondPriceAuction()
    
    def test_mechanism_type(self):
        """Test that the mechanism type is correctly set."""
        assert self.mechanism.mechanism_type == "second_price"
    
    def test_allocate_single_winner(self):
        """Test allocation with a clear single winner."""
        bids = [10.0, 5.0, 8.0]
        allocations = self.mechanism.allocate(bids)
        
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(allocations, expected)
    
    def test_allocate_tie_breaking(self):
        """Test allocation when there are ties (first one wins)."""
        bids = [10.0, 10.0, 5.0]
        allocations = self.mechanism.allocate(bids)
        
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(allocations, expected)
    
    def test_allocate_all_same_bids(self):
        """Test allocation when all bids are the same."""
        bids = [5.0, 5.0, 5.0]
        allocations = self.mechanism.allocate(bids)
        
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(allocations, expected)
    
    def test_payment_second_highest_bid(self):
        """Test that winner pays the second highest bid."""
        bids = [10.0, 5.0, 8.0]
        allocations = np.array([True, False, False])
        payments = self.mechanism.payment(bids, allocations)
        
        expected = np.array([8.0, 0.0, 0.0])  # Winner pays 8.0 (second highest)
        np.testing.assert_array_equal(payments, expected)
    
    def test_payment_single_bidder_error(self):
        """Test that payment raises error with less than 2 bidders."""
        bids = [10.0]
        allocations = np.array([True])
        
        with pytest.raises(ValueError) as exc_info:
            self.mechanism.payment(bids, allocations)
        
        assert "Number of bidders is less than two" in str(exc_info.value)
    
    def test_payment_zero_bidders_error(self):
        """Test that payment raises error with zero bidders."""
        bids = []
        allocations = np.array([])
        
        with pytest.raises(ValueError) as exc_info:
            self.mechanism.payment(bids, allocations)
        
        assert "Number of bidders is less than two" in str(exc_info.value)
    
    def test_payment_tie_second_highest(self):
        """Test payment when there are ties for highest bid."""
        bids = [10.0, 10.0, 5.0]
        allocations = np.array([True, False, False])
        payments = self.mechanism.payment(bids, allocations)
        
        expected = np.array([10.0, 0.0, 0.0])  # Winner pays 10.0 (tied highest)
        np.testing.assert_array_equal(payments, expected)
    
    def test_run_auction_integration(self):
        """Test the complete auction process."""
        bids = [10.0, 5.0, 8.0]
        allocations, payments = self.mechanism.run_auction(bids)
        
        expected_allocations = np.array([True, False, False])
        expected_payments = np.array([8.0, 0.0, 0.0])
        
        np.testing.assert_array_equal(allocations, expected_allocations)
        np.testing.assert_array_equal(payments, expected_payments)


class TestFirstPriceAuction:
    """Test the FirstPriceAuction class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mechanism = FirstPriceAuction()
    
    def test_mechanism_type(self):
        """Test that the mechanism type is correctly set."""
        assert self.mechanism.mechanism_type == "first_price"
    
    def test_allocate_single_winner(self):
        """Test allocation with a clear single winner."""
        bids = [10.0, 5.0, 8.0]
        allocations = self.mechanism.allocate(bids)
        
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(allocations, expected)
    
    def test_allocate_tie_breaking(self):
        """Test allocation when there are ties (first one wins)."""
        bids = [10.0, 10.0, 5.0]
        allocations = self.mechanism.allocate(bids)
        
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(allocations, expected)
    
    def test_payment_highest_bid(self):
        """Test that winner pays their own bid."""
        bids = [10.0, 5.0, 8.0]
        allocations = np.array([True, False, False])
        payments = self.mechanism.payment(bids, allocations)
        
        expected = np.array([10.0, 0.0, 0.0])  # Winner pays 10.0 (their own bid)
        np.testing.assert_array_equal(payments, expected)
    
    def test_payment_single_bidder_error(self):
        """Test that payment raises error with less than 1 bidder."""
        bids = []
        allocations = np.array([])
        
        with pytest.raises(ValueError) as exc_info:
            self.mechanism.payment(bids, allocations)
        
        assert "Number of bidders is less than two" in str(exc_info.value)
    
    def test_payment_tie_highest_bid(self):
        """Test payment when there are ties for highest bid."""
        bids = [10.0, 10.0, 5.0]
        allocations = np.array([True, False, False])
        payments = self.mechanism.payment(bids, allocations)
        
        expected = np.array([10.0, 0.0, 0.0])  # Winner pays 10.0 (highest bid)
        np.testing.assert_array_equal(payments, expected)
    
    def test_run_auction_integration(self):
        """Test the complete auction process."""
        bids = [10.0, 5.0, 8.0]
        allocations, payments = self.mechanism.run_auction(bids)
        
        expected_allocations = np.array([True, False, False])
        expected_payments = np.array([10.0, 0.0, 0.0])
        
        np.testing.assert_array_equal(allocations, expected_allocations)
        np.testing.assert_array_equal(payments, expected_payments)


class TestMechanismComparison:
    """Integration tests for mechanism registration and usage."""
    def test_mechanism_comparison(self):
        """Test that second price and first price auctions behave differently."""
        bids = [10.0, 5.0, 8.0]
        
        second_price = SecondPriceAuction()
        first_price = FirstPriceAuction()
        
        # Allocations should be the same
        sp_allocations, sp_payments = second_price.run_auction(bids)
        fp_allocations, fp_payments = first_price.run_auction(bids)
        
        np.testing.assert_array_equal(sp_allocations, fp_allocations)
        
        # Payments should be different
        assert not np.array_equal(sp_payments, fp_payments)
        assert sp_payments[0] == 8.0  # Second price: pays second highest
        assert fp_payments[0] == 10.0  # First price: pays own bid


class TestMechanismEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_negative_bids(self):
        """Test behavior with negative bids."""
        bids = [-5.0, -10.0, -3.0]
        
        second_price = SecondPriceAuction()
        first_price = FirstPriceAuction()
        
        # Should still work with negative bids
        sp_allocations, sp_payments = second_price.run_auction(bids)
        fp_allocations, fp_payments = first_price.run_auction(bids)
        
        # Winner should be the highest (least negative) bid
        expected_allocations = np.array([False, False, True])
        np.testing.assert_array_equal(sp_allocations, expected_allocations)
        np.testing.assert_array_equal(fp_allocations, expected_allocations)
    
    def test_zero_bids(self):
        """Test behavior with zero bids."""
        bids = [0.0, 0.0, 0.0]
        
        second_price = SecondPriceAuction()
        first_price = FirstPriceAuction()
        
        # Should handle zero bids
        sp_allocations, sp_payments = second_price.run_auction(bids)
        fp_allocations, fp_payments = first_price.run_auction(bids)
        
        # First bidder should win
        expected_allocations = np.array([True, False, False])
        np.testing.assert_array_equal(sp_allocations, expected_allocations)
        np.testing.assert_array_equal(fp_allocations, expected_allocations)
    
    def test_very_large_bids(self):
        """Test behavior with very large bid values."""
        bids = [1e10, 1e9, 1e8]
        
        second_price = SecondPriceAuction()
        first_price = FirstPriceAuction()
        
        # Should handle large numbers
        sp_allocations, sp_payments = second_price.run_auction(bids)
        fp_allocations, fp_payments = first_price.run_auction(bids)
        
        expected_allocations = np.array([True, False, False])
        np.testing.assert_array_equal(sp_allocations, expected_allocations)
        np.testing.assert_array_equal(fp_allocations, expected_allocations)
    
    def test_float_precision_bids(self):
        """Test behavior with floating point precision issues."""
        bids = [1.0000001, 1.0, 0.9999999]
        
        second_price = SecondPriceAuction()
        first_price = FirstPriceAuction()
        
        # Should handle floating point precision
        sp_allocations, sp_payments = second_price.run_auction(bids)
        fp_allocations, fp_payments = first_price.run_auction(bids)
        
        expected_allocations = np.array([True, False, False])
        np.testing.assert_array_equal(sp_allocations, expected_allocations)
        np.testing.assert_array_equal(fp_allocations, expected_allocations)


class TestMechanismPerformance:
    """Test performance characteristics of mechanisms."""
    
    def test_large_number_of_bidders(self):
        """Test behavior with a large number of bidders."""
        n_bidders = 1000
        bids = np.random.uniform(0, 100, n_bidders)
        
        second_price = SecondPriceAuction()
        first_price = FirstPriceAuction()
        
        # Should handle large numbers of bidders
        sp_allocations, sp_payments = second_price.run_auction(bids)
        fp_allocations, fp_payments = first_price.run_auction(bids)
        
        # Should have exactly one winner
        assert np.sum(sp_allocations) == 1
        assert np.sum(fp_allocations) == 1
        
        # Winner should be the highest bidder
        winner_idx = np.argmax(bids)
        assert sp_allocations[winner_idx] == True
        assert fp_allocations[winner_idx] == True
    
    @pytest.mark.parametrize("mechanism_class", [SecondPriceAuction, FirstPriceAuction])
    def test_mechanism_consistency(self, mechanism_class):
        """Test that mechanisms produce consistent results for same inputs."""
        mechanism = mechanism_class()
        bids = [10.0, 5.0, 8.0]
        
        # Run auction multiple times
        results = []
        for _ in range(10):
            allocations, payments = mechanism.run_auction(bids)
            results.append((allocations, payments))
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            np.testing.assert_array_equal(result[0], first_result[0])
            np.testing.assert_array_equal(result[1], first_result[1])
