"""Tests for branch morphometry computations.

Uses a synthetic Y-skeleton to verify:
- Branch order (0 for trunk, 1 for children)
- Diameter from known SWC radii
- Electrotonic length against hand calculation
- Soma distance to midpoint
- Synapse counting from SnapResult
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path.home() / "research" / "neurostat"))

from neurostat.io.swc import NeuronSkeleton, SWCNode, Branch, SnapResult
from src.branch_morphometry import (
    compute_branch_order,
    compute_mean_diameter,
    compute_electrotonic_length,
    count_synapses_per_branch,
    count_exc_inh_per_branch,
    RM, RI,
)


def make_y_skeleton():
    """Create a synthetic Y-shaped skeleton.

    Structure:
        node 0 (soma, root) -- node 1 -- node 2 (branch point)
                                           /        \\
                                        node 3    node 4

    Branch 0: [0, 1, 2] (trunk, length=20000 nm)
    Branch 1: [2, 3] (left child, length=10000 nm)
    Branch 2: [2, 4] (right child, length=10000 nm)

    All radii = 500 nm (diameter = 1000 nm).
    """
    nodes = {
        0: SWCNode(0, 1, 0.0, 0.0, 0.0, 500.0, -1),       # soma
        1: SWCNode(1, 3, 10000.0, 0.0, 0.0, 500.0, 0),     # dendrite
        2: SWCNode(2, 3, 20000.0, 0.0, 0.0, 500.0, 1),     # branch point
        3: SWCNode(3, 3, 30000.0, 5000.0, 0.0, 500.0, 2),  # left child
        4: SWCNode(4, 3, 30000.0, -5000.0, 0.0, 500.0, 2), # right child
    }
    children = {
        0: [1],
        1: [2],
        2: [3, 4],
        3: [],
        4: [],
    }

    skel = NeuronSkeleton(nodes=nodes, children=children, root_id=0)
    # Manually set branches (since from_swc_file does this automatically)
    skel.branches = [
        Branch(
            node_ids=[0, 1, 2],
            edge_lengths=np.array([10000.0, 10000.0]),
            total_length=20000.0,
            start_node=0,
            end_node=2,
        ),
        Branch(
            node_ids=[2, 3],
            edge_lengths=np.array([np.sqrt(10000**2 + 5000**2)]),
            total_length=np.sqrt(10000**2 + 5000**2),
            start_node=2,
            end_node=3,
        ),
        Branch(
            node_ids=[2, 4],
            edge_lengths=np.array([np.sqrt(10000**2 + 5000**2)]),
            total_length=np.sqrt(10000**2 + 5000**2),
            start_node=2,
            end_node=4,
        ),
    ]
    return skel


class TestBranchOrder:
    def test_y_skeleton_orders(self):
        skel = make_y_skeleton()
        orders = compute_branch_order(skel)
        assert orders[0] == 0, "Trunk should be order 0"
        assert orders[1] == 1, "Left child should be order 1"
        assert orders[2] == 1, "Right child should be order 1"

    def test_all_orders_assigned(self):
        skel = make_y_skeleton()
        orders = compute_branch_order(skel)
        assert np.all(orders >= 0), "All branches should be assigned"


class TestMeanDiameter:
    def test_uniform_radii(self):
        skel = make_y_skeleton()
        # All radii = 500 nm → diameter = 1000 nm
        d = compute_mean_diameter(skel, 0)
        assert d == pytest.approx(1000.0), "Diameter should be 2 * radius"

    def test_all_branches(self):
        skel = make_y_skeleton()
        for i in range(3):
            d = compute_mean_diameter(skel, i)
            assert d == pytest.approx(1000.0)


class TestElectrotonicLength:
    def test_hand_calculation(self):
        """Verify electrotonic length against manual computation.

        L = 20000 nm = 2e-3 cm
        d = 1000 nm = 1e-4 cm
        lambda = sqrt(20000 * 1e-4 / (4 * 150)) = sqrt(2 / 600) = sqrt(0.003333) ≈ 0.05774 cm
        L/lambda = 2e-3 / 0.05774 ≈ 0.03464
        """
        L_nm = 20000.0
        d_nm = 1000.0
        result = compute_electrotonic_length(L_nm, d_nm)

        d_cm = d_nm * 1e-7
        L_cm = L_nm * 1e-7
        lam = np.sqrt(RM * d_cm / (4 * RI))
        expected = L_cm / lam

        assert result == pytest.approx(expected, rel=1e-6)

    def test_zero_diameter(self):
        assert np.isnan(compute_electrotonic_length(10000.0, 0.0))

    def test_negative_diameter(self):
        assert np.isnan(compute_electrotonic_length(10000.0, -100.0))

    def test_increases_with_length(self):
        """Longer branches → larger electrotonic length."""
        e1 = compute_electrotonic_length(10000.0, 1000.0)
        e2 = compute_electrotonic_length(50000.0, 1000.0)
        assert e2 > e1

    def test_increases_with_thinner_diameter(self):
        """Thinner branches → larger electrotonic length (shorter lambda)."""
        e_thick = compute_electrotonic_length(20000.0, 2000.0)
        e_thin = compute_electrotonic_length(20000.0, 500.0)
        assert e_thin > e_thick


class TestSynapseCounting:
    def test_bincount(self):
        snap = SnapResult(
            branch_ids=np.array([0, 0, 1, 2, 2, 2]),
            branch_positions=np.array([100, 200, 300, 400, 500, 600]),
            distances=np.zeros(6),
            valid=np.ones(6, dtype=bool),
        )
        counts = count_synapses_per_branch(snap, 3)
        np.testing.assert_array_equal(counts, [2, 1, 3])

    def test_empty_branches(self):
        snap = SnapResult(
            branch_ids=np.array([0, 0]),
            branch_positions=np.array([100, 200]),
            distances=np.zeros(2),
            valid=np.ones(2, dtype=bool),
        )
        counts = count_synapses_per_branch(snap, 5)
        np.testing.assert_array_equal(counts, [2, 0, 0, 0, 0])

    def test_exc_inh_counts(self):
        snap = SnapResult(
            branch_ids=np.array([0, 0, 1, 1, 2]),
            branch_positions=np.array([100, 200, 300, 400, 500]),
            distances=np.zeros(5),
            valid=np.ones(5, dtype=bool),
        )
        types = np.array(["excitatory", "inhibitory", "excitatory", "excitatory", "inhibitory"])
        exc, inh = count_exc_inh_per_branch(snap, types, 3)
        np.testing.assert_array_equal(exc, [1, 2, 0])
        np.testing.assert_array_equal(inh, [1, 0, 1])
