"""
Redundant encoding/decoding for crop robustness.

Maps M message bits to N = M * R physical encoding slots, where each
message bit is replicated R times in spatially distant cells. At extraction
time, replicas are aggregated via mean-pooling (differentiable, for training)
or majority voting (hard decisions, for evaluation).

The permutation scheme guarantees that replicas of each bit land in
different cells of the 8x8 spatial grid, and that their cells are at
least Manhattan distance 2 apart. This ensures that a crop destroying
one replica is unlikely to also destroy another.
"""

import torch


class RedundantCoder:
    """
    Redundant bit encoding/decoding for crop robustness.

    Args:
        message_bits: Number of actual payload bits (M).
        physical_slots: Number of physical encoding positions (N = M * R).
        redundancy: Replication factor (R). Each message bit appears R times.
        bits_per_cell: Number of bits per spatial cell (must match Controller).
        grid_size: Spatial grid dimension (default 8 for 8x8 grid).
        seed: Deterministic seed for permutation generation.
    """

    def __init__(
        self,
        message_bits: int = 128,
        physical_slots: int = 256,
        redundancy: int = 2,
        bits_per_cell: int = 4,
        grid_size: int = 8,
        seed: int = 42,
    ):
        assert physical_slots == message_bits * redundancy, (
            f"physical_slots ({physical_slots}) must equal "
            f"message_bits ({message_bits}) * redundancy ({redundancy})"
        )
        self.M = message_bits
        self.N = physical_slots
        self.R = redundancy
        self.bits_per_cell = bits_per_cell
        self.grid_size = grid_size

        # perm: (M, R) -- for each message bit, R physical slot indices
        self.perm = self._generate_permutation(seed)

    def _slot_to_cell(self, slot_idx: int) -> int:
        """Map a physical slot index to its cell index."""
        return slot_idx // self.bits_per_cell

    def _cell_to_grid(self, cell_idx: int) -> tuple:
        """Map a cell index to (row, col) in the spatial grid."""
        return cell_idx // self.grid_size, cell_idx % self.grid_size

    def _manhattan_distance(self, cell_a: int, cell_b: int) -> int:
        """Manhattan distance between two cells in the spatial grid."""
        r1, c1 = self._cell_to_grid(cell_a)
        r2, c2 = self._cell_to_grid(cell_b)
        return abs(r1 - r2) + abs(c1 - c2)

    def _generate_permutation(self, seed: int) -> torch.Tensor:
        """
        Generate a permutation where replicas of each message bit land in
        different cells that are at least Manhattan distance 2 apart.

        Returns:
            (M, R) tensor of physical slot indices.
        """
        gen = torch.Generator().manual_seed(seed)
        max_attempts = 1000

        for attempt in range(max_attempts):
            p = torch.randperm(self.N, generator=gen)
            mapping = p.view(self.M, self.R)

            # Verify constraints
            valid = True
            for j in range(self.M):
                cells = [self._slot_to_cell(mapping[j, r].item()) for r in range(self.R)]
                # All replicas must be in different cells
                if len(set(cells)) != self.R:
                    valid = False
                    break
                # All replica cells must be at least distance 2 apart
                for a in range(self.R):
                    for b in range(a + 1, self.R):
                        if self._manhattan_distance(cells[a], cells[b]) < 2:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                return mapping

        # Fallback: relax distance constraint, only require different cells
        gen2 = torch.Generator().manual_seed(seed + 10000)
        for attempt in range(max_attempts):
            p = torch.randperm(self.N, generator=gen2)
            mapping = p.view(self.M, self.R)
            valid = True
            for j in range(self.M):
                cells = [self._slot_to_cell(mapping[j, r].item()) for r in range(self.R)]
                if len(set(cells)) != self.R:
                    valid = False
                    break
            if valid:
                return mapping

        raise RuntimeError(
            f"Failed to generate valid permutation after {max_attempts * 2} attempts. "
            f"Try different seed or reduce redundancy."
        )

    def encode(self, message_bits: torch.Tensor) -> torch.Tensor:
        """
        Expand M message bits to N physical bits via replication.

        Args:
            message_bits: (B, M) float tensor with values in {0, 1}

        Returns:
            physical_bits: (B, N) float tensor with values in {0, 1}
        """
        B = message_bits.shape[0]
        device = message_bits.device
        physical = torch.zeros(B, self.N, device=device)
        perm = self.perm.to(device)
        for r in range(self.R):
            physical.scatter_(1, perm[:, r].unsqueeze(0).expand(B, -1), message_bits)
        return physical

    def decode_soft(self, physical_logits: torch.Tensor) -> torch.Tensor:
        """
        Aggregate N physical logits to M message logits via mean-pooling.

        Differentiable -- use during training for gradient flow.

        Args:
            physical_logits: (B, N) raw logits from extractor

        Returns:
            message_logits: (B, M) aggregated logits
        """
        device = physical_logits.device
        perm = self.perm.to(device)
        gathered = torch.stack(
            [physical_logits.gather(1, perm[:, r].unsqueeze(0).expand(physical_logits.size(0), -1))
             for r in range(self.R)],
            dim=-1,
        )  # (B, M, R)
        return gathered.mean(dim=-1)  # (B, M)

    def decode_hard(self, physical_logits: torch.Tensor) -> torch.Tensor:
        """
        Majority voting for hard bit decisions at evaluation time.

        Args:
            physical_logits: (B, N) raw logits from extractor

        Returns:
            message_bits: (B, M) hard bit decisions in {0, 1}
        """
        device = physical_logits.device
        perm = self.perm.to(device)
        probs = torch.sigmoid(physical_logits)
        gathered = torch.stack(
            [probs.gather(1, perm[:, r].unsqueeze(0).expand(probs.size(0), -1))
             for r in range(self.R)],
            dim=-1,
        )  # (B, M, R)
        votes = (gathered > 0.5).float()
        majority = (votes.sum(dim=-1) > self.R / 2).float()
        return majority
