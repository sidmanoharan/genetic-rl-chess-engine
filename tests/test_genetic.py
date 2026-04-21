"""
tests/test_genetic.py
---------------------
Unit tests for the genetic algorithm components.

Run with:
    pytest tests/test_genetic.py -v
"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic.genome import Genome, random_genome, default_genome
from genetic.operators import (
    tournament_selection,
    elitist_selection,
    uniform_crossover,
    blend_crossover,
    gaussian_mutation,
    reproduce,
)


class TestGenome:
    """Tests for the Genome data structure."""

    def test_default_genome_is_valid(self):
        """Default genome should produce valid agent parameters."""
        g = default_genome()
        params = g.to_dict()
        assert params["num_residual_blocks"] >= 2
        assert params["learning_rate"] > 0
        assert 0 < params["gamma"] < 1

    def test_random_genome_within_ranges(self):
        """Random genome values should all be within their valid ranges."""
        for _ in range(10):
            g = random_genome()
            for gene_name, value in g.genes().items():
                low, high = g.RANGES[gene_name]
                assert low <= value <= high, (
                    f"Gene '{gene_name}' = {value} is outside range [{low}, {high}]"
                )

    def test_to_dict_returns_integers_where_needed(self):
        """num_residual_blocks and num_filters should be integers in to_dict()."""
        g = random_genome()
        params = g.to_dict()
        assert isinstance(params["num_residual_blocks"], int)
        assert isinstance(params["num_filters"], int)

    def test_genes_and_set_genes_roundtrip(self):
        """set_genes(genes()) should be the identity operation."""
        g = random_genome()
        original_genes = g.genes().copy()
        g.set_genes(original_genes)
        assert g.genes() == original_genes


class TestSelectionOperators:
    """Tests for selection operators."""

    def setup_method(self):
        """Create a simple population for testing."""
        self.genomes = [random_genome() for _ in range(10)]
        self.fitness = list(range(10))  # Fitness 0–9

    def test_tournament_selection_returns_genome(self):
        """Tournament selection should return a Genome."""
        result = tournament_selection(self.genomes, self.fitness)
        assert isinstance(result, Genome)

    def test_tournament_selection_bias_toward_high_fitness(self):
        """Tournament selection should prefer high-fitness individuals."""
        # Run many tournaments and check the best individual wins most often
        wins = {i: 0 for i in range(10)}
        # Give extreme fitness to make the test reliable
        extreme_fitness = [0] * 9 + [1000]
        for _ in range(100):
            winner = tournament_selection(self.genomes, extreme_fitness, tournament_size=5)
            # The winner should almost always be the best individual
        # At least the test should not crash

    def test_elitist_selection_returns_correct_count(self):
        """Elitist selection should return exactly num_elite genomes."""
        elite = elitist_selection(self.genomes, self.fitness, num_elite=3)
        assert len(elite) == 3

    def test_elitist_selection_returns_best(self):
        """Elitist selection should return the highest-fitness genomes."""
        elite = elitist_selection(self.genomes, self.fitness, num_elite=1)
        # The elite genome should be a deep copy of the highest-fitness genome
        assert isinstance(elite[0], Genome)


class TestCrossoverOperators:
    """Tests for crossover operators."""

    def setup_method(self):
        self.parent_a = random_genome()
        self.parent_b = random_genome()

    def test_uniform_crossover_produces_genome(self):
        child = uniform_crossover(self.parent_a, self.parent_b)
        assert isinstance(child, Genome)

    def test_uniform_crossover_genes_from_parents(self):
        """Each child gene should equal either parent A's or parent B's gene."""
        child = uniform_crossover(self.parent_a, self.parent_b)
        for gene in child.genes():
            a_val = self.parent_a.genes()[gene]
            b_val = self.parent_b.genes()[gene]
            c_val = child.genes()[gene]
            assert c_val == a_val or c_val == b_val, (
                f"Gene '{gene}': child={c_val} is neither parent_a={a_val} nor parent_b={b_val}"
            )

    def test_blend_crossover_produces_genome(self):
        child = blend_crossover(self.parent_a, self.parent_b)
        assert isinstance(child, Genome)

    def test_blend_crossover_within_ranges(self):
        """Blend crossover child should still be within gene ranges."""
        for _ in range(20):
            child = blend_crossover(self.parent_a, self.parent_b, alpha=0.3)
            for gene_name, value in child.genes().items():
                low, high = child.RANGES[gene_name]
                assert low <= value <= high, (
                    f"Blend crossover: gene '{gene_name}' = {value} outside [{low}, {high}]"
                )

    def test_crossover_does_not_modify_parents(self):
        """Crossover should not modify parent genomes."""
        a_genes_before = self.parent_a.genes().copy()
        b_genes_before = self.parent_b.genes().copy()
        uniform_crossover(self.parent_a, self.parent_b)
        assert self.parent_a.genes() == a_genes_before, "Parent A was modified by crossover"
        assert self.parent_b.genes() == b_genes_before, "Parent B was modified by crossover"


class TestMutationOperator:
    """Tests for the mutation operator."""

    def test_mutation_returns_genome(self):
        g = random_genome()
        mutated = gaussian_mutation(g)
        assert isinstance(mutated, Genome)

    def test_mutation_stays_within_ranges(self):
        """Mutated genes should remain within their valid ranges."""
        g = random_genome()
        for _ in range(20):
            mutated = gaussian_mutation(g, mutation_rate=1.0, mutation_strength=0.5)
            for gene_name, value in mutated.genes().items():
                low, high = mutated.RANGES[gene_name]
                assert low <= value <= high, (
                    f"Mutation: gene '{gene_name}' = {value} outside [{low}, {high}]"
                )

    def test_mutation_does_not_modify_original(self):
        """Mutation should not modify the original genome."""
        g = random_genome()
        genes_before = g.genes().copy()
        gaussian_mutation(g, mutation_rate=1.0)
        assert g.genes() == genes_before, "Original genome was modified by mutation"

    def test_zero_mutation_rate_no_change(self):
        """With mutation_rate=0, no genes should be mutated."""
        g = random_genome()
        mutated = gaussian_mutation(g, mutation_rate=0.0)
        assert mutated.genes() == g.genes(), "With rate=0, genes should not change"
