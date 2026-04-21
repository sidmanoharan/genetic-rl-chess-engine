"""
operators.py
------------
Genetic algorithm operators: selection, crossover, and mutation.

These three operations are the engine of evolution:

    Selection:  Which agents get to reproduce? (survival of the fittest)
    Crossover:  How do two parent genomes combine to make a child?
    Mutation:   Random small changes that introduce genetic diversity.

Without mutation, the population converges to a local optimum and stops
improving. Without selection, there's no pressure to improve. Without
crossover, there's no benefit to having a population — it's just independent
parallel training.
"""

import numpy as np
import copy
from typing import List, Tuple

from genetic.genome import Genome


# ---------------------------------------------------------------------------
# Selection: choose which agents survive and reproduce
# ---------------------------------------------------------------------------

def tournament_selection(
    genomes: List[Genome],
    fitness_scores: List[float],
    tournament_size: int = 3,
) -> Genome:
    """
    Tournament selection: run a mini-tournament and pick the winner.

    Randomly select `tournament_size` agents, then pick the one with
    the highest fitness. Repeat for each offspring needed.

    This is preferred over purely rank-based selection because:
        - It maintains selection pressure even when scores are close
        - It doesn't require sorting the entire population
        - The tournament size controls selection pressure (larger = more pressure)

    Args:
        genomes:         All genomes in the population.
        fitness_scores:  ELO rating (or other fitness) for each genome.
        tournament_size: Number of agents competing in each tournament.

    Returns:
        The genome of the tournament winner.
    """
    n = len(genomes)
    tournament_indices = np.random.choice(n, size=tournament_size, replace=False)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return copy.deepcopy(genomes[winner_idx])


def elitist_selection(
    genomes: List[Genome],
    fitness_scores: List[float],
    num_elite: int,
) -> List[Genome]:
    """
    Elitism: always keep the top N agents unchanged in the next generation.

    This prevents the best solution found so far from being lost due to
    unlucky mutation or crossover (a real risk without elitism).

    Args:
        genomes:         All genomes.
        fitness_scores:  Fitness for each genome.
        num_elite:       How many top agents to preserve unchanged.

    Returns:
        List of the top `num_elite` genomes (deep-copied).
    """
    sorted_pairs = sorted(
        zip(fitness_scores, genomes),
        key=lambda x: x[0],
        reverse=True
    )
    return [copy.deepcopy(genome) for _, genome in sorted_pairs[:num_elite]]


# ---------------------------------------------------------------------------
# Crossover: combine two parents to make a child
# ---------------------------------------------------------------------------

def uniform_crossover(parent_a: Genome, parent_b: Genome) -> Genome:
    """
    Uniform crossover: for each gene, randomly pick from parent A or B.

    This is the genetic equivalent of shuffling a deck of cards that
    contains half of each parent's genes.

    Args:
        parent_a: First parent genome.
        parent_b: Second parent genome.

    Returns:
        A new child genome with genes randomly drawn from both parents.
    """
    child = Genome()
    genes_a = parent_a.genes()
    genes_b = parent_b.genes()

    child_genes = {}
    for gene_name in genes_a:
        # 50/50 chance of inheriting from each parent
        if np.random.random() < 0.5:
            child_genes[gene_name] = genes_a[gene_name]
        else:
            child_genes[gene_name] = genes_b[gene_name]

    child.set_genes(child_genes)
    return child


def blend_crossover(parent_a: Genome, parent_b: Genome, alpha: float = 0.3) -> Genome:
    """
    BLX-α crossover: child genes are sampled from an extended range between parents.

    For each gene, the child value is sampled from
        [min(a,b) - α*range, max(a,b) + α*range]
    where range = max(a,b) - min(a,b).

    This allows the child to explore slightly beyond both parents,
    which can help escape local optima.

    Args:
        parent_a: First parent genome.
        parent_b: Second parent genome.
        alpha:    Extension factor. Higher = more exploration.

    Returns:
        A new child genome.
    """
    child = Genome()
    genes_a = parent_a.genes()
    genes_b = parent_b.genes()

    child_genes = {}
    for gene_name in genes_a:
        a_val = genes_a[gene_name]
        b_val = genes_b[gene_name]

        low_limit, high_limit = child.RANGES[gene_name]

        val_min = min(a_val, b_val)
        val_max = max(a_val, b_val)
        val_range = val_max - val_min

        sample_low = max(low_limit, val_min - alpha * val_range)
        sample_high = min(high_limit, val_max + alpha * val_range)

        child_genes[gene_name] = np.random.uniform(sample_low, sample_high)

    child.set_genes(child_genes)
    return child


# ---------------------------------------------------------------------------
# Mutation: random perturbation of genes
# ---------------------------------------------------------------------------

def gaussian_mutation(
    genome: Genome,
    mutation_rate: float = 0.15,
    mutation_strength: float = 0.2,
) -> Genome:
    """
    Gaussian mutation: add small random noise to each gene.

    Each gene has a `mutation_rate` chance of being mutated. When mutated,
    Gaussian noise is added (scaled by the gene's range and mutation_strength).

    Args:
        genome:           The genome to mutate (not modified in-place).
        mutation_rate:    Probability of mutating each gene.
        mutation_strength: How much noise to add (fraction of the gene's range).

    Returns:
        A new mutated genome.
    """
    mutated = copy.deepcopy(genome)
    current_genes = mutated.genes()
    new_genes = {}

    for gene_name, current_value in current_genes.items():
        if np.random.random() < mutation_rate:
            low, high = mutated.RANGES[gene_name]
            gene_range = high - low

            # Add Gaussian noise scaled to the gene's range
            noise = np.random.normal(0, mutation_strength * gene_range)
            new_value = current_value + noise

            # Clamp to valid range
            new_value = float(np.clip(new_value, low, high))
            new_genes[gene_name] = new_value
        else:
            new_genes[gene_name] = current_value

    mutated.set_genes(new_genes)
    return mutated


# ---------------------------------------------------------------------------
# Full reproduction pipeline
# ---------------------------------------------------------------------------

def reproduce(
    parent_a: Genome,
    parent_b: Genome,
    mutation_rate: float = 0.15,
    mutation_strength: float = 0.2,
    use_blend_crossover: bool = True,
) -> Genome:
    """
    Full reproduction: crossover + mutation.

    This is the function called to produce each offspring in the
    next generation.

    Args:
        parent_a:             First parent.
        parent_b:             Second parent.
        mutation_rate:        Per-gene mutation probability.
        mutation_strength:    Noise magnitude for mutations.
        use_blend_crossover:  If True, use BLX-α crossover (more exploration).
                              If False, use uniform crossover.

    Returns:
        A new child genome.
    """
    if use_blend_crossover:
        child = blend_crossover(parent_a, parent_b)
    else:
        child = uniform_crossover(parent_a, parent_b)

    child = gaussian_mutation(child, mutation_rate, mutation_strength)
    return child
