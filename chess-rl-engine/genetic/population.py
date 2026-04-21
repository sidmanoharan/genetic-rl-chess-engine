"""
population.py
-------------
Manages the population of agents across generations.

The population is a group of agents, each with their own neural network
and genome. Every generation:
    1. Agents play against each other to determine fitness (ELO)
    2. The genetic operators select, crossover, and mutate genomes
    3. New agents are created from the evolved genomes
    4. The process repeats

Over many generations, the population should improve on average as
beneficial hyperparameter combinations are found and propagated.
"""

import torch
import numpy as np
from typing import List, Optional
import logging

from genetic.genome import Genome, random_genome, default_genome
from genetic.operators import (
    tournament_selection,
    elitist_selection,
    reproduce,
)
from rl_agent.agent import ChessAgent

logger = logging.getLogger("chess_rl")


class Population:
    """
    A collection of chess agents that evolve together over generations.

    Usage:
        pop = Population(size=16, device=device)
        pop.initialise()

        for generation in range(num_generations):
            fitness = evaluate_all(pop.agents)  # play games, compute ELO
            pop.evolve(fitness)                  # create next generation
    """

    def __init__(
        self,
        size: int = 16,
        survival_rate: float = 0.25,
        mutation_rate: float = 0.15,
        mutation_strength: float = 0.2,
        elitism: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            size:             Number of agents in the population.
            survival_rate:    Fraction of top agents that get to reproduce.
            mutation_rate:    Per-gene mutation probability.
            mutation_strength: Mutation noise magnitude.
            elitism:          If True, always carry the best agent forward unchanged.
            device:           Torch device for all agents.
        """
        self.size = size
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism = elitism
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agents: List[ChessAgent] = []
        self.genomes: List[Genome] = []
        self.generation = 0

    def initialise(self) -> None:
        """
        Create the initial population with random genomes.

        One agent uses the default genome (known-good hyperparameters)
        and the rest are fully randomised. This ensures we don't start
        from a completely random baseline.
        """
        logger.info(f"Initialising population of {self.size} agents...")

        self.genomes = [default_genome()] + [
            random_genome() for _ in range(self.size - 1)
        ]
        self.agents = [self._genome_to_agent(g) for g in self.genomes]

        logger.info(f"Population initialised. Genomes: {[str(g) for g in self.genomes[:3]]}...")

    def evolve(self, fitness_scores: List[float]) -> None:
        """
        Evolve the population: create the next generation from the current one.

        Steps:
            1. Preserve elite agents (if elitism=True)
            2. Use tournament selection to pick parents
            3. Apply crossover + mutation to produce offspring
            4. Create new agents from the offspring genomes

        Args:
            fitness_scores: ELO rating (or other fitness) for each agent.
                            Must be same length as self.agents.
        """
        assert len(fitness_scores) == len(self.agents), (
            f"Expected {len(self.agents)} fitness scores, got {len(fitness_scores)}"
        )

        logger.info(
            f"Evolving generation {self.generation} → {self.generation + 1}. "
            f"Best ELO: {max(fitness_scores):.0f}, Mean: {np.mean(fitness_scores):.0f}"
        )

        new_genomes: List[Genome] = []

        # ── Elitism: preserve the best agent unchanged ────────────────────
        num_elite = 1 if self.elitism else 0
        if num_elite > 0:
            elite_genomes = elitist_selection(self.genomes, fitness_scores, num_elite)
            new_genomes.extend(elite_genomes)
            logger.info(f"Elite agent preserved: {elite_genomes[0]}")

        # ── Fill the rest of the population with offspring ────────────────
        while len(new_genomes) < self.size:
            # Tournament selection for two parents
            parent_a = tournament_selection(self.genomes, fitness_scores)
            parent_b = tournament_selection(self.genomes, fitness_scores)

            # Crossover + mutation
            child = reproduce(
                parent_a,
                parent_b,
                mutation_rate=self.mutation_rate,
                mutation_strength=self.mutation_strength,
            )
            new_genomes.append(child)

        # ── Create new agents from evolved genomes ────────────────────────
        self.genomes = new_genomes
        self.agents = [self._genome_to_agent(g) for g in self.genomes]
        self.generation += 1

        logger.info(f"New generation {self.generation} created with {self.size} agents.")

    def _genome_to_agent(self, genome: Genome) -> ChessAgent:
        """
        Instantiate a ChessAgent from a Genome.

        Args:
            genome: The genome specifying this agent's hyperparameters.

        Returns:
            A fresh ChessAgent (untrained) with the specified hyperparameters.
        """
        params = genome.to_dict()
        return ChessAgent(
            num_residual_blocks=params["num_residual_blocks"],
            num_filters=params["num_filters"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            clip_epsilon=params["clip_epsilon"],
            entropy_coef=params["entropy_coef"],
            gae_lambda=params["gae_lambda"],
            device=self.device,
        )

    def best_agent(self) -> ChessAgent:
        """Return the agent with the highest ELO rating."""
        return max(self.agents, key=lambda a: a.elo)

    def elo_ratings(self) -> List[float]:
        """Return the ELO rating of every agent in the population."""
        return [agent.elo for agent in self.agents]

    def summary(self) -> str:
        """Return a human-readable summary of the current population."""
        elos = self.elo_ratings()
        return (
            f"Generation {self.generation} | "
            f"Pop: {self.size} | "
            f"Best ELO: {max(elos):.0f} | "
            f"Mean ELO: {np.mean(elos):.0f} | "
            f"Std: {np.std(elos):.0f}"
        )
