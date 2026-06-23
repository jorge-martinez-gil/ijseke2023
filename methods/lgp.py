# -*- coding: utf-8 -*-
"""Pure Python Linear Genetic Programming for semantic similarity aggregation.

This module provides a portable alternative to the legacy Java LGP project.
It follows the same dataset and CLI conventions as the other Python methods.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from utils import load_dataset, pearson_score, spearman_score


EPSILON = 1e-12
VALUE_LIMIT = 1e6
DEFAULT_CONSTANTS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -2.0)


def protected_division(x, y):
    """Divide x by y while preserving x where the denominator is zero."""
    return np.divide(x, y, out=np.copy(x), where=np.abs(y) > EPSILON)


def _safe_values(values):
    """Keep evolved register values finite and bounded."""
    finite = np.nan_to_num(values, nan=0.0, posinf=VALUE_LIMIT, neginf=-VALUE_LIMIT)
    return np.clip(finite, -VALUE_LIMIT, VALUE_LIMIT)


BINARY_OPERATORS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": protected_division,
}

UNARY_OPERATORS = {
    "sin": np.sin,
    "cos": np.cos,
}

OPERATOR_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "sin": "sin",
    "cos": "cos",
}

OPERATOR_NAMES = tuple(BINARY_OPERATORS) + tuple(UNARY_OPERATORS)


@dataclass(frozen=True)
class Instruction:
    """One LGP register-machine instruction."""

    target: int
    operator: str
    left: int
    right: int

    @property
    def arity(self):
        """Return the number of operands used by this instruction."""
        return 1 if self.operator in UNARY_OPERATORS else 2

    def execute(self, registers):
        """Apply this instruction in place to the register matrix."""
        if self.arity == 1:
            registers[self.target] = _safe_values(UNARY_OPERATORS[self.operator](registers[self.left]))
            return

        registers[self.target] = _safe_values(
            BINARY_OPERATORS[self.operator](registers[self.left], registers[self.right])
        )

    def to_expression(self, expressions):
        """Render this instruction using the current symbolic register values."""
        left = expressions[self.left]
        if self.arity == 1:
            return f"{OPERATOR_SYMBOLS[self.operator]}({left})"

        right = expressions[self.right]
        return f"({left} {OPERATOR_SYMBOLS[self.operator]} {right})"


@dataclass(frozen=True)
class LGPProgram:
    """A sequence of LGP instructions evaluated against shared registers."""

    instructions: Tuple[Instruction, ...]
    output_register: int = 0

    def predict(self, X, register_count):
        """Execute this program and return the output register values."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D feature matrix.")
        if register_count < X.shape[1]:
            raise ValueError("register_count must be at least the number of input features.")

        registers = np.zeros((register_count, X.shape[0]), dtype=float)
        registers[: X.shape[1], :] = X.T

        for index in range(X.shape[1], register_count):
            constant = DEFAULT_CONSTANTS[(index - X.shape[1]) % len(DEFAULT_CONSTANTS)]
            registers[index, :] = constant

        for instruction in self.instructions:
            instruction.execute(registers)

        return registers[self.output_register].copy()

    def to_lines(self, register_count, n_features):
        """Return a readable instruction listing for this program."""
        expressions = [f"x{i}" for i in range(n_features)]
        for index in range(n_features, register_count):
            constant = DEFAULT_CONSTANTS[(index - n_features) % len(DEFAULT_CONSTANTS)]
            expressions.append(f"{constant:g}")

        lines = []
        for index, instruction in enumerate(self.instructions):
            expression = instruction.to_expression(expressions)
            expressions[instruction.target] = expression
            lines.append(f"{index:03d}: r{instruction.target} = {expression}")

        lines.append(f"output: r{self.output_register} = {expressions[self.output_register]}")
        return lines


class LinearGPRegressor:
    """Small, deterministic Linear GP regressor for tabular similarity features."""

    def __init__(
        self,
        metric="pearson",
        population_size=100,
        generations=100,
        min_program_length=5,
        max_program_length=40,
        register_count=None,
        tournament_size=5,
        crossover_rate=0.5,
        mutation_rate=0.3,
        parsimony_coefficient=0.0,
        random_state=0,
        verbose=0,
    ):
        if metric not in {"pearson", "spearman"}:
            raise ValueError("metric must be 'pearson' or 'spearman'.")
        if population_size < 1:
            raise ValueError("population_size must be positive.")
        if min_program_length < 1:
            raise ValueError("min_program_length must be positive.")
        if max_program_length < min_program_length:
            raise ValueError("max_program_length must be greater than or equal to min_program_length.")
        if tournament_size < 1:
            raise ValueError("tournament_size must be positive.")

        self.metric = metric
        self.population_size = population_size
        self.generations = generations
        self.min_program_length = min_program_length
        self.max_program_length = max_program_length
        self.register_count = register_count
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state
        self.verbose = verbose

        self.best_program_ = None
        self.best_score_ = float("-inf")
        self.best_fitness_ = float("-inf")
        self.n_features_in_ = None
        self.register_count_ = None

    def fit(self, X, y):
        """Evolve a linear GP program on the given training data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be a 2-D feature matrix.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        self.n_features_in_ = X.shape[1]
        self.register_count_ = self.register_count or max(self.n_features_in_ * 3, self.n_features_in_ + 1)
        if self.register_count_ < self.n_features_in_:
            raise ValueError("register_count must be at least the number of input features.")

        rng = np.random.default_rng(self.random_state)
        population = [self._random_program(rng) for _ in range(self.population_size)]

        for generation in range(self.generations + 1):
            scored_population = self._score_population(population, X, y)
            self._remember_best(scored_population, X, y)

            if self.verbose and generation % self.verbose == 0:
                print(
                    f"Generation {generation}: "
                    f"best {self.metric}={self.best_score_:.6f}, fitness={self.best_fitness_:.6f}"
                )

            if generation == self.generations:
                break

            population = self._next_generation(scored_population, rng)

        return self

    def predict(self, X):
        """Predict target values with the best evolved program."""
        if self.best_program_ is None:
            raise ValueError("This LinearGPRegressor instance is not fitted yet.")
        return self.best_program_.predict(X, self.register_count_)

    def program_lines(self):
        """Return a readable listing of the best evolved program."""
        if self.best_program_ is None:
            raise ValueError("This LinearGPRegressor instance is not fitted yet.")
        return self.best_program_.to_lines(self.register_count_, self.n_features_in_)

    def _random_program(self, rng):
        length = int(rng.integers(self.min_program_length, self.max_program_length + 1))
        instructions = tuple(self._random_instruction(rng) for _ in range(length))
        return LGPProgram(instructions=instructions)

    def _random_instruction(self, rng):
        operator = str(rng.choice(OPERATOR_NAMES))
        return Instruction(
            target=int(rng.integers(0, self.register_count_)),
            operator=operator,
            left=int(rng.integers(0, self.register_count_)),
            right=int(rng.integers(0, self.register_count_)),
        )

    def _score_population(self, population, X, y):
        return [(self._fitness(program, X, y), program) for program in population]

    def _fitness(self, program, X, y):
        raw_score = self._score(program, X, y)
        if not np.isfinite(raw_score):
            return float("-inf")
        return raw_score - self.parsimony_coefficient * len(program.instructions)

    def _score(self, program, X, y):
        y_pred = program.predict(X, self.register_count_)
        score = pearson_score(y, y_pred) if self.metric == "pearson" else spearman_score(y, y_pred)
        return score if np.isfinite(score) else float("-inf")

    def _remember_best(self, scored_population, X, y):
        fitness, program = max(scored_population, key=lambda item: item[0])
        if fitness > self.best_fitness_:
            self.best_program_ = program
            self.best_fitness_ = fitness
            self.best_score_ = self._score(program, X, y)

    def _next_generation(self, scored_population, rng):
        scored_population = sorted(scored_population, key=lambda item: item[0], reverse=True)
        next_population = [scored_population[0][1]]

        while len(next_population) < self.population_size:
            if rng.random() < self.crossover_rate:
                parent_a = self._tournament(scored_population, rng)
                parent_b = self._tournament(scored_population, rng)
                child = self._crossover(parent_a, parent_b, rng)
            else:
                child = self._tournament(scored_population, rng)

            if rng.random() < self.mutation_rate:
                child = self._mutate(child, rng)

            next_population.append(child)

        return next_population

    def _tournament(self, scored_population, rng):
        size = min(self.tournament_size, len(scored_population))
        indices = rng.choice(len(scored_population), size=size, replace=False)
        contenders = [scored_population[index] for index in indices]
        return max(contenders, key=lambda item: item[0])[1]

    def _crossover(self, parent_a, parent_b, rng):
        instructions_a = parent_a.instructions
        instructions_b = parent_b.instructions
        cut_a = int(rng.integers(0, len(instructions_a) + 1))
        cut_b = int(rng.integers(0, len(instructions_b) + 1))
        child = instructions_a[:cut_a] + instructions_b[cut_b:]

        if len(child) > self.max_program_length:
            child = child[: self.max_program_length]
        if len(child) < self.min_program_length:
            extra = tuple(self._random_instruction(rng) for _ in range(self.min_program_length - len(child)))
            child = child + extra

        return LGPProgram(instructions=child)

    def _mutate(self, program, rng):
        instructions = list(program.instructions)
        action = str(rng.choice(("replace", "insert", "delete")))

        if action == "insert" and len(instructions) < self.max_program_length:
            index = int(rng.integers(0, len(instructions) + 1))
            instructions.insert(index, self._random_instruction(rng))
        elif action == "delete" and len(instructions) > self.min_program_length:
            index = int(rng.integers(0, len(instructions)))
            del instructions[index]
        else:
            index = int(rng.integers(0, len(instructions)))
            instructions[index] = self._random_instruction(rng)

        return LGPProgram(instructions=tuple(instructions))


def parse_args():
    """Parse command-line options for the pure Python LGP runner."""
    parser = argparse.ArgumentParser(description="Run pure Python Linear GP.")
    parser.add_argument(
        "--dataset",
        choices=["mc", "geresid"],
        default="mc",
        help="Dataset to use for training/validation splits.",
    )
    parser.add_argument(
        "--metric",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Metric optimized during evolution.",
    )
    parser.add_argument("--population-size", type=int, default=100, help="Number of programs per generation.")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations to evolve.")
    parser.add_argument("--min-program-length", type=int, default=5, help="Minimum number of instructions.")
    parser.add_argument("--max-program-length", type=int, default=40, help="Maximum number of instructions.")
    parser.add_argument("--register-count", type=int, default=None, help="Number of registers; defaults to 3x inputs.")
    parser.add_argument("--tournament-size", type=int, default=5, help="Tournament selection size.")
    parser.add_argument("--crossover-rate", type=float, default=0.5, help="Probability of one-point crossover.")
    parser.add_argument("--mutation-rate", type=float, default=0.3, help="Probability of program mutation.")
    parser.add_argument(
        "--parsimony-coefficient",
        type=float,
        default=0.0,
        help="Fitness penalty applied per instruction.",
    )
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Print progress every N generations; 0 disables progress output.",
    )
    return parser.parse_args()


def main():
    """Train pure Python LGP and report validation correlations."""
    import pandas as pd

    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    X_train, y_train = load_dataset(project_root, args.dataset, "training")
    X_test, y_test = load_dataset(project_root, args.dataset, "validation")

    model = LinearGPRegressor(
        metric=args.metric,
        population_size=args.population_size,
        generations=args.generations,
        min_program_length=args.min_program_length,
        max_program_length=args.max_program_length,
        register_count=args.register_count,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        parsimony_coefficient=args.parsimony_coefficient,
        random_state=args.random_state,
        verbose=args.verbose,
    ).fit(X_train, y_train)

    y_pred = model.predict(X_test)
    df_preds = pd.DataFrame({"Actual": y_test.squeeze(), "Predicted": y_pred.squeeze()})
    print(df_preds)

    print(f"Best training {args.metric}: {model.best_score_:.6f}")
    print("Best program:")
    print("\n".join(model.program_lines()))
    print(f"Pearson Correlation: {pearson_score(y_test, y_pred):.6f}")
    print(f"Spearman Correlation: {spearman_score(y_test, y_pred):.6f}")


if __name__ == "__main__":
    main()
