"""
Processing pipeline for the AuthBench benchmark.

Modules:
- config: target distributions and defaults
- datasets: dataset readers and manifest parsing
- chunker/dirty/sampling: core processing steps
- build_benchmark: CLI orchestrating the pipeline
"""

__all__ = [
    "build_benchmark",
]
