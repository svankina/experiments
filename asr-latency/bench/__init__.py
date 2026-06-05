"""Shared ASR latency benchmark harness.

The same package is mounted into every engine container so that timing and result
recording are identical across engines. Engine-specific code lives in `bench.engines`.
"""
