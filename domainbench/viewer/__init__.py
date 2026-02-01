"""
DomainBench Result Viewer - A local web interface for visualizing benchmark results
"""

from .app import create_app, run_viewer

__all__ = ["create_app", "run_viewer"]
