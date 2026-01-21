"""
Entry point for running domainbench as a module.

Usage: python -m domainbench [command] [options]
"""

from domainbench.cli import app

if __name__ == "__main__":
    app()
