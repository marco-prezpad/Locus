"""
__main__.py

Minimal entry point used as a Phase 0 smoke test: loads the configuration,
initialises structured logging, loads the central cost scenario, and
prints a short summary. Intended to be removed once the real pipeline
replaces it in later phases.

Author: Marco Pérez Padilla
Date:   20-10-2026
"""

from locus.infrastructure.config import load_config, load_cost_scenario
from locus.infrastructure.logging import LoggerFactory


def main() -> None:
    """Load configuration and logging, then print a short summary
    confirming that the project boots end to end.
    """
    config = load_config()
    LoggerFactory.configure(config.app.log_level)
    log = LoggerFactory.get("main")
    central_costs = load_cost_scenario("central")

    log.info(
        "app_started",
        extra={
            "event": "app_started",
            "environment": config.app.environment,
            "run_id": "phase0-smoke-test",
        },
    )
    print(f"locus v0.1.0 — {config.app.environment}")
    print(f"Loaded config: {len(config.model_dump())} sections")
    print(f"Default horizon: {config.model.default_horizon_hours}h")
    print(f"Earliest execution: {config.signal_execution.earliest_execution_time}")
    print(f"Impact k (central): {central_costs['impact_k']}")


if __name__ == "__main__":
    main()
