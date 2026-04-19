from __future__ import annotations

import argparse
import json
import logging

from tornado_ml.config import ProjectConfig
from tornado_ml.experiment_runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tornado ML experiment.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = ProjectConfig.from_yaml(args.config)
    result = ExperimentRunner(config).run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
