from __future__ import annotations

import argparse
import json

from tornado_ml.config import ProjectConfig
from tornado_ml.experiment_runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tornado ML experiment.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    result = ExperimentRunner(config).run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
