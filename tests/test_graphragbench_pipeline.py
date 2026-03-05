import unittest
from pathlib import Path

from graphrag_lab.configs.loader import load_config
from graphrag_lab.runners.pipeline import run_toy_pipeline


class GraphRAGBenchPipelineTest(unittest.TestCase):
    def test_graphragbench_profile_runs(self) -> None:
        config = load_config("graphragbench-debug", config_root=Path.cwd() / "configs")
        report = run_toy_pipeline(config)

        self.assertEqual(report["summary"]["mode"], "graphragbench-debug")
        self.assertEqual(report["summary"]["num_samples"], 2)


if __name__ == "__main__":
    unittest.main()
