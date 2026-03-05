import unittest
from pathlib import Path

from graphrag_lab.configs.loader import load_config
from graphrag_lab.runners.pipeline import run_toy_pipeline


class ToyPipelineTest(unittest.TestCase):
    def test_toy_pipeline_runs_and_writes_report(self) -> None:
        config = load_config("local-debug", config_root=Path.cwd() / "configs")
        report = run_toy_pipeline(config)

        self.assertGreaterEqual(report["summary"]["num_samples"], 1)
        self.assertGreaterEqual(report["summary"]["avg_score"], 0.0)
        self.assertLessEqual(report["summary"]["avg_score"], 1.0)

        report_path = config.runtime.output_dir / config.runtime.mode / "report.json"
        self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
