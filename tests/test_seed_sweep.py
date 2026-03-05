import unittest
from pathlib import Path

from graphrag_lab.configs.loader import load_config
from graphrag_lab.runners.pipeline import run_seed_sweep


class SeedSweepTest(unittest.TestCase):
    def test_seed_sweep_writes_aggregate_report(self) -> None:
        config = load_config("local-debug", config_root=Path.cwd() / "configs")
        report = run_seed_sweep(config, [7, 11])

        self.assertEqual(report["aggregate"]["num_runs"], 2)
        self.assertIn("mean_avg_score", report["aggregate"])
        self.assertTrue((Path.cwd() / "artifacts" / "seed_sweep_local-debug.json").exists())


if __name__ == "__main__":
    unittest.main()
