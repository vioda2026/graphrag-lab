import unittest
from pathlib import Path

from graphrag_lab.benchmarks.graphragbench_adapter import GraphRAGBenchAdapter


class GraphRAGBenchAdapterTest(unittest.TestCase):
    def test_split_filtering(self) -> None:
        adapter = GraphRAGBenchAdapter(
            data_path=Path.cwd() / "data/graphragbench/sample.jsonl",
            split="test",
        )
        samples = list(adapter.samples())
        self.assertEqual(len(samples), 2)
        self.assertSetEqual({s.sample_id for s in samples}, {"gqb-2", "gqb-3"})

    def test_eval_f1(self) -> None:
        adapter = GraphRAGBenchAdapter(
            data_path=Path.cwd() / "data/graphragbench/sample.jsonl",
            split="test",
        )
        self.assertGreater(adapter.evaluate("Paris", "The capital is Paris."), 0.5)
        self.assertEqual(adapter.evaluate("Mars", "Jupiter"), 0.0)


if __name__ == "__main__":
    unittest.main()
