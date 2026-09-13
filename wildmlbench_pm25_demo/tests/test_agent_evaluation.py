import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_agent_output import evaluate, validate_predictions
from prepare_aide_workspace import validate_workspace


class AgentEvaluationTests(unittest.TestCase):
    def write(self, directory, name, text):
        path = Path(directory) / name
        path.write_text(text, encoding='utf-8')
        return path

    def test_workspace_allowlist_and_hidden_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('train.csv', 'test_features.csv', 'TASK.md'):
                self.write(directory, name, 'x\n1\n')
            validate_workspace(Path(directory))
            self.write(directory, 'test_labels.csv', 'row_id,target_pm25_next_hour\na,1\n')
            with self.assertRaises(ValueError):
                validate_workspace(Path(directory))

    def test_prediction_validation_rejects_missing_duplicate_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            features = self.write(directory, 'features.csv', 'row_id,x\na,1\nb,2\n')
            for rows in ('a,1\n', 'a,1\na,2\n', 'a,nan\nb,2\n'):
                predictions = self.write(directory, 'predictions.csv', 'row_id,prediction\n' + rows)
                with self.assertRaises(ValueError):
                    validate_predictions(predictions, features)

    def test_evaluator_aligns_by_row_id_and_calculates_rmse(self):
        with tempfile.TemporaryDirectory() as directory:
            features = self.write(directory, 'features.csv', 'row_id,x\na,1\nb,2\n')
            predictions = self.write(directory, 'predictions.csv', 'row_id,prediction\nb,5\na,1\n')
            labels = self.write(directory, 'labels.csv', 'row_id,target_pm25_next_hour\na,1\nb,3\n')
            validate_predictions(predictions, features)
            self.assertAlmostEqual(evaluate(predictions, labels), 2 ** 0.5)

    def test_docker_configuration_mounts_only_agent_inputs(self):
        compose = (Path(__file__).resolve().parents[1] / 'docker-compose.yml').read_text()
        dockerfile = (Path(__file__).resolve().parents[1] / 'docker' / 'entrypoint.sh').read_text()
        self.assertIn('./aide_task:/workspace:ro', compose)
        self.assertNotIn('test_labels.csv', compose)
        self.assertNotIn('wildmlbench_pm25_demo:/', compose)
        self.assertIn("allowed = {'train.csv', 'test_features.csv', 'TASK.md'}", dockerfile)


if __name__ == '__main__':
    unittest.main()
