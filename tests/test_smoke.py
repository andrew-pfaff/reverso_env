"""Smoke tests for the UCI HAR Reverso baseline integration."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reverso.data import ForecastAugmentationConfig, UCIHARReversoDataset
from reverso.models import ReversoConfig, build_model
from reverso.training import (
    TrainingConfig,
    build_dataloaders,
    evaluate,
    run_training,
    set_seed,
    train_one_epoch,
)
from reverso.training.metrics import ForecastMetricAccumulator


class ReversoSmokeTests(unittest.TestCase):
    def test_dataset_and_dataloader_shapes(self) -> None:
        config = TrainingConfig(
            task_name="all_to_acc",
            batch_size=4,
            train_max_windows=8,
            val_max_windows=4,
        )
        loaders = build_dataloaders(config)
        batch = next(iter(loaders["train"]))

        self.assertEqual(tuple(batch["context"].shape), (4, 384, 6))
        self.assertEqual(tuple(batch["target"].shape), (4, 64, 3))
        self.assertEqual(batch["context"].dtype, torch.float32)
        self.assertEqual(batch["target"].dtype, torch.float32)

    def test_model_forward_shape(self) -> None:
        dataset = UCIHARReversoDataset(split="val", task="all_to_all")
        sample = dataset[0]
        config = ReversoConfig.from_task(
            "all_to_all",
            context_len=384,
            pred_len=64,
            d_model=32,
            n_layers=2,
        )
        model = build_model(config)
        prediction = model(sample["context"].unsqueeze(0))
        self.assertEqual(tuple(prediction.shape), (1, 64, 6))

    def test_flash_fft_size_covers_full_convolution(self) -> None:
        config = ReversoConfig.from_task(
            "all_to_all",
            context_len=384,
            pred_len=64,
            d_model=32,
            n_layers=2,
            long_conv_kernel=384,
        )
        self.assertGreaterEqual(config.resolved_flash_fft_size, 384 + 384 - 1)

    def test_amp_requires_half_precision_dtype(self) -> None:
        config = TrainingConfig(amp=True, dtype="fp32")
        with self.assertRaises(ValueError):
            config.resolved_amp_dtype()

    def test_activity_subset_filters_to_stairs_only(self) -> None:
        dataset = UCIHARReversoDataset(
            split="val",
            task="all_to_acc",
            activity_subset="stairs_both",
        )
        labels = {int(dataset[index]["label"].item()) for index in range(min(len(dataset), 32))}
        self.assertTrue(labels.issubset({8, 9}))

    def test_dataset_augmentation_preserves_shapes(self) -> None:
        dataset = UCIHARReversoDataset(
            split="train",
            task="all_to_acc",
            activity_subset="stairs_both",
            augmentation=ForecastAugmentationConfig(
                downsample_prob=1.0,
                amplitude_mod_prob=1.0,
                censor_prob=1.0,
            ),
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["context"].shape), (384, 6))
        self.assertEqual(tuple(sample["target"].shape), (64, 3))
        self.assertTrue(torch.isfinite(sample["context"]).all())
        self.assertTrue(torch.isfinite(sample["target"]).all())

    def test_metric_accumulator_outputs_expected_keys(self) -> None:
        prediction = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0]]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([8], dtype=torch.int64)

        accumulator = ForecastMetricAccumulator(["acc_x", "gyr_x"])
        accumulator.update(prediction, target, labels)
        metrics = accumulator.compute()

        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertIn("mae_channel/acc_x", metrics)
        self.assertIn("mae_channel/gyr_x", metrics)
        self.assertIn("mae_activity/stairs_up", metrics)

    def test_one_optimizer_step_and_eval(self) -> None:
        set_seed(42)
        config = TrainingConfig(
            task_name="all_to_all",
            d_model=32,
            n_layers=2,
            batch_size=4,
            train_max_windows=8,
            val_max_windows=4,
            num_epochs=1,
            device="cpu",
        )
        loaders = build_dataloaders(config)
        model_config = config.make_model_config()
        model = build_model(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            device=torch.device("cpu"),
            target_channels=model_config.target_channels,
            augmentation_config=None,
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            device=torch.device("cpu"),
            target_channels=model_config.target_channels,
        )

        self.assertTrue(torch.isfinite(torch.tensor(train_metrics["loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(val_metrics["loss"])))
        self.assertIn("mae_channel/acc_x", train_metrics)
        self.assertIn("mae_modality/gyroscope", val_metrics)

    def test_training_run_writes_visualization_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                task_name="all_to_acc",
                activity_subset="stairs_both",
                d_model=32,
                n_layers=2,
                batch_size=4,
                num_epochs=1,
                train_max_windows=8,
                device="cpu",
                output_dir=temp_dir,
                downsample_prob=0.5,
                amplitude_mod_prob=0.5,
                censor_prob=0.5,
                mixup_prob=0.5,
                mixup_alpha=0.2,
            )
            result = run_training(config)
            artifact_paths = result["artifact_paths"]

            self.assertIsNotNone(artifact_paths)
            assert artifact_paths is not None
            for required_key in {
                "config",
                "history",
                "summary",
                "training_curves",
                "channel_mae",
                "activity_mae",
                "prediction_example",
                "prediction_example_stairs_up",
                "prediction_example_stairs_down",
                }:
                self.assertIn(required_key, artifact_paths)
                self.assertTrue(Path(artifact_paths[required_key]).is_file())
            self.assertTrue(Path(artifact_paths["checkpoint_final"]).is_file())
            self.assertTrue(Path(artifact_paths["checkpoint_best"]).is_file())

    def test_augmented_stairs_training_stays_finite(self) -> None:
        set_seed(42)
        config = TrainingConfig(
            task_name="all_to_acc",
            activity_subset="stairs_both",
            d_model=32,
            n_layers=2,
            batch_size=16,
            train_max_windows=128,
            val_max_windows=64,
            num_epochs=1,
            device="cpu",
            downsample_prob=0.3,
            amplitude_mod_prob=0.3,
            censor_prob=0.3,
            mixup_prob=0.3,
            mixup_alpha=0.2,
        )
        result = run_training(config)
        history = result["history"]
        self.assertEqual(len(history), 1)
        epoch_metrics = history[0]
        self.assertTrue(torch.isfinite(torch.tensor(epoch_metrics["train_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(epoch_metrics["val_loss"])))


if __name__ == "__main__":
    unittest.main()
