import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
import fiddle as fdl
import lightning as L
import torch
from fiddle.codegen import codegen
from fiddle.printing import as_dict_flattened
from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary

import wandb
from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.utils.config import get_wandb_config, parse_fiddle_config
from src.utils.wandb import WandbEpochAxisCallback

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger
    from src.config.schemas import ExperimentConfig


def _maybe_compile_backbone(model: L.LightningModule, mode: str) -> None:
    if not hasattr(model, "backbone"):
        warnings.warn("--compile_backbone was requested, but model has no `.backbone` attribute.")
        return

    try:
        model.backbone = torch.compile(model.backbone, mode=mode)
    except Exception as exc:  # pragma: no cover - best-effort speed option
        warnings.warn(f"torch.compile failed; continuing without compilation. Error: {exc}")


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, file_okay=True), required=False)
@click.option("--resume_run_name", type=str, default=None)
@click.option("--no_wandb", is_flag=True, default=False)
@click.option("--seed", type=int, default=42)
@click.option("--max_epochs", type=int, default=None, help="Override max_epochs from the config.")
@click.option("--precision", type=str, default="bf16-mixed", show_default=True)
@click.option("--accelerator", type=str, default="gpu", show_default=True)
@click.option("--devices", type=str, default="1", show_default=True)
@click.option("--accumulate_grad_batches", type=int, default=1, show_default=True)
@click.option("--gradient_clip_val", type=float, default=1.0, show_default=True)
@click.option("--num_workers_hint", type=int, default=None, help="Override datamodule.num_workers if present.")
@click.option("--fast_dev_run", type=int, default=0, help="Run N train/val batches for a smoke test.")
@click.option("--overfit_batches", type=float, default=0.0, help="Use e.g. 8 or 0.02 to debug overfitting.")
@click.option("--limit_train_batches", type=float, default=1.0)
@click.option("--limit_val_batches", type=float, default=1.0)
@click.option("--compile_backbone", is_flag=True, default=False)
@click.option("--compile_mode", type=str, default="default", show_default=True)
@click.option("--matmul_precision", type=str, default="high", show_default=True)
def main(
    config_path,
    resume_run_name,
    no_wandb,
    seed,
    max_epochs,
    precision,
    accelerator,
    devices,
    accumulate_grad_batches,
    gradient_clip_val,
    num_workers_hint,
    fast_dev_run,
    overfit_batches,
    limit_train_batches,
    limit_val_batches,
    compile_backbone,
    compile_mode,
    matmul_precision,
):
    """GPU-oriented training entry point.

    Examples:
      uv run python -m scripts.train_model_gpu src/config/resnet50_unet_512.py --no_wandb
      uv run python -m scripts.train_model_gpu src/config/resnet101_unet_512.py --accumulate_grad_batches 2
      uv run python -m scripts.train_model_gpu src/config/resnet50_unet_512.py --fast_dev_run 2 --no_wandb
      uv run python -m scripts.train_model_gpu src/config/resnet50_unet_512.py --overfit_batches 8 --max_epochs 30 --no_wandb
    """

    torch.set_float32_matmul_precision(matmul_precision)
    L.seed_everything(seed, workers=True)

    if resume_run_name is not None:
        cfg: fdl.Config[ExperimentConfig] = get_wandb_config(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{resume_run_name}")
        run_name = resume_run_name
    else:
        cfg: fdl.Config[ExperimentConfig] = parse_fiddle_config(config_path)
        run_name = cfg.name + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_dir = f"logs/{run_name}"

    ckpt_path = None
    if resume_run_name is not None:
        last_ckpt = Path(log_dir) / "last.ckpt"
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
        else:
            ckpts = sorted(Path(log_dir).glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            if ckpts:
                ckpt_path = str(ckpts[-1])
        if ckpt_path is None:
            warnings.warn(f"No checkpoint found in {log_dir}")

    built_cfg: ExperimentConfig = fdl.build(cfg)
    model: L.LightningModule = built_cfg.model

    if compile_backbone:
        _maybe_compile_backbone(model, compile_mode)

    data_module: L.LightningDataModule = built_cfg.data_module
    if num_workers_hint is not None and hasattr(data_module, "num_workers"):
        data_module.num_workers = num_workers_hint
        if hasattr(data_module, "persistent_workers"):
            data_module.persistent_workers = num_workers_hint > 0

    partial_checkpoint_callback = built_cfg.training_cfg.checkpoint_callback
    checkpoint_callback = partial_checkpoint_callback(dirpath=log_dir) if partial_checkpoint_callback is not None else None
    callbacks = list(built_cfg.training_cfg.callbacks)
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)
    callbacks.append(ModelSummary(max_depth=2))

    logger: list[Logger] = []
    partial_wandb_logger = built_cfg.training_cfg.wandb_logger
    if partial_wandb_logger is not None and not no_wandb:
        wandb_logger = partial_wandb_logger(
            id=run_name,
            name=run_name,
            resume="allow",
            tags=[f"seed: {seed}"],
        )
        logger.append(wandb_logger)
        callbacks.append(WandbEpochAxisCallback())
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        wandb_logger.experiment.config.update({"seed": seed}, allow_val_change=True)

        if config_path is not None and resume_run_name is None:
            wandb_logger.experiment.config.update(as_dict_flattened(cfg))
            generated = codegen.codegen_dot_syntax(cfg)
            code_str = "\n".join(generated.lines())
            artifact = wandb.Artifact(name=f"{run_name}_config", type="config")
            with artifact.new_file("config.py", mode="w") as file:
                file.write(code_str)
            wandb_logger.experiment.log_artifact(artifact)

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=int(devices) if devices.isdigit() else devices,
        precision=precision,
        benchmark=True,
        deterministic=False,
        log_every_n_steps=25,
        logger=logger,
        max_epochs=max_epochs if max_epochs is not None else built_cfg.training_cfg.max_epochs,
        default_root_dir=log_dir,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
        fast_dev_run=fast_dev_run if fast_dev_run > 0 else False,
        overfit_batches=overfit_batches,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=2,
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
