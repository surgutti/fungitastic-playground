import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
import fiddle as fdl
import lightning as L
from fiddle.codegen import codegen
from fiddle.printing import as_dict_flattened

import wandb
from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.utils.config import get_wandb_config, parse_fiddle_config
from src.utils.wandb import WandbEpochAxisCallback

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger

    from src.config.schemas import ExperimentConfig

@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, file_okay=True), required=False)
@click.option("--resume_run_name", type=str, default=None)
@click.option("--no_wandb", is_flag=True, default=False)
@click.option("--seed", type=int, default=42)
def main(config_path, resume_run_name, no_wandb, seed):
    L.seed_everything(seed)

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

    partial_checkpoint_callback = built_cfg.training_cfg.checkpoint_callback
    checkpoint_callback = partial_checkpoint_callback(dirpath=log_dir) if partial_checkpoint_callback is not None else None
    callbacks = built_cfg.training_cfg.callbacks + ([checkpoint_callback] if checkpoint_callback is not None else [])

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
        wandb_logger.experiment.config.update({"seed": seed}, allow_val_change=True)

        if config_path is not None and resume_run_name is None:
            wandb_logger.experiment.config.update(
                as_dict_flattened(cfg)
            )
            generated = codegen.codegen_dot_syntax(cfg)
            code_str = "\n".join(generated.lines())
            artifact = wandb.Artifact(name=f"{run_name}_config", type="config")
            with artifact.new_file("config.py", mode="w") as file:
                file.write(code_str)

            wandb_logger.experiment.log_artifact(artifact)

    data_module: L.LightningDataModule = built_cfg.data_module

    trainer = L.Trainer(
        log_every_n_steps=50,
        logger=logger,
        max_epochs=built_cfg.training_cfg.max_epochs,
        default_root_dir=log_dir,
        callbacks=callbacks,
    )
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
