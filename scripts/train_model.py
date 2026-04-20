import lightning as L

import click

@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=True, file_okay=True, readable=True))
@click.option("--seed", type=int, default=42)
def main(config_path, seed):
  L.seed_everything(seed)

  # trainer = L.Trainer()
  # trainer.fit(
  #   model=model,
  #   datamodule=data_module,
  #   ckpt_path=ckpt_path
  # )


if __name__ == "__main__":
  main()