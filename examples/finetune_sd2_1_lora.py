from pathlib import Path

# TODO: AWARE OF RANK 0 LOG
import lightning as pl
import safetensors.torch as st
from lightning import Trainer, Callback
from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    ModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import Logger
from torch.utils.data import Dataset

from petorch import AdapterAPI
from petorch import logger
from petorch.experiments.callbacks.debugger import ClearmlImageDebugger
from petorch.experiments.checkpoints import ModelCheckpoint, DefaultCheckpointIO
from petorch.experiments.loggers import ClearmlLogger
from petorch.modules.diffusions.stable_diffusion import (
    StableDiffusionModule, StableDiffusionDataModule, SDSample,
)
from petorch.prebuilt.configs import LoraConfig

model_id = "stabilityai/stable-diffusion-2-1"

def get_trainer(
    dirpath: str,
    checkpoints_path: str = None,
    version: int | None = None,
    *,
    max_steps=3000,
    max_epochs=None,
    save_every_n_epochs=None,
    save_every_n_train_steps=None,
    save_top_k=1,
    accumulate_grad_batches=4,
    precision="bf16-true",
    addition_callbacks: list[Callback] = None,
    addition_loggers: list[Logger] = None,
    debug: bool = False,
    debug_downscale: float = 0.2,
    **kwargs,
):
    """

    Args:
        addition_loggers:
        version:
        dirpath:
        checkpoints_path:
        max_steps: Number of doing optimizer.step().
        max_epochs:
        save_every_n_epochs:
        save_every_n_train_steps: Number of doing optimizer.step().
        save_top_k:
        precision:
        accumulate_grad_batches:
        addition_callbacks:
        debug:
        debug_downscale:
        **kwargs:

    Returns:

    """
    checkpoints_path = checkpoints_path or f"{dirpath}/checkpoints"
    if version is None:
        version = 0
        while True:
            if not Path(checkpoints_path).joinpath(f"version_{version}").exists():
                break
            version += 1
    assert isinstance(version, int) and version >= 0
    checkpoints_path = (
        Path(checkpoints_path).joinpath(f"version_{version}").resolve().as_posix()
    )
    dirpath = Path(dirpath).resolve().as_posix()
    logger.info(f"Dir path: {dirpath}")
    logger.info(f"Checkpoint path: {checkpoints_path}")

    # Default callbacks
    callbacks: list[pl.Callback] = []
    callbacks.extend(
        [
            # Save last checkpoint
            # see `ModelCheckpoint._save_none_monitor_checkpoint`, it will call `_save_none_monitor_checkpoint` if monitor
            # is not provide, so although we not use save_last, it still save the last each epoch.
            # Not set save_weights_only, save for everything.
            ModelCheckpoint(
                dirpath=checkpoints_path,
                filename="{epoch}-{step}",
                monitor="step",  # For saving top k after every epoch. Compare by `step`.
                mode="max",
                save_top_k=save_top_k,
                every_n_epochs=save_every_n_epochs,
                every_n_train_steps=save_every_n_train_steps,
                save_on_train_epoch_end=True,
                verbose=debug,
                module_state_dict_extractor=AdapterAPI.get_adapter_state_dict,
                file_extension=".safetensors",
            ),
            TQDMProgressBar(leave=True),
            ModelSummary(max_depth=1),
        ]
    )
    if addition_callbacks:
        callbacks.extend(addition_callbacks)
    loggers: list[Logger] = []
    if addition_loggers:
        loggers.extend(addition_loggers)

    if debug:
        assert 0 <= debug_downscale <= 1
        kwargs["limit_train_batches"] = (
            kwargs.get("limit_train_batches") or debug_downscale
        )
        kwargs["limit_val_batches"] = kwargs.get("limit_val_batches") or debug_downscale
        max_steps = debug_downscale * max_steps
        accumulate_grad_batches = 1
        # callbacks.append(DebugCallback(10, verbose=False))
    # else:
        # callbacks.append(DebugCallback(20, verbose=False))

    trainer = Trainer(
        default_root_dir=dirpath,
        callbacks=callbacks,
        logger=loggers,
        precision=precision,
        max_steps=max_steps,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
        plugins=DefaultCheckpointIO(),
        # accelerator="cpu",
        # fast_dev_run=10,
        **kwargs,
    )
    return trainer


def get_sd_module(
    adt_config: LoraConfig = None, adt_ckpt: str = None, **module_kwargs
) -> StableDiffusionModule:
    logger.debug(f"Load model from: {model_id}")
    module = StableDiffusionModule(model_id=model_id, **module_kwargs)
    if adt_config:
        logger.debug(f"Add adapter from config: {adt_config}")
        module.requires_grad_(False)
        AdapterAPI.add_adapter(module, config, activate=True)
        if adt_ckpt:
            logger.debug(f"Load Adapter checkpoint from: {adt_ckpt}")
            adt_state_dict = st.load_file(adt_ckpt)
            AdapterAPI.load_adapter_state_dict(module, adt_state_dict, strict_load=True)
    n, t = 0, 0
    for name, param in module.named_parameters():
        t += param.numel()
        if param.requires_grad:
            assert "lora" in name
            n += param.numel()
    logger.debug(f"Trainable params={n:,} | Total params={t:,} | Ratio={n / t:.4f}")
    return module

class NarutoBlipDataset(Dataset):
    def __init__(self):
        from datasets import load_dataset, Dataset as ArrDataset
        self.dataset: ArrDataset = load_dataset("lambdalabs/naruto-blip-captions", split="train")

    def __getitem__(self, item):
        sample = self.dataset[item]
        return SDSample(
            image=sample["image"],
            text = sample["text"]
        )

    def __len__(self)->int:
        return len(self.dataset)


if __name__ == "__main__":
    PROJECT_NAME = "sd_2_1"
    storage_path = "/home/hieuhieuhieu/petorch/storages/"
    adt_ckpt = None

    # 1. Prepare model
    # Add lora to unet only
    config = LoraConfig(
        adapter_name="default", rank=16, alpha=16, fqname_filter=lambda _n: "unet" in _n
    )
    module = get_sd_module(config, adt_ckpt)

    # 2. Prepare data
    data_module = StableDiffusionDataModule(
        NarutoBlipDataset,
        batch_size=4,
        num_workers=32,
        train_ratio=0.95
    )

    # 3. Prepare trainer
    pl_trainer = get_trainer(
        storage_path + PROJECT_NAME,
        max_steps=5000,
        # save_every_n_train_steps=10,  # change 5 or 10 for debug, train batch step, not optimization step
        save_every_n_epochs=4,
        save_top_k=3,
        accumulate_grad_batches=4,
        addition_loggers=[
            ClearmlLogger(
                log_model_checkpoint=True,
                checkpoint_path="checkpoints",
                project_name=PROJECT_NAME,
                task_name="finetune_naruto_blip_lora_r16",
                output_uri="https://files.clear.ml"
            )
        ],
        addition_callbacks=[
            LearningRateMonitor(),
            ClearmlImageDebugger(
                "A girl with blonde hair and blue eyes.", every_n_epochs=2
            ),
        ],
        log_every_n_steps=10,  # batch steps (training_step), not optimization step.
        # debug=True,
        # debug_downscale=0.1
    )
    pl_trainer.fit(module, datamodule=data_module)
