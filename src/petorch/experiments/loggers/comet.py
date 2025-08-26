import pprint
from typing import (Any, Literal, Optional, cast, TYPE_CHECKING, )
from uuid import uuid4

import lightning.pytorch.loggers as pl_loggers
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from petorch import logger
from petorch.utilities import b64encode

if TYPE_CHECKING:
    from comet_ml import (APIExperiment,
    )


class CometLogger(pl_loggers.CometLogger):

    def __init__(
        self,
        log_model_checkpoint: bool,
        checkpoint_path: str = "checkpoints",  # Will be concat with dirpath_id of checkpoint model.
        encode_dirpath: bool = True,
        *,
        # Lightning CometLogger kwargs.
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        experiment_key: Optional[str] = None,
        mode: Optional[Literal["get_or_create", "get", "create"]] = None,
        online: Optional[bool] = None,
        prefix: Optional[str] = None,
        name: str | None = None,  # experiment name
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key,
            workspace=workspace,
            project=project,
            experiment_key=experiment_key,
            mode=mode,
            online=online,
            prefix=prefix,
            name=name,
            **kwargs,
        )
        self._log_model_checkpoints = log_model_checkpoint
        assert checkpoint_path
        self._model_name = checkpoint_path
        self._encode_dirpath = encode_dirpath

        self._api_expr = None

    @property
    def api_experiment(self) -> APIExperiment:
        if self._api_expr is None:
            from comet_ml import API

            comet_expr = self.experiment
            self._api_expr = API().get(
                comet_expr.workspace,
                comet_expr.project_name,
                experiment=comet_expr.get_key(),
            )
        return self._api_expr

    def _get_model_path(self, dirpath: str) -> str:
        if self._encode_dirpath:
            dirpath = b64encode(dirpath)
        return f"{self._model_name}/{dirpath}"

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        # TODO: Current reupload all checkpoints every time be invocated.
        #  See `alternative` part in https://github.com/Lightning-AI/pytorch-lightning/issues/16770
        #  Can be optimized by storing some metadata, and check it to avoid reupload the same checkpoint.

        if not self._log_model_checkpoints:
            return
        comet_path = self._get_model_path(checkpoint_callback.dirpath)
        comet_expr = self.experiment

        def _log_model(local_model_path: str) -> dict:
            return comet_expr.log_model(comet_path, local_model_path, overwrite=True)

        uid = uuid4()  # For task debug.
        cn = f"{self.__class__.__name__}-ID={uid}"
        # flushed =[]
        logger.info(
            f"{cn}: Uploading checkpoints...\n"
            f"workspace=`{comet_expr.workspace}` - project name=`{comet_expr.project_name}` - model path=`{comet_path}`"
        )
        uploaded_files = dict()
        """Id of files that just upload in this method. Keys are assetId, values are filename (excluding dir)."""

        # log best model path
        if paths := getattr(checkpoint_callback, "best_k_models", None):
            paths = cast(dict[str, torch.Tensor], paths)
            for path, score in paths.items():
                file = path.rsplit("/", maxsplit=1)[-1]

                assert file not in uploaded_files.values()
                logger.debug(
                    f"Uploading `best_model_path` checkpoint with score={score.item()}: {path}"
                )
                aid = _log_model(path)["assetId"]
                uploaded_files[aid] = file

        if path := getattr(checkpoint_callback, "last_model_path", None):
            file = path.rsplit("/", maxsplit=1)[-1]
            if file in uploaded_files.values():
                assert path not in checkpoint_callback.best_k_models
                logger.warning(
                    f"`last_model_path` checkpoint name `{file}` already in hold_files (be one of the best k files)."
                    f" Add postfix `_last` in filename."
                )
                f, extension = file.rsplit(".", 1)
                f += "_last"
                file = f"{f}.{extension}"
                assert file not in uploaded_files.values()
            logger.debug(f"Uploading `last_model_path` checkpoint: {path}")
            aid = _log_model(path)["assetId"]
            uploaded_files[aid] = file

        # None when no running experiment, not error.
        if not (f := comet_expr.flush()) and f is not None:
            logger.error(f"{cn}: Upload checkpoint fails...")
        else:
            logger.info(
                f"{cn}: Uploaded checkpoints successfully: \n{pprint.pformat(uploaded_files)}"
            )

        # remove old models logged to experiment if they are not part of best k models at this point
        # Note that there's a case that model files are the same in Comet, but ID are different, so I use ID as key here.
        api_expr = self.api_experiment

        all_model_files = {
            model["assetId"]: model["fileName"]
            for model in api_expr.get_model_asset_list(comet_path)
        }
        logger.debug(
            f"{cn}: ALl model checkpoints after uploading: \n"
            f"{pprint.pformat(all_model_files)}\n"
            f"Should hold files: \n"
            f"{pprint.pformat(uploaded_files)}"
        )
        assert len(all_model_files) >= len(uploaded_files)

        # Delete all keys not in hold_files. Or if duplicate, delete the one not have ID in uploaded_files.
        del_ids = {}  # assetID: filename
        remain_files = {}  # filename: assetID
        for aid, filename in all_model_files.items():
            if filename not in uploaded_files.values():
                del_ids[aid] = filename
                continue

            # Not allow upload files have the name but differnent IDs, Except the ID just upload.
            if filename in remain_files:  # Duplicate
                if (rid := remain_files[filename]) not in uploaded_files.keys():
                    assert aid in uploaded_files
                    del_ids[rid] = filename
                    remain_files[filename] = aid
                else:
                    assert aid not in del_ids
                    assert aid not in uploaded_files
                    del_ids[aid] = filename
            else:
                remain_files[filename] = aid
        if del_ids:
            logger.info(f"Deleting old checkpoints ... \n" f"{pprint.pformat(del_ids)}")
            for del_id in del_ids.keys():
                api_expr.delete_asset(del_id)
            if not (f := comet_expr.flush()) and f is not None:
                logger.error(f"{cn}: Delete checkpoints fails...")
            else:
                logger.info(f"{cn}: Deleted checkpoints successfully. ")

        logger.info(f"{cn}: End uploading {"~"*30}")
