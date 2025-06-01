import hydra
import logging
from omegaconf import OmegaConf
from models import build_model
from datasets import build_data_module
import pytorch_lightning as pl
import os

# Move temporary compilation files to another drive
os.environ["TORCH_EXTENSIONS_DIR"] = "D:/torch_extensions"
os.environ["TORCH_HOME"] = "D:/torch_cache"
os.environ["TMPDIR"] = "D:/tmp"

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="PVQE_S")
def my_hydra_app(cfg):
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Wassup")
    datamodule = build_data_module(cfg.datasets)
    # datamodule.setup()
    # train_dataloader = datamodule.train_dataloader()
    model = build_model(cfg.model)
    trainer = pl.Trainer(accelerator="gpu", min_epochs=1, max_epochs=1000)
    trainer.fit(model, datamodule)
    # for idx, batch in enumerate(train_dataloader):
    #     enrl, mic, farend_lpb, target = batch["data"]
    #     enrl_length, mic_length, farend_lpb_length, target_length = batch["length"]
    #     enh = model(enrl, enrl_length, mic, mic_length, farend_lpb)
    #     print(mic.shape)
    #     print(enh.shape)
    # print("Okay")

if __name__ == "__main__":
    my_hydra_app()