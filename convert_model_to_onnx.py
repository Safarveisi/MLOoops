import torch
import hydra
import logging

from omegaconf.omegaconf import OmegaConf

from model import ColaModel
from data import DataModule

logger = logging.getLogger(__name__)

@hydra.main(config_path="./configs", config_name="config", version_base="1.3")
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint.ckpt"
    logger.info(f"Loading pre-trained model from: {model_path}")
    cola_model = ColaModel.load_from_checkpoint(model_path)
    data_model = DataModule(
        cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length
    )
    data_model.prepare_data()
    data_model.setup()
    # Get the first batch
    input_batch = next(iter(data_model.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        cola_model,
        (
            input_sample["input_ids"].to("cuda:0"),
            input_sample["attention_mask"].to("cuda:0"),
        ), 
        f"{root_dir}/models/model.onnx",
        export_params=True,
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )

if __name__ == "__main__":
    convert_model()