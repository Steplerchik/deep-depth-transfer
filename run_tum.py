import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.utilities.parsing import AttributeDict

from deep_depth_transfer.data import TumVideoDataModuleFactory
from deep_depth_transfer.models.factory import ModelFactory
from deep_depth_transfer.models.utils import load_undeepvo_checkpoint
from deep_depth_transfer.utils import TensorBoardLogger, MLFlowLogger, LoggerCollection

parser = ArgumentParser(description="Run deep depth transfer on TUM RGB-D")
parser.add_argument("--config", type=str, default="./config/multi_depth_model_tum.yaml")
parser.add_argument("--dataset", type=str, default="./datasets/tum_rgbd/rgbd_dataset_freiburg3_large_cabinet_validation")
parser.add_argument("--load_model", type=bool, default=False)
parser.add_argument("--model_checkpoint", type=str, default="./checkpoints/checkpoint_undeepvo.pth")
parser.add_argument("--experiment_name", type=str, default="TUM RGB-D")
parser = pl.Trainer.add_argparse_args(parser)
arguments = parser.parse_args()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://proxy2.cod.phystech.edu:10086/"
os.environ["AWS_ACCESS_KEY_ID"] = "depth"
os.environ["AWS_SECRET_ACCESS_KEY"] = "depth123"
mlflow_url = "http://proxy2.cod.phystech.edu:10085/"
logger = LoggerCollection(
    [TensorBoardLogger("lightning_logs"),
     # MLFlowLogger(experiment_name=arguments.experiment_name, tracking_uri=mlflow_url)
     ]
)

# Make trainer
trainer = pl.Trainer.from_argparse_args(arguments, logger=logger)

data_model_factory = TumVideoDataModuleFactory(arguments.dataset, use_poses=True)

# Load parameters
params = load_hparams_from_yaml(arguments.config)
params = AttributeDict(params)
print("Load model from params \n" + str(params))
data_model = data_model_factory.make_data_module_from_params(params)

# data_model._train_dataset.indices = data_model._train_dataset.indices[:200]
# data_model._test_dataset.indices = data_model._test_dataset.indices[:20]
# data_model._validation_dataset.indices = data_model._validation_dataset.indices[:20]

model = ModelFactory().make_mono_model(params, data_model.get_cameras_calibration())

if arguments.load_model:
    print("Load checkpoint")
    load_undeepvo_checkpoint(model, arguments.model_checkpoint)

print("Start training")
trainer.fit(model, data_model)
