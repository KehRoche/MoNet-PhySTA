import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch.set_num_threads(3)

from src.engines.monet_engine import MONET_Engine
from src.models.monet import MoNet
from src.utils.args import get_public_config
from src.utils.dataloader import (
    get_dataset_info,
    load_adj_from_numpy,
    load_adj_from_pickle,
    load_dataset,
)
from src.utils.logging import get_logger
from src.utils.metrics import masked_mae


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config(config_path=None):
    parser = get_public_config()
    parser.add_argument("--config", type=str, default=None, help="Path to the YAML config file.")
    parser.add_argument("--cl_epoch", type=int, default=3)
    parser.add_argument("--warm_epoch", type=int, default=30)
    parser.add_argument("--tpd", type=int, default=96)

    parser.add_argument("--lrate", type=float, default=2e-3)
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--clip_grad_value", type=float, default=5)

    parser.add_argument("--mask_ratio", type=float, default=0)
    parser.add_argument("--dataset", type=str, default="PEMS-BAY")
    parser.add_argument("--emd_dim", type=int, default=4, help="Embedding dimension.")
    parser.add_argument("--gfno_hidden", type=int, default=8, help="Hidden width of the graph FNO block.")
    parser.add_argument(
        "--energy_splits",
        nargs=2,
        type=float,
        default=[0.3, 0.7],
        help="Two spectral energy thresholds, for example: 0.7 0.95.",
    )
    parser.add_argument("--topk_edges", type=int, default=3, help="Number of top-k graph edges.")
    parser.add_argument("--ecc_layers", type=int, default=1, help="Number of ECC graph convolution layers.")

    args = parser.parse_args()
    config_file = Path(args.config or config_path or Path(__file__).with_name("config.yaml"))
    if not config_file.is_absolute():
        config_file = (REPO_ROOT / config_file).resolve()

    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["config1"]

    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config[key] = value

    keys_to_log = ["emd_dim", "dataset", "gfno_hidden", "energy_splits", "mask_ratio", "topk_edges", "ecc_layers"]
    log_dir = REPO_ROOT / "eval" / str(config["model_name"]) / str(config["dataset"])
    logger = get_logger(str(log_dir), __name__, f"record_s{config['seed']}.log")
    logger.info("Param: %s", {k: config[k] for k in keys_to_log if k in config})

    return config, str(log_dir), logger


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.normal_(m.bias, mean=0, std=0.01)


def build_engine(config, dataloader, scaler, log_dir, logger):
    model = MoNet(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        model_config=config,
    )
    device = torch.device(config["device"])
    cl_step = config["cl_epoch"] * dataloader["train_loader"].num_batch
    warm_step = config["warm_epoch"] * dataloader["train_loader"].num_batch
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lrate"], weight_decay=config["wdecay"], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[1, 38, 46, 54, 62, 70, 80],
        gamma=0.5,
    )

    return MONET_Engine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=masked_mae,
        lrate=config["lrate"],
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=config["clip_grad_value"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        log_dir=log_dir,
        logger=logger,
        seed=config["seed"],
        cl_step=cl_step,
        warm_step=warm_step,
        horizon=config["horizon"],
        mask_ratio=config["mask_ratio"],
    )


def prepare_data(config, logger):
    device = torch.device(config["device"])
    data_path, adj_path, _ = get_dataset_info(config["dataset"])
    if config["dataset"] == "PEMS-BAY":
        adj = load_adj_from_pickle(adj_path)[2]
    else:
        adj = load_adj_from_numpy(adj_path)

    config["adj"] = torch.tensor(adj).to(device).float()
    return load_dataset(data_path, config, logger)


def objective(trial):
    try:
        import swanlab
    except ImportError as exc:
        raise ImportError("Install optional dependencies with `pip install optuna swanlab` for tuning.") from exc

    config, log_dir, logger = get_config()
    hype_config = {
        "dataset": trial.suggest_categorical("dataset", ["PEMS-BAY", "SD", "KnowAir", "BJAir"]),
    }
    run = swanlab.init(project="Optuna_Tuning", experiment_name=f"Trial_{trial.number}", config=hype_config)
    config.update({k: v for k, v in hype_config.items() if k in config})

    dataloader, scaler = prepare_data(config, logger)
    engine = build_engine(config, dataloader, scaler, log_dir, logger)
    loss = engine.train(run) if config["mode"] == "train" else engine.evaluate(config["mode"], run)
    run.log({"loss": loss})
    swanlab.finish()
    torch.cuda.empty_cache()
    return loss


def main():
    config, log_dir, logger = get_config()
    set_seed(config["seed"])
    dataloader, scaler = prepare_data(config, logger)
    engine = build_engine(config, dataloader, scaler, log_dir, logger)

    if config["mode"] == "train":
        engine.train()
    else:
        engine.evaluate(config["mode"])


if __name__ == "__main__":
    main()
