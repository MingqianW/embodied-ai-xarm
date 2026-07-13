import dataclasses
import functools
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
try:
    import pandas as pd
except ImportError:  # pragma: no cover - only needed for parquet tracing.
    pd = None

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


DEBUG_EVERY_STEPS = 1
DEBUG_PRINT_FIRST_STEPS = 10
DEBUG_TRACE_SAMPLE_LOSS_THRESHOLD = float(os.environ.get("OPENPI_DEBUG_TRACE_SAMPLE_LOSS_THRESHOLD", "10"))
DEBUG_TRACE_LOG_PATH = Path(os.environ.get("OPENPI_DEBUG_TRACE_LOG", "debug_high_loss_traces.jsonl")).expanduser()
PARQUET_TRACER = None


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def _tree_get(obj: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _to_float(x: Any) -> float:
    return float(np.asarray(x))


def _safe_np(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def append_trace_record(record: dict[str, Any]) -> None:
    DEBUG_TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_TRACE_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def _as_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=np.float32)
    except Exception:
        return None


def _find_action_stats(stats: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates: list[tuple[str, dict[str, Any]]] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            lower = prefix.lower()
            has_stats = any(k in value for k in ("mean", "std", "q01", "q99", "min", "max"))
            if has_stats and ("action" in lower or prefix in {"actions", "action"}):
                candidates.append((prefix, value))
            for k, v in value.items():
                walk(f"{prefix}/{k}" if prefix else k, v)

    walk("", stats)
    if not candidates:
        raise ValueError(f"Could not find action stats in norm stats keys: {list(stats.keys())}")
    candidates.sort(key=lambda item: (item[0] != "actions", len(item[0])))
    return candidates[0]


def _align_stat(stat: np.ndarray | None, action_dim: int, default: float) -> np.ndarray:
    if stat is None:
        return np.full(action_dim, default, dtype=np.float32)
    stat = stat.reshape(-1).astype(np.float32)
    if stat.shape[0] == action_dim:
        return stat
    out = np.full(action_dim, default, dtype=np.float32)
    out[: min(action_dim, stat.shape[0])] = stat[:action_dim]
    return out


class ParquetActionTracer:
    """Matches transformed OpenPI action vectors back to likely parquet rows."""

    def __init__(self, dataset_root: Path, norm_stats_path: Path, action_dim: int = 32, delta_dims: int = 6):
        if pd is None:
            raise RuntimeError("pandas is required for parquet tracing")
        self.dataset_root = dataset_root
        self.norm_stats_path = norm_stats_path
        self.action_dim = action_dim
        self.delta_dims = delta_dims
        stats = json.loads(norm_stats_path.read_text())
        self.action_stats_key, self.action_stats = _find_action_stats(stats)
        self.entries: list[dict[str, Any]] = []
        self._build()

    def _pad(self, action7: np.ndarray) -> np.ndarray:
        out = np.zeros(self.action_dim, dtype=np.float32)
        out[: action7.shape[0]] = action7
        return out

    def _normalization_variants(self, action: np.ndarray) -> dict[str, np.ndarray]:
        variants = {"raw_transformed": action}
        mean = _align_stat(_as_float_array(self.action_stats.get("mean")), self.action_dim, 0.0)
        std = _align_stat(_as_float_array(self.action_stats.get("std")), self.action_dim, 1.0)
        variants["zscore_mean_std"] = (action - mean) / np.where(np.abs(std) < 1e-12, np.nan, std)

        q01 = _as_float_array(self.action_stats.get("q01"))
        q99 = _as_float_array(self.action_stats.get("q99"))
        if q01 is not None and q99 is not None:
            q01 = _align_stat(q01, self.action_dim, 0.0)
            q99 = _align_stat(q99, self.action_dim, 1.0)
            variants["bounds_q01_q99"] = 2.0 * (action - q01) / np.where(np.abs(q99 - q01) < 1e-12, np.nan, q99 - q01) - 1.0

        min_v = _as_float_array(self.action_stats.get("min"))
        max_v = _as_float_array(self.action_stats.get("max"))
        if min_v is not None and max_v is not None:
            min_v = _align_stat(min_v, self.action_dim, 0.0)
            max_v = _align_stat(max_v, self.action_dim, 1.0)
            variants["bounds_min_max"] = 2.0 * (action - min_v) / np.where(np.abs(max_v - min_v) < 1e-12, np.nan, max_v - min_v) - 1.0
        return variants

    def _build(self) -> None:
        parquets = sorted(self.dataset_root.rglob("*.parquet"))
        for pf in parquets:
            df = pd.read_parquet(pf, columns=["state", "actions", "episode_index", "frame_index", "task_index"])
            for row_i, row in df.iterrows():
                state = np.asarray(row["state"], dtype=np.float32).reshape(-1)
                action = np.asarray(row["actions"], dtype=np.float32).reshape(-1)
                transformed7 = action.copy()
                transformed7[: self.delta_dims] = action[: self.delta_dims] - state[: self.delta_dims]
                transformed = self._pad(transformed7)
                for norm_name, normed in self._normalization_variants(transformed).items():
                    if np.isfinite(normed).all():
                        self.entries.append(
                            {
                                "norm_name": norm_name,
                                "vector": normed.astype(np.float32),
                                "file": str(pf),
                                "row": int(row_i),
                                "episode_index": int(row["episode_index"]),
                                "frame_index": int(row["frame_index"]),
                                "task_index": int(row["task_index"]),
                                "state": state,
                                "action": action,
                                "transformed7": transformed7,
                            }
                        )

    def match(self, target: np.ndarray, top: int = 5) -> list[dict[str, Any]]:
        target = target.reshape(-1).astype(np.float32)
        results = []
        for entry in self.entries:
            vec = entry["vector"]
            dims = min(len(target), len(vec))
            # First 7 dims carry the robot action. Remaining dims are padded.
            dims = min(dims, 7)
            dist = float(np.linalg.norm(vec[:dims] - target[:dims]))
            max_abs_err = float(np.max(np.abs(vec[:dims] - target[:dims])))
            results.append((dist, max_abs_err, entry))
        results.sort(key=lambda item: item[0])
        out = []
        for dist, max_abs_err, entry in results[:top]:
            out.append({**entry, "distance": dist, "max_abs_err": max_abs_err})
        return out


def maybe_init_parquet_tracer(config: _config.TrainConfig) -> ParquetActionTracer | None:
    dataset_root = os.environ.get("OPENPI_DEBUG_LEROBOT_ROOT")
    norm_stats = os.environ.get("OPENPI_DEBUG_NORM_STATS")

    if dataset_root is None:
        dataset_root = "/home/mw89/.cache/huggingface/lerobot/local/xarm_pickup_v260624"
    if norm_stats is None:
        norm_stats = "assets/pi05_xarm_full_finetune/local/xarm_pickup_v260624/norm_stats.json"

    dataset_root_path = Path(dataset_root).expanduser()
    norm_stats_path = Path(norm_stats).expanduser()
    if not dataset_root_path.exists():
        logging.warning("Parquet tracer disabled; dataset root does not exist: %s", dataset_root_path)
        return None
    if not norm_stats_path.exists():
        logging.warning("Parquet tracer disabled; norm stats do not exist: %s", norm_stats_path)
        return None
    try:
        tracer = ParquetActionTracer(dataset_root_path, norm_stats_path, action_dim=config.model.action_dim)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Parquet tracer disabled after initialization error: %s", exc)
        return None
    logging.info(
        "Parquet tracer ready: entries=%d dataset=%s norm_stats=%s action_stats=%s",
        len(tracer.entries),
        dataset_root_path,
        norm_stats_path,
        tracer.action_stats_key,
    )
    return tracer


def _array_stats(prefix: str, x: Any, out: dict[str, float]) -> None:
    arr = _safe_np(x).astype(np.float32)
    out[f"{prefix}/finite"] = float(np.isfinite(arr).all())
    out[f"{prefix}/absmax"] = float(np.nanmax(np.abs(arr)))
    out[f"{prefix}/min"] = float(np.nanmin(arr))
    out[f"{prefix}/max"] = float(np.nanmax(arr))
    out[f"{prefix}/mean"] = float(np.nanmean(arr))
    out[f"{prefix}/std"] = float(np.nanstd(arr))

    if arr.ndim >= 1 and arr.shape[-1] <= 64:
        flat = arr.reshape(-1, arr.shape[-1])
        for i in range(flat.shape[-1]):
            out[f"{prefix}/dim_{i}_min"] = float(np.nanmin(flat[:, i]))
            out[f"{prefix}/dim_{i}_max"] = float(np.nanmax(flat[:, i]))
            out[f"{prefix}/dim_{i}_mean"] = float(np.nanmean(flat[:, i]))
            out[f"{prefix}/dim_{i}_std"] = float(np.nanstd(flat[:, i]))


def debug_host_batch(step: int, observation: _model.Observation, actions: _model.Actions, info: dict[str, Any]) -> None:
    """Print only samples whose loss is high enough to investigate."""
    sample_loss = info.get("debug/sample_loss")
    if sample_loss is None:
        return

    sample_loss_np = _safe_np(sample_loss).astype(np.float32)
    bad_indices = np.where(sample_loss_np >= DEBUG_TRACE_SAMPLE_LOSS_THRESHOLD)[0]
    if len(bad_indices) == 0:
        return

    actions_np = _safe_np(actions).astype(np.float32)
    scalar_info = {
        k: _to_float(v)
        for k, v in info.items()
        if k != "debug/sample_loss" and np.asarray(jax.device_get(v)).ndim == 0
    }

    print("\n" + "=" * 100)
    print(f"HIGH LOSS SAMPLES DETECTED | step={step} threshold={DEBUG_TRACE_SAMPLE_LOSS_THRESHOLD}")
    print("=" * 100)
    for key in sorted(scalar_info):
        if key == "loss" or key.startswith("debug/chunked_loss") or key.startswith("debug/loss_horizon"):
            print(f"{key}: {scalar_info[key]}")
    print(f"trace_log: {DEBUG_TRACE_LOG_PATH}")

    for rank, batch_i in enumerate(bad_indices[np.argsort(sample_loss_np[bad_indices])[::-1]]):
        sample_actions = actions_np[batch_i]
        first_action = sample_actions[0] if sample_actions.ndim == 2 else sample_actions
        print("\nBAD SAMPLE")
        print(f"rank={rank} batch_i={int(batch_i)} sample_loss={float(sample_loss_np[batch_i]):.6f}")
        print(f"action_absmax={float(np.nanmax(np.abs(sample_actions))):.6f}")
        print(f"first_action={first_action}")
        print(f"sample_action_min={float(np.nanmin(sample_actions)):.6f}")
        print(f"sample_action_max={float(np.nanmax(sample_actions)):.6f}")

        trace_record: dict[str, Any] = {
            "step": int(step),
            "rank": int(rank),
            "batch_i": int(batch_i),
            "sample_loss": float(sample_loss_np[batch_i]),
            "action_absmax": float(np.nanmax(np.abs(sample_actions))),
            "sample_action_min": float(np.nanmin(sample_actions)),
            "sample_action_max": float(np.nanmax(sample_actions)),
            "first_action": first_action,
            "scalar_info": scalar_info,
            "matches": [],
        }

        if PARQUET_TRACER is not None:
            flat_abs_argmax = int(np.nanargmax(np.abs(sample_actions)))
            if sample_actions.ndim == 2:
                horizon_i, dim_i = np.unravel_index(flat_abs_argmax, sample_actions.shape)
                trace_action = sample_actions[horizon_i]
                print(f"trace_action_source=horizon_{int(horizon_i)} dim_{int(dim_i)}")
                trace_record["trace_horizon"] = int(horizon_i)
                trace_record["trace_dim"] = int(dim_i)
            else:
                dim_i = flat_abs_argmax
                trace_action = sample_actions
                print(f"trace_action_source=dim_{int(dim_i)}")
                trace_record["trace_horizon"] = None
                trace_record["trace_dim"] = int(dim_i)
            print(f"trace_action={trace_action}")
            trace_record["trace_action"] = trace_action
            print("PARQUET TRACE MATCHES:")
            matches = PARQUET_TRACER.match(trace_action, top=5)
            trace_record["matches"] = [
                {
                    "distance": match["distance"],
                    "max_abs_err": match["max_abs_err"],
                    "norm_name": match["norm_name"],
                    "file": match["file"],
                    "episode_index": match["episode_index"],
                    "frame_index": match["frame_index"],
                    "row": match["row"],
                    "task_index": match["task_index"],
                    "transformed7": match["transformed7"],
                    "state": match["state"],
                    "action": match["action"],
                }
                for match in matches
            ]
            for match in matches:
                print(
                    " "
                    f"distance={match['distance']:.6f} max_abs_err={match['max_abs_err']:.6f} "
                    f"norm={match['norm_name']} file={match['file']} "
                    f"episode={match['episode_index']} frame={match['frame_index']} "
                    f"row={match['row']} task={match['task_index']}"
                )
                print(f"   transformed7={match['transformed7']}")
                print(f"   state={match['state']}")
                print(f"   action={match['action']}")
        else:
            trace_record["trace_error"] = "PARQUET_TRACER is not initialized"

        append_trace_record(trace_record)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # One extra forward pass for debugging. This is intentionally in train_debug.py only.
    chunked_loss = model.compute_loss(train_rng, observation, actions, train=True)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    action0 = actions[:, 0, :] if actions.ndim == 3 else actions
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "debug/chunked_loss_absmax": jnp.nanmax(jnp.abs(chunked_loss)),
        "debug/chunked_loss_min": jnp.nanmin(chunked_loss),
        "debug/chunked_loss_max": jnp.nanmax(chunked_loss),
        "debug/chunked_loss_std": jnp.nanstd(chunked_loss),
        "debug/action_absmax": jnp.nanmax(jnp.abs(actions)),
        "debug/action_min": jnp.nanmin(actions),
        "debug/action_max": jnp.nanmax(actions),
        "debug/action_std": jnp.nanstd(actions),
        "debug/action_dim6_absmax": jnp.nanmax(jnp.abs(actions[..., 6])),
        "debug/action_dim6_min": jnp.nanmin(actions[..., 6]),
        "debug/action_dim6_max": jnp.nanmax(actions[..., 6]),
        "debug/action_dim6_std": jnp.nanstd(actions[..., 6]),
        "debug/first_action_absmax": jnp.nanmax(jnp.abs(action0)),
    }

    if chunked_loss.ndim >= 2:
        per_horizon = jnp.mean(chunked_loss, axis=0)
        info.update({f"debug/loss_horizon_{i}": per_horizon[i] for i in range(min(16, per_horizon.shape[0]))})
    if chunked_loss.ndim >= 1:
        per_sample = jnp.mean(chunked_loss.reshape((chunked_loss.shape[0], -1)), axis=1)
        info["debug/loss_sample_max"] = jnp.max(per_sample)
        info["debug/loss_sample_min"] = jnp.min(per_sample)
        info["debug/loss_sample_std"] = jnp.std(per_sample)
        info["debug/loss_sample_argmax"] = jnp.argmax(per_sample)
        info["debug/sample_loss"] = per_sample

    return new_state, info


def main(config: _config.TrainConfig):
    global PARQUET_TRACER
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info("Using debug training script. It runs an extra forward pass per step and is slower than scripts/train.py.")
    PARQUET_TRACER = maybe_init_parquet_tracer(config)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    debug_host_batch(0, batch[0], batch[1], {})

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        current_batch = batch
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, current_batch)
        infos.append(info)

        should_debug_print = step < DEBUG_PRINT_FIRST_STEPS or step % DEBUG_EVERY_STEPS == 0
        if should_debug_print:
            host_info = jax.device_get(info)
            debug_host_batch(step, current_batch[0], current_batch[1], host_info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
