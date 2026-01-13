import argparse
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx
from flax.nnx import Module as nn_Module
from jax import numpy as jnp
from pathlib import Path

# alias for model imports
nn = nnx

from config import ModelConfig, TrainConfig
from data import (
    VideoDataset,
    MultiVideoDataset,
    VPTDataset,
    VPTStreamingDataset,
    create_dataset,
)
from model.tokenizer import VTEncoder, VTDecoder
from model.lam import LatentActionModel
from model.dynamics import DynamicsModel


def create_models(cfg: ModelConfig, rngs: nn.Rngs):
    encoder = VTEncoder(
        d_model=cfg.vt_d_model,
        patch_size=cfg.patch_size,
        num_patches=cfg.num_patches,
        max_frames=cfg.max_frames,
        st_blocks=cfg.vt_layers,
        num_heads=cfg.vt_heads,
        kq_size=cfg.vt_kq_size,
        d_ff=cfg.vt_d_ff,
        dropout=cfg.vt_dropout,
        num_codes=cfg.vt_num_codes,
        latent_dim=cfg.vt_latent_dim,
        rngs=rngs,
    )

    decoder = VTDecoder(
        d_model=cfg.vt_d_model,
        latent_dim=cfg.vt_latent_dim,
        patch_size=cfg.patch_size,
        num_patches=cfg.num_patches,
        max_frames=cfg.max_frames,
        st_blocks=cfg.vt_layers,
        num_heads=cfg.vt_heads,
        kq_size=cfg.vt_kq_size,
        d_ff=cfg.vt_d_ff,
        dropout=cfg.vt_dropout,
        rngs=rngs,
    )

    lam = LatentActionModel(
        d_model=cfg.lam_d_model,
        latent_dim=cfg.vt_latent_dim,
        action_dim=cfg.lam_action_dim,
        num_actions=cfg.lam_num_actions,
        num_patches=cfg.num_patches,
        max_frames=cfg.max_frames,
        num_layers=cfg.lam_layers,
        num_heads=cfg.lam_heads,
        kq_size=cfg.lam_kq_size,
        d_ff=cfg.lam_d_ff,
        dropout=cfg.lam_dropout,
        rngs=rngs,
    )

    dynamics = DynamicsModel(
        d_model=cfg.dyn_d_model,
        latent_dim=cfg.vt_latent_dim,
        action_dim=cfg.lam_action_dim,
        num_codes=cfg.vt_num_codes,
        num_patches=cfg.num_patches,
        num_layers=cfg.dyn_layers,
        num_heads=cfg.dyn_heads,
        kq_size=cfg.dyn_kq_size,
        d_ff=cfg.dyn_d_ff,
        dropout=cfg.dyn_dropout,
        rngs=rngs,
    )

    return encoder, decoder, lam, dynamics


def create_optimizer(cfg: TrainConfig, num_steps: int):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        decay_steps=num_steps,
        end_value=cfg.learning_rate * 0.1,
    )
    return optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay)


def train_stage1(
    encoder,
    decoder,
    dataset,
    cfg: ModelConfig,
    train_cfg: TrainConfig,
    rng: np.random.Generator,
):
    print("=== Stage 1: Video Tokenizer ===")

    opt = create_optimizer(train_cfg, train_cfg.vt_steps)
    optimizer = nnx.Optimizer(encoder, opt)
    # need separate optimizer state for decoder
    opt_dec = create_optimizer(train_cfg, train_cfg.vt_steps)
    optimizer_dec = nnx.Optimizer(decoder, opt_dec)

    for step in range(train_cfg.vt_steps):
        batch = dataset.sample_batch(train_cfg.batch_size, rng)
        x = jnp.array(batch)

        def loss_fn(encoder, decoder):
            z_q, vq_loss, _ = encoder(x, deterministic=False)
            h, w = cfg.frame_size
            x_recon = decoder(z_q, img_height=h, img_width=w, deterministic=False)
            recon_loss = jnp.mean((x - x_recon) ** 2)
            return recon_loss + vq_loss, (recon_loss, vq_loss)

        (total_loss, (recon_loss, vq_loss)), (grads_enc, grads_dec) = (
            nnx.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(encoder, decoder)
        )

        optimizer.update(grads_enc)
        optimizer_dec.update(grads_dec)

        metrics = {
            "vt/total_loss": float(total_loss),
            "vt/recon_loss": float(recon_loss),
            "vt/vq_loss": float(vq_loss),
            "step": step,
        }

        if step % train_cfg.log_every == 0:
            wandb.log(metrics, step=step)
            print(f"[VT] step {step}: recon={recon_loss:.4f}, vq={vq_loss:.4f}")

    return encoder, decoder


def train_stage2(
    encoder, lam, dataset, train_cfg: TrainConfig, rng: np.random.Generator
):
    print("=== Stage 2: Latent Action Model ===")

    opt = create_optimizer(train_cfg, train_cfg.lam_steps)
    optimizer = nnx.Optimizer(lam, opt)

    for step in range(train_cfg.lam_steps):
        batch = dataset.sample_batch(train_cfg.batch_size, rng)
        x = jnp.array(batch)

        def loss_fn(lam):
            z_q, _, _ = encoder(x, deterministic=True)
            z_q = jax.lax.stop_gradient(z_q)
            _, action_loss, _ = lam(z_q, deterministic=False)
            return action_loss

        loss, grad = nnx.value_and_grad(loss_fn)(lam)
        optimizer.update(grad)

        if step % train_cfg.log_every == 0:
            wandb.log(
                {"lam/vq_loss": float(loss), "step": step},
                step=train_cfg.vt_steps + step,
            )
            print(f"[LAM] step {step}: loss={loss:.4f}")

    return lam


def train_stage3(
    encoder,
    lam,
    dynamics,
    dataset,
    cfg: ModelConfig,
    train_cfg: TrainConfig,
    rng: np.random.Generator,
):
    print("=== Stage 3: Dynamics Model ===")

    opt = create_optimizer(train_cfg, train_cfg.dyn_steps)
    optimizer = nnx.Optimizer(dynamics, opt)

    for step in range(train_cfg.dyn_steps):
        batch = dataset.sample_batch(train_cfg.batch_size, rng)
        x = jnp.array(batch)

        def loss_fn(dynamics):
            z_q, _, indices = encoder(x, deterministic=True)
            z_q = jax.lax.stop_gradient(z_q)

            actions, _, _ = lam(z_q, deterministic=True)
            actions = jax.lax.stop_gradient(actions)

            B, T, P, _ = z_q.shape
            indices = indices.reshape(B, T, P)
            total_loss = 0.0

            for t in range(T - 1):
                z_t = z_q[:, t, :, :]
                action_t = actions[:, t, :]
                target = indices[:, t + 1, :]
                loss = dynamics.loss(z_t, action_t, target, deterministic=False)
                total_loss = total_loss + loss

            return total_loss / (T - 1)

        loss, grad = nnx.value_and_grad(loss_fn)(dynamics)
        optimizer.update(grad)

        base_step = train_cfg.vt_steps + train_cfg.lam_steps
        if step % train_cfg.log_every == 0:
            wandb.log({"dyn/ce_loss": float(loss), "step": step}, step=base_step + step)
            print(f"[DYN] step {step}: loss={loss:.4f}")

    return dynamics


def save_checkpoint(path: Path, encoder, decoder, lam, dynamics, step: int):
    path.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()

    # save each model's state
    _, encoder_state = nnx.split(encoder)
    _, decoder_state = nnx.split(decoder)
    _, lam_state = nnx.split(lam)
    _, dynamics_state = nnx.split(dynamics)

    state = {
        "encoder": encoder_state,
        "decoder": decoder_state,
        "lam": lam_state,
        "dynamics": dynamics_state,
        "step": step,
    }

    checkpointer.save(path / f"step_{step}", state)
    print(f"Saved checkpoint at step {step}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="paths to video files or frame folders",
    )
    parser.add_argument(
        "--frame-size", type=int, default=64, help="frame size (square)"
    )
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8, help="sequence length")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vt-steps", type=int, default=10000)
    parser.add_argument("--lam-steps", type=int, default=5000)
    parser.add_argument("--dyn-steps", type=int, default=20000)
    parser.add_argument("--wandb-project", type=str, default="genie")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)

    # VPT dataset options
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="auto",
        choices=["auto", "video", "frames", "vpt", "vpt_streaming"],
        help="Dataset type (auto-detected by default)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10,
        help="Number of videos to cache in memory (VPT dataset)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to use (for debugging)",
    )

    args = parser.parse_args()

    # configs
    model_cfg = ModelConfig(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
    )

    train_cfg = TrainConfig(
        data_paths=args.data,
        sequence_length=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        vt_steps=args.vt_steps,
        lam_steps=args.lam_steps,
        dyn_steps=args.dyn_steps,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        checkpoint_dir=args.checkpoint_dir,
    )

    # init wandb
    wandb.init(
        project=train_cfg.wandb_project,
        name=train_cfg.wandb_name,
        config={
            "model": model_cfg.__dict__,
            "train": train_cfg.__dict__,
        },
    )

    # load data
    print(f"Loading data from {train_cfg.data_paths}")

    # Use create_dataset factory for single path with auto-detection
    if len(train_cfg.data_paths) == 1:
        dataset = create_dataset(
            train_cfg.data_paths[0],
            model_cfg.frame_size,
            train_cfg.sequence_length,
            train_cfg.stride,
            dataset_type=args.dataset_type,
            cache_size=args.cache_size,
            max_videos=args.max_videos,
        )
    else:
        # Multiple paths: use MultiVideoDataset for non-VPT data
        if args.dataset_type in ("vpt", "vpt_streaming"):
            raise ValueError("VPT dataset types only support single directory path")
        dataset = MultiVideoDataset(
            [Path(p) for p in train_cfg.data_paths],
            model_cfg.frame_size,
            train_cfg.sequence_length,
            train_cfg.stride,
        )

    # Print dataset info
    if hasattr(dataset, "get_video_info"):
        info = dataset.get_video_info()
        print(f"Dataset info: {info}")
    else:
        print(f"Dataset: {len(dataset)} sequences")

    # create models
    rngs = nn.Rngs(args.seed)
    rng = np.random.default_rng(args.seed)

    encoder, decoder, lam, dynamics = create_models(model_cfg, rngs)

    # training stages
    encoder, decoder = train_stage1(
        encoder, decoder, dataset, model_cfg, train_cfg, rng
    )
    lam = train_stage2(encoder, lam, dataset, train_cfg, rng)
    dynamics = train_stage3(encoder, lam, dynamics, dataset, model_cfg, train_cfg, rng)

    # save final checkpoint
    total_steps = train_cfg.vt_steps + train_cfg.lam_steps + train_cfg.dyn_steps
    save_checkpoint(
        Path(train_cfg.checkpoint_dir), encoder, decoder, lam, dynamics, total_steps
    )

    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
