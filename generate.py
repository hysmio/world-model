import argparse
import jax
import numpy as np
import cv2
from flax import nnx
from jax import numpy as jnp
from pathlib import Path

from config import ModelConfig
from train import create_models


def generate_frames(
    encoder,
    decoder,
    lam,
    dynamics,
    initial_frame: jnp.ndarray,
    num_frames: int,
    action_indices: list[int] | None = None,
    cfg: ModelConfig = None,
    temperature: float = 1.0,
    rng_key: jax.Array = None,
) -> list[jnp.ndarray]:
    # initial_frame: (1, 1, C, H, W)
    frames = [initial_frame[0, 0]]  # store as (C, H, W)

    # encode initial frame
    z_q, _, _ = encoder(initial_frame, deterministic=True)
    z_t = z_q[0, 0, :, :]  # (P, latent_dim)
    z_t = z_t[None, :, :]  # (1, P, latent_dim)

    for i in range(num_frames - 1):
        # get action
        if action_indices is not None:
            action_idx = action_indices[i % len(action_indices)]
            action = lam.action_vq.lookup(jnp.array([action_idx]))  # (1, action_dim)
        else:
            # random action
            action_idx = np.random.randint(0, cfg.lam_num_actions)
            action = lam.action_vq.lookup(jnp.array([action_idx]))

        # predict next frame tokens
        if rng_key is not None:
            rng_key, subkey = jax.random.split(rng_key)
            pred_indices = dynamics.predict_tokens(
                z_t, action, deterministic=True, temperature=temperature, sample=True, rng=subkey
            )
        else:
            pred_indices = dynamics.predict_tokens(z_t, action, deterministic=True)

        # lookup embeddings
        z_t = encoder.vq.lookup(pred_indices)  # (1, P, latent_dim)

        # decode to pixels
        z_decode = z_t[None, :, :, :]  # (1, 1, P, latent_dim)
        h, w = cfg.frame_size
        frame = decoder(z_decode, img_height=h, img_width=w, deterministic=True)
        frames.append(frame[0, 0])  # (C, H, W)

    return frames


def frames_to_video(frames: list[jnp.ndarray], output_path: str, fps: int = 10):
    # frames: list of (C, H, W) in [-1, 1]
    h, w = frames[0].shape[1], frames[0].shape[2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        # (C, H, W) -> (H, W, C), [-1, 1] -> [0, 255]
        frame = np.array(frame)
        frame = np.transpose(frame, (1, 2, 0))
        frame = ((frame + 1) * 127.5).clip(0, 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--initial-frame", type=str, required=True, help="path to initial frame image")
    parser.add_argument("--output", type=str, default="generated.mp4")
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--actions", type=int, nargs="*", help="action indices to use (cycles)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = ModelConfig(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
    )

    # create models
    rngs = nnx.Rngs(args.seed)
    encoder, decoder, lam, dynamics = create_models(cfg, rngs)

    # TODO: load checkpoint weights
    # for now just use random weights for testing

    # load initial frame
    img = cv2.imread(args.initial_frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, cfg.frame_size)
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.transpose(img, (2, 0, 1))  # (C, H, W)
    initial = jnp.array(img)[None, None, :, :, :]  # (1, 1, C, H, W)

    print(f"Generating {args.num_frames} frames...")
    rng_key = jax.random.key(args.seed)

    frames = generate_frames(
        encoder, decoder, lam, dynamics,
        initial, args.num_frames,
        action_indices=args.actions,
        cfg=cfg,
        temperature=args.temperature,
        rng_key=rng_key,
    )

    frames_to_video(frames, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
