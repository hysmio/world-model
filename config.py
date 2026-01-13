from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # image / patches
    frame_size: tuple[int, int] = (64, 64)
    patch_size: int = 8
    max_frames: int = 16

    # video tokenizer
    vt_d_model: int = 256
    vt_layers: int = 4
    vt_heads: int = 4
    vt_kq_size: int = 32
    vt_d_ff: int = 512
    vt_dropout: float = 0.1
    vt_num_codes: int = 512
    vt_latent_dim: int = 32

    # latent action model
    lam_d_model: int = 128
    lam_layers: int = 2
    lam_heads: int = 4
    lam_kq_size: int = 16
    lam_d_ff: int = 256
    lam_dropout: float = 0.1
    lam_num_actions: int = 8
    lam_action_dim: int = 16

    # dynamics model
    dyn_d_model: int = 256
    dyn_layers: int = 6
    dyn_heads: int = 4
    dyn_kq_size: int = 32
    dyn_d_ff: int = 512
    dyn_dropout: float = 0.1

    @property
    def num_patches(self) -> int:
        h, w = self.frame_size
        return (h // self.patch_size) * (w // self.patch_size)


@dataclass
class TrainConfig:
    # data
    data_paths: list[str] = field(default_factory=list)
    sequence_length: int = 8
    stride: int = 1

    # training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    # training steps per stage
    vt_steps: int = 10000
    lam_steps: int = 5000
    dyn_steps: int = 20000

    # logging
    log_every: int = 100

    # wandb
    wandb_project: str = "genie"
    wandb_name: str | None = None

    # checkpoints
    checkpoint_dir: str = "checkpoints"
