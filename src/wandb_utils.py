"""
Optional Weights & Biases logging.

Weights & Biases is **optional** and disabled by default, so the pipeline runs
offline without a W&B account. To enable it, set environment variables before
running:

    export USE_WANDB=true
    export WANDB_ENTITY=<your-wandb-entity>
    # optional: export WANDB_OFFLINE=true   # log to a local ./wandb/ dir only

When disabled, `init_run(...)` returns a no-op run whose `.log()`,
`.config.update()`, `.config[...]`, `.name` and `.finish()` behave sensibly but
do nothing, so the surrounding code does not need to change.
"""

import os


def wandb_enabled():
    """True if W&B logging has been explicitly enabled via USE_WANDB."""
    return os.environ.get("USE_WANDB", "false").lower() in ("1", "true", "yes")


class _NoOpConfig(dict):
    """Dict that also supports the wandb ``config.update(...)`` signature."""

    def update(self, d=None, **kwargs):
        if d:
            super().update(d)

    def __getitem__(self, key):
        # Return None for missing keys instead of raising, matching how the
        # calling code tolerates absent config entries.
        return self.get(key)


class _NoOpRun:
    """Minimal stand-in for a wandb run when logging is disabled."""

    def __init__(self, name=None):
        self.name = name
        self.config = _NoOpConfig()

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


def init_run(name=None, project=None, **kwargs):
    """Return a real W&B run if ``USE_WANDB`` is set, otherwise a no-op run.

    The entity is read from the ``WANDB_ENTITY`` environment variable so that no
    personal account details are hard-coded in the source.
    """
    if not wandb_enabled():
        return _NoOpRun(name=name)
    import wandb
    if os.environ.get("WANDB_OFFLINE", "false").lower() in ("1", "true", "yes"):
        os.environ["WANDB_MODE"] = "offline"
    return wandb.init(entity=os.environ.get("WANDB_ENTITY"), project=project, name=name, **kwargs)
