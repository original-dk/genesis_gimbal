# RSL-RL

This directory contains RSL-RL 6.0.0 source copied from the installed
`rsl-rl-lib` distribution. Upstream: https://github.com/leggedrobotics/rsl_rl.
License: BSD-3-Clause. License notices are preserved in `licenses/`.

The package ships with Genesis. Install its runtime dependencies with:

```bash
uv pip install -e ".[rl]"
```

Install the CUDA build of PyTorch appropriate for the training machine before
installing the extra. Changes to this directory are maintained in this repository.
