# RSL-RL

This directory contains RSL-RL 6.0.0 source copied from the installed
`rsl-rl-lib` distribution. Upstream: https://github.com/leggedrobotics/rsl_rl.
License: BSD-3-Clause. License notices are preserved in `licenses/`.

Run from the repository root to import this source directly:

```bash
PYTHONPATH=. python examples/locomotion/go2_train.py -e go2-velocity-standing
```

The environment must provide PyTorch >=2.6.0, tensordict >=0.7.0, numpy,
tensorboard, GitPython, onnx and onnxscript >=0.5.4. Install missing dependencies
with `uv pip install`. Select the CUDA build of PyTorch for the training machine.
Changes to this directory are maintained in this repository.
