"""railrl.encoders — Stage 4 model input pipeline + encoders (spec 03).

Modules:
    input_pipeline  — snapshot row → encoded tensors (§2). The numpy core
                      (`encode_snapshot`) is torch-free and unit-testable; the
                      PyG HeteroData / Dataset wrappers lazy-import torch.
    hgt / sequence / fusion — encoders (§3-§5)  [built next]
"""
