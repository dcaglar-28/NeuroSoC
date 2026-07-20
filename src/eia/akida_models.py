"""Akida deployment & verification path (BrainChip MetaTF) — ECG (arrhythmia
+ MI + shockable-rhythm), heart sounds, and the synthetic CRM demo.

Deploy sibling of `rockpool_models.py` (which stays untouched). Where
`rockpool_models.py` maps a Rockpool LIF SNN onto Xylo, this module builds
small **quantized Conv2D CNNs** and runs them on **MetaTF**: `quantizeml`
(quantization/calibration) -> `cnn2snn` (Keras -> Akida conversion) ->
`akida.Model` (software simulator, no hardware needed). Model builders,
sharing everything downstream (`quantize_and_convert`/`verify_against_sim`/
`to_akida_input` are fully generic — no modality-specific shape assumptions):
  - `build_akida_model` (ECG-arrhythmia, and reused unchanged for the
    synthetic CRM demo and shockable-rhythm/VF-VT detection): a single-
    channel waveform reshaped to a (window, 1, 1) single-column "image" —
    Conv2D-over-time, one real axis.
  - `build_akida_heart_model` (heart sounds): the `signal_features`
    filterbank's `(n_features, n_subwindows)` map used AS a genuine 2-D
    image (bands x time), both axes real — see its docstring for why this
    needed a different first-layer shape than ECG's.
  - `build_akida_mi_model` (PTB-XL myocardial infarction — deepens ECG from
    arrhythmia to MI, docs/ptbxl_mi_results.md): 12-lead ECG as a `(leads,
    time)` map, the same "genuinely 2-D" class as heart sounds', not a
    reuse of `build_akida_model` — MI needs spatial lead information a
    single-column reshape would destroy.

WHY A CONV2D-OVER-TIME, NOT A NATIVE CONV1D: Akida 2.0's layer catalog (see
`docs/akida_ecg_results.md` Part 0) has no Conv1D — the closest fits are
`Conv2D`/`DepthwiseConv2D` (spatial) or the newer `BufferTempConv`/
`tenn_spatiotemporal` (genuinely temporal, TENN). This first slice uses the
simplest mapping (quantized Conv2D-over-time, treating the 1-D window as a
(window, 1, 1) single-column "image") and leaves TENN for later — TENN is the
temporal-native alternative worth trying if this slice's fidelity is poor.

CONFIRMED AKIDA v2 LAYER CONSTRAINTS (found empirically, converting a Conv2D
stack — not documented anywhere obvious, and NOT specific to 1-D-as-2-D):
  - The first (Input) conv layer's kernel AND stride must be SQUARE
    (`kernel_size[0] == kernel_size[1]`, `strides[0] == strides[1]`), even
    though our real width is 1 — pass e.g. `kernel_size=(7, 7)` with
    `padding="same"`, which zero-pads the phantom width dimension harmlessly.
  - EVERY conv layer's max-pooling must ALSO be square (`cnn2snn` raises
    "should have square pooling" otherwise) — this applies beyond just the
    input layer, contrary to what the first error message's wording implies.
  - `Conv2D -> GlobalAveragePooling2D -> ReLU` is an INVALID block pattern
    for `cnn2snn.convert`; the valid ordering is `Conv2D -> ReLU ->
    GlobalAveragePooling2D` (`akida_models.layer_blocks.conv_block`'s
    `post_relu_gap=True` flag gives this ordering — `post_relu_gap=False`,
    the default, gives the invalid one for a global-avg-pooled block).
  - `quantizeml`/`cnn2snn` require the **`tf_keras`** package (standalone
    Keras 2, NOT `tensorflow.keras`, which is Keras 3 under TF 2.19) for
    every layer, optimizer, and loss — mixing `tf.keras.*` objects into a
    `tf_keras`-built model raises "Could not interpret optimizer identifier".

Input convention: Akida's InputConv2D layer expects a **uint8** `(n, x, y, c)`
tensor (`x`=window/time, `y`=1, `c`=1 here) — `to_akida_input` converts a
real-valued `(n, window)` ECG batch (any scale) into this format via
per-window min-max normalization to [0, 255], the same spirit as image pixel
values. A `Rescaling(1/255)` layer (folded into the quantized graph, not a
separate preprocessing step) converts back to a working float range for the
float-training phase.

Two-stage training (mirrors "train off-device -> quantize -> verify" exactly,
`quantize_and_convert` doing the middle step):
  1. Train the float `tf_keras.Model` normally (supervised, class-weighted
     loss) via `.compile()`/`.fit()` — ordinary float training.
  2. `quantizeml.models.quantize(float_model, samples=calibration_batch, ...)`
     inserts fake-quantization ops and CALIBRATES ranges from the sample
     batch — despite "QAT" framing in BrainChip's own materials, this
     function's actual signature takes no labels, so by itself it is
     **post-training quantization + calibration, not label-driven QAT**.
     Genuine QAT is still available and confirmed working: the quantized
     model it returns is an ordinary differentiable `tf_keras.Model` (its
     `Quantized*` layers are real Keras layers with straight-through-
     estimator fake-quant, not opaque numpy ops), so calling
     `.compile()`/`.fit()` on it AGAIN with real labels (a short, low-LR
     fine-tune) genuinely back-props through the quantization and recovers
     accuracy — confirmed by a working `qmodel.fit()` call in this module's
     development. `quantize_and_convert`'s `qat_epochs` argument does this.
  3. `cnn2snn.convert(quantized_model)` -> `akida.Model`, ready to
     `.forward()`/`.predict()` on the **Software backend** (confirmed no
     hardware device required: `akida.devices()` returns `[]` in this
     container, and the model summary prints a "(Software)" sequence).

FIDELITY CAVEAT (Part 0's load-bearing question — see docs/akida_ecg_results.md
for the full write-up): unlike SynSense's explicit claim that XyloSim's traces
match silicon exactly, a multi-source check of BrainChip's own documentation
(brainchip.com, doc.brainchipinc.com) found **no equivalent explicit bit-exact
or cycle-accurate claim** for the Akida software simulator — only that it is
"a CPU implementation of the Akida Neuromorphic Processor IP" whose inference
"is computed on the host CPU." The model IS fully integer-quantized before
`.forward()` runs (uint8/int8/int32 in and out), which is architecturally
consistent with running the real quantized computation rather than a float
approximation — but this is circumstantial, not a formal guarantee the way
XyloSim's is. Report "Akida-sim agreement," not "verified against silicon."

Install (Linux only — `akida` has no macOS wheel, ever; see Dockerfile.akida):
    docker build -f Dockerfile.akida -t eia-akida .  (or `pip install "eia[akida]"` on Linux)
Docs:
  - MetaTF dev tools: https://brainchip.com/metatf-dev-tools/
  - Akida docs: https://doc.brainchipinc.com/
"""

from __future__ import annotations

import numpy as np

# Akida 2.0 (the current default target — `cnn2snn.get_akida_version()` in
# this toolchain resolves to `AkidaVersion.v2` unless explicitly overridden;
# Akida Pico support also exists in this SDK, `akida.AKD1500`/`Pico_FPGA`,
# but is not exercised here — see docs/akida_ecg_results.md Part 0) 8-bit
# default weight/activation precision. Akida also supports 4-bit and 1-bit
# (1-bit reserved for edge/last-layer learning) — see `weight_bits`/
# `activation_bits` on `quantize_and_convert`.
DEFAULT_WEIGHT_BITS = 8
DEFAULT_ACTIVATION_BITS = 8


def to_akida_input(X: np.ndarray) -> np.ndarray:
    """Convert a real-valued batch into Akida's expected uint8 image format,
    via per-ROW min-max normalization to [0, 255] (the same spirit as image
    pixel values, which is what Akida's InputConv2D layer is built to
    expect) — the last axis is normalized independently per (sample, ...
    leading axes) slice.

    Two input shapes, two output shapes (ECG's `(n, window)` path is
    UNCHANGED from before heart sounds was added — same call, same output):
      - `(n, window)` (ECG: single-channel waveform) -> `(n, window, 1, 1)`.
        `window` becomes the "image height" (time axis); width/channels=1.
      - `(n, H, W)` (heart: `(n_features, n_subwindows)` filterbank map from
        `eia.signal_features` — see `build_akida_heart_model`) -> `(n, H, W,
        1)`. Normalizing per-row (per feature/band) rather than per-whole-
        sample preserves each feature's own dynamic range (line length,
        band power, and spectral entropy are on very different natural
        scales even after `signal_features.normalize_features_train_only`'s
        z-score — this is an additional, deliberately independent [0, 255]
        remap per row, same as ECG's per-window remap).
    """
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=-1, keepdims=True)
    hi = X.max(axis=-1, keepdims=True)
    span = np.where(hi > lo, hi - lo, 1.0)
    scaled = (X - lo) / span * 255.0
    out = scaled.astype(np.uint8)
    if X.ndim == 2:
        return out[..., None, None]  # (n, window) -> (n, window, 1, 1)
    return out[..., None]  # (n, H, W) -> (n, H, W, 1)


def build_akida_model(window: int = 187, n_classes: int = 2):
    """Build the float `tf_keras.Model` ECG classifier: a small quantized-
    1-D-CNN-shaped stack of Conv2D blocks over the (window, 1, 1) input.

    Every conv block uses a SQUARE kernel/stride/pool (see module docstring —
    a confirmed Akida v2 conversion constraint, not a modelling choice; the
    real work only ever happens along the height/time axis since width is
    always 1). Global-average-pools to a `n_classes`-logit head — no softmax
    (`SparseCategoricalCrossentropy(from_logits=True)` downstream, matching
    the Xylo path's raw-logit/spike-count convention).

    Returns an UNTRAINED model — train it with ordinary `.compile()`/
    `.fit()` (class-weighted loss, real labels) before `quantize_and_convert`.
    """
    import tensorflow as tf
    from akida_models.layer_blocks import conv_block, dense_block
    from tf_keras import Model
    from tf_keras.layers import Input, Rescaling

    img_input = Input(shape=(window, 1, 1), name="input", dtype=tf.uint8)
    x = Rescaling(1.0 / 255, name="rescaling")(img_input)
    x = conv_block(x, filters=8, kernel_size=(7, 7), name="block1",
                    padding="same", add_batchnorm=True, strides=(2, 2))
    x = conv_block(x, filters=16, kernel_size=(5, 5), name="block2",
                    padding="same", add_batchnorm=True, pooling="max", pool_size=(2, 2))
    x = conv_block(x, filters=32, kernel_size=(3, 3), name="block3",
                    padding="same", add_batchnorm=True, pooling="global_avg",
                    post_relu_gap=True)
    x = dense_block(x, units=n_classes, name="predictions",
                     add_batchnorm=False, relu_activation=False)
    return Model(img_input, x, name="ecg_akida")


def build_akida_heart_model(n_bands: int = 4, n_subwindows: int = 24, n_classes: int = 2):
    """Build the float `tf_keras.Model` heart-sound classifier over the
    `(n_bands, n_subwindows, 1)` filterbank-feature "image"
    (`eia.signal_features.extract_window_features`'s `(n_features,
    n_subwindows)` output, matching `datasets.PCG_FEATURE_NAMES`'s default 4
    features x `n_subwindows=24` sub-windows) — a Conv2D-native fit, unlike
    ECG's single-column `(window, 1, 1)` reshape: here BOTH spatial axes
    carry real structure (bands x time), not one real axis and one phantom
    one, so this doesn't need `build_akida_model`'s stride-instead-of-pool
    workaround on the input layer.

    Same confirmed Akida v2 constraints apply (square kernel/stride/pool
    every conv layer; `Conv2D -> ReLU -> GlobalAveragePooling2D` ordering
    via `post_relu_gap=True`, not `Conv2D -> GlobalAveragePooling2D ->
    ReLU`) — see `build_akida_model`'s docstring and module docstring for
    where these were found. `n_bands=4` is small, so only ONE square
    max-pool (in block2) is used — a second would collapse the band axis to
    zero; `GlobalAveragePooling2D` in block3 handles whatever's left
    regardless of exact size.

    Returns an UNTRAINED model — train it with ordinary `.compile()`/
    `.fit()` (class-weighted loss, real labels) before `quantize_and_convert`.
    """
    import tensorflow as tf
    from akida_models.layer_blocks import conv_block, dense_block
    from tf_keras import Model
    from tf_keras.layers import Input, Rescaling

    img_input = Input(shape=(n_bands, n_subwindows, 1), name="input", dtype=tf.uint8)
    x = Rescaling(1.0 / 255, name="rescaling")(img_input)
    x = conv_block(x, filters=8, kernel_size=(3, 3), name="block1",
                    padding="same", add_batchnorm=True, strides=(1, 1))
    x = conv_block(x, filters=16, kernel_size=(3, 3), name="block2",
                    padding="same", add_batchnorm=True, pooling="max", pool_size=(2, 2))
    x = conv_block(x, filters=32, kernel_size=(3, 3), name="block3",
                    padding="same", add_batchnorm=True, pooling="global_avg",
                    post_relu_gap=True)
    x = dense_block(x, units=n_classes, name="predictions",
                     add_batchnorm=False, relu_activation=False)
    return Model(img_input, x, name="heart_akida")


def build_akida_mi_model(n_leads: int = 12, n_samples: int = 1000, n_classes: int = 2):
    """Build the float `tf_keras.Model` MI-vs-NORM classifier over the
    `(n_leads, n_samples, 1)` 12-lead ECG "image" (`datasets.PTBXL_LEADS` x
    100 Hz x 10 s = 1000 samples) — a genuinely 2-D fit (leads x time), the
    same class of input as `build_akida_heart_model`'s bands x time map, NOT
    a single-column reshape like `build_akida_model`'s ECG-arrhythmia input:
    MI needs spatial LEAD information (which leads show ST elevation
    localizes the infarct), so leads are a real, not phantom, axis here.

    Deeper / more pooling than the heart model: `n_samples=1000` is ~40x
    heart's `n_subwindows=24`, so the time axis needs more reduction for a
    tractable Akida-sim forward pass. `n_leads=12` (vs. heart's 4 bands)
    tolerates two square max-pools before the lead axis bottoms out.

    Same confirmed Akida v2 constraints as every other model in this module
    (square kernel/stride/pool every conv layer; `Conv2D -> ReLU ->
    GlobalAveragePooling2D` ordering via `post_relu_gap=True`).

    Returns an UNTRAINED model — train it with ordinary `.compile()`/
    `.fit()` (class-weighted loss, real labels) before `quantize_and_convert`.
    """
    import tensorflow as tf
    from akida_models.layer_blocks import conv_block, dense_block
    from tf_keras import Model
    from tf_keras.layers import Input, Rescaling

    img_input = Input(shape=(n_leads, n_samples, 1), name="input", dtype=tf.uint8)
    x = Rescaling(1.0 / 255, name="rescaling")(img_input)
    x = conv_block(x, filters=8, kernel_size=(3, 3), name="block1",
                    padding="same", add_batchnorm=True, strides=(2, 2))
    x = conv_block(x, filters=16, kernel_size=(3, 3), name="block2",
                    padding="same", add_batchnorm=True, pooling="max", pool_size=(2, 2))
    x = conv_block(x, filters=32, kernel_size=(3, 3), name="block3",
                    padding="same", add_batchnorm=True, pooling="max", pool_size=(2, 2))
    x = conv_block(x, filters=64, kernel_size=(3, 3), name="block4",
                    padding="same", add_batchnorm=True, pooling="global_avg",
                    post_relu_gap=True)
    x = dense_block(x, units=n_classes, name="predictions",
                     add_batchnorm=False, relu_activation=False)
    return Model(img_input, x, name="mi_akida")


def quantize_and_convert(model, calibration_samples: np.ndarray,
                          weight_bits: int = DEFAULT_WEIGHT_BITS,
                          activation_bits: int = DEFAULT_ACTIVATION_BITS,
                          num_samples: int | None = None, batch_size: int = 32,
                          qat_epochs: int = 0, qat_lr: float = 1e-4,
                          qat_X=None, qat_y=None, qat_class_weight: dict | None = None):
    """Quantize a trained float model and convert it to an Akida model.

    Args:
        model: a TRAINED `build_akida_model` output (float weights).
        calibration_samples: `(n, window, 1, 1)` uint8 array used to
            calibrate quantization ranges (`quantizeml.models.quantize`'s
            `samples`) — ideally a slice of the real training set, not noise.
        weight_bits/activation_bits: Akida supports 8, 4, or 1-bit (1-bit
            reserved for edge-learning layers) — see module docstring.
        num_samples: calibration sample count; defaults to
            `len(calibration_samples)`.
        batch_size: calibration batch size.
        qat_epochs: if > 0, fine-tunes the QUANTIZED model with real labels
            (`qat_X`/`qat_y`, required if `qat_epochs > 0`) for this many
            epochs at `qat_lr` before conversion — genuine QAT recovering
            accuracy lost to quantization, confirmed to work (see module
            docstring); 0 (default) skips this and uses calibration alone.
        qat_class_weight: optional class-weight dict for the QAT fine-tune
            loss, matching the float-training discipline used everywhere
            else in this repo (real MIT-BIH's imbalance needs it).

    Returns:
        (quantized_keras_model, akida_model) — the quantized `tf_keras.Model`
        (still float-callable for a "does quantization alone break it"
        sanity check) and the converted `akida.Model`, ready to
        `.forward()`/`.predict()` on the software backend.
    """
    import tf_keras
    from quantizeml.models import QuantizationParams, quantize

    import cnn2snn

    qparams = QuantizationParams(weight_bits=weight_bits, activation_bits=activation_bits)
    qmodel = quantize(model, qparams=qparams, samples=calibration_samples,
                       num_samples=num_samples or len(calibration_samples),
                       batch_size=batch_size, epochs=1)

    if qat_epochs > 0:
        if qat_X is None or qat_y is None:
            raise ValueError("qat_epochs > 0 requires qat_X and qat_y")
        qmodel.compile(
            optimizer=tf_keras.optimizers.Adam(qat_lr),
            loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])
        qmodel.fit(qat_X, qat_y, epochs=qat_epochs, batch_size=batch_size,
                   class_weight=qat_class_weight, verbose=0)

    akida_model = cnn2snn.convert(qmodel)
    return qmodel, akida_model


def verify_against_sim(float_model, akida_model, X_uint8: np.ndarray) -> dict:
    """Compare the float `tf_keras.Model`'s predictions vs. the Akida
    (software-simulated) model's predictions on the same uint8 input batch —
    the Akida analog of `rockpool_models.verify_against_sim`.

    Both models emit raw logits (no final activation — see
    `build_akida_model`); the predicted class is each one's argmax.

    Returns dict with both prediction arrays, per-sample agreement, and the
    raw logit arrays (for AUROC/AUPRC — softmax the float logits, and derive
    an Akida score the same way `xylo_verify.py` derives one from summed
    spike counts: normalized positive-class share).
    """
    float_out = np.asarray(float_model.predict(X_uint8, verbose=0))
    akida_out = np.asarray(akida_model.forward(X_uint8)).reshape(X_uint8.shape[0], -1)

    pred_float = float_out.argmax(axis=-1)
    pred_akida = akida_out.argmax(axis=-1)
    return {
        "pred_float": pred_float,
        "pred_akida": pred_akida,
        "out_float": float_out,
        "out_akida": akida_out,
        "match": pred_float == pred_akida,
        "agreement_rate": float((pred_float == pred_akida).mean()),
    }
