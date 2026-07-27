r"""A small attention-based sequence model, in NumPy, with hand-derived gradients.

What this is
------------
A transformer-style volatility forecaster: it reads a window of recent returns and
predicts the distribution of the next one. No autograd, no framework -- the
forward pass and every gradient are written out, which is the only way to build
one under DDR-002 and also the only way to be sure you understand it.

What it forecasts, and why not direction
-----------------------------------------
It outputs a **scale**, not a sign. Direction is not reliably forecastable from
price history; volatility is, because it clusters. Pointing a flexible model at
direction on ten years of daily data is how people fit noise and publish it, so
the target here is one where there is genuinely signal to find.

Trained by minimising **Gaussian negative log-likelihood** in the scale, which is
a proper scoring rule for a distribution: the model cannot lower its loss by
hedging toward the middle.

.. math::
   \mathcal{L} = \frac{1}{2n}\sum_t \left[\log \hat\sigma_t^2
                 + \frac{r_t^2}{\hat\sigma_t^2}\right]

Architecture
------------
Deliberately small: one attention head, one feed-forward layer, and a softplus
output that keeps the predicted scale positive. A few thousand parameters against
a few thousand training points. Anything larger would fit the sample rather than
the process, and the point is a fair comparison against the baselines, not the
largest model that will run.

Honest expectations
-------------------
The EWMA baseline in :mod:`quantos.models.baselines` is very hard to beat at a
one-day horizon. It has no fitted parameters and is already a GARCH(1,1) with
persistence pinned at one. This model may lose to it. That result would be
published as readily as a win -- see ``scripts/benchmark_models.py``, which runs
every forecaster on the same walk-forward split and prints the table whichever way
it falls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["AttentionVolatilityModel", "TrainingHistory", "make_windows"]


def make_windows(
    returns: NDArray[np.float64], window: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Turn a return series into ``(inputs, targets)`` of shape ``(n, window)``, ``(n,)``.

    Strictly causal: row ``i`` holds returns ``[i, i+window)`` and the target is
    return ``i+window``, so no window ever contains its own answer.

    Example
        >>> import numpy as np
        >>> x, y = make_windows(np.arange(6.0), window=3)
        >>> x
        array([[0., 1., 2.],
               [1., 2., 3.],
               [2., 3., 4.]])
        >>> y
        array([3., 4., 5.])
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size <= window:
        raise ValueError(f"need more than {window} returns, got {returns.size}")
    n = returns.size - window
    inputs = np.lib.stride_tricks.sliding_window_view(returns, window)[:n]
    targets = returns[window:]
    return np.ascontiguousarray(inputs), np.ascontiguousarray(targets)


def _softplus(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """``log(1 + e^x)``, computed without overflowing for large ``x``."""
    return np.logaddexp(0.0, x)


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Derivative of softplus, written stably for both signs."""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


@dataclass
class TrainingHistory:
    """What happened during training, including whether it should be believed."""

    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def overfitted(self) -> bool:
        """Report whether validation loss rose while training loss kept falling."""
        if len(self.validation_loss) < 5:
            return False
        return bool(
            self.validation_loss[-1] > min(self.validation_loss) * 1.02
            and self.train_loss[-1] < self.train_loss[len(self.train_loss) // 2]
        )

    def summary(self) -> str:
        lines = [
            f"{len(self.train_loss)} epochs, best at {self.best_epoch}"
            f"{' (stopped early)' if self.stopped_early else ''}",
        ]
        if self.train_loss:
            lines.append(
                f"  train {self.train_loss[0]:.5f} -> {self.train_loss[-1]:.5f}, "
                f"validation {self.validation_loss[0]:.5f} -> {self.validation_loss[-1]:.5f}"
            )
        if self.overfitted:
            lines.append("  validation loss rose while training loss fell: overfitting")
        lines.extend(f"  {note}" for note in self.notes)
        return "\n".join(lines)


@dataclass
class AttentionVolatilityModel:
    r"""Single-head attention over a window of returns, predicting a scale.

    The forward pass, in order:

    1. **Feature map.** Each of the ``window`` positions becomes
       :math:`[\,|r|, r, r^2\,]`, then a learned projection lifts it to
       ``d_model``. Absolute return is included explicitly because it is the
       quantity volatility clustering acts on, and making the model rediscover it
       from raw returns wastes capacity.
    2. **Attention.** One head, so the learned weighting over lags is directly
       readable -- :meth:`attention_profile` returns it, and on trained models it
       reliably concentrates on recent lags, which is the model recovering
       clustering on its own.
    3. **Feed-forward** with a tanh nonlinearity.
    4. **Softplus output**, which keeps the predicted scale strictly positive
       without a clamp that would kill the gradient.

    Gradients are derived by hand for every step. They are verified against
    central finite differences in ``tests/models/test_sequence.py``, which is the
    only honest way to ship hand-written backpropagation.
    """

    window: int = 20
    d_model: int = 8
    seed: int = 20240719

    # Parameters, allocated by `initialise`.
    projection: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    query: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    key: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    value: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    hidden_weight: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    hidden_bias: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    output_weight: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    output_bias: float = 0.0

    #: Set by `fit`; inputs are divided by it so the loss surface is well scaled.
    input_scale: float = 1.0
    history: TrainingHistory = field(default_factory=TrainingHistory)

    N_FEATURES = 3

    def initialise(self) -> AttentionVolatilityModel:
        """Xavier-style initialisation, sized so the first forward pass is sane."""
        rng = np.random.default_rng(self.seed)

        def scaled(shape: tuple[int, ...], fan_in: int) -> NDArray[np.float64]:
            return rng.normal(0.0, np.sqrt(1.0 / fan_in), shape)

        d = self.d_model
        self.projection = scaled((self.N_FEATURES, d), self.N_FEATURES)
        self.query = scaled((d, d), d)
        self.key = scaled((d, d), d)
        self.value = scaled((d, d), d)
        self.hidden_weight = scaled((d, d), d)
        self.hidden_bias = np.zeros(d)
        self.output_weight = scaled((d, 1), d).ravel()
        # Start near softplus(0) ~ 0.69 so the initial scale is order 1 after
        # input normalisation, rather than near zero where the log blows up.
        self.output_bias = 0.0
        return self

    # -- forward ----------------------------------------------------------- #
    def _features(self, windows: NDArray[np.float64]) -> NDArray[np.float64]:
        scaled = windows / self.input_scale
        return np.stack([np.abs(scaled), scaled, scaled**2], axis=-1)

    def _forward(self, windows: NDArray[np.float64]) -> tuple[NDArray[np.float64], dict]:
        """Predicted scale per row, plus the cache the backward pass needs."""
        features = self._features(windows)  # (n, T, F)
        embedded = features @ self.projection  # (n, T, d)

        q = embedded @ self.query  # (n, T, d)
        k = embedded @ self.key
        v = embedded @ self.value

        # Attention pooled to a single vector: the query is the mean position, so
        # the model learns which lags matter rather than producing a sequence.
        q_pooled = q.mean(axis=1)  # (n, d)
        scores = np.einsum("nd,ntd->nt", q_pooled, k) / np.sqrt(self.d_model)
        scores -= scores.max(axis=1, keepdims=True)  # stabilise the softmax
        weights = np.exp(scores)
        weights /= weights.sum(axis=1, keepdims=True)  # (n, T)

        context = np.einsum("nt,ntd->nd", weights, v)  # (n, d)
        pre_hidden = context @ self.hidden_weight + self.hidden_bias
        hidden = np.tanh(pre_hidden)
        pre_output = hidden @ self.output_weight + self.output_bias
        scale = _softplus(pre_output) + 1e-8

        cache = {
            "features": features,
            "embedded": embedded,
            "q": q,
            "k": k,
            "v": v,
            "q_pooled": q_pooled,
            "weights": weights,
            "context": context,
            "pre_hidden": pre_hidden,
            "hidden": hidden,
            "pre_output": pre_output,
            "scale": scale,
        }
        return scale, cache

    def predict_scaled(self, windows: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predicted scale in normalised units."""
        return self._forward(np.atleast_2d(np.asarray(windows, dtype=float)))[0]

    def predict(self, windows: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predicted volatility in the units of the original returns."""
        return self.predict_scaled(windows) * self.input_scale

    # -- loss and gradients ------------------------------------------------- #
    def _loss_and_gradients(
        self, windows: NDArray[np.float64], targets: NDArray[np.float64]
    ) -> tuple[float, dict]:
        r"""Gaussian negative log-likelihood and every parameter gradient.

        With :math:`\hat\sigma` the predicted scale and :math:`y` the normalised
        target,

        .. math:: \mathcal{L} = \frac{1}{2n}\sum \log \hat\sigma^2
                                + \frac{y^2}{\hat\sigma^2}

        so :math:`\partial\mathcal{L}/\partial\hat\sigma =
        \frac{1}{n}(\hat\sigma^{-1} - y^2\hat\sigma^{-3})`, and the rest is the
        chain rule back through softplus, the feed-forward layer, the attention
        weights and the projection.
        """
        scale, cache = self._forward(windows)
        y = targets / self.input_scale
        n = float(targets.size)

        loss = float(np.mean(np.log(scale**2) + y**2 / scale**2) / 2.0)

        # dL/d(scale)
        d_scale = (1.0 / scale - y**2 / scale**3) / n
        # through softplus
        d_pre_output = d_scale * _sigmoid(cache["pre_output"])

        d_output_weight = cache["hidden"].T @ d_pre_output
        d_output_bias = float(np.sum(d_pre_output))

        d_hidden = np.outer(d_pre_output, self.output_weight)
        d_pre_hidden = d_hidden * (1.0 - cache["hidden"] ** 2)
        d_hidden_weight = cache["context"].T @ d_pre_hidden
        d_hidden_bias = d_pre_hidden.sum(axis=0)

        d_context = d_pre_hidden @ self.hidden_weight.T  # (n, d)

        # context = sum_t weights[n,t] * v[n,t,:]
        d_weights = np.einsum("nd,ntd->nt", d_context, cache["v"])
        d_v = cache["weights"][:, :, None] * d_context[:, None, :]

        # softmax Jacobian: dL/ds_t = w_t (dL/dw_t - sum_j w_j dL/dw_j)
        weighted = np.sum(d_weights * cache["weights"], axis=1, keepdims=True)
        d_scores = cache["weights"] * (d_weights - weighted)
        d_scores /= np.sqrt(self.d_model)

        # scores[n,t] = q_pooled[n,:] . k[n,t,:]
        d_q_pooled = np.einsum("nt,ntd->nd", d_scores, cache["k"])
        d_k = d_scores[:, :, None] * cache["q_pooled"][:, None, :]
        d_q = np.repeat(d_q_pooled[:, None, :], self.window, axis=1) / self.window

        d_query = np.einsum("ntd,nte->de", cache["embedded"], d_q)
        d_key = np.einsum("ntd,nte->de", cache["embedded"], d_k)
        d_value = np.einsum("ntd,nte->de", cache["embedded"], d_v)

        d_embedded = d_q @ self.query.T + d_k @ self.key.T + d_v @ self.value.T
        d_projection = np.einsum("ntf,ntd->fd", cache["features"], d_embedded)

        return loss, {
            "projection": d_projection,
            "query": d_query,
            "key": d_key,
            "value": d_value,
            "hidden_weight": d_hidden_weight,
            "hidden_bias": d_hidden_bias,
            "output_weight": d_output_weight,
            "output_bias": d_output_bias,
        }

    def _parameters(self) -> dict[str, NDArray[np.float64] | float]:
        return {
            "projection": self.projection,
            "query": self.query,
            "key": self.key,
            "value": self.value,
            "hidden_weight": self.hidden_weight,
            "hidden_bias": self.hidden_bias,
            "output_weight": self.output_weight,
            "output_bias": self.output_bias,
        }

    def _apply(self, name: str, delta: NDArray[np.float64] | float) -> None:
        if name == "output_bias":
            self.output_bias = float(self.output_bias + delta)
        else:
            setattr(self, name, getattr(self, name) + delta)

    # -- training ----------------------------------------------------------- #
    def fit(
        self,
        returns: NDArray[np.float64],
        *,
        epochs: int = 200,
        learning_rate: float = 0.02,
        validation_fraction: float = 0.2,
        patience: int = 20,
        weight_decay: float = 1e-4,
    ) -> AttentionVolatilityModel:
        """Train with Adam, early stopping on a **chronological** validation split.

        The split is by time, never shuffled. A random split would put future
        observations in the training set and leak the answer backwards, which is
        the single most common way a financial sequence model reports a score it
        did not earn.
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        inputs, targets = make_windows(returns, self.window)

        split = int(len(inputs) * (1.0 - validation_fraction))
        if split < 50 or len(inputs) - split < 20:
            raise ValueError(
                f"need more data: {len(inputs)} windows split into {split} train and "
                f"{len(inputs) - split} validation"
            )

        train_x, train_y = inputs[:split], targets[:split]
        valid_x, valid_y = inputs[split:], targets[split:]

        # Normalise by the TRAINING standard deviation only. Using the whole
        # series would leak the validation period's volatility into training.
        self.input_scale = float(np.std(train_y, ddof=1)) or 1.0
        self.initialise()

        moment1 = {
            k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in self._parameters().items()
        }
        moment2 = {
            k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in self._parameters().items()
        }
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        best_loss = float("inf")
        best_parameters: dict | None = None
        best_epoch = 0
        history = TrainingHistory()

        for epoch in range(1, epochs + 1):
            loss, gradients = self._loss_and_gradients(train_x, train_y)

            for name, raw_gradient in gradients.items():
                gradient = np.asarray(raw_gradient, dtype=float)
                if name not in ("hidden_bias", "output_bias"):
                    # L2 penalty on weights but not biases: shrinking a bias just
                    # biases the output, which is not what regularisation is for.
                    gradient = gradient + weight_decay * np.asarray(
                        self._parameters()[name], dtype=float
                    )
                moment1[name] = beta1 * moment1[name] + (1 - beta1) * gradient
                moment2[name] = beta2 * moment2[name] + (1 - beta2) * gradient**2
                corrected1 = moment1[name] / (1 - beta1**epoch)
                corrected2 = moment2[name] / (1 - beta2**epoch)
                self._apply(name, -learning_rate * corrected1 / (np.sqrt(corrected2) + epsilon))

            validation_loss, _ = self._loss_and_gradients(valid_x, valid_y)
            history.train_loss.append(loss)
            history.validation_loss.append(validation_loss)

            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_epoch = epoch
                best_parameters = {
                    k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in self._parameters().items()
                }
            elif epoch - best_epoch >= patience:
                history.stopped_early = True
                history.notes.append(
                    f"no validation improvement for {patience} epochs; restored epoch {best_epoch}"
                )
                break

        if best_parameters is not None:
            for name, value in best_parameters.items():
                if name == "output_bias":
                    self.output_bias = float(value)
                else:
                    setattr(self, name, value)

        history.best_epoch = best_epoch
        self.history = history
        return self

    def attention_profile(self, windows: NDArray[np.float64]) -> NDArray[np.float64]:
        """Average attention weight per lag -- what the model learned to look at.

        On a trained model this concentrates on recent lags, which is volatility
        clustering recovered from data rather than imposed. A flat profile means
        the attention layer is doing nothing and the model is a feed-forward net
        on the window mean.
        """
        _, cache = self._forward(np.atleast_2d(np.asarray(windows, dtype=float)))
        return np.asarray(cache["weights"].mean(axis=0), dtype=float)

    def n_parameters(self) -> int:
        return int(sum(np.asarray(v, dtype=float).size for v in self._parameters().values()))
