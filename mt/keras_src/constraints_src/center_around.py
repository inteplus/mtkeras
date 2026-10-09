from .. import constraints
from ..ops_compat import ops


class CenterAround(constraints.Constraint):
    """Constrains the last axis to have values centered around `ref_value`.

    After each optimiser update the constraint shifts the weights so that the mean over the last
    axis equals `ref_value`. It is meant for e.g. the bias vector of a logit layer (see the
    ``zero_mean_logit_biases`` option of :func:`mt.keras.applications.create_classifier_block`).

    Parameters
    ----------
    ref_value : float, optional
        the target mean along the last axis. Defaults to 0.0.

    Examples
    --------
    Used as a bias constraint of a dense layer (not executed here):

    .. code-block:: python

       from mt import keras
       layer = keras.layers.Dense(10, bias_constraint=keras.constraints.CenterAround())
    """

    def __init__(self, ref_value: float = 0.0):
        self.ref_value = ref_value

    def __call__(self, w):
        """Returns `w` minus ``(mean(w, axis=-1) - ref_value)``, intended to center the last axis.

        The shift is computed as ``mean(w, axis=-1, keepdims=True) - ref_value`` and then given an
        extra trailing axis (``expand_dims(-1)``) before being subtracted from `w`.

        Parameters
        ----------
        w : tensor-like
            the weight tensor, last axis being the one to center

        Returns
        -------
        tensor-like
            the shifted tensor. Because of the extra ``expand_dims``, broadcasting adds a leading
            axis for a 1D `w` (shape ``(n,)`` gives ``(1, n)``) and, for ND `w` with N > 1, a
            result of a larger shape than `w`; the values are only meaningful for 1D weights up to
            this reshaping.
        """
        mean = ops.reduce_mean(w, axis=-1, keepdims=True)
        ref_mean = mean - self.ref_value
        ref_mean = ops.expand_dims(ref_mean, -1)
        return w - ref_mean

    def get_config(self):
        """Returns ``{"ref_value": ref_value}``."""
        return {"ref_value": self.ref_value}
