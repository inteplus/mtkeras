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
        """Returns `w` minus ``(mean(w, axis=-1) - ref_value)``, centering the last axis.

        The shift is ``mean(w, axis=-1, keepdims=True) - ref_value``, which broadcasts against `w`
        so that the output has the same shape as `w` and its last-axis mean equals `ref_value`.

        Parameters
        ----------
        w : tensor-like
            the weight tensor, last axis being the one to center

        Returns
        -------
        tensor-like
            the shifted tensor, of the same shape as `w`
        """
        mean = ops.reduce_mean(w, axis=-1, keepdims=True)
        ref_mean = mean - self.ref_value
        return w - ref_mean

    def get_config(self):
        """Returns ``{"ref_value": ref_value}``."""
        return {"ref_value": self.ref_value}
