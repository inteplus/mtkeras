"""Regression tests for CenterAround, MobileNetV3Mixer and MHAPool2D (need Keras installed)."""

import importlib.util
import unittest


def importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


HAS_KERAS = importable("keras")


@unittest.skipUnless(HAS_KERAS, "keras is not installed")
class TestCenterAround(unittest.TestCase):
    def _run(self, shape, ref):
        import numpy as np
        from mt.keras_src.constraints_src.center_around import CenterAround

        w = np.random.RandomState(0).rand(*shape).astype("float32")
        out = np.asarray(CenterAround(ref)(w))
        self.assertEqual(out.shape, w.shape)
        np.testing.assert_allclose(out.mean(axis=-1), ref, atol=1e-5)

    def test_1d(self):
        self._run((5,), 0.0)

    def test_2d(self):
        self._run((3, 4), 0.5)

    def test_3d(self):
        self._run((2, 3, 4), -1.0)


@unittest.skipUnless(HAS_KERAS, "keras is not installed")
class TestMixerVariants(unittest.TestCase):
    def _mixer(self, variant):
        from mt import keras
        from mt.base import model as base_model
        from mt.keras_src.applications_src.mobilenet_v3_split import MobileNetV3Mixer

        params = base_model.MobileNetV3MixerParams(variant=variant)
        inp = keras.layers.Input(shape=(4, 4, 8))
        return MobileNetV3Mixer(inp, params, 16)

    def test_mobilenet(self):
        self.assertEqual(tuple(self._mixer("mobilenet").output_shape), (None, 1, 1, 16))

    def test_maxpool(self):
        self.assertEqual(tuple(self._mixer("maxpool").output_shape), (None, 8))


@unittest.skipUnless(HAS_KERAS, "keras is not installed")
class TestSplitModelType(unittest.TestCase):
    def test_invalid_model_type_raises(self):
        from mt.base import model as base_model
        from mt.keras_src.applications_src.mobilenet_v3_split import MobileNetV3Split

        for bad in ("large", "Medium", None):
            with self.assertRaises(base_model.ModelSyntaxError):
                MobileNetV3Split((64, 64, 3), model_type=bad)


@unittest.skipUnless(HAS_KERAS, "keras is not installed")
class TestMHAPoolSize(unittest.TestCase):
    def test_pool_size_used(self):
        from mt import keras
        from mt.keras_src.layers_src import MHAPool2D

        layer = MHAPool2D(2, 4, pool_size=(4, 4))
        out = layer(keras.layers.Input(shape=[8, 8, 8]))
        self.assertEqual(tuple(out.shape), (None, 2, 2, 8))

    def test_default(self):
        from mt import keras
        from mt.keras_src.layers_src import MHAPool2D

        out = MHAPool2D(2, 4)(keras.layers.Input(shape=[8, 8, 8]))
        self.assertEqual(tuple(out.shape), (None, 4, 4, 8))


if __name__ == "__main__":
    unittest.main()
