from types import SimpleNamespace

import numpy as np
import pytest

from voxelmorph_pipeline.inference import predict_moved_and_warp


class FakeInferenceModel:
    def __init__(self, outputs, predictions):
        self.outputs = outputs
        self.predictions = predictions
        self.predict_calls = 0

    def predict(self, inputs, verbose=0):
        assert len(inputs) == 2
        assert verbose == 0
        self.predict_calls += 1
        return self.predictions


def test_moved_image_and_final_warp_share_one_prediction():
    moving = np.ones((1, 4, 5, 6, 1), dtype=np.float32)
    fixed = np.zeros_like(moving)
    moved_prediction = moving * 0.5
    warp_prediction = np.zeros((1, 4, 5, 6, 3), dtype=np.float32)
    source_model = SimpleNamespace(
        inputs=["moving", "fixed"],
        references=SimpleNamespace(y_source="moved_tensor", pos_flow="final_warp_tensor"),
    )
    built = []

    def model_factory(*, inputs, outputs):
        assert inputs is source_model.inputs
        inference_model = FakeInferenceModel(
            outputs,
            [moved_prediction, warp_prediction],
        )
        built.append(inference_model)
        return inference_model

    moved, warp = predict_moved_and_warp(
        source_model,
        moving,
        fixed,
        model_factory=model_factory,
    )

    assert built[0].outputs == ["moved_tensor", "final_warp_tensor"]
    assert built[0].predict_calls == 1
    assert moved.shape == (4, 5, 6)
    assert warp.shape == (4, 5, 6, 3)
    assert np.isfinite(moved).all()
    assert np.isfinite(warp).all()


def test_nonfinite_single_pass_output_is_rejected():
    moving = np.ones((1, 2, 2, 2, 1), dtype=np.float32)
    fixed = np.zeros_like(moving)
    moved = moving.copy()
    moved[0, 0, 0, 0, 0] = np.nan
    warp = np.zeros((1, 2, 2, 2, 3), dtype=np.float32)
    source_model = SimpleNamespace(
        inputs=["moving", "fixed"],
        references=SimpleNamespace(y_source="moved", pos_flow="warp"),
    )

    with pytest.raises(ValueError, match="NaN or Inf"):
        predict_moved_and_warp(
            source_model,
            moving,
            fixed,
            model_factory=lambda **_: FakeInferenceModel(None, [moved, warp]),
        )
