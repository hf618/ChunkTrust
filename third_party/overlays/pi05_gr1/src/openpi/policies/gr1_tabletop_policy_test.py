import numpy as np

from openpi.policies import gr1_tabletop_policy


def test_extract_expand_and_action_dict_round_trip() -> None:
    raw = np.arange(2 * 3 * 44, dtype=np.float32).reshape(2, 3, 44)
    active = gr1_tabletop_policy.extract_active(raw)
    expected_indices = [*range(0, 7), *range(22, 29), *range(7, 13), *range(29, 35), *range(41, 44)]
    np.testing.assert_array_equal(active, raw[..., expected_indices])

    expanded = gr1_tabletop_policy.expand_active(active)
    np.testing.assert_array_equal(gr1_tabletop_policy.extract_active(expanded), active)
    inactive_indices = [*range(13, 22), *range(35, 41)]
    np.testing.assert_array_equal(expanded[..., inactive_indices], 0.0)

    action_dict = gr1_tabletop_policy.active_to_action_dict(active)
    assert list(action_dict) == [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    np.testing.assert_array_equal(gr1_tabletop_policy.action_dict_to_active(action_dict), active)


def test_inputs_use_one_real_camera_two_masked_cameras_and_29d_absolute_actions() -> None:
    raw_state = np.arange(44, dtype=np.float64)
    raw_actions = np.arange(4 * 44, dtype=np.float64).reshape(4, 44)
    channel_first_image = np.arange(3 * 16 * 12, dtype=np.uint8).reshape(3, 16, 12)
    transformed = gr1_tabletop_policy.Gr1TabletopInputs()(
        {
            "observation/image": channel_first_image,
            "observation/state": raw_state,
            "actions": raw_actions,
            "prompt": "place the bottle in the cabinet",
        }
    )

    assert transformed["state"].shape == (29,)
    assert transformed["actions"].shape == (4, 29)
    assert transformed["image"]["base_0_rgb"].shape == (16, 12, 3)
    assert transformed["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert transformed["image"]["right_wrist_0_rgb"].shape == (224, 224, 3)
    assert not np.any(transformed["image"]["left_wrist_0_rgb"])
    assert not np.any(transformed["image"]["right_wrist_0_rgb"])
    assert transformed["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.False_,
    }
    assert transformed["prompt"] == "place the bottle in the cabinet"
    np.testing.assert_array_equal(
        transformed["actions"],
        gr1_tabletop_policy.extract_active(raw_actions).astype(np.float32),
    )


def test_outputs_keep_only_active_dimensions() -> None:
    actions = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)
    outputs = gr1_tabletop_policy.Gr1TabletopOutputs()({"actions": actions, "trace": 1})
    assert outputs["actions"].shape == (5, 29)
    assert outputs["trace"] == 1
    np.testing.assert_array_equal(outputs["actions"], actions[:, :29])
