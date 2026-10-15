import numpy as np


def test_find_bw_targets_fast_path_skips_heavy_helpers_when_base_candidate_exists(
    monkeypatch,
) -> None:
    from stuff_i_moved_away.inference import find_bw_targets as bw

    config = {
        "detector": {
            "class_id": 10,
            "score_threshold": 0.5,
            "template_rescue": {
                "enabled": True,
                "trigger_max_initial_boxes": 1,
            },
            "fast_runtime": {},
        },
        "dataset": {},
        "benchmark": {},
    }

    image = np.zeros((64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(
        bw, "_extract_square_candidates", lambda image_bgr, cfg: [{"score": 0.92}]
    )
    monkeypatch.setattr(
        bw, "_dark_mask", lambda image_bgr, cfg: np.zeros((64, 64), dtype=np.uint8)
    )
    monkeypatch.setattr(bw, "_extract_dark_components", lambda mask, cfg: [])
    monkeypatch.setattr(
        bw, "_extract_local_dark_components", lambda image_bgr, cfg: []
    )
    monkeypatch.setattr(bw, "_group_components", lambda components, cfg: [])
    monkeypatch.setattr(
        bw,
        "_score_square_candidate",
        lambda image_bgr, vote, cfg: {
            "bbox": (10, 10, 20, 20),
            "score": 0.91,
            "source": "square_template",
            "white_ratio": 0.5,
            "black_ratio": 0.5,
            "red_ratio": 0.0,
            "green_ratio": 0.0,
            "contrast_std": 30.0,
            "component_count": 1,
            "significant_component_count": 1,
            "compact_component_count": 1,
            "max_component_solidity": 1.0,
            "edge_ratio": 0.0,
            "template_vote_count": 1,
            "template_mean_score": 0.91,
            "template_patch_score": 0.91,
            "template_corr": 0.91,
            "binary_agreement": 0.91,
            "contrast_separation": 0.91,
            "structure_score": 0.91,
            "refined_center_x": 20.0,
            "refined_center_y": 20.0,
        },
    )
    monkeypatch.setattr(
        bw,
        "_nms",
        lambda candidates, cfg: sorted(
            candidates, key=lambda candidate: candidate["score"], reverse=True
        ),
    )
    monkeypatch.setattr(bw, "_limit_square_candidates", lambda candidates, cfg: candidates)
    monkeypatch.setattr(
        bw, "_dedupe_candidates_by_center", lambda candidates, **kwargs: candidates
    )
    monkeypatch.setattr(
        bw,
        "_reject_border_touching_candidates",
        lambda candidates, image_shape, cfg: candidates,
    )

    def fail(*args, **kwargs):
        raise AssertionError("heavy helper should not be called in fast path")

    monkeypatch.setattr(bw, "_extract_component_template_seed_components", fail)
    monkeypatch.setattr(bw, "_component_seeded_template_candidates", fail)
    monkeypatch.setattr(bw, "_self_template_candidates", fail)
    monkeypatch.setattr(bw, "_template_match_votes", fail)
    monkeypatch.setattr(bw, "_scene_anchor_candidates", fail)

    result = bw.find_bw_targets_in_image(image=image, config=config, debug=False)

    assert result is not None
    assert len(result["boxes"]) == 1
    assert result["boxes"][0]["bbox_px"] == (10, 10, 20, 20)
