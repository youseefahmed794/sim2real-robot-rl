"""
Unit test verifying evaluation protocol rigor:
Asserts that every held-out evaluation scenario lies STRICTLY OUTSIDE
the training Domain Randomization ranges.

Author: Youseef Abdel Tawwab
"""

import os
import yaml
import pytest


def load_configs():
    base_dir = os.path.dirname(__file__)
    dr_path = os.path.abspath(os.path.join(base_dir, "..", "configs", "dr_train.yaml"))
    eval_path = os.path.abspath(os.path.join(base_dir, "..", "configs", "eval_suites.yaml"))

    with open(dr_path, "r") as f:
        dr_cfg = yaml.safe_load(f)
    with open(eval_path, "r") as f:
        eval_cfg = yaml.safe_load(f)

    return dr_cfg, eval_cfg


def test_nominal_is_in_distribution():
    """Verify nominal suite is within training DR bounds."""
    dr_cfg, eval_cfg = load_configs()
    nom = eval_cfg["in_distribution"]["nominal"]

    assert dr_cfg["friction"]["min"] <= nom["friction"] <= dr_cfg["friction"]["max"]
    assert dr_cfg["base_mass_scale"]["min"] <= nom["base_mass_scale"] <= dr_cfg["base_mass_scale"]["max"]
    assert dr_cfg["latency_ms"]["min"] <= nom["latency_ms"] <= dr_cfg["latency_ms"]["max"]
    assert nom["push_force_n"] <= dr_cfg["push_impulse"]["max_force_n"]


def test_all_held_out_suites_strictly_outside_training_ranges():
    """
    Verify that every held-out suite parameter lies strictly outside
    the corresponding training DR bounds.
    """
    dr_cfg, eval_cfg = load_configs()
    held_out = eval_cfg["held_out"]

    assert len(held_out) >= 4, "Must have at least 4 held-out stress suites."

    for suite_name, suite in held_out.items():
        param_name = suite["tested_parameter"]
        test_type = suite["test_type"]
        val = suite[param_name]

        if param_name == "friction":
            train_min = dr_cfg["friction"]["min"]
            train_max = dr_cfg["friction"]["max"]
            if test_type == "below_min":
                assert val < train_min, (
                    f"Held-out '{suite_name}' friction {val} must be strictly below training min {train_min}"
                )
            elif test_type == "above_max":
                assert val > train_max, (
                    f"Held-out '{suite_name}' friction {val} must be strictly above training max {train_max}"
                )

        elif param_name == "base_mass_scale":
            train_min = dr_cfg["base_mass_scale"]["min"]
            train_max = dr_cfg["base_mass_scale"]["max"]
            if test_type == "below_min":
                assert val < train_min, (
                    f"Held-out '{suite_name}' mass scale {val} must be strictly below training min {train_min}"
                )
            elif test_type == "above_max":
                assert val > train_max, (
                    f"Held-out '{suite_name}' mass scale {val} must be strictly above training max {train_max}"
                )

        elif param_name == "latency_ms":
            train_max = dr_cfg["latency_ms"]["max"]
            assert val > train_max, (
                f"Held-out '{suite_name}' latency {val} ms must be strictly above training max {train_max} ms"
            )

        elif param_name == "push_force_n":
            train_max = dr_cfg["push_impulse"]["max_force_n"]
            assert val > train_max, (
                f"Held-out '{suite_name}' push {val} N must be strictly above training max {train_max} N"
            )
        else:
            pytest.fail(f"Unknown tested parameter: {param_name}")
