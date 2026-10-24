"""
Unit tests for robot configuration, URDF model, and MJCF specification.

Author: Youseef Abdel Tawwab
"""

import os
import xml.etree.ElementTree as ET
import yaml
import pytest


def test_robot_yaml_config():
    """Verify robot_config.yaml loads correctly and has complete 12-DOF definitions."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "robot_config.yaml")
    assert os.path.exists(cfg_path), "robot_config.yaml not found"

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg["robot"]["num_legs"] == 4
    assert cfg["robot"]["dof_per_leg"] == 3
    assert cfg["robot"]["total_dof"] == 12
    assert len(cfg["joints"]["names"]) == 12
    assert len(cfg["joints"]["nominal_angles"]) == 12
    assert cfg["joints"]["limits"]["max_torque"] > 0


def test_urdf_xml_structure():
    """Verify URDF file is well-formed XML and contains expected 12 revolute joints."""
    urdf_path = os.path.join(os.path.dirname(__file__), "..", "models", "urdf", "quadruped_12dof.urdf")
    assert os.path.exists(urdf_path), "URDF model not found"

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    assert root.tag == "robot"
    assert root.attrib["name"] == "quadruped_12dof"

    revolute_joints = [j for j in root.findall("joint") if j.attrib.get("type") == "revolute"]
    assert len(revolute_joints) == 12, f"Expected 12 revolute joints, found {len(revolute_joints)}"

    fixed_joints = [j for j in root.findall("joint") if j.attrib.get("type") == "fixed"]
    assert len(fixed_joints) == 4, f"Expected 4 fixed foot joints, found {len(fixed_joints)}"


def test_mjcf_xml_structure():
    """Verify MJCF model is well-formed XML and contains 12 motor actuators."""
    mjcf_path = os.path.join(os.path.dirname(__file__), "..", "models", "mjcf", "quadruped_12dof.xml")
    assert os.path.exists(mjcf_path), "MJCF model not found"

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    assert root.tag == "mujoco"

    actuators = root.find("actuator")
    assert actuators is not None, "actuator block missing in MJCF"

    motors = actuators.findall("motor")
    assert len(motors) == 12, f"Expected 12 motor actuators, found {len(motors)}"
