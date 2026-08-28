"""Exact xArm training-facing field names, shapes, ordering, and units.

The names in this module describe the existing OpenPI/LeRobot contract.  They
are deliberately not a redesign: both real recordings and MuJoCo generation
use six arm joint angles in radians followed by the xArm hardware gripper raw
value.  At the serialization boundary the historical action key is plural
(``actions``), while OpenPI consumes it as the policy action tensor.
"""

from __future__ import annotations


XARM_ARM_JOINT_COLUMNS_RAD = (
    "j1_rad",
    "j2_rad",
    "j3_rad",
    "j4_rad",
    "j5_rad",
    "j6_rad",
)
XARM_GRIPPER_COLUMN_RAW = "gripper_mm"
# ``gripper_mm`` is the historical CSV name.  Its numeric value is the xArm
# controller's raw gripper convention, not a geometric aperture in millimetres.
XARM_STATE_COLUMNS = XARM_ARM_JOINT_COLUMNS_RAD + (XARM_GRIPPER_COLUMN_RAW,)
XARM_STATE_SHAPE = (7,)
XARM_ACTION_SHAPE = XARM_STATE_SHAPE
XARM_IMAGE_SHAPE = (480, 640, 3)
XARM_IMAGE_DTYPE = "uint8"

TRAINING_IMAGE_KEY = "image"
TRAINING_WRIST_IMAGE_KEY = "wrist_image"
TRAINING_STATE_KEY = "state"
TRAINING_ACTION_KEY = "actions"
TRAINING_TASK_KEY = "task"
TRAINING_REQUIRED_KEYS = (
    TRAINING_IMAGE_KEY,
    TRAINING_WRIST_IMAGE_KEY,
    TRAINING_STATE_KEY,
    TRAINING_ACTION_KEY,
    TRAINING_TASK_KEY,
)

# OpenPI remaps the LeRobot fields at ingestion.  Keeping this explicit avoids
# confusing storage keys with model-input keys.
OPENPI_FIELD_MAPPING = {
    TRAINING_IMAGE_KEY: "observation/image",
    TRAINING_WRIST_IMAGE_KEY: "observation/wrist_image",
    TRAINING_STATE_KEY: "observation/state",
    TRAINING_ACTION_KEY: "action",
    TRAINING_TASK_KEY: "task",
}

