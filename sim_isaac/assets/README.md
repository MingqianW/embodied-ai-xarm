# Isaac xArm Assets

Source robot descriptions and meshes remain under
`third_party/xarm_ros2/xarm_description`.

Run `sim_isaac/scripts/prepare_xarm_asset.py --validate-only` to inspect the
source. Expanded URDF, generated USD, and validation reports are written under
`sim_isaac/generated/`, which is intentionally ignored. A USD is only
considered generated when the Isaac importer succeeds and the output file is
present.

