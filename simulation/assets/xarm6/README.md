# Canonical xArm6 MuJoCo model

`xarm6_pick_scene.xml` is the checked-in, authoritative runtime MJCF. Its
compiled geometry, actuators, contacts, solver settings, cameras, and task
bodies are protected by simulation regression tests.

The files in `simulation.tools` are development tooling, not a second runtime
source of truth:

1. `python -m simulation.tools.generate_xarm6_mjcf` regenerates the arm model
   from the pinned xArm ROS description sources.
2. `python -m simulation.tools.build_xarm6_pick_scene` assembles the task scene.
   Runtime loading reapplies the package-owned camera calibration from YAML.

When the canonical MJCF already exists, the builder compiles a temporary
candidate, compares every exposed MuJoCo model array and global option, and
leaves the authoritative file byte-for-byte untouched. A behavior-changing
candidate is preserved for explicit review and is never silently promoted.

Generated MJCF changes must be reviewed through the compiled-model contract
tests. Runtime code always loads the checked-in scene through
`simulation.resources`; it does not regenerate models implicitly.
