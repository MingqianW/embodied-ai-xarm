# Retired gripper/contact diagnostics

Phase 5 retired the one-off friction, split-pad, force-width, scripted-slip,
Menagerie tuning, contact-regression, and policy-matrix programs. Their source
remains available in Git at the pre-Phase-5 checkpoint `df664e8`; executable
copies are intentionally not retained under a `legacy` package.

## Conclusions that remain relevant

- The current compiled scene is the authority for simulation behavior. It uses
  an elliptic friction cone with `impratio=10`, four split fingertip pad geoms,
  pad sliding friction `2.0`, and gripper actuator force range `[-8, 8]`.
  Focused simulation tests lock these values. These choices are not evidence
  of real-robot contact fidelity.
- The historical B/red-block/seed-50000 c5 trace established measurable
  TCP-relative object slip after the scored success point. It did not establish
  friction or table contact as the cause. The retained trace recorder and
  data-only analyzer are the supported way to investigate another episode.
- In converted real and simulated LeRobot data, `actions[6]_t` is a next-state
  imitation label. It is not an independently measured low-level gripper
  command. The retained real-sim audit therefore reports state-only behavior
  and explicitly marks force, friction, servo, and command-delay questions as
  unidentifiable.
- Camera calibration remains an offline image/render workflow. Exact RealSense
  intrinsics and distortion were unavailable, so remaining visual mismatch
  cannot honestly be assigned to extrinsics alone.

No generated result bundle for the retired parameter searches is versioned in
the repository. Consequently, this document does not promote a winning
parameter, claim a causal mechanism, or reconstruct missing evidence.
