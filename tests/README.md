# Tests

Pure logic only (no camera, no Qt window). From the repo root:

    python -m unittest discover -s tests -v

or `pytest tests` if you have pytest. Each file adds `scripts/registration` to the path through `context.py`.

| File | Covers |
|---|---|
| test_error.py | calibrated error radius, grade bands |
| test_selection.py | pedicle selection and stepping |
| test_scene.py | scene JSON: round trip, strict JSON, error summary, version |
| test_jitter.py | rolling jitter: detrending, noise scale, resets |
| test_check.py | click-to-check geometry |
| test_geometry.py | slice_mesh, project_point, zoom_frame |