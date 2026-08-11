# Pedicle target annotations for sub-verse835

## MVP scope

Only pedicle isthmus centre points are defined in this file (10 points: L1-L5, left and right pedicles).

Full trajectory targets (entry, midpoint, tip) are NOT yet defined. These will be added after the first prototype and pipeline are working.

## Rationale

Starting with single points per pedicle enables:
- Direct comparison with baseline (Sachin Daya 2025, cranial)
- Direct comparison with SoTA benchmark (Xu et al. 2020, insertion point error)
- Compatibility with Fitzpatrick TRE formula (single-target evaluation)
- Simpler MVP evaluation before extending scope

## Placement protocol

Isthmus centre defined as the geometric middle of the pedicle bone bridge at its narrowest cross section, verified by three-plane sanity check (axial, sagittal, coronal). Placement done visually per pedicle without enforced left-right symmetry.

## Coordinate system

LPS (Slicer export default). Convert to RAS in Python pipeline if needed by negating L and P components.

## Future work

Extend to 3-point trajectories (entry, midpoint, tip) per pedicle for richer TRE analysis in a second pass. Estimated additional effort: ~1 hour in Slicer, no upstream pipeline changes needed.