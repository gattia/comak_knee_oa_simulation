#!/usr/bin/env python3
"""Turn a COMAK model (with your marker set) into an AddBiomechanics custom skeleton.

    python pipeline/prepare_addbiomechanics_skeleton.py --model MODEL.osim --out SKELETON.osim

Steps (see docs/scaling_comak_model.md for where this sits in the pipeline):

1. ``nsosim.knee_assembly.strip_comak_knee`` removes the COMAK knee bodies, ligaments,
   contact and spanning muscles, and reconnects femur_r -> tibia_r with a PinJoint.
2. The PinJoint is rebuilt to match the input knee's flexion hinge. ``strip_comak_knee``
   (nsosim at the pinned commit) leaves three things for this script to correct:
   - It puts the tibia frame at +6 mm instead of -6 mm along x. It reads the tibia weld
     as if tibia_r were its parent, but COMAK declares that weld tibia_proximal_r ->
     tibia_r. The whole shank and foot end up 12 mm posterior, with every leg length
     intact.
   - It reverses the flexion sign. A PinJoint turns +q about z, but COMAK's knee_flex_r
     turns -q, so positive knee_flex_r would be hyperextension.
   - It leaves the default range of +/-10 rad, where COMAK's is -10 to 160 deg.
   The frames are set to the knee centre as measured on femur_r and tibia_r in the input,
   turned 180 deg about x when the input flexes by -q. The coordinate takes the input's
   range, clamping and default.
3. Every 2-knot ``SimmSpline`` (a straight line) in a joint's transform is replaced by the
   equivalent ``LinearFunction``. The COMAK left leg (knee_l, ankle_l, subtalar_l, mtp_l)
   defines its hinges as y = -x splines. AddBiomechanics' loader (nimblephysics) only
   treats a one-coordinate CustomJoint as a hinge when the function is linear; otherwise
   it fails with "Invalid axis order for Euler joint", which AB reports only as an
   unexplained LoadingError.
4. Meshes AB can load are attached, because a custom skeleton cannot bring its own: AB loads
   meshes only from its Geometry folder (``--ab-geometry``, the AddBiomechanics repo's
   ``server/data/Geometry``), and COMAK's are not there.
   - Feet: AB's ``r_foot``/``r_bofoot`` (and left) on ``calcn_*``/``toes_*``, at the scale of the
     model's own foot meshes. The engine finds the foot contact points through these meshes by
     name. Without them, the dynamics fit crashes (``vector::_M_default_append``).
   - Head: AB's ``hat_skull``/``hat_jaw``, fitted (scale + translation) onto COMAK's own skull and
     jaw on ``neckhead``. They are the same meshes, so the fit is exact. AB measures the generic's
     height from the meshes it can load: without the head and feet it measures 0.58 m, with them
     1.561 m (COMAK's own meshes: 1.562 m). The height prior and AB's first scaling bounds use it.
   Display geometry only: no mass, joint or marker changes.
5. Verified against the input model: every remaining body and every marker is in the same
   place at rest and in a flexed pose (COMAK knee secondary coordinates at zero). Checking
   only against the stripped intermediate would pass a stripping error.

The model is renamed ``<input name>_ab_skeleton``, so AB's copy of it
(``Models/unscaled_generic.osim``) names its source.

The input model must already carry the markers of your motion-capture protocol. Check the
result loads with nimble (separate env, ``pip install nimblephysics``) before uploading:
``nimble.biomechanics.OpenSimParser.parseOsim(SKELETON.osim)``.
"""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import opensim as osim
import pyvista as pv
from nsosim.knee_assembly import strip_comak_knee
from scipy.spatial import cKDTree

TOL = 1e-9  # metres for positions, direction cosines for orientations

# AB's engine can load only meshes in its own Geometry folder (a custom skeleton cannot bring any). It finds the
# foot contact points through these meshes by name (bad_frames_detector/thresholds.py: none found -> the dynamics
# fit crashes), and it measures the generic's height from the meshes it can load (subject.py: getHeight).
# body -> (AB meshes, body whose own mesh scale they take). The feet carry the engine's foot contact points.
# femur_r/tibia_r lose all their meshes with the stripped knee bodies, and AB records a body's scale only in its
# mesh scale_factors, so they get the gait2392 right-leg meshes, at the left leg's scales (same frames).
AB_BODY_MESHES = {
    "calcn_r": (["r_foot.vtp"], "calcn_r"), "toes_r": (["r_bofoot.vtp"], "toes_r"),
    "calcn_l": (["l_foot.vtp"], "calcn_l"), "toes_l": (["l_bofoot.vtp"], "toes_l"),
    "femur_r": (["femur_r.vtp"], "femur_l"), "tibia_r": (["tibia_r.vtp", "fibula_r.vtp"], "tibia_l"),
}
# AB mesh -> the model's own mesh it is fitted onto (scale + translation), on the same body.
AB_HEAD_MESHES = {"hat_skull.vtp": ("neckhead", "skull"), "hat_jaw.vtp": ("neckhead", "jaw")}
FIT_MAX_MM = 1.0  # a fitted AB mesh must lie on the model's own mesh to within this mean distance


def linearise_two_knot_splines(model_el) -> list[str]:
    """Replace each 2-knot SimmSpline inside a TransformAxis with its LinearFunction. Exact."""
    done = []
    for joint in model_el.find(".//JointSet/objects"):
        for axis in joint.iter("TransformAxis"):
            for i, spline in enumerate(list(axis)):
                if spline.tag != "SimmSpline":
                    continue
                xs = [float(v) for v in spline.findtext("x").split()]
                ys = [float(v) for v in spline.findtext("y").split()]
                if len(xs) != 2:
                    continue
                slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
                linear = ET.Element("LinearFunction", {"name": spline.get("name")})
                ET.SubElement(linear, "coefficients").text = f"{slope!r} {ys[0] - slope * xs[0]!r}"
                linear.tail = spline.tail
                axis.remove(spline)
                axis.insert(i, linear)
                done.append(f"{joint.get('name')}/{axis.get('name')}")
    return done


def flatten_multiplier_functions(model_el) -> list[str]:
    """Replace each MultiplierFunction(f, s) with the equivalent function, scaled. Exact.

    AB's joint rationalizer (nimble) does not apply the multiplier. COMAK's left knee keeps its femur length in
    one: translation2 = 0.925 x spline(-0.404) = -0.374 m. Without this step, AB attaches the left tibia at the
    hip (374 mm off), and its scaling then trades femur, tibia and pelvis lengths to compensate.
    """
    done = []
    for joint in model_el.find(".//JointSet/objects"):
        for axis in joint.iter("TransformAxis"):
            for i, mult in enumerate(list(axis)):
                if mult.tag != "MultiplierFunction":
                    continue
                scale = float(mult.findtext("scale"))
                (inner,) = list(mult.find("function"))
                if inner.tag == "Constant":
                    inner.find("value").text = repr(float(inner.findtext("value")) * scale)
                elif inner.tag == "SimmSpline":
                    ys = [float(v) for v in inner.findtext("y").split()]
                    inner.find("y").text = " ".join(repr(v * scale) for v in ys)
                elif inner.tag == "LinearFunction":
                    a, b = (float(v) for v in inner.findtext("coefficients").split())
                    inner.find("coefficients").text = f"{a * scale!r} {b * scale!r}"
                else:
                    raise NotImplementedError(f"{joint.get('name')}/{axis.get('name')}: MultiplierFunction({inner.tag})")
                inner.set("name", mult.get("name", "function"))
                inner.tail = mult.tail
                axis.remove(mult)
                axis.insert(i, inner)
                done.append(f"{joint.get('name')}/{axis.get('name')}")
    return done


def positive_unit_slopes(model_el) -> list[str]:
    """Rewrite each rotation about axis a by -q (LinearFunction slope -1) as +q about -a. Exact.

    AB's joint rationalizer sets such a slope to +1 without flipping the axis, so the joint turns the wrong way:
    COMAK's left knee would bend backwards.
    """
    done = []
    for joint in model_el.find(".//JointSet/objects"):
        for axis in joint.iter("TransformAxis"):
            fn = axis.find("LinearFunction")
            if fn is None or not axis.get("name", "").startswith("rotation"):
                continue
            a, b = (float(v) for v in fn.findtext("coefficients").split())
            if a == -1.0 and b == 0.0:
                fn.find("coefficients").text = "1.0 0.0"
                axis.find("axis").text = " ".join(repr(-float(v) + 0.0) for v in axis.findtext("axis").split())
                done.append(f"{joint.get('name')}/{axis.get('name')}")
    return done


def add_identity_offset_frames(model_el) -> list[str]:
    """Give every CustomJoint socketed directly on its bodies a parent and child offset frame. Exact.

    AB's joint rationalizer rewrites coupled CustomJoints into Rajagopal's form. It subtracts each translation's
    value at q = 0 and makes the first rotation's slope +1, putting both into the joint's offset frames. COMAK's
    knee_l (and pf_l) have no offset frames, so both are lost: the left tibia lands at the hip (374 mm) and the
    left knee bends backwards.
    """
    done = []
    for joint in model_el.find(".//JointSet/objects"):
        if joint.tag != "CustomJoint" or joint.find("frames") is not None:
            continue
        parent, child = joint.findtext("socket_parent_frame"), joint.findtext("socket_child_frame")
        if not (parent.startswith("/bodyset/") and child.startswith("/bodyset/")):
            continue  # e.g. ground -> pelvis
        frames = ET.Element("frames")
        for side, path in (("parent", parent), ("child", child)):
            name = f"{joint.get('name')}_{path.split('/')[-1]}_offset"
            frame = ET.SubElement(frames, "PhysicalOffsetFrame", {"name": name})
            ET.SubElement(frame, "socket_parent").text = path
            ET.SubElement(frame, "translation").text = "0 0 0"
            ET.SubElement(frame, "orientation").text = "0 0 0"
            joint.find(f"socket_{side}_frame").text = name
        coordinates = list(joint).index(joint.find("coordinates"))
        joint.insert(coordinates + 1, frames)
        done.append(joint.get("name"))
    return done


def _body_fixed_xyz(R: np.ndarray) -> list[float]:
    """OpenSim frame orientation (body-fixed X-Y-Z Euler angles) of a rotation matrix."""
    return [float(np.arctan2(-R[1, 2], R[2, 2])), float(np.arcsin(np.clip(R[0, 2], -1, 1))),
            float(np.arctan2(-R[0, 1], R[0, 0]))]


def knee_from_input(model: osim.Model, side: str = "r") -> dict:
    """The input knee's flexion hinge, for rebuilding the stripped PinJoint to match it.

    Returns the knee centre on the femur and on the tibia as (translation, rotation
    matrix), expressed through the weld joints so it does not depend on the pose; the
    flexion slope (+1 or -1 about the joint z axis); and the flexion coordinate's range,
    clamping and default.
    """
    state = model.initSystem()
    model.realizePosition(state)
    knee = osim.CustomJoint.safeDownCast(model.getJointSet().get(f"knee_{side}"))
    flexion = knee.getSpatialTransform().getTransformAxis(0)
    axis = [flexion.getAxis().get(k) for k in range(3)]
    linear = osim.LinearFunction.safeDownCast(flexion.get_function())
    if axis != [0.0, 0.0, 1.0] or linear is None or linear.getIntercept() != 0 or abs(linear.getSlope()) != 1:
        raise NotImplementedError(f"knee_{side} flexion must be +/-q about z to become a PinJoint")
    coordinate = model.getCoordinateSet().get(flexion.getCoordinateNamesInArray().get(0))
    out = dict(
        slope=linear.getSlope(),
        range=[coordinate.getRangeMin(), coordinate.getRangeMax()],
        clamped=coordinate.get_clamped(),
        default=coordinate.getDefaultValue(),
        frames={},
    )
    for frame, body in ((knee.getParentFrame(), f"femur_{side}"), (knee.getChildFrame(), f"tibia_{side}")):
        X = frame.findTransformBetween(state, model.getBodySet().get(body))
        R = np.array([[X.R().get(i, j) for j in range(3)] for i in range(3)])
        out["frames"][f"/bodyset/{body}"] = ([X.p().get(k) for k in range(3)], R)
    return out


def set_knee_hinge(model_el, knee_in: dict, side: str = "r") -> list[str]:
    """Rebuild the knee_<side> PinJoint to match the input knee; return what changed.

    A PinJoint turns +q about its frames' z axis. When the input turns -q (COMAK:
    positive knee_flex_r is flexion), both frames are turned 180 deg about x, so the
    joint axis is -z and the coordinate keeps the input's sign, range and default.
    """
    knee = next(j for j in model_el.find(".//JointSet/objects") if j.get("name") == f"knee_{side}")
    flip = np.diag([1.0, -1.0, -1.0]) if knee_in["slope"] < 0 else np.eye(3)
    changed = []

    def put(el, tag, new: str, unset: str):
        """Set a property tag; `unset` is OpenSim's value when the tag is absent."""
        node = el.find(tag)
        if node is None:
            node = ET.SubElement(el, tag)
            node.text = unset
        if not _same(node.text, new):
            changed.append(f"{el.get('name')}/{tag}: {_show(node.text)} -> {_show(new)}")
        node.text = new

    for offset in knee.find("frames"):
        translation, R = knee_in["frames"][offset.findtext("socket_parent")]
        put(offset, "translation", _fmt(translation), "0 0 0")
        put(offset, "orientation", _fmt(_body_fixed_xyz(R @ flip)), "0 0 0")
    coordinate = knee.find("coordinates/Coordinate")
    put(coordinate, "range", _fmt(knee_in["range"]), "-10 10")
    put(coordinate, "default_value", _fmt([knee_in["default"]]), "0")
    put(coordinate, "clamped", "true" if knee_in["clamped"] else "false", "false")
    return changed


def _fmt(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def _same(a: str, b: str) -> bool:
    a, b = a.split(), b.split()
    try:
        return len(a) == len(b) and np.allclose(np.float64(a), np.float64(b), rtol=0, atol=TOL)
    except ValueError:
        return a == b


def _show(text: str) -> str:
    try:
        return str(np.round(np.float64(text.split()), 4).tolist())
    except ValueError:
        return text.strip()


def _body_meshes(body) -> list:
    """(mesh file, scale factors) of the meshes attached directly to a body."""
    out = []
    for i in range(body.getPropertyByName("attached_geometry").size()):
        mesh = osim.Mesh.safeDownCast(body.get_attached_geometry(i))
        if mesh is not None:
            sf = mesh.get_scale_factors()
            out.append((mesh.get_mesh_file(), np.array([sf.get(k) for k in range(3)])))
    return out


def _fit_scale_translation(a: np.ndarray, b: np.ndarray, iters: int = 50):
    """s, t (no rotation) putting s*a + t onto the surface sampled by b; returns s, t, distances (m)."""
    tree = cKDTree(b)
    s = np.linalg.norm(np.ptp(b, 0)) / np.linalg.norm(np.ptp(a, 0))
    t = b.mean(0) - s * a.mean(0)
    for _ in range(iters):
        nearest = b[tree.query(s * a + t)[1]]
        ac, bc = a - a.mean(0), nearest - nearest.mean(0)
        s = (ac * bc).sum() / (ac * ac).sum()
        t = nearest.mean(0) - s * a.mean(0)
    return s, t, tree.query(s * a + t)[0]


def attach_ab_meshes(model: osim.Model, model_geometry: Path, ab_geometry: Path) -> list[str]:
    """Attach meshes from AB's Geometry folder where the engine needs them; return a report.

    Display geometry only: no mass, joint or marker changes. Feet: AB's foot meshes on the foot bodies, at the
    scale of the model's own foot meshes. They are the gait2392 feet that COMAK's are scaled from. Head: AB's
    skull and jaw, each fitted (scale + translation) onto the model's own skull or jaw, on an offset frame.
    """
    bodies = model.updBodySet()
    report = []
    for body_name, (ab_files, scale_from) in AB_BODY_MESHES.items():
        body = bodies.get(body_name)
        own = _body_meshes(bodies.get(scale_from))
        if not own:
            raise ValueError(f"{scale_from} has no mesh to take a scale from")
        for ab_file in ab_files:
            mesh = osim.Mesh(ab_file)
            mesh.setName(f"ab_{Path(ab_file).stem}")
            mesh.set_scale_factors(osim.Vec3(*own[0][1]))
            body.attachGeometry(mesh)
            report.append(f"{body_name}: {ab_file} at {scale_from}'s mesh scale {np.round(own[0][1], 4).tolist()}")
    for ab_file, (body_name, own_key) in AB_HEAD_MESHES.items():
        body = bodies.get(body_name)
        own = [(f, sf) for f, sf in _body_meshes(body) if own_key in f]
        if len(own) != 1:
            raise ValueError(f"expected one '{own_key}' mesh on {body_name}, found {[f for f, _ in own]}")
        target = pv.read(str(model_geometry / own[0][0])).points * own[0][1]
        s, t, dist = _fit_scale_translation(np.asarray(pv.read(str(ab_geometry / ab_file)).points), target)
        if 1000 * dist.mean() > FIT_MAX_MM:
            raise RuntimeError(f"{ab_file} does not fit {own[0][0]}: mean distance {1000 * dist.mean():.2f} mm")
        frame = osim.PhysicalOffsetFrame(f"{body_name}_ab_{Path(ab_file).stem}", body, osim.Transform(osim.Vec3(*t)))
        mesh = osim.Mesh(ab_file)
        mesh.setName(f"ab_{Path(ab_file).stem}")
        mesh.set_scale_factors(osim.Vec3(s, s, s))
        frame.attachGeometry(mesh)
        body.addComponent(frame)
        report.append(f"{body_name}: {ab_file} fitted onto {own[0][0]} (scale {s:.5f}, offset {np.round(t, 4).tolist()} m; "
                      f"mean {1000 * dist.mean():.2f} mm, max {1000 * dist.max():.2f} mm)")
    return report


def _placements(path: Path, q: dict, bodies: list[str] | None = None) -> dict[str, np.ndarray]:
    """Ground transforms of bodies, and marker positions; every coordinate 0 unless in q."""
    m = osim.Model(str(path))
    s = m.initSystem()
    cs = m.getCoordinateSet()
    for i in range(cs.getSize()):
        cs.get(i).setValue(s, q.get(cs.get(i).getName(), 0.0), False)
    m.realizePosition(s)
    bs = m.getBodySet()
    out = {}
    for n in bodies or [bs.get(i).getName() for i in range(bs.getSize())]:
        X = bs.get(n).getTransformInGround(s)
        R = X.R()
        out[n] = np.r_[[X.p().get(k) for k in range(3)], [R.get(i, j) for i in range(3) for j in range(3)]]
    ms = m.getMarkerSet()
    for i in range(ms.getSize()):
        out["marker " + ms.get(i).getName()] = np.array(
            [ms.get(i).getLocationInGround(s).get(k) for k in range(3)]
        )
    return out


def verify_against(reference: Path, out: Path, poses: list[dict]) -> None:
    """Every body and marker of `out` is placed as in `reference`, at each pose."""
    bodies = [k for k in _placements(out, {}) if not k.startswith("marker ")]
    for q in poses:
        a, b = _placements(reference, q, bodies), _placements(out, q, bodies)
        missing = sorted(set(b) - set(a))
        if missing:
            raise RuntimeError(f"{out} has markers the input lacks: {missing}")
        worst = max(b, key=lambda k: np.abs(a[k] - b[k]).max())
        diff = np.abs(a[worst] - b[worst]).max()
        if diff > TOL:
            raise RuntimeError(f"{worst} differs from the input by {diff:.3g} at pose {q or 'rest'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="COMAK model with your marker set")
    ap.add_argument("--out", required=True, type=Path, help="skeleton .osim to upload to AddBiomechanics")
    ap.add_argument("--ab-geometry", required=True, type=Path,
                    help="AddBiomechanics' Geometry folder (AddBiomechanics repo: server/data/Geometry)")
    args = ap.parse_args()

    model = osim.Model(str(args.model))
    knee_in = knee_from_input(model, side="r")
    name = f"{model.getName()}_ab_skeleton"
    model, _config = strip_comak_knee(model, side="r")
    meshes = attach_ab_meshes(model, args.model.parent / "Geometry", args.ab_geometry)
    model.finalizeConnections()
    model.initSystem()
    with tempfile.TemporaryDirectory() as tmp:
        stripped = Path(tmp) / "stripped.osim"
        model.printToXML(str(stripped))

        tree = ET.parse(stripped)
        model_el = tree.getroot().find("Model")
        model_el.set("name", name)
        corrected = set_knee_hinge(model_el, knee_in, side="r")
        flattened = flatten_multiplier_functions(model_el)
        framed = add_identity_offset_frames(model_el)
        converted = linearise_two_knot_splines(model_el)
        flipped = positive_unit_slopes(model_el)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tree.write(args.out, xml_declaration=True, encoding="UTF-8")

    written = osim.Model(str(args.out))  # keep alive: iterating a temporary's set reads freed memory
    coords = {c.getName() for c in written.getCoordinateSet()}
    # Inside every range: coordinates are clamped, and a clamped value would test nothing.
    flexed = {k: v for k, v in {"hip_flex_r": 0.5, "knee_flex_r": 1.0, "ankle_flex_r": 0.2,
                                "knee_flex_l": 0.9, "ankle_flex_l": 0.3, "subt_angle_l": 0.2,
                                "mtp_angle_l": 0.4}.items() if k in coords}
    verify_against(args.model, args.out, [{}, flexed])

    print(f"wrote {args.out}  (model '{name}')")
    print(f"  knee_r rebuilt to match the input: {'; '.join(corrected) or 'nothing to change'}")
    print(f"  MultiplierFunctions flattened: {len(flattened)}; offset frames added: {', '.join(framed) or 'none'}")
    print(f"  linearised: {', '.join(converted) or 'none'}")
    print(f"  slope -1 rewritten as +1 about the negated axis: {', '.join(flipped) or 'none'}")
    print("  AB meshes:\n    " + "\n    ".join(meshes))
    print("  verified: every body and marker matches the input model at rest and flexed")


if __name__ == "__main__":
    main()
