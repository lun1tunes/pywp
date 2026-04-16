/**
 * fast_replan.js — Client-side analytical trajectory recalculation.
 *
 * Provides instant visual feedback when the user drags control points
 * (t1, t3) in the Three.js 3D viewer.  The result is geometrically
 * approximate — a full Python solver replan should follow on mouse-up.
 *
 * Coordinate convention:
 *   X = East, Y = North, Z = TVD (positive downward in data,
 *   but displayed as -Z in Three.js via Z_DISPLAY_SIGN).
 *
 * Architecture:
 *   1. Python sends the initial trajectory as payload.lines[].
 *   2. On drag, JS calls fastReplan(surface, t1New, t3New, config)
 *      which returns an array of [x, y, z] points (data coords).
 *   3. The caller replaces the BufferGeometry of the target line.
 *   4. On mouse-up, Python re-runs the full solver with the new
 *      t1/t3, and the JS trajectory is replaced by the exact result.
 *
 * Usage (inside viewer_template.html, after scene is initialised):
 *
 *   const replan = new FastReplan({
 *     surface: [surfX, surfY, surfZ],
 *     config: {
 *       entryIncTargetDeg: 86,
 *       maxIncDeg: 95,
 *       dlsBuildMaxDegPer30m: 3.0,
 *       kopMinVerticalM: 550,
 *     },
 *   });
 *
 *   // On drag tick:
 *   const pts = replan.compute([t1x, t1y, t1z], [t3x, t3y, t3z]);
 *   updateLineGeometry(trajectoryLine, pts);
 *
 * STATUS: skeleton — functions contain analytical formulas but are not
 * yet wired into the viewer event loop.  Integration will happen when
 * the interactive control-point editor is implemented.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const DEG2RAD = Math.PI / 180.0;
const RAD2DEG = 180.0 / Math.PI;
const SMALL = 1e-9;

// ---------------------------------------------------------------------------
// Vector helpers (3-element arrays)
// ---------------------------------------------------------------------------
function v3Sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function v3Dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function v3Cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function v3Len(v) {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

function v3Scale(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function v3Add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function v3Normalise(v) {
  const len = v3Len(v);
  return len > SMALL ? v3Scale(v, 1.0 / len) : [0, 0, 1];
}

// ---------------------------------------------------------------------------
// Direction vector from INC/AZI (degrees) — same convention as Python
// [sin(inc)*cos(azi), sin(inc)*sin(azi), cos(inc)]
// This gives X=East component, Y=North component, Z=vertical component.
// ---------------------------------------------------------------------------
function directionVector(incDeg, aziDeg) {
  const incR = incDeg * DEG2RAD;
  const aziR = aziDeg * DEG2RAD;
  return [
    Math.sin(incR) * Math.cos(aziR),
    Math.sin(incR) * Math.sin(aziR),
    Math.cos(incR),
  ];
}

// ---------------------------------------------------------------------------
// Rodrigues rotation: interpolate direction from d1 to d2 at parameter t.
// Returns a unit direction vector.
// ---------------------------------------------------------------------------
function rodriguesInterp(d1, d2, t) {
  const dot = Math.max(-1, Math.min(1, v3Dot(d1, d2)));
  const theta = Math.acos(dot);
  if (theta < 1e-12) {
    return d1.slice();
  }
  const cross = v3Cross(d1, d2);
  const crossLen = v3Len(cross);
  if (crossLen < 1e-10) {
    return d1.slice();
  }
  const rotAxis = v3Scale(cross, 1.0 / crossLen);
  const inPlane = v3Cross(rotAxis, d1);
  const angle = t * theta;
  return v3Normalise(
    v3Add(v3Scale(d1, Math.cos(angle)), v3Scale(inPlane, Math.sin(angle)))
  );
}

// ---------------------------------------------------------------------------
// Minimum curvature position increment (single interval).
// Returns [dN, dE, dTVD].
// ---------------------------------------------------------------------------
function minCurvIncrement(dMD, inc1Deg, azi1Deg, inc2Deg, azi2Deg) {
  const i1 = inc1Deg * DEG2RAD;
  const i2 = inc2Deg * DEG2RAD;
  const a1 = azi1Deg * DEG2RAD;
  const a2 = azi2Deg * DEG2RAD;
  const cosBeta =
    Math.cos(i1) * Math.cos(i2) +
    Math.sin(i1) * Math.sin(i2) * Math.cos(a2 - a1);
  const beta = Math.acos(Math.max(-1, Math.min(1, cosBeta)));
  let rf = 1.0;
  if (Math.abs(beta) > 1e-6) {
    rf = (2.0 / beta) * Math.tan(beta / 2.0);
  }
  const half = (dMD / 2.0) * rf;
  const dN = half * (Math.sin(i1) * Math.cos(a1) + Math.sin(i2) * Math.cos(a2));
  const dE = half * (Math.sin(i1) * Math.sin(a1) + Math.sin(i2) * Math.sin(a2));
  const dZ = half * (Math.cos(i1) + Math.cos(i2));
  return [dN, dE, dZ];
}

// ---------------------------------------------------------------------------
// Fast analytical trajectory: J-profile (VERTICAL → BUILD → HORIZONTAL)
//
// Input:  surface [x,y,z], t1 [x,y,z], t3 [x,y,z], config {}
// Output: array of [x, y, z] points in data coordinates (E, N, TVD)
// ---------------------------------------------------------------------------
function fastReplanPoints(surface, t1, t3, config) {
  const entryIncDeg = config.entryIncTargetDeg || 86.0;
  const maxIncDeg = config.maxIncDeg || 95.0;
  const dlsBuild = config.dlsBuildMaxDegPer30m || 3.0;
  const kopMin = config.kopMinVerticalM || 550.0;
  const mdStep = config.mdStepM || 30.0;

  // --- Section geometry ---
  const dNt1t3 = t3[1] - t1[1]; // North
  const dEt1t3 = t3[0] - t1[0]; // East
  const horizDist = Math.sqrt(dNt1t3 * dNt1t3 + dEt1t3 * dEt1t3);
  if (horizDist < SMALL) {
    return [surface, t1, t3]; // degenerate
  }
  const aziEntryRad = Math.atan2(dEt1t3, dNt1t3);
  const aziEntryDeg = ((aziEntryRad * RAD2DEG) % 360 + 360) % 360;

  // Project surface→t1 onto section plane
  const dNst1 = t1[1] - surface[1];
  const dEst1 = t1[0] - surface[0];
  const cosAz = Math.cos(aziEntryRad);
  const sinAz = Math.sin(aziEntryRad);
  const s1 = dNst1 * cosAz + dEst1 * sinAz; // along-section
  const z1 = t1[2] - surface[2]; // TVD offset

  if (z1 <= 0 || s1 <= SMALL) {
    return [surface, t1, t3]; // infeasible
  }

  // --- J-profile geometry ---
  const incEntryRad = entryIncDeg * DEG2RAD;
  const oneMinusCos = 1.0 - Math.cos(incEntryRad);
  const sinInc = Math.sin(incEntryRad);

  let radiusM, jDls, kopVertical, buildLength;

  if (oneMinusCos > SMALL && s1 > SMALL) {
    radiusM = s1 / oneMinusCos;
    jDls = 30.0 * RAD2DEG / radiusM;
    kopVertical = z1 - radiusM * sinInc;
    buildLength = radiusM * incEntryRad;
  }

  // Fallback if J-profile DLS too high or KOP negative
  if (!radiusM || jDls > dlsBuild * 1.2 || kopVertical < 0) {
    radiusM = 30.0 * RAD2DEG / dlsBuild;
    kopVertical = Math.max(kopMin, z1 - radiusM * sinInc);
    if (kopVertical < 0) kopVertical = 0;
    buildLength = radiusM * incEntryRad;
    jDls = dlsBuild;
  }

  // --- Generate stations ---
  const stations = []; // each: {md, incDeg, aziDeg}
  let md = 0;

  // VERTICAL section
  const vertEnd = Math.max(kopVertical, 0);
  for (let m = 0; m <= vertEnd; m += mdStep) {
    stations.push({ md: m, incDeg: 0, aziDeg: aziEntryDeg });
  }
  if (stations.length === 0 || stations[stations.length - 1].md < vertEnd) {
    stations.push({ md: vertEnd, incDeg: 0, aziDeg: aziEntryDeg });
  }
  md = vertEnd;

  // BUILD section (Rodrigues interpolation at mdStep intervals)
  if (buildLength > SMALL) {
    const d1 = directionVector(0, aziEntryDeg);
    const d2 = directionVector(entryIncDeg, aziEntryDeg);
    const nSteps = Math.max(1, Math.ceil(buildLength / mdStep));
    const dt = 1.0 / nSteps;
    for (let i = 1; i <= nSteps; i++) {
      const t = i * dt;
      const dVec = rodriguesInterp(d1, d2, t);
      const incR = Math.acos(Math.max(-1, Math.min(1, dVec[2])));
      const aziR = Math.atan2(dVec[1], dVec[0]);
      const incDeg = incR * RAD2DEG;
      const aziDeg = ((aziR * RAD2DEG) % 360 + 360) % 360;
      md += buildLength * dt;
      stations.push({ md, incDeg, aziDeg });
    }
  }

  // HORIZONTAL hold (t1 → t3) — straight at entry inclination
  const ds13 =
    (t3[1] - t1[1]) * cosAz +
    (t3[0] - t1[0]) * sinAz;
  const dz13 = t3[2] - t1[2];
  const holdInc = entryIncDeg;
  const sinH = Math.sin(holdInc * DEG2RAD);
  const cosH = Math.cos(holdInc * DEG2RAD);
  let holdLength = ds13 * sinH + dz13 * cosH;
  if (holdLength < 0) holdLength = Math.sqrt(ds13 * ds13 + dz13 * dz13);

  if (holdLength > SMALL) {
    const nSteps = Math.max(1, Math.ceil(holdLength / mdStep));
    const stepLen = holdLength / nSteps;
    for (let i = 1; i <= nSteps; i++) {
      md += stepLen;
      stations.push({ md, incDeg: holdInc, aziDeg: aziEntryDeg });
    }
  }

  // --- Convert MD/INC/AZI → XYZ via minimum curvature ---
  const points = [[surface[0], surface[1], surface[2]]];
  let curN = surface[1];
  let curE = surface[0];
  let curZ = surface[2];

  for (let i = 1; i < stations.length; i++) {
    const prev = stations[i - 1];
    const curr = stations[i];
    const dMD = curr.md - prev.md;
    if (dMD <= 0) continue;
    const [dN, dE, dZ] = minCurvIncrement(
      dMD,
      prev.incDeg,
      prev.aziDeg,
      curr.incDeg,
      curr.aziDeg
    );
    curN += dN;
    curE += dE;
    curZ += dZ;
    points.push([curE, curN, curZ]); // [X=East, Y=North, Z=TVD]
  }

  return points;
}

// ---------------------------------------------------------------------------
// Geometry update helper: replace a THREE.LineSegments buffer in-place.
// ---------------------------------------------------------------------------
function updateLineGeometry(lineObject, points, zDisplaySign) {
  const sign = zDisplaySign || -1.0;
  if (!lineObject || !lineObject.geometry || points.length < 2) return;

  const positions = [];
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    positions.push(prev[0], prev[1], prev[2] * sign);
    positions.push(curr[0], curr[1], curr[2] * sign);
  }

  const attr = lineObject.geometry.getAttribute("position");
  if (attr && attr.count * 3 === positions.length) {
    // Same vertex count — update in place (no allocation).
    attr.array.set(positions);
    attr.needsUpdate = true;
  } else {
    // Different vertex count — recreate attribute.
    lineObject.geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(new Float32Array(positions), 3)
    );
  }
  lineObject.geometry.computeBoundingSphere();
}

// ---------------------------------------------------------------------------
// FastReplan class — holds config, exposes compute().
// ---------------------------------------------------------------------------
class FastReplan {
  /**
   * @param {Object} options
   * @param {number[]} options.surface  [x, y, z] surface location
   * @param {Object}   options.config   Planning config (see top-of-file doc)
   */
  constructor(options) {
    this.surface = options.surface || [0, 0, 0];
    this.config = Object.assign(
      {
        entryIncTargetDeg: 86,
        maxIncDeg: 95,
        dlsBuildMaxDegPer30m: 3.0,
        kopMinVerticalM: 550,
        mdStepM: 30,
      },
      options.config || {}
    );
  }

  /**
   * Compute an analytical trajectory for the given control points.
   *
   * @param {number[]} t1  [x, y, z] entry point
   * @param {number[]} t3  [x, y, z] target point
   * @returns {number[][]} Array of [x, y, z] points in data coordinates
   */
  compute(t1, t3) {
    return fastReplanPoints(this.surface, t1, t3, this.config);
  }
}

// ---------------------------------------------------------------------------
// Export for use inside the viewer IIFE.
// These will be available when this file is injected into the template.
// ---------------------------------------------------------------------------
// window.FastReplan = FastReplan;
// window.updateLineGeometry = updateLineGeometry;
