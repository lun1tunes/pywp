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
 * The preview is intentionally lightweight: when Python sends the
 * already solved survey stations, the client warps that baseline path
 * to the edited t1/t3 anchors.  This preserves the visual character of
 * the planned trajectory while guaranteeing exact edited endpoints.
 * If baseline stations are unavailable, a smooth cubic fallback is used.
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

function v3Lerp(a, b, t) {
  const k = Math.max(0, Math.min(1, Number(t) || 0));
  return [
    a[0] + (b[0] - a[0]) * k,
    a[1] + (b[1] - a[1]) * k,
    a[2] + (b[2] - a[2]) * k,
  ];
}

function v3Distance(a, b) {
  return v3Len(v3Sub(a, b));
}

function clonePoint(point) {
  return [
    Number(point && point[0]) || 0,
    Number(point && point[1]) || 0,
    Number(point && point[2]) || 0,
  ];
}

function normalisePointList(points) {
  if (!Array.isArray(points)) {
    return [];
  }
  return points
    .filter((point) => Array.isArray(point) && point.length >= 3)
    .map(clonePoint)
    .filter((point) => point.every((value) => Number.isFinite(value)));
}

function pushIfSeparated(points, point) {
  const current = clonePoint(point);
  const previous = points.length > 0 ? points[points.length - 1] : null;
  if (!previous || v3Distance(previous, current) > 1e-6) {
    points.push(current);
  }
}

function nearestPointIndex(points, target) {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  points.forEach((point, index) => {
    const distance = v3Distance(point, target);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function cumulativeLengths(points) {
  const result = [0.0];
  for (let i = 1; i < points.length; i++) {
    result.push(result[i - 1] + v3Distance(points[i - 1], points[i]));
  }
  return result;
}

function cubicBezier(p0, p1, p2, p3, t) {
  const u = 1.0 - t;
  const uu = u * u;
  const tt = t * t;
  const uuu = uu * u;
  const ttt = tt * t;
  return [
    p0[0] * uuu + 3 * p1[0] * uu * t + 3 * p2[0] * u * tt + p3[0] * ttt,
    p0[1] * uuu + 3 * p1[1] * uu * t + 3 * p2[1] * u * tt + p3[1] * ttt,
    p0[2] * uuu + 3 * p1[2] * uu * t + 3 * p2[2] * u * tt + p3[2] * ttt,
  ];
}

function appendBezierSamples(points, p0, p1, p2, p3, sampleCount) {
  const count = Math.max(2, Math.floor(sampleCount || 2));
  for (let i = 1; i <= count; i++) {
    pushIfSeparated(points, cubicBezier(p0, p1, p2, p3, i / count));
  }
}

function endpointExact(points, surface, t1, t3) {
  const cleaned = [];
  pushIfSeparated(cleaned, surface);
  let hasT1 = false;
  points.forEach((point) => {
    if (v3Distance(point, t1) <= 1e-6) {
      hasT1 = true;
    }
    pushIfSeparated(cleaned, point);
  });
  if (!hasT1 && cleaned.length > 1) {
    const insertAt = Math.max(1, nearestPointIndex(cleaned, t1));
    cleaned.splice(insertAt, 0, clonePoint(t1));
  }
  if (cleaned.length < 2 || v3Distance(cleaned[cleaned.length - 1], t3) > 1e-6) {
    pushIfSeparated(cleaned, t3);
  } else {
    cleaned[cleaned.length - 1] = clonePoint(t3);
  }
  cleaned[0] = clonePoint(surface);
  return cleaned;
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
// Fast analytical trajectory preview.
//
// Input:  surface [x,y,z], t1 [x,y,z], t3 [x,y,z], config {}
// Output: array of [x, y, z] points in data coordinates (E, N, TVD)
// ---------------------------------------------------------------------------
function warpedBaselineReplanPoints(surface, t1, t3, config) {
  const basePoints = normalisePointList(config.basePoints);
  const originalT1 = clonePoint(config.originalT1 || t1);
  const originalT3 = clonePoint(config.originalT3 || t3);
  if (basePoints.length < 3) {
    return null;
  }

  const lengths = cumulativeLengths(basePoints);
  const totalLength = lengths[lengths.length - 1] || 0.0;
  if (totalLength <= SMALL) {
    return null;
  }

  const t1Index = nearestPointIndex(basePoints, originalT1);
  const t1Arc = Math.max(lengths[t1Index] || 0.0, totalLength * 0.05, SMALL);
  const tailLength = Math.max(totalLength - t1Arc, SMALL);
  const t1Delta = v3Sub(t1, originalT1);
  const t3Delta = v3Sub(t3, originalT3);
  const warped = [clonePoint(surface)];

  for (let i = 1; i < basePoints.length; i++) {
    let point;
    if (i === t1Index) {
      point = clonePoint(t1);
    } else if (i === basePoints.length - 1) {
      point = clonePoint(t3);
    } else {
      const arc = lengths[i] || 0.0;
      let delta;
      if (arc <= t1Arc) {
        delta = v3Scale(t1Delta, arc / t1Arc);
      } else {
        const u = Math.max(0, Math.min(1, (arc - t1Arc) / tailLength));
        delta = v3Lerp(t1Delta, t3Delta, u);
      }
      point = v3Add(basePoints[i], delta);
    }
    pushIfSeparated(warped, point);
  }

  return endpointExact(warped, surface, t1, t3);
}

function bezierFallbackReplanPoints(surface, t1, t3, config) {
  const mdStep = Math.max(Number(config.mdStepM || 30.0), 5.0);
  const points = [clonePoint(surface)];
  const st1 = v3Sub(t1, surface);
  const t1t3 = v3Sub(t3, t1);
  const horizontalToT1 = Math.hypot(st1[0], st1[1]);
  const tailHorizontal = Math.hypot(t1t3[0], t1t3[1]);
  const entryDir = v3Normalise([
    Math.abs(t1t3[0]) + Math.abs(t1t3[1]) > SMALL ? t1t3[0] : st1[0],
    Math.abs(t1t3[0]) + Math.abs(t1t3[1]) > SMALL ? t1t3[1] : st1[1],
    0,
  ]);
  const depthSign = (t1[2] - surface[2]) >= 0 ? 1 : -1;
  const depthSpan = Math.max(Math.abs(t1[2] - surface[2]), 1.0);
  const firstControl = [
    surface[0],
    surface[1],
    surface[2] + depthSign * Math.min(depthSpan * 0.48, Math.max(depthSpan * 0.22, 420.0)),
  ];
  const entryControlLength = Math.max(horizontalToT1 * 0.34, depthSpan * 0.12, 80.0);
  const secondControl = [
    t1[0] - entryDir[0] * entryControlLength,
    t1[1] - entryDir[1] * entryControlLength,
    t1[2] - depthSign * Math.min(depthSpan * 0.08, 160.0),
  ];
  appendBezierSamples(
    points,
    surface,
    firstControl,
    secondControl,
    t1,
    Math.max(12, Math.ceil((horizontalToT1 + depthSpan) / mdStep)),
  );

  const tailLength = Math.max(v3Distance(t1, t3), tailHorizontal, mdStep);
  const tailControlLength = Math.max(tailHorizontal * 0.33, tailLength * 0.18, 60.0);
  const tailFirstControl = [
    t1[0] + entryDir[0] * tailControlLength,
    t1[1] + entryDir[1] * tailControlLength,
    t1[2] + (t3[2] - t1[2]) * 0.18,
  ];
  const tailSecondControl = [
    t3[0] - entryDir[0] * tailControlLength,
    t3[1] - entryDir[1] * tailControlLength,
    t3[2] - (t3[2] - t1[2]) * 0.18,
  ];
  appendBezierSamples(
    points,
    t1,
    tailFirstControl,
    tailSecondControl,
    t3,
    Math.max(8, Math.ceil(tailLength / mdStep)),
  );

  return endpointExact(points, surface, t1, t3);
}

function legacyJProfileReplanPoints(surface, t1, t3, config) {
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
    return endpointExact([surface, t1, t3], surface, t1, t3); // degenerate
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
    return endpointExact([surface, t1, t3], surface, t1, t3); // infeasible
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

  return endpointExact(points, surface, t1, t3);
}

function fastReplanPoints(surface, t1, t3, config) {
  const safeSurface = clonePoint(surface);
  const safeT1 = clonePoint(t1);
  const safeT3 = clonePoint(t3);
  const safeConfig = config || {};
  const warped = warpedBaselineReplanPoints(safeSurface, safeT1, safeT3, safeConfig);
  if (warped && warped.length >= 3) {
    return warped;
  }
  const smooth = bezierFallbackReplanPoints(safeSurface, safeT1, safeT3, safeConfig);
  if (smooth && smooth.length >= 3) {
    return smooth;
  }
  return legacyJProfileReplanPoints(safeSurface, safeT1, safeT3, safeConfig);
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
        basePoints: [],
        originalT1: null,
        originalT3: null,
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
