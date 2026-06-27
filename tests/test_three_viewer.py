from __future__ import annotations

from pathlib import Path
import time

import pywp.three_viewer as three_viewer


def test_viewer_template_contains_safe_custom_3d_controls() -> None:
    html = three_viewer._viewer_template_with_libraries()

    assert 'id="fit-camera-btn"' in html
    assert 'id="fullscreen-btn"' in html
    assert 'id="anti-collision-btn"' in html
    assert 'id="anti-collision-controls"' in html
    assert 'id="world-axis-control-wrap"' in html
    assert 'id="world-axis-btn"' in html
    assert 'id="world-axis-controls"' in html
    assert 'id="world-axis-toggle"' in html
    assert 'id="reference-labels-toggle"' in html
    assert ".toolbar-btn.is-mini" in html
    assert 'data-ac-layer="cones"' in html
    assert 'data-ac-layer="sidetrack_relative_cones"' in html
    assert 'data-ac-layer="overlaps"' in html
    assert 'data-ac-layer="segments"' in html
    assert "function initAntiCollisionControls()" in html
    assert "const WORLD_AXIS_STORAGE_KEY = `pywp:world-axis-overlay:${viewerStateScope}`;" in html
    assert "const WORLD_AXIS_Z_SCALE_STORAGE_KEY = `pywp:world-axis-z-scale:${viewerStateScope}`;" in html
    assert "const REFERENCE_LABELS_STORAGE_KEY = `pywp:reference-labels:${viewerStateScope}`;" in html
    assert "const WORLD_AXIS_PLANE_Z_ROUND_STEP = 100.0;" in html
    assert "const WORLD_AXIS_GRID_SQUARES_PER_SIDE = 4;" in html
    assert "const VIEWER_Z_SCALE_DEFAULT = 1.0;" in html
    assert "const VIEWER_Z_SCALE_MIN = 1.0;" in html
    assert "const VIEWER_Z_SCALE_MAX = 10.0;" in html
    assert "const VIEWER_Z_SCALE_STEP = 0.05;" in html
    assert 'id="world-axis-drag-handle"' in html
    assert 'id="world-axis-z-scale-slider"' in html
    assert 'id="world-axis-z-scale-value"' in html
    assert ".world-axis-handle" in html
    assert ".world-axis-handle-icon" in html
    assert "box-sizing: border-box;" in html
    assert "appearance: none;" in html
    assert "-webkit-appearance: none;" in html
    assert "line-height: 0;" in html
    assert "left: 50%;" in html
    assert "top: 50%;" in html
    assert "pointer-events: none;" in html
    assert ".world-axis-scale-slider" in html
    assert ".world-axis-scale-labels" not in html
    assert 'step="0.05"' in html
    assert "function worldAxisPlaneIntersectionRaw(ndcX, ndcY, rawZ)" in html
    assert "((Number(rawZ) - Number(sceneOrigin.z || 0)) * Z_DISPLAY_SIGN) /" in html
    assert "function normalizedRawBounds(rawBounds)" in html
    assert "function worldAxisDefaultPadFocusId()" in html
    assert "function worldAxisPadFocusIdForFocusId(focusId)" in html
    assert "function worldAxisRawBoundsForPadFocusId(focusId)" in html
    assert "function worldAxisCurrentRawBounds()" in html
    assert "function worldAxisBaseAnchorForBounds(rawBounds)" in html
    assert "function worldAxisResolvedAnchorRaw(xyStep, xTickCount, yTickCount)" in html
    assert "function setWorldAxisActivePadFocus(focusId, options)" in html
    assert "function syncWorldAxisActivePadFocus(options)" in html
    assert "function worldAxisPlaneRawZ(rawBounds)" in html
    assert "function worldAxisTopRawZ(rawBounds)" in html
    assert "function worldAxisRawLevelMatches(leftRawZ, rightRawZ)" in html
    assert "function worldAxisMajorRawZValues(planeRawZ, topRawZ, zStep)" in html
    assert "function worldAxisMinorTickSpecs(planeRawZ, topRawZ, zStep, zMinorSubdivision)" in html
    assert "const deepestRawZ = dataNumber(rawMax[2], 0);" in html
    assert "const roundedRawZ = Math.ceil(deepestRawZ / roundStep) * roundStep;" in html
    assert "roundedRawZ <= deepestRawZ + 1e-6" in html
    assert "const shallowestRawZ = dataNumber(rawMin[2], 0);" in html
    assert "return Math.min(roundedRawZ, 0.0);" in html
    assert "function worldAxisDepthRangeRaw()" in html
    assert "function worldAxisNiceStep(rawValue)" in html
    assert "function formatWorldAxisDistance(rawDistance)" in html
    assert "function worldAxisSubdivisionForPixelSpan(pixelSpan)" in html
    assert "function worldAxisAdaptiveMajorStep(" in html
    assert "function worldAxisAdaptiveZMajorStep(" in html
    assert "const allowedSteps = [10000.0, 5000.0, 2000.0, 1000.0, 500.0, 100.0, 50.0];" in html
    assert "let coarsestWithinCoverage = 0.0;" in html
    assert "let selectedStep = 0.0;" in html
    assert "function worldAxisZScaleFactor()" in html
    assert "function worldAxisZScaleLabel()" in html
    assert "function worldAxisDisplayZ(rawZValue)" in html
    assert "function buildWorldAxisLabelSpecs(" in html
    assert "xyStep," in html
    assert "zStep," in html
    assert "text: formatWorldAxisValue(\n              worldAxisOverlay.anchorRaw.y,\n              xyStep," in html
    assert "text: formatWorldAxisValue(\n              planeRawZ,\n              zStep," in html
    assert "localPosition: new THREE.Vector3(0, 0, 0),\n            offsetX: -14," in html
    assert "localPosition: new THREE.Vector3(0, 0, 0),\n            offsetX: 12," in html
    assert "function rebuildWorldAxisGeometry()" in html
    assert "function updateWorldAxisOverlay()" in html
    assert "function updateWorldAxisDragHandlePosition(" in html
    assert "function startWorldAxisDrag(event)" in html
    assert "function handleWorldAxisDragPointerMove(event)" in html
    assert "function endWorldAxisDrag(event)" in html
    assert "function initWorldAxisControls()" in html
    assert 'activePadFocusId: ""' in html
    assert "dragOffsetRaw: { x: 0.0, y: 0.0 }" in html
    assert "dragging: false" in html
    assert "zScaleFactor: VIEWER_Z_SCALE_DEFAULT," in html
    assert "const unitsPerPixel = worldUnitsPerPixelAt(anchorDisplay);" in html
    assert "syncWorldAxisActivePadFocus({ resetOffset: false });" in html
    assert "const xyCoverageStep = worldAxisNiceStep(" in html
    assert "Math.max(xSpanRaw, ySpanRaw, 40.0) / WORLD_AXIS_GRID_SQUARES_PER_SIDE" in html
    assert "const xyCoverageExtentRaw =" in html
    assert "xyCoverageStep * WORLD_AXIS_GRID_SQUARES_PER_SIDE" in html
    assert "const xyStep = worldAxisAdaptiveMajorStep(" in html
    assert "metrics.unitsPerPixel," in html
    assert "132.0," in html
    assert "100.0," in html
    assert "Math.round(\n              xyCoverageExtentRaw /" in html
    assert "const zCoverageStep = worldAxisNiceStep(" in html
    assert "Math.max(worldAxisDepthRangeRaw() / 5.0, 1000.0)" in html
    assert "const zStep = worldAxisAdaptiveZMajorStep(" in html
    assert "50.0," in html
    assert "worldAxisZScaleFactor()," in html
    assert "const xTickCount = Math.max(" in html
    assert "const yTickCount = Math.max(" in html
    assert "worldAxisOverlay.zScaleFactor = readStoredNumber(" in html
    assert "function readStoredInteger(key, fallback, minValue, maxValue)" in html
    assert "function readStoredNumber(key, fallback, minValue, maxValue)" in html
    assert "function storeStringValue(key, value)" in html
    assert "function sanitizeViewerZScaleFactor(value)" in html
    assert "String(Number(factor.toFixed(2)))" in html
    assert "const zScaleFactor = worldAxisZScaleFactor();" in html
    assert "const xyMajorPixelSpan =" in html
    assert "const zMajorPixelSpan =" in html
    assert "const xyMinorSubdivision = worldAxisSubdivisionForPixelSpan(xyMajorPixelSpan);" in html
    assert "const zMinorSubdivision = worldAxisSubdivisionForPixelSpan(zMajorPixelSpan);" in html
    assert "const xyTickStep = xyStep / xyMinorSubdivision;" in html
    assert "const zMajorRawValues = worldAxisMajorRawZValues(planeRawZ, topRawZ, zStep);" in html
    assert "const zTickSpecs = worldAxisMinorTickSpecs(" in html
    assert "const minorGridPositions = [];" in html
    assert "const tickStep = safeStep / safeSubdivision;" in html
    assert "const safeTopRawZ = Math.min(Number(topRawZ) || 0, 0.0);" in html
    assert "const bottomRawZ =" in html
    assert "Math.floor(Math.max(Number(planeRawZ) || 0, 0.0) / safeStep) * safeStep" in html
    assert "Math.floor(Math.max(Number(planeRawZ) || 0, 0.0) / tickStep) * tickStep" in html
    assert "for (let rawZ = bottomRawZ; rawZ >= safeTopRawZ; rawZ -= safeStep)" in html
    assert "for (let rawZ = bottomRawZ; rawZ >= safeTopRawZ; rawZ -= tickStep)" in html
    assert "values.push(0.0);" in html
    assert "values.push(safeTopRawZ);" in html
    assert "rawZ: safeTopRawZ," in html
    assert "const pushZAxisCoordinateLabels = (" in html
    assert "pushZAxisCoordinateLabels(xMax, yMax, 10, 0, true);" in html
    assert "pushZAxisCoordinateLabels(0, yMax, 14, 0, true);" in html
    assert "showZStepLabels" not in html
    assert "pushZAxisStepLabels" not in html
    assert "const pushZAxisTitleLabel = (axisLocalX, axisLocalY, offsetX, offsetY) => {" in html
    assert "pushZAxisTitleLabel(xMax, yMax, 8, 0);" in html
    assert "pushZAxisTitleLabel(0, yMax, 16, 0);" in html
    assert "const zDisplayLength =" in html
    assert "const zLength = Math.max(Number(planeRawZ) - Number(topRawZ), 0.0);" in html
    assert "const approxZPixelLength =" in html
    assert "const zLabelEvery = Math.max(" in html
    assert "zMajorRawValues.forEach((rawZValue, zIndex) => {" in html
    assert "const isTop = worldAxisRawLevelMatches(rawZValue, topRawZ);" in html
    assert "const isZero = worldAxisRawLevelMatches(rawZValue, 0.0);" in html
    assert "const distanceFromPlane = Number(planeRawZ) - Number(rawZValue);" in html
    assert "worldAxisOverlay.labelSpecs = labelSpecs;" in html
    assert "worldAxisDisplayZ(-zMax)," in html
    assert "offsetY: offsetY - 10," in html
    assert 'color: "#6b7ea6"' in html
    assert "const xyStepLabel = formatWorldAxisDistance(xyStep);" in html
    assert "text: formatWorldAxisDistance(segmentDistance)," not in html
    assert "[0, 0, 0]," in html
    assert "[0, 0, worldAxisDisplayZ(-zLength)]," in html
    assert "x: (dataNumber(bounds.min[0], 0) + dataNumber(bounds.max[0], 0)) * 0.5," in html
    assert "y: (dataNumber(bounds.min[1], 0) + dataNumber(bounds.max[1], 0)) * 0.5," in html
    assert "x: centeredX - (Number(xyStep) * Number(xTickCount)) * 0.5," in html
    assert "y: centeredY - (Number(xyStep) * Number(yTickCount)) * 0.5," in html
    assert "const gridAnchorRaw = worldAxisResolvedAnchorRaw(" in html
    assert "[xValue, yMax, 0]," in html
    assert "[xValue, yMax, worldAxisDisplayZ(-zLength)]," in html
    assert "[0, yValue, 0]," in html
    assert "[0, yValue, worldAxisDisplayZ(-zLength)]," in html
    assert "if (tickSpec.major) {" in html
    assert "pushWorldAxisSegment(\n              minorGridPositions," in html
    assert "Number(rawZValue) < 0.0 &&" in html
    assert "!worldAxisRawLevelMatches(rawZValue, 0.0)" in html
    assert "[xMin, yMax, zValue]," in html
    assert "[xMax, yMax, zValue]," in html
    assert "[0, yMin, zValue]," in html
    assert "[0, yMax, zValue]," in html
    assert "[xValue, yMax - tickHalf, 0]," in html
    assert "[-tickHalf, yValue, 0]," in html
    assert "zTickSpecs.forEach((tickSpec) => {" in html
    assert "[-zTickHalf, 0, worldAxisDisplayZ(-distanceFromPlane)]," in html
    assert "[xMax - zTickHalf, yMax, worldAxisDisplayZ(-distanceFromPlane)]," in html
    assert "[0, yMax - zTickHalf, worldAxisDisplayZ(-distanceFromPlane)]," in html
    assert 'createWorldAxisLineSegments(minorGridPositions, "#9aa3af", 0.12, 1.4)' in html
    assert "focusViewerTarget(targetBounds, String(item.id || \"\"));" in html
    assert "window.addEventListener(\"pointermove\", handleWorldAxisDragPointerMove, true);" in html
    assert "displayPoint([\n            worldAxisOverlay.anchorRaw.x,\n            worldAxisOverlay.anchorRaw.y,\n            0.0,\n          ]).project(camera);" in html
    assert "function syncWorldAxisScaleControls()" in html
    assert "function syncOptionalReferenceLabelsVisibility()" in html
    assert "function applyViewerZScaleFactor(value)" in html
    assert "function rescaleViewerSceneZ(ratio)" in html
    assert "function objectLocksViewerZGeometryScale(object)" in html
    assert "worldAxisZScaleSlider.addEventListener(\"input\", () => {" in html
    assert "applyViewerZScaleFactor(worldAxisZScaleSlider.value);" in html
    assert "syncWorldAxisScaleControls();" in html
    assert "function registerAntiCollisionVisualObject(object, itemOrRole)" in html
    assert 'role === "sidetrack_relative_cone"' in html
    assert 'role === "sidetrack_relative_cone_tip"' in html
    assert 'return "sidetrack_relative_cones";' in html
    assert 'role === "overlap" || role === "overlap_volume"' in html
    assert 'role === "conflict_segment" || role === "conflict_hover"' in html
    assert "function antiCollisionLayerForItem(itemOrRole)" in html
    assert 'name.includes("конфликт") || name.includes("conflict")' in html
    assert "function addOverlapVolumeMesh(item)" in html
    assert "function resampledRingPoint(ring, sampleIndex, sampleCount)" in html
    assert "registerAntiCollisionVisualObject(mesh, item)" in html
    assert "antiCollisionVisualObjects[layer].forEach" in html
    assert 'id="legend-toggle-btn"' in html
    assert 'id="tooltip"' in html
    assert 'id="axes-gizmo"' in html
    assert "#root canvas" in html
    assert "width: 100% !important;" in html
    assert "height: 100% !important;" in html
    assert 'id="label-layer"' in html
    assert ".scene-label" in html
    assert ".scene-label.well-name-label" in html
    assert "max-width: min(220px, calc(100vw - 24px));" in html
    assert "overflow-wrap: anywhere;" in html
    assert "white-space: normal;" in html
    assert "transform: translate(8px, -50%);" in html
    assert "contain: layout paint;" in html
    assert 'labelRole === "well_label" ||' in html
    assert 'labelRole === "pilot_point_label" ||' in html
    assert 'labelRole === "reference_label"' in html
    assert 'labelRole === "reference_label_optional"' in html
    assert 'role === "target_label" || role === "control_point_label"' in html
    assert "font-size: 10px;" in html
    assert ".axes-gizmo-line" in html
    assert ".axes-gizmo-label" in html
    assert "OrbitControls" in html
    assert "controls.rotateSpeed = 0.72;" in html
    assert "controls.zoomSpeed = 0.95;" in html
    assert "controls.panSpeed = 0.47;" in html
    assert "controls.enableDamping = true;" in html
    assert "controls.dampingFactor = 0.12;" in html
    assert "controls.screenSpacePanning = true;" in html
    assert "function updateLabels()" in html
    assert "updateWorldAxisOverlay();" in html
    assert "offsetY: Number((options && options.offsetY) || 0)" in html
    assert 'role === "well_label" ||' in html
    assert 'role === "pilot_point_label" ||' in html
    assert 'role === "reference_label"' in html
    assert 'role === "reference_label_optional"' in html
    assert 'role === "reference_pad_label"' in html
    assert "item.offsetY" in html
    assert "renderer.domElement.getBoundingClientRect()" in html
    assert "labelLayerElement.getBoundingClientRect()" in html
    assert "transform-origin: 50% 100%;" in html
    assert "position.z += labelLift;" not in html
    assert "align-items: center;" in html
    assert "justify-content: center;" in html
    assert 'xmlns="http://www.w3.org/2000/svg"' in html


def test_viewer_template_shows_xyz_hover_for_edit_handles() -> None:
    html = three_viewer._viewer_template_with_libraries()

    assert 'function editHandleHoverData(handle)' in html
    assert '<strong>X:</strong>' in html
    assert '<strong>Y:</strong>' in html
    assert '<strong>Z:</strong>' in html
    assert "pickMesh.userData.editHandleIndex = handleIndex;" in html
    assert "hoverTargets.push(pickMesh);" in html
    assert "const editHandleIndex = Number(hoverObject.userData.editHandleIndex);" in html
    assert "? editHandleHoverData(editHandles[editHandleIndex])" in html
    assert "✋" not in html
    assert "X / East" in html
    assert "Y / North" in html
    assert 'label: "Z"' in html
    assert "Z / TVD" not in html
    assert 'color: "#16A34A"' in html
    assert 'color: "#2563EB"' in html
    assert 'color: "#DC2626"' in html
    assert "scene-axis-label" in html
    assert "function updateAxesGizmo()" in html
    assert "camera.quaternion.clone().invert()" in html
    assert 'Math.abs(dx) < 6 ? "middle"' in html
    assert "#legend.is-collapsed" in html
    assert "#legend {" in html
    assert ".legend-swatch.is-point" in html
    assert 'String(symbol || "line").toLowerCase() === "point"' in html
    assert "item.symbol" in html
    assert "function lineDashPattern(item)" in html
    assert "dashSize: Math.max(maxSpan * 0.006, 6.0)" in html
    assert "gapSize: Math.max(maxSpan * 0.0012, 1.6)" in html
    assert "#collisions-panel {" in html
    assert "z-index: 8;" in html
    assert "#minimap-control.is-expanded #minimap-ruler-btn.is-active {" in html
    assert "#minimap-control.is-expanded #minimap-ruler-btn.is-active:hover {" in html
    assert "function syncLegendVisibility()" in html
    assert "payload.legend_tree" in html
    assert "payload.focus_targets" in html
    assert "function fitCameraToRawBounds(rawBounds)" in html
    assert "const DEFAULT_FIT_LEFT_BIAS_RATIO = 0.12;" in html
    assert "function rawBoundsMatch(leftRawBounds, rightRawBounds)" in html
    assert "const tolerance = 0.001;" in html
    assert "function sharedSinglePadFitBiasRatio(padFocusId, rawBounds)" in html
    assert "return DEFAULT_FIT_LEFT_BIAS_RATIO * 0.5;" in html
    assert "function resolvedDefaultFitBiasRatio()" in html
    assert "function applyCameraHorizontalBias(horizontalRatio)" in html
    assert "crossVectors(viewDirection, camera.up)" in html
    assert "const fovRad = THREE.MathUtils.degToRad(camera.fov);" in html
    assert "applyCameraHorizontalBias(resolvedDefaultFitBiasRatio());" in html
    assert "const sharedBiasRatio = sharedSinglePadFitBiasRatio(padFocusId, rawBounds);" in html
    assert "sceneOrigin = rawBoundsCenter(payload.bounds || {})" in html
    assert "function displayPoint(value)" in html
    assert "function displayXYZ(xValue, yValue, zValue)" in html
    assert "function dataPointFromDisplay(displayPosition)" in html
    assert "dataNumber(value && value[0], 0) - sceneOrigin.x" in html
    assert "dataNumber(value && value[1], 0) - sceneOrigin.y" in html
    assert "((dataNumber(value && value[2], 0) - sceneOrigin.z) * Z_DISPLAY_SIGN) /" in html
    assert "dataNumber(displayPosition && displayPosition.x, 0) + sceneOrigin.x" in html
    assert "dataNumber(displayPosition && displayPosition.y, 0) + sceneOrigin.y" in html
    assert "(dataNumber(displayPosition && displayPosition.z, 0) * zScaleFactor) *" in html
    assert "function updateLineGeometry(lineObject, points, zDisplaySign, displayOrigin, zScaleFactor)" in html
    assert "const originX = Number.isFinite(Number(origin.x)) ? Number(origin.x) : 0.0;" in html
    assert "const scaleFactor = Math.max(Number(zScaleFactor) || 1.0, 1.0);" in html
    assert "updateLineGeometry(\n                editLineObjects[wellIndex],\n                points,\n                Z_DISPLAY_SIGN,\n                sceneOrigin,\n                worldAxisZScaleFactor()," in html
    assert "updateLineGeometry(\n            editLineObjects[wellIndex],\n            pts,\n            Z_DISPLAY_SIGN,\n            sceneOrigin,\n            worldAxisZScaleFactor()," in html
    assert "fitMiniMapToRawBounds(rawBounds || payload.bounds || {})" in html
    assert "function focusMiniMapToDisplayPoint(displayCenter, focusSpan)" in html
    assert "focusMiniMapToDisplayPoint(targetCenter, offsetDistance * 2.0)" in html
    assert "const forcePan = Number(event.button || 0) === 2;" in html
    assert "if (!forcePan && miniMapRulerState.enabled) {" in html
    assert "if (!forcePan && startMiniMapEditDrag(event)) {" in html
    assert "renderer.domElement.addEventListener(\"contextmenu\", (event) => {" in html
    assert "let miniMapOverlayBuilt = false;" in html
    assert "function buildMiniMapOverlaysFromPayload()" in html
    assert "if (!miniMapOverlayBuilt) {" in html
    assert "buildMiniMapOverlaysFromPayload();" in html
    assert "function editWellTargetFocusBounds(index)" in html
    assert "function fitCameraToEditWellTargets(index)" in html
    assert "selectEditWell(editableIndex, { focus: !editModeActive })" in html
    assert "row.dataset.editWellIndex = String(editableIndex)" in html
    assert "targetBounds && !editModeActive" in html
    assert "fitCameraToEditWellTargets(nextIndex)" in html
    assert "data-edit-well-index" in html
    assert "function selectEditWell(index, options)" in html
    assert "function sendEditTargetsToStreamlit(changes)" in html
    assert "font-size: 15px;" in html
    assert "const originalMesh = new THREE.Mesh(handleGeometry, originalMat);" in html
    assert "originalMesh.position.copy(displayPoint(point));" in html
    assert "pointMesh.userData.lockViewerZGeometryScale = true;" in html
    assert "previewMesh.userData.lockViewerZGeometryScale = true;" in html
    assert "originalMesh.userData.lockViewerZGeometryScale = true;" in html
    assert "pickMesh.userData.lockViewerZGeometryScale = true;" in html
    assert "h.originalMesh.visible =" in html
    assert "material.depthTest = applies && selectedWellDirty ? false : item.depthTest;" in html
    assert 'id="edit-toolbox"' in html
    assert "editToolbox.classList.toggle" in html
    assert "function initEditToolboxDrag()" in html
    assert 'data-plane="xy"' in html
    assert 'data-plane="x"' not in html
    assert 'data-plane="y"' not in html
    assert 'id="edit-scope-selector"' in html
    assert 'data-scope="point"' in html
    assert 'data-scope="pair"' in html
    assert "function setEditMoveScope(scope)" in html
    assert 'let editMoveScope = "point";' in html
    assert 'title="Изменить точки скважины"' in html
    assert 'data-scope="pair" type="button">Все</button>' in html
    assert 'id="edit-operation-selector"' in html
    assert 'data-operation="move"' in html
    assert 'data-operation="rotate"' in html
    assert "function setEditTransformMode(operation)" in html
    assert 'let editTransformMode = "move";' in html
    assert "function createEditRotationControl(wellIndex, color)" in html
    assert "function createRotationArrowGeometry(angleRad, direction)" in html
    assert "const turnDirection = Number(direction) < 0 ? -1.0 : 1.0;" in html
    assert "[125, -1]" in html
    assert "multiplyScalar(0.042)" in html
    assert "function startEditRotationDrag(wellIndex, event)" in html
    assert "function updateEditRotationDrag(event)" in html
    assert "let editRotateStartPreviewPoints = null;" in html
    assert "function rotatedPreviewTrajectoryPoints" in html
    assert "function rebuildRotatedPreviewTrajectory" in html
    assert "editRotateStartPreviewPoints = editReplanners[wellIndex].compute" in html
    assert "rebuildRotatedPreviewTrajectory(" in html
    assert "editRotationModeEnabled()" in html
    assert "editRotationPickMeshes" in html
    assert "function editPointDefinitionsForWell(well)" in html
    assert "const editOriginalPoints = []" in html
    assert "function updateWellEditPoints(wellIndex, nextPoints)" in html
    assert "editWellHasExplicitPoints(wellIndex)" in html
    assert "editWellHasComplexExplicitPoints(wellIndex)" in html
    assert "editWellUsesSidetrackWindowPreview(wellIndex)" in html
    assert "function editWellSupportsRotation(wellIndex)" in html
    assert "sidetrackWindowEntryForWell(wellIndex)" in html
    assert "editWellSupportsRotation(selectedEditWellIndex)" in html
    assert "!editWellSupportsRotation(selectedEditWellIndex)" in html
    assert "const targetPoints = pointEntries.filter" in html
    assert "change.points = targetPoints.map" in html
    assert 'editDragMoveScope === "pair"' in html
    assert "updateWellEditTargets(wi, nextT1, nextT3)" in html
    assert 'id="edit-undo-btn"' in html
    assert 'id="edit-redo-btn"' in html
    assert 'id="edit-reset-btn"' in html
    assert 'id="edit-warning-hide-btn"' in html
    assert "editWarningDismissed = true" in html
    assert "editHasAnyChange && !editWarningDismissed" in html
    assert "function undoSelectedEdit()" in html
    assert "function redoSelectedEdit()" in html
    assert "function resetSelectedEdit()" in html
    assert "editDragHistoryStartState = cloneEditState" in html
    assert "function pointerPlaneIntersection(event, displayZ)" in html
    assert "LineDashedMaterial" in html
    assert "edit-delta-label" in html
    assert "background: rgba(255,255,255,0.18);" in html
    assert "box-shadow: none;" in html
    assert "text-shadow:" in html
    assert "initEditDeltaLabelDrag" in html
    assert "formatDeltaMeters" in html
    assert "formatLengthMeters" in html
    assert 'toFixed(1).replace(".", ",")' in html
    assert "edit-lateral-label" in html
    assert "horizontalT1T3Length" in html
    assert "function t1T3Length(wellIndex)" in html
    assert "Math.hypot(t3[0] - t1[0], t3[1] - t1[1], t3[2] - t1[2])" in html
    assert "`Длина ГС ${formatLengthMeters(t1T3Length(wellIndex))}`" in html
    assert "syncEditLateralLabels" in html
    assert "function worldUnitsPerPixelAt(displayPosition)" in html
    assert "function editHandleScale(displayPosition, pixelDiameter)" in html
    assert "camera.position.distanceTo(worldPosition)" in html
    assert "h.mesh.scale.setScalar(editHandleScale(displayPosition, 16.0))" in html
    assert "pickMesh" in html
    assert "previewMesh" in html
    assert "function handleDeltaLength(handle)" in html
    assert "lineObject.visible = editModeActive && isDirty" in html
    assert "BroadcastChannel" in html
    assert "navigator.clipboard" not in html
    assert "window.parent.location" not in html
    assert "function editableIndexForBaseName(nameValue)" in html
    assert "registerEditableBaseMaterial(material, colorValue, nameValue)" in html
    assert "itemWellIndex === selectedWellIndex" in html
    assert "basePoints: Array.isArray(well.base_points)" in html
    assert "warpedBaselineReplanPoints" in html
    assert "return endpointExact(warped, surface, t1, t3);" in html
    assert "function applyViewerPayload(nextPayload, options)" in html
    assert "function applyAntiCollisionOverlayPayload(nextPayload)" in html
    assert "window.__PYWP_VIEWER_UPDATE__ = function (nextPayloadJson, nextPayloadDigest)" in html
    assert "window.__PYWP_VIEWER_APPLY_ANTICOLLISION_OVERLAY__ = function (" in html
    assert "clearViewerDataObjects();" in html
    assert "clearAntiCollisionVisualPayload();" in html
    assert "camera.position.add(originDelta);" in html
    assert "controls.target.add(originDelta);" in html
    assert "sidetrack_window" in html
    assert "nearestSidetrackParentPoint" in html
    assert "snappedSidetrackWindowEntry" in html
    assert "change.sidetrack_window" in html
    assert "sidetrackMdDeltaRow" in html
    assert "<span>ΔMD</span>" in html
    assert "Свернуть / развернуть" in html
    assert "legend-node-btn legend-node-pad" in html
    assert "legend-node-btn legend-node-well" in html
    assert "legend-item.is-focusable" in html
    assert 'id="minimap-frame"' in html
    assert 'id="minimap-label-layer"' in html
    assert 'id="minimap-ruler-layer"' in html
    assert 'id="minimap-ruler-line"' in html
    assert 'id="minimap-ruler-start"' in html
    assert 'id="minimap-ruler-end"' in html
    assert 'id="minimap-ruler-label"' in html
    assert ".minimap-well-label" in html
    assert ".minimap-edit-delta-label" in html
    assert ".minimap-edit-lateral-label" in html
    assert ".minimap-ruler-line" in html
    assert ".minimap-ruler-point" in html
    assert ".minimap-ruler-label" in html
    assert 'id="minimap-axes"' in html
    assert "minimap-axis-x" in html
    assert "minimap-axis-y" in html
    assert "X / E" in html
    assert "Y / N" in html
    assert "const miniMapRulerLayer = document.getElementById(\"minimap-ruler-layer\")" in html
    assert "const miniMapRulerBtn = document.getElementById(\"minimap-ruler-btn\")" in html
    assert "const miniMapAxes = document.getElementById(\"minimap-axes\")" in html
    assert "const miniMapLabels = [];" in html
    assert "miniMapAxes.classList.add(\"is-visible\")" in html
    assert "function updateMiniMapLabels()" in html
    assert "function updateMiniMapRulerOverlay()" in html
    assert 'labelRole === "reference_label" ||' in html
    assert 'labelRole === "reference_label_optional" ||' in html
    assert "function handleMiniMapRulerClick(event)" in html
    assert "function setMiniMapRulerEnabled(enabled)" in html
    assert "function syncMiniMapRulerButton()" in html
    assert "function miniMapRulerRawPointAtEvent(event)" in html
    assert "function miniMapRulerDisplayPoint(rawPoint)" in html
    assert "function miniMapRulerCurrentEndRaw()" in html
    assert "function miniMapRulerDistanceMeters(startRaw, endRaw)" in html
    assert "function formatMiniMapDistanceMeters(distanceMeters)" in html
    assert "function resetMiniMapRulerMeasurement()" in html
    assert "for (let index = miniMapLabels.length - 1; index >= 0; index -= 1) {" in html
    assert "(item && item.sourceLabel && item.sourceLabel.role)" in html
    assert 'labelRole === "edit_delta_label"' in html
    assert 'labelRole === "edit_lateral_label"' in html
    assert "{ offsetY: offsetY, role: role }" in html
    assert "function syncEditWellNameLabelPosition(wellIndex)" in html
    assert 'id="minimap-control"' in html
    assert 'id="minimap-label"' in html
    assert 'id="minimap-ruler-btn"' in html
    assert 'class="minimap-ruler-icon"' in html
    assert "План E-N (вид сверху)" in html
    assert "const miniMapControl = document.getElementById(\"minimap-control\")" in html
    assert "miniMapControl.classList.add(\"is-expanded\")" in html
    assert "miniMapControl.style.width" in html
    assert "miniMapLabel.classList.add(\"is-expanded\")" in html
    assert "miniMapControl.style.left" in html
    assert "pointer-events: none;" in html
    assert "pointer-events: auto;" in html
    assert "screenX >= miniMapLeft" in html
    assert "#minimap-frame" in html
    assert "border: 1px solid rgba(15,23,42,0.16);" in html
    assert "background: rgba(255,255,255,0.035);" in html
    assert "background: transparent;" in html
    assert "box-shadow: none;" in html
    assert "text-shadow: 0 1px 2px rgba(255,255,255,0.95);" in html
    assert 'id="minimap-toggle-btn"' in html
    assert ">План E-N</button>" in html
    assert "initWorldAxisControls();" in html
    assert "План E-N ‹" not in html
    assert "#minimap-control:not(.is-expanded) #minimap-toggle-btn::before" in html
    assert 'content: "‹";' in html
    assert "font-weight: 500;" in html
    assert "#minimap-control:not(.is-expanded) #minimap-label" in html
    assert "#minimap-control.is-expanded #minimap-ruler-btn" in html
    assert "#minimap-control.is-expanded #minimap-ruler-btn.is-active" in html
    assert ".minimap-ruler-icon" in html
    assert "stroke: currentColor;" in html
    assert "max-width: calc(100% - 84px);" in html
    assert 'miniMapToggleBtn.textContent = "План E-N";' in html
    assert "new THREE.OrthographicCamera" in html
    assert "function miniMapCanvasRect()" in html
    assert "function miniMapSizeConfig(canvasWidth, canvasHeight)" in html
    assert "function resizedMiniMapSize(startRect, event)" in html
    assert "function syncMiniMapResizeZoom(rect)" in html
    assert "function handleMiniMapResizePointerDown(event)" in html
    assert "function handleMiniMapResizePointerMove(event)" in html
    assert "function handleMiniMapResizePointerUp(event)" in html
    assert "miniMapState.customSize = resizedMiniMapSize" in html
    assert "miniMapState.resizeStartWorldPerPixel = miniMapWorldPerPixel()" in html
    assert "cursor: nwse-resize;" in html
    assert "border-top: 18px solid rgba(71,85,105,0.62);" in html
    assert "Math.sqrt(targetArea * aspect) * 1.2" in html
    assert "renderer.setScissorTest(true)" in html
    assert "const miniMapHiddenContourObjects = [];" in html
    assert "function renderMiniMapScene()" in html
    assert "registerMiniMapHiddenContourObject(wireframe)" in html
    assert "const miniMapOverlayScene = new THREE.Scene();" in html
    assert "function addMiniMapTrajectoryOverlay(item)" in html
    assert 'const isConflictSegment = role === "conflict_segment";' in html
    assert 'role === "line" || role === "conflict_segment"' in html
    assert 'const dashKind = String((item && item.dash) || "solid").toLowerCase();' in html
    assert "const material = new THREE.LineDashedMaterial({" in html
    assert "line.computeLineDistances();" in html
    assert "mesh.renderOrder = isConflictSegment ? 8 : 0;" in html
    assert "lineSegments.renderOrder = isConflictSegment ? 9 : 0;" in html
    assert "miniMapTrajectoryOverlayWidth() * 1.2" in html
    assert "mesh.renderOrder = 12;" in html
    assert "function addMiniMapConeOverlay(item, sourceGeometry)" in html
    assert 'String((item && item.role) || "") !== "cone"' in html
    assert "addMiniMapConeOverlay(item, geometry)" in html
    assert "mesh.renderOrder = -10;" in html
    assert "function addMiniMapMarkerOverlay(item)" in html
    assert "renderer.clearDepth();" in html
    assert "renderer.render(miniMapOverlayScene, miniMapCamera);" in html
    assert "depthTest: false,\n            depthWrite: false," in html
    assert "role === \"cone_tip\"" in html
    assert "renderMiniMapScene();" in html
    assert "handleMiniMapWheel" in html
    assert "handleMiniMapPointerDown" in html
    assert "isPointerInMiniMap(event)" in html
    assert "function miniMapEditableHandleAtEvent(event)" in html
    assert "function startMiniMapEditDrag(event)" in html
    assert 'editDragInputMode = "minimap"' in html
    assert "editTransformMode === \"rotate\"" in html
    assert "miniMapWorldAtEvent(event)" in html
    assert 'role: "edit_delta_label"' in html
    assert 'role: "edit_lateral_label"' in html
    assert "item.sourceLabel || null" in html
    assert 'item.role === "edit_delta_label"' in html
    assert 'miniMapRulerState.phase = "measuring";' in html
    assert 'miniMapRulerState.phase = "fixed";' in html
    assert "setMiniMapRulerEnabled(false);" in html
    assert "miniMapRulerBtn.addEventListener(\"click\", (event) => {" in html
    assert "handleMiniMapRulerClick(event);" in html
    assert "function sortedLegendItems(items)" in html
    assert '.localeCompare(legendSortLabel(right), "ru"' in html
    assert "numeric: true" in html
    assert "sortedLegendItems(group.children)" in html
    assert "sortedLegendItems(flatItems)" in html
    assert "pad-first-surface-label" not in html
    assert 'role === "pad_first_surface_label"' not in html
    assert "function ensureCircleMarkerTexture()" in html
    assert "new THREE.CanvasTexture(canvas)" in html
    assert "new THREE.PointsMaterial" in html
    assert "function addPolylineTube()" in html
    assert "new THREE.MeshLambertMaterial({" in html
    assert "const depthMaterial = isConflictSegment" in html
    assert "colorWrite: false" in html
    assert "depthMesh.renderOrder = -1" in html
    assert "Math.max(maxSpan * 0.00032, 0.9)" in html
    assert "sceneComplexity > 90000 || pointCount > 2200" in html
    assert "new THREE.AmbientLight(0xffffff, 0.72)" in html
    assert "new THREE.DirectionalLight(0xffffff, 0.58)" in html
    assert "THREE.MeshLambertMaterial" in html
    assert "THREE.WireframeGeometry" in html
    assert "new THREE.LineSegments(wireGeometry, wireMaterial)" in html
    assert 'depthWrite: role === "cone"' not in html
    assert "mesh.renderOrder = role === \"overlap\" ? 4 : isConeSurface ? 1 : 2" in html
    assert "mesh.renderOrder = 4;" in html
    assert "depthWrite: !isConeTip && !isConflictSegment" in html
    assert "wireframe.renderOrder = mesh.renderOrder + 0.05" in html
    assert 'role === "cone" || role === "sidetrack_relative_cone"' in html
    assert 'sidetrack_relative_cones: []' in html
    assert "Math.min(opacity, 0.10)" in html
    assert "opacity: 0.07" in html
    assert 'x: "#16A34A"' in html
    assert 'y: "#2563EB"' in html
    assert 'z: "#DC2626"' in html
    assert "function numberOrDefault(value, fallback)" in html
    assert "numberOrDefault(item.opacity, 0.34)" in html
    assert "numberOrDefault(item.opacity, 0.25)" in html
    assert "markerScale * 0.52" in html
    assert "function addHoverPointCloud(points, item, hoverItems)" in html
    assert 'const defaultHoverColor = hexOrDefault(item.color, "#64748b")' in html
    assert "{ color: defaultHoverColor }" in html
    assert "mesh.userData.lockViewerZGeometryScale = true;" in html
    assert "pointCloud.userData.hover = { color: defaultHoverColor }" in html
    assert "pointCloud.userData.hoverItems = itemHoverData" in html
    assert "raycaster.params.Points.threshold = Math.max(worldMarkerSize * 0.55, 3.0)" in html
    assert "indexedHoverItems && Number.isInteger(intersections[0].index)" in html
    assert "const labelOffsetX = 6;" in html
    assert "rect.width - labelWidth - padding - labelOffsetX" in html
    assert "max-width: min(160px, calc(100% - 16px));" in html
    assert "max-width: calc(100% - 84px);" in html
    assert "<strong>DLS:</strong>" in html
    assert "<strong>INC:</strong>" in html
    assert 'id="reset-camera-btn"' not in html
    assert ">Anti-collision</button>" in html
    assert ">Легенда</button>" in html
    assert 'title.textContent = "Пересечения";' in html


def test_viewer_template_highlights_collisions_for_selected_legend_well() -> None:
    html = three_viewer._viewer_template_with_libraries()

    assert ".collision-item.is-related-legend-well {" in html
    assert "let selectedLegendWellNameKeys = new Set();" in html
    assert "function legendWellNameKeysForItem(item, kind)" in html
    assert "function syncSelectedCollisionHighlights()" in html
    assert "function setSelectedLegendWellNameKeys(keys)" in html
    assert 'setSelectedLegendWellNameKeys(legendWellNameKeysForItem(item, "well"));' in html
    assert "setSelectedLegendWellNameKeys(legendWellNameKeysForItem(item, kind));" in html
    assert "item.dataset.collisionWellA = normalizedWellNameKey(collision.well_a);" in html
    assert "item.dataset.collisionWellB = normalizedWellNameKey(collision.well_b);" in html
    assert 'item.classList.toggle(\n              "is-related-legend-well",' in html


def test_viewer_template_does_not_focus_camera_for_pad_legend_clicks() -> None:
    html = three_viewer._viewer_template_with_libraries()

    assert 'if (kind !== "well") {\n              return;\n            }' in html
    assert 'setSelectedLegendWellNameKeys(legendWellNameKeysForItem(item, kind));' in html
    assert 'focusViewerTarget(targetBounds, String(item.id || ""));' in html


def test_orbit_controls_use_expected_mouse_bindings() -> None:
    controls_js = (three_viewer._VENDOR_DIR / "OrbitControls.js").read_text(
        encoding="utf-8"
    )

    assert "LEFT: THREE.MOUSE.ROTATE" in controls_js
    assert "MIDDLE: THREE.MOUSE.DOLLY" in controls_js
    assert "RIGHT: THREE.MOUSE.PAN" in controls_js
    assert "this.screenSpacePanning = true;" in controls_js


def test_render_local_three_scene_appends_instance_token(monkeypatch) -> None:
    captured: dict[str, object] = {}
    calls: list[str] = []

    def _fake_component(**kwargs):
        calls.append("component")
        captured["component"] = dict(kwargs)
        return {"type": "noop"}

    monkeypatch.setattr(three_viewer, "_viewer_component", _fake_component)
    monkeypatch.setattr(three_viewer, "_viewer_runtime_digest", lambda: "runtime-digest")

    result = three_viewer.render_local_three_scene(
        {"background": "#FFFFFF"},
        height=640,
        instance_token=7,
    )

    assert '"background":"#FFFFFF"' in str(captured["component"]["payload_json"])
    assert '"edit_channel":"pywp_three_edit_' in str(
        captured["component"]["payload_json"]
    )
    assert captured["component"]["payload_digest"] == three_viewer._payload_digest(
        str(captured["component"]["payload_json"])
    )
    assert captured["component"]["runtime_digest"] == "runtime-digest"
    assert captured["component"]["height"] == 640
    assert captured["component"]["instance_token"] == 7
    assert str(captured["component"]["channel"]).startswith("pywp_three_edit_")
    assert captured["component"]["default"] is None
    assert captured["component"]["key"] == "three-viewer-runtime-scene"
    assert captured["component"]["has_anticollision_payload"] is False
    assert calls == ["component"]
    assert result == {"type": "noop"}


def test_render_local_three_scene_marks_anticollision_payload(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_component(**kwargs):
        captured.update(kwargs)
        return {"type": "noop"}

    monkeypatch.setattr(three_viewer, "_viewer_component", _fake_component)
    monkeypatch.setattr(three_viewer, "_viewer_runtime_digest", lambda: "runtime-digest")

    three_viewer.render_local_three_scene(
        {
            "background": "#FFFFFF",
            "collisions": [{"id": "collision-1"}],
        },
        height=320,
    )

    assert captured["has_anticollision_payload"] is True


def test_render_local_three_scene_reuses_serialized_payload_for_same_object(
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []
    json_calls = {"count": 0}
    payload = {"background": "#FFFFFF", "title": "Scene"}
    three_viewer._SERIALIZED_PAYLOAD_CACHE.clear()
    original_json_dumps = three_viewer.json.dumps

    def _fake_component(**kwargs):
        captured.append(dict(kwargs))
        return {"type": "noop"}

    def _counting_json_dumps(*args, **kwargs):
        json_calls["count"] += 1
        return original_json_dumps(*args, **kwargs)

    monkeypatch.setattr(three_viewer, "_viewer_component", _fake_component)
    monkeypatch.setattr(three_viewer, "_viewer_runtime_digest", lambda: "runtime-digest")
    monkeypatch.setattr(three_viewer.json, "dumps", _counting_json_dumps)

    three_viewer.render_local_three_scene(
        payload,
        height=480,
        instance_token=3,
        key="scene",
    )
    three_viewer.render_local_three_scene(
        payload,
        height=480,
        instance_token=3,
        key="scene",
    )

    assert json_calls["count"] == 1
    assert captured[0]["payload_json"] == captured[1]["payload_json"]
    assert captured[0]["payload_digest"] == captured[1]["payload_digest"]


def test_three_viewer_runtime_component_relays_json_events() -> None:
    component_html = (three_viewer._ASSETS_DIR / "index.html").read_text(
        encoding="utf-8"
    )

    assert "streamlit:setComponentValue" in component_html
    assert 'dataType: "json"' in component_html
    assert "new BroadcastChannel(channel)" in component_html
    assert 'data.type !== "pywp:editTargets"' in component_html
    assert 'const assetSuffix = expectedDigest' in component_html
    assert 'fetch("./templates/viewer_template.html" + assetSuffix' in component_html
    assert 'fetch("./vendor/three.min.js" + assetSuffix' in component_html
    assert 'fetch("./vendor/OrbitControls.js" + assetSuffix' in component_html
    assert 'fetch("./fast_replan.js" + assetSuffix' in component_html
    assert 'frame.srcdoc = sceneHtml;' in component_html
    assert 'nextPayloadDigest === currentPayloadDigest' in component_html
    assert "currentHasAntiCollisionPayload" in component_html
    assert "!currentHasAntiCollisionPayload &&" in component_html
    assert "nextHasAntiCollisionPayload &&" in component_html
    assert 'frameWindow.__PYWP_VIEWER_APPLY_ANTICOLLISION_OVERLAY__' in component_html
    assert 'frameWindow.__PYWP_VIEWER_UPDATE__' in component_html
    assert 'setStatus("Обновление 3D...", false);' in component_html


def test_three_viewer_assets_are_declared_as_package_data() -> None:
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.setuptools.package-data]" in pyproject_text
    assert '"three_viewer_assets/*.js"' in pyproject_text
    assert '"three_viewer_assets/*.html"' in pyproject_text
    assert '"three_viewer_assets/component/*.html"' in pyproject_text
    assert '"three_viewer_assets/templates/*.html"' in pyproject_text
    assert '"three_viewer_assets/vendor/*.js"' in pyproject_text


def test_ptc_single_well_view_uses_three_edit_override() -> None:
    page_source = Path("pywp/ptc_page_results.py").read_text(encoding="utf-8")

    assert (
        "def render_3d_override(container: object, payload: dict[str, object]) -> None:"
        in page_source
    )
    assert '"component_key": f"ptc-single-well-{selected.name}"' in page_source
    assert "wt._build_edit_wells_payload(" in page_source
    assert "render_3d_override=render_3d_override" in page_source


def test_three_viewer_edit_state_has_single_preview_line_declaration() -> None:
    html = three_viewer._viewer_template_with_libraries()

    assert (
        html.count(
            "const editLineObjects = []; // THREE.LineSegments per well for preview"
        )
        == 1
    )


def test_three_viewer_pair_move_scope_shifts_only_target_points() -> None:
    html = three_viewer._viewer_template_with_libraries()

    assert "function editEntryMovesWithPairScope(entry)" in html
    assert 'return pointType === "t1" || pointType === "t3";' in html
    assert "position: editEntryMovesWithPairScope(entry)" in html
    assert ": copyEditPoint(entry.position)," in html


def test_three_viewer_sidetrack_window_drag_snaps_by_cursor_ray() -> None:
    html = three_viewer._viewer_template_with_libraries()

    assert "function pointerRayForViewport(event, activeCamera, viewportRect)" in html
    assert "function miniMapViewportClientRect()" in html
    assert "function nearestSidetrackParentPointFromRawTarget(handle, target)" in html
    assert "ray.distanceSqToSegment(" in html
    assert 'const activeCamera = inputMode === "minimap" ? miniMapCamera : camera;' in html
    assert "best = {" in html
    assert "mdM: start[3] + (end[3] - start[3]) * t," in html
    assert "return best || nearestSidetrackParentPointFromRawTarget(handle, fallbackTarget);" in html
    assert "sidetrackWindowUpdate = snappedSidetrackWindowEntry(" in html
    assert "editDragInputMode," in html


def test_three_viewer_asset_loader_reloads_file_after_mtime_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "viewer_template.html"
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    (vendor_dir / "three.min.js").write_text("window.THREE = {};", encoding="utf-8")
    (vendor_dir / "OrbitControls.js").write_text(
        "window.OrbitControls = function () {};",
        encoding="utf-8",
    )
    template_path.write_text(
        "__THREE_LIBRARY__ __ORBIT_CONTROLS__ A __SCENE_PAYLOAD__",
        encoding="utf-8",
    )

    monkeypatch.setattr(three_viewer, "_TEMPLATE_PATH", template_path)
    monkeypatch.setattr(three_viewer, "_VENDOR_DIR", vendor_dir)
    three_viewer._read_text_cached.cache_clear()

    first = three_viewer._viewer_template_with_libraries()
    assert " A " in first

    time.sleep(0.01)
    template_path.write_text(
        "__THREE_LIBRARY__ __ORBIT_CONTROLS__ B __SCENE_PAYLOAD__",
        encoding="utf-8",
    )

    second = three_viewer._viewer_template_with_libraries()
    assert " B " in second
