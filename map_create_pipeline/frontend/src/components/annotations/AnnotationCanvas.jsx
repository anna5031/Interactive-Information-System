import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import PropTypes from 'prop-types';
import { ArrowLeftRight, ArrowUpDown } from 'lucide-react';
import LABEL_CONFIG, { getLabelById, isLineLabel, isBoxLabel, isPointLabel } from '../../config/annotationConfig';
import styles from './AnnotationCanvas.module.css';
import BoxAnnotation from './shapes/BoxAnnotation';
import LineAnnotation from './shapes/LineAnnotation';
import PointAnnotation from './shapes/PointAnnotation';
import useBoxInteractions from './hooks/useBoxInteractions';
import useLineInteractions from './hooks/useLineInteractions';
import usePointInteractions from './hooks/usePointInteractions';
import { BOX_BORDER_WIDTH, BOX_BORDER_OFFSET } from './boxDrawingConstants';
import {
  clamp,
  LINE_SNAP_THRESHOLD,
  DRAW_SNAP_DISTANCE,
  AXIS_LOCK_TOLERANCE,
  buildSnapPoints,
  buildSnapSegments,
  snapPosition,
  findAnchorForPoint,
  projectToClosestAnchor,
  applyAxisLockToLine,
  SNAP_RELEASE_DISTANCE,
  findVerticalSnap,
  findHorizontalSnap,
} from './utils/canvasGeometry';

const MIN_BOX_SIZE = 0.01;
const MIN_LINE_LENGTH = 0;
const MIN_ZOOM = 1;
const MAX_ZOOM = 4;
const KEYBOARD_ZOOM_STEP = 0.2;
const WHEEL_ZOOM_SENSITIVITY = 0.0016;
const GUIDE_COLOR = '#38bdf8';
const GUIDE_SEGMENT_TOLERANCE = 1e-6;
const clampNumber = (value, min, max) => Math.min(Math.max(value, min), max);
const getLabel = (labelId) => getLabelById(labelId) || LABEL_CONFIG[0];
const clampUnit = (value) => clampNumber(value, 0, 1);

const HIGHLIGHT_WIDTH = 2;
const HIGHLIGHT_OFFSET = HIGHLIGHT_WIDTH / 2;
const CALIBRATION_COLOR = '#fb923c';
const CALIBRATION_LABEL = { name: '기준선', color: CALIBRATION_COLOR };

const buildGuidesFromAxisSnaps = (snaps) => {
  if (!snaps || snaps.length === 0) {
    return [];
  }
  const guides = [];
  snaps.forEach((snap) => {
    if (!snap || !snap.snapped || !Array.isArray(snap.sources)) {
      return;
    }
    guides.push({
      type: snap.axis === 'vertical' ? 'vertical' : 'horizontal',
      value: snap.value,
      sources: snap.sources,
      axis: snap.axis,
    });
  });
  return guides;
};

const mergeGuideSegments = (segments) => {
  if (!segments || segments.length === 0) {
    return [];
  }

  const EPS = GUIDE_SEGMENT_TOLERANCE;
  const verticalGroups = new Map();
  const horizontalGroups = new Map();
  const generalGroups = new Map();

  segments.forEach((segment) => {
    const { x1, y1, x2, y2 } = segment;
    const dx = x2 - x1;
    const dy = y2 - y1;
    if (Math.abs(dx) <= EPS && Math.abs(dy) <= EPS) {
      return;
    }

    if (Math.abs(dx) <= EPS) {
      const xCoord = (x1 + x2) * 0.5;
      const key = `v:${xCoord.toFixed(6)}`;
      const current = verticalGroups.get(key);
      if (!current) {
        verticalGroups.set(key, {
          x: xCoord,
          minY: Math.min(y1, y2),
          maxY: Math.max(y1, y2),
          count: 1,
        });
      } else {
        current.x = (current.x * current.count + xCoord) / (current.count + 1);
        current.count += 1;
        current.minY = Math.min(current.minY, y1, y2);
        current.maxY = Math.max(current.maxY, y1, y2);
      }
      return;
    }

    if (Math.abs(dy) <= EPS) {
      const yCoord = (y1 + y2) * 0.5;
      const key = `h:${yCoord.toFixed(6)}`;
      const current = horizontalGroups.get(key);
      if (!current) {
        horizontalGroups.set(key, {
          y: yCoord,
          minX: Math.min(x1, x2),
          maxX: Math.max(x1, x2),
          count: 1,
        });
      } else {
        current.y = (current.y * current.count + yCoord) / (current.count + 1);
        current.count += 1;
        current.minX = Math.min(current.minX, x1, x2);
        current.maxX = Math.max(current.maxX, x1, x2);
      }
      return;
    }

    const length = Math.hypot(dx, dy);
    if (length <= EPS) {
      return;
    }

    let dirX = dx / length;
    let dirY = dy / length;
    if (dirX < 0 || (Math.abs(dirX) <= EPS && dirY < 0)) {
      dirX *= -1;
      dirY *= -1;
    }

    const normalX = -dirY;
    const normalY = dirX;
    const offset = normalX * x1 + normalY * y1;
    const key = `g:${normalX.toFixed(6)}:${normalY.toFixed(6)}:${offset.toFixed(6)}`;
    let group = generalGroups.get(key);
    if (!group) {
      group = {
        originX: x1,
        originY: y1,
        dirX,
        dirY,
        normalX,
        normalY,
        offset,
        minT: Infinity,
        maxT: -Infinity,
      };
      generalGroups.set(key, group);
    }

    const updateBounds = (px, py) => {
      const distance = Math.abs(group.normalX * px + group.normalY * py - group.offset);
      if (distance > EPS * 10) {
        return;
      }
      const t = group.dirX * (px - group.originX) + group.dirY * (py - group.originY);
      group.minT = Math.min(group.minT, t);
      group.maxT = Math.max(group.maxT, t);
    };

    updateBounds(x1, y1);
    updateBounds(x2, y2);
  });

  const merged = [];

  verticalGroups.forEach((group) => {
    const x = clampUnit(group.x);
    merged.push({
      x1: x,
      y1: 0,
      x2: x,
      y2: 1,
    });
  });

  horizontalGroups.forEach((group) => {
    const y = clampUnit(group.y);
    merged.push({
      x1: 0,
      y1: y,
      x2: 1,
      y2: y,
    });
  });

  generalGroups.forEach((group) => {
    if (!Number.isFinite(group.minT) || !Number.isFinite(group.maxT)) {
      return;
    }
    const startX = group.originX + group.dirX * group.minT;
    const startY = group.originY + group.dirY * group.minT;
    const endX = group.originX + group.dirX * group.maxT;
    const endY = group.originY + group.dirY * group.maxT;
    merged.push({
      x1: clampUnit(startX),
      y1: clampUnit(startY),
      x2: clampUnit(endX),
      y2: clampUnit(endY),
    });
  });

  return merged;
};

const AnnotationCanvas = forwardRef(
  (
    {
      imageUrl,
      previewOverlayUrl,
      previewOverlayVisible = false,
      previewOverlayOpacity = 0.4,
      boxes,
      lines,
      points = [],
      selectedItem,
      onSelect,
      onUpdateBox,
      onUpdateLine,
      onUpdatePoint,
      addMode,
      activeLabelId,
      onAddShape,
      hiddenLabelIds,
      isReadOnly = false,
      calibrationLine,
      onCalibrationLineChange,
      calibrationReadOnly = false,
    },
    ref
  ) => {
    const containerRef = useRef(null);
    const imageRef = useRef(null);
    const pointerStateRef = useRef(null);
    const imageBoxRef = useRef({ offsetX: 0, offsetY: 0, width: 0, height: 0 });
    const viewportRef = useRef({ scale: 1, offsetX: 0, offsetY: 0 });
    const capturePointer = useCallback((event) => {
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch (pointerError) {
        // ignore pointer capture errors
      }
    }, []);
    const [calibrationSelected, setCalibrationSelected] = useState(false);

    const [imageBox, setImageBox] = useState({ offsetX: 0, offsetY: 0, width: 0, height: 0 });
    const [draftShape, setDraftShape] = useState(null);
    const [viewport, setViewport] = useState(viewportRef.current);
    const [guides, setGuides] = useState([]);
    const effectiveAddMode = addMode && !isReadOnly;

    const updateViewport = useCallback((updater) => {
      setViewport((prev) => {
        const next = updater(prev);
        viewportRef.current = next;
        return next;
      });
    }, []);

    const setGuidesNormalized = useCallback((next) => {
      if (!next || next.length === 0) {
        setGuides([]);
        return;
      }
      setGuides(next);
    }, []);

    const clearGuides = useCallback(() => {
      setGuides([]);
    }, []);

    const clampOffset = useCallback((offset, scale, size) => {
      if (!size || scale <= 1) {
        return 0;
      }
      const maxShift = size * (scale - 1);
      return clampNumber(offset, -maxShift, 0);
    }, []);

    const applyViewportZoom = useCallback(
      (nextScale, focal) => {
        const metrics = imageBoxRef.current;
        if (!metrics.width || !metrics.height) {
          return;
        }
        let scale = clampNumber(nextScale, MIN_ZOOM, MAX_ZOOM);
        scale = Math.round(scale * 1000) / 1000;

        updateViewport((prev) => {
          if (Math.abs(scale - prev.scale) < 1e-5) {
            return prev;
          }

          let offsetX = prev.offsetX;
          let offsetY = prev.offsetY;

          if (focal) {
            const worldX = focal.x / prev.scale;
            const worldY = focal.y / prev.scale;
            offsetX = focal.x - worldX * scale;
            offsetY = focal.y - worldY * scale;
          } else {
            const ratio = scale / prev.scale;
            offsetX *= ratio;
            offsetY *= ratio;
            if (prev.scale <= 1 && scale > 1) {
              offsetX = -(metrics.width * scale - metrics.width) / 2;
              offsetY = -(metrics.height * scale - metrics.height) / 2;
            }
          }

          const nextOffsetX = clampOffset(offsetX, scale, metrics.width);
          const nextOffsetY = clampOffset(offsetY, scale, metrics.height);

          return { scale, offsetX: nextOffsetX, offsetY: nextOffsetY };
        });
      },
      [clampOffset, updateViewport]
    );

    const applyViewportPan = useCallback(
      (deltaX, deltaY) => {
        const metrics = imageBoxRef.current;
        if (!metrics.width || !metrics.height) {
          return;
        }

        updateViewport((prev) => {
          if (prev.scale <= 1) {
            if (prev.offsetX === 0 && prev.offsetY === 0) {
              return prev;
            }
            return { scale: prev.scale, offsetX: 0, offsetY: 0 };
          }

          let offsetX = prev.offsetX - deltaX;
          let offsetY = prev.offsetY - deltaY;
          offsetX = clampOffset(offsetX, prev.scale, metrics.width);
          offsetY = clampOffset(offsetY, prev.scale, metrics.height);

          if (offsetX === prev.offsetX && offsetY === prev.offsetY) {
            return prev;
          }

          return { scale: prev.scale, offsetX, offsetY };
        });
      },
      [clampOffset, updateViewport]
    );

    const handleWheel = useCallback(
      (event) => {
        const container = containerRef.current;
        const metrics = imageBoxRef.current;
        if (!container || !metrics.width || !metrics.height) {
          return;
        }

        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          const rect = container.getBoundingClientRect();
          const currentViewportState = viewportRef.current;
          const focal = {
            x: event.clientX - rect.left - metrics.offsetX - currentViewportState.offsetX,
            y: event.clientY - rect.top - metrics.offsetY - currentViewportState.offsetY,
          };
          const nextScale = viewportRef.current.scale * Math.exp(-event.deltaY * WHEEL_ZOOM_SENSITIVITY);
          applyViewportZoom(nextScale, focal);
          return;
        }

        if (viewportRef.current.scale > 1) {
          event.preventDefault();
          applyViewportPan(event.deltaX, event.deltaY);
        }
      },
      [applyViewportPan, applyViewportZoom]
    );

    useImperativeHandle(
      ref,
      () => ({
        zoomIn: () => {
          const current = viewportRef.current.scale;
          applyViewportZoom(current * (1 + KEYBOARD_ZOOM_STEP));
        },
        zoomOut: () => {
          const current = viewportRef.current.scale;
          applyViewportZoom(current * (1 - KEYBOARD_ZOOM_STEP));
        },
        resetZoom: () => {
          applyViewportZoom(1);
        },
        setZoom: (scale) => {
          applyViewportZoom(scale);
        },
        panBy: (deltaX, deltaY) => {
          applyViewportPan(deltaX, deltaY);
        },
        getViewport: () => viewportRef.current,
      }),
      [applyViewportPan, applyViewportZoom]
    );

    const visibleBoxes = useMemo(() => {
      if (!hiddenLabelIds || hiddenLabelIds.size === 0) {
        return boxes;
      }
      return boxes.filter((box) => !hiddenLabelIds.has(box.labelId));
    }, [boxes, hiddenLabelIds]);

    const visibleLines = useMemo(() => {
      if (!hiddenLabelIds || hiddenLabelIds.size === 0) {
        return lines;
      }
      return lines.filter((line) => !hiddenLabelIds.has(line.labelId));
    }, [lines, hiddenLabelIds]);

    const visiblePoints = useMemo(() => {
      if (!hiddenLabelIds || hiddenLabelIds.size === 0) {
        return points;
      }
      return points.filter((point) => !hiddenLabelIds.has(point.labelId));
    }, [points, hiddenLabelIds]);

    const anchoredPointIdsByBox = useMemo(() => {
      const map = new Map();
      points.forEach((point) => {
        const anchor = point?.anchor;
        if (anchor?.type === 'box' && anchor.id && point.id) {
          if (!map.has(anchor.id)) {
            map.set(anchor.id, new Set());
          }
          map.get(anchor.id).add(point.id);
        }
      });
      return map;
    }, [points]);

    const boxesMap = useMemo(() => {
      return boxes.reduce((acc, box) => {
        acc[box.id] = box;
        return acc;
      }, {});
    }, [boxes]);

    const linesMap = useMemo(() => {
      return new Map(lines.map((line) => [line.id, line]));
    }, [lines]);

    const snapPoints = useMemo(() => buildSnapPoints(boxes, lines, points), [boxes, lines, points]);

    const snapSegments = useMemo(() => buildSnapSegments(boxes, lines), [boxes, lines]);

    const updateImageBox = useCallback(() => {
      const container = containerRef.current;
      const image = imageRef.current;
      if (!container || !image || !image.naturalWidth || !image.naturalHeight) {
        return;
      }

      const containerRect = container.getBoundingClientRect();
      const containerWidth = containerRect.width;
      const containerHeight = containerRect.height;

      const imageAspect = image.naturalWidth / image.naturalHeight;
      const containerAspect = containerWidth / containerHeight;

      let renderWidth = containerWidth;
      let renderHeight = containerHeight;
      let offsetX = 0;
      let offsetY = 0;

      if (imageAspect > containerAspect) {
        renderWidth = containerWidth;
        renderHeight = containerWidth / imageAspect;
        offsetY = (containerHeight - renderHeight) / 2;
      } else {
        renderHeight = containerHeight;
        renderWidth = containerHeight * imageAspect;
        offsetX = (containerWidth - renderWidth) / 2;
      }

      const next = {
        offsetX,
        offsetY,
        width: renderWidth,
        height: renderHeight,
      };

      imageBoxRef.current = next;
      setImageBox(next);
      updateViewport((prev) => {
        const scale = clampNumber(prev.scale, MIN_ZOOM, MAX_ZOOM);
        const offsetX = clampOffset(prev.offsetX, scale, next.width);
        const offsetY = clampOffset(prev.offsetY, scale, next.height);

        if (scale === prev.scale && offsetX === prev.offsetX && offsetY === prev.offsetY) {
          return prev;
        }

        return { scale, offsetX, offsetY };
      });
    }, [clampOffset, updateViewport]);

    useEffect(() => {
      updateImageBox();
    }, [updateImageBox, imageUrl]);

    useEffect(() => {
      updateViewport(() => ({ scale: 1, offsetX: 0, offsetY: 0 }));
    }, [imageUrl, updateViewport]);

    useEffect(() => {
      const image = imageRef.current;
      if (!image) {
        return () => {};
      }

      if (image.complete) {
        updateImageBox();
      } else {
        image.addEventListener('load', updateImageBox);
      }

      return () => {
        image.removeEventListener('load', updateImageBox);
      };
    }, [updateImageBox, imageUrl]);

    useEffect(() => {
      if (typeof ResizeObserver === 'undefined') {
        window.addEventListener('resize', updateImageBox);
        return () => {
          window.removeEventListener('resize', updateImageBox);
        };
      }

      const observer = new ResizeObserver(() => updateImageBox());
      const container = containerRef.current;
      if (container) {
        observer.observe(container);
      }

      window.addEventListener('resize', updateImageBox);

      return () => {
        observer.disconnect();
        window.removeEventListener('resize', updateImageBox);
      };
    }, [updateImageBox]);

    useEffect(() => {
      const container = containerRef.current;
      return () => {
        if (pointerStateRef.current?.pointerId !== undefined && container) {
          try {
            container.releasePointerCapture(pointerStateRef.current.pointerId);
          } catch (error) {
            // ignore release failures
          }
        }
        pointerStateRef.current = null;
      };
    }, []);

    useEffect(() => {
      if (!effectiveAddMode) {
        setDraftShape(null);
      }
    }, [effectiveAddMode]);

    useEffect(() => {
      const container = containerRef.current;
      if (!container) {
        return;
      }

      container.addEventListener('wheel', handleWheel, { passive: false });

      return () => {
        container.removeEventListener('wheel', handleWheel);
      };
    }, [handleWheel]);

    const normalisePointer = useCallback((event) => {
      const container = containerRef.current;
      const metrics = imageBoxRef.current;
      const viewportState = viewportRef.current;

      if (!container || !metrics.width || !metrics.height) {
        return { x: 0, y: 0, isInside: false };
      }

      const containerRect = container.getBoundingClientRect();
      const rawX = event.clientX - containerRect.left - metrics.offsetX;
      const rawY = event.clientY - containerRect.top - metrics.offsetY;

      const adjustedX = (rawX - viewportState.offsetX) / viewportState.scale;
      const adjustedY = (rawY - viewportState.offsetY) / viewportState.scale;

      const x = clamp(adjustedX / metrics.width);
      const y = clamp(adjustedY / metrics.height);
      const isInside = adjustedX >= 0 && adjustedY >= 0 && adjustedX <= metrics.width && adjustedY <= metrics.height;

      return { x, y, localX: adjustedX, localY: adjustedY, isInside };
    }, []);

    const setSelection = (type, id) => {
      setCalibrationSelected(false);
      if (type && id) {
        onSelect?.({ type, id });
        return;
      }
      onSelect?.(null);
    };

    const snapDrawingPosition = (x, y, options = {}) => snapPosition({ x, y, snapPoints, snapSegments, ...options });

    const getAnchorForPoint = (x, y) => findAnchorForPoint(x, y, lines, boxes, LINE_SNAP_THRESHOLD);

    const projectDraggingPoint = useCallback(
      (x, y) => {
        return projectToClosestAnchor(x, y, lines, boxes);
      },
      [lines, boxes]
    );

    const applyAxisLock = (line, axisLockHint = null) => applyAxisLockToLine(line, AXIS_LOCK_TOLERANCE, axisLockHint);

    const { handleBoxPointerDown, handleBoxPointerMove, handleBoxResizePointerDown, handleBoxResizePointerMove } =
      useBoxInteractions({
        addMode: effectiveAddMode,
        readOnly: isReadOnly,
        pointerStateRef,
        boxesMap,
        hiddenLabelIds,
        normalisePointer,
        snapDrawingPosition,
        snapPoints,
        snapSegments,
        onUpdateBox,
        setSelection,
        clamp,
        minBoxSize: MIN_BOX_SIZE,
        anchoredPointIdsByBox,
        setGuides: setGuidesNormalized,
        clearGuides,
        snapReleaseDistance: SNAP_RELEASE_DISTANCE,
      });

    const { handleLinePointerDown, handleLinePointerMove, handleLineHandlePointerDown, handleLineResizeMove } =
      useLineInteractions({
        addMode: effectiveAddMode,
        readOnly: isReadOnly,
        pointerStateRef,
        linesMap,
        hiddenLabelIds,
        normalisePointer,
        clamp,
        onUpdateLine,
        setSelection,
        setGuides: setGuidesNormalized,
        clearGuides,
        snapReleaseDistance: SNAP_RELEASE_DISTANCE,
        snapPoints,
        snapSegments,
      });

    const { handlePointPointerDown, handlePointPointerMove, cleanupPointMove } = usePointInteractions({
      pointerStateRef,
      normalisePointer,
      projectToClosestAnchor: projectDraggingPoint,
      onUpdatePoint,
      setSelection,
      addMode: effectiveAddMode,
      readOnly: isReadOnly,
      clearGuides,
    });

    const calibrationAvailable = Boolean(calibrationLine);
    const calibrationInteractive = Boolean(
      calibrationAvailable && onCalibrationLineChange && !calibrationReadOnly && !isReadOnly
    );
    const showCalibrationHandles = Boolean(calibrationSelected && calibrationInteractive);

    useEffect(() => {
      if (!calibrationInteractive) {
        setCalibrationSelected(false);
      }
    }, [calibrationInteractive]);

    const handleCalibrationLinePointerDown = useCallback(
      (event) => {
        if (!calibrationInteractive || !calibrationLine) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        const { x, y } = normalisePointer(event);
        clearGuides();
        setCalibrationSelected(true);
        pointerStateRef.current = {
          type: 'move-calibration',
          pointerId: event.pointerId,
          startX: x,
          startY: y,
          startLine: { ...calibrationLine },
          snapSuppressed: false,
          snapOrigin: null,
          lastSnap: null,
        };
        capturePointer(event);
      },
      [
        calibrationInteractive,
        calibrationLine,
        normalisePointer,
        pointerStateRef,
        clearGuides,
        capturePointer,
        setCalibrationSelected,
      ]
    );

    const handleCalibrationLinePointerMove = useCallback(
      (event) => {
        if (!calibrationInteractive) {
          return;
        }
        const state = pointerStateRef.current;
        if (!state || state.type !== 'move-calibration') {
          return;
        }
        event.preventDefault();

        const { x, y } = normalisePointer(event);
        const deltaX = x - state.startX;
        const deltaY = y - state.startY;

        const rawLine = {
          x1: clamp(state.startLine.x1 + deltaX),
          y1: clamp(state.startLine.y1 + deltaY),
          x2: clamp(state.startLine.x2 + deltaX),
          y2: clamp(state.startLine.y2 + deltaY),
        };

        let nextX1 = rawLine.x1;
        let nextY1 = rawLine.y1;
        let nextX2 = rawLine.x2;
        let nextY2 = rawLine.y2;

        let shouldSnap = true;
        if (state.snapSuppressed && state.snapOrigin) {
          const startDelta = Math.hypot(rawLine.x1 - state.snapOrigin.x1, rawLine.y1 - state.snapOrigin.y1);
          const endDelta = Math.hypot(rawLine.x2 - state.snapOrigin.x2, rawLine.y2 - state.snapOrigin.y2);
          if (Math.max(startDelta, endDelta) >= SNAP_RELEASE_DISTANCE) {
            state.snapSuppressed = false;
            state.snapOrigin = null;
            state.lastSnap = null;
          } else {
            shouldSnap = false;
          }
        }

        const activeSnapResults = [];
        if (shouldSnap) {
          const snapOptions = {
            snapPoints,
            snapSegments,
            distance: DRAW_SNAP_DISTANCE,
            excludePointOwners: undefined,
            excludeSegmentOwners: undefined,
          };

          const vCandidates = [
            { value: rawLine.x1, kind: 'start' },
            { value: rawLine.x2, kind: 'end' },
          ];
          let bestVerticalSnap = { snapped: false, distance: Infinity };
          let vSnapSourceKind = 'start';
          vCandidates.forEach((candidate) => {
            const snap = findVerticalSnap({ value: candidate.value, ...snapOptions });
            if (snap.snapped && snap.distance < bestVerticalSnap.distance) {
              bestVerticalSnap = snap;
              vSnapSourceKind = candidate.kind;
            }
          });

          const hCandidates = [
            { value: rawLine.y1, kind: 'start' },
            { value: rawLine.y2, kind: 'end' },
          ];
          let bestHorizontalSnap = { snapped: false, distance: Infinity };
          let hSnapSourceKind = 'start';
          hCandidates.forEach((candidate) => {
            const snap = findHorizontalSnap({ value: candidate.value, ...snapOptions });
            if (snap.snapped && snap.distance < bestHorizontalSnap.distance) {
              bestHorizontalSnap = snap;
              hSnapSourceKind = candidate.kind;
            }
          });

          if (bestVerticalSnap.snapped) {
            const snapDeltaX =
              bestVerticalSnap.value - (vSnapSourceKind === 'start' ? rawLine.x1 : rawLine.x2);
            nextX1 = rawLine.x1 + snapDeltaX;
            nextX2 = rawLine.x2 + snapDeltaX;
            activeSnapResults.push(bestVerticalSnap);
          }

          if (bestHorizontalSnap.snapped) {
            const snapDeltaY =
              bestHorizontalSnap.value - (hSnapSourceKind === 'start' ? rawLine.y1 : rawLine.y2);
            nextY1 = rawLine.y1 + snapDeltaY;
            nextY2 = rawLine.y2 + snapDeltaY;
            activeSnapResults.push(bestHorizontalSnap);
          }
        }

        const nextLine = { x1: nextX1, y1: nextY1, x2: nextX2, y2: nextY2 };

        if (activeSnapResults.length > 0) {
          state.snapSuppressed = true;
          state.snapOrigin = { ...rawLine };
          state.lastSnap = { line: { ...nextLine }, guides: activeSnapResults };
          setGuidesNormalized(buildGuidesFromAxisSnaps(activeSnapResults));
        } else if (state.snapSuppressed && state.lastSnap) {
          Object.assign(nextLine, state.lastSnap.line);
          setGuidesNormalized(buildGuidesFromAxisSnaps(state.lastSnap.guides));
        } else {
          state.lastSnap = null;
          clearGuides();
        }

        onCalibrationLineChange?.(nextLine);
      },
      [
        calibrationInteractive,
        normalisePointer,
        pointerStateRef,
        snapPoints,
        snapSegments,
        setGuidesNormalized,
        clearGuides,
        onCalibrationLineChange,
      ]
    );

    const handleCalibrationLineHandlePointerDown = useCallback(
      (event, _line, handle) => {
        if (!calibrationInteractive || !calibrationLine) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        clearGuides();
        setCalibrationSelected(true);
        pointerStateRef.current = {
          type: 'resize-calibration',
          pointerId: event.pointerId,
          handle,
          startLine: { ...calibrationLine },
          snapSuppressed: false,
          snapOrigin: null,
          lastSnap: null,
        };
        capturePointer(event);
      },
      [
        calibrationInteractive,
        calibrationLine,
        pointerStateRef,
        clearGuides,
        capturePointer,
        setCalibrationSelected,
      ]
    );

    const handleCalibrationLineHandlePointerMove = useCallback(
      (event) => {
        if (!calibrationInteractive) {
          return;
        }
        const state = pointerStateRef.current;
        if (!state || state.type !== 'resize-calibration') {
          return;
        }
        event.preventDefault();

        const { x, y } = normalisePointer(event);
        const rawX = clamp(x);
        const rawY = clamp(y);

        const fixedX = state.handle === 'start' ? state.startLine.x2 : state.startLine.x1;
        const fixedY = state.handle === 'start' ? state.startLine.y2 : state.startLine.y1;

        let nextX = rawX;
        let nextY = rawY;
        let shouldSnap = true;
        if (state.snapSuppressed && state.snapOrigin && state.snapOrigin.handle === state.handle) {
          const dx = rawX - state.snapOrigin.x;
          const dy = rawY - state.snapOrigin.y;
          if (Math.hypot(dx, dy) >= SNAP_RELEASE_DISTANCE) {
            state.snapSuppressed = false;
            state.snapOrigin = null;
            state.lastSnap = null;
          } else {
            shouldSnap = false;
          }
        }

        const activeSnapResults = [];
        if (shouldSnap) {
          const snapOptions = {
            snapPoints,
            snapSegments,
            distance: DRAW_SNAP_DISTANCE,
            excludePointOwners: undefined,
            excludeSegmentOwners: undefined,
          };
          const snapV = findVerticalSnap({ value: rawX, ...snapOptions });
          if (snapV.snapped) {
            nextX = snapV.value;
            activeSnapResults.push(snapV);
          }
          const snapH = findHorizontalSnap({ value: rawY, ...snapOptions });
          if (snapH.snapped) {
            nextY = snapH.value;
            activeSnapResults.push(snapH);
          }
        }

        if (activeSnapResults.length > 0) {
          state.snapSuppressed = true;
          state.snapOrigin = { x: rawX, y: rawY, handle: state.handle };
          state.lastSnap = { x: nextX, y: nextY, guides: activeSnapResults, handle: state.handle };
          setGuidesNormalized(buildGuidesFromAxisSnaps(activeSnapResults));
        } else if (state.snapSuppressed && state.lastSnap && state.lastSnap.handle === state.handle) {
          nextX = state.lastSnap.x;
          nextY = state.lastSnap.y;
          setGuidesNormalized(buildGuidesFromAxisSnaps(state.lastSnap.guides));
        } else {
          state.lastSnap = null;
          const dx = Math.abs(nextX - fixedX);
          const dy = Math.abs(nextY - fixedY);
          const isVertical = dx <= dy * AXIS_LOCK_TOLERANCE;
          const isHorizontal = dy <= dx * AXIS_LOCK_TOLERANCE;
          if (isVertical) {
            nextX = fixedX;
            setGuidesNormalized([
              { type: 'vertical', value: fixedX, sources: [], axis: 'vertical' },
              { type: 'lock_symbol', x: fixedX, y: nextY, lock: 'vertical' },
            ]);
          } else if (isHorizontal) {
            nextY = fixedY;
            setGuidesNormalized([
              { type: 'horizontal', value: fixedY, sources: [], axis: 'horizontal' },
              { type: 'lock_symbol', x: nextX, y: fixedY, lock: 'horizontal' },
            ]);
          } else {
            clearGuides();
          }
        }

        const nextLine = {
          x1: state.handle === 'start' ? nextX : state.startLine.x1,
          y1: state.handle === 'start' ? nextY : state.startLine.y1,
          x2: state.handle === 'end' ? nextX : state.startLine.x2,
          y2: state.handle === 'end' ? nextY : state.startLine.y2,
        };
        onCalibrationLineChange?.(nextLine);
      },
      [
        calibrationInteractive,
        pointerStateRef,
        normalisePointer,
        snapPoints,
        snapSegments,
        setGuidesNormalized,
        clearGuides,
        onCalibrationLineChange,
      ]
    );

    const handlePointerUp = (event) => {
      const state = pointerStateRef.current;
      if (!state) {
        return;
      }

      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }

      if (state.type === 'pan') {
        if (!state.didMove) {
          onSelect?.(null);
        }
        pointerStateRef.current = null;
        return;
      }

      if (state.type === 'add-box') {
        if (draftShape && draftShape.width >= MIN_BOX_SIZE && draftShape.height >= MIN_BOX_SIZE) {
          onAddShape?.(draftShape);
        }
        setDraftShape(null);
      }

      if (state.type === 'add-line') {
        if (draftShape) {
          const length = Math.hypot(draftShape.x2 - draftShape.x1, draftShape.y2 - draftShape.y1);
          if (length >= MIN_LINE_LENGTH) {
            onAddShape?.(draftShape);
          }
        }
        setDraftShape(null);
      }

      const isCalibrationInteraction = state.type === 'move-calibration' || state.type === 'resize-calibration';

      if (isCalibrationInteraction) {
        clearGuides();
        pointerStateRef.current = null;
        return;
      }

      if (state.type === 'move-point') {
        cleanupPointMove?.();
        const { x, y } = normalisePointer(event);
        const anchor = projectDraggingPoint(x, y);

        if (anchor) {
          onUpdatePoint?.(state.id, {
            x: anchor.x,
            y: anchor.y,
            anchor: anchor.anchor,
          });
        } else {
          onUpdatePoint?.(state.id, { x, y, anchor: null });
        }
      }

      clearGuides();
      pointerStateRef.current = null;
      if (!isCalibrationInteraction) {
        setCalibrationSelected(false);
      }
    };

    const handleCanvasPointerDown = (event) => {
      clearGuides();
      setCalibrationSelected(false);
      if (isReadOnly) {
        const { isInside } = normalisePointer(event);
        if (!isInside) {
          onSelect?.(null);
        }
        return;
      }

      const labelIsLine = isLineLabel(activeLabelId);
      const labelIsPoint = isPointLabel(activeLabelId);

      if (!effectiveAddMode) {
        const position = normalisePointer(event);
        if (!position.isInside) {
          onSelect?.(null);
          return;
        }

        if (event.button === 0 || event.button === 1 || event.pointerType === 'touch') {
          pointerStateRef.current = {
            type: 'pan',
            pointerId: event.pointerId,
            lastClientX: event.clientX,
            lastClientY: event.clientY,
            didMove: false,
          };
          event.currentTarget.setPointerCapture(event.pointerId);
          event.preventDefault();
        } else {
          onSelect?.(null);
        }
        return;
      }

      const position = normalisePointer(event);
      if (!position.isInside) {
        return;
      }
      event.preventDefault();

      const snappedStart = snapDrawingPosition(position.x, position.y);

      if (labelIsLine) {
        const draft = {
          type: 'line',
          labelId: activeLabelId,
          x1: snappedStart.x,
          y1: snappedStart.y,
          x2: snappedStart.x,
          y2: snappedStart.y,
        };
        setDraftShape(draft);
        pointerStateRef.current = {
          type: 'add-line',
          pointerId: event.pointerId,
          originX: snappedStart.x,
          originY: snappedStart.y,
        };
        event.currentTarget.setPointerCapture(event.pointerId);
      } else if (labelIsPoint) {
        const anchor = getAnchorForPoint(position.x, position.y);
        if (!anchor) {
          return;
        }
        const draft = {
          type: 'point',
          labelId: activeLabelId,
          x: anchor.x,
          y: anchor.y,
          anchor: anchor.anchor,
        };
        onAddShape?.(draft);
      } else if (isBoxLabel(activeLabelId)) {
        const snapped = snapDrawingPosition(position.x, position.y);
        const draft = {
          type: 'box',
          labelId: activeLabelId,
          x: snapped.x,
          y: snapped.y,
          width: 0,
          height: 0,
        };
        setDraftShape(draft);
        pointerStateRef.current = {
          type: 'add-box',
          pointerId: event.pointerId,
          originX: snapped.x,
          originY: snapped.y,
        };
        event.currentTarget.setPointerCapture(event.pointerId);
      } else {
        return;
      }
    };

    const handleCanvasPointerMove = (event) => {
      if (isReadOnly) {
        return;
      }

      const state = pointerStateRef.current;
      if (!state) {
        return;
      }

      if (state.type === 'move-point') {
        handlePointPointerMove(event);
        return;
      }
      if (state.type === 'move-line') {
        handleLinePointerMove(event);
        return;
      }
      if (state.type === 'resize-line') {
        handleLineResizeMove(event);
        return;
      }
      if (state.type === 'move-box') {
        handleBoxPointerMove(event);
        return;
      }
      if (state.type === 'resize-box') {
        handleBoxResizePointerMove(event);
        return;
      }

      if (state.type === 'pan') {
        event.preventDefault();
        const deltaX = event.clientX - state.lastClientX;
        const deltaY = event.clientY - state.lastClientY;
        if (deltaX !== 0 || deltaY !== 0) {
          applyViewportPan(-deltaX, -deltaY);
          state.lastClientX = event.clientX;
          state.lastClientY = event.clientY;
          state.didMove = state.didMove || Math.abs(deltaX) > 0 || Math.abs(deltaY) > 0;
        }
        return;
      }

      if (state.type !== 'add-box' && state.type !== 'add-line') {
        return;
      }
      event.preventDefault();

      const { x, y } = normalisePointer(event);

      if (state.type === 'add-box') {
        const snapped = snapDrawingPosition(x, y);
        const snappedX = snapped.x;
        const snappedY = snapped.y;
        const boxMinX = Math.min(state.originX, snappedX);
        const boxMinY = Math.min(state.originY, snappedY);
        const width = Math.abs(snappedX - state.originX);
        const height = Math.abs(snappedY - state.originY);

        setDraftShape((prev) =>
          prev
            ? {
                ...prev,
                x: clamp(boxMinX, 0, 1 - MIN_BOX_SIZE),
                y: clamp(boxMinY, 0, 1 - MIN_BOX_SIZE),
                width: clamp(width, 0, 1),
                height: clamp(height, 0, 1),
              }
            : prev
        );
      } else if (state.type === 'add-line') {
        const snapped = snapDrawingPosition(x, y);
        const snappedX = snapped.x;
        const snappedY = snapped.y;
        setDraftShape((prev) =>
          prev
            ? {
                ...prev,
                ...(() => {
                  const line = { x1: state.originX, y1: state.originY, x2: snappedX, y2: snappedY };
                  const lockedLine = applyAxisLock(line);
                  return {
                    x2: lockedLine.x2,
                    y2: lockedLine.y2,
                  };
                })(),
              }
            : prev
        );
      }
    };

    const viewportStyle = useMemo(
      () => ({
        left: `${imageBox.offsetX}px`,
        top: `${imageBox.offsetY}px`,
        width: `${imageBox.width}px`,
        height: `${imageBox.height}px`,
        transform: `translate(${viewport.offsetX}px, ${viewport.offsetY}px)`,
      }),
      [imageBox.height, imageBox.offsetX, imageBox.offsetY, imageBox.width, viewport.offsetX, viewport.offsetY]
    );

    const viewportInnerStyle = useMemo(
      () => ({
        width: `${imageBox.width}px`,
        height: `${imageBox.height}px`,
        transform: `scale(${viewport.scale})`,
        transformOrigin: 'top left',
      }),
      [imageBox.height, imageBox.width, viewport.scale]
    );

    const overlayStyle = useMemo(
      () => ({
        width: `${imageBox.width}px`,
        height: `${imageBox.height}px`,
      }),
      [imageBox.height, imageBox.width]
    );

    const draftElements = [];
    if (draftShape?.type === 'box') {
      const label = getLabel(draftShape.labelId);
      const draftColor = label?.color || '#94a3b8';
      const leftPercent = draftShape.x * 100;
      const topPercent = draftShape.y * 100;
      const widthPercent = draftShape.width * 100;
      const heightPercent = draftShape.height * 100;
      draftElements.push(
        <div
          key='draft-box'
          className={styles.annotation}
          style={{
            left: `calc(${leftPercent}% - ${BOX_BORDER_OFFSET}px)`,
            top: `calc(${topPercent}% - ${BOX_BORDER_OFFSET}px)`,
            width: `calc(${widthPercent}% + ${BOX_BORDER_WIDTH}px)`,
            height: `calc(${heightPercent}% + ${BOX_BORDER_WIDTH}px)`,
            border: `${BOX_BORDER_WIDTH}px dashed ${draftColor}`,
            backgroundColor: 'transparent',
            opacity: 0.85,
            pointerEvents: 'none',
          }}
        />
      );
    } else if (draftShape?.type === 'line') {
      draftElements.push(
        <svg key='draft-line' className={styles.lineDraft} width={imageBox.width} height={imageBox.height}>
          <line
            x1={draftShape.x1 * imageBox.width}
            y1={draftShape.y1 * imageBox.height}
            x2={draftShape.x2 * imageBox.width}
            y2={draftShape.y2 * imageBox.height}
            stroke='#94a3b8'
            strokeWidth={4}
            strokeLinecap='round'
            strokeDasharray='6 4'
          />
        </svg>
      );
    }

    const { edgeMap, lineHighlights } = useMemo(() => {
      const edgeMap = new Map();
      const lineHighlights = new Set();

      if (!guides || guides.length === 0) {
        return { edgeMap, lineHighlights };
      }

      const cornerToEdges = {
        'top-left': ['top', 'left'],
        'top-right': ['top', 'right'],
        'bottom-left': ['bottom', 'left'],
        'bottom-right': ['bottom', 'right'],
      };

      const addEdge = (ownerId, edge) => {
        if (!ownerId || !edge) {
          return;
        }
        if (!edgeMap.has(ownerId)) {
          edgeMap.set(ownerId, new Set());
        }
        edgeMap.get(ownerId).add(edge);
      };

      guides.forEach((guide) => {
        if (!Array.isArray(guide.sources)) {
          return;
        }

        guide.sources.forEach((source) => {
          if (!source) {
            return;
          }

          const { ownerId, type, meta } = source;
          if (!ownerId) {
            return;
          }

          if (type === 'box-edge') {
            const edge = meta?.edge;
            if (!edge) return;

            if (guide.axis === 'vertical' && (edge === 'left' || edge === 'right')) {
              addEdge(ownerId, edge);
            } else if (guide.axis === 'horizontal' && (edge === 'top' || edge === 'bottom')) {
              addEdge(ownerId, edge);
            }
            return;
          }

          if (type === 'box-corner') {
            const edges = cornerToEdges[meta?.corner];
            if (edges) {
              if (guide.axis === 'vertical') {
                if (edges.includes('left')) addEdge(ownerId, 'left');
                if (edges.includes('right')) addEdge(ownerId, 'right');
              } else if (guide.axis === 'horizontal') {
                if (edges.includes('top')) addEdge(ownerId, 'top');
                if (edges.includes('bottom')) addEdge(ownerId, 'bottom');
              }
            }
            return;
          }

          if (type === 'line' || type === 'line-end') {
            lineHighlights.add(ownerId);
          }
        });
      });

      return { edgeMap, lineHighlights };
    }, [guides]);

    const guideElements = useMemo(() => {
      if (!guides || guides.length === 0) {
        return [];
      }

      const iconElements = [];
      const segmentGuides = [];

      guides.forEach((guide, index) => {
        if (guide.type === 'vertical') {
          segmentGuides.push({
            key: `guide-line-${index}`,
            x1: guide.value,
            y1: 0,
            x2: guide.value,
            y2: 1,
          });
        } else if (guide.type === 'horizontal') {
          segmentGuides.push({
            key: `guide-line-${index}`,
            x1: 0,
            y1: guide.value,
            x2: 1,
            y2: guide.value,
          });
        } else if (guide.type === 'segment') {
          segmentGuides.push({ ...guide, key: index });
        } else if (guide.type === 'lock_symbol') {
          const Icon = guide.lock === 'vertical' ? ArrowUpDown : ArrowLeftRight;
          iconElements.push(
            <div
              key={`guide-lock-${index}`}
              className={styles.guideLockIcon}
              style={{
                left: `${guide.x * 100}%`,
                top: `${guide.y * 100}%`,
                transform: 'translate(12px, -8px)',
              }}
            >
              <Icon size={14} strokeWidth={3} />
            </div>
          );
        }
      });

      if (segmentGuides.length > 0) {
        const mergedSegments = mergeGuideSegments(segmentGuides);
        const segmentsToRender = mergedSegments.length > 0 ? mergedSegments : segmentGuides;
        iconElements.push(
          <svg key='guide-segments' className={styles.guideLayer} width={imageBox.width} height={imageBox.height}>
            {segmentsToRender.map((guide, index) => (
              <line
                key={`guide-segment-${guide.key ?? index}`}
                x1={guide.x1 * imageBox.width}
                y1={guide.y1 * imageBox.height}
                x2={guide.x2 * imageBox.width}
                y2={guide.y2 * imageBox.height}
                stroke={GUIDE_COLOR}
                strokeDasharray='6 4'
                strokeWidth={2}
                strokeOpacity={0.65}
                strokeLinecap='round'
              />
            ))}
          </svg>
        );
      }

      return iconElements;
    }, [guides, imageBox.height, imageBox.width]);

    const highlightLayer = useMemo(() => {
      const elements = [];
      if ((!edgeMap || edgeMap.size === 0) && (!lineHighlights || lineHighlights.size === 0)) {
        return elements;
      }

      const boxMap = new Map(visibleBoxes.map((b) => [b.id, b]));
      edgeMap.forEach((edges, boxId) => {
        const box = boxMap.get(boxId);
        if (!box) return;

        edges.forEach((edge) => {
          let style = {};
          const key = `${boxId}-${edge}-highlight`;
          const left = `${box.x * 100}%`;
          const top = `${box.y * 100}%`;
          const width = `${box.width * 100}%`;
          const height = `${box.height * 100}%`;

          switch (edge) {
            case 'left':
              style = {
                top,
                height,
                left,
                width: `${HIGHLIGHT_WIDTH}px`,
                transform: `translateX(-${HIGHLIGHT_OFFSET}px)`,
              };
              break;
            case 'right':
              style = {
                top,
                height,
                left: `calc(${left} + ${width})`,
                width: `${HIGHLIGHT_WIDTH}px`,
                transform: `translateX(-${HIGHLIGHT_OFFSET}px)`,
              };
              break;
            case 'top':
              style = {
                left,
                width,
                top,
                height: `${HIGHLIGHT_WIDTH}px`,
                transform: `translateY(-${HIGHLIGHT_OFFSET}px)`,
              };
              break;
            case 'bottom':
              style = {
                left,
                width,
                top: `calc(${top} + ${height})`,
                height: `${HIGHLIGHT_WIDTH}px`,
                transform: `translateY(-${HIGHLIGHT_OFFSET}px)`,
              };
              break;
            default:
              return;
          }
          elements.push(<div key={key} className={styles.highlightEdge} style={style} />);
        });
      });

      if (imageBox.width > 0 && imageBox.height > 0) {
        lineHighlights.forEach((lineId) => {
          const line = linesMap.get(lineId);
          if (!line) return;

          const x1_pct = line.x1 * 100;
          const y1_pct = line.y1 * 100;

          const dx_px = (line.x2 - line.x1) * imageBox.width;
          const dy_px = (line.y2 - line.y1) * imageBox.height;
          const length_px = Math.hypot(dx_px, dy_px);
          const angle_rad = Math.atan2(dy_px, dx_px);

          const style = {
            left: `${x1_pct}%`,
            top: `${y1_pct}%`,
            width: `${length_px}px`,
            height: `${HIGHLIGHT_WIDTH}px`,
            transform: `rotate(${angle_rad}rad) translateY(-${HIGHLIGHT_OFFSET}px)`,
            transformOrigin: 'top left',
          };
          elements.push(<div key={`${line.id}-highlight`} className={styles.highlightEdge} style={style} />);
        });
      }

      return elements;
    }, [visibleBoxes, edgeMap, linesMap, lineHighlights, imageBox.width, imageBox.height]);

    const canvasStyle = useMemo(
      () => ({
        minWidth: imageBox.width ? `${imageBox.width}px` : undefined,
        minHeight: imageBox.height ? `${imageBox.height}px` : undefined,
      }),
      [imageBox.height, imageBox.width]
    );

    return (
      <div
        ref={containerRef}
        className={`${styles.canvas} ${effectiveAddMode ? styles.adding : ''}`}
        onPointerDown={handleCanvasPointerDown}
        onPointerMove={handleCanvasPointerMove}
        onPointerUp={handlePointerUp}
        style={canvasStyle}
      >
        <div className={styles.viewport} style={viewportStyle}>
          <div className={styles.viewportInner} style={viewportInnerStyle}>
            <img
              ref={imageRef}
              src={imageUrl}
              alt='floor plan'
              className={styles.image}
              style={{ width: '100%', height: '100%' }}
            />
            <div className={styles.overlay} style={overlayStyle}>
              {previewOverlayUrl && previewOverlayVisible && (
                <img
                  src={previewOverlayUrl}
                  alt='free space preview overlay'
                  className={styles.previewMask}
                  style={{ opacity: previewOverlayOpacity }}
                />
              )}
              {guideElements}
              {draftElements}
              {visibleBoxes.map((box) => (
                <BoxAnnotation
                  key={box.id}
                  box={box}
                  label={getLabel(box.labelId)}
                  isSelected={selectedItem?.type === 'box' && selectedItem.id === box.id}
                  viewportScale={viewport.scale}
                  onPointerDown={handleBoxPointerDown}
                  onPointerMove={handleBoxPointerMove}
                  onPointerUp={handlePointerUp}
                  onResizePointerDown={handleBoxResizePointerDown}
                  onResizePointerMove={handleBoxResizePointerMove}
                />
              ))}
              {highlightLayer}

              <svg className={styles.lineLayer} width={imageBox.width} height={imageBox.height}>
                {calibrationAvailable && (
                  <LineAnnotation
                    key='__calibration__'
                    line={{ id: '__calibration__', labelId: 'calibration', ...calibrationLine }}
                    label={CALIBRATION_LABEL}
                    isSelected={showCalibrationHandles}
                    isHighlighted={showCalibrationHandles}
                    imageBox={imageBox}
                    viewportScale={viewport.scale}
                    onPointerDown={handleCalibrationLinePointerDown}
                    onPointerMove={handleCalibrationLinePointerMove}
                    onPointerUp={handlePointerUp}
                    onHandlePointerDown={handleCalibrationLineHandlePointerDown}
                    onHandlePointerMove={handleCalibrationLineHandlePointerMove}
                    strokeWidth={2}
                    hitStrokeWidth={24}
                    handleRadius={4}
                    highlightColor={CALIBRATION_COLOR}
                  />
                )}
                {visibleLines.map((line) => (
                  <LineAnnotation
                    key={line.id}
                    line={line}
                    label={getLabel(line.labelId)}
                    isSelected={selectedItem?.type === 'line' && selectedItem.id === line.id}
                    imageBox={imageBox}
                    viewportScale={viewport.scale}
                    onPointerDown={handleLinePointerDown}
                    onPointerMove={handleLinePointerMove}
                    onPointerUp={handlePointerUp}
                    onHandlePointerDown={handleLineHandlePointerDown}
                    onHandlePointerMove={handleLineResizeMove}
                  />
                ))}
              </svg>
              <svg className={styles.pointLayer} width={imageBox.width} height={imageBox.height}>
                {visiblePoints.map((point) => (
                  <PointAnnotation
                    key={point.id}
                    point={point}
                    label={getLabel(point.labelId)}
                    isSelected={selectedItem?.type === 'point' && selectedItem.id === point.id}
                    imageBox={imageBox}
                    viewportScale={viewport.scale}
                    onPointerDown={handlePointPointerDown}
                    onPointerMove={handlePointPointerMove}
                    onPointerUp={handlePointerUp}
                  />
                ))}
              </svg>
            </div>
          </div>
        </div>
      </div>
    );
  }
);
AnnotationCanvas.displayName = 'AnnotationCanvas';

AnnotationCanvas.propTypes = {
  imageUrl: PropTypes.string.isRequired,
  previewOverlayUrl: PropTypes.string,
  previewOverlayVisible: PropTypes.bool,
  previewOverlayOpacity: PropTypes.number,
  boxes: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      labelId: PropTypes.string.isRequired,
      x: PropTypes.number.isRequired,
      y: PropTypes.number.isRequired,
      width: PropTypes.number.isRequired,
      height: PropTypes.number.isRequired,
    })
  ).isRequired,
  lines: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      labelId: PropTypes.string.isRequired,
      x1: PropTypes.number.isRequired,
      y1: PropTypes.number.isRequired,
      x2: PropTypes.number.isRequired,
      y2: PropTypes.number.isRequired,
    })
  ).isRequired,
  points: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      labelId: PropTypes.string.isRequired,
      x: PropTypes.number.isRequired,
      y: PropTypes.number.isRequired,
      anchor: PropTypes.shape({
        type: PropTypes.oneOf(['line', 'box']),
        id: PropTypes.string,
        t: PropTypes.number,
        edge: PropTypes.oneOf(['top', 'bottom', 'left', 'right']),
      }),
    })
  ).isRequired,
  selectedItem: PropTypes.shape({
    type: PropTypes.oneOf(['box', 'line', 'point']).isRequired,
    id: PropTypes.string.isRequired,
  }),
  onSelect: PropTypes.func,
  onUpdateBox: PropTypes.func,
  onUpdateLine: PropTypes.func,
  onUpdatePoint: PropTypes.func,
  addMode: PropTypes.bool,
  activeLabelId: PropTypes.string,
  onAddShape: PropTypes.func,
  hiddenLabelIds: PropTypes.instanceOf(Set),
  isReadOnly: PropTypes.bool,
  calibrationLine: PropTypes.shape({
    x1: PropTypes.number,
    y1: PropTypes.number,
    x2: PropTypes.number,
    y2: PropTypes.number,
  }),
  onCalibrationLineChange: PropTypes.func,
  calibrationReadOnly: PropTypes.bool,
};

AnnotationCanvas.defaultProps = {
  previewOverlayUrl: null,
  previewOverlayVisible: false,
  previewOverlayOpacity: 0.4,
  selectedItem: null,
  onSelect: undefined,
  onUpdateBox: undefined,
  onUpdateLine: undefined,
  onUpdatePoint: undefined,
  addMode: false,
  activeLabelId: LABEL_CONFIG[0]?.id ?? '0',
  onAddShape: undefined,
  hiddenLabelIds: undefined,
  isReadOnly: false,
  calibrationLine: null,
  onCalibrationLineChange: undefined,
  calibrationReadOnly: false,
};

export default AnnotationCanvas;
