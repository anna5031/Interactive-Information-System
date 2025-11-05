import { useCallback } from 'react';

const useBoxInteractions = ({
  addMode,
  readOnly = false,
  pointerStateRef,
  boxesMap,
  hiddenLabelIds,
  normalisePointer,
  snapDrawingPosition,
  onUpdateBox,
  setSelection,
  clamp,
  minBoxSize,
  anchoredPointIdsByBox,
  setGuides,
  clearGuides,
  snapReleaseDistance,
}) => {
  const releaseDistance = Number.isFinite(snapReleaseDistance) ? snapReleaseDistance : 0.04;

  const createMovingSegment = (kind, snap, { x, y, width, height, id }) => {
    if (!kind) {
      return null;
    }
    const base = {
      ownerId: id,
      type: 'moving-edge',
      axis: snap?.axis ?? null,
    };

    switch (kind) {
      case 'edge-left': {
        const xCoord = x;
        return {
          ...base,
          meta: { edge: 'left', role: 'moving' },
          x1: xCoord,
          y1: y,
          x2: xCoord,
          y2: y + height,
          ax: xCoord,
          ay: y,
          bx: xCoord,
          by: y + height,
        };
      }
      case 'edge-right': {
        const xCoord = x + width;
        return {
          ...base,
          meta: { edge: 'right', role: 'moving' },
          x1: xCoord,
          y1: y,
          x2: xCoord,
          y2: y + height,
          ax: xCoord,
          ay: y,
          bx: xCoord,
          by: y + height,
        };
      }
      case 'edge-top': {
        const yCoord = y;
        return {
          ...base,
          meta: { edge: 'top', role: 'moving' },
          x1: x,
          y1: yCoord,
          x2: x + width,
          y2: yCoord,
          ax: x,
          ay: yCoord,
          bx: x + width,
          by: yCoord,
        };
      }
      case 'edge-bottom': {
        const yCoord = y + height;
        return {
          ...base,
          meta: { edge: 'bottom', role: 'moving' },
          x1: x,
          y1: yCoord,
          x2: x + width,
          y2: yCoord,
          ax: x,
          ay: yCoord,
          bx: x + width,
          by: yCoord,
        };
      }
      default:
        return null;
    }
  };

  const orientationFromKind = (kind) => {
    switch (kind) {
      case 'corner-top-left':
        return { vertical: 'top', horizontal: 'left' };
      case 'corner-top-right':
        return { vertical: 'top', horizontal: 'right' };
      case 'corner-bottom-left':
        return { vertical: 'bottom', horizontal: 'left' };
      case 'corner-bottom-right':
        return { vertical: 'bottom', horizontal: 'right' };
      case 'edge-top':
        return { vertical: 'top', horizontal: null };
      case 'edge-bottom':
        return { vertical: 'bottom', horizontal: null };
      case 'edge-left':
        return { vertical: null, horizontal: 'left' };
      case 'edge-right':
        return { vertical: null, horizontal: 'right' };
      default:
        return { vertical: null, horizontal: null };
    }
  };

  const resolveAxisValue = (axis, snap) => {
    if (!axis || !snap) {
      return null;
    }

    const trySources = [snap.source, ...(snap.relatedSources ?? [])];
    const extract = (source) => {
      if (!source) {
        return null;
      }
      if (axis === 'horizontal') {
        if (Number.isFinite(source.ay)) {
          return source.ay;
        }
        if (Number.isFinite(source.y1) && Number.isFinite(source.y2)) {
          if (Math.abs(source.y1 - source.y2) <= 1e-9) {
            return source.y1;
          }
          return (source.y1 + source.y2) / 2;
        }
        if (Number.isFinite(source.y)) {
          return source.y;
        }
      } else if (axis === 'vertical') {
        if (Number.isFinite(source.ax)) {
          return source.ax;
        }
        if (Number.isFinite(source.x1) && Number.isFinite(source.x2)) {
          if (Math.abs(source.x1 - source.x2) <= 1e-9) {
            return source.x1;
          }
          return (source.x1 + source.x2) / 2;
        }
        if (Number.isFinite(source.x)) {
          return source.x;
        }
      }
      return null;
    };

    for (let i = 0; i < trySources.length; i += 1) {
      const value = extract(trySources[i]);
      if (Number.isFinite(value)) {
        return value;
      }
    }

    if (axis === 'horizontal' && Number.isFinite(snap.y)) {
      return snap.y;
    }
    if (axis === 'vertical' && Number.isFinite(snap.x)) {
      return snap.x;
    }
    return null;
  };

  const resolveOrientation = (axis, kind, snap) => {
    const seed = orientationFromKind(kind);

    if (axis === 'horizontal' && !seed.vertical) {
      const edge = snap?.source?.meta?.edge || snap?.movingSegment?.meta?.edge;
      if (edge === 'top' || edge === 'bottom') {
        return { ...seed, vertical: edge };
      }
    }
    if (axis === 'vertical' && !seed.horizontal) {
      const edge = snap?.source?.meta?.edge || snap?.movingSegment?.meta?.edge;
      if (edge === 'left' || edge === 'right') {
        return { ...seed, horizontal: edge };
      }
    }

    return seed;
  };

  const buildGuides = useCallback((snap) => {
    if (!snap?.source) {
      return [];
    }
    const guides = [];
    const guideMap = new Map();
    const sources = [snap.source, ...(snap.relatedSources ?? [])].filter(Boolean);
    if (snap.movingSegment) {
      sources.push(snap.movingSegment);
    }
    const axis = snap.axis ?? null;
    const LINE_AXIS_TOLERANCE = 0.0005;
    const pushGuide = (type, value, source) => {
      if (!Number.isFinite(value)) {
        return;
      }
      const key = `${type}:${value.toFixed(6)}`;
      let entry = guideMap.get(key);
      if (!entry) {
        entry = { type, value, source: source ?? null, sources: [] };
        guideMap.set(key, entry);
        guides.push(entry);
      }
      if (source) {
        entry.sources.push(source);
        if (!entry.source) {
          entry.source = source;
        }
      }
    };
    const pushSegment = (x1, y1, x2, y2, source) => {
      if (![x1, y1, x2, y2].every(Number.isFinite)) {
        return;
      }
      const key = `segment:${x1.toFixed(6)}:${y1.toFixed(6)}:${x2.toFixed(6)}:${y2.toFixed(6)}`;
      let entry = guideMap.get(key);
      if (!entry) {
        entry = { type: 'segment', x1, y1, x2, y2, source: source ?? null, sources: [] };
        guideMap.set(key, entry);
        guides.push(entry);
      }
      if (source) {
        entry.sources.push(source);
        if (!entry.source) {
          entry.source = source;
        }
      }
    };
    const normalizeSegmentCoords = (source) => {
      const ax = Number.isFinite(source.ax) ? source.ax : source.x1;
      const ay = Number.isFinite(source.ay) ? source.ay : source.y1;
      const bx = Number.isFinite(source.bx) ? source.bx : source.x2;
      const by = Number.isFinite(source.by) ? source.by : source.y2;
      return { ax, ay, bx, by };
    };
    sources.forEach((source) => {
      if (source.type === 'box-edge' || source.type === 'moving-edge') {
        const edge = source.meta?.edge;
        const { ax, ay, bx, by } = normalizeSegmentCoords(source);
        const isVertical = edge === 'left' || edge === 'right';
        const isHorizontal = edge === 'top' || edge === 'bottom';
        if (!axis || axis === 'vertical') {
          if (isVertical) {
            const xCoord = axis === 'vertical' && Number.isFinite(snap.x) ? snap.x : ax;
            pushSegment(xCoord, Math.min(ay, by), xCoord, Math.max(ay, by), source);
          }
        }
        if (!axis || axis === 'horizontal') {
          if (isHorizontal) {
            const yCoord = axis === 'horizontal' && Number.isFinite(snap.y) ? snap.y : ay;
            pushSegment(Math.min(ax, bx), yCoord, Math.max(ax, bx), yCoord, source);
          }
        }
      } else if (source.type === 'line') {
        const isVertical = Math.abs(source.ax - source.bx) <= LINE_AXIS_TOLERANCE;
        const isHorizontal = Math.abs(source.ay - source.by) <= LINE_AXIS_TOLERANCE;
        if (!axis) {
          pushSegment(source.ax, source.ay, source.bx, source.by, source);
          return;
        }
        if (axis === 'vertical' && isVertical) {
          const y1 = Math.min(source.ay, source.by);
          const y2 = Math.max(source.ay, source.by);
          pushSegment(snap.x, y1, snap.x, y2, source);
        }
        if (axis === 'horizontal' && isHorizontal) {
          const x1 = Math.min(source.ax, source.bx);
          const x2 = Math.max(source.ax, source.bx);
          pushSegment(x1, snap.y, x2, snap.y, source);
        }
      } else {
        if (!axis || axis === 'vertical') {
          pushGuide('vertical', snap.x, source);
        }
        if (!axis || axis === 'horizontal') {
          pushGuide('horizontal', snap.y, source);
        }
      }
    });
    return guides;
  }, []);

  const handlePointerCapture = (event) => {
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch (error) {
      // ignore pointer capture issues
    }
  };

  const handleBoxPointerDown = useCallback(
    (event, box) => {
      if (addMode) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      setSelection('box', box.id);

      if (readOnly) {
        return;
      }

      const { x, y } = normalisePointer(event);

      clearGuides?.();
      pointerStateRef.current = {
        type: 'move-box',
        id: box.id,
        offsetX: x - box.x,
        offsetY: y - box.y,
        pointerId: event.pointerId,
        snapSuppressed: false,
        snapSource: null,
        snapOrigin: null,
        lastSnap: null,
      };

      handlePointerCapture(event);
    },
    [addMode, clearGuides, normalisePointer, pointerStateRef, readOnly, setSelection]
  );

  const handleBoxPointerMove = useCallback(
    (event) => {
      if (readOnly) {
        return;
      }

      const state = pointerStateRef.current;
      if (!state || state.type !== 'move-box') {
        return;
      }
      event.preventDefault();

      const box = boxesMap[state.id];
      if (!box || hiddenLabelIds?.has(box.labelId)) {
        clearGuides?.();
        return;
      }

      const { x, y } = normalisePointer(event);

      const rawX = clamp(x - state.offsetX, 0, 1 - box.width);
      const rawY = clamp(y - state.offsetY, 0, 1 - box.height);

      let nextX = rawX;
      let nextY = rawY;

      let shouldSnap = true;
      if (state.snapSuppressed && state.snapOrigin) {
        const dx = rawX - state.snapOrigin.x;
        const dy = rawY - state.snapOrigin.y;
        if (Math.hypot(dx, dy) >= releaseDistance) {
          state.snapSuppressed = false;
          state.snapSource = null;
          state.snapOrigin = null;
          state.lastSnap = null;
        } else {
          shouldSnap = false;
        }
      }

      let bestSnap = null;
      if (shouldSnap) {
        const excludeOwners = new Set([box.id]);
        const anchoredPoints = anchoredPointIdsByBox?.get?.(box.id);
        if (anchoredPoints) {
          anchoredPoints.forEach((pointId) => excludeOwners.add(pointId));
        }

        const halfHeight = box.height / 2;
        const halfWidth = box.width / 2;

        const candidates = [
          {
            x: rawX,
            y: rawY,
            kind: 'corner-top-left',
            apply: (snap) => ({ x: snap.x, y: snap.y }),
          },
          {
            x: rawX + box.width,
            y: rawY,
            kind: 'corner-top-right',
            apply: (snap) => ({ x: snap.x - box.width, y: snap.y }),
          },
          {
            x: rawX,
            y: rawY + box.height,
            kind: 'corner-bottom-left',
            apply: (snap) => ({ x: snap.x, y: snap.y - box.height }),
          },
          {
            x: rawX + box.width,
            y: rawY + box.height,
            kind: 'corner-bottom-right',
            apply: (snap) => ({ x: snap.x - box.width, y: snap.y - box.height }),
          },
          {
            x: rawX,
            y: rawY + halfHeight,
            kind: 'edge-left',
            filter: (snap) => snap.axis === 'vertical',
            apply: (snap) => ({ x: snap.x, y: rawY }),
          },
          {
            x: rawX + box.width,
            y: rawY + halfHeight,
            kind: 'edge-right',
            filter: (snap) => snap.axis === 'vertical',
            apply: (snap) => ({ x: snap.x - box.width, y: rawY }),
          },
          {
            x: rawX + halfWidth,
            y: rawY,
            kind: 'edge-top',
            filter: (snap) => snap.axis === 'horizontal',
            apply: (snap) => ({ x: rawX, y: snap.y }),
          },
          {
            x: rawX + halfWidth,
            y: rawY + box.height,
            kind: 'edge-bottom',
            filter: (snap) => snap.axis === 'horizontal',
            apply: (snap) => ({ x: rawX, y: snap.y - box.height }),
          },
        ];

        candidates.forEach((candidate) => {
          const snap = snapDrawingPosition(candidate.x, candidate.y, {
            excludePointOwners: excludeOwners,
            excludeSegmentOwners: excludeOwners,
          });
          if (!snap.snapped) {
            return;
          }
          if (candidate.filter && !candidate.filter(snap)) {
            return;
          }
          if (!bestSnap || snap.distance < bestSnap.snap.distance) {
            bestSnap = { snap, apply: candidate.apply, candidate };
          }
        });
      }

      if (bestSnap) {
        let applied = bestSnap.apply(bestSnap.snap);
        const { snap } = bestSnap;
        const axis = snap.axis ?? null;
        if (axis) {
          const axisValue = resolveAxisValue(axis, snap);
          if (Number.isFinite(axisValue)) {
            const orientation = resolveOrientation(axis, bestSnap.candidate?.kind ?? null, snap);
            if (axis === 'horizontal') {
              if (orientation.vertical === 'top') {
                applied = { ...applied, y: axisValue };
              } else if (orientation.vertical === 'bottom') {
                applied = { ...applied, y: axisValue - box.height };
              }
            } else if (axis === 'vertical') {
              if (orientation.horizontal === 'left') {
                applied = { ...applied, x: axisValue };
              } else if (orientation.horizontal === 'right') {
                applied = { ...applied, x: axisValue - box.width };
              }
            }
          }
        }
        nextX = clamp(applied.x, 0, 1 - box.width);
        nextY = clamp(applied.y, 0, 1 - box.height);
        state.snapSuppressed = true;
        const enrichedSnap = (() => {
          const movingSegment = createMovingSegment(bestSnap.candidate?.kind, bestSnap.snap, {
            x: nextX,
            y: nextY,
            width: box.width,
            height: box.height,
            id: box.id,
          });
          if (!movingSegment) {
            return bestSnap.snap;
          }
          return { ...bestSnap.snap, movingSegment };
        })();
        state.snapSource = enrichedSnap.source ?? null;
        state.snapOrigin = { x: rawX, y: rawY };
        state.lastSnap = { snap: enrichedSnap, applied, candidate: bestSnap.candidate?.kind ?? null };
        setGuides?.(buildGuides(enrichedSnap));
      } else if (state.snapSuppressed && state.lastSnap) {
        const { applied, snap } = state.lastSnap;
        nextX = clamp(applied.x, 0, 1 - box.width);
        nextY = clamp(applied.y, 0, 1 - box.height);
        setGuides?.(buildGuides(snap));
      } else {
        // state.snapSource = null;
        state.lastSnap = null;
        if (!shouldSnap) {
          // keep suppression until release distance met
        } else {
          state.snapSuppressed = false;
          state.snapSource = null;
          state.snapOrigin = null;
        }
        clearGuides?.();
      }

      onUpdateBox?.(box.id, {
        x: nextX,
        y: nextY,
      });
    },
    [
      buildGuides,
      anchoredPointIdsByBox,
      boxesMap,
      clamp,
      clearGuides,
      hiddenLabelIds,
      normalisePointer,
      onUpdateBox,
      pointerStateRef,
      releaseDistance,
      setGuides,
      snapDrawingPosition,
      readOnly,
    ]
  );

  const handleBoxResizePointerDown = useCallback(
    (event, box, handle) => {
      if (addMode) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      if (readOnly) {
        return;
      }

      clearGuides?.();
      pointerStateRef.current = {
        type: 'resize-box',
        id: box.id,
        handle,
        startBox: { ...box },
        pointerId: event.pointerId,
        snapSuppressed: false,
        snapSource: null,
        snapOrigin: null,
        lastSnap: null,
      };

      handlePointerCapture(event);
    },
    [addMode, clearGuides, pointerStateRef, readOnly]
  );

  const handleBoxResizePointerMove = useCallback(
    (event) => {
      if (readOnly) {
        return;
      }

      const state = pointerStateRef.current;
      if (!state || state.type !== 'resize-box') {
        return;
      }
      event.preventDefault();

      const { id, handle, startBox } = state;
      const { x, y } = normalisePointer(event);

      const excludeOwners = new Set([id]);

      const original = {
        left: startBox.x,
        right: startBox.x + startBox.width,
        top: startBox.y,
        bottom: startBox.y + startBox.height,
      };

      let newLeft = original.left;
      let newRight = original.right;
      let newTop = original.top;
      let newBottom = original.bottom;

      const rawCornerX = clamp(x);
      const rawCornerY = clamp(y);
      let cornerX = rawCornerX;
      let cornerY = rawCornerY;

      const startCornerX = handle?.includes('right') ? original.right : original.left;
      const startCornerY = handle?.includes('bottom') ? original.bottom : original.top;
      const deltaX = rawCornerX - startCornerX;
      const deltaY = rawCornerY - startCornerY;

      let axisPreference = null;
      if (handle === 'left' || handle === 'right') {
        axisPreference = 'vertical';
      } else if (handle === 'top' || handle === 'bottom') {
        axisPreference = 'horizontal';
      } else {
        const absDx = Math.abs(deltaX);
        const absDy = Math.abs(deltaY);
        if (absDx > absDy * 1.15) {
          axisPreference = 'vertical';
        } else if (absDy > absDx * 1.15) {
          axisPreference = 'horizontal';
        }
      }

      let shouldSnap = true;
      if (state.snapSuppressed && state.snapOrigin) {
        const dx = rawCornerX - state.snapOrigin.x;
        const dy = rawCornerY - state.snapOrigin.y;
        if (Math.hypot(dx, dy) >= releaseDistance) {
          state.snapSuppressed = false;
          state.snapSource = null;
          state.snapOrigin = null;
          state.lastSnap = null;
        } else {
          shouldSnap = false;
        }
      }

      let snapResult = null;
      if (shouldSnap) {
        snapResult = snapDrawingPosition(cornerX, cornerY, {
          excludePointOwners: excludeOwners,
          excludeSegmentOwners: excludeOwners,
          axisPreference,
        });
      }

      if (snapResult?.snapped) {
        cornerX = snapResult.x;
        cornerY = snapResult.y;
        state.snapSuppressed = true;
        state.snapSource = snapResult.source ?? null;
        state.snapOrigin = { x: rawCornerX, y: rawCornerY };
        state.lastSnap = { snap: snapResult, x: cornerX, y: cornerY };
        setGuides?.(buildGuides(snapResult));
      } else if (state.snapSuppressed && state.lastSnap) {
        cornerX = state.lastSnap.x;
        cornerY = state.lastSnap.y;
        setGuides?.(buildGuides(state.lastSnap.snap));
      } else {
        state.snapSource = null;
        state.lastSnap = null;
        if (!shouldSnap) {
          // keep suppression until release distance met
        } else {
          state.snapSuppressed = false;
          state.snapOrigin = null;
        }
        clearGuides?.();
      }

      if (handle.includes('left')) {
        newLeft = clamp(Math.min(cornerX, original.right - minBoxSize), 0, original.right - minBoxSize);
      }
      if (handle.includes('right')) {
        newRight = clamp(Math.max(cornerX, original.left + minBoxSize), original.left + minBoxSize, 1);
      }
      if (handle.includes('top')) {
        newTop = clamp(Math.min(cornerY, original.bottom - minBoxSize), 0, original.bottom - minBoxSize);
      }
      if (handle.includes('bottom')) {
        newBottom = clamp(Math.max(cornerY, original.top + minBoxSize), original.top + minBoxSize, 1);
      }

      const boundedLeft = clamp(newLeft, 0, newRight - minBoxSize);
      const boundedRight = clamp(newRight, boundedLeft + minBoxSize, 1);
      const boundedTop = clamp(newTop, 0, newBottom - minBoxSize);
      const boundedBottom = clamp(newBottom, boundedTop + minBoxSize, 1);

      const nextX = boundedLeft;
      const nextY = boundedTop;
      const nextWidth = clamp(boundedRight - boundedLeft, minBoxSize, 1);
      const nextHeight = clamp(boundedBottom - boundedTop, minBoxSize, 1);

      onUpdateBox?.(id, {
        x: nextX,
        y: nextY,
        width: nextWidth,
        height: nextHeight,
      });
    },
    [
      clamp,
      clearGuides,
      minBoxSize,
      normalisePointer,
      onUpdateBox,
      pointerStateRef,
      readOnly,
      releaseDistance,
      setGuides,
      snapDrawingPosition,
      buildGuides,
    ]
  );

  return {
    handleBoxPointerDown,
    handleBoxPointerMove,
    handleBoxResizePointerDown,
    handleBoxResizePointerMove,
  };
};

export default useBoxInteractions;
