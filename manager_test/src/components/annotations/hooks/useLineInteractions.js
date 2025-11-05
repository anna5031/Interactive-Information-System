import { useCallback } from 'react';

const useLineInteractions = ({
  addMode,
  readOnly = false,
  pointerStateRef,
  linesMap,
  hiddenLabelIds,
  normalisePointer,
  clamp,
  applyAxisLock,
  snapLineWithState,
  snapLineEndpointWithState,
  onUpdateLine,
  setSelection,
  setGuides,
  clearGuides,
  snapReleaseDistance,
}) => {
  const releaseDistance = Number.isFinite(snapReleaseDistance) ? snapReleaseDistance : 0.04;

  const createMovingLineSegment = (line, ownerId, axis) => {
    if (!line || !ownerId) {
      return null;
    }
    return {
      ownerId,
      type: 'moving-line',
      axis: axis ?? null,
      x1: line.x1,
      y1: line.y1,
      x2: line.x2,
      y2: line.y2,
      ax: line.x1,
      ay: line.y1,
      bx: line.x2,
      by: line.y2,
    };
  };

  const determineAxisPreference = (line) => {
    if (!line) {
      return null;
    }
    const dx = Math.abs(line.x2 - line.x1);
    const dy = Math.abs(line.y2 - line.y1);
    if (dx <= dy * 0.2) {
      return 'vertical';
    }
    if (dy <= dx * 0.2) {
      return 'horizontal';
    }
    return null;
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
        entry = { type, value, source: source ?? null, sources: source ? [source] : [] };
        guideMap.set(key, entry);
        guides.push(entry);
      } else if (source) {
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
        entry = { type: 'segment', x1, y1, x2, y2, source: source ?? null, sources: source ? [source] : [] };
        guideMap.set(key, entry);
        guides.push(entry);
      } else if (source) {
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
        const isVertical = edge === 'left' || edge === 'right';
        const isHorizontal = edge === 'top' || edge === 'bottom';
        if (!axis || axis === 'vertical') {
          if (isVertical) {
            const { ax, ay, by } = normalizeSegmentCoords(source);
            const xCoord = axis === 'vertical' && Number.isFinite(snap.x) ? snap.x : ax;
            pushSegment(xCoord, Math.min(ay, by), xCoord, Math.max(ay, by), source);
          }
        }
        if (!axis || axis === 'horizontal') {
          if (isHorizontal) {
            const { ay, ax, bx } = normalizeSegmentCoords(source);
            const yCoord = axis === 'horizontal' && Number.isFinite(snap.y) ? snap.y : ay;
            pushSegment(Math.min(ax, bx), yCoord, Math.max(ax, bx), yCoord, source);
          }
        }
      } else if (source.type === 'line-end') {
        const end = source.meta?.end === 'end' ? 'end' : 'start';
        const coords = end === 'end' ? { x: source.bx, y: source.by } : { x: source.ax, y: source.ay };
        pushGuide('vertical', coords.x, source);
        pushGuide('horizontal', coords.y, source);
      } else if (source.type === 'box-corner' || source.type === 'point') {
        pushGuide('vertical', source.x, source);
        pushGuide('horizontal', source.y, source);
      } else if (source.type === 'line' || source.type === 'moving-line') {
        const { ax: x1, ay: y1, bx: x2, by: y2 } = normalizeSegmentCoords(source);
        const isVertical = Math.abs(x1 - x2) <= LINE_AXIS_TOLERANCE;
        const isHorizontal = Math.abs(y1 - y2) <= LINE_AXIS_TOLERANCE;
        if (!axis) {
          pushSegment(x1, y1, x2, y2, source);
          return;
        }
        if (axis === 'vertical' && isVertical) {
          const segY1 = Math.min(y1, y2);
          const segY2 = Math.max(y1, y2);
          pushSegment(snap.x, segY1, snap.x, segY2, source);
        }
        if (axis === 'horizontal' && isHorizontal) {
          const segX1 = Math.min(x1, x2);
          const segX2 = Math.max(x1, x2);
          pushSegment(segX1, snap.y, segX2, snap.y, source);
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

  const handleLinePointerDown = useCallback(
    (event, line) => {
      if (addMode) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      setSelection('line', line.id);

      if (readOnly) {
        return;
      }

      const { x, y } = normalisePointer(event);

      clearGuides?.();
      pointerStateRef.current = {
        type: 'move-line',
        id: line.id,
        startX: x,
        startY: y,
        startLine: { ...line },
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

  const handleLinePointerMove = useCallback(
    (event) => {
      if (readOnly) {
        return;
      }

      const state = pointerStateRef.current;
      if (!state || state.type !== 'move-line') {
        return;
      }
      event.preventDefault();

      const line = linesMap[state.id];
      if (!line || hiddenLabelIds?.has(line.labelId)) {
        clearGuides?.();
        return;
      }

      const { x, y } = normalisePointer(event);
      const deltaX = x - state.startX;
      const deltaY = y - state.startY;

      const rawLine = {
        x1: clamp(state.startLine.x1 + deltaX),
        y1: clamp(state.startLine.y1 + deltaY),
        x2: clamp(state.startLine.x2 + deltaX),
        y2: clamp(state.startLine.y2 + deltaY),
      };

      let next = applyAxisLock(rawLine);

      let shouldSnap = true;
      if (state.snapSuppressed && state.snapOrigin) {
        const startDelta = Math.hypot(rawLine.x1 - state.snapOrigin.x1, rawLine.y1 - state.snapOrigin.y1);
        const endDelta = Math.hypot(rawLine.x2 - state.snapOrigin.x2, rawLine.y2 - state.snapOrigin.y2);
        if (Math.max(startDelta, endDelta) >= releaseDistance) {
          state.snapSuppressed = false;
          state.snapSource = null;
          state.snapOrigin = null;
          state.lastSnap = null;
        } else {
          shouldSnap = false;
        }
      }

      let snapResult = null;
      const axisPreference = determineAxisPreference(state.startLine);
      if (shouldSnap) {
        snapResult = snapLineWithState(next, line.id, axisPreference);
      }

      if (snapResult?.snap) {
        next = snapResult.line;
        next = applyAxisLock(next);
        state.snapSuppressed = true;
        const enrichedSnap = (() => {
          const movingSegment = createMovingLineSegment(next, line.id, snapResult.snap.axis ?? null);
          if (!movingSegment) {
            return snapResult.snap;
          }
          return { ...snapResult.snap, movingSegment };
        })();
        state.snapSource = enrichedSnap.source ?? null;
        state.snapOrigin = { ...rawLine };
        state.lastSnap = { line: { ...next }, snap: enrichedSnap };
        setGuides?.(buildGuides(enrichedSnap));
      } else if (state.snapSuppressed && state.lastSnap) {
        next = { ...state.lastSnap.line };
        setGuides?.(buildGuides(state.lastSnap.snap));
      } else {
        state.snapSource = null;
        state.lastSnap = null;
        if (!shouldSnap) {
          // keep suppression until movement exceeds release distance
        } else {
          state.snapSuppressed = false;
          state.snapOrigin = null;
        }
        clearGuides?.();
      }

      next = applyAxisLock(next);

      onUpdateLine?.(line.id, next);
    },
    [
      applyAxisLock,
      buildGuides,
      clamp,
      clearGuides,
      hiddenLabelIds,
      linesMap,
      normalisePointer,
      onUpdateLine,
      pointerStateRef,
      readOnly,
      releaseDistance,
      setGuides,
      snapLineWithState,
    ]
  );

  const handleLineHandlePointerDown = useCallback(
    (event, line, handle) => {
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
        type: 'resize-line',
        id: line.id,
        handle,
        startLine: { ...line },
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

  const handleLineResizeMove = useCallback(
    (event) => {
      if (readOnly) {
        return;
      }

      const state = pointerStateRef.current;
      if (!state || state.type !== 'resize-line') {
        return;
      }
      event.preventDefault();

      const { id, handle, startLine } = state;
      const { x, y } = normalisePointer(event);

      const rawX = clamp(x);
      const rawY = clamp(y);

      let next = { ...startLine };

      if (handle === 'start') {
        next.x1 = rawX;
        next.y1 = rawY;
      } else {
        next.x2 = rawX;
        next.y2 = rawY;
      }

      let shouldSnap = true;
      if (state.snapSuppressed && state.snapOrigin && state.snapOrigin.handle === handle) {
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

      let snapResult = null;
      const axisPreference = determineAxisPreference(startLine);
      if (shouldSnap) {
        const endpoint = handle === 'start' ? 'start' : 'end';
        snapResult = snapLineEndpointWithState(next, endpoint, id, axisPreference);
      }

      if (snapResult?.snap) {
        next = snapResult.line;
        next = applyAxisLock(next);
        state.snapSuppressed = true;
        const enrichedSnap = (() => {
          const movingSegment = createMovingLineSegment(next, id, snapResult.snap.axis ?? null);
          if (!movingSegment) {
            return snapResult.snap;
          }
          return { ...snapResult.snap, movingSegment };
        })();
        state.snapSource = enrichedSnap.source ?? null;
        state.snapOrigin = { x: rawX, y: rawY, handle };
        state.lastSnap = { line: { ...next }, snap: enrichedSnap, handle };
        setGuides?.(buildGuides(enrichedSnap));
      } else if (state.snapSuppressed && state.lastSnap && state.lastSnap.handle === handle) {
        next = { ...state.lastSnap.line };
        setGuides?.(buildGuides(state.lastSnap.snap));
      } else {
        state.snapSource = null;
        if (!shouldSnap) {
          // keep suppression until release distance met
        } else {
          state.snapSuppressed = false;
          state.snapOrigin = null;
        }
        state.lastSnap = null;
        clearGuides?.();
      }

      next = applyAxisLock(next);

      onUpdateLine?.(id, next);
    },
    [
      applyAxisLock,
      buildGuides,
      clamp,
      clearGuides,
      normalisePointer,
      onUpdateLine,
      pointerStateRef,
      readOnly,
      releaseDistance,
      setGuides,
      snapLineEndpointWithState,
    ]
  );

  return {
    handleLinePointerDown,
    handleLinePointerMove,
    handleLineHandlePointerDown,
    handleLineResizeMove,
  };
};

export default useLineInteractions;
