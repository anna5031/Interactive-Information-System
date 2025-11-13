import { useCallback } from 'react';
import { AXIS_LOCK_TOLERANCE, findVerticalSnap, findHorizontalSnap, DRAW_SNAP_DISTANCE } from '../utils/canvasGeometry';

const useLineInteractions = ({
  addMode,
  readOnly = false,
  pointerStateRef,
  linesMap,
  hiddenLabelIds,
  normalisePointer,
  clamp,
  onUpdateLine,
  setSelection,
  setGuides,
  clearGuides,
  snapReleaseDistance,
  snapPoints,
  snapSegments,
}) => {
  const releaseDistance = Number.isFinite(snapReleaseDistance) ? snapReleaseDistance : 0.04;

  const buildGuidesFromAxisSnaps = useCallback((snaps) => {
    if (!snaps || snaps.length === 0) {
      return [];
    }
    const guides = [];
    snaps.forEach((snap) => {
      if (!snap || !snap.snapped || !Array.isArray(snap.sources) || snap.sources.length === 0) {
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

      const line = linesMap.get(state.id);
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

      let nextX1 = rawLine.x1;
      let nextY1 = rawLine.y1;
      let nextX2 = rawLine.x2;
      let nextY2 = rawLine.y2;

      let shouldSnap = true;
      if (state.snapSuppressed && state.snapOrigin) {
        const startDelta = Math.hypot(rawLine.x1 - state.snapOrigin.x1, rawLine.y1 - state.snapOrigin.y1);
        const endDelta = Math.hypot(rawLine.x2 - state.snapOrigin.x2, rawLine.y2 - state.snapOrigin.y2);
        if (Math.max(startDelta, endDelta) >= releaseDistance) {
          state.snapSuppressed = false;
          state.snapOrigin = null;
          state.lastSnap = null;
        } else {
          shouldSnap = false;
        }
      }

      const activeSnapResults = [];
      if (shouldSnap) {
        const excludeOwners = new Set([line.id]);
        const snapOptions = {
          snapPoints,
          snapSegments,
          distance: DRAW_SNAP_DISTANCE,
          excludePointOwners: excludeOwners,
          excludeSegmentOwners: excludeOwners,
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
          const snapDeltaX = bestVerticalSnap.value - (vSnapSourceKind === 'start' ? rawLine.x1 : rawLine.x2);
          nextX1 = rawLine.x1 + snapDeltaX;
          nextX2 = rawLine.x2 + snapDeltaX;
          activeSnapResults.push(bestVerticalSnap);
        }

        if (bestHorizontalSnap.snapped) {
          const snapDeltaY = bestHorizontalSnap.value - (hSnapSourceKind === 'start' ? rawLine.y1 : rawLine.y2);
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
        setGuides?.(buildGuidesFromAxisSnaps(activeSnapResults));
      } else if (state.snapSuppressed && state.lastSnap) {
        Object.assign(nextLine, state.lastSnap.line);
        setGuides?.(buildGuidesFromAxisSnaps(state.lastSnap.guides));
      } else {
        state.lastSnap = null;
        if (shouldSnap) {
          state.snapSuppressed = false;
          state.snapOrigin = null;
        }
        clearGuides?.();
      }

      onUpdateLine?.(line.id, {
        x1: clamp(nextLine.x1),
        y1: clamp(nextLine.y1),
        x2: clamp(nextLine.x2),
        y2: clamp(nextLine.y2),
      });
    },
    [
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
      snapPoints,
      snapSegments,
      buildGuidesFromAxisSnaps,
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
      const line = linesMap.get(id);
      if (!line) return;

      const { x, y } = normalisePointer(event);

      const rawX = clamp(x);
      const rawY = clamp(y);

      const fixedX = handle === 'start' ? startLine.x2 : startLine.x1;
      const fixedY = handle === 'start' ? startLine.y2 : startLine.y1;

      let nextX = rawX;
      let nextY = rawY;

      let shouldSnap = true;
      if (state.snapSuppressed && state.snapOrigin && state.snapOrigin.handle === handle) {
        const dx = rawX - state.snapOrigin.x;
        const dy = rawY - state.snapOrigin.y;
        if (Math.hypot(dx, dy) >= releaseDistance) {
          state.snapSuppressed = false;
          state.snapOrigin = null;
          state.lastSnap = null;
        } else {
          shouldSnap = false;
        }
      }

      const activeSnapResults = [];
      if (shouldSnap) {
        const excludeOwners = new Set([id]);
        const snapOptions = {
          snapPoints,
          snapSegments,
          distance: DRAW_SNAP_DISTANCE,
          excludePointOwners: excludeOwners,
          excludeSegmentOwners: excludeOwners,
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
        state.snapOrigin = { x: rawX, y: rawY, handle };
        state.lastSnap = { x: nextX, y: nextY, guides: activeSnapResults, handle };
        setGuides?.(buildGuidesFromAxisSnaps(activeSnapResults));
      } else if (state.snapSuppressed && state.lastSnap && state.lastSnap.handle === handle) {
        nextX = state.lastSnap.x;
        nextY = state.lastSnap.y;
        setGuides?.(buildGuidesFromAxisSnaps(state.lastSnap.guides));
      } else {
        state.lastSnap = null;
        if (shouldSnap) {
          state.snapSuppressed = false;
          state.snapOrigin = null;
        }

        const dx = Math.abs(nextX - fixedX);
        const dy = Math.abs(nextY - fixedY);

        const isVertical = dx <= dy * AXIS_LOCK_TOLERANCE;
        const isHorizontal = dy <= dx * AXIS_LOCK_TOLERANCE;

        if (isVertical) {
          nextX = fixedX;
          setGuides?.([
            { type: 'vertical', value: fixedX, sources: [], axis: 'vertical' },
            { type: 'lock_symbol', x: fixedX, y: nextY, lock: 'vertical' },
          ]);
        } else if (isHorizontal) {
          nextY = fixedY;
          setGuides?.([
            { type: 'horizontal', value: fixedY, sources: [], axis: 'horizontal' },
            { type: 'lock_symbol', x: nextX, y: fixedY, lock: 'horizontal' },
          ]);
        } else {
          clearGuides?.();
        }
      }

      // --- Update Logic ---
      const nextLine = { ...line };
      if (handle === 'start') {
        nextLine.x1 = nextX;
        nextLine.y1 = nextY;
      } else {
        nextLine.x2 = nextX;
        nextLine.y2 = nextY;
      }

      onUpdateLine?.(id, nextLine);
    },
    [
      readOnly,
      pointerStateRef,
      normalisePointer,
      clamp,
      onUpdateLine,
      setGuides,
      clearGuides,
      releaseDistance,
      linesMap,
      snapPoints,
      snapSegments,
      buildGuidesFromAxisSnaps,
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
