import { useCallback } from 'react';

const useLineInteractions = ({
  addMode,
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
}) => {
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

      const { x, y } = normalisePointer(event);

      pointerStateRef.current = {
        type: 'move-line',
        id: line.id,
        startX: x,
        startY: y,
        startLine: { ...line },
        pointerId: event.pointerId,
      };

      handlePointerCapture(event);
    },
    [addMode, normalisePointer, pointerStateRef, setSelection]
  );

  const handleLinePointerMove = useCallback(
    (event) => {
      const state = pointerStateRef.current;
      if (!state || state.type !== 'move-line') {
        return;
      }
      event.preventDefault();

      const line = linesMap[state.id];
      if (!line || hiddenLabelIds?.has(line.labelId)) {
        return;
      }

      const { x, y } = normalisePointer(event);
      const deltaX = x - state.startX;
      const deltaY = y - state.startY;

      let next = {
        x1: clamp(state.startLine.x1 + deltaX),
        y1: clamp(state.startLine.y1 + deltaY),
        x2: clamp(state.startLine.x2 + deltaX),
        y2: clamp(state.startLine.y2 + deltaY),
      };

      next = applyAxisLock(next);
      next = snapLineWithState(next, line.id);
      next = applyAxisLock(next);

      onUpdateLine?.(line.id, next);
    },
    [applyAxisLock, clamp, hiddenLabelIds, linesMap, normalisePointer, onUpdateLine, pointerStateRef, snapLineWithState]
  );

  const handleLineHandlePointerDown = useCallback(
    (event, line, handle) => {
      if (addMode) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      pointerStateRef.current = {
        type: 'resize-line',
        id: line.id,
        handle,
        startLine: { ...line },
        pointerId: event.pointerId,
      };

      handlePointerCapture(event);
    },
    [addMode, pointerStateRef]
  );

  const handleLineResizeMove = useCallback(
    (event) => {
      const state = pointerStateRef.current;
      if (!state || state.type !== 'resize-line') {
        return;
      }
      event.preventDefault();

      const { id, handle, startLine } = state;
      const { x, y } = normalisePointer(event);

      let next = { ...startLine };

      if (handle === 'start') {
        next.x1 = clamp(x);
        next.y1 = clamp(y);
        next = snapLineEndpointWithState(next, 'start', id);
      } else {
        next.x2 = clamp(x);
        next.y2 = clamp(y);
        next = snapLineEndpointWithState(next, 'end', id);
      }

      next = applyAxisLock(next);

      onUpdateLine?.(id, next);
    },
    [applyAxisLock, clamp, normalisePointer, onUpdateLine, pointerStateRef, snapLineEndpointWithState]
  );

  return {
    handleLinePointerDown,
    handleLinePointerMove,
    handleLineHandlePointerDown,
    handleLineResizeMove,
  };
};

export default useLineInteractions;
