import { useCallback, useRef } from 'react';

const usePointInteractions = ({
  pointerStateRef,
  normalisePointer,
  projectToClosestAnchor,
  onUpdatePoint,
  setSelection,
  addMode,
  readOnly = false,
  clearGuides,
}) => {
  const animationFrameRef = useRef(null);
  const latestPointerEventRef = useRef(null);

  const handlePointerCapture = (event) => {
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch (error) {
      // ignore pointer capture issues
    }
  };

  const processPointMove = useCallback(() => {
    const event = latestPointerEventRef.current;
    const state = pointerStateRef.current;

    animationFrameRef.current = null;

    if (!event || !state || state.type !== 'move-point' || readOnly) {
      return;
    }

    const { x, y, isInside } = normalisePointer(event);
    if (!isInside) {
      return;
    }
    const anchor = projectToClosestAnchor(x, y);

    if (!anchor) {
      return;
    }

    onUpdatePoint?.(state.id, {
      x: anchor.x,
      y: anchor.y,
      anchor: anchor.anchor,
    });
  }, [projectToClosestAnchor, normalisePointer, onUpdatePoint, pointerStateRef, readOnly]);

  const handlePointPointerDown = useCallback(
    (event, point) => {
      event.preventDefault();
      event.stopPropagation();

      setSelection('point', point.id);

      if (addMode || readOnly) {
        return;
      }

      clearGuides?.();
      pointerStateRef.current = {
        type: 'move-point',
        id: point.id,
      };

      handlePointerCapture(event);
    },
    [addMode, clearGuides, pointerStateRef, readOnly, setSelection]
  );

  const handlePointPointerMove = useCallback(
    (event) => {
      if (readOnly) {
        return;
      }

      const state = pointerStateRef.current;
      if (!state || state.type !== 'move-point') {
        return;
      }
      event.preventDefault();

      latestPointerEventRef.current = event;

      if (!animationFrameRef.current) {
        animationFrameRef.current = requestAnimationFrame(processPointMove);
      }
    },
    [readOnly, pointerStateRef, processPointMove]
  );

  const cleanupPointMove = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    latestPointerEventRef.current = null;
  }, []);

  return {
    handlePointPointerDown,
    handlePointPointerMove,
    cleanupPointMove,
  };
};

export default usePointInteractions;
