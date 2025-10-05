import { useCallback } from 'react';

const usePointInteractions = ({
  pointerStateRef,
  normalisePointer,
  getAnchorForPoint,
  onUpdatePoint,
  setSelection,
  addMode,
}) => {
  const handlePointerCapture = (event) => {
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch (error) {
      // ignore pointer capture issues
    }
  };

  const handlePointPointerDown = useCallback(
    (event, point) => {
      event.preventDefault();
      event.stopPropagation();

      setSelection('point', point.id);

      if (addMode) {
        return;
      }

      pointerStateRef.current = {
        type: 'move-point',
        id: point.id,
      };

      handlePointerCapture(event);
    },
    [addMode, pointerStateRef, setSelection]
  );

  const handlePointPointerMove = useCallback(
    (event) => {
      const state = pointerStateRef.current;
      if (!state || state.type !== 'move-point') {
        return;
      }
      event.preventDefault();

      const { x, y, isInside } = normalisePointer(event);
      if (!isInside) {
        return;
      }

      const anchor = getAnchorForPoint(x, y);
      if (!anchor) {
        return;
      }

      onUpdatePoint?.(state.id, {
        x: anchor.x,
        y: anchor.y,
        anchor: anchor.anchor,
      });
    },
    [getAnchorForPoint, normalisePointer, onUpdatePoint, pointerStateRef]
  );

  return {
    handlePointPointerDown,
    handlePointPointerMove,
  };
};

export default usePointInteractions;
