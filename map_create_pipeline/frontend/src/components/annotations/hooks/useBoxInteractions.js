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
}) => {
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

      pointerStateRef.current = {
        type: 'move-box',
        id: box.id,
        offsetX: x - box.x,
        offsetY: y - box.y,
        pointerId: event.pointerId,
      };

      handlePointerCapture(event);
    },
    [addMode, normalisePointer, pointerStateRef, readOnly, setSelection]
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
        return;
      }

      const { x, y } = normalisePointer(event);

      let nextX = clamp(x - state.offsetX, 0, 1 - box.width);
      let nextY = clamp(y - state.offsetY, 0, 1 - box.height);

      const excludeOwners = new Set([box.id]);
      const anchoredPoints = anchoredPointIdsByBox?.get?.(box.id);
      if (anchoredPoints) {
        anchoredPoints.forEach((pointId) => excludeOwners.add(pointId));
      }
      const candidates = [
        {
          x: nextX,
          y: nextY,
          apply: (snapX, snapY) => ({ x: snapX, y: snapY }),
        },
        {
          x: nextX + box.width,
          y: nextY,
          apply: (snapX, snapY) => ({ x: snapX - box.width, y: snapY }),
        },
        {
          x: nextX,
          y: nextY + box.height,
          apply: (snapX, snapY) => ({ x: snapX, y: snapY - box.height }),
        },
        {
          x: nextX + box.width,
          y: nextY + box.height,
          apply: (snapX, snapY) => ({ x: snapX - box.width, y: snapY - box.height }),
        },
      ];

      let bestSnap = null;
      candidates.forEach((candidate) => {
        const snap = snapDrawingPosition(candidate.x, candidate.y, {
          excludePointOwners: excludeOwners,
          excludeSegmentOwners: excludeOwners,
        });
        if (snap.snapped && (!bestSnap || snap.distance < bestSnap.distance)) {
          bestSnap = { ...snap, apply: candidate.apply };
        }
      });

      if (bestSnap) {
        const applied = bestSnap.apply(bestSnap.x, bestSnap.y);
        nextX = clamp(applied.x, 0, 1 - box.width);
        nextY = clamp(applied.y, 0, 1 - box.height);
      }

      onUpdateBox?.(box.id, {
        x: nextX,
        y: nextY,
      });
    },
    [
      anchoredPointIdsByBox,
      boxesMap,
      clamp,
      hiddenLabelIds,
      normalisePointer,
      onUpdateBox,
      pointerStateRef,
      snapDrawingPosition,
      readOnly,
    ]
  );

  const handleBoxResizePointerDown = useCallback(
    (event, box, corner) => {
      if (addMode) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      if (readOnly) {
        return;
      }

      pointerStateRef.current = {
        type: 'resize-box',
        id: box.id,
        corner,
        startBox: { ...box },
        pointerId: event.pointerId,
      };

      handlePointerCapture(event);
    },
    [addMode, pointerStateRef, readOnly]
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

      const { id, corner, startBox } = state;
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

      let cornerX = clamp(x);
      let cornerY = clamp(y);
      const snap = snapDrawingPosition(cornerX, cornerY, {
        excludePointOwners: excludeOwners,
        excludeSegmentOwners: excludeOwners,
      });
      if (snap.snapped) {
        cornerX = snap.x;
        cornerY = snap.y;
      }

      if (corner.includes('left')) {
        newLeft = clamp(Math.min(cornerX, original.right - minBoxSize), 0, original.right - minBoxSize);
      }
      if (corner.includes('right')) {
        newRight = clamp(Math.max(cornerX, original.left + minBoxSize), original.left + minBoxSize, 1);
      }
      if (corner.includes('top')) {
        newTop = clamp(Math.min(cornerY, original.bottom - minBoxSize), 0, original.bottom - minBoxSize);
      }
      if (corner.includes('bottom')) {
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
    [clamp, minBoxSize, normalisePointer, onUpdateBox, pointerStateRef, readOnly, snapDrawingPosition]
  );

  return {
    handleBoxPointerDown,
    handleBoxPointerMove,
    handleBoxResizePointerDown,
    handleBoxResizePointerMove,
  };
};

export default useBoxInteractions;
