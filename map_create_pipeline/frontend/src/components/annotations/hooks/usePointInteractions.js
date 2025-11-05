import { useCallback, useRef } from 'react';

const usePointInteractions = ({
  pointerStateRef,
  normalisePointer,
  projectToClosestAnchor, // *** 수정: prop 이름 변경 (getAnchorForPoint -> projectToClosestAnchor) ***
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

    // *** 수정: threshold 없는 projectToClosestAnchor 함수 사용 ***
    const anchor = projectToClosestAnchor(x, y);

    // *** 수정: 앵커를 못찾으면(null) 마지막 위치를 그대로 사용 (업데이트 안함) ***
    if (!anchor) {
      // 캔버스에 선/박스가 아예 없는 엣지 케이스.
      // 이 경우, 포인트는 마지막으로 알려진 x, y에 머무름 (분리되지 않음)
      // 혹은 원한다면 마우스 위치를 따르게 할 수도 있음:
      // onUpdatePoint?.(state.id, { x, y, anchor: null });
      return;
    }

    // 앵커를 찾으면 스냅된 위치로 업데이트
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
