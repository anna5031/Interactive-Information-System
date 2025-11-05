import { useCallback } from 'react';
import { findVerticalSnap, findHorizontalSnap, DRAW_SNAP_DISTANCE } from '../utils/canvasGeometry';

const useBoxInteractions = ({
  addMode,
  readOnly = false,
  pointerStateRef,
  boxesMap,
  hiddenLabelIds,
  normalisePointer,
  // snapDrawingPosition, // 더 이상 handleBoxPointerMove에서 사용되지 않음
  snapPoints,
  snapSegments,
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

  // const createMovingSegment = (kind, snap, { x, y, width, height, id }) => {
  //   if (!kind) {
  //     return null;
  //   }
  //   const base = {
  //     ownerId: id,
  //     type: 'moving-edge',
  //     axis: snap?.axis ?? null,
  //   };

  //   switch (kind) {
  //     case 'edge-left': {
  //       const xCoord = x;
  //       return {
  //         ...base,
  //         meta: { edge: 'left', role: 'moving' },
  //         x1: xCoord,
  //         y1: y,
  //         x2: xCoord,
  //         y2: y + height,
  //         ax: xCoord,
  //         ay: y,
  //         bx: xCoord,
  //         by: y + height,
  //       };
  //     }
  //     case 'edge-right': {
  //       const xCoord = x + width;
  //       return {
  //         ...base,
  //         meta: { edge: 'right', role: 'moving' },
  //         x1: xCoord,
  //         y1: y,
  //         x2: xCoord,
  //         y2: y + height,
  //         ax: xCoord,
  //         ay: y,
  //         bx: xCoord,
  //         by: y + height,
  //       };
  //     }
  //     case 'edge-top': {
  //       const yCoord = y;
  //       return {
  //         ...base,
  //         meta: { edge: 'top', role: 'moving' },
  //         x1: x,
  //         y1: yCoord,
  //         x2: x + width,
  //         y2: yCoord,
  //         ax: x,
  //         ay: yCoord,
  //         bx: x + width,
  //         by: yCoord,
  //       };
  //     }
  //     case 'edge-bottom': {
  //       const yCoord = y + height;
  //       return {
  //         ...base,
  //         meta: { edge: 'bottom', role: 'moving' },
  //         x1: x,
  //         y1: yCoord,
  //         x2: x + width,
  //         y2: yCoord,
  //         ax: x,
  //         ay: yCoord,
  //         bx: x + width,
  //         by: yCoord,
  //       };
  //     }
  //     default:
  //       return null;
  //   }
  // };

  // const orientationFromKind = (kind) => {
  //   switch (kind) {
  //     case 'corner-top-left':
  //       return { vertical: 'top', horizontal: 'left' };
  //     case 'corner-top-right':
  //       return { vertical: 'top', horizontal: 'right' };
  //     case 'corner-bottom-left':
  //       return { vertical: 'bottom', horizontal: 'left' };
  //     case 'corner-bottom-right':
  //       return { vertical: 'bottom', horizontal: 'right' };
  //     case 'edge-top':
  //       return { vertical: 'top', horizontal: null };
  //     case 'edge-bottom':
  //       return { vertical: 'bottom', horizontal: null };
  //     case 'edge-left':
  //       return { vertical: null, horizontal: 'left' };
  //     case 'edge-right':
  //       return { vertical: null, horizontal: 'right' };
  //     default:
  //       return { vertical: null, horizontal: null };
  //   }
  // };

  // const resolveAxisValue = (axis, snap) => {
  //   if (!axis || !snap) {
  //     return null;
  //   }

  //   const trySources = snap.sources ?? [];
  //   if (trySources.length === 0 && snap.source) {
  //     trySources.push(snap.source);
  //   }

  //   const extract = (source) => {
  //     if (!source) {
  //       return null;
  //     }
  //     if (axis === 'horizontal') {
  //       if (Number.isFinite(source.ay)) {
  //         return source.ay;
  //       }
  //       if (Number.isFinite(source.y1) && Number.isFinite(source.y2)) {
  //         if (Math.abs(source.y1 - source.y2) <= 1e-9) {
  //           return source.y1;
  //         }
  //         return (source.y1 + source.y2) / 2;
  //       }
  //       if (Number.isFinite(source.y)) {
  //         return source.y;
  //       }
  //     } else if (axis === 'vertical') {
  //       if (Number.isFinite(source.ax)) {
  //         return source.ax;
  //       }
  //       if (Number.isFinite(source.x1) && Number.isFinite(source.x2)) {
  //         if (Math.abs(source.x1 - source.x2) <= 1e-9) {
  //           return source.x1;
  //         }
  //         return (source.x1 + source.x2) / 2;
  //       }
  //       if (Number.isFinite(source.x)) {
  //         return source.x;
  //       }
  //     }
  //     return null;
  //   };

  //   for (let i = 0; i < trySources.length; i += 1) {
  //     const value = extract(trySources[i]);
  //     if (Number.isFinite(value)) {
  //       return value;
  //     }
  //   }

  //   if (axis === 'horizontal' && Number.isFinite(snap.value)) {
  //     return snap.value;
  //   }
  //   if (axis === 'vertical' && Number.isFinite(snap.value)) {
  //     return snap.value;
  //   }
  //   return null;
  // };

  // const resolveOrientation = (axis, kind, snap) => {
  //   const seed = orientationFromKind(kind);

  //   const source = snap?.sources?.[0] ?? snap?.source ?? null;

  //   if (axis === 'horizontal' && !seed.vertical) {
  //     const edge = source?.meta?.edge || snap?.movingSegment?.meta?.edge;
  //     if (edge === 'top' || edge === 'bottom') {
  //       return { ...seed, vertical: edge };
  //     }
  //   }
  //   if (axis === 'vertical' && !seed.horizontal) {
  //     const edge = source?.meta?.edge || snap?.movingSegment?.meta?.edge;
  //     if (edge === 'left' || edge === 'right') {
  //       return { ...seed, horizontal: edge };
  //     }
  //   }

  //   return seed;
  // };

  // const buildGuides = useCallback((snap) => {
  //   const sources = (snap?.sources ?? (snap?.source ? [snap.source] : [])).filter(Boolean);
  //   if (sources.length === 0) {
  //     return [];
  //   }

  //   const guides = [];
  //   const guideMap = new Map();
  //   if (snap.movingSegment) {
  //     sources.push(snap.movingSegment);
  //   }
  //   const axis = snap.axis ?? null;
  //   const LINE_AXIS_TOLERANCE = 0.0005;

  //   const pushGuide = (type, value, source) => {
  //     if (!Number.isFinite(value)) {
  //       return;
  //     }
  //     const key = `${type}:${value.toFixed(6)}`;
  //     let entry = guideMap.get(key);
  //     if (!entry) {
  //       entry = { type, value, source: source ?? null, sources: [] };
  //       guideMap.set(key, entry);
  //       guides.push(entry);
  //     }
  //     if (source) {
  //       entry.sources.push(source);
  //       if (!entry.source) {
  //         entry.source = source;
  //       }
  //     }
  //   };

  //   const pushSegment = (x1, y1, x2, y2, source) => {
  //     if (![x1, y1, x2, y2].every(Number.isFinite)) {
  //       return;
  //     }
  //     const key = `segment:${x1.toFixed(6)}:${y1.toFixed(6)}:${x2.toFixed(6)}:${y2.toFixed(6)}`;
  //     let entry = guideMap.get(key);
  //     if (!entry) {
  //       entry = { type: 'segment', x1, y1, x2, y2, source: source ?? null, sources: [] };
  //       guideMap.set(key, entry);
  //       guides.push(entry);
  //     }
  //     if (source) {
  //       entry.sources.push(source);
  //       if (!entry.source) {
  //         entry.source = source;
  //       }
  //     }
  //   };

  //   const normalizeSegmentCoords = (source) => {
  //     const ax = Number.isFinite(source.ax) ? source.ax : source.x1;
  //     const ay = Number.isFinite(source.ay) ? source.ay : source.y1;
  //     const bx = Number.isFinite(source.bx) ? source.bx : source.x2;
  //     const by = Number.isFinite(source.by) ? source.by : source.y2;
  //     return { ax, ay, bx, by };
  //   };

  //   const snapValueX = axis === 'vertical' ? snap.value : null;
  //   const snapValueY = axis === 'horizontal' ? snap.value : null;

  //   sources.forEach((source) => {
  //     if (source.type === 'box-edge' || source.type === 'moving-edge') {
  //       const edge = source.meta?.edge;
  //       const { ax, ay, bx, by } = normalizeSegmentCoords(source);
  //       const isVertical = edge === 'left' || edge === 'right';
  //       const isHorizontal = edge === 'top' || edge === 'bottom';
  //       if (!axis || axis === 'vertical') {
  //         if (isVertical) {
  //           const xCoord = snapValueX ?? ax;
  //           pushSegment(xCoord, Math.min(ay, by), xCoord, Math.max(ay, by), source);
  //         }
  //       }
  //       if (!axis || axis === 'horizontal') {
  //         if (isHorizontal) {
  //           const yCoord = snapValueY ?? ay;
  //           pushSegment(Math.min(ax, bx), yCoord, Math.max(ax, bx), yCoord, source);
  //         }
  //       }
  //     } else if (source.type === 'line') {
  //       const { ax, ay, bx, by } = normalizeSegmentCoords(source);
  //       const isVertical = Math.abs(ax - bx) <= LINE_AXIS_TOLERANCE;
  //       const isHorizontal = Math.abs(ay - by) <= LINE_AXIS_TOLERANCE;
  //       if (!axis) {
  //         pushSegment(ax, ay, bx, by, source);
  //         return;
  //       }
  //       if (axis === 'vertical' && isVertical) {
  //         const y1_val = Math.min(ay, by);
  //         const y2_val = Math.max(ay, by);
  //         pushSegment(snapValueX, y1_val, snapValueX, y2_val, source);
  //       }
  //       if (axis === 'horizontal' && isHorizontal) {
  //         const x1_val = Math.min(ax, bx);
  //         const x2_val = Math.max(ax, bx);
  //         pushSegment(x1_val, snapValueY, x2_val, snapValueY, source);
  //       }
  //     } else {
  //       if (!axis || axis === 'vertical') {
  //         pushGuide('vertical', snapValueX ?? source.x, source);
  //       }
  //       if (!axis || axis === 'horizontal') {
  //         pushGuide('horizontal', snapValueY ?? source.y, source);
  //       }
  //     }
  //   });
  //   return guides;
  // }, []);

  const buildGuidesFromAxisSnaps = useCallback((snaps) => {
    // ... (이 함수는 변경 없음)
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
    // ... (이 함수는 변경 없음)
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch (error) {
      // ignore pointer capture issues
    }
  };

  const handleBoxPointerDown = useCallback(
    (event, box) => {
      // ... (이 함수는 변경 없음)
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
        snapOrigin: null,
        lastSnap: null,
      };

      handlePointerCapture(event);
    },
    [addMode, clearGuides, normalisePointer, pointerStateRef, readOnly, setSelection]
  );

  // *** 수정된 부분 시작 ***
  // `handleBoxPointerMove` 함수를 축 기반 스냅 로직으로 교체
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

      // 1. 계산: 마우스 위치 기준 박스의 새 원본(top-left) 좌표
      const rawX = clamp(x - state.offsetX, 0, 1 - box.width);
      const rawY = clamp(y - state.offsetY, 0, 1 - box.height);

      let nextX = rawX;
      let nextY = rawY;

      const activeSnapResults = [];

      // 2. 스냅 억제 로직 (기존과 유사)
      let shouldSnap = true;
      if (state.snapSuppressed && state.snapOrigin) {
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

      // 3. 스냅 탐색 (축 기반)
      if (shouldSnap) {
        const excludeOwners = new Set([box.id]);
        const anchoredPoints = anchoredPointIdsByBox?.get?.(box.id);
        if (anchoredPoints) {
          anchoredPoints.forEach((pointId) => excludeOwners.add(pointId));
        }

        const snapOptions = {
          snapPoints,
          snapSegments,
          distance: DRAW_SNAP_DISTANCE,
          excludePointOwners: excludeOwners,
          excludeSegmentOwners: excludeOwners,
        };

        // 박스의 3개 수직 축 (left, center, right)
        const vCandidates = [
          { value: rawX, kind: 'left' },
          { value: rawX + box.width / 2, kind: 'center' },
          { value: rawX + box.width, kind: 'right' },
        ];

        // 박스의 3개 수평 축 (top, center, bottom)
        const hCandidates = [
          { value: rawY, kind: 'top' },
          { value: rawY + box.height / 2, kind: 'center' },
          { value: rawY + box.height, kind: 'bottom' },
        ];

        let bestVerticalSnap = { snapped: false, distance: Infinity };
        let vSnapSourceKind = 'left';

        vCandidates.forEach((candidate) => {
          const snap = findVerticalSnap({ value: candidate.value, ...snapOptions });
          if (snap.snapped && snap.distance < bestVerticalSnap.distance) {
            bestVerticalSnap = snap;
            vSnapSourceKind = candidate.kind;
          }
        });

        let bestHorizontalSnap = { snapped: false, distance: Infinity };
        let hSnapSourceKind = 'top';

        hCandidates.forEach((candidate) => {
          const snap = findHorizontalSnap({ value: candidate.value, ...snapOptions });
          if (snap.snapped && snap.distance < bestHorizontalSnap.distance) {
            bestHorizontalSnap = snap;
            hSnapSourceKind = candidate.kind;
          }
        });

        // 4. 스냅 적용
        if (bestVerticalSnap.snapped) {
          if (vSnapSourceKind === 'left') nextX = bestVerticalSnap.value;
          else if (vSnapSourceKind === 'center') nextX = bestVerticalSnap.value - box.width / 2;
          else if (vSnapSourceKind === 'right') nextX = bestVerticalSnap.value - box.width;

          activeSnapResults.push(bestVerticalSnap);
        }

        if (bestHorizontalSnap.snapped) {
          if (hSnapSourceKind === 'top') nextY = bestHorizontalSnap.value;
          else if (hSnapSourceKind === 'center') nextY = bestHorizontalSnap.value - box.height / 2;
          else if (hSnapSourceKind === 'bottom') nextY = bestHorizontalSnap.value - box.height;

          activeSnapResults.push(bestHorizontalSnap);
        }
      } // End if (shouldSnap)

      // 5. 스냅 상태 및 가이드라인 처리
      if (activeSnapResults.length > 0) {
        state.snapSuppressed = true;
        state.snapOrigin = { x: rawX, y: rawY };
        state.lastSnap = { x: nextX, y: nextY, guides: activeSnapResults };
        setGuides?.(buildGuidesFromAxisSnaps(activeSnapResults));
      } else if (state.snapSuppressed && state.lastSnap) {
        // 스냅 유지
        nextX = state.lastSnap.x;
        nextY = state.lastSnap.y;
        setGuides?.(buildGuidesFromAxisSnaps(state.lastSnap.guides));
      } else {
        // 스냅 없음
        state.lastSnap = null;
        if (shouldSnap) {
          // 스냅 억제 풀기
          state.snapSuppressed = false;
          state.snapOrigin = null;
        }
        clearGuides?.();
      }

      // 6. 최종 위치 클램핑 및 업데이트
      const finalX = clamp(nextX, 0, 1 - box.width);
      const finalY = clamp(nextY, 0, 1 - box.height);

      onUpdateBox?.(box.id, {
        x: finalX,
        y: finalY,
      });
    },
    [
      // buildGuides, // buildGuides는 이제 사용되지 않음
      buildGuidesFromAxisSnaps, // 새 가이드 빌더 사용
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
      snapPoints, // 필요
      snapSegments, // 필요
      // snapDrawingPosition, // 더 이상 사용되지 않음
      readOnly,
    ]
  );
  // *** 수정된 부분 끝 ***

  const handleBoxResizePointerDown = useCallback(
    (event, box, handle) => {
      // ... (이 함수는 이전 단계에서 수정 완료됨)
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
        snapOrigin: null,
        lastSnap: null,
      };

      handlePointerCapture(event);
    },
    [addMode, clearGuides, pointerStateRef, readOnly]
  );

  const handleBoxResizePointerMove = useCallback(
    (event) => {
      // ... (이 함수는 이전 단계에서 수정 완료됨)
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

      const rawCornerX = clamp(x);
      const rawCornerY = clamp(y);

      let cornerX = rawCornerX;
      let cornerY = rawCornerY;
      let snapX = null;
      let snapY = null;
      let activeSnapResults = [];

      const snapOptions = {
        snapPoints,
        snapSegments,
        distance: DRAW_SNAP_DISTANCE,
        excludePointOwners: excludeOwners,
        excludeSegmentOwners: excludeOwners,
      };

      const isVerticalEdge = handle === 'top' || handle === 'bottom';
      const isHorizontalEdge = handle === 'left' || handle === 'right';
      const isCornerHandle = !isVerticalEdge && !isHorizontalEdge;

      // --- 스냅 로직 ---
      if (isCornerHandle || isHorizontalEdge) {
        snapX = findVerticalSnap({ value: rawCornerX, ...snapOptions });
        if (snapX.snapped) {
          cornerX = snapX.value;
          activeSnapResults.push(snapX);
        }
      }

      if (isCornerHandle || isVerticalEdge) {
        snapY = findHorizontalSnap({ value: rawCornerY, ...snapOptions });
        if (snapY.snapped) {
          cornerY = snapY.value;
          activeSnapResults.push(snapY);
        }
      }

      // --- 스냅 억제 및 가이드 로직 ---
      if (activeSnapResults.length > 0) {
        state.snapSuppressed = true;
        state.snapOrigin = { x: rawCornerX, y: rawCornerY };
        state.lastSnap = { x: cornerX, y: cornerY, guides: activeSnapResults };
        setGuides?.(buildGuidesFromAxisSnaps(activeSnapResults));
      } else if (state.snapSuppressed && state.snapOrigin) {
        const dx = rawCornerX - state.snapOrigin.x;
        const dy = rawCornerY - state.snapOrigin.y;
        if (Math.hypot(dx, dy) < releaseDistance) {
          cornerX = state.lastSnap.x;
          cornerY = state.lastSnap.y;
          setGuides?.(buildGuidesFromAxisSnaps(state.lastSnap.guides));
        } else {
          state.snapSuppressed = false;
          state.snapOrigin = null;
          state.lastSnap = null;
          clearGuides?.();
        }
      } else {
        clearGuides?.();
      }

      // --- 좌표 적용 로직 ---
      let newLeft = original.left;
      let newRight = original.right;
      let newTop = original.top;
      let newBottom = original.bottom;

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
      readOnly,
      pointerStateRef,
      normalisePointer,
      clamp,
      snapPoints,
      snapSegments,
      minBoxSize,
      onUpdateBox,
      setGuides,
      clearGuides,
      releaseDistance,
      buildGuidesFromAxisSnaps,
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
