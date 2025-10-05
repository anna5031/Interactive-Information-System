import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import PropTypes from 'prop-types';
import LABEL_CONFIG, { getLabelById, isLineLabel, isBoxLabel, isPointLabel } from '../../config/annotationConfig';
import styles from './AnnotationCanvas.module.css';
import BoxAnnotation from './shapes/BoxAnnotation';
import LineAnnotation from './shapes/LineAnnotation';
import PointAnnotation from './shapes/PointAnnotation';
import useBoxInteractions from './hooks/useBoxInteractions';
import useLineInteractions from './hooks/useLineInteractions';
import usePointInteractions from './hooks/usePointInteractions';
import {
  clamp,
  LINE_SNAP_THRESHOLD,
  AXIS_LOCK_TOLERANCE,
  buildSnapPoints,
  buildSnapSegments,
  snapPosition,
  findAnchorForPoint,
  applyAxisLockToLine,
  snapLineEndpoints,
  snapSpecificLineEndpoint,
} from './utils/canvasGeometry';

const MIN_BOX_SIZE = 0.01;
const MIN_LINE_LENGTH = 0.01;
const getLabel = (labelId) => getLabelById(labelId) || LABEL_CONFIG[0];

const AnnotationCanvas = ({
  imageUrl,
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
}) => {
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const pointerStateRef = useRef(null);
  const imageBoxRef = useRef({ offsetX: 0, offsetY: 0, width: 0, height: 0 });

  const [imageBox, setImageBox] = useState({ offsetX: 0, offsetY: 0, width: 0, height: 0 });
  const [draftShape, setDraftShape] = useState(null);

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

  const boxesMap = useMemo(() => {
    return boxes.reduce((acc, box) => {
      acc[box.id] = box;
      return acc;
    }, {});
  }, [boxes]);

  const linesMap = useMemo(() => {
    return lines.reduce((acc, line) => {
      acc[line.id] = line;
      return acc;
    }, {});
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
  }, []);

  useEffect(() => {
    updateImageBox();
  }, [updateImageBox, imageUrl]);

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

  const normalisePointer = (event) => {
    const container = containerRef.current;
    const metrics = imageBoxRef.current;

    if (!container || !metrics.width || !metrics.height) {
      return { x: 0, y: 0, isInside: false };
    }

    const containerRect = container.getBoundingClientRect();
    const localX = event.clientX - containerRect.left - metrics.offsetX;
    const localY = event.clientY - containerRect.top - metrics.offsetY;

    const x = clamp(localX / metrics.width);
    const y = clamp(localY / metrics.height);
    const isInside = localX >= 0 && localY >= 0 && localX <= metrics.width && localY <= metrics.height;

    return { x, y, localX, localY, isInside };
  };

  const setSelection = (type, id) => {
    onSelect?.({ type, id });
  };

  const snapDrawingPosition = (x, y, options = {}) =>
    snapPosition({ x, y, snapPoints, snapSegments, ...options });

  const getAnchorForPoint = (x, y) =>
    findAnchorForPoint(x, y, lines, boxes, LINE_SNAP_THRESHOLD);

  const applyAxisLock = (line) => applyAxisLockToLine(line, AXIS_LOCK_TOLERANCE);

  const snapLineWithState = (line, excludeId) =>
    snapLineEndpoints({ line, snapPoints, snapSegments, excludeId });

  const snapLineEndpointWithState = (line, endpoint, excludeId) =>
    snapSpecificLineEndpoint({ line, endpoint, snapPoints, snapSegments, excludeId });

  const handleBoxPointerDown = (event, box) => {
    if (addMode) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    setSelection('box', box.id);

    const { x, y } = normalisePointer(event);

    pointerStateRef.current = {
      type: 'move-box',
      id: box.id,
      offsetX: x - box.x,
      offsetY: y - box.y,
      pointerId: event.pointerId,
    };

    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handleBoxPointerMove = (event) => {
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
  };

  const handleBoxResizePointerDown = (event, box, corner) => {
    if (addMode) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    const { x, y } = normalisePointer(event);

    pointerStateRef.current = {
      type: 'resize-box',
      id: box.id,
      corner,
      startX: x,
      startY: y,
      startBox: { ...box },
      pointerId: event.pointerId,
    };

    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handleBoxResizePointerMove = (event) => {
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
      newLeft = clamp(Math.min(cornerX, original.right - MIN_BOX_SIZE), 0, original.right - MIN_BOX_SIZE);
    }
    if (corner.includes('right')) {
      newRight = clamp(Math.max(cornerX, original.left + MIN_BOX_SIZE), original.left + MIN_BOX_SIZE, 1);
    }
    if (corner.includes('top')) {
      newTop = clamp(Math.min(cornerY, original.bottom - MIN_BOX_SIZE), 0, original.bottom - MIN_BOX_SIZE);
    }
    if (corner.includes('bottom')) {
      newBottom = clamp(Math.max(cornerY, original.top + MIN_BOX_SIZE), original.top + MIN_BOX_SIZE, 1);
    }

    const boundedLeft = clamp(newLeft, 0, newRight - MIN_BOX_SIZE);
    const boundedRight = clamp(newRight, boundedLeft + MIN_BOX_SIZE, 1);
    const boundedTop = clamp(newTop, 0, newBottom - MIN_BOX_SIZE);
    const boundedBottom = clamp(newBottom, boundedTop + MIN_BOX_SIZE, 1);

    const nextX = boundedLeft;
    const nextY = boundedTop;
    const nextWidth = clamp(boundedRight - boundedLeft, MIN_BOX_SIZE, 1);
    const nextHeight = clamp(boundedBottom - boundedTop, MIN_BOX_SIZE, 1);

    onUpdateBox?.(id, {
      x: nextX,
      y: nextY,
      width: nextWidth,
      height: nextHeight,
    });
  };

  const handleLinePointerDown = (event, line) => {
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

    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handleLinePointerMove = (event) => {
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
  };

  const handleLineHandlePointerDown = (event, line, handle) => {
    if (addMode) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    const { x, y } = normalisePointer(event);

    pointerStateRef.current = {
      type: 'resize-line',
      id: line.id,
      handle,
      startX: x,
      startY: y,
      startLine: { ...line },
      pointerId: event.pointerId,
    };

    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handleLineResizeMove = (event) => {
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
  };

  const handlePointPointerDown = (event, point) => {
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

    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointPointerMove = (event) => {
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
  };

  const handlePointerUp = (event) => {
    const state = pointerStateRef.current;
    if (!state) {
      return;
    }

    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
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

    if (state.type === 'move-point') {
      // point updates already applied during move
    }

    pointerStateRef.current = null;
  };

  const handleCanvasPointerDown = (event) => {
    const labelIsLine = isLineLabel(activeLabelId);
    const labelIsPoint = isPointLabel(activeLabelId);

    if (!addMode) {
      const { isInside } = normalisePointer(event);
      if (!isInside) {
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
    const state = pointerStateRef.current;
    if (!state || (state.type !== 'add-box' && state.type !== 'add-line')) {
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
                const dx = snappedX - state.originX;
                const dy = snappedY - state.originY;
                const absDx = Math.abs(dx);
                const absDy = Math.abs(dy);

                if (absDy <= absDx * AXIS_LOCK_TOLERANCE) {
                  return {
                    x2: snappedX,
                    y2: clamp(state.originY),
                  };
                }

                if (absDx <= absDy * AXIS_LOCK_TOLERANCE) {
                  return {
                    x2: clamp(state.originX),
                    y2: snappedY,
                  };
                }

                return {
                  x2: snappedX,
                  y2: snappedY,
                };
              })(),
            }
          : prev
      );
    }
  };

  const overlayStyle = useMemo(() => {
    return {
      left: `${imageBox.offsetX}px`,
      top: `${imageBox.offsetY}px`,
      width: `${imageBox.width}px`,
      height: `${imageBox.height}px`,
    };
  }, [imageBox]);

  const draftElements = [];
  if (draftShape?.type === 'box') {
    const label = getLabel(draftShape.labelId);
    draftElements.push(
      <div
        key='draft-box'
        className={styles.annotation}
        style={{
          left: `${draftShape.x * 100}%`,
          top: `${draftShape.y * 100}%`,
          width: `${draftShape.width * 100}%`,
          height: `${draftShape.height * 100}%`,
          borderColor: label?.color || '#94a3b8',
          opacity: 0.6,
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

  return (
    <div
      ref={containerRef}
      className={`${styles.canvas} ${addMode ? styles.adding : ''}`}
      onPointerDown={handleCanvasPointerDown}
      onPointerMove={handleCanvasPointerMove}
      onPointerUp={handlePointerUp}
    >
      <img ref={imageRef} src={imageUrl} alt='floor plan' className={styles.image} />
      <div className={styles.overlay} style={overlayStyle}>
        {draftElements}
        {visibleBoxes.map((box) => (
          <BoxAnnotation
            key={box.id}
            box={box}
            label={getLabel(box.labelId)}
            isSelected={selectedItem?.type === 'box' && selectedItem.id === box.id}
            onPointerDown={handleBoxPointerDown}
            onPointerMove={handleBoxPointerMove}
            onPointerUp={handlePointerUp}
            onResizePointerDown={handleBoxResizePointerDown}
            onResizePointerMove={handleBoxResizePointerMove}
          />
        ))}
        <svg className={styles.lineLayer} width={imageBox.width} height={imageBox.height}>
          {visibleLines.map((line) => (
            <LineAnnotation
              key={line.id}
              line={line}
              label={getLabel(line.labelId)}
              isSelected={selectedItem?.type === 'line' && selectedItem.id === line.id}
              imageBox={imageBox}
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
              onPointerDown={handlePointPointerDown}
              onPointerMove={handlePointPointerMove}
              onPointerUp={handlePointerUp}
            />
          ))}
        </svg>
      </div>
    </div>
  );
};

AnnotationCanvas.propTypes = {
  imageUrl: PropTypes.string.isRequired,
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
};

AnnotationCanvas.defaultProps = {
  selectedItem: null,
  onSelect: undefined,
  onUpdateBox: undefined,
  onUpdateLine: undefined,
  onUpdatePoint: undefined,
  addMode: false,
  activeLabelId: LABEL_CONFIG[0]?.id ?? '0',
  onAddShape: undefined,
  hiddenLabelIds: undefined,
};

export default AnnotationCanvas;
