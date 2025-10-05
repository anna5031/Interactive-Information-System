import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import PropTypes from 'prop-types';
import LABEL_CONFIG, { getLabelById, isLineLabel, isBoxLabel, isPointLabel } from '../../config/annotationConfig';
import styles from './AnnotationCanvas.module.css';

const clamp = (value, min = 0, max = 1) => {
  if (Number.isNaN(value)) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
};

const MIN_BOX_SIZE = 0.01;
const MIN_LINE_LENGTH = 0.01;
const LINE_SNAP_THRESHOLD = 0.02;
const AXIS_LOCK_ANGLE_DEG = 5;
const AXIS_LOCK_TOLERANCE = Math.tan((AXIS_LOCK_ANGLE_DEG * Math.PI) / 180);

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

  const projectPointToSegment = (px, py, ax, ay, bx, by) => {
    const dx = bx - ax;
    const dy = by - ay;
    const lengthSquared = dx * dx + dy * dy;
    if (lengthSquared <= Number.EPSILON) {
      const distance = Math.hypot(px - ax, py - ay);
      return { t: 0, x: ax, y: ay, distance };
    }

    const t = clamp(((px - ax) * dx + (py - ay) * dy) / lengthSquared, 0, 1);
    const x = ax + dx * t;
    const y = ay + dy * t;
    const distance = Math.hypot(px - x, py - y);

    return { t, x, y, distance };
  };

  const findAnchorForPoint = (x, y) => {
    let best = null;

    lines.forEach((line, lineIndex) => {
      const projection = projectPointToSegment(x, y, line.x1, line.y1, line.x2, line.y2);
      if (projection.distance <= LINE_SNAP_THRESHOLD && (!best || projection.distance < best.distance)) {
        best = {
          x: projection.x,
          y: projection.y,
          distance: projection.distance,
          anchor: {
            type: 'line',
            id: line.id,
            index: lineIndex,
            t: projection.t,
          },
        };
      }
    });

    boxes.forEach((box, boxIndex) => {
      const edges = [
        { edge: 'top', ax: box.x, ay: box.y, bx: box.x + box.width, by: box.y },
        { edge: 'bottom', ax: box.x, ay: box.y + box.height, bx: box.x + box.width, by: box.y + box.height },
        { edge: 'left', ax: box.x, ay: box.y, bx: box.x, by: box.y + box.height },
        { edge: 'right', ax: box.x + box.width, ay: box.y, bx: box.x + box.width, by: box.y + box.height },
      ];

      edges.forEach(({ edge, ax, ay, bx, by }) => {
        const projection = projectPointToSegment(x, y, ax, ay, bx, by);
        if (projection.distance <= LINE_SNAP_THRESHOLD && (!best || projection.distance < best.distance)) {
          best = {
            x: projection.x,
            y: projection.y,
            distance: projection.distance,
            anchor: {
              type: 'box',
              id: box.id,
              index: boxIndex,
              edge,
              t: projection.t,
            },
          };
        }
      });
    });

    return best;
  };

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

    const nextX = clamp(x - state.offsetX, 0, 1 - box.width);
    const nextY = clamp(y - state.offsetY, 0, 1 - box.height);

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

    const { id, corner, startX, startY, startBox } = state;
    const { x, y } = normalisePointer(event);

    let { x: nextX, y: nextY, width: nextWidth, height: nextHeight } = startBox;

    if (corner.includes('right')) {
      const delta = x - startX;
      nextWidth = clamp(startBox.width + delta, MIN_BOX_SIZE, 1);
    }

    if (corner.includes('left')) {
      const delta = x - startX;
      const proposedWidth = startBox.width - delta;
      const proposedX = startBox.x + delta;
      if (proposedWidth >= MIN_BOX_SIZE) {
        nextX = clamp(proposedX, 0, 1 - MIN_BOX_SIZE);
        nextWidth = clamp(proposedWidth, MIN_BOX_SIZE, 1);
      } else {
        nextX = startBox.x + startBox.width - MIN_BOX_SIZE;
        nextWidth = MIN_BOX_SIZE;
      }
    }

    if (corner.includes('bottom')) {
      const delta = y - startY;
      nextHeight = clamp(startBox.height + delta, MIN_BOX_SIZE, 1);
    }

    if (corner.includes('top')) {
      const delta = y - startY;
      const proposedHeight = startBox.height - delta;
      const proposedY = startBox.y + delta;
      if (proposedHeight >= MIN_BOX_SIZE) {
        nextY = clamp(proposedY, 0, 1 - MIN_BOX_SIZE);
        nextHeight = clamp(proposedHeight, MIN_BOX_SIZE, 1);
      } else {
        nextY = startBox.y + startBox.height - MIN_BOX_SIZE;
        nextHeight = MIN_BOX_SIZE;
      }
    }

    nextX = clamp(nextX, 0, 1 - nextWidth);
    nextY = clamp(nextY, 0, 1 - nextHeight);

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

    const next = {
      x1: clamp(state.startLine.x1 + deltaX),
      y1: clamp(state.startLine.y1 + deltaY),
      x2: clamp(state.startLine.x2 + deltaX),
      y2: clamp(state.startLine.y2 + deltaY),
    };

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
      next = {
        ...next,
        x1: clamp(x),
        y1: clamp(y),
      };
    } else {
      next = {
        ...next,
        x2: clamp(x),
        y2: clamp(y),
      };
    }

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

    const anchor = findAnchorForPoint(x, y);
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

    if (labelIsLine) {
      const draft = {
        type: 'line',
        labelId: activeLabelId,
        x1: position.x,
        y1: position.y,
        x2: position.x,
        y2: position.y,
      };
      setDraftShape(draft);
      pointerStateRef.current = {
        type: 'add-line',
        pointerId: event.pointerId,
        originX: position.x,
        originY: position.y,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
    } else if (labelIsPoint) {
      const anchor = findAnchorForPoint(position.x, position.y);
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
      const draft = {
        type: 'box',
        labelId: activeLabelId,
        x: position.x,
        y: position.y,
        width: 0,
        height: 0,
      };
      setDraftShape(draft);
      pointerStateRef.current = {
        type: 'add-box',
        pointerId: event.pointerId,
        originX: position.x,
        originY: position.y,
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
      const minX = Math.min(state.originX, x);
      const minY = Math.min(state.originY, y);
      const width = Math.abs(x - state.originX);
      const height = Math.abs(y - state.originY);

      setDraftShape((prev) =>
        prev
          ? {
              ...prev,
              x: clamp(minX, 0, 1 - MIN_BOX_SIZE),
              y: clamp(minY, 0, 1 - MIN_BOX_SIZE),
              width: clamp(width, 0, 1),
              height: clamp(height, 0, 1),
            }
          : prev
      );
    } else if (state.type === 'add-line') {
      setDraftShape((prev) =>
        prev
          ? {
              ...prev,
              ...(() => {
                const dx = x - state.originX;
                const dy = y - state.originY;
                const absDx = Math.abs(dx);
                const absDy = Math.abs(dy);

                if (absDy <= absDx * AXIS_LOCK_TOLERANCE) {
                  return {
                    x2: clamp(x),
                    y2: clamp(state.originY),
                  };
                }

                if (absDx <= absDy * AXIS_LOCK_TOLERANCE) {
                  return {
                    x2: clamp(state.originX),
                    y2: clamp(y),
                  };
                }

                return {
                  x2: clamp(x),
                  y2: clamp(y),
                };
              })(),
            }
          : prev
      );
    }
  };

  const renderBoxHandles = (box) => {
    const corners = ['top-left', 'top-right', 'bottom-left', 'bottom-right'];
    const cursorMap = {
      'top-left': 'nwse-resize',
      'top-right': 'nesw-resize',
      'bottom-left': 'nesw-resize',
      'bottom-right': 'nwse-resize',
    };

    return corners.map((corner) => {
      const [vertical, horizontal] = corner.split('-');
      const style = {
        top: vertical === 'top' ? 0 : undefined,
        bottom: vertical === 'bottom' ? 0 : undefined,
        left: horizontal === 'left' ? 0 : undefined,
        right: horizontal === 'right' ? 0 : undefined,
        cursor: cursorMap[corner] || 'pointer',
      };

      return (
        <div
          key={corner}
          role='presentation'
          className={styles.resizeHandle}
          style={style}
          onPointerDown={(event) => handleBoxResizePointerDown(event, box, corner)}
          onPointerMove={handleBoxResizePointerMove}
          onPointerUp={handlePointerUp}
        />
      );
    });
  };

  const renderBox = (box) => {
    const label = getLabelById(box.labelId) || LABEL_CONFIG[0];
    const borderColor = label?.color || '#f59e0b';
    const isSelected = selectedItem?.type === 'box' && selectedItem.id === box.id;

    return (
      <div
        key={box.id}
        role='presentation'
        className={`${styles.annotation} ${isSelected ? styles.selected : ''}`}
        style={{
          left: `${box.x * 100}%`,
          top: `${box.y * 100}%`,
          width: `${box.width * 100}%`,
          height: `${box.height * 100}%`,
          borderColor,
        }}
        onPointerDown={(event) => handleBoxPointerDown(event, box)}
        onPointerMove={handleBoxPointerMove}
        onPointerUp={handlePointerUp}
      >
        <span className={styles.label} style={{ backgroundColor: borderColor }}>
          {label?.name || box.labelId}
        </span>
        {isSelected && renderBoxHandles(box)}
      </div>
    );
  };

  const renderLine = (line) => {
    const label = getLabelById(line.labelId) || LABEL_CONFIG[0];
    const stroke = label?.color || '#f59e0b';

    const isSelected = selectedItem?.type === 'line' && selectedItem.id === line.id;

    return (
      <g key={line.id} className={styles.lineGroup}>
        <line
          x1={line.x1 * imageBox.width}
          y1={line.y1 * imageBox.height}
          x2={line.x2 * imageBox.width}
          y2={line.y2 * imageBox.height}
          stroke={stroke}
          strokeWidth={4}
          strokeLinecap='round'
          className={styles.line}
          onPointerDown={(event) => handleLinePointerDown(event, line)}
          onPointerMove={handleLinePointerMove}
          onPointerUp={handlePointerUp}
        />
        {isSelected && (
          <>
            <circle
              cx={line.x1 * imageBox.width}
              cy={line.y1 * imageBox.height}
              r={8}
              className={styles.lineHandle}
              onPointerDown={(event) => handleLineHandlePointerDown(event, line, 'start')}
              onPointerMove={handleLineResizeMove}
              onPointerUp={handlePointerUp}
            />
            <circle
              cx={line.x2 * imageBox.width}
              cy={line.y2 * imageBox.height}
              r={8}
              className={styles.lineHandle}
              onPointerDown={(event) => handleLineHandlePointerDown(event, line, 'end')}
              onPointerMove={handleLineResizeMove}
              onPointerUp={handlePointerUp}
            />
          </>
        )}
      </g>
    );
  };

  const renderPoint = (point) => {
    const label = getLabelById(point.labelId) || LABEL_CONFIG[0];
    const fill = label?.color || '#ef4444';
    const isSelected = selectedItem?.type === 'point' && selectedItem.id === point.id;

    return (
      <circle
        key={point.id}
        cx={point.x * imageBox.width}
        cy={point.y * imageBox.height}
        r={isSelected ? 10 : 7}
        className={`${styles.point} ${isSelected ? styles.pointSelected : ''}`}
        fill={fill}
        onPointerDown={(event) => handlePointPointerDown(event, point)}
        onPointerMove={handlePointPointerMove}
        onPointerUp={handlePointerUp}
      />
    );
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
    const label = getLabelById(draftShape.labelId) || LABEL_CONFIG[0];
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
        {visibleBoxes.map((box) => renderBox(box))}
        <svg className={styles.lineLayer} width={imageBox.width} height={imageBox.height}>
          {visibleLines.map((line) => renderLine(line))}
        </svg>
        <svg className={styles.pointLayer} width={imageBox.width} height={imageBox.height}>
          {visiblePoints.map((point) => renderPoint(point))}
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
