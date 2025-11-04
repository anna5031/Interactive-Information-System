const clamp = (value, min = 0, max = 1) => {
  if (Number.isNaN(value)) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
};

const LINE_SNAP_THRESHOLD = 0.02;
const DRAW_SNAP_DISTANCE = 0.02;
const AXIS_LOCK_ANGLE_DEG = 5;
const AXIS_LOCK_TOLERANCE = Math.tan((AXIS_LOCK_ANGLE_DEG * Math.PI) / 180);
const EMPTY_SET = Object.freeze(new Set());

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

const buildSnapPoints = (boxes = [], lines = [], points = []) => {
  const list = [];

  boxes.forEach((box) => {
    list.push({ x: box.x, y: box.y, ownerId: box.id, type: 'box-corner', meta: { corner: 'top-left' } });
    list.push({ x: box.x + box.width, y: box.y, ownerId: box.id, type: 'box-corner', meta: { corner: 'top-right' } });
    list.push({ x: box.x, y: box.y + box.height, ownerId: box.id, type: 'box-corner', meta: { corner: 'bottom-left' } });
    list.push({ x: box.x + box.width, y: box.y + box.height, ownerId: box.id, type: 'box-corner', meta: { corner: 'bottom-right' } });
  });

  lines.forEach((line) => {
    list.push({ x: line.x1, y: line.y1, ownerId: line.id, type: 'line-end', meta: { end: 'start' } });
    list.push({ x: line.x2, y: line.y2, ownerId: line.id, type: 'line-end', meta: { end: 'end' } });
  });

  points.forEach((point) => {
    list.push({ x: point.x, y: point.y, ownerId: point.id, type: 'point' });
  });

  return list;
};

const buildSnapSegments = (boxes = [], lines = []) => {
  const segments = [];

  boxes.forEach((box) => {
    segments.push({ ax: box.x, ay: box.y, bx: box.x + box.width, by: box.y, ownerId: box.id, type: 'box-edge', meta: { edge: 'top' } });
    segments.push({ ax: box.x, ay: box.y + box.height, bx: box.x + box.width, by: box.y + box.height, ownerId: box.id, type: 'box-edge', meta: { edge: 'bottom' } });
    segments.push({ ax: box.x, ay: box.y, bx: box.x, by: box.y + box.height, ownerId: box.id, type: 'box-edge', meta: { edge: 'left' } });
    segments.push({ ax: box.x + box.width, ay: box.y, bx: box.x + box.width, by: box.y + box.height, ownerId: box.id, type: 'box-edge', meta: { edge: 'right' } });
  });

  lines.forEach((line) => {
    segments.push({ ax: line.x1, ay: line.y1, bx: line.x2, by: line.y2, ownerId: line.id, type: 'line' });
  });

  return segments;
};

const shouldSkipOwner = (ownerId, excludeSet) => ownerId && excludeSet && excludeSet.has(ownerId);

const snapPosition = ({
  x,
  y,
  snapPoints,
  snapSegments,
  distance = DRAW_SNAP_DISTANCE,
  excludePointOwners = EMPTY_SET,
  excludeSegmentOwners = EMPTY_SET,
  includePointTypes = null,
  includeSegmentTypes = null,
  clampFn = clamp,
}) => {
  let best = null;

  if (Array.isArray(snapPoints)) {
    snapPoints.forEach((point) => {
      if (shouldSkipOwner(point.ownerId, excludePointOwners)) {
        return;
      }
      if (includePointTypes && !includePointTypes.has(point.type)) {
        return;
      }
      const pointDistance = Math.hypot(x - point.x, y - point.y);
      if (pointDistance <= distance && (!best || pointDistance < best.distance)) {
        best = { x: point.x, y: point.y, distance: pointDistance, source: point };
      }
    });
  }

  if (Array.isArray(snapSegments)) {
    snapSegments.forEach((segment) => {
      if (shouldSkipOwner(segment.ownerId, excludeSegmentOwners)) {
        return;
      }
      if (includeSegmentTypes && !includeSegmentTypes.has(segment.type)) {
        return;
      }
      const projection = projectPointToSegment(x, y, segment.ax, segment.ay, segment.bx, segment.by);
      if (projection.distance <= distance && (!best || projection.distance < best.distance)) {
        best = { x: projection.x, y: projection.y, distance: projection.distance, source: segment };
      }
    });
  }

  if (best) {
    return { x: clampFn(best.x), y: clampFn(best.y), snapped: true, distance: best.distance, source: best.source };
  }

  return { x: clampFn(x), y: clampFn(y), snapped: false, distance: Infinity, source: null };
};

const findAnchorForPoint = (x, y, lines = [], boxes = [], threshold = LINE_SNAP_THRESHOLD) => {
  let best = null;

  lines.forEach((line, lineIndex) => {
    const projection = projectPointToSegment(x, y, line.x1, line.y1, line.x2, line.y2);
    if (projection.distance <= threshold && (!best || projection.distance < best.distance)) {
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
      if (projection.distance <= threshold && (!best || projection.distance < best.distance)) {
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

  if (best) {
    return { x: clamp(best.x), y: clamp(best.y), anchor: best.anchor };
  }

  return null;
};

const applyAxisLockToLine = (line, tolerance = AXIS_LOCK_TOLERANCE) => {
  if (!line) {
    return line;
  }

  const dx = line.x2 - line.x1;
  const dy = line.y2 - line.y1;

  if (Math.abs(dx) < Number.EPSILON && Math.abs(dy) < Number.EPSILON) {
    return line;
  }

  if (Math.abs(dy) <= Math.abs(dx) * tolerance) {
    const midY = clamp((line.y1 + line.y2) / 2);
    return { ...line, y1: midY, y2: midY };
  }

  if (Math.abs(dx) <= Math.abs(dy) * tolerance) {
    const midX = clamp((line.x1 + line.x2) / 2);
    return { ...line, x1: midX, x2: midX };
  }

  return line;
};

const snapLineEndpoints = ({
  line,
  snapPoints,
  snapSegments,
  excludeId,
  distance = DRAW_SNAP_DISTANCE,
}) => {
  if (!line) {
    return line;
  }

  const excludeOwners = excludeId ? new Set([excludeId]) : EMPTY_SET;
  const endpoints = [
    { key: 'start', x: line.x1, y: line.y1 },
    { key: 'end', x: line.x2, y: line.y2 },
  ];

  let best = null;
  endpoints.forEach((endpoint) => {
    const snap = snapPosition({
      x: endpoint.x,
      y: endpoint.y,
      snapPoints,
      snapSegments,
      distance,
      excludePointOwners: excludeOwners,
      excludeSegmentOwners: excludeOwners,
    });
    if (snap.snapped && (!best || snap.distance < best.distance)) {
      best = { endpoint: endpoint.key, snap };
    }
  });

  if (!best) {
    return line;
  }

  const { endpoint, snap } = best;
  if (endpoint === 'start') {
    const deltaX = snap.x - line.x1;
    const deltaY = snap.y - line.y1;
    return {
      ...line,
      x1: clamp(snap.x),
      y1: clamp(snap.y),
      x2: clamp(line.x2 + deltaX),
      y2: clamp(line.y2 + deltaY),
    };
  }

  const deltaX = snap.x - line.x2;
  const deltaY = snap.y - line.y2;
  return {
    ...line,
    x1: clamp(line.x1 + deltaX),
    y1: clamp(line.y1 + deltaY),
    x2: clamp(snap.x),
    y2: clamp(snap.y),
  };
};

const snapSpecificLineEndpoint = ({
  line,
  endpoint,
  snapPoints,
  snapSegments,
  excludeId,
  distance = DRAW_SNAP_DISTANCE,
}) => {
  if (!line) {
    return line;
  }

  const excludeOwners = excludeId ? new Set([excludeId]) : EMPTY_SET;
  const targetX = endpoint === 'start' ? line.x1 : line.x2;
  const targetY = endpoint === 'start' ? line.y1 : line.y2;

  const snap = snapPosition({
    x: targetX,
    y: targetY,
    snapPoints,
    snapSegments,
    distance,
    excludePointOwners: excludeOwners,
    excludeSegmentOwners: excludeOwners,
  });

  if (!snap.snapped) {
    return line;
  }

  if (endpoint === 'start') {
    return { ...line, x1: clamp(snap.x), y1: clamp(snap.y) };
  }
  return { ...line, x2: clamp(snap.x), y2: clamp(snap.y) };
};

export {
  clamp,
  LINE_SNAP_THRESHOLD,
  DRAW_SNAP_DISTANCE,
  AXIS_LOCK_ANGLE_DEG,
  AXIS_LOCK_TOLERANCE,
  EMPTY_SET,
  projectPointToSegment,
  buildSnapPoints,
  buildSnapSegments,
  snapPosition,
  findAnchorForPoint,
  applyAxisLockToLine,
  snapLineEndpoints,
  snapSpecificLineEndpoint,
};
