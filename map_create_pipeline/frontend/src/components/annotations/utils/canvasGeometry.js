const clamp = (value, min = 0, max = 1) => {
  if (Number.isNaN(value)) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
};

const LINE_SNAP_THRESHOLD = 0.002;
const DRAW_SNAP_DISTANCE = 0.002;
const SNAP_RELEASE_DISTANCE = 0.008;
const AXIS_LOCK_ANGLE_DEG = 5;
const AXIS_LOCK_TOLERANCE = Math.tan((AXIS_LOCK_ANGLE_DEG * Math.PI) / 180);
const EMPTY_SET = Object.freeze(new Set());
const POINT_MATCH_TOLERANCE = 0.0004;
const AXIS_TOLERANCE = 0.0005;
const RELATED_AXIS_TOLERANCE = Math.max(0.001, DRAW_SNAP_DISTANCE * 0.4);

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
    list.push({
      x: box.x,
      y: box.y + box.height,
      ownerId: box.id,
      type: 'box-corner',
      meta: { corner: 'bottom-left' },
    });
    list.push({
      x: box.x + box.width,
      y: box.y + box.height,
      ownerId: box.id,
      type: 'box-corner',
      meta: { corner: 'bottom-right' },
    });
  });

  lines.forEach((line) => {
    list.push({ x: line.x1, y: line.y1, ownerId: line.id, type: 'line-end', meta: { end: 'start' } });
    list.push({ x: line.x2, y: line.y2, ownerId: line.id, type: 'line-end', meta: { end: 'end' } });
  });

  points.forEach((point) => {
    if (point.labelId !== '0') {
      list.push({ x: point.x, y: point.y, ownerId: point.id, type: 'point' });
    }
  });

  return list;
};

const buildSnapSegments = (boxes = [], lines = []) => {
  const segments = [];

  boxes.forEach((box) => {
    segments.push({
      ax: box.x,
      ay: box.y,
      bx: box.x + box.width,
      by: box.y,
      ownerId: box.id,
      type: 'box-edge',
      meta: { edge: 'top' },
    });
    segments.push({
      ax: box.x,
      ay: box.y + box.height,
      bx: box.x + box.width,
      by: box.y + box.height,
      ownerId: box.id,
      type: 'box-edge',
      meta: { edge: 'bottom' },
    });
    segments.push({
      ax: box.x,
      ay: box.y,
      bx: box.x,
      by: box.y + box.height,
      ownerId: box.id,
      type: 'box-edge',
      meta: { edge: 'left' },
    });
    segments.push({
      ax: box.x + box.width,
      ay: box.y,
      bx: box.x + box.width,
      by: box.y + box.height,
      ownerId: box.id,
      type: 'box-edge',
      meta: { edge: 'right' },
    });
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
  axisPreference = null,
}) => {
  let best = null;

  const consider = (candidate) => {
    if (!candidate) {
      return;
    }
    if (candidate.distance > distance) {
      return;
    }

    let weight = 1;
    if (candidate.axis) {
      if (axisPreference && candidate.axis !== axisPreference) {
        weight = 3;
      } else if (axisPreference && candidate.axis === axisPreference) {
        weight = 0.4;
      } else {
        weight = 0.9;
      }
    }

    const effectiveDistance = candidate.distance * weight;
    const scored = { ...candidate, effectiveDistance };

    const shouldReplace =
      !best ||
      effectiveDistance < best.effectiveDistance ||
      (axisPreference &&
        candidate.axis === axisPreference &&
        best.axis !== axisPreference &&
        Math.abs(effectiveDistance - best.effectiveDistance) <= Number.EPSILON);

    if (shouldReplace) {
      best = scored;
    }
  };

  if (Array.isArray(snapPoints)) {
    snapPoints.forEach((point) => {
      if (shouldSkipOwner(point.ownerId, excludePointOwners)) {
        return;
      }
      if (includePointTypes && !includePointTypes.has(point.type)) {
        return;
      }
      const pointDistance = Math.hypot(x - point.x, y - point.y);
      consider({
        x: point.x,
        y: point.y,
        distance: pointDistance,
        source: point,
        axis: null,
      });
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

      if (segment.type === 'box-edge' && segment.meta?.edge) {
        const edge = segment.meta.edge;
        const isVertical = edge === 'left' || edge === 'right';
        if (isVertical) {
          const axisDistance = Math.abs(x - segment.ax);
          consider({
            x: segment.ax,
            y,
            distance: axisDistance,
            source: segment,
            axis: 'vertical',
          });
        } else {
          const axisDistance = Math.abs(y - segment.ay);
          consider({
            x,
            y: segment.ay,
            distance: axisDistance,
            source: segment,
            axis: 'horizontal',
          });
        }

        return;
      }

      if (segment.type === 'line') {
        consider({
          x: segment.ax,
          y: segment.ay,
          distance: Math.hypot(x - segment.ax, y - segment.ay),
          source: { ...segment, type: 'line-end', meta: { end: 'start' } },
          axis: null,
        });
        consider({
          x: segment.bx,
          y: segment.by,
          distance: Math.hypot(x - segment.bx, y - segment.by),
          source: { ...segment, type: 'line-end', meta: { end: 'end' } },
          axis: null,
        });
        return;
      }

      const projection = projectPointToSegment(x, y, segment.ax, segment.ay, segment.bx, segment.by);
      consider({
        x: projection.x,
        y: projection.y,
        distance: projection.distance,
        source: segment,
        axis: null,
      });
    });
  }

  if (best) {
    const relatedSources = [];

    if (best.axis && Array.isArray(snapSegments)) {
      const axis = best.axis;
      const axisValue = axis === 'vertical' ? best.x : best.y;

      snapSegments.forEach((segment) => {
        if (segment === best.source) {
          return;
        }
        if (segment.type !== 'box-edge' && segment.type !== 'line') {
          return;
        }
        if (shouldSkipOwner(segment.ownerId, excludeSegmentOwners)) {
          return;
        }

        if (axis === 'vertical') {
          let segX = null;
          if (segment.type === 'box-edge') {
            const edge = segment.meta?.edge;
            if (edge !== 'left' && edge !== 'right') {
              return;
            }
            segX = segment.ax;
          } else {
            if (Math.abs(segment.ax - segment.bx) > AXIS_TOLERANCE) {
              return;
            }
            segX = segment.ax;
          }
          if (Math.abs(segX - axisValue) <= RELATED_AXIS_TOLERANCE) {
            relatedSources.push(segment);
          }
        } else if (axis === 'horizontal') {
          let segY = null;
          if (segment.type === 'box-edge') {
            const edge = segment.meta?.edge;
            if (edge !== 'top' && edge !== 'bottom') {
              return;
            }
            segY = segment.ay;
          } else {
            if (Math.abs(segment.ay - segment.by) > AXIS_TOLERANCE) {
              return;
            }
            segY = segment.ay;
          }
          if (Math.abs(segY - axisValue) <= RELATED_AXIS_TOLERANCE) {
            relatedSources.push(segment);
          }
        }
      });
    }

    if (Array.isArray(snapPoints)) {
      snapPoints.forEach((point) => {
        if (point === best.source) {
          return;
        }
        if (shouldSkipOwner(point.ownerId, excludePointOwners)) {
          return;
        }
        if (
          Math.abs(point.x - best.x) <= POINT_MATCH_TOLERANCE &&
          Math.abs(point.y - best.y) <= POINT_MATCH_TOLERANCE
        ) {
          relatedSources.push(point);
        }
      });
    }
    // eslint-disable-next-line no-unused-vars
    const { effectiveDistance, ...result } = best;

    return {
      x: clampFn(result.x),
      y: clampFn(result.y),
      snapped: true,
      distance: result.distance,
      source: result.source,
      axis: result.axis ?? null,
      relatedSources: relatedSources.length > 0 ? relatedSources : null,
    };
  }

  return { x: clampFn(x), y: clampFn(y), snapped: false, distance: Infinity, source: null, axis: null };
};

const projectToClosestAnchor = (x, y, lines = [], boxes = []) => {
  let best = null;

  lines.forEach((line, lineIndex) => {
    const projection = projectPointToSegment(x, y, line.x1, line.y1, line.x2, line.y2);
    if (!best || projection.distance < best.distance) {
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
      if (!best || projection.distance < best.distance) {
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

const applyAxisLockToLine = (line, tolerance = AXIS_LOCK_TOLERANCE, axisLockHint = null) => {
  if (!line) {
    return line;
  }

  if (axisLockHint?.axis === 'horizontal' && Number.isFinite(axisLockHint.value)) {
    const lockedY = clamp(axisLockHint.value);
    return { ...line, y1: lockedY, y2: lockedY };
  }

  if (axisLockHint?.axis === 'vertical' && Number.isFinite(axisLockHint.value)) {
    const lockedX = clamp(axisLockHint.value);
    return { ...line, x1: lockedX, x2: lockedX };
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
  axisPreference = null,
}) => {
  if (!line) {
    return { line, snap: null };
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
      axisPreference,
    });
    if (snap.snapped && (!best || snap.distance < best.distance)) {
      best = { endpoint: endpoint.key, snap };
    }
  });

  if (!best) {
    return { line, snap: null };
  }

  const { endpoint, snap } = best;
  let nextLine;
  if (endpoint === 'start') {
    const deltaX = snap.x - line.x1;
    const deltaY = snap.y - line.y1;
    nextLine = {
      ...line,
      x1: clamp(snap.x),
      y1: clamp(snap.y),
      x2: clamp(line.x2 + deltaX),
      y2: clamp(line.y2 + deltaY),
    };
  } else {
    const deltaX = snap.x - line.x2;
    const deltaY = snap.y - line.y2;
    nextLine = {
      ...line,
      x1: clamp(line.x1 + deltaX),
      y1: clamp(line.y1 + deltaY),
      x2: clamp(snap.x),
      y2: clamp(snap.y),
    };
  }

  return { line: nextLine, snap: { ...snap, endpoint } };
};

const snapSpecificLineEndpoint = ({
  line,
  endpoint,
  snapPoints,
  snapSegments,
  excludeId,
  distance = DRAW_SNAP_DISTANCE,
  axisPreference = null,
}) => {
  if (!line) {
    return { line, snap: null };
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
    axisPreference,
  });

  if (!snap.snapped) {
    return { line, snap: null };
  }

  if (endpoint === 'start') {
    return { line: { ...line, x1: clamp(snap.x), y1: clamp(snap.y) }, snap: { ...snap, endpoint } };
  }
  return { line: { ...line, x2: clamp(snap.x), y2: clamp(snap.y) }, snap: { ...snap, endpoint } };
};

const findVerticalSnap = ({ value, snapPoints, snapSegments, distance, excludePointOwners, excludeSegmentOwners }) => {
  let bestDistance = distance;
  let bestValue = value;
  let snapped = false;

  let bestEffectiveDistance = distance;
  const SNAP_PRIORITY_BOOST = 0.5;

  if (Array.isArray(snapSegments)) {
    snapSegments.forEach((segment) => {
      if (shouldSkipOwner(segment.ownerId, excludeSegmentOwners)) return;
      let segX = null;
      if (segment.type === 'box-edge' && (segment.meta?.edge === 'left' || segment.meta?.edge === 'right'))
        segX = segment.ax;
      else if (segment.type === 'line' && Math.abs(segment.ax - segment.bx) < AXIS_TOLERANCE) segX = segment.ax;

      if (segX !== null) {
        const d = Math.abs(value - segX);
        const effectiveDistance = d * SNAP_PRIORITY_BOOST;

        if (d < distance && effectiveDistance < bestEffectiveDistance) {
          bestEffectiveDistance = effectiveDistance;
          bestDistance = d;
          bestValue = segX;
          snapped = true;
        }
      }
    });
  }

  if (Array.isArray(snapPoints)) {
    snapPoints.forEach((point) => {
      if (shouldSkipOwner(point.ownerId, excludePointOwners)) return;
      const d = Math.abs(value - point.x);
      const effectiveDistance = d;

      if (d < distance && effectiveDistance < bestEffectiveDistance) {
        bestEffectiveDistance = effectiveDistance;
        bestDistance = d;
        bestValue = point.x;
        snapped = true;
      }
    });
  }

  if (!snapped) {
    return { snapped: false, value, distance: Infinity, sources: [], axis: 'vertical' };
  }

  const sources = [];
  const addedSourceIds = new Set();
  const relatedTolerance = Math.max(AXIS_TOLERANCE, RELATED_AXIS_TOLERANCE, 1e-5);

  if (Array.isArray(snapPoints)) {
    snapPoints.forEach((point) => {
      if (shouldSkipOwner(point.ownerId, excludePointOwners)) return;
      if (Math.abs(bestValue - point.x) < relatedTolerance) {
        if (!addedSourceIds.has(point.ownerId)) {
          sources.push(point);
          addedSourceIds.add(point.ownerId);
        }
      }
    });
  }
  if (Array.isArray(snapSegments)) {
    snapSegments.forEach((segment) => {
      if (shouldSkipOwner(segment.ownerId, excludeSegmentOwners)) return;
      let segX = null;
      if (segment.type === 'box-edge' && (segment.meta?.edge === 'left' || segment.meta?.edge === 'right'))
        segX = segment.ax;
      else if (segment.type === 'line' && Math.abs(segment.ax - segment.bx) < AXIS_TOLERANCE) segX = segment.ax;

      if (segX !== null && Math.abs(bestValue - segX) < relatedTolerance) {
        if (!addedSourceIds.has(segment.ownerId)) {
          sources.push(segment);
          addedSourceIds.add(segment.ownerId);
        }
      }
    });
  }

  return { snapped: true, value: bestValue, distance: bestDistance, sources, axis: 'vertical' };
};

const findHorizontalSnap = ({
  value,
  snapPoints,
  snapSegments,
  distance,
  excludePointOwners,
  excludeSegmentOwners,
}) => {
  let bestDistance = distance;
  let bestValue = value;
  let snapped = false;

  let bestEffectiveDistance = distance;
  const SNAP_PRIORITY_BOOST = 0.5;

  if (Array.isArray(snapSegments)) {
    snapSegments.forEach((segment) => {
      if (shouldSkipOwner(segment.ownerId, excludeSegmentOwners)) return;
      let segY = null;
      if (segment.type === 'box-edge' && (segment.meta?.edge === 'top' || segment.meta?.edge === 'bottom'))
        segY = segment.ay;
      else if (segment.type === 'line' && Math.abs(segment.ay - segment.by) < AXIS_TOLERANCE) segY = segment.ay;

      if (segY !== null) {
        const d = Math.abs(value - segY);
        const effectiveDistance = d * SNAP_PRIORITY_BOOST;

        if (d < distance && effectiveDistance < bestEffectiveDistance) {
          bestEffectiveDistance = effectiveDistance;
          bestDistance = d;
          bestValue = segY;
          snapped = true;
        }
      }
    });
  }

  if (Array.isArray(snapPoints)) {
    snapPoints.forEach((point) => {
      if (shouldSkipOwner(point.ownerId, excludePointOwners)) return;
      const d = Math.abs(value - point.y);
      const effectiveDistance = d;

      if (d < distance && effectiveDistance < bestEffectiveDistance) {
        bestEffectiveDistance = effectiveDistance;
        bestDistance = d;
        bestValue = point.y;
        snapped = true;
      }
    });
  }

  if (!snapped) {
    return { snapped: false, value, distance: Infinity, sources: [], axis: 'horizontal' };
  }

  const sources = [];
  const addedSourceIds = new Set();
  const relatedTolerance = Math.max(AXIS_TOLERANCE, RELATED_AXIS_TOLERANCE, 1e-5);

  if (Array.isArray(snapPoints)) {
    snapPoints.forEach((point) => {
      if (shouldSkipOwner(point.ownerId, excludePointOwners)) return;
      if (Math.abs(bestValue - point.y) < relatedTolerance) {
        if (!addedSourceIds.has(point.ownerId)) {
          sources.push(point);
          addedSourceIds.add(point.ownerId);
        }
      }
    });
  }
  if (Array.isArray(snapSegments)) {
    snapSegments.forEach((segment) => {
      if (shouldSkipOwner(segment.ownerId, excludeSegmentOwners)) return;
      let segY = null;
      if (segment.type === 'box-edge' && (segment.meta?.edge === 'top' || segment.meta?.edge === 'bottom'))
        segY = segment.ay;
      else if (segment.type === 'line' && Math.abs(segment.ay - segment.by) < AXIS_TOLERANCE) segY = segment.ay;

      if (segY !== null && Math.abs(bestValue - segY) < relatedTolerance) {
        if (!addedSourceIds.has(segment.ownerId)) {
          sources.push(segment);
          addedSourceIds.add(segment.ownerId);
        }
      }
    });
  }

  return { snapped: true, value: bestValue, distance: bestDistance, sources, axis: 'horizontal' };
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
  projectToClosestAnchor,
  findAnchorForPoint,
  applyAxisLockToLine,
  snapLineEndpoints,
  snapSpecificLineEndpoint,
  SNAP_RELEASE_DISTANCE,
  findVerticalSnap,
  findHorizontalSnap,
};
