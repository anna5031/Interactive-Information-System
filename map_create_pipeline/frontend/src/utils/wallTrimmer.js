const EPSILON = 1e-6;
const DEFAULT_CUT_MARGIN = 0; // default slack applied when trimming walls against boxes
const DEFAULT_EDGE_PROXIMITY_MARGIN = 0; // removes wall if it sits within this margin outside of the box
const MIN_SEGMENT_LENGTH = 0.01; // discard wall leftovers below 1% of image span
let splitCounter = 0;

const isValidBox = (box) =>
  box &&
  typeof box.x === 'number' &&
  typeof box.y === 'number' &&
  typeof box.width === 'number' &&
  typeof box.height === 'number';

const generateSegmentId = (baseId) => {
  splitCounter += 1;
  return `${baseId || 'line'}-split-${splitCounter}`;
};

const toBoxBounds = (box, cutMargin = DEFAULT_CUT_MARGIN) => {
  const x1 = Number(box.x) || 0;
  const y1 = Number(box.y) || 0;
  const width = Number(box.width) || 0;
  const height = Number(box.height) || 0;
  const minX = Math.min(x1, x1 + width) - cutMargin;
  const maxX = Math.max(x1, x1 + width) + cutMargin;
  const minY = Math.min(y1, y1 + height) - cutMargin;
  const maxY = Math.max(y1, y1 + height) + cutMargin;
  return { minX, maxX, minY, maxY };
};

const pointOnSegment = (segment, t) => ({
  x: segment.x1 + (segment.x2 - segment.x1) * t,
  y: segment.y1 + (segment.y2 - segment.y1) * t,
});

const segmentLength = (segment) => Math.hypot(segment.x2 - segment.x1, segment.y2 - segment.y1);

const cloneSegment = (source, startT, endT, options = undefined) => {
  const deleteTinySegments = options?.deleteTinySegments ?? true;
  const minSegmentThreshold = Number.isFinite(options?.minSegmentLength)
    ? Math.max(EPSILON, options.minSegmentLength)
    : Math.max(EPSILON, MIN_SEGMENT_LENGTH);
  const start = pointOnSegment(source, startT);
  const end = pointOnSegment(source, endT);
  const span = Math.hypot(end.x - start.x, end.y - start.y);
  if (span <= EPSILON) {
    return null;
  }
  if (deleteTinySegments && span <= minSegmentThreshold) {
    return null;
  }
  return {
    ...source,
    id: generateSegmentId(source.id),
    x1: start.x,
    y1: start.y,
    x2: end.x,
    y2: end.y,
  };
};

const intervalGap = (aMin, aMax, bMin, bMax) => {
  if (aMax < bMin) {
    return bMin - aMax;
  }
  if (bMax < aMin) {
    return aMin - bMax;
  }
  return 0;
};

const getSegmentExtents = (segment) => {
  const minX = Math.min(segment.x1, segment.x2);
  const maxX = Math.max(segment.x1, segment.x2);
  const minY = Math.min(segment.y1, segment.y2);
  const maxY = Math.max(segment.y1, segment.y2);
  return { minX, maxX, minY, maxY };
};

const expandBounds = (bounds, margin) => {
  if (!Number.isFinite(margin) || margin <= 0) {
    return bounds;
  }
  return {
    minX: bounds.minX - margin,
    maxX: bounds.maxX + margin,
    minY: bounds.minY - margin,
    maxY: bounds.maxY + margin,
  };
};

const needsEdgeProximityTrim = (segment, bounds, tolerance) => {
  if (!Number.isFinite(tolerance) || tolerance <= 0) {
    return false;
  }

  const { minX, maxX, minY, maxY } = getSegmentExtents(segment);
  const dx = intervalGap(minX, maxX, bounds.minX, bounds.maxX);
  const dy = intervalGap(minY, maxY, bounds.minY, bounds.maxY);

  if (dx === 0 && dy === 0) {
    return false;
  }

  const overlapsX = dx === 0;
  const overlapsY = dy === 0;

  if (overlapsY && dx > 0 && dx <= tolerance) {
    return true;
  }

  if (overlapsX && dy > 0 && dy <= tolerance) {
    return true;
  }

  return false;
};

const clipSegmentOutsideBox = (segment, bounds, options) => {
  if (!segment || !bounds) {
    return segment ? [segment] : [];
  }

  const { minX, maxX, minY, maxY } = bounds;
  if (
    Number.isNaN(minX) ||
    Number.isNaN(maxX) ||
    Number.isNaN(minY) ||
    Number.isNaN(maxY) ||
    minX >= maxX ||
    minY >= maxY
  ) {
    return [segment];
  }

  const dx = segment.x2 - segment.x1;
  const dy = segment.y2 - segment.y1;
  if (!Number.isFinite(dx) || !Number.isFinite(dy) || segmentLength(segment) <= EPSILON) {
    return [segment];
  }

  let tEnter = 0;
  let tExit = 1;
  const constraints = [
    [-dx, segment.x1 - minX],
    [dx, maxX - segment.x1],
    [-dy, segment.y1 - minY],
    [dy, maxY - segment.y1],
  ];

  const pEpsilon = 1e-9;

  for (const [p, q] of constraints) {
    if (Math.abs(p) < pEpsilon) {
      if (q < 0) {
        return [segment];
      }
      continue;
    }
    const r = q / p;
    if (p < 0) {
      if (r > tExit) {
        return [segment];
      }
      if (r > tEnter) {
        tEnter = r;
      }
    } else if (p > 0) {
      if (r < tEnter) {
        return [segment];
      }
      if (r < tExit) {
        tExit = r;
      }
    }
  }

  if (tEnter >= tExit) {
    return [segment];
  }

  const insideStart = Math.max(0, Math.min(1, tEnter));
  const insideEnd = Math.max(0, Math.min(1, tExit));

  if (insideEnd - insideStart <= EPSILON) {
    return [segment];
  }

  if (insideStart <= 0 && insideEnd >= 1) {
    return [];
  }

  const result = [];

  if (insideStart > 0) {
    const first = cloneSegment(segment, 0, insideStart, options);
    if (first) {
      result.push(first);
    }
  }

  if (insideEnd < 1) {
    const second = cloneSegment(segment, insideEnd, 1, options);
    if (second) {
      result.push(second);
    }
  }

  return result.length > 0 ? result : [];
};

const trimLineAgainstBoxes = (line, boxBounds, options) => {
  if (!boxBounds || boxBounds.length === 0) {
    return [line];
  }

  if (
    !line ||
    !Number.isFinite(line.x1) ||
    !Number.isFinite(line.y1) ||
    !Number.isFinite(line.x2) ||
    !Number.isFinite(line.y2)
  ) {
    return [line];
  }

  const resolvedBoxes = boxBounds
    .map((entry) => {
      if (!entry) {
        return null;
      }
      if (entry.bounds) {
        return entry;
      }
      return { bounds: entry, edgeProximityMargin: resolveEdgeProximityMargin(options) };
    })
    .filter((entry) => entry?.bounds);

  if (resolvedBoxes.length === 0) {
    return [line];
  }

  let segments = [{ ...line, type: line.type ?? 'line', labelId: line.labelId ?? '4' }];

  for (const { bounds, edgeProximityMargin } of resolvedBoxes) {
    const proximityMargin = Number.isFinite(edgeProximityMargin)
      ? Math.max(0, edgeProximityMargin)
      : resolveEdgeProximityMargin(options);
    const expandedBounds =
      proximityMargin > 0 ? expandBounds(bounds, proximityMargin) : bounds;
    const nextSegments = [];
    segments.forEach((segment) => {
      const targetBounds =
        proximityMargin > 0 && needsEdgeProximityTrim(segment, bounds, proximityMargin)
          ? expandedBounds
          : bounds;
      const clipped = clipSegmentOutsideBox(segment, targetBounds, options);
      if (clipped.length === 0) {
        return;
      }
      clipped.forEach((item) => {
        nextSegments.push(item);
      });
    });
    segments = nextSegments;
    if (segments.length === 0) {
      break;
    }
  }

  return segments;
};

const resolveCutMargin = (options) => {
  const candidate = Number(options?.cutMargin);
  if (!Number.isFinite(candidate) || candidate < 0) {
    return DEFAULT_CUT_MARGIN;
  }
  return candidate;
};

const resolveEdgeProximityMargin = (options, fallback = DEFAULT_EDGE_PROXIMITY_MARGIN) => {
  const candidate = Number(options?.edgeProximityMargin);
  if (Number.isFinite(candidate) && candidate >= 0) {
    return candidate;
  }
  if (Number.isFinite(fallback) && fallback >= 0) {
    return fallback;
  }
  return DEFAULT_EDGE_PROXIMITY_MARGIN;
};

export const subtractBoxesFromLines = (lines, boxes, options) => {
  if (!Array.isArray(lines)) {
    return [];
  }
  if (!Array.isArray(boxes) || boxes.length === 0) {
    return lines;
  }

  const cutMargin = resolveCutMargin(options);
  const edgeProximityMargin = resolveEdgeProximityMargin(options, cutMargin);
  const trimOptions = {
    ...options,
    cutMargin,
    edgeProximityMargin,
  };

  const usableBoxes = boxes
    .filter(isValidBox)
    .map((box) => ({
      bounds: toBoxBounds(box, cutMargin),
      edgeProximityMargin,
    }));

  if (usableBoxes.length === 0) {
    return lines;
  }

  return lines.flatMap((line) => {
    if (line?.type !== 'line' || line?.labelId !== '4') {
      return [line];
    }
    return trimLineAgainstBoxes(line, usableBoxes, trimOptions);
  });
};

export const subtractBoxFromLines = (lines, box, options) => {
  if (!box) {
    return lines;
  }
  return subtractBoxesFromLines(lines, [box], options);
};

export default subtractBoxesFromLines;
