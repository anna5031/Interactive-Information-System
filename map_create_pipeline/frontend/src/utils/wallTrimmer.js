const EPSILON = 1e-6;
const CUT_MARGIN = 0.004; // extra slack so near-overlaps still trim the wall
const MIN_SEGMENT_LENGTH = 0.01; // discard wall leftovers below 1% of image span
let splitCounter = 0;

const isHorizontalLine = (line) =>
  typeof line?.y1 === 'number' && typeof line?.y2 === 'number' && Math.abs(line.y1 - line.y2) < EPSILON;
const isVerticalLine = (line) =>
  typeof line?.x1 === 'number' && typeof line?.x2 === 'number' && Math.abs(line.x1 - line.x2) < EPSILON;

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

const clampIntervalToRange = (start, end, rangeStart, rangeEnd) => {
  const clampedStart = Math.max(start, rangeStart);
  const clampedEnd = Math.min(end, rangeEnd);
  if (clampedEnd - clampedStart <= EPSILON) {
    return null;
  }
  return [clampedStart, clampedEnd];
};

const subtractIntervals = (rangeStart, rangeEnd, intervals) => {
  if (!intervals || intervals.length === 0) {
    return [[rangeStart, rangeEnd]];
  }

  const filtered = intervals
    .map(([start, end]) => clampIntervalToRange(start, end, rangeStart, rangeEnd))
    .filter(Boolean)
    .sort((a, b) => a[0] - b[0]);

  const segments = [];
  let cursor = rangeStart;

  filtered.forEach(([start, end]) => {
    if (start > cursor + EPSILON) {
      segments.push([cursor, Math.min(start, rangeEnd)]);
    }
    cursor = Math.max(cursor, end);
  });

  if (cursor < rangeEnd - EPSILON) {
    segments.push([cursor, rangeEnd]);
  }

  return segments;
};

const collectHorizontalCuts = (line, boxes) => {
  const { x1, x2, y1 } = line;
  const constantY = y1;
  const start = Math.min(x1, x2);
  const end = Math.max(x1, x2);

  return boxes
    .map((box) => {
      if (!isValidBox(box)) {
        return null;
      }
      const boxTop = box.y - CUT_MARGIN;
      const boxBottom = box.y + box.height + CUT_MARGIN;
      if (constantY < boxTop - EPSILON || constantY > boxBottom + EPSILON) {
        return null;
      }
      const boxStart = box.x;
      const boxEnd = box.x + box.width;
      return clampIntervalToRange(boxStart, boxEnd, start, end);
    })
    .filter(Boolean);
};

const collectVerticalCuts = (line, boxes) => {
  const { y1, y2, x1 } = line;
  const constantX = x1;
  const start = Math.min(y1, y2);
  const end = Math.max(y1, y2);

  return boxes
    .map((box) => {
      if (!isValidBox(box)) {
        return null;
      }
      const boxLeft = box.x - CUT_MARGIN;
      const boxRight = box.x + box.width + CUT_MARGIN;
      if (constantX < boxLeft - EPSILON || constantX > boxRight + EPSILON) {
        return null;
      }
      const boxStart = box.y;
      const boxEnd = box.y + box.height;
      return clampIntervalToRange(boxStart, boxEnd, start, end);
    })
    .filter(Boolean);
};

const splitLineByBoxes = (line, boxes) => {
  if (!boxes || boxes.length === 0) {
    return [line];
  }

  if (!isHorizontalLine(line) && !isVerticalLine(line)) {
    return [line];
  }

  const { id, x1, y1, x2, y2, ...rest } = line;
  const preserved = { ...rest, type: line.type ?? 'line', labelId: line.labelId ?? '4' };

  const segments = [];

  if (isHorizontalLine(line)) {
    const cuts = collectHorizontalCuts(line, boxes);
    if (cuts.length === 0) {
      return [line];
    }
    const rangeStart = Math.min(x1, x2);
    const rangeEnd = Math.max(x1, x2);
    const available = subtractIntervals(rangeStart, rangeEnd, cuts);
    const isPositive = x2 >= x1;

    available.forEach(([start, end]) => {
      const span = end - start;
      if (span <= Math.max(EPSILON, MIN_SEGMENT_LENGTH)) {
        return;
      }
      const segment = {
        ...preserved,
        id: generateSegmentId(id),
        x1: isPositive ? start : end,
        y1,
        x2: isPositive ? end : start,
        y2,
      };
      segments.push(segment);
    });
  } else if (isVerticalLine(line)) {
    const cuts = collectVerticalCuts(line, boxes);
    if (cuts.length === 0) {
      return [line];
    }
    const rangeStart = Math.min(y1, y2);
    const rangeEnd = Math.max(y1, y2);
    const available = subtractIntervals(rangeStart, rangeEnd, cuts);
    const isPositive = y2 >= y1;

    available.forEach(([start, end]) => {
      const span = end - start;
      if (span <= Math.max(EPSILON, MIN_SEGMENT_LENGTH)) {
        return;
      }
      const segment = {
        ...preserved,
        id: generateSegmentId(id),
        x1,
        y1: isPositive ? start : end,
        x2,
        y2: isPositive ? end : start,
      };
      segments.push(segment);
    });
  }

  return segments.length > 0 ? segments : [];
};

export const subtractBoxesFromLines = (lines, boxes) => {
  if (!Array.isArray(lines)) {
    return [];
  }
  if (!Array.isArray(boxes) || boxes.length === 0) {
    return lines;
  }

  return lines.flatMap((line) => {
    if (line?.type !== 'line' || line?.labelId !== '4') {
      return [line];
    }
    return splitLineByBoxes(line, boxes);
  });
};

export const subtractBoxFromLines = (lines, box) => {
  if (!box) {
    return lines;
  }
  return subtractBoxesFromLines(lines, [box]);
};

export default subtractBoxesFromLines;
