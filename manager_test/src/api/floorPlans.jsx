import yoloTxtPath from '../dummy/yolo.txt';
import wallTxtPath from '../dummy/wall.txt';
import { isBoxLabel, isPointLabel } from '../config/annotationConfig';
import { subtractBoxesFromLines } from '../utils/wallTrimmer';

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const readFileAsDataUrl = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });

const normaliseValue = (value) => {
  if (Number.isNaN(value)) {
    return 0;
  }
  return Math.min(Math.max(value, 0), 1);
};

const DOOR_LABEL_ID = '0';
const SMALL_NUMBER = 1e-9;

const clampToRange = (value, start, end) => {
  const min = Math.min(start, end);
  const max = Math.max(start, end);
  if (!Number.isFinite(value)) {
    return min;
  }
  if (value <= min) {
    return min;
  }
  if (value >= max) {
    return max;
  }
  return value;
};

export const parseYoloBoxes = (text) => {
  return text
    .split('\n')
    .map((line, index) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return null;
      }
      const parts = trimmed.split(/\s+/);
      if (parts.length < 5) {
        return null;
      }

      const [labelId, centerX, centerY, width, height] = parts;
      if (!isBoxLabel(labelId)) {
        return null;
      }

      const parsedWidth = Number.parseFloat(width);
      const parsedHeight = Number.parseFloat(height);
      const parsedCenterX = Number.parseFloat(centerX);
      const parsedCenterY = Number.parseFloat(centerY);

      const normalisedWidth = normaliseValue(parsedWidth);
      const normalisedHeight = normaliseValue(parsedHeight);
      const normalisedX = normaliseValue(parsedCenterX - normalisedWidth / 2);
      const normalisedY = normaliseValue(parsedCenterY - normalisedHeight / 2);

      return {
        id: `${labelId}-box-${index}`,
        type: 'box',
        labelId,
        x: normalisedX,
        y: normalisedY,
        width: normalisedWidth,
        height: normalisedHeight,
      };
    })
    .filter(Boolean);
};

export const parseWallLines = (text) => {
  return text
    .split('\n')
    .map((line, index) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return null;
      }
      const parts = trimmed.split(/\s+/);
      if (parts.length < 4) {
        return null;
      }

      const [x1, y1, x2, y2] = parts.map((value) => normaliseValue(Number.parseFloat(value)));

      return {
        id: `wall-${index}`,
        type: 'line',
        labelId: '4',
        x1,
        y1,
        x2,
        y2,
      };
    })
    .filter(Boolean);
};

const parseDoorCandidatesFromYolo = (text) => {
  return text
    .split('\n')
    .map((line, index) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return null;
      }

      const parts = trimmed.split(/\s+/);
      if (parts.length < 5) {
        return null;
      }

      const [labelId, centerX, centerY, width, height] = parts;
      if (labelId !== DOOR_LABEL_ID) {
        return null;
      }

      const parsedWidth = Number.parseFloat(width);
      const parsedHeight = Number.parseFloat(height);
      const parsedCenterX = Number.parseFloat(centerX);
      const parsedCenterY = Number.parseFloat(centerY);

      if (
        !Number.isFinite(parsedWidth) ||
        !Number.isFinite(parsedHeight) ||
        !Number.isFinite(parsedCenterX) ||
        !Number.isFinite(parsedCenterY) ||
        parsedWidth <= 0 ||
        parsedHeight <= 0
      ) {
        return null;
      }

      const normalisedWidth = normaliseValue(parsedWidth);
      const normalisedHeight = normaliseValue(parsedHeight);
      if (normalisedWidth <= 0 || normalisedHeight <= 0) {
        return null;
      }

      const normalisedCenterX = normaliseValue(parsedCenterX);
      const normalisedCenterY = normaliseValue(parsedCenterY);
      const x = normaliseValue(normalisedCenterX - normalisedWidth / 2);
      const y = normaliseValue(normalisedCenterY - normalisedHeight / 2);

      return {
        id: `door-candidate-${index}`,
        x,
        y,
        width: normalisedWidth,
        height: normalisedHeight,
        centerX: normaliseValue(x + normalisedWidth / 2),
        centerY: normaliseValue(y + normalisedHeight / 2),
      };
    })
    .filter(Boolean);
};

const clampFraction = (value) => {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
};

const projectPointToSegment = (px, py, ax, ay, bx, by) => {
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= SMALL_NUMBER) {
    return { x: ax, y: ay, t: 0 };
  }
  const rawT = ((px - ax) * dx + (py - ay) * dy) / lengthSquared;
  const t = clampFraction(rawT);
  return {
    x: ax + dx * t,
    y: ay + dy * t,
    t,
  };
};

const closestPointsBetweenSegments = (ax, ay, bx, by, cx, cy, dx, dy) => {
  const ux = bx - ax;
  const uy = by - ay;
  const vx = dx - cx;
  const vy = dy - cy;
  const wx = ax - cx;
  const wy = ay - cy;

  const a = ux * ux + uy * uy;
  const b = ux * vx + uy * vy;
  const c = vx * vx + vy * vy;
  const d = ux * wx + uy * wy;
  const e = vx * wx + vy * wy;

  const denom = a * c - b * b;

  const lengthUSquared = a;
  const lengthVSquared = c;

  if (lengthUSquared <= SMALL_NUMBER && lengthVSquared <= SMALL_NUMBER) {
    const distance = Math.hypot(ax - cx, ay - cy);
    return {
      distance,
      t1: 0,
      t2: 0,
      point1: { x: ax, y: ay },
      point2: { x: cx, y: cy },
    };
  }

  if (lengthUSquared <= SMALL_NUMBER) {
    const projection = projectPointToSegment(ax, ay, cx, cy, dx, dy);
    const distance = Math.hypot(ax - projection.x, ay - projection.y);
    return {
      distance,
      t1: 0,
      t2: projection.t,
      point1: { x: ax, y: ay },
      point2: { x: projection.x, y: projection.y },
    };
  }

  if (lengthVSquared <= SMALL_NUMBER) {
    const projection = projectPointToSegment(cx, cy, ax, ay, bx, by);
    const distance = Math.hypot(projection.x - cx, projection.y - cy);
    return {
      distance,
      t1: projection.t,
      t2: 0,
      point1: { x: projection.x, y: projection.y },
      point2: { x: cx, y: cy },
    };
  }

  let sNumerator = denom;
  let tNumerator = denom;
  let sDenominator = denom;
  let tDenominator = denom;

  if (Math.abs(denom) <= SMALL_NUMBER) {
    sNumerator = 0;
    sDenominator = 1;
    tNumerator = e;
    tDenominator = c;
  } else {
    sNumerator = b * e - c * d;
    tNumerator = a * e - b * d;

    if (sNumerator < 0) {
      sNumerator = 0;
      tNumerator = e;
      tDenominator = c;
    } else if (sNumerator > sDenominator) {
      sNumerator = sDenominator;
      tNumerator = e + b;
      tDenominator = c;
    }
  }

  if (tNumerator < 0) {
    tNumerator = 0;
    if (-d < 0) {
      sNumerator = 0;
      sDenominator = 1;
    } else if (-d > a) {
      sNumerator = sDenominator;
    } else {
      sNumerator = -d;
      sDenominator = a;
    }
  } else if (tNumerator > tDenominator) {
    tNumerator = tDenominator;
    if (-d + b < 0) {
      sNumerator = 0;
      sDenominator = 1;
    } else if (-d + b > a) {
      sNumerator = sDenominator;
    } else {
      sNumerator = -d + b;
      sDenominator = a;
    }
  }

  const s = sDenominator === 0 ? 0 : sNumerator / sDenominator;
  const t = tDenominator === 0 ? 0 : tNumerator / tDenominator;

  const clampedS = clampFraction(s);
  const clampedT = clampFraction(t);

  const closestPoint1 = {
    x: ax + ux * clampedS,
    y: ay + uy * clampedS,
  };
  const closestPoint2 = {
    x: cx + vx * clampedT,
    y: cy + vy * clampedT,
  };

  return {
    distance: Math.hypot(closestPoint1.x - closestPoint2.x, closestPoint1.y - closestPoint2.y),
    t1: clampedS,
    t2: clampedT,
    point1: closestPoint1,
    point2: closestPoint2,
  };
};

const buildAnchorSegments = (boxes = [], lines = []) => {
  const segments = [];

  lines.forEach((line, index) => {
    if (!line) {
      return;
    }
    segments.push({
      type: 'line',
      id: line.id,
      index,
      ax: normaliseValue(line.x1),
      ay: normaliseValue(line.y1),
      bx: normaliseValue(line.x2),
      by: normaliseValue(line.y2),
    });
  });

  boxes.forEach((box, index) => {
    if (!box) {
      return;
    }
    const minX = normaliseValue(box.x);
    const minY = normaliseValue(box.y);
    const maxX = normaliseValue(box.x + box.width);
    const maxY = normaliseValue(box.y + box.height);

    segments.push({
      type: 'box',
      id: box.id,
      index,
      edge: 'top',
      ax: minX,
      ay: minY,
      bx: maxX,
      by: minY,
    });
    segments.push({
      type: 'box',
      id: box.id,
      index,
      edge: 'bottom',
      ax: minX,
      ay: maxY,
      bx: maxX,
      by: maxY,
    });
    segments.push({
      type: 'box',
      id: box.id,
      index,
      edge: 'left',
      ax: minX,
      ay: minY,
      bx: minX,
      by: maxY,
    });
    segments.push({
      type: 'box',
      id: box.id,
      index,
      edge: 'right',
      ax: maxX,
      ay: minY,
      bx: maxX,
      by: maxY,
    });
  });

  return segments;
};

const findClosestAnchorForDoor = (doorRect, anchorSegments) => {
  if (!doorRect || !Array.isArray(anchorSegments) || anchorSegments.length === 0) {
    return null;
  }

  const rectEdges = [
    { ax: doorRect.minX, ay: doorRect.minY, bx: doorRect.maxX, by: doorRect.minY },
    { ax: doorRect.maxX, ay: doorRect.minY, bx: doorRect.maxX, by: doorRect.maxY },
    { ax: doorRect.minX, ay: doorRect.maxY, bx: doorRect.maxX, by: doorRect.maxY },
    { ax: doorRect.minX, ay: doorRect.minY, bx: doorRect.minX, by: doorRect.maxY },
  ];

  let best = null;

  anchorSegments.forEach((segment) => {
    rectEdges.forEach((edge) => {
      const result = closestPointsBetweenSegments(segment.ax, segment.ay, segment.bx, segment.by, edge.ax, edge.ay, edge.bx, edge.by);
      if (!Number.isFinite(result.distance)) {
        return;
      }
      if (!best || result.distance < best.distance - SMALL_NUMBER) {
        best = {
          distance: result.distance,
          x: result.point1.x,
          y: result.point1.y,
          t: clampFraction(result.t1),
          segment,
        };
      }
    });
  });

  return best;
};

const createDoorPointsFromCandidates = (candidates, boxes, lines) => {
  if (!Array.isArray(candidates) || candidates.length === 0) {
    return [];
  }

  const anchorSegments = buildAnchorSegments(boxes, lines);

  return candidates.map((candidate, index) => {
    const minX = normaliseValue(candidate.x);
    const minY = normaliseValue(candidate.y);
    const maxX = normaliseValue(candidate.x + candidate.width);
    const maxY = normaliseValue(candidate.y + candidate.height);

    const anchorMatch = findClosestAnchorForDoor(
      {
        minX,
        minY,
        maxX,
        maxY,
      },
      anchorSegments
    );

    let x = normaliseValue(candidate.centerX);
    let y = normaliseValue(candidate.centerY);
    let anchor;

    if (anchorMatch) {
      const { segment } = anchorMatch;
      const isHorizontal = Math.abs(segment.ay - segment.by) <= SMALL_NUMBER;
      const isVertical = Math.abs(segment.ax - segment.bx) <= SMALL_NUMBER;

      if (isHorizontal) {
        const segmentMinX = Math.min(segment.ax, segment.bx);
        const segmentMaxX = Math.max(segment.ax, segment.bx);
        x = clampToRange(x, segmentMinX, segmentMaxX);
        y = normaliseValue(segment.ay);
        const span = segment.bx - segment.ax;
        const resolvedT = span === 0 ? 0 : clampFraction((x - segment.ax) / span);
        if (segment.type === 'line') {
          anchor = {
            type: 'line',
            id: segment.id,
            index: segment.index,
            t: resolvedT,
          };
        } else if (segment.type === 'box') {
          anchor = {
            type: 'box',
            id: segment.id,
            index: segment.index,
            edge: segment.edge,
            t: resolvedT,
          };
        }
      } else if (isVertical) {
        const segmentMinY = Math.min(segment.ay, segment.by);
        const segmentMaxY = Math.max(segment.ay, segment.by);
        y = clampToRange(y, segmentMinY, segmentMaxY);
        x = normaliseValue(segment.ax);
        const span = segment.by - segment.ay;
        const resolvedT = span === 0 ? 0 : clampFraction((y - segment.ay) / span);
        if (segment.type === 'line') {
          anchor = {
            type: 'line',
            id: segment.id,
            index: segment.index,
            t: resolvedT,
          };
        } else if (segment.type === 'box') {
          anchor = {
            type: 'box',
            id: segment.id,
            index: segment.index,
            edge: segment.edge,
            t: resolvedT,
          };
        }
      } else {
        x = normaliseValue(anchorMatch.x);
        y = normaliseValue(anchorMatch.y);
        if (segment.type === 'line') {
          anchor = {
            type: 'line',
            id: segment.id,
            index: segment.index,
            t: anchorMatch.t,
          };
        } else if (segment.type === 'box') {
          anchor = {
            type: 'box',
            id: segment.id,
            index: segment.index,
            edge: segment.edge,
            t: anchorMatch.t,
          };
        }
      }
    }

    const point = {
      id: `door-${index}`,
      type: 'point',
      labelId: DOOR_LABEL_ID,
      x,
      y,
    };

    if (anchor) {
      point.anchor = anchor;
    }

    return point;
  });
};

const serialiseBoxesToYoloText = (boxes) => {
  if (!Array.isArray(boxes)) {
    return '';
  }

  return boxes
    .filter((annotation) => annotation.type === 'box')
    .map((annotation, index) => {
      const { labelId, x, y, width, height, id } = annotation;
      const centerX = normaliseValue(x + width / 2);
      const centerY = normaliseValue(y + height / 2);
      const normalisedWidth = normaliseValue(width);
      const normalisedHeight = normaliseValue(height);

      const identifier = id ?? `${labelId}-box-${index}`;

      return `${labelId} ${centerX} ${centerY} ${normalisedWidth} ${normalisedHeight} ${identifier}`.trim();
    })
    .join('\n');
};

const serialiseLinesToWallText = (lines) => {
  if (!Array.isArray(lines)) {
    return '';
  }

  return lines
    .filter((annotation) => annotation.type === 'line')
    .map((annotation, index) => {
      const { x1, y1, x2, y2, id } = annotation;
      const identifier = id ?? `wall-${index}`;
      return `${normaliseValue(x1)} ${normaliseValue(y1)} ${normaliseValue(x2)} ${normaliseValue(y2)} ${identifier}`.trim();
    })
    .join('\n');
};

const parseDoorPoints = (text, boxes, lines) => {
  return text
    .split('\n')
    .map((line, index) => {
      const trimmed = line.trim();
      if (!trimmed) {
        return null;
      }
      const parts = trimmed.split(/\s+/);
      if (parts.length < 2) {
        return null;
      }

      let cursor = 0;
      let labelId = parts[cursor];
      if (isPointLabel(labelId)) {
        cursor += 1;
      } else {
        labelId = DOOR_LABEL_ID;
      }

      const xValue = parts[cursor];
      const yValue = parts[cursor + 1];
      const rest = parts.slice(cursor + 2);

      if (!Number.isFinite(Number.parseFloat(xValue)) || !Number.isFinite(Number.parseFloat(yValue))) {
        return null;
      }

      const x = normaliseValue(Number.parseFloat(xValue));
      const y = normaliseValue(Number.parseFloat(yValue));

      const point = {
        id: `door-${index}`,
        type: 'point',
        labelId,
        x,
        y,
      };

      if (rest.length > 0) {
        const [anchorType, ...anchorRest] = rest;
        if (anchorType === 'line' && anchorRest.length >= 2) {
          let lineIndex = -1;
          let lineId;
          let tValue;

          if (anchorRest.length >= 3) {
            const [indexToken, idToken, tToken] = anchorRest;
            lineIndex = Number.parseInt(indexToken, 10);
            lineId = idToken;
            tValue = Number.parseFloat(tToken);
          } else {
            const [idToken, tToken] = anchorRest;
            lineId = idToken;
            tValue = Number.parseFloat(tToken);
          }

          if (Number.isFinite(tValue)) {
            const normalisedIndex = Number.isInteger(lineIndex) ? lineIndex : -1;
            const lineRef =
              normalisedIndex >= 0 && normalisedIndex < lines.length
                ? lines[normalisedIndex]
                : lines.find((candidate) => candidate.id === lineId);
            const resolvedId = lineRef?.id || lineId;
            if (resolvedId) {
              point.anchor = {
                type: 'line',
                id: resolvedId,
                index: lineRef ? normalisedIndex : undefined,
                t: normaliseValue(tValue),
              };
            }
          }
        } else if (anchorType === 'box' && anchorRest.length >= 3) {
          let boxIndex = -1;
          let boxId;
          let edge;
          let tValue;

          if (anchorRest.length >= 4) {
            const [indexToken, idToken, edgeToken, tToken] = anchorRest;
            boxIndex = Number.parseInt(indexToken, 10);
            boxId = idToken;
            edge = edgeToken;
            tValue = Number.parseFloat(tToken);
          } else {
            const [idToken, edgeToken, tToken] = anchorRest;
            boxId = idToken;
            edge = edgeToken;
            tValue = Number.parseFloat(tToken);
          }

          if (Number.isFinite(tValue) && edge) {
            const normalisedIndex = Number.isInteger(boxIndex) ? boxIndex : -1;
            const boxRef =
              normalisedIndex >= 0 && normalisedIndex < boxes.length
                ? boxes[normalisedIndex]
                : boxes.find((candidate) => candidate.id === boxId);
            const resolvedId = boxRef?.id || boxId;
            if (resolvedId) {
              point.anchor = {
                type: 'box',
                id: resolvedId,
                index: boxRef ? normalisedIndex : undefined,
                edge,
                t: normaliseValue(tValue),
              };
            }
          }
        }
      }

      return point;
    })
    .filter(Boolean);
};
const serialisePointsToDoorText = (points, boxes, lines) => {
  if (!Array.isArray(points)) {
    return '';
  }

  return points
    .filter((annotation) => annotation.type === 'point')
    .map((annotation) => {
      const { labelId, x, y, anchor } = annotation;
      const prefix = isPointLabel(labelId) && labelId !== DOOR_LABEL_ID ? `${labelId} ` : '';
      if (anchor?.type === 'line') {
        const lineIndex =
          Number.isInteger(anchor.index) && anchor.index >= 0
            ? anchor.index
            : lines.findIndex((line) => line.id === anchor.id);
        const identifier = lineIndex >= 0 ? lineIndex : anchor.id ?? 'unknown';
        return `${prefix}${normaliseValue(x)} ${normaliseValue(y)} line ${identifier} ${anchor.id ?? 'unknown'} ${normaliseValue(anchor.t ?? 0)}`.trim();
      }
      if (anchor?.type === 'box') {
        const boxIndex =
          Number.isInteger(anchor.index) && anchor.index >= 0
            ? anchor.index
            : boxes.findIndex((box) => box.id === anchor.id);
        const identifier = boxIndex >= 0 ? boxIndex : anchor.id ?? 'unknown';
        return `${prefix}${normaliseValue(x)} ${normaliseValue(y)} box ${identifier} ${anchor.id ?? 'unknown'} ${anchor.edge ?? 'unknown'} ${normaliseValue(anchor.t ?? 0)}`.trim();
      }
      return `${prefix}${normaliseValue(x)} ${normaliseValue(y)}`.trim();
    })
    .join('\n');
};

export const uploadFloorPlan = async (file) => {
  await delay(2000);

  const [yoloResponse, wallResponse] = await Promise.all([fetch(yoloTxtPath), fetch(wallTxtPath)]);
  const [yoloText, wallText] = await Promise.all([yoloResponse.text(), wallResponse.text()]);

  const boxes = parseYoloBoxes(yoloText);
  const doorCandidates = parseDoorCandidatesFromYolo(yoloText);
  const rawLines = parseWallLines(wallText);
  const lines = subtractBoxesFromLines(rawLines, boxes);
  const points = createDoorPointsFromCandidates(doorCandidates, boxes, lines);
  const doorText = serialisePointsToDoorText(points, boxes, lines);
  const imageUrl = await readFileAsDataUrl(file);

  return {
    fileName: file.name,
    imageUrl,
    boxes,
    lines,
    points,
    rawYoloText: yoloText,
    rawWallText: wallText,
    rawDoorText: doorText,
  };
};

export const saveAnnotations = async ({ boxes, lines, points }) => {
  await delay(2000);

  const savedYoloText = serialiseBoxesToYoloText(boxes);
  const savedWallText = serialiseLinesToWallText(lines);
  const savedDoorText = serialisePointsToDoorText(points, boxes, lines);

  return {
    savedYoloText,
    savedWallText,
    savedDoorText,
  };
};
