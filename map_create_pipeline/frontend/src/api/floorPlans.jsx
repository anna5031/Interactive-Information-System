import api from './client.jsx';
import yoloTxtPath from '../dummy/yolo.txt';
import wallTxtPath from '../dummy/wall.txt';
import { isBoxLabel, isPointLabel } from '../config/annotationConfig';
import { subtractBoxesFromLines } from '../utils/wallTrimmer';
import { saveStepOneResult } from './stepOneResults';

const ENABLE_INITIAL_WALL_PRESERVATION = true;

const DOOR_LABEL_ID = '0';
const SMALL_NUMBER = 1e-9;
const DEFAULT_CLASS_NAMES = ['room', 'stairs', 'wall', 'elevator', 'door'];
const NUMERIC_TOKEN_REGEX = /^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$/;

const isNumericToken = (token) => {
  if (typeof token !== 'string') {
    return false;
  }
  const trimmed = token.trim();
  return trimmed.length > 0 && NUMERIC_TOKEN_REGEX.test(trimmed);
};

const readFileAsDataUrl = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });

const loadImageDimensions = (url) =>
  new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () =>
      resolve({
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
    image.onerror = () => reject(new Error('이미지 크기를 불러오지 못했습니다.'));
    image.src = url;
  });

const normaliseApiUrl = (value) => {
  if (!value || typeof value !== 'string') {
    return null;
  }
  if (value.startsWith('data:') || /^https?:\/\//i.test(value)) {
    return value;
  }
  const baseUrl = api?.defaults?.baseURL;
  if (!baseUrl) {
    return value;
  }
  const trimmedBase = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  const normalisedPath = value.startsWith('/') ? value : `/${value}`;
  return `${trimmedBase}${normalisedPath}`;
};

const normaliseValue = (value) => {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(Math.max(value, 0), 1);
};

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

      const [labelId, cxToken, cyToken, wToken, hToken, ...restTokens] = parts;
      if (!isBoxLabel(labelId)) {
        return null;
      }

      const width = Number.parseFloat(wToken);
      const height = Number.parseFloat(hToken);
      const cx = Number.parseFloat(cxToken);
      const cy = Number.parseFloat(cyToken);
      if (![width, height, cx, cy].every(Number.isFinite)) {
        return null;
      }

      const normalisedWidth = normaliseValue(width);
      const normalisedHeight = normaliseValue(height);
      const normalisedX = normaliseValue(cx - normalisedWidth / 2);
      const normalisedY = normaliseValue(cy - normalisedHeight / 2);

      let identifierToken = null;
      let confidenceToken = null;
      let trailingTokens = [];

      if (restTokens.length >= 2 && isNumericToken(restTokens[0])) {
        confidenceToken = restTokens[0];
        identifierToken = restTokens[1];
        trailingTokens = restTokens.slice(2);
      } else if (restTokens.length >= 1) {
        identifierToken = restTokens[0];
        trailingTokens = restTokens.slice(1);
      }

      const resolvedId = identifierToken ?? `${labelId}-box-${index}`;

      return {
        id: resolvedId,
        type: 'box',
        labelId,
        x: normalisedX,
        y: normalisedY,
        width: normalisedWidth,
        height: normalisedHeight,
        meta: {
          identifierToken,
          confidenceToken,
          trailingTokens,
        },
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

      const [x1Token, y1Token, x2Token, y2Token, ...restTokens] = parts;
      const coords = [x1Token, y1Token, x2Token, y2Token].map((value) => Number.parseFloat(value));
      if (!coords.every(Number.isFinite)) {
        return null;
      }

      const [x1, y1, x2, y2] = coords.map(normaliseValue);

      const identifierToken = restTokens.length > 0 ? restTokens[0] : null;
      const trailingTokens = restTokens.length > 1 ? restTokens.slice(1) : [];
      const resolvedId = identifierToken ?? `wall-${index}`;

      return {
        id: resolvedId,
        type: 'line',
        labelId: '4',
        x1,
        y1,
        x2,
        y2,
        meta: {
          identifierToken,
          trailingTokens,
        },
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

      const [labelId, cxToken, cyToken, widthToken, heightToken] = parts;
      if (labelId !== DOOR_LABEL_ID) {
        return null;
      }

      const width = Number.parseFloat(widthToken);
      const height = Number.parseFloat(heightToken);
      const cx = Number.parseFloat(cxToken);
      const cy = Number.parseFloat(cyToken);
      if (![width, height, cx, cy].every(Number.isFinite)) {
        return null;
      }
      if (width <= 0 || height <= 0) {
        return null;
      }

      const normalisedWidth = normaliseValue(width);
      const normalisedHeight = normaliseValue(height);
      if (normalisedWidth <= 0 || normalisedHeight <= 0) {
        return null;
      }

      const normalisedCx = normaliseValue(cx);
      const normalisedCy = normaliseValue(cy);
      const x = normaliseValue(normalisedCx - normalisedWidth / 2);
      const y = normaliseValue(normalisedCy - normalisedHeight / 2);

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

export const parseDoorPoints = (text, boxes, lines) => {
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

      const xToken = parts[cursor];
      const yToken = parts[cursor + 1];
      const rest = parts.slice(cursor + 2);

      const xValue = Number.parseFloat(xToken);
      const yValue = Number.parseFloat(yToken);
      if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) {
        return null;
      }

      const point = {
        id: `door-${index}`,
        type: 'point',
        labelId,
        x: normaliseValue(xValue),
        y: normaliseValue(yValue),
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
      if (-d < 0) {
        tNumerator = 0;
        tDenominator = 1;
      } else if (-d > a) {
        tNumerator = tDenominator;
      } else {
        tNumerator = -d;
        tDenominator = a;
      }
    } else if (sNumerator > sDenominator) {
      sNumerator = sDenominator;
      if (-d + b < 0) {
        tNumerator = 0;
        tDenominator = 1;
      } else if (-d + b > a) {
        tNumerator = tDenominator;
      } else {
        tNumerator = -d + b;
        tDenominator = a;
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

    segments.push({ type: 'box', id: box.id, index, edge: 'top', ax: minX, ay: minY, bx: maxX, by: minY });
    segments.push({ type: 'box', id: box.id, index, edge: 'bottom', ax: minX, ay: maxY, bx: maxX, by: maxY });
    segments.push({ type: 'box', id: box.id, index, edge: 'left', ax: minX, ay: minY, bx: minX, by: maxY });
    segments.push({ type: 'box', id: box.id, index, edge: 'right', ax: maxX, ay: minY, bx: maxX, by: maxY });
  });

  return segments;
};

const findClosestAnchorForDoor = (doorRect, anchorSegments) => {
  if (!doorRect || !Array.isArray(anchorSegments) || anchorSegments.length === 0) {
    return null;
  }

  const doorCenterX = normaliseValue(doorRect.x + doorRect.width / 2);
  const doorCenterY = normaliseValue(doorRect.y + doorRect.height / 2);

  let bestMatch = null;
  let bestDistance = Number.POSITIVE_INFINITY;

  anchorSegments.forEach((segment) => {
    const result = closestPointsBetweenSegments(
      doorRect.x,
      doorRect.y,
      doorRect.x + doorRect.width,
      doorRect.y + doorRect.height,
      segment.ax,
      segment.ay,
      segment.bx,
      segment.by
    );

    if (!result) {
      return;
    }

    const doorPoint = { x: doorCenterX, y: doorCenterY };
    const distance = Math.hypot(doorPoint.x - result.point2.x, doorPoint.y - result.point2.y);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestMatch = {
        segment,
        anchorPoint: result.point2,
        t: result.t2,
      };
    }
  });

  return bestMatch;
};

const createDoorPointsFromCandidates = (candidates, boxes, lines) => {
  if (!Array.isArray(candidates) || candidates.length === 0) {
    return [];
  }

  const anchorSegments = buildAnchorSegments(boxes, lines);

  return candidates.map((candidate, index) => {
    let anchor = null;
    let x = normaliseValue(candidate.x + candidate.width / 2);
    let y = normaliseValue(candidate.y + candidate.height / 2);

    const anchorMatch = findClosestAnchorForDoor(candidate, anchorSegments);
    if (anchorMatch) {
      const { segment, anchorPoint, t } = anchorMatch;
      if (segment.type === 'line') {
        anchor = {
          type: 'line',
          id: segment.id,
          index: segment.index,
          t: clampFraction(t ?? 0),
        };
        x = normaliseValue(anchorPoint.x);
        y = normaliseValue(anchorPoint.y);
      } else if (segment.type === 'box') {
        anchor = {
          type: 'box',
          id: segment.id,
          index: segment.index,
          edge: segment.edge,
          t: clampFraction(t ?? 0),
        };
        if (segment.edge === 'top' || segment.edge === 'bottom') {
          x = clampToRange(anchorPoint.x, segment.ax, segment.bx);
          y = normaliseValue(segment.ay);
        } else {
          y = clampToRange(anchorPoint.y, segment.ay, segment.by);
          x = normaliseValue(segment.ax);
        }
      }
    }

    const point = {
      id: `door-${index}`,
      type: 'point',
      labelId: DOOR_LABEL_ID,
      x: normaliseValue(x),
      y: normaliseValue(y),
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
      const { labelId, x, y, width, height, id, meta } = annotation;
      const centerX = normaliseValue(x + width / 2);
      const centerY = normaliseValue(y + height / 2);
      const normalisedWidth = normaliseValue(width);
      const normalisedHeight = normaliseValue(height);

      const identifierToken = meta?.identifierToken ?? id ?? `${labelId}-box-${index}`;
      const tokens = [labelId, `${centerX}`, `${centerY}`, `${normalisedWidth}`, `${normalisedHeight}`];

      if (meta?.confidenceToken && meta.identifierToken) {
        tokens.push(meta.confidenceToken);
      }

      tokens.push(identifierToken);

      if (Array.isArray(meta?.trailingTokens) && meta.trailingTokens.length > 0) {
        tokens.push(...meta.trailingTokens);
      }

      return tokens.join(' ').trim();
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
      const { x1, y1, x2, y2, id, meta } = annotation;
      const identifierToken = meta?.identifierToken ?? id ?? `wall-${index}`;
      const baseTokens = [
        `${normaliseValue(x1)}`,
        `${normaliseValue(y1)}`,
        `${normaliseValue(x2)}`,
        `${normaliseValue(y2)}`,
        identifierToken,
      ];

      if (Array.isArray(meta?.trailingTokens) && meta.trailingTokens.length > 0) {
        baseTokens.push(...meta.trailingTokens);
      }

      return baseTokens.join(' ').trim();
    })
    .join('\n');
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
        const identifier = lineIndex >= 0 ? lineIndex : (anchor.id ?? 'unknown');
        return `${prefix}${normaliseValue(x)} ${normaliseValue(y)} line ${identifier} ${anchor.id ?? 'unknown'} ${normaliseValue(anchor.t ?? 0)}`.trim();
      }
      if (anchor?.type === 'box') {
        const boxIndex =
          Number.isInteger(anchor.index) && anchor.index >= 0
            ? anchor.index
            : boxes.findIndex((box) => box.id === anchor.id);
        const identifier = boxIndex >= 0 ? boxIndex : (anchor.id ?? 'unknown');
        return `${prefix}${normaliseValue(x)} ${normaliseValue(y)} box ${identifier} ${anchor.id ?? 'unknown'} ${anchor.edge ?? 'unknown'} ${normaliseValue(anchor.t ?? 0)}`.trim();
      }
      return `${prefix}${normaliseValue(x)} ${normaliseValue(y)}`.trim();
    })
    .join('\n');
};

const cloneBox = (box) => ({
  ...box,
  meta: box?.meta
    ? {
        ...box.meta,
        trailingTokens: Array.isArray(box.meta.trailingTokens) ? [...box.meta.trailingTokens] : box.meta.trailingTokens,
      }
    : undefined,
});

const cloneLine = (line) => ({
  ...line,
  meta: line?.meta
    ? {
        ...line.meta,
        trailingTokens: Array.isArray(line.meta.trailingTokens)
          ? [...line.meta.trailingTokens]
          : line.meta.trailingTokens,
      }
    : undefined,
});

const clonePoint = (point) => ({ ...point, anchor: point?.anchor ? { ...point.anchor } : undefined });

const subtractBoxesFromLinesForInitialLoad = (lines, boxes) =>
  ENABLE_INITIAL_WALL_PRESERVATION ? lines : subtractBoxesFromLines(lines, boxes);

export const uploadFloorPlan = async (file) => {
  const [yoloResponse, wallResponse, doorResponse] = await Promise.all([fetch(yoloTxtPath), fetch(wallTxtPath)]);

  const [yoloText, wallText, doorFileText] = await Promise.all([
    yoloResponse.text(),
    wallResponse.text(),
    doorResponse ? doorResponse.text() : Promise.resolve(''),
  ]);

  const boxes = parseYoloBoxes(yoloText);
  const doorCandidates = parseDoorCandidatesFromYolo(yoloText);
  const rawLines = parseWallLines(wallText);
  const lines = subtractBoxesFromLinesForInitialLoad(rawLines, boxes);

  const parsedDoorPoints = parseDoorPoints(doorFileText, boxes, lines);
  const autoDoorPoints = createDoorPointsFromCandidates(doorCandidates, boxes, lines);
  const points = parsedDoorPoints.length > 0 ? parsedDoorPoints : autoDoorPoints;
  const doorText = parsedDoorPoints.length > 0 ? doorFileText.trim() : serialisePointsToDoorText(points, boxes, lines);

  const imageUrl = await readFileAsDataUrl(file);
  const { width: imageWidth, height: imageHeight } = await loadImageDimensions(imageUrl);

  return {
    fileName: file.name,
    imageUrl,
    imageWidth,
    imageHeight,
    boxes,
    lines,
    points,
    rawYoloText: yoloText,
    rawWallText: wallText,
    rawDoorText: doorText,
  };
};

export const saveAnnotations = async ({
  fileName,
  imageUrl,
  boxes,
  lines,
  points,
  rawYoloText,
  rawWallText,
  rawDoorText,
  imageWidth,
  imageHeight,
  classNames = DEFAULT_CLASS_NAMES,
  sourceImagePath,
  sourceOriginalId,
}) => {
  const clonedBoxes = Array.isArray(boxes) ? boxes.map(cloneBox) : [];
  const clonedLines = Array.isArray(lines) ? lines.map(cloneLine) : [];
  const clonedPoints = Array.isArray(points) ? points.map(clonePoint) : [];

  const savedYoloText = serialiseBoxesToYoloText(clonedBoxes);
  const savedWallText = serialiseLinesToWallText(clonedLines);
  const savedDoorText = serialisePointsToDoorText(clonedPoints, clonedBoxes, clonedLines);

  let processingResult = null;

  if (
    Number.isFinite(imageWidth) &&
    Number.isFinite(imageHeight) &&
    imageWidth > 0 &&
    imageHeight > 0 &&
    savedYoloText &&
    savedWallText &&
    savedDoorText
  ) {
    try {
      const payload = {
        yolo_text: savedYoloText,
        wall_text: savedWallText,
        door_text: savedDoorText,
        image_width: Math.round(imageWidth),
        image_height: Math.round(imageHeight),
        class_names: classNames,
        source_image_path: sourceImagePath,
      };
      if (imageUrl) {
        payload.imageDataUrl = imageUrl;
      }
      if (sourceOriginalId) {
        payload.source_original_id = sourceOriginalId;
      }

      const response = await api.post('/api/floorplans/process', payload);
      processingResult = response?.data ?? null;
    } catch (error) {
      console.error('백엔드 처리에 실패했습니다.', error);
    }
  }

  const backendImageUrl =
    processingResult?.metadata?.image_url ?? processingResult?.metadata?.imageUrl ?? null;
  const backendImageDataUrl =
    processingResult?.metadata?.image_data_url ?? processingResult?.metadata?.imageDataUrl ?? null;

  const resolvedImageUrl = normaliseApiUrl(backendImageUrl) ?? imageUrl ?? null;
  const resolvedImageDataUrl = backendImageUrl ? null : backendImageDataUrl ?? imageUrl ?? null;

  const stepOneResult = await saveStepOneResult({
    yolo: {
      raw: rawYoloText ?? '',
      text: savedYoloText,
      boxes: clonedBoxes,
    },
    wall: {
      raw: rawWallText ?? '',
      text: savedWallText,
      lines: clonedLines,
    },
    door: {
      raw: rawDoorText ?? '',
      text: savedDoorText,
      points: clonedPoints,
    },
    imageUrl: resolvedImageUrl,
    imageDataUrl: resolvedImageDataUrl,
    metadata: {
      fileName: fileName ?? 'floor-plan.png',
      imageUrl: resolvedImageUrl,
      imageWidth: imageWidth ?? 0,
      imageHeight: imageHeight ?? 0,
    },
    processingResult,
    sourceOriginalId: sourceOriginalId ?? null,
  });

  return {
    stepOneResult,
    processingResult,
    savedYoloText,
    savedWallText,
    savedDoorText,
  };
};

export const fetchProcessingResultById = async (requestId) => {
  if (!requestId) {
    throw new Error('requestId가 필요합니다.');
  }
  const response = await api.get(`/api/floorplans/${requestId}`);
  return response?.data ?? null;
};

export const fetchStoredFloorPlanSummaries = async () => {
  const response = await api.get('/api/floorplans');
  return response?.data ?? [];
};

export const fetchStoredFloorPlanByStepOneId = async (stepOneId) => {
  if (!stepOneId) {
    throw new Error('stepOneId가 필요합니다.');
  }
  const response = await api.get(`/api/floorplans/by-step-one/${stepOneId}`);
  return response?.data ?? null;
};
