import api from './client.jsx';
import { isBoxLabel, isPointLabel } from '../config/annotationConfig';
import { subtractBoxesFromLines } from '../utils/wallTrimmer';
import { saveStepOneResult } from './stepOneResults';
import { computeMetersPerPixel, parseLengthInput, sanitizeScaleLine } from '../utils/scaleReference';

const ENABLE_INITIAL_WALL_PRESERVATION = false;
const INITIAL_WALL_TRIM_MARGIN = 0.004; // trims ~0.2% of image span around RF-DETR boxes
const INITIAL_WALL_TRIM_OPTIONS = Object.freeze({
  cutMargin: INITIAL_WALL_TRIM_MARGIN,
  edgeProximityMargin: INITIAL_WALL_TRIM_MARGIN,
});

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

const normalisePreviewPayload = (preview, fallbackWidth, fallbackHeight) => {
  if (!preview) {
    return null;
  }
  const artifactBundle = preview.artifactBundle ?? preview.artifact_bundle;
  if (!artifactBundle) {
    return null;
  }
  const rawSize = preview.imageSize || preview.image_size || {};
  const safeWidth = Number.isFinite(rawSize.width)
    ? Math.round(rawSize.width)
    : Number.isFinite(fallbackWidth)
      ? Math.round(fallbackWidth)
      : 0;
  const safeHeight = Number.isFinite(rawSize.height)
    ? Math.round(rawSize.height)
    : Number.isFinite(fallbackHeight)
      ? Math.round(fallbackHeight)
      : 0;
  const payload = {
    imageSize: {
      width: safeWidth,
      height: safeHeight,
    },
    artifactBundle,
  };
  const ratio = preview.freeSpaceRatio ?? preview.free_space_ratio;
  if (Number.isFinite(ratio)) {
    payload.freeSpaceRatio = Number(ratio);
  }
  if (preview.config) {
    payload.config = preview.config;
  }
  return payload;
};

const buildPreviewPersistencePayload = (preview, fallbackWidth, fallbackHeight) => {
  if (!preview) {
    return null;
  }

  const resolveImageSize = (value) => {
    if (value && Number.isFinite(value.width) && Number.isFinite(value.height)) {
      return {
        width: Math.round(value.width),
        height: Math.round(value.height),
      };
    }
    if (Number.isFinite(fallbackWidth) && Number.isFinite(fallbackHeight)) {
      return {
        width: Math.round(fallbackWidth),
        height: Math.round(fallbackHeight),
      };
    }
    return { width: 0, height: 0 };
  };

  const directArtifactBundle = preview.artifactBundle ?? preview.artifact_bundle ?? null;
  if (directArtifactBundle) {
    const payload = {
      imageSize: resolveImageSize(preview.imageSize ?? preview.image_size),
      artifactBundle: directArtifactBundle,
    };
    const ratio = preview.freeSpaceRatio ?? preview.free_space_ratio;
    if (typeof ratio === 'number') {
      payload.freeSpaceRatio = ratio;
    }
    if (preview.config) {
      payload.config = preview.config;
    }
    return payload;
  }

  const normalised = normalisePreviewPayload(preview, fallbackWidth, fallbackHeight);
  if (!normalised) {
    return null;
  }
  const artifactBundle = normalised.artifactBundle ?? normalised.artifact_bundle;
  if (!artifactBundle) {
    return null;
  }
  const payload = {
    imageSize: normalised.imageSize,
    artifactBundle,
  };
  if (typeof normalised.freeSpaceRatio === 'number') {
    payload.freeSpaceRatio = normalised.freeSpaceRatio;
  }
  if (normalised.config) {
    payload.config = normalised.config;
  }
  return payload;
};

export const parseDetectionBoxes = (text) => {
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

const parseDoorCandidatesFromDetections = (text) => {
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
            let lineRef = null;
            if (lineId) {
              lineRef = lines.find((candidate) => candidate.id === lineId) || null;
            }
            if (!lineRef && normalisedIndex >= 0 && normalisedIndex < lines.length) {
              lineRef = lines[normalisedIndex];
            }
            const resolvedId = lineRef?.id || lineId;
            if (resolvedId) {
              point.anchor = {
                type: 'line',
                id: resolvedId,
                index: lineRef ? lines.indexOf(lineRef) : undefined,
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
            let boxRef = null;
            if (boxId) {
              boxRef = boxes.find((candidate) => candidate.id === boxId) || null;
            }
            if (!boxRef && normalisedIndex >= 0 && normalisedIndex < boxes.length) {
              boxRef = boxes[normalisedIndex];
            }
            const resolvedId = boxRef?.id || boxId;
            if (resolvedId) {
              point.anchor = {
                type: 'box',
                id: resolvedId,
                index: boxRef ? boxes.indexOf(boxRef) : undefined,
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

const serialiseBoxesToDetectionText = (boxes) => {
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
      const identifierToken = id ?? meta?.identifierToken ?? `wall-${index}`;
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

const ensureLinesCoverDoorAnchors = (lines, baseLines, points) => {
  const resolvedLines = Array.isArray(lines) ? lines.filter(Boolean) : [];
  const seenLineIds = new Set();
  resolvedLines.forEach((line) => {
    if (line?.id) {
      seenLineIds.add(line.id);
    }
  });

  const fallbackLookup = new Map();
  const registerFallback = (source) => {
    if (!Array.isArray(source)) {
      return;
    }
    source.forEach((line) => {
      if (line?.id && !fallbackLookup.has(line.id)) {
        fallbackLookup.set(line.id, line);
      }
    });
  };

  registerFallback(baseLines);
  registerFallback(lines);

  let recoveredCount = 0;
  (Array.isArray(points) ? points : []).forEach((point) => {
    if (point?.anchor?.type !== 'line') {
      return;
    }
    const anchorId = point.anchor.id;
    if (!anchorId || seenLineIds.has(anchorId)) {
      return;
    }
    const fallback = fallbackLookup.get(anchorId);
    if (!fallback) {
      return;
    }
    resolvedLines.push(fallback);
    seenLineIds.add(anchorId);
    recoveredCount += 1;
  });

  return {
    lines: resolvedLines,
    addedCount: recoveredCount,
  };
};

const validateDoorAnchors = (points, boxes, lines) => {
  const boxesMap = new Map((Array.isArray(boxes) ? boxes : []).map((box) => [box?.id, box]).filter(([id]) => id));
  const linesMap = new Map((Array.isArray(lines) ? lines : []).map((line) => [line?.id, line]).filter(([id]) => id));

  const validPoints = [];
  const invalidPoints = [];

  (Array.isArray(points) ? points : []).forEach((point) => {
    const anchor = point?.anchor;
    if (!anchor) {
      invalidPoints.push(point);
      return;
    }
    if (anchor.type === 'line') {
      if (anchor.id && linesMap.has(anchor.id)) {
        validPoints.push(point);
        return;
      }
      invalidPoints.push(point);
      return;
    }
    if (anchor.type === 'box') {
      if (anchor.id && boxesMap.has(anchor.id) && anchor.edge) {
        validPoints.push(point);
        return;
      }
      invalidPoints.push(point);
      return;
    }
    invalidPoints.push(point);
  });

  return { validPoints, invalidPoints };
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
  ENABLE_INITIAL_WALL_PRESERVATION ? lines : subtractBoxesFromLines(lines, boxes, INITIAL_WALL_TRIM_OPTIONS);

export const uploadFloorPlan = async (file) => {
  const imageUrl = await readFileAsDataUrl(file);
  const { width: localWidth, height: localHeight } = await loadImageDimensions(imageUrl);

  const formData = new FormData();
  formData.append('file', file);
  formData.append('image_width', Math.round(localWidth));
  formData.append('image_height', Math.round(localHeight));

  let inferenceResult = null;
  try {
    const response = await api.post('/api/floorplans/inference', formData);
    inferenceResult = response?.data ?? null;
  } catch (error) {
    console.error('RF-DETR 추론 요청에 실패했습니다.', error);
    throw error;
  }

  const objectDetectionText = inferenceResult?.object_detection_text ?? inferenceResult?.objectDetectionText ?? '';
  const wallText = inferenceResult?.wall_text ?? inferenceResult?.wallText ?? '';
  const doorFileText = inferenceResult?.door_text ?? inferenceResult?.doorText ?? '';
  const fileName = inferenceResult?.file_name ?? inferenceResult?.fileName ?? file.name;

  const boxes = parseDetectionBoxes(objectDetectionText);
  const doorCandidates = parseDoorCandidatesFromDetections(objectDetectionText);
  const rawLines = parseWallLines(wallText);
  const lines = subtractBoxesFromLinesForInitialLoad(rawLines, boxes);

  const parsedDoorPoints = parseDoorPoints(doorFileText, boxes, lines);
  const autoDoorPoints = createDoorPointsFromCandidates(doorCandidates, boxes, lines);
  const points = parsedDoorPoints.length > 0 ? parsedDoorPoints : autoDoorPoints;
  const doorText = parsedDoorPoints.length > 0 ? doorFileText.trim() : serialisePointsToDoorText(points, boxes, lines);

  const imageWidth = inferenceResult?.image_width ?? inferenceResult?.imageWidth ?? localWidth;
  const imageHeight = inferenceResult?.image_height ?? inferenceResult?.imageHeight ?? localHeight;

  return {
    fileName,
    imageUrl,
    imageWidth,
    imageHeight,
    boxes,
    lines,
    points,
    rawObjectDetectionText: objectDetectionText,
    rawWallText: wallText,
    rawDoorText: doorText,
  };
};

export const saveAnnotations = async ({
  fileName,
  floorLabel,
  floorValue,
  calibrationLine,
  calibrationLengthMeters,
  imageUrl,
  boxes,
  lines,
  points,
  baseLines,
  wallFilter,
  rawObjectDetectionText,
  rawWallText,
  rawDoorText,
  imageWidth,
  imageHeight,
  classNames = DEFAULT_CLASS_NAMES,
  sourceImagePath,
  sourceOriginalId,
  requestId,
  freeSpacePreview = null,
  mode = 'server-only',
}) => {
  const clonedBoxes = Array.isArray(boxes) ? boxes.map(cloneBox) : [];
  const clonedLines = Array.isArray(lines) ? lines.map(cloneLine) : [];
  const clonedPoints = Array.isArray(points) ? points.map(clonePoint) : [];
  const clonedBaseLines = Array.isArray(baseLines) ? baseLines.map(cloneLine) : clonedLines;

  const { lines: resolvedLinesForSaving, addedCount: anchorLineRecoveryCount } = ensureLinesCoverDoorAnchors(
    clonedLines,
    clonedBaseLines,
    clonedPoints
  );

  const { validPoints: anchoredPoints, invalidPoints: invalidDoorPoints } = validateDoorAnchors(
    clonedPoints,
    clonedBoxes,
    resolvedLinesForSaving
  );
  if (invalidDoorPoints.length > 0) {
    const suffix = invalidDoorPoints.length === 1 ? '문 1개' : `문 ${invalidDoorPoints.length}개`;
    throw new Error(
      `${suffix}가 기준 벽/공간과 연결이 끊어졌습니다. 해당 문을 삭제하거나 다시 벽/박스에 붙인 뒤 저장해 주세요.`
    );
  }

  const savedObjectDetectionText = serialiseBoxesToDetectionText(clonedBoxes);
  const savedWallText = serialiseLinesToWallText(resolvedLinesForSaving);
  const savedWallBaseText = serialiseLinesToWallText(clonedBaseLines);
  const savedDoorText = serialisePointsToDoorText(anchoredPoints, clonedBoxes, resolvedLinesForSaving);

  const previewPayloadForLegacy = normalisePreviewPayload(freeSpacePreview, imageWidth, imageHeight);
  const previewArtifactsPayload = buildPreviewPersistencePayload(freeSpacePreview, imageWidth, imageHeight);

  let processingResult = null;
  let saveStepOneResponse = null;
  let requestIdentifier = requestId ?? null;

  const canPersistOnServer =
    Number.isFinite(imageWidth) &&
    Number.isFinite(imageHeight) &&
    imageWidth > 0 &&
    imageHeight > 0 &&
    savedObjectDetectionText &&
    savedWallText &&
    savedDoorText;

  const calibrationLengthValue = parseLengthInput(calibrationLengthMeters);
  const sanitizedCalibrationLine = sanitizeScaleLine(calibrationLine, { enforceValid: true });
  if (!sanitizedCalibrationLine || !Number.isFinite(calibrationLengthValue) || calibrationLengthValue <= 0) {
    throw new Error('기준선 길이와 실제 길이를 먼저 입력해 주세요.');
  }
  const derivedMetersPerPixel = computeMetersPerPixel(
    sanitizedCalibrationLine,
    calibrationLengthValue,
    imageWidth,
    imageHeight
  );
  if (!Number.isFinite(derivedMetersPerPixel) || derivedMetersPerPixel <= 0) {
    throw new Error('이미지 크기 또는 기준선 정보를 확인해 주세요.');
  }

  const scaleReferencePayload = {
    ...sanitizedCalibrationLine,
    lengthMeters: calibrationLengthValue,
  };

  if (canPersistOnServer && mode === 'legacy') {
    try {
      const payload = {
        object_detection_text: savedObjectDetectionText,
        wall_text: savedWallText,
        door_text: savedDoorText,
        wallBaseText: savedWallBaseText,
        image_width: Math.round(imageWidth),
        image_height: Math.round(imageHeight),
        class_names: classNames,
        source_image_path: sourceImagePath,
        skipGraph: true,
        scale_reference: scaleReferencePayload,
      };
      if (imageUrl) {
        payload.imageDataUrl = imageUrl;
      }
      if (sourceOriginalId) {
        payload.source_original_id = sourceOriginalId;
      }
      if (requestId) {
        payload.requestId = requestId;
      }
      if (floorLabel) {
        payload.floorLabel = floorLabel;
      }
      if (floorValue) {
        payload.floorValue = floorValue;
      }
      if (previewPayloadForLegacy) {
        payload.free_space_preview = previewPayloadForLegacy;
      }
      const response = await api.post('/api/floorplans/process', payload);
      processingResult = response?.data ?? null;
      requestIdentifier = processingResult?.request_id ?? requestIdentifier;
    } catch (error) {
      console.error('Step 1 저장을 위한 백엔드 처리에 실패했습니다.', error);
      const detail = error?.response?.data?.detail;
      const message =
        (typeof detail === 'string' && detail.trim().length > 0 ? detail : null) ||
        error?.message ||
        '백엔드 저장 중 오류가 발생했습니다.';
      throw new Error(message);
    }
  } else if (canPersistOnServer && mode !== 'legacy') {
    try {
      const payload = {
        objectDetectionText: savedObjectDetectionText,
        wallText: savedWallText,
        wallBaseText: savedWallBaseText,
        doorText: savedDoorText,
        imageWidth: Math.round(imageWidth),
        imageHeight: Math.round(imageHeight),
        classNames,
        scaleReference: scaleReferencePayload,
      };
      if (sourceImagePath) {
        payload.sourceImagePath = sourceImagePath;
      }
      if (imageUrl) {
        payload.imageDataUrl = imageUrl;
      }
      if (requestId) {
        payload.requestId = requestId;
      }
      if (floorLabel) {
        payload.floorLabel = floorLabel;
      }
      if (floorValue) {
        payload.floorValue = floorValue;
      }
      if (previewArtifactsPayload) {
        payload.freeSpacePreview = previewArtifactsPayload;
      }
      const response = await api.post('/api/floorplans/save-step-one', payload);
      saveStepOneResponse = response?.data ?? null;
      requestIdentifier = saveStepOneResponse?.requestId ?? requestIdentifier;
      if (saveStepOneResponse) {
        const stubMetadata = {
          ...(saveStepOneResponse.metadata ?? {}),
        };
        if (saveStepOneResponse.preview) {
          stubMetadata.preview = saveStepOneResponse.preview;
        }
        processingResult = {
          request_id: saveStepOneResponse.requestId ?? requestIdentifier ?? null,
          created_at: saveStepOneResponse.createdAt ?? new Date().toISOString(),
          image_size: saveStepOneResponse.imageSize ?? { width: Math.round(imageWidth), height: Math.round(imageHeight) },
          class_names: saveStepOneResponse.classNames ?? classNames,
          objects: null,
          graph: null,
          metadata: stubMetadata,
        };
      }
    } catch (error) {
      console.error('Step 1 서버 저장에 실패했습니다.', error);
      const detail = error?.response?.data?.detail;
      const message =
        (typeof detail === 'string' && detail.trim().length > 0 ? detail : null) ||
        error?.message ||
        'Step1 저장 중 오류가 발생했습니다.';
      throw new Error(message);
    }
  }

  const metadataFromServer =
    processingResult?.metadata ??
    saveStepOneResponse?.metadata ??
    null;

  const resolvedFloorLabel =
    metadataFromServer?.floorLabel ??
    metadataFromServer?.floor_label ??
    floorLabel ??
    null;
  const resolvedFloorValue =
    metadataFromServer?.floorValue ??
    metadataFromServer?.floor_value ??
    floorValue ??
    null;

  const backendImageUrl = metadataFromServer?.image_url ?? metadataFromServer?.imageUrl ?? null;
  const backendImageDataUrl = metadataFromServer?.image_data_url ?? metadataFromServer?.imageDataUrl ?? null;

  const resolvedImageUrl = normaliseApiUrl(backendImageUrl) ?? imageUrl ?? null;
  const resolvedImageDataUrl =
    backendImageUrl && !backendImageUrl.startsWith('data:')
      ? (backendImageDataUrl ?? null)
      : (backendImageDataUrl ?? (imageUrl?.startsWith('data:') ? imageUrl : null));

  const stepOneResult = await saveStepOneResult({
    objectDetection: {
      raw: rawObjectDetectionText ?? '',
      text: savedObjectDetectionText,
      boxes: clonedBoxes,
    },
    wall: {
      raw: rawWallText ?? '',
      text: savedWallText,
      baseText: savedWallBaseText,
      lines: resolvedLinesForSaving,
      baseLines: clonedBaseLines,
      filter: wallFilter ?? null,
    },
    door: {
      raw: rawDoorText ?? '',
      text: savedDoorText,
      points: anchoredPoints,
    },
    wallBaseText: savedWallBaseText,
    imageUrl: resolvedImageUrl,
      metadata: {
        ...(metadataFromServer ?? {}),
        fileName: fileName ?? 'floor-plan.png',
        imageUrl: resolvedImageUrl,
        imageWidth: imageWidth ?? 0,
        imageHeight: imageHeight ?? 0,
        floorLabel: resolvedFloorLabel ?? null,
        floorValue: resolvedFloorValue ?? null,
        scaleReference: scaleReferencePayload,
      },
    imageDataUrl: resolvedImageDataUrl,
    processingResult,
    requestId: requestIdentifier ?? processingResult?.request_id ?? requestId ?? null,
    sourceOriginalId: sourceOriginalId ?? null,
    floorLabel: resolvedFloorLabel ?? null,
    floorValue: resolvedFloorValue ?? null,
    calibrationLine: sanitizedCalibrationLine,
    calibrationLengthMeters: String(calibrationLengthValue),
    preview:
      saveStepOneResponse?.preview ??
      processingResult?.metadata?.preview ??
      previewArtifactsPayload ??
      previewPayloadForLegacy,
  });

  return {
    stepOneResult,
    processingResult,
    savedObjectDetectionText,
    savedWallText,
    savedDoorText,
    anchorLineRecoveryCount,
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

export const fetchFloorPlanFloors = async () => {
  const response = await api.get('/api/floorplans/floors');
  return response?.data ?? [];
};

export const fetchStoredFloorPlanByStepOneId = async (stepOneId) => {
  if (!stepOneId) {
    throw new Error('stepOneId가 필요합니다.');
  }
  const response = await api.get(`/api/floorplans/by-step-one/${stepOneId}`);
  return response?.data ?? null;
};

export const deleteStoredFloorPlan = async ({ requestId, stepOneId }) => {
  if (!requestId) {
    throw new Error('requestId가 필요합니다.');
  }
  const params = new URLSearchParams();
  if (stepOneId) {
    params.append('stepOneId', stepOneId);
  }
  const query = params.toString() ? `?${params.toString()}` : '';
  const response = await api.delete(`/api/floorplans/${requestId}${query}`);
  return response?.data ?? null;
};

export const fetchFloorPlanGraph = async (requestId) => {
  if (!requestId) {
    throw new Error('requestId가 필요합니다.');
  }
  const response = await api.get(`/api/floorplans/${requestId}/graph`);
  return response?.data ?? null;
};

export const saveFloorPlanGraph = async (requestId, graphPayload) => {
  if (!requestId) {
    throw new Error('requestId가 필요합니다.');
  }
  const response = await api.put(`/api/floorplans/${requestId}/graph`, {
    graph: graphPayload,
  });
  return response?.data ?? null;
};

export const processStepOneForStepTwo = async ({ stepOneResult, requestId: overrideRequestId }) => {
  const resolvedRequestId =
    overrideRequestId ??
    stepOneResult?.processingResult?.request_id ??
    stepOneResult?.requestId ??
    null;
  if (!resolvedRequestId) {
    throw new Error('재계산할 requestId를 찾을 수 없습니다.');
  }

  const response = await api.post(`/api/floorplans/${resolvedRequestId}/prepare-graph`, {});
  const processingResult = response?.data ?? null;

  const mergedMetadata = {
    ...(stepOneResult?.metadata || {}),
    ...(processingResult?.metadata || {}),
  };

  const mergedStepOne = await saveStepOneResult({
    sourceOriginalId: stepOneResult?.id ?? processingResult?.request_id ?? resolvedRequestId,
    createdAt: stepOneResult?.createdAt ?? processingResult?.created_at ?? new Date().toISOString(),
    fileName: stepOneResult?.fileName ?? mergedMetadata?.fileName ?? `${resolvedRequestId}.json`,
    filePath: stepOneResult?.filePath,
    imageUrl:
      stepOneResult?.imageUrl ??
      mergedMetadata?.imageUrl ??
      mergedMetadata?.image_url ??
      null,
    imageDataUrl: stepOneResult?.imageDataUrl ?? mergedMetadata?.image_data_url ?? mergedMetadata?.imageDataUrl ?? null,
    metadata: mergedMetadata,
    objectDetection: stepOneResult?.objectDetection,
    wall: stepOneResult?.wall,
    door: stepOneResult?.door,
    processingResult,
    requestId: processingResult?.request_id ?? resolvedRequestId,
    wallBaseText: stepOneResult?.wall?.baseText ?? null,
    preview: processingResult?.metadata?.preview ?? stepOneResult?.preview ?? null,
  });

  return {
    stepOneResult: mergedStepOne,
    processingResult,
  };
};

export const autoCorrectLayout = async ({ boxes, lines, imageWidth, imageHeight }) => {
  const payload = {
    boxes: Array.isArray(boxes) ? boxes.map(cloneBox) : [],
    lines: Array.isArray(lines) ? lines.map(cloneLine) : [],
  };
  if (Number.isFinite(imageWidth) && imageWidth > 0) {
    payload.imageWidth = Math.round(imageWidth);
  }
  if (Number.isFinite(imageHeight) && imageHeight > 0) {
    payload.imageHeight = Math.round(imageHeight);
  }
  const response = await api.post('/api/floorplans/auto-correct', payload);
  return response?.data ?? null;
};

export const fetchFreeSpacePreview = async ({
  boxes,
  lines,
  baseLines,
  points,
  imageWidth,
  imageHeight,
  options,
}) => {
  if (!Number.isFinite(imageWidth) || !Number.isFinite(imageHeight) || imageWidth <= 0 || imageHeight <= 0) {
    throw new Error('미리보기를 생성하려면 이미지 크기 정보가 필요합니다.');
  }

  const clonedBoxes = Array.isArray(boxes) ? boxes.map(cloneBox) : [];
  const clonedLines = Array.isArray(lines) ? lines.map(cloneLine) : [];
  const clonedPoints = Array.isArray(points) ? points.map(clonePoint) : [];
  const resolvedBaseLines = Array.isArray(baseLines) && baseLines.length > 0 ? baseLines.map(cloneLine) : clonedLines;

  const { lines: resolvedLines } = ensureLinesCoverDoorAnchors(clonedLines, resolvedBaseLines, clonedPoints);

  const objectDetectionText = serialiseBoxesToDetectionText(clonedBoxes);
  const wallText = serialiseLinesToWallText(resolvedLines);
  const doorText = serialisePointsToDoorText(clonedPoints, clonedBoxes, resolvedLines);

  if (!objectDetectionText || !wallText) {
    throw new Error('박스와 벽 정보가 필요합니다. 편집 내용을 확인해 주세요.');
  }

  const payload = {
    objectDetectionText,
    wallText,
    doorText: doorText ?? '',
    imageWidth: Math.round(imageWidth),
    imageHeight: Math.round(imageHeight),
  };

  if (options && typeof options === 'object') {
    payload.options = {};
    if (Number.isFinite(options.doorProbeDistance)) {
      payload.options.doorProbeDistance = Math.max(1, Math.round(options.doorProbeDistance));
    }
    if (
      Array.isArray(options.morphOpenKernel) &&
      options.morphOpenKernel.length === 2 &&
      options.morphOpenKernel.every((value) => Number.isFinite(value) && value > 0)
    ) {
      payload.options.morphOpenKernel = options.morphOpenKernel.map((value) => Math.max(1, Math.round(value)));
    }
    if (Number.isFinite(options.morphOpenIterations)) {
      payload.options.morphOpenIterations = Math.max(1, Math.round(options.morphOpenIterations));
    }
  }

  const response = await api.post('/api/floorplans/free-space-preview', payload);
  return response?.data ?? null;
};
