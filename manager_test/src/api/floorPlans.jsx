import yoloTxtPath from '../dummy/yolo.txt';
import wallTxtPath from '../dummy/wall.txt';
import doorTxtPath from '../dummy/door.txt';
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
        labelId = '0';
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
      const prefix = isPointLabel(labelId) && labelId !== '0' ? `${labelId} ` : '';
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

  const [yoloResponse, wallResponse, doorResponse] = await Promise.all([
    fetch(yoloTxtPath),
    fetch(wallTxtPath),
    fetch(doorTxtPath),
  ]);
  const [yoloText, wallText, doorText] = await Promise.all([
    yoloResponse.text(),
    wallResponse.text(),
    doorResponse.text(),
  ]);

  const boxes = parseYoloBoxes(yoloText);
  const rawLines = parseWallLines(wallText);
  const lines = subtractBoxesFromLines(rawLines, boxes);
  const points = parseDoorPoints(doorText, boxes, lines);
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
