import yoloTxtPath from '../dummy/yolo.txt';
import wallTxtPath from '../dummy/wall.txt';
import { isLineLabel } from '../config/annotationConfig';
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
      if (isLineLabel(labelId)) {
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
    .map((annotation) => {
      const { labelId, x, y, width, height } = annotation;
      const centerX = normaliseValue(x + width / 2);
      const centerY = normaliseValue(y + height / 2);
      const normalisedWidth = normaliseValue(width);
      const normalisedHeight = normaliseValue(height);

      return `${labelId} ${centerX} ${centerY} ${normalisedWidth} ${normalisedHeight}`;
    })
    .join('\n');
};

const serialiseLinesToWallText = (lines) => {
  if (!Array.isArray(lines)) {
    return '';
  }

  return lines
    .filter((annotation) => annotation.type === 'line')
    .map((annotation) => {
      const { x1, y1, x2, y2 } = annotation;
      return `${normaliseValue(x1)} ${normaliseValue(y1)} ${normaliseValue(x2)} ${normaliseValue(y2)}`;
    })
    .join('\n');
};

export const uploadFloorPlan = async (file) => {
  await delay(2000);

  const [yoloResponse, wallResponse] = await Promise.all([fetch(yoloTxtPath), fetch(wallTxtPath)]);
  const [yoloText, wallText] = await Promise.all([yoloResponse.text(), wallResponse.text()]);

  const boxes = parseYoloBoxes(yoloText);
  const rawLines = parseWallLines(wallText);
  const lines = subtractBoxesFromLines(rawLines, boxes);
  const imageUrl = await readFileAsDataUrl(file);

  return {
    fileName: file.name,
    imageUrl,
    boxes,
    lines,
    rawYoloText: yoloText,
    rawWallText: wallText,
  };
};

export const saveAnnotations = async ({ boxes, lines }) => {
  await delay(2000);

  const savedYoloText = serialiseBoxesToYoloText(boxes);
  const savedWallText = serialiseLinesToWallText(lines);

  return {
    savedYoloText,
    savedWallText,
  };
};
