const DEFAULT_SCALE_LINE = Object.freeze({
  x1: 0.25,
  y1: 0.9,
  x2: 0.75,
  y2: 0.9,
});

const clampUnit = (value) => {
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

export const createDefaultScaleLine = () => ({ ...DEFAULT_SCALE_LINE });

export const sanitizeScaleLine = (line, { enforceValid = false } = {}) => {
  if (!line || typeof line !== 'object') {
    return enforceValid ? null : createDefaultScaleLine();
  }
  const coords = [line.x1, line.y1, line.x2, line.y2].map((value) => {
    if (value == null) {
      return null;
    }
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return null;
    }
    return clampUnit(numeric);
  });
  if (coords.some((value) => value == null)) {
    return enforceValid ? null : createDefaultScaleLine();
  }
  let [x1, y1, x2, y2] = coords;
  if (x1 === x2 && y1 === y2) {
    if (enforceValid) {
      return null;
    }
    x2 = clampUnit(x2 + 0.1);
  }
  return { x1, y1, x2, y2 };
};

export const parseLengthInput = (value) => {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const numeric = Number.parseFloat(trimmed);
    return Number.isFinite(numeric) ? numeric : null;
  }
  return null;
};

export const computeMetersPerPixel = (line, lengthMeters, width, height) => {
  if (!line || !Number.isFinite(lengthMeters) || lengthMeters <= 0) {
    return null;
  }
  if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
    return null;
  }
  const dx = (line.x2 - line.x1) * width;
  const dy = (line.y2 - line.y1) * height;
  const pixelLength = Math.hypot(dx, dy);
  if (!Number.isFinite(pixelLength) || pixelLength <= 0) {
    return null;
  }
  return lengthMeters / pixelLength;
};

export const deriveMetersPerPixel = (line, lengthInput, width, height) => {
  const numericLength = parseLengthInput(lengthInput);
  if (!Number.isFinite(numericLength) || numericLength <= 0) {
    return null;
  }
  return computeMetersPerPixel(line, numericLength, width, height);
};
