const clamp = (value, min, max) => {
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

export const DEFAULT_WALL_FILTER = Object.freeze({
  enabled: true,
  percentile: 18,
  minPixels: 15,
  minSegments: 8,
});

export const createDefaultWallFilter = () => ({ ...DEFAULT_WALL_FILTER });

export const normalizeWallFilter = (config) => {
  const base = createDefaultWallFilter();
  const merged = config ? { ...base, ...config } : base;
  return {
    enabled: Boolean(merged.enabled),
    percentile:
      merged.percentile === null || merged.percentile === undefined
        ? null
        : clamp(Number.parseFloat(merged.percentile), 0, 100),
    minPixels: Number.isFinite(merged.minPixels) ? Math.max(0, Number(merged.minPixels)) : base.minPixels,
    minSegments: Number.isFinite(merged.minSegments) ? Math.max(1, Math.floor(merged.minSegments)) : base.minSegments,
  };
};

const computePercentile = (values, percentile) => {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }
  if (!Number.isFinite(percentile)) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 1) {
    return sorted[0];
  }
  const rank = ((percentile / 100) * (sorted.length - 1));
  const lowerIndex = Math.floor(rank);
  const upperIndex = Math.min(sorted.length - 1, Math.ceil(rank));
  if (lowerIndex === upperIndex) {
    return sorted[lowerIndex];
  }
  const weight = rank - lowerIndex;
  return sorted[lowerIndex] * (1 - weight) + sorted[upperIndex] * weight;
};

const resolveDimensions = (width, height) => {
  const resolvedWidth = Number.isFinite(width) && width > 0 ? width : null;
  const resolvedHeight = Number.isFinite(height) && height > 0 ? height : null;
  return {
    width: resolvedWidth,
    height: resolvedHeight,
    hasValidDimensions: Number.isFinite(resolvedWidth) && Number.isFinite(resolvedHeight),
  };
};

const computeLineLengthPixels = (line, width, height) => {
  if (!line) {
    return Number.NaN;
  }
  const { x1, y1, x2, y2 } = line;
  const dx = Number(x2) - Number(x1);
  const dy = Number(y2) - Number(y1);
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
    return Number.NaN;
  }
  const scaledDx = dx * width;
  const scaledDy = dy * height;
  if (!Number.isFinite(scaledDx) || !Number.isFinite(scaledDy)) {
    return Number.NaN;
  }
  return Math.hypot(scaledDx, scaledDy);
};

const restoreProtectedLines = (filteredLines, sourceLines, protectedLineIds) => {
  const result = Array.isArray(filteredLines) ? filteredLines : [];
  if (!(protectedLineIds instanceof Set) || protectedLineIds.size === 0) {
    return { lines: result, protectedCount: 0 };
  }

  const existingIds = new Set(result.map((line) => line?.id).filter(Boolean));
  let protectedCount = 0;

  sourceLines.forEach((line) => {
    const id = line?.id;
    if (!id || existingIds.has(id) || !protectedLineIds.has(id)) {
      return;
    }
    result.push(line);
    existingIds.add(id);
    protectedCount += 1;
  });

  return { lines: result, protectedCount };
};

export const filterWallLinesByLength = (
  lines,
  filterConfig,
  imageWidth,
  imageHeight,
  options = {}
) => {
  const normalized = normalizeWallFilter(filterConfig);
  const source = Array.isArray(lines) ? lines : [];
  const total = source.length;
  const protectedLineIds =
    options?.protectedLineIds instanceof Set
      ? options.protectedLineIds
      : Array.isArray(options?.protectedLineIds)
        ? new Set(options.protectedLineIds)
        : new Set();

  const { width, height, hasValidDimensions } = resolveDimensions(imageWidth, imageHeight);
  const widthScale = hasValidDimensions ? width : 1;
  const heightScale = hasValidDimensions ? height : 1;
  const guardThreshold =
    hasValidDimensions && Number.isFinite(normalized.minPixels) && normalized.minPixels > 0 ? normalized.minPixels : null;

  if (!normalized.enabled || total === 0) {
    return {
      lines: source,
      stats: {
        total,
        kept: total,
        removed: 0,
        percentile: normalized.percentile,
        percentileValuePx: null,
        guardLimitPx: guardThreshold,
        thresholdPx: null,
        sampleSize: 0,
        minSegments: normalized.minSegments,
        applied: false,
        reason: normalized.enabled ? 'NOT_READY' : 'DISABLED',
        protectedCount: 0,
        hasValidDimensions,
      },
    };
  }

  const lengths = source.map((line) => computeLineLengthPixels(line, widthScale, heightScale));
  const finiteLengths = lengths.filter((value) => Number.isFinite(value) && value >= 0);

  if (normalized.percentile === 0) {
    return {
      lines: source,
      stats: {
        total,
        kept: total,
        removed: 0,
        percentile: normalized.percentile,
        percentileValuePx: null,
        guardLimitPx: guardThreshold,
        thresholdPx: null,
        sampleSize: finiteLengths.length,
        minSegments: normalized.minSegments,
        applied: false,
        reason: 'ZERO_PERCENTILE',
        protectedCount: 0,
        hasValidDimensions,
      },
    };
  }

  const percentileReady = normalized.percentile !== null && finiteLengths.length >= normalized.minSegments;

  if (!percentileReady) {
    return {
      lines: source,
      stats: {
        total,
        kept: total,
        removed: 0,
        percentile: normalized.percentile,
        percentileValuePx: null,
        guardLimitPx: guardThreshold,
        thresholdPx: null,
        sampleSize: finiteLengths.length,
        minSegments: normalized.minSegments,
        applied: false,
        reason: normalized.percentile === null ? 'NO_PERCENTILE' : 'NOT_ENOUGH_SEGMENTS',
        protectedCount: 0,
        hasValidDimensions,
      },
    };
  }

  const percentileValue = computePercentile(finiteLengths, normalized.percentile);
  const baseThreshold = Number.isFinite(percentileValue) ? percentileValue : null;

  if (normalized.percentile >= 100) {
    const { lines: protectedLinesOnly, protectedCount } = restoreProtectedLines([], source, protectedLineIds);
    const kept = protectedLinesOnly.length;
    const removed = Math.max(0, total - kept);
    const reason = protectedCount > 0 ? 'PERCENTILE_100_PROTECTED' : 'PERCENTILE_100';
    return {
      lines: protectedLinesOnly,
      stats: {
        total,
        kept,
        removed,
        percentile: normalized.percentile,
        percentileValuePx: percentileValue ?? null,
        guardLimitPx: guardThreshold,
        thresholdPx: Number.POSITIVE_INFINITY,
        sampleSize: finiteLengths.length,
        minSegments: normalized.minSegments,
        applied: removed > 0 || protectedCount > 0,
        reason,
        protectedCount,
        hasValidDimensions,
      },
    };
  }

  const thresholdPx =
    baseThreshold === null
      ? null
      : guardThreshold === null
        ? baseThreshold
        : Math.min(baseThreshold, guardThreshold);

  if (!Number.isFinite(thresholdPx) || thresholdPx <= 0) {
    return {
      lines: source,
      stats: {
        total,
        kept: total,
        removed: 0,
        percentile: normalized.percentile,
        percentileValuePx: percentileValue ?? null,
        guardLimitPx: guardThreshold,
        thresholdPx: thresholdPx ?? null,
        sampleSize: finiteLengths.length,
        minSegments: normalized.minSegments,
        applied: false,
        reason: 'NON_POSITIVE_THRESHOLD',
        protectedCount: 0,
        hasValidDimensions,
      },
    };
  }

  const filtered = source.filter((line, index) => {
    const length = lengths[index];
    if (!Number.isFinite(length)) {
      return true;
    }
    return length >= thresholdPx;
  });
  const { lines: filteredWithProtected, protectedCount } = restoreProtectedLines(filtered, source, protectedLineIds);

  const kept = filteredWithProtected.length;
  const removed = Math.max(0, total - kept);
  const reason =
    protectedCount > 0
      ? 'PROTECTED_LINES'
      : removed > 0
        ? 'APPLIED'
        : 'THRESHOLD_NO_EFFECT';

  return {
    lines: filteredWithProtected,
    stats: {
      total,
      kept,
      removed,
      percentile: normalized.percentile,
      percentileValuePx: percentileValue ?? null,
      guardLimitPx: guardThreshold,
      thresholdPx,
      sampleSize: finiteLengths.length,
      minSegments: normalized.minSegments,
      applied: removed > 0 || protectedCount > 0,
      reason,
      protectedCount,
      hasValidDimensions,
    },
  };
};
