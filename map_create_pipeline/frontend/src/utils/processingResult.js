import api from '../api/client.jsx';
import { parseDoorPoints, parseWallLines, parseDetectionBoxes } from '../api/floorPlans';
import { computeMetersPerPixel, parseLengthInput, sanitizeScaleLine } from './scaleReference';

const normaliseImageSize = (imageSize = {}) => {
  const width = Number.isFinite(Number(imageSize.width)) ? Number(imageSize.width) : 0;
  const height = Number.isFinite(Number(imageSize.height)) ? Number(imageSize.height) : 0;
  return { width, height };
};

const resolveImageUrl = (value) => {
  if (!value) {
    return null;
  }
  if (typeof value !== 'string') {
    return null;
  }
  if (value.startsWith('data:') || /^https?:\/\//i.test(value)) {
    return value;
  }
  const baseUrl = api?.defaults?.baseURL;
  if (!baseUrl) {
    return value;
  }
  const normalisedBase = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  const normalisedPath = value.startsWith('/') ? value : `/${value}`;
  return `${normalisedBase}${normalisedPath}`;
};

const computeMetersPerPixelFromScaleReference = (scaleReference, imageSize) => {
  if (!scaleReference || typeof scaleReference !== 'object') {
    return null;
  }
  const sanitizedLine = sanitizeScaleLine(scaleReference, { enforceValid: true });
  if (!sanitizedLine) {
    return null;
  }
  const lengthMeters =
    parseLengthInput(scaleReference.lengthMeters ?? scaleReference.length_meters ?? scaleReference.length) ?? null;
  if (!Number.isFinite(lengthMeters) || lengthMeters <= 0) {
    return null;
  }
  const width = Number.isFinite(Number(imageSize?.width)) ? Number(imageSize.width) : null;
  const height = Number.isFinite(Number(imageSize?.height)) ? Number(imageSize.height) : null;
  if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
    return null;
  }
  return computeMetersPerPixel(sanitizedLine, lengthMeters, width, height);
};

const buildStepOneRecord = ({
  stepOneId,
  requestId,
  createdAt,
  imageSize,
  classNames,
  sourceImagePath,
  imageUrl,
  imageDataUrl,
  floorLabel,
  floorValue,
  metersPerPixel,
  scaleReference,
  objectDetectionText,
  wallText,
  wallBaseLines,
  wallBaseText,
  wallFilter,
  doorText,
  processingResult,
}) => {
  const safeStepOneId = stepOneId || (requestId ? `step_one_${requestId}` : `step_one_${Date.now()}`);
  const sanitisedImageSize = normaliseImageSize(imageSize);
  const resolvedScaleReference = scaleReference ?? null;
  const derivedMetersPerPixel = computeMetersPerPixelFromScaleReference(resolvedScaleReference, sanitisedImageSize);
  const fallbackMetersPerPixel =
    Number.isFinite(metersPerPixel) && metersPerPixel > 0 ? metersPerPixel : null;
  const resolvedMetersPerPixel = derivedMetersPerPixel ?? fallbackMetersPerPixel;

  const boxes = parseDetectionBoxes(objectDetectionText ?? '');
  const lines = parseWallLines(wallText ?? '');
  const hasBaseLines = Array.isArray(wallBaseLines) && wallBaseLines.length > 0;
  const points = parseDoorPoints(doorText ?? '', boxes, lines);

  const resolvedProcessingResult =
    processingResult != null
      ? { ...processingResult }
      : requestId
        ? {
            request_id: requestId,
            created_at: createdAt ?? null,
            image_size: sanitisedImageSize,
            class_names: Array.isArray(classNames) ? [...classNames] : [],
          }
        : null;

  const resolvedFileName = sourceImagePath || requestId || `${safeStepOneId}.json`;
  const resolvedImageUrl = resolveImageUrl(imageUrl) ?? (imageDataUrl?.startsWith('data:') ? imageDataUrl : null);
  const resolvedImageDataUrl =
    imageDataUrl && imageDataUrl.startsWith('data:')
      ? imageDataUrl
      : resolvedImageUrl?.startsWith('data:')
        ? resolvedImageUrl
        : null;

  const wallPayload = {
    raw: wallText ?? '',
    text: wallText ?? '',
    lines,
  };
  if (hasBaseLines) {
    wallPayload.baseLines = wallBaseLines;
    wallPayload.baseText = wallBaseText ?? wallText ?? '';
  } else if (wallBaseText) {
    wallPayload.baseText = wallBaseText;
  }
  if (wallFilter !== undefined) {
    wallPayload.filter = wallFilter;
  }

  return {
    id: safeStepOneId,
    fileName: resolvedFileName,
    filePath: `src/dummy/step_one_result/${safeStepOneId}.json`,
    createdAt: createdAt ?? new Date().toISOString(),
    floorLabel: floorLabel ?? null,
    floorValue: floorValue ?? null,
    metersPerPixel: resolvedMetersPerPixel,
    scaleReference: resolvedScaleReference,
    metadata: {
      fileName: resolvedFileName,
      imageUrl: resolvedImageUrl ?? null,
      imageWidth: sanitisedImageSize.width,
      imageHeight: sanitisedImageSize.height,
      floorLabel: floorLabel ?? null,
      floorValue: floorValue ?? null,
      scaleReference: resolvedScaleReference,
    },
    imageUrl: resolvedImageUrl ?? null,
    imageDataUrl: resolvedImageDataUrl,
    requestId: requestId ?? null,
    objectDetection: {
      raw: objectDetectionText ?? '',
      text: objectDetectionText ?? '',
      boxes,
    },
    wall: wallPayload,
    door: {
      raw: doorText ?? '',
      text: doorText ?? '',
      points,
    },
    processingResult: resolvedProcessingResult,
  };
};

const buildProcessingResultStubFromStoredSummary = (stored) => {
  const requestId = stored?.requestId ?? null;
  if (!requestId) {
    return null;
  }
  const metadata = {
    graph_summary: stored?.graphSummary ?? null,
    image_url: stored?.imageUrl ?? null,
    image_size: stored?.imageSize ?? null,
    scale_reference: stored?.scaleReference ?? stored?.scale_reference ?? null,
    floor_label: stored?.floorLabel ?? null,
    floor_value: stored?.floorValue ?? null,
  };
  return {
    request_id: requestId,
    created_at: stored?.createdAt ?? null,
    image_size: stored?.imageSize ?? null,
    class_names: Array.isArray(stored?.classNames) ? [...stored.classNames] : [],
    metadata,
  };
};

export const buildStepOneRecordFromStoredResult = (stored) => {
  if (!stored) {
    return null;
  }
  const wallBaseText = stored?.wallBaseText ?? stored?.wall_base_text ?? null;
  let storedBaseLines = null;
  if (wallBaseText && typeof wallBaseText === 'string') {
    const parsedBaseLines = parseWallLines(wallBaseText);
    if (parsedBaseLines.length > 0) {
      storedBaseLines = parsedBaseLines;
    }
  }
  if (!storedBaseLines) {
    storedBaseLines =
      Array.isArray(stored?.wall?.baseLines) && stored.wall.baseLines.length > 0
        ? stored.wall.baseLines
        : Array.isArray(stored?.wallBaseLines) && stored.wallBaseLines.length > 0
          ? stored.wallBaseLines
          : null;
  }
  return buildStepOneRecord({
    stepOneId: stored.stepOneId,
    requestId: stored.requestId,
    createdAt: stored.createdAt,
    imageSize: stored.imageSize,
    classNames: stored.classNames,
    sourceImagePath: stored.sourceImagePath,
    imageUrl: stored.imageUrl,
    imageDataUrl: stored.imageDataUrl,
    floorLabel: stored.floorLabel ?? stored.floor_label ?? stored?.metadata?.floorLabel ?? null,
    floorValue: stored.floorValue ?? stored.floor_value ?? stored?.metadata?.floorValue ?? null,
    metersPerPixel:
      stored.metersPerPixel ??
      stored.meters_per_pixel ??
      stored?.metadata?.metersPerPixel ??
      stored?.metadata?.meters_per_pixel ??
      null,
    scaleReference:
      stored.scaleReference ??
      stored.scale_reference ??
      stored?.metadata?.scaleReference ??
      stored?.metadata?.scale_reference ??
      null,
    objectDetectionText: stored.objectDetectionText,
    wallText: stored.wallText,
    wallBaseText: wallBaseText ?? '',
    wallBaseLines: storedBaseLines,
    wallFilter: stored?.wall?.filter ?? stored.wallFilter,
    doorText: stored.doorText,
    processingResult: stored.processingResult ?? buildProcessingResultStubFromStoredSummary(stored),
  });
};

export const buildStepOneRecordFromProcessingData = (processingData, { stepOneId, sourceImagePath } = {}) => {
  if (!processingData) {
    return null;
  }
  const texts = processingData?.metadata?.texts ?? {};
  return buildStepOneRecord({
    stepOneId: stepOneId ?? null,
    requestId: processingData.request_id ?? null,
    createdAt: processingData.created_at ?? null,
    imageSize: processingData.image_size ?? null,
    classNames: processingData.class_names ?? null,
    sourceImagePath: sourceImagePath ?? processingData?.metadata?.source_image_path ?? null,
    imageUrl: processingData?.metadata?.image_url ?? null,
    imageDataUrl: processingData?.metadata?.image_data_url ?? processingData?.metadata?.imageDataUrl ?? null,
    floorLabel: processingData?.metadata?.floor_label ?? processingData?.metadata?.floorLabel ?? null,
    floorValue: processingData?.metadata?.floor_value ?? processingData?.metadata?.floorValue ?? null,
    metersPerPixel:
      processingData?.metadata?.meters_per_pixel ?? processingData?.metadata?.metersPerPixel ?? null,
    scaleReference:
      processingData?.metadata?.scale_reference ?? processingData?.metadata?.scaleReference ?? null,
    objectDetectionText: texts.object_detection ?? '',
    wallText: texts.wall ?? '',
    wallBaseText: texts.wall_base ?? '',
    doorText: texts.door ?? '',
    processingResult: processingData,
  });
};
