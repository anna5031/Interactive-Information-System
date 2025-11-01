import api from '../api/client.jsx';
import { parseDoorPoints, parseWallLines, parseYoloBoxes } from '../api/floorPlans';

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

const buildStepOneRecord = ({
  stepOneId,
  requestId,
  createdAt,
  imageSize,
  classNames,
  sourceImagePath,
  imageUrl,
  imageDataUrl,
  yoloText,
  wallText,
  doorText,
  processingResult,
}) => {
  const safeStepOneId = stepOneId || (requestId ? `step_one_${requestId}` : `step_one_${Date.now()}`);
  const sanitisedImageSize = normaliseImageSize(imageSize);

  const boxes = parseYoloBoxes(yoloText ?? '');
  const lines = parseWallLines(wallText ?? '');
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
    imageDataUrl && imageDataUrl.startsWith('data:') ? imageDataUrl : resolvedImageUrl?.startsWith('data:') ? resolvedImageUrl : null;

  return {
    id: safeStepOneId,
    fileName: resolvedFileName,
    filePath: `src/dummy/step_one_result/${safeStepOneId}.json`,
    createdAt: createdAt ?? new Date().toISOString(),
    metadata: {
      fileName: resolvedFileName,
      imageUrl: resolvedImageUrl ?? null,
      imageWidth: sanitisedImageSize.width,
      imageHeight: sanitisedImageSize.height,
    },
    imageUrl: resolvedImageUrl ?? null,
    imageDataUrl: resolvedImageDataUrl,
    requestId: requestId ?? null,
    yolo: {
      raw: yoloText ?? '',
      text: yoloText ?? '',
      boxes,
    },
    wall: {
      raw: wallText ?? '',
      text: wallText ?? '',
      lines,
    },
    door: {
      raw: doorText ?? '',
      text: doorText ?? '',
      points,
    },
    processingResult: resolvedProcessingResult,
  };
};

export const buildStepOneRecordFromStoredResult = (stored) => {
  if (!stored) {
    return null;
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
    yoloText: stored.yoloText,
    wallText: stored.wallText,
    doorText: stored.doorText,
    processingResult: stored.processingResult,
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
    imageDataUrl:
      processingData?.metadata?.image_data_url ??
      processingData?.metadata?.imageDataUrl ??
      null,
    yoloText: texts.yolo ?? '',
    wallText: texts.wall ?? '',
    doorText: texts.door ?? '',
    processingResult: processingData,
  });
};
