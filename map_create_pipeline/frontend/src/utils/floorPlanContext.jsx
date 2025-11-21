import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import PropTypes from 'prop-types';
import { generateIdWithPrefix } from './id';
import { createDefaultWallFilter } from './wallFilter';
import { getStoredStepOneResultById } from '../api/stepOneResults';
import { fetchProcessingResultById } from '../api/floorPlans';
import { buildStepOneRecordFromProcessingData } from './processingResult';
import { useAuth } from './authContext.jsx';
import {
  createDefaultScaleLine,
  deriveMetersPerPixel,
  parseLengthInput,
  sanitizeScaleLine,
} from './scaleReference';

const STORAGE_KEY = 'floor_plan_workflow_state';

const MAX_DATA_URL_LENGTH = 2 * 1024 * 1024; // ~2MB

const stripLargeDataUrl = (value) => {
  if (typeof value !== 'string') {
    return value;
  }
  if (value.startsWith('data:') && value.length > MAX_DATA_URL_LENGTH) {
    return null;
  }
  return value;
};

const sanitizeMetadataPayload = (metadata) => {
  if (!metadata || typeof metadata !== 'object') {
    return metadata ?? null;
  }
  const sanitized = { ...metadata };
  if (sanitized.preview) {
    delete sanitized.preview;
  }
  if (typeof sanitized.image_data_url === 'string') {
    sanitized.image_data_url = stripLargeDataUrl(sanitized.image_data_url);
  }
  if (typeof sanitized.imageDataUrl === 'string') {
    sanitized.imageDataUrl = stripLargeDataUrl(sanitized.imageDataUrl);
  }
  if (typeof sanitized.imageUrl === 'string') {
    sanitized.imageUrl = stripLargeDataUrl(sanitized.imageUrl);
  }
  return sanitized;
};

const sanitizeProcessingResultForStorage = (processingResult) => {
  if (!processingResult || typeof processingResult !== 'object') {
    return processingResult ?? null;
  }
  const sanitized = {
    ...processingResult,
  };
  if (processingResult.metadata) {
    sanitized.metadata = sanitizeMetadataPayload(processingResult.metadata);
  }
  return sanitized;
};

const sanitizeStepOneResultForStorage = (result) => {
  if (!result || typeof result !== 'object') {
    return result ?? null;
  }
  const sanitized = {
    ...result,
  };
  sanitized.metadata = sanitizeMetadataPayload(result.metadata);
  if (result.processingResult) {
    sanitized.processingResult = sanitizeProcessingResultForStorage(result.processingResult);
  }
  if (typeof sanitized.imageUrl === 'string') {
    sanitized.imageUrl = stripLargeDataUrl(sanitized.imageUrl);
  }
  if (typeof sanitized.imageDataUrl === 'string') {
    sanitized.imageDataUrl = stripLargeDataUrl(sanitized.imageDataUrl);
  }
  return sanitized;
};

const sanitizeWorkflowStateForStorage = (state) => ({
  ...state,
  imageUrl: stripLargeDataUrl(state.imageUrl),
  processingResult: sanitizeProcessingResultForStorage(state.processingResult),
  stepOneResult: sanitizeStepOneResultForStorage(state.stepOneResult),
  calibrationLine: sanitizeScaleLine(state.calibrationLine),
  calibrationLengthMeters: typeof state.calibrationLengthMeters === 'string' ? state.calibrationLengthMeters : '',
});

const createDefaultState = () => ({
  stage: 'upload',
  fileName: '',
  floorLabel: '',
  floorValue: '',
  metersPerPixel: null,
  calibrationLine: createDefaultScaleLine(),
  calibrationLengthMeters: '',
  imageUrl: null,
  imageWidth: 0,
  imageHeight: 0,
  boxes: [],
  lines: [],
  wallBaseLines: [],
  points: [],
  rawObjectDetectionText: '',
  rawWallText: '',
  rawDoorText: '',
  stepOneResult: null,
  stepOneOriginalId: null,
  processingResult: null,
  skipUploadRedirect: false,
  wallFilter: createDefaultWallFilter(),
});

const mergeWithDefaults = (partial) => {
  const base = createDefaultState();
  const resolvedFilter = partial?.wallFilter
    ? { ...createDefaultWallFilter(), ...partial.wallFilter }
    : base.wallFilter;
  return {
    ...base,
    ...partial,
    floorLabel: partial?.floorLabel ?? base.floorLabel,
    floorValue: partial?.floorValue ?? base.floorValue,
    metersPerPixel: partial?.metersPerPixel ?? base.metersPerPixel,
    calibrationLine: partial?.calibrationLine ? sanitizeScaleLine(partial.calibrationLine) : base.calibrationLine,
    calibrationLengthMeters: partial?.calibrationLengthMeters ?? base.calibrationLengthMeters,
    wallFilter: resolvedFilter,
  };
};

const FloorPlanContext = createContext();

const readStoredState = () => {
  if (typeof window === 'undefined') {
    return createDefaultState();
  }

  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return createDefaultState();
    }
    const parsed = JSON.parse(stored);

    if (!parsed.stepOneResult && (parsed.savedObjectDetectionText || parsed.savedWallText || parsed.savedDoorText)) {
      const legacyId = generateIdWithPrefix('legacy_step_one');
      parsed.stepOneResult = {
        id: legacyId,
        fileName: `${legacyId}.json`,
        filePath: null,
        createdAt: new Date().toISOString(),
        objectDetection: {
          text: parsed.savedObjectDetectionText || '',
          raw: parsed.rawObjectDetectionText || '',
          boxes: parsed.boxes || [],
        },
        wall: {
          text: parsed.savedWallText || '',
          raw: parsed.rawWallText || '',
          lines: parsed.lines || [],
        },
        door: {
          text: parsed.savedDoorText || '',
          raw: parsed.rawDoorText || '',
          points: parsed.points || [],
        },
        metadata: {
          fileName: parsed.fileName || '',
          imageUrl: parsed.imageUrl || null,
          imageWidth: parsed.imageWidth || 0,
          imageHeight: parsed.imageHeight || 0,
        },
        processingResult: parsed.processingResult ?? null,
      };
      delete parsed.savedObjectDetectionText;
      delete parsed.savedWallText;
      delete parsed.savedDoorText;
    }

    return mergeWithDefaults(parsed);
  } catch (error) {
    console.error('Failed to parse stored workflow state', error);
    return createDefaultState();
  }
};

export const FloorPlanProvider = ({ children }) => {
  const [state, setState] = useState(readStoredState);
  const { user } = useAuth();

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    if (
      state.stage === 'upload' &&
      !state.fileName &&
      !state.imageUrl &&
      (!state.boxes || state.boxes.length === 0) &&
      (!state.lines || state.lines.length === 0) &&
      (!state.points || state.points.length === 0)
    ) {
      window.localStorage.removeItem(STORAGE_KEY);
      return;
    }

    try {
      const safeState = sanitizeWorkflowStateForStorage(state);
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(safeState));
    } catch (storageError) {
      console.warn('Failed to persist workflow state', storageError);
    }
  }, [state]);

  const setStage = (stage, options = {}) => {
    setState((prev) => ({
      ...prev,
      stage,
      skipUploadRedirect: options.skipUploadRedirect ?? false,
    }));
  };

  const setUploadData = ({
    fileName,
    floorLabel,
    floorValue,
    metersPerPixel,
    imageUrl,
    imageWidth,
    imageHeight,
    boxes,
    lines,
    points,
    rawObjectDetectionText,
    rawWallText,
    rawDoorText,
  }) => {
    setState((prev) =>
      mergeWithDefaults({
        stage: 'editor',
        fileName,
        floorLabel: floorLabel ?? prev.floorLabel ?? '',
        floorValue: floorValue ?? prev.floorValue ?? '',
        metersPerPixel: metersPerPixel ?? null,
        calibrationLine: createDefaultScaleLine(),
        calibrationLengthMeters: '',
        imageUrl,
        imageWidth: imageWidth ?? 0,
        imageHeight: imageHeight ?? 0,
        boxes,
        lines,
        wallBaseLines: Array.isArray(lines) ? lines : [],
        points,
        rawObjectDetectionText,
        rawWallText,
        rawDoorText,
        stepOneResult: null,
        stepOneOriginalId: null,
        processingResult: null,
        skipUploadRedirect: false,
        wallFilter: prev.wallFilter ?? createDefaultWallFilter(),
      })
    );
  };

  const updateBoxes = (boxes) => {
    setState((prev) => ({
      ...prev,
      boxes,
    }));
  };

  const updateLines = (lines) => {
    setState((prev) => ({
      ...prev,
      lines,
    }));
  };

  const updateWallBaseLines = (lines) => {
    setState((prev) => ({
      ...prev,
      wallBaseLines: Array.isArray(lines) ? lines : [],
    }));
  };

  const updatePoints = (points) => {
    setState((prev) => ({
      ...prev,
      points,
    }));
  };

  const setCalibrationLine = (nextLine) => {
    setState((prev) => {
      const sanitizedLine = sanitizeScaleLine(nextLine);
      const metersPerPixel = deriveMetersPerPixel(
        sanitizedLine,
        prev.calibrationLengthMeters,
        prev.imageWidth,
        prev.imageHeight
      );
      return {
        ...prev,
        calibrationLine: sanitizedLine,
        metersPerPixel,
      };
    });
  };

  const setCalibrationLengthMeters = (value) => {
    setState((prev) => {
      const nextValue = typeof value === 'string' ? value : value == null ? '' : String(value);
      const metersPerPixel = deriveMetersPerPixel(prev.calibrationLine, nextValue, prev.imageWidth, prev.imageHeight);
      return {
        ...prev,
        calibrationLengthMeters: nextValue,
        metersPerPixel,
      };
    });
  };

  const setStepOneResult = (result, { processingResult } = {}) => {
    setState((prev) => ({
      ...prev,
      stepOneResult: result ?? null,
      stepOneOriginalId: result?.id ?? null,
      processingResult: processingResult ?? prev.processingResult ?? null,
    }));
  };

  const setProcessingResult = (result) => {
    setState((prev) => ({
      ...prev,
      processingResult: result ?? null,
      stepOneResult: prev.stepOneResult
        ? { ...prev.stepOneResult, processingResult: result ?? null }
        : prev.stepOneResult,
    }));
  };

  const loadStepOneResultForEdit = useCallback(
    async (result) => {
      if (!result) {
        return null;
      }

      let resolved = result;
      if (result?.id) {
        const stored = getStoredStepOneResultById(result.id);
        if (stored) {
          resolved = stored;
        }
      }

      const requestId =
        resolved?.processingResult?.request_id ??
        resolved?.processingResult?.requestId ??
        resolved?.requestId ??
        resolved?.id ??
        null;

      if (requestId) {
        try {
          const processingPayload = await fetchProcessingResultById(requestId);
          if (processingPayload) {
            const rebuilt = buildStepOneRecordFromProcessingData(processingPayload, {
              stepOneId: resolved?.id ?? resolved?.stepOneId ?? null,
              sourceImagePath: resolved?.sourceImagePath ?? null,
            });
            if (rebuilt) {
              resolved = {
                ...rebuilt,
                processingResult: processingPayload,
              };
            }
          }
        } catch (fetchError) {
          console.error('저장된 편집 데이터를 다시 불러오지 못했습니다.', fetchError);
        }
      }

      const { metadata, objectDetection, wall, door } = resolved;
      const fallbackServerImageUrl = requestId
        ? `/api/floorplans/${requestId}/image${user?.id ? `?userId=${encodeURIComponent(user.id)}` : ''}`
        : null;
      const resolvedImageUrl =
        metadata?.imageUrl ||
        resolved?.imageUrl ||
        resolved?.imageDataUrl ||
        resolved?.processingResult?.metadata?.image_url ||
        resolved?.processingResult?.metadata?.imageUrl ||
        fallbackServerImageUrl ||
        null;

      let resolvedScaleReference =
        metadata?.scaleReference || metadata?.scale_reference || resolved?.scaleReference || null;
      if (!resolvedScaleReference && resolved?.calibrationLine) {
        const fallbackLine = sanitizeScaleLine(resolved.calibrationLine);
        const fallbackLength = parseLengthInput(resolved?.calibrationLengthMeters);
        if (fallbackLine && Number.isFinite(fallbackLength) && fallbackLength > 0) {
          resolvedScaleReference = { ...fallbackLine, lengthMeters: fallbackLength };
        }
      }
      const calibrationLine = resolvedScaleReference ? sanitizeScaleLine(resolvedScaleReference) : createDefaultScaleLine();
      const calibrationLengthValue =
        resolvedScaleReference?.lengthMeters ?? resolvedScaleReference?.length_meters ?? null;
      const calibrationLengthMeters =
        calibrationLengthValue != null && Number.isFinite(Number(calibrationLengthValue))
          ? String(calibrationLengthValue)
          : '';
      const computedMetersPerPixel = deriveMetersPerPixel(
        calibrationLine,
        calibrationLengthMeters,
        metadata?.imageWidth || 0,
        metadata?.imageHeight || 0
      );
      const resolvedMetersPerPixel =
        computedMetersPerPixel ??
        resolved?.metersPerPixel ??
        metadata?.metersPerPixel ??
        metadata?.meters_per_pixel ??
        null;

      setState((prev) =>
        mergeWithDefaults({
          stage: 'editor',
          fileName: metadata?.fileName || resolved.fileName || 'step-one-result.json',
          floorLabel: metadata?.floorLabel || prev.floorLabel || '',
          floorValue: metadata?.floorValue || prev.floorValue || '',
          metersPerPixel: resolvedMetersPerPixel,
          calibrationLine,
          calibrationLengthMeters,
          imageUrl: resolvedImageUrl,
          imageWidth: metadata?.imageWidth || 0,
          imageHeight: metadata?.imageHeight || 0,
          boxes: Array.isArray(objectDetection?.boxes) ? objectDetection.boxes : [],
          lines: Array.isArray(wall?.lines) ? wall.lines : [],
          wallBaseLines: Array.isArray(wall?.baseLines)
            ? wall.baseLines
            : Array.isArray(wall?.lines)
              ? wall.lines
              : [],
          points: Array.isArray(door?.points) ? door.points : [],
          rawObjectDetectionText: objectDetection?.raw || '',
          rawWallText: wall?.raw || '',
          rawDoorText: door?.raw || '',
          stepOneResult: resolved,
          stepOneOriginalId: resolved.id || null,
          processingResult: resolved.processingResult ?? null,
          skipUploadRedirect: false,
          wallFilter: wall?.filter ?? prev.wallFilter ?? createDefaultWallFilter(),
        })
      );

      return resolved;
    },
    [user?.id]
  );

  const setWallFilter = (updater) => {
    setState((prev) => {
      const previous = prev.wallFilter ?? createDefaultWallFilter();
      const nextValue = typeof updater === 'function' ? updater(previous) : updater;
      const merged = {
        ...createDefaultWallFilter(),
        ...previous,
        ...(nextValue || {}),
      };
      return {
        ...prev,
        wallFilter: merged,
      };
    });
  };

  const resetWorkflow = ({ skipUploadRedirect = false } = {}) => {
    setState((prev) =>
      mergeWithDefaults({
        stage: 'upload',
        floorLabel: '',
        floorValue: '',
        metersPerPixel: null,
        calibrationLine: createDefaultScaleLine(),
        calibrationLengthMeters: '',
        skipUploadRedirect,
        wallFilter: prev.wallFilter ?? createDefaultWallFilter(),
      })
    );

    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  };

  const value = useMemo(
    () => ({
      state,
      setStage,
      setUploadData,
      updateBoxes,
      updateLines,
      updatePoints,
      setStepOneResult,
      setProcessingResult,
      loadStepOneResultForEdit,
      resetWorkflow,
      wallFilter: state.wallFilter ?? createDefaultWallFilter(),
      setWallFilter,
      wallBaseLines: state.wallBaseLines ?? state.lines ?? [],
      updateWallBaseLines,
      setCalibrationLine,
      setCalibrationLengthMeters,
    }),
    [state, loadStepOneResultForEdit]
  );

  return <FloorPlanContext.Provider value={value}>{children}</FloorPlanContext.Provider>;
};

FloorPlanProvider.propTypes = {
  children: PropTypes.node.isRequired,
};

export const useFloorPlan = () => {
  const context = useContext(FloorPlanContext);
  if (!context) {
    throw new Error('useFloorPlan must be used within a FloorPlanProvider');
  }
  return context;
};
