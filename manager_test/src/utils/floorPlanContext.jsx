import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import PropTypes from 'prop-types';
import { generateIdWithPrefix } from './id';

const STORAGE_KEY = 'floor_plan_workflow_state';

const defaultState = {
  stage: 'upload',
  fileName: '',
  imageUrl: null,
  boxes: [],
  lines: [],
  points: [],
  rawYoloText: '',
  rawWallText: '',
  rawDoorText: '',
  stepOneResult: null,
  stepOneOriginalId: null,
  skipUploadRedirect: false,
};

const mergeWithDefaults = (partial) => ({
  ...defaultState,
  ...partial,
});

const FloorPlanContext = createContext();

const readStoredState = () => {
  if (typeof window === 'undefined') {
    return defaultState;
  }

  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return defaultState;
    }
    const parsed = JSON.parse(stored);

    if (!parsed.stepOneResult && (parsed.savedYoloText || parsed.savedWallText || parsed.savedDoorText)) {
      const legacyId = generateIdWithPrefix('legacy_step_one');
      parsed.stepOneResult = {
        id: legacyId,
        fileName: `${legacyId}.json`,
        filePath: null,
        createdAt: new Date().toISOString(),
        yolo: {
          text: parsed.savedYoloText || '',
          raw: parsed.rawYoloText || '',
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
        },
      };
      delete parsed.savedYoloText;
      delete parsed.savedWallText;
      delete parsed.savedDoorText;
    }

    return mergeWithDefaults(parsed);
  } catch (error) {
    console.error('Failed to parse stored workflow state', error);
    return defaultState;
  }
};

export const FloorPlanProvider = ({ children }) => {
  const [state, setState] = useState(readStoredState);

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

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [state]);

  const setStage = (stage, options = {}) => {
    setState((prev) => ({
      ...prev,
      stage,
      skipUploadRedirect: options.skipUploadRedirect ?? false,
    }));
  };

  const setUploadData = ({ fileName, imageUrl, boxes, lines, points, rawYoloText, rawWallText, rawDoorText }) => {
    setState(
      mergeWithDefaults({
        stage: 'editor',
        fileName,
        imageUrl,
        boxes,
        lines,
        points,
        rawYoloText,
        rawWallText,
        rawDoorText,
        stepOneResult: null,
        stepOneOriginalId: null,
        skipUploadRedirect: false,
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

  const updatePoints = (points) => {
    setState((prev) => ({
      ...prev,
      points,
    }));
  };

  const setStepOneResult = (result) => {
    setState((prev) => ({
      ...prev,
      stepOneResult: result,
      stepOneOriginalId: null,
    }));
  };

  const loadStepOneResultForEdit = (result) => {
    if (!result) {
      return;
    }

    const { metadata, yolo, wall, door } = result;

    setState(
      mergeWithDefaults({
        stage: 'editor',
        fileName: metadata?.fileName || result.fileName || 'step-one-result.json',
        imageUrl: metadata?.imageUrl || null,
        boxes: Array.isArray(yolo?.boxes) ? yolo.boxes : [],
        lines: Array.isArray(wall?.lines) ? wall.lines : [],
        points: Array.isArray(door?.points) ? door.points : [],
        rawYoloText: yolo?.raw || '',
        rawWallText: wall?.raw || '',
        rawDoorText: door?.raw || '',
        stepOneResult: result,
        stepOneOriginalId: result.id || null,
        skipUploadRedirect: false,
      })
    );
  };

  const resetWorkflow = ({ skipUploadRedirect = false } = {}) => {
    const nextState = mergeWithDefaults({
      stage: 'upload',
      skipUploadRedirect,
    });

    setState(nextState);

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
      loadStepOneResultForEdit,
      resetWorkflow,
    }),
    [state]
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
