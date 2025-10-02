import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import PropTypes from 'prop-types';

const STORAGE_KEY = 'floor_plan_workflow_state';

const defaultState = {
  stage: 'upload',
  fileName: '',
  imageUrl: null,
  boxes: [],
  lines: [],
  rawYoloText: '',
  rawWallText: '',
  savedYoloText: '',
  savedWallText: '',
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
      (!state.lines || state.lines.length === 0)
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

  const setUploadData = ({ fileName, imageUrl, boxes, lines, rawYoloText, rawWallText }) => {
    setState(
      mergeWithDefaults({
        stage: 'editor',
        fileName,
        imageUrl,
        boxes,
        lines,
        rawYoloText,
        rawWallText,
        savedYoloText: '',
        savedWallText: '',
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

  const setSavedTexts = ({ yolo, wall }) => {
    setState((prev) => ({
      ...prev,
      savedYoloText: yolo,
      savedWallText: wall,
    }));
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
      setSavedTexts,
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
