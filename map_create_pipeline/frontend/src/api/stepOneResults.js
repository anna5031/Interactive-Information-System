import { generateStepOneId } from '../utils/id';

const STORAGE_KEY = 'dummy_step_one_results';

const readLocalStorage = () => {
  if (typeof window === 'undefined' || !window.localStorage) {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed;
    }
    if (parsed && typeof parsed === 'object') {
      return Object.values(parsed);
    }
  } catch (error) {
    console.warn('Failed to parse step one results', error);
  }
  return [];
};

const writeLocalStorage = (records) => {
  if (typeof window === 'undefined' || !window.localStorage) {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch (error) {
    console.warn('Failed to write step one results', error);
  }
};

const mergeWallData = (previous = {}, current = {}) => {
  const merged = {
    ...previous,
    ...current,
  };
  if ((current?.floorLabel == null || current?.floorLabel === '') && previous?.floorLabel) {
    merged.floorLabel = previous.floorLabel;
  }
  if ((current?.floorValue == null || current?.floorValue === '') && previous?.floorValue) {
    merged.floorValue = previous.floorValue;
  }
  const hasNextBaseLines = Array.isArray(current.baseLines) && current.baseLines.length > 0;
  if (!hasNextBaseLines && Array.isArray(previous.baseLines) && previous.baseLines.length > 0) {
    merged.baseLines = previous.baseLines;
  }
  if (!merged.baseText) {
    merged.baseText = current.baseText || previous.baseText || current.text || previous.text || '';
  }
  if ((merged.filter === undefined || merged.filter === null) && (previous.filter !== undefined && previous.filter !== null)) {
    merged.filter = previous.filter;
  }
  if (!Array.isArray(merged.baseLines)) {
    merged.baseLines = [];
  }
  return merged;
};

const mergeRecords = (previous, current) => {
  if (!previous) {
    return current;
  }

  const merged = {
    ...previous,
    ...current,
  };

  if (previous.metadata || current.metadata) {
    merged.metadata = {
      ...(previous?.metadata || {}),
      ...(current?.metadata || {}),
    };
  }
  if (previous.objectDetection || current.objectDetection) {
    merged.objectDetection = {
      ...(previous?.objectDetection || {}),
      ...(current?.objectDetection || {}),
    };
  }
  if (previous.door || current.door) {
    merged.door = {
      ...(previous?.door || {}),
      ...(current?.door || {}),
    };
  }
  merged.wall = mergeWallData(previous?.wall || {}, current?.wall || {});

  const requestHistorySet = new Set();
  const appendHistory = (record) => {
    if (!record) {
      return;
    }
    if (Array.isArray(record.requestHistory)) {
      record.requestHistory.forEach((entry) => {
        if (entry) {
          requestHistorySet.add(entry);
        }
      });
    }
    if (record.requestId) {
      requestHistorySet.add(record.requestId);
    }
    if (record.processingResult?.request_id) {
      requestHistorySet.add(record.processingResult.request_id);
    }
  };

  appendHistory(previous);
  appendHistory(current);

  merged.requestId = current?.requestId ?? current?.processingResult?.request_id ?? previous?.requestId ?? previous?.processingResult?.request_id ?? null;
  merged.requestHistory = Array.from(requestHistorySet);

  return merged;
};

const buildRecord = (payload) => {
  const {
    sourceOriginalId,
    createdAt,
    fileName: providedFileName,
    filePath: providedFilePath,
    requestId,
    requestHistory,
    ...rest
  } = payload || {};
  const id = sourceOriginalId || generateStepOneId();
  const fileName = providedFileName || `${id}.json`;
  const filePath = providedFilePath || `src/dummy/step_one_result/${fileName}`;
  const historySet = new Set();
  if (Array.isArray(requestHistory)) {
    requestHistory.forEach((entry) => {
      if (entry) {
        historySet.add(entry);
      }
    });
  }
  if (requestId) {
    historySet.add(requestId);
  }
  return {
    id,
    fileName,
    filePath,
    createdAt: createdAt ?? new Date().toISOString(),
    requestId: requestId ?? null,
    requestHistory: Array.from(historySet),
    ...rest,
  };
};

export const saveStepOneResult = async (payload) => {
  const record = buildRecord(payload);
  const existing = readLocalStorage();
  const previous = existing.find((item) => item.id === record.id);
  const mergedRecord = mergeRecords(previous, record);
  const next = existing.filter((item) => item.id !== mergedRecord.id).concat(mergedRecord);
  writeLocalStorage(next);
  return mergedRecord;
};

export const deleteStepOneResult = (id) => {
  if (!id) {
    return;
  }
  const existing = readLocalStorage();
  const next = existing.filter((item) => item.id !== id);
  if (next.length === existing.length) {
    return;
  }
  writeLocalStorage(next);
};

export const getStoredStepOneResults = () => readLocalStorage();

export const getStoredStepOneResultById = (id) => readLocalStorage().find((item) => item.id === id) ?? null;
