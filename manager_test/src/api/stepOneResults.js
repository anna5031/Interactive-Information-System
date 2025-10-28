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

const buildRecord = (payload) => {
  const { sourceOriginalId, ...rest } = payload || {};
  const id = sourceOriginalId || generateStepOneId();
  const fileName = `${id}.json`;
  return {
    id,
    fileName,
    filePath: `src/dummy/step_one_result/${fileName}`,
    createdAt: new Date().toISOString(),
    ...rest,
  };
};

export const saveStepOneResult = async (payload) => {
  const record = buildRecord(payload);
  const existing = readLocalStorage();
  const next = existing.filter((item) => item.id !== record.id).concat(record);
  writeLocalStorage(next);
  return record;
};

export const getStoredStepOneResults = () => readLocalStorage();

export const getStoredStepOneResultById = (id) => readLocalStorage().find((item) => item.id === id) ?? null;
