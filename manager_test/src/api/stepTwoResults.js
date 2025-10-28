import { generateStepTwoId } from '../utils/id';

const STORAGE_KEY = 'dummy_step_two_results';

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
    console.warn('Failed to parse step two results', error);
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
    console.warn('Failed to persist step two results', error);
  }
};

const buildRecord = (payload) => {
  const id = generateStepTwoId();
  const fileName = `${id}.json`;
  return {
    id,
    fileName,
    filePath: `src/dummy/step_two_result/${fileName}`,
    createdAt: new Date().toISOString(),
    ...payload,
  };
};

export const saveStepTwoResult = async (payload) => {
  const record = buildRecord(payload);
  const existing = readLocalStorage();
  writeLocalStorage(existing.concat(record));
  return record;
};

export const getStoredStepTwoResults = () => readLocalStorage();

export const getStoredStepTwoResultById = (id) => readLocalStorage().find((item) => item.id === id) ?? null;

