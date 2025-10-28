const randomPart = () => Math.random().toString(36).slice(2, 10);

export const generateIdWithPrefix = (prefix) => {
  const base = typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${randomPart()}`;
  return prefix ? `${prefix}_${base}` : base;
};

export const generateStepOneId = () => generateIdWithPrefix('step_one');

export const generateStepTwoId = () => generateIdWithPrefix('step_two');

