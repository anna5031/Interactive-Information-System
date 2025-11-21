import api from './client.jsx';

export const fetchStepThreeStatuses = async () => {
  const response = await api.get('/api/step-three');
  return response.data;
};

export const fetchStepThreeResultById = async (stepOneId) => {
  if (!stepOneId) {
    return null;
  }
  const response = await api.get(`/api/step-three/${stepOneId}`);
  return response.data;
};

export const saveStepThree = async ({ stepOneId, requestId, floorLabel, floorValue, rooms, doors }) => {
  const response = await api.put(`/api/step-three/${stepOneId}`, {
    requestId: requestId ?? null,
    floorLabel: floorLabel ?? null,
    floorValue: floorValue ?? null,
    rooms: rooms ?? [],
    doors: doors ?? [],
  });
  return response.data;
};
