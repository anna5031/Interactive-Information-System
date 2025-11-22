import api from './client.jsx';

export const fetchStepThreeStatuses = async () => {
  const response = await api.get('/api/step-three');
  return response.data;
};

export const fetchStepThreeResultById = async (requestId) => {
  if (!requestId) {
    return null;
  }
  const response = await api.get(`/api/step-three/${requestId}`);
  return response.data;
};

export const saveStepThree = async ({ requestId, floorLabel, floorValue, rooms, doors }) => {
  const response = await api.put(`/api/step-three/${requestId}`, {
    requestId: requestId ?? null,
    floorLabel: floorLabel ?? null,
    floorValue: floorValue ?? null,
    rooms: rooms ?? [],
    doors: doors ?? [],
  });
  return response.data;
};
