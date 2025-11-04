import api from './client.jsx';

export const fetchStepTwoStatuses = async () => {
  const response = await api.get('/api/step-two');
  return response.data;
};

export const fetchStepTwoResultById = async (stepOneId) => {
  if (!stepOneId) {
    return null;
  }
  const response = await api.get(`/api/step-two/${stepOneId}`);
  return response.data;
};

export const saveStepTwo = async ({ stepOneId, requestId, rooms, doors }) => {
  const response = await api.put(`/api/step-two/${stepOneId}`, {
    requestId: requestId ?? null,
    rooms: rooms ?? [],
    doors: doors ?? [],
  });
  return response.data;
};
