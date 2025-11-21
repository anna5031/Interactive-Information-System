import api from './client';

export const registerBuilding = async (buildingName) => {
  const response = await api.post('/buildings/register', {
    building_name: buildingName,
  });
  return response.data;
};

export default { registerBuilding };
