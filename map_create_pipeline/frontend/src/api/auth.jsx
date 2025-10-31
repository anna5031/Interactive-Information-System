import api from './client';

// 사용자 로그인
export const userLogin = async ({ id, name, email }) => {
  const response = await api.post('/users/login', {
    user_id: id,
    name,
    email,
  });
  return response.data;
};

// 사용자 프로필 조회
export const getUserProfile = async () => {
  try {
    const response = await api.get('/users/profile');
    return response.data;
  } catch (error) {
    console.error('Failed to get user profile:', error);
    throw error;
  }
};
