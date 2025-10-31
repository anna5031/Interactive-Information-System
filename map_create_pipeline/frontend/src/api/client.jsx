import axios from 'axios';

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';
const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || DEFAULT_API_BASE_URL;

let currentUser = null;

export const setUserForApiRequests = (user) => {
  currentUser = user;
};

const api = axios.create({
  baseURL: apiBaseUrl,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  if (currentUser?.id) {
    config.headers['X-User-ID'] = currentUser.id;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API 에러:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export default api;
