import { createContext, useContext, useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { setUserForApiRequests } from '../api/client';
import { userLogin } from '../api/auth';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children, initialLoading = true }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(initialLoading);

  const tokenKey = 'auth_token';
  const userDataKey = 'user_data';

  useEffect(() => {
    // 페이지 로드 시 기존 로그인 상태 확인
    const token = localStorage.getItem(tokenKey);
    const userData = localStorage.getItem(userDataKey);

    if (token && userData) {
      try {
        const parsedUserData = JSON.parse(userData);
        setUser(parsedUserData);
        setUserForApiRequests(parsedUserData);
      } catch (error) {
        console.error('Failed to parse user data:', error);
        localStorage.removeItem(tokenKey);
        localStorage.removeItem(userDataKey);
      }
    }
    setLoading(false);
  }, [tokenKey, userDataKey]);

  const login = async (id, name, email, token = 'local_token', extra = {}) => {
    const userData = {
      id,
      name,
      email,
      // Provide fallback school metadata for pages that expect them.
      school_id: extra.school_id ?? id,
      school_name: extra.school_name ?? name ?? 'Demo School',
    };

    try {
      // 백엔드에 로그인 정보 전송
      await userLogin({ id, name, email });
    } catch (error) {
      console.error('Backend login failed:', error);
    } finally {
      // 백엔드 로그인 실패시에도 프론트엔드에서는 로그인 상태 유지
      setUser(userData);
      setUserForApiRequests(userData);
      localStorage.setItem(tokenKey, token);
      localStorage.setItem(userDataKey, JSON.stringify(userData));
    }
  };

  const logout = () => {
    setUser(null);
    setUserForApiRequests(null);
    localStorage.removeItem(tokenKey);
    localStorage.removeItem(userDataKey);
  };

  const value = {
    user,
    login,
    logout,
    loading,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

AuthProvider.propTypes = {
  children: PropTypes.node.isRequired,
  initialLoading: PropTypes.bool,
};
