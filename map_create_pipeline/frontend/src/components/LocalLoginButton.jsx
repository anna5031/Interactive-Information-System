import { useState, useCallback } from 'react';
import { useAuth } from '../utils/authContext';
import { registerBuilding } from '../api/buildings';
import styles from './LocalLoginButton.module.css';

const LocalLoginButton = () => {
  const { login } = useAuth();
  const [buildingName, setBuildingName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = useCallback(async () => {
    const trimmed = buildingName.trim();
    if (!trimmed) {
      setError('건물 이름을 입력해주세요.');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const result = await registerBuilding(trimmed);
      const buildingId = result?.building_id;
      if (!buildingId) {
        throw new Error('Missing building ID');
      }
      await login(buildingId, trimmed, `${buildingId}@local.building`, 'local_token', {
        school_id: buildingId,
        school_name: trimmed,
      });
    } catch (err) {
      console.error(err);
      setError('건물 폴더를 준비하는 동안 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.');
    } finally {
      setLoading(false);
    }
  }, [buildingName, login]);

  return (
    <div className={styles.container}>
      <div className={styles.formGroup}>
        <label htmlFor='building-name' className={styles.label}>
          사용할 건물 이름을 입력하세요
        </label>
        <input
          id='building-name'
          type='text'
          placeholder='예: 모두의 학교 본관'
          value={buildingName}
          onChange={(e) => setBuildingName(e.target.value)}
          className={styles.input}
          onKeyDown={(e) => e.key === 'Enter' && !loading && handleSubmit()}
          disabled={loading}
        />
        <p className={styles.helperText}>
          입력한 한글 건물 이름과 매칭되는 안전한 폴더 ID를 자동으로 만들어 data 폴더에 저장합니다.
        </p>
      </div>
      {error && <div className={styles.error}>{error}</div>}
      <button onClick={handleSubmit} className={styles.button} disabled={loading}>
        {loading ? '폴더 준비 중...' : '시작하기'}
      </button>
    </div>
  );
};

export default LocalLoginButton;
