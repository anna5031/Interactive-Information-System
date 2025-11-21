import { useState, useCallback } from 'react';
import { useAuth } from '../utils/authContext';
import api from '../api/client';
import styles from './FacilityUploadPage.module.css';
import Header from '../components/Header';
import Footer from '../components/Footer';

export default function FacilityUploadPage() {
  const { user } = useAuth();
  const [file, setFile] = useState(null);
  const [name, setName] = useState('');
  const [msg, setMsg] = useState('');
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer?.files?.[0];
    if (f) setFile(f);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return setMsg('파일을 선택하세요');
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('name', name);
      const schoolId = user?.school_id;
      if (!schoolId) return setMsg('관리자에 소속된 학교 ID가 없습니다.');
      const resp = await api.post(`/schools/${schoolId}/images`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMsg('업로드 성공: ' + (resp.data.filename || ''));
      setFile(null);
      setName('');
    } catch (err) {
      console.error(err);
      setMsg('업로드 실패');
    }
  };

  return (
    <div className={styles.container}>
      <Header />
      <main className={styles.body}>
        <div className={styles.contentWrap}>
          <h2 className={styles.breadcrumb}>{user?.school_name || '학교'} &gt; 본관 &gt; 화장실 사진 관리</h2>
          <ul className={styles.notes}>
            <li>이미지는 최대 10개 첨부 가능해요</li>
            <li>권한 및 개인정보에 주의해주세요</li>
          </ul>

          <div
            className={`${styles.dropzone} ${dragOver ? styles.dragOver : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <input type="file" accept="image/*" id="file-input" onChange={handleFileChange} className={styles.hiddenInput} />
            <label htmlFor="file-input" className={styles.dropLabel}>
              <div className={styles.uploadIcon}>⬆</div>
              <div>
                <span className={styles.clickHere}>Click here</span> to upload or drop media here
              </div>
            </label>
          </div>

          <form onSubmit={handleSubmit} className={styles.formArea}>
            <input type="text" placeholder="사진 설명(예: 3층 장애인 화장실)" value={name} onChange={(e) => setName(e.target.value)} className={styles.textInput} />
            <button type="submit" className={styles.uploadButton}>업로드하기</button>
          </form>

          {file && <div className={styles.preview}>선택된 파일: {file.name}</div>}
          {msg && <div className={styles.msg}>{msg}</div>}
        </div>
      </main>
      <Footer />
    </div>
  );
}
