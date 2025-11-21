import React from 'react';
import styles from './Footer.module.css';

export default function Footer() {
  return (
    <footer className={styles.bottombar}>
      <div className={styles.bottomLinks}>
        <div>프로젝트 소개</div>
        <div className={styles.sep} />
        <div>무의 바로가기</div>
      </div>
    </footer>
  );
}
