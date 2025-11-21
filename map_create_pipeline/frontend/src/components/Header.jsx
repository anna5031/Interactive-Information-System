import React from 'react';
import PropTypes from 'prop-types';
import styles from './Header.module.css';

export default function Header({ logoText = '모두의 학교' }) {
  return (
    <header className={styles.topbar}>
      <div className={styles.container}>
        <div className={styles.logoArea}>
          <div className={styles.logoText}>{logoText}</div>
        </div>
      </div>
    </header>
  );
}

Header.propTypes = {
  logoText: PropTypes.string,
};
