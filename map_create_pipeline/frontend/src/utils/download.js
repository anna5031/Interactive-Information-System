const ensureJsonExtension = (name) => {
  if (!name) {
    return 'result.json';
  }
  const trimmed = name.trim();
  if (!trimmed) {
    return 'result.json';
  }
  return trimmed.toLowerCase().endsWith('.json') ? trimmed : `${trimmed}.json`;
};

export const downloadJson = (fileName, data, fallbackName = 'result.json') => {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    const resolved = ensureJsonExtension(fileName || fallbackName);
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = resolved;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.setTimeout(() => window.URL.revokeObjectURL(url), 0);
  } catch (error) {
    console.error('Failed to trigger JSON download', error);
  }
};

