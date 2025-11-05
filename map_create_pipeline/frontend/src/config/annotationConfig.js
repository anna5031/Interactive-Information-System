const LABEL_CONFIG = [
  { id: '0', name: 'Door', color: '#ef4444', type: 'point' },
  { id: '1', name: 'Elevator', color: '#a855f7', type: 'box' },
  { id: '2', name: 'Room', color: '#22c55e', type: 'box' },
  { id: '3', name: 'Stair', color: '#0ea5e9', type: 'box' },
  { id: '4', name: 'Wall', color: '#ffbb00ff', type: 'line' },
];

export const getLabelById = (labelId) => LABEL_CONFIG.find((label) => label.id === labelId);

export const getDefaultLabelId = () => LABEL_CONFIG[0]?.id ?? '0';

export const isLineLabel = (labelId) => getLabelById(labelId)?.type === 'line';

export const isBoxLabel = (labelId) => getLabelById(labelId)?.type === 'box';

export const isPointLabel = (labelId) => getLabelById(labelId)?.type === 'point';

export default LABEL_CONFIG;
