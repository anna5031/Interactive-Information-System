const IDENTITY_MATRIX3D = 'matrix3d(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)';

function formatValue(value) {
  if (!Number.isFinite(value)) {
    return '0';
  }
  const rounded = Math.abs(value) < 1e-9 ? 0 : value;
  return rounded.toFixed(6);
}

export function homographyToCssMatrix(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 3) {
    return null;
  }

  const [row1, row2, row3] = matrix;
  if (![row1, row2, row3].every((row) => Array.isArray(row) && row.length === 3)) {
    return null;
  }

  const [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]] = matrix;

  const cssMatrix = [
    h11,
    h21,
    0,
    h31,
    h12,
    h22,
    0,
    h32,
    0,
    0,
    1,
    0,
    h13,
    h23,
    0,
    h33 || 1,
  ];

  return `matrix3d(${cssMatrix.map(formatValue).join(',')})`;
}

export { IDENTITY_MATRIX3D };
