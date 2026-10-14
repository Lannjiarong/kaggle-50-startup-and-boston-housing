// Test OLS solver in pure JavaScript
function solveOLS(X, y) {
  const n = X.length;
  const p = X[0].length; // includes intercept if X has 1s column
  
  // X^T * X
  const XtX = Array.from({ length: p }, () => Array(p).fill(0));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += X[k][i] * X[k][j];
      }
      XtX[i][j] = sum;
    }
  }

  // X^T * y
  const Xty = Array(p).fill(0);
  for (let i = 0; i < p; i++) {
    let sum = 0;
    for (let k = 0; k < n; k++) {
      sum += X[k][i] * y[k];
    }
    Xty[i] = sum;
  }

  // Invert XtX using Gauss-Jordan with partial pivoting and small ridge regularization for collinearity safety
  const A = XtX.map((row, i) => {
    const r = [...row];
    r[i] += 1e-9; // tiny ridge for stability
    return r;
  });
  const I = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => (i === j ? 1 : 0)));

  for (let i = 0; i < p; i++) {
    let maxRow = i;
    for (let k = i + 1; k < p; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
        maxRow = k;
      }
    }
    [A[i], A[maxRow]] = [A[maxRow], A[i]];
    [I[i], I[maxRow]] = [I[maxRow], I[i]];

    const pivot = A[i][i];
    if (Math.abs(pivot) < 1e-12) continue;

    for (let j = 0; j < p; j++) {
      A[i][j] /= pivot;
      I[i][j] /= pivot;
    }

    for (let k = 0; k < p; k++) {
      if (k !== i) {
        const factor = A[k][i];
        for (let j = 0; j < p; j++) {
          A[k][j] -= factor * A[i][j];
          I[k][j] -= factor * I[i][j];
        }
      }
    }
  }

  // Beta = (X^T X)^(-1) * X^T y
  const beta = Array(p).fill(0);
  for (let i = 0; i < p; i++) {
    let sum = 0;
    for (let j = 0; j < p; j++) {
      sum += I[i][j] * Xty[j];
    }
    beta[i] = sum;
  }

  // Predictions and metrics
  const yPred = Array(n).fill(0);
  const residuals = Array(n).fill(0);
  let rss = 0;
  let ySum = 0;
  for (let i = 0; i < n; i++) {
    let pred = 0;
    for (let j = 0; j < p; j++) {
      pred += X[i][j] * beta[j];
    }
    yPred[i] = pred;
    residuals[i] = y[i] - pred;
    rss += residuals[i] ** 2;
    ySum += y[i];
  }
  const yMean = ySum / n;
  let tss = 0;
  for (let i = 0; i < n; i++) {
    tss += (y[i] - yMean) ** 2;
  }

  const rmse = Math.sqrt(rss / n);
  const r2 = 1 - rss / tss;
  const kFeatures = p - 1;
  const adjR2 = 1 - (1 - r2) * (n - 1) / (n - kFeatures - 1);

  return { beta, rmse, r2, adjR2, yPred, residuals, invXtX: I };
}

// Test with dummy data
const X = [
  [1, 10, 2],
  [1, 20, 4],
  [1, 30, 5],
  [1, 40, 7],
  [1, 50, 8]
];
const y = [25, 45, 62, 85, 102];
const res = solveOLS(X, y);
console.log('Test OLS results:');
console.log('Beta:', res.beta);
console.log('RMSE:', res.rmse);
console.log('R2:', res.r2);
