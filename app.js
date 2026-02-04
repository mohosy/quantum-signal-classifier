const canvas = document.getElementById("playground");
const ctx = canvas.getContext("2d");

const classZeroBtn = document.getElementById("classZeroBtn");
const classOneBtn = document.getElementById("classOneBtn");
const trainBtn = document.getElementById("trainBtn");
const autoBtn = document.getElementById("autoBtn");
const seedBtn = document.getElementById("seedBtn");
const resetBtn = document.getElementById("resetBtn");
const learningRateInput = document.getElementById("learningRate");
const epochsInput = document.getElementById("epochsPerStep");
const learningRateValue = document.getElementById("learningRateValue");
const epochsValue = document.getElementById("epochsPerStepValue");

const pointCountEl = document.getElementById("pointCount");
const lossValueEl = document.getElementById("lossValue");
const accuracyValueEl = document.getElementById("accuracyValue");
const iterationValueEl = document.getElementById("iterationValue");

const WIDTH = canvas.width;
const HEIGHT = canvas.height;
const EPS = 1e-9;

let points = [];
let labelMode = 0;
let autoTrainId = null;
let iterationCount = 0;

const model = {
  w0: (Math.random() - 0.5) * 0.6,
  w1: (Math.random() - 0.5) * 0.6,
  b: (Math.random() - 0.5) * 0.4,
  loss: null,
  accuracy: null,
};

function toFeatureSpace(px, py) {
  return {
    x: (px / WIDTH) * 2 - 1,
    y: 1 - (py / HEIGHT) * 2,
  };
}

function toCanvasSpace(x, y) {
  return {
    x: ((x + 1) / 2) * WIDTH,
    y: ((1 - y) / 2) * HEIGHT,
  };
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function predictProbability(features) {
  const z = model.w0 * features.x + model.w1 * features.y + model.b;
  return sigmoid(z);
}

function updateStats() {
  pointCountEl.textContent = points.length;
  lossValueEl.textContent = Number.isFinite(model.loss) ? model.loss.toFixed(4) : "n/a";
  accuracyValueEl.textContent = Number.isFinite(model.accuracy) ? `${(model.accuracy * 100).toFixed(1)}%` : "n/a";
  iterationValueEl.textContent = iterationCount;
}

function drawHeatmap() {
  const cell = 12;
  for (let y = 0; y < HEIGHT; y += cell) {
    for (let x = 0; x < WIDTH; x += cell) {
      const features = toFeatureSpace(x + cell * 0.5, y + cell * 0.5);
      const p = predictProbability(features);
      const alpha = 0.18 + Math.abs(p - 0.5) * 0.3;
      if (p >= 0.5) {
        ctx.fillStyle = `rgba(255,95,162,${alpha})`;
      } else {
        ctx.fillStyle = `rgba(68,242,198,${alpha})`;
      }
      ctx.fillRect(x, y, cell, cell);
    }
  }
}

function drawAxes() {
  ctx.strokeStyle = "rgba(180,210,240,0.24)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(WIDTH / 2, 0);
  ctx.lineTo(WIDTH / 2, HEIGHT);
  ctx.moveTo(0, HEIGHT / 2);
  ctx.lineTo(WIDTH, HEIGHT / 2);
  ctx.stroke();
}

function drawDecisionBoundary() {
  if (Math.abs(model.w1) < 0.0001) return;

  const xLeft = -1;
  const xRight = 1;
  const yLeft = -(model.w0 * xLeft + model.b) / model.w1;
  const yRight = -(model.w0 * xRight + model.b) / model.w1;

  const a = toCanvasSpace(xLeft, yLeft);
  const b = toCanvasSpace(xRight, yRight);

  ctx.strokeStyle = "rgba(245,249,255,0.95)";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawPoints() {
  points.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.px, point.py, 6.2, 0, Math.PI * 2);
    ctx.fillStyle = point.label ? "#ff5fa2" : "#44f2c6";
    ctx.fill();

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(6,12,22,0.95)";
    ctx.stroke();
  });
}

function render() {
  ctx.clearRect(0, 0, WIDTH, HEIGHT);
  drawHeatmap();
  drawAxes();
  drawDecisionBoundary();
  drawPoints();
  updateStats();
}

function train(epochs) {
  if (points.length < 3) return;

  const lr = Number(learningRateInput.value);

  for (let step = 0; step < epochs; step += 1) {
    let gradW0 = 0;
    let gradW1 = 0;
    let gradB = 0;
    let loss = 0;
    let correct = 0;

    for (const point of points) {
      const features = toFeatureSpace(point.px, point.py);
      const pred = predictProbability(features);
      const target = point.label;
      const error = pred - target;

      gradW0 += error * features.x;
      gradW1 += error * features.y;
      gradB += error;
      loss += -(target * Math.log(pred + EPS) + (1 - target) * Math.log(1 - pred + EPS));

      if ((pred >= 0.5 ? 1 : 0) === target) {
        correct += 1;
      }
    }

    const n = points.length;
    model.w0 -= (lr * gradW0) / n;
    model.w1 -= (lr * gradW1) / n;
    model.b -= (lr * gradB) / n;

    model.loss = loss / n;
    model.accuracy = correct / n;
    iterationCount += 1;
  }

  render();
}

function seedDataset() {
  points = [];

  function pushCluster(cx, cy, spread, label, n) {
    for (let i = 0; i < n; i += 1) {
      const angle = Math.random() * Math.PI * 2;
      const radius = (Math.random() ** 0.75) * spread;
      const px = cx + Math.cos(angle) * radius;
      const py = cy + Math.sin(angle) * radius;

      points.push({
        px: Math.min(WIDTH - 14, Math.max(14, px)),
        py: Math.min(HEIGHT - 14, Math.max(14, py)),
        label,
      });
    }
  }

  pushCluster(WIDTH * 0.35, HEIGHT * 0.68, 115, 0, 55);
  pushCluster(WIDTH * 0.7, HEIGHT * 0.34, 130, 1, 55);

  iterationCount = 0;
  model.loss = null;
  model.accuracy = null;

  render();
}

function resetAll() {
  points = [];
  model.w0 = (Math.random() - 0.5) * 0.6;
  model.w1 = (Math.random() - 0.5) * 0.6;
  model.b = (Math.random() - 0.5) * 0.4;
  model.loss = null;
  model.accuracy = null;
  iterationCount = 0;
  if (autoTrainId) {
    clearInterval(autoTrainId);
    autoTrainId = null;
    autoBtn.textContent = "Start Auto-Train";
  }
  render();
}

canvas.addEventListener("click", (event) => {
  const rect = canvas.getBoundingClientRect();
  const px = ((event.clientX - rect.left) / rect.width) * WIDTH;
  const py = ((event.clientY - rect.top) / rect.height) * HEIGHT;

  points.push({ px, py, label: labelMode });
  render();
});

classZeroBtn.addEventListener("click", () => {
  labelMode = 0;
  classZeroBtn.classList.add("active");
  classOneBtn.classList.remove("active");
});

classOneBtn.addEventListener("click", () => {
  labelMode = 1;
  classOneBtn.classList.add("active");
  classZeroBtn.classList.remove("active");
});

learningRateInput.addEventListener("input", () => {
  learningRateValue.textContent = Number(learningRateInput.value).toFixed(2);
});

epochsInput.addEventListener("input", () => {
  epochsValue.textContent = epochsInput.value;
});

trainBtn.addEventListener("click", () => {
  train(Number(epochsInput.value));
});

autoBtn.addEventListener("click", () => {
  if (autoTrainId) {
    clearInterval(autoTrainId);
    autoTrainId = null;
    autoBtn.textContent = "Start Auto-Train";
    return;
  }

  autoTrainId = setInterval(() => {
    train(Number(epochsInput.value));
  }, 150);
  autoBtn.textContent = "Stop Auto-Train";
});

seedBtn.addEventListener("click", () => {
  seedDataset();
});

resetBtn.addEventListener("click", () => {
  resetAll();
});

seedDataset();
render();
