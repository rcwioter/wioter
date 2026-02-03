const form = document.getElementById("upload-form");
const statusEl = document.getElementById("status");
const preview = document.getElementById("preview");
const canvas = document.getElementById("overlay");
const stats = document.getElementById("stats");
const legend = document.getElementById("legend");

const COLORS = ["#f59f00", "#15aabf", "#ff6b6b", "#845ef7"]; // palette

const resetOverlay = () => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
};

const renderStats = (summary) => {
  stats.innerHTML = "";
  legend.innerHTML = "";
  const entries = Object.entries(summary);

  if (entries.length === 0) {
    stats.innerHTML = '<p class="muted">未检测到病斑。</p>';
    return;
  }

  entries.forEach(([label, count], index) => {
    const color = COLORS[index % COLORS.length];
    const card = document.createElement("div");
    card.className = "stat-card";
    card.innerHTML = `<span>${label}</span><strong>${count}</strong>`;
    stats.appendChild(card);

    const legendItem = document.createElement("div");
    legendItem.className = "legend-item";
    legendItem.innerHTML = `<span class="legend-color" style="background:${color}"></span>${label}`;
    legend.appendChild(legendItem);
  });
};

const drawDetections = (detections) => {
  const ctx = canvas.getContext("2d");
  ctx.lineWidth = 2;
  ctx.font = "14px sans-serif";

  detections.forEach((det, index) => {
    const color = COLORS[index % COLORS.length];
    ctx.strokeStyle = color;
    ctx.fillStyle = color;

    ctx.strokeRect(det.x, det.y, det.width, det.height);
    ctx.fillRect(det.x, det.y - 20, ctx.measureText(det.label).width + 12, 20);
    ctx.fillStyle = "#fff";
    ctx.fillText(`${det.label} ${(det.confidence * 100).toFixed(1)}%`, det.x + 6, det.y - 6);
  });
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("image");

  if (!fileInput.files[0]) {
    statusEl.textContent = "请先选择图片。";
    return;
  }

  statusEl.textContent = "检测中，请稍候...";
  resetOverlay();

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  try {
    const response = await fetch("/detect", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();

    if (!response.ok) {
      statusEl.textContent = payload.error || "检测失败。";
      return;
    }

    preview.src = payload.image;
    preview.onload = () => {
      preview.style.display = "block";
      canvas.width = preview.naturalWidth;
      canvas.height = preview.naturalHeight;
      canvas.style.width = "100%";
      canvas.style.height = "auto";
      drawDetections(payload.detections);
    };

    renderStats(payload.summary);
    statusEl.textContent = "检测完成。";
  } catch (error) {
    statusEl.textContent = "检测服务暂不可用。";
  }
});
