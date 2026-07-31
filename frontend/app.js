const pages = [...document.querySelectorAll(".page")];
const menuButtons = [...document.querySelectorAll("#menu button[data-page]")];
const historyKey = "punyguard-history-v2";

function showPage(name) {
  pages.forEach(page => page.classList.toggle("active", page.id === name));
  menuButtons.forEach(button => button.classList.toggle("active", button.dataset.page === name));
  document.querySelector("#page-name").textContent =
    menuButtons.find(button => button.dataset.page === name)?.textContent.trim() || "Dashboard";
  document.querySelector(".sidebar").classList.remove("open");
  if (name === "history") renderHistory();
}

menuButtons.forEach(button => button.addEventListener("click", () => showPage(button.dataset.page)));
document.querySelectorAll(".go-predict").forEach(button => button.addEventListener("click", () => showPage("predict")));
document.querySelector("#mobile-menu").addEventListener("click", () => document.querySelector(".sidebar").classList.toggle("open"));

async function healthCheck() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    document.querySelector("#status-dot").classList.add("ready");
    document.querySelector("#server-label").textContent = "Model siap";
  } catch {
    document.querySelector("#server-label").textContent = "Server tidak tersedia";
  }
}

function historyEntries() {
  try { return JSON.parse(localStorage.getItem(historyKey) || "[]"); }
  catch { return []; }
}

function saveHistory(result) {
  const entries = historyEntries();
  entries.unshift({...result, timestamp: new Date().toISOString()});
  localStorage.setItem(historyKey, JSON.stringify(entries.slice(0, 100)));
}

function renderHistory() {
  const entries = historyEntries();
  const list = document.querySelector("#history-list");
  document.querySelector("#history-empty").classList.toggle("hidden", entries.length > 0);
  list.innerHTML = entries.map(entry => `
    <div class="history-row">
      <b>${escapeHtml(entry.hostname)}</b>
      <span class="tag ${entry.prediction}">${entry.prediction === "phishing" ? "Phishing" : "Legitimate"}</span>
      <span>${(entry.probability * 100).toFixed(2)}%</span>
      <span>${escapeHtml(entry.model)}</span>
      <span>${new Date(entry.timestamp).toLocaleString("id-ID")}</span>
    </div>`).join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  })[character]);
}

document.querySelector("#clear-history").addEventListener("click", () => {
  localStorage.removeItem(historyKey);
  renderHistory();
});

document.querySelectorAll("[data-example]").forEach(button => button.addEventListener("click", () => {
  document.querySelector("#url-input").value = button.dataset.example;
}));

document.querySelector("#predict-form").addEventListener("submit", async event => {
  event.preventDefault();
  const submit = event.submitter;
  submit.disabled = true;
  submit.textContent = "Menganalisis…";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({url: document.querySelector("#url-input").value})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediksi gagal.");
    showResult(result);
    saveHistory(result);
  } catch (error) {
    alert(error.message);
  } finally {
    submit.disabled = false;
    submit.textContent = "Analisis";
  }
});

function showResult(result) {
  document.querySelector("#result-empty").classList.add("hidden");
  const container = document.querySelector("#result");
  container.classList.remove("hidden");
  container.classList.toggle("risky", result.prediction === "phishing");
  document.querySelector("#prediction").textContent =
    result.prediction === "phishing" ? "Terindikasi phishing" : "Cenderung legitimate";
  document.querySelector("#probability").textContent = `${(result.probability * 100).toFixed(2)}%`;
  document.querySelector("#risk-fill").style.width = `${result.probability * 100}%`;
  document.querySelector("#hostname").textContent = result.hostname;
  document.querySelector("#decoded").textContent = result.decoded_domain;
  document.querySelector("#active-model").textContent = result.model;
  document.querySelector("#homograph").textContent = result.homograph ? "Terdeteksi" : "Tidak terdeteksi";
  document.querySelector("#unicode-scripts").textContent =
    result.scripts?.length ? result.scripts.join(", ") : "ASCII / tidak teridentifikasi";
  document.querySelector("#url-indicators").textContent =
    result.url_indicators?.length
      ? result.url_indicators.map(value => value.replaceAll("__", "").replaceAll("_", " ")).join(", ")
      : "Tidak ada indikator struktural";
  document.querySelector("#interpretation").textContent = result.interpretation;
  const alertBox = document.querySelector("#homograph-alert");
  alertBox.classList.toggle("hidden", !result.homograph);
  document.querySelector("#homograph-reason").textContent = result.homograph_reasons.join(" · ");
}

healthCheck();
renderHistory();
