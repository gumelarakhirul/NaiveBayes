const form = document.querySelector("#predict-form");
const input = document.querySelector("#url-input");
const submit = document.querySelector("#submit-button");
const errorBox = document.querySelector("#form-error");
const loading = document.querySelector("#loading-status");
const resultBox = document.querySelector("#result");
const emptyBox = document.querySelector("#result-empty");

async function healthCheck() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    document.querySelector("#status-dot").classList.add("ready");
    document.querySelector("#server-label").textContent = "Siap menganalisis";
  } catch {
    document.querySelector("#server-label").textContent = "Layanan belum terhubung";
  }
}

document.querySelectorAll("[data-example]").forEach(button => {
  button.addEventListener("click", () => {
    input.value = button.dataset.example;
    input.focus();
  });
});

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (submit.disabled) return;
  const url = input.value.trim();
  if (!url) {
    errorBox.textContent = "Masukkan URL atau domain terlebih dahulu.";
    errorBox.classList.remove("hidden");
    input.focus();
    return;
  }
  submit.disabled = true;
  submit.textContent = "Menganalisis...";
  form.setAttribute("aria-busy", "true");
  errorBox.classList.add("hidden");
  resultBox.classList.add("hidden");
  emptyBox.classList.add("hidden");
  loading.classList.remove("hidden");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({url}),
      signal: controller.signal
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Analisis gagal. Silakan coba kembali.");
    showResult(result);
  } catch (error) {
    errorBox.textContent = error.name === "AbortError"
      ? "Analisis terlalu lama. Silakan coba kembali."
      : error instanceof TypeError ? "Tidak dapat menghubungi layanan. Pastikan server berjalan, lalu coba kembali." : error.message;
    errorBox.classList.remove("hidden");
    emptyBox.classList.remove("hidden");
  } finally {
    clearTimeout(timeout);
    submit.disabled = false;
    submit.textContent = "Analisis tautan \u2197";
    form.removeAttribute("aria-busy");
    loading.classList.add("hidden");
  }
});

const indicatorLabels = {
  host_ip: "Alamat IP sebagai host", at_symbol: "Simbol @ pada alamat",
  many_subdomains: "Subdomain berlapis", long_url: "URL panjang",
  punycode: "Domain Punycode", plain_http: "Menggunakan HTTP",
  suspicious_path: "Kata sensitif pada jalur URL", percent_encoding: "Karakter tersandi pada URL",
  unusual_port: "Port tidak umum", invalid_port: "Port tidak valid"
};

function showResult(result) {
  resultBox.classList.remove("hidden");
  resultBox.classList.toggle("risky", result.prediction === "phishing");
  document.querySelector("#prediction").textContent = result.prediction === "phishing" ? "Terindikasi phishing" : "Cenderung legitimate";
  document.querySelector("#probability").textContent = `${(result.probability * 100).toLocaleString("id-ID", {minimumFractionDigits: 2, maximumFractionDigits: 2})}%`;
  document.querySelector("#risk-fill").style.width = `${Math.max(0, Math.min(100, result.probability * 100))}%`;
  for (const [id, value] of Object.entries({
    hostname: result.hostname, decoded: result.decoded_domain,
    "active-model": result.model, homograph: result.homograph ? "Terdeteksi" : "Tidak terdeteksi",
    "unicode-scripts": result.scripts?.length ? result.scripts.join(", ") : "Tidak teridentifikasi",
    "url-indicators": result.url_indicators?.length ? result.url_indicators.map(value => {
      const key = value.replaceAll("__", "");
      return indicatorLabels[key] || key.replaceAll("_", " ");
    }).join(", ") : "Tidak ada indikator struktural",
    interpretation: result.interpretation,
    "homograph-reason": (result.homograph_reasons || []).join(" \u00b7 ")
  })) document.getElementById(id).textContent = value;
  document.querySelector("#homograph-alert").classList.toggle("hidden", !result.homograph);
}

healthCheck();
