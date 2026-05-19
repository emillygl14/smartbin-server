/**
 * Tambahkan snippet ini ke web/templates/index.html di Railway
 * Letakkan di dalam <script> tag, atau di file web/static/js/stream.js
 *
 * Cara kerja:
 * 1. Setiap 30 detik, cek /api/stream-url dari Railway
 * 2. Kalau URL ada dan berubah, update <img> di halaman
 * 3. Kalau tidak ada URL (Pi belum online), tampilkan placeholder
 */

const STREAM_REFRESH_INTERVAL = 30000; // 30 detik

function initVideoStream() {
  const container = document.getElementById("stream-container");
  if (!container) return;

  async function fetchAndUpdateStream() {
    try {
      const response = await fetch("/api/stream-url");
      const data = await response.json();
      const url = data.stream_url;

      if (url && url !== "") {
        // Ada stream URL dari Pi
        container.innerHTML = `
          <img 
            id="camera-stream"
            src="${url}"
            alt="Live Camera SmartBin"
            style="width:100%;border-radius:8px;background:#000;"
            onerror="onStreamError(this)"
          />
          <p style="font-size:12px;color:#888;margin-top:4px;text-align:center;">
            📡 Live via Cloudflare Tunnel
          </p>
        `;
      } else {
        // Pi belum online atau tunnel belum ready
        container.innerHTML = `
          <div style="
            width:100%;
            aspect-ratio:4/3;
            background:#1a1a1a;
            border-radius:8px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            color:#666;
          ">
            <span style="font-size:32px;">📷</span>
            <p style="margin:8px 0 0;font-size:13px;">Kamera belum tersambung</p>
            <p style="margin:4px 0 0;font-size:11px;">Menunggu Raspberry Pi online...</p>
          </div>
        `;
      }
    } catch (err) {
      console.warn("Stream URL fetch error:", err);
    }
  }

  // Jalankan sekali langsung, lalu setiap 30 detik
  fetchAndUpdateStream();
  setInterval(fetchAndUpdateStream, STREAM_REFRESH_INTERVAL);
}

function onStreamError(img) {
  // Kalau stream putus (Pi mati / tunnel expired)
  img.parentElement.innerHTML = `
    <div style="
      width:100%;
      aspect-ratio:4/3;
      background:#1a1a1a;
      border-radius:8px;
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
      color:#666;
    ">
      <span style="font-size:32px;">⚠️</span>
      <p style="margin:8px 0 0;font-size:13px;">Stream terputus</p>
      <p style="margin:4px 0 0;font-size:11px;">Raspberry Pi mungkin sedang offline</p>
    </div>
  `;
}

// Panggil saat halaman siap
document.addEventListener("DOMContentLoaded", initVideoStream);

/**
 * Tambahkan elemen ini di HTML dashboard di tempat yang kamu inginkan:
 *
 * <div id="stream-container" style="width:100%;max-width:640px;">
 *   <!-- stream akan diisi otomatis oleh JS di atas -->
 * </div>
 */
