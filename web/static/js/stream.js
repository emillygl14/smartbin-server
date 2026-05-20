const STREAM_REFRESH_INTERVAL = 30000;

function initVideoStream() {
  const container = document.getElementById("stream-container");
  if (!container) return;

  async function fetchAndUpdateStream() {
    try {
      const response = await fetch("/api/stream-url");
      const data = await response.json();
      const url = data.stream_url;

      if (url && url !== "") {
        container.innerHTML = `
          <img
            id="camera-stream"
            src="${url}"
            alt="Live Camera SmartBin"
            style="width:100%;display:block;border-radius:18px;"
            onerror="onStreamError(this)"
          />
          <p style="font-size:12px;color:#64748b;margin:6px 0 0;text-align:center;padding-bottom:8px;">
            📡 Live via Cloudflare Tunnel
          </p>
        `;
      } else {
        container.innerHTML = `
          <div style="
            width:100%;
            aspect-ratio:16/9;
            background:#f1f5f9;
            border-radius:18px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            color:#94a3b8;
            gap:10px;
            border:2px dashed rgba(22,163,74,0.2);
          ">
            <span style="font-size:38px;">📷</span>
            <p style="margin:0;font-size:14px;font-weight:600;color:#64748b;">Kamera belum tersambung</p>
            <p style="margin:0;font-size:12px;color:#94a3b8;">Menunggu Raspberry Pi online...</p>
          </div>
        `;
      }
    } catch (err) {
      console.warn("Stream URL fetch error:", err);
    }
  }

  fetchAndUpdateStream();
  setInterval(fetchAndUpdateStream, STREAM_REFRESH_INTERVAL);
}

function onStreamError(img) {
  img.parentElement.innerHTML = `
    <div style="
      width:100%;
      aspect-ratio:16/9;
      background:#fff7f7;
      border-radius:18px;
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
      color:#94a3b8;
      gap:10px;
      border:2px dashed rgba(220,38,38,0.2);
    ">
      <span style="font-size:38px;">⚠️</span>
      <p style="margin:0;font-size:14px;font-weight:600;color:#dc2626;">Stream terputus</p>
      <p style="margin:0;font-size:12px;color:#94a3b8;">Raspberry Pi mungkin sedang offline</p>
    </div>
  `;
}

document.addEventListener("DOMContentLoaded", initVideoStream);
