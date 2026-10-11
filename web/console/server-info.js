import { sessionUrl } from './protocol.js';

export function showServerInfo(data) {
  const footer = document.getElementById('server-info');
  if (!footer || !data.git_rev) return;
  footer.textContent = `Server ${data.version || 'unknown'} · ${data.git_rev} · started ${data.started_at || 'unknown'}`;
}

// Health works before capture starts. Telemetry refreshes this when the socket changes.
const health = new URL(sessionUrl(window.location), window.location.href);
health.protocol = health.protocol === 'wss:' ? 'https:' : 'http:';
health.pathname = '/health';
health.search = '';
fetch(health).then(r => r.ok ? r.json() : null).then(data => {
  if (data) showServerInfo(data);
}).catch(() => {}); // Cross-origin health may be blocked; telemetry still supplies identity.
