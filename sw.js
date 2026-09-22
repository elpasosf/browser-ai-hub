/* App-shell SW — do not intercept Hugging Face / esm CDN model traffic */
const SHELL = 'bah-shell-v3';
const ASSETS = ['./', './index.html', './style.css', './app.js', './llm-worker.js', './sw.js'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(SHELL).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((k) => k.startsWith('bah-shell-') && k !== SHELL)
          .map((k) => caches.delete(k))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    (async () => {
      const cached = await caches.match(request);
      if (cached) {
        event.waitUntil(
          fetch(request)
            .then((res) => {
              if (res.ok) caches.open(SHELL).then((c) => c.put(request, res.clone()));
            })
            .catch(() => {})
        );
        return cached;
      }
      try {
        const res = await fetch(request);
        if (res.ok) {
          const copy = res.clone();
          caches.open(SHELL).then((c) => c.put(request, copy));
        }
        return res;
      } catch {
        return (await caches.match('./index.html')) || Response.error();
      }
    })()
  );
});
