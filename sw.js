/* App-shell service worker — caches UI assets for instant reloads */
const SHELL = 'bah-shell-v2';
const ASSETS = ['./', './index.html', './style.css', './app.js', './worker.js'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL).then((cache) => cache.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.filter((k) => k !== SHELL && !k.startsWith('browser-ai-hub-models')).map((k) => caches.delete(k)));
      await self.clients.claim();
    })()
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  // Only cache same-origin app shell; never intercept model CDNs here
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    (async () => {
      const cached = await caches.match(request);
      if (cached) {
        // Stale-while-revalidate for shell
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
        if (res.ok && ASSETS.some((a) => url.pathname.endsWith(a.replace('./', '')) || url.pathname.endsWith('/'))) {
          const copy = res.clone();
          caches.open(SHELL).then((c) => c.put(request, copy));
        }
        return res;
      } catch (err) {
        const fallback = await caches.match('./index.html');
        if (fallback) return fallback;
        throw err;
      }
    })()
  );
});
