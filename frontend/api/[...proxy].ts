export default async function handler(req, res) {
  const base = (process.env.VITE_API_BASE || '').replace(/\/$/, '')
  if (!base) {
    return res.status(500).json({ error: 'VITE_API_BASE not set' })
  }
  const url = `${base}${req.url.replace('/api/proxy', '')}`

  const response = await fetch(url, {
    method: req.method,
    headers: {
      'Content-Type': 'application/json',
    },
    body: req.method !== 'GET' ? JSON.stringify(req.body) : undefined,
  });

  const data = await response.text();
  res.status(response.status).send(data);
}
