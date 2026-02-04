const readBody = (req) =>
  new Promise((resolve, reject) => {
    const chunks = []
    req.on('data', (chunk) => chunks.push(Buffer.from(chunk)))
    req.on('end', () => resolve(Buffer.concat(chunks)))
    req.on('error', reject)
  })

export default async function handler(req, res) {
  const base = (process.env.VITE_API_BASE || '').replace(/\/$/, '')
  if (!base) {
    return res.status(500).json({ error: 'VITE_API_BASE not set' })
  }

  const method = req.method || 'GET'
  const targetPath = (req.url || '').replace('/api/proxy', '')
  const url = `${base}${targetPath}`

  const headers = {}
  for (const [key, value] of Object.entries(req.headers || {})) {
    if (value === undefined) continue
    if (key.toLowerCase() === 'host') continue
    headers[key] = Array.isArray(value) ? value.join(',') : value
  }

  let body
  if (!['GET', 'HEAD'].includes(method)) {
    if (typeof req.body !== 'undefined') {
      if (Buffer.isBuffer(req.body) || typeof req.body === 'string') {
        body = req.body
      } else {
        body = JSON.stringify(req.body)
        if (!headers['content-type']) {
          headers['content-type'] = 'application/json'
        }
      }
    } else {
      body = await readBody(req)
    }
  }

  const response = await fetch(url, { method, headers, body })
  res.status(response.status)
  response.headers.forEach((value, key) => {
    if (key.toLowerCase() === 'transfer-encoding') return
    res.setHeader(key, value)
  })
  const buffer = Buffer.from(await response.arrayBuffer())
  res.send(buffer)
}
