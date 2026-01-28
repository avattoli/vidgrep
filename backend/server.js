import cors from 'cors'
import express from 'express'
import multer from 'multer'
import crypto from 'crypto'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'
import { spawn } from 'child_process'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const projectRoot = path.resolve(__dirname, '..')
const dataDir = path.join(projectRoot, 'data')
const videosDir = path.join(dataDir, 'videos')
const framesDir = path.join(dataDir, 'frames')
const indexDir = path.join(dataDir, 'index')
const metadataDir = path.join(dataDir, 'metadata')
const metadataPath = path.join(metadataDir, 'metadata.json')
const indexPath = path.join(indexDir, 'faiss.index')
const resultsDir = path.join(projectRoot, 'results')
const resultsVideoDir = path.join(projectRoot, 'results_video')
const videoExtensions = new Set(['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'])

for (const dir of [dataDir, videosDir, framesDir, indexDir, metadataDir, resultsDir, resultsVideoDir]) {
  fs.mkdirSync(dir, { recursive: true })
}

const app = express()
app.use(cors())
app.use(express.json({ limit: '10mb' }))

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, videosDir),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname || '').toLowerCase() || '.mp4'
    const unique = crypto.randomUUID().replace(/-/g, '')
    cb(null, `${unique}${ext}`)
  }
})

const upload = multer({ storage })
const pythonBin = process.env.PYTHON_BIN || 'python'

const runPython = (args, { quiet = true } = {}) =>
  new Promise((resolve, reject) => {
    const proc = spawn(pythonBin, args, {
      cwd: projectRoot,
      stdio: ['ignore', 'pipe', 'pipe']
    })

    let stdout = ''
    let stderr = ''

    proc.stdout.on('data', (data) => {
      const chunk = data.toString()
      stdout += chunk
      if (!quiet) {
        process.stdout.write(chunk)
      }
    })

    proc.stderr.on('data', (data) => {
      const chunk = data.toString()
      stderr += chunk
      if (!quiet) {
        process.stderr.write(chunk)
      }
    })

    proc.on('close', (code) => {
      if (code === 0) {
        resolve({ stdout, stderr })
      } else {
        reject(new Error(stderr || `Python exited with code ${code}`))
      }
    })
  })

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' })
})

app.get('/api/status', (_req, res) => {
  let videoCount = 0
  try {
    if (fs.existsSync(videosDir)) {
      const entries = fs.readdirSync(videosDir)
      videoCount = entries.filter((entry) => videoExtensions.has(path.extname(entry).toLowerCase())).length
    }
  } catch {
    videoCount = 0
  }

  let embeddings = 0
  const hasMetadata = fs.existsSync(metadataPath)
  if (hasMetadata) {
    try {
      const raw = fs.readFileSync(metadataPath, 'utf-8')
      const parsed = JSON.parse(raw)
      embeddings = Array.isArray(parsed) ? parsed.length : 0
    } catch {
      embeddings = 0
    }
  }

  const hasIndex = fs.existsSync(indexPath)

  res.json({
    hasVideos: videoCount > 0,
    videoCount,
    embeddings,
    hasIndex,
    hasMetadata
  })
})

app.use('/frames', express.static(path.join(projectRoot, 'data', 'frames')))
app.use('/results', express.static(path.join(projectRoot, 'results')))
app.use('/videos', express.static(videosDir))
app.use(
  '/results_video',
  express.static(resultsVideoDir, {
    etag: false,
    lastModified: false,
    cacheControl: false,
    maxAge: 0,
  })
)

// Quiet the browser's favicon request on the API port
app.get('/favicon.ico', (_req, res) => res.status(204).end())

// Explicit clip handler (helps when static middleware is skipped or misconfigured)
app.get('/api/clip/:name', (req, res) => {
  const filePath = path.join(resultsVideoDir, req.params.name)
  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'Clip not found' })
  }

  const stat = fs.statSync(filePath)
  const fileSize = stat.size
  const range = req.headers.range

  if (range) {
    const [startStr, endStr] = range.replace(/bytes=/, '').split('-')
    const start = parseInt(startStr, 10)
    const end = endStr ? parseInt(endStr, 10) : fileSize - 1
    const chunkSize = end - start + 1
    const file = fs.createReadStream(filePath, { start, end })
    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunkSize,
      'Content-Type': 'video/mp4'
    })
    file.pipe(res)
  } else {
    res.writeHead(200, {
      'Content-Length': fileSize,
      'Content-Type': 'video/mp4',
      'Accept-Ranges': 'bytes'
    })
    fs.createReadStream(filePath).pipe(res)
  }
})

app.post('/api/ingest', upload.single('video'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' })
  }

  try {
    await runPython(['ingest.py', req.file.path], { quiet: false })
    return res.json({
      ok: true,
      filename: req.file.filename,
      path: req.file.path
    })
  } catch (error) {
    return res.status(500).json({ error: error.message })
  }
})

app.post('/api/search', async (req, res) => {
  const { query, top_k } = req.body ?? {}
  if (!query) {
    return res.status(400).json({ error: 'query is required' })
  }

  try {
    const args = ['backend/scripts/search_api.py', query]
    if (top_k) {
      args.push('--top-k', String(top_k))
    }
    const { stdout } = await runPython(args)
    const payload = JSON.parse(stdout.trim() || '{}')
    const results = Array.isArray(payload.results) ? payload.results : []
    const withUrls = results.map((result) => {
      const filename = result?.video_path ? path.basename(result.video_path) : null
      return {
        ...result,
        video_url: filename ? `/videos/${filename}` : null
      }
    })
    return res.json({ results: withUrls })
  } catch (error) {
    return res.status(500).json({ error: error.message })
  }
})

const port = process.env.PORT || 5175
app.listen(port, () => {
  console.log(`VidGrep backend listening on http://localhost:${port}`)
})
