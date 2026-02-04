import { useCallback, useEffect, useState } from 'react'
import { Button } from '@heroui/react'
import { Loading } from '../components/Loading'
import './Chat.css'

const API_BASE = '/api/proxy'
const proxyUrl = (path: string) => `${API_BASE}?path=${encodeURIComponent(path)}`

type SearchResult = {
  video_id?: string
  timestamp?: number
  score?: number
  video_url?: string | null
  clip_url?: string | null
  image_url?: string | null
}

type Message = {
  role: 'user' | 'assistant'
  content: string
}

export default function Chat() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [results, setResults] = useState<SearchResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasData, setHasData] = useState<boolean | null>(null)
  const [datasetInfo, setDatasetInfo] = useState<string | null>(null)
  const [videos, setVideos] = useState<{ video_id: string; count: number }[]>([])
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null)
  const [deleteLoading, setDeleteLoading] = useState(false)

  useEffect(() => {
    const controller = new AbortController()

    const loadStatus = async () => {
      try {
        const response = await fetch(proxyUrl('/api/status'), {
          signal: controller.signal
        })
        const payload = await response.json().catch(() => ({}))
        if (!response.ok) {
          setHasData(false)
          return
        }

        const videoCount = Number(payload?.videoCount ?? 0)
        const embeddings = Number(payload?.embeddings ?? 0)
        const dataAvailable = videoCount > 0 || embeddings > 0
        setHasData(dataAvailable)
        if (dataAvailable) {
          setDatasetInfo(
            videoCount > 0
              ? `${videoCount} video${videoCount === 1 ? '' : 's'} ready`
              : `${embeddings} embeddings ready`
          )
        }
      } catch {
        setHasData(false)
      }
    }

    loadStatus()

    return () => controller.abort()
  }, [])

  const loadVideos = useCallback(async () => {
    const controller = new AbortController()
    try {
      const r = await fetch(proxyUrl('/api/videos'), { signal: controller.signal })
      const j = await r.json().catch(() => ({}))
      const list = Array.isArray(j.videos) ? j.videos : []
      setVideos(list)
      if (list.length > 0) {
        if (!selectedVideo || !list.find((v: { video_id: string }) => v.video_id === selectedVideo)) {
          setSelectedVideo(list[0].video_id)
        }
      } else {
        setSelectedVideo(null)
      }
    } catch {
      // ignore
    }
    return () => controller.abort()
  }, [selectedVideo])

  useEffect(() => {
    loadVideos()
  }, [loadVideos])

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const trimmed = query.trim()
    if (!trimmed || isLoading || hasData === false) {
      return
    }

    setIsLoading(true)
    setError(null)
    setResults([])
    setMessages((prev) => [...prev, { role: 'user', content: trimmed + (selectedVideo ? ` [${selectedVideo}]` : '') }])
    setQuery('')

    try {
      const response = await fetch(proxyUrl('/api/search'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmed, top_k: 9, video_id: selectedVideo || undefined })
      })

      const payload = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(payload?.error || 'Search failed')
      }

      const nextResults = Array.isArray(payload.results) ? payload.results : []
      setResults(nextResults)
      if (payload.clip_count === 0 && payload.debug) {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: 'No clips were generated. Debug info:' },
          { role: 'assistant', content: payload.debug }
        ])
      } else {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Found ${nextResults.length} matching clips.` }
        ])
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Search failed'
      setError(message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDelete = async () => {
    if (!selectedVideo || deleteLoading) return
    setDeleteLoading(true)
    setError(null)
    try {
      const resp = await fetch(proxyUrl(`/api/video/${selectedVideo}/delete`), { method: 'POST' })
      const json = await resp.json().catch(() => ({}))
      if (!resp.ok) {
        throw new Error(json?.error || 'Delete failed')
      }
      setMessages((prev) => [...prev, { role: 'assistant', content: `Deleted ${selectedVideo}. Rebuilding index (${json.remaining} videos enqueued).` }])
      setResults([])
      await loadVideos()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Delete failed'
      setError(message)
    } finally {
      setDeleteLoading(false)
    }
  }

  return (
    <div className="chat-page">
      <header className="chat-header">
        <p className="eyebrow">VidGrep</p>
        <h1>Ask about your video</h1>
        <p className="subhead">Describe the moment you want to find.</p>
        <p className="banner-warning">Notice: video playback is temporarily disabled; showing frames only.</p>
      </header>

      {datasetInfo ? <p className="subhead">{datasetInfo}</p> : null}
      {hasData === false ? (
        <p className="upload-error">No videos are indexed yet. Upload one on the home page first.</p>
      ) : null}

      <section className="chat-thread">
        {messages.map((message, index) => (
          <div key={`${message.role}-${index}`} className={`chat-bubble ${message.role}`}>
            {message.content}
          </div>
        ))}
      </section>

      <div className="chat-filters">
        <label className="filter-label">
          Video:
          <select
            value={selectedVideo ?? ''}
            onChange={(e) => setSelectedVideo(e.target.value || null)}
            className="video-select"
          >
            {videos.map((v) => (
              <option key={v.video_id} value={v.video_id}>
                {v.video_id} ({v.count})
              </option>
            ))}
          </select>
        </label>
        <Button
          variant="ghost"
          size="sm"
          isDisabled={!selectedVideo || deleteLoading}
          onPress={handleDelete}
        >
          {deleteLoading ? 'Deleting...' : 'Delete video'}
        </Button>
      </div>

      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          className="chat-input"
          type="text"
          placeholder="e.g., a dog running on the beach"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          disabled={hasData === false}
        />
        {isLoading ? (
          <Loading />
        ) : (
          <Button variant="ghost" type="submit" isDisabled={!query.trim() || hasData === false}>
            Ask
          </Button>
        )}
      </form>

      {error ? <p className="upload-error">{error}</p> : null}

      <section className="results-grid">
        {results.map((result, index) => {
          const startTime = Math.max(0, Number(result.timestamp ?? 0))
          const imageSrc = result.image_url ? proxyUrl(result.image_url) : null
          const key = `${result.video_id ?? 'video'}-${index}`

          return (
            <div key={key} className="result-card">
              {imageSrc ? (
                <div className="frame-wrapper">
                  <img className="result-image" src={imageSrc} alt="Result frame" />
                  <span className="timestamp-chip">{startTime.toFixed(2)}s</span>
                </div>
              ) : (
                <div className="result-missing">Frame unavailable</div>
              )}
              <div className="result-meta">
                <span>{result.video_id ?? 'Video'}</span>
              </div>
            </div>
          )
        })}
      </section>
    </div>
  )
}

// ResultCard (kept for reference; not used in current render)
// type ResultCardProps = {
//   clipSrc: string | null
//   videoSrc: string | null
//   poster?: string
//   label: string
//   timestamp: number
// }
//
// function ResultCard({ clipSrc, videoSrc, poster, label, timestamp }: ResultCardProps) {
//   const src = clipSrc ?? videoSrc
//   return (
//     <div className="result-card">
//       {src ? (
//         <video
//           className="result-video"
//           controls
//           preload="metadata"
//           crossOrigin="anonymous"
//           poster={poster}
//         >
//           <source src={src} type="video/mp4" />
//           Your browser does not support the video tag.
//         </video>
//       ) : (
//         <div className="result-missing">Video unavailable</div>
//       )}
//       <div className="result-meta">
//         <span>{label}</span>
//         <span>{timestamp.toFixed(2)}s</span>
//       </div>
//     </div>
//   )
// }
