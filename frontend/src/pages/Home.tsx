import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './Home.css'
import { Button } from '@heroui/react'
import { Loading } from '../components/Loading'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:5175'

export default function Home() {
    const [isDragging, setIsDragging] = useState(false)
    const [file, setFile] = useState<File | null>(null)
    const [previewUrl, setPreviewUrl] = useState<string | null>(null)
    const [isUploading, setIsUploading] = useState(false)
    const [uploadError, setUploadError] = useState<string | null>(null)
    const [uploadMessage, setUploadMessage] = useState<string | null>(null)
    const [isUploaded, setIsUploaded] = useState(false)
    const [hasExistingData, setHasExistingData] = useState(false)
    const [existingCount, setExistingCount] = useState<number | null>(null)
    const [uploadProgress, setUploadProgress] = useState<number>(0)
    const [jobStatus, setJobStatus] = useState<string | null>(null)
    const [jobId, setJobId] = useState<string | null>(null)
    const jobPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
    const navigate = useNavigate()

    useEffect(() => {
        if (!file) {
            setPreviewUrl(null)
            return
        }
        const url = URL.createObjectURL(file)
        setPreviewUrl(url)
        return () => URL.revokeObjectURL(url)
    }, [file])

    useEffect(() => {
        const controller = new AbortController()

        const loadStatus = async () => {
            try {
        const response = await fetch('http://localhost:5175/api/status', {
                    signal: controller.signal
                })
                const payload = await response.json().catch(() => ({}))
                if (!response.ok) {
                    return
                }
                const videoCount = Number(payload?.videoCount ?? 0)
                const embeddings = Number(payload?.embeddings ?? 0)
                const hasData = videoCount > 0 || embeddings > 0
                setHasExistingData(hasData)
                setExistingCount(videoCount > 0 ? videoCount : null)
            } catch {
                setHasExistingData(false)
            }
        }

        loadStatus()

        return () => controller.abort()
    }, [])

    const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault()
        setIsDragging(true)
    }

    const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault()
        setIsDragging(false)
    }

    const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault()
        setIsDragging(false)
        const droppedFile = event.dataTransfer.files?.[0]
        if (droppedFile) {
            setFile(droppedFile)
            setUploadError(null)
            setUploadMessage(null)
            setIsUploaded(false)
        }
    }

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = event.target.files?.[0]
        if (selectedFile) {
            setFile(selectedFile)
            setUploadError(null)
            setUploadMessage(null)
            setIsUploaded(false)
        }
    }

    const handleSubmit = async () => {
        if (!file || isUploading) {
            return
        }

        setIsUploading(true)
        setUploadError(null)
        setUploadMessage(null)
        setUploadProgress(0)
        setJobStatus(null)
        setJobId(null)

        try {
            // 1) init
            const initRes = await fetch(`${API_BASE}/api/upload/init`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: file.name, size: file.size })
            })
            const initJson = await initRes.json().catch(() => ({}))
            if (!initRes.ok || !initJson.uploadId || !initJson.chunkSize) {
                throw new Error(initJson?.error || 'Upload init failed')
            }
            const { uploadId, chunkSize } = initJson

            // 2) upload chunks
            const totalChunks = Math.ceil(file.size / chunkSize)
            for (let i = 0; i < totalChunks; i++) {
                const start = i * chunkSize
                const end = Math.min(file.size, start + chunkSize)
                const blob = file.slice(start, end)
                const chunkRes = await fetch(`${API_BASE}/api/upload/chunk`, {
                    method: 'POST',
                    headers: {
                        'upload-id': uploadId,
                        'chunk-index': String(i),
                        'total-chunks': String(totalChunks)
                    },
                    body: blob
                })
                const chunkJson = await chunkRes.json().catch(() => ({}))
                if (!chunkRes.ok) {
                    throw new Error(chunkJson?.error || 'Chunk upload failed')
                }
                setUploadProgress(Math.round(((i + 1) / totalChunks) * 100))
            }

            // 3) complete + enqueue ingest job
            const completeRes = await fetch(`${API_BASE}/api/upload/complete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ uploadId })
            })
            const completeJson = await completeRes.json().catch(() => ({}))
            if (!completeRes.ok || !completeJson.jobId) {
                throw new Error(completeJson?.error || 'Finalize failed')
            }

            const job = completeJson.jobId as string
            setJobId(job)
            setUploadMessage('Upload complete. Processing...')
            setIsUploaded(false)

            // 4) poll job status
            if (jobPollRef.current) clearInterval(jobPollRef.current)
            jobPollRef.current = setInterval(async () => {
                try {
                    const jr = await fetch(`${API_BASE}/api/job/${job}`)
                    const jj = await jr.json().catch(() => ({}))
                    if (!jr.ok) return
                    setJobStatus(jj.status || null)
                    if (jj.status === 'done') {
                        setIsUploaded(true)
                        setUploadMessage('Processing finished. Click the preview to ask a question.')
                        clearInterval(jobPollRef.current!)
                        jobPollRef.current = null
                    }
                    if (jj.status === 'error') {
                        setUploadError(jj.error || 'Ingest failed')
                        clearInterval(jobPollRef.current!)
                        jobPollRef.current = null
                    }
                } catch {
                    // ignore transient polling errors
                }
            }, 2000)
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Upload failed'
            setUploadError(message)
        } finally {
            setIsUploading(false)
        }
    }

    useEffect(() => {
        return () => {
            if (jobPollRef.current) clearInterval(jobPollRef.current)
        }
    }, [])

    return (
        <div className="home">
            <div className="home-header">
                <p className="eyebrow">VidGrep</p>
                <h1>Drop a video to start searching</h1>
                <p className="subhead">Drag and drop a file below, or click to browse.</p>
            </div>

            <div
                className={`dropzone ${isDragging ? 'is-dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <input
                    className="file-input"
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                />
                <div className="dropzone-label">
                    <span className="dropzone-title">Drag &amp; drop your video</span>
                    <span className="dropzone-hint">MP4, MOV, MKV · up to whatever your browser allows</span>
                </div>
            </div>

            <div className="file-meta">
                {file ? (
                    <>
                        <span className="file-name">{file.name}</span>
                        <span className="file-size">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                    </>
                ) : (
                    <span>No file selected yet.</span>
                )}
            </div>


            <div className="submit-row">
                {isUploading ? (
                    <Loading />
                ) : (
                    <Button
                        variant="ghost"
                        isDisabled={!file}
                        className="submit-button"
                        onPress={handleSubmit}
                    >
                        Submit
                    </Button>
                )}
            </div>

            {hasExistingData && !file ? (
                <div className="existing-data">
                    <p className="subhead">
                        Existing dataset detected{existingCount ? ` (${existingCount} video${existingCount === 1 ? '' : 's'})` : ''}.
                    </p>
                    <Button variant="ghost" className="submit-button" onPress={() => navigate('/chat')}>
                        Start chatting
                    </Button>
                </div>
            ) : null}

            {uploadError ? <p className="upload-error">{uploadError}</p> : null}
            {uploadMessage ? <p className="upload-message">{uploadMessage}</p> : null}
            {uploadProgress > 0 && uploadProgress < 100 ? (
                <p className="upload-progress">Uploading... {uploadProgress}%</p>
            ) : null}
            {jobStatus ? <p className="upload-message">Ingest: {jobStatus}</p> : null}


            {previewUrl ? (
                <div
                    className={`video-preview ${isUploaded ? 'is-clickable' : ''}`}
                    onClick={() => {
                        if (isUploaded) {
                            navigate('/chat')
                        }
                    }}
                    role={isUploaded ? 'button' : undefined}
                    tabIndex={isUploaded ? 0 : -1}
                >
                    <video
                        className="video-thumb"
                        src={previewUrl}
                        preload="metadata"
                        muted
                        playsInline
                    />
                </div>
            ) : null}
        </div>
    )
}
