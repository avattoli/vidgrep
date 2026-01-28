import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './Home.css'
import { Button } from '@heroui/react'
import { Loading } from '../components/Loading'

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

        const formData = new FormData()
        formData.append('video', file)

        try {
            const response = await fetch('http://localhost:5175/api/ingest', {
                method: 'POST',
                body: formData
            })
            const payload = await response.json().catch(() => ({}))

            if (!response.ok) {
                throw new Error(payload?.error || 'Upload failed')
            }

            setUploadMessage('Upload complete. Click the preview to ask a question.')
            setIsUploaded(true)
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Upload failed'
            setUploadError(message)
        } finally {
            setIsUploading(false)
        }
    }

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
