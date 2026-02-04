import './App.css'
import Home from './pages/Home'
import Chat from './pages/Chat'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Analytics } from '@vercel/analytics/react'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/chat" element={<Chat />} />
      </Routes>
      <Analytics />
    </BrowserRouter>
  )
}

export default App
