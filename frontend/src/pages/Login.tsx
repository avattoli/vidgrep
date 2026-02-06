import { useState } from 'react'
import { Button } from '@heroui/react'
import './Login.css'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [notice, setNotice] = useState<string | null>(null)

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setNotice('Login is UI-only for now. Backend auth not implemented.')
  }

  return (
    <div className="login">
      <header className="login-header">
        <p className="eyebrow">VidGrep</p>
        <h1>Sign in</h1>
        <p className="subhead">Use your email to continue.</p>
      </header>

      <form className="login-card" onSubmit={handleSubmit}>
        <label className="login-field">
          <span>Email</span>
          <input
            type="email"
            autoComplete="email"
            placeholder="you@company.com"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            required
          />
        </label>

        <label className="login-field">
          <span>Password</span>
          <input
            type="password"
            autoComplete="current-password"
            placeholder="••••••••"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            required
          />
        </label>

        <Button className="submit-button" variant="ghost" type="submit">
          Continue
        </Button>

        {notice ? <p className="login-note">{notice}</p> : null}
      </form>
    </div>
  )
}
