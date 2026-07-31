import { useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

type Form = {
  age: number; gender: number; height: number; weight: number
  ap_hi: number; ap_lo: number; cholesterol: number; gluc: number
  smoke: number; alco: number; active: number
}
type Risk = {
  risk_percent: string; risk_label: string; risk_probability: number
  bmi: number; pulse_pressure: number
  top_factors: { feature: string; magnitude: number }[]; disclaimer: string
}
type Ecg = {
  recording_index: number; anomaly_score: number; risk_level: string
  interpretation: string; limitation_note: string; waveform: number[]
  waveform_sample_indices: number[]; segment_scores: number[]
}

const initial: Form = {
  age: 55, gender: 2, height: 172, weight: 78, ap_hi: 128, ap_lo: 82,
  cholesterol: 1, gluc: 1, smoke: 0, alco: 0, active: 1,
}
const labels: Record<string, string> = {
  ap_hi: 'Systolic blood pressure', age_years: 'Age', cholesterol: 'Cholesterol',
  bmi: 'Body mass index', active: 'Physical activity', ap_lo: 'Diastolic blood pressure',
}

function NumberField({ label, unit, value, onChange, min, max }: {
  label: string; unit: string; value: number; onChange: (value: number) => void; min: number; max: number
}) {
  return <label className="field">
    {label}
    <span className="number">
      <input type="number" value={value} min={min} max={max}
        onChange={event => onChange(event.currentTarget.value === '' ? Number.NaN : Number(event.currentTarget.value))} />
      <small>{unit}</small>
    </span>
  </label>
}

function SelectField({ label, value, onChange, options }: {
  label: string; value: number; onChange: (value: number) => void; options: [number, string][]
}) {
  return <label className="field">
    {label}
    <select value={value} onChange={event => onChange(Number(event.currentTarget.value))}>
      {options.map(([optionValue, text]) => <option value={optionValue} key={optionValue}>{text}</option>)}
    </select>
  </label>
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return <label className="toggle"><input type="checkbox" checked={checked} onChange={event => onChange(event.currentTarget.checked)} /><span />{label}</label>
}

function App() {
  const [tab, setTab] = useState<'risk' | 'ecg'>('risk')
  const [form, setForm] = useState<Form>(initial)
  const [risk, setRisk] = useState<Risk | null>(null)
  const [ecg, setEcg] = useState<Ecg | null>(null)
  const [index, setIndex] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const update = (key: keyof Form, value: number) => setForm(previous => ({ ...previous, [key]: value }))
  const switchTab = (next: 'risk' | 'ecg') => { setTab(next); setError('') }

  const validate = () => {
    if (!Object.values(form).every(Number.isFinite)) return 'Please enter a number in every clinical field.'
    if (form.ap_hi <= form.ap_lo) return 'Systolic BP must be greater than diastolic BP.'
    if (form.age < 1 || form.age > 120 || form.height < 100 || form.height > 220 || form.weight < 20 || form.weight > 300) {
      return 'Please use the supported clinical ranges shown in the form.'
    }
    return ''
  }

  async function assess(event: FormEvent) {
    event.preventDefault()
    setError('')
    const message = validate()
    if (message) { setError(message); return }
    setBusy(true)
    try {
      const response = await fetch(`${API}/predict/tabular`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form),
      })
      const body = await response.json()
      if (!response.ok) {
        const detail = Array.isArray(body.detail) ? body.detail.map((item: { msg?: string }) => item.msg).join(' ') : body.detail
        throw new Error(typeof detail === 'string' ? detail : 'Unable to assess this profile.')
      }
      setRisk(body)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to assess this profile.')
    } finally { setBusy(false) }
  }

  async function scoreEcg() {
    setError('')
    setBusy(true)
    try {
      const response = await fetch(`${API}/predict/ecg`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ recording_index: index }),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Unable to calculate ECG score.')
      setEcg(body)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to calculate ECG score.')
    } finally { setBusy(false) }
  }

  const waveformPath = useMemo(() => {
    if (!ecg) return ''
    const lastIndex = ecg.waveform_sample_indices.at(-1) || 1
    return ecg.waveform.map((value, position) => {
      const x = (ecg.waveform_sample_indices[position] / lastIndex) * 1000
      return `${position ? 'L' : 'M'} ${x} ${(1 - value) * 250}`
    }).join(' ')
  }, [ecg])

  return <main className="app-shell">
    <header className="topbar"><span className="brand">CARDIO<span>SCOPE</span></span><span className="status"><i /> RESEARCH DEMO · NOT FOR DIAGNOSIS</span></header>
    <section className="hero"><p className="eyebrow">MULTI-MODAL CARDIOVASCULAR INTELLIGENCE</p><h1>See the signals.<br /><em>Understand the risk.</em></h1></section>
    <nav className="mode-tabs" aria-label="Analysis mode">
      <button className={tab === 'risk' ? 'active' : ''} onClick={() => switchTab('risk')}>Clinical risk calculator</button>
      <button className={tab === 'ecg' ? 'active' : ''} onClick={() => switchTab('ecg')}>ECG anomaly explorer</button>
    </nav>
    {error && <div className="notice error" role="alert">{error}</div>}

    {tab === 'risk' ? <section className="workspace">
      <form className="glass form-card" onSubmit={assess}>
        <p className="eyebrow">INPUT</p><h2>Clinical profile</h2>
        <div className="form-grid">
          <NumberField label="Age" unit="years" value={form.age} onChange={value => update('age', value)} min={1} max={120} />
          <SelectField label="Biological sex" value={form.gender} onChange={value => update('gender', value)} options={[[2, 'Male'], [1, 'Female']]} />
          <NumberField label="Height" unit="cm" value={form.height} onChange={value => update('height', value)} min={100} max={220} />
          <NumberField label="Weight" unit="kg" value={form.weight} onChange={value => update('weight', value)} min={20} max={300} />
          <NumberField label="Systolic BP" unit="mmHg" value={form.ap_hi} onChange={value => update('ap_hi', value)} min={60} max={300} />
          <NumberField label="Diastolic BP" unit="mmHg" value={form.ap_lo} onChange={value => update('ap_lo', value)} min={30} max={200} />
          <SelectField label="Cholesterol" value={form.cholesterol} onChange={value => update('cholesterol', value)} options={[[1, 'Normal'], [2, 'Above normal'], [3, 'Well above normal']]} />
          <SelectField label="Glucose" value={form.gluc} onChange={value => update('gluc', value)} options={[[1, 'Normal'], [2, 'Above normal'], [3, 'Well above normal']]} />
        </div>
        <div className="toggles">
          <Toggle label="Smokes" checked={Boolean(form.smoke)} onChange={value => update('smoke', value ? 1 : 0)} />
          <Toggle label="Alcohol use" checked={Boolean(form.alco)} onChange={value => update('alco', value ? 1 : 0)} />
          <Toggle label="Physically active" checked={Boolean(form.active)} onChange={value => update('active', value ? 1 : 0)} />
        </div>
        <button className="primary" disabled={busy}>{busy ? 'ANALYSING…' : 'ASSESS CARDIOVASCULAR RISK →'}</button>
      </form>
      <aside className="glass result">
        {risk ? <><p className="eyebrow">MODEL OUTPUT</p><div className="risk-number">{risk.risk_percent}</div><h2>{risk.risk_label} estimated risk</h2><div className="meter"><i style={{ width: `${risk.risk_probability * 100}%` }} /></div><div className="stats"><span>BMI<b>{risk.bmi}</b></span><span>Pulse pressure<b>{risk.pulse_pressure} mmHg</b></span></div><h3>Leading model factors</h3><ul>{risk.top_factors.map(item => <li key={item.feature}>{labels[item.feature] ?? item.feature}<small>↑ risk</small><b>{item.magnitude.toFixed(3)}</b></li>)}</ul><p className="disclaimer">{risk.disclaimer}</p></> : <><p className="eyebrow">MODEL OUTPUT</p><h2>Awaiting profile</h2><p>Complete the clinical profile to receive an explained population-level risk estimate.</p></>}
      </aside>
    </section> : <>
      <section className="workspace">
        <div className="glass form-card">
          <p className="eyebrow">UNSUPERVISED ECG ANALYSIS</p><h2>Explore a recording</h2>
          <NumberField label="Recording index" unit="of 186" value={index} onChange={value => setIndex(Math.max(0, Math.min(186, value)))} min={0} max={186} />
          <div className="index-row"><button onClick={() => setIndex(Math.max(0, index - 1))}>← Previous</button><button onClick={() => setIndex(Math.min(186, index + 1))}>Next →</button></div>
          <button className="primary" onClick={scoreEcg} disabled={busy}>{busy ? 'CALCULATING…' : 'CALCULATE ANOMALY SCORE →'}</button>
        </div>
        <aside className="glass result">{ecg ? <><p className="eyebrow">MODEL OUTPUT</p><div className="risk-number">{ecg.anomaly_score.toFixed(5)}</div><h2>{ecg.risk_level} relative anomaly</h2><p>{ecg.interpretation}</p><p className="disclaimer">{ecg.limitation_note}</p></> : <><p className="eyebrow">MODEL OUTPUT</p><h2>Awaiting ECG selection</h2><p>Select a recording and calculate its reconstruction-error anomaly score.</p></>}</aside>
      </section>
      {ecg && <section className="glass waveform-card">
        <div className="waveform-header"><div><p className="eyebrow">RAW ECG SIGNAL · DISPLAY DECIMATED</p><h2>Recording #{ecg.recording_index}</h2></div><p>cyan: amplitude · bands: local reconstruction error</p></div>
        <svg viewBox="0 0 1000 280" aria-label="ECG waveform with reconstruction error bands">
          {ecg.segment_scores.map((segmentScore, position) => <rect className="error-band" key={position} x={position / ecg.segment_scores.length * 1000} y="0" width={1000 / ecg.segment_scores.length + 1} height="250" opacity={0.03 + 0.2 * segmentScore / Math.max(...ecg.segment_scores)} />)}
          <path className="ecg-path" d={waveformPath} />
        </svg>
      </section>}
    </>}
  </main>
}

export default App
