import axios from 'axios'

// Use relative URL when VITE_API_BASE not set -> works behind vite proxy and in Docker
const BASE = import.meta.env.VITE_API_BASE || ''

export const api = axios.create({ baseURL: BASE || undefined, timeout: 120000 })

export async function alignSync(form: FormData) {
  const { data } = await api.post('/api/v1/align/sync', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
  return data
}

export async function getStatus(taskId: string) {
  const { data } = await api.get(`/api/v1/status/${taskId}`)
  return data
}

export async function health() {
  const { data } = await api.get('/api/v1/health')
  return data
}
