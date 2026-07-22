// const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''
/**
 * Sends a user question to the RAG chatbot backend.
 * @param {string} question
 * @returns {Promise<{reply: string, sources: Array<object>}>}
 */
export async function sendChatMessage(question) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`)
  }

  const data = await response.json()
  return {
    reply: data.reply ?? 'No response received.',
    sources: data.sources ?? [],
  }
}
