import { useCallback, useState } from 'react'
import { sendChatMessage } from '../services/api.js'

let idCounter = 0
const nextId = () => `msg-${Date.now()}-${idCounter++}`

/**
 * Manages the full lifecycle of a chat conversation:
 * message history, sending, loading state, and error handling.
 */
export function useChat() {
  const [messages, setMessages] = useState([])
  const [isSending, setIsSending] = useState(false)

  const sendMessage = useCallback(async (text) => {
    const trimmed = text.trim()
    if (!trimmed || isSending) return

    const userMessage = { id: nextId(), role: 'user', text: trimmed }
    setMessages((prev) => [...prev, userMessage])
    setIsSending(true)

    try {
      const { reply, sources } = await sendChatMessage(trimmed)
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: 'assistant', text: reply, sources },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: 'assistant',
          text: "Sorry, I couldn't reach the nutrition assistant. Please make sure the backend is running and try again.",
          isError: true,
        },
      ])
    } finally {
      setIsSending(false)
    }
  }, [isSending])

  return { messages, isSending, sendMessage }
}
