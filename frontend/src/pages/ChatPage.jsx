import { useEffect, useRef } from 'react'
import { Sparkles } from 'lucide-react'
import TopBar from '../components/TopBar.jsx'
import ChatInput from '../components/ChatInput.jsx'
import ChatBubble from '../components/ChatBubble.jsx'
import TypingIndicator from '../components/TypingIndicator.jsx'
import { useChat } from '../hooks/useChat.js'

export default function ChatPage() {
  const { messages, isSending, sendMessage } = useChat()
  const scrollRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, isSending])

  const hasMessages = messages.length > 0

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar showBack />

      <div className="flex-1 flex flex-col px-5 pb-6 max-w-md w-full mx-auto">
        {!hasMessages && (
          <div className="pt-2 pb-6">
            <div className="inline-flex items-center gap-1.5 bg-petnutri-green/10 text-petnutri-green-dark text-xs font-medium px-3 py-1.5 rounded-full mb-5">
              <Sparkles size={13} />
              Nutrition Expert AI
            </div>

            <h1 className="font-display font-bold text-3xl leading-tight text-petnutri-text mb-2">
              How can I help your pet today?
            </h1>
            <p className="text-petnutri-muted text-sm">
              Expert guidance for a healthier, happier life.
            </p>
          </div>
        )}

        {hasMessages && (
          <div
            ref={scrollRef}
            className="chat-scroll flex-1 overflow-y-auto py-4 space-y-3"
          >
            {messages.map((message) => (
              <ChatBubble key={message.id} message={message} />
            ))}
            {isSending && <TypingIndicator />}
          </div>
        )}

        <div className={hasMessages ? '' : 'mt-auto'}>
          <ChatInput onSend={sendMessage} disabled={isSending} />
        </div>
      </div>
    </div>
  )
}
