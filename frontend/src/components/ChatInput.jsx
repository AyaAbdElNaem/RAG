import { useState } from 'react'
import { ArrowRight } from 'lucide-react'

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState('')

  const handleSend = () => {
    if (!value.trim() || disabled) return
    onSend(value)
    setValue('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="bg-white border border-petnutri-orange-light/25 rounded-2xl shadow-sm p-3 sm:p-4">
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        rows={2}
        placeholder="Ask about ingredients, safety, or diet plans..."
        className="w-full resize-none bg-transparent outline-none text-sm text-petnutri-text placeholder:text-petnutri-muted disabled:opacity-60"
      />
      <div className="flex justify-end mt-2">
        <button
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          className="flex items-center gap-2 bg-petnutri-orange hover:bg-petnutri-orange/90 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium text-sm px-5 py-2.5 rounded-full transition-colors"
        >
          Consult AI
          <ArrowRight size={16} strokeWidth={2.5} />
        </button>
      </div>
    </div>
  )
}
