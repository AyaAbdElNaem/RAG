export default function ChatBubble({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? 'bg-petnutri-green text-white rounded-br-sm'
            : message.isError
            ? 'bg-red-50 text-red-700 border border-red-200 rounded-bl-sm'
            : 'bg-white text-petnutri-text border border-petnutri-orange-light/20 shadow-sm rounded-bl-sm'
        }`}
      >
        {message.text}

        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-petnutri-muted/20">
            <p className="text-xs font-semibold text-petnutri-muted mb-1">Sources</p>
            <ul className="space-y-1">
              {message.sources.map((s, i) => (
                <li key={i} className="text-xs text-petnutri-muted">
                  {s.title || 'Untitled source'}
                  {s.category ? ` · ${s.category}` : ''}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
