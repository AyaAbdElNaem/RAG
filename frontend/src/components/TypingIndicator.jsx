export default function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="bg-white border border-petnutri-orange-light/20 shadow-sm rounded-2xl rounded-bl-sm px-4 py-3 flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-petnutri-orange-light animate-bounce [animation-delay:-0.3s]" />
        <span className="w-1.5 h-1.5 rounded-full bg-petnutri-orange-light animate-bounce [animation-delay:-0.15s]" />
        <span className="w-1.5 h-1.5 rounded-full bg-petnutri-orange-light animate-bounce" />
      </div>
    </div>
  )
}
