import { ArrowLeft, PawPrint } from 'lucide-react'
import { Link } from 'react-router-dom'

export default function TopBar({ showBack = false }) {
  return (
    <header className="flex items-center justify-between px-5 py-4">
      {showBack ? (
        <Link
          to="/"
          className="flex items-center gap-1.5 text-petnutri-orange-light font-medium text-sm hover:opacity-80 transition-opacity"
        >
          <ArrowLeft size={18} strokeWidth={2.25} />
          Back to Home
        </Link>
      ) : (
        <span />
      )}

      <div className="flex items-center gap-1.5 text-petnutri-brown font-display font-semibold text-base">
        <PawPrint size={18} className="text-petnutri-orange-light" fill="currentColor" />
        PetNutri
      </div>
    </header>
  )
}
