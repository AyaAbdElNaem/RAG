import { PawPrint, Dog } from 'lucide-react'
import { Link } from 'react-router-dom'
// Drop your own licensed photo at src/assets/hero-dog.jpg and swap the
// placeholder block below for an <img src={dogImage} .../> tag.
// import dogImage from '../assets/hero-dog.jpg'

const HERO_IMG = 'https://images.unsplash.com/photo-1552053831-71594a27632d?auto=format&fit=crop&w=900&q=80'
const HERO_IMG_FALLBACK = 'https://images.pexels.com/photos/58997/pexels-photo-58997.jpeg?auto=compress&cs=tinysrgb&w=900'

export default function LandingPage() {
  return (
    <div className="relative min-h-screen flex flex-col items-center px-6 py-10 overflow-hidden">
      {/* Decorative animated pet graphics — purely visual, so they're
          aria-hidden and pointer-events-none to stay out of the way of
          real interactive elements and screen readers. */}
      <span
        aria-hidden="true"
        className="pointer-events-none select-none absolute top-6 right-6 text-3xl animate-bounce"
      >
        🐾
      </span>
      <span
        aria-hidden="true"
        className="pointer-events-none select-none absolute top-28 left-4 text-2xl animate-wiggle"
      >
        🐶
      </span>
      <span
        aria-hidden="true"
        className="pointer-events-none select-none absolute bottom-10 right-8 text-3xl animate-pulse"
      >
        🐱
      </span>

      <div className="w-full max-w-md relative z-10">
        {/* Logo */}
        <div className="flex items-center gap-2 mb-10">
          <PawPrint className="w-5 h-5 text-clay" strokeWidth={2.5} />
          <span className="font-display font-bold text-lg text-clay tracking-tight">PetNutri</span>
        </div>

        {/* Hero copy */}
        <h1 className="font-display font-extrabold text-4xl leading-tight text-espresso mb-4">
          Personalized Nutrition
          <br />
          for a{' '}
          <span className="text-petnutri-orange-light">Happier Paw</span>
          <span aria-hidden="true" className="inline-block ml-2 text-3xl align-middle animate-wiggle">🐾</span>
        </h1>

        <p className="text-espresso/70 text-base leading-relaxed mb-8 max-w-sm">
          Every pet is unique. Our AI-driven platform crafts perfectly balanced diets tailored
          to your companion's breed, age, and health needs.
        </p>

        {/* CTA */}
        <Link
          to="/chat"
          className="block w-full text-center bg-petnutri-green hover:bg-petnutri-green-dark transition-colors text-white font-medium text-base py-4 rounded-full shadow-sm mb-8"
        >
          Start AI Consultation
        </Link>

        {/* Hero image — replace with your own licensed pet photo */}
        {/* <div className="rounded-2xl overflow-hidden shadow-md aspect-[4/3] bg-gradient-to-br from-petnutri-green/10 to-petnutri-orange-light/10 border border-petnutri-muted/10 flex items-center justify-center">
          <Dog size={72} strokeWidth={1.2} className="text-petnutri-green/40" />
          
        </div> */}
                <div className="rounded-3xl overflow-hidden shadow-lg shadow-espresso/10 border border-peach-border">
          <img
            src={HERO_IMG}
            onError={(e) => { e.currentTarget.onerror = null; e.currentTarget.src = HERO_IMG_FALLBACK }}
            alt="Happy golden retriever relaxing at home"
            className="w-full h-72 object-cover"
         />
        </div>
      </div>
    </div>
  )
}


