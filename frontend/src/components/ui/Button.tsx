import type { ButtonHTMLAttributes, ReactNode } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  icon?: ReactNode
  children?: ReactNode
}

const VARIANT_CLASSES: Record<Variant, string> = {
  primary: 'bg-ink text-canvas hover:bg-ink-muted disabled:bg-ink-subtle',
  secondary:
    'bg-surface text-ink border border-border hover:bg-surface-sunken disabled:text-ink-subtle',
  ghost: 'bg-transparent text-ink-muted hover:bg-surface-sunken hover:text-ink',
  danger: 'bg-transparent text-danger hover:bg-danger-soft',
}

export function Button({
  variant = 'secondary',
  icon,
  children,
  className = '',
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium
        transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-60
        ${VARIANT_CLASSES[variant]} ${className}`}
      disabled={disabled}
      {...rest}
    >
      {icon}
      {children}
    </button>
  )
}
