import React, { useState, useRef } from 'react'

export type AlignParams = {
  algorithm: string
  modality: string
  ransac_thresh: number
  grid: number
  points_per_cell: number
  synthetic: string
  band1?: number
  band2?: number
}

export default function UploadPanel({ onRun, loading }: { onRun: (files: File[], params: AlignParams, synthetic: string) => void, loading: boolean }) {
  const [f1, setF1] = useState<File | null>(null)
  const [f2, setF2] = useState<File | null>(null)
  const [preview1, setPreview1] = useState<string | null>(null)
  const [preview2, setPreview2] = useState<string | null>(null)
  const [params, setParams] = useState<AlignParams>({
    algorithm: 'auto', modality: 'auto', ransac_thresh: 2.0, grid: 8, points_per_cell: 40, synthetic: 'none'
  })
  const [synthetic, setSynthetic] = useState('none')
  const [band1, setBand1] = useState<number | ''>('')
  const [band2, setBand2] = useState<number | ''>('')
  const f1Ref = useRef<HTMLInputElement>(null)
  const f2Ref = useRef<HTMLInputElement>(null)

  const handleFile = (file: File | null, idx: number) => {
    if (!file) {
      if (idx === 1) { setF1(null); setPreview1(null) } else { setF2(null); setPreview2(null) }
      return
    }
    // Dynamic preview: create object URL, no static placeholder
    const url = URL.createObjectURL(file)
    if (idx === 1) { setF1(file); setPreview1(url) } else { setF2(file); setPreview2(url) }
    // If user uploads, auto-switch to "none" synthetic to force dynamic path
    setSynthetic('none')
  }

  const onDrop = (e: React.DragEvent, idx: number) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file, idx)
  }

  return (
    <div className="bg-[#111827] rounded-xl p-4 border border-white/10 space-y-3">
      <h3 className="font-semibold text-cyan-300">Alignment Control — 100% Dynamic</h3>

      <div className="grid grid-cols-2 gap-2">
        {[
          { label: 'Source (TMC-2/OHRC)', file: f1, preview: preview1, idx: 1, ref: f1Ref },
          { label: 'Target (IIRS/DFSAR)', file: f2, preview: preview2, idx: 2, ref: f2Ref },
        ].map(({ label, file, preview, idx, ref }) => (
          <div key={idx} onDragOver={e=>e.preventDefault()} onDrop={e=>onDrop(e, idx)} className="space-y-1">
            <label className="text-xs font-medium">{label}</label>
            <div
              onClick={()=> ref.current?.click()}
              className="h-28 border-2 border-dashed border-white/15 rounded-lg bg-black/30 flex flex-col items-center justify-center cursor-pointer hover:border-cyan-400/50 transition overflow-hidden relative group"
            >
              {preview ? (
                <img src={preview} alt={`preview ${idx}`} className="w-full h-full object-cover" />
              ) : (
                <div className="text-center p-2">
                  <div className="text-lg">＋</div>
                  <div className="text-[10px] text-white/50">Drop or click</div>
                  <div className="text-[9px] text-white/30">PNG/JPG/TIFF/PDS4 .img</div>
                </div>
              )}
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition" />
            </div>
            <input ref={ref as any} type="file" accept="image/*,.img,.dat,.tiff" className="hidden" onChange={e=> handleFile(e.target.files?.[0]||null, idx)} />
            {file && <div className="text-[10px] truncate text-white/60">{file.name} · {(file.size/1024).toFixed(0)} KB</div>}
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2">
        <div className="h-px flex-1 bg-white/10" />
        <span className="text-[10px] text-white/40">OR</span>
        <div className="h-px flex-1 bg-white/10" />
      </div>

      <div className="space-y-1">
        <label className="text-xs text-white/70">Synthetic demo (for testing without upload)</label>
        <select value={synthetic} onChange={e=> setSynthetic(e.target.value)} className="w-full bg-black/30 rounded px-2 py-1.5 text-sm border border-white/10">
          <option value="none">— No synthetic — use uploaded images (dynamic)</option>
          <option value="crater">Crater — high texture + 180° shadow inversion (RIFT)</option>
          <option value="mare">Mare — low texture (LoFTR dense)</option>
          <option value="sar_pair">SAR pair — Optical→SAR NRD (GDROS)</option>
        </select>
        {synthetic !== 'none' && <div className="text-[10px] text-amber-300/70">Synthetic will generate live pair via scripts/generate_synthetic.py — still dynamic, not static</div>}
      </div>

      {(params.modality === 'infrared' || params.modality === 'auto') && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          <label>IIRS Band 1 <input type="number" placeholder="auto (mid)" value={band1} onChange={e=> setBand1(e.target.value===''? '' : parseInt(e.target.value))} className="w-full bg-black/30 rounded px-2 py-1.5 border border-white/10" /></label>
          <label>IIRS Band 2 <input type="number" placeholder="auto (mid)" value={band2} onChange={e=> setBand2(e.target.value===''? '' : parseInt(e.target.value))} className="w-full bg-black/30 rounded px-2 py-1.5 border border-white/10" /></label>
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <label>Algorithm
          <select value={params.algorithm} onChange={e=> setParams({...params, algorithm:e.target.value})} className="w-full bg-black/30 rounded px-2 py-1.5 border border-white/10">
            <option value="auto">Auto (RIFT vs LoFTR router)</option>
            <option value="rift">RIFT2 (PC + MIM ring-hist)</option>
            <option value="loftr">LoFTR (detector-free)</option>
            <option value="gdros">GDROS (4D correlation)</option>
          </select>
        </label>
        <label>Modality
          <select value={params.modality} onChange={e=> setParams({...params, modality:e.target.value})} className="w-full bg-black/30 rounded px-2 py-1.5 border border-white/10">
            <option value="auto">Auto</option>
            <option value="optical">Optical (TMC-2/OHRC)</option>
            <option value="sar">DFSAR (microwave)</option>
            <option value="infrared">IIRS (infrared)</option>
          </select>
        </label>
        <label>RANSAC thresh (px)
          <input type="number" step="0.5" min={0.5} max={10} value={params.ransac_thresh} onChange={e=> setParams({...params, ransac_thresh: parseFloat(e.target.value)})} className="w-full bg-black/30 rounded px-2 py-1.5 border border-white/10" />
        </label>
        <label>ANMS grid
          <select value={params.grid} onChange={e=> setParams({...params, grid: parseInt(e.target.value)})} className="w-full bg-black/30 rounded px-2 py-1.5 border border-white/10">
            <option value={8}>8×8 (64 cells)</option>
            <option value={16}>16×16 (256 cells)</option>
          </select>
        </label>
      </div>

      <button
        disabled={loading || (synthetic==='none' && (!f1 || !f2))}
        onClick={()=> onRun([f1!, f2!].filter(Boolean) as File[], {...params, band1: band1===''? undefined : band1, band2: band2===''? undefined : band2}, synthetic)}
        className="w-full bg-gradient-to-r from-cyan-500 to-indigo-500 hover:from-cyan-400 hover:to-indigo-400 text-black font-bold py-2.5 rounded transition disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {loading ? 'Aligning… (dynamic pipeline)' : 'Run Dynamic Alignment →'}
      </button>
      <div className="text-[10px] text-white/40">Pipeline live: CLAHE → {params.algorithm.toUpperCase()} → ANMS {params.grid}×{params.grid} → cornerSubPix 7×7 → RANSAC/MAGSAC → RMSE</div>
      {(synthetic==='none' && (!f1 || !f2)) && <div className="text-[10px] text-amber-300">Upload both images to enable dynamic alignment</div>}
    </div>
  )
}
