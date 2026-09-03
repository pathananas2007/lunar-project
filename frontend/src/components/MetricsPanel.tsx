export default function MetricsPanel({ data }: { data: any }) {
  if (!data) return <div className="bg-[#111827] rounded-xl p-4 border border-white/10 text-sm text-white/50">No run yet. Execute alignment.</div>
  const m = data
  const ok = m.rmse < 1.0 && m.inlier_ratio > 0.5
  return (
    <div className="bg-[#111827] rounded-xl p-4 border border-white/10 space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-cyan-300">Metrics</h3>
        <span className={`text-xs px-2 py-1 rounded ${ok ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'}`}>{ok ? 'PASS' : 'REVIEW'}</span>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-black/30 rounded p-2"><div className="text-white/50">Algorithm</div><div className="font-mono font-bold">{m.algorithm}</div></div>
        <div className="bg-black/30 rounded p-2"><div className="text-white/50">Latency</div><div className="font-mono font-bold">{m.latency_ms?.toFixed(0)} ms</div></div>
        <div className="bg-black/30 rounded p-2"><div className="text-white/50">Total KP</div><div className="font-mono font-bold">{m.total_keypoints}</div></div>
        <div className="bg-black/30 rounded p-2"><div className="text-white/50">Inliers</div><div className="font-mono font-bold">{m.inlier_count} ({(m.inlier_ratio*100).toFixed(1)}%)</div></div>
        <div className="bg-black/30 rounded p-2"><div className="text-white/50">RMSE</div><div className={`font-mono font-bold ${m.rmse < 0.5 ? 'text-emerald-300' : m.rmse < 1 ? 'text-amber-300' : 'text-red-400'}`}>{m.rmse?.toFixed(3)} px</div></div>
        <div className="bg-black/30 rounded p-2"><div className="text-white/50">Uniformity</div><div className="font-mono font-bold">{(m.uniformity*100).toFixed(1)}%</div></div>
      </div>
      {m.homography && (
        <div className="text-[10px] font-mono bg-black/30 rounded p-2 overflow-x-auto">
          H = [{m.homography[0].map((v:number)=> v.toFixed(3)).join(', ')}]<br/>
          &nbsp;&nbsp;&nbsp;[{m.homography[1].map((v:number)=> v.toFixed(3)).join(', ')}]<br/>
          &nbsp;&nbsp;&nbsp;[{m.homography[2].map((v:number)=> v.toFixed(3)).join(', ')}]
        </div>
      )}
      <div className="text-[10px] text-white/30">Target: RMSE &lt;1.0px, Inlier &gt;75% multi-modal, Latency &lt;5s CPU</div>
    </div>
  )
}
