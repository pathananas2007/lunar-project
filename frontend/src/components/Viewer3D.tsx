import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import * as THREE from 'three'
import { useMemo, useEffect, useState } from 'react'

function Terrain({ heightScale = 6, heightMap }: { heightScale?: number; heightMap?: string | null }) {
  const [displacement, setDisplacement] = useState<number[] | null>(null)

  // Dynamic: if heightMap (dataURL) provided, sample it for displacement; else fallback synthetic
  useEffect(() => {
    if (!heightMap) {
      setDisplacement(null)
      return
    }
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.onload = () => {
      const canvas = document.createElement('canvas')
      const size = 128
      canvas.width = size
      canvas.height = size
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0, size, size)
      const data = ctx.getImageData(0, 0, size, size).data
      const heights: number[] = []
      for (let i = 0; i < size * size; i++) {
        const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2]
        // Luminance -> height, invert so craters (dark) are wells
        const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        // Normalize around 0.5, scale to -1..1, with crater exaggeration
        let h = (0.5 - lum) * 2.5
        // Add contrast for rims
        if (lum > 0.55) h *= 0.6 // highlands lower
        heights.push(h)
      }
      setDisplacement(heights)
    }
    img.onerror = () => setDisplacement(null)
    img.src = heightMap
  }, [heightMap])

  const geom = useMemo(() => {
    const g = new THREE.PlaneGeometry(20, 20, 128, 128)
    const pos = g.attributes.position as THREE.BufferAttribute
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i)
      let h = 0
      if (displacement) {
        h = displacement[i] * heightScale * 0.18
      } else {
        const t = Date.now() * 0.0001
        const craters: [number, number, number][] = [
          [-4 + Math.sin(t) * 0.2, -3, 2.5],
          [3, 4 + Math.cos(t) * 0.2, 1.8],
          [0, 0, 3.2]
        ]
        for (const [cx, cy, r] of craters) {
          const d = Math.hypot(x - cx, y - cy)
          if (d < r) h -= Math.cos((d / r) * Math.PI / 2) * 1.2
          else if (d < r * 1.4) h += 0.15 * Math.exp(-(d - r))
        }
        h += (Math.sin(x * 3) * Math.cos(y * 3)) * 0.05
        h = h * heightScale * 0.15
      }
      pos.setZ(i, h)
    }
    g.computeVertexNormals()
    return g
  }, [heightScale, displacement])

  // ISRO fix: dispose geometry on unmount / heightMap change to prevent leak
  useEffect(() => {
    return () => { geom.dispose() }
  }, [geom])

  // Dynamic material: use heightMap as texture if available
  const texture = useMemo(() => {
    if (!heightMap) return null
    const loader = new THREE.TextureLoader()
    const tex = loader.load(heightMap)
    tex.colorSpace = THREE.SRGBColorSpace
    return tex
  }, [heightMap])

  useEffect(() => {
    return () => { if (texture) texture.dispose() }
  }, [texture])

  return (
    <mesh geometry={geom} rotation={[-Math.PI / 2, 0, 0]}>
      {texture ? (
        <meshStandardMaterial map={texture} roughness={0.85} metalness={0.05} />
      ) : (
        <meshStandardMaterial color="#8a8d93" roughness={0.9} metalness={0.05} />
      )}
    </mesh>
  )
}

function OverlayTexture({ overlayMap, opacity = 0.55 }: { overlayMap?: string | null; opacity?: number }) {
  const texture = useMemo(() => {
    if (!overlayMap) return null
    const loader = new THREE.TextureLoader()
    const tex = loader.load(overlayMap)
    tex.colorSpace = THREE.SRGBColorSpace
    return tex
  }, [overlayMap])

  const geom = useMemo(() => new THREE.PlaneGeometry(20, 20), [])
  useEffect(() => {
    return () => { geom.dispose(); if (texture) texture.dispose() }
  }, [geom, texture])

  if (!texture) return null
  return (
    <mesh geometry={geom} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
      <meshBasicMaterial map={texture} transparent opacity={opacity} depthWrite={false} />
    </mesh>
  )
}

function MatchLines({ correspondences, imageSize = 1024 }: { correspondences: any[]; imageSize?: number }) {
  const subset = useMemo(() => correspondences.slice(0, 140), [correspondences])
  if (!subset.length) return null
  const toWorld = (x: number, y: number) => [(x / imageSize) * 20 - 10, 10 - (y / imageSize) * 20] as const
  // Pre-create geometries once per subset to allow dispose
  const lines = useMemo(() => {
    return subset.map(c => {
      const [x1, y1] = toWorld(c.x1, c.y1)
      const [x2, y2] = toWorld(c.x2, c.y2)
      const color = c.inlier ? (c.confidence > 0.8 ? '#10b981' : '#facc15') : '#ef4444'
      const opacity = c.inlier ? 0.88 : 0.22
      const pts = [new THREE.Vector3(x1, 0.25, y1), new THREE.Vector3(x2, 0.35, y2)]
      const geom = new THREE.BufferGeometry().setFromPoints(pts)
      const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity })
      return { geom, mat }
    })
  }, [subset, imageSize])
  useEffect(() => {
    return () => { lines.forEach(l => { l.geom.dispose(); l.mat.dispose() }) }
  }, [lines])
  return (
    <group>
      {lines.map((l, i) => (
        <primitive key={i} object={new THREE.Line(l.geom, l.mat)} />
      ))}
    </group>
  )
}

export default function Viewer3D({ result, imageUrls }: { result: any; imageUrls?: { base?: string | null; overlay?: string | null } }) {
  const corrs = result?.correspondences || []
  const imgSize = result?.warped_shape ? result.warped_shape[1] : result?.homography ? 1024 : 1024
  const heightMap = imageUrls?.base || null
  const overlayMap = imageUrls?.overlay || null

  return (
    <div className="w-full h-[520px] bg-black rounded-xl border border-white/10 overflow-hidden relative">
      <Canvas camera={{ position: [8, 10, 8], fov: 45 }}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[5, 10, 5]} intensity={1.2} />
        <Terrain heightMap={heightMap} />
        {overlayMap && <OverlayTexture overlayMap={overlayMap} opacity={0.5} />}
        <Grid position={[0, -0.6, 0]} args={[20, 20]} cellColor="#1f2937" sectionColor="#38bdf8" fadeDistance={22} />
        {corrs.length > 0 && <MatchLines correspondences={corrs} imageSize={imgSize} />}
        <OrbitControls enableDamping minDistance={4} maxDistance={28} />
      </Canvas>
      <div className="absolute top-2 left-2 text-[10px] bg-black/60 px-2 py-1 rounded border border-white/10">
        {heightMap ? "Dynamic DEM from uploaded image" : "Synthetic DEM"} — Drag to orbit · Scroll to zoom · <span className="text-emerald-300">inlier</span> / <span className="text-red-400">outlier</span>
      </div>
      {overlayMap && <div className="absolute top-2 right-2 text-[10px] bg-cyan-500/20 text-cyan-200 px-2 py-1 rounded">Overlay: warped source (dynamic)</div>}
      {result && (
        <div className="absolute bottom-2 left-2 right-2 flex gap-2 text-[10px]">
          <span className="bg-emerald-500/20 text-emerald-200 px-2 py-1 rounded">Green: confident &gt;0.8</span>
          <span className="bg-amber-500/20 text-amber-200 px-2 py-1 rounded">Yellow: threshold</span>
          <span className="bg-white/10 px-2 py-1 rounded ml-auto">{corrs.filter((c: any) => c.inlier).length} / {corrs.length} inliers · {result.latency_ms?.toFixed(0)} ms · {result.algorithm}</span>
        </div>
      )}
    </div>
  )
}
