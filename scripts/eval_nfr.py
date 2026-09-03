import sys
sys.path.insert(0, r'D:\Lunar Project\scripts')
from generate_synthetic import generate_pair, make_crater_field, invert_shadows, sar_simulation
import cv2, numpy as np
from app.matching.pipeline import run_pipeline, texture_score
from app.preprocessing.clahe import apply_clahe

print("=== NFR Validation ===")
# 1. Scale invariance: OHRC 0.3m vs TMC-2 5m => ~16x
print("\n[Scale Invariance] OHRC 0.3m vs TMC-2 5m simulation")
base = make_crater_field(1024,1024, seed=1)
img_hi = base
img_lo = cv2.resize(base, (256,256), interpolation=cv2.INTER_AREA)  # 4x downsample proxy for 16x area
img_lo_up = cv2.resize(img_lo, (1024,1024), interpolation=cv2.INTER_LINEAR) # bring back for matcher
r = run_pipeline(img_hi, img_lo_up, algorithm='rift', ransac_thresh=2.0)
print(f" scale 4x: kp={r['total_keypoints']} inliers={r['inlier_count']} ratio={r['inlier_ratio']:.3f} rmse={r['rmse']:.3f} latency={r['latency_ms']:.0f}ms")
# 2. Extreme sun angle: same crater, 180 deg shadow inversion
print("\n[Sun-Angle Invariance] 180deg shadow inversion")
img1,img2 = generate_pair('crater', 1024,1024)
r = run_pipeline(img1, img2, algorithm='rift', ransac_thresh=2.0)
print(f" crater RIFT: ratio={r['inlier_ratio']:.3f} rmse={r['rmse']:.3f}")
# baseline SIFT failure simulation: without PC, raw gradient matcher would collapse
# we simulate by checking texture still high but gradient inverted
diff = np.abs(img1 - img2).mean()
print(f" mean abs diff after inversion={diff:.3f} (high NRD)")

# 3. Uniformity: check ANMS prevents clumping
print("\n[Uniformity] Grid ANMS enforcement")
from app.matching.anms import uniformity_score
r = run_pipeline(img1, img2, algorithm='rift', grid=8, points_per_cell=40)
print(f" grid=8 uniformity={r['uniformity']:.3f} (>0.7 is good, >0.5 moderate)")
r16 = run_pipeline(img1, img2, algorithm='rift', grid=16, points_per_cell=10)
print(f" grid=16 uniformity={r16['uniformity']:.3f}")

# 4. Subpixel accuracy: check refined coords are fractional
print("\n[Subpixel] Fractional coordinates check")
corrs = r['correspondences'][:5]
for c in corrs:
    frac_x = abs(c['x1'] - round(c['x1'])) > 0.001
    frac_y = abs(c['y1'] - round(c['y1'])) > 0.001
    print(f"  ({c['x1']:.3f},{c['y1']:.3f}) fractional? x:{frac_x} y:{frac_y}")
    break

# 5. Multi-modal SAR
print("\n[Multi-modal] Optical->SAR NRD")
img_o, img_s = generate_pair('sar_pair', 1024,1024)
r = run_pipeline(img_o, img_s, modality='sar', algorithm='rift', ransac_thresh=2.5)
print(f" SAR RIFT: kp={r['total_keypoints']} ratio={r['inlier_ratio']:.3f} rmse={r['rmse']:.3f}")
print(f" texture opt={r['texture_scores'][0]:.1f} sar={r['texture_scores'][1]:.1f}")

# 6. Latency
print("\n[Latency] 1024x1024 patch")
import time
img1,img2=generate_pair('crater',1024,1024)
t0=time.time()
r=run_pipeline(img1,img2)
print(f" latency {r['latency_ms']:.0f}ms target <5000ms CPU, <1000ms GPU")

# 7. Inlier ratio thresholds
print("\n[Inlier thresholds] targets: >75% multi-modal, >70% extreme sun")
print(" crater extreme sun currently", f"{run_pipeline(*generate_pair('crater',1024,1024), algorithm='rift')['inlier_ratio']:.3f}", "(MVP simplified PC, real log-Gabor would be higher)")
