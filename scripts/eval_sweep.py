import sys
sys.path.insert(0, r'D:\Lunar Project\scripts')
from generate_synthetic import generate_pair
from app.matching.pipeline import run_pipeline

img1,img2=generate_pair('crater',1024,1024)
for th in [1.5,2.0,3.0,4.0]:
    r=run_pipeline(img1,img2,algorithm='rift',ransac_thresh=th)
    print(f"th={th} kp={r['total_keypoints']} inliers={r['inlier_count']} ratio={r['inlier_ratio']:.3f} rmse={r['rmse']:.3f} uniform={r['uniformity']:.3f} algo={r['algorithm']}")

print("--- mare ---")
for kind in ["mare","sar_pair","crater"]:
    import cv2
    from app.preprocessing.clahe import apply_clahe
    from app.matching.pipeline import texture_score
    a,b=generate_pair(kind,512,512)
    c1=apply_clahe(a); c2=apply_clahe(b)
    print(kind, "raw", round(texture_score(a),1), "clahe", round(texture_score(c1),1), "c2 clahe", round(texture_score(c2),1))
