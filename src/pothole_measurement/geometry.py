"""CPU road-plane footprint estimation from the authors' frozen pipeline.

The fit, interpolation, ring order, dense validity gate, and fallback formulas
are preserved from p1130_area_validation; see docs/implementation.md.
Depth must be on the rectified grid; boxes/masks are raw when camera is given,
and already rectified otherwise. Depth units determine raw area units.
"""
from __future__ import annotations
from time import perf_counter
import cv2
import numpy as np

def fit_plane(points, rng, trials=75, threshold=0.20):
    if len(points)<100:return None
    if len(points)>600: points=points[rng.choice(len(points),600,replace=False)]
    best=None; bestn=0
    for _ in range(trials):
        a,b,c=points[rng.choice(len(points),3,replace=False)]; n=np.cross(b-a,c-a); norm=np.linalg.norm(n)
        if norm<1e-8:continue
        n/=norm; d=-n@a; keep=np.abs(points@n+d)<threshold
        if keep.sum()>bestn:best,bestn=(n,d,keep),keep.sum()
    if best is None or bestn<80:return None
    q=points[best[2]]; centre=q.mean(axis=0); _,_,vh=np.linalg.svd(q-centre,full_matrices=False); n=vh[-1];n/=np.linalg.norm(n);return n,-n@centre

def mbtp_xy_area(depth: np.ndarray, box: tuple[float, float, float, float], k: np.ndarray) -> float | None:
    """Paper Eqs. (6)--(12), using the paper's X-Y triangle formula."""
    h, w = depth.shape
    x0, y0, x1, y1 = box
    # A detection box is discrete in image coordinates.  Clamp the inclusive
    # grid so that adjacent 2x2 pixels always exist.
    u0 = max(0, int(np.floor(x0))); u1 = min(w - 1, int(np.ceil(x1)))
    v0 = max(0, int(np.floor(y0))); v1 = min(h - 1, int(np.ceil(y1)))
    if u1 - u0 < 1 or v1 - v0 < 1:
        return None
    z = depth[v0:v1 + 1, u0:u1 + 1].astype(np.float64)
    if not np.isfinite(z).all() or np.any(z <= 0):
        return None
    u, v = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
    x = (u - k[0, 2]) * z / k[0, 0]
    y = (v - k[1, 2]) * z / k[1, 1]
    # P0=(u,v); P1=(u+1,v); P2=(u,v+1); P3=(u+1,v+1)
    x0g, y0g = x[:-1, :-1], y[:-1, :-1]
    x1g, y1g = x[:-1, 1:], y[:-1, 1:]
    x2g, y2g = x[1:, :-1], y[1:, :-1]
    x3g, y3g = x[1:, 1:], y[1:, 1:]
    a1 = 0.5 * np.abs((x1g - x0g) * (y2g - y0g) - (y1g - y0g) * (x2g - x0g))
    a2 = 0.5 * np.abs((x2g - x0g) * (y3g - y0g) - (y2g - y0g) * (x3g - x0g))
    return float((a1 + a2).sum() * np.pi / 4.0)

def project_area(contour,n,d,k):
 p=contour.reshape(-1,2);rays=np.c_[(p[:,0]-k[0,2])/k[0,0],(p[:,1]-k[1,2])/k[1,1],np.ones(len(p))];t=-d/(rays@n)
 if np.any(~np.isfinite(t)) or np.any(t<=0):return None
 v=rays*t[:,None];helper=np.array([1.,0,0]) if abs(n[0])<=.9 else np.array([0.,1.,0]);u=np.cross(n,helper);u/=np.linalg.norm(u);w=np.cross(n,u);q=np.c_[v@u,v@w]
 return float(.5*abs(np.dot(q[:,0],np.roll(q[:,1],-1))-np.dot(q[:,1],np.roll(q[:,0],-1))))

def ring_coordinates_original(shape, box):
    """Reference ring construction, exposed for direct coordinate checks."""
    h, w = shape
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)/2, (y0+y1)/2
    ow, oh = (x1-x0)*1.5, (y1-y0)*1.5
    inner = np.zeros((h, w), np.uint8)
    outer = np.zeros((h, w), np.uint8)
    cv2.rectangle(inner, (round(x0), round(y0)), (round(x1), round(y1)), 1, -1)
    cv2.rectangle(outer, (round(cx-ow/2), round(cy-oh/2)),
                  (round(cx+ow/2), round(cy+oh/2)), 1, -1)
    v, u = np.nonzero(outer.astype(bool) & ~inner.astype(bool))
    take = (u % 2 == 0) & (v % 2 == 0)
    return u[take], v[take]

def ring_coordinates_roi(shape, box):
    """Return exactly the old even-coordinate ring in global row-major order.

    Filled OpenCV rectangles include both rounded endpoints. Clipping occurs
    before constructing only the outer rectangle's even-coordinate grid. The
    row-major nonzero order is the same as scanning the full image first and
    decimating later, which preserves fit_plane's random sample indices.
    """
    h, w = shape
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)/2, (y0+y1)/2
    ow, oh = (x1-x0)*1.5, (y1-y0)*1.5
    ix0, ix1 = sorted((round(x0), round(x1)))
    iy0, iy1 = sorted((round(y0), round(y1)))
    ox0, ox1 = sorted((round(cx-ow/2), round(cx+ow/2)))
    oy0, oy1 = sorted((round(cy-oh/2), round(cy+oh/2)))
    ox0, ox1 = max(0, ox0), min(w-1, ox1)
    oy0, oy1 = max(0, oy0), min(h-1, oy1)
    if ox0 > ox1 or oy0 > oy1:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    ox0 += ox0 % 2
    oy0 += oy0 % 2
    cols = np.arange(ox0, ox1+1, 2, dtype=np.int64)
    rows = np.arange(oy0, oy1+1, 2, dtype=np.int64)
    ring = ((rows[:, None] < iy0) | (rows[:, None] > iy1)
            | (cols[None, :] < ix0) | (cols[None, :] > ix1))
    vr, ur = np.nonzero(ring)
    return cols[ur], rows[vr]

def _geometry(depth, rawbox, k, seed, mask, *, roi, profile, camera):
    tic = perf_counter() if profile is not None else None

    def stamp(name):
        nonlocal tic
        if profile is not None:
            now = perf_counter()
            profile[name] = profile.get(name, 0.0) + (now-tic)*1000.0
            tic = now

    h, w = depth.shape
    box = camera.rectify_bbox(tuple(rawbox), k) if camera is not None else tuple(rawbox)
    x0, y0, x1, y1 = box
    stamp("bbox_rectify_ms")
    u, v = (ring_coordinates_roi if roi else ring_coordinates_original)(depth.shape, box)
    z = depth[v, u]
    ok = np.isfinite(z) & (z > 0)
    u, v, z = u[ok], v[ok], z[ok]
    pts = np.c_[(u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z]
    stamp("ring_prepare_ms")
    plane = fit_plane(pts, np.random.default_rng(seed))
    stamp("fit_plane_ms")
    bp = None
    if plane is not None:
        bp = project_area(np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1]]), *plane, k)
    stamp("bbox_projection_ms")
    # Preserve the original dense validity gate even when the plane succeeds.
    dense = mbtp_xy_area(depth, box, k)
    dense = None if dense is None else dense*4/np.pi
    valid = bool(dense is not None and np.isfinite(dense) and dense > 0)
    route = 'plane' if bp is not None and bp > 0 else 'fallback'
    raw = bp if route == 'plane' else dense
    result = {
        'bbox_valid': valid, 'bbox_route': route,
        'bbox_raw': None if raw is None else float(raw),
        'mask_valid': False, 'mask_route': None, 'mask_raw': None,
        'mask_pixels': None,
    }
    stamp("bbox_dense_ms")
    if mask is None:
        return result
    if mask.shape != depth.shape:
        raise ValueError("mask and depth must have the same spatial shape")
    maps = camera.maps((w, h), k) if camera is not None else None
    stamp("mask_map_lookup_ms")
    rm = (cv2.remap(mask.astype(np.uint8, copy=not roi), *maps,
                    cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
          if maps is not None else mask.astype(np.uint8, copy=not roi))
    stamp("mask_remap_ms")
    # sumElems returns the same integer sum for practical uint8 image sizes.
    # Keep numpy's exact original operation in the cache-only ablation.
    result['mask_pixels'] = int(cv2.sumElems(rm)[0]) if roi else int(rm.sum())
    occupied = result['mask_pixels'] > 0
    stamp("mask_scan_ms")
    mp = None
    if route == 'plane' and (occupied if roi else rm.any()):
        contours, hierarchy = cv2.findContours(rm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        total = 0.
        allok = True
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                if len(contour) < 3:
                    continue
                a = project_area(contour, *plane, k)
                if a is None:
                    allok = False
                    break
                nesting = 0
                parent = int(hierarchy[0, i, 3])
                while parent >= 0:
                    nesting += 1
                    parent = int(hierarchy[0, parent, 3])
                total += (-1 if nesting % 2 else 1)*a
            if allok and total > 0:
                mp = total
    stamp("mask_contour_projection_ms")
    if mp is None and (occupied if roi else rm.any()):
        if roi:
            x, y, bw, bh = cv2.boundingRect(rm)
            xa, xb = max(0, x-1), min(w-1, x+bw)
            ya, yb = max(0, y-1), min(h-1, y+bh)
        else:
            vv, uu = np.nonzero(rm)
            xa, xb = max(0, int(uu.min())-1), min(w-1, int(uu.max())+1)
            ya, yb = max(0, int(vv.min())-1), min(h-1, int(vv.max())+1)
        zz = depth[ya:yb+1, xa:xb+1]
        u2, v2 = np.meshgrid(np.arange(xa, xb+1), np.arange(ya, yb+1))
        xx = (u2-k[0, 2])*zz/k[0, 0]
        yy = (v2-k[1, 2])*zz/k[1, 1]
        a = .5*np.abs((xx[:-1,1:]-xx[:-1,:-1])*(yy[1:,:-1]-yy[:-1,:-1])
                     -(yy[:-1,1:]-yy[:-1,:-1])*(xx[1:,:-1]-xx[:-1,:-1]))
        b = .5*np.abs((xx[1:,:-1]-xx[:-1,:-1])*(yy[1:,1:]-yy[:-1,:-1])
                     -(yy[1:,:-1]-yy[:-1,:-1])*(xx[1:,1:]-xx[:-1,:-1]))
        total = float(((a+b)*rm[ya:yb, xa:xb]).sum())
        if np.isfinite(total) and total > 0:
            mp = total
        mr = 'fallback'
    else:
        mr = 'plane'
    result.update(
        mask_valid=bool(mp is not None and np.isfinite(mp) and mp > 0),
        mask_route=mr, mask_raw=None if mp is None else float(mp),
    )
    stamp("mask_dense_fallback_ms")
    return result

def geometry(depth, rawbox, k, seed=1130, mask=None, *, camera=None, profile=None):
    """Return bbox and optional comparison-mask footprints and validity flags.

    The paper adopts bbox_raw after route-specific calibration. A valid plane
    result retains the original dense-support gate for bbox_valid.
    """
    depth = np.asarray(depth)
    k = np.asarray(k, dtype=np.float64)
    box = np.asarray(rawbox, dtype=np.float64)
    if depth.ndim != 2 or min(depth.shape) < 2:
        raise ValueError("depth must be a two-dimensional array of at least 2 by 2")
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("box must be finite xyxy coordinates with positive width and height")
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("k must be a finite 3 by 3 camera matrix with positive focal lengths")
    if mask is not None and not np.isin(mask, [0, 1]).all():
        raise ValueError("mask must be binary 0/1 (convert 0/255 images before calling)")
    return _geometry(depth, box, k, seed, mask, roi=True, profile=profile, camera=camera)
