"""Analytic geometry, route semantics, and optimized/reference equivalence."""
import numpy as np
import unittest
from pothole_measurement.camera import Camera
from pothole_measurement.geometry import (
    _geometry, fit_plane, geometry, mbtp_xy_area, project_area,
    ring_coordinates_original, ring_coordinates_roi,
)

K = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])


def test_roi_ring_exactly_matches_full_opencv_rectangles():
    rng = np.random.default_rng(9231130)
    boxes = [(0,0,0,0), (-10,-20,-2,-1), (90,90,100,110),
             (.5,.5,9.5,10.5), (-.5,-1.5,8.5,12.5), (20,30,10,15)]
    boxes += [tuple(rng.uniform(-40, 120, 4)) for _ in range(500)]
    for box in boxes:
        reference = ring_coordinates_original((63,81), box)
        optimized = ring_coordinates_roi((63,81), box)
        assert all(np.array_equal(a,b) for a,b in zip(reference, optimized)), box


def test_robust_plane_rejects_outliers_and_preserves_sampling():
    rng = np.random.default_rng(29)
    xy = rng.uniform(-2, 2, (500, 2))
    points = np.c_[xy, 4+.2*xy[:,0]-.1*xy[:,1]]
    points = np.r_[points, rng.uniform(-10, 10, (100, 3))]
    fitted = fit_plane(points, np.random.default_rng(1130))
    assert fitted is not None
    normal, offset = fitted
    assert np.max(np.abs(points[:500]@normal+offset)) < .03
    again = fit_plane(points, np.random.default_rng(1130))
    assert np.array_equal(normal, again[0])
    assert offset == again[1]
    assert fit_plane(points[:99], np.random.default_rng(1130)) is None


def test_constant_depth_matches_analytic_enclosing_footprint():
    depth = np.full((480,640), 4.)
    box = (220,160,420,320)
    contour = np.array([[220,160], [420,160], [420,320], [220,320]])
    projected = project_area(contour, np.array([0.,0,1]), -4., K)
    dense = mbtp_xy_area(depth, box, K)*4/np.pi
    result = geometry(depth, box, K)
    assert np.isclose(projected, 2.048, atol=1e-12, rtol=0)
    assert np.isclose(dense, projected, atol=1e-12, rtol=0)
    assert result['bbox_route'] == 'plane'
    assert result['bbox_valid']
    assert np.isclose(result['bbox_raw'], projected, atol=1e-12, rtol=0)


def test_dense_support_gate_remains_required_even_if_plane_succeeds():
    depth = np.full((480,640), 4.)
    depth[230,300] = np.nan  # Inside bbox, exterior ring remains intact.
    result = geometry(depth, (220,160,420,320), K)
    assert result['bbox_route'] == 'plane'
    assert np.isclose(result['bbox_raw'], 2.048)
    assert not result['bbox_valid']


def test_optimized_mask_hole_and_fallback_match_reference():
    import cv2
    depth = np.full((480,640), 4.)
    mask = np.zeros(depth.shape, np.uint8)
    cv2.rectangle(mask, (240,180), (400,300), 1, -1)
    cv2.rectangle(mask, (280,220), (340,260), 0, -1)
    inputs = (depth, (220,160,420,320), K, 1130, mask)
    reference = _geometry(*inputs, roi=False, profile=None, camera=None)
    optimized = geometry(*inputs)
    assert optimized == reference
    assert optimized['mask_valid']
    assert optimized['mask_raw'] < 160*120*16/500**2
    sparse = np.zeros_like(depth)
    sparse[160:321,220:421] = 4  # no valid exterior support, dense bbox is valid
    fallback = geometry(sparse, inputs[1], K)
    assert fallback['bbox_route'] == 'fallback'
    assert fallback['bbox_valid']
    assert np.isclose(fallback['bbox_raw'], 2.048)


def test_pinhole_camera_identity_and_resolution_guard():
    camera = Camera(K, np.zeros(4), model='pinhole', image_size=(640,480))
    np.testing.assert_allclose(camera.rectify_bbox((220,160,420,320), K), (220,160,420,320))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'camera expects'):
        camera.matrix((320,240))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'binary'):
        geometry(np.full((480,640), 4.), (220,160,420,320), K, mask=np.full((480,640),255))


def test_rgb_fisheye_cache_reproduces_undistort_image_exactly():
    import cv2
    image = np.random.default_rng(17).integers(0,256,(96,128,3),dtype=np.uint8)
    raw_k = np.array([[100.,0,64],[0,101.,48],[0,0,1]])
    distortion = np.array([-.037439,-.004983,.004292,-.002258])
    camera = Camera(raw_k, distortion, image_size=(128,96))
    k = camera.matrix((128,96))
    original = cv2.fisheye.undistortImage(image, raw_k, distortion, Knew=k)
    optimized = camera.rectify_image(image,k)
    assert np.array_equal(original,optimized)
    # The binary-mask map representation stays separate from the RGB cache.
    assert camera.maps((128,96),k)[0].dtype == np.float32
    assert camera.maps((128,96),k,map_type=cv2.CV_16SC2)[0].dtype == np.int16


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items() if name.startswith("test_") and callable(value))
