import numpy as np
import unittest
from pothole_measurement.calibration import calibrated_area, fit_calibration


def test_log_fit_uses_training_only_and_keeps_routes_separate():
    records = [dict(split='train', bbox_valid=True, bbox_route='plane', bbox_raw=x,
                    reference_m2=2*x**.8) for x in (.1,.2,.4,.8)]
    records += [dict(split='test', bbox_valid=True, bbox_route='plane', bbox_raw=.2,
                     reference_m2=99999)]
    coefficients = fit_calibration(records)
    np.testing.assert_allclose(coefficients['bbox_plane'], [np.log(2), .8])
    assert coefficients['bbox_plane_train_n'] == 4
    assert 'bbox_fallback' not in coefficients
    geo = dict(bbox_valid=True, bbox_route='plane', bbox_raw=.3)
    assert np.isclose(calibrated_area(geo, coefficients), 2*.3**.8)
    assert calibrated_area(dict(geo, bbox_route='fallback'), coefficients) is None
    assert calibrated_area(dict(geo, bbox_valid=False), coefficients) is None
    with unittest.TestCase().assertRaisesRegex(ValueError, 'split=train'):
        fit_calibration([dict(records[0], split='test')])


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items() if name.startswith("test_") and callable(value))
