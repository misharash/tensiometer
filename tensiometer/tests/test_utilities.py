###############################################################################
# initial imports:

import unittest

import tensiometer.utilities as ttu

import numpy as np

###############################################################################


class test_confidence_to_sigma(unittest.TestCase):

    def setUp(self):
        pass

    # test against known output:
    def test_from_confidence_to_sigma_result(self):
        result = ttu.from_confidence_to_sigma(np.array([0.68, 0.95, 0.997]))
        known_result = np.array([0.99445788, 1.95996398, 2.96773793])
        assert np.allclose(result, known_result)

    def test_from_sigma_to_confidence_result(self):
        result = ttu.from_sigma_to_confidence(np.array([1., 2., 3.]))
        known_result = np.array([0.68268949, 0.95449974, 0.9973002])
        assert np.allclose(result, known_result)

    # test that one function is the inverse of the other:
    def test_sigma_confidence_inverse(self):
        test_numbers = np.arange(1, 6)
        test_confidence = ttu.from_sigma_to_confidence(test_numbers)
        test_sigma = ttu.from_confidence_to_sigma(test_confidence)
        assert np.allclose(test_numbers, test_sigma)

    # test raises:
    def test_errors(self):
        with self.assertRaises(ValueError):
            ttu.from_confidence_to_sigma(-1.)
        with self.assertRaises(ValueError):
            ttu.from_confidence_to_sigma(2.)
        with self.assertRaises(ValueError):
            ttu.from_sigma_to_confidence(-1.)

###############################################################################


class test_chi2_to_sigma(unittest.TestCase):

    def setUp(self):
        pass

    # test values:
    def test_values(self):
        assert np.allclose(ttu.from_chi2_to_sigma(1., 1.), 1.0)
        assert np.allclose(ttu.from_chi2_to_sigma(20.0, 2.),
                           ttu.from_chi2_to_sigma(20.0, 2., 1000))

    # test raises:
    def test_errors(self):
        with self.assertRaises(ValueError):
            ttu.from_chi2_to_sigma(-2., 2.)
        with self.assertRaises(ValueError):
            ttu.from_chi2_to_sigma(2., -2.)

###############################################################################


class test_KL_decomposition(unittest.TestCase):

    def setUp(self):
        import numpy as np
        pass

    # test values:
    def test_values(self):
        pass

    # test raises:
    def test_errors(self):
        pass

###############################################################################


class test_QR_inverse(unittest.TestCase):

    def setUp(self):
        pass

    # test values:
    def test_values(self):
        pass

    # test raises:
    def test_errors(self):
        pass

###############################################################################


class test_clopper_pearson_binomial_trial(unittest.TestCase):

    def setUp(self):
        pass

    # test values:
    def test_values(self):
        low, high = ttu.clopper_pearson_binomial_trial(1., 2.)

    # test raises:
    def test_errors(self):
        pass

###############################################################################

class test_PDM_vectorization(unittest.TestCase):

    def setUp(self):
        pass

    # test values:
    def test_values(self):
        # generate a random vector between -1, 1 (seeded so reproducible):
        np.random.seed(0)
        # sweep dimensions from low to medium
        for d in range(2, 20):
            num = d*(d+1)//2
            # get some random matrices:
            for i in range(10):
                vec = 2.*np.random.rand(num) -1.
                # get the corresponding PDM matrix:
                mat = vector_to_PDM(vec, d)
                # check that it is positive definite:
                assert np.all(np.linalg.eig(mat)[0] > 0)
                # transform back. This can be different from the previous one
                # because of many discrete symmetries in defining the
                # eigenvectors
                vec2 = PDM_to_vector(mat, d)
                # transform again. This has to be equal, discrete symmetries for
                # eigenvectors do not matter once they are paired with eigenvalues:
                mat2 = vector_to_PDM(vec2, d)
                assert np.allclose(mat, mat2)

    # test raises:
    def test_errors(self):
        pass

###############################################################################


if __name__ == '__main__':
    unittest.main(verbosity=2)
