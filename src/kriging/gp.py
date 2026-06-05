"""
Gaussian Process model for the kriging calibration.

Kept dependency-light (only torch + gpytorch) so it can be imported on its own —
e.g. by examples/minimal_kriging_example.py — without pulling in the geospatial
stack (rasterio / geopandas) that the rest of kriging/core.py requires.
"""
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    # Constant mean + scaled Matern kernel (the configuration used in the paper).
    def __init__(self, train_x, train_y, likelihood, ndims=None, mean_function='constant', kernel_function='matern', matern_nu=0.5):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=matern_nu, ard_num_dims=ndims))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
