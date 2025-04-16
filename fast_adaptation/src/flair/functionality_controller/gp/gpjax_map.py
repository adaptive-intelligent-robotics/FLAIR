# Written or Adapted by the Imperial College London Team for the FLAIR project, 2023
# Authors for this file: 
# Maxime Allard
# Manon Flageat

# Copyright 2022 The MOGPJax Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import copy
import os
import time
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax as ox
from gpjax.abstractions import InferenceState, progress_bar_scan
from gpjax.config import get_global_config
from gpjax.gaussian_distribution import GaussianDistribution
from gpjax.gps import AbstractPosterior, AbstractPrior, construct_posterior
from gpjax.kernels import AbstractKernel
from gpjax.likelihoods import AbstractLikelihood
from gpjax.mean_functions import AbstractMeanFunction, Zero
from gpjax.parameters import ParameterState, constrain, trainable_params, unconstrain
from jax.random import KeyArray
from jaxkern.base import AbstractKernel
from jaxlinop import DiagonalLinearOperator, identity
from jaxtyping import Array, Float
from jaxutils import Dataset


class MAPMean(AbstractMeanFunction):
    """
    A zero mean function. This function returns a repeated scalar value for all inputs.
    The scalar value itself can be treated as a model hyperparameter and learned during training.
    """

    def __init__(
        self,
        maps: Sequence[jnp.array],
        ref_map: jnp.array,
        output_dim: Optional[int] = 1,
        chosen_bd: Optional[int] = 0,
        name: Optional[str] = "Mean function",
    ):
        """Initialise the constant-mean function.
        Args:
            output_dim (Optional[int]): The output dimension of the mean function. Defaults to 1.
            name (Optional[str]): The name of the mean function. Defaults to "Mean function".
        """
        self.maps_descriptors = maps
        self.ref_map = ref_map
        self.num_maps = len(maps)
        self.chosen_bd = chosen_bd
        super().__init__(output_dim, name)

    def get_behaviours(
        self,
        input_points: Float[Array, "N D"],
        params: Dict,
        indices: Optional[jnp.array] = None,
        testing=False,
    ):

        # errors = jnp.zeros(shape = (input_points.shape[0], self.output_dim))
        # scaling = params['scaling']
        p1, p2 = params["rotation"]  # /jnp.max(params['rotation'])
        p3, p4 = params["offset"]

        # p1,p2 = jnp.clip(p1,0,1),jnp.clip(p2,0,1)
        # p2 = 1.0
        w = 0.33
        scaling = jnp.asarray(
            [[(p1 + p2) / 2, (p2 - p1) * w / 4], [(p2 - p1) / w, (p1 + p2) / 2]]
        ).T
        constants = jnp.ones(shape=(1, 1))  # params['constant']
        all_maps = copy.deepcopy(self.maps_descriptors)

        offset = jnp.round(jnp.asarray([[p3, p4]]), decimals=1)

        all_maps[0] = jnp.matmul(self.maps_descriptors[0], scaling) + offset
        # jax.tree_map(lambda x: jnp.matmul(x,scaling),self.maps_descriptors)
        # @partial(jax.jit, static_argnames=("k_nn",))
        # @jax.jit
        def _get_cells_indices(
            descriptors: jnp.ndarray,
            centroids: jnp.ndarray,
            k_nn: int,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            set_of_descriptors of shape (1, num_descriptors)
            centroids of shape (num_centroids, num_descriptors)
            """
            distances = jax.vmap(jnp.linalg.norm)(descriptors - centroids)
            ## Negating distances because we want the smallest ones
            min_args = jnp.argmin(distances)
            # min_dist,min_args = jax.lax.top_k(-1*distances,k_nn)
            return min_args  # ,-1*min_dist

        func = jax.vmap(
            _get_cells_indices,
            in_axes=(
                0,
                None,
                None,
            ),
        )

        if testing:
            error_per_map = jax.tree_map(lambda x: (x - self.ref_map), all_maps)
        else:
            if not indices:
                ## get all the indices of the input points (The position of the genotypes needs to be unchanged across all maps)
                # distances = jax.vmap(jnp.linalg.norm)(input_points-self.ref_map)
                indices = func(input_points, self.ref_map, 1)
                # jax.lax.top_k(-distances,k=1)
                ## Get BD from each Map for a Point
                # distances = jax.tree_map(lambda x: jax.vmap(jnp.linalg.norm)(input_points-x),self.maps_descriptors)
                # indices =  jax.tree_map(lambda x: jax.lax.top_k(-x,k=1)[1],distances)
                ## Calulating the movement of the BDs per map
                if input_points.shape[0] > 1:
                    indices = indices.squeeze()
                else:
                    indices = indices.reshape(
                        -1,
                    )
                error_per_map = jax.tree_map(
                    lambda x: (x.at[indices].get() - self.ref_map.at[indices].get()),
                    all_maps,
                )

        # indices,points = func(input_points,self.ref_map,1)
        # # jax.lax.top_k(-distances,k=1)
        # ## Get BD from each Map for a Point
        # # distances = jax.tree_map(lambda x: jax.vmap(jnp.linalg.norm)(input_points-x),self.maps_descriptors)
        # # indices =  jax.tree_map(lambda x: jax.lax.top_k(-x,k=1)[1],distances)
        # ## Calulating the movement of the BDs per map
        # if input_points.shape[0] > 1:
        #     indices = indices.squeeze()
        # else:
        #     indices = indices.reshape(-1,)
        # error_per_map = jax.tree_map(lambda x: (x.at[indices].get()-self.ref_map.at[indices].get()),all_maps)

        error_per_map = jnp.asarray(error_per_map)
        # Weighted Sum of all the Errors per BD
        errors = jnp.sum(
            error_per_map.at[:, :, self.chosen_bd].get() * constants, axis=0
        ).reshape(-1, 1)
        # errors = errors.at[:,[self.chosen_bd]].get()
        return errors

    @jax.jit
    def get_behaviours_inputs(
        self,
        input_points: Float[Array, "N D"],
        params: Dict,
        indices: Optional[jnp.array] = None,
    ):

        p1, p2 = params["rotation"]  # /jnp.max(params['rotation'])
        p3, p4 = params["offset"]
        w = 0.33
        scaling = jnp.asarray(
            [[(p1 + p2) / 2, (p2 - p1) * w / 4], [(p2 - p1) / w, (p1 + p2) / 2]]
        ).T
        constants = jnp.ones(shape=(1, 1))  # params['constant']
        all_maps = copy.deepcopy(self.maps_descriptors)

        offset = jnp.round(jnp.asarray([[p3, p4]]), decimals=1)

        all_maps[0] = jnp.matmul(self.maps_descriptors[0], scaling) + offset
        # jax.tree_map(lambda x: jnp.matmul(x,scaling),self.maps_descriptors)
        # @partial(jax.jit, static_argnames=("k_nn",))
        @jax.jit
        def _get_cells_indices(
            descriptors: jnp.ndarray,
            centroids: jnp.ndarray,
            k_nn: int,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            set_of_descriptors of shape (1, num_descriptors)
            centroids of shape (num_centroids, num_descriptors)
            """
            distances = jax.vmap(jnp.linalg.norm)(descriptors - centroids)
            ## Negating distances because we want the smallest ones
            min_args = jnp.argmin(distances)
            # min_dist,min_args = jax.lax.top_k(-1*distances,k_nn)
            return min_args  # ,-1*min_dist

        func = jax.vmap(
            _get_cells_indices,
            in_axes=(
                0,
                None,
                None,
            ),
        )

        indices = func(input_points, self.ref_map, 1)
        # jax.lax.top_k(-distances,k=1)
        ## Get BD from each Map for a Point
        # distances = jax.tree_map(lambda x: jax.vmap(jnp.linalg.norm)(input_points-x),self.maps_descriptors)
        # indices =  jax.tree_map(lambda x: jax.lax.top_k(-x,k=1)[1],distances)
        ## Calulating the movement of the BDs per map
        if input_points.shape[0] > 1:
            indices = indices.squeeze()
        else:
            indices = indices.reshape(
                -1,
            )
        error_per_map = jax.tree_map(
            lambda x: (x.at[indices].get() - self.ref_map.at[indices].get()), all_maps
        )

        error_per_map = jnp.asarray(error_per_map)
        # Weighted Sum of all the Errors per BD
        errors = jnp.sum(
            error_per_map.at[:, :, self.chosen_bd].get() * constants, axis=0
        ).reshape(-1, 1)
        # errors = errors.at[:,[self.chosen_bd]].get()
        return errors

    @jax.jit
    def get_behaviours_full_map(self, params: Dict):

        p1, p2 = params["rotation"]  # /jnp.max(params['rotation'])
        p3, p4 = params["offset"]
        w = 0.33
        scaling = jnp.asarray(
            [[(p1 + p2) / 2, (p2 - p1) * w / 4], [(p2 - p1) / w, (p1 + p2) / 2]]
        ).T
        constants = jnp.ones(shape=(1, 1))  # params['constant']
        all_maps = copy.deepcopy(self.maps_descriptors)

        offset = jnp.round(jnp.asarray([[p3, p4]]), decimals=1)

        all_maps[0] = jnp.matmul(self.maps_descriptors[0], scaling) + offset

        error_per_map = jax.tree_map(lambda x: (x - self.ref_map), all_maps)

        error_per_map = jnp.asarray(error_per_map)
        # Weighted Sum of all the Errors per BD
        errors = jnp.sum(
            error_per_map.at[:, :, self.chosen_bd].get() * constants, axis=0
        ).reshape(-1, 1)
        # errors = errors.at[:,[self.chosen_bd]].get()
        return errors

    @jax.jit
    def get_behaviours_simple(self, input_points: Float[Array, "N D"], params: Dict):

        p1, p2 = params["rotation"]  # /jnp.max(params['rotation'])
        p3, p4 = params["offset"]
        w = 0.33
        scaling = jnp.asarray(
            [[(p1 + p2) / 2, (p2 - p1) * w / 4], [(p2 - p1) / w, (p1 + p2) / 2]]
        ).T
        constants = jnp.ones(shape=(1, 1))  # params['constant']

        offset = jnp.round(jnp.asarray([[p3, p4]]), decimals=1)

        new_map = jnp.matmul(input_points, scaling) + offset

        # error_per_map = jax.tree_map(lambda x: (x-self.ref_map),all_maps)

        # error_per_map = jnp.asarray(error_per_map)
        # # Weighted Sum of all the Errors per BD
        # errors = jnp.sum(error_per_map.at[:,:,self.chosen_bd].get()*constants,axis=0).reshape(-1, 1)
        # return errors

        return new_map.at[:, [self.chosen_bd]].get()

    def __call__(
        self,
        params: Dict,
        x: Float[Array, "N D"],
        indices: Optional[jnp.array] = None,
        testing=False,
    ) -> Float[Array, "N Q"]:
        """Evaluate the mean function at the given points.
        Args:
            params (Dict): The parameters of the mean function.
            x (Float[Array, "N D"]): The input points at which to evaluate the mean function.
        Returns:
            Float[Array, "N Q"]: A vector of repeated constant values.
        """

        out_shape = (x.shape[0], self.output_dim)

        ### NOTE THAT THE LINEAR AND ANGULAR VELOCITY NEED TO BE AT THE END OF THE INPUT X
        # mean_errors = jnp.zeros(shape=out_shape) #self.get_behaviours(x[:,-2:],params,indices=None,testing=testing) # jnp.zeros(shape=out_shape) #

        # if testing:
        #      mean_errors = self.get_behaviours_full_map(params)
        # else:
        #     mean_errors = self.get_behaviours_inputs(x[:,-2:],params,indices=None)

        # return mean_errors
        transformed_map = self.get_behaviours_simple(x[:, -2:], params)
        return transformed_map

    def init_params(self, key: KeyArray) -> Dict:
        """The parameters of the mean function. For the constant-mean function, this is a dictionary with a single value.
        Args:
            key (KeyArray): The PRNG key to use for initialising the parameters.
        Returns:
            Dict: The parameters of the mean function.
        """

        # return {"constant": jnp.ones(shape=(self.num_maps,1))/self.num_maps,"scaling":jnp.identity(2)}

        parameters = {"rotation": jnp.ones(shape=(2,)), "offset": jnp.zeros(shape=(2,))}
        # parameters['capacity'] = parameters['capacity'].at[:2].set(1.0)
        # return {"scaling":jnp.identity(2)}
        return parameters


#######################
# GP Priors
#######################


class MAPPrior(AbstractPrior):
    """A Gaussian process prior object. The GP is parameterised by a
    `mean <https://gpjax.readthedocs.io/en/latest/api.html#module-gpjax.mean_functions>`_
    and `kernel <https://gpjax.readthedocs.io/en/latest/api.html#module-gpjax.kernels>`_ function.
    A Gaussian process prior parameterised by a mean function :math:`m(\\cdot)` and a kernel
    function :math:`k(\\cdot, \\cdot)` is given by
    .. math::
        p(f(\\cdot)) = \mathcal{GP}(m(\\cdot), k(\\cdot, \\cdot)).
    To invoke a ``Prior`` distribution, only a kernel function is required. By
    default, the mean function will be set to zero. In general, this assumption
    will be reasonable assuming the data being modelled has been centred.
    Example:
        >>> import gpjax as gpx
        >>>
        >>> kernel = gpx.kernels.RBF()
        >>> prior = gpx.Prior(kernel = kernel)
    """

    def __init__(
        self,
        kernel: AbstractKernel,
        mean_function: Optional[AbstractMeanFunction] = Zero(),
        name: Optional[str] = "GP prior",
    ) -> None:
        """Initialise the GP prior.
        Args:
            kernel (AbstractKernel): The kernel function used to parameterise the prior.
            mean_function (Optional[MeanFunction]): The mean function used to parameterise the
                prior. Defaults to zero.
            name (Optional[str]): The name of the GP prior. Defaults to "GP prior".
        """
        self.kernel = kernel
        self.mean_function = mean_function
        self.name = name

    def __mul__(self, other: AbstractLikelihood):
        """The product of a prior and likelihood is proportional to the
        posterior distribution. By computing the product of a GP prior and a
        likelihood object, a posterior GP object will be returned. Mathetically,
        this can be described by:
         .. math::
             p(f(\\cdot) | y) \\propto p(y | f(\\cdot)) p(f(\\cdot)).
         where :math:`p(y | f(\\cdot))` is the likelihood and :math:`p(f(\\cdot))`
         is the prior.
         Example:
             >>> import gpjax as gpx
             >>>
             >>> kernel = gpx.kernels.RBF()
             >>> prior = gpx.Prior(kernel = kernel)
             >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=100)
             >>>
             >>> prior * likelihood
         Args:
             other (Likelihood): The likelihood distribution of the observed dataset.
         Returns:
             Posterior: The relevant GP posterior for the given prior and
                likelihood. Special cases are accounted for where the model
                is conjugate.
        """

        return MAPConjugatePosterior(prior=self, likelihood=other)
        # return construct_posterior(prior=self, likelihood=other)

    def __rmul__(self, other: AbstractLikelihood):
        """Reimplement the multiplication operator to allow for order-invariant
        product of a likelihood and a prior i.e., likelihood * prior.
        Args:
            other (Likelihood): The likelihood distribution of the observed
                dataset.
        Returns:
            Posterior: The relevant GP posterior for the given prior and
                likelihood. Special cases are accounted for where the model
                is conjugate.
        """
        return self.__mul__(other)

    def predict(
        self, params: Dict
    ) -> Callable[[Float[Array, "N D"]], GaussianDistribution]:
        """Compute the predictive prior distribution for a given set of
        parameters. The output of this function is a function that computes
        a distrx distribution for a given set of inputs.
        In the following example, we compute the predictive prior distribution
        and then evaluate it on the interval :math:`[0, 1]`:
        Example:
            >>> import gpjax as gpx
            >>> import jax.numpy as jnp
            >>>
            >>> kernel = gpx.kernels.RBF()
            >>> prior = gpx.Prior(kernel = kernel)
            >>>
            >>> parameter_state = gpx.initialise(prior)
            >>> prior_predictive = prior.predict(parameter_state.params)
            >>> prior_predictive(jnp.linspace(0, 1, 100))
        Args:
            params (Dict): The specific set of parameters for which the mean
            function should be defined for.
        Returns:
            Callable[[Float[Array, "N D"]], GaussianDistribution]: A mean
            function that accepts an input array for where the mean function
            should be evaluated at. The mean function's value at these points is
            then returned.
        """
        jitter = get_global_config()["jitter"]

        # Unpack mean function and kernel
        mean_function = self.mean_function
        kernel = self.kernel

        def predict_fn(
            test_inputs: Float[Array, "N D"], indices: Optional[jnp.array] = None
        ) -> GaussianDistribution:

            # Unpack test inputs
            t = test_inputs
            n_test = test_inputs.shape[0]

            μt = mean_function(params["mean_function"], t, testing=True)
            Ktt = kernel.gram(params["kernel"], t)
            Ktt += identity(n_test) * jitter

            return GaussianDistribution(jnp.atleast_1d(μt.squeeze()), Ktt)

        return predict_fn

    def init_params(self, key: KeyArray) -> Dict:
        """Initialise the GP prior's parameter set.
        Args:
            key (KeyArray): The PRNG key.
        Returns:
            Dict: The initialised parameter set.
        """
        return {
            "kernel": self.kernel.init_params(key),
            "mean_function": self.mean_function.init_params(key),
        }


class MAPConjugatePosterior(AbstractPosterior):
    """A Gaussian process posterior distribution when the constituent likelihood
    function is a Gaussian distribution. In such cases, the latent function values
    :math:`f` can be analytically integrated out of the posterior distribution.
    As such, many computational operations can be simplified; something we make use
    of in this object.
    For a Gaussian process prior :math:`p(\mathbf{f})` and a Gaussian likelihood
    :math:`p(y | \\mathbf{f}) = \\mathcal{N}(y\\mid \mathbf{f}, \\sigma^2))` where
    :math:`\mathbf{f} = f(\\mathbf{x})`, the predictive posterior distribution at
    a set of inputs :math:`\\mathbf{x}` is given by
    .. math::
        p(\\mathbf{f}^{\\star}\mid \mathbf{y}) & = \\int p(\\mathbf{f}^{\\star} \\mathbf{f} \\mid \\mathbf{y})\\\\
        & =\\mathcal{N}(\\mathbf{f}^{\\star} \\boldsymbol{\mu}_{\mid \mathbf{y}}, \\boldsymbol{\Sigma}_{\mid \mathbf{y}}
    where
    .. math::
        \\boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\\mathbf{x}^{\\star}, \\mathbf{x})\\left(k(\\mathbf{x}, \\mathbf{x}')+\\sigma^2\\mathbf{I}_n\\right)^{-1}\\mathbf{y}  \\\\
        \\boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\\mathbf{x}^{\\star}, \\mathbf{x}^{\\star\\prime}) -k(\\mathbf{x}^{\\star}, \\mathbf{x})\\left( k(\\mathbf{x}, \\mathbf{x}') + \\sigma^2\\mathbf{I}_n \\right)^{-1}k(\\mathbf{x}, \\mathbf{x}^{\\star}).
    Example:
        >>> import gpjax as gpx
        >>> import jax.numpy as jnp
        >>>
        >>> prior = gpx.Prior(kernel = gpx.kernels.RBF())
        >>> likelihood = gpx.likelihoods.Gaussian()
        >>>
        >>> posterior = prior * likelihood
    """

    def __init__(
        self,
        prior: AbstractPrior,
        likelihood: AbstractLikelihood,
        name: Optional[str] = "GP posterior",
    ) -> None:
        """Initialise the conjugate GP posterior object.
        Args:
            prior (Prior): The prior distribution of the GP.
            likelihood (AbstractLikelihood): The likelihood distribution of the observed dataset.
            name (Optional[str]): The name of the GP posterior. Defaults to "GP posterior".
        """
        self.prior = prior
        self.likelihood = likelihood
        self.name = name

    def predict(
        self,
        params: Dict,
        train_data: Dataset,
        weights_noise: jnp.array,
        indices: Optional[jnp.array] = None,
    ) -> Callable[[Float[Array, "N D"]], GaussianDistribution]:
        """Conditional on a training data set, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned
        function can be evaluated at a set of test inputs to compute the
        corresponding predictive density.
        The predictive distribution of a conjugate GP is given by
        .. math::
            p(\\mathbf{f}^{\\star}\mid \mathbf{y}) & = \\int p(\\mathbf{f}^{\\star} \\mathbf{f} \\mid \\mathbf{y})\\\\
            & =\\mathcal{N}(\\mathbf{f}^{\\star} \\boldsymbol{\mu}_{\mid \mathbf{y}}, \\boldsymbol{\Sigma}_{\mid \mathbf{y}}
        where
        .. math::
            \\boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\\mathbf{x}^{\\star}, \\mathbf{x})\\left(k(\\mathbf{x}, \\mathbf{x}')+\\sigma^2\\mathbf{I}_n\\right)^{-1}\\mathbf{y}  \\\\
            \\boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\\mathbf{x}^{\\star}, \\mathbf{x}^{\\star\\prime}) -k(\\mathbf{x}^{\\star}, \\mathbf{x})\\left( k(\\mathbf{x}, \\mathbf{x}') + \\sigma^2\\mathbf{I}_n \\right)^{-1}k(\\mathbf{x}, \\mathbf{x}^{\\star}).
        The conditioning set is a GPJax ``Dataset`` object, whilst predictions
        are made on a regular Jax array.
        £xample:
            For a ``posterior`` distribution, the following code snippet will
            evaluate the predictive distribution.
            >>> import gpjax as gpx
            >>>
            >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
            >>> ytrain = jnp.sin(xtrain)
            >>> xtest = jnp.linspace(0, 1).reshape(-1, 1)
            >>>
            >>> params = gpx.initialise(posterior)
            >>> predictive_dist = posterior.predict(params, gpx.Dataset(X=xtrain, y=ytrain))
            >>> predictive_dist(xtest)
        Args:
            params (Dict): A dictionary of parameters that should be used to
                compute the posterior.
            train_data (Dataset): A `gpx.Dataset` object that contains the
                input and output data used for training dataset.
        Returns:
            Callable[[Float[Array, "N D"]], GaussianDistribution]: A
                function that accepts an input array and returns the predictive
                distribution as a ``GaussianDistribution``.
        """
        jitter = get_global_config()["jitter"]

        # Unpack training data
        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        start_t = time.time()
        # Observation noise σ²
        obs_noise = params["likelihood"]["obs_noise"]
        μx = mean_function(params["mean_function"], x, indices)
        # μx = mean_function.get_behaviours_inputs(x[:,-2:],params["mean_function"],indices)

        print("Mean Function Optim", {time.time() - start_t}, flush=True)
        start_t = time.time()
        # Precompute Gram matrix, Kxx, at training inputs, x
        Kxx = kernel.gram(params["kernel"], x)
        Kxx += identity(n) * jitter

        print("Kxx Optim", {time.time() - start_t}, flush=True)
        # Σ = Kxx + Dσ²

        start_t = time.time()
        Sigma = Kxx + DiagonalLinearOperator(weights_noise) * obs_noise
        print("Kxx Optim", {time.time() - start_t}, flush=True)
        # Σ = Kxx + Iσ²
        # Sigma = Kxx + identity(n) * obs_noise

        def predict(
            test_inputs: Float[Array, "N D"], testing=True
        ) -> GaussianDistribution:
            """Compute the predictive distribution at a set of test inputs.
            Args:
                test_inputs (Float[Array, "N D"]): A Jax array of test inputs.
            Returns:
                GaussianDistribution: A ``GaussianDistribution``
                object that represents the predictive distribution.
            """

            # Unpack test inputs
            t = test_inputs
            n_test = test_inputs.shape[0]

            μt = mean_function(params["mean_function"], t, testing=testing)
            # μt = mean_function.get_behaviours_full_map(params["mean_function"])

            Ktt = kernel.gram(params["kernel"], t)
            Kxt = kernel.cross_covariance(params["kernel"], x, t)

            # Σ⁻¹ Kxt
            Sigma_inv_Kxt = Sigma.solve(Kxt)
            # μt  +  Ktx (Kxx + Iσ²)⁻¹ (y  -  μx)
            mean = μt + jnp.matmul(Sigma_inv_Kxt.T, y - μx)

            # Ktt  -  Ktx (Kxx + Iσ²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
            covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
            covariance += identity(n_test) * jitter

            return GaussianDistribution(jnp.atleast_1d(mean.squeeze()), covariance)

        return predict

    def marginal_log_likelihood(
        self,
        train_data: Dataset,
        negative: bool = False,
    ) -> Callable[[Dict], Float[Array, "1"]]:
        """Compute the marginal log-likelihood function of the Gaussian process.
        The returned function can then be used for gradient based optimisation
        of the model's parameters or for model comparison. The implementation
        given here enables exact estimation of the Gaussian process' latent
        function values.
        For a training dataset :math:`\\{x_n, y_n\\}_{n=1}^N`, set of test
        inputs :math:`\\mathbf{x}^{\\star}` the corresponding latent function
        evaluations are given by :math:`\\mathbf{f} = f(\\mathbf{x})`
        and :math:`\\mathbf{f}^{\\star} = f(\\mathbf{x}^{\\star})`, the marginal
        log-likelihood is given by:
        .. math::
            \\log p(\\mathbf{y}) & = \\int p(\\mathbf{y}\\mid\\mathbf{f})p(\\mathbf{f}, \\mathbf{f}^{\\star}\\mathrm{d}\\mathbf{f}^{\\star}\\\\
            &=0.5\\left(-\\mathbf{y}^{\\top}\\left(k(\\mathbf{x}, \\mathbf{x}') +\\sigma^2\\mathbf{I}_N  \\right)^{-1}\\mathbf{y}-\\log\\lvert k(\\mathbf{x}, \\mathbf{x}') + \\sigma^2\\mathbf{I}_N\\rvert - n\\log 2\\pi \\right).
        Example:
        For a given ``ConjugatePosterior`` object, the following code snippet shows
        how the marginal log-likelihood can be evaluated.
        >>> import gpjax as gpx
        >>>
        >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
        >>> ytrain = jnp.sin(xtrain)
        >>> D = gpx.Dataset(X=xtrain, y=ytrain)
        >>>
        >>> params = gpx.initialise(posterior)
        >>> mll = posterior.marginal_log_likelihood(train_data = D)
        >>> mll(params)
        Our goal is to maximise the marginal log-likelihood. Therefore, when
        optimising the model's parameters with respect to the parameters, we
        use the negative marginal log-likelihood. This can be realised through
        >>> mll = posterior.marginal_log_likelihood(train_data = D, negative=True)
        Further, prior distributions can be passed into the marginal log-likelihood
        >>> mll = posterior.marginal_log_likelihood(train_data = D)
        For optimal performance, the marginal log-likelihood should be ``jax.jit``
        compiled.
        >>> mll = jit(posterior.marginal_log_likelihood(train_data = D))
        Args:
            train_data (Dataset): The training dataset used to compute the
                marginal log-likelihood.
            negative (Optional[bool]): Whether or not the returned function
                should be negative. For optimisation, the negative is useful
                as minimisation of the negative marginal log-likelihood is
                equivalent to maximisation of the marginal log-likelihood.
                Defaults to False.
        Returns:
            Callable[[Dict], Float[Array, "1"]]: A functional representation
                of the marginal log-likelihood that can be evaluated at a
                given parameter set.
        """
        jitter = get_global_config()["jitter"]

        # Unpack training data
        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack mean function and kernel
        mean_function = self.prior.mean_function
        kernel = self.prior.kernel

        # The sign of the marginal log-likelihood depends on whether we are maximising or minimising
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def mll(
            params: Dict,
        ):
            """Compute the marginal log-likelihood of the Gaussian process.
            Args:
                params (Dict): The model's parameters.
            Returns:
                Float[Array, "1"]: The marginal log-likelihood.
            """

            # Observation noise σ²
            obs_noise = params["likelihood"]["obs_noise"]
            μx = mean_function(params["mean_function"], x)
            # μx = mean_function.get_behaviours_inputs(x[:,-2:],params["mean_function"])

            # TODO: This implementation does not take advantage of the covariance operator structure.
            # Future work concerns implementation of a custom Gaussian distribution / measure object that accepts a covariance operator.

            # Σ = (Kxx + Iσ²) = LLᵀ
            Kxx = kernel.gram(params["kernel"], x)
            Kxx += identity(n) * jitter
            Sigma = Kxx + identity(n) * obs_noise

            # p(y | x, θ), where θ are the model hyperparameters:
            marginal_likelihood = GaussianDistribution(
                jnp.atleast_1d(μx.squeeze()), Sigma
            )

            return constant * (
                marginal_likelihood.log_prob(jnp.atleast_1d(y.squeeze())).squeeze()
            )

        return mll


def map_fit(
    objective1: Callable,
    objective2: Callable,
    parameter_state: ParameterState,
    optax_optim: ox.GradientTransformation,
    num_iters: Optional[int] = 100,
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
) -> InferenceState:
    """Abstracted method for fitting a GP model with respect to a supplied objective function.
    Optimisers used here should originate from Optax.
    Args:
        objective (Callable): The objective function that we are optimising with respect to.
        parameter_state (ParameterState): The initial parameter state.
        optax_optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        num_iters (Optional[int]): The number of optimisation steps to run. Defaults to 100.
        log_rate (Optional[int]): How frequently the objective function's value should be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults to True.
    Returns:
        InferenceState: An InferenceState object comprising the optimised parameters and training history respectively.
    """

    params, trainables, bijectors = parameter_state.unpack()

    # Define optimisation loss function on unconstrained space, with a stop gradient rule for trainables that are set to False
    def loss(params: Dict) -> Float[Array, "1"]:
        params = trainable_params(params, trainables)
        params = constrain(params, bijectors)
        return objective1(params) + objective2(params)
        # return -jnp.sqrt(objective1(params)**2+objective2(params)**2)

    # Transform params to unconstrained space
    params = unconstrain(params, bijectors)

    # Initialise optimiser state
    opt_state = optax_optim.init(params)

    # Iteration loop numbers to scan over
    iter_nums = jnp.arange(num_iters)

    # Optimisation step
    def step(carry, iter_num: int):
        params, opt_state = carry
        loss_val, loss_gradient = jax.value_and_grad(loss)(params)
        updates, opt_state = optax_optim.update(loss_gradient, opt_state, params)
        params = ox.apply_updates(params, updates)
        carry = params, opt_state
        return carry, loss_val

    # Display progress bar if verbose is True
    if verbose:
        step = progress_bar_scan(num_iters, log_rate)(step)

    # Run the optimisation loop
    (params, _), history = jax.lax.scan(step, (params, opt_state), iter_nums)

    # Transform final params to constrained space
    params = constrain(params, bijectors)

    return InferenceState(params=params, history=history)


def fit_simple(
    objective: Callable,
    parameter_state: ParameterState,
    optax_optim: ox.GradientTransformation,
    num_iters: Optional[int] = 100,
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = False,
) -> InferenceState:
    """Abstracted method for fitting a GP model with respect to a supplied objective function.
    Optimisers used here should originate from Optax.
    Args:
        objective (Callable): The objective function that we are optimising with respect to.
        parameter_state (ParameterState): The initial parameter state.
        optax_optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        num_iters (Optional[int]): The number of optimisation steps to run. Defaults to 100.
        log_rate (Optional[int]): How frequently the objective function's value should be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults to True.
    Returns:
        InferenceState: An InferenceState object comprising the optimised parameters and training history respectively.
    """

    params, trainables, bijectors = parameter_state.unpack()

    # Define optimisation loss function on unconstrained space, with a stop gradient rule for trainables that are set to False
    def loss(params: Dict) -> Float[Array, "1"]:
        params = trainable_params(params, trainables)
        params = constrain(params, bijectors)
        return objective(params)

    start_t = time.time()
    # Tranform params to unconstrained space
    params = unconstrain(params, bijectors)
    print("Unconstrain Optim", {time.time() - start_t}, flush=True)
    start_t = time.time()
    # Initialise optimiser state
    opt_state = optax_optim.init(params)
    print("Time Init Optim", {time.time() - start_t}, flush=True)
    # Iteration loop numbers to scan over
    iter_nums = jnp.arange(num_iters)

    # Optimisation step
    @jax.jit
    def step(carry, iter_num: int):
        params, opt_state = carry
        loss_val, loss_gradient = jax.value_and_grad(loss)(params)
        updates, opt_state = optax_optim.update(loss_gradient, opt_state, params)
        params = ox.apply_updates(params, updates)
        carry = params, opt_state
        return carry, loss_val

    # Display progress bar if verbose is True
    if verbose:
        step = progress_bar_scan(num_iters, log_rate)(step)

    start_t = time.time()
    # Run the optimisation loop
    (params, _), history = jax.lax.scan(step, (params, opt_state), iter_nums)

    print("Lax Scan Optim", {time.time() - start_t}, flush=True)

    # Tranform final params to constrained space
    params = constrain(params, bijectors)

    return InferenceState(params=params, history=history)
