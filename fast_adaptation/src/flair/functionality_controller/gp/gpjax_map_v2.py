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


from abc import abstractmethod
from dataclasses import dataclass
from flax import linen as nn
from functools import partial

from beartype.typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    Dict,
    Union,
    Tuple,
)
import jax.numpy as jnp
from jax.random import (
    PRNGKey,
    normal,
)
from jaxtyping import (
    Float,
    Num,
)
from dataclasses import (
    dataclass,
    field,
)
from gpjax.base import (
    Module,
    param_field,
    static_field,
)
import flax
from gpjax.dataset import Dataset
from gpjax.gaussian_distribution import GaussianDistribution
from gpjax.kernels import RFF, RBF
from gpjax.kernels.base import AbstractKernel
from gpjax.likelihoods import (
    AbstractLikelihood,
    Gaussian,
)
from gpjax.linops import identity,DiagonalLinearOperator,to_dense
from gpjax.mean_functions import AbstractMeanFunction, Zero
from gpjax.typing import (
    Array,
    FunctionalSample,
    KeyArray,
    Array,
    ScalarFloat,
)

import time
from gpjax.gps import AbstractPrior,construct_posterior,AbstractPosterior
import jax
import copy
import optax as ox
# from gpjax.parameters import constrain, trainable_params, unconstrain,ParameterState
from gpjax.objectives import AbstractObjective
from gpjax.scan import vscan
from gpjax.fit import get_batch,_check_model, _check_train_data,_check_optim,_check_num_iters,_check_batch_size,_check_prng_key,_check_log_rate,_check_verbose

@dataclass
class MAPMean(AbstractMeanFunction):
    """
    A zero mean function. This function returns a repeated scalar value for all inputs.
    The scalar value itself can be treated as a model hyperparameter and learned during training.
    """
    chosen_bd: Float[Array, "1"] =  static_field(default_factory=0)
    rotation: Float[Array, "1"] = param_field(jnp.zeros(shape=(1,)))
    # rotation: Float[Array, "2"] = param_field(jnp.ones(shape=(2,)))
    offset: Float[Array, "2"] = param_field(jnp.zeros(shape=(2,)))
    

    # def __init__(
    #     self, maps: Sequence[jnp.array],ref_map:jnp.array, output_dim: Optional[int] = 1,chosen_bd: Optional[int] = 0, name: Optional[str] = "Mean function", 
    # ):
    #     """Initialise the constant-mean function.
    #     Args:
    #         output_dim (Optional[int]): The output dimension of the mean function. Defaults to 1.
    #         name (Optional[str]): The name of the mean function. Defaults to "Mean function".
    #     """
    #     self.maps_descriptors = maps
    #     self.ref_map = ref_map
    #     self.num_maps = len(maps)
    #     self.chosen_bd = chosen_bd
    #     super().__init__()

    # @jax.jit
    def get_behaviours_with_state(
        self,input_points: Float[Array, "N D"],state,
    ):
        ## State Dependent P
        ## alpha*x**2+beta
        # alpha = self.offset[0]
        # beta = self.offset[1]
        # offset = self.offset[2]
        # # new_p = alpha*state**2+beta
        # new_p = alpha*jnp.clip(state,a_min=-0.4,a_max=0.4)+beta
        # # new_p = alpha*jnp.clip(state,a_min=-0.6,a_max=0.6)+beta
        # # new_p = alpha*state**3+beta


        ### a+b*x+c*x**2+d*x**3
        a = self.offset[0]
        b = self.offset[1]
        c = self.offset[2]
        d = self.offset[3]
        offset = self.offset[4]
        # clip_state = jnp.clip(state,a_min=-0.4,a_max=0.4)
        clip_state = jnp.clip(state,a_min=-0.6,a_max=0.6)
        new_p = a+b*clip_state+c*clip_state**2+d*clip_state**3
        w= 0.33

        

        # p1,p2 = jnp.min(jnp.asarray([1,1-new_p])),jnp.min(jnp.asarray([1,1+new_p]))
        p1,p2 = jnp.clip(1-new_p,a_max=1.0),jnp.clip(1+new_p,a_max=1.0)

        diag = (p1+p2) / 2
        # diag = jnp.clip((p1+p2) / 2, 0.5, 1.0)

        scaling = jnp.asarray([[diag,(p2-p1)*w/4],[(p2-p1)/w,diag]]).T

        ##TODO Einsum
        
        a = jnp.concatenate([diag,(p2-p1)*w/4],axis=1)
        b = jnp.concatenate([(p2-p1)/w,diag],axis=1)
        # print("SHAPES",a.shape,b.shape,new_p.shape,p1.shape,p2.shape,diag.shape,flush=True)
        # print("New Map P1,P2",jnp.multiply(input_points,a).sum(axis=1).shape,b.shape,input_points.shape,jnp.asarray([diag,(p2-p1)*w/4]).shape,jnp.multiply(input_points,jnp.asarray([diag,(p2-p1)*w/4])).shape,flush=True)

        new_map_mean = jnp.stack([jnp.multiply(input_points,a).sum(axis=1)+offset,jnp.multiply(input_points,b).sum(axis=1)],axis=-1)
        # print("New Map ",new_map_mean.shape,state.shape,flush=True)
        # new_map = jnp.matmul(all_descriptors,scaling)

        # return new_map
        return new_map_mean

    
    def get_behaviours_simple(self,input_points: Float[Array, "N D"],chosen_bd):

        # print("ROTATION",self.chosen_bd,flush=True)

        p = self.rotation[0]
        p = jnp.clip(p,-1.0,1.0)
        # p1,p2 = self.rotation[0],self.rotation[1] #/jnp.max(params['rotation'])
        p1,p2 = jnp.min(jnp.asarray([1,1-p])),jnp.min(jnp.asarray([1,1+p]))
        p3,p4  = self.offset[0],self.offset[1]
        w = 0.33
        diag = jnp.clip((p1+p2) / 2, 0.5, 1.0)
        scaling = jnp.asarray([[diag,(p2-p1)*w/4],[(p2-p1)/w,diag]]).T
        constants = jnp.ones(shape=(1,1)) #params['constant']

        offset = jnp.round(jnp.asarray([[p3,p4]]),decimals=2)

        new_map = jnp.matmul(input_points,scaling) + offset
        
        # error_per_map = jax.tree_map(lambda x: (x-self.ref_map),all_maps)
        
        # error_per_map = jnp.asarray(error_per_map)
        # # Weighted Sum of all the Errors per BD
        # errors = jnp.sum(error_per_map.at[:,:,self.chosen_bd].get()*constants,axis=0).reshape(-1, 1)
        # return errors
        
        return new_map.at[:,[chosen_bd]].get()

    def get_behaviours_all(self,input_points: Float[Array, "N D"]):

        # print("ROTATION",self.chosen_bd,flush=True)

        p = self.rotation[0]
        p = jnp.clip(p,-1.0,1.0)
        # p1,p2 = self.rotation[0],self.rotation[1] #/jnp.max(params['rotation'])
        p1,p2 = jnp.min(jnp.asarray([1,1-p])),jnp.min(jnp.asarray([1,1+p]))
        p3,p4  = self.offset[0],self.offset[1]
        w = 0.33
        diag = jnp.clip((p1+p2) / 2, 0.5, 1.0)
        scaling = jnp.asarray([[diag,(p2-p1)*w/4],[(p2-p1)/w,diag]]).T
        constants = jnp.ones(shape=(1,1)) #params['constant']

        offset = jnp.round(jnp.asarray([[p3,p4]]),decimals=2)

        new_map = jnp.matmul(input_points,scaling) + offset
        
        # error_per_map = jax.tree_map(lambda x: (x-self.ref_map),all_maps)
        
        # error_per_map = jnp.asarray(error_per_map)
        # # Weighted Sum of all the Errors per BD
        # errors = jnp.sum(error_per_map.at[:,:,self.chosen_bd].get()*constants,axis=0).reshape(-1, 1)
        # return errors
        
        # return new_map.at[:,[chosen_bd]].get()
        return new_map


    def pred_state(self, x: Float[Array, "N D"],state) -> Float[Array, "N Q"]:
        return self.get_behaviours_with_state(x[:,-2:],state)
    # @jax.jit
    def __call__(self, x: Float[Array, "N D"],chosen_bd=None,state=None) -> Float[Array, "N Q"]:
        """Evaluate the mean function at the given points.
        Args:
            params (Dict): The parameters of the mean function.
            x (Float[Array, "N D"]): The input points at which to evaluate the mean function.
        Returns:
            Float[Array, "N Q"]: A vector of repeated constant values.
        """
        ### NOTE THAT THE LINEAR AND ANGULAR VELOCITY NEED TO BE AT THE END OF THE INPUT X
        # mean_errors = jnp.zeros(shape=out_shape) #self.get_behaviours(x[:,-2:],params,indices=None,testing=testing) # jnp.zeros(shape=out_shape) #

        # if testing:
        #      mean_errors = self.get_behaviours_full_map(params)
        # else:
        #     mean_errors = self.get_behaviours_inputs(x[:,-2:],params,indices=None)

        # return mean_errors
        # print("NEW X :",x,flush=True)
        if chosen_bd == None:
            chosen_bd = self.chosen_bd
        elif chosen_bd ==-1:
            return self.get_behaviours_all(x[:,-2:])
        transformed_map = self.get_behaviours_simple(x[:,-2:],chosen_bd)
        # print("NEW MAP :",transformed_map.shape,flush=True)
        return transformed_map


    def init_params(self, key: KeyArray) -> Dict:
        """The parameters of the mean function. For the constant-mean function, this is a dictionary with a single value.
        Args:
            key (KeyArray): The PRNG key to use for initialising the parameters.
        Returns:
            Dict: The parameters of the mean function.
        """

        # return {"constant": jnp.ones(shape=(self.num_maps,1))/self.num_maps,"scaling":jnp.identity(2)}

        
        parameters = {"rotation":jnp.ones(shape=(2,)),"offset":jnp.zeros(shape=(2,))}
        # parameters['capacity'] = parameters['capacity'].at[:2].set(1.0)
        # return {"scaling":jnp.identity(2)}
        return parameters



#######################
# GP Priors
#######################

@dataclass
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

    # def __init__(
    #     self,
    #     kernel: AbstractKernel,
    #     mean_function: Optional[AbstractMeanFunction] = Zero(),
    #     name: Optional[str] = "GP prior",
    # ) -> None:
    #     """Initialise the GP prior.
    #     Args:
    #         kernel (AbstractKernel): The kernel function used to parameterise the prior.
    #         mean_function (Optional[MeanFunction]): The mean function used to parameterise the
    #             prior. Defaults to zero.
    #         name (Optional[str]): The name of the GP prior. Defaults to "GP prior".
    #     """
    #     # self.kernel = kernel
    #     # self.mean_function = mean_function
    #     self.name = name
    #     super().__init__(kernel=kernel,mean_function=mean_function,jitter=0.0001)

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

        # Unpack mean function and kernel
        mean_function = self.mean_function
        kernel = self.kernel

        def predict_fn(
            test_inputs: Float[Array, "N D"],indices: Optional[jnp.array]=None
        ) -> GaussianDistribution:

            # Unpack test inputs
            t = test_inputs
            n_test = test_inputs.shape[0]

            μt = mean_function(params["mean_function"], t,testing=True)
            Ktt = kernel.gram(params["kernel"], t)
            Ktt += identity(n_test) * self.jitter

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

@dataclass
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
    def predict(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: Dataset,
        weights_noise: Num[Array, "N D"],
        chosen_bd: int = None,
    ) -> GaussianDistribution:
        r"""Query the predictive posterior distribution.

        Conditional on a training data set, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned function
        can be evaluated at a set of test inputs to compute the corresponding
        predictive density.

        The predictive distribution of a conjugate GP is given by
        $$
            p(\mathbf{f}^{\star}\mid \mathbf{y}) & = \int p(\mathbf{f}^{\star} \mathbf{f} \mid \mathbf{y})\\
            & =\mathcal{N}(\mathbf{f}^{\star} \boldsymbol{\mu}_{\mid \mathbf{y}}, \boldsymbol{\Sigma}_{\mid \mathbf{y}}
        $$
        where
        $$
            \boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\mathbf{x}^{\star}, \mathbf{x})\left(k(\mathbf{x}, \mathbf{x}')+\sigma^2\mathbf{I}_n\right)^{-1}\mathbf{y}  \\
            \boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\mathbf{x}^{\star}, \mathbf{x}^{\star\prime}) -k(\mathbf{x}^{\star}, \mathbf{x})\left( k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_n \right)^{-1}k(\mathbf{x}, \mathbf{x}^{\star}).
        $$

        The conditioning set is a GPJax `Dataset` object, whilst predictions
        are made on a regular Jax array.

        Example:
            For a `posterior` distribution, the following code snippet will
            evaluate the predictive distribution.
            ```python
                >>> import gpjax as gpx
                >>> import jax.numpy as jnp
                >>>
                >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
                >>> ytrain = jnp.sin(xtrain)
                >>> D = gpx.Dataset(X=xtrain, y=ytrain)
                >>> xtest = jnp.linspace(0, 1).reshape(-1, 1)
                >>>
                >>> prior = gpx.Prior(mean_function = gpx.Zero(), kernel = gpx.RBF())
                >>> posterior = prior * gpx.Gaussian(num_datapoints = D.n)
                >>> predictive_dist = posterior(xtest, D)
            ```

        Args:
            test_inputs (Num[Array, "N D"]): A Jax array of test inputs at which the
                predictive distribution is evaluated.
            train_data (Dataset): A `gpx.Dataset` object that contains the input and
                output data used for training dataset.

        Returns
        -------
            GaussianDistribution: A
                function that accepts an input array and returns the predictive
                    distribution as a `GaussianDistribution`.
        """
        # Unpack training data


        x, y, n = train_data.X, train_data.y[:,[chosen_bd]], train_data.n

        # Unpack test inputs
        t, n_test = test_inputs, test_inputs.shape[0]

        # Observation noise o²
        obs_noise = self.likelihood.obs_noise
        mx = self.prior.mean_function(x,chosen_bd)

        # start_t = time.time()
        # Precompute Gram matrix, Kxx, at training inputs, x
        Kxx = self.prior.kernel.gram(x) + (identity(n) * self.prior.jitter)
        # print("Kxx Optim",{time.time() - start_t},flush=True)
        # Σ = Kxx + Io²
        Sigma = Kxx + identity(n) * obs_noise*weights_noise

        mean_t = self.prior.mean_function(t,chosen_bd)

        # start_t = time.time()
        # print("MEAN,", mean_t.shape,y.shape,flush=True)
        # Ktt = self.prior.kernel.gram(t)
        # bin/r = identity(n_test)
        Kxt = self.prior.kernel.cross_covariance(x, t)
        # print("Ktt Optim",{time.time() - start_t},flush=True)

        # start_t = time.time()
        # Σ⁻¹ Kxt
        #Sigma_inv_Kxt = Sigma.solve(Kxt)
        #logger.warning(f'size of SIGMA mat: {Sigma_inv_Kxt.shape}')
        # print("Sigma_inv_Kxt Optim",{time.time() - start_t},flush=True)
        # μt  +  Ktx (Kxx + Io²)⁻¹ (y  -  μx)
        #mean = mean_t + jnp.matmul(Sigma_inv_Kxt.T, y - mx)

        L = jnp.linalg.cholesky(Sigma.to_dense())
        alpha = jnp.linalg.solve(jnp.transpose(L), jnp.linalg.solve(L, y-mx) )
        mean = mean_t + jnp.matmul(jnp.transpose(Kxt),alpha)
        
        #logger.warning(f'size of mean mat: {mean.shape}')

        
        # start_t = time.time()
        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        # v = jnp.linalg.solve(L,Kxt)
        # covariance = 1 - jnp.sum(jnp.square(v),axis=0) # Variance of the GP

        covariance = jnp.ones(shape=(n_test,))
        #covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        #covariance += identity(n_test) * self.prior.jitter
        #logger.warning(f'size of uncertainty mat: {covariance}')
        
        # print("Covariance Optim",{time.time() - start_t},flush=True)

        return mean.squeeze(), covariance

    def predict_all(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: Dataset,
        weights_noise: Num[Array, "N D"],
    ) -> GaussianDistribution:
        r"""Query the predictive posterior distribution.

        Conditional on a training data set, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned function
        can be evaluated at a set of test inputs to compute the corresponding
        predictive density.

        The predictive distribution of a conjugate GP is given by
        $$
            p(\mathbf{f}^{\star}\mid \mathbf{y}) & = \int p(\mathbf{f}^{\star} \mathbf{f} \mid \mathbf{y})\\
            & =\mathcal{N}(\mathbf{f}^{\star} \boldsymbol{\mu}_{\mid \mathbf{y}}, \boldsymbol{\Sigma}_{\mid \mathbf{y}}
        $$
        where
        $$
            \boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\mathbf{x}^{\star}, \mathbf{x})\left(k(\mathbf{x}, \mathbf{x}')+\sigma^2\mathbf{I}_n\right)^{-1}\mathbf{y}  \\
            \boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\mathbf{x}^{\star}, \mathbf{x}^{\star\prime}) -k(\mathbf{x}^{\star}, \mathbf{x})\left( k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_n \right)^{-1}k(\mathbf{x}, \mathbf{x}^{\star}).
        $$

        The conditioning set is a GPJax `Dataset` object, whilst predictions
        are made on a regular Jax array.

        Example:
            For a `posterior` distribution, the following code snippet will
            evaluate the predictive distribution.
            ```python
                >>> import gpjax as gpx
                >>> import jax.numpy as jnp
                >>>
                >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
                >>> ytrain = jnp.sin(xtrain)
                >>> D = gpx.Dataset(X=xtrain, y=ytrain)
                >>> xtest = jnp.linspace(0, 1).reshape(-1, 1)
                >>>
                >>> prior = gpx.Prior(mean_function = gpx.Zero(), kernel = gpx.RBF())
                >>> posterior = prior * gpx.Gaussian(num_datapoints = D.n)
                >>> predictive_dist = posterior(xtest, D)
            ```

        Args:
            test_inputs (Num[Array, "N D"]): A Jax array of test inputs at which the
                predictive distribution is evaluated.
            train_data (Dataset): A `gpx.Dataset` object that contains the input and
                output data used for training dataset.

        Returns
        -------
            GaussianDistribution: A
                function that accepts an input array and returns the predictive
                    distribution as a `GaussianDistribution`.
        """
        # Unpack training data


        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack test inputs
        t, n_test = test_inputs, test_inputs.shape[0]

        # Observation noise o²
        obs_noise = self.likelihood.obs_noise
        mx = self.prior.mean_function(x,-1)

        # start_t = time.time()
        # Precompute Gram matrix, Kxx, at training inputs, x
        Kxx = self.prior.kernel.gram(x) + (identity(n) * self.prior.jitter)
        # print("Kxx Optim",{time.time() - start_t},flush=True)
        # Σ = Kxx + Io²
        Sigma = Kxx + identity(n) * obs_noise*weights_noise

        mean_t = self.prior.mean_function(t,-1) ## Make quicker
    
        Kxt = self.prior.kernel.cross_covariance(x, t)


        L = jnp.linalg.cholesky(Sigma.to_dense())
        alpha = jnp.linalg.solve(jnp.transpose(L), jnp.linalg.solve(L, y-mx) )
        cov_term = jnp.matmul(jnp.transpose(Kxt),alpha)
        mean = mean_t + cov_term
        
        #logger.warning(f'size of mean mat: {mean.shape}')

        
        # start_t = time.time()
        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        # v = jnp.linalg.solve(L,Kxt)
        # covariance = 1 - jnp.sum(jnp.square(v),axis=0) # Variance of the GP
        covariance = jnp.ones(shape=(n_test,))

        #covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        #covariance += identity(n_test) * self.prior.jitter
        #logger.warning(f'size of uncertainty mat: {covariance}')
        
        # print("Covariance Optim",{time.time() - start_t},flush=True)

        return mean.squeeze(), covariance,cov_term


    def predict_all_state(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: Dataset,
        weights_noise: Num[Array, "N D"],
        state_train,
        state_test,
    ) -> GaussianDistribution:
        r"""Query the predictive posterior distribution.

        Conditional on a training data set, compute the GP's posterior
        predictive distribution for a given set of parameters. The returned function
        can be evaluated at a set of test inputs to compute the corresponding
        predictive density.

        The predictive distribution of a conjugate GP is given by
        $$
            p(\mathbf{f}^{\star}\mid \mathbf{y}) & = \int p(\mathbf{f}^{\star} \mathbf{f} \mid \mathbf{y})\\
            & =\mathcal{N}(\mathbf{f}^{\star} \boldsymbol{\mu}_{\mid \mathbf{y}}, \boldsymbol{\Sigma}_{\mid \mathbf{y}}
        $$
        where
        $$
            \boldsymbol{\mu}_{\mid \mathbf{y}} & = k(\mathbf{x}^{\star}, \mathbf{x})\left(k(\mathbf{x}, \mathbf{x}')+\sigma^2\mathbf{I}_n\right)^{-1}\mathbf{y}  \\
            \boldsymbol{\Sigma}_{\mid \mathbf{y}} & =k(\mathbf{x}^{\star}, \mathbf{x}^{\star\prime}) -k(\mathbf{x}^{\star}, \mathbf{x})\left( k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_n \right)^{-1}k(\mathbf{x}, \mathbf{x}^{\star}).
        $$

        The conditioning set is a GPJax `Dataset` object, whilst predictions
        are made on a regular Jax array.

        Example:
            For a `posterior` distribution, the following code snippet will
            evaluate the predictive distribution.
            ```python
                >>> import gpjax as gpx
                >>> import jax.numpy as jnp
                >>>
                >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
                >>> ytrain = jnp.sin(xtrain)
                >>> D = gpx.Dataset(X=xtrain, y=ytrain)
                >>> xtest = jnp.linspace(0, 1).reshape(-1, 1)
                >>>
                >>> prior = gpx.Prior(mean_function = gpx.Zero(), kernel = gpx.RBF())
                >>> posterior = prior * gpx.Gaussian(num_datapoints = D.n)
                >>> predictive_dist = posterior(xtest, D)
            ```

        Args:
            test_inputs (Num[Array, "N D"]): A Jax array of test inputs at which the
                predictive distribution is evaluated.
            train_data (Dataset): A `gpx.Dataset` object that contains the input and
                output data used for training dataset.

        Returns
        -------
            GaussianDistribution: A
                function that accepts an input array and returns the predictive
                    distribution as a `GaussianDistribution`.
        """
        # Unpack training data


        x, y, n = train_data.X, train_data.y, train_data.n

        # Unpack test inputs
        t, n_test = test_inputs, test_inputs.shape[0]

        # Observation noise o²
        obs_noise = self.likelihood.obs_noise
        mx = self.prior.mean_function.pred_state(x,state_train)
        # mx =  self.prior.mean_function(x,-1)

        # start_t = time.time()
        # Precompute Gram matrix, Kxx, at training inputs, x
        # Kxx = self.prior.kernel.gram(x) + (identity(n) * self.prior.jitter)   
        x_state = jnp.concatenate([x,state_train],axis=1)
        x_test_state = jnp.concatenate([t,state_test],axis=1)

        Kxx = self.prior.kernel.gram(x_state) + (identity(n) * self.prior.jitter)
    
        # print("Kxx Optim",{time.time() - start_t},flush=True)
        # Σ = Kxx + Io²
        Sigma = Kxx + identity(n) * obs_noise*weights_noise

        mean_t = self.prior.mean_function.pred_state(t,state_test) ## Make quicker
        # mean_t = self.prior.mean_function(t,-1) ## Make quicker

        Kxt = self.prior.kernel.cross_covariance(x, t)

        base_kernel = RBF(active_dims=[0]).replace(
            lengthscale=jnp.array([10.0]),
            variance=jnp.array([1.0]), # HACK, needs to be 1
        )
        # Kxt_state = self.prior.kernel.cross_covariance(state_train, state_test)
        Kxt_state = base_kernel.cross_covariance(state_train, state_test)
        
        # Kxt_full = Kxt 
        Kxt_full = jnp.multiply(Kxt,Kxt_state)
        # print("KXT_FULL",Kxt_state,flush=True)

        L = jnp.linalg.cholesky(Sigma.to_dense())
        alpha = jnp.linalg.solve(jnp.transpose(L), jnp.linalg.solve(L, y-mx) )
        cov_term = jnp.matmul(jnp.transpose(Kxt_full),alpha)
        # cov_term = jnp.matmul(jnp.transpose(Kxt),alpha)

        # alpha = jnp.linalg.solve(L, jnp.linalg.solve(jnp.transpose(L), Kxt))
        # cov_term = jnp.matmul(alpha,y-mx)
        mean = mean_t + cov_term

        # cov_term = cov_term.at[:,1].set(0.0)
        
        #logger.warning(f'size of mean mat: {mean.shape}')
        
        # start_t = time.time()
        # Ktt  -  Ktx (Kxx + Io²)⁻¹ Kxt, TODO: Take advantage of covariance structure to compute Schur complement more efficiently.
        
        # v = jnp.linalg.solve(L,Kxt_full)
        # covariance = 1 - jnp.sum(jnp.square(v),axis=0) # Variance of the GP

        covariance = jnp.ones(shape=(n_test,))
        #covariance = Ktt - jnp.matmul(Kxt.T, Sigma_inv_Kxt)
        #covariance += identity(n_test) * self.prior.jitter
        #logger.warning(f'size of uncertainty mat: {covariance}')
        
        # print("Covariance Optim",{time.time() - start_t},flush=True)

        return mean.squeeze(), covariance,alpha, Kxt

def map_fit(  # noqa: PLR0913
    *,
    model1: Module,
    model2: Module,
    objective: Union[AbstractObjective, Callable[[Module, Dataset], ScalarFloat]],
    train_data: Dataset,
    optim: ox.GradientTransformation,
    key: KeyArray,
    num_iters: Optional[int] = 100,
    batch_size: Optional[int] = -1,
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
    unroll: Optional[int] = 1,
    safe: Optional[bool] = True,
) -> Tuple[Module, Array]:
    r"""Train a Module model with respect to a supplied Objective function.
    Optimisers used here should originate from Optax.

    Example:
    ```python
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import optax as ox
        >>> import gpjax as gpx
        >>>
        >>> # (1) Create a dataset:
        >>> X = jnp.linspace(0.0, 10.0, 100)[:, None]
        >>> y = 2.0 * X + 1.0 + 10 * jr.normal(jr.PRNGKey(0), X.shape)
        >>> D = gpx.Dataset(X, y)
        >>>
        >>> # (2) Define your model:
        >>> class LinearModel(gpx.Module):
                weight: float = gpx.param_field()
                bias: float = gpx.param_field()

                def __call__(self, x):
                    return self.weight * x + self.bias

        >>> model = LinearModel(weight=1.0, bias=1.0)
        >>>
        >>> # (3) Define your loss function:
        >>> class MeanSquareError(gpx.AbstractObjective):
                def evaluate(self, model: LinearModel, train_data: gpx.Dataset) -> float:
                    return jnp.mean((train_data.y - model(train_data.X)) ** 2)
        >>>
        >>> loss = MeanSqaureError()
        >>>
        >>> # (4) Train!
        >>> trained_model, history = gpx.fit(
                model=model, objective=loss, train_data=D, optim=ox.sgd(0.001), num_iters=1000
            )
    ```

    Args:
        model (Module): The model Module to be optimised.
        objective (Objective): The objective function that we are optimising with
            respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        optim (GradientTransformation): The Optax optimiser that is to be used for
            learning a parameter set.
        num_iters (Optional[int]): The number of optimisation steps to run. Defaults
            to 100.
        batch_size (Optional[int]): The size of the mini-batch to use. Defaults to -1
            (i.e. full batch).
        key (Optional[KeyArray]): The random key to use for the optimisation batch
            selection. Defaults to jr.PRNGKey(42).
        log_rate (Optional[int]): How frequently the objective function's value should
            be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults
            to True.
        unroll (int): The number of unrolled steps to use for the optimisation.
            Defaults to 1.

    Returns
    -------
        Tuple[Module, Array]: A Tuple comprising the optimised model and training
            history respectively.
    """
    if safe:
        # Check inputs.
        _check_model(model)
        _check_train_data(train_data)
        _check_optim(optim)
        _check_num_iters(num_iters)
        _check_batch_size(batch_size)
        _check_prng_key(key)
        _check_log_rate(log_rate)
        _check_verbose(verbose)

    # Unconstrained space loss function with stop-gradient rule for non-trainable params.
    def loss(model1: Module,model2: Module, batch1: Dataset,batch2: Dataset) -> ScalarFloat:
        model1 = model1.stop_gradient()
        model2 = model2.stop_gradient()
        # print(batch1.X.values, flush=True)
        l1 = objective(model1.constrain(), batch1)
        l2 = objective(model2.constrain(), batch2)
        return l1+l2

    # Unconstrained space model.
    model1 = model1.unconstrain()
    model2 = model2.unconstrain()

    # Initialise optimiser state.
    # state = optim.init(model1)
    state = optim.init({"model1":model1,"model2":model2})

    # Mini-batch random keys to scan over.
    iter_keys = jax.random.split(key, num_iters)

    batch1 = train_data.replace(y=train_data.y[:,[0]])
    batch2 = train_data.replace(y=train_data.y[:,[1]])

    # Optimisation step.
    def step(carry, key):
        model1,model2, opt_state = carry

        # if batch_size != -1:
        #     batch = get_batch(train_data, batch_size, key)
        # else:
        
        # batch1.y = train_data.y[:,0]
        # batch2.y = train_data.y[:,1]
        loss_val, loss_gradient = jax.value_and_grad(loss)(model1,model2, batch1,batch2)

        updates, opt_state = optim.update(loss_gradient, opt_state, {"model1":model1,"model2":model2})
        full_updates = ox.apply_updates({"model1":model1,"model2":model2}, updates)


        # ## Updating the mean function for the updates of the second model
        # new_mean = updates.prior.mean_function.replace(chosen_bd=1)
        # updates = updates.replace(prior=updates.prior.replace(mean_function=new_mean))
        
        # # updates, opt_state2 = optim.update(loss_gradient, opt_state2, model2)
        # model2 = ox.apply_updates(model2, updates)
        carry = full_updates['model1'],full_updates['model2'],opt_state
        # carry = model1, model2, opt_state
        return carry, loss_val

    # Optimisation scan.
    scan = vscan if verbose else jax.lax.scan
    start_t = time.time()
    # Optimisation loop.
    (model1,model2, _), history = scan(step, (model1,model2, state), (iter_keys), unroll=unroll)
    print("Scan Optim",{time.time() - start_t},flush=True)
    # Constrained space.
    model1 = model1.constrain()
    model2 = model2.constrain()

    return model1,model2, history


def map_fit_test(  # noqa: PLR0913
    *,
    model1: Module,
    model2: Module,
    objective: Union[AbstractObjective, Callable[[Module, Dataset], ScalarFloat]],
    train_data: Dataset,
    optim: ox.GradientTransformation,
    key: KeyArray,
    num_iters: Optional[int] = 100,
    batch_size: Optional[int] = -1,
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
    unroll: Optional[int] = 1,
    safe: Optional[bool] = True,
) -> Tuple[Module, Array]:
    r"""Train a Module model with respect to a supplied Objective function.
    Optimisers used here should originate from Optax.

    Example:
    ```python
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import optax as ox
        >>> import gpjax as gpx
        >>>
        >>> # (1) Create a dataset:
        >>> X = jnp.linspace(0.0, 10.0, 100)[:, None]
        >>> y = 2.0 * X + 1.0 + 10 * jr.normal(jr.PRNGKey(0), X.shape)
        >>> D = gpx.Dataset(X, y)
        >>>
        >>> # (2) Define your model:
        >>> class LinearModel(gpx.Module):
                weight: float = gpx.param_field()
                bias: float = gpx.param_field()

                def __call__(self, x):
                    return self.weight * x + self.bias

        >>> model = LinearModel(weight=1.0, bias=1.0)
        >>>
        >>> # (3) Define your loss function:
        >>> class MeanSquareError(gpx.AbstractObjective):
                def evaluate(self, model: LinearModel, train_data: gpx.Dataset) -> float:
                    return jnp.mean((train_data.y - model(train_data.X)) ** 2)
        >>>
        >>> loss = MeanSqaureError()
        >>>
        >>> # (4) Train!
        >>> trained_model, history = gpx.fit(
                model=model, objective=loss, train_data=D, optim=ox.sgd(0.001), num_iters=1000
            )
    ```

    Args:
        model (Module): The model Module to be optimised.
        objective (Objective): The objective function that we are optimising with
            respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        optim (GradientTransformation): The Optax optimiser that is to be used for
            learning a parameter set.
        num_iters (Optional[int]): The number of optimisation steps to run. Defaults
            to 100.
        batch_size (Optional[int]): The size of the mini-batch to use. Defaults to -1
            (i.e. full batch).
        key (Optional[KeyArray]): The random key to use for the optimisation batch
            selection. Defaults to jr.PRNGKey(42).
        log_rate (Optional[int]): How frequently the objective function's value should
            be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults
            to True.
        unroll (int): The number of unrolled steps to use for the optimisation.
            Defaults to 1.

    Returns
    -------
        Tuple[Module, Array]: A Tuple comprising the optimised model and training
            history respectively.
    """
    if safe:
        # Check inputs.
        _check_model(model)
        _check_train_data(train_data)
        _check_optim(optim)
        _check_num_iters(num_iters)
        _check_batch_size(batch_size)
        _check_prng_key(key)
        _check_log_rate(log_rate)
        _check_verbose(verbose)

    # Unconstrained space loss function with stop-gradient rule for non-trainable params.
    def loss(model1: Module,model2: Module, batch1: Dataset,batch2: Dataset) -> ScalarFloat:
        model1 = model1.stop_gradient()
        model2 = model2.stop_gradient()
        l1 = objective(model1.constrain(), batch1)
        l2 = objective(model2.constrain(), batch2)
        return l1+l2

    # Unconstrained space model.
    model1 = model1.unconstrain()
    model2 = model2.unconstrain()

    # Initialise optimiser state.
    state = optim.init(model1)

    # Mini-batch random keys to scan over.
    iter_keys = jax.random.split(key, num_iters)
    batch1 = Dataset(X=train_data.X,y=train_data.y.at[:,[0]].get())#train_data.replace(y=train_data.y[:,[0]])
    batch2 = Dataset(X=train_data.X,y=train_data.y.at[:,[1]].get())

    # Optimisation step.
    def step(carry, key):
        model1,model2, opt_state = carry

        loss_val, loss_gradient = jax.value_and_grad(loss)(model1,model2, batch1,batch2)

        updates, opt_state = optim.update(loss_gradient, opt_state, model1)
        model1 = ox.apply_updates(model1, updates)


        ## Updating the mean function for the updates of the second model
        new_mean = updates.prior.mean_function.replace(chosen_bd=1)
        updates = updates.replace(prior=updates.prior.replace(mean_function=new_mean))
        
        # updates, opt_state2 = optim.update(loss_gradient, opt_state2, model2)
        model2 = ox.apply_updates(model2, updates)

        carry = model1, model2, opt_state
        return carry, loss_val

    # Optimisation scan.
    scan = vscan if verbose else jax.lax.scan

    # Optimisation loop.
    (model1,model2, _), history = scan(step, (model1,model2, state), (iter_keys), unroll=unroll)

    # Constrained space.
    model1 = model1.constrain()
    model2 = model2.constrain()

    return model1,model2, history



feature_space_dim = 2


class Network(nn.Module):
    """A simple MLP."""

    @nn.compact
    def __call__(self, x):
        hidden = nn.Dense(features=4)(x)
        # x = nn.relu(x)
        hidden = nn.LayerNorm()(hidden)
        x = nn.Dense(features=2)(hidden)
        x = nn.tanh(x)
        return x

@dataclass
class DeepKernelFunction(AbstractKernel):
    base_kernel: AbstractKernel = None
    network: nn.Module = static_field(None)
    # dummy_x: jax.Array = static_field(None)
    obs_dimensions: Float = static_field(None)
    # key: jax.random.PRNGKeyArray = static_field(jax.random.PRNGKey(123))
    nn_params: Any = field(init=False, repr=False)

    def __post_init__(self):
        if self.base_kernel is None:
            raise ValueError("base_kernel must be specified")
        if self.network is None:
            raise ValueError("network must be specified")
        # self.nn_params = flax.core.unfreeze(self.network.init(self.key, self.dummy_x))
        dummy_batch = jnp.ones(shape=(1,self.obs_dimensions))
        self.nn_params = flax.core.unfreeze(self.network.init(jax.random.PRNGKey(123), dummy_batch))

    def __call__(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, "1"]:

        action_x = x.at[-2:].get()
        action_y = y.at[-2:].get()
        xt = self.network.apply(self.nn_params, x)
        yt = self.network.apply(self.nn_params, y)

        inputs_gp_xt = jnp.concatenate([action_x,xt],axis=0)
        inputs_gp_yt = jnp.concatenate([action_y,yt],axis=0)
        print("SHAPE INPUT ",inputs_gp_xt.shape,flush=True)
        return self.base_kernel(inputs_gp_xt, inputs_gp_yt)
        # return self.base_kernel(xt, yt)


class MAPConjugateMLL(AbstractObjective):
    def step(
        self,
        posterior: "gpjax.gps.ConjugatePosterior",  # noqa: F821
        train_data: Dataset,  # noqa: F821
    ) -> ScalarFloat:
        r"""Evaluate the marginal log-likelihood of the Gaussian process.

        Compute the marginal log-likelihood function of the Gaussian process.
        The returned function can then be used for gradient based optimisation
        of the model's parameters or for model comparison. The implementation
        given here enables exact estimation of the Gaussian process' latent
        function values.

        For a training dataset $`\{x_n, y_n\}_{n=1}^N`$, set of test inputs
        $`\mathbf{x}^{\star}`$ the corresponding latent function evaluations are given
        by $`\mathbf{f}=f(\mathbf{x})`$ and $`\mathbf{f}^{\star}f(\mathbf{x}^{\star})`$,
        the marginal log-likelihood is given by:
        ```math
        \begin{align}
            \log p(\mathbf{y}) & = \int p(\mathbf{y}\mid\mathbf{f})p(\mathbf{f}, \mathbf{f}^{\star}\mathrm{d}\mathbf{f}^{\star}\\
            &=0.5\left(-\mathbf{y}^{\top}\left(k(\mathbf{x}, \mathbf{x}') +\sigma^2\mathbf{I}_N  \right)^{-1}\mathbf{y}-\log\lvert k(\mathbf{x}, \mathbf{x}') + \sigma^2\mathbf{I}_N\rvert - n\log 2\pi \right).
        \end{align}
        ```

        For a given ``ConjugatePosterior`` object, the following code snippet shows
        how the marginal log-likelihood can be evaluated.

        Example:
        ```python
            >>> import gpjax as gpx
            >>>
            >>> xtrain = jnp.linspace(0, 1).reshape(-1, 1)
            >>> ytrain = jnp.sin(xtrain)
            >>> D = gpx.Dataset(X=xtrain, y=ytrain)
            >>>
            >>> meanf = gpx.mean_functions.Constant()
            >>> kernel = gpx.kernels.RBF()
            >>> likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
            >>> prior = gpx.Prior(mean_function = meanf, kernel=kernel)
            >>> posterior = prior * likelihood
            >>>
            >>> mll = gpx.ConjugateMLL(negative=True)
            >>> mll(posterior, train_data = D)
        ```

        Our goal is to maximise the marginal log-likelihood. Therefore, when optimising
        the model's parameters with respect to the parameters, we use the negative
        marginal log-likelihood. This can be realised through

        ```python
            mll = gpx.ConjugateMLL(negative=True)
        ```

        For optimal performance, the marginal log-likelihood should be ``jax.jit``
        compiled.
        ```python
            mll = jit(gpx.ConjugateMLL(negative=True))
        ```

        Args:
            posterior (ConjugatePosterior): The posterior distribution for which
                we want to compute the marginal log-likelihood.
            train_data (Dataset): The training dataset used to compute the
                marginal log-likelihood.

        Returns
        -------
            ScalarFloat: The marginal log-likelihood of the Gaussian process for the
                current parameter set.
        """
        x, y, n = train_data.X, train_data.y, train_data.n

        # Observation noise o²
        obs_noise = posterior.likelihood.obs_noise
        # mx1 = posterior.prior.mean_function(x,chosen_bd = 0)
        # mx2 = posterior.prior.mean_function(x,chosen_bd = 1)

        mx = posterior.prior.mean_function(x,chosen_bd = -1)
        mx1, mx2 = mx[:,[0]],mx[:,[1]]

        # Σ = (Kxx + Io²) = LLᵀ
        Kxx = posterior.prior.kernel.gram(x)
        Kxx += identity(n) * posterior.prior.jitter
        Sigma = Kxx + identity(n) * obs_noise

        # p(y | x, θ), where θ are the model hyperparameters:
        mll1 = GaussianDistribution(jnp.atleast_1d(mx1.squeeze()), Sigma)
        mll2 = GaussianDistribution(jnp.atleast_1d(mx2.squeeze()), Sigma)

        l1 = self.constant * (mll1.log_prob(jnp.atleast_1d(y.at[:,[0]].get().squeeze())).squeeze())
        l2 = self.constant * (mll2.log_prob(jnp.atleast_1d(y.at[:,[1]].get().squeeze())).squeeze())

        return l1+l2
