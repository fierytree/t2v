# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Implementation of multiheaded FAVOR-attention & FAVOR-self-attention layers.

Prefix Sum Tensorflow implementation by Valerii Likhosherstov.
"""
import math
import numpy as np
import torch
import torch.nn as nn
# import tensorflow as tf
# from performer.fast_attention.tensorflow import util
from .layers import contract_inner,default_init


class NIN(torch.nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = torch.nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = torch.nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    y = contract_inner(x, self.W) + self.b
    return y

BIG_CONSTANT = 1e8


def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
  r"""Constructs the matrix of random projections.

  Constructs a matrix of random orthogonal projections. Each projection vector
  has direction chosen uniformly at random and either deterministic length
  \sqrt{d} or length taken from the \chi(d) distribution (in the latter case
  marginal distributions of the projections are d-dimensional Gaussian vectors
  with associated identity covariance matrix).

  Args:
    m: number of random projections.
    d: dimensionality of each random projection.
    seed: random seed used to construct projections.
    scaling: 1 if all the random projections need to be renormalized to have
      length \sqrt{d}, 0 if the lengths of random projections should follow
      \chi(d) distribution.
    struct_mode: if True then products of Givens rotations will be used to
      construct random orthogonal matrix. This bypasses Gram-Schmidt
      orthogonalization.

  Returns:
    The matrix of random projections of the shape [m, d].
  """
  nb_full_blocks = int(m / d)
  block_list = []
  current_seed = seed
  for _ in range(nb_full_blocks):
    if struct_mode:
      q = create_products_of_givens_rotations(d, seed)
    else:
      unstructured_block = torch.random.normal((d, d), seed=current_seed)
      q, _ = torch.linalg.qr(unstructured_block)
      q = torch.transpose(q)
    block_list.append(q)
    current_seed += 1
  remaining_rows = m - nb_full_blocks * d
  if remaining_rows > 0:
    if struct_mode:
      q = create_products_of_givens_rotations(d, seed)
    else:
      unstructured_block = torch.random.normal((d, d), seed=current_seed)
      q, _ = torch.linalg.qr(unstructured_block)
      q = torch.transpose(q)
    block_list.append(q[0:remaining_rows])
  final_matrix = torch.experimental.numpy.vstack(block_list)
  current_seed += 1

  if scaling == 0:
    multiplier = torch.norm(torch.random.normal((m, d), seed=current_seed), axis=1)
  elif scaling == 1:
    multiplier = torch.sqrt(float(d)) * torch.ones((m)).to(d.device)
  else:
    raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

  return torch.mm(torch.diag(multiplier), final_matrix)


def create_products_of_givens_rotations(dim, seed):
  r"""Constructs a 2D-tensor which is a product of Givens random rotations.

  Constructs a 2D-tensor of the form G_1 * ... * G_k, where G_i is a Givens
  random rotation. The resulting tensor mimics a matrix taken uniformly at
  random form the orthogonal group.

  Args:
    dim: number of rows/columns of the resulting 2D-tensor.
    seed: random seed.

  Returns:
    The product of Givens random rotations.
  """
  nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
  q = np.eye(dim, dim)
  np.random.seed(seed)
  for _ in range(nb_givens_rotations):
    random_angle = math.pi * np.random.uniform()
    random_indices = np.random.choice(dim, 2)
    index_i = min(random_indices[0], random_indices[1])
    index_j = max(random_indices[0], random_indices[1])
    slice_i = q[index_i]
    slice_j = q[index_j]
    new_slice_i = math.cos(random_angle) * slice_i + math.sin(
        random_angle) * slice_j
    new_slice_j = -math.sin(random_angle) * slice_i + math.cos(
        random_angle) * slice_j
    q[index_i] = new_slice_i
    q[index_j] = new_slice_j
  return torch.from_numpy(q).to(torch.float32)


def relu_kernel_transformation(data,
                               is_query,
                               projection_matrix=None,
                               numerical_stabilizer=0.001):
  """Computes features for the ReLU-kernel.

  Computes random features for the ReLU kernel from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  del is_query
  if projection_matrix is None:
    return torch.relu(data) + numerical_stabilizer
  else:
    ratio = 1.0 / torch.sqrt(
        projection_matrix.shape[0].to(torch.float32))
    data_dash = ratio * torch.einsum("blhd,md->blhm", data, projection_matrix)
    return torch.relu(data_dash) + numerical_stabilizer


def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix=None,
                                  numerical_stabilizer=0.000001):
  """Computes random features for the softmax kernel using FAVOR+ mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
  data_normalizer = 1.0 / (
      torch.sqrt(torch.sqrt(data.shape[-1].to(torch.float32))))
  data = data_normalizer * data
  ratio = 1.0 / torch.sqrt(projection_matrix.shape[0].to(torch.float32))
  data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)
  diag_data = torch.square(data)
  diag_data = torch.sum(
      diag_data, dim=len(data.shape) - 1)
  diag_data = diag_data / 2.0
  diag_data = diag_data.unsqueeze(-1)
  last_dims_t = (len(data_dash.shape) - 1,)
  attention_dims_t = (len(data_dash.shape) - 3,)
  if is_query:
    data_dash = ratio * (
        torch.exp(data_dash - diag_data - torch.max(
            data_dash, dim=last_dims_t, keepdim=True)) + numerical_stabilizer)
  else:
    data_dash = ratio * (
        torch.exp(data_dash - diag_data - torch.max(
            data_dash, dim=last_dims_t + attention_dims_t, keepdim=True)) +
        numerical_stabilizer)

  return data_dash


def noncausal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR noncausal attention AV.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR noncausal attention AV.
  """
  kvs = torch.einsum("lbhm,lbhd->bhmd", ks, vs)
  return torch.einsum("lbhm,bhmd->lbhd", qs, kvs)


def noncausal_denominator(qs, ks):
  """Computes FAVOR normalizer in noncausal attention.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in noncausal attention.
  """
  all_ones = torch.ones([ks.shape[0]]).to(ks.device)
  ks_sum = torch.einsum("lbhm,l->bhm", ks, all_ones)
  return torch.einsum("lbhm,bhm->lbh", qs, ks_sum)


def causal_numerator(qs, ks, vs):
  """Computes not-normalized FAVOR causal attention A_{masked}V.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
  """

  result = []
  sums = torch.zeros_like(torch.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

  for index in range(qs.shape[0]):
    sums = sums + torch.einsum("ijk,ijl->ijkl", ks[index], vs[index])
    result.append(torch.einsum("ijkl,ijk->ijl", sums, qs[index])[None, Ellipsis])

  result = torch.concat(result, axis=0)

  def grad(res_grad):

    grads = torch.zeros_like(torch.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

    gr_sums = sums

    q_grads = []
    k_grads = []
    v_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          torch.einsum("ijkl,ijl->ijk", gr_sums, res_grad[index])[None, Ellipsis])
      grads = grads + torch.einsum("ijk,ijl->ijkl", qs[index], res_grad[index])
      k_grads.append(torch.einsum("ijkl,ijl->ijk", grads, vs[index])[None, Ellipsis])
      v_grads.append(torch.einsum("ijkl,ijk->ijl", grads, ks[index])[None, Ellipsis])
      gr_sums = gr_sums - torch.einsum("ijk,ijl->ijkl", ks[index], vs[index])

    q_grads = torch.concat(q_grads[::-1], axis=0)
    k_grads = torch.concat(k_grads[::-1], axis=0)
    v_grads = torch.concat(v_grads[::-1], axis=0)

    return q_grads, k_grads, v_grads

  return result, grad


def causal_denominator(qs, ks):
  """Computes FAVOR normalizer in causal attention.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].

  Returns:
    FAVOR normalizer in causal attention.
  """

  result = []
  sums = torch.zeros_like(ks[0])

  for index in range(qs.shape[0]):
    sums = sums + ks[index]
    result.append(torch.sum(qs[index] * sums, dim=2)[None, Ellipsis])

  result = torch.concat(result, dim=0)

  def grad(res_grad):

    k_grad = torch.zeros_like(ks[0])

    gr_sums = sums

    q_grads = []
    k_grads = []

    for index in range(qs.shape[0] - 1, -1, -1):

      q_grads.append(
          torch.einsum("ijk,ij->ijk", gr_sums, res_grad[index])[None, Ellipsis])
      k_grad = k_grad + torch.einsum("ijk,ij->ijk", qs[index], res_grad[index])
      k_grads.append(k_grad[None, Ellipsis])
      gr_sums = gr_sums - ks[index]

    q_grads = torch.concat(q_grads[::-1], axis=0)
    k_grads = torch.concat(k_grads[::-1], axis=0)

    return q_grads, k_grads

  return result, grad


def favor_attention(query,
                    key,
                    value,
                    kernel_transformation,
                    causal,
                    projection_matrix=None):
  """Computes FAVOR normalized attention.

  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    kernel_transformation: transformation used to get finite kernel features.
    causal: whether attention is causal or not.
    projection_matrix: projection matrix to be used.

  Returns:
    FAVOR normalized attention.
  """
  query_prime = kernel_transformation(query, True,
                                      projection_matrix)  # [B,L,H,M]
  key_prime = kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]
  query_prime = query_prime.permute(1, 0, 2, 3)  # [L,B,H,M]
  key_prime = key_prime.permute(1, 0, 2, 3)  # [L,B,H,M]
  value = value.permute(1, 0, 2, 3)  # [L,B,H,D]

  if causal:
    av_attention = causal_numerator(query_prime, key_prime, value)
    attention_normalizer = causal_denominator(query_prime, key_prime)
  else:
    av_attention = noncausal_numerator(query_prime, key_prime, value)
    attention_normalizer = noncausal_denominator(query_prime, key_prime)
  # TODO(kchoro): Add more comments.
  av_attention = av_attention.permute(1, 0, 2, 3)
  attention_normalizer = attention_normalizer.permute(1, 0, 2)
  attention_normalizer = attention_normalizer.unsqueeze(-1)
  return av_attention / attention_normalizer


class Attention(nn.Module):
  """Multi-headed attention layer."""

  def __init__(self,
               hidden_size,
               context_size,
               num_heads,
               attention_dropout,
               kernel_transformation=relu_kernel_transformation,
               numerical_stabilizer=0.001,
               causal=False,
               projection_matrix_type=None,
               nb_random_features=0,
               n_head_channels=-1,
               init_scale=0.,):
    """Initialize Attention.

    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
      kernel_transformation: transformation used to produce kernel features for
        attention.
      numerical_stabilizer: used to bound away from zero kernel values.
      causal: whether attention is causal or not.
      projection_matrix_type: None if Identity should be used, otherwise random
        projection matrix will be applied.
      nb_random_features: number of random features to be used (relevant only if
        projection_matrix is not None).
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    _num_heads=num_heads
    if n_head_channels == -1:
        _num_heads = num_heads
    else:
        if hidden_size < n_head_channels:
          _num_heads = 1
        else:
          assert hidden_size % n_head_channels == 0
          _num_heads = hidden_size // n_head_channels

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = _num_heads
    self.attention_dropout = attention_dropout
    self.kernel_transformation = kernel_transformation
    self.numerical_stabilizer = numerical_stabilizer
    self.causal = causal
    self.projection_matrix_type = projection_matrix_type
    self.nb_random_features = nb_random_features

    hidden_dim = self.hidden_size
    self.query_dense_layer = NIN(hidden_dim, hidden_dim)
    self.key_dense_layer = NIN(context_size, hidden_dim)
    self.value_dense_layer = NIN(context_size, hidden_dim)
    self.output_dense_layer = NIN(hidden_dim, hidden_dim, init_scale)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
    }

  def forward(self,
            query_input,
            source_input,
            bias,
            training,
            cache=None,
            decode_loop_step=None):
    """Apply attention mechanism to query_input and source_input.

    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      source_input: A tensor with shape [batch_size, length_source,
        hidden_size].
      bias: A tensor with shape [batch_size, 1, length_query, length_source],
        the attention bias that will be added to the result of the dot product.
      training: A bool, whether in training mode or not.
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, heads, dim_per_head],
             "v": tensor with shape [batch_size, i, heads, dim_per_head]} where
               i is the current decoded length for non-padded decode, or max
               sequence length for padded decode.
      decode_loop_step: An integer, step number of the decoding loop. Used only
        for autoregressive inference on TPU.

    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
    """
    # Linearly project the query, key and value using different learned
    # projections. Splitting heads is automatically done during the linear
    # projections --> [batch_size, length, num_heads, dim_per_head].
    query = self.query_dense_layer(query_input)
    key = self.key_dense_layer(source_input)
    value = self.value_dense_layer(source_input)


    query = query.reshape(
        [query.shape[0], query.shape[1], self.num_heads, -1])
    key = key.reshape([key.shape[0], key.shape[1], self.num_heads, -1])
    value = value.reshape(
        [value.shape[0], value.shape[1], self.num_heads, -1])

    if self.projection_matrix_type is None:
      projection_matrix = None
    else:
      dim = query.shape[-1]
      seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT))
      seed = seed.to(torch.int32)
      projection_matrix = create_projection_matrix(
          self.nb_random_features, dim, seed=seed)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      if decode_loop_step is not None:
        cache_k_shape = cache["k"].shape
        indices = torch.reshape(
            nn.functional.one_hot(decode_loop_step, cache_k_shape[1], dtype=key.dtype),
            [1, cache_k_shape[1], 1, 1])
        key = cache["k"] + key * indices
        cache_v_shape = cache["v"].shape
        indices = torch.reshape(
            nn.functional.one_hot(decode_loop_step, cache_v_shape[1], dtype=value.dtype),
            [1, cache_v_shape[1], 1, 1])
        value = cache["v"] + value * indices
      else:
        key = torch.concat([cache["k"].to(key.dtype), key], axis=1)
        value = torch.concat([cache["v"].to(value.dtype), value], axis=1)

      # Update cache
      cache["k"] = key
      cache["v"] = value

    # attention_output = h
    attention_output = favor_attention(query, key, value,
                                       self.kernel_transformation, self.causal,
                                       projection_matrix)
    attention_output = attention_output.reshape(attention_output.shape[0],attention_output.shape[1],-1)
    attention_output = self.output_dense_layer(attention_output)
    return attention_output


from inspect import isfunction
import torch.nn.functional as F

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
  """Multiheaded self-attention layer."""
  def __init__(self,
               hidden_size,
               context_size,
               num_heads,
               attention_dropout,
               kernel_transformation=relu_kernel_transformation,
               numerical_stabilizer=0.001,
               causal=False,
               projection_matrix_type=None,
               nb_random_features=0,
               n_head_channels=-1,
               init_scale=0.,):
    super(SelfAttention, self).__init__()

    self.self_attention = Attention(hidden_size,hidden_size,num_heads,attention_dropout,
      kernel_transformation,numerical_stabilizer,causal,projection_matrix_type,
      nb_random_features,n_head_channels,init_scale,)
    self.cross_attention = Attention(hidden_size,context_size,num_heads,attention_dropout,
      kernel_transformation,numerical_stabilizer,causal,projection_matrix_type,
      nb_random_features,n_head_channels,init_scale,)
    self.ff = FeedForward(hidden_size, dropout=attention_dropout, glu=True)
    self.norm1 = torch.nn.LayerNorm(hidden_size)
    self.norm2 = torch.nn.LayerNorm(hidden_size)
    self.norm3 = torch.nn.LayerNorm(hidden_size)

    self.hidden_size = hidden_size
    self.num_heads = self.self_attention.num_heads
    self.attention_dropout = attention_dropout
    self.kernel_transformation = kernel_transformation
    self.numerical_stabilizer = numerical_stabilizer
    self.causal = causal
    self.projection_matrix_type = projection_matrix_type
    self.nb_random_features = nb_random_features

  def forward(self,
           query_input,
           source_input,
           training=True,
           cache=None,
           decode_loop_step=None):

    x = self.norm1(query_input)
    bias = torch.ones(1).to(query_input.device)
    dim_per_head = self.hidden_size // self.num_heads
    # cache = {
    #     "k": torch.zeros([1, 0, self.num_heads, dim_per_head]),
    #     "v": torch.zeros([1, 0, self.num_heads, dim_per_head]),
    # }
    
    x = self.self_attention(x, x, bias, training, cache, decode_loop_step) + x
    x = self.norm2(query_input)
    x = self.cross_attention(x, source_input, bias, training, cache, decode_loop_step) + x
    x = self.ff(self.norm3(x)) + x
    return x
