import copy

from functorch.einops import rearrange
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def max_one_hot(logits: torch.Tensor, straight_through_grads: bool = True, dim: int = -1):
  """
  Gets one hot of argmax over logits with straight-through gradients.

  Args:
    logits - shape (..., n_classes, ...)
  """
  indices = logits.argmax(dim=dim)
  ohs = F.one_hot(indices, num_classes=logits.shape[dim])
  if dim != -1 and dim != len(logits.shape) - 1:
    ohs = ohs.transpose(dim, -1)

  if straight_through_grads:
    ohs = ohs + logits - logits.detach()

  return ohs


class ReshapeLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)
    
class CReLU(nn.Module):
  def forward(self, x):
    return F.relu(torch.cat([x, -x], dim=-1))


# Source: https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        # Size of the codebook
        self._num_embeddings = n_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(n_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(n_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def quantized_decode(self, oh_encodings):
      """ Decodes from one-hot encodings. """
      batch_size, n_latents, latent_dim = oh_encodings.shape
      flat_oh_encodings = oh_encodings.reshape(-1, latent_dim)
      # Quantize and unflatten
      quantized = torch.matmul(flat_oh_encodings, self._embedding.weight.detach())
      quantized = quantized.view(batch_size, n_latents, self._embedding_dim)
      return quantized.permute(0, 2, 1)
      
    def decode(self, encoding_indices):
      """ Decodes from `LongTensor` encodings. """
      batch_size, n_latents = encoding_indices.shape
      encoding_indices = encoding_indices.reshape(-1, 1)
      encodings = F.one_hot(encoding_indices, self._num_embeddings).float()
      # Quantize and unflatten
      quantized = torch.matmul(encodings, self._embedding.weight)
      quantized = quantized.view(batch_size, n_latents, self._embedding_dim)
      return quantized.permute(0, 2, 1)

    def forward(self, inputs, codebook_mask=None, masked_input=True, masked_output=True):
      """
      Input can be BC(WH) or BCWH, C is embedding dim, (WH) is n latents
      convert inputs from BCHW -> BHWC

      codebook_mask only masks codebook for quantization output unless
      masked_input is True, in which case it masks the input as well.
      """

      if len(inputs.shape) == 4:
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
      else:
        inputs = inputs.permute(0, 2, 1).contiguous()
      input_shape = inputs.shape
      
      # Flatten input
      flat_input = inputs.view(-1, self._embedding_dim)
      
      # Get the codebook weights
      codebook = self._embedding.weight
      if codebook_mask is not None and masked_input:
        codebook = codebook * codebook_mask
      
      # Calculate distances
      distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                  + torch.sum(codebook**2, dim=1)
                  - 2 * torch.matmul(flat_input, codebook.t()))
      
      # Encoding
      flat_oh_encodings = max_one_hot(-distances)
      oh_encodings = rearrange(flat_oh_encodings,
        '(b n) d -> b d n', b=inputs.shape[0])
      
      # Quantize and unflatten
      with torch.no_grad():
        codebook = self._embedding.weight
        if codebook_mask is not None and masked_output:
          codebook = codebook * codebook_mask
        quantized = torch.matmul(flat_oh_encodings, codebook).view(input_shape)
      
      # Use EMA to update the embedding vectors
      if self.training:
          self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                    (1 - self._decay) * torch.sum(flat_oh_encodings.detach(), 0)
          
          # Laplace smoothing of the cluster size
          n = torch.sum(self._ema_cluster_size.data)
          self._ema_cluster_size = (
              (self._ema_cluster_size + self._epsilon)
              / (n + self._num_embeddings * self._epsilon) * n)
          
          dw = torch.matmul(flat_oh_encodings.detach().t(), flat_input)
          self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
          
          self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
      
      # Loss
      e_latent_loss = F.mse_loss(quantized, inputs)
      loss = self._commitment_cost * e_latent_loss
      
      # Straight Through Estimator
      quantized = inputs + (quantized - inputs).detach()
      avg_probs = torch.mean(flat_oh_encodings, dim=0)
      perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
      
      # convert quantized from BHWC -> BCHW
      if len(input_shape) == 4:
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, oh_encodings
      return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, oh_encodings


class VQVAEModel(nn.Module):
  def __init__(self, obs_dim, codebook_size, embedding_dim, encoder=None,
               decoder=None, n_latents=None, quantized_enc=False, sparsity=0.0,
               sparsity_type='random'):
    super().__init__()
    if quantized_enc:
      self.encoder_type = 'soft_vqvae'
    else:
      self.encoder_type = 'vqvae'
    self.codebook_size = codebook_size
    self.embedding_dim = embedding_dim
    self.n_embeddings = embedding_dim if quantized_enc else codebook_size
    self.quantized_enc = quantized_enc
    self.encoder = encoder
    self.quantizer = VectorQuantizerEMA(
      codebook_size, embedding_dim,
      commitment_cost=0.25, decay=0.99)
    self.decoder = decoder

    test_input = torch.ones(1, *obs_dim, dtype=torch.float32)
    self.encoder_out_shape = self.encoder(test_input).shape[1:]
    self.n_latent_embeds = np.prod(self.encoder_out_shape[1:])
    
    if n_latents is not None:
      if len(self.encoder_out_shape) == 1:
        # When the encoder has a flat output, make a layer to create channels
        self.encoder = nn.Sequential(
          self.encoder,
          nn.Linear(self.encoder_out_shape[0], embedding_dim * n_latents),
          ReshapeLayer(-1, embedding_dim, n_latents))
        self.decoder = nn.Sequential(
          ReshapeLayer(-1, embedding_dim * n_latents),
          nn.Linear(embedding_dim * n_latents, self.encoder_out_shape[0]),
          self.decoder)
        
        self.encoder_out_shape = (embedding_dim, n_latents)
        self.n_latent_embeds = n_latents
      else:
        raise NotImplementedError('`n_latents` param not supported for >1-D encoder outputs!')
    elif len(self.encoder_out_shape) == 1:
      raise ValueError('VQVAEs with dense encoders must have a value for `n_latents`!')

    # Create sparsity masks if needed
    sparsity_mask = self.create_sparsity_mask(sparsity, sparsity_type)
    if sparsity_mask is None:
      self.sparsity_mask = None
    else:
      self.register_buffer('sparsity_mask', sparsity_mask)
    self.sparsity_enabled = False

    # Flat representation size
    # num classes x num vectors in a single latent vector
    self.latent_dim = self.n_embeddings * self.n_latent_embeds

  def get_codebook(self):
    codebook = self.quantizer._embedding.weight
    if self.sparsity_enabled and self.sparsity_mask is not None:
      codebook = codebook * self.sparsity_mask
    # (n_embeddings, embedding_dim)
    return codebook

  def create_sparsity_mask(self, sparsity=0.0, sparsity_type='random'):
    if self.encoder_type == 'soft_vqvae' and \
       (sparsity > 0 or sparsity_type == 'identity'):
      if sparsity_type == 'random':
        random_idxs = torch.randn((self.n_embeddings, self.embedding_dim))
        sorted_indices = torch.sort(random_idxs, dim=1)[1]
        n_zeros = int(self.embedding_dim * sparsity)
        sparsity_mask = sorted_indices >= n_zeros
        sparsity_mask = sparsity_mask.float()
      elif sparsity_type == 'identity':
        assert self.n_embeddings == self.embedding_dim, \
          'Identity sparsity only supported for square matrices!'
        sparsity_mask = torch.eye(self.embedding_dim)
    else:
      sparsity_mask = None
    
    return sparsity_mask

  def enable_sparsity(self):
    self.sparsity_enabled = True

  def disable_sparsity(self):
    self.sparsity_enabled = False

  def get_encoder(self):
    new_quantizer = copy.copy(self.quantizer)
    new_quantizer.full_forward = self.quantizer.forward
    if self.quantized_enc:
      new_quantizer.forward = lambda x: new_quantizer.full_forward(x)[1]
    else:
      new_quantizer.forward = lambda x: new_quantizer.full_forward(x)[3]

    encoder = nn.Sequential(
      self.encoder,
      new_quantizer
    )

    encoder.encoder_type = self.encoder_type
    encoder.encoder_out_shape = self.encoder_out_shape
    encoder.n_latent_embeds = self.n_latent_embeds
    encoder.n_embeddings = self.n_embeddings
    encoder.embedding_dim = self.embedding_dim

    return encoder

  def forward(self, x):
    encoder_out = self.encoder(x)
    if self.sparsity_enabled and self.sparsity_mask is not None:
      quantizer_loss, quantized, perplexity, oh_encodings = self.quantizer(
        encoder_out, self.sparsity_mask, masked_input=False, masked_output=False)
      x_hat = self.decoder(quantized)
      quantized = self.quantizer(encoder_out, self.sparsity_mask, masked_input=False)[1]
    else:
      quantizer_loss, quantized, perplexity, oh_encodings = self.quantizer(encoder_out)
      x_hat = self.decoder(quantized)
    out_encodings = quantized.reshape(x.shape[0], self.n_embeddings, -1) \
      if self.quantized_enc else oh_encodings

    return x_hat, {
      'loss': quantizer_loss,
      'perplexity': perplexity,
      'encodings': out_encodings,
    }

  def encode(self, x, return_one_hot=False, as_long=True):
    encoder_out = self.encoder(x)
    mask = self.sparsity_mask if self.sparsity_enabled else None
    _, quantized, quantizer_loss, oh_encodings = self.quantizer(
      encoder_out, mask, masked_input=False)
    if self.quantized_enc:
      quantized = quantized.reshape(x.shape[0], self.n_embeddings, -1)
      return quantized
    elif return_one_hot or not as_long:
      return oh_encodings
    else:
      return oh_encodings.argmax(dim=1)

  def decode(self, encodings):
    if self.quantized_enc:
      quantized = encodings
      if self.sparsity_enabled and self.sparsity_mask is not None:
        quantized = quantized.view(
          quantized.shape[0], self.n_embeddings, self.n_latent_embeds)
        quantized = self.quantizer(
          quantized, self.sparsity_mask, masked_output=False)[1]
    else:
      quantized = self.quantizer.decode(encodings.long())
    quantized = quantized.view(encodings.shape[0], *self.encoder_out_shape)
    return self.decoder(quantized)

  def quantize_logits(self, logits):
    quantizer_loss, quantized = self.quantizer(logits)[:2]
    return quantized, quantizer_loss

  def decode_from_quantized(self, quantized):
    quantized = quantized.view(quantized.shape[0], *self.encoder_out_shape)
    return self.decoder(quantized)