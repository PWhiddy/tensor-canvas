import eagerpy as ep
import torch

def make_uv(t):
  '''
  Generates UV coordinates
  Returns tensors of x coords and y coords each with shape matching t
  '''
  uvx = ep.expand_dims(ep.arange(t, 0.0, t.shape[1], 1), axis=0).tile([t.shape[0],1])
  uvy = ep.expand_dims(ep.arange(t, 0.0, t.shape[0], 1), axis=0).tile([t.shape[1],1]).transpose()
  return uvx, uvy

def dist_to_col(dist, color, blend, t):
  msk = ep.clip((dist+blend) / (2.0*blend), 0.0, 1.0)
  msk = msk * msk * (3.0 - 2.0 * msk)
  msk = msk.expand_dims(axis=2).tile([1,1,3])
  col_t = ep.astensor(color).expand_dims(axis=0).expand_dims(axis=0).tile([t.shape[0],t.shape[1],1])
  return msk*t + (1.0-msk)*col_t

def draw_circle(xp, yp, radius, color, t, blend=0.75):
  '''
  Draws a circle onto an image tensor. All units are in pixels
  Parameters:
    xp: x position
    yp: y position
    radius: circle radius
    color: rgb color tensor with shape (3,) and values in the range 0.0-1.0
    blend (optional): blending distance
  Returns:
    tensor with circle drawn onto it
  '''
  if type(t) == torch.Tensor:
    t = t.permute(1, 2, 0)
  t = ep.astensor(t)
  uvx, uvy = make_uv(t)
  dist = ((xp-uvx)**2.0 + (yp-uvy)**2.0 + 1.0).sqrt()-radius
  t = dist_to_col(dist, color, blend, t)
  t = t.raw
  if type(t) == torch.Tensor:
    t = t.permute(2, 0, 1)
  return t

def draw_line(x1, y1, x2, y2, radius, color, t, blend=0.75):
  '''
  Draws a line onto an image tensor. All units are in pixels
  Parameters:
    x1: x position for endpoint 1
    y1: y position for endpoint 1
    x2: x position for endpoint 2
    y2: y position for endpoint 2
    radius: line width (radius)
    color: rgb color tensor with shape (3,) and values in the range 0.0-1.0
    blend (optional): blending distance
  Returns:
    tensor with circle drawn onto it
  '''
  if type(t) == torch.Tensor:
    t = t.permute(1, 2, 0)
  t = ep.astensor(t)
  uvx, uvy = make_uv(t)
  pax, pay, bax, bay = uvx-x1, uvy-y1, x2-x1, y2-y1
  h = ep.clip( (pax*bax+pay*bay)/(bax*bax+bay*bay) , 0.0, 1.0)
  dlx, dly = pax-bax*h, pay-bay*h
  dist = (dlx*dlx+dly*dly).sqrt()-radius
  t = dist_to_col(dist, color, blend, t)
  t = t.raw
  if type(t) == torch.Tensor:
    t = t.permute(2, 0, 1)
  return t
