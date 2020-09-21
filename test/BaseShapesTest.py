import tensorcanvas as tc
import torch
import tensorflow as tf
import numpy as np

def test_circle():
    test_pt = tc.draw_circle(34.8, 5.3, 2.0, torch.tensor([0.3, 0.2, 1.0]), torch.zeros(3,32,64))
    test_pt = tc.draw_circle(14.8, 15.3, 2.0, torch.tensor([0.1, 0.9, 0.8]), test_pt)
    test_pt = tc.draw_circle(24.8, 2.3, 2.0, torch.tensor([0.0, 0.9, 0.0]), test_pt)

    test_tf = tc.draw_circle(34.8, 5.3, 2.0, tf.convert_to_tensor([0.3, 0.2, 1.0]), tf.zeros([32,64,3]))
    test_tf = tc.draw_circle(14.8, 15.3, 2.0, tf.convert_to_tensor([0.1, 0.9, 0.8]), test_tf)
    test_tf = tc.draw_circle(24.8, 2.3, 2.0, tf.convert_to_tensor([0.0, 0.9, 0.0]), test_tf)

    assert(np.allclose( test_tf.numpy(),  test_pt.permute(1,2,0).numpy()) == True)

    