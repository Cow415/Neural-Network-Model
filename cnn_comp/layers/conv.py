"""
2D Convolutional Layer Class
"""
import numpy as np

class Conv2D:
    """2D Convolutional layer."""
    def __init__(self, in_c, out_c, k, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.k = k

        scale = np.sqrt(2 / (in_c * k * k))
        self.W = np.random.randn(out_c, in_c, k, k) * scale
        self.b = np.zeros(out_c)

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        F, _, KH, KW = self.W.shape

        H_out = (H + 2*self.padding - KH) // self.stride + 1
        W_out = (W + 2*self.padding - KW) // self.stride + 1

        x_pad = np.pad(x,
            ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding))
        )
        self.x_pad = x_pad

        out = np.zeros((N, F, H_out, W_out))

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h = i * self.stride
                        w = j * self.stride
                        out[n,f,i,j] = np.sum(
                            x_pad[n,:,h:h+KH,w:w+KW] * self.W[f]
                        ) + self.b[f]

        return out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        F, _, KH, KW = self.W.shape

        dx = np.zeros_like(self.x_pad)
        self.dW = np.zeros_like(self.W)
        self.db = np.sum(dout, axis=(0,2,3))

        H_out, W_out = dout.shape[2:]

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h = i * self.stride
                        w = j * self.stride
                        dx[n,:,h:h+KH,w:w+KW] += self.W[f] * dout[n,f,i,j]
                        self.dW[f] += self.x_pad[n,:,h:h+KH,w:w+KW] * dout[n,f,i,j]

        return dx[:,:,self.padding:-self.padding,self.padding:-self.padding]

    def params(self): return [self.W, self.b]
    def grads(self): return [self.dW, self.db]
