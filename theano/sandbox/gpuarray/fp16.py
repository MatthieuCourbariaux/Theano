from theano import Op, Apply

from .basic_ops import as_gpuarray_variable

try:
    from nervanagpu.nervanagpu import GPUTensor
except ImportError:
    GPUTensor = None


def to_gputensor(a):
    assert a.flags.c_contiguous or a.flags.f_contiguous
    return GPUTensor(a.shape, dtype=a.dtype, base=a, gpudata=a.gpudata,
                     strides=a.strides, is_trans=a.flags.f_contiguous)


class Gemm16(Op):
    __props__ = ('relu', 'inplace')

    def __init__(self, relu=False, inplace=False):
        self.relu = relu
        self.inplace = inplace

    def make_node(self, C, alpha, A, B, beta):
        if GPUTensor is None:
            raise RuntimeError("Can't use Gemm16: nervanagpu not found")

        A = as_gpuarray_variable(A)
        B = as_gpuarray_variable(B)
        C = as_gpuarray_variable(C)

        assert C.dtype == A.dtype == B.dtype == 'float16'

        return Apply(self, [C, alpha, A, B, beta], [C.type()])

    def perform(self, node, inputs, outputs):
        C, alpha, A, B, beta = inputs
        inplace = self.inplace
        if inplace and not C.flags.forc:
            inplace = False
        if not inplace:
            C = C.copy()
        At = to_gputensor(A)
        Bt = to_gputensor(B)
        Ct = to_gputensor(C)
        outputs[0][0] = At.dot(At, Bt, Ct, alpha=alpha, beta=beta,
                               relu=self.relu)
