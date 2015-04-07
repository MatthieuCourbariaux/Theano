
import numpy
# from theano.scalar.basic import UnaryScalarOp, float32, int16, as_scalar, Apply
from theano.scalar.basic import UnaryScalarOp, as_scalar, Apply, Cast
from theano.scalar.basic import Scalar, float32
# from theano.tensor.basic import TensorType
from theano.tensor.elemwise import Elemwise

class Float16(Scalar):
    
    def __init__(self):
        Scalar.__init__(self, dtype='float16')
    
    def dtype_specs(self):
        return (numpy.float16, 'npy_uint16', 'Float16')

float16 = Float16()

# as in scalar.basic
class Float2half(Cast):
    """
    This operation converts a float32 to a float16
    """
    def __init__(self, name=None):
        Cast.__init__(self, float16, name)
    
    # Do not forget to theano-cache clear after each modification
    def c_code(self, node, name, (x,), (y,), sub):
        return "%(y)s = __float2half_rn(%(x)s);" % locals()  

float2_half_scalar = Float2half(name='float2half')

# as in tensor.basic
float2half = Elemwise(float2_half_scalar)

class Half2float(UnaryScalarOp):
    """
    This operation converts a float16 to a float32.
    """
    def make_node(self, x):
        x = as_scalar(x)
        y = float32()
        return Apply(self, [x], [y])
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = __half2float(%(x)s);" % locals()  
        
half2float_scalar = Half2float(name='half2float')
half2float = Elemwise(half2float_scalar)
