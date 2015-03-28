
from theano.scalar.basic import UnaryScalarOp, int16, float32, as_scalar, Apply
from theano.tensor.elemwise import Elemwise

class Float2half(UnaryScalarOp):
    """
    This operation converts a float32 to a float16 which is contained in a int16 variable.
    """
    def make_node(self, x):
        x = as_scalar(x)
        y = int16()
        return Apply(self, [x], [y])
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = __float2half_rn(%(x)s);" % locals()  
        
float2half_scalar = Float2half(name='float2half')
float2half = Elemwise(float2half_scalar)

class Half2float(UnaryScalarOp):
    """
    This operation converts a float16 which is contained in a int16 variable to a float32.
    """
    def make_node(self, x):
        x = as_scalar(x)
        y = float32()
        return Apply(self, [x], [y])
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = __half2float(%(x)s);" % locals()  
        
half2float_scalar = Half2float(name='half2float')
half2float = Elemwise(half2float_scalar)