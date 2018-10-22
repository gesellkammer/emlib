"""
Utilities related to bpfs
"""
import bpf4 as bpf
from bpf4 import BpfInterface as _Bpf
from emlib.iterlib import pairwise
from . import typehints as t


def zigzag(b0, b1, xs, shape='linear'):
    # type: (_Bpf, _Bpf, t.Seq, str) -> _Bpf 
    """
    Crea una seq. de curvas que van de b0(x) a b1(x) por
    cada x en xs             
                                                                      
     *.                                                                   
      *...  b0                                                              
       *  ...                                                             
       *     ...                                                          
        *       ....                                                      
         *          ...                                                   
          *         :  ...                                                
           *        :*    ...                                             
           *        : *      ...                                          
            *       :  **       ...                                       
             *      :    *         :*.                                    
              *     :     *        : **...                                
               *    :      *       :   *  ...                             
               *    :       *      :    *    ...                          
                *   :        *     :     **     .:.                       
                 *  :         *    :       *     :**..                    
                  * :          **  :        **   :  ****.                 
                   *:            * :          *  :      ****              
      -----------  *:             *:           * :          ****          
        b1       ---*--------------*---         **:             ****      
                                       -----------*----------      .**    
                                                             ----------- 
      x0            x1              x2                       x3

     objective: a point (x, y). b0 and b1 should 
    """
    # dibujo creado con asciipaint.com
    curves = []
    for x0, x1 in pairwise(xs):
        X = [x0, x1]
        Y = [b0(x0), b1(x1)]
        curve = bpf.util.makebpf(shape, X, Y)
        curves.append(curve)
    jointcurve = bpf.max_(*[c.outbound(0, 0) for c in curves])
    return jointcurve


def bpfavg(b, dx):
    # type: (_Bpf, float) -> _Bpf
    """
    Return a Bpf which is the average of b over the range `dx`
    """
    dx2 = dx/2
    avg = ((b<<dx2)+b+(b>>dx2))/3.0
    return avg[b.x0:b.x1]