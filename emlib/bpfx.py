"""
Utilities related to bpfs
"""
from __future__ import annotations
import bpf4 as bpf
from bpf4 import BpfInterface
from emlib.iterlib import pairwise
from typing import Sequence as Seq

def zigzag(b0: BpfInterface, b1: BpfInterface, xs: Seq[float], shape='linear'
           ) -> BpfInterface:
    """
    Creates a curve formed of lines from b0(x) to b1(x) for each x in xs
    
    Args:
        b0: a bpf
        b1: a bpf
        xs: a seq. of x values to evaluate b0 and b1
        shape: the shape of each segment

    Returns:
        The resulting bpf

    :: 

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

    """
    curves = []
    for x0, x1 in pairwise(xs):
        X = [x0, x1]
        Y = [b0(x0), b1(x1)]
        curve = bpf.util.makebpf(shape, X, Y)
        curves.append(curve)
    jointcurve = bpf.max_(*[c.outbound(0, 0) for c in curves])
    return jointcurve


def bpfavg(b: BpfInterface, dx: float) -> BpfInterface:
    """
    Return a Bpf which is the average of b over the range `dx`
    """
    dx2 = dx/2
    avg = ((b<<dx2)+b+(b>>dx2))/3.0
    return avg[b.x0:b.x1]