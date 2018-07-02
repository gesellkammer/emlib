"""
Utilities related to bpfs
"""
import bpf4 as bpf
from emlib.iterlib import pairwise


def zigzag(b0, b1, xs, shape='linear'):
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
