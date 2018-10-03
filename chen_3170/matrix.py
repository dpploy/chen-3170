#!/usr/bin/env python
#--*-- coding: utf-8 -*-
# This file is part of the ChEn-3170 Computational Methods in Chemical Engineering
# course at https://github.com/dpploy/chen-3170
def get_triangular_matrix( mode='lower', ndim=None, mtrx=None ):
    '''
    Returns a triangular matrix.

    If a matrix is given, it transforms the input into a triangular matrix.
    Otherwise, the function generates a random triangular matrix.

    Parameters
    ----------
    mode: string, optional
          Type of triangular matrix: 'lower' or 'upper'. Defaults to lower 
          triangular.
    ndim: int, optional
          Dimension of the square matrix. If a matrix is not provided this 
          argument is required. 
    mtrx: numpy.ndarray, optional
          square matrix to be turned into a triangular matrix
    
    Returns
    -------
    mtrx: numpy.ndarray
          If a matrix was not passed the return is random array. If a matrix
          was passed, its view is modified.

    Examples
    --------

    >>> a_mtrx = ce.get_triangular_matrx('lower',3)
    >>> a_mtrx
    array([[0.38819556, 0.    , 0.        ],
       [0.12304746, 0.07522054, 0.        ],
       [0.96357929, 0.69187941, 0.2878785 ]])

    '''

    assert ndim is None or mtrx is None, 'ndim or mtrx must be given; not both.'
    assert mode =='lower' or mode =='upper', 'invalid mode %r.'%mode
    if ndim is None and mtrx is None:
       assert False,'ndim or mtrx must be given.'
    
    if mtrx is None:
        import numpy as np
        mtrx = np.random.random((ndim,ndim))
    else:
        assert mtrx.shape[0] == mtrx.shape[1], 'matrix not square.' 
    
    # ready to return matrix  
    if mode == 'lower':
        for i in range(mtrx.shape[0]):
            mtrx[i,i+1:] = 0.0
    elif mode == 'upper':
        for j in range(mtrx.shape[1]):
            mtrx[j+1:,j] = 0.0
    else:
        assert False, 'oops. something is very wrong.'

    return mtrx     
