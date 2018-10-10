#!/usr/bin/env python
#--*-- coding: utf-8 -*-
# This file is part of the ChEn-3170 Computational Methods in Chemical Engineering
# course at https://github.com/dpploy/chen-3170
def get_triangular_matrix( mode='lower', ndim=None, mtrx=None ):
    '''
    Returns a triangular matrix in-place.

    If a matrix is given, the function will modify the input, in place, into a 
    triangular matrix. The mtrx object will be modified and reflected on the callee side.
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
    assert not (ndim is None and mtrx is None), 'either ndim or mtrx must be given.'
    assert mode =='lower' or mode =='upper', 'invalid mode %r.'%mode
    
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
#*********************************************************************************
def forward_solve(l_mtrx, b_vec, loop_option='use-dot-product'):
    '''
    Performs a forward solve with a lower triangular matrix and right side vector.
    
    Parameters
    ----------
    l_mtrx: numpy.ndarray, required
            Lower triangular matrix.
    b_vec:  numpy.ndarray, required
            Right-side vector.
    loop_option: string, optional
            This is an internal option to demonstrate the usage of an explicit 
            double loop or an implicit loop using a dot product. 
            Default: 'use-dot-product'
            
    Returns
    -------
    x_vec: numpy.narray
           Solution vector returned.
           
    Examples
    --------
    
    '''        
    import numpy as np
    
    # sanity test
    assert isinstance(l_mtrx,np.ndarray)      # l_mtrx must be np.ndarray
    assert l_mtrx.shape[0] == l_mtrx.shape[1],'non-square matrix.' # l_mtrx must be square
    assert np.all(np.abs(np.diagonal(l_mtrx)) > 0.0),'zero value on diagonal.'
    rows_ids, cols_ids = np.where(np.abs(l_mtrx) > 0) # get i, j of non zero entries
    assert np.all(rows_ids >= cols_ids),'non-triangular matrix.' # test i >= j
    assert b_vec.shape[0] == l_mtrx.shape[0],'incompatible l_mtrx @ b_vec dimensions'  # b_vec must be compatible to l_mtrx
    assert loop_option == 'use-dot-product' or loop_option == 'use-double-loop'
    # end of sanity test
    
    m_rows = l_mtrx.shape[0]
    n_cols = m_rows
    x_vec = np.zeros(n_cols)
    
    if loop_option == 'use-dot-product':
        
        for i in range(m_rows):
            sum_lx = np.dot( l_mtrx[i,:i], x_vec[:i] )
            #sum_lx = l_mtrx[i,:i] @ x_vec[:i] # matrix-vec mult. alternative to dot product
            x_vec[i] = b_vec[i] - sum_lx
            x_vec[i] /= l_mtrx[i,i]
            
    elif loop_option == 'use-double-loop':
             
        for i in range(m_rows):
            sum_lx = 0.0
            for j in range(i):
                sum_lx += l_mtrx[i,j] * x_vec[j]
            x_vec[i] = b_vec[i] - sum_lx
            x_vec[i] /= l_mtrx[i,i]
               
    else:
        assert False, 'not allowed option: %r'%loop_option
        
    return x_vec  
#*********************************************************************************
