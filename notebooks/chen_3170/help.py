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
          square matrix to be turned into a triangular matrix.
    
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
def plot_matrix(mtrx, color_map='bw', title=None):
    '''
    Plot matrix as an image.
    
    Parameters
    ----------
    mtrx: numpy.ndarray, required
          Matrix data. 
    color_map: str, optional 
               Color map for image: 'bw' black and white
    title: str, optional 
           Title for plot.     
            
    Returns
    -------
    None: 
           
    Examples
    --------

    '''

    # sanity check
    import numpy as np
    assert isinstance(mtrx,np.ndarray)
    import numpy as np
    from matplotlib import pyplot as plt # import the pyplot function of the matplotlib package
    # end of sanity check

    plt.rcParams['figure.figsize'] = [20, 4] # extend the figure size on screen output

    plt.figure(1)
    if color_map == 'bw':
        plt.imshow(np.abs(mtrx),cmap='gray')
    else:
        plt.imshow(mtrx,cmap=color_map)
    if title is not None: 
        plt.title(title,fontsize=14)
    print('matrix shape =',mtrx.shape)  # inspect the array shape

    plt.show()

    return
#*********************************************************************************
def print_reactions(reactions):
    '''
    Nice printout of a reactions list.
    
    Parameters
    ----------
    reactions: list(str)
          Reactions in the form of a list.
            
    Returns
    -------
    None: 
           
    Examples
    --------

    '''
    # sanity check
    assert isinstance(reactions,list)
    # end of sanity check

    for r in reactions: 
        i = reactions.index(r)
        print('r%s'%i,': ',r)
    
    n_reactions = len(reactions)  
    print('n_reactions =',n_reactions)

    return
#*********************************************************************************
def print_reaction_mechanisms( mechanisms, mode=None, print_n_mechanisms=None ):
    '''
    Nice printout of a scored reaction mechanims list
    
    Parameters
    ----------
    mechanims: list(str), required
          Sorted reaction mechanims in the form of a list.

    mode: string, optional
          Printing mode: all, top, None. Default: all
            
    Returns
    -------
    None: 
           
    Examples
    --------

    '''
    # sanity check
    assert mode is None or print_n_mechanisms is None
    assert mode =='top' or mode =='all' or mode==None
    assert isinstance(print_n_mechanisms,int) or print_n_mechanisms is None
    # end of sanity check

    if mode is None and print_n_mechanisms is None:
        mode = 'all'

    if print_n_mechanisms is None:
        if mode == 'all':
            print_n_mechanisms = len(mechanisms)
        elif mode == 'top':
            scores = [sm[3] for sm in mechanisms]
            max_score = max(scores)
            tmp = list()
            for s in scores:
                if s == max_score:
                    tmp.append(s)
            print_n_mechanisms = len(tmp)
        else:
            assert False, 'illegal mode %r'%mode

    for rm in mechanisms:
        if mechanisms.index(rm) > print_n_mechanisms-1: continue
        print('Reaction Mechanism: %s (score %4.2f)'%(mechanisms.index(rm),rm[3]))
        for (i,r) in zip( rm[0], rm[1] ):
            print('r%s'%i,r)

    return
#*********************************************************************************
def read_arrhenius_experimental_data(filename):
    '''
    Read k versus T data for fitting an Arrhenius rate constant expression.
    
    Parameters
    ----------
    filename: string, required
            File name of data file including the path.
            
    Returns
    -------
    r_cte: float
        Universal gas constant.
    r_cte: string
        Universal gas constant unit.
    n_pts: int
        Number of data points
    temp: np.ndarray, float
        Temperature data.
    k_cte: np.ndarray, float
        Reaction rate constant data.
           
    Examples
    --------
    
    '''        

    import io                     # import io module
    finput = open(filename, 'rt') # create file object

    import numpy as np

    for line in finput:
    
        line = line.strip()
    
        if line[0] == '#': # skip comments in the file
            continue
        
        var_line = line.split(' = ')
    
        if var_line[0] == 'r_cte':
            r_cte = float(var_line[1].split(' ')[0])
            r_cte_units = var_line[1].split(' ')[1]
        elif var_line[0] == 'n_pts':
            n_pts = int(var_line[1])
            temp  = np.zeros(n_pts)
            k_cte = np.zeros(n_pts)
            idx   = 0 # counter
        else:
            data = line.split(' ')
            temp[idx]  = float(data[0])
            k_cte[idx] = float(data[1])
            idx += 1
            
    return (r_cte, r_cte_units, n_pts, temp, k_cte)

#*********************************************************************************
def plot_arrhenius_experimental_data( temp, k_cte ):
    
    '''
    Plot T versus k data for fitting an Arrhenius rate constant expression.
    
    Parameters
    ----------
    temp: nd.array, required
        Temperature data. 
    k_cte: nd.array, required
        Reaction rate constant data.
            
    Returns
    -------
    None: None
           
    Examples
    --------
    
    '''   

    import matplotlib.pyplot as plt

    plt.figure(1, figsize=(7, 7))

    plt.plot(temp, k_cte,'r*',label='experimental')
    plt.xlabel(r'$T$ [K]',fontsize=14)
    plt.ylabel(r'$k$ [s$^{-1}$]',fontsize=14)
    plt.title('Arrhenius Rxn Rate Constant Data',fontsize=20)
    plt.legend(loc='best',fontsize=12)
    plt.grid(True)
    plt.show()
    print('')

    return

#*********************************************************************************
def color_map( num_colors ):

    assert num_colors >= 1

    import numpy as np

    # primary colors
    # use the RGBA decimal code
    red     = np.array((1,0,0,1))
    blue    = np.array((0,0,1,1))
    magenta = np.array((1,0,1,1))
    green   = np.array((0,1,0,1))
    orange  = np.array((1,0.5,0,1))
    black   = np.array((0,0,0,1))
    yellow  = np.array((1,1,0,1))
    cyan    = np.array((0,1,1,1))

    # order the primary colors here
    color_map = list()
    color_map = [ red, blue, orange, magenta, green, yellow, black, cyan]

    num_primary_colors = len(color_map)

    if num_colors <= num_primary_colors:
        return color_map[:num_colors]

    # interpolate primary colors
    while len(color_map) < num_colors:
        j = 0
        for i in range(len(color_map)-1):
            color_a = color_map[2*i]
            color_b = color_map[2*i+1]
            mid_color = (color_a+color_b)/2.0
            j = 2*i+1
            color_map.insert(j,mid_color) # insert before index
            if len(color_map) == num_colors:
                break

    return color_map
