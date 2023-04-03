#!/usr/bin/env python
#--*-- coding: utf-8 -*-

import numpy as np

def get_triangular_matrix( mode='lower', ndim=None, mtrx=None ):
    """Returns a triangular matrix in-place.

    If a matrix is given, the function will modify the input, in place, into a
    triangular matrix. The mtrx object will be modified and reflected on the callee
    side. Otherwise, the function generates a random triangular matrix.

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

    >>> from chen_3170.help import get_triangular_matrix
    >>> a_mtrx = get_triangular_matrx('lower',3)
    >>> a_mtrx
    array([[0.38819556, 0.    , 0.        ],
       [0.12304746, 0.07522054, 0.        ],
       [0.96357929, 0.69187941, 0.2878785 ]])

    """

    assert ndim is None or mtrx is None, 'ndim or mtrx must be given; not both.'
    assert not (ndim is None and mtrx is None), 'either ndim or mtrx must be given.'
    assert mode =='lower' or mode =='upper', 'invalid mode %r.'%mode

    if ndim is not None:
        assert isinstance(ndim, int)

    if mtrx is None:
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
def forward_solve(l_mtrx, b_vec, loop_option='use-dot-product', zero_tol=1e-12):
    """Performs a forward solve with a lower triangular matrix and right side vector.

    Parameters
    ----------
    l_mtrx: numpy.ndarray
       Lower triangular matrix.
    b_vec:  numpy.ndarray
       Right-side vector.
    loop_option: string
       This is an internal option to demonstrate the usage of an explicit
       double loop or an implicit loop using a dot product.
    zero_tol: float
       Tolerance for non-zero values in the upper triangular portion.

    Returns
    -------
    x_vec: numpy.narray
           Solution vector returned.

    Examples
    --------

    """
    import numpy as np

    # sanity tests

    # l_mtrx must be np.ndarray
    assert isinstance(l_mtrx, np.ndarray)

    # l_mtrx must be square
    assert l_mtrx.shape[0] == l_mtrx.shape[1], 'non-square matrix.'

    assert np.all(np.abs(np.diagonal(l_mtrx)) > 0.0), 'zero value on diagonal.'

    # get i, j of all non zero entries
    rows_ids, cols_ids = np.where(np.abs(l_mtrx) > zero_tol)

    # non-zero number must be in the lower triangular portion
    assert np.all(rows_ids >= cols_ids), \
            'non-triangular matrix. l_mtrx =%r\n'%l_mtrx # test i >= j

    # b_vec must be compatible to l_mtrx
    assert b_vec.shape[0] == l_mtrx.shape[0], 'incompatible l_mtrx @ b_vec dimensions'

    assert loop_option in ('use-dot-product', 'use-double-loop')
    # end of sanity test

    m_rows = l_mtrx.shape[0]
    n_cols = m_rows
    x_vec = np.zeros(n_cols)

    if loop_option == 'use-dot-product':

        for i in range(m_rows):
            sum_lx = np.dot(l_mtrx[i, :i], x_vec[:i])
            #sum_lx = l_mtrx[i,:i] @ x_vec[:i] # matrix-vec mult. alternative to dot product
            x_vec[i] = (b_vec[i] - sum_lx) / l_mtrx[i, i]

    elif loop_option == 'use-double-loop':

        for i in range(m_rows):
            sum_lx = 0.0
            for j in range(i):
                sum_lx += l_mtrx[i, j] * x_vec[j]
            x_vec[i] = (b_vec[i] - sum_lx) / l_mtrx[i, i]

    else:
        assert False, 'not allowed option: %r'%loop_option

    return x_vec
#*********************************************************************************
def plot_matrix(mtrx, color_map='bw', title=None, style=None, yaxis=True, xaxis=True, xlabels=None, ylabels=None):
    '''
    Plot matrix as an image.

    Parameters
    ----------
    mtrx: numpy.ndarray
        Matrix data.
    color_map: str
        Color map for image: 'bw' black and white, use any other valid colormap.
    title: str, optional
        Title for plot.
    style: str
        Matplotlib style: 'dark', 'default', 'gray'
    xlabels: list(str)
        Labels for the x axis.
    ylables: list(str)
        Labels for the y axis.
    Returns
    -------
    None:

    Examples
    --------

    '''

    assert isinstance(mtrx, np.ndarray) # sanity check

    from matplotlib import pyplot as plt # import the pyplot function of the matplotlib package

    # See matplotlib options: help(plt.style.available())
    if style == 'default' or style is None:
        plt.style.use('default')
    elif style == 'dark':
        plt.style.use('dark_background')
    elif style == 'gray':
        plt.style.use('seaborn-dark')
    else:
        assert False

    plt.rcParams['figure.figsize'] = [20, 4] # extend the figure size on screen output

    plt.figure(1)
    if color_map == 'bw':
        plt.imshow(np.abs(mtrx), cmap='gray')
    else:
        plt.imshow(mtrx, cmap=color_map)
    if title is not None:
        plt.title(title,fontsize=14)
    print('matrix shape =',mtrx.shape)  # inspect the array shape

    assert yaxis in [True, False]
    if yaxis:
        if ylabels is not None:
            assert isinstance(ylabels, list)
            assert mtrx.shape[0] == len(ylabels)
            plt.yticks( range(mtrx.shape[0]), ylabels, rotation=0, fontsize=10 )
    else:
        plt.yticks([])

    assert xaxis in [True, False]
    if xaxis:
        if xlabels is not None:
            assert isinstance(xlabels, list)
            assert mtrx.shape[1] == len(xlabels)
            plt.xticks( range(mtrx.shape[1]), xlabels, rotation=60, fontsize=10 )
    else:
        plt.xticks([])


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
def print_reaction_sub_mechanisms(sub_mechanisms, mode=None, print_n_sub_mech=None):
    '''
    Nice printout of a scored reaction sub-mechanism list

    Parameters
    ----------
    sub_mechanims: list(str), required
          Sorted reaction mechanims in the form of a list.

    mode: string, optional
          Printing mode: all, top, None.

    Returns
    -------
    None:

    Examples
    --------

    '''
    # sanity check
    assert mode is None or print_n_sub_mech is None
    assert mode =='top' or mode =='all' or mode==None
    assert isinstance(print_n_sub_mech,int) or print_n_sub_mech is None
    # end of sanity check

    if mode is None and print_n_sub_mech is None:
        mode = 'all'

    if print_n_sub_mech is None:
        if mode == 'all':
            print_n_sub_mech = len(sub_mechanisms)
        elif mode == 'top':
            scores = [sm[3] for sm in sub_mechanisms]
            max_score = max(scores)
            tmp = list()
            for s in scores:
                if s == max_score:
                    tmp.append(s)
            print_n_sub_mech = len(tmp)
        else:
            assert False, 'illegal mode %r'%mode

    for rm in sub_mechanisms:
        if sub_mechanisms.index(rm) > print_n_sub_mech-1: continue
        print('Reaction Sub Mechanism: %s (score %4.2f)'%(sub_mechanisms.index(rm),rm[3]))
        for (i,r) in zip(rm[0], rm[1]):
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
def plot_arrhenius_experimental_data(temp, k_cte, style=None):

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

    # See matplotlib options: help(plt.style.available())
    if style == 'default' or style is None:
        plt.style.use('default')
    elif style == 'dark':
        plt.style.use('dark_background')
    elif style == 'gray':
        plt.style.use('seaborn-dark')
    else:
        assert False

    plt.figure(1, figsize=(12, 7))

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
def color_map(num_colors):
    """Nice colormap for plotting.

    Parameters
    ----------
    num_colors: int
        Number of colors.

    Returns
    -------
    color_map: list(tuple(R,G,B,A))
        List with colors interpolated from internal list of primary colors.
    """

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
    color_map = [red, blue, orange, magenta, green, yellow, cyan, black]

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
#*********************************************************************************
def get_covid_19_us_data( type='deaths' ):
    '''
    Load COVID-19 pandemic cumulative data from:

     https://github.com/CSSEGISandData/COVID-19.

    Parameters
    ----------
    type:  str, optional
            Type of data. Deaths ('deaths') and confirmed cases ('confirmed').
            Default: 'deaths'.

    Returns
    -------
    data: tuple(int, list(str), list(int))
           (population, dates, cases)

    '''

    import pandas as pd

    if type == 'deaths':
        #df.to_html('covid_19_deaths.html')
        df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

    elif type == 'confirmed':
        df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
        df_pop = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
        #df.to_html('covid_19_deaths.html')
        #df.to_html('covid_19_confirmed.html')

    else:
        assert True, 'invalid query type: %r (valid: "deaths", "confirmed"'%(type)

    df = df.drop(['UID','iso2','iso3','Combined_Key','code3','FIPS','Lat', 'Long_','Country_Region'],axis=1)
    df = df.rename(columns={'Province_State':'state/province','Admin2':'city'})

    import numpy as np

    state_names = list()

    state_names_tmp = list()

    for (i,istate) in enumerate(df['state/province']):
        if istate.strip() == 'Wyoming' and df.loc[i,'city']=='Weston':
            break
        state_names_tmp.append(istate)

    state_names_set = set(state_names_tmp)

    state_names = list(state_names_set)
    state_names = sorted(state_names)

    dates = np.array(list(df.columns[3:]))

    population = [0]*len(state_names)
    cases = np.zeros( (len(df.columns[3:]),len(state_names)), dtype=np.float64)

    for (i,istate) in enumerate(df['state/province']):
        if istate.strip() == 'Wyoming' and df.loc[i,'city']=='Weston':
            break

        state_id = state_names.index(istate)
        if type == 'confirmed':
            population[state_id] += int(df_pop.loc[i,'Population'])
        else:
            population[state_id] += int(df.loc[i,'Population'])

        cases[:,state_id] += np.array(list(df.loc[i, df.columns[3:]]))

    return ( state_names, population, dates, cases )
#*********************************************************************************
def get_covid_19_global_data( type='deaths', distribution=True, cumulative=False ):
    '''
    Load COVID-19 pandemic cumulative data from:

        https://github.com/CSSEGISandData/COVID-19

    Parameters
    ----------
    type: str, optional
        Type of data. Deaths ('deaths') and confirmed cases ('confirmed').
        Default: 'deaths'.

    distribution: bool, optional
        Distribution of new cases over dates.
        Default: True

    cumulative: bool, optional
        Cumulative number of cases over dates.
        Default: False

    Returns
    -------
    data: tuple(int, list(str), list(int))
           (contry_names, dates, cases)

    '''

    if cumulative is True:
        distribution = False

    import pandas as pd

    if type == 'deaths':
        df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
        #df.to_html('covid_19_global_deaths.html')

    else:
        assert True, 'invalid query type: %r (valid: "deaths"'%(type)

    df = df.drop(['Lat', 'Long'],axis=1)
    df = df.rename(columns={'Province/State':'state/province','Country/Region':'country/region'})

    import numpy as np

    country_names = list()

    country_names_tmp = list()

    for (i,icountry) in enumerate(df['country/region']):
        country_names_tmp.append(icountry)

    country_names_set = set(country_names_tmp)

    country_names = list(country_names_set)
    country_names = sorted(country_names)

    dates = np.array(list(df.columns[2:]))

    cases = np.zeros( (len(df.columns[2:]),len(country_names)), dtype=np.float64)

    for (i,icountry) in enumerate(df['country/region']):

        country_id = country_names.index(icountry)

        cases[:,country_id] += np.array(list(df.loc[i, df.columns[2:]]))

    if distribution:

        for j in range(cases.shape[1]):
            cases[:,j] = np.round(np.gradient( cases[:,j] ),0)

    return ( country_names, dates, cases )
#*********************************************************************************
