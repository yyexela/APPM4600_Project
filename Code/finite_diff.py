import numpy as np
import matplotlib.pyplot as plt

def generate_D(FD_list):
    '''
    Generates the derivative matrix `D` given a list `FD_list` that specifies the type of finite difference method for each row

    input:\n
    `FD_list`: A list of tuples of the form (FD, N), where FD is the FD option with stride N (see below) for each row

    ouptut:\n
    A matrix `D` with the values filled in from `FD_list`

    finite difference (FD) options for various strides (N): \n
    0: empty row
    1: centered difference            `f'(x) ~   (f(x-N)-f(x-N))/(2*N)`
    2: forward difference             `f'(x) ~   (f(x+N) - f(x))/N`
    3: backward difference            `f'(x) ~   (f(x) - f(x-N))/N`
    4: centered difference 2nd order  `f''(x) ~  (f(x+N)-2*f(x)+f(x-N))/N**2`
    5: forward difference 2nd order   `f''(x) ~  (f(x+2*N)-2*f(x+N)+f(x))/N**2`
    6: backward difference 2nd order  `f''(x) ~  (f(x)-2*f(x-N)+f(x-2*N))/N**2`
    '''

    # Size of our D matrix
    size = len(FD_list)
    
    # Initialize the D matrix
    D = np.zeros((size, size))

    # Populate D matrix
    for i in range(size):
        # Extract values from FD_list
        FD = FD_list[i][0]
        N = FD_list[i][1]

        # Get the row we're inserting and a helper value for padding (see `_get_FD`'s `diff`)
        row, diff = _get_FD(FD, N)

        # Calculate padding
        padding_before = i - diff
        padding_after = size - padding_before - len(row)

        # Verify the padding makes sense
        if padding_before < 0 or padding_after < 0:
            raise Exception(f"generate_D: Row {i} of the matrix D has negative padding: FD {FD}, N {N}, padding_before {padding_before}, padding_after {padding_after}")

        if padding_before + padding_after + len(row) != size:
            raise Exception(f"generate_D: Row {i} of the matrix D is not the right size: FD {FD}, N {N}, row size {len(row)}")

        # Generate padded row with zeros prepended
        row = np.pad(row, (padding_before, padding_after), constant_values=0)

        # Replace row D with our row
        D[i,:] = row
    
    return D

def _get_FD(FD, N):
    '''
    Helper function to get the smallest row-matrix for the finite difference method and calculate how many indices behind the current x we're at (used in `generate_D`) \n

    input:\n
    `FD`: Finite difference method  
    `N`: Stride length

    output:\n
    Tuple (row, diff)
    `row`: The finite difference row
    `diff`: Number of values before the current x that to start of the row

    (ie. for centered difference with an N of 2, return [-1/4, 0, 0, 0, 1/4])
    '''
    if N < 1 or not isinstance(N, int):
        raise Exception(f"get_FD: N ({N}) must be a positive integer")
    
    if FD not in [0,1,2,3,4,5,6]:
        raise Exception(f"get_FD: FD ({FD}) must be an integer in [0,6]")

    if FD == 0:
        ret = np.zeros(0)
        return (ret, 0)
    if FD == 1:
        ret = np.zeros(2*N+1)
        ret[0] = -1/(2*N)
        ret[-1] = 1/(2*N)
        return (ret, N)
    if FD == 2 or FD == 3:
        # Row is the same, but its position changes on the method
        ret = np.zeros(N+1)
        ret[0] = -1/(N)
        ret[-1] = 1/(N)

        if FD == 2:
            return (ret, 0)
        else:
            return (ret, N)
    if FD == 4 or FD == 5 or FD == 6:
        # Row is the same, but its position changes on the method
        ret = np.zeros(2*N+1)
        ret[-1] = 1/(N**2)
        ret[N] = -2/(N**2)
        ret[0] = 1/(N**2)

        if FD == 4:
            return (ret, N)
        elif FD == 5:
            return (ret, 0)
        else:
            return (ret, 2*N)

def generate_centered_D(N):
    N -= 2
    FD_list = [(0, 1)] + [(1,1)]*N + [(0, 1)]
    D = generate_D(FD_list)
    D = D[1:-1]
    return D

def generate_forward_D(N):
    N -= 1
    FD_list = [(2, 1)] * (N) + [(0, 1)]
    D = generate_D(FD_list)
    return D

def generate_backwards_D(N):
    N -= 1
    FD_list = [(0,1)] + [(3, 1)] * (N)
    D = generate_D(FD_list)
    return D

def generate_2nd_centered_D(N):
    N -= 2
    FD_list = [(0, 1)] + [(4,1)]*N + [(0, 1)]
    D = generate_D(FD_list)
    D = D[1:-1]
    return D


def test_D_1():
    '''
    Test function to make sure D is created properly
    '''
    FD_list = [(0, 1), (1, 1), (1, 1), (6, 1), (4, 1), (5, 1), (1, 1), (1, 1), (0, 1)]
    D = generate_D(FD_list)
    plt.matshow(D)
    plt.show()

def test_D_2():
    '''
    Test function to make sure D is created properly
    '''
    N = 5
    D = generate_centered_D(N)
    plt.matshow(D)
    plt.show()

#test_D_1()
#test_D_2()
