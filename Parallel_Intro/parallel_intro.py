# iPyParallel - Intro to Parallel Programming
from ipyparallel import Client
import numpy as np
import time
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as sparse on
    all engines. Return the DirectView.
    """
    client = Client()   # initialize Client object
    dview = client[:]   # create DirectView w/ all available engines
    dview.execute("import scipy.sparse as sparse")     # import scipy.sparse as sparse on all engines
    client.close()

    return dview

# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    # initialize Client object & create DirectView
    client = Client()
    dview = client[:]
    dview.block = True
    
    # distribute the variables
    dview.push(dx)

    # pull the variables back & check
    for key in dx.keys():
        if dx[key] != dview.pull(key)[0]:
            raise ValueError('uh oh')
    
    client.close()
    
def test2():
    dx = {'a':10, 'b':5}
    variables(dx)

# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    # initialize Client object & create DirectView
    client = Client()
    dview = client[:]
    dview.block = True

    # construct function to make draws
    f = lambda n: np.random.normal(size=n)
    
    # evaluate
    dview.execute("import numpy as np")
    results = dview.apply_async(f, n)

    # calculate mean min and max
    means = [np.mean(r) for r in results]
    mins  = [np.min(r)  for r in results]
    maxs  = [np.max(r)  for r in results]

    client.close()

    return means, mins, maxs

def test3():
    means, mins, maxs = prob3()
    print(f'means\n{means}\nmins\n{mins}\nmaxs\n{maxs}')

# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    # intialize
    n_vals = [1000000, 5000000, 10000000, 15000000]
    parallel_times, serial_times = [], []
    # initialize Client object & create DirectView
    client = Client()

    for n in n_vals:
        # parallel
        start = time.time()
        prob3(n)
        parallel_times.append(time.time() - start)

        # serial
        start = time.time()
        for i in range(len(client.ids)):
            draws = np.random.normal(size=n)
            np.mean(draws)
            np.min(draws)
            np.max(draws)
        serial_times.append(time.time() - start)
    
    client.close()

    # plot
    plt.plot(n_vals, parallel_times, label="Parallel")
    plt.plot(n_vals, serial_times, label="Serial")
    plt.legend()
    plt.title("Execution Times")
    plt.xlabel('n')
    plt.ylabel('Time')
    plt.tight_layout()
    plt.show()

# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    # construct x & h
    x, h = np.linspace(a,b,n), (b-a)/(n-1)

    # initialize Client object & create DirectView
    client = Client()
    dview = client[:]
    dview.block = True

    # construct intervals to distribute
    intervals = [(a, b) for a, b in zip(x[:-1], x[1:])]
    dview.scatter('intervals', intervals)

    # define trapezoid function
    def trapezoid(x):
        return f(x[0]) + f(x[1])
    
    results = dview.map(trapezoid, intervals)
    solution = np.sum(results)*(h/2)
    
    client.close()
    
    return solution

def test5():
    f, a, b = lambda x: x, 0, 1
    print(parallel_trapezoidal_rule(f,a,b,n=1000))
