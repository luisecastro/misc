import numpy as np
import scipy.integrate as integrate


# Helper functions used for further calculations
def factorial(x: int) -> int:
    """
    Returns the factorial of a number.
    :param x: number
    :return: factorial of x
    """
    result = 1
    for i in range(1, x+1):
        result *= i
    return result    


def e(steps: int=9) -> float:
    """
    Calculate de value of e with a particular precision
    :param steps: value of n in e formula (steps=9 good for 6 decimals)
    :return: value of e
    """
    result = 0
    for i in range(0, steps+1):
        result += 1 / factorial(i)
    return result


def pi(limit: int=20) -> float:
    """
    Calculate the value of pi to a particular precision
    :param limit: number of iterations to run, more better precision, 20 seems good.
    :return: value of pi
    """
    def helper(step, limit):
        if step >= 2*limit:
            return 0
        return (step ** 2) / (6 + helper(step + 2, limit))
    return 3 + helper(1, limit)


def standard_deviation(x: list, ddof=1) -> float:
    """
    Calculate the standard deviation of a group of numbers
    :param x: group of numbers for calculation
    :param ddof: degrees of freedom
    :return: standard deviation
    """
    result = 0
    n = len(x)
    mean = sum(x) / n
    
    for val in x:
        result += (val-mean)**2
        
    return (result / (n-ddof))**0.5


#Calculation of binomial distribution values: individual and cummulative
def binomial(n: int, k: int, p: float) -> float:
    """
    Returns the evaluation of the Binomial Distribution 
    for the given parameters.
    :param n: number of trials
    :param k: number of successes
    :param p: probability of success
    :return: Binomial evaluation of parameters
    """
    return (factorial(n) / (factorial(k) * factorial(n - k))) * (p**k) * (1-p) ** (n-k)


def binomial_range(n: int, k_lo: int, k_hi: int, p: float) -> (float, list):
    """
    Returns the sum of probabilities, individual evaluations and
    of the cummulative Binomial Distribution for a range of successes.
    :param n: number of trials
    :param k_lo: min number of successes
    :param k_hi: max number of successes
    :param p: probability of success
    :return: sum and individual and cummulative binomial values
    """
    result = []
    accum = []
    
    for i in range(k_lo, k_hi + 1):
        temp = binomial(n, i, p)
        result.append(temp)
        accum.append(sum(result))
        
    return sum(result), result, accum


#Poisson distribution calculation and cumulative values over a provided range.
def poisson(lambd: float, x: int) -> float:
    """
    Evaluation of the Poisson formula for lambda and x
    :param lambd: mean number of successes
    :param x: specific successes to be seen
    :return: value of evaluation
    """
    return ((e()**(-lambd))*(lambd**x))/factorial(x)


def poisson_cdf(lambd: float, x_hi:int) -> list:
    """
    Cummulative evaluation of poisson from 0 to x_hi
    :param lambd: mean number of successes
    :param x_hi: upper limit
    :return: cummulative evaluation of Poisson formula
    """ 
    result = 0
    
    for i in range(x_hi+1):
        result += poisson(lambd, i)

    return result

#Calculation of the probability and the cummulative distribution for the normal
#distribution, to calculate the cummulative value we need to integrate done with
#scipy integration.
def norm_pdf(x: float, mean: float, std: float) -> float:
    """
    Calculate probability from a normal distribution parameters
    :param x: value of the variable
    :param mean: mean of the values
    :param std: standard deviation of the values
    :return: probability value
    """
    return (1/(std*(2*pi())**0.5))*e()**((-1/2)*((x-mean)/std)**2)
    
    
def norm_cdf(mean: float, std: float,  x_lo: float = -np.inf, x_hi: float = np.inf) -> float:
    """
    Calculate cummulative probability for a range in a normal distribution
    :param mean: mean of the values
    :param std: standard deviation of the values
    :x_lo: starting point for the calculation
    :x_hi: ending point for the calculation
    :return: cummulative probability for a the Gaussian distribution
    """
    return integrate.quad(lambda x: norm_pdf(x, mean, std), x_lo, x_hi)[0]
    
    
def z_value(x: float, mean: float, std: float) -> float:
    """
    Calculation of the Z value, # of std deviations for a value
    away from the mean
    :param x: value to calculate the Z value
    :param mean: mean of the values
    :param std: standard deviation of the values
    """
    return (x-mean) / std
