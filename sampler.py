"""
This is a subset sampler I built to sample subsets of network features from In-Q-Tel's NetworkML dataset.

NetworkML is an open source network role classifier that parses .pcap files and trains supervised learning models to indentify
network traffic consistent with certain devices. The orginal dataset contained 221 features per device sample. My job at In-Q-Tel
was to interpret the NetworkML's models, but this was challenging to do with such a high dimensional dataset.
Using another custom algorithm I built, I was able to group features into bins based off multicollinearity, knowing that the models only required
a subset of features from each bin to remain accurate.

An example of subset sampling from bins of features:

Given feature bins:

BIN_1: [max_frame_len, min_frame_len, 25q_frame_len]

BIN_2: [ipv4_multicast, upd_port_67_in, upd_port_67_out]

valid subset samples of these bins might be:

[max_frame_len] U [udp_port_67_in, udp_port_67_out] -> [max_frame_len, udp_port_67_in, udp_port_67_out]

or

[min_frame_len, 25q_frame_len] U [ipv4_multicast, upd_port_67_out] -> [min_frame_len, 25q_frame_len, ipv4_multicast, upd_port_67_out]

I built an earlier version of this subset sampler to brute force a sampling of a few features from each bin and fit a model to see if it was still accurate.
I ran into many issues with memory efficency because my first implementaion relied on itertools.product(). The itertool.product() docs claim to be memory efficient, but reading the
source code reveals that the lower order products are cached to generate the higher order products later. As a result I struggled to use this method with the large bins of features
because itertools.product() would eat up my memory. On the job I managed to make the sampler more memory efficeint so I could sample from larger bins of features.
On my own time, I rebuilt this super memory efficient sampler that completly ommits the use of itertools.product() and greatly reduces the amount of caching needed.
More information on itertools can be found at https://docs.python.org/3/library/itertools.html.
"""


import itertools
import numpy as np


def cycling_powerset(s):
    """
    Creates an infintly cycling iterator over the powerset of s, starting with the largest combinations
    
    Args:
    
    s -- List, the set to compute the powerset of
    
    Returns:
    
    powerset -- Iterator, a cycling iterator aver the powerset of s
    """
    return itertools.cycle(itertools.chain.from_iterable(
            [itertools.combinations(s, r) for r in range(len(s), 0, -1)]))


def card_powerset(s):
    """
    Returns the cardinality of the powerset of s. This is 2^|s|
    
    Args:
    
    s -- List, the set to compute the powerset cardinality of
    
    Returns:
    
    length -- int, The cardinality of the powerset of s
    """
    return np.power(2,len(s)) - 1


def dec_count(digits_max):
    """
    Generates a List that numerically decreases the numbers in digits_max down to [1] * len(digits_max).
    For example, if digits_max = [3, 4, 2], then the methods generates [3,4,2], [3,4,1], [3,3,2], [3,3,1],
    [3,2,2], [3,2,1], ... [1,1,2], [1,1,1].

    This method is logically equivalent to itertools.product([np.arange(1, l+1) for l in digits_max]), but is
    more memory efficeint. itertools.product() will store the running setwise product (see https://docs.python.org/3/library/itertools.html),
    while this method generates each result in place in memory.

    Args:

    digits_max -- List, the maximum value of each digit. Also the first result returned

    Yields:

    result -- List, the i-th decremented value from digits_max

    Raises:

    ValueError -- if any element of digits_max is < 1
    """
    # Exception handling
    if np.any(np.array(digits_max) < 1):
        raise ValueError("All digits must have a max value >= 1")

    n_digits = len(digits_max)

    # Start by yielding the value of max_digits
    result = digits_max.copy()
    yield result

    digit_idx = n_digits - 1  # The digit index pointer. Stores the index of the current index being examined

    """
    Starting at the -1 indexed digit, examine digits to see if they have been iterated down to 1.
    If so, reset that digit to its max value and check the next digit. Otherwise, decrement this digit.
    """
    while result != [1] * n_digits:  # The stopping case is a result of all 1s
        while result[digit_idx] == 1:  # Check if a digit is 1
            result[digit_idx] = digits_max[digit_idx]  # If so, reset this digit to its max value
            digit_idx -= 1  # Go check the next digit

        # Found a digit > 1
        result[digit_idx] -= 1  # Decrement this digit
        digit_idx = n_digits - 1  # Reset the digit index pointer

        yield result
    

def subset_sampler(set_list):
    """
    A generator method that samples subsets from each set in set_list. The iterator unions all samples before returning. For example,
    if set_list = [S0, S1, S2], where Si is a set in set_list. The generator gets a subset s0 of S0, a subset s1 of S1, and a subset s2 of S2
    and returns (s1 U s2 U s3). The generator will sample subsets for each set in decreasing order of size. There are several optimizations
    to make this method memory efficient.

    Args:
    
    set_list-- List, A List of sets, whose subset unions are iterated over
    
    Yields:
    
    sample -- List, a union of subsets from each set in set_list. The largest subsets are returned first

    Raises:

    ValueError -- if set_list contains empty sets
    """

    # Exception Handling 
    for s in set_list:
        if len(s) == 0:
            raise ValueError("All sets in set_list must have at least 1 element")

    n_sets= len(set_list)

    # Build a list of cycling iterators over the powerset of each set in set_list
    powerset_iters = [cycling_powerset(s) for s in set_list]

    # Build a list containing the cardinality of each powerset
    powerset_cards = [card_powerset(s) for s in set_list]

    """
    A typical pythonic way to implement this method would be with itertools.product(), which stores the entire running product in memory (see https://docs.python.org/3/library/itertools.html).
    Depending on the size and data types of each set in set_list, this can consume a lot of memory. To mitigate this we use dec_count()
    to generate the index number of powerset element we desire from each set per iteration. See the method docs for dec_count() to understand why this method is also memory efficient.
    If set_list contained 3 sets of powerset lengths 8, 4, and 16 respectivly, we would use dec_count to generate the set: ([7,3,15], [7,3,14], [7,3,13], [7,3,12], [7,3,11], ..., [1,1,3], [1,1,2], [1,1,1]).
    [7,3,15] represents the 7th, 3rd, and 15th elements of each powerset in powerset_iters respectively. 
    
    Note we always want at least 1 element from each set (we omit the empty set in these powersets).
    """
    # Build an iterator over the powerset cardinality for each set in decreasing order
    powerset_idxs_iter = dec_count(powerset_cards)  # very memory efficient compared to itertools.product()

    """
    To conserve memory, we cache the previously generated subset of each set. For example, if we generate subsets based on powerset elements [3, 4, 2], the next
    iteration will generate subsets based on the indicies [3, 4, 1]. We do not need to store an addition copy of the last subset of size 3 and size 4 for the first
    and second sets, we already have done so in the previous iteration. Only a third set needs a new powerset generated. itertools.product() would normally, redudantly
    regenerate all copies.
    """
    # Allocate space to cache previous subsets
    prev_powerset_idxs = np.zeros(n_sets)  # The index being cached. Used to determine if a cached subset should be used
    prev_powerset_elem = [None] * n_sets  # The subset at the above index

    # Get a subset from each set and perform the union
    for ps_idx in powerset_idxs_iter:  # for example, ps_idx might be [7, 3, 14]
        result = ()
        for i in range(0, n_sets):  
            if ps_idx[i] == prev_powerset_idxs[i]:  # Note that len(ps_idx) == len(prev_powerset_idxs) == n_sets
                # The powerset index for this set has NOT changed. Use the cached powerset element
                result = np.union1d(result, prev_powerset_elem[i])
            else:
                # The powerset index has changed. Get the next powerset element
                nxt = next(powerset_iters[i])

                # Union with the running result
                result = np.union1d(result, nxt)

                # Update the cached index and powerset element
                prev_powerset_idxs[i] = ps_idx[i]
                prev_powerset_elem[i] = nxt

        yield result


def test_subset_sampler(set_list, test_name=None):
    """
    A simple test that ensures the correct number of samples were generated from subset_sampler().

    Args:

    set_list -- List, the set_list to test for subset_sampler()

    Keyword Args:

    test_name -- str, if a string is passed, the method outputs a success message if the test passes that includes test_name (Default=None)
    """
    # count the number of results returned from subset_sampler()
    result_count = 0
    for res in subset_sampler(set_list):
        result_count += 1
    
    # count the number of reults that should have been returned
    true_count = 1
    for s in set_list:
        true_count *= card_powerset(s)

    # Ensure the result counts from above match
    try:
        assert true_count == result_count, "true count: {} != result count: {}".format(true_count, result_count)

        # Prints a success message if the test passes and test_name is specified
        if test_name is not None:
            print("PASSED Test: {}".format(test_name))

    except AssertionError as e:
        print(e)


def main():
    # Define test set_lists
    normal_1 = [[1,2,3], [1,2], [4,6,8]]
    normal_2 = [['A','B'], [1,2,3], ['c'], ['!','@']]
    small_1 = [[0], ['$','%']]
    small_2 = [['apple']]
    mixed_1 = [[1,'A',12.2], [6,'a','orange'], [123,'peach',-1.6]]
    mixed_2 = [[4.45, -1,'watermelon'], [5,':)','A'], [100,2,'peach']]

    # Test each of the above set_lists
    test_subset_sampler(normal_1, 'normal_1')
    test_subset_sampler(normal_2, 'normal_2')
    test_subset_sampler(small_1, 'small_1')
    test_subset_sampler(small_2, 'small_2')
    test_subset_sampler(mixed_1, 'mixed_1')
    test_subset_sampler(mixed_2, 'mixed_2')


if __name__ == '__main__':
    main()