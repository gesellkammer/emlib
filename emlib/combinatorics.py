"""
Functions for combinatorics (combinations with repetitions, derangements, etc.)
"""
from __future__ import annotations
from operator import mul
import random
from . import iterlib as _iterlib
import numpy as _np
from . import misc
import itertools
from functools import reduce
import time as _time
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeVar, TypeAlias, Union, Iterator, Sequence
    seq_t: TypeAlias = Union[list, tuple, _np.ndarray]
    T = TypeVar("T")


def combinations_with_repetition(seq: Iterator[T], size: int) -> Iterator[tuple[T, ...]]:
    """
    Yield all combinations of the elements of seq

    * items can be repeated: (0, 1, 1) is a valid answer
    * position is relevant: (0, 1, 2) is not the same as (2, 1, 0)

    .. note:: this is in fact a product of the sequence with itself (see ``itertools.product``)

    If position is not relevant, use ``itertools.combinations_with_replacement``
    """
    for group in itertools.product(seq, repeat=size):
        yield group


def derangements(seq: Sequence[T]) -> Iterator[list[T]]:
    """Permutations of seq where each element is not in its original position"""
    queue = [-1]
    lenlst = len(seq)
    while queue:
        i = queue[-1] + 1
        if i == lenlst:
            queue.pop()
        elif i not in queue and i != len(queue) - 1:
            queue[-1] = i
            if len(queue) == lenlst:
                yield [seq[x] for x in queue]
            queue.append(-1)
        else:
            queue[-1] = i


def _distance_from_original(seqa: Sequence[T], seqb: Sequence[T]) -> float:
    distance = 0
    for ia, a in enumerate(seqa):
        ib = seqb.index(a)
        distance += abs(ib - ia)
    return distance / len(seqa)


def _max_distance(xs: Sequence) -> float:
    return _distance_from_original(xs, list(reversed(xs)))


def random_range(length: int) -> _np.ndarray:
    """
    Return an array of ints from 0 to length-1, in random order

    Args:
        length: the length of the generated sequence

    Returns:
        the generated array
    """
    s = _np.arange(length)
    _np.random.shuffle(s)
    return s


def distance_from_sorted(seq: seq_t, offset=0) -> int:
    """
    Distance between seq and a sorted seq. of the same length
    """
    if isinstance(seq, (list, tuple)):
        dist = sum(abs(x - i) for i, x in enumerate(seq, start=offset))
    elif isinstance(seq, _np.ndarray):
        indices = _np.arange(offset, offset+len(seq))
        arr = _np.asarray(seq)
        dist = _np.abs(arr - indices).sum()
    else:
        raise TypeError(f"Expected a seq or a np.ndarray, got {type(seq)}")
    return dist


_cached_random_distances = [0,   0,  1,  2,  4,  8, 11, 15, 21, 26,
                            33, 39, 47, 56, 64, 74, 84, 96, 107, 119,
                            132, 146, 160, 175, 191, 208, 225, 242,
                            260, 280, 300, 320, 340, 363, 385, 407, 431, 456,
                            480, 506, 532, 560, 587, 615, 645, 674, 705, 734,
                            769, 799, 834, 866, 900, 937, 970, 1007, 1043, 1082,
                            1120, 1159, 1198, 1240, 1280, 1323, 1366, 1408, 1450, 1495,
                            1540, 1584, 1634, 1679, 1728, 1775, 1823, 1874, 1925, 1975,
                            2029, 2077, 2135, 2185, 2241, 2296, 2352, 2405, 2460, 2524,
                            2579, 2641, 2698, 2756, 2821, 2882, 2945, 3007, 3072, 3131,
                            3200, 3266]


def random_distance(length: int, numseqs=1000) -> float:
    """
    Calculate the distance between a sorted seq. and a random seq. for the given length.

    This is used to measure entropy in a seq.
    """
    if length < len(_cached_random_distances):
        return _cached_random_distances[length]
    accum = 0
    for _ in range(numseqs):
        seq = random_range(length)
        distance = distance_from_sorted(seq)
        accum += distance
    avg = accum / numseqs
    return avg


def unsortedness(seq: seq_t) -> float:
    """
    The entropy of the ordering in this seq.

    Args:
        seq: the seq to evaluate

    Returns:
        a value between 0 and 1, where 0 means sorted and 1 means random order

    """
    return distance_from_sorted(seq) / float(random_distance(len(seq)))


def _unsortx(seq, entropy, margin=0, debug=False, calculate_rating=False):
    if isinstance(margin, int):
        margin = (margin, margin)
    N = len(seq) - sum(margin)
    orig_idx = _np.arange(0, N)
    random_idx = _np.arange(0, orig_idx.size)
    _np.random.shuffle(random_idx)
    interpolated_idx = entropy * (random_idx - orig_idx) + orig_idx
    distances = abs(interpolated_idx - orig_idx)
    mean_dist = distances.sum() / distances.size

    def max_distance(N):
        delta = int(N * 0.5 + 0.5)
        idxs = _np.arange(N, dtype=int)
        return _np.abs((idxs - ((idxs + delta) % N))).sum()

    def worst_distribution(N):
        distance_left = max_distance(N)
        out = []
        max_individual_distance = N - 1
        for i in range(N):
            d = max_individual_distance if distance_left > max_individual_distance else distance_left
            distance_left -= d
            out.append(d)
        return out
    max_dist = max_distance(distances.size)
    resulting_entropy = mean_dist / (max_dist / distances.size)
    subseq_indices = _np.argsort(interpolated_idx) + margin[0]
    seq_array = _np.asarray(seq)
    out = seq_array.copy()
    if margin != (0, 0):
        out[margin[0]:-margin[1]] = out[subseq_indices]
    else:
        out = out[subseq_indices]
    if debug:
        assert len(out) == len(seq)
        assert set(out) == set(seq)
    if not calculate_rating:
        rating = 0
    else:
        deviation = distances.std()
        maxdeviation = _np.array(worst_distribution(N)).std()
        rel_deviation = deviation / maxdeviation
        entropy_weight = 10
        deviation_weight = 10
        # rel_entropy_gap = abs(resulting_entropy - entropy) / entropy
        # the highest the rating, the better
        # rel_entropy_gap: the higher, the further away from desired entropy -> the worse
        # rel_deviation: the higher, the more concentrated the distances in
        # individual indices --> the worse
        rating = 1 - ((abs(resulting_entropy - entropy) / entropy) *
                      10 + rel_deviation * 10) / (entropy_weight + deviation_weight)
    if entropy > 0:
        if _np.all(out == seq_array):
            rating = 0
    return out, rating


def unsort(seq: list, entropy: float, margin=0, tolerance=0.05, numiter=100
           ) -> _np.ndarray:
    """
    Generate a permutation of xs unsorted according to the given entropy.

    Args:
        seq: a sequence to be unsorted
        entropy: 0=the original sequence is returned; 1=random sequence is returned
        margin: a number or tuple (left, right). These elements are left untouched
        numiter: the number of times the algorithm is run. The best result will be returned

    Returns:
        un unsorted version of *seq*, as numpy array, or None if


    * If entropy == 0: the original sequence is returned
    * If entropy == 1: a sequence is generated which is as random as possible (this does
      not mean that there cannot be any fixed points, it refers to the general result)

    Examples
    --------

    .. code::

        # unsort the first 10 numbers, leave 0 and 9 untouched at their places
        unsort(range(10), 0.5, margin=1)

        # unsort the given seq., do not touch the first too elements
        unsort((1,3, 5, 4, 0), 0.2, margin=(2, 0))

    """
    minentropy = entropy - tolerance*0.5
    maxentropy = entropy + tolerance*0.5
    results = []
    for i in range(numiter):
        result, rating = _unsortx(seq, entropy, margin)
        if minentropy <= unsortedness(result) <= maxentropy:
            return result
        results.append((result, rating))
    if not results:
        raise ValueError("Could not unsort")
    results.sort(key=lambda result: abs(unsortedness(result[0]) - entropy))
    return results[0][0]


def unsort2(xs, entropy=1, margin=0, error=0.01, numiter=100):
    minentropy = entropy - error
    maxentropy = entropy + error
    bestentropy = 100
    bestsol = None
    for i in range(numiter):
        sol = _unsort(xs, entropy, margin)
        ent = unsortedness(sol)
        if minentropy <= ent <= maxentropy:
            return sol
        if abs(ent - entropy) < abs(entropy - bestentropy):
            bestentropy = ent
            bestsol = sol
    assert bestsol is not None
    return bestsol


def _unsort(xs: Sequence, entropy=1, margin: int | tuple[int, int] = 0):
    """
    generate a permutation of xs unsorted according to the given
    entropy.

    if entropy == 0: the original sequence is returned
    if entropy == 1: a sequence is generated which is as random as possible
        (this does not mean that there cannot be any fixed points, it refers
         to the general result)

    margin determines a range at the beginning and/or ending that will be left
    untouched.

    Examples:

    # unsort the first 10 numbers, leave 0 and 9 untouched at their places
    unsort(range(10), 0.5, margin=1)

    # unsort the given seq., do not touch the first too elements
    unsort((1,3, 5, 4, 0), 0.2, margin=(2, 0))
    """

    if margin != 0:
        if isinstance(margin, int):
            margin0, margin1 = margin, margin
        else:
            margin0, margin1 = margin
        margin1 = len(xs) - margin1
        unsorted = unsort(xs[margin0:margin1], entropy, 0)
        out = misc.copyseq(xs)
        out[margin0:margin1] = unsorted
        return out
    if entropy == 0:
        return xs
    L = len(xs)
    idxs0 = _np.arange(L)
    random_distr = idxs0.copy()
    NAN = _np.nan
    _np.random.shuffle(random_distr)
    if entropy == 1:
        return _np.asarray(xs)[random_distr]
    random_distr_entropy = _np.abs(random_distr - idxs0).sum() / (L ** 2)
    out = _np.ones_like(idxs0) * NAN
    pick_order = idxs0.copy()
    _np.random.shuffle(pick_order)
    poss_iss = []
    dx = entropy
    for i in pick_order:
        i0 = idxs0[i]
        i1 = random_distr[i]
        best_i = i0 + int((i1 - i0) * dx + 0.5)
        poss_is = [best_i]
        for j in range(1, L):
            n = best_i + j
            if n not in poss_is:
                poss_is.append(n)
            n = best_i - j
            if n not in poss_is:
                poss_is.append(n)
        poss_is = [poss_i for poss_i in poss_is if 0 <= poss_i < L]
        poss_iss.append(poss_is)
    for i, poss_is in zip(pick_order, poss_iss):
        i0 = idxs0[i]
        for poss_i in poss_is:
            if out[poss_i] != NAN:
                out[poss_i] = i0
                break
    n_missing = out[_np.isnan(out)].size
    if n_missing > 0:
        missing = [i for i in idxs0 if i not in out]
        nans = [i for i in range(L) if _np.isnan(out[i])]
        distances = [(abs(i - j), i, j) for i in missing for j in nans]
        distances2 = sorted(distances)
        missing_index = dict((m, True) for m in missing)
        for _, i, j in distances2:
            if missing_index[i]:
                if _np.isnan(out[j]):
                    missing_index[i] = False
                    out[j] = i
                    n_missing -= 1
                    if n_missing <= 0:
                        break
    indices = out.astype(int)
    xs = _np.asarray(xs)
    return xs[indices]


def permutation_further_than(xs: Sequence[T], min_distance: float, rand=True) -> list[T]:
    """
    Return a permutation of xs with a min. distance to it

    min_distance is an indication of entropy, where if min_distance == 0 then the seq
    should be xs and if min_distance == 1 then the elements are ordered as far away
    from the originals as poss.

    Args:
        xs: the seq. to permute
        min_distance: a min. distance (a value between 0 and 1)
        rand: ???

    Returns:
        the permuted seq.
    """
    acceptable_difference = _max_distance(xs) / len(xs) * 0.5

    def distance_from_origin(seq):
        return sum(abs(x - i) for i, x in enumerate(seq)) / len(seq)
    scaled_distance = min_distance * _max_distance(xs)
    best_distance = _max_distance(xs)
    #best_result = range(len(xs))[::-1]
    all_perm = itertools.permutations(len(xs))
    if rand:
        num_perm = reduce(mul, range(1, len(xs) * 1), 1)
        i = random.randint(0, int(num_perm / 5 + 1))
        i = min(i, 5000)
        all_perm0 = _iterlib.take(i, all_perm)
        all_perm = _iterlib.chain(all_perm, all_perm0)
    perm = None
    for perm0, perm1 in all_perm:
        dist = distance_from_origin(perm0)
        if scaled_distance <= dist <= best_distance:
            best_result0 = perm0
            best_distance0 = dist
            if abs(dist - scaled_distance) < acceptable_difference:
                perm = best_result0
                distance = best_distance0
                break
        dist = distance_from_origin(perm1)
        if scaled_distance <= dist <= best_distance:
            best_result1 = perm1
            best_distance1 = dist
            if abs(dist - scaled_distance) < acceptable_difference:
                perm = best_result1
                distance = best_distance1
                break
    if not perm:
        print("solution not found!")
        perm = best_result0 if best_distance0 < best_distance1 else best_result1
    return map(xs.__getitem__, perm)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

del mul
