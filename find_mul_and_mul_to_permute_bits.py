from collections import namedtuple

class SelectorMask:
    def __init__(self, mask, offset):
        self.mask = mask
        self.offset = offset
        pass

# single bit mask is tuple (length, index, value)
# e.g. (8, 2, 7) is mask (0, 0, 7, 0, 0, 0, 0, 0)
SingleBitMask = namedtuple('SingleBitMask', 'length index value')

# selector mask is like SingleBitMask with an added offset relative to input mask
#SelectorMask = namedtuple('SelectorMask', 'length index value offset')

# Selector masks can overlap in equal values (like zeros) only,
# except when the masks are at the same position, in which case they can be joined.
def do_single_bit_masks_conflict(bit1, start1, bit2, start2):
    stop1 = start1 + bit1.length
    stop2 = start2 + bit2.length
    start = max(start1, start2)
    stop = min(stop1, stop2)
    if start >= stop:
        return False  # no overlap => no conflict

    i1 = start1 + bit1.index
    i2 = start2 + bit2.index
    if i1 == i2 and bit1.value == bit2.value:
        return False  # both masks select the same => no conflict

    if start1 == start2:
        #print("joining:", bit1, start1, bit2, start2)
        return False  # masks are at the same position and can be joined => no conflict

    if start <= i1 < stop or start <= i2 < stop:
        return True  # non-equal overlap => conflict

    return False  # no conflict found


# Selector mask must match the input mask.
def does_single_bit_mask_match_input(bit, start, input):
    i = start + bit.index
    return 0 <= i < len(input) and input[i] == bit.value


def get_zeros_around(input, bits, offsets):
    bits = [b for b, o in zip(bits, offsets) if o >= 0]

    zeros_left, zeros_right = len(input), len(input)
    for i in range(len(bits)):
        start = offsets[i]
        stop = start + bits[i].length
        for j in range(len(bits)):
            pos = offsets[j] + bits[j].index
            if pos >= stop:
                zeros_right = min(zeros_right, pos - stop)
            elif pos < start:
                zeros_left = min(zeros_left, start - pos - 1)

    return zeros_left, zeros_right




def print_solution(input, input_multiplier, bits, offsets):
    def bit_value_to_char(value):
        if value == 0:
            return '.'
        if value <= 26:
            return chr(ord('a') - 1 + value)
        if value <= 2 * 26:
            return chr(ord('A') - 1 - 26 + value)
        return '?'

    indent = "    "
    print()

    zeros_left, zeros_right = get_zeros_around(input, bits, offsets)
    print(indent + ''.join(bit_value_to_char(i) for i in input) + '   zeros l/r: {}/{}'.format(zeros_left, zeros_right))

    lines = []
    for bit, offset in zip(bits, offsets):
        mask = bit_value_to_char(0) * bit.index + bit_value_to_char(bit.value) + bit_value_to_char(0) * (bit.length - 1 - bit.index)
        lines.append(" " * (offset if offset >= 0 else len(input)) + mask)

    lines_length = max(len(input), max(len(l) for l in lines))
    for i in range(len(lines)):
        bin = ['0'] * len(input)
        if offsets[i] >= 0:
            bin[len(bin) - 1 - offsets[i]] = '1'
            lines[i] = lines[i] + ' ' * (lines_length - len(lines[i])) + '   <<{:2d} '.format(offsets[i]) + ''.join(bin)

    for line in lines:
        print(indent + line)

    permutation_length = max(l for l, i, v in bits)
    permutation = [bit_value_to_char(0)] * permutation_length
    for i in range(len(bits)):
        if offsets[i] >= 0:
            permutation[bits[i].index] = bit_value_to_char(bits[i].value)

    print(indent + "-" * len(input))
    print(indent + ''.join(permutation))

    print()
    selector_mask = 0
    shift_multiplier = 0
    for bit, offset in zip(bits, offsets):
        if offset < 0:
            continue
        selector_mask = selector_mask | (1 << (len(input) - 1 - (offset + bit.index)))
        shift_multiplier = shift_multiplier | (1 << offset)

    print("f(x) = (x * 0x{:x}u & 0x{:x}u) * 0x{:x}u".format(input_multiplier, selector_mask, shift_multiplier))
    print("f(x) = (x * 0b{:b}u & 0b{:b}u) * 0b{:b}u".format(input_multiplier, selector_mask, shift_multiplier))

    print()


verbose = 0
show_some_partial_solutions = True
best_level = 0


# 0 is real binary zero, >0 identifies the bit value
def find_combined_mask(input, input_multiplier, permutation, permutation_priority = None):
    if verbose >= 1: print('input:', input)
    if verbose >= 2: print('permutation:', permutation)
    global best_level

    if permutation_priority is None:
        permutation_priority = (1,) * len(permutation)

    # Permutation of bits is to be generated at the highest position to reduce duplicate solutions;
    # if the permuation can be generated in a lower position, it can also be generated in highest position,
    # but not necessarily the other way round.
    #
    # Each position in permutation that is > 0 generates a mask.
    length = len(permutation)
    assert(len(permutation_priority) == length)
    bits = [SingleBitMask(length, i, v) for i, v, p in sorted(zip(range(length), permutation, permutation_priority), key=lambda t: -t[2]) if v != 0]
    if verbose >= 2: print('bits:', len(bits), bits)

    # We try all possible positions of single bit masks to find combinations that fit into input mask.
    # Masks can not shift left beyond highest bit position.
    # Masks can shift right beyond lowest bit position in zeros only.
    #
    offsets = [-1] * len(bits)
    max_offset = len(input) - 1
    level = 0  # recursion depth, is also position in bits and offsets
    max_level = len(offsets) - 1
    while level >= 0:
        #print("iteration:", offsets, level)

        # find next matching position of bits[level]
        offsets[level] += 1

        if offsets[level] > max_offset:
            # no solution found => reset this level, return to previous level
            offsets[level] = -1

            if (best_level <= level) if verbose >= 2 else (best_level < level):
                best_level = level
                if show_some_partial_solutions:
                    print("partial:", offsets)
                    print_solution(input, input_multiplier, bits, offsets)

            level -= 1
        else:

            if does_single_bit_mask_match_input(bits[level], offsets[level], input):
                has_conflict = False
                for l in range(0, level):
                    has_conflict = do_single_bit_masks_conflict(bits[level], offsets[level], bits[l], offsets[l])
                    if has_conflict:
                        break

                if not has_conflict:
                    if level < max_level:
                        # partial solution found => go to deeper level
                        level += 1
                        assert(offsets[level] == -1)
                    else:
                        # full solution found => print
                        print("full:", offsets)
                        print_solution(input, input_multiplier, bits, offsets)

    if verbose >= 2: print("Done.")


def enumerate_const_sum(sum, items):
    if items <= 1:
        yield (sum,)
    else:
        for first_value in range(sum + 1):
            for other_values in enumerate_const_sum(sum - first_value, items - 1):
                yield (first_value,) + other_values


#for t in enumerate_const_sum(4, 3):
#    print(t)


def find_padding_and_combined_mask(number_of_bits, input, permutation, permutation_priority = None, right_shift_max = 0):
    max_padding = min(16, number_of_bits // 2)  # much padding has never been observed to improve a solution, but it takes much time
    for padding in range(max_padding + 1):
        print('padding:', padding)
        # generate all combinations with total sum of padding
        # p[0] + input + p[1] + input + p[3] + input
        repetitions = (number_of_bits - padding + len(input) - 1) // len(input)
        for pad_perm in enumerate_const_sum(padding, repetitions):
            for right_shift in range(right_shift_max + 1):
                #if right_shift_max:
                #    print('right_shift:', right_shift)
                full_input = []
                input_multiplier = 0
                reversed_input = list(reversed(input))
                for p in pad_perm:
                    full_input += [0] * p
                    input_multiplier = input_multiplier | (1<<len(full_input))
                    full_input += reversed_input

                full_input = tuple(reversed(full_input[:number_of_bits]))
                assert(len(full_input) == number_of_bits)
                if right_shift:
                    full_input = (0,) * right_shift + full_input[:number_of_bits-right_shift]
                    assert(len(full_input) == number_of_bits)
                find_combined_mask(full_input, input_multiplier, permutation, permutation_priority)

#find_combined_mask(
#    (8, 7, 6, 5, 4, 3, 2, 1) * 4, 0x01010101,
#    (1, 2, 3, 4, 5, 6, 7, 8),
#    (1, 3, 3, 3, 3, 3, 3, 2))

if False:  # no solutions
    find_padding_and_combined_mask(
        32,
        (8, 7, 6, 5, 4, 3, 2, 1),
        (1, 2, 3, 4, 5, 6, 7, 8),
        right_shift_max = 7)

if False:  # a few solutions, but none that generates a constant zero bit to the right
    find_padding_and_combined_mask(
        32,
        (8, 7, 6, 5, 4, 3, 2, 1),
        (1, 2, 3, 4, 5, 6, 7))

if False:  # no solutions, as was to be expected from previous case
    find_padding_and_combined_mask(
        32,
        (8, 7, 6, 5, 4, 3, 2, 1),
        (1, 2, 3, 4, 5, 6, 7, 0))

if True:  # 2 solutions
    find_padding_and_combined_mask(
        32,
        (8, 7, 6, 5, 4, 3, 2, 1),
        (2, 3, 4, 5, 6, 7, 8))

if False:  # a few more solutions, some with "zeros l/r" of 1/0 or 0/1
    find_padding_and_combined_mask(
        32,
        (8, 7, 6, 5, 4, 3, 2, 1),
        (2, 3, 4, 5, 6, 7, 8),
        right_shift_max = 7)

if False:  # many solutions
    find_padding_and_combined_mask(
        64,
        (10, 9, 8, 7, 6, 5, 4, 3, 2, 1),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

if False:  # no solutions
    find_padding_and_combined_mask(
        64,
        (11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), right_shift_max = 7)

if False:  # no solutions
    find_padding_and_combined_mask(
        128,
        (16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))

if False:  # no solutions
    find_padding_and_combined_mask(
        128,
        (15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))

if False:  # many solutions
    find_padding_and_combined_mask(
        128,
        (14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))

if False:  # many solutions
    find_padding_and_combined_mask(
        32,
        (8, 7, 6, 5, 4, 3, 2, 1),
        (1, 2, 3, 4, 0, 0, 0, 0))

if False:  # many solutions, some with zeros l/r: 4/0
    find_padding_and_combined_mask(
        32,
        (8, 7, 6, 5, 4, 3, 2, 1),
        (5, 6, 7, 8, 0, 0, 0, 0))

if False:  # no solutions?
    find_padding_and_combined_mask(
        32,
        (8, 7, 6, 5, 4, 3, 2, 1),
        (0, 0, 0, 0, 5, 6, 7, 8))
