import itertools
from functools import reduce
from copy import deepcopy
import logging
from .helpers import ensure_list

logger = logging.getLogger("pydra")

# dj: might create a new class or move to State


# Function to change user provided splitter to "reverse polish notation" used in State
def splitter2rpn(splitter, other_states=None, state_fields=True):
    """ Functions that translate splitter to "reverse polish notation."""
    if not splitter:
        return []
    output_splitter = []
    _ordering(
        deepcopy(splitter),
        i=0,
        output_splitter=output_splitter,
        other_states=other_states,
        state_fields=state_fields,
    )
    return output_splitter


def _ordering(
    el, i, output_splitter, current_sign=None, other_states=None, state_fields=True
):
    """ Used in the splitter2rpn to get a proper order of fields and signs. """
    if type(el) is tuple:
        # checking if the splitter dont contain splitter from previous nodes, i.e. has str "_NA", etc.
        if type(el[0]) is str and el[0].startswith("_"):
            node_nm = el[0][1:]
            if node_nm not in other_states:
                raise Exception(
                    "can't ask for splitter from {}, other nodes that are connected: ".format(
                        node_nm, other_states.keys()
                    )
                )
            splitter_mod = change_splitter(
                splitter=other_states[node_nm][0].splitter_final, name=node_nm
            )
            if state_fields:
                el = (splitter_mod, el[1])
            if other_states[node_nm][0].other_states:
                other_states.update(other_states[node_nm][0].other_states)
        if type(el[1]) is str and el[1].startswith("_"):
            node_nm = el[1][1:]
            if node_nm not in other_states:
                raise Exception(
                    "can't ask for splitter from {}, other nodes that are connected: ".format(
                        node_nm, other_states.keys()
                    )
                )
            splitter_mod = change_splitter(
                splitter=other_states[node_nm][0].splitter_final, name=node_nm
            )
            if state_fields:
                el = (el[0], splitter_mod)
            if other_states[node_nm][0].other_states:
                other_states.update(other_states[node_nm][0].other_states)
        _iterate_list(
            el,
            ".",
            other_states,
            output_splitter=output_splitter,
            state_fields=state_fields,
        )
    elif type(el) is list:
        if type(el[0]) is str and el[0].startswith("_"):
            node_nm = el[0][1:]
            if node_nm not in other_states:
                raise Exception(
                    "can't ask for splitter from {}, other nodes that are connected: ".format(
                        node_nm, other_states.keys()
                    )
                )
            splitter_mod = change_splitter(
                splitter=other_states[node_nm][0].splitter_final, name=node_nm
            )
            if state_fields:
                el[0] = splitter_mod
            if other_states[node_nm][0].other_states:
                other_states.update(other_states[node_nm][0].other_states)
        if type(el[1]) is str and el[1].startswith("_"):
            node_nm = el[1][1:]
            if node_nm not in other_states:
                raise Exception(
                    "can't ask for splitter from {}, other nodes that are connected: ".format(
                        node_nm, other_states.keys()
                    )
                )
            splitter_mod = change_splitter(
                splitter=other_states[node_nm][0].splitter_final, name=node_nm
            )
            if state_fields:
                el[1] = splitter_mod
            if other_states[node_nm][0].other_states:
                other_states.update(other_states[node_nm][0].other_states)
        _iterate_list(
            el,
            "*",
            other_states,
            output_splitter=output_splitter,
            state_fields=state_fields,
        )
    elif type(el) is str:
        if el.startswith("_"):
            node_nm = el[1:]
            if node_nm not in other_states:
                raise Exception(
                    "can't ask for splitter from {}, other nodes that are connected: ".format(
                        node_nm, other_states.keys()
                    )
                )
            splitter_mod = change_splitter(
                splitter=other_states[node_nm][0].splitter_final, name=node_nm
            )
            if state_fields:
                el = splitter_mod
            if other_states[node_nm][0].other_states:
                other_states.update(other_states[node_nm][0].other_states)
        if type(el) is str:
            output_splitter.append(el)
        elif type(el) is tuple:
            _iterate_list(
                el,
                ".",
                other_states,
                output_splitter=output_splitter,
                state_fields=state_fields,
            )
        elif type(el) is list:
            _iterate_list(
                el,
                "*",
                other_states,
                output_splitter=output_splitter,
                state_fields=state_fields,
            )
    else:
        raise Exception("splitter has to be a string, a tuple or a list")
    if i > 0:
        output_splitter.append(current_sign)


def _iterate_list(element, sign, other_states, output_splitter, state_fields=True):
    """ Used in the splitter2rpn to get recursion. """
    for i, el in enumerate(element):
        _ordering(
            deepcopy(el),
            i,
            current_sign=sign,
            other_states=other_states,
            output_splitter=output_splitter,
            state_fields=state_fields,
        )


# functions used in State to know which element should be used for a specific axis


def splitting_axis(state_inputs, splitter_rpn):
    """Having inputs and splitter (in rpn notation), functions returns the axes of output for every input."""
    axis_for_input = {}
    stack = []
    # to remember current axis
    current_axis = None
    # to remember shapes and axes for partial results
    out_shape = {}
    out_axes = {}
    # to remember imput names for partial results
    out_inputname = {}
    for el in splitter_rpn:
        # scalar splitter
        if el == ".":
            right = stack.pop()
            left = stack.pop()
            # when both, left and right, are already products of partial splitter
            if left.startswith("OUT") and right.startswith("OUT"):
                if out_shape[left] != out_shape[right]:
                    raise Exception(
                        "arrays for scalar operations should have the same size"
                    )
                current_inputs = out_inputname[left] + out_inputname[right]
            # when left is already product of partial splitter
            elif left.startswith("OUT"):
                if (
                    state_inputs[right].shape == out_shape[left]
                ):  # todo:should we allow for one-element array?
                    axis_for_input[right] = out_axes[left]
                else:
                    raise Exception(
                        "arrays for scalar operations should have the same size"
                    )
                current_inputs = out_inputname[left] + [right]
            # when right is already product of partial splitter
            elif right.startswith("OUT"):
                if state_inputs[left].shape == out_shape[right]:
                    axis_for_input[left] = out_axes[right]
                else:
                    raise Exception(
                        "arrays for scalar operations should have the same size"
                    )
                current_inputs = out_inputname[right] + [left]

            else:
                if state_inputs[right].shape == state_inputs[left].shape:
                    current_axis = list(range(state_inputs[right].ndim))
                    current_shape = state_inputs[left].shape
                    axis_for_input[left] = current_axis
                    axis_for_input[right] = current_axis
                    current_inputs = [left, right]
                else:
                    raise Exception(
                        "arrays for scalar operations should have the same size"
                    )
            # adding partial output to the stack
            stack.append("OUT_{}".format(len(out_shape)))
            out_inputname["OUT_{}".format(len(out_shape))] = current_inputs
            out_axes["OUT_{}".format(len(out_shape))] = current_axis
            out_shape["OUT_{}".format(len(out_shape))] = current_shape

        # outer splitter
        elif el == "*":
            right = stack.pop()
            left = stack.pop()
            # when both, left and right, are already products of partial splitter
            if left.startswith("OUT") and right.startswith("OUT"):
                # changing all axis_for_input for inputs from right
                for key in out_inputname[right]:
                    axis_for_input[key] = [
                        i + len(out_axes[left]) for i in axis_for_input[key]
                    ]
                current_axis = out_axes[left] + [
                    i + (out_axes[left][-1] + 1) for i in out_axes[right]
                ]
                current_shape = tuple([i for i in out_shape[left] + out_shape[right]])
                current_inputs = out_inputname[left] + out_inputname[right]
            # when left is already product of partial splitter
            elif left.startswith("OUT"):
                axis_for_input[right] = [
                    i + (out_axes[left][-1] + 1)
                    for i in range(state_inputs[right].ndim)
                ]
                current_axis = out_axes[left] + axis_for_input[right]
                current_shape = tuple(
                    [i for i in out_shape[left] + state_inputs[right].shape]
                )
                current_inputs = out_inputname[left] + [right]
            # when right is already product of partial splitter
            elif right.startswith("OUT"):
                # changing all axis_for_input for inputs from right
                for key in out_inputname[right]:
                    axis_for_input[key] = [
                        i + state_inputs[left].ndim for i in axis_for_input[key]
                    ]
                axis_for_input[left] = [
                    i - len(out_shape[right]) + (out_axes[right][-1] + 1)
                    for i in range(state_inputs[left].ndim)
                ]
                current_axis = out_axes[right] + [
                    i + (out_axes[right][-1] + 1)
                    for i in range(state_inputs[left].ndim)
                ]
                current_shape = tuple(
                    [i for i in state_inputs[left].shape + out_shape[right]]
                )
                current_inputs = out_inputname[right] + [left]
            else:
                axis_for_input[left] = list(range(state_inputs[left].ndim))
                axis_for_input[right] = [
                    i + state_inputs[left].ndim for i in range(state_inputs[right].ndim)
                ]
                current_axis = axis_for_input[left] + axis_for_input[right]
                current_shape = tuple(
                    [i for i in state_inputs[left].shape + state_inputs[right].shape]
                )
                current_inputs = [left, right]
            # adding partial output to the stack
            stack.append("OUT_{}".format(len(out_shape)))
            out_inputname["OUT_{}".format(len(out_shape))] = current_inputs
            out_axes["OUT_{}".format(len(out_shape))] = current_axis
            out_shape["OUT_{}".format(len(out_shape))] = current_shape

        # just a name of input
        else:
            stack.append(el)

    if len(stack) == 0:
        pass
    elif len(stack) > 1:
        raise Exception("exception from splitting_axis")
    elif not stack[0].startswith("OUT"):
        current_axis = [i for i in range(state_inputs[stack[0]].ndim)]
        axis_for_input[stack[0]] = current_axis

    if current_axis:
        ndim = max(current_axis) + 1
    else:
        ndim = 0
    return axis_for_input, ndim


def converter_groups_to_input(group_for_inputs):
    """
    Having axes for all the input fields,
    the function returns fields for each axis and number of all groups.
    """
    input_for_axis = {}
    ngr = 0
    for inp, grs in group_for_inputs.items():
        for gr in ensure_list(grs):
            if gr in input_for_axis.keys():
                input_for_axis[gr].append(inp)
            else:
                ngr += 1
                input_for_axis[gr] = [inp]
    return input_for_axis, ngr


# TODO: currently not used
def groups_stack_input(group_for_inputs, groups_stack):
    """ function that helps testing groups_stack_final
        returns groups_stack_final with input names
    """
    inputs_for_groups = converter_groups_to_input(group_for_inputs)[0]
    groups_stack_input = []
    for stack_el in groups_stack:
        stack_el_inp = []
        for gr in stack_el:
            stack_el_inp += inputs_for_groups[gr]
        groups_stack_input.append(stack_el_inp)
    return groups_stack_input


# TODO: not used currently, think if I need it
def converting_axis2input(axis_for_input, ndim, state_inputs=None):
    """ Having axes for all the input fields, the function returns fields for each axis. """
    input_for_axis = []
    shape = []
    for i in range(ndim):
        input_for_axis.append([])
        shape.append(0)

    for inp, axis in axis_for_input.items():
        for (i, ax) in enumerate(axis):
            input_for_axis[ax].append(inp)
            if state_inputs is not None:
                shape[ax] = state_inputs[inp].shape[i]

    if state_inputs is not None:
        return input_for_axis, shape
    else:
        return input_for_axis


# function used in State if combiner

# TODO: not used currently, think if I need it
def matching_input_from_splitter(splitter_rpn):
    """similar to splitting_axis, but without state_input,
        finding inputs that are for the same axes.
        can't find the final dimensions without inputs.
    """
    axes_for_inputs = {}
    output_inputs = {}
    stack_inp = []
    for el in splitter_rpn:
        if el == ".":
            right, left = stack_inp.pop(), stack_inp.pop()
            out_nm = "OUT{}".format(len(output_inputs))
            if left.startswith("OUT") and right.startswith("OUT"):
                output_inputs[out_nm] = output_inputs[left] + output_inputs[right]
                axes_for_inputs[out_nm] = axes_for_inputs[left].copy()
            elif right.startswith("OUT"):
                output_inputs[out_nm] = output_inputs[right] + [left]
                axes_for_inputs[out_nm] = axes_for_inputs[right].copy()
                axes_for_inputs[left] = axes_for_inputs[right].copy()
            elif left.startswith("OUT"):
                output_inputs[out_nm] = output_inputs[left] + [right]
                axes_for_inputs[out_nm] = axes_for_inputs[left].copy()
                axes_for_inputs[right] = axes_for_inputs[left].copy()
            else:
                output_inputs[out_nm] = [left, right]
                axes_for_inputs[left] = [0]
                axes_for_inputs[out_nm] = [0]
                axes_for_inputs[right] = [0]
            stack_inp.append(out_nm)
        elif el == "*":
            right, left = stack_inp.pop(), stack_inp.pop()
            out_nm = "OUT{}".format(len(output_inputs))
            if left.startswith("OUT") and right.startswith("OUT"):
                output_inputs[out_nm] = output_inputs[left] + output_inputs[right]
                for inp in output_inputs[right] + [right]:
                    axes_for_inputs[inp] = [
                        i + len(axes_for_inputs[left]) for i in axes_for_inputs[inp]
                    ]
                axes_for_inputs[out_nm] = axes_for_inputs[left] + axes_for_inputs[right]
            elif right.startswith("OUT"):
                output_inputs[out_nm] = output_inputs[right] + [left]
                axes_for_inputs[left] = [min(axes_for_inputs[right]) - 1]
                axes_for_inputs[out_nm] = axes_for_inputs[left] + axes_for_inputs[right]
            elif left.startswith("OUT"):
                output_inputs[out_nm] = output_inputs[left] + [right]
                axes_for_inputs[right] = [max(axes_for_inputs[left]) + 1]
                axes_for_inputs[out_nm] = axes_for_inputs[left] + axes_for_inputs[right]
            else:
                output_inputs[out_nm] = [left, right]
                axes_for_inputs[left] = [0]
                axes_for_inputs[right] = [1]
                axes_for_inputs[out_nm] = [0, 1]
            stack_inp.append(out_nm)
        else:
            stack_inp.append(el)

    # checking if at the end I have only one element
    if len(stack_inp) == 1 and stack_inp[0].startswith("OUT"):
        pass
    elif len(stack_inp) == 1:
        axes_for_inputs[stack_inp[0]] = [0]
    else:
        raise Exception("something wrong with the splittper")

    # removing "OUT*" elements
    axes_for_inputs = dict(
        (key, val)
        for (key, val) in axes_for_inputs.items()
        if not key.startswith("OUT")
    )

    # checking if I have any axes below 0
    all_axes = []
    for _, val in axes_for_inputs.items():
        all_axes += val
    min_ax = min(all_axes)
    # moving all axes in case min_ax <0 , so everything starts from 0
    if min_ax < 0:
        axes_for_inputs = dict(
            (key, [v + abs(min_ax) for v in val])
            for (key, val) in axes_for_inputs.items()
        )

    # dimensions
    ndim = len(set(all_axes))

    return axes_for_inputs, ndim


def remove_inp_from_splitter_rpn(splitter_rpn, inputs_to_remove):
    """modifying splitter_rpn: removing inputs due to combining"""
    splitter_rpn_copy = splitter_rpn.copy()
    # reverting order
    splitter_rpn_copy.reverse()
    stack_inp = []
    stack_sgn = []
    from_last_sign = []
    for (ii, el) in enumerate(splitter_rpn_copy):
        # element is a sign
        if el == "." or el == "*":
            stack_sgn.append((ii, el))
            from_last_sign.append(0)
        # it's an input but not to remove
        elif el not in inputs_to_remove:
            if from_last_sign:
                from_last_sign[-1] += 1
            stack_inp.append((ii, el))
        # it'a an input that should be removed
        else:
            if not from_last_sign:
                pass
            elif from_last_sign[-1] <= 1:
                stack_sgn.pop()
                from_last_sign.pop()
            else:
                stack_sgn.pop(-1 * from_last_sign.pop())

    # creating the final splitter_rpn after combining
    remaining_elements = stack_sgn + stack_inp
    remaining_elements.sort(reverse=True)
    splitter_rpn_combined = [el for (i, el) in remaining_elements]
    return splitter_rpn_combined


def rpn2splitter(splitter_rpn):
    """recurrent algorithm to move from splitter_rpn to splitter.
       every time combines pairs of input in one input,
       ends when the length is one
     """
    if splitter_rpn == []:
        return None
    if len(splitter_rpn) == 1:
        return splitter_rpn[0]

    splitter_rpn_copy = splitter_rpn.copy()
    signs = [".", "*"]
    splitter_modified = []

    while splitter_rpn_copy:
        el = splitter_rpn_copy.pop()
        # element is a sign
        if el in signs:
            if (
                splitter_rpn_copy[-1] not in signs
                and splitter_rpn_copy[-2] not in signs
            ):
                right, left = splitter_rpn_copy.pop(), splitter_rpn_copy.pop()
                if el == ".":
                    splitter_modified.append((left, right))
                elif el == "*":
                    splitter_modified.append([left, right])
            else:
                splitter_modified.append(el)
        else:
            splitter_modified.append(el)

    # reversing the list and combining more
    splitter_modified.reverse()
    return rpn2splitter(splitter_modified)


# used in the Node to change names in a splitter and combiner


def change_combiner(combiner, name):
    combiner_changed = []
    for comb in combiner:
        if "." not in comb:
            combiner_changed.append("{}.{}".format(name, comb))
        else:
            combiner_changed.append(comb)
    return combiner_changed


def change_splitter(splitter, name):
    """changing names of splitter: adding names of the node"""
    if isinstance(splitter, str):
        return _add_name([splitter], name)[0]
    elif isinstance(splitter, list):
        return _add_name(splitter, name)
    elif isinstance(splitter, tuple):
        splitter_l = list(splitter)
        return tuple(_add_name(splitter_l, name))


def _add_name(mlist, name):
    for i, elem in enumerate(mlist):
        if isinstance(elem, str):
            if "." in elem or elem.startswith("_"):
                pass
            else:
                mlist[i] = "{}.{}".format(name, mlist[i])
        elif isinstance(elem, list):
            mlist[i] = _add_name(elem, name)
        elif isinstance(elem, tuple):
            mlist[i] = list(elem)
            mlist[i] = _add_name(mlist[i], name)
            mlist[i] = tuple(mlist[i])
    return mlist


op = {".": zip, "*": itertools.product}


def flatten(vals, cur_depth=0, max_depth=None):
    if max_depth is None:
        max_depth = len(list(input_shape(vals)))
    values = []
    if cur_depth >= max_depth:
        values.append([vals])
    else:
        for val in vals:
            if isinstance(val, (list, tuple)):
                values.append(flatten(val, cur_depth + 1, max_depth))
            else:
                values.append([val])
    return itertools.chain.from_iterable(values)


def iter_splits(iterable, keys):
    for iter in list(iterable):
        yield dict(zip(keys, list(flatten(iter, max_depth=1000))))


def input_shape(in1):
    # TODO: have to be changed for inner splitter (sometimes different length)
    shape = [len(in1)]
    last_shape = None
    for value in in1:
        if isinstance(value, list):
            cur_shape = input_shape(value)
            if last_shape is None:
                last_shape = cur_shape
            elif last_shape != cur_shape:
                last_shape = None
                break
        else:
            last_shape = None
            break
    if last_shape is not None:
        shape.extend(last_shape)
    return tuple(shape)


# dj: changing the function so it takes splitter_rpn
def _splits(splitter_rpn, inputs, inner_inputs=None):
    """ Process splitter rpn from left to right
    """
    stack = []
    keys = []
    shapes_var = {}
    if inner_inputs:
        previous_states_ind = {
            "_{}".format(v.name): (v.ind_l_final, v.keys_final)
            for _, v in inner_inputs.items()
        }
        inner_inputs = {k: v for k, v in inner_inputs.items() if k in splitter_rpn}
        keys_fromLeftSpl = ["_{}".format(st.name) for _, st in inner_inputs.items()]
    else:
        previous_states_ind = {}
        inner_inputs = {}
        keys_fromLeftSpl = []

    # when splitter is a single element (no operators)
    if len(splitter_rpn) == 1:
        op_single = splitter_rpn[0]
        return _single_op_splits(
            op_single,
            inputs,
            inner_inputs,
            shapes_var,
            previous_states_ind,
            keys_fromLeftSpl,
        )

    terms = {}
    trm_val = {}
    trm_str = {}
    shape = {}
    # iterating splitter_rpn
    for token in splitter_rpn:
        if token in [".", "*"]:
            terms["R"] = stack.pop()
            terms["L"] = stack.pop()
            # checking if terms are strings, shapes, etc.
            for lr in ["L", "R"]:
                term = terms[lr]
                if isinstance(term, str):
                    if term.startswith("_"):
                        trm_val[lr] = previous_states_ind[term][0]
                        shape[lr] = (len(trm_val[lr]),)
                    else:
                        shape[lr] = input_shape(inputs[term])
                        trm_val[lr] = range(reduce(lambda x, y: x * y, shape[lr]))
                    trm_str[lr] = True
                    shapes_var[term] = shape[lr]
                else:
                    trm_val[lr], shape[lr] = term
                    trm_str[lr] = False

            # checking shapes and creating newshape
            if token == ".":
                if shape["L"] != shape["R"]:
                    raise ValueError(
                        "Operands {} and {} do not have same shape.".format(
                            terms["R"], terms["L"]
                        )
                    )
                newshape = shape["R"]
            if token == "*":
                newshape = tuple(list(shape["L"]) + list(shape["R"]))

            # creating list with keys
            new_keys = {}
            for lr in ["R", "L"]:
                if trm_str[lr]:
                    if terms[lr].startswith("_"):
                        new_keys[lr] = previous_states_ind[terms[lr]][1]
                    else:
                        new_keys[lr] = [terms[lr]]

            if trm_str["L"] and trm_str["R"]:
                # TODO: check why i'm using this order
                keys = keys + new_keys["L"] + new_keys["R"]
            elif trm_str["L"]:
                keys = new_keys["L"] + keys
            elif trm_str["R"]:
                keys = keys + new_keys["R"]

            #
            newtrm_val = {}
            for lr in ["R", "L"]:
                # TODO: rewrite once I have more tests
                if isinstance(terms[lr], str) and terms[lr] in inner_inputs:
                    # TODO: have to be changed if differ length
                    inner_len = [shape[lr][-1]] * reduce(
                        lambda x, y: x * y, shape[lr][:-1]
                    )
                    # this come from the previous node
                    outer_ind = inner_inputs[terms[lr]].ind_l
                    trmval_out = itertools.chain.from_iterable(
                        itertools.repeat(x, n) for x, n in zip(outer_ind, inner_len)
                    )
                    newtrm_val[lr] = op["."](trmval_out, trm_val[lr])
                    keys = (
                        keys[: keys.index(terms[lr])]
                        + inner_inputs[terms[lr]].keys_final
                        + keys[keys.index(terms[lr]) :]
                    )
                else:
                    newtrm_val[lr] = trm_val[lr]

            pushval = (op[token](newtrm_val["L"], newtrm_val["R"]), newshape)
            stack.append(pushval)
        else:  # name of one of the inputs (token not in [".", "*"])
            stack.append(token)

    val = stack.pop()
    if isinstance(val, tuple):
        val = val[0]
    return val, keys, shapes_var, keys_fromLeftSpl


# dj: TODO: do I need keys?
def _splits_groups(splitter_rpn, combiner=None, inner_inputs=None):
    """ Process splitter rpn from left to right
    """
    if not splitter_rpn:
        return [], {}, [], []
    stack = []
    keys = []
    groups = {}
    group_count = None
    if not combiner:
        combiner = []
    if inner_inputs:
        previous_states_ind = {
            "_{}".format(v.name): v.keys_final for _, v in inner_inputs.items()
        }
        inner_inputs = {k: v for k, v in inner_inputs.items() if k in splitter_rpn}
    else:
        previous_states_ind = {}
        inner_inputs = {}

    # when splitter is a single element (no operators)
    if len(splitter_rpn) == 1:
        op_single = splitter_rpn[0]
        return _single_op_splits_groups(
            op_single, combiner, inner_inputs, previous_states_ind, groups
        )

    # len(splitter_rpn) > 1
    # iterating splitter_rpn
    for token in splitter_rpn:
        if token in [".", "*"]:
            terms = {}
            terms["R"] = stack.pop()
            terms["L"] = stack.pop()

            # checking if opL/R are strings
            trm_str = {"L": False, "R": False}
            oldgroups = {}

            for lr in ["L", "R"]:
                if isinstance(terms[lr], str):
                    trm_str[lr] = True
                else:
                    oldgroups[lr] = terms[lr]

            if token == ".":
                if all(trm_str.values()):
                    if group_count is None:
                        group_count = 0
                    else:
                        group_count += 1
                    oldgroup = groups[terms["L"]] = groups[terms["R"]] = group_count
                elif trm_str["R"]:
                    groups[terms["R"]] = oldgroups["L"]
                    oldgroup = oldgroups["L"]
                elif trm_str["L"]:
                    groups[terms["L"]] = oldgroups["R"]
                    oldgroup = oldgroups["R"]
                else:
                    if len(ensure_list(oldgroups["L"])) != len(
                        ensure_list(oldgroups["R"])
                    ):
                        raise ValueError(
                            "Operands do not have same shape "
                            "(left one is {}d and right one is {}d.".format(
                                len(ensure_list(oldgroups["L"])),
                                len(ensure_list(oldgroups["R"])),
                            )
                        )
                    oldgroup = oldgroups["L"]
                    # dj: changing axes for Right part of the scalar op.
                    for k, v in groups.items():
                        if v in ensure_list(oldgroups["R"]):
                            groups[k] = ensure_list(oldgroups["L"])[
                                ensure_list(oldgroups["R"]).index(v)
                            ]
            else:  # if token == "*":
                if all(trm_str.values()):
                    if group_count is None:
                        group_count = 0
                    else:
                        group_count += 1
                    groups[terms["L"]] = group_count
                    group_count += 1
                    groups[terms["R"]] = group_count
                    oldgroup = [groups[terms["L"]], groups[terms["R"]]]
                elif trm_str["R"]:
                    group_count += 1
                    groups[terms["R"]] = group_count
                    oldgroup = ensure_list(oldgroups["L"]) + [groups[terms["R"]]]
                elif trm_str["L"]:
                    group_count += 1
                    groups[terms["L"]] = group_count
                    oldgroup = [groups[terms["L"]]] + ensure_list(oldgroups["R"])
                else:
                    oldgroup = ensure_list(oldgroups["L"]) + ensure_list(oldgroups["R"])

            # creating list of keys
            if trm_str["L"]:
                if terms["L"].startswith("_"):
                    keys = previous_states_ind[terms["L"]] + keys
                else:
                    keys.insert(0, terms["L"])
            if trm_str["R"]:
                if terms["R"].startswith("_"):
                    keys += previous_states_ind[terms["R"]]
                else:
                    keys.append(terms["R"])

            pushgroup = oldgroup
            stack.append(pushgroup)

        else:  # name of one of the inputs
            stack.append(token)

    groups_stack = stack.pop()
    if isinstance(groups_stack, int):
        groups_stack = [groups_stack]
    if inner_inputs:
        groups_stack = [[], groups_stack]
    else:
        groups_stack = [groups_stack]

    if combiner:
        keys_final, groups_final, groups_stack_final, combiner_all = combine_final_groups(
            combiner, groups, groups_stack, keys
        )
        return keys_final, groups_final, groups_stack_final, combiner_all
    else:
        return keys, groups, groups_stack, []


def _single_op_splits(
    op_single, inputs, inner_inputs, shapes_var, previous_states_ind, keys_fromLeftSpl
):
    import numpy as np

    if op_single.startswith("_"):
        return (
            previous_states_ind[op_single][0],
            previous_states_ind[op_single][1],
            None,
            keys_fromLeftSpl,
        )
    shape = input_shape(inputs[op_single])
    shapes_var[op_single] = shape
    trmval = range(reduce(lambda x, y: x * y, shape))
    if op_single in inner_inputs:
        # TODO: have to be changed if differ length
        inner_len = [shape[-1]] * reduce(lambda x, y: x * y, shape[:-1])
        # this come from the previous node
        outer_ind = inner_inputs[op_single].ind_l
        op_out = itertools.chain.from_iterable(
            itertools.repeat(x, n) for x, n in zip(outer_ind, inner_len)
        )
        res = op["."](op_out, trmval)
        val = res
        keys = inner_inputs[op_single].keys_final + [op_single]
        return val, keys, shapes_var, keys_fromLeftSpl
    else:
        val = op["*"](trmval)
        keys = [op_single]
        return val, keys, shapes_var, keys_fromLeftSpl


def _single_op_splits_groups(
    op_single, combiner, inner_inputs, previous_states_ind, groups
):
    if op_single in inner_inputs:
        # TODO: have to be changed if differ length
        # TODO: i think I don't want to add here from left part
        # keys = inner_inputs[op_single].keys_final + [op_single]
        keys = [op_single]
        groups[op_single], groups_stack = 0, [[], [0]]
    else:
        keys = [op_single]
        groups[op_single], groups_stack = 0, [[0]]
    if combiner:
        if combiner == [op_single]:
            return [], {}, [], combiner
        else:
            raise Exception(
                "combiner {} not in splitter_rpn: {}".format(combiner[0], [op_single])
            )
    else:
        return keys, groups, groups_stack, []


def combine_final_groups(combiner, groups, groups_stack, keys):
    input_for_groups, _ = converter_groups_to_input(groups)
    combiner_all = []
    for comb in combiner:
        for gr in ensure_list(groups[comb]):
            combiner_all += input_for_groups[gr]
    combiner_all = list(set(combiner_all))
    combiner_all.sort()

    # groups that were removed (so not trying to remove twice)
    grs_removed = []
    groups_stack_final = deepcopy(groups_stack)
    for comb in combiner:
        grs = groups[comb]
        for gr in ensure_list(grs):
            if gr in groups_stack_final[-1]:
                grs_removed.append(gr)
                groups_stack_final[-1].remove(gr)
            elif gr in grs_removed:
                pass
            else:
                raise Exception(
                    "input {} not ready to combine, you have to combine {} "
                    "first".format(comb, groups_stack[-1])
                )
    groups_final = {inp: gr for (inp, gr) in groups.items() if inp not in combiner_all}
    gr_final = list(set(groups_final.values()))
    map_gr_nr = {nr: i for (i, nr) in enumerate(sorted(gr_final))}
    groups_final = {inp: map_gr_nr[gr] for (inp, gr) in groups_final.items()}
    for i, groups_l in enumerate(groups_stack_final):
        groups_stack_final[i] = [map_gr_nr[gr] for gr in groups_l]

    keys_final = [key for key in keys if key not in combiner_all]
    # TODO: not sure if I have to calculate and return keys, groups, groups_stack
    return keys_final, groups_final, groups_stack_final, combiner_all


def map_splits(split_iter, inputs):
    for split in split_iter:
        yield {k: list(flatten(ensure_list(inputs[k])))[v] for k, v in split.items()}


""" Functions for merging and completing splitters in states.
 Used only in State, could be moved to that class
"""


def connect_splitters(splitter, other_states):
    if splitter:
        # if splitter is string, have to check if this is Left or Right part (Left is required)
        if isinstance(splitter, str):
            # so this is the Left part
            if splitter.startswith("_"):
                left_part = _complete_left(left=splitter, other_states=other_states)
                right_part = None
            else:  # this is Right part
                left_part = _complete_left(other_states=other_states)
                right_part = splitter
        # if splitter is tuple, it has to be either Left or Right part, you can't have (Left, Right)
        elif isinstance(splitter, tuple):
            lr_flag = _left_right_check(splitter, other_states=other_states)
            if lr_flag == "Left":
                left_part = _complete_left(left=splitter, other_states=other_states)
                right_part = None
            elif lr_flag == "Right":
                left_part = _complete_left(other_states=other_states)
                right_part = splitter
            else:
                raise Exception("splitter mix Left and Right parts in scalar splitter")
        elif isinstance(splitter, list):
            lr_flag = _left_right_check(splitter, other_states=other_states)
            if lr_flag == "Left":
                left_part = _complete_left(left=splitter, other_states=other_states)
                right_part = None
            elif lr_flag == "Right":
                left_part = _complete_left(other_states=other_states)
                right_part = splitter
            elif (
                _left_right_check(splitter[0], other_states=other_states) == "Left"
                and _left_right_check(splitter[1], other_states=other_states) == "Right"
            ):
                left_part = _complete_left(left=splitter[0], other_states=other_states)
                right_part = splitter[1]
            else:
                raise Exception("splitter doesn't have separated Left and Right parts")
        else:
            raise Exception(
                "splitter has to be str, tuple or list, "
                "{} was provided".format(type(splitter))
            )
    else:
        # if there is no splitter, I create the Left part
        left_part = _complete_left(other_states=other_states)
        right_part = None
    if right_part:
        splitter = [deepcopy(left_part), deepcopy(right_part)]
    else:
        splitter = deepcopy(left_part)
    return splitter, left_part, right_part


def _complete_left(other_states, left=None):
    """completing Left part: adding all splitters from previous nodes"""
    if left:
        rpn_left = splitter2rpn(left, other_states=other_states, state_fields=False)
        for name, (st, inp) in list(other_states.items())[::-1]:
            if "_{}".format(name) not in rpn_left and st.splitter_final:
                left = ["_{}".format(name), left]
    else:
        left = ["_{}".format(name) for name in other_states]
        if len(left) == 1:
            left = left[0]
    return left


def _left_right_check(splitter_part, other_states):
    """checking if splitter_part is purely Left or Right - string is returned.
    If the splitter_part is mixed None is returned.
    """
    rpn_part = splitter2rpn(
        splitter_part, other_states=other_states, state_fields=False
    )
    inputs_in_splitter = [i for i in rpn_part if i not in ["*", "."]]
    others_in_splitter = [
        True if el.startswith("_") else False for el in inputs_in_splitter
    ]
    if all(others_in_splitter):
        return "Left"
    elif (not all(others_in_splitter)) and (not any(others_in_splitter)):
        return "Right"


def inputs_types_to_dict(name, inputs):
    """converting type.Inputs to dictionary"""
    # dj: any better option?
    input_names = [nm for nm in inputs.__dataclass_fields__.keys() if nm != "_func"]
    inputs_dict = {}
    for field in input_names:
        inputs_dict["{}.{}".format(name, field)] = getattr(inputs, field)
    return inputs_dict
