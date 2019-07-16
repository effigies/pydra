from nipype.utils.filemanip import ensure_list
from collections import OrderedDict
import itertools
import pdb

from . import auxiliary as aux

"""
node = Node(..., split=..., combine=...)

node.get_execnodes()

node.complete
"""


class Dim(dict):
    def __init__(self, labels, values=None):
        super(Dim, self).__init__()
        self._length = None
        if values is None:
            values = [None for label in labels]
        self[labels] = values

    def __getitem__(self, label):
        if isinstance(label, tuple):
            return list(zip(*(self[l] for l in label)))
        return super().__getitem__(label)

    def __setitem__(self, labels, values):
        if isinstance(labels, tuple):
            for label, value in zip(labels, values):
                self[label] = value
            return

        if values is None and self.get(labels) is None:
            super().__setitem__(labels, None)
            return

        values = ensure_list(values)
        if self.get(labels) is not None:
            if values != self[labels]:
                raise KeyError("Label '{}' already exists: {!r}\n"
                               "Cannot set new value {!r}".format(labels, self[labels], values))
            return

        if self._length is None:
            self._length = len(values)
        elif len(values) != self._length:
            raise ValueError("Label {!r} does not have correct shape".format(labels))

        super().__setitem__(labels, values)


class Dimension(object):
    def __init__(self, labels, values=None):
        self.labels = ensure_list(labels)
        self.values = [values] if self.labels is labels else values

    def __eq__(self, obj):
        if isinstance(obj, Dimension):
            return bool(set(self.values) & set(obj.values))
        return any(obj == val for val in self.values)

    def __and__(self, obj):
        for selflabel, selfvalue in zip(selflabel, selfvalue):
            if selflabel in obj.labels:
                idx = obj.labels.index
        new_dim = deepcopy(self)
        for label, value in zip(obj.labels, obj.values):
            pass

        return new_dim


class Unset:
    obj = None
    def __new__(cls, *args, **kwargs):
        if cls.obj is None:
            cls.obj = super().__new__(cls, *args, **kwargs)
        return cls.obj

    def __repr__(self):
        return '<unset value>'


unset = Unset()


class Future:
    """ Basic future API
    Examples
    --------
    >>> exists = Future(5)
    >>> exists.ready
    True
    >>> exists.get()
    5

    >>> dne = Future()
    >>> dne.ready
    True
    >>> try:
    ...     dne.get()
    ... except ValueError as e:
    ...     print(e)
    Unset future
    >>> dne.set(6)
    >>> dne.get()
    6
    """
    def __init__(self, val=unset):
        if isinstance(val, Future):
            val = val._val
        self._exists = val is not unset
        self._val = val

    def set(self, val):
        self._val = val
        self._exists = True

    @property
    def ready(self):
        return self._exists

    def get(self):
        if not self._exists:
            raise ValueError("Unset future")
        return self._val


class ExecNode:
    def __init__(self,
                 name,
                 runnable,
                 inputs,
                 node):
        self._name = name
        self._runnable = runnable
        self._inputs = inputs
        self._node = node

    @property
    def ready(self):
        return all(Future(val).ready for val in self._inputs.values())

class Node:
    """Implements a graph as a node with parents"""
    def __init__(self,
                 name,
                 runnable,
                 split=None,
                 combine=None,
                 ):
        self._name = name
        self._parents = []
        self._runnable = runnable
        self._state = State(self, split, combine)

    @property
    def parents(self):
        return self._parents

    def add_parent(self, node):
        self._parents.append(node)

    def remove_parent(self, node):
        self._parents.remove(node)

    @property
    def name(self):
        return self._name

    @property
    def origin(self):
        return not self._parents

    def state(self):
        return self._state.resolve([parent.state for parent in self._parents])

    def ready(self):
        if not self._state.ready:
            return False

        return True

    #def execnodes(self):


class Workflow(Node):
    def __init__(self,
                 name,
                 split=None,
                 combine=None,
                 ):
        super().__init__(name, self, split, combine)
        self._subnodes = []

    def __call__(self, *args, **kwargs):
        pass

    def add(name, nodelike):
        if isinstance(nodelike, Node):
            #self._children.
            if nodelike.name != name:
                logger.warn('Resetting name on {!r} to {}'.format(nodelike, name))
                #nodelike.


def split_to_dims(split):
    if isinstance(split, str):
        return Dim((split,))
    elif isinstance(split, list):
        return [split_to_dims(elem) for elem in split]
    elif isinstance(split, tuple):
        return 


class State(object):
    """ing state manager

    A mapping state is a set of fields over which a compute node is to be mapped, along with their
    values, which may be known when the node is created or only when it is time to be run.

    Any two given fields may be mapped independently or together, equivalent to the Cartesian
    product or a natural join by index, respectively. When describing multiple mappings in
    relation to one another, the Cartesian product is indicated by a tuple and the natural join
    by a list.

    For lists ``A == [a1, a2]`` and ``B == [b1, b2]``, the Cartesian product ``[A, B]`` produces
    states

        [[(a1, b1), (a1, b2)],
         [(a2, b1), (a2, b2)]]

    Similarly the natural join ``(A, B)`` will produce states

        [(a1, b1), (a2, b2)]

    Note that the natural join is simply the diagonal of the Cartesian product. For brevity, we
    refer to the Cartesian product as the "outer product" and the natural-join-by-index as the
    "inner product", but this is idiosyncratic to Pydra.

    Mapping state is passed from one node to the next, and the initial node has a "unit" state.
    If an additional field is mapped over, it is implicitly an outer product, adding a
    dimension to the state object. If an explicit outer product of N fields is mapped over, then
    N dimensions are added to the state object.
    
    An inner product, by contrast, identifies dimensions. If an inner product of N fields is
    mapped over, generally one new dimension will be added, and may be referred to with N labels.
    However, if any of the fields is one already being mapped over, then no dimensions will be
    added to the state object, and the new fields will become additional labels on the existing
    dimension.

    Although order affects the outputs when state is collapsed, the following principles
    generally hold true:

    Commutativity: [a, b] = [b, a], (a, b) = (b, a)
    Associativity:  [[a, b], c] = [a, [b, c]], ((a, b), c) == (a, (b, c))

    Because of associativity, [a, b, c] is acceptable shorthand for [[a, b], c] or [a, [b, c]],
    and will thus be normalized to [a, [b, c]]

    Distribution: ([a, b], [c, d]) = [(a, c), (b, d)]

    A mapper is a string describing a field or a list or tuple of mappers

    Tuples of mappers indicate identity o

    """

    def __init__(self, node, split=None, combine=None):
        self._node = node
        self._combine = combine
        self.node_name = node.name

        self._dims = []
        if split:
            for elem in split:
                if isinstance

        self._split = []
        self._split_rpn = []
        if split:
            # changing mapper (as in rpn), so I can read from left to right
            # e.g. if mapper=('d', ['e', 'r']), _mapper_rpn=['d', 'e', 'r', '*', '.']
            self._split_rpn = [self.node_name + '.' + axis if axis not in ('*', '.') else axis
                               for axis in aux.mapper2rpn(split)]
            self._split = aux.rpn2mapper(self._split_rpn)

        self._axis_names = [axis for axis in self._split_rpn if axis not in ("*", ".")]

    def resolve(self, states):
        if not states:
            return self

        new_state = states[0].clone


    def prepare_state_input(self, state_inputs):
        """prepare all inputs, should be called once all input is available"""

        # dj TOTHINK: I actually stopped using state_inputs for now, since people wanted to have mapper not only
        # for state inputs. Might have to come back....
        self.state_inputs = state_inputs

        # not all input field have to be use in the mapper, can be an extra scalar
        self._input_names = list(self.state_inputs.keys())

        # dictionary[key=input names] = list of axes related to
        # e.g. {'r': [1], 'e': [0], 'd': [0, 1]}
        # ndim - int, number of dimension for the "final array" (that is not created)
        self._axis_for_input, self._ndim = aux.mapping_axis(self.state_inputs, self._mapper_rpn)

        # list of inputs variable for each axis
        # e.g. [['e', 'd'], ['r', 'd']]
        # shape - list, e.g. [2,3]
        self._input_for_axis, self._shape = aux.converting_axis2input(
            self.state_inputs, self._axis_for_input, self._ndim)

        # list of all possible indexes in each dim, will be use to iterate
        # e.g. [[0, 1], [0, 1, 2]]
        self.all_elements = [range(i) for i in self._shape]
        self.index_generator = itertools.product(*self.all_elements)

    def __getitem__(self, ind):
        if type(ind) is int:
            ind = (ind, )
        return self.state_values(ind)

    # not used?
    #@property
    #def mapper(self):
    #    return self._mapper

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return self._shape

    def state_values(self, ind):
        """returns state input as a dictionary (input name, value)"""
        if len(ind) > self._ndim:
            raise IndexError("too many indices")

        for ii, index in enumerate(ind):
            if index > self._shape[ii] - 1:
                raise IndexError("index {} is out of bounds for axis {} with size {}".format(
                    index, ii, self._shape[ii]))

        state_dict = {}
        for input, ax in self._axis_for_input.items():
            # checking which axes are important for the input
            sl_ax = slice(ax[0], ax[-1] + 1)
            # taking the indexes for the axes
            ind_inp = tuple(ind[sl_ax])  #used to be list
            state_dict[input] = self.state_inputs[input][ind_inp]
        # adding values from input that are not used in the mapper
        for input in set(self._input_names) - set(self._input_names_mapper):
            state_dict[input] = self.state_inputs[input]

        # in py3.7 we can skip OrderedDict
        # returning a named tuple?
        return OrderedDict(sorted(state_dict.items(), key=lambda t: t[0]))

    def state_ind(self, ind):
        """similar to state value but returns indices (not values)"""
        if len(ind) > self._ndim:
            raise IndexError("too many indices")

        for ii, index in enumerate(ind):
            if index > self._shape[ii] - 1:
                raise IndexError("index {} is out of bounds for axis {} with size {}".format(
                    index, ii, self._shape[ii]))

        state_dict = {}
        for input, ax in self._axis_for_input.items():
            # checking which axes are important for the input
            sl_ax = slice(ax[0], ax[-1] + 1)
            # taking the indexes for the axes
            ind_inp = tuple(ind[sl_ax])  #used to be list
            ind_inp_str = "x".join([str(el) for el in ind_inp])
            state_dict[input] = ind_inp_str
        # adding inputs that are not used in the mapper
        for input in set(self._input_names) - set(self._input_names_mapper):
            state_dict[input] = None

        # in py3.7 we can skip OrderedDict
        # returning a named tuple?
        return OrderedDict(sorted(state_dict.items(), key=lambda t: t[0]))
