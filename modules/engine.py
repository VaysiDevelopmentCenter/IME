import re
import random


class SimplePol:
    """
    Class to make from one asm file another one but polymorphic to the input file.
    """

    def __init__(self, path):
        """
        Initialisation of path to file and lists for parsing.
        :param path: string.
        """
        self.stack_register = None
        self.add_sub_register = None
        self.border_pos = None
        self.all_registers_lst = ['rdx', 'rax', 'rcx', 'rsi', 'r11', 'rdi', 'rbx', 'r8', 'r10', 'r9', 'r12', 'r13',
                                  'r14', 'r15', 'rbp', 'rsp']
        self.path = path
        self.content = list()
        self.mov_xor_jmp_je_jk_lst = list()
        self.add_sub_lst = list()
        self.add_sub_lst_im = list()
        self.add_sub_lst_reg = list()
        self.register_lst = list()
        self.mul_lst = list()
        self.cmp_lst = list()

    def reader(self):
        """
        Read asm file and parse it by \n.
        :return: None.
        """
        with open(self.path) as f:
            for line in f:
                self.content.append(line.strip().split('\n'))

    def parser(self, word, lst):
        """
        Find some command in the asm file.
        :param word: string
        :param lst: list of strings
        :return: None.
        """
        for i in range(len(self.content)):
            if re.match(word, self.content[i][0]):
                lst.append(self.content[i])

    def parser_register(self):
        """
        Find all registers that used in asm file.
        :return: None
        """
        for i in range(len(self.content)):
            length = len(self.content[i][0])
            register = str()
            while length != 0:
                if self.content[i][0][length - 1] == ',':
                    while self.content[i][0][length - 1] != ' ' and self.content[i][0][length - 1] != '\t':
                        length -= 1
                        if self.content[i][0][length - 1].isalnum():
                            register += self.content[i][0][length - 1]
                        else:
                            break
                    break
                length -= 1
            if register:
                register = register[::-1]
                self.register_lst.append(register)
        self.register_lst = list(set(self.register_lst))
        trash = list()
        for i in range(len(self.register_lst)):
            if len(self.register_lst[i]) > 3 or len(self.register_lst[i]) < 2:
                trash.append(self.register_lst[i])
        for i in range(len(trash)):
            self.register_lst.remove(trash[i])

    def parser_commands(self):
        """
        Find all mov, cmp, jmp,je,jk commands in asm file.
        :return: None
        """
        self.parser(r"mov|jmp|je|jk|xor", self.mov_xor_jmp_je_jk_lst)

    def parser_add_sub(self):
        """
        Find all add and sub commands in asm file.
        :return: None
        """
        self.parser(r"add|sub", self.add_sub_lst)

    def parser_cmp(self):
        """
        Find all cmp commands in asm file.
        :return: None
        """
        self.parser(r"cmp", self.cmp_lst)

    def parser_mul(self):
        """
        Find in asm file all mul commands amd related nov commands.
        :return: None
        """
        self.parser(r"mul", self.mul_lst)
        for i in range(len(self.mul_lst)):
            index = self.content.index(self.mul_lst[i])
            self.mul_lst.insert(i, self.content[index - 1])
            self.mul_lst.insert(i, self.content[index - 2])

    def set_border(self):
        """
        Function to detect the first command in asm code
        :return: None
        """
        for i in range(len(self.content)):
            if self.content[i] in self.mov_xor_jmp_je_jk_lst or self.content[i] in self.cmp_lst or \
                    self.content[i] in self.mul_lst or self.content[i] in self.add_sub_lst:
                self.border_pos = self.content[i]
                break

    def classification_add_sub(self):
        """
        Function to divide to different lists add and sub command. Add and sub with
        immediate and add with register.
        :return: None
        """
        for i in range(len(self.add_sub_lst)):
            if re.search(r', ?[0-9]+', self.add_sub_lst[i][0]):
                self.add_sub_lst_im.append(self.add_sub_lst[i])
            else:
                self.add_sub_lst_reg.append(self.add_sub_lst[i])

    @staticmethod
    def number_division(number):
        """
        Return a random number where param number is sup.
        :param number: int
        :return: None
        """
        return random.randrange(number)

    @staticmethod
    def line_maker(line, number, reverse=False):
        """
        Function to generate new line of asm code with add or sub and new
        numeric value.
        :param line: string
        :param number: int
        :param reverse: bool
        :return: string
        """
        new_line = str()
        i = 0
        while line[i] != ',':
            new_line += line[i]
            i += 1
        new_line += ','
        new_line += str(number)
        if reverse and new_line[0] == 'a':
            new_line = list(new_line)
            new_line[0], new_line[1], new_line[2] = 's', 'u', 'b'
            new_line = ''.join(new_line)
        elif reverse and new_line[0] == 's':
            new_line = list(new_line)
            new_line[0], new_line[1], new_line[2] = 'a', 'd', 'd'
            new_line = ''.join(new_line)
        return new_line

    def nope_adder(self, element):
        """
        Add to asm code nop.
        :param element: list of elements
        :return: None
        """
        index = self.content.index(element)
        self.content.insert(index, ['nop'])

    def division_adder_im(self, element):
        """
        Extract a number from a line and choose how to divide it.
        :param element: list
        :return: None
        """
        length = len(element[0])
        number = str()
        exact_number = str()
        while element[0][length - 1] != ',':
            number += element[0][length - 1]
            length -= 1
        number = number.split()
        for i in range(len(number)):
            if number[i].isdecimal():
                exact_number = number[i]
                break
        number = int(exact_number[::-1])
        div = self.number_division(number)
        choice = random.choice([1, 2, 3])
        if choice == 1:
            self.division_adder_im_2(element, div, number)
        elif choice == 2:
            self.division_adder_sub(element, div, number)
        else:
            self.division_adder_im_3(element, div, number)

    def division_adder_im_2(self, element, div, number):
        """
        Add to asm code divided command of add or sub like:
        add eax 5
        add eax 3
        Instead of add eax 8
        :param element: list
        :param div: number
        :param number: number
        :return: None
        """
        new_line = self.line_maker(element[0], div)
        self.content.insert(self.content.index(element), [new_line])
        self.content[self.content.index(element)] = \
            [self.line_maker(element[0], number - div)]

    def division_adder_im_3(self, element, div, number):
        """
        Add to asm code divided command of add or sub like:
        add eax 1
        add eax 4
        add eax 3
        Instead of add eax 8
        :param element: list
        :param div: number
        :param number: number
        :return: None
        """
        if div == 0:
            new_div = 0
        else:
            new_div = self.number_division(div)
        self.content.insert(self.content.index(element), [self.line_maker(element[0], new_div)])
        self.content.insert(self.content.index(element), [self.line_maker(element[0], div - new_div)])
        self.content[self.content.index(element)] = \
            [self.line_maker(element[0], number - div)]

    def division_adder_sub(self, element, div, number):
        """
        Function which transform single add or sub line.
        F.E.:
        Make from add eax, 10:
        add eax, 13
        sub eax, 3
        :param element: list of string
        :param div: number
        :param number: number
        :return: None
        """
        new_div = random.randint(number + 1, number + div + 1)
        self.content.insert(self.content.index(element), [self.line_maker(element[0], new_div)])
        self.content[self.content.index(element)] = \
            [self.line_maker(element[0], new_div - number, True)]

    def add_sub_adder(self, element):
        """
        Function which add two lines with add some number to eax and sub
        this number from eax register.
        :param element: list of elements
        :return: None
        """
        reg = str()
        if not self.add_sub_register:
            for i in range(len(self.all_registers_lst)):
                if self.all_registers_lst[i] not in self.register_lst\
                        and self.all_registers_lst[i] != self.stack_register:
                    reg = self.all_registers_lst[i]
                    self.add_sub_register = self.all_registers_lst[i]
                    break
        else:
            reg = self.add_sub_register
        index = self.content.index(element)
        number = self.number_division(10)
        self.content.insert(index, ['sub {}, {}'.format(reg, str(number))])
        self.content.insert(index, ['add {}, {}'.format(reg, str(number))])

    def stack_adder(self, element):
        """
        Add to asm code push and pop of some register.
        :param element: list of elements
        :return: None
        """
        reg = str()
        if not self.stack_register:
            for i in range(len(self.all_registers_lst)):
                if self.all_registers_lst[i] not in self.register_lst\
                        and self.add_sub_register != self.all_registers_lst[i]:
                    reg = self.all_registers_lst[i]
                    self.stack_register = self.all_registers_lst[i]
                    break
        else:
            reg = self.stack_register
        index = self.content.index(element)
        self.content.insert(index, [f'pop {reg}'])
        self.content.insert(index, [f'push {reg}'])

    def stack_nop_adder(self, element):
        """
        Add to code push pop of the register and nop between them.
        :param element: list of string
        :return: None
        """
        reg = str()
        if not self.stack_register:
            for i in range(len(self.all_registers_lst)):
                if self.all_registers_lst[i] not in self.register_lst \
                        and self.add_sub_register != self.all_registers_lst[i]:
                    reg = self.all_registers_lst[i]
                    self.stack_register = self.all_registers_lst[i]
                    break
        else:
            reg = self.stack_register
        index = self.content.index(element)
        self.content.insert(index, [f'pop {reg}'])
        self.content.insert(index, ['nop'])
        self.content.insert(index, [f'push {reg}'])

    def swap_of_reg(self, element):
        """
        Swap to registers in cmp command
        :param element: list of elements
        :return: None
        """
        if len(element[0]) > 16:
            return 0
        l_reg = str()
        f_reg = str()
        length = len(element[0]) - 1
        while element[0][length] != ',':
            l_reg += element[0][length]
            length -= 1
        while element[0][length] != 'p':
            f_reg += element[0][length]
            length -= 1
        if l_reg[-1] == ' ':
            l_reg = l_reg[:-1]
        l_reg = l_reg[::-1]
        if f_reg[-1] == ' ':
            f_reg = f_reg[:-1]
        f_reg = f_reg[::-1]
        f_reg = f_reg[:-1]
        if f_reg.isdecimal() or l_reg.isdecimal():
            return 0
        self.content[self.content.index(element)] = [f"cmp {l_reg}, {f_reg}"]

    def commands_transformer(self):
        """
        Modify every mov, jmp, jk, je command.
        :return: None
        """
        for i in range(len(self.mov_xor_jmp_je_jk_lst)):
            choice = random.choice([1, 2, 3, 4])
            if choice == 1:
                self.nope_adder(self.mov_xor_jmp_je_jk_lst[i])
            elif choice == 2:
                self.add_sub_adder(self.mov_xor_jmp_je_jk_lst[i])
            elif choice == 3:
                self.stack_nop_adder(self.mov_xor_jmp_je_jk_lst[i])
            else:
                self.stack_adder(self.mov_xor_jmp_je_jk_lst[i])

    def add_sub_transformer(self):
        """
        Modify every add and sub command.
        :return: None
        """
        self.set_border()
        for i in range(len(self.add_sub_lst_im)):
            choice = random.choice([1, 2, 3])
            if choice == 1:
                self.nope_adder(self.add_sub_lst_im[i])
            elif choice == 2:
                self.stack_adder(self.add_sub_lst_im[i])
            else:
                self.division_adder_im(self.add_sub_lst_im[i])
        for i in range(len(self.add_sub_lst_reg)):
            choice = random.choice([1, 2, 3])
            if choice == 1:
                self.nope_adder(self.add_sub_lst_reg[i])
            elif choice == 2:
                self.stack_adder(self.add_sub_lst_reg[i])
            else:
                self.add_sub_adder(self.add_sub_lst_reg[i])

    def mul_transform(self):
        """
        Function to transform a mul command.
        :return: None
        """
        for i in range(2, len(self.mul_lst), 3):
            choice = random.choice([i - 1, i - 2])
            element = self.mul_lst[choice]
            length = len(element[0])
            number = str()
            register = str()
            while element[0][length - 1] != ',':
                number += element[0][length - 1]
                length -= 1
            number = int(number[::-1])
            div = self.number_division(number)
            line = self.line_maker(element[0], div)
            self.content[self.content.index(element)] = [line]
            while element[0][length - 1] != ' ':
                register += element[0][length - 1]
                length -= 1
            register = register[::-1]
            self.content.insert(self.content.index([line]) + 1, ['add {} {}'.format(register, str(number - div))])

    def cmp_transform(self):
        """
        Function to transform cmp command.
        :return: None
        """
        for i in range(len(self.cmp_lst)):
            choice = random.choice([1, 2, 3])
            choice = 3
            if choice == 1:
                self.nope_adder(self.cmp_lst[i])
            elif choice == 2:
                self.stack_adder(self.cmp_lst[i])
            else:
                self.swap_of_reg(self.cmp_lst[i])

    def polymorph(self):
        """
        Make code polymorphous and write it to new asm file.
        :return:  None
        """
        self.reader()
        self.parser_add_sub()
        self.parser_mul()
        self.parser_commands()
        self.parser_register()
        self.parser_cmp()
        self.set_border()
        self.classification_add_sub()
        self.commands_transformer()
        self.add_sub_transformer()
        self.mul_transform()
        self.cmp_transform()
        content = str()
        for i in range(len(self.content)):
            content += self.content[i][0]
            if i != len(self.content) - 1:
                content += '\n'
        with open(f"{self.path[:-4]}_pol.asm", 'w') as f:
            f.write(content)


if __name__ == "__main__":
    a = SimplePol("simple.asm")
    a.polymorph()
    print(a.register_lst)

# --- Core IME Framework ---
from abc import ABC, abstractmethod
# import copy # May be useful for entities if mutations sometimes return new objects

class MutableEntity:
    """
    Represents an entity that can be mutated by MutationOperators.
    It acts as a wrapper around the actual data/object being mutated.
    """
    def __init__(self, data):
        self._data = data # The actual content to be mutated
        # In the future, we might add metadata, type hints, constraints, etc.

    @property
    def data(self):
        """Provides access to the internal data.
           Mutation operators will modify internal _data directly or via methods.
        """
        return self._data

    @data.setter
    def data(self, new_data):
        """Allows direct setting of data. Useful for initialization or reset.
           Careful consideration if this should be broadly used vs. mutation.
        """
        self._data = new_data

    def __str__(self):
        return f"MutableEntity(data={str(self._data)})"

    # Consider adding a clone method if needed for specific mutation strategies
    # def clone(self):
    #     import copy
    #     return MutableEntity(copy.deepcopy(self._data))

class MutationOperator(ABC):
    """
    Abstract Base Class for all mutation operators.
    A mutation operator defines a specific way to change a MutableEntity.
    """

    @abstractmethod
    def apply(self, entity: MutableEntity) -> None:
        """
        Applies the mutation to the given MutableEntity.
        This method should modify the entity's internal data.
        """
        pass

    def can_apply(self, entity: MutableEntity) -> bool:
        """
        Checks if this operator can be applied to the given entity.
        By default, operators can apply to any entity. Subclasses can override this
        for more specific applicability checks (e.g., based on data type).
        Returns True if applicable, False otherwise.
        """
        return True

# --- Basic Mutation Operators ---

class IntegerPerturbationOperator(MutationOperator):
    """
    Mutates an integer by adding a small random value to it.
    """
    def __init__(self, perturbation_range=(-5, 5)):
        super().__init__()
        if not (isinstance(perturbation_range, tuple) and
                len(perturbation_range) == 2 and
                isinstance(perturbation_range[0], int) and
                isinstance(perturbation_range[1], int) and
                perturbation_range[0] <= perturbation_range[1]):
            raise ValueError("perturbation_range must be a tuple of two integers (min, max) with min <= max.")
        self.min_perturb, self.max_perturb = perturbation_range

    def can_apply(self, entity: MutableEntity) -> bool:
        return isinstance(entity.data, int)

    def apply(self, entity: MutableEntity) -> None:
        if not self.can_apply(entity):
            # Or raise an error, or log a warning
            return

        perturbation = random.randint(self.min_perturb, self.max_perturb)
        # Ensure perturbation is not zero, to guarantee a change (optional, depends on desired behavior)
        while perturbation == 0 and self.min_perturb != self.max_perturb : # Avoid infinite loop if range is (0,0)
            perturbation = random.randint(self.min_perturb, self.max_perturb)

        entity.data += perturbation

class StringReplaceOperator(MutationOperator):
    """
    Mutates a string by replacing a random character with another random printable ASCII character.
    """
    def can_apply(self, entity: MutableEntity) -> bool:
        return isinstance(entity.data, str) and len(entity.data) > 0

    def apply(self, entity: MutableEntity) -> None:
        if not self.can_apply(entity):
            return

        s = entity.data
        idx = random.randrange(len(s))
        # ASCII printable characters range from 32 (space) to 126 (~)
        new_char_code = random.randint(32, 126)
        original_char = s[idx]
        new_char = chr(new_char_code)

        # Ensure the new character is different from the original (optional)
        while new_char == original_char and len(set(s)) > 1 : # Avoid infinite loop if all chars are the same
            new_char_code = random.randint(32, 126)
            new_char = chr(new_char_code)

        entity.data = s[:idx] + new_char + s[idx+1:]

class ListElementSwapOperator(MutationOperator):
    """
    Mutates a list by swapping two randomly selected distinct elements.
    """
    def can_apply(self, entity: MutableEntity) -> bool:
        return isinstance(entity.data, list) and len(entity.data) >= 2

    def apply(self, entity: MutableEntity) -> None:
        if not self.can_apply(entity):
            return

        lst = entity.data
        idx1, idx2 = random.sample(range(len(lst)), 2) # Ensures two distinct indices

        lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
        # No need to set entity.data = lst as list is modified in-place

# --- Mutation Engine ---

class MutationEngine:
    """
    Orchestrates the mutation process on a MutableEntity using a set of MutationOperators.
    """
    def __init__(self, operators: list[MutationOperator], mutation_probability: float = 1.0):
        if not operators or not all(isinstance(op, MutationOperator) for op in operators):
            raise ValueError("MutationEngine requires a non-empty list of MutationOperator instances.")
        if not (0.0 <= mutation_probability <= 1.0):
            raise ValueError("Mutation probability must be between 0.0 and 1.0.")
        self.operators = operators
        self.mutation_probability = mutation_probability # Placeholder for adaptive rate

    def evaluate_entity(self, entity: MutableEntity) -> float:
        """
        Evaluates the fitness or quality of a MutableEntity.
        Placeholder: In a real scenario, this would involve a complex assessment.
        Returns a numerical score (higher is generally better).
        """
        # print(f"Placeholder: Evaluating entity {entity}. Score: 0.0 (Not implemented)")
        return 0.0 # Default score

    def mutate_once(self, entity: MutableEntity) -> bool:
        """
        Attempts to apply a single, randomly chosen, applicable mutation operator
        to the given entity.

        Args:
            entity: The MutableEntity to mutate.

        Returns:
            True if a mutation was successfully applied, False otherwise.
        """
        if random.random() >= self.mutation_probability:
            # Optional: print("Mutation skipped due to probability.")
            return False # Skipped due to probability

        applicable_operators = [op for op in self.operators if op.can_apply(entity)]

        if not applicable_operators:
            # Optional: print(f"Warning: No applicable operators found for entity: {entity}")
            return False

        operator_to_apply = random.choice(applicable_operators)
        # Optional: print(f"Applying {operator_to_apply.__class__.__name__} to {entity}")
        operator_to_apply.apply(entity)

        # Placeholder for where evaluation might be used after a mutation
        # current_score = self.evaluate_entity(entity)
        # print(f"Entity score after mutation: {current_score}")
        return True

    def run(self, entity: MutableEntity, num_mutations: int) -> int:
        """
        Applies a sequence of mutations to the entity.

        Args:
            entity: The MutableEntity to mutate.
            num_mutations: The desired number of mutation steps to perform.

        Returns:
            The number of successful mutations applied.
        """
        if num_mutations < 0:
            raise ValueError("Number of mutations cannot be negative.")

        successful_mutations = 0
        for _ in range(num_mutations):
            if self.mutate_once(entity):
                successful_mutations += 1
        return successful_mutations

# --- Architectural Graph Representation for Neural Networks (and potentially other structures) ---

import uuid # For generating unique IDs

class ArchitecturalNode:
    """Represents a node in an architectural graph (e.g., a neuron, a layer component)."""
    def __init__(self, node_id: str = None, node_type: str = "generic", properties: dict = None):
        self.id = node_id if node_id is not None else str(uuid.uuid4())
        self.node_type = node_type
        self.properties = properties if properties is not None else {} # e.g., activation_function, bias, value

    def __str__(self):
        return f"Node(id={self.id}, type={self.node_type}, props={self.properties})"

class ArchitecturalEdge:
    """Represents an edge (connection) between two ArchitecturalNodes."""
    def __init__(self, source_node_id: str, target_node_id: str, edge_id: str = None, weight: float = 0.0, properties: dict = None):
        self.id = edge_id if edge_id is not None else str(uuid.uuid4())
        if not source_node_id or not target_node_id:
            raise ValueError("Source and target node IDs must be provided for an edge.")
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.weight = weight
        self.properties = properties if properties is not None else {} # e.g., is_recurrent, delay

    def __str__(self):
        return f"Edge(id={self.id}, from={self.source_node_id}, to={self.target_node_id}, w={self.weight}, props={self.properties})"

class PartitionSchema:
    """
    Represents a way of partitioning a set of elements within a NetworkGraph,
    inspired by Bell numbers and set partitions.
    """
    def __init__(self, name: str, target_element_type: str,
                 partitions: list[set[str]], schema_id: str = None, metadata: dict = None):
        self.id = schema_id if schema_id is not None else str(uuid.uuid4())
        self.name = name # e.g., "LayerPartitioning", "ModuleAffinity"
        # e.g., "nodes", "edges", or a custom tag from node/edge properties
        self.target_element_type = target_element_type
        self.partitions = partitions # list of sets, where each set contains IDs of elements in that partition
        self.metadata = metadata if metadata is not None else {} # e.g., description, parameters of the partitioning algorithm

    def __str__(self):
        return (f"PartitionSchema(id={self.id}, name='{self.name}', type='{self.target_element_type}', "
                f"num_partitions={len(self.partitions)})")

class NetworkGraph:
    """Represents a neural network (or other system) as a graph of nodes and edges."""
    def __init__(self, graph_id: str = None, properties: dict = None):
        self.id = graph_id if graph_id is not None else str(uuid.uuid4())
        self.nodes: dict[str, ArchitecturalNode] = {}
        self.edges: dict[str, ArchitecturalEdge] = {}
        # Adjacency list for quick lookup of connections:
        # node_id -> {'in': [edge_id, ...], 'out': [edge_id, ...]}
        self.adj: dict[str, dict[str, list[str]]] = {}
        self.properties = properties if properties is not None else {} # e.g., name, description
        self.partition_schemas: list[PartitionSchema] = [] # For Bell Superalgebra concepts

    def add_node(self, node: ArchitecturalNode):
        if node.id in self.nodes:
            raise ValueError(f"Node with ID {node.id} already exists.")
        self.nodes[node.id] = node
        self.adj[node.id] = {'in': [], 'out': []}

    def remove_node(self, node_id: str):
        if node_id not in self.nodes:
            # Or raise error: raise ValueError(f"Node with ID {node_id} not found.")
            return

        # Remove associated edges first
        # Use list(self.edges.keys()) to avoid issues with modifying dict during iteration
        for edge_id in list(self.edges.keys()):
            edge = self.edges[edge_id]
            if edge.source_node_id == node_id or edge.target_node_id == node_id:
                self.remove_edge(edge_id) # remove_edge will also update adjacency list

        if node_id in self.nodes: # Check if node still exists before deleting
            del self.nodes[node_id]
        if node_id in self.adj: # Check if adj entry still exists
            del self.adj[node_id]

    def add_edge(self, edge: ArchitecturalEdge):
        if edge.id in self.edges:
            raise ValueError(f"Edge with ID {edge.id} already exists.")
        if edge.source_node_id not in self.nodes:
            raise ValueError(f"Source node with ID {edge.source_node_id} not found for edge {edge.id}.")
        if edge.target_node_id not in self.nodes:
            raise ValueError(f"Target node with ID {edge.target_node_id} not found for edge {edge.id}.")

        self.edges[edge.id] = edge
        if edge.source_node_id in self.adj:
            self.adj[edge.source_node_id]['out'].append(edge.id)
        else: # Should not happen if nodes are added correctly
            self.adj[edge.source_node_id] = {'in': [], 'out': [edge.id]}

        if edge.target_node_id in self.adj:
            self.adj[edge.target_node_id]['in'].append(edge.id)
        else: # Should not happen
            self.adj[edge.target_node_id] = {'in': [edge.id], 'out': []}


    def remove_edge(self, edge_id: str):
        if edge_id not in self.edges:
            # Or raise error
            return

        edge = self.edges[edge_id]
        if edge.source_node_id in self.adj and edge_id in self.adj[edge.source_node_id]['out']:
            self.adj[edge.source_node_id]['out'].remove(edge.id)
        if edge.target_node_id in self.adj and edge_id in self.adj[edge.target_node_id]['in']:
            self.adj[edge.target_node_id]['in'].remove(edge.id)

        if edge_id in self.edges: # Check if edge still exists
            del self.edges[edge_id]

    def get_node(self, node_id: str) -> ArchitecturalNode | None:
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> ArchitecturalEdge | None:
        return self.edges.get(edge_id)

    def get_incoming_edges(self, node_id: str) -> list[ArchitecturalEdge]:
        if node_id not in self.adj: return [] # Changed from self.nodes to self.adj
        return [self.edges[edge_id] for edge_id in self.adj[node_id]['in'] if edge_id in self.edges]

    def get_outgoing_edges(self, node_id: str) -> list[ArchitecturalEdge]:
        if node_id not in self.adj: return [] # Changed from self.nodes to self.adj
        return [self.edges[edge_id] for edge_id in self.adj[node_id]['out'] if edge_id in self.edges]

    def add_partition_schema(self, schema: PartitionSchema):
        self.partition_schemas.append(schema)

    def __str__(self):
        return (f"NetworkGraph(id={self.id}, nodes={len(self.nodes)}, edges={len(self.edges)}, "
                f"partitions={len(self.partition_schemas)}, props={self.properties})")

# Note: The MutableEntity class defined earlier would now be expected to hold an instance of NetworkGraph
# in its self._data field when working with neural network architectures.
# MutationOperators will need to be designed to operate on this NetworkGraph structure.

# --- Superalgebraic Reconfiguration Operator (SRO) Stubs ---

class RepartitionGraphOperator(MutationOperator):
    """
    Performs a superalgebraic reconfiguration by changing how a NetworkGraph's
    elements (e.g., nodes) are partitioned, potentially leading to new
    PartitionSchema objects or modification of existing ones.
    This can then be used by other operators to guide structural changes.
    """
    def __init__(self, target_element_type: str = "nodes",
                 min_partitions: int = 1,
                 max_partitions: int = -1, # -1 means up to num_elements
                 new_schema_name_prefix: str = "Repartitioned"):
        super().__init__()
        self.target_element_type = target_element_type # "nodes", "edges", or based on a property
        self.min_partitions = min_partitions
        self.max_partitions = max_partitions
        self.new_schema_name_prefix = new_schema_name_prefix
        # Future params: partitioning strategy (e.g., random, community detection based, feature-based)

    def can_apply(self, entity: MutableEntity) -> bool:
        if not isinstance(entity.data, NetworkGraph):
            return False
        graph: NetworkGraph = entity.data
        if self.target_element_type == "nodes":
            return len(graph.nodes) > 0
        # TODO: Add checks for other element types if supported (e.g., edges)
        # For now, only "nodes" is a valid target_element_type for this stub
        return False

    def apply(self, entity: MutableEntity) -> None:
        """
        Modifies entity.data (a NetworkGraph) by:
        1. Selecting a set of target elements (currently only 'nodes').
        2. Generating a new partitioning of these elements.
        3. Storing this as a new PartitionSchema in the graph's `partition_schemas` list.
        """
        if not isinstance(entity.data, NetworkGraph): # Double check, though can_apply should catch
            print(f"Warning: {self.__class__.__name__} called on non-NetworkGraph entity.")
            return

        graph: NetworkGraph = entity.data

        # Explicitly check for node partitioning as it's the only one semi-implemented
        if self.target_element_type != "nodes" or not graph.nodes:
            # print(f"Info: {self.__class__.__name__} cannot apply to target_element_type '{self.target_element_type}' or graph has no nodes.")
            return

        print(f"SRO Stub: Applying {self.__class__.__name__} to graph {graph.id}")
        print(f"  Targeting element type: {self.target_element_type}")

        elements_to_partition_ids = list(graph.nodes.keys())
        num_elements = len(elements_to_partition_ids)

        if num_elements == 0:
            print("  No elements to partition.")
            return

        # Determine number of partitions (k)
        max_k = self.max_partitions if self.max_partitions > 0 and self.max_partitions <= num_elements else num_elements
        min_k = max(1, self.min_partitions)
        if min_k > max_k : min_k = max_k

        if max_k == 0:
            print("  Cannot create partitions with max_k=0.")
            return
        if min_k == 0 and max_k == 0 : # Should be caught by previous line but defensive
             print("  min_k and max_k are zero, no partitions to form.")
             return

        k = random.randint(min_k, max_k)
        if k == 0 and num_elements > 0 : # If somehow k becomes 0, but there are elements, default to 1 partition
            k = 1

        new_partitions_list: list[set[str]] = [set() for _ in range(k)]

        # Ensure all elements are assigned and partitions (if k <= num_elements) are non-empty.
        temp_elements = list(elements_to_partition_ids)
        random.shuffle(temp_elements)

        # Assign first k elements to k distinct partitions to ensure non-emptiness if possible
        for i in range(k):
            if temp_elements: # Check if there are elements left to assign
                new_partitions_list[i].add(temp_elements.pop())

        # Assign remaining elements randomly to any of the k partitions
        for remaining_element_id in temp_elements:
            if k > 0: # Ensure k is not zero before using randrange
                 target_partition_idx = random.randrange(k)
                 new_partitions_list[target_partition_idx].add(remaining_element_id)
            # else: if k is 0, this loop shouldn't run anyway due to temp_elements being empty.

        final_partitions = [p for p in new_partitions_list if p]

        if final_partitions:
            schema_name = f"{self.new_schema_name_prefix}_{self.target_element_type}_{str(uuid.uuid4())[:4]}"
            new_schema = PartitionSchema(name=schema_name,
                                       target_element_type=self.target_element_type,
                                       partitions=final_partitions)
            graph.add_partition_schema(new_schema)
            print(f"  Added new PartitionSchema: {new_schema.name} with {len(final_partitions)} partitions.")
            # print(f"  Partitions: {final_partitions}") # For debugging
        else:
            print("  No valid partitions were formed.")


class PartitionBasedRewireOperator(MutationOperator):
    """
    (Stub) Rewires a NetworkGraph based on an existing PartitionSchema.
    It changes connections *between* and *within* partitions according
    to a specified strategy (e.g., fully connect, sparsely connect, create bottlenecks).
    """
    def __init__(self, partition_schema_name_or_id: str = None,
                 inter_partition_strategy: str = "random_sparse",
                 intra_partition_strategy: str = "maintain",
                 connection_density: float = 0.1, # General density for random strategies
                 default_weight_sampler=lambda: random.uniform(-1,1) # How to sample new weights
                 ):
        super().__init__()
        self.partition_schema_name_or_id = partition_schema_name_or_id
        self.inter_partition_strategy = inter_partition_strategy
        self.intra_partition_strategy = intra_partition_strategy
        self.connection_density = connection_density
        self.default_weight_sampler = default_weight_sampler

    def can_apply(self, entity: MutableEntity) -> bool:
        if not isinstance(entity.data, NetworkGraph):
            return False
        graph: NetworkGraph = entity.data
        if not graph.partition_schemas:
            return False
        if self.partition_schema_name_or_id:
            return any(s.id == self.partition_schema_name_or_id or s.name == self.partition_schema_name_or_id
                       for s in graph.partition_schemas)
        return True

    def _get_nodes_from_partition(self, graph: NetworkGraph, partition_set: set[str], schema: PartitionSchema) -> list[ArchitecturalNode]:
        """Helper to get actual node objects from a set of IDs in a partition."""
        if schema.target_element_type == "nodes":
            return [graph.get_node(node_id) for node_id in partition_set if graph.get_node(node_id) is not None]
        # Extend for other types later if needed
        return []

    def apply(self, entity: MutableEntity) -> None:
        if not self.can_apply(entity):
            return

        graph: NetworkGraph = entity.data

        selected_schema: PartitionSchema | None = None
        if self.partition_schema_name_or_id:
            for s_candidate in graph.partition_schemas:
                if s_candidate.id == self.partition_schema_name_or_id or s_candidate.name == self.partition_schema_name_or_id:
                    selected_schema = s_candidate
                    break
        elif graph.partition_schemas:
            selected_schema = random.choice(graph.partition_schemas)

        if not selected_schema:
            print(f"SRO Stub ({self.__class__.__name__}): No suitable PartitionSchema found or provided for graph {graph.id}.")
            return

        print(f"SRO Stub: Applying {self.__class__.__name__} to graph {graph.id} using PartitionSchema: {selected_schema.name}")
        # print(f"  Inter-partition strategy: {self.inter_partition_strategy}")
        # print(f"  Intra-partition strategy: {self.intra_partition_strategy}")

        # --- Placeholder for actual rewiring logic ---
        # This would be very complex. For demonstration, let's imagine one simple strategy:
        # If inter_partition_strategy is "random_sparse", add a few random edges between partitions.

        if selected_schema.target_element_type != "nodes":
            print(f"  Skipping: Rewiring currently only implemented for node-based partitions. Schema is for '{selected_schema.target_element_type}'.")
            return

        # Example: Simple inter-partition random sparse connection
        if self.inter_partition_strategy == "random_sparse" and len(selected_schema.partitions) >= 2:
            num_new_edges = 0
            for i in range(len(selected_schema.partitions)):
                for j in range(i + 1, len(selected_schema.partitions)):
                    partition1_nodes = self._get_nodes_from_partition(graph, selected_schema.partitions[i], selected_schema)
                    partition2_nodes = self._get_nodes_from_partition(graph, selected_schema.partitions[j], selected_schema)

                    if not partition1_nodes or not partition2_nodes:
                        continue

                    # Add some random connections based on density
                    # For each pair of nodes (n1 in P1, n2 in P2), add edge with prob=connection_density
                    for n1 in partition1_nodes:
                        for n2 in partition2_nodes:
                            if random.random() < self.connection_density:
                                # Try adding edge from n1 to n2
                                new_edge_fwd = ArchitecturalEdge(source_node_id=n1.id, target_node_id=n2.id, weight=self.default_weight_sampler())
                                try:
                                    graph.add_edge(new_edge_fwd)
                                    num_new_edges +=1
                                except ValueError: # Edge might exist or other issue
                                    pass
                                # Try adding edge from n2 to n1 (for non-directed consideration or bi-directional)
                                if random.random() < self.connection_density: # Separate probability
                                    new_edge_bwd = ArchitecturalEdge(source_node_id=n2.id, target_node_id=n1.id, weight=self.default_weight_sampler())
                                    try:
                                        graph.add_edge(new_edge_bwd)
                                        num_new_edges +=1
                                    except ValueError:
                                        pass
            if num_new_edges > 0:
                print(f"  Added {num_new_edges} new inter-partition edges using 'random_sparse' strategy.")
            else:
                print(f"  No new inter-partition edges added with 'random_sparse' (density: {self.connection_density}).")

        # TODO: Implement other strategies for inter and intra partition rewiring.
        # This would involve removing existing edges and/or adding new ones based on the strategy.
        # For example:
        # 'full_connect_inter': ensure all nodes in P_i connect to all in P_j.
        # 'sparsify_intra': randomly remove some existing connections within P_i.
        # 'maintain_intra': do nothing to connections within P_i.
        print(f"SRO Stub ({self.__class__.__name__}): Placeholder rewiring logic executed.")
        # --- End Placeholder ---
        pass

# --- Fractal Scale Mutation Operator (FMO) Stubs ---
import copy # For deepcopying substructures

class HierarchicalNoiseInjectionFMO(MutationOperator):
    """
    (Stub) Applies noise or perturbations fractally across different scales
    of a NetworkGraph, based on the FractalMutationProtocol.
    """
    def __init__(self, base_noise_level: float = 0.05,
                 target_scales: list[str] = None, # e.g., ["micro", "meso", "macro", "meta"]
                 decay_factor: float = 0.5): # How noise level might change across scales
        super().__init__()
        self.base_noise_level = base_noise_level
        self.target_scales = target_scales if target_scales is not None else ["micro", "meso"]
        self.decay_factor = decay_factor # Example of a fractal-like parameter

    def can_apply(self, entity: MutableEntity) -> bool:
        return isinstance(entity.data, NetworkGraph)

    def apply(self, entity: MutableEntity) -> None:
        """
        Applies noise thematically across specified scales:
        - Micro (Level 3): Perturb weights/biases in ArchitecturalEdge/Node properties.
        - Meso (Level 2): E.g., randomly disable/enable a small percentage of nodes in a partition/layer,
                          or perturb properties of a layer if nodes represent layers.
        - Macro (Level 1): E.g., slightly alter connectivity density between major partitions,
                          or add/remove a small number of random edges globally.
        - Meta (Level 0): E.g., perturb a global property of the NetworkGraph itself,
                          or (outside this operator) signal the MutationEngine to adjust its own params.
        The actual noise application would be scaled by base_noise_level and decay_factor.
        """
        if not self.can_apply(entity):
            return

        graph: NetworkGraph = entity.data
        # Ensure random is available if not already imported globally in the file
        # import random # Already imported by SimplePol, and used by SROs

        print(f"FMO Stub: Applying {self.__class__.__name__} to graph {graph.id}")
        print(f"  Base noise level: {self.base_noise_level}, Target scales: {self.target_scales}")

        current_noise_level = self.base_noise_level

        if "meta" in self.target_scales:
            print(f"  META: Perturbing graph.properties (e.g., a global learning rate if stored there).")
            # Example: if 'learning_rate' in graph.properties:
            #    if isinstance(graph.properties['learning_rate'], (int, float)):
            #        graph.properties['learning_rate'] *= (1 + random.uniform(-current_noise_level, current_noise_level))
            #    else:
            #        print(f"    Warning: graph.properties['learning_rate'] is not a number.")
            # else:
            #    graph.properties['learning_rate_perturbation_factor'] = (1 + random.uniform(-current_noise_level, current_noise_level))
            #    print(f"    Conceptual: Added 'learning_rate_perturbation_factor' to graph properties.")
            current_noise_level *= self.decay_factor

        if "macro" in self.target_scales:
            print(f"  MACRO: Perturbing overall topology (e.g., add/remove few random global edges, alter partition connectivity rules).")
            # Example: Add one random edge if possible
            # if len(graph.nodes) >= 2:
            #     node_ids = list(graph.nodes.keys())
            #     source_id = random.choice(node_ids)
            #     target_id = random.choice(node_ids)
            #     if source_id != target_id:
            #         try:
            #             graph.add_edge(ArchitecturalEdge(source_node_id=source_id, target_node_id=target_id, weight=random.uniform(-0.1,0.1)))
            #             print(f"    Added a random edge between {source_id} and {target_id}")
            #         except ValueError: # Edge might exist
            #             pass
            current_noise_level *= self.decay_factor

        if "meso" in self.target_scales:
            print(f"  MESO: Perturbing layer/module properties (e.g., temporarily disable some nodes, alter shared activation functions).")
            # Example: For each node, small chance to add a 'disabled: True' to its properties.
            # This would require a PartitionSchema to define layers/modules.
            # num_nodes_affected = 0
            # for node_id in graph.nodes:
            #     if random.random() < current_noise_level * 0.1: # Smaller chance for disabling
            #         graph.nodes[node_id].properties['temp_disabled_by_meso_noise'] = True
            #         num_nodes_affected +=1
            # if num_nodes_affected > 0:
            #    print(f"    {num_nodes_affected} nodes conceptually marked as 'temp_disabled_by_meso_noise'.")
            current_noise_level *= self.decay_factor

        if "micro" in self.target_scales:
            print(f"  MICRO: Perturbing node/edge properties (e.g., weights, biases).")
            num_edges_perturbed = 0
            for edge in graph.edges.values():
                edge.weight += random.uniform(-current_noise_level, current_noise_level)
                num_edges_perturbed+=1

            num_nodes_perturbed = 0
            for node in graph.nodes.values():
                if 'bias' in node.properties and isinstance(node.properties['bias'], (int, float)):
                     node.properties['bias'] += random.uniform(-current_noise_level, current_noise_level)
                     num_nodes_perturbed+=1
            print(f"    Applied noise to {num_edges_perturbed} edge weights and {num_nodes_perturbed} node biases (if 'bias' property exists and is numeric).")

        print(f"FMO Stub ({self.__class__.__name__}): Placeholder noise injection logic executed.")
        pass


class SelfSimilarGrowthFMO(MutationOperator):
    """
    (Stub) Grows a NetworkGraph by adding new structures (nodes, edges) that are
    self-similar to existing structures or based on a fractal generation rule.
    """
    def __init__(self, growth_complexity: int = 1,
                 target_scales: list[str] = None,
                 min_motif_size: int = 2, max_motif_size: int = 5):
        super().__init__()
        self.growth_complexity = growth_complexity
        self.target_scales = target_scales if target_scales is not None else ["meso"] # Default to meso for adding to layers/modules
        self.min_motif_size = min_motif_size
        self.max_motif_size = max_motif_size


    def can_apply(self, entity: MutableEntity) -> bool:
        if not isinstance(entity.data, NetworkGraph):
            return False
        return len(entity.data.nodes) >= self.min_motif_size

    def _find_structural_motif(self, graph: NetworkGraph) -> tuple[list[ArchitecturalNode], list[ArchitecturalEdge]] | None:
        """
        (Placeholder) Finds a small, representative structural motif (subgraph) in the graph.
        Actual implementation would be complex (e.g., graph mining, pattern detection).
        """
        # print(f"    FMO Stub ({self.__class__.__name__}): Searching for structural motif...") # Verbose
        if len(graph.nodes) < self.min_motif_size:
            return None

        motif_nodes_map = {} # Store node_id -> node_object for quick lookup
        motif_edges_list = [] # Store edge objects

        # Simplistic: pick a few random connected nodes and their edges via BFS/DFS
        all_node_ids = list(graph.nodes.keys())
        if not all_node_ids: return None

        start_node_id = random.choice(all_node_ids)

        q = [start_node_id]
        visited_for_motif = {start_node_id} # Keep track of nodes added to this motif search

        while q and len(motif_nodes_map) < self.max_motif_size:
            curr_id = q.pop(0)
            node = graph.get_node(curr_id)
            if node:
                motif_nodes_map[node.id] = copy.deepcopy(node)

            # Add outgoing edges if target is also part of motif (or will be)
            for edge_obj in graph.get_outgoing_edges(curr_id):
                if len(motif_nodes_map) + len(q) >= self.max_motif_size and edge_obj.target_node_id not in visited_for_motif:
                    continue # Don't expand if we are about to exceed max_motif_size for nodes

                if edge_obj.target_node_id not in visited_for_motif:
                    q.append(edge_obj.target_node_id)
                    visited_for_motif.add(edge_obj.target_node_id)

                # If target is (or will be) in motif, add the edge
                if edge_obj.target_node_id in visited_for_motif:
                     motif_edges_list.append(copy.deepcopy(edge_obj))

            # Add incoming edges if source is already part of motif
            for edge_obj in graph.get_incoming_edges(curr_id):
                if edge_obj.source_node_id in motif_nodes_map:
                     # Avoid duplicates if already added via outgoing search from source
                     is_duplicate = any(e.id == edge_obj.id for e in motif_edges_list) or \
                                    any(e.source_node_id == edge_obj.source_node_id and \
                                        e.target_node_id == edge_obj.target_node_id for e in motif_edges_list)
                     if not is_duplicate:
                        motif_edges_list.append(copy.deepcopy(edge_obj))

        if len(motif_nodes_map) >= self.min_motif_size:
            # print(f"    Found motif with {len(motif_nodes_map)} nodes and {len(motif_edges_list)} edges.")
            return list(motif_nodes_map.values()), motif_edges_list
        # print("    No suitable motif found with current simplistic logic.")
        return None

    def _replicate_and_attach_motif(self, graph: NetworkGraph,
                                   motif_nodes: list[ArchitecturalNode],
                                   motif_edges: list[ArchitecturalEdge]):
        if not motif_nodes: return
        # print(f"    FMO Stub ({self.__class__.__name__}): Replicating and attaching motif...")

        id_map = {}
        new_nodes_added_this_replication = []

        for motif_node in motif_nodes:
            new_node_id = str(uuid.uuid4())
            id_map[motif_node.id] = new_node_id

            new_node = ArchitecturalNode(node_id=new_node_id,
                                         node_type=motif_node.node_type,
                                         properties=copy.deepcopy(motif_node.properties))
            try:
                graph.add_node(new_node)
                new_nodes_added_this_replication.append(new_node)
            except ValueError as e:
                print(f"      Warning: Could not add node during motif replication: {e}")
                # If node couldn't be added, we can't map its ID
                del id_map[motif_node.id]


        for motif_edge in motif_edges:
            original_source_id = motif_edge.source_node_id
            original_target_id = motif_edge.target_node_id

            if original_source_id in id_map and original_target_id in id_map:
                new_source_id = id_map[original_source_id]
                new_target_id = id_map[original_target_id]

                new_edge = ArchitecturalEdge(source_node_id=new_source_id,
                                             target_node_id=new_target_id,
                                             weight=motif_edge.weight,
                                             properties=copy.deepcopy(motif_edge.properties))
                try:
                    graph.add_edge(new_edge)
                except ValueError as e:
                    # This can happen if an edge with a default UUID4 ID conflicts, though unlikely.
                    # Or if somehow source/target nodes weren't added to graph despite being in id_map.
                    print(f"      Warning: Could not add edge {new_source_id}->{new_target_id} during motif replication: {e}")

        if new_nodes_added_this_replication:
            existing_graph_node_ids = [nid for nid in graph.nodes.keys() if nid not in id_map.values()]
            if existing_graph_node_ids:
                random_new_node_to_connect = random.choice(new_nodes_added_this_replication)
                random_existing_node_id_to_connect = random.choice(existing_graph_node_ids)

                # Decide direction of connection randomly
                if random.choice([True, False]):
                    source, target = random_existing_node_id_to_connect, random_new_node_to_connect.id
                else:
                    source, target = random_new_node_to_connect.id, random_existing_node_id_to_connect

                connection_edge = ArchitecturalEdge(source_node_id=source,
                                                    target_node_id=target,
                                                    weight=random.uniform(-0.1, 0.1))
                try:
                    graph.add_edge(connection_edge)
                    # print(f"      Attached new motif by connecting {source} to {target}")
                except ValueError as e:
                     print(f"      Warning: Could not attach motif by connecting {source} to {target}: {e}")
            # else:
                # print("      Motif replicated, but no other existing nodes to attach to.")
        # else:
            # print("    No new nodes were successfully added in this replication.")


    def apply(self, entity: MutableEntity) -> None:
        if not self.can_apply(entity):
            return

        graph: NetworkGraph = entity.data
        print(f"FMO Stub: Applying {self.__class__.__name__} to graph {graph.id}")
        # print(f"  Growth complexity: {self.growth_complexity}, Target scales: {self.target_scales}")

        num_successful_growths = 0
        for i in range(self.growth_complexity):
            # print(f"  Growth iteration {i+1}/{self.growth_complexity}") # Verbose
            motif_tuple = self._find_structural_motif(graph)
            if motif_tuple:
                motif_nodes, motif_edges = motif_tuple
                if motif_nodes: # Ensure motif actually has nodes
                    self._replicate_and_attach_motif(graph, motif_nodes, motif_edges)
                    num_successful_growths += 1
                else:
                    # print("    Motif found was empty.") # Verbose
                    break
            else:
                # print("    Could not find/define a motif to replicate for growth.") # Verbose
                break

        if num_successful_growths > 0:
            print(f"  Completed {num_successful_growths} growth operations by motif replication.")
        else:
            print(f"  No growth operations were successfully completed.")

        # print(f"FMO Stub ({self.__class__.__name__}): Placeholder growth logic executed.")
        pass
