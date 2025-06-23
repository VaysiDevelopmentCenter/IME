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
