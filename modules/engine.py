import re
import random
from abc import ABC, abstractmethod
import copy
import uuid
from typing import Any, List, Optional, Dict, Type, Union # Added Union

try:
    import torch
except ImportError:
    pass

# --- Imports for SmartMutationEngine ---
# Changed to absolute imports
try:
    from modules.reprogrammable_selector_nn import ReprogrammableSelectorNN
    # PYG_AVAILABLE will be sourced from reprogrammable_selector_nn if needed, or feature_extractors
except ImportError as e_rsel:
    print(f"Engine.py: Failed to import ReprogrammableSelectorNN: {e_rsel}")
    ReprogrammableSelectorNN = type(None)

# Independent PyGData import for engine.py's own potential needs (if any beyond type hints from other modules)
# and for SmartMutationEngine's _get_rl_state_representation type checks.
try:
    from torch_geometric.data import Data as PyGData
    PYG_AVAILABLE_IN_ENGINE = True # Local flag for engine's direct PyG capabilities
except ImportError:
    class PyGData: # type: ignore
        def __init__(self, x=None, edge_index=None, **kwargs): pass # Basic stub
    PYG_AVAILABLE_IN_ENGINE = False

try:
    from modules import feature_extractors
    # Access PYG_AVAILABLE from feature_extractors if needed, as it also handles PyG imports
    # For SmartMutationEngine, PYG_AVAILABLE from feature_extractors is more relevant for data conversion.
    DEFAULT_FEATURE_CONFIG_FROM_MODULE = feature_extractors.DEFAULT_FEATURE_CONFIG
except ImportError as e_fe:
    print(f"Engine.py: Failed to import feature_extractors: {e_fe}")
    feature_extractors = None
    DEFAULT_FEATURE_CONFIG_FROM_MODULE = {}

try:
    from modules.rl_utils import ExperienceReplayBuffer
except ImportError as e_rl:
    print(f"Engine.py: Failed to import ExperienceReplayBuffer: {e_rl}")
    ExperienceReplayBuffer = type(None)


class SimplePol:
    def __init__(self, path):
        self.stack_register = None; self.add_sub_register = None; self.border_pos = None
        self.all_registers_lst = ['rdx','rax','rcx','rsi','r11','rdi','rbx','r8','r10','r9','r12','r13','r14','r15','rbp','rsp']
        self.path=path; self.content=list(); self.mov_xor_jmp_je_jk_lst=list(); self.add_sub_lst=list()
        self.add_sub_lst_im=list(); self.add_sub_lst_reg=list(); self.register_lst=list(); self.mul_lst=list(); self.cmp_lst=list()
    def reader(self):
        with open(self.path) as f:
            for line in f: self.content.append(line.strip().split('\n'))
    def parser(self,word,lst):
        for i in range(len(self.content)):
            if re.match(word, self.content[i][0]): lst.append(self.content[i])
    def parser_register(self):
        for i in range(len(self.content)):
            length=len(self.content[i][0]); register=str()
            while length!=0:
                if self.content[i][0][length-1]==',':
                    while self.content[i][0][length-1]!=' ' and self.content[i][0][length-1]!='\t':
                        length-=1
                        if self.content[i][0][length-1].isalnum(): register+=self.content[i][0][length-1]
                        else: break
                    break
                length-=1
            if register: register=register[::-1]; self.register_lst.append(register)
        self.register_lst=list(set(self.register_lst)); trash=list()
        for i in range(len(self.register_lst)):
            if len(self.register_lst[i])>3 or len(self.register_lst[i])<2: trash.append(self.register_lst[i])
        for i in range(len(trash)): self.register_lst.remove(trash[i])
    def parser_commands(self): self.parser(r"mov|jmp|je|jk|xor",self.mov_xor_jmp_je_jk_lst)
    def parser_add_sub(self): self.parser(r"add|sub",self.add_sub_lst)
    def parser_cmp(self): self.parser(r"cmp",self.cmp_lst)
    def parser_mul(self):
        self.parser(r"mul",self.mul_lst)
        for i in range(len(self.mul_lst)):
            index=self.content.index(self.mul_lst[i])
            self.mul_lst.insert(i,self.content[index-1]); self.mul_lst.insert(i,self.content[index-2])
    def set_border(self):
        for i in range(len(self.content)):
            if self.content[i] in self.mov_xor_jmp_je_jk_lst or self.content[i] in self.cmp_lst or \
               self.content[i] in self.mul_lst or self.content[i] in self.add_sub_lst:
                self.border_pos=self.content[i]; break
    def classification_add_sub(self):
        for i in range(len(self.add_sub_lst)):
            if re.search(r', ?[0-9]+',self.add_sub_lst[i][0]): self.add_sub_lst_im.append(self.add_sub_lst[i])
            else: self.add_sub_lst_reg.append(self.add_sub_lst[i])
    @staticmethod
    def number_division(number): return random.randrange(number)
    @staticmethod
    def line_maker(line,number,reverse=False):
        new_line=str(); i=0
        while line[i]!=',': new_line+=line[i]; i+=1
        new_line+=','; new_line+=str(number)
        if reverse and new_line[0]=='a': new_line=list(new_line); new_line[0:3]='s','u','b'; new_line=''.join(new_line)
        elif reverse and new_line[0]=='s': new_line=list(new_line); new_line[0:3]='a','d','d'; new_line=''.join(new_line)
        return new_line
    def nope_adder(self,element): self.content.insert(self.content.index(element),['nop'])
    def division_adder_im(self,element):
        length=len(element[0]); number=str(); exact_number=str()
        while element[0][length-1]!=',': number+=element[0][length-1]; length-=1
        number=number.split()
        for i in range(len(number)):
            if number[i].isdecimal(): exact_number=number[i]; break
        number=int(exact_number[::-1]); div=self.number_division(number); choice=random.choice([1,2,3])
        if choice==1: self.division_adder_im_2(element,div,number)
        elif choice==2: self.division_adder_sub(element,div,number)
        else: self.division_adder_im_3(element,div,number)
    def division_adder_im_2(self,element,div,number):
        new_line=self.line_maker(element[0],div); self.content.insert(self.content.index(element),[new_line])
        self.content[self.content.index(element)]=[self.line_maker(element[0],number-div)]
    def division_adder_im_3(self,element,div,number):
        new_div=self.number_division(div) if div!=0 else 0
        self.content.insert(self.content.index(element),[self.line_maker(element[0],new_div)])
        self.content.insert(self.content.index(element),[self.line_maker(element[0],div-new_div)])
        self.content[self.content.index(element)]=[self.line_maker(element[0],number-div)]
    def division_adder_sub(self,element,div,number):
        new_div=random.randint(number+1,number+div+1)
        self.content.insert(self.content.index(element),[self.line_maker(element[0],new_div)])
        self.content[self.content.index(element)]=[self.line_maker(element[0],new_div-number,True)]
    def add_sub_adder(self,element):
        reg=self.add_sub_register
        if not reg:
            for r_cand in self.all_registers_lst:
                if r_cand not in self.register_lst and r_cand!=self.stack_register: reg=r_cand;self.add_sub_register=r_cand;break
        if reg: index=self.content.index(element);num=self.number_division(10);self.content.insert(index,[f'sub {reg}, {num}']);self.content.insert(index,[f'add {reg}, {num}'])
    def stack_adder(self,element):
        reg=self.stack_register
        if not reg:
            for r_cand in self.all_registers_lst:
                if r_cand not in self.register_lst and r_cand!=self.add_sub_register: reg=r_cand;self.stack_register=r_cand;break
        if reg: index=self.content.index(element);self.content.insert(index,[f'pop {reg}']);self.content.insert(index,[f'push {reg}'])
    def stack_nop_adder(self,element):
        reg=self.stack_register
        if not reg:
            for r_cand in self.all_registers_lst:
                if r_cand not in self.register_lst and r_cand!=self.add_sub_register: reg=r_cand;self.stack_register=r_cand;break
        if reg: index=self.content.index(element);self.content.insert(index,[f'pop {reg}']);self.content.insert(index,['nop']);self.content.insert(index,[f'push {reg}'])
    def swap_of_reg(self,element):
        if len(element[0])>16:return;l_reg,f_reg="", "";length=len(element[0])-1
        while element[0][length]!=',':l_reg+=element[0][length];length-=1
        length-=1
        while length>=0 and element[0][length] not in [' ','\t']:f_reg+=element[0][length];length-=1
        l_reg=l_reg.strip()[::-1]; f_reg=f_reg.strip()[::-1]
        if f_reg.isdecimal() or l_reg.isdecimal() or not f_reg or not l_reg:return
        self.content[self.content.index(element)]=[f"cmp {l_reg}, {f_reg}"]
    def commands_transformer(self):
        for i in range(len(self.mov_xor_jmp_je_jk_lst)): # This might have issues if list is modified during iteration; use copy if so.
            item_to_transform = self.mov_xor_jmp_je_jk_lst[i]
            if item_to_transform not in self.content: continue # If already modified/removed
            idx=self.content.index(item_to_transform); choice=random.choice([1,2,3,4])
            if choice==1:self.nope_adder(self.content[idx])
            elif choice==2:self.add_sub_adder(self.content[idx])
            elif choice==3:self.stack_nop_adder(self.content[idx])
            else:self.stack_adder(self.content[idx])
    def add_sub_transformer(self):
        self.set_border()
        for item in list(self.add_sub_lst_im): # Iterate over a copy
            if item not in self.content:continue;idx=self.content.index(item);choice=random.choice([1,2,3])
            if choice==1:self.nope_adder(self.content[idx])
            elif choice==2:self.stack_adder(self.content[idx])
            else:self.division_adder_im(self.content[idx])
        for item in list(self.add_sub_lst_reg): # Iterate over a copy
            if item not in self.content:continue;idx=self.content.index(item);choice=random.choice([1,2,3])
            if choice==1:self.nope_adder(self.content[idx])
            elif choice==2:self.stack_adder(self.content[idx])
            else:self.add_sub_adder(self.content[idx])
    def mul_transform(self): pass
    def cmp_transform(self):
        for item in list(self.cmp_lst):
            if item not in self.content:continue;idx=self.content.index(item);choice=random.choice([1,2,3])
            if choice==1:self.nope_adder(self.content[idx])
            elif choice==2:self.stack_adder(self.content[idx])
            else:self.swap_of_reg(self.content[idx])
    def polymorph(self):
        self.reader();self.parser_add_sub();self.parser_mul();self.parser_commands();self.parser_register();self.parser_cmp()
        self.set_border();self.classification_add_sub();self.commands_transformer();self.add_sub_transformer();self.cmp_transform()
        final_content="\n".join(line[0] for line in self.content)
        with open(f"{self.path[:-4]}_pol.asm",'w') as f:f.write(final_content)

# --- Core IME Framework ---
class MutableEntity:
    def __init__(self, data: Any): self._data: Any = data; self.properties: Dict[str, Any] = {}
    @property
    def data(self) -> Any: return self._data
    @data.setter
    def data(self, new_data: Any) -> None: self._data = new_data
    def __str__(self) -> str:
        ds=str(self._data); return f"MutableEntity(type={type(self._data).__name__}, data={ds[:100]+'...' if len(ds)>100 else ds})"

class MutationOperator(ABC):
    @abstractmethod
    def apply(self, entity: MutableEntity) -> None: pass
    def can_apply(self, entity: MutableEntity) -> bool: return True

class IntegerPerturbationOperator(MutationOperator):
    def __init__(self, perturbation_range: tuple[int,int]=(-5,5)):
        super().__init__()
        if not (isinstance(pr:=perturbation_range,tuple) and len(pr)==2 and all(isinstance(x,int) for x in pr) and pr[0]<=pr[1]):
            raise ValueError("perturbation_range must be tuple of two ints (min,max).")
        self.min_p,self.max_p = pr
    def can_apply(self, entity:MutableEntity)->bool: return isinstance(entity.data,int)
    def apply(self, entity:MutableEntity)->None:
        if not self.can_apply(entity):return
        p=random.randint(self.min_p,self.max_p)
        while p==0 and (self.min_p!=0 or self.max_p!=0): p=random.randint(self.min_p,self.max_p)
        if isinstance(entity.data,(int,float)): entity.data+=p

class StringReplaceOperator(MutationOperator):
    def can_apply(self, entity: MutableEntity) -> bool: return isinstance(entity.data, str) and len(entity.data) > 0
    def apply(self, entity: MutableEntity) -> None:
        if not self.can_apply(entity): return
        s_list = list(entity.data)
        if not s_list: return
        idx = random.randrange(len(s_list))
        original_char = s_list[idx]
        new_char = chr(random.randint(32, 126)) # ASCII printable
        counter = 0
        while new_char == original_char and counter < 100:
            if len(set(s_list)) == 1: break
            new_char = chr(random.randint(32, 126))
            counter += 1
        s_list[idx] = new_char
        entity.data = "".join(s_list)

class ListElementSwapOperator(MutationOperator):
    def can_apply(self,entity:MutableEntity)->bool:return isinstance(entity.data,list) and len(entity.data)>=2
    def apply(self,entity:MutableEntity)->None:
        if not self.can_apply(entity):
            return
        lst = entity.data # Ensure lst is assigned after the guard
        idx1, idx2 = random.sample(range(len(lst)), 2)
        lst[idx1], lst[idx2] = lst[idx2], lst[idx1]

class MutationEngine: # Basic engine
    def __init__(self,operators:List[MutationOperator],mutation_probability:float=1.0):
        if not operators or not all(isinstance(op,MutationOperator) for op in operators):raise ValueError("Need list of MutationOperator.")
        if not (0.0<=mutation_probability<=1.0):raise ValueError("Prob must be [0,1].")
        self.operators=operators;self.mutation_probability=mutation_probability
    def evaluate_entity(self,entity:MutableEntity)->float:return 0.0
    def mutate_once(self,entity:MutableEntity)->bool:
        if random.random()>=self.mutation_probability:return False
        app_ops=[op for op in self.operators if op.can_apply(entity)]
        if not app_ops:return False
        random.choice(app_ops).apply(entity);return True
    def run(self,entity:MutableEntity,num_mutations:int)->int:
        if num_mutations<0:raise ValueError("Num mutations non-negative.");succ=0
        for _ in range(num_mutations):
            if self.mutate_once(entity):succ+=1
        return succ

# Import graph schema classes from the new module
from modules.graph_schema import ArchitecturalNode, ArchitecturalEdge, PartitionSchema, NetworkGraph
# Ensure ast is imported if SmartMutationEngine._extract_features uses it for type checking
import ast


# --- SRO & FMO Stubs (Further Simplified) ---
# These operators might use NetworkGraph, so they are defined after its import.
class RepartitionGraphOperator(MutationOperator):
    def __init__(self, **kwargs): super().__init__()
    def can_apply(self, entity:MutableEntity)->bool:return isinstance(entity.data,NetworkGraph) and len(entity.data.nodes)>0
    def apply(self, entity:MutableEntity): pass # print(f"SRO Repartition Applied to {entity.data.id if hasattr(entity.data,'id') else 'N/A'}")
class PartitionBasedRewireOperator(MutationOperator):
    def __init__(self, **kwargs): super().__init__()
    def can_apply(self, entity:MutableEntity)->bool:return isinstance(entity.data,NetworkGraph) and entity.data.partition_schemas
    def apply(self, entity:MutableEntity): pass # print(f"SRO Rewire Applied to {entity.data.id if hasattr(entity.data,'id') else 'N/A'}")
class HierarchicalNoiseInjectionFMO(MutationOperator):
    def __init__(self, **kwargs): super().__init__()
    def can_apply(self, entity:MutableEntity)->bool:return isinstance(entity.data,NetworkGraph)
    def apply(self, entity:MutableEntity): pass # print(f"FMO Noise Applied to {entity.data.id if hasattr(entity.data,'id') else 'N/A'}")
class SelfSimilarGrowthFMO(MutationOperator):
    def __init__(self, **kwargs): super().__init__()
    def can_apply(self, entity:MutableEntity)->bool:return isinstance(entity.data,NetworkGraph) and len(entity.data.nodes)>0
    def apply(self, entity:MutableEntity): pass # print(f"FMO Growth Applied to {entity.data.id if hasattr(entity.data,'id') else 'N/A'}")

# --- Smart Mutation Engine ---
class SmartMutationEngine:
    def __init__(self, operators:List[MutationOperator], nn_selector:ReprogrammableSelectorNN,
                 feature_config:Optional[Dict[str,Any]]=None, replay_buffer_capacity:int=10000,
                 rl_batch_size:int=32, train_frequency:int=4, fallback_to_random:bool=True,
                 mutation_probability:float=1.0, max_steps_per_episode:int=200):
        if not operators: raise ValueError("Need operators.")
        if ReprogrammableSelectorNN is not type(None) and not isinstance(nn_selector,ReprogrammableSelectorNN): raise ValueError("Bad nn_selector.")
        if feature_extractors is None: raise ImportError("feature_extractors module failed for SmartMutationEngine.")
        self.feature_config = feature_config if feature_config is not None else DEFAULT_FEATURE_CONFIG_FROM_MODULE # Use the imported one
        if not (self.feature_config and "entity_type_dispatch" in self.feature_config and \
                "extractor_functions_map" in self.feature_config and "max_vector_size" in self.feature_config):
            raise ValueError("Invalid feature_config.")
        self.operators=operators
        self.operator_to_index={op:i for i,op in enumerate(operators)}
        self.index_to_operator={i:op for i,op in enumerate(operators)}
        self.nn_selector=nn_selector; self.fallback_to_random=fallback_to_random
        if not (0.0<=mutation_probability<=1.0): raise ValueError("Prob must be [0,1].")
        self.mutation_probability=mutation_probability
        if ReprogrammableSelectorNN is not type(None) and hasattr(self.nn_selector, 'get_config'):
            nn_out_size = self.nn_selector.get_config().get("output_size")
            if nn_out_size != len(self.operators): print(f"Warn: NN output ({nn_out_size}) != op count ({len(self.operators)}).")
        if ExperienceReplayBuffer is type(None): raise ImportError("ExperienceReplayBuffer missing.")
        self.replay_buffer=ExperienceReplayBuffer(replay_buffer_capacity)
        self.rl_batch_size=rl_batch_size; self.train_frequency=train_frequency
        self.mutation_steps_count=0; self.current_episode_step_count=0; self.max_steps_per_episode=max_steps_per_episode

    def _get_rl_state_representation(self, entity: MutableEntity) -> Optional[Any]:
        # _extract_features is now expected to return:
        # - PyGData object if entity is NetworkGraph and GNN features are extracted.
        # - List[float] if entity is for FFN features.
        # - None on failure.
        extracted_output = self._extract_features(entity)

        if extracted_output is None:
            return None

        is_gnn_selector = (ReprogrammableSelectorNN is not type(None) and
                           hasattr(self.nn_selector, 'model_type') and
                           self.nn_selector.model_type == "gnn")

        # Determine PyGData type robustly (it might be a stub if torch_geometric is not installed)
        pyg_data_type_actual = None
        if feature_extractors is not None and hasattr(feature_extractors, 'PyGData_import') and feature_extractors.PyGData_import is not None:
            pyg_data_type_actual = feature_extractors.PyGData_import
        elif PYG_AVAILABLE_IN_ENGINE : # Fallback to engine's PyGData if feature_extractors didn't provide it
             pyg_data_type_actual = PyGData

        is_extracted_pyg = pyg_data_type_actual is not None and isinstance(extracted_output, pyg_data_type_actual)

        if is_gnn_selector:
            if is_extracted_pyg:
                return extracted_output  # Expected: PyGData for GNN
            else:
                print(f"Warning: GNN selector received non-PyGData features (type: {type(extracted_output)}). Ensure NetworkGraph entities use a GNN-compatible feature extractor.")
                return None
        else:  # FFN selector
            if isinstance(extracted_output, list):
                if torch is not None:
                    return torch.tensor([extracted_output], dtype=torch.float32) # Expected: List[float] for FFN
                return extracted_output # No torch, return list
            elif is_extracted_pyg:
                 print(f"Warning: FFN selector received PyGData features: {type(extracted_output)}. This is not directly usable by an FFN. Ensure non-NetworkGraph entities use an FFN-compatible feature extractor.")
                 return None
            else:
                print(f"Warning: FFN selector received unexpected feature type: {type(extracted_output)}")
                return None

    def _extract_features(self, entity:MutableEntity)->Optional[Union[List[float], PyGData]]:
        # Type hint for PyGData here refers to the one defined/imported in this file (engine.py)
        et_name=type(entity.data).__name__

        # Determine entity type string for dispatch
        # Check for ast.Module first, as its __name__ is 'Module'
        # We need a way to distinguish ast.Module from a user-defined 'Module' class potentially
        # For now, assume 'Module' in config refers to ast.Module if data is an AST node.
        # A more robust way would be to register types rather than rely on __name__.
        if 'ast' in globals() and isinstance(entity.data, ast.AST): # Check if ast was imported
             if isinstance(entity.data, ast.Module): et_name = "Module_AST" # Or just "Module" if config uses that
        elif NetworkGraph is not type(None) and isinstance(entity.data, NetworkGraph):
            et_name = "NetworkGraph"
        # else et_name is already type(entity.data).__name__ for list, int, str etc.

        disp_info=self.feature_config["entity_type_dispatch"].get(et_name)
        if not disp_info: return None
        ext_fn_name=disp_info["extractor_function_name"]
        if feature_extractors is None: return None
        ext_fn = self.feature_config["extractor_functions_map"].get(ext_fn_name) # CORRECTED ACCESS
        if not ext_fn: return None
        try:raw_f=ext_fn(entity.data)
        except Exception as e:print(f"Err in extract {ext_fn_name}:{e}");return None
        if not isinstance(raw_f,list):return None
        m_size=self.feature_config["max_vector_size"]
        return (raw_f+[0.0]*(m_size-len(raw_f))) if len(raw_f)<m_size else raw_f[:m_size]

    def select_operator_intelligently(self, entity:MutableEntity, current_state_repr:Any)->Optional[MutationOperator]:
        sel_op=None
        if current_state_repr is not None and ReprogrammableSelectorNN is not type(None) and hasattr(self.nn_selector, 'predict'):
            try:
                scores_t=self.nn_selector.predict(current_state_repr);op_scores=scores_t[0].tolist()
                if len(op_scores)!=len(self.operators):print(f"Err: NN out({len(op_scores)})!=op_cnt({len(self.operators)}).")
                else:
                    ops_s=[{'op':op,'score':op_scores[i],'idx':i} for i,op in enumerate(self.operators)]
                    app_ops=[item for item in ops_s if item['op'].can_apply(entity)]
                    if app_ops:app_ops.sort(key=lambda x:x['score'],reverse=True);sel_op=app_ops[0]['op']
            except Exception as e:print(f"Err NN predict/select:{e}")
        if sel_op is None and self.fallback_to_random:
            fall_ops=[op for op in self.operators if op.can_apply(entity)]
            if fall_ops:sel_op=random.choice(fall_ops)
        return sel_op

    def evaluate_entity(self, entity:MutableEntity)->float:
        data = entity.data
        if isinstance(data, list):
            if not data: # Empty list
                return 1.0

            numeric_data = [x for x in data if isinstance(x, (int, float))] # Define numeric_data here

            if len(numeric_data) != len(data): # Check if all items were numeric
                return 0.0  # Penalize mixed/non-numeric for sorting task

            if len(numeric_data) <= 1: # Single numeric element or empty after filtering
                return 1.0 # Considered sorted

            # Now numeric_data is guaranteed to have at least 2 elements
            inversions = 0
            for i in range(len(numeric_data) - 1):
                if numeric_data[i] > numeric_data[i+1]:
                    inversions += 1
            return 1.0 / (1.0 + float(inversions))
        return 0.0

    def calculate_reward(self,old_s:float,new_s:float)->float:
        rew = new_s-old_s
        if new_s==1.0 and old_s < 1.0: rew+=1.0
        return rew

    def mutate_once(self, entity:MutableEntity)->bool:
        if random.random()>=self.mutation_probability:return False
        self.current_episode_step_count+=1;self.mutation_steps_count+=1
        state_r=self._get_rl_state_representation(entity);score_curr=self.evaluate_entity(entity)
        op_to_apply=self.select_operator_intelligently(entity,state_r)
        if op_to_apply:
            act_idx = self.operator_to_index.get(op_to_apply)
            if act_idx is None: return False
            state_for_buffer=copy.deepcopy(state_r) if not isinstance(state_r,torch.Tensor if 'torch' in globals() and torch is not None else type(None)) else state_r.clone() if 'torch' in globals() and torch is not None and isinstance(state_r, torch.Tensor) else state_r
            op_to_apply.apply(entity);score_new=self.evaluate_entity(entity)
            rew=self.calculate_reward(score_curr,score_new)
            next_s_r=self._get_rl_state_representation(entity)
            done=(score_new==1.0)or(self.current_episode_step_count>=self.max_steps_per_episode)
            if state_for_buffer is not None and ExperienceReplayBuffer is not type(None) and self.replay_buffer is not None:
                self.replay_buffer.push(state_for_buffer,act_idx,rew,next_s_r,done)
            if self.mutation_steps_count%self.train_frequency==0 and ExperienceReplayBuffer is not type(None) and \
               self.replay_buffer is not None and len(self.replay_buffer)>=self.rl_batch_size and \
               ReprogrammableSelectorNN is not type(None) and hasattr(self.nn_selector,'train_on_batch'):
                exps=self.replay_buffer.sample(self.rl_batch_size)
                if exps:self.nn_selector.train_on_batch(exps)
            if done:self.current_episode_step_count=0
            return True
        return False

    def run(self, entity:MutableEntity, num_muts:int)->int:
        if num_muts<0:raise ValueError("Num muts non-negative.");succ_muts=0
        for _ in range(num_muts):
            if self.mutate_once(entity):succ_muts+=1
        return succ_muts
