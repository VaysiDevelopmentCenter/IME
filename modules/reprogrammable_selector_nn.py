# modules/reprogrammable_selector_nn.py
import sys # For sys.path modification
import os  # For sys.path modification

# Ensure the project root is in sys.path for absolute imports when running script directly
# This needs to be at the very top before any 'from modules.xyz' imports are attempted.
PACKAGE_PARENT_DIR_FROM_SCRIPT = '..'
# Get the directory of the current script
SCRIPT_REAL_PATH = os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
SCRIPT_PARENT_DIR = os.path.dirname(SCRIPT_REAL_PATH)
PROJECT_ROOT_PATH = os.path.normpath(os.path.join(SCRIPT_PARENT_DIR, PACKAGE_PARENT_DIR_FROM_SCRIPT))
if PROJECT_ROOT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_PATH)
# print(f"DEBUG: sys.path in reprogrammable_selector_nn.py (top): {sys.path}")


import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
import json # For saving/loading nn_config
import os # For save/load paths
from typing import List, Union, Optional, Dict, Any # For type hints
from collections import namedtuple # For Experience in __main__

# --- PyTorch Geometric Imports & Fallbacks ---
try:
    from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool # type: ignore
    from torch_geometric.data import Data as PyGData, Batch as PyGBatch # type: ignore
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    class GCNConv(nn.Module): # type: ignore
        def __init__(self, *args, **kwargs): super().__init__(); raise NotImplementedError("GCNConv stub: PyG not available.")
    def global_mean_pool(x, batch): raise NotImplementedError("global_mean_pool stub: PyG not available.") # type: ignore
    def global_add_pool(x, batch): raise NotImplementedError("global_add_pool stub: PyG not available.") # type: ignore
    def global_max_pool(x, batch): raise NotImplementedError("global_max_pool stub: PyG not available.") # type: ignore
    class PyGData: # type: ignore
        def __init__(self, x=None, edge_index=None, batch=None, **kwargs): self.x=x;self.edge_index=edge_index;self.batch=batch; [setattr(self,k,v) for k,v in kwargs.items()] # type: ignore
    class PyGBatch: # type: ignore
        @staticmethod
        def from_data_list(data_list): raise NotImplementedError("PyGBatch stub: PyG not available.")

from typing import TYPE_CHECKING # For type hinting

# --- IME Component Imports & Fallbacks ---
# NetworkGraph is now imported from graph_schema, breaking the cycle with engine.py
try:
    from modules.graph_schema import NetworkGraph
except ImportError as e_graph_schema:
    print(f"Warning: Could not import NetworkGraph from modules.graph_schema. Error: {e_graph_schema}")
    NetworkGraph = type(None) # type: ignore


try:
    # Import the new GNN feature extractor and its dimension from feature_extractors
    # Changed to absolute import assuming 'modules' package is in sys.path
    from modules.feature_extractors import extract_network_graph_pyg_data, GNN_NODE_FEATURE_DIM as EXTRACTOR_GNN_NODE_FEATURE_DIM
    # Alias for clarity if used extensively, or use directly
    network_graph_to_pyg_data_via_extractor = extract_network_graph_pyg_data
    IMPORTED_GNN_NODE_FEATURE_DIM = EXTRACTOR_GNN_NODE_FEATURE_DIM
except ImportError as e_feature_extractors:
    def network_graph_to_pyg_data_dummy(graph_unused: Any) -> None: return None # type: ignore
    network_graph_to_pyg_data_via_extractor = network_graph_to_pyg_data_dummy # type: ignore
    IMPORTED_GNN_NODE_FEATURE_DIM = 1 # Fallback if feature_extractors is not available or GNN_NODE_FEATURE_DIM is not set yet
    print(f"Warning: Could not import GNN feature extractor utilities from modules.feature_extractors for ReprogrammableSelectorNN. Error: {e_feature_extractors}")


class ReprogrammableSelectorNN(nn.Module):
    def __init__(self, nn_config: Dict[str, Any], learning_rate: float = 0.001):
        super().__init__()
        if not nn_config or not isinstance(nn_config, dict): raise ValueError("nn_config dict needed.")
        self.nn_config = copy.deepcopy(nn_config)
        self.model_type = self.nn_config.get("model_type", "ffn").lower()
        self.learning_rate = learning_rate

        if self.model_type == "gnn" and not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric required for GNN model_type but not found.")

        # For GNN, ensure node_feature_size is set, using the imported dimension if not specified in config
        if self.model_type == "gnn":
            if "node_feature_size" not in self.nn_config:
                if IMPORTED_GNN_NODE_FEATURE_DIM is None: # Check if it was successfully imported and set
                    raise ValueError("IMPORTED_GNN_NODE_FEATURE_DIM from feature_extractors is None. Cannot configure GNN.")
                if IMPORTED_GNN_NODE_FEATURE_DIM == 1 and NetworkGraph is type(None): # Fallback check
                     print("Warning: Using fallback GNN_NODE_FEATURE_DIM=1. Ensure feature_extractors.GNN_NODE_FEATURE_DIM is correctly set.")
                self.nn_config["node_feature_size"] = IMPORTED_GNN_NODE_FEATURE_DIM

        self.model_internal = self._build_model()
        self.optimizer = optim.Adam(self.model_internal.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _get_activation_function(self, name: Optional[str]) -> Optional[nn.Module]:
        if name is None or name.lower()=="none": return None
        act_map = {"relu":nn.ReLU,"sigmoid":nn.Sigmoid,"tanh":nn.Tanh,"softmax":lambda: nn.Softmax(dim=-1)}
        if (fn := act_map.get(name.lower())): return fn()
        raise ValueError(f"Unsupported activation: {name}")

    def _build_ffn_model(self) -> nn.Sequential:
        cfg=self.nn_config; layers_cfg=cfg.get("layers",[]); ins=cfg.get("input_size");outs=cfg.get("output_size");out_act=cfg.get("output_activation","none")
        if ins is None or outs is None: raise ValueError("FFN config: input/output_size missing.")
        ml:List[nn.Module]=[]; cs=ins
        for i,lc in enumerate(layers_cfg):
            lt=lc.get("type","").lower();sz=lc.get("size");act=lc.get("activation")
            if lt=="linear":
                if sz is None or not isinstance(sz,int) or sz<=0: raise ValueError(f"FFN Linear {i} needs +int 'size'.")
                ml.append(nn.Linear(cs,sz));cs=sz
                if (af:=self._get_activation_function(act)):ml.append(af)
            elif lt=="dropout":
                r=lc.get("rate",0.5);
                if not (isinstance(r,float)and 0.0<=r<1.0):raise ValueError(f"FFN Dropout {i} rate float [0,1).")
                ml.append(nn.Dropout(r))
            else:raise ValueError(f"Unsupported FFN layer: {lt}")
        ml.append(nn.Linear(cs,outs))
        if(oaf:=self._get_activation_function(out_act)):ml.append(oaf)
        return nn.Sequential(*ml)

    def _build_gnn_model(self) -> nn.Module:
        if not PYG_AVAILABLE:raise ImportError("PyG required for GNN.")
        cfg=self.nn_config;nfs=cfg.get("node_feature_size");glc=cfg.get("gnn_layers",[]);gpm=cfg.get("global_pooling","mean").lower()
        pglc=cfg.get("post_gnn_mlp_layers",[]);os=cfg.get("output_size");oan=cfg.get("output_activation","none")
        if nfs is None or os is None:raise ValueError("GNN config: node_feature_size/output_size missing.")

        class GNNSubModel(nn.Module):
            def __init__(self_sub, parent_nn_instance):
                super().__init__(); self_sub.p = parent_nn_instance; self_sub.convs = nn.ModuleList(); cc = nfs
                for i,lc in enumerate(glc):
                    lt=lc.get("type","").lower();oc=lc.get("out_channels")
                    if oc is None: raise ValueError(f"GNN layer {i} needs 'out_channels'.")
                    if lt=="gcnconv": self_sub.convs.append(GCNConv(cc,oc)) # Fixed: s_sub to self_sub
                    else: raise ValueError(f"Unsupported GNN layer: {lt}")
                    cc=oc
                    if(af := self_sub.p._get_activation_function(lc.get("activation"))): self_sub.convs.append(af) # Fixed: s_sub to self_sub

                pool_map={"mean":global_mean_pool,"add":global_add_pool,"max":global_max_pool}
                pool_fn = pool_map.get(gpm) # Corrected assignment
                if not pool_fn:
                    raise ValueError(f"Unsupported pooling method: {gpm}")
                self_sub.pool = pool_fn # Corrected assignment

                self_sub.post_mlp=nn.ModuleList();mcs=cc
                for i,lc in enumerate(pglc):
                    lt=lc.get("type","").lower()
                    if lt=="linear":
                        sz=lc.get("size");
                        if sz is None:raise ValueError(f"Post-GNN MLP Linear {i} needs 'size'.")
                        self_sub.post_mlp.append(nn.Linear(mcs,sz));mcs=sz
                        if(af := self_sub.p._get_activation_function(lc.get("activation"))): self_sub.post_mlp.append(af) # Fixed: s_sub to self_sub
                    elif lt=="dropout": self_sub.post_mlp.append(nn.Dropout(lc.get("rate",0.5))) # Fixed: s_sub to self_sub
                    else:raise ValueError(f"Unsupported Post-GNN MLP layer: {lt}")
                self_sub.out_l=nn.Linear(mcs,os); self_sub.out_act=self_sub.p._get_activation_function(oan)

            def __repr__(self_sub) -> str:
                # Simplified custom repr to avoid recursion issues.
                convs_str = f"convs_count={len(self_sub.convs)}"
                pool_str = f"pool={self_sub.pool.__name__ if hasattr(self_sub.pool, '__name__') else str(self_sub.pool)}"
                mlp_str = f"mlp_count={len(self_sub.post_mlp)}"
                out_l_str = f"out_l={self_sub.out_l.__class__.__name__}"
                out_act_str = f"out_act={self_sub.out_act.__class__.__name__ if self_sub.out_act is not None else 'None'}"
                return f"GNNSubModel({convs_str}, {pool_str}, {mlp_str}, {out_l_str}, {out_act_str})"

            def forward(self_sub,d:PyGData):
                x,ei,b=d.x,d.edge_index,getattr(d,'batch',None)
                if x.dtype!=torch.float32:x=x.to(torch.float32)
                for l_ in self_sub.convs:x=l_(x,ei) if isinstance(l_,GCNConv) else l_(x)
                if x.numel()==0:
                    n_out_ch=nfs;
                    if self_sub.convs: lc_=next((l__ for l__ in reversed(self_sub.convs) if hasattr(l__,'out_channels')),None);
                    if lc_:n_out_ch=lc_.out_channels # type: ignore
                    n_g=int(b.max().item()+1) if b is not None and b.numel()>0 else 1;
                    x=torch.zeros((n_g,n_out_ch),device=x.device)
                else:
                    if b is None:b=torch.zeros(x.shape[0],dtype=torch.long,device=x.device)
                    x=self_sub.pool(x,b)
                for l_ in self_sub.post_mlp:x=l_(x)
                x=self_sub.out_l(x);
                if self_sub.out_act:x=self_sub.out_act(x)
                return x
        return GNNSubModel(self)

    def _build_model(self)->nn.Module:
        if self.model_type=="gnn":return self._build_gnn_model()
        elif self.model_type=="ffn":return self._build_ffn_model()
        else:raise ValueError(f"Unknown model_type:{self.model_type}")

    def forward(self,input_data:Union[torch.Tensor,PyGData])->torch.Tensor:
        if self.model_type=="gnn":
            if not isinstance(input_data,PyGData):raise TypeError(f"GNN expects PyGData,got {type(input_data)}")
            return self.model_internal(input_data)
        else:
            if not isinstance(input_data,torch.Tensor):raise TypeError(f"FFN expects Tensor,got {type(input_data)}")
            feats=input_data.to(torch.float32);exp_in=self.nn_config.get("input_size")
            if feats.shape[-1]!=exp_in:raise ValueError(f"Input feats {feats.shape[-1]}!=FFN expected {exp_in}")
            return self.model_internal(feats)

    def predict(self,features_or_graph_data:Union[List[float],torch.Tensor,Any,PyGData],verbose:bool=False)->torch.Tensor:
        self.model_internal.eval()
        with torch.no_grad():
            input_val:Union[torch.Tensor,PyGData]
            if self.model_type=="gnn":
                if isinstance(features_or_graph_data,PyGData): # type: ignore # PyGData might be a stub
                    input_val=features_or_graph_data
                # NetworkGraph should be directly usable now due to import from graph_schema
                elif NetworkGraph is not type(None) and isinstance(features_or_graph_data, NetworkGraph):
                    # Use the newly imported conversion function
                    if network_graph_to_pyg_data_via_extractor is None: # Should not happen if imports are correct
                        raise RuntimeError("network_graph_to_pyg_data_via_extractor (from feature_extractors) is missing.")
                    conv_data = network_graph_to_pyg_data_via_extractor(features_or_graph_data)
                    if conv_data is None:
                        # Check if PyG is actually available, as the extractor might return None if not.
                        if not PYG_AVAILABLE:
                            raise RuntimeError("PyTorch Geometric not available, cannot convert NetworkGraph to PyGData for GNN prediction.")
                        raise ValueError("NetworkGraph to PyGData conversion failed using feature_extractors.extract_network_graph_pyg_data.")
                    input_val=conv_data
                else:raise TypeError(f"GNN predict expects PyGData or NetworkGraph, got {type(features_or_graph_data)}")
                if verbose and isinstance(input_val, PyGData) and hasattr(input_val, 'num_nodes') and hasattr(input_val, 'num_edges'):
                    print(f"  NN Predict (GNN) Input: Nodes={input_val.num_nodes}, Edges={input_val.num_edges}, x_shape={input_val.x.shape if hasattr(input_val,'x') else 'N/A'}")
            else: # FFN
                if isinstance(features_or_graph_data,torch.Tensor):input_val=features_or_graph_data
                elif isinstance(features_or_graph_data,list):input_val=torch.tensor(features_or_graph_data,dtype=torch.float32)
                else:raise TypeError(f"FFN predict expects list/Tensor,got {type(features_or_graph_data)}")
                if input_val.ndim==1:input_val=input_val.unsqueeze(0)
                if verbose:print(f"  NN Predict (FFN) Input (shape {input_val.shape}): {input_val.tolist()}")

            output=self.forward(input_val)
            if verbose:print(f"  NN Predict Output Scores: {output.tolist()}")
            return output

    def train_on_batch(self,experiences:List[Any],gamma:float=0.99,verbose:bool=False)->float:
        if not experiences:return 0.0
        self.model_internal.train()
        states_orig=[e.state for e in experiences];actions=torch.tensor([e.action for e in experiences],dtype=torch.long)
        rewards=torch.tensor([e.reward for e in experiences],dtype=torch.float32);next_states_orig=[e.next_state for e in experiences]
        dones=torch.tensor([e.done for e in experiences],dtype=torch.bool)

        current_q_s_a: torch.Tensor
        next_q_s_prime_max: torch.Tensor

        if self.model_type=="ffn":
            # states_orig is a list of tensors, each [1, feature_size]
            # Concatenate them along dimension 0 to make a [batch_size, feature_size] tensor
            states_batch = torch.cat(states_orig, dim=0)
            current_q_s_all_a=self.forward(states_batch)
            current_q_s_a=current_q_s_all_a.gather(1,actions.unsqueeze(-1)).squeeze(-1)

            next_q_s_prime_max = torch.zeros(len(experiences), device=states_batch.device) # Initialize for all experiences
            non_final_mask = ~dones # Mask for experiences that are not terminal

            # Identify the indices in the original batch that correspond to non-terminal states
            non_final_indices = torch.where(non_final_mask)[0]

            if len(non_final_indices) > 0: # If there are any non-terminal states
                # Collect next_state tensors ONLY for these non-terminal experiences,
                # and only if they are valid tensors.
                valid_next_states_for_non_final_experiences = []
                # Keep track of which original batch indices these valid_next_states correspond to.
                # This list of indices will be used to update next_q_s_prime_max.
                original_indices_of_valid_next_states = []

                for idx_in_batch in non_final_indices:
                    next_state_candidate = next_states_orig[idx_in_batch.item()]
                    if next_state_candidate is not None and isinstance(next_state_candidate, torch.Tensor):
                        valid_next_states_for_non_final_experiences.append(next_state_candidate)
                        original_indices_of_valid_next_states.append(idx_in_batch) # Store the original tensor index

                if valid_next_states_for_non_final_experiences: # If we found any valid tensors for non-terminal states
                    # Concatenate these valid next_state tensors into a batch
                    nfns_tensor = torch.cat(valid_next_states_for_non_final_experiences, dim=0)

                    with torch.no_grad():
                        # Get Q-values for all actions for these next_states
                        q_values_for_all_actions_nfns = self.forward(nfns_tensor)
                        # Select the max Q-value for each of these next_states
                        max_q_values_nfns = q_values_for_all_actions_nfns.max(1)[0]

                    # Update next_q_s_prime_max at the specific original_indices_of_valid_next_states
                    # Need to convert list of tensor indices to a tensor for indexing
                    if original_indices_of_valid_next_states: # Should be true if valid_next_states_for_non_final_experiences is true
                        update_indices = torch.stack(original_indices_of_valid_next_states).squeeze() # Ensure it's 1D
                        if update_indices.ndim == 0: # Handle if only one non-final state
                            update_indices = update_indices.unsqueeze(0)

                        next_q_s_prime_max[update_indices] = max_q_values_nfns

            # If an experience is non-terminal (non_final_mask[i] is True) but its next_state was None or not a Tensor,
            # its entry in next_q_s_prime_max will remain 0. This is acceptable (treat as 0 future reward).

        elif self.model_type=="gnn" and PYG_AVAILABLE and PyGBatch is not None:
            # GNN path needs similar careful handling of non_final_mask and next_state processing
            valid_states_pyg=[s for s in states_orig if isinstance(s,PyGData)]
            if len(valid_states_pyg)!=len(states_orig):raise ValueError("GNN training needs all states as PyGData.")
            states_batch_pyg=PyGBatch.from_data_list(valid_states_pyg)
            current_q_s_all_a=self.forward(states_batch_pyg)
            current_q_s_a=current_q_s_all_a.gather(1,actions.unsqueeze(-1)).squeeze(-1)
            non_final_mask=~dones
            non_final_next_states_pyg=[ns for ns in next_states_orig if ns is not None and isinstance(ns,PyGData)]
            next_q_s_prime_max=torch.zeros(len(experiences),device=current_q_s_a.device)
            if non_final_next_states_pyg:
                # This assumes that the non_final_mask aligns with non_final_next_states_pyg
                # which is true if all next_states are either None or PyGData.
                if len(non_final_next_states_pyg) == non_final_mask.sum().item():
                    next_states_batch_pyg = PyGBatch.from_data_list(non_final_next_states_pyg)
                    with torch.no_grad():next_q_s_prime_all_a=self.forward(next_states_batch_pyg)
                    next_q_s_prime_max[non_final_mask]=next_q_s_prime_all_a.max(1)[0]
                elif non_final_mask.sum().item() > 0 : # Some non-final states were not PyGData
                     print("Warning: GNN train_on_batch: Some non-final next_states were not PyGData. Their Q-values will be 0.")

        else:raise RuntimeError(f"Unsupported model_type '{self.model_type}' or PyG unavailable for training.")

        q_target=rewards+(gamma*next_q_s_prime_max*(~dones).float())
        loss=self.criterion(current_q_s_a,q_target.detach())
        self.optimizer.zero_grad();loss.backward();self.optimizer.step()

        if verbose and experiences:
            exp0=experiences[0];sr_str=str(exp0.state)[:70]+"..." if len(str(exp0.state))>70 else str(exp0.state)
            nsr_str=str(exp0.next_state)[:70]+"..." if exp0.next_state and len(str(exp0.next_state))>70 else str(exp0.next_state)
            print(f"  NNTrn(S:'{sr_str}',A:{exp0.action},R:{exp0.reward:.2f},NS:'{nsr_str}',D:{exp0.done}) Qpred:{current_q_s_a[0].item():.4f},Qtarg:{q_target[0].item():.4f},Loss:{loss.item():.6f}")

        if not hasattr(self,"_train_c"):self._train_c=0
        self._train_c+=1 # Corrected increment
        return loss.item()

    def get_config(self)->Dict[str,Any]:return copy.deepcopy(self.nn_config)
    def get_architecture_summary(self) -> str:
        sl=[];sl.append(f"Model Type: {self.model_type.upper()}")
        if self.model_type=="ffn":
            sl.append(f"  Input Size: {self.nn_config.get('input_size')}")
            sl.append("  FFN Hidden Layers Config:")
            for i,lc in enumerate(self.nn_config.get('layers',[])):
                ls=f"    L{i}:T={lc.get('type')}"
                if lc.get('type')=='linear':ls+=f",S={lc.get('size')},A={lc.get('activation','none')}"
                elif lc.get('type')=='dropout':ls+=f",R={lc.get('rate')}"
                sl.append(ls)
            sl.append(f"  Output Layer:S={self.nn_config.get('output_size')},A={self.nn_config.get('output_activation','none')}")
        elif self.model_type=="gnn":
            sl.append(f"  Node Feature Size: {self.nn_config.get('node_feature_size')}")
            sl.append("  GNN Conv Layers Config:")
            for i,lc in enumerate(self.nn_config.get('gnn_layers',[])):
                ls=f"    L{i}:T={lc.get('type')}"
                if lc.get('type') in ['gcnconv','gatconv']:ls+=f",OC={lc.get('out_channels')},A={lc.get('activation','none')}"
                sl.append(ls)
            sl.append(f"  Global Pooling: {self.nn_config.get('global_pooling','N/A')}")
            if self.nn_config.get('post_gnn_mlp_layers'):
                sl.append("  Post-GNN MLP Layers Config:")
                for i,lc in enumerate(self.nn_config.get('post_gnn_mlp_layers',[])):
                    ls=f"    L{i}:T={lc.get('type')}"
                    if lc.get('type')=='linear':ls+=f",S={lc.get('size')},A={lc.get('activation','none')}"
                    elif lc.get('type')=='dropout':ls+=f",R={lc.get('rate')}"
                    sl.append(ls)
            sl.append(f"  Output Layer:S={self.nn_config.get('output_size')},A={self.nn_config.get('output_activation','none')}")
        sl.append("\n  PyTorch Model Structure:");sl.append(str(self.model_internal))
        return "\n".join(sl)

    def reconfigure(self, new_nn_config: Dict[str, Any]):
        print(f"Reconfiguring ReprogrammableSelectorNN.")
        if not new_nn_config or not isinstance(new_nn_config, dict): raise ValueError("Invalid new_nn_config: Must be dict.")
        cm_type=new_nn_config.get("model_type",self.model_type).lower();new_nn_config["model_type"]=cm_type
        if cm_type=="ffn":
            if 'input_size' not in new_nn_config or 'output_size' not in new_nn_config: raise ValueError("FFN config needs input/output_size.")
        elif cm_type=="gnn":
            if 'node_feature_size' not in new_nn_config:
                # NetworkGraph should be directly usable
                if IMPORTED_GNN_NODE_FEATURE_DIM is None or \
                   (IMPORTED_GNN_NODE_FEATURE_DIM == 1 and NetworkGraph is type(None)):
                   raise ValueError("GNN config needs node_feature_size or valid IMPORTED_GNN_NODE_FEATURE_DIM from feature_extractors (NetworkGraph might be None).")
                new_nn_config['node_feature_size'] = IMPORTED_GNN_NODE_FEATURE_DIM
            if 'output_size' not in new_nn_config: raise ValueError("GNN config needs output_size.")
        else: raise ValueError(f"Unsupported model_type: {cm_type}")
        self.nn_config=copy.deepcopy(new_nn_config);self.model_type=cm_type
        print(f"  NN config updated. New model_type: {self.model_type}")
        try:
            self.model_internal=self._build_model();print("  Model rebuilt.")
            self.optimizer=optim.Adam(self.model_internal.parameters(),lr=self.learning_rate)
            self.criterion=nn.MSELoss()
        except Exception as e: print(f"  Error rebuilding model: {e}.")

    def save_model(self, directory_path: str, model_filename: str = "nn_model.pth", config_filename: str = "nn_config.json"):
        os.makedirs(directory_path, exist_ok=True)
        model_path = os.path.join(directory_path, model_filename)
        config_path = os.path.join(directory_path, config_filename)
        torch.save(self.model_internal.state_dict(), model_path)
        with open(config_path, 'w') as f:
            json.dump(self.nn_config, f, indent=4)
        print(f"Model saved to {model_path} and config to {config_path}")

    @staticmethod
    def load_model(directory_path: str, model_filename: str = "nn_model.pth", config_filename: str = "nn_config.json", learning_rate: Optional[float] = None) -> 'ReprogrammableSelectorNN':
        config_path = os.path.join(directory_path, config_filename)
        model_path = os.path.join(directory_path, model_filename)
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model or config file not found in {directory_path}")
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)

        # Allow overriding learning rate from loaded config
        final_learning_rate = learning_rate if learning_rate is not None else loaded_config.get('learning_rate', 0.001) # Original LR or default

        # The ReprogrammableSelectorNN __init__ needs nn_config, not the full loaded_config directly if it contains extra stuff.
        # And learning_rate is a direct param to __init__.
        # So, we pass the loaded_config (which is the nn_config) and the potentially overridden learning_rate.
        instance = ReprogrammableSelectorNN(nn_config=loaded_config, learning_rate=final_learning_rate)
        instance.model_internal.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path} and config from {config_path}")
        return instance

# Example usage
if __name__ == '__main__':
    import sys
    import os

    # Explicitly add parent of 'modules' to sys.path if not already there
    # This is to ensure 'from modules.feature_extractors' works when script is run directly.
    PACKAGE_PARENT = '..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    NEW_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
    if NEW_PATH not in sys.path:
        sys.path.insert(0, NEW_PATH)

    # print(f"DEBUG: sys.path in reprogrammable_selector_nn.py: {sys.path}") # For debugging path issues

    if not PYG_AVAILABLE: print("Warning: PyTorch Geometric not available, GNN tests will be limited or fail.")
    print("--- ReprogrammableSelectorNN (FFN & GNN path) Direct Test ---")
    Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))
    ffn_conf = {"model_type":"ffn", "input_size":10, "layers":[{"type":"linear","size":8,"activation":"relu"}], "output_size":3, "output_activation":"none"}
    print("\nTesting FFN configuration:")
    nn_ffn = ReprogrammableSelectorNN(ffn_conf, learning_rate=0.01)
    print(nn_ffn.get_architecture_summary())

    # Prepare FFN dummy states as tensors [1, feature_size]
    ffn_state1_tensor = torch.tensor([random.random() for _ in range(10)], dtype=torch.float32).unsqueeze(0)
    ffn_next_state1_tensor = torch.randn(1, 10, dtype=torch.float32) # For the second experience
    ffn_state2_tensor = torch.randn(1, 10, dtype=torch.float32)


    # Test FFN predict with a list (as might come from some raw feature source)
    ffn_predict_list_test = ffn_state1_tensor.squeeze(0).tolist()
    print(f"FFN Predict (single list): {nn_ffn.predict(ffn_predict_list_test, verbose=True)}")

    ffn_feats_batch = torch.randn(4, 10) # Batch of tensors for prediction
    print(f"FFN Predict (batch tensor): shape {nn_ffn.predict(ffn_feats_batch).shape}")

    # Dummy experiences for FFN should use tensors for states, as train_on_batch expects list of tensors
    dummy_exp_ffn = [
        Experience(ffn_state1_tensor, 0, 1.0, None, True),
        Experience(ffn_state2_tensor, 1, -1.0, ffn_next_state1_tensor, False)
    ]
    print("\nTesting FFN train_on_batch:")
    loss_ffn = nn_ffn.train_on_batch(dummy_exp_ffn, verbose=True); print(f"  FFN batch loss: {loss_ffn}")

    # For GNN test, use the new extractor and its dimension
    # network_graph_to_pyg_data_via_extractor is already imported at the top

    # --- GNN Test Debug Prints ---
    print(f"\n--- GNN Test Pre-condition Debug ---")
    print(f"PYG_AVAILABLE: {PYG_AVAILABLE}")
    print(f"network_graph_to_pyg_data_via_extractor is None: {network_graph_to_pyg_data_via_extractor is None}")
    if network_graph_to_pyg_data_via_extractor is not None:
        print(f"type(network_graph_to_pyg_data_via_extractor): {type(network_graph_to_pyg_data_via_extractor)}")
    # Use direct NetworkGraph for these runtime checks now
    print(f"NetworkGraph is type(None): {NetworkGraph is type(None)}")
    if NetworkGraph is not type(None):
        print(f"type(NetworkGraph): {type(NetworkGraph)}")
    print(f"IMPORTED_GNN_NODE_FEATURE_DIM: {IMPORTED_GNN_NODE_FEATURE_DIM}") # This might be None if feature_extractors hasn't run GNNFeatureExtractor yet
    # --- End GNN Test Debug Prints ---

    # Condition should use direct NetworkGraph for the check
    if PYG_AVAILABLE and network_graph_to_pyg_data_via_extractor is not None and NetworkGraph is not type(None):
        print("\nAttempting GNN configuration test...")
        # First, create a sample graph and extract features to determine node_feature_size dynamically for the test
        temp_g = NetworkGraph("temp_g_for_dim_check") # Use direct NetworkGraph to instantiate
        temp_g.add_layer_node("l1", layer_type='Linear', node_attributes={'out_features': 10})
        temp_pyg_data = network_graph_to_pyg_data_via_extractor(temp_g)

        if temp_pyg_data is not None and hasattr(temp_pyg_data, 'x') and temp_pyg_data.x is not None and temp_pyg_data.x.ndim == 2 and temp_pyg_data.x.shape[1] > 0:
            dynamic_node_feature_size = temp_pyg_data.x.shape[1]
            print(f"  Dynamically determined node_feature_size for GNN test: {dynamic_node_feature_size}")

            gnn_conf = {"model_type":"gnn","node_feature_size":dynamic_node_feature_size,
                        "gnn_layers":[{"type":"gcnconv","out_channels":16,"activation":"relu"}],
                        "global_pooling":"mean","post_gnn_mlp_layers":[{"type":"linear","size":8,"activation":"relu"}],
                        "output_size":2,"output_activation":"none"}
            print("  Testing GNN with dynamically determined node_feature_size:")
            nn_gnn = ReprogrammableSelectorNN(gnn_conf, learning_rate=0.01)
            print(nn_gnn.get_architecture_summary())
            print("  GNN architecture summary printed.")

            try:
                print("  GNN Test: Inside try block.")
                from modules.graph_schema import ArchitecturalNode # Explicit import for test context

                # Create a graph with layer_types known to GNNFeatureExtractor
                print("  GNN Test: Creating g1 NetworkGraph...")
                g1 = NetworkGraph("g1_test") # Use direct NetworkGraph
                print("  GNN Test: g1 created. Adding layers to g1...")
                g1.add_layer_node("input_node", layer_type='Input', node_attributes={'out_features': 64})
                g1.add_layer_node("linear1", layer_type='Linear', node_attributes={'in_features': 64, 'out_features': 32})
                g1.add_layer_node("relu1", layer_type='ReLU', node_attributes={})
                g1.add_layer_node("output_node", layer_type='Linear', node_attributes={'in_features': 32, 'out_features': 2})
                g1.connect_layers("input_node", "linear1")
                g1.connect_layers("linear1", "relu1")
                g1.connect_layers("relu1", "output_node")

                g2 = NetworkGraph("g2_test") # Use direct NetworkGraph
                g2.add_layer_node("input_node2", layer_type='Input', node_attributes={'out_features': 64})
                g2.add_layer_node("linear_alt", layer_type='Linear', node_attributes={'in_features': 64, 'out_features': 2})
                g2.connect_layers("input_node2", "linear_alt")

                # Convert using the new extractor function
                d1 = network_graph_to_pyg_data_via_extractor(g1)
                d2 = network_graph_to_pyg_data_via_extractor(g2)
                # For next_state, can reuse or create another graph
                g_next = NetworkGraph("g_next_test") # Use direct NetworkGraph
                g_next.add_layer_node("next_in", layer_type='Input', node_attributes={'out_features':64})
                g_next.add_layer_node("next_l", layer_type='Linear', node_attributes={'in_features':64,'out_features':2})
                g_next.connect_layers("next_in","next_l")
                d_next = network_graph_to_pyg_data_via_extractor(g_next)

                if d1 and hasattr(d1,'num_nodes') and d1.num_nodes > 0 and \
                   d2 and hasattr(d2,'num_nodes') and d2.num_nodes > 0 and \
                   d_next and hasattr(d_next,'num_nodes') and d_next.num_nodes > 0:
                    print(f"GNN Predict (d1): {nn_gnn.predict(d1, verbose=True)}")
                    dummy_exp_gnn = [Experience(d1, 0, 0.5, d_next, False), Experience(d2, 1, -0.2, None, True)]
                    print("\nTesting GNN train_on_batch:")
                    loss_gnn = nn_gnn.train_on_batch(dummy_exp_gnn, verbose=True)
                    print(f"  GNN batch loss: {loss_gnn}")
                else:
                    print("Failed to create valid PyGData for GNN test or graph was empty.")
                    if d1: print(f"  d1 check: x={d1.x.shape if hasattr(d1,'x') and d1.x is not None else 'N/A'}, edge_index={d1.edge_index.shape if hasattr(d1,'edge_index') and d1.edge_index is not None else 'N/A'}")
                    if d2: print(f"  d2 check: x={d2.x.shape if hasattr(d2,'x') and d2.x is not None else 'N/A'}, edge_index={d2.edge_index.shape if hasattr(d2,'edge_index') and d2.edge_index is not None else 'N/A'}")
                    if d_next: print(f"  d_next check: x={d_next.x.shape if hasattr(d_next,'x') and d_next.x is not None else 'N/A'}, edge_index={d_next.edge_index.shape if hasattr(d_next,'edge_index') and d_next.edge_index is not None else 'N/A'}")

            except Exception as e:
                print(f"Error in GNN test block: {e}")
    else:
        print("\nSkipping GNN test (PyG not available, or network_graph_to_pyg_data_via_extractor missing, or NetworkGraph type is None).")

    print("\nTesting Save/Load:")
    save_dir = "./temp_nn_save"
    try:
        nn_ffn.save_model(save_dir)
        loaded_nn = ReprogrammableSelectorNN.load_model(save_dir, learning_rate=0.005)
        print("Loaded NN architecture summary:")
        print(loaded_nn.get_architecture_summary())
        print(f"Loaded NN learning rate (should be 0.005): {loaded_nn.learning_rate}")
        assert loaded_nn.learning_rate == 0.005
        # Use ffn_predict_list_test which was defined earlier for prediction consistency
        print(f"Prediction from loaded model: {loaded_nn.predict(ffn_predict_list_test)}")
    except Exception as e:
        print(f"Error during save/load test: {e}")
    finally:
        # Clean up created directory and files
        if os.path.exists(save_dir):
            for f in os.listdir(save_dir): os.remove(os.path.join(save_dir,f))
            os.rmdir(save_dir)

    print("--- Test Complete ---")
