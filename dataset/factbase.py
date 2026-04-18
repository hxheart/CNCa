import ast

from torch_geometric.data import Data
import pandas as pd

from feature import *
from coders import onehot_encoding, binary_encoding, boolean_encoding
import enum

NUM_INT_VALUES = 64
NUM_IDS = 2048
MAX_NUM_PREDICATES = 64

def init_feature_registry(feature_registry):
    feature_registry.feature("type", onehot_encoding(3))
    feature_registry.feature("id", binary_encoding(NUM_IDS))
    feature_registry.feature("predicate", onehot_encoding(MAX_NUM_PREDICATES))
    feature_registry.feature("holds", boolean_encoding())

class Constant:
    def __init__(self, name, id=None):
        self.name = name
        self.id = id

    def __repr__(self):
        return self.name

class Node:
    def __init__(self, id, name, features):
        self.id = id
        self.name = name
        self.features = features

    def feature(self, name):
        for f in self.features:
            if f.feature.name == name: return f.value
        return None

class FactBase:
    def __init__(self, predicate_declarations):
        self.nodes = {}
        self.edges = []
        self.predicate_truth_values = {}
        
        self.feature_registry = FeatureRegistry()
        init_feature_registry(self.feature_registry)

        self.predicate_declarations = {}
        for i, p in enumerate(sorted(list(predicate_declarations.keys()))):
            assert i < MAX_NUM_PREDICATES, "the program uses more than the maximum number of allowed predicates"
            
            decl = predicate_declarations[p]
            decl._predicate_feature_value = i
            self.predicate_declarations[p] = decl 
            if len(decl.arg_types) == 1:
                decl.arg_feature_mapping = {0: self.predicate_feature(p)}
            else:
                for i, a in enumerate(decl.arg_types):
                    if a is not Constant:
                        decl.arg_feature_mapping[i] = self.predicate_feature(p, argidx = i)

    def _arg_key(self, a): 
        if type(a) is int: return a
        elif type(a) is Constant: return self.get_or_create_node(a).id
        else: assert False, f"unsupported argument type {type(a)}"

    def set_predicate_value(self, name, args, value):
        res = self.query(name, *args)
        if len(res) != 0: 
            for row in res: 
                if row[-1] != value: return False

        df = pd.DataFrame([[self._arg_key(a) for a in args] + [value]], columns=[f"arg{i}" for i in range(len(args))] + ["holds"])
        if not name in self.predicate_truth_values.keys():
            self.predicate_truth_values[name] = df
        else:
            self.predicate_truth_values[name] = pd.concat([self.predicate_truth_values[name], df])
        return True

    def predicate_feature(self, name, argidx = None):
        name = "predicate_" + name + (f"_arg{argidx}" if argidx is not None else "")
        encoding = boolean_encoding() if argidx is None else binary_encoding(NUM_INT_VALUES)
        return self.feature_registry.feature(name, encoding)

    def get_or_create_node(self, c):
        if not c.name in self.nodes: 
            self.nodes[c.name] = Node(len(self.nodes.values()), c.name, [
                self.feature_registry.feature_value("type", NodeType.constant),
                self.feature_registry.feature_value("id", len(self.nodes.values()))
            ])
        return self.nodes[c.name]

    def create_predicate_node(self, name):
        node_name = "predicate_" + str(len(self.nodes.values()))
        self.nodes[node_name] = Node(len(self.nodes.values()), node_name, [
            self.feature_registry.feature_value("type", NodeType.predicate),
            self.feature_registry.feature_value("predicate", self.predicate_declarations[name].predicate_feature_value),
            self.feature_registry.feature_value("id", len(self.nodes.values()))
        ])
        return self.nodes[node_name]

    def create_query_node(self, name):
        node_name = "query_" + str(len(self.nodes.values()))
        self.nodes[node_name] = Node(len(self.nodes.values()), node_name, [
            self.feature_registry.feature_value("type", NodeType.predicate),
            self.feature_registry.feature_value("predicate", self.predicate_declarations[name].predicate_feature_value),
            self.feature_registry.feature_value("id", len(self.nodes.values()))
        ])
        return self.nodes[node_name]

    def add_fact(self, name, is_negated, *args):
        res = self.set_predicate_value(name, args, False if is_negated else True)
        if not res: return

        if len(args) == 1:
            # negated single arg facts are simply omitted
            if not is_negated: 
                self._add_single_arg_fact(name, args[0])
        elif name == "connected" or name == "ibgp" or name == "ebgp":
            # TODO: maybe use type annotation to declare a predicate to be symmetric
            self._add_multi_arg_symmetric_fact(name, is_negated, args)
        else:
            self._add_multi_arg_fact(name, is_negated, args)

    def add_query(self, name, *args):
        assert name in self.predicate_declarations.keys(), f"unknown predicate {name}"
        decl: PredicateDeclaration = self.predicate_declarations[name]

        query_node = self.create_query_node(name)
        for i,arg in enumerate(args):
            if type(arg) == Constant:
                self.edges.append([self.get_or_create_node(arg).id, query_node.id, i])
            else:
                assert type(arg) is int, f"cannot query facts with argument type {type(arg)}"
                query_node.features.append(decl.feature_value(i, arg))
        query_node.features.append(self.feature_registry.feature_value("holds", -2))

    def _add_multi_arg_fact(self, name, is_negated, args):
        assert name in self.predicate_declarations.keys(), f"unknown predicate {name}"
        decl: PredicateDeclaration = self.predicate_declarations[name]
        
        # forward direction
        predicate_node = self.create_predicate_node(name)
        for i,arg in enumerate(args):
            if type(arg) == Constant:
                self.edges.append([self.get_or_create_node(arg).id, predicate_node.id, i])
            else:
                assert type(arg) is int, f"not a supported predicate argument {arg}"
                predicate_node.features.append(decl.feature_value(i, arg))
        predicate_node.features.append(self.feature_registry.feature_value("holds", 0 if is_negated else 1))


    def _add_multi_arg_symmetric_fact(self, name, is_negated, args):
        assert name in self.predicate_declarations.keys(), f"unknown predicate {name}"
        decl: PredicateDeclaration = self.predicate_declarations[name]
        predicate_node = self.create_predicate_node(name)
        for i,arg in enumerate(args):
            if type(arg) == Constant:
                self.edges.append([predicate_node.id, self.get_or_create_node(arg).id, i])
            else:
                assert type(arg) is int, f"not a supported predicate argument {arg}"
                predicate_node.features.append(decl.feature_value(i, arg))
        predicate_node.features.append(self.feature_registry.feature_value("holds", 0 if is_negated else 1))

    def _add_single_arg_fact(self, name, arg):
        assert type(arg) is Constant, f"cannot add predicate over non-referential constant {arg}"
        assert name in self.predicate_declarations.keys(), f"unknown predicate {name}"
        
        self.get_or_create_node(arg).features.append(self.predicate_feature(name).value(1))
    
    @staticmethod
    def parse(s):
        return parse(s)

    def constants(self, selector):
        if selector not in self.predicate_truth_values.keys():
            print("unmatched selector", selector, self.predicate_truth_values.keys())
            return []
        return self.predicate_truth_values[selector]["arg0"].values.tolist()

    def query(self, name, *args):
        from functools import reduce

        if name not in self.predicate_truth_values.keys(): return []

        frame = self.predicate_truth_values[name]

        filter_args = [self._arg_key(a) for a in args]
        filtered_indices = [self.predicate_truth_values[name][f"arg{i}"] == filter_args[i] for i in range(len(filter_args)) if filter_args[i] != -1]
        if len(filtered_indices) > 0:
            filtered_indices = reduce(lambda f1,f2: f1 & f2, filtered_indices)
            result = frame[filtered_indices]
        else:
            result = frame

        return result.to_numpy().tolist()

    def get_all_facts(self, return_constant_name_mapping=False):
        facts = [] # fact = [PredicateAst]

        predicate_nodes = [node for node in self.nodes.values() if node.feature("type") == NodeType.predicate]
        constant_nodes = [node for node in self.nodes.values() if node.feature("type") == NodeType.constant]

        predicate_lookup_table = {}
        single_arg_decls = []

        for decl in self.predicate_declarations.values(): 
            predicate_lookup_table[decl.predicate_feature_value] = decl
            if len(decl.arg_types) == 1: single_arg_decls.append(decl)

        predicate_arguments = {}
        predicate_negated = {}
        predicate_is_query = {}
        for node in predicate_nodes:
            predicate_feature_value = node.feature("predicate")
            assert predicate_feature_value in predicate_lookup_table.keys(), f"cannot decode predicate with feature value {predicate_feature_value}"
            decl: PredicateDeclaration = predicate_lookup_table[predicate_feature_value]
            predicate_arguments[node.id] = [None for i in range(len(decl.arg_types))]
            
            assert node.feature("holds") is not None, "predicate node without holds feature"
            predicate_negated[node.id] = node.feature("holds") == 0
            predicate_is_query[node.id] = node.feature("holds") == -2

            # check for directly mapped arguments
            if decl.has_directly_mapped_arguments:
                for idx in decl.arg_feature_mapping.keys():
                    feature_name = decl.arg_feature_mapping[idx].name
                    predicate_arguments[node.id][idx] = node.feature(feature_name)

        # decode single-arg facts
        for node in constant_nodes:
            for decl in single_arg_decls:
                arg_feature_mapping = list(decl.arg_feature_mapping.values())
                assert len(arg_feature_mapping) == 1
                feature = arg_feature_mapping[0]
                if node.feature(feature.name): facts.append(PredicateAst(decl.name, [Constant(node.name)], False, False))
                    

        nodes_by_id = {}
        for node in self.nodes.values(): nodes_by_id[node.id] = node

        for src, tgt, edge_type in self.edges:
            src = nodes_by_id[src]
            tgt = nodes_by_id[tgt]
            constant_node = [n for n in [src,tgt] if n.feature("type") == NodeType.constant]
            predicate_node = [n for n in [src,tgt] if n.feature("type") == NodeType.predicate]
            assert len(constant_node) != 0 and len(predicate_node) != 0, f"edge between constant nodes {constant_node} and predicate nodes {predicate_node}"

            constant_node = constant_node[0]
            predicate_node = predicate_node[0]
            
            predicate_arguments[predicate_node.id][edge_type] = constant_node
        
        def arg_repr(arg):
            if type(arg) is int: return arg
            elif type(arg) is Node and arg.feature("type") == NodeType.constant: return Constant(arg.name)
            else: assert False, f"unsupported argument type {type(arg)}"
        
        for id, args in predicate_arguments.items(): 
            predicate_feature_value = nodes_by_id[id].feature("predicate")
            # make sure all predicate arguments are resolved correctly
            assert predicate_feature_value in predicate_lookup_table.keys(), f"cannot decode predicate with feature value {predicate_feature_value}"
            decl = predicate_lookup_table[predicate_feature_value]
            if not all([a is not None for a in args]):
                print(f"warning: failed to resolve arguments for predicate {decl}")

            facts.append(PredicateAst(decl.name, [arg_repr(a) for a in args], predicate_negated[id], predicate_is_query[id]))

        if return_constant_name_mapping:
            mapping = {}
            for n in self.nodes.values():
                if n.feature("type") == NodeType.constant:
                    mapping[n.name] = n.id
            return facts, mapping

        return facts

    def __repr__(self):
        return "\n".join([str(decl) for decl in self.predicate_declarations.values()]) + "\n\n" + \
            "\n".join([str(p) for p in self.get_all_facts()])

    def to_torch_data(self, return_node_names=False):
        d,names = self.to_data(return_node_names=True)

        d = Data(
            x=d["x"], 
            edge_index=d["edge_index"], 
            edge_type=d["edge_type"],
            predicate_declarations=d["predicate_declarations"],
        )
        if return_node_names:
            return d, names
        else:
            return d
    
    def  to_data(self, return_node_names=False):
        nodes = list(self.nodes.values())
        node_features = []
        node_indices = {}
        node_names = {}
        for i, node in enumerate(nodes):
            node_features.append(self.feature_registry.embed_features(node.features).unsqueeze(0))
            node_indices[node.id] = i
            node_names[i] = node.name
        node_features = torch.cat(node_features, axis=0)

        edge_index = [[], []]
        edge_type = []
        for src, tgt, et in self.edges:
            edge_index[0].append(node_indices[src])
            edge_index[1].append(node_indices[tgt])
            edge_type.append(et)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        data = {
            "x": node_features,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "predicate_declarations": self.predicate_declarations
        }

        #data = Data(x=node_features, edge_index=edge_index, edge_type=edge_type, predicate_declarations=self.predicate_declarations)

        if return_node_names: 
            return data, node_names
        else:
            return data

    @staticmethod
    def from_data(data_or_dict, decls=None, names=None):
        d = None
        if type(data_or_dict) is dict:
            d = Data(
                x=data_or_dict["x"],
                edge_index=data_or_dict["edge_index"],
                edge_type=data_or_dict["edge_type"],
                predicate_declarations=data_or_dict["predicate_declarations"]
            )
        else:
            d = data_or_dict

        p = FactBase(decls if decls is not None else d.predicate_declarations)
        node_features = d.x
        edge_index = d.edge_index
        edge_type = d.edge_type

        def node_name(id):
            if names is not None and id in names.keys(): return names[id]
            return f"constant{id}"

        for id, node in enumerate(node_features):
            # skip node data padding
            if torch.all(node == -1): continue
            features = p.feature_registry.unembed_features(node)
            node = Node(id, None, features)
            node.name = node_name(id) if node.feature("type") == NodeType.constant else f"predicate{id}"
            p.nodes[node.name] = node

        for src, tgt, et in zip(edge_index[0], edge_index[1], edge_type):
            p.edges.append([src.item(), tgt.item(), et.item()])

        p._rebuild_predicate_truth_value_index()

        return p

    def _rebuild_predicate_truth_value_index(self):
        self.predicate_truth_values = {}
        facts = self.get_all_facts()
        s_repr = str(self)

        self.nodes = {}
        self.edges = []

        for fact in facts: self.add_fact(fact.name, fact.is_negated, *fact.args)

        s_after_repr = str(self)

        assert s_repr == s_after_repr

class PredicateDeclaration:
    def __init__(self, name, arg_types):
        self.name = name
        self._predicate_feature_value = None
        self.arg_types = arg_types
        self.arg_feature_mapping = {}
        self.is_referential = all([at == Constant for at in self.arg_types]) and len(arg_types) > 1

    @property
    def predicate_feature_value(self):
        if self._predicate_feature_value is None:
            assert False, "illegal state: tried to access predicate_feature_value of a non-fully processed PredicateDeclaration"
        return self._predicate_feature_value

    @property
    def has_directly_mapped_arguments(self):
        return len(self.arg_feature_mapping)

    def feature_value(self, arg_idx, value):
        assert arg_idx < len(self.arg_types), f"predicate {self.name} only has {len(self.arg_types)} arguments, argument index {arg_idx} is out of range"
        assert arg_idx in self.arg_feature_mapping.keys(), f"not a directly mapped argument of predicate {self.name}({self.arg_types}): argument {arg_idx}"
        return self.arg_feature_mapping[arg_idx].value(value)   

    def __repr__(self):
        def type_repr(t): 
            if t is Constant: return "Constant"
            elif t is int: return "int"
            else: return str(t)
        return f"{self.name}: " + " × ".join([type_repr(t) for t in self.arg_types])

class PredicateAst:
    def __init__(self, name, args, is_negated, is_query):
        self.name = name
        self.args = args
        self.is_negated = is_negated
        self.is_query = is_query
    def __repr__(self):
        def argstr(a): return str(a) if a != -2 else "?"
        args_string = ",".join([argstr(a) for a in self.args])
        is_negated = "not " if self.is_negated else ""
        is_query = "? " if self.is_query else ""
        return f"{is_query}{is_negated}{self.name}({args_string})"

HOLE_ID = "__hole__"

def parse_arg(a):
    if type(a) == ast.Name:
        # holes are coded as int value -2
        if a.id == "HOLE_ID": return -2
        return Constant(a.id)
    elif type(a) == ast.Constant:
        if type(a.value) is int:
            return a.value
        else:
            raise AssertionError("expected integer number literal, but got literal of unsupported type '{type(a.value)}'")
    else:
        raise AssertionError(f"expected constant or number literal, but got argument of unsupported AST type '{type(a)}'")

def parse(s):
    # preprocessing
    def process(line):
        if line.startswith("? "):
            return line.replace("?", "-")
        return line.replace("?", "HOLE_ID")
    s = "\n".join([process(l) for l in s.split("\n")])

    facts = []
    tree = ast.parse(s)
    predicate_declarations = {}

    def argtype(a):
        if type(a) != Constant: return type(a)
        if a.name == "HOLE_ID": return int
        else: return type(a)

    def register_predicate_declaration(name, args):
        arg_types = [argtype(a) for a in args]

        if name in predicate_declarations.keys():
            decl = predicate_declarations[name]
            assert len(decl.arg_types) == len(arg_types), f"predicate does not match previously implied predicate signature {decl.arg_types}"
            for at1, at2 in zip(decl.arg_types, arg_types):
                assert at1 == at2, f"predicate does not match previously implied predicate signature {decl.arg_types}"
        else:
            predicate_declarations[name] = PredicateDeclaration(name, arg_types)

    for exp in tree.body:
        is_negated = False
        is_query = False
        if type(exp.value) is ast.UnaryOp:
            if type(exp.value.op) == ast.Not:
                exp.value = exp.value.operand
                is_negated = True
            elif type(exp.value.op) == ast.USub:
                exp.value = exp.value.operand
                is_query = True
            else:
                assert False, f"not a supported top-level statement {str(exp)}"

        assert type(exp.value) is ast.Call, f"expected predicate expression, but got '{str(exp.value)}'"
        call: ast.Call = exp.value
        assert type(call.func) is ast.Name, f"expected predicate name, but got {str(call.func)}"
        
        predicate_name = call.func.id
        predicate_arguments = [parse_arg(a) for a in call.args]

        facts.append(PredicateAst(predicate_name, predicate_arguments, is_negated, is_query))
        register_predicate_declaration(predicate_name, predicate_arguments)

    p = FactBase(predicate_declarations)
    
    for fact in facts:
        if fact.is_query:
            p.add_query(fact.name, *fact.args)
        else:
            p.add_fact(fact.name, fact.is_negated, *fact.args)
    return p

def to_cytoscape(data, p:FactBase = None, return_json_dict=False):
    import json
    """
    {
        "nodes": [
            { "data": { "id": "n0" } },
            { "data": { "id": "n1" } }
        ],
        "edges": [
            { "data": { "source": "n0", "target": "n1" } }
        ]
    }
    """
    predicates_by_id = {}
    nodes_by_id = {}
    if p is not None:
        for name, decl in p.predicate_declarations.items():
            predicates_by_id[decl.predicate_feature_value] = decl
        for node in p.nodes.values():
            nodes_by_id[node.id] = node

    feature = p.feature_registry.feature

    def get_name(idx, features):
        if p is None: return str(idx)
        node_id = features[feature("id").idx].long().item()
        if not node_id in nodes_by_id.keys(): return str(idx)
        node = nodes_by_id[node_id]
        
        node_type = features[feature("type").idx].long().item()
        if node_type == NodeType.constant:
            return node.name
        elif node_type == NodeType.predicate:
            predicate_id = features[feature("predicate").idx].long().item()
            if not predicate_id in predicates_by_id.keys(): return node.name
            predicate = predicates_by_id[predicate_id]
            return predicate.name
        elif node_type == NodeType.predicate:
            predicate_id = features[feature("predicate").idx].long().item()
            if not predicate_id in predicates_by_id.keys(): return node.name
            predicate = predicates_by_id[predicate_id]
            return predicate.name
        else:
            return "<UNKNOWN>"

    graph = {}
    graph["nodes"] = []
    for i, x in enumerate(data.x):
        graph["nodes"].append({
            "data": {
                "id": str(i),
                "label": get_name(i, x),
                "features": [x.long().item() for x in x],
                "color": "blue" if x[feature("type").idx] == NodeType.constant else "grey"
            }
        })
    graph["edges"] = []
    for i,et in zip(range(len(data.edge_index[0])), data.edge_type):
        src = data.edge_index[0,i].item()
        tgt = data.edge_index[1,i].item()
        assert src < len(data.x)
        assert tgt < len(data.x)
        graph["edges"].append({
            "data": {
                "source": str(src),
                "target": str(tgt),
                "label": f"arg{et.item()}"
            }
        })

    if p is not None:
        graph["features"] = [f.name for f in p.feature_registry.get_all_features()]

    
    if return_json_dict:
        return graph
    
    with open("viewer/data.json", "w") as f:
        json.dump(graph, f)

class NodeType(enum.IntEnum):
    constant = 0
    predicate = 1