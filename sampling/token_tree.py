import torch
from typing import List, Tuple, Dict


class Node:
    def __init__(self, token_id: int, q: float, tokenizer=None):
        self.id: int = None
        self.token_id: int = token_id
        self.q: float = q
        self.tokenizer = tokenizer
        self.children: Dict[int, Node] = {}

    def _tree_str_helper(self, level=0) -> str:
        ret = self.__str__() + "\n"
        for i, child in enumerate(self.children.values()):
            if i == len(self.children) - 1:
                ret += "    " * (level) + "└── " + child._tree_str_helper(level + 1)
            else:
                ret += "    " * (level) + "├── " + child._tree_str_helper(level + 1)
        return ret

    def __str__(self):
        if self.token_id is None:
            return "(Root)"
        if type(self.tokenizer) == dict:
            return f"({self.token_id}: {self.tokenizer[self.token_id]}, q={self.q})"
        else:
            return f"({self.token_id}: {self.tokenizer.decode([self.token_id])}, q={self.q})"

    def __repr__(self):
        return self.__str__()


class TokenTree:
    def __init__(self, root_token_id, tokenizer=None):
        self.root: Node = Node(root_token_id, None, tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.root.id = 0
        self.nodelist = [self.root]

    def insert(self, sequence: List[Tuple[int, float]]):
        """
        Insert a sequence of tokens into the tree.
            sequence: list of tuples (token_id, q).
        """
        node = self.root
        for token_id, q in sequence:
            if token_id not in node.children:
                newnode = Node(token_id, q, tokenizer=self.tokenizer)
                node.children[token_id] = newnode
                newnode.id = len(self.nodelist)
                self.nodelist.append(node.children[token_id])
            else:
                assert node.children[token_id].q == q, f"Token ID {token_id} already exists with different q value."
            node = node.children[token_id]

    def attention_mask(self, debug=False) -> torch.Tensor:
        """
        Generate the attention mask for the tree.
            debug: whether to print debug information.
        Returns:
            attention_mask: tensor of shape (1, num_nodes, num_nodes).
        """

        num_nodes = len(self.nodelist)
        attention_mask = torch.zeros((1, num_nodes, num_nodes), dtype=torch.long)

        def _dfs(node, attend_list):
            # attend to previous nodes
            for attend_node in attend_list:
                attention_mask[0, node.id, attend_node.id] = 1
            # attend to self
            attention_mask[0, node.id, node.id] = 1
            attend_list.append(node)

            for child in node.children.values():
                _dfs(child, attend_list)
            attend_list.pop()

        _dfs(self.root, [])

        if debug:
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if attention_mask[0, i, j] == 1:
                        token_i = str(self.nodelist[i])
                        token_j = str(self.nodelist[j])
                        print(f"Node {token_i} attends to Node {token_j}")
        return attention_mask

    def longest_accepted_sequence(self, target_prob: torch.Tensor, debug=False) -> List[Tuple[int, float]]:
        """
        Get the longest accepted sequence from the tree.
            target_prob: tensor of shape (1, num_nodes, vocab_size).
        Returns:
            token_ids: list of token ids in the longest accepted sequence.
            p: target_prob filtered by the longest accepted sequence. shape (1, len(accepted_tokens), vocab_size)
        """
        assert target_prob.shape == (
            1,
            len(self.nodelist),
            len(self.tokenizer),
        ), f"target_prob shape {target_prob.shape} does not match expected shape {(1, len(self.nodelist), len(self.tokenizer))}"

        accepted_node_ids = []
        longest_accepted_node_ids = []

        def _accept(node):
            nonlocal longest_accepted_node_ids

            accepted_node_ids.append(node.id)
            if len(accepted_node_ids) > len(longest_accepted_node_ids):
                longest_accepted_node_ids = accepted_node_ids.copy()

            for child in node.children.values():
                if debug:
                    print(f"child: {child.token_id}, p={target_prob[0, node.id, child.token_id]}, q={child.q}")
                if torch.rand(1, device=target_prob.device) < target_prob[0, node.id, child.token_id] / child.q:
                    # accept the child
                    _accept(child)
                else:
                    # reject the child
                    pass

            accepted_node_ids.pop()

        _accept(self.root)

        token_ids = [self.nodelist[i].token_id for i in longest_accepted_node_ids]
        p = target_prob[:, longest_accepted_node_ids, :]

        return token_ids, p

    def __str__(self):
        return self.root._tree_str_helper()

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    # 序列1：machine learning algorithm
    # 序列2：machine learning system implementation
    # 序列3：machine learning system design
    # 序列4：machine translation model
    # 序列5：machine translation application
    # 序列6：machine time

    token2id = {
        "<root>": -1,
        "machine": 0,
        "learning": 1,
        "algorithm": 2,
        "system": 3,
        "implementation": 4,
        "design": 5,
        "translation": 6,
        "model": 7,
        "application": 8,
        "time": 9,
    }

    id2token = {v: k for k, v in token2id.items()}

    sentences = [
        [("machine", 0.1), ("learning", 0.2), ("algorithm", 0.3)],
        [("machine", 0.1), ("learning", 0.2), ("system", 0.3), ("implementation", 0.4)],
        [("machine", 0.1), ("learning", 0.2), ("system", 0.3), ("design", 0.4)],
        [("machine", 0.1), ("translation", 0.2), ("model", 0.3)],
        [("machine", 0.1), ("translation", 0.2), ("application", 0.3)],
        [("machine", 0.1), ("time", 0.2)],
    ]

    sentences = [[(token2id[token], q) for token, q in sentence] for sentence in sentences]

    # Create a tree and insert the sequences
    tree = TokenTree(root_token_id=None, tokenizer=id2token)
    for sentence in sentences:
        tree.insert(sentence)

    print(tree)
    print(tree.attention_mask(debug=True))
