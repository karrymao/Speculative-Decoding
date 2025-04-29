from typing import List, Tuple, Dict

import torch


class Node:
    def __init__(self, token_id: int, q: float, tokenizer=None, weight: int = 1):
        self.id: int = None
        self.token_id: int = token_id
        self.q: float = q
        self.tokenizer = tokenizer
        self.children: Dict[int, Node] = {}
        self.weight: int = weight  # New attribute to track node weight

    def _tree_str_helper(self, prefix="") -> str:
        """
        Helper function to generate string representation of the tree

        Args:
            prefix: Current line prefix for proper tree formatting

        Returns:
            String representation of this node and its children
        """
        # Start with current node
        result = self.__str__() + "\n"

        # Get list of children
        children = list(self.children.values())

        # Process each child
        for i, child in enumerate(children):
            is_last = i == len(children) - 1

            # Add appropriate branch character and child representation
            if is_last:
                result += prefix + "└── " + child._tree_str_helper(prefix + "    ")
            else:
                result += prefix + "├── " + child._tree_str_helper(prefix + "│   ")

        return result

    def __str__(self):
        if self.token_id is None:
            return "(Root)"
        if type(self.tokenizer) == dict:
            return f"({self.token_id}: {self.tokenizer[self.token_id]}, weight={self.weight})"
        else:
            return f"({self.token_id}: {self.tokenizer.decode([self.token_id])}, weight={self.weight})"

    def __repr__(self):
        return self.__str__()


class TokenTree:
    def __init__(self, root_token_id, tokenizer=None):
        self.root: Node = Node(root_token_id, None, tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.root.id = 0
        self.nodelist = [self.root]

    def insert(self, sequence: List[Tuple[int, float]], weight: int = 1):
        """
        Insert a sequence of tokens into the tree.
            sequence: list of tuples (token_id, q).
            weight: weight to add to the nodes (default: 1)
        """
        node = self.root
        for token_id, q in sequence:
            if token_id not in node.children:
                # Create new node if it doesn't exist
                newnode = Node(token_id, q, tokenizer=self.tokenizer, weight=weight)
                node.children[token_id] = newnode
                newnode.id = len(self.nodelist)
                self.nodelist.append(node.children[token_id])
            else:
                # Node exists, increment weight and assert q value
                node.children[token_id].weight += weight
                assert node.children[token_id].q == q, f"Token ID {token_id} already exists with different q value."
            node = node.children[token_id]

    def get_nodes_by_weight(self, min_weight: int = 1) -> List[Node]:
        """
        Get all nodes with weight >= min_weight.
            min_weight: minimum weight to filter nodes.
        Returns:
            List of nodes with weight >= min_weight.
        """
        return [node for node in self.nodelist if node.weight >= min_weight]

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

            # Sort children by weight (highest first) before traversal
            sorted_children = sorted(node.children.values(), key=lambda x: x.weight, reverse=True)
            for child in sorted_children:
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
        # assert target_prob.shape == (
        #     1,
        #     len(self.nodelist),
        #     len(self.tokenizer),
        # ), f"target_prob shape {target_prob.shape} does not match expected shape {(1, len(self.nodelist), len(self.tokenizer))}"

        accepted_node_ids = []
        longest_accepted_node_ids = []

        def _accept(node):
            nonlocal longest_accepted_node_ids

            accepted_node_ids.append(node.id)
            if len(accepted_node_ids) > len(longest_accepted_node_ids):
                longest_accepted_node_ids = accepted_node_ids.copy()

            # Sort children by weight (highest first) before traversal
            sorted_children = sorted(node.children.items(), key=lambda x: x[1].weight, reverse=True)

            for token_id, child in sorted_children:
                if debug:
                    print(
                        f"child: {child.token_id}, p={target_prob[0, node.id, child.token_id]}, q={child.q}, weight={child.weight}")
                # Scale acceptance probability by weight (optional)
                # weight_factor = min(1.0, child.weight * 0.1)  # Scale factor based on weight
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

    def to_linear_sequence(self, prioritize_weight: bool = True) -> List[int]:
        """
        Convert the tree to a linear sequence using breadth-first traversal.
            prioritize_weight: whether to prioritize nodes with higher weight.
        Returns:
            List of token IDs in breadth-first order.
        """
        sequence = []
        if self.root.token_id is not None:  # Don't include root if it has None token_id
            sequence.append(self.root.token_id)

        queue = [self.root]

        while queue:
            node = queue.pop(0)

            # Get children, sorting by weight if requested
            if prioritize_weight:
                children = sorted(node.children.values(), key=lambda x: x.weight, reverse=True)
            else:
                children = node.children.values()

            for child in children:
                sequence.append(child.token_id)
                queue.append(child)

        return sequence

    def insert_with_original_weight(self, sequence: List[Tuple[int, float]], original_weight: int):
        """
        Insert a sequence of tokens into the tree with the original weight.
        Different from insert() which increments weights of existing nodes.

        Args:
            sequence: list of tuples (token_id, q)
            original_weight: original weight to set (not increment)
        """
        node = self.root
        for token_id, q in sequence:
            if token_id not in node.children:
                # Create new node if it doesn't exist
                newnode = Node(token_id, q, tokenizer=self.tokenizer, weight=original_weight)
                node.children[token_id] = newnode
                newnode.id = len(self.nodelist)
                self.nodelist.append(node.children[token_id])
            else:
                # Node exists, but DON'T increment weight - this is different from insert()
                # Just ensure q value is consistent
                assert node.children[token_id].q == q, f"Token ID {token_id} already exists with different q value."
            node = node.children[token_id]

    def prune_with_beam_search(self, beam_width: int) -> 'TokenTree':
        """
        Prune the tree using beam search algorithm, keeping only the most promising paths.

        Args:
            beam_width: Maximum number of branches to keep at each level

        Returns:
            A new pruned TokenTree
        """
        if beam_width <= 0:
            raise ValueError("Beam width must be positive")

        # Create a new tree with the same root token and tokenizer
        pruned_tree = TokenTree(self.root.token_id, self.tokenizer)

        # If the tree is empty or just has the root, return a copy
        if len(self.nodelist) <= 1:
            return pruned_tree

        # Track current level nodes and their paths from root
        level_nodes = {self.root: []}

        # Keep track of nodes already added to the pruned tree
        added_paths = set()

        # Breadth-first traversal with beam search pruning
        while level_nodes:
            next_level = {}

            # Collect all children of the current level
            all_children = []
            for parent, path in level_nodes.items():
                for token_id, child in parent.children.items():
                    # Score is primarily based on weight, then q value as tiebreaker
                    # Negative q because we want higher probability (lower q) to be preferred
                    score = (child.weight, -child.q if child.q is not None else 0)
                    all_children.append((child, path + [(token_id, child.q)], score))

            # Sort children by score (weight and q)
            all_children.sort(key=lambda x: x[2], reverse=True)

            # Keep only the top-k children according to beam width
            kept_children = all_children[:beam_width]

            # Add kept children to next level
            for child, path, _ in kept_children:
                next_level[child] = path

                # Only add the path to the pruned tree if it hasn't been added yet
                path_tuple = tuple((t, q) for t, q in path)
                if path_tuple not in added_paths:
                    # Copy the original weight instead of incrementing it
                    original_weight = child.weight
                    pruned_tree.insert_with_original_weight(path, original_weight)
                    added_paths.add(path_tuple)

            # Move to next level
            level_nodes = next_level

        return pruned_tree

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
        "is": 0,
        "and": 1,
        "what": 2,
        "need": 3,
        "Mellon": 4,
        "provide": 5,
        "to": 6,
        "'s": 7,
        "it": 8,
        "I": 9,
        "known": 10,
        "about": 11,
    }

    id2token = {v: k for k, v in token2id.items()}

    sentences = [
        [("and", 0.1), ("I", 0.2), ("need", 0.3), ("to", 0.3), ("provide", 0.3)],
        [("and", 0.1), ("I", 0.2), ("need", 0.3), ("to", 0.3), ("provide", 0.3)],
        [("and", 0.1), ("what", 0.2), ("'s", 0.3), ("known", 0.3), ("about", 0.3)],
        [("and", 0.1), ("what", 0.2), ("it", 0.3), ("is", 0.3), ("known", 0.3)],
        [("and", 0.1), ("what", 0.2), ("it", 0.3), ("'s", 0.3), ("known", 0.3)],
    ]

    sentences = [[(token2id[token], q) for token, q in sentence] for sentence in sentences]

    # Create a tree and insert the sequences
    tree = TokenTree(root_token_id=None, tokenizer=id2token)

    # Insert first sequence with weight 1
    tree.insert(sentences[0])

    # Insert second sequence with weight 2
    tree.insert(sentences[1], weight=2)

    # Insert remaining sequences with default weight
    for sentence in sentences[2:]:
        tree.insert(sentence)

    print("Token Tree with Weights:")
    print(tree)

    # # Test getting nodes by weight
    # high_weight_nodes = tree.get_nodes_by_weight(min_weight=2)
    # print("\nNodes with weight >= 2:")
    # for node in high_weight_nodes:
    #     print(node)

    # Test beam search pruning
    beam_width = 2
    print(f"\nPruning tree with beam width {beam_width}:")
    pruned_tree = tree.prune_with_beam_search(beam_width)
    print(pruned_tree)

    # # Test to_linear_sequence with prioritize_weight
    # sequence = tree.to_linear_sequence(prioritize_weight=True)
    # print("\nLinear sequence prioritizing weight:")
    # print([id2token[token_id] for token_id in sequence])
    #
    # # Test attention mask
    # print("\nAttention mask:")
    # tree.attention_mask(debug=True)
