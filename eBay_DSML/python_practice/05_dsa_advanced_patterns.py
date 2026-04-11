"""
eBay DS/ML Interview — Advanced DSA Patterns (Python)
======================================================
Patterns: Trees, Graphs, Linked Lists, Dynamic Programming,
          Tries, Backtracking, Recursion, Bit Manipulation
Each problem has:
  - Question statement
  - Pattern tag
  - Solution with inline comments
  - Time / Space complexity
  - eBay relevance note where applicable
"""

from typing import List, Optional
from collections import defaultdict, deque
import heapq


# ─────────────────────────────────────────────────────────────
# PATTERN 9: LINKED LISTS
# ─────────────────────────────────────────────────────────────

class ListNode:
    def __init__(self, val=0, nxt=None):
        self.val = val
        self.next = nxt

    @staticmethod
    def from_list(arr):
        dummy = ListNode()
        cur = dummy
        for v in arr:
            cur.next = ListNode(v); cur = cur.next
        return dummy.next

    def to_list(self):
        res, cur = [], self
        while cur:
            res.append(cur.val); cur = cur.next
        return res


def reverse_linked_list(head: ListNode) -> ListNode:
    """
    Q: Reverse a singly linked list.
    Pattern: Iterative pointer swap
    TC: O(n)  SC: O(1)
    """
    prev, cur = None, head
    while cur:
        nxt = cur.next      # save next
        cur.next = prev     # reverse pointer
        prev = cur          # advance prev
        cur = nxt           # advance cur
    return prev             # new head


def has_cycle(head: ListNode) -> bool:
    """
    Q: Detect if a linked list has a cycle.
    Pattern: Floyd's Tortoise & Hare (fast/slow pointers)
    TC: O(n)  SC: O(1)
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def find_cycle_start(head: ListNode) -> Optional[ListNode]:
    """
    Q: Return the node where the cycle begins (or None).
    Pattern: Floyd's — after meeting point, reset one pointer to head
    TC: O(n)  SC: O(1)
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            slow = head          # reset slow to head
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return slow          # meeting point = cycle start
    return None


def merge_two_sorted_lists(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Q: Merge two sorted linked lists.
    Pattern: Dummy head + pointer comparison
    TC: O(m+n)  SC: O(1)
    eBay: Merging sorted bid streams from two parallel auction shards.
    """
    dummy = cur = ListNode()
    while l1 and l2:
        if l1.val <= l2.val:
            cur.next = l1; l1 = l1.next
        else:
            cur.next = l2; l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next


def remove_nth_from_end(head: ListNode, n: int) -> ListNode:
    """
    Q: Remove the n-th node from the end in one pass.
    Pattern: Two-pointer gap of n nodes
    TC: O(L)  SC: O(1)
    """
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):       # advance fast by n+1
        fast = fast.next
    while fast:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next   # skip the target node
    return dummy.next


def reorder_list(head: ListNode) -> None:
    """
    Q: Reorder L0→L1→...→Ln to L0→Ln→L1→Ln-1→...
    Pattern: Find mid (slow/fast) → reverse second half → merge
    TC: O(n)  SC: O(1)
    """
    # 1. Find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next; fast = fast.next.next

    # 2. Reverse second half
    prev, cur = None, slow.next
    slow.next = None
    while cur:
        nxt = cur.next; cur.next = prev; prev = cur; cur = nxt
    second = prev

    # 3. Interleave
    first = head
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second; second.next = tmp1
        first = tmp1; second = tmp2


# ─────────────────────────────────────────────────────────────
# PATTERN 10: TREES
# ─────────────────────────────────────────────────────────────

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def from_list(arr):
        """Build tree from BFS-level list (None = absent node)."""
        if not arr:
            return None
        root = TreeNode(arr[0])
        q = deque([root]); i = 1
        while q and i < len(arr):
            node = q.popleft()
            if i < len(arr) and arr[i] is not None:
                node.left = TreeNode(arr[i]); q.append(node.left)
            i += 1
            if i < len(arr) and arr[i] is not None:
                node.right = TreeNode(arr[i]); q.append(node.right)
            i += 1
        return root


def inorder(root: TreeNode) -> List[int]:
    """Left → Root → Right. TC: O(n), SC: O(h)"""
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []


def level_order(root: TreeNode) -> List[List[int]]:
    """
    Q: Level-order (BFS) traversal, grouped by level.
    TC: O(n)  SC: O(n)
    eBay: Useful for category hierarchy tree traversal.
    """
    if not root:
        return []
    result, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        result.append(level)
    return result


def max_depth(root: TreeNode) -> int:
    """
    Q: Maximum depth of a binary tree.
    Pattern: DFS recursion
    TC: O(n)  SC: O(h)
    """
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def is_balanced(root: TreeNode) -> bool:
    """
    Q: Check if binary tree is height-balanced (|left_h - right_h| <= 1).
    Pattern: Post-order DFS; return -1 as sentinel for unbalanced.
    TC: O(n)  SC: O(h)
    """
    def check(node):
        if not node:
            return 0
        lh = check(node.left)
        if lh == -1: return -1
        rh = check(node.right)
        if rh == -1: return -1
        if abs(lh - rh) > 1: return -1
        return 1 + max(lh, rh)

    return check(root) != -1


def diameter_of_binary_tree(root: TreeNode) -> int:
    """
    Q: Longest path between any two nodes (may not pass through root).
    Pattern: Post-order DFS; track global max
    TC: O(n)  SC: O(h)
    """
    ans = [0]
    def depth(node):
        if not node: return 0
        l, r = depth(node.left), depth(node.right)
        ans[0] = max(ans[0], l + r)   # path through this node
        return 1 + max(l, r)
    depth(root)
    return ans[0]


def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Q: Find LCA of nodes p and q in a binary tree.
    Pattern: Post-order DFS — if both subtrees return non-null, current node is LCA.
    TC: O(n)  SC: O(h)
    eBay: LCA of two category nodes in an eBay taxonomy tree.
    """
    if not root or root is p or root is q:
        return root
    left  = lowest_common_ancestor(root.left,  p, q)
    right = lowest_common_ancestor(root.right, p, q)
    return root if left and right else (left or right)


def right_side_view(root: TreeNode) -> List[int]:
    """
    Q: Return values of nodes visible from the right side (rightmost per level).
    Pattern: BFS, take last element of each level
    TC: O(n)  SC: O(n)
    """
    if not root: return []
    result, q = [], deque([root])
    while q:
        for i in range(len(q)):
            node = q.popleft()
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
            if i == len(q): result.append(node.val)   # last of level
    # simpler:
    result2 = []
    q = deque([root])
    while q:
        level_size = len(q)
        for i in range(level_size):
            node = q.popleft()
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
            if i == level_size - 1:
                result2.append(node.val)
    return result2


def path_sum_ii(root: TreeNode, target: int) -> List[List[int]]:
    """
    Q: Find ALL root-to-leaf paths that sum to target.
    Pattern: DFS backtracking with path tracking
    TC: O(n)  SC: O(h)
    """
    result = []
    def dfs(node, remaining, path):
        if not node:
            return
        path.append(node.val)
        if not node.left and not node.right and remaining == node.val:
            result.append(list(path))    # found a valid path
        else:
            dfs(node.left,  remaining - node.val, path)
            dfs(node.right, remaining - node.val, path)
        path.pop()                       # backtrack
    dfs(root, target, [])
    return result


def serialize_deserialize(root: TreeNode):
    """
    Q: Serialize binary tree to string and deserialize back.
    Pattern: BFS encoding with '#' as null marker
    TC: O(n)  SC: O(n)
    eBay: Store/restore category tree snapshots.
    """
    # Serialize
    def serialize(node):
        if not node: return '#'
        return f"{node.val},{serialize(node.left)},{serialize(node.right)}"

    # Deserialize
    def deserialize(data):
        vals = iter(data.split(','))
        def build():
            v = next(vals)
            if v == '#': return None
            node = TreeNode(int(v))
            node.left  = build()
            node.right = build()
            return node
        return build()

    s = serialize(root)
    return deserialize(s)


# ─────────────────────────────────────────────────────────────
# PATTERN 11: GRAPHS
# ─────────────────────────────────────────────────────────────

def num_islands(grid: List[List[str]]) -> int:
    """
    Q: Count the number of islands ('1's surrounded by '0's/boundary).
    Pattern: DFS flood-fill
    TC: O(m*n)  SC: O(m*n) stack
    eBay: Count isolated seller clusters in a marketplace graph.
    """
    if not grid: return 0
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'     # mark visited
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            dfs(r+dr, c+dc)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    return count


def clone_graph(node):
    """
    Q: Deep clone an undirected graph.
    Pattern: BFS + visited hashmap (original → clone)
    TC: O(V+E)  SC: O(V)
    """
    if not node: return None
    clones = {node: type(node)(node.val)}
    q = deque([node])
    while q:
        cur = q.popleft()
        for nbr in cur.neighbors:
            if nbr not in clones:
                clones[nbr] = type(nbr)(nbr.val)
                q.append(nbr)
            clones[cur].neighbors.append(clones[nbr])
    return clones[node]


def course_schedule(numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Q: Can you finish all courses given prerequisites? (Detect cycle in a DAG)
    Pattern: Topological sort — Kahn's algorithm (BFS with in-degree)
    TC: O(V+E)  SC: O(V+E)
    eBay: Validate that a listing processing pipeline has no circular dependencies.
    """
    graph   = defaultdict(list)
    in_deg  = [0] * numCourses
    for dest, src in prerequisites:
        graph[src].append(dest)
        in_deg[dest] += 1

    q = deque(c for c in range(numCourses) if in_deg[c] == 0)
    completed = 0
    while q:
        node = q.popleft()
        completed += 1
        for nbr in graph[node]:
            in_deg[nbr] -= 1
            if in_deg[nbr] == 0:
                q.append(nbr)
    return completed == numCourses


def course_schedule_ii(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Q: Return one valid course ordering (topological sort), or [] if impossible.
    Pattern: Kahn's BFS — same as above but collect ordering.
    TC: O(V+E)  SC: O(V+E)
    """
    graph  = defaultdict(list)
    in_deg = [0] * numCourses
    for dest, src in prerequisites:
        graph[src].append(dest)
        in_deg[dest] += 1

    q      = deque(c for c in range(numCourses) if in_deg[c] == 0)
    order  = []
    while q:
        node = q.popleft()
        order.append(node)
        for nbr in graph[node]:
            in_deg[nbr] -= 1
            if in_deg[nbr] == 0:
                q.append(nbr)
    return order if len(order) == numCourses else []


def dijkstra(graph: dict, start: int) -> dict:
    """
    Q: Shortest path from source to all nodes (non-negative weights).
    Pattern: Dijkstra with min-heap
    TC: O((V+E) log V)  SC: O(V)
    eBay: Find shortest delivery route between warehouses.
    """
    dist = defaultdict(lambda: float('inf'))
    dist[start] = 0
    heap = [(0, start)]    # (distance, node)

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:    # stale entry
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dict(dist)


def number_of_connected_components(n: int, edges: List[List[int]]) -> int:
    """
    Q: Count connected components in an undirected graph.
    Pattern: Union-Find (Disjoint Set Union)
    TC: O(E * α(n)) ≈ O(E)  SC: O(n)
    eBay: Count isolated buyer/seller network clusters.
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return False
        parent[ra] = rb
        return True

    components = n
    for a, b in edges:
        if union(a, b):
            components -= 1
    return components


def pacific_atlantic_water_flow(heights: List[List[int]]) -> List[List[int]]:
    """
    Q: Return all cells that can flow to both Pacific and Atlantic oceans.
    Pattern: Reverse BFS from both ocean boundaries simultaneously.
    TC: O(m*n)  SC: O(m*n)
    """
    rows, cols = len(heights), len(heights[0])
    dirs = [(0,1),(0,-1),(1,0),(-1,0)]

    def bfs(starts):
        visited = set(starts)
        q = deque(starts)
        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and (nr, nc) not in visited
                        and heights[nr][nc] >= heights[r][c]):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return visited

    pacific  = bfs([(r, 0) for r in range(rows)] + [(0, c) for c in range(cols)])
    atlantic = bfs([(r, cols-1) for r in range(rows)] + [(rows-1, c) for c in range(cols)])
    return [[r, c] for r, c in pacific & atlantic]


# ─────────────────────────────────────────────────────────────
# PATTERN 12: DYNAMIC PROGRAMMING — CLASSIC DP
# ─────────────────────────────────────────────────────────────

def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Q: Length of longest common subsequence between two strings.
    Pattern: 2D DP table
    TC: O(m*n)  SC: O(m*n)
    eBay: Similarity matching between listing titles for duplicate detection.
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    Q: Length of longest strictly increasing subsequence.
    Pattern: Patience sorting + binary search
    TC: O(n log n)  SC: O(n)
    """
    import bisect
    tails = []
    for n in nums:
        pos = bisect.bisect_left(tails, n)
        if pos == len(tails):
            tails.append(n)
        else:
            tails[pos] = n
    return len(tails)


def word_break(s: str, wordDict: List[str]) -> bool:
    """
    Q: Can string s be segmented into space-separated dictionary words?
    Pattern: 1D DP — dp[i] = can we form s[:i]
    TC: O(n^2)  SC: O(n)
    eBay: Check if a product query can be tokenised into known category words.
    """
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True                        # empty string is always valid

    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]


def unique_paths(m: int, n: int) -> int:
    """
    Q: Number of unique paths from top-left to bottom-right (only right/down).
    Pattern: 2D DP (optimized to 1D)
    TC: O(m*n)  SC: O(n)
    """
    dp = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    return dp[-1]


def edit_distance(word1: str, word2: str) -> int:
    """
    Q: Minimum insert/delete/replace operations to transform word1 → word2.
    Pattern: 2D DP (Levenshtein Distance)
    TC: O(m*n)  SC: O(m*n)
    eBay: Fuzzy title matching for listing deduplication.
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # delete
                                   dp[i][j-1],    # insert
                                   dp[i-1][j-1])  # replace
    return dp[m][n]


def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Q: 0/1 Knapsack — max value with weight constraint (each item used once).
    Pattern: 2D DP (rows=items, cols=capacity); iterate capacity in reverse for 1D.
    TC: O(n*W)  SC: O(W)
    eBay: Select listings to feature in a homepage ad slot given a relevance budget.
    """
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):    # iterate right-to-left to avoid reuse
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]


def partition_equal_subset_sum(nums: List[int]) -> bool:
    """
    Q: Can nums be partitioned into two subsets with equal sum?
    Pattern: 0/1 Knapsack variant — find if subset sums to total/2
    TC: O(n*S)  SC: O(S) where S = sum(nums)
    """
    total = sum(nums)
    if total % 2: return False
    target = total // 2
    dp = {0}
    for n in nums:
        dp |= {s + n for s in dp if s + n <= target}
    return target in dp


def house_robber(nums: List[int]) -> int:
    """
    Q: Max money you can rob without robbing adjacent houses.
    Pattern: DP with two variables
    TC: O(n)  SC: O(1)
    """
    prev2 = prev1 = 0
    for n in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + n)
    return prev1


def house_robber_ii(nums: List[int]) -> int:
    """
    Q: Same as above but houses arranged in a circle.
    Pattern: Run house_robber twice — skip first or skip last.
    TC: O(n)  SC: O(1)
    """
    def rob(arr):
        p2 = p1 = 0
        for n in arr:
            p2, p1 = p1, max(p1, p2 + n)
        return p1

    return max(rob(nums[1:]), rob(nums[:-1]))


# ─────────────────────────────────────────────────────────────
# PATTERN 13: BACKTRACKING
# ─────────────────────────────────────────────────────────────

def permutations(nums: List[int]) -> List[List[int]]:
    """
    Q: Return all permutations of a list of distinct integers.
    Pattern: Backtracking — swap & recurse
    TC: O(n! * n)  SC: O(n)
    """
    result = []
    def bt(start):
        if start == len(nums):
            result.append(nums[:])
            return
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            bt(start + 1)
            nums[start], nums[i] = nums[i], nums[start]   # backtrack
    bt(0)
    return result


def subsets(nums: List[int]) -> List[List[int]]:
    """
    Q: Return all possible subsets (power set).
    Pattern: Backtracking — include or exclude each element
    TC: O(2^n * n)  SC: O(n)
    eBay: Generate all possible feature-flag combinations for a product test.
    """
    result = []
    def bt(start, current):
        result.append(list(current))
        for i in range(start, len(nums)):
            current.append(nums[i])
            bt(i + 1, current)
            current.pop()
    bt(0, [])
    return result


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Q: All combinations that sum to target (reuse allowed).
    Pattern: Backtracking with pruning
    TC: O(2^t)  SC: O(t/min_candidate) where t = target
    """
    result = []
    candidates.sort()
    def bt(start, remaining, path):
        if remaining == 0:
            result.append(list(path))
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break                         # pruning
            path.append(candidates[i])
            bt(i, remaining - candidates[i], path)    # i (not i+1) allows reuse
            path.pop()
    bt(0, target, [])
    return result


def word_search(board: List[List[str]], word: str) -> bool:
    """
    Q: Search for word in a 2D character grid (adjacent cells, no reuse).
    Pattern: DFS backtracking with visited marking
    TC: O(m*n * 4^L) where L = len(word)  SC: O(L)
    """
    rows, cols = len(board), len(board[0])
    def dfs(r, c, idx):
        if idx == len(word): return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[idx]:
            return False
        tmp, board[r][c] = board[r][c], '#'     # mark visited
        found = any(dfs(r+dr, c+dc, idx+1) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])
        board[r][c] = tmp                        # restore
        return found

    return any(dfs(r, c, 0) for r in range(rows) for c in range(cols))


def n_queens(n: int) -> List[List[str]]:
    """
    Q: Place n queens on n×n board so no two attack. Return all solutions.
    Pattern: Backtracking with column/diagonal sets
    TC: O(n!)  SC: O(n)
    """
    result = []
    cols = set(); diag1 = set(); diag2 = set()

    def bt(row, board):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row-col) in diag1 or (row+col) in diag2:
                continue
            cols.add(col); diag1.add(row-col); diag2.add(row+col)
            board[row][col] = 'Q'
            bt(row + 1, board)
            board[row][col] = '.'; cols.discard(col)
            diag1.discard(row-col); diag2.discard(row+col)

    bt(0, [['.']*n for _ in range(n)])
    return result


def generate_parentheses(n: int) -> List[str]:
    """
    Q: Generate all valid combinations of n pairs of parentheses.
    Pattern: Backtracking — only add '(' if open < n, ')' if close < open
    TC: O(4^n / sqrt(n)) (Catalan number)  SC: O(n)
    """
    result = []
    def bt(s, open_cnt, close_cnt):
        if len(s) == 2 * n:
            result.append(s); return
        if open_cnt < n:
            bt(s + '(', open_cnt + 1, close_cnt)
        if close_cnt < open_cnt:
            bt(s + ')', open_cnt, close_cnt + 1)
    bt('', 0, 0)
    return result


# ─────────────────────────────────────────────────────────────
# PATTERN 14: TRIE (PREFIX TREE)
# ─────────────────────────────────────────────────────────────

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end   = False


class Trie:
    """
    Q: Implement a Trie (insert, search, startsWith).
    TC: O(L) per operation where L = word length
    eBay: Power eBay's search autocomplete (prefix suggestions for product queries).
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children: return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node.children: return False
            node = node.children[ch]
        return True

    def autocomplete(self, prefix: str) -> List[str]:
        """Return all words starting with the given prefix."""
        node = self.root
        for ch in prefix:
            if ch not in node.children: return []
            node = node.children[ch]

        results = []
        def dfs(n, path):
            if n.is_end: results.append(prefix + path)
            for ch, child in n.children.items():
                dfs(child, path + ch)
        dfs(node, '')
        return results


def find_words_on_board(board: List[List[str]], words: List[str]) -> List[str]:
    """
    Q: Find all words from a dictionary that exist in a 2D character board.
    Pattern: Trie + DFS backtracking — classic "Word Search II"
    TC: O(m*n*4^L) worst case, but Trie prunes aggressively  SC: O(W*L)
    eBay: Search for product tags matching any of thousands of category keywords.
    """
    # Build trie
    root = TrieNode()
    for word in words:
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    rows, cols = len(board), len(board[0])
    result = []

    def dfs(r, c, node, path):
        ch = board[r][c]
        if ch not in node.children: return
        nxt = node.children[ch]
        path += ch
        if nxt.is_end:
            result.append(path)
            nxt.is_end = False      # de-duplicate

        board[r][c] = '#'           # mark visited
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                dfs(nr, nc, nxt, path)
        board[r][c] = ch            # restore

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root, '')
    return result


# ─────────────────────────────────────────────────────────────
# PATTERN 15: BIT MANIPULATION
# ─────────────────────────────────────────────────────────────

def single_number(nums: List[int]) -> int:
    """
    Q: Every element appears twice except one. Find that element.
    Pattern: XOR — a^a=0, a^0=a
    TC: O(n)  SC: O(1)
    """
    result = 0
    for n in nums:
        result ^= n
    return result


def count_bits(n: int) -> List[int]:
    """
    Q: Return array[0..n] where array[i] = number of 1 bits in i.
    Pattern: DP — dp[i] = dp[i >> 1] + (i & 1)
    TC: O(n)  SC: O(n)
    """
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = dp[i >> 1] + (i & 1)
    return dp


def is_power_of_two(n: int) -> bool:
    """
    Q: Check if n is a power of two.
    Pattern: Bit trick — n & (n-1) clears lowest set bit; POT has exactly one set bit.
    TC: O(1)  SC: O(1)
    """
    return n > 0 and (n & (n - 1)) == 0


def missing_number(nums: List[int]) -> int:
    """
    Q: Find the missing number in range [0, n].
    Pattern: XOR or Gauss formula (n*(n+1)//2 - sum)
    TC: O(n)  SC: O(1)
    """
    return sum(range(len(nums) + 1)) - sum(nums)


def reverse_bits(n: int) -> int:
    """
    Q: Reverse the 32-bit binary representation of n.
    TC: O(32)  SC: O(1)
    """
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


# ─────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # --- Linked List ---
    head = ListNode.from_list([1, 2, 3, 4, 5])
    rev  = reverse_linked_list(head)

    l1 = ListNode.from_list([1, 2, 4])
    l2 = ListNode.from_list([1, 3, 4])
    merged = merge_two_sorted_lists(l1, l2)

    ll_tests = [
        ("Reverse Linked List",    rev.to_list() == [5, 4, 3, 2, 1]),
        ("Merge Two Sorted Lists", merged.to_list() == [1, 1, 2, 3, 4, 4]),
        ("Remove Nth From End",    remove_nth_from_end(ListNode.from_list([1,2,3,4,5]), 2)
                                       .to_list() == [1, 2, 3, 5]),
    ]

    # --- Tree ---
    root = TreeNode.from_list([3, 9, 20, None, None, 15, 7])
    tree_tests = [
        ("Level Order",           level_order(root) == [[3],[9,20],[15,7]]),
        ("Max Depth",             max_depth(root) == 3),
        ("Is Balanced",           is_balanced(TreeNode.from_list([1,2,2,3,3,None,None,4,4]))==False),
        ("Diameter",              diameter_of_binary_tree(TreeNode.from_list([1,2,3,4,5])) == 3),
        ("Right Side View",       right_side_view(root) == [3, 20, 7]),
        ("Path Sum II",           path_sum_ii(TreeNode.from_list([5,4,8,11,None,13,4,7,2,None,None,5,1]),22)
                                      == [[5,4,11,2],[5,8,4,5]]),
    ]

    # --- Graph ---
    grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
    graph_tests = [
        ("Num Islands",           num_islands(grid) == 1),
        ("Course Schedule",       course_schedule(2, [[1,0]]) == True),
        ("Course Schedule II",    course_schedule_ii(4, [[1,0],[2,0],[3,1],[3,2]]) == [0,1,2,3] or len(course_schedule_ii(4, [[1,0],[2,0],[3,1],[3,2]])) == 4),
        ("Num Components (Union-Find)", number_of_connected_components(5, [[0,1],[1,2],[3,4]]) == 2),
        ("Dijkstra",              dijkstra({0:[(1,4),(2,1)], 1:[(3,1)], 2:[(1,2),(3,5)], 3:[]}, 0) == {0:0,1:3,2:1,3:4}),
    ]

    # --- DP ---
    dp_tests = [
        ("LCS",                   longest_common_subsequence("abcde","ace") == 3),
        ("LIS",                   longest_increasing_subsequence([10,9,2,5,3,7,101,18]) == 4),
        ("Word Break",            word_break("leetcode", ["leet","code"]) == True),
        ("Unique Paths",          unique_paths(3, 7) == 28),
        ("Edit Distance",         edit_distance("horse","ros") == 3),
        ("Knapsack 0/1",          knapsack_01([1,3,4,5],[1,4,5,7],7) == 9),
        ("Partition Equal Sum",   partition_equal_subset_sum([1,5,11,5]) == True),
        ("House Robber",          house_robber([2,7,9,3,1]) == 12),
        ("House Robber II",       house_robber_ii([2,3,2]) == 3),
    ]

    # --- Backtracking ---
    bt_tests = [
        ("Permutations count",    len(permutations([1,2,3])) == 6),
        ("Subsets count",         len(subsets([1,2,3])) == 8),
        ("Combination Sum",       combination_sum([2,3,6,7],7) == [[2,2,3],[7]]),
        ("Generate Parentheses",  sorted(generate_parentheses(3)) == sorted(["((()))","(()())","(())()","()(())","()()()"])),
        ("N-Queens (1 soln)",     len(n_queens(4)) == 2),
        ("Word Search",           word_search([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]],"ABCCED") == True),
    ]

    # --- Trie ---
    t = Trie()
    for w in ["iphone","ipad","macbook","mac","ebay","ecommerce"]:
        t.insert(w)
    trie_tests = [
        ("Trie Search",           t.search("iphone") == True),
        ("Trie Search miss",      t.search("ipho")   == False),
        ("Trie Prefix",           t.starts_with("mac") == True),
        ("Trie Autocomplete",     sorted(t.autocomplete("e")) == sorted(["ebay","ecommerce"])),
    ]

    # --- Bit Manipulation ---
    bit_tests = [
        ("Single Number",         single_number([2,2,1]) == 1),
        ("Count Bits",            count_bits(5) == [0,1,1,2,1,2]),
        ("Power of Two",          is_power_of_two(16) == True and is_power_of_two(3) == False),
        ("Missing Number",        missing_number([3,0,1]) == 2),
        ("Reverse Bits",          reverse_bits(0b00000010100101000001111010011100) == 964176192),
    ]

    # ─── Print results ───
    all_tests = [
        ("LINKED LISTS", ll_tests),
        ("TREES",         tree_tests),
        ("GRAPHS",        graph_tests),
        ("DYNAMIC PROG",  dp_tests),
        ("BACKTRACKING",  bt_tests),
        ("TRIE",          trie_tests),
        ("BIT MANIP",     bit_tests),
    ]

    print("=" * 65)
    print("  eBay DSA Advanced Patterns — Full Test Suite")
    print("=" * 65)
    total = passed = 0
    for section, tests in all_tests:
        print(f"\n  ── {section} ──")
        for name, ok in tests:
            icon = "✅" if ok else "❌"
            print(f"    {icon} {name}")
            total += 1
            passed += ok
    print(f"\n  {'='*40}")
    print(f"  Total: {passed}/{total} passed")
    print("=" * 65)
