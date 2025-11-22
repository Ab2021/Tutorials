import os

def create_lab_content(title, difficulty, time, objectives, problem, starter_code):
    return f"""# {title}

## Difficulty
{difficulty}

## Estimated Time
{time}

## Learning Objectives
{objectives}

## Problem Statement
{problem}

## Starter Code
```python
{starter_code}
```

## Hints
<details>
<summary>Hint 1</summary>
Focus on the core logic first.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>
Solution will be provided after you attempt the problem.
</details>
"""

def update_lab(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated {path}")

# ==============================================================================
# DATA STRUCTURES & ALGORITHMS (DSA)
# ==============================================================================
dsa_root = r"G:\My Drive\Codes & Repos\DSA_Learning_Course"

dsa_content = {
    "Phase1_Foundations/Week2_LinkedLists_Stacks": [
        ("Lab 01: Reverse Linked List", "🟢 Easy", "30 mins", "- Pointer manipulation\n- In-place reversal", "Reverse a singly linked list iteratively and recursively.", "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverseList(head):\n    pass"),
        ("Lab 02: Detect Cycle", "🟢 Easy", "30 mins", "- Floyd's Cycle Finding Algorithm", "Determine if a linked list has a cycle in it.", "def hasCycle(head):\n    pass"),
        ("Lab 03: Merge Two Sorted Lists", "🟢 Easy", "30 mins", "- Merge logic", "Merge two sorted linked lists and return it as a sorted list.", "def mergeTwoLists(list1, list2):\n    pass"),
        ("Lab 04: Valid Parentheses", "🟢 Easy", "30 mins", "- Stack usage", "Given a string containing '(', ')', '{', '}', '[' and ']', determine if the input string is valid.", "def isValid(s):\n    pass"),
        ("Lab 05: Min Stack", "🟡 Medium", "45 mins", "- Stack design\n- Auxiliary stack", "Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.", "class MinStack:\n    def __init__(self):\n        pass"),
    ],
    "Phase1_Foundations/Week3_Queues_Hashing": [
        ("Lab 01: Implement Queue using Stacks", "🟢 Easy", "30 mins", "- Queue operations", "Implement a first in first out (FIFO) queue using only two stacks.", "class MyQueue:\n    pass"),
        ("Lab 02: Two Sum", "🟢 Easy", "30 mins", "- Hash Map", "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.", "def twoSum(nums, target):\n    pass"),
        ("Lab 03: Group Anagrams", "🟡 Medium", "45 mins", "- String hashing", "Given an array of strings, group the anagrams together.", "def groupAnagrams(strs):\n    pass"),
        ("Lab 04: Longest Substring Without Repeating Characters", "🟡 Medium", "45 mins", "- Sliding Window\n- Hash Set", "Find the length of the longest substring without repeating characters.", "def lengthOfLongestSubstring(s):\n    pass"),
        ("Lab 05: Top K Frequent Elements", "🟡 Medium", "45 mins", "- Heap or Bucket Sort", "Given an integer array nums and an integer k, return the k most frequent elements.", "def topKFrequent(nums, k):\n    pass"),
    ],
    "Phase2_Trees_Graphs/Week4_Binary_Trees": [
        ("Lab 01: Maximum Depth of Binary Tree", "🟢 Easy", "20 mins", "- Recursion (DFS)", "Find the maximum depth of a binary tree.", "def maxDepth(root):\n    pass"),
        ("Lab 02: Invert Binary Tree", "🟢 Easy", "20 mins", "- Tree Traversal", "Invert a binary tree (mirror image).", "def invertTree(root):\n    pass"),
        ("Lab 03: Symmetric Tree", "🟢 Easy", "30 mins", "- Recursion", "Check if a binary tree is a mirror of itself.", "def isSymmetric(root):\n    pass"),
        ("Lab 04: Level Order Traversal", "🟡 Medium", "40 mins", "- BFS (Queue)", "Return the level order traversal of its nodes' values.", "def levelOrder(root):\n    pass"),
        ("Lab 05: Validate BST", "🟡 Medium", "45 mins", "- BST Properties", "Determine if a binary tree is a valid binary search tree.", "def isValidBST(root):\n    pass"),
    ],
    "Phase2_Trees_Graphs/Week5_Advanced_Trees": [
        ("Lab 01: Lowest Common Ancestor", "🟡 Medium", "45 mins", "- Recursion", "Find the lowest common ancestor (LCA) of two nodes in a BST.", "def lowestCommonAncestor(root, p, q):\n    pass"),
        ("Lab 02: Kth Smallest Element in BST", "🟡 Medium", "40 mins", "- In-order Traversal", "Find the kth smallest element in a BST.", "def kthSmallest(root, k):\n    pass"),
        ("Lab 03: Construct Binary Tree from Preorder and Inorder", "🟡 Medium", "60 mins", "- Tree Construction", "Construct a binary tree given preorder and inorder traversal arrays.", "def buildTree(preorder, inorder):\n    pass"),
        ("Lab 04: Serialize and Deserialize Binary Tree", "🔴 Hard", "60 mins", "- Serialization", "Design an algorithm to serialize and deserialize a binary tree.", "class Codec:\n    def serialize(self, root):\n        pass\n    def deserialize(self, data):\n        pass"),
        ("Lab 05: Binary Tree Maximum Path Sum", "🔴 Hard", "60 mins", "- DFS", "Find the maximum path sum in a binary tree (path can start and end anywhere).", "def maxPathSum(root):\n    pass"),
    ],
    "Phase2_Trees_Graphs/Week6_Graph_Fundamentals": [
        ("Lab 01: Number of Islands", "🟡 Medium", "45 mins", "- DFS/BFS", "Count the number of islands in a 2D grid map of '1's (land) and '0's (water).", "def numIslands(grid):\n    pass"),
        ("Lab 02: Clone Graph", "🟡 Medium", "45 mins", "- Graph Traversal", "Return a deep copy (clone) of the graph.", "def cloneGraph(node):\n    pass"),
        ("Lab 03: Course Schedule", "🟡 Medium", "50 mins", "- Topological Sort", "Determine if you can finish all courses given prerequisites.", "def canFinish(numCourses, prerequisites):\n    pass"),
        ("Lab 04: Pacific Atlantic Water Flow", "🟡 Medium", "50 mins", "- DFS/BFS", "Find grid coordinates where water flows to both Pacific and Atlantic oceans.", "def pacificAtlantic(heights):\n    pass"),
        ("Lab 05: Number of Connected Components", "🟡 Medium", "40 mins", "- Union Find", "Count connected components in an undirected graph.", "def countComponents(n, edges):\n    pass"),
    ],
    "Phase3_Advanced_Algorithms/Week7_Sorting_Searching": [
        ("Lab 01: Merge Intervals", "🟡 Medium", "45 mins", "- Sorting", "Merge all overlapping intervals.", "def merge(intervals):\n    pass"),
        ("Lab 02: Search in Rotated Sorted Array", "🟡 Medium", "45 mins", "- Binary Search", "Search for a target value in a rotated sorted array.", "def search(nums, target):\n    pass"),
        ("Lab 03: Find Minimum in Rotated Sorted Array", "🟡 Medium", "40 mins", "- Binary Search", "Find the minimum element in a rotated sorted array.", "def findMin(nums):\n    pass"),
        ("Lab 04: Median of Two Sorted Arrays", "🔴 Hard", "60 mins", "- Binary Search", "Find the median of two sorted arrays of different sizes.", "def findMedianSortedArrays(nums1, nums2):\n    pass"),
        ("Lab 05: Kth Largest Element in an Array", "🟡 Medium", "40 mins", "- QuickSelect / Heap", "Find the kth largest element in an unsorted array.", "def findKthLargest(nums, k):\n    pass"),
    ],
    "Phase3_Advanced_Algorithms/Week8_Dynamic_Programming_I": [
        ("Lab 01: Climbing Stairs", "🟢 Easy", "30 mins", "- DP Fundamentals", "Count distinct ways to climb to the top.", "def climbStairs(n):\n    pass"),
        ("Lab 02: Coin Change", "🟡 Medium", "45 mins", "- DP (Knapsack)", "Compute the fewest number of coins that you need to make up that amount.", "def coinChange(coins, amount):\n    pass"),
        ("Lab 03: Longest Increasing Subsequence", "🟡 Medium", "45 mins", "- DP", "Find the length of the longest strictly increasing subsequence.", "def lengthOfLIS(nums):\n    pass"),
        ("Lab 04: Longest Common Subsequence", "🟡 Medium", "45 mins", "- 2D DP", "Find the length of the longest common subsequence of two strings.", "def longestCommonSubsequence(text1, text2):\n    pass"),
        ("Lab 05: Word Break", "🟡 Medium", "50 mins", "- DP", "Determine if a string can be segmented into a space-separated sequence of dictionary words.", "def wordBreak(s, wordDict):\n    pass"),
    ],
    "Phase3_Advanced_Algorithms/Week9_Dynamic_Programming_II": [
        ("Lab 01: Unique Paths", "🟡 Medium", "40 mins", "- 2D DP", "Find the number of unique paths from top-left to bottom-right.", "def uniquePaths(m, n):\n    pass"),
        ("Lab 02: House Robber II", "🟡 Medium", "45 mins", "- DP", "Maximize amount you can rob from circular houses.", "def rob(nums):\n    pass"),
        ("Lab 03: Edit Distance", "🔴 Hard", "60 mins", "- 2D DP", "Find minimum operations to convert word1 to word2.", "def minDistance(word1, word2):\n    pass"),
        ("Lab 04: Burst Balloons", "🔴 Hard", "60 mins", "- Interval DP", "Maximize coins collected by bursting balloons.", "def maxCoins(nums):\n    pass"),
        ("Lab 05: Palindromic Substrings", "🟡 Medium", "40 mins", "- DP / Expansion", "Count how many palindromic substrings are in the string.", "def countSubstrings(s):\n    pass"),
    ],
    "Phase4_Advanced_Topics/Week10_Greedy_Backtracking": [
        ("Lab 01: Jump Game", "🟡 Medium", "40 mins", "- Greedy", "Determine if you can reach the last index.", "def canJump(nums):\n    pass"),
        ("Lab 02: Permutations", "🟡 Medium", "45 mins", "- Backtracking", "Return all possible permutations of distinct integers.", "def permute(nums):\n    pass"),
        ("Lab 03: Subsets", "🟡 Medium", "40 mins", "- Backtracking", "Return all possible subsets (the power set).", "def subsets(nums):\n    pass"),
        ("Lab 04: Combination Sum", "🟡 Medium", "45 mins", "- Backtracking", "Return a list of all unique combinations of candidates where the chosen numbers sum to target.", "def combinationSum(candidates, target):\n    pass"),
        ("Lab 05: N-Queens", "🔴 Hard", "60 mins", "- Backtracking", "Place N queens on an NxN chessboard such that no two queens attack each other.", "def solveNQueens(n):\n    pass"),
    ],
    "Phase4_Advanced_Topics/Week11_Advanced_Graph_Algorithms": [
        ("Lab 01: Network Delay Time", "🟡 Medium", "50 mins", "- Dijkstra", "Find the time it takes for all nodes to receive a signal.", "def networkDelayTime(times, n, k):\n    pass"),
        ("Lab 02: Cheapest Flights Within K Stops", "🟡 Medium", "50 mins", "- Bellman-Ford / BFS", "Find the cheapest price from src to dst with at most k stops.", "def findCheapestPrice(n, flights, src, dst, k):\n    pass"),
        ("Lab 03: Alien Dictionary", "🔴 Hard", "60 mins", "- Topological Sort", "Derive the order of letters in an alien language.", "def alienOrder(words):\n    pass"),
        ("Lab 04: Redundant Connection", "🟡 Medium", "45 mins", "- Union Find", "Find an edge that can be removed to make the graph a tree.", "def findRedundantConnection(edges):\n    pass"),
        ("Lab 05: Word Ladder", "🔴 Hard", "60 mins", "- BFS", "Find the length of shortest transformation sequence from beginWord to endWord.", "def ladderLength(beginWord, endWord, wordList):\n    pass"),
    ],
    "Phase4_Advanced_Topics/Week12_String_Algorithms": [
        ("Lab 01: Implement Trie (Prefix Tree)", "🟡 Medium", "45 mins", "- Trie", "Implement a Trie with insert, search, and startsWith methods.", "class Trie:\n    pass"),
        ("Lab 02: Longest Palindromic Substring", "🟡 Medium", "45 mins", "- Expansion", "Find the longest palindromic substring in s.", "def longestPalindrome(s):\n    pass"),
        ("Lab 03: Word Search II", "🔴 Hard", "60 mins", "- Trie + DFS", "Find all words from the board that exist in the dictionary.", "def findWords(board, words):\n    pass"),
        ("Lab 04: Minimum Window Substring", "🔴 Hard", "60 mins", "- Sliding Window", "Find the minimum window in s which will contain all the characters in t.", "def minWindow(s, t):\n    pass"),
        ("Lab 05: Encode and Decode Strings", "🟡 Medium", "40 mins", "- String Manipulation", "Design an algorithm to encode a list of strings to a string and decode it back.", "class Codec:\n    pass"),
    ],
}

# ==============================================================================
# COMPUTER VISION (CV)
# ==============================================================================
cv_root = r"G:\My Drive\Codes & Repos\Computer_Vision_Course"

cv_content = {
    "Phase1_Foundations/Week2_DeepLearning": [
        ("Lab 01: Perceptron Implementation", "🟢 Easy", "45 mins", "- Neural Network Basics", "Implement a single-layer perceptron from scratch to solve logic gates (AND, OR).", "class Perceptron:\n    pass"),
        ("Lab 02: Backpropagation from Scratch", "🔴 Hard", "90 mins", "- Gradients", "Implement the backpropagation algorithm for a simple MLP using only NumPy.", "def backward_pass(y_true, y_pred, cache):\n    pass"),
        ("Lab 03: Activation Functions", "🟢 Easy", "30 mins", "- Non-linearity", "Implement Sigmoid, Tanh, and ReLU functions and their derivatives.", "def sigmoid(x):\n    pass"),
        ("Lab 04: Loss Functions", "🟢 Easy", "30 mins", "- Optimization", "Implement MSE and Cross-Entropy Loss functions.", "def cross_entropy(y_true, y_pred):\n    pass"),
        ("Lab 05: Training Loop", "🟡 Medium", "60 mins", "- Optimization", "Build a complete training loop with forward pass, loss calculation, backward pass, and weight update.", "def train(model, X, y, epochs):\n    pass"),
    ],
    "Phase2_Classification/Week3_Classic_Architectures": [
        ("Lab 01: LeNet-5 Implementation", "🟡 Medium", "60 mins", "- CNN Architecture", "Implement the LeNet-5 architecture using PyTorch.", "class LeNet5(nn.Module):\n    pass"),
        ("Lab 02: AlexNet Implementation", "🟡 Medium", "60 mins", "- CNN Architecture", "Implement AlexNet with appropriate layers and dropout.", "class AlexNet(nn.Module):\n    pass"),
        ("Lab 03: VGG Blocks", "🟡 Medium", "45 mins", "- Modular Design", "Implement a function to generate VGG blocks.", "def make_vgg_block(num_convs, in_channels, out_channels):\n    pass"),
        ("Lab 04: ResNet Skip Connections", "🟡 Medium", "45 mins", "- Residual Learning", "Implement a Residual Block with skip connections.", "class ResidualBlock(nn.Module):\n    pass"),
        ("Lab 05: Inception Module", "🔴 Hard", "60 mins", "- Complex Architectures", "Implement the Inception module with parallel convolutions.", "class InceptionModule(nn.Module):\n    pass"),
    ],
    "Phase2_Classification/Week4_Advanced_Recognition": [
        ("Lab 01: Data Augmentation", "🟢 Easy", "45 mins", "- Regularization", "Implement random crops, flips, and rotations using torchvision.transforms.", "transforms = None"),
        ("Lab 02: Transfer Learning", "🟡 Medium", "60 mins", "- Fine-tuning", "Load a pretrained ResNet and fine-tune it on a new dataset (e.g., CIFAR-10).", "model = models.resnet18(pretrained=True)"),
        ("Lab 03: Learning Rate Scheduler", "🟡 Medium", "45 mins", "- Optimization", "Implement StepLR and CosineAnnealing schedulers.", "scheduler = None"),
        ("Lab 04: Visualizing Feature Maps", "🟡 Medium", "60 mins", "- Interpretability", "Extract and visualize feature maps from intermediate layers of a CNN.", "def visualize_features(model, image):\n    pass"),
        ("Lab 05: Class Activation Maps (CAM)", "🔴 Hard", "90 mins", "- Interpretability", "Implement CAM to visualize which parts of the image the model focuses on.", "def generate_cam(model, image):\n    pass"),
    ],
    "Phase3_Detection/Week5_Object_Detection": [
        ("Lab 01: Intersection over Union (IoU)", "🟢 Easy", "30 mins", "- Metrics", "Calculate IoU between two bounding boxes.", "def calculate_iou(box1, box2):\n    pass"),
        ("Lab 02: Non-Maximum Suppression (NMS)", "🟡 Medium", "60 mins", "- Post-processing", "Implement NMS to remove redundant overlapping bounding boxes.", "def nms(boxes, scores, threshold):\n    pass"),
        ("Lab 03: Anchor Box Generation", "🟡 Medium", "60 mins", "- Anchors", "Generate anchor boxes for a feature map grid.", "def generate_anchors(base_size, ratios, scales):\n    pass"),
        ("Lab 04: YOLO Loss Function", "🔴 Hard", "90 mins", "- Loss Design", "Implement the multi-part loss function for YOLO (coord, obj, noobj, class).", "def yolo_loss(pred, target):\n    pass"),
        ("Lab 05: Bounding Box Visualization", "🟢 Easy", "30 mins", "- Visualization", "Draw bounding boxes and class labels on an image.", "def draw_boxes(image, boxes, labels):\n    pass"),
    ],
    "Phase3_Detection/Week6_Segmentation": [
        ("Lab 01: Transposed Convolution", "🟡 Medium", "45 mins", "- Upsampling", "Implement transposed convolution (deconvolution) for upsampling.", "layer = nn.ConvTranspose2d(...)"),
        ("Lab 02: U-Net Architecture", "🔴 Hard", "90 mins", "- Segmentation", "Implement the U-Net architecture with encoder-decoder and skip connections.", "class UNet(nn.Module):\n    pass"),
        ("Lab 03: Dice Loss", "🟡 Medium", "45 mins", "- Loss Functions", "Implement Dice Loss for segmentation tasks.", "def dice_loss(pred, target):\n    pass"),
        ("Lab 04: Semantic Segmentation Inference", "🟡 Medium", "60 mins", "- Inference", "Run inference using a pretrained DeepLabV3 model.", "model = models.segmentation.deeplabv3_resnet50(...)"),
        ("Lab 05: Mask R-CNN Basics", "🔴 Hard", "90 mins", "- Instance Segmentation", "Understand and implement RoI Align.", "def roi_align(features, rois):\n    pass"),
    ],
}

# ==============================================================================
# SYSTEM DESIGN
# ==============================================================================
sd_root = r"G:\My Drive\Codes & Repos\System_Design_Course"

sd_content = {
    "Phase2_Building_Blocks": [
        ("Lab 01: Distributed ID Generator", "🟡 Medium", "60 mins", "- Snowflake ID", "Design a unique ID generator that works across distributed systems (like Twitter Snowflake).", "class SnowflakeID:\n    pass"),
        ("Lab 02: Key-Value Store", "🔴 Hard", "90 mins", "- Storage Engines", "Implement a simple in-memory KV store with WAL (Write Ahead Log) for durability.", "class KVStore:\n    pass"),
        ("Lab 03: Message Queue", "🟡 Medium", "60 mins", "- Pub/Sub", "Implement a basic message queue with publish and subscribe methods.", "class MessageQueue:\n    pass"),
        ("Lab 04: Rate Limiter (Sliding Window)", "🟡 Medium", "60 mins", "- Rate Limiting", "Implement a Sliding Window Log rate limiter.", "class SlidingWindowLimiter:\n    pass"),
        ("Lab 05: Bloom Filter", "🟡 Medium", "45 mins", "- Probabilistic Data Structures", "Implement a Bloom Filter for efficient set membership testing.", "class BloomFilter:\n    pass"),
    ],
    "Phase3_Advanced_Architectures": [
        ("Lab 01: MapReduce Simulation", "🔴 Hard", "90 mins", "- Distributed Processing", "Simulate a MapReduce job to count word frequencies across multiple 'nodes'.", "def map_function(text):\n    pass\ndef reduce_function(key, values):\n    pass"),
        ("Lab 02: Circuit Breaker", "🟡 Medium", "60 mins", "- Fault Tolerance", "Implement a Circuit Breaker pattern to prevent cascading failures.", "class CircuitBreaker:\n    pass"),
        ("Lab 03: Service Discovery", "🟡 Medium", "60 mins", "- Microservices", "Implement a simple Service Registry where services can register and discover others.", "class ServiceRegistry:\n    pass"),
        ("Lab 04: Distributed Lock", "🔴 Hard", "90 mins", "- Concurrency", "Implement a distributed lock using a simulated Redis/ZooKeeper backend.", "class DistributedLock:\n    pass"),
        ("Lab 05: Gossip Protocol", "🔴 Hard", "90 mins", "- Consensus", "Simulate a Gossip Protocol for disseminating information across nodes.", "class Node:\n    def gossip(self):\n        pass"),
    ],
    "Phase4_Case_Studies": [
        ("Lab 01: URL Shortener Design", "🟡 Medium", "60 mins", "- System Design", "Implement the core logic for a URL shortener (Base62 encoding).", "def encode(num):\n    pass\ndef decode(str):\n    pass"),
        ("Lab 02: Chat System Message Sync", "🔴 Hard", "90 mins", "- Real-time", "Design a data structure to handle message synchronization and ordering for a chat app.", "class ChatRoom:\n    pass"),
        ("Lab 03: Typeahead Search", "🔴 Hard", "90 mins", "- Tries", "Implement a Trie-based backend for autocomplete suggestions.", "class Typeahead:\n    pass"),
        ("Lab 04: Notification System", "🟡 Medium", "60 mins", "- Async Processing", "Design a notification dispatcher that handles email, SMS, and push notifications.", "class NotificationDispatcher:\n    pass"),
        ("Lab 05: Metrics Monitoring", "🟡 Medium", "60 mins", "- Time Series", "Implement a system to aggregate and query metrics (counters, gauges).", "class MetricsStore:\n    pass"),
    ],
}

# ==============================================================================
# ML SYSTEM DESIGN
# ==============================================================================
ml_root = r"G:\My Drive\Codes & Repos\ML_System_Design_Course"

ml_content = {
    "Phase1_Foundations/Week2_Math_Foundations": [
        ("Lab 01: Linear Algebra with NumPy", "🟢 Easy", "45 mins", "- Matrix Ops", "Implement matrix multiplication, inversion, and eigendecomposition using NumPy.", "import numpy as np"),
        ("Lab 02: Gradient Descent", "🟡 Medium", "60 mins", "- Optimization", "Implement Vanilla Gradient Descent to minimize a quadratic function.", "def gradient_descent(start, lr, epochs):\n    pass"),
        ("Lab 03: PCA from Scratch", "🔴 Hard", "90 mins", "- Dimensionality Reduction", "Implement Principal Component Analysis using covariance matrix and eigen decomposition.", "def pca(X, n_components):\n    pass"),
        ("Lab 04: Probability Distributions", "🟢 Easy", "45 mins", "- Statistics", "Visualize Normal, Binomial, and Poisson distributions.", "import matplotlib.pyplot as plt"),
        ("Lab 05: Hypothesis Testing", "🟡 Medium", "60 mins", "- Statistics", "Perform T-test and Chi-Square test on sample data.", "from scipy import stats"),
    ],
    "Phase2_Core_Algorithms/Week3_Supervised_Learning": [
        ("Lab 01: Linear Regression", "🟢 Easy", "45 mins", "- Regression", "Implement Linear Regression using Scikit-Learn.", "from sklearn.linear_model import LinearRegression"),
        ("Lab 02: Logistic Regression", "🟢 Easy", "45 mins", "- Classification", "Implement Logistic Regression for binary classification.", "from sklearn.linear_model import LogisticRegression"),
        ("Lab 03: Decision Trees", "🟡 Medium", "60 mins", "- Tree Models", "Train and visualize a Decision Tree classifier.", "from sklearn.tree import DecisionTreeClassifier"),
        ("Lab 04: Random Forest", "🟡 Medium", "60 mins", "- Ensembling", "Implement Random Forest and analyze feature importance.", "from sklearn.ensemble import RandomForestClassifier"),
        ("Lab 05: SVM", "🟡 Medium", "60 mins", "- Classification", "Train a Support Vector Machine with different kernels.", "from sklearn.svm import SVC"),
    ],
    "Phase2_Core_Algorithms/Week4_Unsupervised_DeepLearning": [
        ("Lab 01: K-Means Clustering", "🟡 Medium", "60 mins", "- Clustering", "Implement K-Means clustering algorithm from scratch.", "class KMeans:\n    pass"),
        ("Lab 02: DBSCAN", "🟡 Medium", "60 mins", "- Clustering", "Use DBSCAN to cluster non-linear data.", "from sklearn.cluster import DBSCAN"),
        ("Lab 03: Gaussian Mixture Models", "🔴 Hard", "90 mins", "- Density Estimation", "Implement GMM using Expectation-Maximization.", "from sklearn.mixture import GaussianMixture"),
        ("Lab 04: Autoencoders", "🟡 Medium", "60 mins", "- Deep Learning", "Build a simple Autoencoder for image denoising.", "class Autoencoder(nn.Module):\n    pass"),
        ("Lab 05: t-SNE Visualization", "🟡 Medium", "45 mins", "- Visualization", "Use t-SNE to visualize high-dimensional data.", "from sklearn.manifold import TSNE"),
    ],
    "Phase3_System_Design/Week5_Design_Principles": [
        ("Lab 01: Model Registry", "🟡 Medium", "60 mins", "- MLOps", "Design a simple Model Registry schema to track model versions and metadata.", "class ModelRegistry:\n    pass"),
        ("Lab 02: Feature Store", "🔴 Hard", "90 mins", "- Data Engineering", "Simulate a Feature Store with online (Redis) and offline (Parquet) retrieval.", "class FeatureStore:\n    pass"),
        ("Lab 03: Data Validation", "🟡 Medium", "60 mins", "- Data Quality", "Implement data validation checks (schema, drift) using Great Expectations concepts.", "def validate_schema(df, schema):\n    pass"),
        ("Lab 04: A/B Testing", "🟡 Medium", "60 mins", "- Experimentation", "Simulate an A/B test and calculate statistical significance.", "def calculate_significance(control, treatment):\n    pass"),
        ("Lab 05: Pipeline Orchestration", "🟡 Medium", "60 mins", "- Workflows", "Design a DAG for a simple ML pipeline (Load -> Train -> Eval).", "class Pipeline:\n    pass"),
    ],
}

# ==============================================================================
# PYTORCH DEEP LEARNING
# ==============================================================================
pt_root = r"G:\My Drive\Codes & Repos\PyTorch_Deep_Learning_Course"

pt_content = {
    "Phase1_Foundations/Week2_Data_Workflow": [
        ("Lab 01: Custom Dataset", "🟢 Easy", "45 mins", "- Data Loading", "Create a custom Dataset class for loading images and labels.", "class CustomDataset(Dataset):\n    pass"),
        ("Lab 02: Data Loaders", "🟢 Easy", "30 mins", "- Batching", "Use DataLoader to iterate through data in batches with shuffling.", "loader = DataLoader(...)"),
        ("Lab 03: Transforms", "🟢 Easy", "45 mins", "- Preprocessing", "Apply transforms for normalization and augmentation.", "transform = transforms.Compose(...)"),
        ("Lab 04: Handling Imbalanced Data", "🟡 Medium", "60 mins", "- Sampling", "Implement WeightedRandomSampler to handle class imbalance.", "sampler = WeightedRandomSampler(...)"),
        ("Lab 05: Splitting Data", "🟢 Easy", "30 mins", "- Validation", "Split dataset into train, validation, and test sets.", "train_set, val_set = random_split(...)"),
    ],
    "Phase2_Computer_Vision/Week3_CNNs_Architectures": [
        ("Lab 01: Simple CNN", "🟢 Easy", "45 mins", "- Architecture", "Build a simple CNN for MNIST classification.", "class SimpleCNN(nn.Module):\n    pass"),
        ("Lab 02: Pooling Layers", "🟢 Easy", "30 mins", "- Downsampling", "Experiment with MaxPool and AvgPool layers.", "pool = nn.MaxPool2d(...)"),
        ("Lab 03: Batch Normalization", "🟡 Medium", "45 mins", "- Regularization", "Add Batch Normalization to a CNN and observe convergence.", "bn = nn.BatchNorm2d(...)"),
        ("Lab 04: Dropout", "🟢 Easy", "30 mins", "- Regularization", "Implement Dropout to prevent overfitting.", "dropout = nn.Dropout(...)"),
        ("Lab 05: Global Average Pooling", "🟡 Medium", "45 mins", "- Architecture", "Replace fully connected layers with GAP.", "gap = nn.AdaptiveAvgPool2d(...)"),
    ],
}

# ==============================================================================
# REINFORCEMENT LEARNING
# ==============================================================================
rl_root = r"G:\My Drive\Codes & Repos\Reinforcement_Learning_Course"

rl_content = {
    "Phase2_Value_Based_Deep_RL": [
        ("Lab 01: DQN Implementation", "🔴 Hard", "90 mins", "- Deep Q-Learning", "Implement a Deep Q-Network with Experience Replay.", "class DQN(nn.Module):\n    pass"),
        ("Lab 02: Experience Replay Buffer", "🟡 Medium", "45 mins", "- Data Structure", "Implement a cyclic buffer to store transitions.", "class ReplayBuffer:\n    pass"),
        ("Lab 03: Target Network", "🟡 Medium", "45 mins", "- Stability", "Implement target network updates (soft and hard updates).", "def update_target(model, target):\n    pass"),
        ("Lab 04: Double DQN", "🔴 Hard", "60 mins", "- Optimization", "Modify DQN to implement Double DQN loss.", "def ddqn_loss(...):\n    pass"),
        ("Lab 05: Dueling DQN", "🔴 Hard", "60 mins", "- Architecture", "Implement Dueling Network architecture (Value + Advantage).", "class DuelingDQN(nn.Module):\n    pass"),
    ],
    "Phase3_Policy_Based": [
        ("Lab 01: REINFORCE Algorithm", "🟡 Medium", "60 mins", "- Policy Gradient", "Implement the REINFORCE algorithm (Monte Carlo Policy Gradient).", "def reinforce_update(...):\n    pass"),
        ("Lab 02: Actor-Critic", "🔴 Hard", "90 mins", "- AC Methods", "Implement a basic Actor-Critic agent.", "class ActorCritic(nn.Module):\n    pass"),
        ("Lab 03: A2C", "🔴 Hard", "90 mins", "- Parallelism", "Implement Advantage Actor-Critic (A2C).", "class A2CAgent:\n    pass"),
        ("Lab 04: PPO Clipping", "🔴 Hard", "90 mins", "- Optimization", "Implement the PPO clipped objective function.", "def ppo_loss(...):\n    pass"),
        ("Lab 05: Continuous Action Space", "🟡 Medium", "60 mins", "- Distributions", "Handle continuous actions using Gaussian distribution.", "dist = Normal(mu, std)"),
    ],
}

# ==============================================================================
# EXECUTION
# ==============================================================================

def process_course(root, content_dict):
    for folder, labs in content_dict.items():
        base_path = os.path.join(root, folder, "labs")
        for i, (title, diff, time, obj, prob, code) in enumerate(labs):
            filename = f"lab_{i+1:02d}.md"
            path = os.path.join(base_path, filename)
            content = create_lab_content(title, diff, time, obj, prob, code)
            update_lab(path, content)

print("🚀 Starting massive lab population...")

process_course(dsa_root, dsa_content)
process_course(cv_root, cv_content)
process_course(sd_root, sd_content)
process_course(ml_root, ml_content)
process_course(pt_root, pt_content)
process_course(rl_root, rl_content)

print("✅ Massive lab population complete! All specified labs have been upgraded.")
