"""
eBay DS/ML Interview — DSA Coding Patterns (Python)
25 problems by PATTERN. Focus: HashMap, Two Pointers, Sliding Window,
Sorting, Heaps, Stacks, Binary Search, Greedy/DP
"""
from typing import List
from collections import Counter, defaultdict
import heapq

# === PATTERN 1: HASH MAPS ===

def two_sum(nums, target):
    """Return indices of two numbers summing to target. O(n)"""
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
    return []

def group_anagrams(strs):
    """Group anagram strings. O(n*k*log(k))"""
    groups = defaultdict(list)
    for s in strs:
        groups[tuple(sorted(s))].append(s)
    return list(groups.values())

def top_k_frequent(nums, k):
    """Return k most frequent elements. O(n log k)"""
    return [x for x, _ in Counter(nums).most_common(k)]

def first_unique_char(s):
    """Index of first non-repeating character, -1 if none."""
    freq = Counter(s)
    for i, ch in enumerate(s):
        if freq[ch] == 1:
            return i
    return -1

def subarray_sum(nums, k):
    """Count subarrays summing to k. Prefix Sum + HashMap O(n)"""
    count = prefix = 0
    prefix_map = {0: 1}
    for num in nums:
        prefix += num
        count += prefix_map.get(prefix - k, 0)
        prefix_map[prefix] = prefix_map.get(prefix, 0) + 1
    return count

# === PATTERN 2: TWO POINTERS ===

def is_palindrome(s):
    """Check palindrome ignoring non-alphanumeric. O(n)"""
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and not s[l].isalnum(): l += 1
        while l < r and not s[r].isalnum(): r -= 1
        if s[l].lower() != s[r].lower(): return False
        l += 1; r -= 1
    return True

def three_sum(nums):
    """All unique triplets summing to zero. Sort+2ptr O(n^2)"""
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]: continue
        l, r = i+1, len(nums)-1
        while l < r:
            s = nums[i]+nums[l]+nums[r]
            if s < 0: l += 1
            elif s > 0: r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l]==nums[l+1]: l+=1
                while l < r and nums[r]==nums[r-1]: r-=1
                l+=1; r-=1
    return res

def max_area(heights):
    """Container with most water. O(n)"""
    l, r = 0, len(heights)-1
    best = 0
    while l < r:
        best = max(best, (r-l)*min(heights[l], heights[r]))
        if heights[l] < heights[r]: l += 1
        else: r -= 1
    return best

def remove_duplicates(nums):
    """Remove duplicates in sorted array in-place. O(n)"""
    if not nums: return 0
    w = 1
    for r in range(1, len(nums)):
        if nums[r] != nums[r-1]:
            nums[w] = nums[r]; w += 1
    return w

# === PATTERN 3: SLIDING WINDOW ===

def max_sum_subarray_k(nums, k):
    """Max sum of contiguous subarray of size k. O(n)"""
    ws = sum(nums[:k]); best = ws
    for i in range(k, len(nums)):
        ws += nums[i] - nums[i-k]
        best = max(best, ws)
    return best

def longest_unique_substring(s):
    """Longest substring without repeating chars. O(n)"""
    seen = set(); l = best = 0
    for r in range(len(s)):
        while s[r] in seen:
            seen.remove(s[l]); l += 1
        seen.add(s[r])
        best = max(best, r-l+1)
    return best

def min_window(s, t):
    """Min window in s containing all chars of t. O(n)"""
    if not t or not s: return ""
    need = Counter(t); have = 0; required = len(need)
    l = 0; res = ""; min_len = float('inf')
    wc = defaultdict(int)
    for r in range(len(s)):
        wc[s[r]] += 1
        if s[r] in need and wc[s[r]] == need[s[r]]: have += 1
        while have == required:
            if r-l+1 < min_len:
                min_len = r-l+1; res = s[l:r+1]
            wc[s[l]] -= 1
            if s[l] in need and wc[s[l]] < need[s[l]]: have -= 1
            l += 1
    return res

# === PATTERN 4: SORTING & INTERVALS ===

def merge_intervals(intervals):
    """Merge overlapping intervals. O(n log n)"""
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
        else: merged.append([s, e])
    return merged

def min_meeting_rooms(intervals):
    """Min conference rooms needed. Sort+Heap O(n log n)"""
    if not intervals: return 0
    intervals.sort()
    heap = []
    for s, e in intervals:
        if heap and heap[0] <= s: heapq.heappop(heap)
        heapq.heappush(heap, e)
    return len(heap)

# === PATTERN 5: HEAPS ===

def kth_largest(nums, k):
    """Kth largest element. Min-heap of size k. O(n log k)"""
    h = nums[:k]; heapq.heapify(h)
    for n in nums[k:]:
        if n > h[0]: heapq.heapreplace(h, n)
    return h[0]

def merge_k_sorted(lists):
    """Merge k sorted lists. O(N log k)"""
    h = []
    for i, lst in enumerate(lists):
        if lst: heapq.heappush(h, (lst[0], i, 0))
    res = []
    while h:
        val, li, ei = heapq.heappop(h)
        res.append(val)
        if ei+1 < len(lists[li]):
            heapq.heappush(h, (lists[li][ei+1], li, ei+1))
    return res

# === PATTERN 6: BINARY SEARCH ===

def search_rotated(nums, target):
    """Search in rotated sorted array. O(log n)"""
    l, r = 0, len(nums)-1
    while l <= r:
        m = (l+r)//2
        if nums[m] == target: return m
        if nums[l] <= nums[m]:
            if nums[l] <= target < nums[m]: r = m-1
            else: l = m+1
        else:
            if nums[m] < target <= nums[r]: l = m+1
            else: r = m-1
    return -1

def find_peak(nums):
    """Find peak element index. O(log n)"""
    l, r = 0, len(nums)-1
    while l < r:
        m = (l+r)//2
        if nums[m] > nums[m+1]: r = m
        else: l = m+1
    return l

# === PATTERN 7: STACKS ===

def valid_parens(s):
    """Check valid parentheses. O(n)"""
    st = []; mp = {')':'(','}':'{',']':'['}
    for c in s:
        if c in mp:
            if not st or st.pop() != mp[c]: return False
        else: st.append(c)
    return not st

def daily_temperatures(temps):
    """Days until warmer temp. Monotonic stack O(n)"""
    res = [0]*len(temps); st = []
    for i, t in enumerate(temps):
        while st and temps[st[-1]] < t:
            j = st.pop(); res[j] = i-j
        st.append(i)
    return res

# === PATTERN 8: GREEDY / DP ===

def max_profit(prices):
    """Best time to buy/sell stock. Greedy O(n)"""
    mn = float('inf'); best = 0
    for p in prices:
        mn = min(mn, p); best = max(best, p-mn)
    return best

def max_subarray(nums):
    """Maximum subarray sum (Kadane). O(n)"""
    cur = best = nums[0]
    for n in nums[1:]:
        cur = max(n, cur+n); best = max(best, cur)
    return best

def coin_change(coins, amount):
    """Min coins for amount. DP O(n*amount)"""
    dp = [float('inf')]*(amount+1); dp[0] = 0
    for i in range(1, amount+1):
        for c in coins:
            if c <= i: dp[i] = min(dp[i], dp[i-c]+1)
    return dp[amount] if dp[amount] != float('inf') else -1

# === TESTS ===
if __name__ == '__main__':
    tests = [
        ("Two Sum", two_sum([2,7,11,15],9)==[0,1]),
        ("Group Anagrams", len(group_anagrams(["eat","tea","tan","ate","nat","bat"]))==3),
        ("Top K Frequent", set(top_k_frequent([1,1,1,2,2,3],2))=={1,2}),
        ("First Unique", first_unique_char("leetcode")==0),
        ("Subarray Sum K", subarray_sum([1,1,1],2)==2),
        ("Palindrome", is_palindrome("A man, a plan, a canal: Panama")),
        ("3Sum", three_sum([-1,0,1,2,-1,-4])==[[-1,-1,2],[-1,0,1]]),
        ("Container Water", max_area([1,8,6,2,5,4,8,3,7])==49),
        ("Remove Dups", remove_duplicates([1,1,2,2,3])==3),
        ("Max Sum K", max_sum_subarray_k([2,1,5,1,3,2],3)==9),
        ("Longest Unique", longest_unique_substring("abcabcbb")==3),
        ("Min Window", min_window("ADOBECODEBANC","ABC")=="BANC"),
        ("Merge Intervals", merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]),
        ("Meeting Rooms", min_meeting_rooms([[0,30],[5,10],[15,20]])==2),
        ("Kth Largest", kth_largest([3,2,1,5,6,4],2)==5),
        ("Merge K Sorted", merge_k_sorted([[1,4,5],[1,3,4],[2,6]])==[1,1,2,3,4,4,5,6]),
        ("Search Rotated", search_rotated([4,5,6,7,0,1,2],0)==4),
        ("Find Peak", find_peak([1,2,3,1])==2),
        ("Valid Parens", valid_parens("()[]{}")),
        ("Daily Temps", daily_temperatures([73,74,75,71,69,72,76,73])==[1,1,4,2,1,1,0,0]),
        ("Max Profit", max_profit([7,1,5,3,6,4])==5),
        ("Max Subarray", max_subarray([-2,1,-3,4,-1,2,1,-5,4])==6),
        ("Coin Change", coin_change([1,5,10,25],30)==2),
    ]
    print("="*60)
    print("DSA Coding Patterns — All Tests")
    print("="*60)
    for name, ok in tests:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  {sum(1 for _,p in tests if p)}/{len(tests)} passed")
