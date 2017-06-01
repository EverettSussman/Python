# Search and Sort Algorithms
# By Everett Sussman

# Insert Sort, Ascending
def ins_sort_asc(A):
    for j in range(1, len(A)):
        key = A[j]
        i = j-1
        while i >= 0 and A[i] > key:
            A[i+1] = A[i]
            i = i-1
        A[i+1] = key
    return A

# Insertion-Sort Algorithm, descending order
def ins_sort_dec(A):
    for j in range(1, len(A)):
        key = A[j]
        i = j-1
        while i >= 0 and A[i] < key:
            A[i+1] = A[i]
            i = i-1
        A[i+1] = key
    return A

# Recursive Insertion Sort Algorithm
# About 50% faster than ins_sort_asc

def rec_ins_sort(A, i):
    if i < len(A):
        j = i - 1
        key = A[i]
        while j >= 0 and A[j] > key:
            A[j+1] = A[j]
            j = j-1
        A[j+1] = key
        return rec_ins_sort(A, i+1)
    else:
        return A

# Merge Sort Algorithm
# Merge - creates one sorted pile from two smaller sorted piles

def merge(A, p, q, r):
    n1 = q-p + 1
    n2 = r-q
    L, R = [], []
    for i in range(0,n1):
        L.append(A[p+i])
    for j in range(0, n2):
        R.append(A[q+ j + 1])
    L.append(sys.maxint)
    R.append(sys.maxint)
    i, j = 0, 0
    for k in range(p, r + 1):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
    return A

# Merge Sort with O(n lg(n))

def merge_sort(A, p, r):
    if p < r:
        q = int(np.floor((p+r)/2))
        merge_sort(A, p, q)
        merge_sort(A, q+1, r)
        merge(A, p, q, r)
    return A



# Linear Search Algorithm
def lin_search(A, v):
    # A is list of values, v is key
    for j in range(0, len(A)):
        if A[j] == v:
            return j
    return None

# Binary search algorithm with running time log(n)
def bin_search(A, v):
    # A is the list to search, v is value to search
    low, high = 0, len(A)-1
    while low <= high:
        mid = int(np.floor((low + high)/2))
        if A[mid] == v:
            return mid
        elif A[mid] < v:
            low = mid +1
        else:
            high = mid - 1
    return None




