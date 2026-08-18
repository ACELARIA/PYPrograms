import numpy as np

# ============================================================================
# Problem 1: Cricket, Football and Hockey
# ============================================================================
# In a school, there are total 100 students numbered from 1 to 100. We are given three lists named C, F, and H, representing students who play cricket,football, and hockey, respectively.

# Initialize student lists
C = [7, 8, 9, 18, 20, 21, 25, 26, 27, 31, 32, 34, 35, 36, 40, 43, 45, 47, 53, 58, 62, 67, 68, 71, 72, 74, 75, 76, 80, 81, 82, 90, 93, 95, 97, 99]
F = [1, 7, 10, 13, 16, 22, 24, 29, 30, 32, 34, 39, 40, 43, 44, 48, 56, 60, 65, 68, 69, 73, 77, 78, 90, 93, 94, 95, 96]
H = [5, 12, 14, 17, 20, 21, 22, 25, 28, 30, 37, 38, 39, 40, 42, 44, 57, 59, 61, 62, 67, 71, 75, 76, 77, 82, 83, 86, 87, 92, 94, 95]

# Q1.1: Which are the students who play all the three sports?
# Options: [22, 39], [39, 82], [40, 95] (Correct), [82, 94]
all_three = sorted(list(set(C) & set(F) & set(H)))
print("Q1.1 - Students who play all three sports:", all_three)

# Q1.2: Which are the players who play both cricket and hockey but don't play football?
# Options: [20, 21, 25, 62, 67, 71, 75, 76, 82] (Correct), [20, 21, 22, 25, 30, 32, 34], etc.
cricket_hockey_no_football = sorted(list((set(C) & set(H)) - set(F)))
print("Q1.2 - Players who play both cricket and hockey but don't play football:", cricket_hockey_no_football)

# Q1.3: How many players play exactly two sports?
# Options: 19, 20, 21, 22 (Correct)
exactly_two = 0
for i in range(1, 101):
    count = 0
    if i in C: count += 1
    if i in F: count += 1
    if i in H: count += 1
    if count == 2:
        exactly_two += 1
print("Q1.3 - Number of players playing exactly two sports:", exactly_two)

# Q1.4: Which of these players do not play any of the sports?
# Options (Check which of 41, 48, 63, 85 do not play any): 41, 63, 85 (Correct)
options_list = [41, 48, 63, 85]
none_play = [x for x in options_list if x not in C and x not in F and x not in H]
print("Q1.4 - From the options [41, 48, 63, 85], those who don't play any sport:", none_play)

# ============================================================================
# Problem 2: Netflix Views
# ============================================================================
# The netflix dictionary contains a list of 100 titles along with their total number of views in the past.

netflix = {
    'The Debt Collector': 1821195, 'Act of Vengeance': 2804479, 'Paradise Lost': 2195477, 
    "Gerald's Game": 3650626, 'Long Shot': 638440, 'Mak Cun': 1010311, 'Our Souls at Night': 3028705, 
    'Out of Thin Air': 2681938, "Paul Hollywood's Big Continental Road Trip": 230689, 'Satu Hari': 1012350, 
    'Monster High: Boo York, Boo York': 1526025, 
    'Cultivating the Seas: History and Future of the Full-Cycle Cultured Kindai Tuna': 2762103, 
    'Domino': 872663, 'TUNA GIRL': 2574816, '5CM': 2647713, 'Animal World': 220789, 'Hold the Dark': 2307432, 
    'Lessons from a School Shooting: Notes from Dunblane': 2419534, 'Made in Mexico': 3653712, 
    'Single': 239278, 'The 3rd Eye': 196138, 'The Sinking Of Van Der Wijck': 2831856, 'Two Catalonias': 1337119, 
    'Bobby Sands: 66 Days': 2005623, 'Bard of Blood': 744998, 'Deliha 2': 3638900, 
    'Dragons: Rescue Riders': 3241239, 'In the Shadow of the Moon': 1213329, 'Skylines': 1840094, 
    'Sturgill Simpson Presents Sound & Fury': 1915540, 'The Politician': 1019394, 'Weeds on Fire': 2273090, 
    'Much Loved': 3039852, 'Joseph: King of Dreams': 1734449, 'Malaal': 1926128, 'The Grandmaster': 2549410, 
    'The Inmate': 130965, 'The Hurricane Heist': 2191538, 'Def Comedy Jam 25': 1392697, 
    'Restless Creature: Wendy Whelan': 2474412, 'Print the Legend': 548250, 'Birders': 3478883, 
    'Furie': 2755115, 'Leap!': 1401638, 'Oh! Baby (Malayalam)': 839139, 'Oh! Baby (Tamil)': 2691505, 
    'USS Indianapolis: Men of Courage': 3286014, 'A Wrinkle in Time': 1258861, 'Teach Us All': 2428947, 
    'White Island': 1152905, 'Inside Man: Most Wanted': 2812661, 'Jeff Dunham: Beside Himself': 2930368, 
    'China Salesman': 3377292, 'Swearnet: The Movie': 949689, 'The Bar': 620655, 'Manmadhudu 2': 1842761, 
    'Team Kaylie': 359856, 'Under the Eiffel Tower': 2298237, 'Audrie & Daisy': 1573055, 
    'Iliza Shlesinger: Confirmed Kills': 365220, 'BONDING': 1664242, 
    'Do Paise Ki Dhoop Chaar Aane Ki Baarish': 2770879, '20 Feet From Stardom': 2931730, 
    'In Darkness': 3248749, 'Gaga: Five Foot Two': 3719550, 'The Bad Batch': 3656264, 
    'SMOSH: The Movie': 3697506, 'King of Boys': 584112, 'Merry Men: The Real Yoruba Demons': 2035595, 
    "Sarah's Key": 1337902, 'The Wedding Party 2: Destination Dubai': 1471204, 'Vagabond': 1600432, 
    'Battlefish': 723021, 'DRAGON PILOT: Hisone & Masotan': 602919, 'Hilda': 54948, 'Maniac': 1348196, 
    'Quincy': 2601704, 'Rafinha Bastos: Ultimatum': 3469996, 'The Good Cop': 2265521, 
    'Between Two Ferns: The Movie': 3146508, 'Criminal: France': 2021409, 'Criminal: Germany': 658824, 
    'Criminal: Spain': 3078634, 'Criminal: UK': 2721398, 'Daddy Issues': 2119035, 
    "Inside Bill's Brain: Decoding Bill Gates": 1755104, 'The Hockey Girls': 2000507, 'Travel Mates 2': 707260, 
    'True: Tricky Treat Day': 3213199, 'Two Sentence Horror Stories': 2438329, 'Mad World': 2281773, 
    'Mobile Suit Gundam UC': 778879, 'The Bund': 1729892, 'The First Line': 674591, 'Maynard': 84986, 
    'Monkey Twins': 3277179, 'Bitcoin Heist': 1450158, 'I Am Not Madame Bovary': 2763830, 
    'Vincent N Roxxy': 1961810, "Chef's Table: France": 2065738
}

# Q2.1: There's an Indian movie in the dictionary named 'Do Paise Ki Dhoop Chaar Aane Ki Baarish'. 
# What is the number of views for this movie?
# Options: 90,854, 658,824, 1,023,597, 2,770,879 (Correct)
views_indian_movie = netflix.get('Do Paise Ki Dhoop Chaar Aane Ki Baarish')
print("Q2.1 - Views of 'Do Paise Ki Dhoop Chaar Aane Ki Baarish':", f"{views_indian_movie:,}")

# Q2.2: Which of the shows is the most and least popular in terms of number of views?
# Options: Monkey Twins + Maynard, Gaga: Five Foot Two + Hilda (Correct), etc.
most_popular = max(netflix, key=netflix.get)
least_popular = min(netflix, key=netflix.get)
print(f"Q2.2 - Most popular: {most_popular} ({netflix[most_popular]:,} views)")
print(f"Q2.2 - Least popular: {least_popular} ({netflix[least_popular]:,} views)")

# Q2.3: Netflix has just received the number of views for the last week as well. 
# Add these views to the total views and find the most popular title in terms of views.
# Options: Title: Deliha 2, Views: 3943452 (Correct), etc.
netflix_last_week = {
    'The Debt Collector': 229218, 'Act of Vengeance': 191468, 'Paradise Lost': 167846, 
    "Gerald's Game": 113298, 'Long Shot': 15190, 'Mak Cun': 244581, 'Our Souls at Night': 188095, 
    'Out of Thin Air': 183961, "Paul Hollywood's Big Continental Road Trip": 207967, 'Satu Hari': 226211, 
    'Monster High: Boo York, Boo York': 265919, 
    'Cultivating the Seas: History and Future of the Full-Cycle Cultured Kindai Tuna': 192246, 
    'Domino': 190538, 'TUNA GIRL': 301819, '5CM': 154371, 'Animal World': 228227, 'Hold the Dark': 222938, 
    'Lessons from a School Shooting: Notes from Dunblane': 256580, 'Made in Mexico': 199790, 
    'Single': 155517, 'The 3rd Eye': 251349, 'The Sinking Of Van Der Wijck': 153797, 'Two Catalonias': 189422, 
    'Bobby Sands: 66 Days': 202461, 'Bard of Blood': 241928, 'Deliha 2': 304552, 
    'Dragons: Rescue Riders': 155149, 'In the Shadow of the Moon': 222999, 'Skylines': 118737, 
    'Sturgill Simpson Presents Sound & Fury': 43590, 'The Politician': 11902, 'Weeds on Fire': 243382, 
    'Much Loved': 190123, 'Joseph: King of Dreams': 46104, 'Malaal': 70961, 'The Grandmaster': 44908, 
    'The Inmate': 19574, 'The Hurricane Heist': 167696, 'Def Comedy Jam 25': 208075, 
    'Restless Creature: Wendy Whelan': 22976, 'Print the Legend': 161584, 'Birders': 257668, 
    'Furie': 282266, 'Leap!': 124338, 'Oh! Baby (Malayalam)': 18203, 'Oh! Baby (Tamil)': 131481, 
    'USS Indianapolis: Men of Courage': 76558, 'A Wrinkle in Time': 169365, 'Teach Us All': 50378, 
    'White Island': 299779, 'Inside Man: Most Wanted': 123346, 'Jeff Dunham: Beside Himself': 283561, 
    'China Salesman': 182041, 'Swearnet: The Movie': 63809, 'The Bar': 41243, 'Manmadhudu 2': 261349, 
    'Team Kaylie': 101283, 'Under the Eiffel Tower': 83650, 'Audrie & Daisy': 106092, 
    'Iliza Shlesinger: Confirmed Kills': 255103, 'BONDING': 202683, 
    'Do Paise Ki Dhoop Chaar Aane Ki Baarish': 32329, '20 Feet From Stardom': 165178, 
    'In Darkness': 95259, 'Gaga: Five Foot Two': 36638, 'The Bad Batch': 197820, 
    'SMOSH: The Movie': 196519, 'King of Boys': 49332, 'Merry Men: The Real Yoruba Demons': 293229, 
    "Sarah's Key": 178523, 'The Wedding Party 2: Destination Dubai': 51211, 'Vagabond': 116831, 
    'Battlefish': 5448, 'DRAGON PILOT: Hisone & Masotan': 281873, 'Hilda': 21348, 'Maniac': 268187, 
    'Quincy': 139696, 'Rafinha Bastos: Ultimatum': 105449, 'The Good Cop': 73230, 
    'Between Two Ferns: The Movie': 135600, 'Criminal: France': 243469, 'Criminal: Germany': 285211, 
    'Criminal: Spain': 45184, 'Criminal: UK': 117056, 'Daddy Issues': 267329, 
    "Inside Bill's Brain: Decoding Bill Gates": 64083, 'The Hockey Girls': 114146, 'Travel Mates 2': 21088, 
    'True: Tricky Treat Day': 116504, 'Two Sentence Horror Stories': 274471, 'Mad World': 225244, 
    'Mobile Suit Gundam UC': 47306, 'The Bund': 146872, 'The First Line': 148861, 'Maynard': 87252, 
    'Monkey Twins': 199274, 'Bitcoin Heist': 41105, 'I Am Not Madame Bovary': 57841, 
    'Vincent N Roxxy': 207453, "Chef's Table: France": 200536
}

total_views = {}
for title in netflix:
    total_views[title] = netflix[title] + netflix_last_week.get(title, 0)

most_popular_overall = max(total_views, key=total_views.get)
print(f"Q2.3 - Most popular overall: {most_popular_overall} ({total_views[most_popular_overall]:,} views)")

# ============================================================================
# Problem 3: Multiples of 3 Cubes Dictionary
# ============================================================================
# Create a dictionary where the keys are multiples of 3 among the first 100 natural numbers and each value is the cube of the key.

cubes_of_3 = {k: k**3 for k in range(3, 101, 3)}
print("\nProblem 3 - Dictionary containing cubes of multiples of 3:")
print(cubes_of_3)

# ============================================================================
# Problem 4: One-liner Python Code
# ============================================================================
# Write code in one line to print the following results:
# 1. Input: List 1 = ['a', 'b', 'c'], List 2 = ['d', 'e', 'f'] 
#    Output: ['ad', 'ae', 'af', 'bd', 'be', 'bf', 'cd', 'ce', 'cf']
# 2. Input: List 1 = ['a', 'c', 'd', 'b'] 
#    Output: {'A': 'aaa', 'C': 'ccc', 'B': 'bbb', 'D': 'ddd'}

list1 = ['a', 'b', 'c']
list2 = ['d', 'e', 'f']
print("\nProblem 4.1 - Cartesian Product concatenation:", [x + y for x in list1 for y in list2])

list_chars = ['a', 'c', 'd', 'b']
print("Problem 4.2 - Char Mapping dictionary:", {char.upper(): char * 3 for char in list_chars})

# ============================================================================
# Problem 5: 4x7 Array 'nola'
# ============================================================================
# Make the following 4x7 array called nola that starts with 1 and steps by 2. 
# The first element in each row is always 4 more than the last element in the previous row.

nola = np.arange(1, 14, 2) + np.arange(4).reshape(-1, 1) * 16
print("\nProblem 5 - nola array:")
print(nola)

# ============================================================================
# Problem 6: Roux Array
# ============================================================================
# Define a roux array as a 1-D array such that, when it's reversed, it represents the sequence of square numbers 1, 4, 9, 16, ... with 0s interwoven between them.

def make_roux(length):
    # Generate the reversed sequence first
    arr = np.zeros(length, dtype=int)
    for j in range(length):
        if j % 2 == 0:
            arr[j] = (j // 2 + 1) ** 2
    # Reverse it back to represent the original roux array
    return arr[::-1]

print("\nProblem 6 - roux array of length 5:", make_roux(5))
print("Problem 6 - roux array of length 8:", make_roux(8))

# ============================================================================
# Problem 7: Element Presence Check
# ============================================================================
# Write a NumPy program to test whether each element of a 1-D array is also present in a second array.

array1 = np.array([0, 10, 20, 40, 60])
array2 = np.array([0, 40])
output = np.isin(array1, array2)
print("\nProblem 7:")
print("Array1:", array1)
print("Array2:", array2)
print("Output:", output)

# ============================================================================
# Problem 8: Unique Elements
# ============================================================================
# Write a NumPy program to get the unique elements of an array.

arr1 = np.array([10, 10, 20, 20, 30, 30])
print("\nProblem 8:")
print("Original 1D array:", arr1)
print("Unique elements:", np.unique(arr1))

arr2 = np.array([[1, 1], [2, 3]])
print("\nOriginal 2D array:\n", arr2)
print("Unique elements:", np.unique(arr2))

# ============================================================================
# Problem 9: Sorting Along Axes
# ============================================================================
# Write a NumPy program to sort an array along the first and last axis.

a = np.array([[4, 6], [2, 1]])
print("\nProblem 9:")
print("Original array:\n", a)

sort_axis0 = np.sort(a, axis=0)
print("\nSort along the first axis (axis=0):\n", sort_axis0)

# Sorting the resulting array along the last axis yields the expected output [[1, 2], [4, 6]]
sort_both = np.sort(sort_axis0, axis=-1)
print("\nSort along the last axis (axis=-1) of sort_axis0:\n", sort_both)

# ============================================================================
# Problem 10: Count Occurrences and Remove NaNs
# ============================================================================
# 1. Count the occurrence of a specified item in a given NumPy array.
# 2. Remove nan values from a given array.

arr_counts = np.array([10, 20, 20, 20, 20, 0, 20, 30, 30, 30, 0, 0, 20, 20, 0])
unique, counts = np.unique(arr_counts, return_counts=True)
print("\nProblem 10:")
print("Original array:", arr_counts)
print("Value occurrences:", dict(zip(unique, counts)))

arr_nan = np.array([200., 300., np.nan, np.nan, np.nan, 700.])
arr_clean = arr_nan[~np.isnan(arr_nan)]
print("\nOriginal array with NaNs:", arr_nan)
print("After removing nan values:", arr_clean)

# ============================================================================
# Problem 11: Calculate Percentiles
# ============================================================================
# Write a NumPy program to calculate percentiles for a sequence or single-dimensional NumPy array.

seq = np.array([1, 2, 3, 4, 5])
p50 = np.percentile(seq, 50)
p40 = np.percentile(seq, 40)
p90 = np.percentile(seq, 90)
print("\nProblem 11:")
print("Original sequence:", seq)
print("50th percentile (median):", p50)
print("40th percentile:", p40)
print("90th percentile:", p90)

# ============================================================================
# Problem 12: Find Zero Indices
# ============================================================================
# Write a NumPy program to find indices of elements equal to zero in a NumPy array.

arr_zeros = np.array([1, 0, 2, 0, 3, 0, 4, 5, 6, 7, 8])
zeros_idx = np.where(arr_zeros == 0)[0]
print("\nProblem 12:")
print("Original array:", arr_zeros)
print("Indices of elements equal to zero:", zeros_idx)

# ============================================================================
# Problem 13: Indices of Sorted Elements
# ============================================================================
# Write a NumPy program to get the indices of the sorted elements of a given array.

arr_unsorted = np.array([1023, 5202, 6230, 1671, 1682, 5241, 4532])
sorted_idx = np.argsort(arr_unsorted)
print("\nProblem 13:")
print("Original array:", arr_unsorted)
print("Indices of the sorted elements of the array:", sorted_idx)

# ============================================================================
# Problem 14: Partition Array
# ============================================================================
# Write a NumPy program to partition a given array in a specified position and move all the smaller elements to the left of the partition, and the remaining values to the right (in arbitrary order).

arr_partition = np.array([70, 50, 20, 30, -11, 60, 50, 40])
partitioned_result = np.partition(arr_partition, 4)
print("\nProblem 14:")
print("Original array:", arr_partition)
print("After partitioning on 4th position (index 4):", partitioned_result)

# ============================================================================
# Problem 15: Euclidean Distance
# ============================================================================
# Compute the Euclidean distance between two arrays a and b.

a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7, 8])
dist = np.linalg.norm(a - b)
print("\nProblem 15:")
print("Array a:", a)
print("Array b:", b)
print(f"Euclidean distance is: {dist:.4f}")

# ============================================================================
# Problem 16: Find Peaks
# ============================================================================
# Find all the peaks in a 1D NumPy array a. Peaks are points surrounded by smaller values on both sides.

a_peaks = np.array([1, 3, 7, 1, 2, 6, 0, 1])
# Vectorized comparison (excluding boundary elements)
peaks_idx = np.where((a_peaks[1:-1] > a_peaks[:-2]) & (a_peaks[1:-1] > a_peaks[2:]))[0] + 1
print("\nProblem 16:")
print("Original array:", a_peaks)
print("Peak positions/indices:", peaks_idx)

# ============================================================================
# Problem 17: Index of N-th Repetition
# ============================================================================
# Find the index of the nth repetition of a number i.

arr_rep = np.array([1, 2, 1, 1, 3, 4, 3, 1, 1, 2, 1, 1, 2])

def nth_rep_index(arr, i, n):
    indices = np.where(arr == i)[0]
    if len(indices) >= n:
        return indices[n-1]
    return -1

# Given parameters
i_val = 1
n_val = 5
ans_idx = nth_rep_index(arr_rep, i_val, n_val)
print("\nProblem 17:")
print("Original array:", arr_rep)
print(f"Index of {n_val}-th repetition of {i_val}: {ans_idx}")

# ============================================================================
# Problem 18: Neighboring Difference, Prepend, and Append
# ============================================================================
# Write a NumPy program to calculate the difference between neighboring elements, element-wise, and prepend [0, 0] and append [200] to a given array.

arr_orig = np.array([1, 3, 5, 7, 0])
diff_vals = np.diff(arr_orig)
final_res = np.concatenate(([0, 0], diff_vals, [200]))
print("\nProblem 18:")
print("Original array:", arr_orig)
print("Differences:", diff_vals)
print("Resulting array:", final_res)

# ============================================================================
# Problem 19: Round and Absolute Value
# ============================================================================
# Write a NumPy program to round elements of the array to the nearest integer, and then calculate the absolute value element-wise.

arr_float = np.array([-0.7, -1.5, -1.7, 0.3, 1.5, 1.8, 2. ])
arr_rounded_abs = np.abs(np.round(arr_float))
print("\nProblem 19:")
print("Original array:", arr_float)
print("Rounded and absolute values:", arr_rounded_abs)
