import cv2
import numpy as np
import os
import time

def remap_label(label_img):
    rows, cols = label_img.shape
    # Re-map the labels
    unique_labels = np.unique(label_img)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    for y in range(rows):
        for x in range(cols):
            label_img[y, x] = label_map[label_img[y, x]]
    return label_img

def remove_small_region(label_img, label_num, threshold=500):
    # Removing small region
    for i in range(1, label_num):
        if len(label_img[label_img==i]) < threshold: # Remove small region
            label_img[label_img == i] = 0
    return label_img

"""
TODO Binary transfer
"""
def to_binary(img, preprocess=0):
    ##### Otsu's thresholding method #####
    total_pixels = img.size
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    hist = hist / total_pixels # Probability distribution of gray levels

    current_max = 0
    threshold = 0
    m = np.dot(np.arange(256), hist)
    a, b, m_a = 0, 0, 0

    for t in range(256):
        a += hist[t]  # Cumulative probability of the first class (below threshold)
        b = 1 - a     # Cumulative probability of the second class (above threshold)
        m_a += t * hist[t] # Cumulative mean intensity of the first class
        if a == 0 or b == 0:
            continue
        var = ((m_a - (m * a)) ** 2) / (a * b) # Compute between-class variance
        if var > current_max: # Update the maximum variance and optimal threshold
            current_max = var
            threshold = t
    
    # Create a binary image using threshold
    binary_img = (img < threshold).astype(np.uint8)
    if preprocess:
        # Fillimg small region
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        return closed_img
    else:
        return binary_img

"""
TODO Two-pass algorithm
"""
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Union by rank
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

def two_pass(binary_img, connectivity, postprocess=0):
    if connectivity == 4:
        neighbors = [(0, -1), (-1, 0)]
    else :
        neighbors = [(0, -1), (-1, 0), (-1, 1), (-1, -1)]
    
    rows, cols = binary_img.shape
    labels = np.zeros_like(binary_img, dtype=np.int64)
    uf = UnionFind(rows * cols)

    # First pass: Assign labels and record equivalences
    label_num = 1
    for y in range(rows):
        for x in range(cols):
            if binary_img[y, x]:
                adjacent_labels = []
                for dy, dx in neighbors: # Check neighbor's label
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols and binary_img[ny, nx] and labels[ny, nx] > 0:
                        adjacent_labels.append(labels[ny, nx]) # Collect adjacent labels
                
                if adjacent_labels: # If neighbors have labels
                    min_label = min(l for l in adjacent_labels)
                    labels[y, x] = min_label # Assign adjacent minimum label
                    for l in adjacent_labels:
                        if l != min_label:
                            uf.union(l, min_label) # Union other labels
                else: # Assign new label
                    labels[y, x] = label_num
                    label_num += 1

    # Second pass: Relabel
    for y in range(rows):
        for x in range(cols):
            # Set the same set to the root label
            if labels[y, x] > 0: labels[y, x] = uf.find(labels[y, x]) 

    # Removing small region
    if postprocess:
        labels = remove_small_region(labels, label_num, threshold=500)

    # Re-map the labels
    labels = remap_label(labels)
        
    return labels



"""
TODO Seed filling algorithm
"""
def seed_filling(binary_img, connectivity, postprocess=0):
    if connectivity == 4:
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    else :
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
    rows, cols = binary_img.shape
    labels = np.zeros_like(binary_img, dtype=np.int64)

    label_num = 1
    for y in range(rows):
        for x in range(cols):
            if binary_img[y, x] and labels[y, x] == 0:
                stack = [(y, x)] # Initialize the seed stack
                while stack: # Filling
                    current_y, current_x = stack.pop() # Pop next adjacent labels
                    if labels[current_y, current_x] != 0: # Skip the marked pixels
                        continue
                    labels[current_y, current_x] = label_num # Assign label
                    for dy, dx in neighbors: # Check neighbors
                        ny, nx = current_y + dy, current_x + dx
                        if 0 <= ny < rows and 0 <= nx < cols and binary_img[ny, nx] and labels[ny, nx] == 0:
                            stack.append((ny, nx)) # Push adjacent labels into stack
                label_num += 1

    # Removing small region
    if postprocess:
        labels = remove_small_region(labels, label_num, threshold=500)
        # Re-map the labels
        labels = remap_label(labels)

    return labels


"""
Bonus
"""
def rle_encode(binary_img):
        runs = []
        rows, cols = binary_img.shape
        for row in range(rows):
            start = None
            for col in range(cols):
                if binary_img[row, col] == 1:
                    if start is None:
                        start = col # Start of a run
                elif start is not None:
                    runs.append((row, start, col - 1)) # End of a run
                    start = None
            if start is not None:  # Handle runs that end at the row's edge
                runs.append((row, start, cols - 1))
        return runs

"""
Run-Length Encoding (RLE)-Based Methods
"""
def other_cca_algorithm(binary_img, connectivity, postprocess=0):
    runs = rle_encode(binary_img) # Encode the image into runs
    uf = UnionFind(len(runs))
    run_labels = {}  # Store the labels for each run
    current_label = 1

    for i, run in enumerate(runs):
        row, start, end = run
        neighbors = [] # Neighbors from the previous row
        
        # Check for neighbors in the previous row
        for j in range(i): # Only consider the previous row
            prev_row, prev_start, prev_end = runs[j]
            if prev_row == row - 1:
                # 8-connectivity: check horizontal and diagonal
                if connectivity == 8 and max(start, prev_start-1) <= min(end, prev_end+1):
                    neighbors.append(j)
                # 4-connectivity: check horizontal
                if connectivity == 4 and max(start, prev_start) <= min(end, prev_end):
                    neighbors.append(j)
        
        # Assign label to the current run
        if not neighbors:  # No neighbors, assign a new label
            run_labels[i] = current_label
            current_label += 1
        else: # Use the smallest label among neighbors
            first_label = run_labels[neighbors[0]]
            run_labels[i] = first_label
            for neighbor in neighbors:
                uf.union(first_label, run_labels[neighbor])

    # Compress labels to their roots
    for i in run_labels:
        run_labels[i] = uf.find(run_labels[i])

    # Write labels back to the image
    labeled_image = np.zeros_like(binary_img, dtype=np.int32)
    for i, run in enumerate(runs):
        row, start, end = run
        labeled_image[row, start:end + 1] = run_labels[i]

    if postprocess:
        labels = remove_small_region(labeled_image, current_label, threshold=500)
        # Re-map the labels
        labels = remap_label(labels)
    return labels


"""
TODO Color mapping
"""
def color_mapping(label_img):
    color_map = set()
    while len(color_map) < np.max(label_img) + 1:
        color = tuple(np.random.randint(0, 256, size=3))
        color_map.add(color)
    color_map = list(color_map)
    color_map[0] = [0,0,0]

    rows, cols = label_img.shape
    color_image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for y in range(rows):
        for x in range(cols):
            color_image[y, x] = color_map[label_img[y, x]]
    return color_image


"""
Main function
"""
def main():
    os.makedirs("result/connected_component/two_pass", exist_ok=True)
    os.makedirs("result/connected_component/seed_filling", exist_ok=True)
    os.makedirs("result/connected_component/RLE", exist_ok=True)
    connectivity_type = [4, 8]
    preprocess = 1
    postprocess = 1

    for i in range(2):
        img = cv2.imread("data/connected_component/input{}.png".format(i + 1), cv2.IMREAD_GRAYSCALE)

        for connectivity in connectivity_type:

            # TODO Part1: Transfer to binary image
            binary_img = to_binary(img, preprocess)

            # TODO Part2: CCA algorithm
            two_pass_label = two_pass(binary_img, connectivity, postprocess)
            seed_filling_label = seed_filling(binary_img, connectivity, postprocess)
            RLE_label = other_cca_algorithm(binary_img, connectivity, postprocess)

            # TODO Part3: Color mapping
            two_pass_color = color_mapping(two_pass_label)
            seed_filling_color = color_mapping(seed_filling_label)
            RLE_color = color_mapping(RLE_label)

            cv2.imwrite("result/connected_component/two_pass/input{}_c{}_p{}.png".format(i + 1, connectivity, preprocess), two_pass_color)
            cv2.imwrite("result/connected_component/seed_filling/input{}_c{}_p{}.png".format(i + 1, connectivity, preprocess), seed_filling_color)
            cv2.imwrite("result/connected_component/RLE/input{}_c{}_p{}.png".format(i + 1, connectivity, preprocess), RLE_color)


if __name__ == "__main__":
    main()