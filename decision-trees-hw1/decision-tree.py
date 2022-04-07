# CS460G Spring 2022
# Assignment 1 - Decision Tree Classifier
# Jackson Chumbler

import csv
import math
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from anytree import Node, RenderTree


def read_file(file_name):
    # Read .csv
    file = open(file_name)
    csvreader = csv.reader(file)

    rows = []
    for row in csvreader:
        row[0] = float(row[0])
        row[1] = float(row[1])
        row[2] = int(row[2])
        rows.append(row)

    file.close()
    # 'rows' holds .csv data
    # For clarity, add to separate lists
    x_vals = []
    y_vals = []
    labels = []
    for row in rows:
        x_vals.append(float(row[0]))
        y_vals.append(float(row[1]))
        labels.append(int(row[2]))
    return rows, x_vals, y_vals


def discretize_data(rows, x_vals, y_vals, axis, num_of_bins):
    if axis != "x" and axis != "y":
        print("Error: bad axis")
        return

    elif axis == "x":
        # We are going to split the data such that there are
        # roughly the same number of entries in each bin.
        min_x = min(x_vals)
        max_x = max(x_vals)
        x_range = (max_x - min_x)
        bin_size = x_range / num_of_bins
        # Put data in the bins
        # 10 empty bins
        bins = []
        for i in range(num_of_bins):
            bins.append([])

        for row in rows:
            what_bin = int((row[0] - min_x) / bin_size)
            if what_bin >= num_of_bins:  # max value
                what_bin = num_of_bins - 1
            bins[what_bin].append(row)
    elif axis == "y":
        # We are going to split the data such that there are
        # roughly the same number of entries in each bin.
        min_y = min(y_vals)
        max_y = max(y_vals)
        y_range = (max_y - min_y)
        bin_size = y_range / num_of_bins

        # Put data in the bins
        # 10 empty bins
        bins = []
        for i in range(num_of_bins):
            bins.append([])

        for row in rows:
            what_bin = int((row[1] - min_y) / bin_size)
            if what_bin >= num_of_bins:  # max value
                what_bin = num_of_bins - 1
            bins[what_bin].append(row)

    return bins

# Return entropy of group of data in binary classification.


def entropy(entries):
    zero_total = 0
    one_total = 0
    for entry in entries:
        if entry[2] == 0:
            zero_total += 1
        elif entry[2] == 1:
            one_total += 1
    if zero_total == 0 or one_total == 0:
        return 0
    zero_ratio = zero_total / (zero_total + one_total)
    one_ratio = one_total / (zero_total + one_total)
    entropy = (-1) * zero_ratio * math.log(zero_ratio, 2) + \
        (-1) * one_ratio * math.log(one_ratio, 2)
    return entropy

# Data is discretized before hand. We must discretize and call best_split twice.
# We have to discretize once on X and again on the Y val to see the best fit.


def best_split(bins):
    num_bins = len(bins)
    entries = []  # [x, y, val]
    for bin in bins:
        for entry in bin:
            entries.append(entry)
    initial_entropy = entropy(entries)  # not necessary yet...
    lowest_entropy = 999  # impossible value
    split_after = -1  # what bin to split
    for i in range(num_bins - 1):  # if there are 5 bins, can split after 0,1,2,3
        # Measure entropy of left size, weighted with ratio of total entries
        lower_entries = []
        for j in range(i):
            # Get all entries in these bins
            for entry in bins[j]:
                lower_entries.append(entry)

        # Measure entropy for right size, weighted with ratio of total entries
        upper_entries = []
        for j in range(i, num_bins):
            for entry in bins[j]:
                upper_entries.append(entry)
        # Now that we have a list of all data in bins below and above the potential split
        # We need to check the entropy of splitting on the x-val and y-val
        lower_ratio = len(lower_entries) / (len(lower_entries) + len(upper_entries))
        upper_ratio = len(upper_entries) / (len(lower_entries) + len(upper_entries))
        x_split_entropy = lower_ratio*entropy(lower_entries) + upper_ratio*entropy(upper_entries)
        if x_split_entropy < lowest_entropy:
            lowest_entropy = x_split_entropy
            split_after = i
            #print("Best Split for x: ", i, "entropy of: ", x_split_entropy)
            #print("Lower:", len(lower_entries), "Upper:", len(upper_entries))
        #print("bin level: ", i)
    information_gain = initial_entropy - lowest_entropy
    #print("Entropy:", initial_entropy, " to", lowest_entropy)
    #print("IG:", information_gain)
    #print("Split after", split_after)
    return split_after, information_gain

# While best split chooses the best linear separation of bins,
# we need a function which tests best_split on x & y discretizations.


def decide_next_split(rows, x_vals, y_vals, num_of_bins):
    # We start with x.
    bins_x = discretize_data(rows, x_vals, y_vals, "x", num_of_bins)
    split_after_x, information_gain_x = best_split(bins_x)

    # and then Y
    bins_y = discretize_data(rows, x_vals, y_vals, "y", num_of_bins)
    split_after_y, information_gain_y = best_split(bins_y)

    if information_gain_x >= information_gain_y:
        return "x", bins_x, split_after_x
    else:
        return "y", bins_y, split_after_y
# Return a left and right list of rows, given bin # to split after.


def split_data(bins, split_after):
    left = []
    right = []
    for i in range(split_after):
        for entry in bins[i]:
            left.append(entry)
    for i in range(split_after, len(bins)):
        for entry in bins[i]:
            right.append(entry)
    return left, right


# Each node of decision tree should have:
# -axis that its branches split on
# -value that is split after
# -depth
# Return what value is the upper bound for the left
# side of a split.
def split_after_val(bins, split_after, x_or_y):
    max_val = float('-inf')
    for i in range(split_after):
        if len(bins[i]) > 0:
            if x_or_y == "x":
                max_val = max(max_val, max(bins[i], key=lambda i: i[0])[0])
            elif x_or_y == "y":
                max_val = max(max_val, max(bins[i], key=lambda i: i[1])[1])
    return max_val


def decision_tree(rows, x_vals, y_vals, curr_node, depth, max_depth, num_of_bins):

    if(depth < max_depth):
        x_or_y, bins, split_after = decide_next_split(rows, x_vals, y_vals, num_of_bins)
        left, right = split_data(bins, split_after)

        # Left Child
        if len(left) > 0 and entropy(left) > 0:
            left_x_vals = []
            left_y_vals = []
            for row in left:
                left_x_vals.append(float(row[0]))
                left_y_vals.append(float(row[1]))
            nodename = "left"
            nodename += str(depth)

            left_x_or_y, left_bins, left_split_after = decide_next_split(
                left, left_x_vals, left_y_vals, num_of_bins)
            L_left, L_right = split_data(left_bins, left_split_after)

            L_split_after_num = split_after_val(left_bins, left_split_after, left_x_or_y)
            left_node = Node(nodename, x_or_y=left_x_or_y, split_after=L_split_after_num, depth=depth+1,
                             left_points=L_left, right_points=L_right, parent=curr_node)  # split_after=left_split_after
            left_node = decision_tree(left, left_x_vals, left_y_vals,
                                      left_node, depth+1, max_depth, num_of_bins)

        # RIGHT CHILD
        if len(right) > 0 and entropy(right) > 0:
            right_x_vals = []
            right_y_vals = []
            for row in right:
                right_x_vals.append(float(row[0]))
                right_y_vals.append(float(row[1]))
            nodename = "right"
            nodename += str(depth)

            right_x_or_y, right_bins, right_split_after = decide_next_split(
                right, right_x_vals, right_y_vals, num_of_bins)
            R_left, R_right = split_data(right_bins, right_split_after)

            R_split_after_num = split_after_val(right_bins, right_split_after, right_x_or_y)

            right_node = Node(nodename, x_or_y=right_x_or_y, split_after=R_split_after_num,
                              depth=depth+1, left_points=R_left, right_points=R_right, parent=curr_node)
            right_node = decision_tree(right, right_x_vals, right_y_vals,
                                       right_node, depth+1, max_depth, num_of_bins)

        depth += 1
    return curr_node

# This func provides the dominant label in a list of entries.
# Simply done by finding the mode.


def dominant_label(points):
    sum_0 = 0
    sum_1 = 0
    for point in points:
        if point[2] == 0:
            sum_0 += 1
        elif point[2] == 1:
            sum_1 += 1
    if sum_1 > sum_0:
        return 1
    else:
        return 0

# This returns the bounds for a node.
# This is done by viewing itself and all of its parents.


def rectangle_bounds(node):
    minx = float('-inf')
    miny = float('-inf')
    maxx = float('inf')
    maxy = float('inf')
    while node.parent is not None:
        if "left" in node.name:
            if node.parent.x_or_y == "x":
                maxx = min(maxx, node.parent.split_after)
                #print(node.parent.x_or_y, " <= ", node.parent.split_after)
            elif node.parent.x_or_y == "y":
                maxy = min(maxy, node.parent.split_after)
        elif "right" in node.name:
            #print(node.parent.x_or_y, " > ", node.parent.split_after)
            if node.parent.x_or_y == "x":
                minx = max(minx, node.parent.split_after)
            elif node.parent.x_or_y == "y":
                miny = max(miny, node.parent.split_after)
        node = node.parent
    return minx, maxx, miny, maxy

# This function is a bit obtuse. It traces the decision tree,
# and creates a list of rectangles associated with predicted label.
# We do this by starting with infinite bounds, and narrowing them down.


def find_decision_range(root_node):
    decisions = []  # of the form [[[minx, maxx],[miny, maxy]], label]

    if len(root_node.children) == 1:
        #print("root splits on", root_node.x_or_y, " ->", root_node.split_after)
        if "left" in root_node.children[0].name:
            if root_node.x_or_y == "x":
                decisions.append([[[root_node.split_after, float('inf')], [float(
                    '-inf'), float('inf')]], dominant_label(root_node.right_points)])
            elif root_node.x_or_y == "y":
                decisions.append([[[float('-inf'), float('inf')], [root_node.split_after,
                                 float('inf')]], dominant_label(root_node.right_points)])
        elif "right" in root_node.children[0].name:
            if root_node.x_or_y == "x":
                decisions.append([[[float('-inf'), root_node.split_after],
                                 [float('-inf'), float('inf')]], dominant_label(root_node.left_points)])
            elif root_node.x_or_y == "y":
                decisions.append([[[float('-inf'), float('inf')], [float('-inf'),
                                 root_node.split_after]], dominant_label(root_node.right_points)])
    elif len(root_node.children) == 0:
        if root_node.x_or_y == "x":
            decisions.append([[[root_node.split_after, float('inf')], [float(
                '-inf'), float('inf')]], dominant_label(root_node.right_points)])
            decisions.append([[[float('-inf'), root_node.split_after],
                             [float('-inf'), float('inf')]], dominant_label(root_node.left_points)])
        elif root_node.x_or_y == "y":
            decisions.append([[[float('-inf'), float('inf')], [root_node.split_after,
                             float('inf')]], dominant_label(root_node.right_points)])
            decisions.append([[[float('-inf'), float('inf')], [float('-inf'),
                             root_node.split_after]], dominant_label(root_node.left_points)])

    for node in root_node.descendants:
        if len(node.children) == 0:
            minx, maxx, miny, maxy = rectangle_bounds(node)
            left_range = [[minx, maxx], [miny, maxy]]
            right_range = [[minx, maxx], [miny, maxy]]
            if node.x_or_y == "x":
                # Left Range
                left_range[0][1] = node.split_after  # modify max x for left side of x-split
                # Right Range
                right_range[0][0] = node.split_after
            elif node.x_or_y == "y":
                # Left range
                left_range[1][1] = node.split_after
                # Right range
                right_range[1][0] = node.split_after
            decisions.append([left_range, dominant_label(node.left_points)])
            decisions.append([right_range, dominant_label(node.right_points)])
        elif len(node.children) == 1:
            minx, maxx, miny, maxy = rectangle_bounds(node)
            if "left" in node.children[0].name:
                right_range = [[minx, maxx], [miny, maxy]]
                if node.x_or_y == "x":
                    right_range[0][0] = node.split_after
                elif node.x_or_y == "y":
                    right_range[1][0] = node.split_after
                decisions.append([right_range, dominant_label(node.right_points)])
            elif "right" in node.children[0].name:
                left_range = [[minx, maxx], [miny, maxy]]
                if node.x_or_y == "x":
                    left_range[0][1] = node.split_after
                elif node.x_or_y == "y":
                    left_range[1][1] = node.split_after
                decisions.append([left_range, dominant_label(node.left_points)])
    # END FOR NODE
    return decisions


def test(data, decisions):
    num_correct = 0
    num_incorrect = 0
    for entry in data:
        for dec in decisions:
            # if entry x val fits between max & min accepted values
            if entry[0] > dec[0][0][0] and entry[0] <= dec[0][0][1]:
                # if entry y val fits between max&min accepted values
                if entry[1] > dec[0][1][0] and entry[1] <= dec[0][1][1]:
                    # Increment when label correct/incorrect
                    if entry[2] == dec[1]:
                        num_correct += 1
                    else:
                        num_incorrect += 1
    accuracy = num_correct / (num_correct + num_incorrect)
    print("#correct:", num_correct)
    print("#numincorrect:", num_incorrect)
    print("Accuracy:", accuracy)
    print("")


def print_tree(root_node):
    print_node = root_node
    # Remove large lists of points.
    # This is done to make the rendered tree readable in terminal.
    delattr(print_node, 'left_points')
    delattr(print_node, 'right_points')
    for print_child in print_node.descendants:
        delattr(print_child, 'left_points')
        delattr(print_child, 'right_points')
    print(RenderTree(print_node))

# Create a function which visualizes the data set, adding the
# output splits of the decision tree.
# decision_inequalities = list of [minx, maxx, label]


def graph_data(rows, x_vals, y_vals, decisions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x feat')
    ax.set_ylabel('f feat')
    ax.set_title('')
    categories = []

    for row in rows:
        categories.append(row[2])
    ax.scatter(x_vals, y_vals, c=categories)
    axes = plt.gca()

    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()

    for decision in decisions:
        list_of_x = []
        list_of_y = []
        for x_val in decision[0][0]:
            if x_val == float('inf'):
                x_val = x_max
            elif x_val == float('-inf'):
                x_val = x_min
            list_of_x.append(x_val)

        for y_val in decision[0][1]:
            if y_val == float('inf'):
                y_val = y_max
            elif y_val == float('-inf'):
                y_val = y_min
            list_of_y.append(y_val)
        min_x = min(list_of_x)
        max_x = max(list_of_x)
        min_y = min(list_of_y)
        max_y = max(list_of_y)
        rect_color = 'yellow'
        if decision[1] == 0:
            rect_color = 'purple'
        ax.add_patch(Rectangle(xy=(min_x, min_y), width=(max_x-min_x),
                     height=(max_y-min_y), alpha=0.2, color=rect_color, linewidth=0))
    ax.scatter(x_vals, y_vals, c=categories)
    yellow_patch = mpatches.Patch(color='purple', label='Label: 1')
    purple_patch = mpatches.Patch(color='yellow', label='Label: 0')
    ax.legend(handles=[purple_patch, yellow_patch])
    plt.show()


def main():
    max_depth = 3
    num_of_bins = 30
    # ->The line below is the only Pokemon data I am using.
    # Swap the comments or change synthetic-# to change dataset.
    rows, x_vals, y_vals = read_file('data/synthetic-4.csv')
    # rows, x_vals, y_vals = read_file('data/totalHpLegendary.csv')
    x_or_y, bins, split_after = decide_next_split(rows, x_vals, y_vals, num_of_bins)
    left, right = split_data(bins, split_after)

    # Create Tree
    # First, we create and initialize the root node.
    split_after_num = split_after_val(bins, split_after, x_or_y)
    root_node = Node("root", x_or_y=x_or_y, split_after=split_after_num,
                     left_points=left, right_points=right)
    # Calling decision tree on root_node completes the tree.
    root_node = decision_tree(rows, x_vals, y_vals, root_node, 0, max_depth, num_of_bins)

    # Decision Rectangles to terminal
    decisions = find_decision_range(root_node)
    print("\nDecisions Surface: [[xmin, xmax],[ymin, ymax], predicted_label]")
    for d in decisions:
        print(d)
    print("")

    # Test and print accuracy
    test(rows, decisions)
    # Render the tree to terminal
    print_tree(root_node)
    # Provide matplotlib graph
    graph_data(rows, x_vals, y_vals, decisions)


if __name__ == "__main__":
    main()
